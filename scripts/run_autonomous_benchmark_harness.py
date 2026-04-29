from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import os
import subprocess
import time
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_run_log(log_json: Path, log: dict[str, Any]) -> None:
    log_json.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_harness_spec(spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if spec.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if spec.get("report_kind") != "autonomous_benchmark_harness_spec":
        failures.append("report_kind must be autonomous_benchmark_harness_spec")
    if not str(spec.get("benchmark", "")).strip():
        failures.append("benchmark is required")
    prerequisites = spec.get("prerequisites", [])
    if prerequisites is not None and not isinstance(prerequisites, list):
        failures.append("prerequisites must be a list when provided")
        prerequisites = []
    for index, prerequisite in enumerate(prerequisites, start=1):
        if not isinstance(prerequisite, dict):
            failures.append(f"prerequisite {index} must be an object")
            continue
        name = str(prerequisite.get("name", "")).strip()
        if not name:
            failures.append(f"prerequisite {index} name is required")
        kind = str(prerequisite.get("kind", "")).strip()
        if kind not in {"account", "artifact", "environment"}:
            failures.append(f"prerequisite {name or index} kind must be account, artifact, or environment")
        required_env = prerequisite.get("required_env", [])
        if required_env is not None and (
            not isinstance(required_env, list) or any(not isinstance(value, str) or not value for value in required_env)
        ):
            failures.append(f"prerequisite {name or index} required_env must be a string list")
        proof_path = str(prerequisite.get("proof_path", "")).strip()
        if kind == "account" and not required_env and not proof_path:
            failures.append(f"account prerequisite {name or index} must declare required_env or proof_path")
        satisfied_by = str(prerequisite.get("satisfied_by", "env_or_proof")).strip()
        if satisfied_by not in {"env", "proof", "env_or_proof", "env_and_proof"}:
            failures.append(f"prerequisite {name or index} satisfied_by has unsupported value")
    contract = spec.get("autonomy_contract") if isinstance(spec.get("autonomy_contract"), dict) else {}
    if contract.get("operator_role") != "launch_and_monitor_only":
        failures.append("autonomy_contract.operator_role must be launch_and_monitor_only")
    if not str(contract.get("selection_mode", "")).strip():
        failures.append("autonomy_contract.selection_mode is required")
    prohibited = contract.get("prohibited_manual_interventions")
    if not isinstance(prohibited, list) or not prohibited:
        failures.append("autonomy_contract.prohibited_manual_interventions must be a non-empty list")
    kernel_owned = contract.get("kernel_owned_phases")
    if not isinstance(kernel_owned, list) or not kernel_owned:
        failures.append("autonomy_contract.kernel_owned_phases must be a non-empty list")
    countable = contract.get("countable_evidence")
    if not isinstance(countable, list) or not countable:
        failures.append("autonomy_contract.countable_evidence must be a non-empty list")
    phases = spec.get("phases")
    if not isinstance(phases, list) or not phases:
        failures.append("phases must be a non-empty list")
        return failures
    seen_names: set[str] = set()
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            failures.append(f"phase {index} must be an object")
            continue
        name = str(phase.get("name", "")).strip()
        if not name:
            failures.append(f"phase {index} name is required")
        elif name in seen_names:
            failures.append(f"duplicate phase name: {name}")
        seen_names.add(name)
        if phase.get("kind") != "command":
            failures.append(f"phase {name or index} kind must be command")
        argv = phase.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(part, str) or not part for part in argv):
            failures.append(f"phase {name or index} argv must be a non-empty string list")
        preflight_argv = phase.get("preflight_argv", [])
        if preflight_argv is not None and (
            not isinstance(preflight_argv, list)
            or any(not isinstance(part, str) or not part for part in preflight_argv)
        ):
            failures.append(f"phase {name or index} preflight_argv must be a string list")
        env = phase.get("env", {})
        if env is not None and (
            not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
        ):
            failures.append(f"phase {name or index} env must be a string map")
        phase_cwd = phase.get("cwd", "")
        if phase_cwd is not None and not isinstance(phase_cwd, str):
            failures.append(f"phase {name or index} cwd must be a string")
        outputs = phase.get("expected_outputs", [])
        if outputs is not None and (
            not isinstance(outputs, list) or any(not isinstance(value, str) or not value for value in outputs)
        ):
            failures.append(f"phase {name or index} expected_outputs must be a string list")
        inputs = phase.get("required_inputs", [])
        if inputs is not None and (
            not isinstance(inputs, list) or any(not isinstance(value, str) or not value for value in inputs)
        ):
            failures.append(f"phase {name or index} required_inputs must be a string list")
    if isinstance(kernel_owned, list):
        phase_names = {str(phase.get("name", "")).strip() for phase in phases if isinstance(phase, dict)}
        for phase_name in kernel_owned:
            if str(phase_name).strip() not in phase_names:
                failures.append(f"autonomy_contract.kernel_owned_phases references unknown phase: {phase_name}")
    return failures


def _path_exists(path: str, *, cwd: Path) -> bool:
    raw = Path(path)
    return raw.exists() if raw.is_absolute() else (cwd / raw).exists()


def check_phase_required_inputs(phases: list[dict[str, Any]], *, cwd: Path) -> list[str]:
    missing: list[str] = []
    for phase in phases:
        name = str(phase.get("name", "")).strip() or "unnamed_phase"
        for required_input in phase.get("required_inputs", []):
            if not _path_exists(str(required_input), cwd=cwd):
                missing.append(f"{name} missing required input: {required_input}")
    return missing


def check_harness_required_inputs(phases: list[dict[str, Any]], *, cwd: Path) -> list[str]:
    missing: list[str] = []
    produced_outputs: set[str] = set()
    for phase in phases:
        name = str(phase.get("name", "")).strip() or "unnamed_phase"
        for required_input in phase.get("required_inputs", []):
            required_input = str(required_input)
            if required_input in produced_outputs:
                continue
            if not _path_exists(required_input, cwd=cwd):
                missing.append(f"{name} missing required input: {required_input}")
        produced_outputs.update(str(output) for output in phase.get("expected_outputs", []))
    return missing


def check_phase_preflight(
    phases: list[dict[str, Any]],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> list[str]:
    failures: list[str] = []
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    for phase in phases:
        name = str(phase.get("name", "")).strip() or "unnamed_phase"
        preflight_argv = phase.get("preflight_argv", [])
        if not preflight_argv:
            continue
        missing_inputs = check_phase_required_inputs([phase], cwd=cwd)
        if missing_inputs:
            continue
        phase_env = base_env.copy()
        phase_env.update({str(k): str(v) for k, v in dict(phase.get("env", {})).items()})
        phase_cwd = Path(str(phase.get("cwd") or cwd))
        if not phase_cwd.is_absolute():
            phase_cwd = cwd / phase_cwd
        completed = subprocess.run(
            [str(part) for part in preflight_argv],
            cwd=str(phase_cwd),
            env=phase_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if int(completed.returncode) != 0:
            stderr = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
            stdout = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
            detail = stderr or stdout or f"returncode={completed.returncode}"
            failures.append(f"{name} preflight failed: {detail}")
    return failures


def check_harness_prerequisites(spec: dict[str, Any], *, env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    failures: list[str] = []
    prerequisites = spec.get("prerequisites", [])
    if not isinstance(prerequisites, list):
        return ["prerequisites must be a list when provided"]
    for prerequisite in prerequisites:
        if not isinstance(prerequisite, dict):
            continue
        if prerequisite.get("blocking") is False:
            continue
        name = str(prerequisite.get("name", "unnamed_prerequisite")).strip() or "unnamed_prerequisite"
        required_env = [str(value) for value in prerequisite.get("required_env", []) if str(value).strip()]
        proof_path = str(prerequisite.get("proof_path", "")).strip()
        satisfied_by = str(prerequisite.get("satisfied_by", "env_or_proof")).strip() or "env_or_proof"
        env_ok = bool(required_env) and all(str(env.get(key, "")).strip() for key in required_env)
        proof_ok = bool(proof_path) and Path(proof_path).exists()
        if satisfied_by == "env":
            ok = env_ok
        elif satisfied_by == "proof":
            ok = proof_ok
        elif satisfied_by == "env_and_proof":
            ok = env_ok and proof_ok
        else:
            ok = env_ok or proof_ok
        if not ok:
            failures.append(
                f"{name} unsatisfied: required_env={','.join(required_env) or '-'} "
                f"proof_path={proof_path or '-'} satisfied_by={satisfied_by}"
            )
    return failures


def harness_status(
    spec: dict[str, Any],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    run_preflight: bool = False,
) -> dict[str, Any]:
    validation_failures = validate_harness_spec(spec)
    prerequisite_failures = [] if validation_failures else check_harness_prerequisites(spec, env=env)
    phases = spec.get("phases") if isinstance(spec.get("phases"), list) else []
    initial_input_failures: list[str] = []
    preflight_failures: list[str] = []
    if phases:
        initial_input_failures = check_harness_required_inputs(phases, cwd=cwd)
        if run_preflight and not validation_failures and not prerequisite_failures and not initial_input_failures:
            preflight_failures = check_phase_preflight([phases[0]], cwd=cwd, env=env)
    blockers = validation_failures + prerequisite_failures + initial_input_failures + preflight_failures
    return {
        "benchmark": str(spec.get("benchmark", "")).strip(),
        "runnable": not blockers,
        "validation_failures": validation_failures,
        "prerequisite_failures": prerequisite_failures,
        "initial_input_failures": initial_input_failures,
        "preflight_failures": preflight_failures,
        "blockers": blockers,
    }


def _phase_range(phases: list[dict[str, Any]], *, start_at: str, stop_after: str) -> list[dict[str, Any]]:
    selected = phases
    if start_at:
        names = [str(phase.get("name", "")) for phase in phases]
        if start_at not in names:
            raise SystemExit(f"unknown start phase: {start_at}")
        selected = phases[names.index(start_at) :]
    if stop_after:
        names = [str(phase.get("name", "")) for phase in selected]
        if stop_after not in names:
            raise SystemExit(f"unknown stop phase: {stop_after}")
        selected = selected[: names.index(stop_after) + 1]
    return selected


def run_harness_spec(
    spec: dict[str, Any],
    *,
    cwd: Path,
    log_json: Path,
    start_at: str = "",
    stop_after: str = "",
    heartbeat_seconds: float = 30.0,
) -> dict[str, Any]:
    failures = validate_harness_spec(spec)
    if failures:
        raise SystemExit("autonomous benchmark harness validation failed: " + "; ".join(failures))
    prerequisite_failures = check_harness_prerequisites(spec)
    phases = _phase_range(spec["phases"], start_at=start_at, stop_after=stop_after)
    log: dict[str, Any] = {
        "report_kind": "autonomous_benchmark_harness_run_log",
        "benchmark": spec.get("benchmark", ""),
        "created_at": _now(),
        "cwd": str(cwd),
        "phase_results": [],
        "success": False,
    }
    log_json.parent.mkdir(parents=True, exist_ok=True)
    if prerequisite_failures:
        log["prerequisite_failures"] = prerequisite_failures
        _write_run_log(log_json, log)
        raise SystemExit("harness prerequisites not satisfied: " + "; ".join(prerequisite_failures))
    heartbeat_seconds = max(1.0, float(heartbeat_seconds))
    for phase in phases:
        phase_name = str(phase["name"])
        missing_inputs = check_phase_required_inputs([phase], cwd=cwd)
        if missing_inputs:
            phase_result = {
                "name": phase_name,
                "started_at": _now(),
                "completed_at": _now(),
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "missing_inputs": missing_inputs,
                "missing_outputs": [],
            }
            log["phase_results"].append(phase_result)
            log["failed_phase"] = phase_name
            _write_run_log(log_json, log)
            raise SystemExit("harness phase missing required inputs: " + "; ".join(missing_inputs))
        started_at = _now()
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in dict(phase.get("env", {})).items()})
        argv = [str(part) for part in phase["argv"]]
        phase_cwd = Path(str(phase.get("cwd") or cwd))
        if not phase_cwd.is_absolute():
            phase_cwd = cwd / phase_cwd
        started_monotonic = time.monotonic()
        log["active_phase"] = {
            "argv": argv,
            "cwd": str(phase_cwd),
            "elapsed_seconds": 0.0,
            "heartbeat_at": started_at,
            "name": phase_name,
            "pid": None,
            "started_at": started_at,
        }
        _write_run_log(log_json, log)
        process = subprocess.Popen(
            argv,
            cwd=str(phase_cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log["active_phase"]["pid"] = int(process.pid)
        _write_run_log(log_json, log)
        while True:
            try:
                stdout, stderr = process.communicate(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                log["active_phase"]["heartbeat_at"] = _now()
                log["active_phase"]["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
                _write_run_log(log_json, log)
        missing_outputs = [
            output
            for output in phase.get("expected_outputs", [])
            if not (cwd / str(output)).exists() and not Path(str(output)).exists()
        ]
        phase_result = {
            "name": phase_name,
            "started_at": started_at,
            "completed_at": _now(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "pid": int(process.pid),
            "returncode": int(process.returncode or 0),
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "cwd": str(phase_cwd),
            "missing_outputs": missing_outputs,
        }
        log["phase_results"].append(phase_result)
        log.pop("active_phase", None)
        _write_run_log(log_json, log)
        if int(process.returncode or 0) != 0 or missing_outputs:
            log["failed_phase"] = phase_name
            _write_run_log(log_json, log)
            raise SystemExit(f"harness phase failed: {phase_name}")
    log["success"] = True
    log["completed_at"] = _now()
    _write_run_log(log_json, log)
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--harness-json", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--harness-json", action="append", default=None)
    status_parser.add_argument("--harness-dir", default="")
    status_parser.add_argument("--cwd", default=".")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--preflight", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--harness-json", required=True)
    run_parser.add_argument("--log-json", required=True)
    run_parser.add_argument("--cwd", default=".")
    run_parser.add_argument("--start-at", default="")
    run_parser.add_argument("--stop-after", default="")
    run_parser.add_argument("--heartbeat-seconds", type=float, default=30.0)

    args = parser.parse_args()
    if args.mode == "status":
        paths = [Path(path) for path in (args.harness_json or [])]
        if args.harness_dir:
            paths.extend(sorted(Path(args.harness_dir).glob("*.json")))
        if not paths:
            raise SystemExit("--harness-json or --harness-dir is required for status")
        statuses = [
            {
                "path": str(path),
                **harness_status(_read_json(path), cwd=Path(args.cwd).resolve(), run_preflight=bool(args.preflight)),
            }
            for path in paths
        ]
        if args.json:
            print(json.dumps({"harness_count": len(statuses), "statuses": statuses}, indent=2, sort_keys=True))
        else:
            for item in statuses:
                state = "runnable" if item["runnable"] else "blocked"
                blockers = "; ".join(item["blockers"])
                print(f"{state} benchmark={item['benchmark']} harness={item['path']} blockers={blockers}")
        return

    spec = _read_json(Path(args.harness_json))
    failures = validate_harness_spec(spec)
    if failures:
        raise SystemExit("autonomous benchmark harness validation failed: " + "; ".join(failures))
    if args.mode == "validate":
        print(f"verified_autonomous_benchmark_harness={args.harness_json}")
        return
    log = run_harness_spec(
        spec,
        cwd=Path(args.cwd).resolve(),
        log_json=Path(args.log_json),
        start_at=args.start_at,
        stop_after=args.stop_after,
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    print(
        f"benchmark={log['benchmark']} "
        f"phase_count={len(log['phase_results'])} "
        f"log_json={args.log_json}"
    )


if __name__ == "__main__":
    main()
