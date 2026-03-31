from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
from datetime import datetime, timezone

from agent_kernel.config import KernelConfig


def _load_plan(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read promotion plan {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"promotion plan is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"promotion plan is not a JSON object: {path}")
    return payload


def _run_shell_command(command: str, *, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout).strip(),
        "stderr": str(completed.stderr).strip(),
    }


def _is_missing_prior_retained_baseline(result: dict[str, object]) -> bool:
    text = " ".join(
        [
            str(result.get("stdout", "")).strip(),
            str(result.get("stderr", "")).strip(),
        ]
    ).strip()
    return "no prior retained baseline exists for subsystem=" in text


def _compare_classification(result: dict[str, object]) -> str:
    if int(result.get("returncode", 0) or 0) == 0:
        return "compared"
    if _is_missing_prior_retained_baseline(result):
        return "bootstrap_first_retain"
    return "compare_failed"


def _finalize_execution_command(command: str, *, apply_finalize: bool) -> str:
    text = str(command).strip()
    if apply_finalize:
        return text.replace(" --dry-run", "")
    return text


def _parse_finalize_result(stdout: str) -> dict[str, str]:
    text = str(stdout).strip()
    result = {
        "cycle_id": "",
        "subsystem": "",
        "state": "",
        "reason": "",
    }
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in result:
            result[key] = value
    return result


def _should_run_finalize(
    *,
    subsystem: str,
    compare_status: str,
    apply_finalize: bool,
    allow_bootstrap_finalize: bool,
    allowed_bootstrap_subsystems: set[str],
) -> tuple[bool, str]:
    normalized = str(compare_status).strip()
    if normalized == "compare_failed":
        return False, "compare_failed"
    if (
        normalized == "bootstrap_first_retain"
        and allowed_bootstrap_subsystems
        and str(subsystem).strip() not in allowed_bootstrap_subsystems
    ):
        return False, "bootstrap_subsystem_not_allowed"
    if normalized == "bootstrap_first_retain" and apply_finalize and not allow_bootstrap_finalize:
        return False, "bootstrap_requires_review"
    return True, ""


def _candidate_pass_record(
    candidate: dict[str, object],
    *,
    compare_result: dict[str, object],
    finalize_result: dict[str, object],
    apply_finalize: bool,
) -> dict[str, object]:
    finalize_stdout = str(finalize_result.get("stdout", "")).strip()
    parsed_finalize = _parse_finalize_result(finalize_stdout)
    compare_classification = _compare_classification(compare_result)
    return {
        "scope_id": str(candidate.get("scope_id", "")).strip(),
        "cycle_id": str(candidate.get("cycle_id", "")).strip(),
        "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
        "selected_variant_id": str(candidate.get("selected_variant_id", "")).strip(),
        "candidate_artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
        "promotion_score": float(candidate.get("promotion_score", 0.0) or 0.0),
        "apply_finalize": bool(apply_finalize),
        "promotion_route": compare_classification,
        "compare": compare_result,
        "compare_status": compare_classification,
        "finalize": finalize_result,
        "finalize_skipped": bool(finalize_result.get("skipped", False)),
        "finalize_skip_reason": str(finalize_result.get("skip_reason", "")).strip(),
        "finalize_state": str(parsed_finalize.get("state", "")).strip(),
        "finalize_reason": str(parsed_finalize.get("reason", "")).strip(),
        "finalize_cycle_id": str(parsed_finalize.get("cycle_id", "")).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--promotion-plan",
        default="",
        help="Promotion plan from scripts/report_frontier_promotion_plan.py. Defaults to the shared report path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on plan entries to execute")
    parser.add_argument(
        "--apply-finalize",
        action="store_true",
        help="Actually run finalize without --dry-run. Omit to keep finalize in dry-run mode.",
    )
    parser.add_argument(
        "--allow-bootstrap-finalize",
        action="store_true",
        help="Allow --apply-finalize to retain bootstrap_first_retain candidates without manual review.",
    )
    parser.add_argument(
        "--allow-bootstrap-subsystem",
        action="append",
        default=[],
        help="Optional repeated subsystem allowlist for bootstrap finalize. When set, bootstrap finalize runs only for matching subsystems.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--allow-subsystem",
        action="append",
        default=[],
        help="Optional repeated subsystem allowlist. When set, only matching candidates are executed.",
    )
    parser.add_argument(
        "--block-subsystem",
        action="append",
        default=[],
        help="Optional repeated subsystem blocklist. Matching candidates are skipped and recorded in the pass report.",
    )
    parser.add_argument("--output-path", default="", help="Optional explicit report path")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    plan_path = (
        Path(str(args.promotion_plan).strip())
        if str(args.promotion_plan).strip()
        else config.improvement_reports_dir / "supervised_frontier_promotion_plan.json"
    )
    plan = _load_plan(plan_path)
    candidates = plan.get("promotion_candidates", [])
    if not isinstance(candidates, list):
        raise SystemExit(f"promotion plan is missing promotion_candidates list: {plan_path}")
    if int(args.limit) > 0:
        candidates = candidates[: int(args.limit)]
    allowed_subsystems = {
        str(value).strip()
        for value in list(args.allow_subsystem or [])
        if str(value).strip()
    }
    blocked_subsystems = {
        str(value).strip()
        for value in list(args.block_subsystem or [])
        if str(value).strip()
    }
    allowed_bootstrap_subsystems = {
        str(value).strip()
        for value in list(args.allow_bootstrap_subsystem or [])
        if str(value).strip()
    }

    repo_root = Path(__file__).resolve().parents[1]
    started_at = datetime.now(timezone.utc)
    results: list[dict[str, object]] = []
    compare_failures = 0
    bootstrap_candidates = 0
    finalize_failures = 0
    retained = 0
    rejected = 0
    skipped_candidates = 0
    skipped_bootstrap_finalize = 0
    skipped_bootstrap_subsystem_not_allowed = 0
    skipped_compare_failed_finalize = 0

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        if allowed_subsystems and subsystem not in allowed_subsystems:
            skipped_candidates += 1
            results.append(
                {
                    "scope_id": str(candidate.get("scope_id", "")).strip(),
                    "cycle_id": str(candidate.get("cycle_id", "")).strip(),
                    "selected_subsystem": subsystem,
                    "selected_variant_id": str(candidate.get("selected_variant_id", "")).strip(),
                    "candidate_artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                    "promotion_route": "skipped_by_policy",
                    "skip_reason": f"subsystem_not_allowed:{subsystem or 'unknown'}",
                }
            )
            continue
        if subsystem in blocked_subsystems:
            skipped_candidates += 1
            results.append(
                {
                    "scope_id": str(candidate.get("scope_id", "")).strip(),
                    "cycle_id": str(candidate.get("cycle_id", "")).strip(),
                    "selected_subsystem": subsystem,
                    "selected_variant_id": str(candidate.get("selected_variant_id", "")).strip(),
                    "candidate_artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                    "promotion_route": "skipped_by_policy",
                    "skip_reason": f"subsystem_blocked:{subsystem}",
                }
            )
            continue
        compare_command = str(candidate.get("compare_command", "")).strip()
        finalize_command = _finalize_execution_command(
            str(candidate.get("finalize_command", "")).strip(),
            apply_finalize=bool(args.apply_finalize),
        )
        if not compare_command or not finalize_command:
            entry = {
                "scope_id": str(candidate.get("scope_id", "")).strip(),
                "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                "error": "candidate is missing compare_command or finalize_command",
            }
            results.append(entry)
            compare_failures += 1
            finalize_failures += 1
            if args.stop_on_error:
                break
            continue

        compare_result = _run_shell_command(compare_command, cwd=repo_root)
        compare_status = _compare_classification(compare_result)
        if compare_status == "bootstrap_first_retain":
            bootstrap_candidates += 1
        elif compare_status == "compare_failed":
            compare_failures += 1
            if args.stop_on_error:
                results.append(
                    _candidate_pass_record(
                        candidate,
                        compare_result=compare_result,
                        finalize_result={"command": finalize_command, "returncode": -1, "stdout": "", "stderr": ""},
                        apply_finalize=bool(args.apply_finalize),
                    )
                )
                break

        should_run_finalize, finalize_skip_reason = _should_run_finalize(
            subsystem=subsystem,
            compare_status=compare_status,
            apply_finalize=bool(args.apply_finalize),
            allow_bootstrap_finalize=bool(args.allow_bootstrap_finalize),
            allowed_bootstrap_subsystems=allowed_bootstrap_subsystems,
        )
        if should_run_finalize:
            finalize_result = _run_shell_command(finalize_command, cwd=repo_root)
        else:
            if finalize_skip_reason == "bootstrap_requires_review":
                skipped_bootstrap_finalize += 1
            elif finalize_skip_reason == "bootstrap_subsystem_not_allowed":
                skipped_bootstrap_subsystem_not_allowed += 1
            elif finalize_skip_reason == "compare_failed":
                skipped_compare_failed_finalize += 1
            finalize_result = {
                "command": finalize_command,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "skipped": True,
                "skip_reason": finalize_skip_reason,
            }
        if int(finalize_result.get("returncode", 0) or 0) != 0:
            finalize_failures += 1
        parsed = _parse_finalize_result(str(finalize_result.get("stdout", "")).strip())
        state = str(parsed.get("state", "")).strip()
        if state == "retain":
            retained += 1
        elif state == "reject":
            rejected += 1
        results.append(
            _candidate_pass_record(
                candidate,
                compare_result=compare_result,
                finalize_result=finalize_result,
                apply_finalize=bool(args.apply_finalize),
            )
        )
        if args.stop_on_error and (
            int(compare_result.get("returncode", 0) or 0) != 0 or int(finalize_result.get("returncode", 0) or 0) != 0
        ):
            break

    completed_at = datetime.now(timezone.utc)
    payload = {
        "report_kind": "supervised_frontier_promotion_pass",
        "generated_at": completed_at.isoformat(),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "promotion_plan_path": str(plan_path),
        "summary": {
            "candidate_count": len(candidates),
            "executed_candidates": len(results),
            "skipped_candidates": skipped_candidates,
            "apply_finalize": bool(args.apply_finalize),
            "compare_failures": compare_failures,
            "bootstrap_candidates": bootstrap_candidates,
            "finalize_failures": finalize_failures,
            "skipped_bootstrap_finalize": skipped_bootstrap_finalize,
            "skipped_bootstrap_subsystem_not_allowed": skipped_bootstrap_subsystem_not_allowed,
            "skipped_compare_failed_finalize": skipped_compare_failed_finalize,
            "retained": retained,
            "rejected": rejected,
        },
        "policy": {
            "allow_bootstrap_finalize": bool(args.allow_bootstrap_finalize),
            "allowed_bootstrap_subsystems": sorted(allowed_bootstrap_subsystems),
            "allowed_subsystems": sorted(allowed_subsystems),
            "blocked_subsystems": sorted(blocked_subsystems),
        },
        "results": results,
    }
    output_path = (
        Path(str(args.output_path).strip())
        if str(args.output_path).strip()
        else config.improvement_reports_dir / "supervised_frontier_promotion_pass.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
