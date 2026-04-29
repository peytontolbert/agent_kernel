from __future__ import annotations

from pathlib import Path
import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a8_benchmark_adapter_specs import build_default_adapter_spec
from prepare_swe_bench_a8_run import build_swe_autonomous_harness_spec, build_swe_bench_command


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(
    name: str,
    argv: list[str],
    *,
    preflight_argv: list[str] | None = None,
    required_inputs: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    gate: str = "",
) -> dict[str, Any]:
    phase: dict[str, Any] = {
        "name": name,
        "kind": "command",
        "argv": argv,
    }
    if preflight_argv:
        phase["preflight_argv"] = preflight_argv
    if required_inputs:
        phase["required_inputs"] = required_inputs
    if expected_outputs:
        phase["expected_outputs"] = expected_outputs
    if gate:
        phase["gate"] = gate
    return phase


def _default_prerequisites(benchmark: str) -> list[dict[str, Any]]:
    if benchmark != "codeforces":
        return []
    return [
        {
            "blocking": True,
            "kind": "account",
            "name": "codeforces_account",
            "proof_path": "/data/agentkernel/benchmarks/codeforces/account_gate.json",
            "reason": "Codeforces benchmark execution and rating evidence require a real Codeforces account/session outside this repo.",
            "required_env": ["CODEFORCES_HANDLE"],
            "satisfied_by": "env_or_proof",
        }
    ]


def _prerequisites_from_run_spec(spec: dict[str, Any], *, benchmark: str) -> list[dict[str, Any]]:
    raw = spec.get("prerequisites")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]
    return _default_prerequisites(benchmark)


def _adapter_command(
    *,
    benchmark: str,
    adapter: dict[str, Any],
    runner: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(adapter.get("script", "scripts/run_a8_benchmark_adapter.py")),
        "--benchmark",
        benchmark,
        "--summary-json",
        str(adapter["summary_json"]),
        "--output-json",
        str(adapter["output_packet_json"]),
    ]
    if adapter.get("conservative_comparison_report") is True:
        command.append("--conservative-comparison-report")
    adapter_spec_json = str(adapter.get("adapter_spec_json", "")).strip()
    if adapter_spec_json:
        command.extend(["--adapter-spec-json", adapter_spec_json])
    runner = runner or {}
    command_json = str(runner.get("runner_command_json", runner.get("command_json", ""))).strip()
    if command_json:
        command.extend(["--runner-command-json", command_json])
    runner_cwd = str(runner.get("runner_cwd", runner.get("cwd", ""))).strip()
    if runner_cwd:
        command.extend(["--runner-cwd", runner_cwd])
    runner_log_json = str(runner.get("runner_log_json", "")).strip()
    if runner_log_json:
        command.extend(["--runner-log-json", runner_log_json])
    return command


def _adapter_required_inputs(adapter: dict[str, Any], *inputs: str) -> list[str]:
    required = [value for value in inputs if str(value).strip()]
    adapter_spec_json = str(adapter.get("adapter_spec_json", "")).strip()
    if adapter_spec_json:
        required.append(adapter_spec_json)
    return required


def _adapter_spec_validation_phases(
    *,
    benchmark: str,
    adapter: dict[str, Any],
    python_bin: str,
) -> list[dict[str, Any]]:
    adapter_spec_json = str(adapter.get("adapter_spec_json", "")).strip()
    if not adapter_spec_json:
        return []
    return [
        _phase(
            "validate_adapter_spec",
            [
                python_bin,
                "scripts/a8_benchmark_adapter_specs.py",
                "validate",
                "--adapter-spec-json",
                adapter_spec_json,
                "--benchmark",
                benchmark,
            ],
            required_inputs=[adapter_spec_json],
            gate="generated benchmark adapter spec must be valid before benchmark adaptation",
        )
    ]


def _runner_text(runner: dict[str, Any], key: str, default: str = "") -> str:
    return str(runner.get(key, default)).strip() or default


def _runner_int(runner: dict[str, Any], key: str, default: int) -> int:
    raw = runner.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _swe_autonomous_path(runner: dict[str, Any], key: str, artifact_root: str, filename: str) -> str:
    explicit = _runner_text(runner, key)
    if explicit:
        return explicit
    if not artifact_root:
        raise ValueError(f"runner.{key} or runner.artifact_root is required for swebench_autonomous_queue")
    return str(Path(artifact_root) / filename)


def _attach_adapter_spec_to_swe_autonomous_harness(
    harness: dict[str, Any],
    *,
    benchmark: str,
    adapter: dict[str, Any],
    python_bin: str,
) -> None:
    adapter_spec_json = str(adapter.get("adapter_spec_json", "")).strip()
    if not adapter_spec_json:
        return
    harness.setdefault("artifacts", {})["adapter_spec_json"] = adapter_spec_json
    for phase in harness.get("phases", []):
        if not isinstance(phase, dict) or phase.get("name") != "adapt_a8_packet":
            continue
        for key in ("argv", "preflight_argv"):
            argv = phase.get(key)
            if isinstance(argv, list) and "--adapter-spec-json" not in argv:
                argv.extend(["--adapter-spec-json", adapter_spec_json])
        required_inputs = phase.setdefault("required_inputs", [])
        if isinstance(required_inputs, list) and adapter_spec_json not in required_inputs:
            required_inputs.append(adapter_spec_json)
        break
    validation_phases = _adapter_spec_validation_phases(
        benchmark=benchmark,
        adapter=adapter,
        python_bin=python_bin,
    )
    if validation_phases:
        harness["phases"] = validation_phases + list(harness.get("phases", []))
        harness["autonomy_contract"]["kernel_owned_phases"] = [
            str(phase.get("name", "")).strip()
            for phase in harness["phases"]
            if isinstance(phase, dict) and str(phase.get("name", "")).strip()
        ]


def build_autonomous_harness_from_run_spec(
    spec: dict[str, Any],
    *,
    spec_path: str = "",
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    if spec.get("report_kind") not in {"a8_benchmark_run_spec", "standalone_benchmark_run_spec"}:
        raise ValueError("run spec report_kind must be a8_benchmark_run_spec or standalone_benchmark_run_spec")
    benchmark = str(spec.get("benchmark", "")).strip()
    if not benchmark:
        raise ValueError("run spec benchmark is required")
    runner = spec.get("runner") if isinstance(spec.get("runner"), dict) else {}
    adapter = spec.get("adapter") if isinstance(spec.get("adapter"), dict) else {}
    runner_kind = str(runner.get("kind", "")).strip()
    phases: list[dict[str, Any]]
    artifacts: dict[str, str] = {
        "run_spec_json": spec_path,
        "summary_json": str(adapter.get("summary_json", "")),
        "output_packet_json": str(adapter.get("output_packet_json", "")),
    }
    adapter_spec_json = str(adapter.get("adapter_spec_json", "")).strip()
    if adapter_spec_json:
        artifacts["adapter_spec_json"] = adapter_spec_json
    open_limits = [
        "This harness only makes benchmark execution reproducible; benchmark claims require verified output packets.",
        "The operator should launch or monitor the harness, not hand-edit benchmark summaries or predictions.",
    ]
    prerequisites = _prerequisites_from_run_spec(spec, benchmark=benchmark)
    if any(item.get("kind") == "account" for item in prerequisites):
        open_limits.insert(0, f"{benchmark} is account-gated; satisfy harness prerequisites before execution.")
    if runner_kind == "summary_only":
        summary_source = str(runner.get("summary_source", adapter.get("summary_json", "")))
        adapter_command = _adapter_command(benchmark=benchmark, adapter=adapter)
        phases = _adapter_spec_validation_phases(benchmark=benchmark, adapter=adapter, python_bin=python_bin) + [
            _phase(
                "adapt_summary_packet",
                adapter_command,
                preflight_argv=adapter_command + ["--validate-only"],
                required_inputs=_adapter_required_inputs(adapter, summary_source),
                expected_outputs=[str(adapter.get("output_packet_json", ""))],
                gate="summary source must exist and adapter must verify the packet",
            )
        ]
        artifacts["summary_source"] = summary_source
    elif runner_kind == "external_harness":
        command_json = str(runner.get("runner_command_json", runner.get("command_json", ""))).strip()
        summary_source = str(runner.get("summary_source", adapter.get("summary_json", "")))
        required_inputs = [command_json] if command_json else [summary_source]
        adapter_command = _adapter_command(benchmark=benchmark, adapter=adapter, runner=runner)
        phases = _adapter_spec_validation_phases(benchmark=benchmark, adapter=adapter, python_bin=python_bin) + [
            _phase(
                "run_external_harness_and_adapt",
                adapter_command,
                required_inputs=_adapter_required_inputs(adapter, *required_inputs),
                expected_outputs=[str(adapter.get("output_packet_json", ""))],
                gate="external harness command must succeed and adapter must verify the packet",
            )
        ]
        artifacts["summary_source"] = summary_source
    elif runner_kind == "swebench_harness":
        predictions_path = str(runner.get("predictions_path", "")).strip()
        results_json = str(runner.get("results_json", "")).strip()
        summary_json = str(adapter.get("summary_json", "")).strip()
        output_packet_json = str(adapter.get("output_packet_json", "")).strip()
        run_id = str(runner.get("run_id", "")).strip()
        namespace = str(runner.get("namespace", "swebench")).strip() or "swebench"
        report_dir = str(runner.get("report_dir", "")).strip()
        phases = _adapter_spec_validation_phases(benchmark=benchmark, adapter=adapter, python_bin=python_bin) + [
            _phase(
                "validate_predictions",
                [
                    python_bin,
                    "scripts/prepare_swe_bench_predictions.py",
                    "validate",
                    "--predictions-jsonl",
                    predictions_path,
                ],
                required_inputs=[predictions_path],
                gate="predictions JSONL must be structurally valid before official evaluation",
            ),
            _phase(
                "official_harness",
                build_swe_bench_command(
                    python_bin=python_bin,
                    dataset_name=str(runner.get("dataset_name", "")).strip(),
                    split=str(runner.get("split", "test")).strip() or "test",
                    predictions_path=predictions_path,
                    run_id=run_id,
                    max_workers=int(runner.get("max_workers", 1)),
                    timeout=int(runner.get("timeout", 1800)),
                    cache_level=str(runner.get("cache_level", "env")).strip() or "env",
                    namespace=namespace,
                    report_dir=report_dir,
                    instance_ids=list(runner.get("instance_ids", [])) if isinstance(runner.get("instance_ids"), list) else None,
                ),
                gate="official SWE-bench harness must complete",
            ),
            _phase(
                "materialize_results",
                [
                    python_bin,
                    "scripts/prepare_swe_bench_a8_run.py",
                    "materialize-results",
                    "--run-id",
                    run_id,
                    "--namespace",
                    namespace,
                    "--report-dir",
                    report_dir,
                    "--output-results-json",
                    results_json,
                ],
                expected_outputs=[results_json],
            ),
            _phase(
                "summarize_results",
                [
                    python_bin,
                    "scripts/prepare_swe_bench_a8_run.py",
                    "summarize",
                    "--results-json",
                    results_json,
                    "--output-summary-json",
                    summary_json,
                ],
                required_inputs=[results_json],
                expected_outputs=[summary_json],
            ),
            _phase(
                "adapt_a8_packet",
                _adapter_command(benchmark=benchmark, adapter=adapter),
                preflight_argv=_adapter_command(benchmark=benchmark, adapter=adapter) + ["--validate-only"],
                required_inputs=_adapter_required_inputs(adapter, summary_json),
                expected_outputs=[output_packet_json],
                gate="adapter verifies a8_benchmark_result packet",
            ),
        ]
        artifacts.update(
            {
                "predictions_jsonl": predictions_path,
                "report_dir": report_dir,
                "results_json": results_json,
            }
        )
        open_limits.insert(0, "This generic SWE harness assumes predictions already exist; use the SWE patch-generation harness to create them autonomously.")
    elif runner_kind in {"swebench_autonomous_queue", "swebench_live_autonomous_queue"}:
        artifact_root = _runner_text(runner, "artifact_root")
        run_id = _runner_text(runner, "run_id")
        if not run_id:
            raise ValueError(f"runner.run_id is required for {runner_kind}")
        dataset_json = _runner_text(runner, "dataset_json")
        repo_cache_root = _runner_text(runner, "repo_cache_root")
        if not dataset_json:
            raise ValueError(f"runner.dataset_json is required for {runner_kind}")
        if not repo_cache_root:
            raise ValueError(f"runner.repo_cache_root is required for {runner_kind}")
        prediction_task_manifest = _swe_autonomous_path(
            runner,
            "prediction_task_manifest",
            artifact_root,
            f"prediction_tasks_{run_id}.json",
        )
        patch_dir = _swe_autonomous_path(runner, "patch_dir", artifact_root, f"patches_{run_id}")
        queue_manifest = _swe_autonomous_path(runner, "queue_manifest", artifact_root, f"queue_manifest_{run_id}.json")
        queue_root = _swe_autonomous_path(runner, "queue_root", artifact_root, f"queue_{run_id}")
        workspace_root = _runner_text(runner, "workspace_root", str(Path(artifact_root) / "workspaces"))
        workspace_prefix = _runner_text(runner, "workspace_prefix", f"swe_patch_{run_id}")
        predictions_jsonl = _swe_autonomous_path(runner, "predictions_path", artifact_root, f"predictions_{run_id}.jsonl")
        apply_check_json = _swe_autonomous_path(
            runner,
            "apply_check_json",
            artifact_root,
            f"predictions_{run_id}_apply_check.json",
        )
        run_spec_json = _swe_autonomous_path(runner, "generated_run_spec_json", artifact_root, f"a8_run_spec_{run_id}.json")
        results_json = _runner_text(runner, "results_json")
        if not results_json:
            results_json = str(Path(_runner_text(runner, "report_dir", artifact_root)) / "results.json")
        summary_json = str(adapter.get("summary_json", "")).strip()
        output_packet_json = str(adapter.get("output_packet_json", "")).strip()
        if not summary_json:
            raise ValueError(f"adapter.summary_json is required for {runner_kind}")
        if not output_packet_json:
            raise ValueError(f"adapter.output_packet_json is required for {runner_kind}")
        live_predictions_json = _swe_autonomous_path(
            runner,
            "predictions_patch_json",
            artifact_root,
            f"preds_{run_id}.json",
        ) if runner_kind == "swebench_live_autonomous_queue" else ""
        swe_harness = build_swe_autonomous_harness_spec(
            benchmark=benchmark,
            dataset_json=dataset_json,
            dataset_name=_runner_text(runner, "dataset_name"),
            split=_runner_text(runner, "split", "test"),
            repo_cache_root=repo_cache_root,
            prediction_task_manifest=prediction_task_manifest,
            patch_dir=patch_dir,
            queue_manifest=queue_manifest,
            queue_root=queue_root,
            workspace_root=workspace_root,
            workspace_prefix=workspace_prefix,
            predictions_jsonl=predictions_jsonl,
            apply_check_json=apply_check_json,
            run_spec_json=run_spec_json,
            run_id=run_id,
            harness_root=_runner_text(runner, "harness_root", "/data/agiattempt/agi_dw/third_party/swe-bench"),
            max_workers=_runner_int(runner, "max_workers", 1),
            timeout=_runner_int(runner, "timeout", 1800),
            cache_level=_runner_text(runner, "cache_level", "env"),
            namespace=_runner_text(runner, "namespace", "swebench"),
            report_dir=_runner_text(runner, "report_dir", str(Path(artifact_root) / "evaluation_results")),
            results_json=results_json,
            summary_json=summary_json,
            output_packet_json=output_packet_json,
            python_bin=python_bin,
            model_name_or_path=_runner_text(runner, "model_name_or_path", "agentkernel"),
            provider=_runner_text(runner, "provider", "vllm"),
            drain_limit=_runner_int(runner, "drain_limit", _runner_int(runner, "limit", 0)),
            max_source_context_bytes=_runner_int(runner, "max_source_context_bytes", 30000),
            instance_ids=list(runner.get("instance_ids", [])) if isinstance(runner.get("instance_ids"), list) else None,
            limit=_runner_int(runner, "limit", 0),
            prepare_repo_cache=bool(runner.get("prepare_repo_cache", False)),
            repo_cache_manifest_json=_runner_text(runner, "repo_cache_manifest_json"),
            fetch_repo_cache=bool(runner.get("fetch_repo_cache", False)),
            official_harness_kind=(
                "swebench_live" if runner_kind == "swebench_live_autonomous_queue" else "swebench"
            ),
            live_predictions_json=live_predictions_json,
            live_platform=_runner_text(runner, "platform", "linux"),
            live_overwrite=_runner_int(runner, "overwrite", 0),
            live_start_month=_runner_text(runner, "start_month"),
            live_end_month=_runner_text(runner, "end_month"),
            live_submission_dir=_runner_text(runner, "submission_dir"),
            live_submission_subset=_runner_text(runner, "submission_subset", _runner_text(runner, "split", "verified")),
            live_system_name=_runner_text(runner, "system_name", "Agent Kernel"),
        )
        _attach_adapter_spec_to_swe_autonomous_harness(
            swe_harness,
            benchmark=benchmark,
            adapter=adapter,
            python_bin=python_bin,
        )
        swe_harness["source_run_spec"] = spec_path
        swe_harness["prerequisites"] = prerequisites
        swe_harness.setdefault("artifacts", {})["run_spec_json"] = spec_path
        return swe_harness
    else:
        raise ValueError(f"unsupported runner.kind: {runner_kind}")
    return {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_benchmark_harness_spec",
        "benchmark": benchmark,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_spec": spec_path,
        "prerequisites": prerequisites,
        "autonomy_contract": {
            "operator_role": "launch_and_monitor_only",
            "selection_mode": "run_spec_defined",
            "kernel_owned_phases": [phase["name"] for phase in phases],
            "prohibited_manual_interventions": [
                "manual benchmark summary editing",
                "manual prediction JSONL editing",
                "manual result packet editing",
                "manual generated adapter spec editing after harness launch",
                "claiming harness readiness as benchmark performance",
            ],
            "countable_evidence": [
                "verified a8_benchmark_result packet",
            ],
        },
        "artifacts": artifacts,
        "phases": phases,
        "open_limits": open_limits,
    }


def adapter_spec_from_run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("report_kind") not in {"a8_benchmark_run_spec", "standalone_benchmark_run_spec"}:
        raise ValueError("run spec report_kind must be a8_benchmark_run_spec or standalone_benchmark_run_spec")
    benchmark = str(spec.get("benchmark", "")).strip()
    if not benchmark:
        raise ValueError("run spec benchmark is required")
    adapter = spec.get("adapter") if isinstance(spec.get("adapter"), dict) else {}
    inline_spec = adapter.get("adapter_spec") if isinstance(adapter.get("adapter_spec"), dict) else None
    if inline_spec is None:
        return build_default_adapter_spec(benchmark)
    generated = deepcopy(inline_spec)
    generated.setdefault("spec_version", "asi_v1")
    generated.setdefault("report_kind", "a8_benchmark_adapter_spec")
    generated.setdefault("created_at", datetime.now(UTC).isoformat())
    generated.setdefault("benchmark", benchmark)
    generated.setdefault(
        "open_limits",
        [
            "This adapter spec maps an already-produced benchmark summary into an A8 result packet.",
            "It does not execute the benchmark and does not make summary values trustworthy by itself.",
        ],
    )
    return generated


def _run_spec_with_materialized_adapter_spec(
    spec: dict[str, Any],
    *,
    spec_stem: str,
    adapter_spec_dir: Path,
) -> tuple[dict[str, Any], Path]:
    materialized = deepcopy(spec)
    adapter_spec_dir.mkdir(parents=True, exist_ok=True)
    adapter_spec_path = adapter_spec_dir / f"{spec_stem}_adapter_spec.json"
    _write_json(adapter_spec_path, adapter_spec_from_run_spec(materialized))
    adapter = dict(materialized.get("adapter") if isinstance(materialized.get("adapter"), dict) else {})
    adapter["adapter_spec_json"] = str(adapter_spec_path)
    adapter.pop("adapter_spec", None)
    materialized["adapter"] = adapter
    return materialized, adapter_spec_path


def build_harnesses_from_run_spec_dir(
    run_spec_dir: Path,
    *,
    output_dir: Path,
    adapter_spec_dir: Path | None = None,
    python_bin: str = sys.executable,
) -> list[Path]:
    if not run_spec_dir.exists() or not run_spec_dir.is_dir():
        raise FileNotFoundError(f"run spec dir does not exist: {run_spec_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec_path in sorted(run_spec_dir.glob("*.json")):
        run_spec = _read_json(spec_path)
        if adapter_spec_dir is not None:
            run_spec, _ = _run_spec_with_materialized_adapter_spec(
                run_spec,
                spec_stem=spec_path.stem,
                adapter_spec_dir=adapter_spec_dir,
            )
        harness = build_autonomous_harness_from_run_spec(
            run_spec,
            spec_path=str(spec_path),
            python_bin=python_bin,
        )
        output_path = output_dir / f"{spec_path.stem}_harness.json"
        _write_json(output_path, harness)
        written.append(output_path)
    if not written:
        raise ValueError(f"no run spec JSON files found in {run_spec_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec-json", default="")
    parser.add_argument("--output-harness-json", default="")
    parser.add_argument("--run-spec-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--adapter-spec-dir", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()
    if args.run_spec_dir or args.output_dir:
        if not args.run_spec_dir or not args.output_dir:
            raise SystemExit("--run-spec-dir and --output-dir must be provided together")
        written = build_harnesses_from_run_spec_dir(
            Path(args.run_spec_dir),
            output_dir=Path(args.output_dir),
            adapter_spec_dir=Path(args.adapter_spec_dir) if args.adapter_spec_dir else None,
            python_bin=args.python_bin,
        )
        print(f"generated_harness_count={len(written)} output_dir={args.output_dir}")
        for path in written:
            print(f"harness_json={path}")
        return
    if not args.run_spec_json or not args.output_harness_json:
        raise SystemExit("--run-spec-json and --output-harness-json are required for single-spec mode")
    spec_path = Path(args.run_spec_json)
    run_spec = _read_json(spec_path)
    if args.adapter_spec_dir:
        run_spec, adapter_spec_path = _run_spec_with_materialized_adapter_spec(
            run_spec,
            spec_stem=spec_path.stem,
            adapter_spec_dir=Path(args.adapter_spec_dir),
        )
        print(f"adapter_spec_json={adapter_spec_path}")
    harness = build_autonomous_harness_from_run_spec(
        run_spec,
        spec_path=str(spec_path),
        python_bin=args.python_bin,
    )
    _write_json(Path(args.output_harness_json), harness)
    print(
        f"benchmark={harness['benchmark']} "
        f"phase_count={len(harness['phases'])} "
        f"harness_json={args.output_harness_json}"
    )


if __name__ == "__main__":
    main()
