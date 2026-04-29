from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import shlex
import shutil
import sys
from typing import Any


DEFAULT_DATASETS = {
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified",
    "swe_rebench": "",
    "swe_bench_live": "SWE-bench-Live/SWE-bench-Live",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_label(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "retry"


def build_swe_bench_command(
    *,
    python_bin: str,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str,
    report_dir: str,
    instance_ids: list[str] | None = None,
) -> list[str]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if cache_level not in {"none", "base", "env", "instance"}:
        raise ValueError("cache_level must be one of none, base, env, instance")
    command = [
        python_bin,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--cache_level",
        cache_level,
        "--run_id",
        run_id,
        "--namespace",
        namespace,
        "--report_dir",
        report_dir,
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def build_swe_bench_live_command(
    *,
    python_bin: str,
    dataset_name: str,
    split: str,
    predictions_json: str,
    platform: str,
    output_dir: str,
    workers: int,
    overwrite: int,
    instance_ids: list[str] | None = None,
    start_month: str = "",
    end_month: str = "",
) -> list[str]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if platform not in {"linux", "windows"}:
        raise ValueError("platform must be linux or windows")
    if overwrite not in {0, 1}:
        raise ValueError("overwrite must be 0 or 1")
    command = [
        python_bin,
        "-m",
        "evaluation.evaluation",
        "--dataset",
        dataset_name,
        "--split",
        split,
        "--platform",
        platform,
        "--patch_dir",
        predictions_json,
        "--output_dir",
        output_dir,
        "--workers",
        str(workers),
        "--overwrite",
        str(overwrite),
    ]
    if start_month:
        command.extend(["--start-month", start_month])
    if end_month:
        command.extend(["--end-month", end_month])
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def _count_from_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list | tuple | set):
            return len(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value.strip()))
            except ValueError:
                continue
    return None


def summarize_swe_bench_results(results: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
    resolved = _count_from_keys(
        results,
        (
            "resolved_count",
            "instances_resolved",
            "num_resolved",
            "resolved",
            "resolved_ids",
            "resolved_instances",
            "success",
            "success_ids",
        ),
    )
    total = _count_from_keys(
        results,
        (
            "total",
            "total_instances",
            "instance_count",
            "instances_submitted",
            "submitted_instances",
            "submitted_ids",
            "submitted",
            "all_instances",
        ),
    )
    if total is None:
        unresolved = _count_from_keys(results, ("unresolved", "unresolved_ids", "unresolved_instances"))
        errors = _count_from_keys(results, ("errors", "error_ids", "error_instances"))
        if resolved is not None and unresolved is not None:
            total = resolved + unresolved + int(errors or 0)
    if resolved is None:
        raise ValueError("SWE-bench results must include resolved count or resolved ID list")
    if total is None or total <= 0:
        raise ValueError("SWE-bench results must include positive total/submitted count")
    if resolved < 0 or resolved > total:
        raise ValueError("resolved count must be between zero and total")
    return {
        "report_kind": "official_swe_bench_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "resolved_count": int(resolved),
        "task_count": int(total),
        "resolve_rate": round(float(resolved) / float(total), 6),
    }


def build_a8_swe_benchmark_run_spec(
    *,
    benchmark: str,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str,
    harness_root: str,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str,
    report_dir: str,
    results_json: str,
    summary_json: str,
    output_packet_json: str,
    ready_to_run: bool | None = None,
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    if benchmark not in DEFAULT_DATASETS:
        raise ValueError("benchmark must be one of " + ",".join(DEFAULT_DATASETS))
    resolved_dataset = str(dataset_name or DEFAULT_DATASETS[benchmark]).strip()
    if not resolved_dataset:
        raise ValueError("dataset_name is required")
    path = Path(predictions_path)
    computed_ready = path.exists() and path.is_file() if ready_to_run is None else bool(ready_to_run)
    open_limits = [
        "This spec is execution readiness only; it is not A8 evidence until the official SWE harness runs.",
        "Do not claim SWE-bench Verified from local apply-check or sparse pytest smoke tests.",
    ]
    if not computed_ready:
        open_limits.insert(0, "ready_to_run remains false until predictions_path exists and is intentionally selected.")
    runner = {
        "kind": "swebench_harness",
        "harness_root": harness_root,
        "dataset_name": resolved_dataset,
        "split": split,
        "predictions_path": predictions_path,
        "run_id": run_id,
        "max_workers": max_workers,
        "timeout": timeout,
        "cache_level": cache_level,
        "namespace": namespace,
        "report_dir": report_dir,
        "results_json": results_json,
    }
    selected_ids = [value.strip() for value in (instance_ids or []) if value.strip()]
    if selected_ids:
        runner["instance_ids"] = selected_ids
    return {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": benchmark,
        "ready_to_run": computed_ready,
        "runner": runner,
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": summary_json,
            "output_packet_json": output_packet_json,
            "conservative_comparison_report": True,
        },
        "open_limits": open_limits,
    }


def _phase(
    name: str,
    argv: list[str],
    *,
    cwd: str = "",
    env: dict[str, str] | None = None,
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
    if cwd:
        phase["cwd"] = cwd
    if env:
        phase["env"] = env
    if preflight_argv:
        phase["preflight_argv"] = preflight_argv
    if required_inputs:
        phase["required_inputs"] = required_inputs
    if expected_outputs:
        phase["expected_outputs"] = expected_outputs
    if gate:
        phase["gate"] = gate
    return phase


def _queue_env(queue_root: str) -> dict[str, str]:
    root = Path(queue_root)
    return {
        "AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH": str(root / "queue.json"),
        "AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH": str(root / "runtime_state.json"),
        "AGENT_KERNEL_RUNTIME_DATABASE_PATH": str(root / "agentkernel.sqlite3"),
        "AGENT_KERNEL_RUN_REPORTS_DIR": str(root / "reports"),
        "AGENT_KERNEL_RUN_CHECKPOINTS_DIR": str(root / "checkpoints"),
    }


def _queue_max_queued_budget(*, selected_ids: list[str], limit: int, drain_limit: int) -> int:
    """Return a queue budget that can admit the intended harness slice."""
    if not selected_ids and limit <= 0:
        return 0
    intended_jobs = len(selected_ids) if selected_ids else max(0, limit)
    return max(1, intended_jobs, max(0, drain_limit))


def build_swe_autonomous_harness_spec(
    *,
    benchmark: str,
    dataset_json: str,
    dataset_name: str,
    split: str,
    repo_cache_root: str,
    prediction_task_manifest: str,
    patch_dir: str,
    queue_manifest: str,
    queue_root: str,
    workspace_root: str,
    workspace_prefix: str,
    predictions_jsonl: str,
    apply_check_json: str,
    run_spec_json: str,
    run_id: str,
    harness_root: str,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str,
    report_dir: str,
    results_json: str,
    summary_json: str,
    output_packet_json: str,
    python_bin: str,
    model_name_or_path: str,
    provider: str,
    drain_limit: int,
    max_source_context_bytes: int,
    instance_ids: list[str] | None = None,
    limit: int = 0,
    prepare_repo_cache: bool = False,
    repo_cache_manifest_json: str = "",
    fetch_repo_cache: bool = False,
    official_harness_kind: str = "swebench",
    live_predictions_json: str = "",
    live_platform: str = "linux",
    live_overwrite: int = 0,
    live_start_month: str = "",
    live_end_month: str = "",
    live_submission_dir: str = "",
    live_submission_subset: str = "",
    live_system_name: str = "Agent Kernel",
) -> dict[str, Any]:
    if benchmark not in DEFAULT_DATASETS:
        raise ValueError("benchmark must be one of " + ",".join(DEFAULT_DATASETS))
    if official_harness_kind not in {"swebench", "swebench_live"}:
        raise ValueError("official_harness_kind must be swebench or swebench_live")
    resolved_dataset = str(dataset_name or DEFAULT_DATASETS[benchmark]).strip()
    if not resolved_dataset:
        raise ValueError("dataset_name is required")
    is_live = official_harness_kind == "swebench_live"
    selected_ids = [value.strip() for value in (instance_ids or []) if value.strip()]
    if selected_ids:
        selection_mode = "operator_selected_instance_ids"
    elif limit > 0:
        selection_mode = "dataset_limit"
    else:
        selection_mode = "full_split"
    prediction_task_progress_json = str(
        Path(prediction_task_manifest).with_name(f"prediction_task_progress_{run_id}.json")
    )

    prepare_args = [
        python_bin,
        "scripts/prepare_swe_bench_prediction_tasks.py",
        "--dataset-json",
        dataset_json,
        "--output-manifest-json",
        prediction_task_manifest,
        "--output-patch-dir",
        patch_dir,
        "--model-name-or-path",
        model_name_or_path,
        "--repo-cache-root",
        repo_cache_root,
        "--max-source-context-bytes",
        str(max_source_context_bytes),
        "--progress-json",
        prediction_task_progress_json,
    ]
    if selected_ids:
        prepare_args.extend(["--instance-ids", *selected_ids])
    if limit > 0:
        prepare_args.extend(["--limit", str(limit)])
    repo_cache_manifest_json = str(repo_cache_manifest_json).strip()
    if prepare_repo_cache and not repo_cache_manifest_json:
        repo_cache_manifest_json = str(Path(prediction_task_manifest).with_name(f"repo_cache_{run_id}.json"))
    repo_cache_args = [
        python_bin,
        "scripts/prepare_swe_bench_repo_cache.py",
        "--dataset-json",
        dataset_json,
        "--repo-cache-root",
        repo_cache_root,
        "--output-json",
        repo_cache_manifest_json,
    ]
    if fetch_repo_cache:
        repo_cache_args.append("--fetch")
    if selected_ids:
        repo_cache_args.extend(["--instance-ids", *selected_ids])
    if limit > 0:
        repo_cache_args.extend(["--limit", str(limit)])

    queue_args = [
        python_bin,
        "scripts/prepare_swe_bench_queue_manifest.py",
        "--prediction-task-manifest",
        prediction_task_manifest,
        "--output-manifest-json",
        queue_manifest,
        "--workspace-prefix",
        workspace_prefix,
    ]
    queue_env = _queue_env(queue_root)
    queue_env["AGENT_KERNEL_WORKSPACE_ROOT"] = workspace_root
    # Benchmark queues must retain every terminal task until prediction
    # collection; the default delegated-queue terminal pruning is too small
    # for full SWE-Bench Verified.
    queue_env["AGENT_KERNEL_STORAGE_KEEP_TERMINAL_JOB_RECORDS"] = "1000"
    queue_env["AGENT_KERNEL_STORAGE_PRUNE_TERMINAL_JOB_ARTIFACTS"] = "0"
    patch_jobs_verification_json = str(Path(queue_root) / "patch_jobs_verification.json")
    queue_max_queued_per_budget_group = _queue_max_queued_budget(
        selected_ids=selected_ids,
        limit=limit,
        drain_limit=drain_limit,
    )
    enqueue_args = [
        python_bin,
        "scripts/run_job_queue.py",
        "enqueue-manifest",
        "--manifest-path",
        queue_manifest,
        "--limit",
        str(limit if limit > 0 else 0),
        "--priority-start",
        "100",
        "--budget-group",
        f"{benchmark}_{run_id}",
        "--max-queued-per-budget-group",
        str(queue_max_queued_per_budget_group),
        "--skip-existing-task-ids",
        "--json",
    ]
    drain_args = [
        python_bin,
        "scripts/run_job_queue.py",
        "drain",
        "--limit",
        str(drain_limit),
        "--provider",
        provider,
        "--model",
        model_name_or_path,
        "--enforce-preflight",
        "0",
        "--use-tolbert-context",
        "0",
        "--use-graph-memory",
        "0",
        "--use-world-model",
        "0",
        "--use-retrieval-proposals",
        "0",
        "--use-trust-proposals",
        "0",
        "--asi-coding-require-live-llm",
        "1",
    ]
    verify_patch_jobs_args = [
        python_bin,
        "scripts/verify_swe_bench_patch_jobs.py",
        "--queue-json",
        queue_env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"],
        "--queue-manifest",
        queue_manifest,
        "--workspace-root",
        workspace_root,
        "--output-json",
        patch_jobs_verification_json,
    ]
    collect_args = [
        python_bin,
        "scripts/collect_swe_bench_predictions.py",
        "--prediction-task-manifest",
        prediction_task_manifest,
        "--queue-manifest",
        queue_manifest,
        "--workspace-root",
        workspace_root,
        "--output-jsonl",
        predictions_jsonl,
        "--patch-job-verification-json",
        patch_jobs_verification_json,
    ]
    apply_check_args = [
        python_bin,
        "scripts/validate_swe_bench_predictions_against_repo_cache.py",
        "--dataset-json",
        dataset_json,
        "--predictions-jsonl",
        predictions_jsonl,
        "--repo-cache-root",
        repo_cache_root,
        "--output-json",
        apply_check_json,
    ]
    if selected_ids:
        apply_check_args.extend(["--instance-ids", *selected_ids])
    run_spec_args = [
        python_bin,
        "scripts/prepare_swe_bench_a8_run.py",
        "spec",
        "--benchmark",
        benchmark,
        "--dataset-name",
        resolved_dataset,
        "--split",
        split,
        "--predictions-path",
        predictions_jsonl,
        "--run-id",
        run_id,
        "--max-workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--cache-level",
        cache_level,
        "--namespace",
        namespace,
        "--report-dir",
        report_dir,
        "--results-json",
        results_json,
        "--summary-json",
        summary_json,
        "--output-packet-json",
        output_packet_json,
        "--swe-bench-root",
        harness_root,
        "--ready-to-run",
        "auto",
        "--output-spec-json",
        run_spec_json,
    ]
    if selected_ids:
        run_spec_args.extend(["--instance-ids", *selected_ids])
    live_predictions_json = str(live_predictions_json).strip()
    if is_live and not live_predictions_json:
        live_predictions_json = str(Path(predictions_jsonl).with_suffix(".live_preds.json"))
    if is_live:
        official_args = build_swe_bench_live_command(
            python_bin=python_bin,
            dataset_name=resolved_dataset,
            split=split,
            predictions_json=live_predictions_json,
            platform=live_platform,
            output_dir=report_dir,
            workers=max_workers,
            overwrite=live_overwrite,
            instance_ids=selected_ids or None,
            start_month=live_start_month,
            end_month=live_end_month,
        )
        materialize_args: list[str] = []
    else:
        official_args = build_swe_bench_command(
            python_bin=python_bin,
            dataset_name=resolved_dataset,
            split=split,
            predictions_path=predictions_jsonl,
            run_id=run_id,
            max_workers=max_workers,
            timeout=timeout,
            cache_level=cache_level,
            namespace=namespace,
            report_dir=report_dir,
            instance_ids=selected_ids or None,
        )
        materialize_args = [
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
        ]
    summarize_args = [
        python_bin,
        "scripts/prepare_swe_bench_a8_run.py",
        "summarize",
        "--results-json",
        results_json,
        "--output-summary-json",
        summary_json,
    ]
    adapt_args = [
        python_bin,
        "scripts/run_a8_benchmark_adapter.py",
        "--benchmark",
        benchmark,
        "--summary-json",
        summary_json,
        "--output-json",
        output_packet_json,
        "--conservative-comparison-report",
    ]
    live_prediction_args = [
        python_bin,
        "scripts/prepare_swe_bench_live_submission.py",
        "preds",
        "--predictions-jsonl",
        predictions_jsonl,
        "--output-json",
        live_predictions_json,
    ]
    live_submission_subset = str(live_submission_subset).strip() or split
    if is_live and not live_submission_dir:
        live_submission_dir = str(Path(report_dir).parent / "submission" / live_submission_subset / "agentkernel")
    live_submission_args = [
        python_bin,
        "scripts/prepare_swe_bench_live_submission.py",
        "package",
        "--predictions-jsonl",
        predictions_jsonl,
        "--results-json",
        results_json,
        "--output-dir",
        live_submission_dir,
        "--model-name",
        model_name_or_path,
        "--system-name",
        live_system_name,
        "--subset",
        live_submission_subset,
        "--run-command",
        " ".join(shlex.quote(part) for part in official_args),
    ]
    open_limits = [
        "The harness is A8 evidence only after the official SWE harness completes and the adapter packet verifies.",
        "Repo-cache apply-check and patch-generation success are not benchmark scores.",
        "Do not add per-instance queue hooks or manually edit predictions after this harness starts.",
    ]
    if selection_mode != "full_split":
        open_limits.insert(0, "This is selected-slice or limited-run evidence, not full benchmark evidence.")
    if is_live:
        open_limits.append(
            "SWE-bench Live leaderboard ranking requires packaging preds.json/results.json/README.md and submitting them through the official PR flow."
        )
    artifacts = {
        "prediction_task_manifest": prediction_task_manifest,
        "prediction_task_progress_json": prediction_task_progress_json,
        "patch_dir": patch_dir,
        "queue_manifest": queue_manifest,
        "queue_root": queue_root,
        "patch_jobs_verification_json": patch_jobs_verification_json,
        "workspace_root": workspace_root,
        "workspace_prefix": workspace_prefix,
        "predictions_jsonl": predictions_jsonl,
        "apply_check_json": apply_check_json,
        "run_spec_json": run_spec_json,
        "report_dir": report_dir,
        "results_json": results_json,
        "summary_json": summary_json,
        "output_packet_json": output_packet_json,
    }
    phases = [
        *(
            [
                _phase(
                    "prepare_repo_cache",
                    repo_cache_args,
                    required_inputs=[dataset_json],
                    expected_outputs=[repo_cache_manifest_json],
                    gate="repository cache must contain the dataset repositories before prediction tasks are built",
                )
            ]
            if prepare_repo_cache
            else []
        ),
        _phase(
            "prepare_prediction_tasks",
            prepare_args,
            required_inputs=[dataset_json, repo_cache_root],
            expected_outputs=[prediction_task_manifest, prediction_task_progress_json],
        ),
        _phase(
            "prepare_queue_manifest",
            queue_args,
            required_inputs=[prediction_task_manifest],
            expected_outputs=[queue_manifest],
        ),
        _phase(
            "enqueue_patch_jobs",
            enqueue_args,
            env=queue_env,
            required_inputs=[queue_manifest],
            expected_outputs=[queue_env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]],
        ),
        _phase(
            "drain_patch_jobs",
            drain_args,
            env=queue_env,
            required_inputs=[queue_env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]],
        ),
        _phase(
            "verify_patch_jobs",
            verify_patch_jobs_args,
            required_inputs=[queue_env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"], queue_manifest],
            expected_outputs=[patch_jobs_verification_json],
            gate="nonterminal patch jobs must finish; terminal abstentions become no-op predictions",
        ),
        _phase(
            "collect_predictions",
            collect_args,
            required_inputs=[prediction_task_manifest, queue_manifest, patch_jobs_verification_json],
            expected_outputs=[predictions_jsonl],
        ),
        _phase(
            "repo_cache_apply_check",
            apply_check_args,
            required_inputs=[dataset_json, predictions_jsonl, repo_cache_root],
            expected_outputs=[apply_check_json],
            gate="all non-empty patches must apply; empty no-op predictions are valid abstentions",
        ),
    ]
    if is_live:
        artifacts["live_predictions_json"] = live_predictions_json
        artifacts["leaderboard_submission_dir"] = live_submission_dir
    if prepare_repo_cache:
        artifacts["repo_cache_manifest_json"] = repo_cache_manifest_json
    if is_live:
        phases.extend(
            [
                _phase(
                    "build_live_predictions_json",
                    live_prediction_args,
                    required_inputs=[predictions_jsonl],
                    expected_outputs=[live_predictions_json],
                    gate="SWE-bench Live predictions must use the official instance-id keyed object format",
                ),
                _phase(
                    "official_harness",
                    official_args,
                    cwd=harness_root,
                    required_inputs=[live_predictions_json],
                    expected_outputs=[results_json],
                    gate="official SWE-bench Live evaluator must complete",
                ),
                _phase("summarize_results", summarize_args, required_inputs=[results_json], expected_outputs=[summary_json]),
                _phase(
                    "adapt_a8_packet",
                    adapt_args,
                    preflight_argv=adapt_args + ["--validate-only"],
                    required_inputs=[summary_json],
                    expected_outputs=[output_packet_json],
                    gate="adapter verifies a8_benchmark_result packet",
                ),
                _phase(
                    "package_live_leaderboard_submission",
                    live_submission_args,
                    required_inputs=[predictions_jsonl, results_json],
                    expected_outputs=[
                        str(Path(live_submission_dir) / "preds.json"),
                        str(Path(live_submission_dir) / "results.json"),
                        str(Path(live_submission_dir) / "README.md"),
                    ],
                    gate="leaderboard submission package must include predictions, results, and system README",
                ),
            ]
        )
    else:
        phases.extend(
            [
                _phase("build_run_spec", run_spec_args, required_inputs=[predictions_jsonl], expected_outputs=[run_spec_json]),
                _phase(
                    "official_harness",
                    official_args,
                    required_inputs=[predictions_jsonl],
                    gate="official SWE-bench harness must complete",
                ),
                _phase("materialize_results", materialize_args, expected_outputs=[results_json]),
                _phase("summarize_results", summarize_args, required_inputs=[results_json], expected_outputs=[summary_json]),
                _phase(
                    "adapt_a8_packet",
                    adapt_args,
                    preflight_argv=adapt_args + ["--validate-only"],
                    required_inputs=[summary_json],
                    expected_outputs=[output_packet_json],
                    gate="adapter verifies a8_benchmark_result packet",
                ),
            ]
        )
    return {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_benchmark_harness_spec",
        "benchmark": benchmark,
        "created_at": datetime.now(UTC).isoformat(),
        "autonomy_contract": {
            "operator_role": "launch_and_monitor_only",
            "selection_mode": selection_mode,
            "kernel_owned_phases": [phase["name"] for phase in phases],
            "prohibited_manual_interventions": [
                "manual per-instance patch authoring",
                "manual prediction JSONL editing",
                "new per-instance queue hook edits during the run",
                "claiming local apply-checks as official benchmark results",
            ],
            "countable_evidence": [
                "official SWE harness resolved_instances/resolved_ids",
                "verified a8_benchmark_result packet",
                *(
                    ["SWE-bench Live leaderboard submission package"]
                    if is_live
                    else []
                ),
            ],
        },
        "inputs": {
            "dataset_json": dataset_json,
            "dataset_name": resolved_dataset,
            "split": split,
            "repo_cache_root": repo_cache_root,
            "instance_ids": selected_ids,
            "limit": limit,
            "max_source_context_bytes": max_source_context_bytes,
        },
        "run_config": {
            "run_id": run_id,
            "harness_root": harness_root,
            "max_workers": max_workers,
            "timeout": timeout,
            "cache_level": cache_level,
            "namespace": namespace,
            "python_bin": python_bin,
            "model_name_or_path": model_name_or_path,
            "provider": provider,
            "drain_limit": drain_limit,
            "queue_max_queued_per_budget_group": queue_max_queued_per_budget_group,
            "official_harness_kind": official_harness_kind,
            "live_platform": live_platform,
            "live_overwrite": live_overwrite,
            "live_start_month": live_start_month,
            "live_end_month": live_end_month,
            "live_submission_subset": live_submission_subset,
        },
        "artifacts": artifacts,
        "phases": phases,
        "open_limits": open_limits,
    }


def build_swe_retry_harness_spec(
    *,
    source_harness: dict[str, Any],
    patch_job_verification: dict[str, Any],
    retry_label: str,
    artifact_dir: Path,
    output_harness_json: str,
    run_id: str = "",
) -> dict[str, Any]:
    if source_harness.get("report_kind") != "autonomous_benchmark_harness_spec":
        raise ValueError("source harness report_kind must be autonomous_benchmark_harness_spec")
    retry_instance_ids = [
        str(value).strip()
        for value in patch_job_verification.get("retry_instance_ids", [])
        if str(value).strip()
    ]
    if not retry_instance_ids:
        raise ValueError("patch job verification does not contain retry_instance_ids")
    inputs = source_harness.get("inputs") if isinstance(source_harness.get("inputs"), dict) else {}
    run_config = source_harness.get("run_config") if isinstance(source_harness.get("run_config"), dict) else {}
    artifacts = source_harness.get("artifacts") if isinstance(source_harness.get("artifacts"), dict) else {}
    label = _safe_label(retry_label)
    benchmark = str(source_harness.get("benchmark", "")).strip()
    if benchmark not in DEFAULT_DATASETS:
        raise ValueError("source harness benchmark must be one of " + ",".join(DEFAULT_DATASETS))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workspace_prefix = str(artifacts.get("workspace_prefix", "swe_bench_retry")).strip().strip("/") or "swe_bench_retry"
    resolved_run_id = str(run_id).strip() or f"{str(run_config.get('run_id', 'swe_retry')).strip() or 'swe_retry'}_{label}"
    spec = build_swe_autonomous_harness_spec(
        benchmark=benchmark,
        dataset_json=str(inputs.get("dataset_json", "")).strip(),
        dataset_name=str(inputs.get("dataset_name", DEFAULT_DATASETS[benchmark])).strip(),
        split=str(inputs.get("split", "test")).strip() or "test",
        repo_cache_root=str(inputs.get("repo_cache_root", "")).strip(),
        prediction_task_manifest=str(artifact_dir / f"prediction_tasks_{label}.json"),
        patch_dir=str(artifact_dir / f"patches_{label}"),
        queue_manifest=str(artifact_dir / f"queue_manifest_{label}.json"),
        queue_root=str(artifact_dir / f"queue_{label}"),
        workspace_root=str(artifacts.get("workspace_root", "workspace")).strip() or "workspace",
        workspace_prefix=f"{workspace_prefix}_{label}",
        predictions_jsonl=str(artifact_dir / f"predictions_{label}.jsonl"),
        apply_check_json=str(artifact_dir / f"predictions_{label}_apply_check.json"),
        run_spec_json=str(artifact_dir / f"a8_run_spec_{label}.json"),
        run_id=resolved_run_id,
        harness_root=str(run_config.get("harness_root", "/data/agiattempt/agi_dw/third_party/swe-bench")).strip()
        or "/data/agiattempt/agi_dw/third_party/swe-bench",
        max_workers=int(run_config.get("max_workers", 1) or 1),
        timeout=int(run_config.get("timeout", 1800) or 1800),
        cache_level=str(run_config.get("cache_level", "env")).strip() or "env",
        namespace=str(run_config.get("namespace", "swebench")).strip() or "swebench",
        report_dir=str(artifact_dir / f"evaluation_results_{label}"),
        results_json=str(artifact_dir / f"evaluation_results_{label}" / "results.json"),
        summary_json=str(artifact_dir / f"evaluation_results_{label}" / "summary.json"),
        output_packet_json=str(artifact_dir / f"evaluation_results_{label}" / "a8_benchmark_result.json"),
        python_bin=str(run_config.get("python_bin", sys.executable)).strip() or sys.executable,
        model_name_or_path=str(run_config.get("model_name_or_path", "agentkernel")).strip() or "agentkernel",
        provider=str(run_config.get("provider", "vllm")).strip() or "vllm",
        drain_limit=len(retry_instance_ids),
        max_source_context_bytes=int(inputs.get("max_source_context_bytes", 30000) or 30000),
        instance_ids=retry_instance_ids,
        limit=0,
    )
    spec["source_retry"] = {
        "source_harness_json": str(source_harness.get("source_harness_json", "")),
        "patch_job_verification_json": str(patch_job_verification.get("patch_job_verification_json", "")),
        "retry_label": label,
        "retry_instance_ids": retry_instance_ids,
        "output_harness_json": output_harness_json,
    }
    spec["open_limits"].insert(0, "This retry harness was generated from failed SWE patch-job verifier output; it retries only retry_instance_ids.")
    return spec


def build_swe_success_continuation_harness_spec(
    *,
    source_harness: dict[str, Any],
    patch_job_verification: dict[str, Any],
    success_label: str,
    artifact_dir: Path,
    output_harness_json: str,
    run_id: str = "",
) -> dict[str, Any]:
    if source_harness.get("report_kind") != "autonomous_benchmark_harness_spec":
        raise ValueError("source harness report_kind must be autonomous_benchmark_harness_spec")
    successful_instance_ids = [
        str(value).strip()
        for value in patch_job_verification.get("successful_instance_ids", [])
        if str(value).strip()
    ]
    if not successful_instance_ids:
        raise ValueError("patch job verification does not contain successful_instance_ids")
    inputs = source_harness.get("inputs") if isinstance(source_harness.get("inputs"), dict) else {}
    run_config = source_harness.get("run_config") if isinstance(source_harness.get("run_config"), dict) else {}
    source_artifacts = source_harness.get("artifacts") if isinstance(source_harness.get("artifacts"), dict) else {}
    label = _safe_label(success_label)
    benchmark = str(source_harness.get("benchmark", "")).strip()
    if benchmark not in DEFAULT_DATASETS:
        raise ValueError("source harness benchmark must be one of " + ",".join(DEFAULT_DATASETS))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_dataset = str(inputs.get("dataset_name", DEFAULT_DATASETS[benchmark])).strip()
    resolved_run_id = str(run_id).strip() or f"{str(run_config.get('run_id', 'swe_success')).strip() or 'swe_success'}_{label}"
    python_bin = str(run_config.get("python_bin", sys.executable)).strip() or sys.executable
    namespace = str(run_config.get("namespace", "swebench")).strip() or "swebench"
    timeout = int(run_config.get("timeout", 1800) or 1800)
    max_workers = int(run_config.get("max_workers", 1) or 1)
    cache_level = str(run_config.get("cache_level", "env")).strip() or "env"
    harness_root = str(run_config.get("harness_root", "/data/agiattempt/agi_dw/third_party/swe-bench")).strip()
    if not harness_root:
        harness_root = "/data/agiattempt/agi_dw/third_party/swe-bench"

    predictions_jsonl = str(artifact_dir / f"predictions_{label}.jsonl")
    apply_check_json = str(artifact_dir / f"predictions_{label}_apply_check.json")
    run_spec_json = str(artifact_dir / f"a8_run_spec_{label}.json")
    report_dir = str(artifact_dir / f"evaluation_results_{label}")
    results_json = str(artifact_dir / f"evaluation_results_{label}" / "results.json")
    summary_json = str(artifact_dir / f"evaluation_results_{label}" / "summary.json")
    output_packet_json = str(artifact_dir / f"evaluation_results_{label}" / "a8_benchmark_result.json")
    verification_json = str(patch_job_verification.get("patch_job_verification_json", "")).strip()
    if not verification_json:
        raise ValueError("patch_job_verification_json is required for success continuation harness")

    collect_args = [
        python_bin,
        "scripts/collect_swe_bench_predictions.py",
        "--prediction-task-manifest",
        str(source_artifacts.get("prediction_task_manifest", "")).strip(),
        "--queue-manifest",
        str(source_artifacts.get("queue_manifest", "")).strip(),
        "--workspace-root",
        str(source_artifacts.get("workspace_root", "workspace")).strip() or "workspace",
        "--output-jsonl",
        predictions_jsonl,
        "--patch-job-verification-json",
        verification_json,
    ]
    apply_check_args = [
        python_bin,
        "scripts/validate_swe_bench_predictions_against_repo_cache.py",
        "--dataset-json",
        str(inputs.get("dataset_json", "")).strip(),
        "--predictions-jsonl",
        predictions_jsonl,
        "--repo-cache-root",
        str(inputs.get("repo_cache_root", "")).strip(),
        "--output-json",
        apply_check_json,
        "--instance-ids",
        *successful_instance_ids,
    ]
    run_spec_args = [
        python_bin,
        "scripts/prepare_swe_bench_a8_run.py",
        "spec",
        "--benchmark",
        benchmark,
        "--dataset-name",
        resolved_dataset,
        "--split",
        str(inputs.get("split", "test")).strip() or "test",
        "--predictions-path",
        predictions_jsonl,
        "--run-id",
        resolved_run_id,
        "--max-workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--cache-level",
        cache_level,
        "--namespace",
        namespace,
        "--report-dir",
        report_dir,
        "--results-json",
        results_json,
        "--summary-json",
        summary_json,
        "--output-packet-json",
        output_packet_json,
        "--swe-bench-root",
        harness_root,
        "--ready-to-run",
        "auto",
        "--output-spec-json",
        run_spec_json,
        "--instance-ids",
        *successful_instance_ids,
    ]
    official_args = build_swe_bench_command(
        python_bin=python_bin,
        dataset_name=resolved_dataset,
        split=str(inputs.get("split", "test")).strip() or "test",
        predictions_path=predictions_jsonl,
        run_id=resolved_run_id,
        max_workers=max_workers,
        timeout=timeout,
        cache_level=cache_level,
        namespace=namespace,
        report_dir=report_dir,
        instance_ids=successful_instance_ids,
    )
    materialize_args = [
        python_bin,
        "scripts/prepare_swe_bench_a8_run.py",
        "materialize-results",
        "--run-id",
        resolved_run_id,
        "--namespace",
        namespace,
        "--report-dir",
        report_dir,
        "--output-results-json",
        results_json,
    ]
    summarize_args = [
        python_bin,
        "scripts/prepare_swe_bench_a8_run.py",
        "summarize",
        "--results-json",
        results_json,
        "--output-summary-json",
        summary_json,
    ]
    adapt_args = [
        python_bin,
        "scripts/run_a8_benchmark_adapter.py",
        "--benchmark",
        benchmark,
        "--summary-json",
        summary_json,
        "--output-json",
        output_packet_json,
        "--conservative-comparison-report",
    ]
    return {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_benchmark_harness_spec",
        "benchmark": benchmark,
        "created_at": datetime.now(UTC).isoformat(),
        "autonomy_contract": {
            "operator_role": "launch_and_monitor_only",
            "selection_mode": "verified_patch_success_continuation",
            "kernel_owned_phases": [
                "collect_predictions",
                "repo_cache_apply_check",
                "official_harness",
                "summarize_results",
                "adapt_a8_packet",
            ],
            "prohibited_manual_interventions": [
                "manual per-instance patch authoring",
                "manual prediction JSONL editing",
                "claiming local apply-checks as official benchmark results",
            ],
            "countable_evidence": [
                "official SWE harness resolved_instances/resolved_ids",
                "verified a8_benchmark_result packet",
            ],
        },
        "inputs": {
            "dataset_json": str(inputs.get("dataset_json", "")).strip(),
            "dataset_name": resolved_dataset,
            "split": str(inputs.get("split", "test")).strip() or "test",
            "repo_cache_root": str(inputs.get("repo_cache_root", "")).strip(),
            "instance_ids": successful_instance_ids,
            "source_harness_json": str(source_harness.get("source_harness_json", "")),
            "patch_job_verification_json": verification_json,
        },
        "run_config": {
            "run_id": resolved_run_id,
            "harness_root": harness_root,
            "max_workers": max_workers,
            "timeout": timeout,
            "cache_level": cache_level,
            "namespace": namespace,
            "python_bin": python_bin,
        },
        "artifacts": {
            "source_prediction_task_manifest": str(source_artifacts.get("prediction_task_manifest", "")).strip(),
            "source_queue_manifest": str(source_artifacts.get("queue_manifest", "")).strip(),
            "source_workspace_root": str(source_artifacts.get("workspace_root", "workspace")).strip() or "workspace",
            "predictions_jsonl": predictions_jsonl,
            "apply_check_json": apply_check_json,
            "run_spec_json": run_spec_json,
            "report_dir": report_dir,
            "results_json": results_json,
            "summary_json": summary_json,
            "output_packet_json": output_packet_json,
        },
        "phases": [
            _phase(
                "collect_predictions",
                collect_args,
                required_inputs=[
                    str(source_artifacts.get("prediction_task_manifest", "")).strip(),
                    str(source_artifacts.get("queue_manifest", "")).strip(),
                    verification_json,
                ],
                expected_outputs=[predictions_jsonl],
                gate="collect verifier-successful patches and no-op abstentions",
            ),
            _phase(
                "repo_cache_apply_check",
                apply_check_args,
                required_inputs=[str(inputs.get("dataset_json", "")).strip(), predictions_jsonl, str(inputs.get("repo_cache_root", "")).strip()],
                expected_outputs=[apply_check_json],
                gate="all non-empty patches must apply; empty no-op predictions are valid abstentions",
            ),
            _phase("build_run_spec", run_spec_args, required_inputs=[predictions_jsonl], expected_outputs=[run_spec_json]),
            _phase(
                "official_harness",
                official_args,
                required_inputs=[predictions_jsonl],
                gate="official SWE-bench harness must complete",
            ),
            _phase("materialize_results", materialize_args, expected_outputs=[results_json]),
            _phase("summarize_results", summarize_args, required_inputs=[results_json], expected_outputs=[summary_json]),
            _phase(
                "adapt_a8_packet",
                adapt_args,
                preflight_argv=adapt_args + ["--validate-only"],
                required_inputs=[summary_json],
                expected_outputs=[output_packet_json],
                gate="adapter verifies a8_benchmark_result packet",
            ),
        ],
        "open_limits": [
            "This continuation harness evaluates only patch jobs already verified successful by a prior harness verifier.",
            "This is selected-slice evidence, not full benchmark evidence.",
            "Do not claim patch verification or repo-cache apply-check as official benchmark success.",
        ],
        "source_success_continuation": {
            "source_harness_json": str(source_harness.get("source_harness_json", "")),
            "patch_job_verification_json": verification_json,
            "success_label": label,
            "successful_instance_ids": successful_instance_ids,
            "output_harness_json": output_harness_json,
        },
    }


def materialize_swe_bench_results(
    *,
    run_id: str,
    namespace: str,
    report_dir: str,
    output_results_json: str,
    search_root: str = ".",
) -> Path:
    filename = f"{namespace}.{run_id}.json"
    candidates = [
        Path(report_dir) / filename,
        Path(search_root) / filename,
    ]
    for root in (Path(report_dir), Path(search_root)):
        if root.exists():
            candidates.extend(sorted(root.glob(f"*.{run_id}.json")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            output_path = Path(output_results_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != output_path.resolve():
                shutil.copyfile(candidate, output_path)
            return output_path
    raise FileNotFoundError("could not locate official SWE-bench report at " + ", ".join(str(path) for path in candidates))


def _command_mode(args: argparse.Namespace) -> None:
    dataset_name = str(args.dataset_name or DEFAULT_DATASETS[args.benchmark]).strip()
    if not dataset_name:
        raise SystemExit("--dataset-name is required for this benchmark")
    instance_ids = [value.strip() for value in args.instance_ids if value.strip()] if args.instance_ids else None
    command = build_swe_bench_command(
        python_bin=str(args.python_bin),
        dataset_name=dataset_name,
        split=str(args.split),
        predictions_path=str(args.predictions_path),
        run_id=str(args.run_id),
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=str(args.cache_level),
        namespace=str(args.namespace),
        report_dir=str(args.report_dir),
        instance_ids=instance_ids,
    )
    _write_json(Path(args.output_command_json), command)
    if args.output_spec_json:
        _write_json(
            Path(args.output_spec_json),
            {
                "spec_version": "asi_v1",
                "report_kind": "a8_swe_bench_runner_spec",
                "benchmark": args.benchmark,
                "dataset_name": dataset_name,
                "split": args.split,
                "predictions_path": args.predictions_path,
                "run_id": args.run_id,
                "runner_cwd": args.swe_bench_root,
                "summary_json": args.summary_json,
                "output_packet_json": args.output_packet_json,
                "runner_command_json": args.output_command_json,
            },
        )
    print(f"benchmark={args.benchmark} command_json={args.output_command_json}")


def _spec_mode(args: argparse.Namespace) -> None:
    dataset_name = str(args.dataset_name or DEFAULT_DATASETS[args.benchmark]).strip()
    if not dataset_name:
        raise SystemExit("--dataset-name is required for this benchmark")
    ready_to_run: bool | None = None if args.ready_to_run == "auto" else args.ready_to_run == "true"
    spec = build_a8_swe_benchmark_run_spec(
        benchmark=args.benchmark,
        dataset_name=dataset_name,
        split=args.split,
        predictions_path=args.predictions_path,
        run_id=args.run_id,
        harness_root=args.swe_bench_root,
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=args.cache_level,
        namespace=args.namespace,
        report_dir=args.report_dir,
        results_json=args.results_json,
        summary_json=args.summary_json,
        output_packet_json=args.output_packet_json,
        ready_to_run=ready_to_run,
        instance_ids=args.instance_ids,
    )
    _write_json(Path(args.output_spec_json), spec)
    print(
        f"benchmark={args.benchmark} "
        f"ready_to_run={str(spec['ready_to_run']).lower()} "
        f"spec_json={args.output_spec_json}"
    )


def _summarize_mode(args: argparse.Namespace) -> None:
    results_path = Path(args.results_json)
    summary = summarize_swe_bench_results(_read_json(results_path), source_path=str(results_path))
    _write_json(Path(args.output_summary_json), summary)
    print(
        f"resolved_count={summary['resolved_count']} "
        f"task_count={summary['task_count']} "
        f"resolve_rate={summary['resolve_rate']} "
        f"summary_json={args.output_summary_json}"
    )


def _materialize_results_mode(args: argparse.Namespace) -> None:
    output_path = materialize_swe_bench_results(
        run_id=args.run_id,
        namespace=args.namespace,
        report_dir=args.report_dir,
        output_results_json=args.output_results_json,
        search_root=args.search_root,
    )
    print(f"results_json={output_path}")


def _harness_mode(args: argparse.Namespace) -> None:
    spec = build_swe_autonomous_harness_spec(
        benchmark=args.benchmark,
        dataset_json=args.dataset_json,
        dataset_name=args.dataset_name,
        split=args.split,
        repo_cache_root=args.repo_cache_root,
        prediction_task_manifest=args.prediction_task_manifest,
        patch_dir=args.patch_dir,
        queue_manifest=args.queue_manifest,
        queue_root=args.queue_root,
        workspace_root=args.workspace_root,
        workspace_prefix=args.workspace_prefix,
        predictions_jsonl=args.predictions_jsonl,
        apply_check_json=args.apply_check_json,
        run_spec_json=args.run_spec_json,
        run_id=args.run_id,
        harness_root=args.swe_bench_root,
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=args.cache_level,
        namespace=args.namespace,
        report_dir=args.report_dir,
        results_json=args.results_json,
        summary_json=args.summary_json,
        output_packet_json=args.output_packet_json,
        python_bin=args.python_bin,
        model_name_or_path=args.model_name_or_path,
        provider=args.provider,
        drain_limit=int(args.drain_limit),
        max_source_context_bytes=int(args.max_source_context_bytes),
        instance_ids=args.instance_ids,
        limit=int(args.limit),
        prepare_repo_cache=bool(args.prepare_repo_cache),
        repo_cache_manifest_json=args.repo_cache_manifest_json,
        fetch_repo_cache=bool(args.fetch_repo_cache),
        official_harness_kind=args.official_harness_kind,
        live_predictions_json=args.live_predictions_json,
        live_platform=args.live_platform,
        live_overwrite=int(args.live_overwrite),
        live_start_month=args.live_start_month,
        live_end_month=args.live_end_month,
        live_submission_dir=args.live_submission_dir,
        live_submission_subset=args.live_submission_subset,
        live_system_name=args.live_system_name,
    )
    _write_json(Path(args.output_harness_json), spec)
    print(
        f"benchmark={args.benchmark} "
        f"selection_mode={spec['autonomy_contract']['selection_mode']} "
        f"harness_json={args.output_harness_json}"
    )


def _retry_harness_mode(args: argparse.Namespace) -> None:
    source_harness = _read_json(Path(args.source_harness_json))
    source_harness["source_harness_json"] = args.source_harness_json
    patch_job_verification = _read_json(Path(args.patch_job_verification_json))
    patch_job_verification["patch_job_verification_json"] = args.patch_job_verification_json
    spec = build_swe_retry_harness_spec(
        source_harness=source_harness,
        patch_job_verification=patch_job_verification,
        retry_label=args.retry_label,
        artifact_dir=Path(args.artifact_dir),
        output_harness_json=args.output_harness_json,
        run_id=args.run_id,
    )
    _write_json(Path(args.output_harness_json), spec)
    print(
        f"benchmark={spec['benchmark']} "
        f"retry_count={len(spec['source_retry']['retry_instance_ids'])} "
        f"harness_json={args.output_harness_json}"
    )


def _success_continuation_harness_mode(args: argparse.Namespace) -> None:
    source_harness = _read_json(Path(args.source_harness_json))
    source_harness["source_harness_json"] = args.source_harness_json
    patch_job_verification = _read_json(Path(args.patch_job_verification_json))
    patch_job_verification["patch_job_verification_json"] = args.patch_job_verification_json
    spec = build_swe_success_continuation_harness_spec(
        source_harness=source_harness,
        patch_job_verification=patch_job_verification,
        success_label=args.success_label,
        artifact_dir=Path(args.artifact_dir),
        output_harness_json=args.output_harness_json,
        run_id=args.run_id,
    )
    _write_json(Path(args.output_harness_json), spec)
    print(
        f"benchmark={spec['benchmark']} "
        f"success_count={len(spec['source_success_continuation']['successful_instance_ids'])} "
        f"harness_json={args.output_harness_json}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    command_parser = subparsers.add_parser("command")
    command_parser.add_argument("--benchmark", choices=tuple(DEFAULT_DATASETS), required=True)
    command_parser.add_argument("--dataset-name", default="")
    command_parser.add_argument("--split", default="test")
    command_parser.add_argument("--predictions-path", required=True)
    command_parser.add_argument("--run-id", required=True)
    command_parser.add_argument("--max-workers", type=int, default=4)
    command_parser.add_argument("--timeout", type=int, default=1800)
    command_parser.add_argument("--cache-level", default="env")
    command_parser.add_argument("--namespace", default="swebench")
    command_parser.add_argument("--report-dir", default=".")
    command_parser.add_argument("--python-bin", default=sys.executable)
    command_parser.add_argument("--swe-bench-root", default="/data/agiattempt/agi_dw/third_party/swe-bench")
    command_parser.add_argument("--summary-json", default="")
    command_parser.add_argument("--output-packet-json", default="")
    command_parser.add_argument("--output-command-json", required=True)
    command_parser.add_argument("--output-spec-json", default="")
    command_parser.add_argument("--instance-ids", nargs="*")
    command_parser.set_defaults(func=_command_mode)

    spec_parser = subparsers.add_parser("spec")
    spec_parser.add_argument("--benchmark", choices=tuple(DEFAULT_DATASETS), required=True)
    spec_parser.add_argument("--dataset-name", default="")
    spec_parser.add_argument("--split", default="test")
    spec_parser.add_argument("--predictions-path", required=True)
    spec_parser.add_argument("--run-id", required=True)
    spec_parser.add_argument("--max-workers", type=int, default=4)
    spec_parser.add_argument("--timeout", type=int, default=1800)
    spec_parser.add_argument("--cache-level", default="env")
    spec_parser.add_argument("--namespace", default="swebench")
    spec_parser.add_argument("--report-dir", required=True)
    spec_parser.add_argument("--results-json", required=True)
    spec_parser.add_argument("--summary-json", required=True)
    spec_parser.add_argument("--output-packet-json", required=True)
    spec_parser.add_argument("--swe-bench-root", default="/data/agiattempt/agi_dw/third_party/swe-bench")
    spec_parser.add_argument("--ready-to-run", choices=("auto", "true", "false"), default="auto")
    spec_parser.add_argument("--instance-ids", nargs="*")
    spec_parser.add_argument("--output-spec-json", required=True)
    spec_parser.set_defaults(func=_spec_mode)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results-json", required=True)
    summarize_parser.add_argument("--output-summary-json", required=True)
    summarize_parser.set_defaults(func=_summarize_mode)

    materialize_parser = subparsers.add_parser("materialize-results")
    materialize_parser.add_argument("--run-id", required=True)
    materialize_parser.add_argument("--namespace", default="swebench")
    materialize_parser.add_argument("--report-dir", required=True)
    materialize_parser.add_argument("--output-results-json", required=True)
    materialize_parser.add_argument("--search-root", default=".")
    materialize_parser.set_defaults(func=_materialize_results_mode)

    harness_parser = subparsers.add_parser("harness")
    harness_parser.add_argument("--benchmark", choices=tuple(DEFAULT_DATASETS), required=True)
    harness_parser.add_argument("--dataset-json", required=True)
    harness_parser.add_argument("--dataset-name", default="")
    harness_parser.add_argument("--split", default="test")
    harness_parser.add_argument("--repo-cache-root", required=True)
    harness_parser.add_argument("--prediction-task-manifest", required=True)
    harness_parser.add_argument("--patch-dir", required=True)
    harness_parser.add_argument("--queue-manifest", required=True)
    harness_parser.add_argument("--queue-root", required=True)
    harness_parser.add_argument("--workspace-root", default="workspace")
    harness_parser.add_argument("--workspace-prefix", required=True)
    harness_parser.add_argument("--predictions-jsonl", required=True)
    harness_parser.add_argument("--apply-check-json", required=True)
    harness_parser.add_argument("--run-spec-json", required=True)
    harness_parser.add_argument("--run-id", required=True)
    harness_parser.add_argument("--max-workers", type=int, default=1)
    harness_parser.add_argument("--timeout", type=int, default=1800)
    harness_parser.add_argument("--cache-level", default="env")
    harness_parser.add_argument("--namespace", default="swebench")
    harness_parser.add_argument("--report-dir", required=True)
    harness_parser.add_argument("--results-json", required=True)
    harness_parser.add_argument("--summary-json", required=True)
    harness_parser.add_argument("--output-packet-json", required=True)
    harness_parser.add_argument("--swe-bench-root", default="/data/agiattempt/agi_dw/third_party/swe-bench")
    harness_parser.add_argument("--python-bin", default=sys.executable)
    harness_parser.add_argument("--model-name-or-path", default="agentkernel")
    harness_parser.add_argument("--provider", default="vllm")
    harness_parser.add_argument("--drain-limit", type=int, default=0)
    harness_parser.add_argument("--max-source-context-bytes", type=int, default=30000)
    harness_parser.add_argument("--limit", type=int, default=0)
    harness_parser.add_argument("--instance-ids", nargs="*")
    harness_parser.add_argument("--prepare-repo-cache", action="store_true")
    harness_parser.add_argument("--repo-cache-manifest-json", default="")
    harness_parser.add_argument("--fetch-repo-cache", action="store_true")
    harness_parser.add_argument("--official-harness-kind", choices=("swebench", "swebench_live"), default="swebench")
    harness_parser.add_argument("--live-predictions-json", default="")
    harness_parser.add_argument("--live-platform", choices=("linux", "windows"), default="linux")
    harness_parser.add_argument("--live-overwrite", type=int, choices=(0, 1), default=0)
    harness_parser.add_argument("--live-start-month", default="")
    harness_parser.add_argument("--live-end-month", default="")
    harness_parser.add_argument("--live-submission-dir", default="")
    harness_parser.add_argument("--live-submission-subset", default="")
    harness_parser.add_argument("--live-system-name", default="Agent Kernel")
    harness_parser.add_argument("--output-harness-json", required=True)
    harness_parser.set_defaults(func=_harness_mode)

    retry_harness_parser = subparsers.add_parser("retry-harness")
    retry_harness_parser.add_argument("--source-harness-json", required=True)
    retry_harness_parser.add_argument("--patch-job-verification-json", required=True)
    retry_harness_parser.add_argument("--retry-label", default="retry")
    retry_harness_parser.add_argument("--artifact-dir", required=True)
    retry_harness_parser.add_argument("--run-id", default="")
    retry_harness_parser.add_argument("--output-harness-json", required=True)
    retry_harness_parser.set_defaults(func=_retry_harness_mode)

    success_harness_parser = subparsers.add_parser("success-continuation-harness")
    success_harness_parser.add_argument("--source-harness-json", required=True)
    success_harness_parser.add_argument("--patch-job-verification-json", required=True)
    success_harness_parser.add_argument("--success-label", default="success")
    success_harness_parser.add_argument("--artifact-dir", required=True)
    success_harness_parser.add_argument("--run-id", default="")
    success_harness_parser.add_argument("--output-harness-json", required=True)
    success_harness_parser.set_defaults(func=_success_continuation_harness_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
