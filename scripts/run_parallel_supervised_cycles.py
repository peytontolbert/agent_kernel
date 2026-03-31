from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess
from threading import Thread
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner
from evals.harness import scoped_improvement_cycle_config
from evals.metrics import EvalMetrics

_BATCH_HISTORY_FILENAME = "parallel_supervised_preview_history.jsonl"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _normalized_optional_values(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        token = str(value).strip()
        if token:
            normalized.append(token)
    return normalized


def _value_for_worker(values: list[str], *, worker_index: int, field_name: str, worker_count: int) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) != worker_count:
        raise SystemExit(
            f"{field_name} expects either one value or exactly --workers values; "
            f"received {len(values)} for {worker_count} workers"
        )
    return values[worker_index]


def _child_scope_id(scope_prefix: str, worker_index: int) -> str:
    return f"{scope_prefix}_{worker_index + 1}"


def _child_progress_label(progress_prefix: str, worker_index: int) -> str:
    return f"{progress_prefix}_{worker_index + 1}"


def _prefixed_progress_line(scope_id: str, line: str) -> str:
    text = str(line).rstrip("\n")
    return f"[parallel:{scope_id}] {text}" if text else f"[parallel:{scope_id}]"


def _stream_pipe(pipe, *, scope_id: str, sink: list[str]) -> None:
    try:
        for line in pipe:
            sink.append(line)
            print(_prefixed_progress_line(scope_id, line), file=sys.stderr, flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_streaming_command(cmd: list[str], *, cwd: Path, env: dict[str, str], scope_id: str) -> CompletedProcess[str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_thread = Thread(target=_stream_pipe, args=(process.stdout,), kwargs={"scope_id": scope_id, "sink": stdout_lines})
    stderr_thread = Thread(target=_stream_pipe, args=(process.stderr,), kwargs={"scope_id": scope_id, "sink": stderr_lines})
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return CompletedProcess(
        cmd,
        returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def _eval_metrics_from_summary(summary: object) -> EvalMetrics:
    payload = summary if isinstance(summary, dict) else {}
    allowed_fields = {field.name for field in fields(EvalMetrics)}
    kwargs = {
        name: payload[name]
        for name in allowed_fields
        if name in payload
    }
    kwargs.setdefault("total", int(payload.get("total", 0) or 0) if isinstance(payload, dict) else 0)
    kwargs.setdefault("passed", int(payload.get("passed", 0) or 0) if isinstance(payload, dict) else 0)
    return EvalMetrics(**kwargs)


def _latest_observe_metrics(config: KernelConfig) -> EvalMetrics:
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    records = planner.load_cycle_records(config.improvement_cycles_path)
    for record in reversed(records):
        if str(record.get("state", "")).strip() != "observe":
            continue
        return _eval_metrics_from_summary(record.get("metrics_summary", {}))
    return EvalMetrics(total=0, passed=0)


def _load_cycle_records(path: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    if config is not None and config.uses_sqlite_storage():
        records = config.sqlite_store().load_cycle_records(output_path=path)
        if records:
            return records
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, object]] = []
    for line in lines:
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _batch_history_path(config: KernelConfig) -> Path:
    return config.improvement_reports_dir / _BATCH_HISTORY_FILENAME


def _load_recent_batch_history(config: KernelConfig, *, recent_limit: int = 12) -> list[dict[str, object]]:
    path = _batch_history_path(config)
    records = _load_cycle_records(path, config=config)
    if recent_limit > 0:
        records = records[-recent_limit:]
    return records


def _subsystem_parallel_history_penalty(subsystem: str, history: list[dict[str, object]]) -> float:
    penalty = 0.0
    bonus = 0.0
    for batch in history:
        runs = batch.get("runs", [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            target = str(run.get("requested_subsystem", "") or run.get("selected_subsystem", "")).strip()
            if target != subsystem:
                continue
            if int(run.get("returncode", 0) or 0) != 0:
                penalty += 0.08
            if bool(run.get("observation_timed_out", False)):
                penalty += 0.07
            if not bool(run.get("generated_candidate", False)):
                penalty += 0.05
            if bool(run.get("generated_candidate", False)) and int(run.get("returncode", 0) or 0) == 0:
                bonus += 0.03
    return round(max(-0.08, min(0.25, penalty - bonus)), 4)


def _planner_for_config(config: KernelConfig) -> ImprovementPlanner:
    return ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )


def _planner_ranked_experiments(config: KernelConfig, *, workers: int) -> tuple[EvalMetrics, ImprovementPlanner, list[object]]:
    metrics = _latest_observe_metrics(config)
    planner = _planner_for_config(config)
    candidates = planner.select_portfolio_campaign(metrics, max_candidates=max(int(workers) * 3, int(workers)))
    if not candidates:
        candidates = planner.rank_experiments(metrics)
    return metrics, planner, list(candidates)


def _planner_diversified_subsystems(config: KernelConfig, *, workers: int) -> list[str]:
    metrics, planner, candidates = _planner_ranked_experiments(config, workers=workers)
    history = _load_recent_batch_history(config)
    ranked = []
    for candidate in candidates:
        penalty = _subsystem_parallel_history_penalty(candidate.subsystem, history)
        ranked.append((float(candidate.score) - penalty, -float(candidate.expected_gain), candidate.subsystem))
    ranked.sort(reverse=True)
    subsystems = [subsystem for _, _, subsystem in ranked]
    if not subsystems:
        return []
    selected: list[str] = []
    while len(selected) < int(workers):
        for subsystem in subsystems:
            selected.append(subsystem)
            if len(selected) >= int(workers):
                break
    return selected[: int(workers)]


def _planner_variant_ids_for_subsystems(config: KernelConfig, *, requested_subsystems: list[str]) -> list[str]:
    normalized_subsystems = [str(subsystem).strip() for subsystem in requested_subsystems]
    if not normalized_subsystems:
        return []
    metrics, planner, ranked_experiments = _planner_ranked_experiments(config, workers=len(normalized_subsystems))
    experiments_by_subsystem: dict[str, object] = {}
    for experiment in ranked_experiments:
        subsystem = str(getattr(experiment, "subsystem", "")).strip()
        if subsystem and subsystem not in experiments_by_subsystem:
            experiments_by_subsystem[subsystem] = experiment

    variant_ids = [""] * len(normalized_subsystems)
    subsystem_indexes: dict[str, list[int]] = {}
    for index, subsystem in enumerate(normalized_subsystems):
        if subsystem:
            subsystem_indexes.setdefault(subsystem, []).append(index)

    for subsystem, indexes in subsystem_indexes.items():
        experiment = experiments_by_subsystem.get(subsystem)
        if experiment is None:
            continue
        ranked_variants = planner.rank_variants(experiment, metrics)
        if not ranked_variants:
            continue
        variant_width = max(1, len(indexes))
        variant_budget = planner.recommend_variant_budget(experiment, metrics, max_width=variant_width)
        selected_variant_ids: list[str] = [
            str(variant_id).strip()
            for variant_id in variant_budget.selected_ids
            if str(variant_id).strip()
        ]
        for variant in ranked_variants:
            variant_id = str(getattr(variant, "variant_id", "")).strip()
            if variant_id and variant_id not in selected_variant_ids:
                selected_variant_ids.append(variant_id)
            if len(selected_variant_ids) >= len(indexes):
                break
        if not selected_variant_ids:
            continue
        for offset, index in enumerate(indexes):
            variant_ids[index] = selected_variant_ids[min(offset, len(selected_variant_ids) - 1)]
    return variant_ids


def _child_command(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    worker_index: int,
    scope_id: str,
    progress_label: str,
    requested_subsystem: str,
    requested_variant_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_supervised_improvement_cycle.py"),
        "--generate-only",
        "--scope-id",
        scope_id,
        "--progress-label",
        progress_label,
    ]
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.task_limit > 0:
        cmd.extend(["--task-limit", str(args.task_limit)])
    if args.max_observation_seconds > 0.0:
        cmd.extend(["--max-observation-seconds", str(args.max_observation_seconds)])
    if requested_subsystem:
        cmd.extend(["--subsystem", requested_subsystem])
    if requested_variant_id:
        cmd.extend(["--variant-id", requested_variant_id])
    notes = str(args.notes).strip()
    if notes:
        cmd.extend(["--notes", f"{notes} [{scope_id}]"])
    for family in args.priority_benchmark_family:
        token = str(family).strip()
        if token:
            cmd.extend(["--priority-benchmark-family", token])
    for weighted_family in args.priority_benchmark_family_weight:
        token = str(weighted_family).strip()
        if token:
            cmd.extend(["--priority-benchmark-family-weight", token])
    for flag, enabled in (
        ("--include-episode-memory", args.include_episode_memory),
        ("--include-skill-memory", args.include_skill_memory),
        ("--include-skill-transfer", args.include_skill_transfer),
        ("--include-operator-memory", args.include_operator_memory),
        ("--include-tool-memory", args.include_tool_memory),
        ("--include-verifier-memory", args.include_verifier_memory),
        ("--include-curriculum", args.include_curriculum),
        ("--include-failure-curriculum", args.include_failure_curriculum),
    ):
        if enabled:
            cmd.append(flag)
    if args.generated_curriculum_budget_seconds > 0.0:
        cmd.extend(["--generated-curriculum-budget-seconds", str(args.generated_curriculum_budget_seconds)])
    if args.failure_curriculum_budget_seconds > 0.0:
        cmd.extend(["--failure-curriculum-budget-seconds", str(args.failure_curriculum_budget_seconds)])
    return cmd


def _records_for_latest_cycle(records: list[dict[str, object]]) -> list[dict[str, object]]:
    latest_cycle_id = ""
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if cycle_id:
            latest_cycle_id = cycle_id
            break
    if not latest_cycle_id:
        return []
    return [record for record in records if str(record.get("cycle_id", "")).strip() == latest_cycle_id]


def _record_for_state(records: list[dict[str, object]], state: str) -> dict[str, object]:
    for record in records:
        if str(record.get("state", "")).strip() == state:
            return record
    return {}


def _scoped_run_summary(base_config: KernelConfig, *, scope_id: str) -> dict[str, object]:
    scoped_config = scoped_improvement_cycle_config(base_config, scope_id, seed_from_base=False)
    records = _load_cycle_records(scoped_config.improvement_cycles_path, config=base_config)
    cycle_records = _records_for_latest_cycle(records)
    observe_record = _record_for_state(cycle_records, "observe")
    select_record = _record_for_state(cycle_records, "select")
    generate_record = _record_for_state(cycle_records, "generate")
    observe_metrics = observe_record.get("metrics_summary", {}) if isinstance(observe_record.get("metrics_summary", {}), dict) else {}
    select_metrics = select_record.get("metrics_summary", {}) if isinstance(select_record.get("metrics_summary", {}), dict) else {}
    candidate_artifact_path = str(
        generate_record.get("candidate_artifact_path") or generate_record.get("artifact_path") or ""
    ).strip()
    return {
        "scope_id": scope_id,
        "workspace_root": str(scoped_config.workspace_root),
        "cycles_path": str(scoped_config.improvement_cycles_path),
        "candidate_artifacts_root": str(scoped_config.candidate_artifacts_root),
        "cycle_id": str(cycle_records[-1].get("cycle_id", "")).strip() if cycle_records else "",
        "selected_subsystem": str(select_record.get("subsystem", "")).strip(),
        "selected_variant_id": str(select_metrics.get("selected_variant_id", "")).strip()
        or str((select_metrics.get("selected_variant", {}) or {}).get("variant_id", "")).strip(),
        "candidate_artifact_path": candidate_artifact_path,
        "generated_candidate": bool(generate_record),
        "observation_timed_out": bool(observe_metrics.get("observation_timed_out", False)),
        "observation_warning": str(observe_metrics.get("observation_warning", "")).strip(),
        "observation_elapsed_seconds": float(observe_metrics.get("observation_elapsed_seconds", 0.0) or 0.0),
        "records_written": len(cycle_records),
        "states": [str(record.get("state", "")).strip() for record in cycle_records],
    }


def _run_single_worker(
    *,
    repo_root: Path,
    base_config: KernelConfig,
    args: argparse.Namespace,
    worker_index: int,
    scope_prefix: str,
    progress_prefix: str,
    requested_subsystem: str,
    requested_variant_id: str,
) -> dict[str, object]:
    scope_id = _child_scope_id(scope_prefix, worker_index)
    progress_label = _child_progress_label(progress_prefix, worker_index)
    cmd = _child_command(
        repo_root=repo_root,
        args=args,
        worker_index=worker_index,
        scope_id=scope_id,
        progress_label=progress_label,
        requested_subsystem=requested_subsystem,
        requested_variant_id=requested_variant_id,
    )
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(base_config.to_env())
    completed = _run_streaming_command(
        cmd,
        cwd=repo_root,
        env=env,
        scope_id=scope_id,
    )
    summary = _scoped_run_summary(base_config, scope_id=scope_id)
    return {
        "worker_index": worker_index + 1,
        "scope_id": scope_id,
        "progress_label": progress_label,
        "requested_subsystem": requested_subsystem,
        "requested_variant_id": requested_variant_id,
        "command": cmd,
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout).strip(),
        "stderr": str(completed.stderr).strip(),
        **summary,
    }


def _batch_report_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    generated_runs = sum(1 for run in runs if bool(run.get("generated_candidate", False)))
    timed_out_runs = sum(1 for run in runs if bool(run.get("observation_timed_out", False)))
    failed_runs = sum(1 for run in runs if int(run.get("returncode", 0) or 0) != 0)
    selected_subsystems = sorted(
        {
            str(run.get("selected_subsystem", "")).strip()
            for run in runs
            if str(run.get("selected_subsystem", "")).strip()
        }
    )
    requested_subsystems = sorted(
        {
            str(run.get("requested_subsystem", "")).strip()
            for run in runs
            if str(run.get("requested_subsystem", "")).strip()
        }
    )
    requested_variant_ids = sorted(
        {
            str(run.get("requested_variant_id", "")).strip()
            for run in runs
            if str(run.get("requested_variant_id", "")).strip()
        }
    )
    return {
        "completed_runs": len(runs),
        "generated_runs": generated_runs,
        "timed_out_runs": timed_out_runs,
        "failed_runs": failed_runs,
        "selected_subsystems": selected_subsystems,
        "requested_subsystems": requested_subsystems,
        "requested_variant_ids": requested_variant_ids,
    }


def _batch_history_record(
    *,
    report_path: Path,
    started_at: datetime,
    completed_at: datetime,
    scope_prefix: str,
    summary: dict[str, object],
    runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "report_kind": "parallel_supervised_preview_history",
        "report_path": str(report_path),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "scope_prefix": scope_prefix,
        "summary": summary,
        "runs": [
            {
                "scope_id": str(run.get("scope_id", "")).strip(),
                "requested_subsystem": str(run.get("requested_subsystem", "")).strip(),
                "requested_variant_id": str(run.get("requested_variant_id", "")).strip(),
                "selected_subsystem": str(run.get("selected_subsystem", "")).strip(),
                "selected_variant_id": str(run.get("selected_variant_id", "")).strip(),
                "generated_candidate": bool(run.get("generated_candidate", False)),
                "observation_timed_out": bool(run.get("observation_timed_out", False)),
                "returncode": int(run.get("returncode", 0) or 0),
            }
            for run in runs
        ],
    }


def _append_batch_history(config: KernelConfig, payload: dict[str, object]) -> None:
    history_path = _batch_history_path(config)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-observation-seconds", type=float, default=0.0)
    parser.add_argument("--scope-prefix", default="")
    parser.add_argument("--progress-label-prefix", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--subsystem", action="append", default=[])
    parser.add_argument("--variant-id", action="append", default=[])
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--priority-benchmark-family-weight", action="append", default=[])
    parser.add_argument("--auto-diversify-subsystems", dest="auto_diversify_subsystems", action="store_true")
    parser.add_argument("--no-auto-diversify-subsystems", dest="auto_diversify_subsystems", action="store_false")
    parser.set_defaults(auto_diversify_subsystems=True)
    parser.add_argument("--auto-diversify-variants", dest="auto_diversify_variants", action="store_true")
    parser.add_argument("--no-auto-diversify-variants", dest="auto_diversify_variants", action="store_false")
    parser.set_defaults(auto_diversify_variants=False)
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--generated-curriculum-budget-seconds", type=float, default=0.0)
    parser.add_argument("--failure-curriculum-budget-seconds", type=float, default=0.0)
    args = parser.parse_args()

    args.subsystem = _normalized_optional_values(args.subsystem)
    args.variant_id = _normalized_optional_values(args.variant_id)
    if max(1, int(args.workers)) != int(args.workers):
        raise SystemExit("--workers must be at least 1")

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    repo_root = Path(__file__).resolve().parents[1]
    scope_prefix = str(args.scope_prefix).strip() or f"supervised_parallel_{_utc_stamp()}"
    progress_prefix = str(args.progress_label_prefix).strip() or scope_prefix

    if args.subsystem:
        requested_subsystems = [
            _value_for_worker(
                args.subsystem,
                worker_index=index,
                field_name="--subsystem",
                worker_count=args.workers,
            )
            for index in range(int(args.workers))
        ]
    elif args.auto_diversify_subsystems:
        requested_subsystems = _planner_diversified_subsystems(config, workers=int(args.workers))
    else:
        requested_subsystems = [""] * int(args.workers)
    if len(requested_subsystems) < int(args.workers):
        requested_subsystems.extend([""] * (int(args.workers) - len(requested_subsystems)))

    if args.variant_id:
        requested_variant_ids = [
            _value_for_worker(
                args.variant_id,
                worker_index=index,
                field_name="--variant-id",
                worker_count=args.workers,
            )
            for index in range(int(args.workers))
        ]
    elif args.auto_diversify_variants:
        requested_variant_ids = _planner_variant_ids_for_subsystems(
            config,
            requested_subsystems=requested_subsystems[: int(args.workers)],
        )
    else:
        requested_variant_ids = [""] * int(args.workers)
    if len(requested_variant_ids) < int(args.workers):
        requested_variant_ids.extend([""] * (int(args.workers) - len(requested_variant_ids)))

    started_at = datetime.now(timezone.utc)
    runs: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [
            executor.submit(
                _run_single_worker,
                repo_root=repo_root,
                base_config=config,
                args=args,
                worker_index=worker_index,
                scope_prefix=scope_prefix,
                progress_prefix=progress_prefix,
                requested_subsystem=requested_subsystems[worker_index],
                requested_variant_id=requested_variant_ids[worker_index],
            )
            for worker_index in range(int(args.workers))
        ]
        for future in as_completed(futures):
            runs.append(future.result())
    runs.sort(key=lambda run: int(run.get("worker_index", 0) or 0))

    completed_at = datetime.now(timezone.utc)
    summary = _batch_report_summary(runs)
    report_payload = {
        "report_kind": "parallel_supervised_preview_report",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "worker_count": int(args.workers),
        "scope_prefix": scope_prefix,
        "progress_label_prefix": progress_prefix,
        "auto_diversify_subsystems": bool(args.auto_diversify_subsystems),
        "summary": summary,
        "runs": runs,
    }
    report_path = config.improvement_reports_dir / f"parallel_supervised_preview_{_utc_stamp()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    _append_batch_history(
        config,
        _batch_history_record(
            report_path=report_path,
            started_at=started_at,
            completed_at=completed_at,
            scope_prefix=scope_prefix,
            summary=summary,
            runs=runs,
        ),
    )
    print(str(report_path))


if __name__ == "__main__":
    main()
