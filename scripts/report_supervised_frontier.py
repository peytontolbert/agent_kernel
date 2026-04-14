from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.improvement.artifact_support_evidence import (
    artifact_retrieval_reuse_evidence,
    tool_shared_repo_bundle_evidence,
)
from agent_kernel.ops.runtime_supervision import atomic_write_json


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
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _record_for_state(records: list[dict[str, object]], state: str) -> dict[str, object]:
    for record in records:
        if str(record.get("state", "")).strip() == state:
            return record
    return {}


def _record_scope_id(record: dict[str, object]) -> str:
    metrics = _metrics_summary(record)
    return str(metrics.get("scope_id", "")).strip()


def _latest_cycle_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    latest_cycle_id = ""
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if cycle_id:
            latest_cycle_id = cycle_id
            break
    if not latest_cycle_id:
        return []
    return [record for record in records if str(record.get("cycle_id", "")).strip() == latest_cycle_id]


def _metrics_summary(record: dict[str, object]) -> dict[str, object]:
    metrics = record.get("metrics_summary", {})
    return metrics if isinstance(metrics, dict) else {}


def _record_variant_id(record: dict[str, object]) -> str:
    metrics = _metrics_summary(record)
    direct = str(metrics.get("selected_variant_id", "")).strip()
    if direct:
        return direct
    selected_variant = metrics.get("selected_variant", {})
    if isinstance(selected_variant, dict):
        return str(selected_variant.get("variant_id", "")).strip()
    return ""


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _followup_summary_metrics(observe_metrics: dict[str, object], kind: str) -> tuple[int, int]:
    followups = observe_metrics.get("observation_curriculum_followups", [])
    if not isinstance(followups, list):
        return 0, 0
    for item in followups:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind", "")).strip() != kind:
            continue
        return (
            int(item.get("generated_total", 0) or 0),
            int(item.get("generated_passed", 0) or 0),
        )
    return 0, 0


def _candidate_sha256(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return sha256(data).hexdigest()


def _load_json_object(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _difficulty_slice(metrics: dict[str, object], field: str, difficulty: str) -> dict[str, object]:
    payload = metrics.get(field, {})
    if not isinstance(payload, dict):
        return {}
    item = payload.get(difficulty, {})
    return dict(item) if isinstance(item, dict) else {}


def _benchmark_family_slice(metrics: dict[str, object], field: str, family: str) -> dict[str, object]:
    payload = metrics.get(field, {})
    if not isinstance(payload, dict):
        return {}
    item = payload.get(family, {})
    return dict(item) if isinstance(item, dict) else {}


def _observed_benchmark_families(observe_metrics: dict[str, object]) -> list[str]:
    families: set[str] = set()
    for field in (
        "total_by_benchmark_family",
        "passed_by_benchmark_family",
        "generated_by_benchmark_family",
        "generated_passed_by_benchmark_family",
        "total_by_origin_benchmark_family",
        "passed_by_origin_benchmark_family",
        "generated_by_origin_benchmark_family",
        "generated_passed_by_origin_benchmark_family",
        "proposal_metrics_by_benchmark_family",
        "world_feedback_by_benchmark_family",
    ):
        payload = observe_metrics.get(field, {})
        if not isinstance(payload, dict):
            continue
        for family in payload:
            normalized = str(family).strip()
            if normalized:
                families.add(normalized)
    return sorted(families)


def _long_horizon_summary(observe_metrics: dict[str, object]) -> dict[str, object]:
    difficulty = "long_horizon"
    total_by_difficulty = observe_metrics.get("total_by_difficulty", {})
    passed_by_difficulty = observe_metrics.get("passed_by_difficulty", {})
    total = int(total_by_difficulty.get(difficulty, 0) or 0) if isinstance(total_by_difficulty, dict) else 0
    passed = int(passed_by_difficulty.get(difficulty, 0) or 0) if isinstance(passed_by_difficulty, dict) else 0
    proposal = _difficulty_slice(observe_metrics, "proposal_metrics_by_difficulty", difficulty)
    world_feedback = _difficulty_slice(observe_metrics, "world_feedback_by_difficulty", difficulty)
    if total <= 0 and not proposal and not world_feedback:
        return {}
    world_feedback_summary = {
        key: value
        for key, value in world_feedback.items()
        if key
        in {
            "step_count",
            "progress_calibration_mae",
            "risk_calibration_mae",
            "decoder_progress_calibration_mae",
            "decoder_risk_calibration_mae",
        }
    }
    return {
        "difficulty": difficulty,
        "task_count": total,
        "passed": passed,
        "pass_rate": round((float(passed) / float(total)) if total > 0 else 0.0, 4),
        "proposal_selected_steps": int(proposal.get("proposal_selected_steps", 0) or 0),
        "novel_valid_command_steps": int(proposal.get("novel_valid_command_steps", 0) or 0),
        "novel_valid_command_rate": round(float(proposal.get("novel_valid_command_rate", 0.0) or 0.0), 4),
        "world_feedback_step_count": int(world_feedback.get("step_count", 0) or 0),
        "world_feedback": world_feedback_summary,
    }


def _benchmark_family_summary(observe_metrics: dict[str, object], family: str) -> dict[str, object]:
    total_by_family = observe_metrics.get("total_by_benchmark_family", {})
    passed_by_family = observe_metrics.get("passed_by_benchmark_family", {})
    generated_by_family = observe_metrics.get("generated_by_benchmark_family", {})
    generated_passed_by_family = observe_metrics.get("generated_passed_by_benchmark_family", {})
    primary_total = int(total_by_family.get(family, 0) or 0) if isinstance(total_by_family, dict) else 0
    primary_passed = int(passed_by_family.get(family, 0) or 0) if isinstance(passed_by_family, dict) else 0
    generated_total = int(generated_by_family.get(family, 0) or 0) if isinstance(generated_by_family, dict) else 0
    generated_passed = (
        int(generated_passed_by_family.get(family, 0) or 0) if isinstance(generated_passed_by_family, dict) else 0
    )
    proposal = _benchmark_family_slice(observe_metrics, "proposal_metrics_by_benchmark_family", family)
    world_feedback = _benchmark_family_slice(observe_metrics, "world_feedback_by_benchmark_family", family)
    if primary_total <= 0 and generated_total <= 0 and not proposal and not world_feedback:
        return {}
    world_feedback_summary = {
        key: value
        for key, value in world_feedback.items()
        if key
        in {
            "step_count",
            "progress_calibration_mae",
            "risk_calibration_mae",
            "decoder_progress_calibration_mae",
            "decoder_risk_calibration_mae",
        }
    }
    return {
        "benchmark_family": family,
        "primary_task_count": primary_total,
        "primary_passed": primary_passed,
        "primary_pass_rate": round((float(primary_passed) / float(primary_total)) if primary_total > 0 else 0.0, 4),
        "generated_task_count": generated_total,
        "generated_passed": generated_passed,
        "generated_pass_rate": round(
            (float(generated_passed) / float(generated_total)) if generated_total > 0 else 0.0,
            4,
        ),
        "proposal_selected_steps": int(proposal.get("proposal_selected_steps", 0) or 0),
        "novel_valid_command_steps": int(proposal.get("novel_valid_command_steps", 0) or 0),
        "novel_valid_command_rate": round(float(proposal.get("novel_valid_command_rate", 0.0) or 0.0), 4),
        "world_feedback_step_count": int(world_feedback.get("step_count", 0) or 0),
        "world_feedback": world_feedback_summary,
    }


def _validation_family_compare_guard_reasons(summary: dict[str, object]) -> list[str]:
    if not isinstance(summary, dict) or not summary:
        return []
    reasons: list[str] = []
    primary_task_count = int(summary.get("primary_task_count", 0) or 0)
    generated_task_count = int(summary.get("generated_task_count", 0) or 0)
    world_feedback_step_count = int(summary.get("world_feedback_step_count", 0) or 0)
    if primary_task_count > 0:
        reasons.append("validation_family_pass_rate_regressed")
    if generated_task_count > 0:
        reasons.append("validation_family_generated_pass_rate_regressed")
    if primary_task_count + generated_task_count > 0:
        reasons.append("validation_family_novel_command_rate_regressed")
    if world_feedback_step_count > 0:
        reasons.append("validation_family_world_feedback_regressed")
    return reasons


def _state_rank(state: str) -> int:
    order = {
        "retain": 5,
        "reject": 4,
        "generate": 3,
        "select": 2,
        "observe": 1,
    }
    return order.get(str(state).strip(), 0)


def _scope_summary(*, cycles_path: Path, config: KernelConfig) -> dict[str, object] | None:
    records = _load_cycle_records(cycles_path, config=config)
    expected_scope_id = cycles_path.stem.removeprefix("cycles_")
    scoped_records = [
        record
        for record in records
        if _record_scope_id(record) == expected_scope_id
    ]
    if scoped_records:
        records = scoped_records
    cycle_records = _latest_cycle_records(records)
    if not cycle_records:
        return None
    latest = cycle_records[-1]
    observe_record = _record_for_state(cycle_records, "observe")
    select_record = _record_for_state(cycle_records, "select")
    generate_record = _record_for_state(cycle_records, "generate")
    observe_metrics = _metrics_summary(observe_record)
    select_metrics = _metrics_summary(select_record)
    latest_metrics = _metrics_summary(latest)

    scope_id = (
        str(observe_metrics.get("scope_id", "")).strip()
        or str(select_metrics.get("scope_id", "")).strip()
        or str(latest_metrics.get("scope_id", "")).strip()
        or expected_scope_id
    )
    selected_subsystem = (
        str(select_record.get("subsystem", "")).strip()
        or str(generate_record.get("subsystem", "")).strip()
        or str(latest.get("subsystem", "")).strip()
    )
    selected_variant_id = (
        _record_variant_id(select_record)
        or _record_variant_id(generate_record)
        or _record_variant_id(latest)
    )
    candidate_artifact_path = str(
        generate_record.get("candidate_artifact_path")
        or generate_record.get("artifact_path")
        or latest.get("candidate_artifact_path")
        or ""
    ).strip()
    candidate_path = Path(candidate_artifact_path) if candidate_artifact_path else None
    candidate_exists = bool(candidate_path and candidate_path.exists())
    candidate_digest = _candidate_sha256(candidate_path) if candidate_path is not None and candidate_exists else ""
    candidate_payload = (
        _load_json_object(candidate_path)
        if candidate_exists and selected_subsystem in {"skills", "tooling"}
        else {}
    )
    observed_benchmark_families = _observed_benchmark_families(observe_metrics)
    long_horizon_summary = _long_horizon_summary(observe_metrics)
    validation_family_summary = _benchmark_family_summary(observe_metrics, "validation")
    validation_family_compare_guard_reasons = _validation_family_compare_guard_reasons(validation_family_summary)
    retrieval_reuse_summary = (
        artifact_retrieval_reuse_evidence(candidate_payload, subsystem=selected_subsystem)
        if isinstance(candidate_payload, dict) and candidate_payload
        else {}
    )
    shared_repo_bundle_summary = (
        tool_shared_repo_bundle_evidence(candidate_payload)
        if selected_subsystem == "tooling" and isinstance(candidate_payload, dict) and candidate_payload
        else {}
    )
    observation_timed_out = bool(observe_metrics.get("observation_timed_out", False))
    observation_budget_exceeded = bool(observe_metrics.get("observation_budget_exceeded", False))
    primary_total = int(observe_metrics.get("total", 0) or 0)
    primary_passed = int(observe_metrics.get("passed", 0) or 0)
    generated_total = int(observe_metrics.get("generated_total", 0) or 0)
    generated_passed = int(observe_metrics.get("generated_passed", 0) or 0)
    generated_success_total, generated_success_passed = _followup_summary_metrics(
        observe_metrics,
        "generated_success",
    )
    generated_failure_total, generated_failure_passed = _followup_summary_metrics(
        observe_metrics,
        "generated_failure",
    )
    if generated_total <= 0 and generated_success_total > 0:
        generated_total = generated_success_total
    if generated_passed <= 0 and generated_success_passed > 0:
        generated_passed = generated_success_passed
    observation_returncode = int(observe_metrics.get("observation_returncode", 0) or 0)
    run_completed = bool(observe_record)
    process_succeeded = bool(run_completed and observation_returncode == 0 and not observation_timed_out)
    healthy_run = bool(process_succeeded and not str(observe_metrics.get("observation_warning", "")).strip())
    status = str(latest.get("state", "")).strip() or "unknown"
    if run_completed:
        status = "healthy" if healthy_run else ("completed_with_warnings" if process_succeeded else "failed")
    return {
        "scope_id": scope_id,
        "scope_label": scope_id,
        "cycles_path": str(cycles_path),
        "cycle_id": str(latest.get("cycle_id", "")).strip(),
        "protocol": str(
            observe_metrics.get("protocol", "")
            or select_metrics.get("protocol", "")
            or latest_metrics.get("protocol", "")
        ).strip(),
        "scoped_run": bool(
            observe_metrics.get("scoped_run", False)
            or select_metrics.get("scoped_run", False)
            or latest_metrics.get("scoped_run", False)
        ),
        "selected_subsystem": selected_subsystem,
        "selected_variant_id": selected_variant_id,
        "last_state": str(latest.get("state", "")).strip(),
        "status": status,
        "run_completed": run_completed,
        "run_succeeded": process_succeeded,
        "process_succeeded": process_succeeded,
        "healthy_run": healthy_run,
        "observation_returncode": observation_returncode,
        "generated_candidate": bool(generate_record) and bool(candidate_artifact_path),
        "candidate_artifact_path": candidate_artifact_path,
        "candidate_exists": candidate_exists,
        "candidate_sha256": candidate_digest,
        "observed_benchmark_families": observed_benchmark_families,
        "long_horizon_summary": long_horizon_summary,
        "validation_family_summary": validation_family_summary,
        "validation_family_compare_guard_reasons": validation_family_compare_guard_reasons,
        "retrieval_reuse_summary": retrieval_reuse_summary,
        "shared_repo_bundle_summary": shared_repo_bundle_summary,
        "candidate_artifact_kind": str(
            generate_record.get("artifact_kind", "") or latest.get("artifact_kind", "")
        ).strip(),
        "primary_total": primary_total,
        "primary_passed": primary_passed,
        "primary_pass_rate": _safe_float(observe_metrics.get("pass_rate", 0.0)),
        "generated_success_total": generated_total,
        "generated_success_passed": generated_passed,
        "generated_success_pass_rate": _safe_float(observe_metrics.get("generated_pass_rate", 0.0)),
        "generated_failure_total": generated_failure_total,
        "generated_failure_passed": generated_failure_passed,
        "observation_timed_out": observation_timed_out,
        "observation_budget_exceeded": observation_budget_exceeded,
        "observation_warning": str(observe_metrics.get("observation_warning", "")).strip(),
        "observation_elapsed_seconds": _safe_float(observe_metrics.get("observation_elapsed_seconds", 0.0)),
        "observation_timeout_budget_source": str(
            observe_metrics.get("observation_current_task_timeout_budget_source", "")
        ).strip(),
        "observation_timeout_budget_seconds": _safe_float(
            observe_metrics.get("observation_current_task_timeout_budget_seconds", 0.0)
        ),
    }


def _frontier_sort_key(summary: dict[str, object]) -> tuple[object, ...]:
    return (
        0 if bool(summary.get("generated_candidate", False)) else 1,
        0 if bool(summary.get("candidate_exists", False)) else 1,
        0 if not bool(summary.get("observation_timed_out", False)) else 1,
        -_state_rank(str(summary.get("last_state", "")).strip()),
        _safe_float(summary.get("observation_elapsed_seconds", 0.0)),
        str(summary.get("scope_id", "")).strip(),
    )


def _dedupe_frontier(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        digest = str(run.get("candidate_sha256", "")).strip()
        candidate_path = str(run.get("candidate_artifact_path", "")).strip()
        fingerprint = digest or candidate_path or str(run.get("scope_id", "")).strip()
        key = (
            str(run.get("selected_subsystem", "")).strip(),
            str(run.get("selected_variant_id", "")).strip(),
            fingerprint,
        )
        groups[key].append(run)
    frontier: list[dict[str, object]] = []
    for siblings in groups.values():
        ordered = sorted(siblings, key=_frontier_sort_key)
        winner = dict(ordered[0])
        winner["duplicate_scope_ids"] = [
            str(item.get("scope_id", "")).strip()
            for item in ordered[1:]
            if str(item.get("scope_id", "")).strip()
        ]
        winner["duplicate_count"] = len(ordered) - 1
        frontier.append(winner)
    frontier.sort(key=_frontier_sort_key)
    return frontier


def _summary_payload(runs: list[dict[str, object]], frontier: list[dict[str, object]]) -> dict[str, object]:
    subsystem_counter = Counter(
        str(run.get("selected_subsystem", "")).strip()
        for run in runs
        if str(run.get("selected_subsystem", "")).strip()
    )
    timeout_source_counter = Counter(
        str(run.get("observation_timeout_budget_source", "")).strip()
        for run in runs
        if str(run.get("observation_timeout_budget_source", "")).strip()
    )
    validation_compare_guard_counter = Counter(
        str(reason).strip()
        for run in runs
        for reason in (run.get("validation_family_compare_guard_reasons", []) if isinstance(run.get("validation_family_compare_guard_reasons", []), list) else [])
        if str(reason).strip()
    )
    retrieval_reuse_runs = [
        run for run in runs if isinstance(run.get("retrieval_reuse_summary", {}), dict) and run.get("retrieval_reuse_summary")
    ]
    validation_runs = [
        run for run in runs if isinstance(run.get("validation_family_summary", {}), dict) and run.get("validation_family_summary")
    ]
    return {
        "scoped_run_count": len(runs),
        "completed_runs": sum(1 for run in runs if bool(run.get("run_completed", False))),
        "successful_runs": sum(1 for run in runs if bool(run.get("process_succeeded", False))),
        "healthy_runs": sum(1 for run in runs if bool(run.get("healthy_run", False))),
        "warning_runs": sum(
            1
            for run in runs
            if bool(run.get("process_succeeded", False)) and not bool(run.get("healthy_run", False))
        ),
        "primary_passed_runs": sum(1 for run in runs if int(run.get("primary_passed", 0) or 0) > 0),
        "generated_success_runs": sum(1 for run in runs if int(run.get("generated_success_passed", 0) or 0) > 0),
        "generated_failure_runs": sum(1 for run in runs if int(run.get("generated_failure_passed", 0) or 0) > 0),
        "primary_total": sum(int(run.get("primary_total", 0) or 0) for run in runs),
        "primary_passed": sum(int(run.get("primary_passed", 0) or 0) for run in runs),
        "generated_success_total": sum(int(run.get("generated_success_total", 0) or 0) for run in runs),
        "generated_success_passed": sum(int(run.get("generated_success_passed", 0) or 0) for run in runs),
        "generated_failure_total": sum(int(run.get("generated_failure_total", 0) or 0) for run in runs),
        "generated_failure_passed": sum(int(run.get("generated_failure_passed", 0) or 0) for run in runs),
        "frontier_candidate_count": len(frontier),
        "generated_candidate_runs": sum(1 for run in runs if bool(run.get("generated_candidate", False))),
        "timed_out_runs": sum(1 for run in runs if bool(run.get("observation_timed_out", False))),
        "budget_exceeded_runs": sum(1 for run in runs if bool(run.get("observation_budget_exceeded", False))),
        "deduped_runs": sum(int(item.get("duplicate_count", 0) or 0) for item in frontier),
        "retrieval_reuse_runs": len(retrieval_reuse_runs),
        "retrieval_backed_procedure_total": sum(
            int(((run.get("retrieval_reuse_summary", {}) or {}).get("retrieval_backed_procedure_count", 0) or 0))
            for run in retrieval_reuse_runs
        ),
        "trusted_retrieval_procedure_total": sum(
            int(((run.get("retrieval_reuse_summary", {}) or {}).get("trusted_retrieval_procedure_count", 0) or 0))
            for run in retrieval_reuse_runs
        ),
        "verified_retrieval_command_total": sum(
            int(((run.get("retrieval_reuse_summary", {}) or {}).get("verified_retrieval_command_count", 0) or 0))
            for run in retrieval_reuse_runs
        ),
        "validation_family_runs": len(validation_runs),
        "validation_generated_runs": sum(
            1
            for run in validation_runs
            if int(((run.get("validation_family_summary", {}) or {}).get("generated_task_count", 0) or 0)) > 0
        ),
        "validation_generated_total": sum(
            int(((run.get("validation_family_summary", {}) or {}).get("generated_task_count", 0) or 0))
            for run in validation_runs
        ),
        "validation_generated_passed": sum(
            int(((run.get("validation_family_summary", {}) or {}).get("generated_passed", 0) or 0))
            for run in validation_runs
        ),
        "validation_family_compare_guard_reason_counts": dict(sorted(validation_compare_guard_counter.items())),
        "subsystems": dict(sorted(subsystem_counter.items())),
        "timeout_budget_sources": dict(sorted(timeout_source_counter.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycles-glob",
        default="cycles_*.jsonl",
        help="Glob relative to trajectories/improvement/ for scoped cycle files to ingest.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional explicit output path. Defaults to improvement_reports_dir/supervised_parallel_frontier.json",
    )
    parser.add_argument(
        "--include-unscoped",
        action="store_true",
        help="Include cycle files whose latest cycle does not report scoped_run=true.",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    search_root = config.improvement_cycles_path.parent
    output_path = (
        Path(args.output_path)
        if str(args.output_path).strip()
        else config.improvement_reports_dir / "supervised_parallel_frontier.json"
    )

    runs: list[dict[str, object]] = []
    cycle_paths = (
        sorted(config.sqlite_store().list_cycle_paths(parent=search_root, pattern=str(args.cycles_glob)))
        if config.uses_sqlite_storage()
        else sorted(search_root.glob(str(args.cycles_glob)))
    )
    for cycles_path in cycle_paths:
        summary = _scope_summary(cycles_path=cycles_path, config=config)
        if summary is None:
            continue
        if not args.include_unscoped and not bool(summary.get("scoped_run", False)):
            continue
        runs.append(summary)

    frontier = _dedupe_frontier(runs)
    payload = {
        "report_kind": "supervised_parallel_frontier_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_root": str(search_root),
        "cycles_glob": str(args.cycles_glob),
        "summary": _summary_payload(runs, frontier),
        "frontier_candidates": frontier,
        "scoped_runs": sorted(runs, key=_frontier_sort_key),
    }
    atomic_write_json(output_path, payload, config=config)
    print(output_path)


if __name__ == "__main__":
    main()
