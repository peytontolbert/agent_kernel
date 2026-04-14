from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping

from ..config import KernelConfig
from ..improvement import ImprovementCycleRecord, ImprovementPlanner
from ..ops.runtime_supervision import atomic_write_json
from ..strategy_memory import finalize_strategy_node


def semantic_progress_phase_family(phase: str) -> str:
    normalized = str(phase).strip()
    if not normalized:
        return "unknown"
    if normalized == "holdout_eval":
        return "holdout"
    if normalized.startswith("preview"):
        return "preview"
    if normalized in {
        "apply_decision",
        "materialize_retrieval_bundle",
        "decision_reject_reason",
        "decision_retain_reason",
        "done",
    }:
        return "apply"
    if normalized in {"observe"} or normalized.startswith("observe_"):
        return "observe"
    if normalized in {"generated_success"} or normalized.startswith("generated_success"):
        return "finalize"
    if normalized in {"variant_search", "variant_generate"}:
        return "preview"
    if normalized in {"confirmation_eval", "confidence_aggregate", "finalize"}:
        return "finalize"
    if normalized in {"generated_failure", "recovery"} or normalized.startswith("generated_failure"):
        return "recovery"
    return "active"


def semantic_progress_state(
    *,
    phase: str,
    now: float,
    started_at: float,
    last_progress_at: float,
    max_progress_stall_seconds: float,
    max_runtime_seconds: float = 0.0,
    current_task: Mapping[str, object] | None = None,
    observe_summary: Mapping[str, object] | None = None,
    pending_decision_state: str = "",
    preview_state: str = "",
    current_task_verification_passed: bool | None = None,
) -> dict[str, object]:
    normalized_phase = str(phase).strip()
    phase_family = semantic_progress_phase_family(normalized_phase)
    if not normalized_phase:
        return {
            "phase": "",
            "phase_family": phase_family,
            "status": "sampling",
            "progress_class": "unknown",
            "decision_distance": "unknown",
            "detail": "waiting for progress events",
        }
    task = current_task if isinstance(current_task, Mapping) else {}
    observe = observe_summary if isinstance(observe_summary, Mapping) else {}
    progress_silence_seconds = max(0.0, float(now) - float(last_progress_at or now))
    runtime_elapsed_seconds = max(0.0, float(now) - float(started_at or now))
    stall_threshold = max(0.0, float(max_progress_stall_seconds or 0.0))
    runtime_pressure = 0.0
    if float(max_runtime_seconds or 0.0) > 0.0:
        runtime_pressure = min(1.0, runtime_elapsed_seconds / max(1.0, float(max_runtime_seconds)))
    pending_state = str(pending_decision_state).strip()
    preview_decision = str(preview_state).strip()
    task_index = int(task.get("index", 0) or 0)
    task_total = int(task.get("total", 0) or 0)
    task_started = task_index > 0
    task_complete = task_started and task_total > 0 and task_index >= task_total
    task_verification_failed = current_task_verification_passed is False
    observe_total = int(observe.get("total", 0) or 0)
    observe_complete = observe_total > 0
    observe_degraded_threshold = 45.0
    observe_stuck_threshold = 120.0
    if stall_threshold > 0.0:
        observe_stuck_threshold = min(observe_stuck_threshold, stall_threshold)
        observe_degraded_threshold = min(observe_degraded_threshold, max(15.0, observe_stuck_threshold / 2.0))

    status = "active"
    progress_class = "healthy"
    decision_distance = "far"
    detail = f"{phase_family} work is progressing"
    if normalized_phase == "done":
        status = "complete"
        progress_class = "complete"
        decision_distance = "complete"
        detail = "finalize completed"
    elif pending_state in {"retain", "reject"} or preview_decision in {"retain", "reject"}:
        status = "decision_emitted"
        progress_class = "complete" if phase_family == "apply" else "healthy"
        decision_distance = "decision_emitted"
        detail = "decision has been emitted and is awaiting durable completion"
    else:
        if phase_family in {"preview", "apply", "finalize", "holdout"}:
            decision_distance = "near"
        elif phase_family in {"observe", "recovery"}:
            decision_distance = "far"
        else:
            decision_distance = "active"
        if (
            phase_family == "observe"
            and not observe_complete
            and task_started
            and progress_silence_seconds >= observe_stuck_threshold
        ):
            status = "stalled"
            progress_class = "stuck"
            detail = "observe throughput stalled before decision evidence"
        elif (
            phase_family == "observe"
            and not observe_complete
            and task_started
            and progress_silence_seconds >= observe_degraded_threshold
        ):
            status = "active"
            progress_class = "degraded"
            detail = "observe throughput is advancing too slowly to support timely intervention"
        elif stall_threshold > 0.0 and progress_silence_seconds >= stall_threshold:
            status = "stalled"
            progress_class = "stuck"
            detail = f"{phase_family} work has stopped making forward progress"
        elif phase_family in {"preview", "apply", "finalize"} and runtime_pressure >= 0.6:
            status = "active"
            progress_class = "degraded"
            detail = f"{phase_family} work is advancing slowly relative to the runtime budget"
        elif phase_family == "observe" and observe_complete:
            status = "complete"
            progress_class = "complete"
            decision_distance = "near"
            detail = "observe phase completed and is ready to hand off"
        elif task_verification_failed:
            status = "active"
            progress_class = "degraded"
            detail = (
                f"{phase_family} task {task_index}/{max(task_total, task_index)} failed verification and is awaiting recovery"
                if task_started
                else f"{phase_family} verification failed and is awaiting recovery"
            )
        elif task_complete:
            detail = (
                f"{phase_family} task {task_index}/{max(task_total, task_index)} reached the execution boundary "
                "and is awaiting verification"
            )
        elif task_started:
            detail = f"{phase_family} task {task_index}/{max(task_total, task_index)} is progressing"
        elif phase_family == "recovery":
            detail = "recovery work is active"
    return {
        "phase": normalized_phase,
        "phase_family": phase_family,
        "status": status,
        "progress_class": progress_class,
        "decision_distance": decision_distance,
        "progress_silence_seconds": progress_silence_seconds,
        "runtime_elapsed_seconds": runtime_elapsed_seconds,
        "detail": detail,
    }


def prior_retained_metrics_summary(comparison: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(comparison, dict):
        return {}
    summary: dict[str, object] = {
        "prior_retained_available": bool(comparison.get("available", False)),
        "prior_retained_baseline_cycle_id": str(comparison.get("baseline_cycle_id", "")),
        "prior_retained_baseline_snapshot_path": str(comparison.get("baseline_snapshot_path", "")),
    }
    if not bool(comparison.get("available", False)):
        reason = str(comparison.get("reason", "")).strip()
        if reason:
            summary["prior_retained_reason"] = reason
        return summary
    baseline_metrics = comparison.get("baseline_metrics", {})
    if isinstance(baseline_metrics, dict):
        summary["prior_retained_baseline_pass_rate"] = float(baseline_metrics.get("pass_rate", 0.0))
        summary["prior_retained_baseline_average_steps"] = float(baseline_metrics.get("average_steps", 0.0))
        summary["prior_retained_baseline_generated_pass_rate"] = float(baseline_metrics.get("generated_pass_rate", 0.0))
        summary["prior_retained_baseline_proposal_selected_steps"] = int(
            baseline_metrics.get("proposal_selected_steps", 0) or 0
        )
        summary["prior_retained_baseline_novel_valid_command_steps"] = int(
            baseline_metrics.get("novel_valid_command_steps", 0) or 0
        )
        summary["prior_retained_baseline_novel_valid_command_rate"] = float(
            baseline_metrics.get("novel_valid_command_rate", 0.0)
        )
        summary["prior_retained_baseline_tolbert_primary_episodes"] = int(
            baseline_metrics.get("tolbert_primary_episodes", 0) or 0
        )
    current_metrics = comparison.get("current_metrics", {})
    if isinstance(current_metrics, dict):
        summary["prior_retained_current_pass_rate"] = float(current_metrics.get("pass_rate", 0.0))
        summary["prior_retained_current_average_steps"] = float(current_metrics.get("average_steps", 0.0))
        summary["prior_retained_current_generated_pass_rate"] = float(current_metrics.get("generated_pass_rate", 0.0))
        summary["prior_retained_current_proposal_selected_steps"] = int(
            current_metrics.get("proposal_selected_steps", 0) or 0
        )
        summary["prior_retained_current_novel_valid_command_steps"] = int(
            current_metrics.get("novel_valid_command_steps", 0) or 0
        )
        summary["prior_retained_current_novel_valid_command_rate"] = float(
            current_metrics.get("novel_valid_command_rate", 0.0)
        )
        summary["prior_retained_current_tolbert_primary_episodes"] = int(
            current_metrics.get("tolbert_primary_episodes", 0) or 0
        )
    summary["prior_retained_pass_rate_delta"] = float(comparison.get("pass_rate_delta", 0.0))
    summary["prior_retained_average_step_delta"] = float(comparison.get("average_step_delta", 0.0))
    summary["prior_retained_generated_pass_rate_delta"] = float(comparison.get("generated_pass_rate_delta", 0.0))
    evidence = comparison.get("evidence", {})
    if isinstance(evidence, dict):
        if "regressed_family_count" in evidence:
            summary["prior_retained_regressed_family_count"] = int(evidence.get("regressed_family_count", 0))
        if "worst_family_delta" in evidence:
            summary["prior_retained_worst_family_delta"] = float(evidence.get("worst_family_delta", 0.0))
    return summary


def is_runtime_managed_artifact_path(path: str) -> bool:
    normalized = str(path).strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    return not (lowered.startswith("/tmp/") or "pytest-" in lowered or "/tests/" in lowered)


def production_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if str(record.get("state", "")) in {"retain", "reject"}
        and is_runtime_managed_artifact_path(str(record.get("artifact_path", "")))
    ]


def yield_summary_for(records: list[dict[str, object]]) -> dict[str, object]:
    retained = [record for record in records if str(record.get("state", "")) == "retain"]
    rejected = [record for record in records if str(record.get("state", "")) == "reject"]
    retained_by_subsystem: dict[str, int] = {}
    rejected_by_subsystem: dict[str, int] = {}
    for record in retained:
        key = str(record.get("subsystem", "unknown"))
        retained_by_subsystem[key] = retained_by_subsystem.get(key, 0) + 1
    for record in rejected:
        key = str(record.get("subsystem", "unknown"))
        rejected_by_subsystem[key] = rejected_by_subsystem.get(key, 0) + 1

    def _average_delta(rows: list[dict[str, object]], *, baseline_key: str, candidate_key: str) -> float:
        deltas: list[float] = []
        for row in rows:
            metrics = row.get("metrics_summary", {})
            if not isinstance(metrics, dict):
                continue
            try:
                baseline = float(metrics.get(baseline_key, 0.0))
                candidate = float(metrics.get(candidate_key, 0.0))
            except (TypeError, ValueError):
                continue
            deltas.append(candidate - baseline)
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    return {
        "retained_cycles": len(retained),
        "rejected_cycles": len(rejected),
        "total_decisions": len(records),
        "retained_by_subsystem": retained_by_subsystem,
        "rejected_by_subsystem": rejected_by_subsystem,
        "average_retained_pass_rate_delta": _average_delta(
            retained,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        "average_retained_step_delta": _average_delta(
            retained,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
        "average_rejected_pass_rate_delta": _average_delta(
            rejected,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        "average_rejected_step_delta": _average_delta(
            rejected,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
    }


def cycle_id_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._") or "cycle"


def phase_gate_metrics_summary(report: dict[str, object]) -> dict[str, object]:
    failures = report.get("failures", [])
    if not isinstance(failures, list):
        failures = []
    normalized_failures = [str(failure) for failure in failures if str(failure).strip()]
    return {
        "phase_gate_passed": bool(report.get("passed", False)),
        "phase_gate_failures": normalized_failures,
        "phase_gate_failure_count": len(normalized_failures),
        "phase_gate_generated_lane_included": bool(report.get("generated_lane_included", False)),
        "phase_gate_failure_recovery_lane_included": bool(report.get("failure_recovery_lane_included", False)),
    }


def decision_state_for_cycle_report(
    *,
    final_state: str,
    final_reason: str,
    preview_reason_code: str = "",
    decision_reason_code: str = "",
) -> dict[str, object]:
    retention_state = str(final_state).strip()
    if retention_state not in {"retain", "reject", "incomplete"}:
        retention_state = "undecided"
    retention_basis = (
        str(decision_reason_code).strip()
        or str(preview_reason_code).strip()
        or str(final_reason).strip()
        or retention_state
    )
    return {
        "decision_owner": "child_native",
        "decision_credit": "child_native",
        "decision_conversion_state": "runtime_managed" if retention_state in {"retain", "reject"} else "incomplete",
        "retention_state": retention_state,
        "retention_basis": retention_basis,
        "closeout_mode": "natural",
        "controller_intervention_reason_code": "",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def write_cycle_report(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    cycle_id: str,
    subsystem: str,
    artifact_path: Path,
    final_state: str,
    final_reason: str,
    artifact_update: dict[str, object],
    evidence: dict[str, object],
    baseline,
    candidate,
    phase_gate_report: dict[str, object],
    prior_retained_comparison: dict[str, object] | None = None,
    prior_retained_guard_reason: str = "",
    prior_retained_guard_reason_code: str = "",
    preview_reason_code: str = "",
    decision_reason_code: str = "",
    promotion_block_reason_code: str = "",
    protocol_match_id: str = "",
    strategy_candidate: dict[str, object] | None = None,
    tolbert_runtime_summary: dict[str, object] | None = None,
) -> Path:
    records = planner.load_cycle_records(config.improvement_cycles_path)
    cycle_records = [record for record in records if str(record.get("cycle_id", "")) == cycle_id]
    managed_decisions = production_decisions(records)
    summary = planner.retained_gain_summary(config.improvement_cycles_path)
    safe_cycle_id = cycle_id_safe(cycle_id)
    report_path = config.improvement_reports_dir / f"cycle_report_{safe_cycle_id}.json"
    report = {
        "spec_version": "asi_v1",
        "report_kind": "improvement_cycle_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "cycle_id": cycle_id,
        "subsystem": subsystem,
        "strategy_candidate": dict(strategy_candidate or {}),
        "strategy_candidate_id": str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
        "strategy_candidate_kind": str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
        "strategy_origin": str(
            dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
        ).strip(),
        "artifact_path": str(artifact_path),
        "candidate_artifact_path": str(artifact_update.get("candidate_artifact_path", "")),
        "active_artifact_path": str(artifact_update.get("active_artifact_path", "")),
        "artifact_kind": str(artifact_update.get("artifact_kind", "")),
        "final_state": final_state,
        "final_reason": final_reason,
        "decision_state": decision_state_for_cycle_report(
            final_state=final_state,
            final_reason=final_reason,
            preview_reason_code=preview_reason_code,
            decision_reason_code=decision_reason_code,
        ),
        "promotion_blocked": str(final_state).strip() != "retain",
        "promotion_block_reason_code": str(promotion_block_reason_code).strip(),
        "prior_retained_guard_reason": str(prior_retained_guard_reason).strip(),
        "prior_retained_guard_reason_code": str(prior_retained_guard_reason_code).strip()
        or str(promotion_block_reason_code).strip(),
        "preview_reason_code": str(preview_reason_code).strip(),
        "decision_reason_code": str(decision_reason_code).strip(),
        "artifact_lifecycle_state": str(artifact_update.get("artifact_lifecycle_state", "")),
        "artifact_sha256": str(artifact_update.get("artifact_sha256", "")),
        "previous_artifact_sha256": str(artifact_update.get("previous_artifact_sha256", "")),
        "rollback_artifact_path": str(artifact_update.get("rollback_artifact_path", "")),
        "artifact_snapshot_path": str(artifact_update.get("artifact_snapshot_path", "")),
        "candidate_isolation_summary": {
            "candidate_artifact_path": str(artifact_update.get("candidate_artifact_path", "")),
            "active_artifact_path": str(artifact_update.get("active_artifact_path", "")),
            "candidate_artifact_snapshot_path": str(artifact_update.get("candidate_artifact_snapshot_path", "")),
            "active_artifact_snapshot_path": str(artifact_update.get("active_artifact_snapshot_path", "")),
            "paths_are_distinct": bool(
                str(artifact_update.get("candidate_artifact_path", "")).strip()
                and str(artifact_update.get("active_artifact_path", "")).strip()
                and str(artifact_update.get("candidate_artifact_path", "")).strip()
                != str(artifact_update.get("active_artifact_path", "")).strip()
            ),
            "runtime_managed_artifact_path": is_runtime_managed_artifact_path(
                str(artifact_update.get("active_artifact_path", ""))
            ),
        },
        "compatibility": dict(artifact_update.get("compatibility", {})),
        "baseline_metrics": {
            "pass_rate": baseline.pass_rate,
            "average_steps": baseline.average_steps,
            "generated_pass_rate": baseline.generated_pass_rate,
            "proposal_selected_steps": baseline.proposal_selected_steps,
            "novel_valid_command_steps": baseline.novel_valid_command_steps,
            "novel_valid_command_rate": baseline.novel_valid_command_rate,
            "tolbert_primary_episodes": baseline.tolbert_primary_episodes,
        },
        "candidate_metrics": {
            "pass_rate": candidate.pass_rate,
            "average_steps": candidate.average_steps,
            "generated_pass_rate": candidate.generated_pass_rate,
            "proposal_selected_steps": candidate.proposal_selected_steps,
            "novel_valid_command_steps": candidate.novel_valid_command_steps,
            "novel_valid_command_rate": candidate.novel_valid_command_rate,
            "tolbert_primary_episodes": candidate.tolbert_primary_episodes,
        },
        "phase_gate_report": {
            "passed": bool(phase_gate_report.get("passed", False)),
            "failures": [str(failure) for failure in phase_gate_report.get("failures", []) if str(failure).strip()],
            "generated_lane_included": bool(phase_gate_report.get("generated_lane_included", False)),
            "failure_recovery_lane_included": bool(phase_gate_report.get("failure_recovery_lane_included", False)),
        },
        "prior_retained_comparison": dict(prior_retained_comparison or {}),
        "tolbert_runtime_summary": dict(tolbert_runtime_summary or {}),
        "evidence": evidence,
        "yield_summary": {
            "retained_cycles": summary.retained_cycles,
            "rejected_cycles": summary.rejected_cycles,
            "total_decisions": summary.total_decisions,
            "retained_by_subsystem": summary.retained_by_subsystem,
            "rejected_by_subsystem": summary.rejected_by_subsystem,
            "average_retained_pass_rate_delta": summary.average_retained_pass_rate_delta,
            "average_retained_step_delta": summary.average_retained_step_delta,
            "average_rejected_pass_rate_delta": summary.average_rejected_pass_rate_delta,
            "average_rejected_step_delta": summary.average_rejected_step_delta,
        },
        "production_yield_summary": yield_summary_for(managed_decisions),
        "current_cycle_records": cycle_records,
    }
    strategy_node = finalize_strategy_node(config, report)
    report["strategy_memory"] = {
        "strategy_node_id": strategy_node.strategy_node_id,
        "parent_strategy_node_ids": list(strategy_node.parent_strategy_node_ids),
        "retention_state": strategy_node.retention_state,
        "retained_gain": float(strategy_node.retained_gain),
        "analysis_lesson": strategy_node.analysis_lesson,
        "reuse_conditions": list(strategy_node.reuse_conditions),
        "avoid_conditions": list(strategy_node.avoid_conditions),
        "execution_evidence": dict(strategy_node.execution_evidence),
    }
    atomic_write_json(report_path, report, config=config)
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="record",
            subsystem=subsystem,
            action="write_cycle_report",
            artifact_path=str(report_path),
            artifact_kind="improvement_cycle_report",
            reason="persisted single-cycle improvement evidence report",
            metrics_summary={
                "final_state": final_state,
                "preview_reason_code": preview_reason_code,
                "decision_reason_code": decision_reason_code,
                **phase_gate_metrics_summary(phase_gate_report),
                "production_total_decisions": report["production_yield_summary"]["total_decisions"],
                "production_retained_cycles": report["production_yield_summary"]["retained_cycles"],
                "production_rejected_cycles": report["production_yield_summary"]["rejected_cycles"],
                "protocol": "autonomous",
                "protocol_match_id": str(protocol_match_id).strip(),
                "strategy_candidate_id": str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                "strategy_candidate_kind": str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                "strategy_origin": str(
                    dict(strategy_candidate or {}).get(
                        "origin",
                        dict(strategy_candidate or {}).get("strategy_origin", ""),
                    )
                ).strip(),
            },
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            strategy_origin=str(
                dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
            ).strip(),
        ),
    )
    return report_path
