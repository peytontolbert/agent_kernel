from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
import inspect
import json
import math
import re

from evals.harness import compare_abstraction_transfer_modes, run_eval, scoped_eval_config

from .config import KernelConfig
from .runtime_supervision import atomic_write_json
from .strategy_memory import finalize_strategy_node
from .improvement import (
    _generated_kind_pass_rate,
    _has_generated_kind,
    artifact_sha256,
    effective_artifact_payload_for_retention,
    ImprovementCycleRecord,
    ImprovementPlanner,
    apply_artifact_retention_decision,
    evaluate_artifact_retention,
    payload_with_active_artifact_context,
    persist_replay_verified_tool_artifact,
    proposal_gate_failure_reason,
    retention_gate_for_payload,
    retention_evidence,
)
from .modeling.evaluation.liftoff import build_liftoff_gate_report
from .subsystems import (
    active_artifact_path_for_subsystem,
    baseline_candidate_flags,
    base_subsystem_for,
    comparison_config_for_subsystem_artifact,
    config_with_subsystem_artifact_path,
)
from .tolbert_assets import materialize_retained_retrieval_asset_bundle


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


def _candidate_matches_active_artifact(candidate_artifact_path: Path, active_artifact_path: Path) -> bool:
    if candidate_artifact_path == active_artifact_path:
        return False
    if not candidate_artifact_path.exists() or not active_artifact_path.exists():
        return False
    try:
        candidate_payload = json.loads(candidate_artifact_path.read_text(encoding="utf-8"))
        active_payload = json.loads(active_artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        candidate_payload = None
        active_payload = None
    if isinstance(candidate_payload, dict) and isinstance(active_payload, dict):
        low_signal_keys = {"artifact_kind", "spec_version", "lifecycle_state"}
        candidate_identity_keys = {str(key).strip() for key in candidate_payload if str(key).strip()} - low_signal_keys
        active_identity_keys = {str(key).strip() for key in active_payload if str(key).strip()} - low_signal_keys
        if not candidate_identity_keys or not active_identity_keys:
            return False
    candidate_sha256 = artifact_sha256(candidate_artifact_path)
    active_sha256 = artifact_sha256(active_artifact_path)
    return bool(candidate_sha256) and candidate_sha256 == active_sha256


def comparison_flags(subsystem: str, *, config: KernelConfig | None = None) -> dict[str, bool]:
    capability_modules_path = None if config is None else config.capability_modules_path
    _, candidate = baseline_candidate_flags(subsystem, capability_modules_path)
    return dict(candidate)


def _resolved_runtime_eval_flag(
    flags: dict[str, bool],
    key: str,
    *,
    auto_enable: bool = False,
) -> bool:
    if key in flags:
        return bool(flags.get(key, False))
    return bool(auto_enable)


def autonomous_runtime_eval_flags(config: KernelConfig, flags: dict[str, bool]) -> dict[str, bool]:
    enriched = dict(flags)
    enriched["include_discovered_tasks"] = _resolved_runtime_eval_flag(
        enriched,
        "include_discovered_tasks",
    )
    enriched["include_episode_memory"] = _resolved_runtime_eval_flag(
        enriched,
        "include_episode_memory",
    )
    enriched["include_skill_memory"] = _resolved_runtime_eval_flag(
        enriched,
        "include_skill_memory",
        auto_enable=config.skills_path.exists(),
    )
    enriched["include_skill_transfer"] = _resolved_runtime_eval_flag(
        enriched,
        "include_skill_transfer",
        auto_enable=config.skills_path.exists(),
    )
    enriched["include_operator_memory"] = _resolved_runtime_eval_flag(
        enriched,
        "include_operator_memory",
        auto_enable=config.operator_classes_path.exists(),
    )
    enriched["include_tool_memory"] = _resolved_runtime_eval_flag(
        enriched,
        "include_tool_memory",
        auto_enable=config.tool_candidates_path.exists(),
    )
    enriched["include_verifier_memory"] = _resolved_runtime_eval_flag(
        enriched,
        "include_verifier_memory",
        auto_enable=config.trajectories_root.exists(),
    )
    enriched["include_benchmark_candidates"] = _resolved_runtime_eval_flag(
        enriched,
        "include_benchmark_candidates",
        auto_enable=config.benchmark_candidates_path.exists(),
    )
    enriched["include_verifier_candidates"] = _resolved_runtime_eval_flag(
        enriched,
        "include_verifier_candidates",
        auto_enable=config.verifier_contracts_path.exists(),
    )
    enriched["include_generated"] = _resolved_runtime_eval_flag(enriched, "include_generated")
    enriched["include_failure_generated"] = _resolved_runtime_eval_flag(enriched, "include_failure_generated")
    return enriched


def evaluate_subsystem_metrics(
    *,
    config: KernelConfig,
    subsystem: str,
    flags: dict[str, object],
    progress_label: str | None = None,
):
    task_limit = flags.get("task_limit")
    if not isinstance(task_limit, int) or task_limit <= 0:
        task_limit = None
    if base_subsystem_for(subsystem, config.capability_modules_path) == "operators":
        return compare_abstraction_transfer_modes(
            config=config,
            include_discovered_tasks=flags["include_discovered_tasks"],
            include_episode_memory=flags["include_episode_memory"],
            include_verifier_memory=flags["include_verifier_memory"],
            include_benchmark_candidates=flags["include_benchmark_candidates"],
            include_verifier_candidates=flags["include_verifier_candidates"],
            include_generated=flags["include_generated"],
            include_failure_generated=flags["include_failure_generated"],
            task_limit=task_limit,
            progress_label_prefix=progress_label,
        ).operator_metrics
    return run_eval(config=config, progress_label=progress_label, **flags)


def _is_retryable_tolbert_startup_failure(error_text: str) -> bool:
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return (
        "tolbert service failed to become ready" in normalized
        or "tolbert service exited before startup ready" in normalized
    )


def _new_tolbert_runtime_summary(*, use_tolbert_context: bool) -> dict[str, object]:
    return {
        "configured_to_use_tolbert": bool(use_tolbert_context),
        "stages_attempted": [],
        "successful_tolbert_stages": [],
        "startup_failure_stages": [],
        "recovered_without_tolbert_stages": [],
        "bypassed_stages": [],
        "startup_failure_count": 0,
        "outcome": "pending" if bool(use_tolbert_context) else "bypassed",
    }


def _ordered_unique_stage_names(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _normalize_tolbert_runtime_summary(
    summary: Mapping[str, object] | None,
    *,
    use_tolbert_context: bool | None = None,
) -> dict[str, object]:
    payload = dict(summary) if isinstance(summary, Mapping) else {}
    configured = (
        bool(payload.get("configured_to_use_tolbert", False))
        if use_tolbert_context is None
        else bool(use_tolbert_context)
    )
    normalized = _new_tolbert_runtime_summary(use_tolbert_context=configured)
    for key in (
        "stages_attempted",
        "successful_tolbert_stages",
        "startup_failure_stages",
        "recovered_without_tolbert_stages",
        "bypassed_stages",
    ):
        normalized[key] = _ordered_unique_stage_names(payload.get(key, []))
    normalized["startup_failure_count"] = len(normalized["startup_failure_stages"])
    if payload.get("outcome"):
        normalized["outcome"] = str(payload.get("outcome", "")).strip()
    return _finalize_tolbert_runtime_summary(normalized)


def _mark_tolbert_stage(summary: dict[str, object] | None, key: str, stage_name: str) -> None:
    if not isinstance(summary, dict):
        return
    token = str(stage_name).strip()
    if not token:
        return
    values = summary.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        summary[key] = values
    if token not in values:
        values.append(token)


def _finalize_tolbert_runtime_summary(summary: Mapping[str, object] | None) -> dict[str, object]:
    normalized = _new_tolbert_runtime_summary(
        use_tolbert_context=bool(dict(summary or {}).get("configured_to_use_tolbert", False))
    )
    normalized.update(dict(summary or {}))
    for key in (
        "stages_attempted",
        "successful_tolbert_stages",
        "startup_failure_stages",
        "recovered_without_tolbert_stages",
        "bypassed_stages",
    ):
        normalized[key] = _ordered_unique_stage_names(normalized.get(key, []))
    normalized["startup_failure_count"] = len(normalized["startup_failure_stages"])
    configured = bool(normalized.get("configured_to_use_tolbert", False))
    if normalized["startup_failure_stages"]:
        outcome = "failed_recovered"
    elif normalized["successful_tolbert_stages"]:
        outcome = "succeeded"
    elif normalized["bypassed_stages"] or not configured:
        outcome = "bypassed"
    elif normalized["stages_attempted"]:
        outcome = "pending"
    else:
        outcome = "not_exercised"
    normalized["outcome"] = outcome
    normalized["used_tolbert_successfully"] = bool(normalized["successful_tolbert_stages"])
    normalized["recovered_without_tolbert"] = bool(normalized["recovered_without_tolbert_stages"])
    normalized["bypassed"] = outcome == "bypassed"
    return normalized


def _evaluate_subsystem_metrics_with_tolbert_startup_retry(
    *,
    config: KernelConfig,
    subsystem: str,
    flags: dict[str, object],
    progress_label: str | None,
    phase_name: str,
    progress: Callable[[str], None] | None,
    tolbert_runtime_summary: dict[str, object] | None = None,
):
    if isinstance(tolbert_runtime_summary, dict):
        tolbert_runtime_summary["configured_to_use_tolbert"] = bool(config.use_tolbert_context)
        _mark_tolbert_stage(tolbert_runtime_summary, "stages_attempted", phase_name)
    try:
        result = evaluate_subsystem_metrics(
            config=config,
            subsystem=subsystem,
            flags=flags,
            progress_label=progress_label,
        )
        if bool(config.use_tolbert_context):
            _mark_tolbert_stage(tolbert_runtime_summary, "successful_tolbert_stages", phase_name)
        else:
            _mark_tolbert_stage(tolbert_runtime_summary, "bypassed_stages", phase_name)
        return result
    except RuntimeError as exc:
        if not bool(config.use_tolbert_context) or not _is_retryable_tolbert_startup_failure(str(exc)):
            raise
        _mark_tolbert_stage(tolbert_runtime_summary, "startup_failure_stages", phase_name)
        if progress is not None:
            progress(
                f"finalize phase={phase_name}_retry subsystem={subsystem} "
                "reason=tolbert_startup_failure use_tolbert_context=0"
            )
        _mark_tolbert_stage(tolbert_runtime_summary, "recovered_without_tolbert_stages", phase_name)
        _mark_tolbert_stage(tolbert_runtime_summary, "bypassed_stages", phase_name)
        return evaluate_subsystem_metrics(
            config=replace(config, use_tolbert_context=False),
            subsystem=subsystem,
            flags=flags,
            progress_label=progress_label,
        )


def _call_tolbert_preview_eval(
    *,
    config: KernelConfig,
    subsystem: str,
    flags: dict[str, object],
    progress_label: str | None,
    phase_name: str,
    progress: Callable[[str], None] | None,
    tolbert_runtime_summary: dict[str, object] | None = None,
):
    retry_evaluator = _evaluate_subsystem_metrics_with_tolbert_startup_retry
    parameters = inspect.signature(retry_evaluator).parameters
    kwargs = {
        "config": config,
        "subsystem": subsystem,
        "flags": flags,
        "progress_label": progress_label,
        "phase_name": phase_name,
        "progress": progress,
    }
    if "tolbert_runtime_summary" in parameters:
        kwargs["tolbert_runtime_summary"] = tolbert_runtime_summary
    return retry_evaluator(**kwargs)


def _retention_eval_config(
    *,
    base_config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
    scope: str,
) -> KernelConfig:
    scoped = scoped_eval_config(
        base_config,
        scope,
        trajectories_root=base_config.trajectories_root,
        persist_episode_memory=False,
    )
    preview_runtime_root = scoped.improvement_reports_dir / "preview_runtime"
    scoped = replace(
        scoped,
        run_checkpoints_dir=preview_runtime_root / "checkpoints",
        unattended_workspace_snapshot_root=preview_runtime_root / "workspace_snapshots",
    )
    scoped.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scoped.unattended_workspace_snapshot_root.mkdir(parents=True, exist_ok=True)
    return comparison_config_for_subsystem_artifact(scoped, subsystem, artifact_path)


def _comparison_task_limit_for_retention(
    subsystem: str,
    *,
    task_limit: int | None,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> int | None:
    preview_comparison_task_limit_cap = 16
    retrieval_preview_comparison_task_limit_cap = 8
    retrieval_bounded_compare_fallback = 8
    if not isinstance(task_limit, int) or task_limit <= 0:
        task_limit = None
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval" or not isinstance(payload, dict):
        if task_limit is None:
            return None
        # Retention preview already follows breadth-aware observe plus holdout gates.
        # Cap non-retrieval preview slices so autonomous cycles can reach decisions in bounded time.
        return min(task_limit, preview_comparison_task_limit_cap)
    preview_controls = payload.get("preview_controls", {})
    if isinstance(preview_controls, dict):
        comparison_task_limit_floor = preview_controls.get("comparison_task_limit_floor")
        try:
            comparison_task_limit_floor = int(comparison_task_limit_floor)
        except (TypeError, ValueError):
            comparison_task_limit_floor = 0
        if comparison_task_limit_floor > 0:
            bounded_floor = max(retrieval_preview_comparison_task_limit_cap, comparison_task_limit_floor)
            if task_limit is None:
                return bounded_floor
            return max(task_limit, bounded_floor)
    gate = retention_gate_for_payload(
        subsystem,
        payload,
        capability_modules_path=capability_modules_path,
    )
    if bool(gate.get("require_trusted_carryover_repair_improvement", False)):
        bounded_limit = max(retrieval_preview_comparison_task_limit_cap, retrieval_bounded_compare_fallback)
        if task_limit is None:
            return bounded_limit
        return max(task_limit, bounded_limit)
    if task_limit is None:
        return retrieval_preview_comparison_task_limit_cap
    return min(task_limit, retrieval_preview_comparison_task_limit_cap)


def _holdout_task_limit_for_retention(
    subsystem: str,
    *,
    comparison_task_limit: int | None,
    capability_modules_path: Path | None = None,
) -> int | None:
    if not isinstance(comparison_task_limit, int) or comparison_task_limit <= 0:
        return None
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval":
        return None
    return max(4, comparison_task_limit)


def _holdout_generated_schedule_limit_for_retention(
    subsystem: str,
    *,
    comparison_task_limit: int | None,
    capability_modules_path: Path | None = None,
) -> int:
    holdout_task_limit = _holdout_task_limit_for_retention(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=capability_modules_path,
    )
    if not isinstance(holdout_task_limit, int) or holdout_task_limit <= 0:
        return 0
    return holdout_task_limit


def _retrieval_preview_priority_overrides(
    payload: dict[str, object] | None,
) -> tuple[list[str], dict[str, float]]:
    if not isinstance(payload, dict):
        return [], {}
    preview_controls = payload.get("preview_controls", {})
    if not isinstance(preview_controls, dict):
        return [], {}
    families = [
        str(family).strip()
        for family in list(preview_controls.get("priority_benchmark_families", []))
        if str(family).strip()
    ]
    weights_value = preview_controls.get("priority_benchmark_family_weights", {})
    weights: dict[str, float] = {}
    if isinstance(weights_value, dict):
        for family, weight in weights_value.items():
            normalized_family = str(family).strip()
            if not normalized_family:
                continue
            try:
                weights[normalized_family] = float(weight)
            except (TypeError, ValueError):
                continue
    return families, weights


def _retrieval_bounded_preview_required(
    subsystem: str,
    *,
    payload: dict[str, object] | None,
    capability_modules_path: Path | None = None,
) -> bool:
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval" or not isinstance(payload, dict):
        return False
    preview_controls = payload.get("preview_controls", {})
    if not isinstance(preview_controls, dict):
        return False
    return bool(preview_controls.get("bounded_comparison_required", False))


def _apply_retrieval_bounded_preview_filters(
    subsystem: str,
    *,
    flags: dict[str, object],
    payload: dict[str, object] | None,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    scoped_flags = dict(flags)
    if not _retrieval_bounded_preview_required(
        subsystem,
        payload=payload,
        capability_modules_path=capability_modules_path,
    ):
        return scoped_flags
    # Bounded retrieval previews need discriminative non-replay slices rather than replay-family drift.
    scoped_flags["include_episode_memory"] = False
    scoped_flags["include_verifier_memory"] = False
    return scoped_flags


def _merge_priority_families(
    explicit_families: list[str] | None,
    explicit_weights: dict[str, float] | None,
    payload: dict[str, object] | None,
    *,
    allow_payload_overrides: bool = True,
) -> tuple[list[str], dict[str, float]]:
    merged_families: list[str] = []
    merged_weights: dict[str, float] = {}
    for family in list(explicit_families or []):
        normalized = str(family).strip()
        if not normalized:
            continue
        if normalized not in merged_families:
            merged_families.append(normalized)
    for family, weight in dict(explicit_weights or {}).items():
        normalized = str(family).strip()
        if not normalized:
            continue
        try:
            merged_weights[normalized] = float(weight)
        except (TypeError, ValueError):
            continue
        if normalized not in merged_families:
            merged_families.append(normalized)
    if allow_payload_overrides:
        retrieval_families, retrieval_weights = _retrieval_preview_priority_overrides(payload)
        for family in retrieval_families:
            if family not in merged_families:
                merged_families.append(family)
        for family, weight in retrieval_weights.items():
            merged_weights[family] = max(weight, float(merged_weights.get(family, 0.0) or 0.0))
            if family not in merged_families:
                merged_families.append(family)
    return merged_families, merged_weights


def confirmation_confidence_report(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    gate: dict[str, object] | None = None,
) -> dict[str, object]:
    gate = dict(gate or {})
    baseline_pass_rates = [float(run.pass_rate) for run in baseline_runs]
    candidate_pass_rates = [float(run.pass_rate) for run in candidate_runs]
    baseline_step_deltas = [float(run.average_steps) for run in baseline_runs]
    candidate_step_deltas = [float(run.average_steps) for run in candidate_runs]
    baseline_passed = sum(int(run.passed) for run in baseline_runs)
    baseline_total = sum(max(0, int(run.total)) for run in baseline_runs)
    candidate_passed = sum(int(run.passed) for run in candidate_runs)
    candidate_total = sum(max(0, int(run.total)) for run in candidate_runs)
    baseline_rate = 0.0 if baseline_total <= 0 else baseline_passed / baseline_total
    candidate_rate = 0.0 if candidate_total <= 0 else candidate_passed / candidate_total
    z = max(0.0, float(gate.get("confirmation_confidence_z", 1.0)))
    baseline_var = 0.0 if baseline_total <= 0 else baseline_rate * (1.0 - baseline_rate) / baseline_total
    candidate_var = 0.0 if candidate_total <= 0 else candidate_rate * (1.0 - candidate_rate) / candidate_total
    delta_stderr = math.sqrt(max(0.0, baseline_var + candidate_var))
    delta_mean = candidate_rate - baseline_rate
    mean_step_delta = (
        0.0
        if not baseline_step_deltas or not candidate_step_deltas
        else sum(candidate - baseline for baseline, candidate in zip(baseline_step_deltas, candidate_step_deltas))
        / min(len(baseline_step_deltas), len(candidate_step_deltas))
    )
    step_deltas = [
        float(candidate - baseline)
        for baseline, candidate in zip(baseline_step_deltas, candidate_step_deltas)
    ]
    step_delta_stderr = 0.0
    if len(step_deltas) >= 2:
        step_delta_mean = sum(step_deltas) / len(step_deltas)
        variance = sum((delta - step_delta_mean) ** 2 for delta in step_deltas) / (len(step_deltas) - 1)
        step_delta_stderr = math.sqrt(max(0.0, variance) / len(step_deltas))
    baseline_wilson_lower, baseline_wilson_upper = _wilson_interval(baseline_passed, baseline_total, z=z)
    candidate_wilson_lower, candidate_wilson_upper = _wilson_interval(candidate_passed, candidate_total, z=z)
    paired_deltas = [
        float(candidate.pass_rate - baseline.pass_rate)
        for baseline, candidate in zip(baseline_runs, candidate_runs)
    ]
    paired_non_regression_rate = (
        0.0 if not paired_deltas else sum(1 for delta in paired_deltas if delta >= 0.0) / len(paired_deltas)
    )
    paired_improvement_rate = (
        0.0 if not paired_deltas else sum(1 for delta in paired_deltas if delta > 0.0) / len(paired_deltas)
    )
    paired_task_report = _paired_task_trace_report(baseline_runs, candidate_runs, z=z)
    paired_trajectory_report = _paired_trajectory_report(baseline_runs, candidate_runs, z=z)
    family_bounds = _family_conservative_bounds(baseline_runs, candidate_runs, z=z)
    return {
        "run_count": min(len(baseline_runs), len(candidate_runs)),
        "baseline_pass_rate_mean": baseline_rate,
        "candidate_pass_rate_mean": candidate_rate,
        "pass_rate_delta_mean": delta_mean,
        "pass_rate_delta_stderr": delta_stderr,
        "pass_rate_delta_lower_bound": delta_mean - (z * delta_stderr),
        "pass_rate_delta_upper_bound": delta_mean + (z * delta_stderr),
        "baseline_pass_rate_spread": 0.0 if not baseline_pass_rates else max(baseline_pass_rates) - min(baseline_pass_rates),
        "candidate_pass_rate_spread": 0.0 if not candidate_pass_rates else max(candidate_pass_rates) - min(candidate_pass_rates),
        "baseline_pass_rate_wilson_lower": baseline_wilson_lower,
        "baseline_pass_rate_wilson_upper": baseline_wilson_upper,
        "candidate_pass_rate_wilson_lower": candidate_wilson_lower,
        "candidate_pass_rate_wilson_upper": candidate_wilson_upper,
        "pass_rate_delta_conservative_lower_bound": candidate_wilson_lower - baseline_wilson_upper,
        "paired_non_regression_rate": paired_non_regression_rate,
        "paired_improvement_rate": paired_improvement_rate,
        **paired_task_report,
        **paired_trajectory_report,
        "mean_step_delta": mean_step_delta,
        "step_delta_stderr": step_delta_stderr,
        "step_delta_upper_bound": mean_step_delta + (z * step_delta_stderr),
        "step_delta_lower_bound": mean_step_delta - (z * step_delta_stderr),
        "step_delta_spread": (
            0.0
            if not baseline_step_deltas or not candidate_step_deltas
            else max(candidate - baseline for baseline, candidate in zip(baseline_step_deltas, candidate_step_deltas))
            - min(candidate - baseline for baseline, candidate in zip(baseline_step_deltas, candidate_step_deltas))
        ),
        "family_conservative_bounds": family_bounds,
        "worst_family_conservative_lower_bound": min(
            (float(item.get("delta_conservative_lower_bound", 0.0)) for item in family_bounds.values()),
            default=0.0,
        ),
        "regressed_family_conservative_count": sum(
            1 for item in family_bounds.values() if float(item.get("delta_conservative_lower_bound", 0.0)) < 0.0
        ),
    }


def confirmation_confidence_failures(report: dict[str, object], *, gate: dict[str, object] | None = None) -> list[str]:
    gate = dict(gate or {})
    failures: list[str] = []
    run_count = int(report.get("run_count", 0))
    max_stderr = float(gate.get("max_confirmation_pass_rate_delta_stderr", 0.12))
    max_spread = float(gate.get("max_confirmation_pass_rate_spread", 0.15))
    max_step_spread = float(gate.get("max_confirmation_step_delta_spread", 1.5))
    min_lower_bound = float(gate.get("min_confirmation_pass_rate_delta_lower_bound", -0.05))
    min_conservative_lower_bound = float(
        gate.get("min_confirmation_pass_rate_delta_conservative_lower_bound", min_lower_bound)
    )
    max_step_upper_bound = float(gate.get("max_confirmation_step_delta_upper_bound", 0.25))
    min_paired_non_regression_rate = float(gate.get("min_confirmation_paired_non_regression_rate", 0.0))
    min_confirmation_paired_task_pair_count = int(gate.get("min_confirmation_paired_task_pair_count", 0) or 0)
    min_paired_task_non_regression_rate_lower_bound = float(
        gate.get("min_confirmation_paired_task_non_regression_rate_lower_bound", 0.0)
    )
    max_confirmation_paired_task_non_regression_p_value = gate.get(
        "max_confirmation_paired_task_non_regression_p_value"
    )
    min_confirmation_paired_trace_pair_count = int(gate.get("min_confirmation_paired_trace_pair_count", 0) or 0)
    min_confirmation_paired_trace_non_regression_rate_lower_bound = float(
        gate.get("min_confirmation_paired_trace_non_regression_rate_lower_bound", 0.0)
    )
    max_confirmation_paired_trace_non_regression_p_value = gate.get(
        "max_confirmation_paired_trace_non_regression_p_value"
    )
    min_confirmation_paired_trajectory_pair_count = int(
        gate.get("min_confirmation_paired_trajectory_pair_count", 0) or 0
    )
    min_confirmation_paired_trajectory_non_regression_rate_lower_bound = float(
        gate.get("min_confirmation_paired_trajectory_non_regression_rate_lower_bound", 0.0)
    )
    max_confirmation_paired_trajectory_non_regression_p_value = gate.get(
        "max_confirmation_paired_trajectory_non_regression_p_value"
    )
    min_confirmation_paired_trajectory_exact_match_rate_lower_bound = float(
        gate.get("min_confirmation_paired_trajectory_exact_match_rate_lower_bound", 0.0)
    )
    min_worst_family_conservative_lower_bound = gate.get("min_confirmation_worst_family_conservative_lower_bound")
    max_regressed_family_conservative_count = gate.get("max_confirmation_regressed_family_conservative_count")
    max_confirmation_regressed_task_count = gate.get("max_confirmation_regressed_task_count")
    max_confirmation_regressed_trace_task_count = gate.get("max_confirmation_regressed_trace_task_count")
    max_confirmation_regressed_trajectory_task_count = gate.get("max_confirmation_regressed_trajectory_task_count")
    if (
        float(report.get("pass_rate_delta_stderr", 0.0)) > max_stderr
        and (run_count >= 3 or "max_confirmation_pass_rate_delta_stderr" in gate)
    ):
        failures.append("confirmation pass-rate uncertainty remained too high")
    if (
        float(report.get("baseline_pass_rate_spread", 0.0)) > max_spread
        and (run_count >= 3 or "max_confirmation_pass_rate_spread" in gate)
    ):
        failures.append("baseline confirmation pass-rate spread remained too high")
    if (
        float(report.get("candidate_pass_rate_spread", 0.0)) > max_spread
        and (run_count >= 3 or "max_confirmation_pass_rate_spread" in gate)
    ):
        failures.append("candidate confirmation pass-rate spread remained too high")
    if (
        float(report.get("step_delta_spread", 0.0)) > max_step_spread
        and (run_count >= 3 or "max_confirmation_step_delta_spread" in gate)
    ):
        failures.append("confirmation step-delta spread remained too high")
    if (
        float(report.get("pass_rate_delta_lower_bound", 0.0)) < min_lower_bound
        and (run_count >= 3 or "min_confirmation_pass_rate_delta_lower_bound" in gate)
    ):
        failures.append("candidate confirmation pass-rate lower bound remained too weak")
    if (
        float(report.get("pass_rate_delta_conservative_lower_bound", 0.0)) < min_conservative_lower_bound
        and (run_count >= 2 or "min_confirmation_pass_rate_delta_conservative_lower_bound" in gate)
    ):
        failures.append("candidate confirmation conservative pass-rate bound remained too weak")
    if (
        float(report.get("step_delta_upper_bound", 0.0)) > max_step_upper_bound
        and (run_count >= 2 or "max_confirmation_step_delta_upper_bound" in gate)
    ):
        failures.append("candidate confirmation step-delta upper bound remained too weak")
    if (
        float(report.get("paired_non_regression_rate", 0.0)) < min_paired_non_regression_rate
        and (run_count >= 2 or "min_confirmation_paired_non_regression_rate" in gate)
    ):
        failures.append("candidate confirmation paired non-regression rate remained too weak")
    if (
        int(report.get("paired_task_pair_count", 0) or 0) < min_confirmation_paired_task_pair_count
        and min_confirmation_paired_task_pair_count > 0
    ):
        failures.append("candidate confirmation paired task evidence remained too small")
    if (
        float(report.get("paired_task_non_regression_rate_lower_bound", 0.0))
        < min_paired_task_non_regression_rate_lower_bound
        and (run_count >= 2 or "min_confirmation_paired_task_non_regression_rate_lower_bound" in gate)
    ):
        failures.append("candidate confirmation paired task non-regression bound remained too weak")
    if (
        max_confirmation_paired_task_non_regression_p_value is not None
        and float(report.get("paired_task_non_regression_significance_p_value", 1.0))
        > float(max_confirmation_paired_task_non_regression_p_value)
        and int(report.get("paired_task_pair_count", 0) or 0) > 0
    ):
        failures.append("candidate confirmation paired task significance remained too weak")
    if (
        int(report.get("paired_trace_pair_count", 0) or 0) < min_confirmation_paired_trace_pair_count
        and min_confirmation_paired_trace_pair_count > 0
    ):
        failures.append("candidate confirmation paired trace evidence remained too small")
    if (
        float(report.get("paired_trace_non_regression_rate_lower_bound", 0.0))
        < min_confirmation_paired_trace_non_regression_rate_lower_bound
        and (run_count >= 2 or "min_confirmation_paired_trace_non_regression_rate_lower_bound" in gate)
    ):
        failures.append("candidate confirmation paired trace non-regression bound remained too weak")
    if (
        max_confirmation_paired_trace_non_regression_p_value is not None
        and float(report.get("paired_trace_non_regression_significance_p_value", 1.0))
        > float(max_confirmation_paired_trace_non_regression_p_value)
        and int(report.get("paired_trace_pair_count", 0) or 0) > 0
    ):
        failures.append("candidate confirmation paired trace significance remained too weak")
    if (
        int(report.get("paired_trajectory_pair_count", 0) or 0) < min_confirmation_paired_trajectory_pair_count
        and min_confirmation_paired_trajectory_pair_count > 0
    ):
        failures.append("candidate confirmation paired trajectory evidence remained too small")
    if (
        float(report.get("paired_trajectory_non_regression_rate_lower_bound", 0.0))
        < min_confirmation_paired_trajectory_non_regression_rate_lower_bound
        and (run_count >= 2 or "min_confirmation_paired_trajectory_non_regression_rate_lower_bound" in gate)
    ):
        failures.append("candidate confirmation paired trajectory non-regression bound remained too weak")
    if (
        max_confirmation_paired_trajectory_non_regression_p_value is not None
        and float(report.get("paired_trajectory_non_regression_significance_p_value", 1.0))
        > float(max_confirmation_paired_trajectory_non_regression_p_value)
        and int(report.get("paired_trajectory_pair_count", 0) or 0) > 0
    ):
        failures.append("candidate confirmation paired trajectory significance remained too weak")
    if (
        float(report.get("paired_trajectory_exact_match_rate_lower_bound", 0.0))
        < min_confirmation_paired_trajectory_exact_match_rate_lower_bound
        and (run_count >= 2 or "min_confirmation_paired_trajectory_exact_match_rate_lower_bound" in gate)
    ):
        failures.append("candidate confirmation paired trajectory exact-match bound remained too weak")
    if (
        min_worst_family_conservative_lower_bound is not None
        and float(report.get("worst_family_conservative_lower_bound", 0.0))
        < float(min_worst_family_conservative_lower_bound)
    ):
        failures.append("candidate confirmation family conservative lower bound remained too weak")
    if (
        max_regressed_family_conservative_count is not None
        and int(report.get("regressed_family_conservative_count", 0) or 0)
        > int(max_regressed_family_conservative_count)
    ):
        failures.append("candidate confirmation family conservative regression count remained too high")
    if (
        max_confirmation_regressed_task_count is not None
        and int(report.get("regressed_task_count", 0) or 0) > int(max_confirmation_regressed_task_count)
    ):
        failures.append("candidate confirmation regressed-task count remained too high")
    if (
        max_confirmation_regressed_trace_task_count is not None
        and int(report.get("regressed_trace_task_count", 0) or 0) > int(max_confirmation_regressed_trace_task_count)
    ):
        failures.append("candidate confirmation regressed-trace task count remained too high")
    if (
        max_confirmation_regressed_trajectory_task_count is not None
        and int(report.get("regressed_trajectory_task_count", 0) or 0)
        > int(max_confirmation_regressed_trajectory_task_count)
    ):
        failures.append("candidate confirmation regressed-trajectory task count remained too high")
    return failures


def _wilson_interval(passed: int, total: int, *, z: float) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = max(0.0, min(1.0, passed / total))
    denominator = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _one_sided_sign_test_p_value(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    successes = max(0, min(int(successes), int(trials)))
    trials = int(trials)
    if trials <= 200:
        denominator = float(2**trials)
        tail = 0.0
        for count in range(successes, trials + 1):
            tail += math.comb(trials, count) / denominator
        return min(1.0, max(0.0, tail))
    mean = trials * 0.5
    variance = trials * 0.25
    if variance <= 0.0:
        return 1.0
    z = ((successes - 0.5) - mean) / math.sqrt(variance)
    return min(1.0, max(0.0, 0.5 * math.erfc(z / math.sqrt(2.0))))


def _family_conservative_bounds(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    z: float,
) -> dict[str, dict[str, object]]:
    families: set[str] = set()
    for run in baseline_runs:
        families.update(str(key) for key in getattr(run, "total_by_benchmark_family", {}).keys())
        families.update(str(key) for key in getattr(run, "passed_by_benchmark_family", {}).keys())
    for run in candidate_runs:
        families.update(str(key) for key in getattr(run, "total_by_benchmark_family", {}).keys())
        families.update(str(key) for key in getattr(run, "passed_by_benchmark_family", {}).keys())
    bounds: dict[str, dict[str, object]] = {}
    for family in sorted(families):
        baseline_total = sum(int(getattr(run, "total_by_benchmark_family", {}).get(family, 0) or 0) for run in baseline_runs)
        candidate_total = sum(int(getattr(run, "total_by_benchmark_family", {}).get(family, 0) or 0) for run in candidate_runs)
        baseline_passed = sum(
            int(getattr(run, "passed_by_benchmark_family", {}).get(family, 0) or 0) for run in baseline_runs
        )
        candidate_passed = sum(
            int(getattr(run, "passed_by_benchmark_family", {}).get(family, 0) or 0) for run in candidate_runs
        )
        if baseline_total <= 0 and candidate_total <= 0:
            continue
        baseline_lower, baseline_upper = _wilson_interval(baseline_passed, baseline_total, z=z)
        candidate_lower, candidate_upper = _wilson_interval(candidate_passed, candidate_total, z=z)
        bounds[family] = {
            "baseline_total": baseline_total,
            "baseline_passed": baseline_passed,
            "candidate_total": candidate_total,
            "candidate_passed": candidate_passed,
            "baseline_wilson_lower": baseline_lower,
            "baseline_wilson_upper": baseline_upper,
            "candidate_wilson_lower": candidate_lower,
            "candidate_wilson_upper": candidate_upper,
            "delta_conservative_lower_bound": candidate_lower - baseline_upper,
        }
    return bounds


def _paired_task_trace_report(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    z: float,
) -> dict[str, object]:
    traces: dict[str, dict[str, object]] = {}
    for baseline_run, candidate_run in zip(baseline_runs, candidate_runs):
        baseline_outcomes = getattr(baseline_run, "task_outcomes", {}) or {}
        candidate_outcomes = getattr(candidate_run, "task_outcomes", {}) or {}
        if not isinstance(baseline_outcomes, dict) or not isinstance(candidate_outcomes, dict):
            continue
        for task_id in sorted(set(baseline_outcomes) & set(candidate_outcomes)):
            baseline_payload = baseline_outcomes.get(task_id, {})
            candidate_payload = candidate_outcomes.get(task_id, {})
            if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
                continue
            baseline_success = 1 if bool(baseline_payload.get("success", False)) else 0
            candidate_success = 1 if bool(candidate_payload.get("success", False)) else 0
            baseline_steps = int(baseline_payload.get("steps", 0) or 0)
            candidate_steps = int(candidate_payload.get("steps", 0) or 0)
            trace = traces.setdefault(
                str(task_id),
                {
                    "task_id": str(task_id),
                    "benchmark_family": str(
                        candidate_payload.get("benchmark_family", baseline_payload.get("benchmark_family", "bounded"))
                    ).strip()
                    or "bounded",
                    "pair_count": 0,
                    "baseline_successes": 0,
                    "candidate_successes": 0,
                    "non_regression_count": 0,
                    "improvement_count": 0,
                    "regression_count": 0,
                    "step_deltas": [],
                },
            )
            trace["pair_count"] = int(trace.get("pair_count", 0) or 0) + 1
            trace["baseline_successes"] = int(trace.get("baseline_successes", 0) or 0) + baseline_success
            trace["candidate_successes"] = int(trace.get("candidate_successes", 0) or 0) + candidate_success
            if candidate_success >= baseline_success:
                trace["non_regression_count"] = int(trace.get("non_regression_count", 0) or 0) + 1
            if candidate_success > baseline_success:
                trace["improvement_count"] = int(trace.get("improvement_count", 0) or 0) + 1
            if candidate_success < baseline_success:
                trace["regression_count"] = int(trace.get("regression_count", 0) or 0) + 1
            step_deltas = list(trace.get("step_deltas", []))
            step_deltas.append(candidate_steps - baseline_steps)
            trace["step_deltas"] = step_deltas
    paired_task_pair_count = sum(int(trace.get("pair_count", 0) or 0) for trace in traces.values())
    paired_task_non_regression_count = sum(
        int(trace.get("non_regression_count", 0) or 0) for trace in traces.values()
    )
    paired_task_improvement_count = sum(
        int(trace.get("improvement_count", 0) or 0) for trace in traces.values()
    )
    non_regression_lower, non_regression_upper = _wilson_interval(
        paired_task_non_regression_count,
        paired_task_pair_count,
        z=z,
    )
    improvement_lower, improvement_upper = _wilson_interval(
        paired_task_improvement_count,
        paired_task_pair_count,
        z=z,
    )
    non_regression_p_value = _one_sided_sign_test_p_value(
        paired_task_non_regression_count,
        paired_task_pair_count,
    )
    improvement_p_value = _one_sided_sign_test_p_value(
        paired_task_improvement_count,
        paired_task_pair_count,
    )
    summarized_traces: dict[str, dict[str, object]] = {}
    regressed_task_count = 0
    most_regressed: list[tuple[float, str, dict[str, object]]] = []
    for task_id, trace in traces.items():
        pair_count = max(1, int(trace.get("pair_count", 0) or 0))
        baseline_success_rate = int(trace.get("baseline_successes", 0) or 0) / pair_count
        candidate_success_rate = int(trace.get("candidate_successes", 0) or 0) / pair_count
        mean_step_delta = sum(trace.get("step_deltas", [])) / len(trace.get("step_deltas", []) or [0])
        regression_score = baseline_success_rate - candidate_success_rate
        if regression_score > 0.0:
            regressed_task_count += 1
        summarized = {
            "task_id": task_id,
            "benchmark_family": str(trace.get("benchmark_family", "bounded")),
            "pair_count": pair_count,
            "baseline_success_rate": baseline_success_rate,
            "candidate_success_rate": candidate_success_rate,
            "non_regression_rate": int(trace.get("non_regression_count", 0) or 0) / pair_count,
            "improvement_rate": int(trace.get("improvement_count", 0) or 0) / pair_count,
            "regression_rate": int(trace.get("regression_count", 0) or 0) / pair_count,
            "mean_step_delta": mean_step_delta,
        }
        summarized_traces[task_id] = summarized
        most_regressed.append((regression_score, task_id, summarized))
    most_regressed.sort(key=lambda item: (-item[0], item[2]["mean_step_delta"], item[1]))
    return {
        "paired_task_count": len(summarized_traces),
        "paired_task_pair_count": paired_task_pair_count,
        "paired_task_non_regression_rate": 0.0
        if paired_task_pair_count == 0
        else paired_task_non_regression_count / paired_task_pair_count,
        "paired_task_non_regression_rate_lower_bound": non_regression_lower,
        "paired_task_non_regression_rate_upper_bound": non_regression_upper,
        "paired_task_non_regression_significance_p_value": non_regression_p_value,
        "paired_task_improvement_rate": 0.0
        if paired_task_pair_count == 0
        else paired_task_improvement_count / paired_task_pair_count,
        "paired_task_improvement_rate_lower_bound": improvement_lower,
        "paired_task_improvement_rate_upper_bound": improvement_upper,
        "paired_task_improvement_significance_p_value": improvement_p_value,
        "regressed_task_count": regressed_task_count,
        "paired_task_traces": summarized_traces,
        "most_regressed_tasks": [item[2] for item in most_regressed[:10] if item[0] > 0.0],
        **_paired_trace_regression_report(baseline_runs, candidate_runs, z=z),
    }


def _paired_trace_regression_report(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    z: float,
) -> dict[str, object]:
    traces: dict[str, dict[str, object]] = {}
    for baseline_run, candidate_run in zip(baseline_runs, candidate_runs):
        baseline_outcomes = getattr(baseline_run, "task_outcomes", {}) or {}
        candidate_outcomes = getattr(candidate_run, "task_outcomes", {}) or {}
        if not isinstance(baseline_outcomes, dict) or not isinstance(candidate_outcomes, dict):
            continue
        for task_id in sorted(set(baseline_outcomes) & set(candidate_outcomes)):
            baseline_trace = baseline_outcomes.get(task_id, {})
            candidate_trace = candidate_outcomes.get(task_id, {})
            if not isinstance(baseline_trace, dict) or not isinstance(candidate_trace, dict):
                continue
            baseline_score = _task_trace_severity_score(baseline_trace)
            candidate_score = _task_trace_severity_score(candidate_trace)
            trace = traces.setdefault(
                str(task_id),
                {
                    "task_id": str(task_id),
                    "benchmark_family": str(
                        candidate_trace.get("benchmark_family", baseline_trace.get("benchmark_family", "bounded"))
                    ).strip()
                    or "bounded",
                    "pair_count": 0,
                    "non_regression_count": 0,
                    "regression_count": 0,
                    "improvement_count": 0,
                    "baseline_scores": [],
                    "candidate_scores": [],
                },
            )
            trace["pair_count"] = int(trace.get("pair_count", 0) or 0) + 1
            baseline_scores = list(trace.get("baseline_scores", []))
            candidate_scores = list(trace.get("candidate_scores", []))
            baseline_scores.append(float(baseline_score))
            candidate_scores.append(float(candidate_score))
            trace["baseline_scores"] = baseline_scores
            trace["candidate_scores"] = candidate_scores
            if candidate_score <= baseline_score:
                trace["non_regression_count"] = int(trace.get("non_regression_count", 0) or 0) + 1
            if candidate_score < baseline_score:
                trace["improvement_count"] = int(trace.get("improvement_count", 0) or 0) + 1
            if candidate_score > baseline_score:
                trace["regression_count"] = int(trace.get("regression_count", 0) or 0) + 1
    pair_count = sum(int(trace.get("pair_count", 0) or 0) for trace in traces.values())
    non_regression_count = sum(int(trace.get("non_regression_count", 0) or 0) for trace in traces.values())
    improvement_count = sum(int(trace.get("improvement_count", 0) or 0) for trace in traces.values())
    non_regression_lower, non_regression_upper = _wilson_interval(non_regression_count, pair_count, z=z)
    improvement_lower, improvement_upper = _wilson_interval(improvement_count, pair_count, z=z)
    non_regression_p_value = _one_sided_sign_test_p_value(non_regression_count, pair_count)
    improvement_p_value = _one_sided_sign_test_p_value(improvement_count, pair_count)
    regressed_trace_task_count = 0
    trace_summaries: dict[str, dict[str, object]] = {}
    ranked: list[tuple[float, str, dict[str, object]]] = []
    for task_id, trace in traces.items():
        count = max(1, int(trace.get("pair_count", 0) or 0))
        baseline_mean = sum(trace.get("baseline_scores", [])) / len(trace.get("baseline_scores", []) or [1.0])
        candidate_mean = sum(trace.get("candidate_scores", [])) / len(trace.get("candidate_scores", []) or [1.0])
        delta = candidate_mean - baseline_mean
        if delta > 0.0:
            regressed_trace_task_count += 1
        summary = {
            "task_id": task_id,
            "benchmark_family": str(trace.get("benchmark_family", "bounded")),
            "pair_count": count,
            "baseline_trace_score": baseline_mean,
            "candidate_trace_score": candidate_mean,
            "trace_score_delta": delta,
            "non_regression_rate": int(trace.get("non_regression_count", 0) or 0) / count,
            "improvement_rate": int(trace.get("improvement_count", 0) or 0) / count,
            "regression_rate": int(trace.get("regression_count", 0) or 0) / count,
        }
        trace_summaries[task_id] = summary
        ranked.append((delta, task_id, summary))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {
        "paired_trace_task_count": len(trace_summaries),
        "paired_trace_pair_count": pair_count,
        "paired_trace_non_regression_rate": 0.0 if pair_count == 0 else non_regression_count / pair_count,
        "paired_trace_non_regression_rate_lower_bound": non_regression_lower,
        "paired_trace_non_regression_rate_upper_bound": non_regression_upper,
        "paired_trace_non_regression_significance_p_value": non_regression_p_value,
        "paired_trace_improvement_rate": 0.0 if pair_count == 0 else improvement_count / pair_count,
        "paired_trace_improvement_rate_lower_bound": improvement_lower,
        "paired_trace_improvement_rate_upper_bound": improvement_upper,
        "paired_trace_improvement_significance_p_value": improvement_p_value,
        "regressed_trace_task_count": regressed_trace_task_count,
        "paired_trace_summaries": trace_summaries,
        "most_regressed_trace_tasks": [item[2] for item in ranked[:10] if item[0] > 0.0],
    }


def _task_trace_severity_score(trace: dict[str, object]) -> float:
    failure_signals = {
        str(signal).strip()
        for signal in list(trace.get("failure_signals", []))
        if str(signal).strip()
    }
    score = 0.0
    if not bool(trace.get("success", False)):
        score += 10.0
    score += 4.0 if bool(trace.get("unsafe_ambiguous", False)) else 0.0
    score += 4.0 if bool(trace.get("hidden_side_effect_risk", False)) else 0.0
    score += 2.5 if not bool(trace.get("first_step_verified", False)) else 0.0
    score += 1.5 * int(trace.get("present_forbidden_artifact_count", 0) or 0)
    score += 1.5 * int(trace.get("changed_preserved_artifact_count", 0) or 0)
    score += 0.75 * int(trace.get("missing_expected_artifact_count", 0) or 0)
    score += 0.75 * int(trace.get("no_state_progress_steps", 0) or 0)
    score += 0.5 * int(trace.get("state_regression_steps", 0) or 0)
    score += 0.2 * int(trace.get("total_state_regression_count", 0) or 0)
    score += 0.15 * int(trace.get("low_confidence_steps", 0) or 0)
    score += 0.1 * max(0, int(trace.get("steps", 0) or 0) - 1)
    score += 0.5 * max(0.0, 1.0 - float(trace.get("completion_ratio", 0.0) or 0.0))
    if "no_state_progress" in failure_signals:
        score += 1.0
    if "state_regression" in failure_signals:
        score += 1.0
    termination_reason = str(trace.get("termination_reason", "")).strip()
    score += {
        "success": 0.0,
        "policy_terminated": 0.5,
        "max_steps_reached": 1.5,
        "no_state_progress": 2.0,
        "repeated_failed_action": 2.5,
        "setup_failed": 3.0,
        "setup_pending": 2.0,
    }.get(termination_reason, 1.0 if termination_reason else 0.0)
    return float(score)


def _paired_trajectory_report(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    z: float,
) -> dict[str, object]:
    trajectories: dict[str, dict[str, object]] = {}
    for baseline_run, candidate_run in zip(baseline_runs, candidate_runs):
        baseline_trajectories = getattr(baseline_run, "task_trajectories", {}) or {}
        candidate_trajectories = getattr(candidate_run, "task_trajectories", {}) or {}
        if not isinstance(baseline_trajectories, dict) or not isinstance(candidate_trajectories, dict):
            continue
        for task_id in sorted(set(baseline_trajectories) & set(candidate_trajectories)):
            baseline_payload = baseline_trajectories.get(task_id, {})
            candidate_payload = candidate_trajectories.get(task_id, {})
            if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
                continue
            baseline_signature = _trajectory_signature(baseline_payload)
            candidate_signature = _trajectory_signature(candidate_payload)
            baseline_score = _trajectory_severity_score(baseline_payload)
            candidate_score = _trajectory_severity_score(candidate_payload)
            summary = trajectories.setdefault(
                str(task_id),
                {
                    "task_id": str(task_id),
                    "benchmark_family": str(
                        candidate_payload.get("benchmark_family", baseline_payload.get("benchmark_family", "bounded"))
                    ).strip()
                    or "bounded",
                    "pair_count": 0,
                    "exact_match_count": 0,
                    "non_regression_count": 0,
                    "regression_count": 0,
                    "improvement_count": 0,
                    "alignment_rates": [],
                    "baseline_scores": [],
                    "candidate_scores": [],
                },
            )
            summary["pair_count"] = int(summary.get("pair_count", 0) or 0) + 1
            if baseline_signature == candidate_signature:
                summary["exact_match_count"] = int(summary.get("exact_match_count", 0) or 0) + 1
            alignment_rates = list(summary.get("alignment_rates", []))
            alignment_rates.append(_trajectory_alignment_rate(baseline_signature, candidate_signature))
            summary["alignment_rates"] = alignment_rates
            baseline_scores = list(summary.get("baseline_scores", []))
            candidate_scores = list(summary.get("candidate_scores", []))
            baseline_scores.append(float(baseline_score))
            candidate_scores.append(float(candidate_score))
            summary["baseline_scores"] = baseline_scores
            summary["candidate_scores"] = candidate_scores
            if candidate_score <= baseline_score:
                summary["non_regression_count"] = int(summary.get("non_regression_count", 0) or 0) + 1
            if candidate_score < baseline_score:
                summary["improvement_count"] = int(summary.get("improvement_count", 0) or 0) + 1
            if candidate_score > baseline_score:
                summary["regression_count"] = int(summary.get("regression_count", 0) or 0) + 1
    pair_count = sum(int(item.get("pair_count", 0) or 0) for item in trajectories.values())
    exact_match_count = sum(int(item.get("exact_match_count", 0) or 0) for item in trajectories.values())
    non_regression_count = sum(int(item.get("non_regression_count", 0) or 0) for item in trajectories.values())
    improvement_count = sum(int(item.get("improvement_count", 0) or 0) for item in trajectories.values())
    exact_lower, exact_upper = _wilson_interval(exact_match_count, pair_count, z=z)
    non_regression_lower, non_regression_upper = _wilson_interval(non_regression_count, pair_count, z=z)
    improvement_lower, improvement_upper = _wilson_interval(improvement_count, pair_count, z=z)
    exact_match_p_value = _one_sided_sign_test_p_value(exact_match_count, pair_count)
    non_regression_p_value = _one_sided_sign_test_p_value(non_regression_count, pair_count)
    improvement_p_value = _one_sided_sign_test_p_value(improvement_count, pair_count)
    regressed_trajectory_task_count = 0
    ranked: list[tuple[float, str, dict[str, object]]] = []
    summaries: dict[str, dict[str, object]] = {}
    for task_id, item in trajectories.items():
        count = max(1, int(item.get("pair_count", 0) or 0))
        baseline_score = sum(item.get("baseline_scores", [])) / len(item.get("baseline_scores", []) or [1.0])
        candidate_score = sum(item.get("candidate_scores", [])) / len(item.get("candidate_scores", []) or [1.0])
        delta = candidate_score - baseline_score
        if delta > 0.0:
            regressed_trajectory_task_count += 1
        summary = {
            "task_id": task_id,
            "benchmark_family": str(item.get("benchmark_family", "bounded")),
            "pair_count": count,
            "exact_match_rate": int(item.get("exact_match_count", 0) or 0) / count,
            "non_regression_rate": int(item.get("non_regression_count", 0) or 0) / count,
            "improvement_rate": int(item.get("improvement_count", 0) or 0) / count,
            "mean_alignment_rate": sum(item.get("alignment_rates", [])) / len(item.get("alignment_rates", []) or [1.0]),
            "baseline_trajectory_score": baseline_score,
            "candidate_trajectory_score": candidate_score,
            "trajectory_score_delta": delta,
        }
        summaries[task_id] = summary
        ranked.append((delta, task_id, summary))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {
        "paired_trajectory_task_count": len(summaries),
        "paired_trajectory_pair_count": pair_count,
        "paired_trajectory_exact_match_rate": 0.0 if pair_count == 0 else exact_match_count / pair_count,
        "paired_trajectory_exact_match_rate_lower_bound": exact_lower,
        "paired_trajectory_exact_match_rate_upper_bound": exact_upper,
        "paired_trajectory_exact_match_significance_p_value": exact_match_p_value,
        "paired_trajectory_non_regression_rate": 0.0 if pair_count == 0 else non_regression_count / pair_count,
        "paired_trajectory_non_regression_rate_lower_bound": non_regression_lower,
        "paired_trajectory_non_regression_rate_upper_bound": non_regression_upper,
        "paired_trajectory_non_regression_significance_p_value": non_regression_p_value,
        "paired_trajectory_improvement_rate": 0.0 if pair_count == 0 else improvement_count / pair_count,
        "paired_trajectory_improvement_rate_lower_bound": improvement_lower,
        "paired_trajectory_improvement_rate_upper_bound": improvement_upper,
        "paired_trajectory_improvement_significance_p_value": improvement_p_value,
        "regressed_trajectory_task_count": regressed_trajectory_task_count,
        "paired_trajectory_summaries": summaries,
        "most_regressed_trajectory_tasks": [item[2] for item in ranked[:10] if item[0] > 0.0],
    }


def _trajectory_signature(payload: dict[str, object]) -> list[tuple[object, ...]]:
    signature: list[tuple[object, ...]] = []
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        signature.append(
            (
                str(step.get("action", "")).strip(),
                str(step.get("content", "")).strip(),
                str(step.get("command", "")).strip(),
                int(step.get("exit_code", 0) or 0),
                bool(step.get("timed_out", False)),
                bool(step.get("verification_passed", False)),
                tuple(str(value).strip() for value in step.get("failure_signals", []) if str(value).strip()),
                int(step.get("state_regression_count", 0) or 0),
                str(step.get("decision_source", "")).strip(),
                str(step.get("tolbert_route_mode", "")).strip(),
                bool(step.get("retrieval_influenced", False)),
                bool(step.get("trust_retrieval", False)),
            )
        )
    signature.append(
        (
            "__terminal__",
            str(payload.get("termination_reason", "")).strip(),
            bool(payload.get("success", False)),
        )
    )
    return signature


def _trajectory_alignment_rate(
    baseline_signature: list[tuple[object, ...]],
    candidate_signature: list[tuple[object, ...]],
) -> float:
    if not baseline_signature and not candidate_signature:
        return 1.0
    matched = 0
    for baseline_step, candidate_step in zip(baseline_signature, candidate_signature):
        if baseline_step != candidate_step:
            break
        matched += 1
    denominator = max(len(baseline_signature), len(candidate_signature), 1)
    return matched / denominator


def _trajectory_severity_score(payload: dict[str, object]) -> float:
    score = 0.0
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    if not bool(payload.get("success", False)):
        score += 10.0
    termination_reason = str(payload.get("termination_reason", "")).strip()
    score += {
        "success": 0.0,
        "policy_terminated": 0.5,
        "max_steps_reached": 1.5,
        "no_state_progress": 2.0,
        "repeated_failed_action": 2.5,
        "setup_failed": 3.0,
        "setup_pending": 2.0,
    }.get(termination_reason, 1.0 if termination_reason else 0.0)
    for step in steps:
        if not isinstance(step, dict):
            continue
        score += 0.1
        if not bool(step.get("verification_passed", False)):
            score += 1.0
        score += 0.75 * int(step.get("state_regression_count", 0) or 0)
        if bool(step.get("timed_out", False)):
            score += 1.5
        if int(step.get("exit_code", 0) or 0) != 0:
            score += 0.5
        failure_signals = {
            str(value).strip()
            for value in step.get("failure_signals", [])
            if str(value).strip()
        }
        if "no_state_progress" in failure_signals:
            score += 1.0
        if "state_regression" in failure_signals:
            score += 1.0
        if bool(step.get("retrieval_influenced", False)) and not bool(step.get("trust_retrieval", False)):
            score += 0.15
    return float(score)


def compare_to_prior_retained(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    subsystem: str,
    artifact_path: Path,
    cycles_path: Path,
    before_cycle_id: str,
    flags: dict[str, object],
    payload: dict[str, object] | None = None,
    task_limit: int | None = None,
    progress_label_prefix: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object] | None:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    prior_record = planner.prior_retained_artifact_record(
        cycles_path,
        subsystem,
        before_cycle_id=before_cycle_id,
    )
    if prior_record is None:
        return None
    baseline_cycle_id = str(prior_record.get("cycle_id", "")).strip()
    snapshot_value = str(prior_record.get("artifact_snapshot_path", "")).strip()
    snapshot_path = Path(snapshot_value) if snapshot_value else Path()
    if not snapshot_value or not snapshot_path.exists():
        return {
            "available": False,
            "baseline_cycle_id": baseline_cycle_id,
            "baseline_snapshot_path": snapshot_value,
            "reason": "prior retained artifact snapshot does not exist",
        }

    comparison_task_limit = _comparison_task_limit_for_retention(
        subsystem,
        task_limit=task_limit,
        payload=payload,
        capability_modules_path=config.capability_modules_path,
    )
    scoped_flags = _apply_retrieval_bounded_preview_filters(
        subsystem,
        flags=flags,
        payload=payload,
        capability_modules_path=config.capability_modules_path,
    )
    explicit_priority_families = [
        str(family).strip()
        for family in list(scoped_flags.get("priority_benchmark_families", []) or [])
        if str(family).strip()
    ]
    explicit_priority_weights: dict[str, float] = {}
    for family, weight in dict(scoped_flags.get("priority_benchmark_family_weights", {}) or {}).items():
        normalized_family = str(family).strip()
        if not normalized_family:
            continue
        try:
            explicit_priority_weights[normalized_family] = float(weight)
        except (TypeError, ValueError):
            continue
    restrict_to_priority_families = bool(scoped_flags.get("restrict_to_priority_benchmark_families", False)) and bool(
        explicit_priority_families
    )
    merged_priority_families, merged_priority_family_weights = _merge_priority_families(
        explicit_priority_families,
        explicit_priority_weights,
        payload,
        allow_payload_overrides=not restrict_to_priority_families,
    )
    if isinstance(comparison_task_limit, int) and comparison_task_limit > 0:
        scoped_flags["task_limit"] = comparison_task_limit
    holdout_generated_schedule_limit = _holdout_generated_schedule_limit_for_retention(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
    if holdout_generated_schedule_limit > 0:
        scoped_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
        scoped_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
    if merged_priority_families:
        scoped_flags["priority_benchmark_families"] = list(merged_priority_families)
    if merged_priority_family_weights:
        scoped_flags["priority_benchmark_family_weights"] = dict(merged_priority_family_weights)
    if restrict_to_priority_families and merged_priority_families:
        scoped_flags["restrict_to_priority_benchmark_families"] = True
        scoped_flags["prefer_low_cost_tasks"] = True
        scoped_flags["include_generated"] = False
        scoped_flags["include_failure_generated"] = False
    baseline_config = _retention_eval_config(
        base_config=config,
        subsystem=subsystem,
        artifact_path=snapshot_path,
        scope=f"{cycle_id_safe(before_cycle_id)}_prior_retained_baseline",
    )
    current_config = _retention_eval_config(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe(before_cycle_id)}_prior_retained_candidate",
    )
    _emit(
        f"finalize phase=preview_prior_retained_baseline_eval subsystem={subsystem} "
        f"baseline_cycle_id={baseline_cycle_id}"
    )
    baseline_metrics = _call_tolbert_preview_eval(
        config=baseline_config,
        subsystem=subsystem,
        flags=scoped_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_baseline",
        phase_name="preview_prior_retained_baseline",
        progress=progress,
    )
    _emit(
        f"finalize phase=preview_prior_retained_candidate_eval subsystem={subsystem} "
        f"baseline_cycle_id={baseline_cycle_id}"
    )
    current_metrics = _call_tolbert_preview_eval(
        config=current_config,
        subsystem=subsystem,
        flags=scoped_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_candidate",
        phase_name="preview_prior_retained_candidate",
        progress=progress,
    )
    _emit(
        f"finalize phase=preview_prior_retained_complete subsystem={subsystem} "
        f"baseline_cycle_id={baseline_cycle_id} "
        f"baseline_pass_rate={baseline_metrics.pass_rate:.4f} "
        f"candidate_pass_rate={current_metrics.pass_rate:.4f}"
    )
    baseline_payload: dict[str, object] | None = None
    try:
        loaded_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded_snapshot, dict):
            baseline_payload = loaded_snapshot
    except (OSError, json.JSONDecodeError):
        baseline_payload = None
    comparison_payload = payload_with_active_artifact_context(
        payload,
        active_artifact_path=snapshot_path,
        active_artifact_payload=baseline_payload,
    )
    evidence = retention_evidence(
        subsystem,
        baseline_metrics,
        current_metrics,
        payload=comparison_payload,
        capability_modules_path=config.capability_modules_path,
    )
    return {
        "available": True,
        "baseline_cycle_id": baseline_cycle_id,
        "baseline_snapshot_path": str(snapshot_path),
        "reason": "measured current artifact against the prior retained snapshot",
        "baseline_metrics": {
            "pass_rate": baseline_metrics.pass_rate,
            "average_steps": baseline_metrics.average_steps,
            "generated_pass_rate": baseline_metrics.generated_pass_rate,
            "proposal_selected_steps": baseline_metrics.proposal_selected_steps,
            "novel_valid_command_steps": baseline_metrics.novel_valid_command_steps,
            "novel_valid_command_rate": baseline_metrics.novel_valid_command_rate,
            "tolbert_primary_episodes": baseline_metrics.tolbert_primary_episodes,
        },
        "current_metrics": {
            "pass_rate": current_metrics.pass_rate,
            "average_steps": current_metrics.average_steps,
            "generated_pass_rate": current_metrics.generated_pass_rate,
            "proposal_selected_steps": current_metrics.proposal_selected_steps,
            "novel_valid_command_steps": current_metrics.novel_valid_command_steps,
            "novel_valid_command_rate": current_metrics.novel_valid_command_rate,
            "tolbert_primary_episodes": current_metrics.tolbert_primary_episodes,
        },
        "pass_rate_delta": current_metrics.pass_rate - baseline_metrics.pass_rate,
        "average_step_delta": current_metrics.average_steps - baseline_metrics.average_steps,
        "generated_pass_rate_delta": current_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
        "evidence": evidence,
    }


def preview_candidate_retention(
    *,
    config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
    cycle_id: str,
    active_artifact_path: Path | None = None,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_curriculum: bool = False,
    include_failure_curriculum: bool = False,
    task_limit: int | None = None,
    priority_benchmark_families: list[str] | None = None,
    priority_benchmark_family_weights: dict[str, float] | None = None,
    restrict_to_priority_benchmark_families: bool = False,
    progress_label_prefix: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    tolbert_runtime_summary = _new_tolbert_runtime_summary(use_tolbert_context=bool(config.use_tolbert_context))
    artifact_payload = None
    if artifact_path.exists():
        parsed = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            artifact_payload = effective_artifact_payload_for_retention(
                subsystem,
                parsed,
                capability_modules_path=config.capability_modules_path,
            )
    comparison_task_limit = _comparison_task_limit_for_retention(
        subsystem,
        task_limit=task_limit,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    merged_priority_families, merged_priority_family_weights = _merge_priority_families(
        priority_benchmark_families,
        priority_benchmark_family_weights,
        artifact_payload,
        allow_payload_overrides=not (
            restrict_to_priority_benchmark_families and bool(priority_benchmark_families)
        ),
    )
    managed_active_artifact_path = (
        active_artifact_path if active_artifact_path is not None else active_artifact_path_for_subsystem(config, subsystem)
    )
    baseline_config = _retention_eval_config(
        base_config=config,
        subsystem=subsystem,
        artifact_path=managed_active_artifact_path,
        scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_preview_baseline",
    )
    candidate_config = _retention_eval_config(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_preview_candidate",
    )
    baseline_flags, candidate_flags = baseline_candidate_flags(subsystem, config.capability_modules_path)
    baseline_flags.update(
        {
            "include_discovered_tasks": include_discovered_tasks or baseline_flags["include_discovered_tasks"],
            "include_episode_memory": include_episode_memory or baseline_flags["include_episode_memory"],
            "include_skill_memory": include_skill_memory or baseline_flags["include_skill_memory"],
            "include_skill_transfer": include_skill_transfer or baseline_flags["include_skill_transfer"],
            "include_operator_memory": include_operator_memory or baseline_flags["include_operator_memory"],
            "include_tool_memory": include_tool_memory or baseline_flags["include_tool_memory"],
            "include_verifier_memory": include_verifier_memory or baseline_flags["include_verifier_memory"],
            "include_generated": include_curriculum or baseline_flags["include_generated"],
            "include_failure_generated": include_failure_curriculum or baseline_flags["include_failure_generated"],
        }
    )
    baseline_flags = autonomous_runtime_eval_flags(baseline_config, baseline_flags)
    baseline_flags = _apply_retrieval_bounded_preview_filters(
        subsystem,
        flags=baseline_flags,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    if isinstance(comparison_task_limit, int) and comparison_task_limit > 0:
        baseline_flags["task_limit"] = comparison_task_limit
    if merged_priority_families:
        baseline_flags["priority_benchmark_families"] = list(merged_priority_families)
    if merged_priority_family_weights:
        baseline_flags["priority_benchmark_family_weights"] = dict(merged_priority_family_weights)
    if restrict_to_priority_benchmark_families and merged_priority_families:
        baseline_flags["restrict_to_priority_benchmark_families"] = True
        baseline_flags["prefer_low_cost_tasks"] = True
        baseline_flags["include_generated"] = False
        baseline_flags["include_failure_generated"] = False
    candidate_flags.update(
        {
            "include_discovered_tasks": include_discovered_tasks or candidate_flags["include_discovered_tasks"],
            "include_episode_memory": include_episode_memory or candidate_flags["include_episode_memory"],
            "include_skill_memory": include_skill_memory or candidate_flags["include_skill_memory"],
            "include_skill_transfer": include_skill_transfer or candidate_flags["include_skill_transfer"],
            "include_operator_memory": include_operator_memory or candidate_flags["include_operator_memory"],
            "include_tool_memory": include_tool_memory or candidate_flags["include_tool_memory"],
            "include_verifier_memory": include_verifier_memory or candidate_flags["include_verifier_memory"],
            "include_generated": include_curriculum or candidate_flags["include_generated"],
            "include_failure_generated": include_failure_curriculum or candidate_flags["include_failure_generated"],
        }
    )
    candidate_flags = autonomous_runtime_eval_flags(candidate_config, candidate_flags)
    candidate_flags = _apply_retrieval_bounded_preview_filters(
        subsystem,
        flags=candidate_flags,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    if isinstance(comparison_task_limit, int) and comparison_task_limit > 0:
        candidate_flags["task_limit"] = comparison_task_limit
    if merged_priority_families:
        candidate_flags["priority_benchmark_families"] = list(merged_priority_families)
    if merged_priority_family_weights:
        candidate_flags["priority_benchmark_family_weights"] = dict(merged_priority_family_weights)
    if restrict_to_priority_benchmark_families and merged_priority_families:
        candidate_flags["restrict_to_priority_benchmark_families"] = True
        candidate_flags["prefer_low_cost_tasks"] = True
        candidate_flags["include_generated"] = False
        candidate_flags["include_failure_generated"] = False
    _emit(f"finalize phase=preview_baseline_eval subsystem={subsystem}")
    baseline = _call_tolbert_preview_eval(
        config=baseline_config,
        subsystem=subsystem,
        flags=baseline_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_baseline",
        phase_name="preview_baseline",
        progress=progress,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    _emit(
        f"finalize phase=preview_baseline_complete subsystem={subsystem} "
        f"baseline_pass_rate={baseline.pass_rate:.4f}"
    )
    _emit(f"finalize phase=preview_candidate_eval subsystem={subsystem}")
    candidate = _call_tolbert_preview_eval(
        config=candidate_config,
        subsystem=subsystem,
        flags=candidate_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_candidate",
        phase_name="preview_candidate",
        progress=progress,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    _emit(
        f"finalize phase=preview_candidate_complete subsystem={subsystem} "
        f"candidate_pass_rate={candidate.pass_rate:.4f}"
    )
    evidence = retention_evidence(
        subsystem,
        baseline,
        candidate,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    compatibility = {}
    if isinstance(artifact_payload, dict):
        compatibility = dict(artifact_payload.get("compatibility", {}))
    candidate_matches_active = (
        base_subsystem_for(subsystem, config.capability_modules_path) == "retrieval"
        and _candidate_matches_active_artifact(artifact_path, managed_active_artifact_path)
    )
    if candidate_matches_active:
        state, reason = ("reject", "candidate artifact is identical to the active retained artifact")
        evidence = {
            **evidence,
            "artifact_content_unchanged": True,
            "candidate_artifact_path": str(artifact_path),
            "active_artifact_path": str(managed_active_artifact_path),
        }
    else:
        state, reason = evaluate_artifact_retention(
            subsystem,
            baseline,
            candidate,
            artifact_path=artifact_path,
            payload=artifact_payload,
            capability_modules_path=config.capability_modules_path,
        )
    gate = retention_gate_for_payload(
        subsystem,
        artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    phase_gate_report = autonomous_phase_gate_report(
        subsystem=subsystem,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        candidate_flags=candidate_flags,
        gate=gate,
        capability_modules_path=config.capability_modules_path,
    )
    if state == "retain" and not bool(phase_gate_report.get("passed", False)):
        phase_gate_failures = phase_gate_report.get("failures", [])
        first_failure = ""
        if isinstance(phase_gate_failures, list) and phase_gate_failures:
            first_failure = str(phase_gate_failures[0]).strip()
        reason = first_failure or "candidate failed autonomous phase gates"
        state = "reject"
    prior_retained_comparison: dict[str, object] | None = None
    prior_retained_guard_reason_value = ""
    prior_retained_guard_reason_code = ""
    if state == "retain":
        planner = ImprovementPlanner(
            memory_root=config.trajectories_root,
            prompt_proposals_path=config.prompt_proposals_path,
            use_prompt_proposals=config.use_prompt_proposals,
            capability_modules_path=config.capability_modules_path,
            trust_ledger_path=config.unattended_trust_ledger_path,
        )
        prior_retained_comparison = compare_to_prior_retained(
            config=config,
            planner=planner,
            subsystem=subsystem,
            artifact_path=artifact_path,
            cycles_path=config.improvement_cycles_path,
            before_cycle_id=cycle_id,
            flags=dict(candidate_flags),
            payload=artifact_payload,
            task_limit=task_limit,
            progress_label_prefix=(
                None if not progress_label_prefix else f"{progress_label_prefix}_prior_retained"
            ),
            progress=progress,
        )
        prior_guard_reason = prior_retained_guard_reason(
            subsystem=subsystem,
            gate=gate,
            comparison=prior_retained_comparison,
            capability_modules_path=config.capability_modules_path,
        )
        prior_retained_guard_reason_value = str(prior_guard_reason or "").strip()
        prior_retained_guard_reason_code = _prior_retained_guard_reason_code(prior_retained_guard_reason_value)
        if prior_guard_reason:
            baseline_cycle_id = ""
            if isinstance(prior_retained_comparison, dict):
                baseline_cycle_id = str(prior_retained_comparison.get("baseline_cycle_id", "")).strip()
            if baseline_cycle_id:
                reason = (
                    f"candidate failed prior retained comparison against {baseline_cycle_id}: "
                    f"{prior_guard_reason}"
                )
            else:
                reason = f"candidate failed prior retained comparison: {prior_guard_reason}"
            state = "reject"
    reason_code = _retention_reason_code(
        subsystem=subsystem,
        state=state,
        reason=reason,
        phase_gate_report=phase_gate_report,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
    )
    return {
        "state": state,
        "reason": reason,
        "reason_code": reason_code,
        "gate": gate,
        "phase_gate_report": phase_gate_report,
        "baseline": baseline,
        "candidate": candidate,
        "evidence": evidence,
        "compatibility": compatibility,
        "payload": artifact_payload,
        "prior_retained_comparison": prior_retained_comparison,
        "prior_retained_guard_reason": prior_retained_guard_reason_value,
        "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
        "baseline_flags": baseline_flags,
        "candidate_flags": candidate_flags,
        "active_artifact_path": managed_active_artifact_path,
        "tolbert_runtime_summary": _finalize_tolbert_runtime_summary(tolbert_runtime_summary),
    }


def _prior_retained_metrics_summary(comparison: dict[str, object] | None) -> dict[str, object]:
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
        family_pass_rate_delta = evidence.get("family_pass_rate_delta", {})
        if isinstance(family_pass_rate_delta, dict):
            summary["prior_retained_family_pass_rate_delta"] = {
                str(family): float(delta)
                for family, delta in family_pass_rate_delta.items()
            }
        if "generated_regressed_family_count" in evidence:
            summary["prior_retained_generated_regressed_family_count"] = int(
                evidence.get("generated_regressed_family_count", 0)
            )
        if "generated_worst_family_delta" in evidence:
            summary["prior_retained_generated_worst_family_delta"] = float(
                evidence.get("generated_worst_family_delta", 0.0)
            )
        generated_family_pass_rate_delta = evidence.get("generated_family_pass_rate_delta", {})
        if isinstance(generated_family_pass_rate_delta, dict):
            summary["prior_retained_generated_family_pass_rate_delta"] = {
                str(family): float(delta)
                for family, delta in generated_family_pass_rate_delta.items()
            }
        if "failure_recovery_pass_rate_delta" in evidence:
            summary["prior_retained_failure_recovery_pass_rate_delta"] = float(
                evidence.get("failure_recovery_pass_rate_delta", 0.0)
            )
        if "proposal_selected_steps_delta" in evidence:
            summary["prior_retained_proposal_selected_steps_delta"] = int(
                evidence.get("proposal_selected_steps_delta", 0) or 0
            )
        if "novel_valid_command_rate_delta" in evidence:
            summary["prior_retained_novel_valid_command_rate_delta"] = float(
                evidence.get("novel_valid_command_rate_delta", 0.0)
            )
        if "tolbert_primary_episodes_delta" in evidence:
            summary["prior_retained_tolbert_primary_episodes_delta"] = int(
                evidence.get("tolbert_primary_episodes_delta", 0) or 0
            )
    return summary


def autonomous_phase_gate_report(
    *,
    subsystem: str,
    baseline_metrics,
    candidate_metrics,
    candidate_flags: dict[str, bool],
    gate: dict[str, object],
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    failures: list[str] = []
    base_subsystem = base_subsystem_for(subsystem, capability_modules_path)
    generated_lane_included = bool(candidate_flags.get("include_generated", False))
    failure_recovery_lane_included = bool(candidate_flags.get("include_failure_generated", False))
    require_generated_lane_output = bool(
        gate.get(
            "require_autonomous_generated_lane_output",
            base_subsystem not in {"retrieval", "tolbert_model"},
        )
    )
    require_failure_recovery_output = bool(
        gate.get(
            "require_autonomous_failure_recovery_output",
            base_subsystem not in {"retrieval", "tolbert_model"},
        )
    )
    if not generated_lane_included:
        failures.append("generated-task lane was not included in autonomous cycle evaluation")
    if not failure_recovery_lane_included:
        failures.append("failure-recovery lane was not included in autonomous cycle evaluation")
    if generated_lane_included and require_generated_lane_output and int(candidate_metrics.generated_total) <= 0:
        failures.append("generated-task lane produced no tasks during autonomous evaluation")
    if (
        failure_recovery_lane_included
        and require_failure_recovery_output
        and int(candidate_metrics.generated_by_kind.get("failure_recovery", 0)) <= 0
    ):
        failures.append("failure-recovery lane produced no generated tasks during autonomous evaluation")
    if base_subsystem in {"retrieval", "tolbert_model"}:
        if candidate_metrics.trusted_retrieval_steps < baseline_metrics.trusted_retrieval_steps:
            failures.append("retrieval candidate reduced trusted retrieval usage under autonomous phase gates")
        if candidate_metrics.low_confidence_episodes > baseline_metrics.low_confidence_episodes:
            failures.append("retrieval candidate increased low-confidence episodes under autonomous phase gates")
        retrieval_influence_required = (
            candidate_metrics.trusted_retrieval_steps > 0
            or candidate_metrics.retrieval_guided_steps > 0
            or candidate_metrics.retrieval_selected_steps > 0
            or candidate_metrics.retrieval_ranked_skill_steps > 0
        )
        if retrieval_influence_required and candidate_metrics.retrieval_influenced_steps <= 0:
            failures.append("retrieval candidate showed no retrieval influence during autonomous evaluation")
        if retrieval_influence_required and (
            candidate_metrics.retrieval_ranked_skill_steps <= 0 and candidate_metrics.retrieval_selected_steps <= 0
        ):
            failures.append("retrieval candidate showed no retrieval selection or skill ranking during autonomous evaluation")
    if bool(gate.get("require_failure_recovery_non_regression", False)) and (
        _has_generated_kind(baseline_metrics, "failure_recovery")
        or _has_generated_kind(candidate_metrics, "failure_recovery")
    ):
        if _generated_kind_pass_rate(candidate_metrics, "failure_recovery") < _generated_kind_pass_rate(
            baseline_metrics,
            "failure_recovery",
        ):
            failures.append("failure-recovery lane regressed under autonomous phase gates")
    return {
        "passed": not failures,
        "failures": failures,
        "generated_lane_included": generated_lane_included,
        "failure_recovery_lane_included": failure_recovery_lane_included,
    }


def prior_retained_guard_reason(
    *,
    subsystem: str,
    gate: dict[str, object],
    comparison: dict[str, object] | None,
    capability_modules_path: Path | None = None,
) -> str | None:
    if not isinstance(comparison, dict) or not bool(comparison.get("available", False)):
        return None
    baseline_metrics = comparison.get("baseline_metrics", {})
    current_metrics = comparison.get("current_metrics", {})
    evidence = comparison.get("evidence", {})
    if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict):
        return None
    baseline_pass_rate = float(baseline_metrics.get("pass_rate", 0.0))
    current_pass_rate = float(current_metrics.get("pass_rate", 0.0))
    baseline_average_steps = float(baseline_metrics.get("average_steps", 0.0))
    current_average_steps = float(current_metrics.get("average_steps", 0.0))
    baseline_generated_pass_rate = float(baseline_metrics.get("generated_pass_rate", 0.0))
    current_generated_pass_rate = float(current_metrics.get("generated_pass_rate", 0.0))
    if current_pass_rate < baseline_pass_rate:
        return "candidate regressed pass rate against the prior retained baseline"
    max_step_regression = float(gate.get("max_step_regression", 0.0))
    if current_average_steps - baseline_average_steps > max_step_regression:
        return "candidate increased average steps against the prior retained baseline"
    if bool(gate.get("require_generated_lane_non_regression", False)) and (
        current_generated_pass_rate < baseline_generated_pass_rate
    ):
        return "candidate regressed the generated-task lane against the prior retained baseline"
    if "max_regressed_families" in gate and isinstance(evidence, dict):
        if int(evidence.get("regressed_family_count", 0)) > int(gate.get("max_regressed_families", 0)):
            return "candidate regressed one or more benchmark families against the prior retained baseline"
    if "max_generated_regressed_families" in gate and isinstance(evidence, dict):
        if int(evidence.get("generated_regressed_family_count", 0)) > int(
            gate.get("max_generated_regressed_families", 0)
        ):
            return "candidate regressed one or more generated benchmark families against the prior retained baseline"
    if (
        base_subsystem_for(subsystem, capability_modules_path) == "curriculum"
        and bool(gate.get("require_failure_recovery_improvement", True))
        and isinstance(evidence, dict)
    ):
        if float(evidence.get("failure_recovery_pass_rate_delta", 0.0)) < 0.0:
            return "candidate regressed failure-recovery performance against the prior retained baseline"
    if bool(gate.get("require_failure_recovery_non_regression", False)) and isinstance(evidence, dict):
        if float(evidence.get("failure_recovery_pass_rate_delta", 0.0)) < 0.0:
            return "candidate regressed failure-recovery performance against the prior retained baseline"
    if bool(gate.get("require_primary_routing_signal", False)) and int(
        current_metrics.get("tolbert_primary_episodes", 0) or 0
    ) < int(gate.get("min_primary_episodes", 0) or 0):
        return "candidate never entered retained Tolbert primary routing against the prior retained baseline"
    if bool(gate.get("require_novel_command_signal", False)) and int(
        current_metrics.get("proposal_selected_steps", 0) or 0
    ) <= 0:
        if not _tolbert_prior_retained_selection_signal_fallback_satisfied(gate, comparison):
            return "candidate produced no proposal-selected commands against the prior retained baseline"
    if int(evidence.get("proposal_selected_steps_delta", 0)) < int(gate.get("min_proposal_selected_steps_delta", 0)):
        return "candidate regressed proposal-selected command usage against the prior retained baseline"
    if int(current_metrics.get("novel_valid_command_steps", 0) or 0) < int(
        gate.get("min_novel_valid_command_steps", 0)
    ):
        return "candidate did not produce enough verifier-valid novel commands against the prior retained baseline"
    if float(evidence.get("novel_valid_command_rate_delta", 0.0)) < float(
        gate.get("min_novel_valid_command_rate_delta", 0.0)
    ):
        return "candidate regressed verifier-valid novel-command rate against the prior retained baseline"
    long_horizon_summary = evidence.get("long_horizon_summary", {})
    if isinstance(long_horizon_summary, dict):
        long_horizon_task_count = int(long_horizon_summary.get("baseline_task_count", 0) or 0) + int(
            long_horizon_summary.get("candidate_task_count", 0) or 0
        )
        if bool(gate.get("require_long_horizon_non_regression", False)) and long_horizon_task_count > 0:
            if float(long_horizon_summary.get("pass_rate_delta", 0.0) or 0.0) < 0.0:
                return "candidate regressed long-horizon pass rate against the prior retained baseline"
        if bool(gate.get("require_long_horizon_novel_command_non_regression", False)) and long_horizon_task_count > 0:
            if float(long_horizon_summary.get("novel_valid_command_rate_delta", 0.0) or 0.0) < 0.0:
                return "candidate regressed long-horizon verifier-valid novel-command rate against the prior retained baseline"
        long_horizon_world_feedback = long_horizon_summary.get("world_feedback", {})
        if not isinstance(long_horizon_world_feedback, dict):
            long_horizon_world_feedback = {}
        long_horizon_feedback_steps = int(long_horizon_summary.get("baseline_world_feedback_step_count", 0) or 0) + int(
            long_horizon_summary.get("candidate_world_feedback_step_count", 0) or 0
        )
        if bool(gate.get("require_long_horizon_world_feedback_non_regression", False)) and long_horizon_feedback_steps > 0:
            if float(long_horizon_world_feedback.get("progress_calibration_mae_gain", 0.0) or 0.0) < 0.0:
                return "candidate regressed long-horizon world-feedback calibration against the prior retained baseline"
    validation_family_summary = evidence.get("validation_family_summary", {})
    if isinstance(validation_family_summary, dict):
        validation_primary_task_count = int(validation_family_summary.get("baseline_primary_task_count", 0) or 0) + int(
            validation_family_summary.get("candidate_primary_task_count", 0) or 0
        )
        if bool(gate.get("require_validation_family_non_regression", False)) and validation_primary_task_count > 0:
            if float(validation_family_summary.get("primary_pass_rate_delta", 0.0) or 0.0) < 0.0:
                return "candidate regressed validation-family pass rate against the prior retained baseline"
        validation_generated_task_count = int(
            validation_family_summary.get("baseline_generated_task_count", 0) or 0
        ) + int(validation_family_summary.get("candidate_generated_task_count", 0) or 0)
        if bool(gate.get("require_validation_family_generated_non_regression", False)) and validation_generated_task_count > 0:
            if float(validation_family_summary.get("generated_pass_rate_delta", 0.0) or 0.0) < 0.0:
                return "candidate regressed validation-family generated pass rate against the prior retained baseline"
        validation_total_task_count = validation_primary_task_count + validation_generated_task_count
        if bool(gate.get("require_validation_family_novel_command_non_regression", False)) and validation_total_task_count > 0:
            if float(validation_family_summary.get("novel_valid_command_rate_delta", 0.0) or 0.0) < 0.0:
                return "candidate regressed validation-family verifier-valid novel-command rate against the prior retained baseline"
        validation_world_feedback = validation_family_summary.get("world_feedback", {})
        if not isinstance(validation_world_feedback, dict):
            validation_world_feedback = {}
        validation_feedback_steps = int(
            validation_family_summary.get("baseline_world_feedback_step_count", 0) or 0
        ) + int(validation_family_summary.get("candidate_world_feedback_step_count", 0) or 0)
        if bool(gate.get("require_validation_family_world_feedback_non_regression", False)) and validation_feedback_steps > 0:
            if float(validation_world_feedback.get("progress_calibration_mae_gain", 0.0) or 0.0) < 0.0:
                return "candidate regressed validation-family world-feedback calibration against the prior retained baseline"
    shared_repo_bundle_summary = evidence.get("shared_repo_bundle_summary", {})
    if (
        base_subsystem_for(subsystem, capability_modules_path) == "tooling"
        and bool(gate.get("require_shared_repo_bundle_coherence", False))
        and isinstance(shared_repo_bundle_summary, dict)
    ):
        shared_repo_candidate_count = int(
            shared_repo_bundle_summary.get("baseline_shared_repo_candidate_count", 0) or 0
        ) + int(shared_repo_bundle_summary.get("candidate_shared_repo_candidate_count", 0) or 0)
        if shared_repo_candidate_count > 0:
            if int(shared_repo_bundle_summary.get("candidate_bundle_coherence_delta", 0) or 0) < 0:
                return "candidate regressed shared-repo bundle coherence against the prior retained baseline"
            if int(shared_repo_bundle_summary.get("shared_repo_incomplete_integrator_candidate_count_delta", 0) or 0) > 0:
                return "candidate increased incomplete shared-repo integrator histories against the prior retained baseline"
            if int(shared_repo_bundle_summary.get("shared_repo_complete_candidate_count_delta", 0) or 0) < 0:
                return "candidate reduced complete shared-repo bundle evidence against the prior retained baseline"
    family_gate_failure = proposal_gate_failure_reason(
        gate,
        evidence,
        subject="candidate",
    )
    if family_gate_failure is not None:
        return f"{family_gate_failure} against the prior retained baseline"
    if current_pass_rate <= baseline_pass_rate and current_average_steps > baseline_average_steps:
        return "candidate did not beat the prior retained baseline on pass rate or steps"
    return None


def _tolbert_prior_retained_selection_signal_fallback_satisfied(
    gate: dict[str, object],
    comparison: dict[str, object],
) -> bool:
    if not bool(gate.get("allow_selection_signal_fallback", False)):
        return False
    baseline_metrics = comparison.get("baseline_metrics", {})
    current_metrics = comparison.get("current_metrics", {})
    evidence = comparison.get("evidence", {})
    if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict) or not isinstance(evidence, dict):
        return False
    if float(current_metrics.get("pass_rate", 0.0) or 0.0) < float(baseline_metrics.get("pass_rate", 0.0) or 0.0):
        return False
    selection_deltas = (
        int(evidence.get("trusted_retrieval_delta", 0) or 0),
        int(evidence.get("tolbert_primary_episodes_delta", 0) or 0),
    )
    return any(delta > 0 for delta in selection_deltas)


def _prior_retained_guard_reason_code(reason: str) -> str:
    normalized = str(reason).strip()
    if not normalized:
        return ""
    return {
        "candidate regressed long-horizon pass rate against the prior retained baseline": "long_horizon_pass_rate_regressed",
        "candidate regressed long-horizon verifier-valid novel-command rate against the prior retained baseline": "long_horizon_novel_command_rate_regressed",
        "candidate regressed long-horizon world-feedback calibration against the prior retained baseline": "long_horizon_world_feedback_regressed",
        "candidate regressed validation-family pass rate against the prior retained baseline": "validation_family_pass_rate_regressed",
        "candidate regressed validation-family generated pass rate against the prior retained baseline": "validation_family_generated_pass_rate_regressed",
        "candidate regressed validation-family verifier-valid novel-command rate against the prior retained baseline": "validation_family_novel_command_rate_regressed",
        "candidate regressed validation-family world-feedback calibration against the prior retained baseline": "validation_family_world_feedback_regressed",
        "candidate regressed shared-repo bundle coherence against the prior retained baseline": "shared_repo_bundle_coherence_regressed",
        "candidate increased incomplete shared-repo integrator histories against the prior retained baseline": "shared_repo_incomplete_integrator_histories_increased",
        "candidate reduced complete shared-repo bundle evidence against the prior retained baseline": "shared_repo_complete_bundle_evidence_regressed",
    }.get(normalized, "")


def _retention_reason_code_for_text(reason: str) -> str:
    normalized = str(reason).strip()
    if not normalized:
        return ""
    def _nested_reason_code(text: str) -> str:
        parts = str(text).rsplit(": ", 1)
        if len(parts) != 2:
            return ""
        return _retention_reason_code_for_text(parts[-1])
    prior_retained_code = _prior_retained_guard_reason_code(normalized)
    if prior_retained_code:
        return prior_retained_code
    if normalized.startswith("candidate failed prior retained comparison"):
        nested_code = _nested_reason_code(normalized)
        if nested_code:
            return nested_code
        return "prior_retained_comparison_failed"
    if normalized.startswith("candidate failed confirmation run"):
        nested_code = _nested_reason_code(normalized)
        return nested_code or "confirmation_run_failed"
    if normalized.startswith("candidate failed holdout run"):
        nested_code = _nested_reason_code(normalized)
        return nested_code or "holdout_run_failed"
    if normalized.startswith("candidate failed autonomous phase gates"):
        return "autonomous_phase_gates_failed"
    return {
        "candidate artifact is identical to the active retained artifact": "candidate_artifact_unchanged",
        "retrieval candidate did not satisfy the retained retrieval gate": "retrieval_retained_gate_failed",
        "retrieval candidate produced no material change from the retained artifact": "retrieval_no_material_change",
        "retrieval candidate regressed one or more benchmark families": "retrieval_family_regressed",
        "retrieval candidate regressed failure-recovery generation": "retrieval_failure_recovery_regressed",
        "Tolbert model candidate did not produce a checkpoint": "tolbert_checkpoint_missing",
        "Tolbert model candidate did not produce a retrieval cache": "tolbert_retrieval_cache_missing",
        "Tolbert model candidate regressed base success": "tolbert_base_success_regressed",
        "Qwen adapter candidate did not produce a training dataset": "qwen_training_dataset_missing",
        "Qwen adapter candidate attempted to claim primary runtime authority": "qwen_runtime_authority_violation",
        "Qwen adapter candidate did not declare a runtime target": "qwen_runtime_target_missing",
        "Qwen adapter candidate disabled teacher-generation support": "qwen_teacher_generation_disabled",
        "Qwen adapter candidate regressed base success": "qwen_base_success_regressed",
        "candidate regressed pass rate against the prior retained baseline": "prior_retained_pass_rate_regressed",
        "candidate increased average steps against the prior retained baseline": "prior_retained_average_steps_regressed",
        "candidate regressed the generated-task lane against the prior retained baseline": "prior_retained_generated_lane_regressed",
        "candidate regressed one or more benchmark families against the prior retained baseline": "prior_retained_family_regressed",
        "candidate regressed one or more generated benchmark families against the prior retained baseline": "prior_retained_generated_family_regressed",
        "candidate regressed failure-recovery performance against the prior retained baseline": "prior_retained_failure_recovery_regressed",
        "candidate produced no proposal-selected commands against the prior retained baseline": "prior_retained_proposal_selected_commands_missing",
        "candidate regressed proposal-selected command usage against the prior retained baseline": "prior_retained_proposal_selected_commands_regressed",
        "candidate did not produce enough verifier-valid novel commands against the prior retained baseline": "prior_retained_novel_valid_commands_missing",
        "candidate regressed verifier-valid novel-command rate against the prior retained baseline": "prior_retained_novel_valid_command_rate_regressed",
        "candidate regressed long-horizon pass rate against the prior retained baseline": "long_horizon_pass_rate_regressed",
        "candidate regressed long-horizon verifier-valid novel-command rate against the prior retained baseline": "long_horizon_novel_command_rate_regressed",
        "candidate regressed long-horizon world-feedback calibration against the prior retained baseline": "long_horizon_world_feedback_regressed",
        "candidate regressed validation-family pass rate against the prior retained baseline": "validation_family_pass_rate_regressed",
        "candidate regressed validation-family generated pass rate against the prior retained baseline": "validation_family_generated_pass_rate_regressed",
        "candidate regressed validation-family verifier-valid novel-command rate against the prior retained baseline": "validation_family_novel_command_rate_regressed",
        "candidate regressed validation-family world-feedback calibration against the prior retained baseline": "validation_family_world_feedback_regressed",
        "candidate regressed shared-repo bundle coherence against the prior retained baseline": "shared_repo_bundle_coherence_regressed",
        "candidate increased incomplete shared-repo integrator histories against the prior retained baseline": "shared_repo_incomplete_integrator_histories_increased",
        "candidate reduced complete shared-repo bundle evidence against the prior retained baseline": "shared_repo_complete_bundle_evidence_regressed",
        "candidate did not beat the prior retained baseline on pass rate or steps": "prior_retained_baseline_not_beaten",
        "generated-task lane was not included in autonomous cycle evaluation": "autonomous_generated_lane_missing",
        "failure-recovery lane was not included in autonomous cycle evaluation": "autonomous_failure_recovery_lane_missing",
        "generated-task lane produced no tasks during autonomous evaluation": "autonomous_generated_lane_empty",
        "failure-recovery lane produced no generated tasks during autonomous evaluation": "autonomous_failure_recovery_lane_empty",
        "retrieval candidate reduced trusted retrieval usage under autonomous phase gates": "autonomous_retrieval_trusted_usage_regressed",
        "retrieval candidate increased low-confidence episodes under autonomous phase gates": "autonomous_retrieval_low_confidence_increased",
        "retrieval candidate showed no retrieval influence during autonomous evaluation": "autonomous_retrieval_influence_missing",
        "retrieval candidate showed no retrieval selection or skill ranking during autonomous evaluation": "autonomous_retrieval_selection_missing",
        "failure-recovery lane regressed under autonomous phase gates": "autonomous_failure_recovery_regressed",
    }.get(normalized, "")


def _retention_reason_code(
    *,
    subsystem: str,
    state: str,
    reason: str,
    phase_gate_report: dict[str, object] | None = None,
    prior_retained_guard_reason_code: str = "",
) -> str:
    del subsystem
    if str(state).strip() != "reject":
        return ""
    if str(prior_retained_guard_reason_code).strip():
        return str(prior_retained_guard_reason_code).strip()
    candidate_reasons = [reason]
    if isinstance(phase_gate_report, dict) and not bool(phase_gate_report.get("passed", False)):
        candidate_reasons.extend(
            str(failure)
            for failure in phase_gate_report.get("failures", [])
            if str(failure).strip()
        )
    for candidate_reason in candidate_reasons:
        code = _retention_reason_code_for_text(candidate_reason)
        if code:
            return code
    return "retention_reject_unknown"


def _promotion_block_reason_code(*, final_reason: str, prior_retained_guard_reason: str = "") -> str:
    prior_code = _prior_retained_guard_reason_code(prior_retained_guard_reason)
    if prior_code:
        return prior_code
    normalized = str(final_reason).strip()
    if normalized.startswith("candidate failed prior retained comparison"):
        parts = normalized.rsplit(": ", 1)
        if parts:
            return _prior_retained_guard_reason_code(parts[-1])
    return ""


def _is_runtime_managed_artifact_path(path: str) -> bool:
    normalized = str(path).strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    return not (lowered.startswith("/tmp/") or "pytest-" in lowered or "/tests/" in lowered)


def _production_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if str(record.get("state", "")) in {"retain", "reject"}
        and _is_runtime_managed_artifact_path(str(record.get("artifact_path", "")))
    ]


def _yield_summary_for(records: list[dict[str, object]]) -> dict[str, object]:
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

    productive_depth_step_deltas: list[float] = []
    depth_drift_step_deltas: list[float] = []
    productive_depth_retained_cycles = 0
    depth_drift_cycles = 0
    long_horizon_retained_cycles = 0
    for row in records:
        metrics = row.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            continue
        try:
            baseline_steps = float(metrics.get("baseline_average_steps", 0.0) or 0.0)
            candidate_steps = float(metrics.get("candidate_average_steps", 0.0) or 0.0)
            baseline_pass_rate = float(metrics.get("baseline_pass_rate", 0.0) or 0.0)
            candidate_pass_rate = float(metrics.get("candidate_pass_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        step_delta = candidate_steps - baseline_steps
        pass_delta = candidate_pass_rate - baseline_pass_rate
        long_horizon_summary = metrics.get("long_horizon_summary", {})
        long_horizon_task_count = 0
        long_horizon_negative = False
        if isinstance(long_horizon_summary, dict):
            long_horizon_task_count = int(long_horizon_summary.get("baseline_task_count", 0) or 0) + int(
                long_horizon_summary.get("candidate_task_count", 0) or 0
            )
            long_horizon_negative = (
                float(long_horizon_summary.get("pass_rate_delta", 0.0) or 0.0) < 0.0
                or float(long_horizon_summary.get("novel_valid_command_rate_delta", 0.0) or 0.0) < 0.0
            )
            world_feedback = long_horizon_summary.get("world_feedback", {})
            if isinstance(world_feedback, dict):
                long_horizon_negative = long_horizon_negative or (
                    float(world_feedback.get("progress_calibration_mae_gain", 0.0) or 0.0) < 0.0
                )
        if str(row.get("state", "")) == "retain" and long_horizon_task_count > 0:
            long_horizon_retained_cycles += 1
        if step_delta <= 0.0:
            continue
        productive_depth = (
            str(row.get("state", "")) == "retain"
            and bool(metrics.get("phase_gate_passed", True))
            and pass_delta >= 0.0
            and not long_horizon_negative
        )
        if productive_depth:
            productive_depth_retained_cycles += 1
            productive_depth_step_deltas.append(step_delta)
            continue
        depth_drift_cycles += 1
        depth_drift_step_deltas.append(step_delta)

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
        "productive_depth_retained_cycles": productive_depth_retained_cycles,
        "average_productive_depth_step_delta": (
            0.0
            if not productive_depth_step_deltas
            else sum(productive_depth_step_deltas) / len(productive_depth_step_deltas)
        ),
        "max_productive_depth_step_delta": max(productive_depth_step_deltas, default=0.0),
        "depth_drift_cycles": depth_drift_cycles,
        "average_depth_drift_step_delta": (
            0.0 if not depth_drift_step_deltas else sum(depth_drift_step_deltas) / len(depth_drift_step_deltas)
        ),
        "max_depth_drift_step_delta": max(depth_drift_step_deltas, default=0.0),
        "long_horizon_retained_cycles": long_horizon_retained_cycles,
    }


def cycle_id_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._") or "cycle"


def _phase_gate_metrics_summary(report: dict[str, object]) -> dict[str, object]:
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


def _decision_state_for_cycle_report(
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


def _write_cycle_report(
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
    protocol_match_id: str = "",
    strategy_candidate: dict[str, object] | None = None,
    tolbert_runtime_summary: dict[str, object] | None = None,
) -> Path:
    records = planner.load_cycle_records(config.improvement_cycles_path)
    cycle_records = [record for record in records if str(record.get("cycle_id", "")) == cycle_id]
    production_decisions = _production_decisions(records)
    summary = planner.retained_gain_summary(config.improvement_cycles_path)
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    report_path = config.improvement_reports_dir / f"cycle_report_{safe_cycle_id}.json"
    promotion_block_reason_code = _promotion_block_reason_code(
        final_reason=final_reason,
        prior_retained_guard_reason=prior_retained_guard_reason,
    )
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
        "artifact_path": str(artifact_path),
        "candidate_artifact_path": str(artifact_update.get("candidate_artifact_path", "")),
        "active_artifact_path": str(artifact_update.get("active_artifact_path", "")),
        "artifact_kind": str(artifact_update.get("artifact_kind", "")),
        "final_state": final_state,
        "final_reason": final_reason,
        "decision_state": _decision_state_for_cycle_report(
            final_state=final_state,
            final_reason=final_reason,
            preview_reason_code=preview_reason_code,
            decision_reason_code=decision_reason_code,
        ),
        "promotion_blocked": str(final_state).strip() != "retain",
        "promotion_block_reason_code": promotion_block_reason_code,
        "prior_retained_guard_reason": str(prior_retained_guard_reason).strip(),
        "prior_retained_guard_reason_code": str(prior_retained_guard_reason_code).strip() or promotion_block_reason_code,
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
            "runtime_managed_artifact_path": _is_runtime_managed_artifact_path(
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
        "tolbert_runtime_summary": _finalize_tolbert_runtime_summary(tolbert_runtime_summary),
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
        "production_yield_summary": _yield_summary_for(production_decisions),
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
                **_phase_gate_metrics_summary(phase_gate_report),
                "production_total_decisions": report["production_yield_summary"]["total_decisions"],
                "production_retained_cycles": report["production_yield_summary"]["retained_cycles"],
                "production_rejected_cycles": report["production_yield_summary"]["rejected_cycles"],
                "protocol": "autonomous",
                "protocol_match_id": str(protocol_match_id).strip(),
                "strategy_candidate_id": str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                "strategy_candidate_kind": str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            },
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
        ),
    )
    return report_path


def finalize_cycle(
    *,
    config: KernelConfig,
    subsystem: str,
    cycle_id: str,
    artifact_path: Path,
    active_artifact_path: Path | None = None,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_curriculum: bool = False,
    include_failure_curriculum: bool = False,
    comparison_task_limit: int | None = None,
    priority_benchmark_families: list[str] | None = None,
    priority_benchmark_family_weights: dict[str, float] | None = None,
    restrict_to_priority_benchmark_families: bool = False,
    preview: dict[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
    protocol_match_id: str = "",
    strategy_candidate: dict[str, object] | None = None,
) -> tuple[str, str]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    repo_root = Path(__file__).resolve().parents[1]
    _emit(f"finalize phase=preview subsystem={subsystem}")
    preview_result = preview
    if preview_result is None:
        preview_result = preview_candidate_retention(
            config=config,
            subsystem=subsystem,
            artifact_path=artifact_path,
            cycle_id=cycle_id,
            active_artifact_path=active_artifact_path,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_memory=include_skill_memory,
            include_skill_transfer=include_skill_transfer,
            include_operator_memory=include_operator_memory,
            include_tool_memory=include_tool_memory,
            include_verifier_memory=include_verifier_memory,
            include_curriculum=include_curriculum,
            include_failure_curriculum=include_failure_curriculum,
            task_limit=comparison_task_limit,
            priority_benchmark_families=priority_benchmark_families,
            priority_benchmark_family_weights=priority_benchmark_family_weights,
            restrict_to_priority_benchmark_families=restrict_to_priority_benchmark_families,
            progress_label_prefix=f"{cycle_id}_{subsystem}_preview",
            progress=_emit,
        )
    else:
        _emit(f"finalize phase=preview_reused subsystem={subsystem}")
    managed_active_artifact_path = Path(preview_result["active_artifact_path"])
    baseline = preview_result["baseline"]
    candidate = preview_result["candidate"]
    evidence = dict(preview_result["evidence"])
    compatibility = dict(preview_result["compatibility"])
    artifact_payload = preview_result["payload"]
    state = str(preview_result["state"])
    reason = str(preview_result["reason"])
    gate = dict(preview_result["gate"])
    phase_gate_report = dict(preview_result.get("phase_gate_report", {}))
    prior_retained_comparison = preview_result["prior_retained_comparison"]
    prior_retained_guard_reason = str(preview_result.get("prior_retained_guard_reason", "")).strip()
    prior_retained_guard_reason_code = str(preview_result.get("prior_retained_guard_reason_code", "")).strip()
    preview_reason_code = str(preview_result.get("reason_code", "")).strip()
    tolbert_runtime_summary = _normalize_tolbert_runtime_summary(
        preview_result.get("tolbert_runtime_summary"),
        use_tolbert_context=bool(config.use_tolbert_context),
    )
    _emit(
        f"finalize phase=preview_complete subsystem={subsystem} "
        f"preview_state={state} baseline_pass_rate={baseline.pass_rate:.4f} "
        f"candidate_pass_rate={candidate.pass_rate:.4f}"
    )
    if state == "reject" and preview_reason_code:
        _emit(
            f"finalize phase=preview_reject_reason subsystem={subsystem} "
            f"reason_code={preview_reason_code} reason={reason}"
        )
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
    )
    baseline_flags = dict(preview_result["baseline_flags"])
    candidate_flags = dict(preview_result["candidate_flags"])
    protocol_metrics = {
        "protocol": "autonomous",
        "protocol_match_id": str(protocol_match_id).strip(),
    }

    evaluate_record_kwargs = {}
    if base_subsystem_for(subsystem, config.capability_modules_path) == "tooling":
        replay_verified_update = persist_replay_verified_tool_artifact(
            artifact_path,
            cycle_id=cycle_id,
            runtime_config=config,
        )
        evaluate_record_kwargs = {
            "artifact_lifecycle_state": str(replay_verified_update["artifact_lifecycle_state"]),
            "artifact_sha256": str(replay_verified_update["artifact_sha256"]),
            "previous_artifact_sha256": str(replay_verified_update["previous_artifact_sha256"]),
            "rollback_artifact_path": str(replay_verified_update["rollback_artifact_path"]),
            "artifact_snapshot_path": str(replay_verified_update["artifact_snapshot_path"]),
        }

    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="evaluate",
            subsystem=subsystem,
            action="compare_candidate_to_baseline",
            artifact_path=str(managed_active_artifact_path),
            artifact_kind="retention_evaluation",
            reason="measured baseline and candidate lanes for artifact retention",
            metrics_summary={
                "baseline_pass_rate": baseline.pass_rate,
                "candidate_pass_rate": candidate.pass_rate,
                "baseline_average_steps": baseline.average_steps,
                "candidate_average_steps": candidate.average_steps,
                "baseline_generated_pass_rate": baseline.generated_pass_rate,
                "candidate_generated_pass_rate": candidate.generated_pass_rate,
                **protocol_metrics,
                **_phase_gate_metrics_summary(phase_gate_report),
                **evidence,
            },
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            compatibility=compatibility,
            **evaluate_record_kwargs,
        ),
    )
    required_confirmation_runs = max(1, int(gate.get("required_confirmation_runs", 1)))
    confirmation_baseline_runs = [baseline]
    confirmation_candidate_runs = [candidate]
    if state == "retain" and required_confirmation_runs > 1:
        baseline_config = _retention_eval_config(
            base_config=config,
            subsystem=subsystem,
            artifact_path=managed_active_artifact_path,
            scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_confirmation_baseline",
        )
        candidate_config = _retention_eval_config(
            base_config=config,
            subsystem=subsystem,
            artifact_path=artifact_path,
            scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_confirmation_candidate",
        )
        def _evaluate_metrics_pair() -> tuple[object, object]:
            if base_subsystem_for(subsystem, config.capability_modules_path) == "operators":
                comparison = compare_abstraction_transfer_modes(
                    config=baseline_config,
                    include_discovered_tasks=baseline_flags["include_discovered_tasks"],
                    include_episode_memory=baseline_flags["include_episode_memory"],
                    include_verifier_memory=baseline_flags["include_verifier_memory"],
                    include_benchmark_candidates=baseline_flags["include_benchmark_candidates"],
                    include_verifier_candidates=baseline_flags["include_verifier_candidates"],
                    include_generated=baseline_flags["include_generated"],
                    include_failure_generated=baseline_flags["include_failure_generated"],
                    task_limit=baseline_flags.get("task_limit"),
                    progress_label_prefix=f"{cycle_id}_{subsystem}_confirmation",
                )
                return comparison.raw_skill_metrics, comparison.operator_metrics
            return (
                _evaluate_subsystem_metrics_with_tolbert_startup_retry(
                    config=baseline_config,
                    subsystem=subsystem,
                    flags=baseline_flags,
                    progress_label=f"{cycle_id}_{subsystem}_confirmation_baseline",
                    phase_name="confirmation_baseline",
                    progress=_emit,
                    tolbert_runtime_summary=tolbert_runtime_summary,
                ),
                _evaluate_subsystem_metrics_with_tolbert_startup_retry(
                    config=candidate_config,
                    subsystem=subsystem,
                    flags=candidate_flags,
                    progress_label=f"{cycle_id}_{subsystem}_confirmation_candidate",
                    phase_name="confirmation_candidate",
                    progress=_emit,
                    tolbert_runtime_summary=tolbert_runtime_summary,
                ),
            )
        for confirmation_index in range(2, required_confirmation_runs + 1):
            _emit(
                f"finalize phase=confirmation_eval subsystem={subsystem} "
                f"run={confirmation_index}/{required_confirmation_runs}"
            )
            confirmation_baseline, confirmation_candidate = _evaluate_metrics_pair()
            confirmation_baseline_runs.append(confirmation_baseline)
            confirmation_candidate_runs.append(confirmation_candidate)
            confirmation_evidence = retention_evidence(
                subsystem,
                confirmation_baseline,
                confirmation_candidate,
                payload=artifact_payload,
                capability_modules_path=config.capability_modules_path,
            )
            planner.append_cycle_record(
                config.improvement_cycles_path,
                ImprovementCycleRecord(
                    cycle_id=cycle_id,
                    state="evaluate",
                    subsystem=subsystem,
                    action="confirm_candidate_to_baseline",
                    artifact_path=str(managed_active_artifact_path),
                    artifact_kind="retention_confirmation",
                    reason=f"confirmation run {confirmation_index} of {required_confirmation_runs}",
                    metrics_summary={
                        "confirmation_run_index": confirmation_index,
                        "confirmation_run_count": required_confirmation_runs,
                        "baseline_pass_rate": confirmation_baseline.pass_rate,
                        "candidate_pass_rate": confirmation_candidate.pass_rate,
                        "baseline_average_steps": confirmation_baseline.average_steps,
                        "candidate_average_steps": confirmation_candidate.average_steps,
                        **protocol_metrics,
                        **confirmation_evidence,
                    },
                    strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                    strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                    candidate_artifact_path=str(artifact_path),
                    active_artifact_path=str(managed_active_artifact_path),
                    compatibility=compatibility,
                ),
            )
            confirmation_state, confirmation_reason = evaluate_artifact_retention(
                subsystem,
                confirmation_baseline,
                confirmation_candidate,
                artifact_path=artifact_path,
                payload=artifact_payload,
                capability_modules_path=config.capability_modules_path,
            )
            if confirmation_state != "retain":
                state = "reject"
                reason = (
                    f"candidate failed confirmation run {confirmation_index} "
                    f"of {required_confirmation_runs}: {confirmation_reason}"
                )
                break
    if state == "retain" and isinstance(comparison_task_limit, int) and comparison_task_limit > 0:
        _emit(f"finalize phase=holdout_eval subsystem={subsystem}")
        holdout_baseline_config = _retention_eval_config(
            base_config=config,
            subsystem=subsystem,
            artifact_path=managed_active_artifact_path,
            scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_holdout_baseline",
        )
        holdout_candidate_config = _retention_eval_config(
            base_config=config,
            subsystem=subsystem,
            artifact_path=artifact_path,
            scope=f"{cycle_id_safe(cycle_id)}_{subsystem}_holdout_candidate",
        )
        holdout_baseline_flags = {key: value for key, value in baseline_flags.items() if key != "task_limit"}
        holdout_candidate_flags = {key: value for key, value in candidate_flags.items() if key != "task_limit"}
        holdout_task_limit = _holdout_task_limit_for_retention(
            subsystem,
            comparison_task_limit=comparison_task_limit,
            capability_modules_path=config.capability_modules_path,
        )
        holdout_generated_schedule_limit = _holdout_generated_schedule_limit_for_retention(
            subsystem,
            comparison_task_limit=comparison_task_limit,
            capability_modules_path=config.capability_modules_path,
        )
        if isinstance(holdout_task_limit, int) and holdout_task_limit > 0:
            holdout_baseline_flags["task_limit"] = holdout_task_limit
            holdout_candidate_flags["task_limit"] = holdout_task_limit
        if holdout_generated_schedule_limit > 0:
            holdout_baseline_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
            holdout_baseline_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
            holdout_candidate_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
            holdout_candidate_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
        holdout_baseline = _evaluate_subsystem_metrics_with_tolbert_startup_retry(
            config=holdout_baseline_config,
            subsystem=subsystem,
            flags=holdout_baseline_flags,
            progress_label=f"{cycle_id}_{subsystem}_holdout_baseline",
            phase_name="holdout_baseline",
            progress=_emit,
            tolbert_runtime_summary=tolbert_runtime_summary,
        )
        holdout_candidate = _evaluate_subsystem_metrics_with_tolbert_startup_retry(
            config=holdout_candidate_config,
            subsystem=subsystem,
            flags=holdout_candidate_flags,
            progress_label=f"{cycle_id}_{subsystem}_holdout_candidate",
            phase_name="holdout_candidate",
            progress=_emit,
            tolbert_runtime_summary=tolbert_runtime_summary,
        )
        holdout_evidence = retention_evidence(
            subsystem,
            holdout_baseline,
            holdout_candidate,
            payload=artifact_payload,
            capability_modules_path=config.capability_modules_path,
        )
        holdout_phase_gate_report = autonomous_phase_gate_report(
            subsystem=subsystem,
            baseline_metrics=holdout_baseline,
            candidate_metrics=holdout_candidate,
            candidate_flags=holdout_candidate_flags,
            gate=gate,
            capability_modules_path=config.capability_modules_path,
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="evaluate",
                subsystem=subsystem,
                action="holdout_candidate_to_baseline",
                artifact_path=str(managed_active_artifact_path),
                artifact_kind="retention_holdout",
                reason="validated capped preview on an uncapped holdout lane",
                metrics_summary={
                    "baseline_pass_rate": holdout_baseline.pass_rate,
                    "candidate_pass_rate": holdout_candidate.pass_rate,
                    "baseline_average_steps": holdout_baseline.average_steps,
                    "candidate_average_steps": holdout_candidate.average_steps,
                    **protocol_metrics,
                    **_phase_gate_metrics_summary(holdout_phase_gate_report),
                    **holdout_evidence,
                },
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                candidate_artifact_path=str(artifact_path),
                active_artifact_path=str(managed_active_artifact_path),
                compatibility=compatibility,
            ),
        )
        holdout_state, holdout_reason = evaluate_artifact_retention(
            subsystem,
            holdout_baseline,
            holdout_candidate,
            artifact_path=artifact_path,
            payload=artifact_payload,
            capability_modules_path=config.capability_modules_path,
        )
        if holdout_state == "retain" and prior_retained_comparison is not None:
            holdout_prior_retained = compare_to_prior_retained(
                config=config,
                planner=planner,
                subsystem=subsystem,
                artifact_path=artifact_path,
                cycles_path=config.improvement_cycles_path,
                before_cycle_id=cycle_id,
                flags=dict(holdout_candidate_flags),
                payload=artifact_payload,
                task_limit=holdout_task_limit,
                progress_label_prefix=f"{cycle_id}_{subsystem}_holdout_prior_retained",
                progress=progress,
            )
            holdout_prior_guard_reason = prior_retained_guard_reason(
                subsystem=subsystem,
                gate=gate,
                comparison=holdout_prior_retained,
                capability_modules_path=config.capability_modules_path,
            )
            if holdout_prior_guard_reason:
                prior_retained_guard_reason = str(holdout_prior_guard_reason).strip()
                prior_retained_guard_reason_code = _prior_retained_guard_reason_code(prior_retained_guard_reason)
                holdout_state = "reject"
                holdout_reason = holdout_prior_guard_reason
            prior_retained_comparison = holdout_prior_retained
        if holdout_state == "retain" and not bool(holdout_phase_gate_report.get("passed", False)):
            holdout_failures = holdout_phase_gate_report.get("failures", [])
            holdout_reason = (
                str(holdout_failures[0]).strip()
                if isinstance(holdout_failures, list) and holdout_failures
                else "candidate failed autonomous phase gates on holdout"
            )
            holdout_state = "reject"
        confirmation_baseline_runs.append(holdout_baseline)
        confirmation_candidate_runs.append(holdout_candidate)
        baseline = holdout_baseline
        candidate = holdout_candidate
        evidence = holdout_evidence
        phase_gate_report = holdout_phase_gate_report
        state = holdout_state
        reason = holdout_reason
    if state == "retain":
        _emit(
            f"finalize phase=confidence_aggregate subsystem={subsystem} "
            f"confirmation_runs={len(confirmation_candidate_runs)}"
        )
        confirmation_report = confirmation_confidence_report(
            confirmation_baseline_runs,
            confirmation_candidate_runs,
            gate=gate,
        )
        confirmation_failures = confirmation_confidence_failures(confirmation_report, gate=gate)
        evidence.update(
            {
                "confirmation_run_count": int(confirmation_report.get("run_count", 0)),
                "confirmation_pass_rate_delta_stderr": float(
                    confirmation_report.get("pass_rate_delta_stderr", 0.0)
                ),
                "confirmation_pass_rate_delta_lower_bound": float(
                    confirmation_report.get("pass_rate_delta_lower_bound", 0.0)
                ),
                "confirmation_pass_rate_delta_conservative_lower_bound": float(
                    confirmation_report.get("pass_rate_delta_conservative_lower_bound", 0.0)
                ),
                "confirmation_paired_non_regression_rate": float(
                    confirmation_report.get("paired_non_regression_rate", 0.0)
                ),
                "confirmation_paired_task_non_regression_rate_lower_bound": float(
                    confirmation_report.get("paired_task_non_regression_rate_lower_bound", 0.0)
                ),
                "confirmation_paired_task_non_regression_significance_p_value": float(
                    confirmation_report.get("paired_task_non_regression_significance_p_value", 1.0)
                ),
                "confirmation_regressed_task_count": int(
                    confirmation_report.get("regressed_task_count", 0) or 0
                ),
                "confirmation_paired_trace_non_regression_rate_lower_bound": float(
                    confirmation_report.get("paired_trace_non_regression_rate_lower_bound", 0.0)
                ),
                "confirmation_paired_trace_non_regression_significance_p_value": float(
                    confirmation_report.get("paired_trace_non_regression_significance_p_value", 1.0)
                ),
                "confirmation_regressed_trace_task_count": int(
                    confirmation_report.get("regressed_trace_task_count", 0) or 0
                ),
                "confirmation_paired_trajectory_non_regression_rate_lower_bound": float(
                    confirmation_report.get("paired_trajectory_non_regression_rate_lower_bound", 0.0)
                ),
                "confirmation_paired_trajectory_non_regression_significance_p_value": float(
                    confirmation_report.get("paired_trajectory_non_regression_significance_p_value", 1.0)
                ),
                "confirmation_paired_trajectory_exact_match_rate_lower_bound": float(
                    confirmation_report.get("paired_trajectory_exact_match_rate_lower_bound", 0.0)
                ),
                "confirmation_paired_trajectory_exact_match_significance_p_value": float(
                    confirmation_report.get("paired_trajectory_exact_match_significance_p_value", 1.0)
                ),
                "confirmation_regressed_trajectory_task_count": int(
                    confirmation_report.get("regressed_trajectory_task_count", 0) or 0
                ),
                "confirmation_candidate_pass_rate_spread": float(
                    confirmation_report.get("candidate_pass_rate_spread", 0.0)
                ),
                "confirmation_baseline_pass_rate_spread": float(
                    confirmation_report.get("baseline_pass_rate_spread", 0.0)
                ),
                "confirmation_worst_family_conservative_lower_bound": float(
                    confirmation_report.get("worst_family_conservative_lower_bound", 0.0)
                ),
                "confirmation_regressed_family_conservative_count": int(
                    confirmation_report.get("regressed_family_conservative_count", 0) or 0
                ),
                "confirmation_step_delta_stderr": float(confirmation_report.get("step_delta_stderr", 0.0)),
                "confirmation_step_delta_upper_bound": float(
                    confirmation_report.get("step_delta_upper_bound", 0.0)
                ),
                "confirmation_step_delta_spread": float(confirmation_report.get("step_delta_spread", 0.0)),
            }
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="evaluate",
                subsystem=subsystem,
                action="aggregate_confirmation_confidence",
                artifact_path=str(managed_active_artifact_path),
                artifact_kind="retention_confirmation_confidence",
                reason="aggregated confirmation-run stability and pass-rate uncertainty",
                metrics_summary={
                    **protocol_metrics,
                    **dict(confirmation_report),
                },
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                candidate_artifact_path=str(artifact_path),
                active_artifact_path=str(managed_active_artifact_path),
                compatibility=compatibility,
            ),
        )
        if confirmation_failures:
            state = "reject"
            reason = confirmation_failures[0]
    if state == "retain" and not bool(phase_gate_report.get("passed", False)):
        phase_gate_failures = phase_gate_report.get("failures", [])
        first_failure = ""
        if isinstance(phase_gate_failures, list) and phase_gate_failures:
            first_failure = str(phase_gate_failures[0]).strip()
        reason = first_failure or "candidate failed autonomous phase gates"
        state = "reject"
    if prior_retained_comparison is not None:
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="evaluate",
                subsystem=subsystem,
                action="compare_candidate_to_prior_retained",
                artifact_path=str(managed_active_artifact_path),
                artifact_kind="prior_retained_comparison",
                reason=str(prior_retained_comparison.get("reason", "")),
                metrics_summary={
                    **protocol_metrics,
                    **_prior_retained_metrics_summary(prior_retained_comparison),
                    "prior_retained_guard_reason": prior_retained_guard_reason,
                    "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
                },
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                candidate_artifact_path=str(artifact_path),
                active_artifact_path=str(managed_active_artifact_path),
                compatibility=compatibility,
            ),
        )
    decision_reason_code = _retention_reason_code(
        subsystem=subsystem,
        state=state,
        reason=reason,
        phase_gate_report=phase_gate_report,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
    )
    tolbert_runtime_summary = _finalize_tolbert_runtime_summary(tolbert_runtime_summary)
    evidence["tolbert_runtime_summary"] = dict(tolbert_runtime_summary)
    evidence["tolbert_runtime_outcome"] = str(tolbert_runtime_summary.get("outcome", "")).strip()
    evidence["tolbert_runtime_startup_failure_count"] = int(
        tolbert_runtime_summary.get("startup_failure_count", 0) or 0
    )
    _emit(f"finalize phase=apply_decision subsystem={subsystem} state={state}")
    artifact_update = apply_artifact_retention_decision(
        artifact_path=artifact_path,
        active_artifact_path=managed_active_artifact_path,
        subsystem=subsystem,
        cycle_id=cycle_id,
        decision_state=state,
        decision_reason=reason,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        capability_modules_path=config.capability_modules_path,
        runtime_config=config,
    )
    if base_subsystem_for(subsystem, config.capability_modules_path) == "retrieval" and state == "retain":
        _emit(f"finalize phase=materialize_retrieval_bundle subsystem={subsystem}")
        bundle_manifest_path = materialize_retained_retrieval_asset_bundle(
            repo_root=repo_root,
            config=config,
            cycle_id=cycle_id,
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="record",
                subsystem=subsystem,
                action="materialize_retrieval_asset_bundle",
                artifact_path=str(bundle_manifest_path),
                artifact_kind="tolbert_retrieval_asset_bundle",
                reason="materialized retained retrieval controls into a Tolbert runtime bundle",
                metrics_summary={
                    "baseline_pass_rate": baseline.pass_rate,
                    "candidate_pass_rate": candidate.pass_rate,
                    "decision_pass_rate_delta": candidate.pass_rate - baseline.pass_rate,
                    **protocol_metrics,
                },
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                active_artifact_path=str(managed_active_artifact_path),
            ),
        )
    if base_subsystem_for(subsystem, config.capability_modules_path) == "tolbert_model" and state == "retain":
        payload = {}
        if managed_active_artifact_path.exists():
            try:
                payload = json.loads(managed_active_artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        liftoff_report = build_liftoff_gate_report(
            candidate_metrics=candidate,
            baseline_metrics=baseline,
            artifact_payload=payload,
        )
        atomic_write_json(
            config.tolbert_liftoff_report_path,
            {
                "spec_version": "asi_v1",
                "artifact_kind": "liftoff_gate_report",
                "cycle_id": cycle_id,
                "subsystem": subsystem,
                "report": liftoff_report.to_dict(),
            },
            config=config,
        )
        if isinstance(payload, dict):
            runtime_policy = payload.get("runtime_policy", {})
            if not isinstance(runtime_policy, dict):
                runtime_policy = {}
            if liftoff_report.state == "retain":
                runtime_policy["primary_benchmark_families"] = list(liftoff_report.primary_takeover_families)
                runtime_policy["shadow_benchmark_families"] = list(liftoff_report.shadow_only_families)
            else:
                runtime_policy["shadow_benchmark_families"] = sorted(
                    {
                        *[
                            str(value).strip()
                            for value in runtime_policy.get("shadow_benchmark_families", [])
                            if str(value).strip()
                        ],
                        *liftoff_report.primary_takeover_families,
                        *liftoff_report.shadow_only_families,
                    }
                )
            payload["runtime_policy"] = runtime_policy
            atomic_write_json(
                managed_active_artifact_path,
                payload,
                config=config,
            )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="record",
                subsystem=subsystem,
                action="write_tolbert_liftoff_gate_report",
                artifact_path=str(config.tolbert_liftoff_report_path),
                artifact_kind="liftoff_gate_report",
                reason=liftoff_report.reason,
                metrics_summary={
                    **protocol_metrics,
                    **liftoff_report.to_dict(),
                },
                active_artifact_path=str(managed_active_artifact_path),
            ),
        )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state=state,
            subsystem=subsystem,
            action="finalize_cycle",
            artifact_path=str(managed_active_artifact_path),
            artifact_kind=str(artifact_update["artifact_kind"] or "retention_decision"),
            reason=reason,
            metrics_summary={
                "baseline_pass_rate": baseline.pass_rate,
                "candidate_pass_rate": candidate.pass_rate,
                "baseline_average_steps": baseline.average_steps,
                "candidate_average_steps": candidate.average_steps,
                "preview_reason_code": preview_reason_code,
                "decision_reason_code": decision_reason_code,
                "promotion_block_reason_code": _promotion_block_reason_code(
                    final_reason=reason,
                    prior_retained_guard_reason=prior_retained_guard_reason,
                ),
                "prior_retained_guard_reason": prior_retained_guard_reason,
                "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
                **protocol_metrics,
                **_phase_gate_metrics_summary(phase_gate_report),
                **evidence,
                **_prior_retained_metrics_summary(prior_retained_comparison),
            },
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            artifact_lifecycle_state=str(artifact_update["artifact_lifecycle_state"]),
            artifact_sha256=str(artifact_update["artifact_sha256"]),
            previous_artifact_sha256=str(artifact_update["previous_artifact_sha256"]),
            rollback_artifact_path=str(artifact_update["rollback_artifact_path"]),
            artifact_snapshot_path=str(artifact_update["artifact_snapshot_path"]),
            compatibility=dict(artifact_update["compatibility"]),
        ),
    )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="record",
            subsystem=subsystem,
            action="persist_retention_outcome",
            artifact_path=str(managed_active_artifact_path),
            artifact_kind=str(artifact_update["artifact_kind"] or "retention_record"),
            reason="persisted artifact lifecycle and cycle-lineage metadata",
            metrics_summary={
                "baseline_pass_rate": baseline.pass_rate,
                "candidate_pass_rate": candidate.pass_rate,
                "decision_pass_rate_delta": candidate.pass_rate - baseline.pass_rate,
                "preview_reason_code": preview_reason_code,
                "decision_reason_code": decision_reason_code,
                "promotion_block_reason_code": _promotion_block_reason_code(
                    final_reason=reason,
                    prior_retained_guard_reason=prior_retained_guard_reason,
                ),
                "prior_retained_guard_reason": prior_retained_guard_reason,
                "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
                **protocol_metrics,
                **_phase_gate_metrics_summary(phase_gate_report),
                **evidence,
                **_prior_retained_metrics_summary(prior_retained_comparison),
            },
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            artifact_lifecycle_state=str(artifact_update["artifact_lifecycle_state"]),
            artifact_sha256=str(artifact_update["artifact_sha256"]),
            previous_artifact_sha256=str(artifact_update["previous_artifact_sha256"]),
            rollback_artifact_path=str(artifact_update["rollback_artifact_path"]),
            artifact_snapshot_path=str(artifact_update["artifact_snapshot_path"]),
            compatibility=dict(artifact_update["compatibility"]),
        ),
    )
    _write_cycle_report(
        config=config,
        planner=planner,
        cycle_id=cycle_id,
        subsystem=subsystem,
        artifact_path=managed_active_artifact_path,
        final_state=state,
        final_reason=reason,
        artifact_update=artifact_update,
        evidence=evidence,
        baseline=baseline,
        candidate=candidate,
        phase_gate_report=phase_gate_report,
        prior_retained_comparison=prior_retained_comparison,
        prior_retained_guard_reason=prior_retained_guard_reason,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
        preview_reason_code=preview_reason_code,
        decision_reason_code=decision_reason_code,
        protocol_match_id=protocol_match_id,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    if state == "reject" and decision_reason_code:
        _emit(
            f"finalize phase=decision_reject_reason subsystem={subsystem} "
            f"reason_code={decision_reason_code} reason={reason}"
        )
    _emit(f"finalize phase=done subsystem={subsystem} state={state}")
    return state, reason
