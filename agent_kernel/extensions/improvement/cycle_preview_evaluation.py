from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ...config import KernelConfig
from ...improvement import ImprovementPlanner
from ...strategy_memory import finalize_strategy_node
from ...extensions.strategy.subsystems import (
    active_artifact_path_for_subsystem,
    baseline_candidate_flags,
    base_subsystem_for,
)
from .artifacts import (
    effective_artifact_payload_for_retention,
    payload_with_active_artifact_context,
    retention_gate_for_payload,
)
from .cycle_preview_support import retrieval_bounded_preview_required


def _allow_payload_priority_overrides(
    *,
    subsystem: str,
    payload: dict[str, object] | None,
    restrict_to_priority_families: bool,
    explicit_priority_families: list[str] | None,
    capability_modules_path: Path | None,
) -> bool:
    if not restrict_to_priority_families or not explicit_priority_families:
        return True
    # Retrieval retention previews need their artifact-defined discrimination probe
    # families even when the parent campaign is running with a restricted proof lane.
    return retrieval_bounded_preview_required(
        subsystem,
        payload=payload,
        capability_modules_path=capability_modules_path,
    )


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
    comparison_task_limit_for_retention_fn: Callable[..., int | None],
    apply_retrieval_bounded_preview_filters_fn: Callable[..., dict[str, object]],
    merge_priority_families_fn: Callable[..., tuple[list[str], dict[str, float]]],
    holdout_generated_schedule_limit_for_retention_fn: Callable[..., int],
    apply_priority_family_restriction_fn: Callable[..., None],
    apply_direct_a4_transition_model_compare_preferences_fn: Callable[..., None],
    retention_eval_config_fn: Callable[..., KernelConfig],
    call_tolbert_preview_eval_fn: Callable[..., object],
    retention_evidence_fn: Callable[..., dict[str, object]],
    cycle_id_safe_fn: Callable[[str], str],
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

    comparison_task_limit = comparison_task_limit_for_retention_fn(
        subsystem,
        task_limit=task_limit,
        payload=payload,
        flags=flags,
        capability_modules_path=config.capability_modules_path,
    )
    scoped_flags = apply_retrieval_bounded_preview_filters_fn(
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
    preserve_generated_lanes = bool(scoped_flags.get("include_generated", False)) or bool(
        scoped_flags.get("include_failure_generated", False)
    )
    allow_payload_overrides = _allow_payload_priority_overrides(
        subsystem=subsystem,
        payload=payload,
        restrict_to_priority_families=restrict_to_priority_families,
        explicit_priority_families=explicit_priority_families,
        capability_modules_path=config.capability_modules_path,
    )
    merged_priority_families, merged_priority_family_weights = merge_priority_families_fn(
        explicit_priority_families,
        explicit_priority_weights,
        payload,
        allow_payload_overrides=allow_payload_overrides,
    )
    if isinstance(comparison_task_limit, int) and comparison_task_limit > 0:
        scoped_flags["task_limit"] = comparison_task_limit
    holdout_generated_schedule_limit = holdout_generated_schedule_limit_for_retention_fn(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        baseline_flags=scoped_flags,
        candidate_flags=scoped_flags,
        capability_modules_path=config.capability_modules_path,
    )
    if holdout_generated_schedule_limit > 0:
        scoped_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
        scoped_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
    if merged_priority_families:
        scoped_flags["priority_benchmark_families"] = list(merged_priority_families)
    if merged_priority_family_weights:
        scoped_flags["priority_benchmark_family_weights"] = dict(merged_priority_family_weights)
    apply_priority_family_restriction_fn(
        scoped_flags,
        restrict_to_priority_families=restrict_to_priority_families,
        merged_priority_families=merged_priority_families,
        preserve_generated_lanes=preserve_generated_lanes,
    )
    apply_direct_a4_transition_model_compare_preferences_fn(
        subsystem,
        flags=scoped_flags,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
    baseline_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=snapshot_path,
        scope=f"{cycle_id_safe_fn(before_cycle_id)}_prior_retained_baseline",
    )
    current_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe_fn(before_cycle_id)}_prior_retained_candidate",
    )
    _emit(
        f"finalize phase=preview_prior_retained_baseline_eval subsystem={subsystem} "
        f"baseline_cycle_id={baseline_cycle_id}"
    )
    baseline_metrics = call_tolbert_preview_eval_fn(
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
    current_metrics = call_tolbert_preview_eval_fn(
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
    evidence = retention_evidence_fn(
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
    preview_scope_suffix: str | None = None,
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
    new_tolbert_runtime_summary_fn: Callable[..., dict[str, object]],
    comparison_task_limit_for_retention_fn: Callable[..., int | None],
    merge_priority_families_fn: Callable[..., tuple[list[str], dict[str, float]]],
    retention_eval_config_fn: Callable[..., KernelConfig],
    autonomous_runtime_eval_flags_fn: Callable[..., dict[str, bool]],
    apply_retrieval_bounded_preview_filters_fn: Callable[..., dict[str, object]],
    apply_priority_family_restriction_fn: Callable[..., None],
    apply_direct_a4_transition_model_compare_preferences_fn: Callable[..., None],
    call_tolbert_preview_eval_fn: Callable[..., object],
    retention_evidence_fn: Callable[..., dict[str, object]],
    candidate_matches_active_artifact_fn: Callable[[Path, Path], bool],
    evaluate_artifact_retention_fn: Callable[..., tuple[str, str]],
    autonomous_phase_gate_report_fn: Callable[..., dict[str, object]],
    compare_to_prior_retained_fn: Callable[..., dict[str, object] | None],
    prior_retained_guard_reason_fn: Callable[..., str | None],
    prior_retained_guard_reason_code_fn: Callable[[str], str],
    retention_reason_code_fn: Callable[..., str],
    finalize_tolbert_runtime_summary_fn: Callable[..., dict[str, object]],
    cycle_id_safe_fn: Callable[[str], str],
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    tolbert_runtime_summary = new_tolbert_runtime_summary_fn(use_tolbert_context=bool(config.use_tolbert_context))
    artifact_payload = None
    if artifact_path.exists():
        parsed = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            artifact_payload = effective_artifact_payload_for_retention(
                subsystem,
                parsed,
                capability_modules_path=config.capability_modules_path,
            )
    comparison_flags = {
        "task_limit": task_limit,
        "priority_benchmark_families": list(priority_benchmark_families or []),
        "restrict_to_priority_benchmark_families": restrict_to_priority_benchmark_families,
    }
    comparison_task_limit = comparison_task_limit_for_retention_fn(
        subsystem,
        task_limit=task_limit,
        payload=artifact_payload,
        flags=comparison_flags,
        capability_modules_path=config.capability_modules_path,
    )
    allow_payload_overrides = _allow_payload_priority_overrides(
        subsystem=subsystem,
        payload=artifact_payload,
        restrict_to_priority_families=(
            restrict_to_priority_benchmark_families and bool(priority_benchmark_families)
        ),
        explicit_priority_families=priority_benchmark_families,
        capability_modules_path=config.capability_modules_path,
    )
    merged_priority_families, merged_priority_family_weights = merge_priority_families_fn(
        priority_benchmark_families,
        priority_benchmark_family_weights,
        artifact_payload,
        allow_payload_overrides=allow_payload_overrides,
    )
    managed_active_artifact_path = (
        active_artifact_path if active_artifact_path is not None else active_artifact_path_for_subsystem(config, subsystem)
    )
    scope_suffix = ""
    if preview_scope_suffix is not None:
        normalized_scope_suffix = cycle_id_safe_fn(str(preview_scope_suffix).strip())
        if normalized_scope_suffix:
            scope_suffix = f"_{normalized_scope_suffix}"
    baseline_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=managed_active_artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_preview_baseline{scope_suffix}",
    )
    candidate_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_preview_candidate{scope_suffix}",
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
    baseline_flags = autonomous_runtime_eval_flags_fn(baseline_config, baseline_flags)
    baseline_flags = apply_retrieval_bounded_preview_filters_fn(
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
    preserve_generated_lanes = bool(include_curriculum) or bool(include_failure_curriculum)
    apply_priority_family_restriction_fn(
        baseline_flags,
        restrict_to_priority_families=restrict_to_priority_benchmark_families,
        merged_priority_families=merged_priority_families,
        preserve_generated_lanes=preserve_generated_lanes,
    )
    apply_direct_a4_transition_model_compare_preferences_fn(
        subsystem,
        flags=baseline_flags,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
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
    candidate_flags = autonomous_runtime_eval_flags_fn(candidate_config, candidate_flags)
    candidate_flags = apply_retrieval_bounded_preview_filters_fn(
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
    apply_priority_family_restriction_fn(
        candidate_flags,
        restrict_to_priority_families=restrict_to_priority_benchmark_families,
        merged_priority_families=merged_priority_families,
        preserve_generated_lanes=preserve_generated_lanes,
    )
    apply_direct_a4_transition_model_compare_preferences_fn(
        subsystem,
        flags=candidate_flags,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
    _emit(f"finalize phase=preview_baseline_eval subsystem={subsystem}")
    baseline = call_tolbert_preview_eval_fn(
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
    candidate = call_tolbert_preview_eval_fn(
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
    evidence = retention_evidence_fn(
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
        and candidate_matches_active_artifact_fn(artifact_path, managed_active_artifact_path)
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
        state, reason = evaluate_artifact_retention_fn(
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
    phase_gate_report = autonomous_phase_gate_report_fn(
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
        prior_retained_comparison = compare_to_prior_retained_fn(
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
        prior_guard_reason = prior_retained_guard_reason_fn(
            subsystem=subsystem,
            gate=gate,
            comparison=prior_retained_comparison,
            capability_modules_path=config.capability_modules_path,
        )
        prior_retained_guard_reason_value = str(prior_guard_reason or "").strip()
        prior_retained_guard_reason_code = prior_retained_guard_reason_code_fn(prior_retained_guard_reason_value)
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
    reason_code = retention_reason_code_fn(
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
        "tolbert_runtime_summary": finalize_tolbert_runtime_summary_fn(tolbert_runtime_summary),
    }
