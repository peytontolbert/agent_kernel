from __future__ import annotations

from pathlib import Path
from typing import Callable
import json

from evals.harness import compare_abstraction_transfer_modes, run_eval

from .config import KernelConfig
from .improvement_confirmation import (
    confirmation_confidence_failures as _confirmation_confidence_failures,
    confirmation_confidence_report as _confirmation_confidence_report,
)
from .extensions.improvement.artifacts import (
    apply_artifact_retention_decision,
    artifact_sha256,
    persist_replay_verified_tool_artifact,
)
from .extensions.improvement import (
    cycle_decision_support,
    cycle_finalize_support,
    cycle_preview_evaluation,
    cycle_preview_support,
    cycle_retention_reasoning,
    cycle_tolbert_runtime,
    improvement_plugins as improvement_plugins_ext,
)
from .improvement_retention import evaluate_artifact_retention, retention_evidence
from .ops.improvement_reporting import (
    cycle_id_safe,
    decision_state_for_cycle_report as _reporting_decision_state_for_cycle_report,
    phase_gate_metrics_summary as _reporting_phase_gate_metrics_summary,
    prior_retained_metrics_summary as _reporting_prior_retained_metrics_summary,
    production_decisions as _reporting_production_decisions,
    write_cycle_report as _persist_cycle_report,
    yield_summary_for as _reporting_yield_summary_for,
)
from .improvement import ImprovementPlanner
from .extensions.strategy.subsystems import (
    baseline_candidate_flags,
    base_subsystem_for,
)

materialize_retained_retrieval_asset_bundle = (
    improvement_plugins_ext.materialize_retained_retrieval_asset_bundle
)


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
    runtime_flags = dict(flags)
    runtime_flags.setdefault("write_unattended_reports", True)
    task_limit = runtime_flags.get("task_limit")
    if not isinstance(task_limit, int) or task_limit <= 0:
        task_limit = None
    if base_subsystem_for(subsystem, config.capability_modules_path) == "operators":
        return compare_abstraction_transfer_modes(
            config=config,
            include_discovered_tasks=runtime_flags["include_discovered_tasks"],
            include_episode_memory=runtime_flags["include_episode_memory"],
            include_verifier_memory=runtime_flags["include_verifier_memory"],
            include_benchmark_candidates=runtime_flags["include_benchmark_candidates"],
            include_verifier_candidates=runtime_flags["include_verifier_candidates"],
            include_generated=runtime_flags["include_generated"],
            include_failure_generated=runtime_flags["include_failure_generated"],
            task_limit=task_limit,
            progress_label_prefix=progress_label,
        ).operator_metrics
    return run_eval(config=config, progress_label=progress_label, **runtime_flags)


_is_retryable_tolbert_startup_failure = cycle_tolbert_runtime.is_retryable_tolbert_startup_failure
_new_tolbert_runtime_summary = cycle_tolbert_runtime.new_tolbert_runtime_summary
_ordered_unique_stage_names = cycle_tolbert_runtime.ordered_unique_stage_names
_apply_priority_family_restriction = cycle_preview_support.apply_priority_family_restriction
_normalize_tolbert_runtime_summary = cycle_tolbert_runtime.normalize_tolbert_runtime_summary
_mark_tolbert_stage = cycle_tolbert_runtime.mark_tolbert_stage
_finalize_tolbert_runtime_summary = cycle_tolbert_runtime.finalize_tolbert_runtime_summary


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
    return cycle_tolbert_runtime.evaluate_subsystem_metrics_with_tolbert_startup_retry(
        evaluate_subsystem_metrics=evaluate_subsystem_metrics,
        config=config,
        subsystem=subsystem,
        flags=flags,
        progress_label=progress_label,
        phase_name=phase_name,
        progress=progress,
        tolbert_runtime_summary=tolbert_runtime_summary,
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
    return cycle_tolbert_runtime.call_tolbert_preview_eval(
        retry_evaluator=_evaluate_subsystem_metrics_with_tolbert_startup_retry,
        config=config,
        subsystem=subsystem,
        flags=flags,
        progress_label=progress_label,
        phase_name=phase_name,
        progress=progress,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )


_retention_eval_config = cycle_preview_support.retention_eval_config
_comparison_task_limit_for_retention = cycle_preview_support.comparison_task_limit_for_retention
_holdout_task_limit_for_retention = cycle_preview_support.holdout_task_limit_for_retention
_holdout_generated_schedule_limit_for_retention = (
    cycle_preview_support.holdout_generated_schedule_limit_for_retention
)
_retrieval_preview_priority_overrides = cycle_preview_support.retrieval_preview_priority_overrides
_retrieval_bounded_preview_required = cycle_preview_support.retrieval_bounded_preview_required
_apply_retrieval_bounded_preview_filters = cycle_preview_support.apply_retrieval_bounded_preview_filters
_merge_priority_families = cycle_preview_support.merge_priority_families


def confirmation_confidence_report(
    baseline_runs: list[object],
    candidate_runs: list[object],
    *,
    gate: dict[str, object] | None = None,
) -> dict[str, object]:
    return _confirmation_confidence_report(
        baseline_runs,
        candidate_runs,
        gate=gate,
    )


def confirmation_confidence_failures(report: dict[str, object], *, gate: dict[str, object] | None = None) -> list[str]:
    return _confirmation_confidence_failures(report, gate=gate)


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
    return cycle_preview_evaluation.compare_to_prior_retained(
        config=config,
        planner=planner,
        subsystem=subsystem,
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id=before_cycle_id,
        flags=flags,
        payload=payload,
        task_limit=task_limit,
        progress_label_prefix=progress_label_prefix,
        progress=progress,
        comparison_task_limit_for_retention_fn=_comparison_task_limit_for_retention,
        apply_retrieval_bounded_preview_filters_fn=_apply_retrieval_bounded_preview_filters,
        merge_priority_families_fn=_merge_priority_families,
        holdout_generated_schedule_limit_for_retention_fn=_holdout_generated_schedule_limit_for_retention,
        apply_priority_family_restriction_fn=_apply_priority_family_restriction,
        retention_eval_config_fn=_retention_eval_config,
        call_tolbert_preview_eval_fn=_call_tolbert_preview_eval,
        retention_evidence_fn=retention_evidence,
        cycle_id_safe_fn=cycle_id_safe,
    )


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
) -> dict[str, object]:
    return cycle_preview_evaluation.preview_candidate_retention(
        config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        cycle_id=cycle_id,
        preview_scope_suffix=preview_scope_suffix,
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
        task_limit=task_limit,
        priority_benchmark_families=priority_benchmark_families,
        priority_benchmark_family_weights=priority_benchmark_family_weights,
        restrict_to_priority_benchmark_families=restrict_to_priority_benchmark_families,
        progress_label_prefix=progress_label_prefix,
        progress=progress,
        new_tolbert_runtime_summary_fn=_new_tolbert_runtime_summary,
        comparison_task_limit_for_retention_fn=_comparison_task_limit_for_retention,
        merge_priority_families_fn=_merge_priority_families,
        retention_eval_config_fn=_retention_eval_config,
        autonomous_runtime_eval_flags_fn=autonomous_runtime_eval_flags,
        apply_retrieval_bounded_preview_filters_fn=_apply_retrieval_bounded_preview_filters,
        apply_priority_family_restriction_fn=_apply_priority_family_restriction,
        call_tolbert_preview_eval_fn=_call_tolbert_preview_eval,
        retention_evidence_fn=retention_evidence,
        candidate_matches_active_artifact_fn=_candidate_matches_active_artifact,
        evaluate_artifact_retention_fn=evaluate_artifact_retention,
        autonomous_phase_gate_report_fn=autonomous_phase_gate_report,
        compare_to_prior_retained_fn=compare_to_prior_retained,
        prior_retained_guard_reason_fn=prior_retained_guard_reason,
        prior_retained_guard_reason_code_fn=_prior_retained_guard_reason_code,
        retention_reason_code_fn=_retention_reason_code,
        finalize_tolbert_runtime_summary_fn=_finalize_tolbert_runtime_summary,
        cycle_id_safe_fn=cycle_id_safe,
    )


def _prior_retained_metrics_summary(comparison: dict[str, object] | None) -> dict[str, object]:
    return _reporting_prior_retained_metrics_summary(comparison)


autonomous_phase_gate_report = cycle_retention_reasoning.autonomous_phase_gate_report
prior_retained_guard_reason = cycle_retention_reasoning.prior_retained_guard_reason
_tolbert_prior_retained_selection_signal_fallback_satisfied = (
    cycle_retention_reasoning.tolbert_prior_retained_selection_signal_fallback_satisfied
)
_prior_retained_guard_reason_code = cycle_retention_reasoning.prior_retained_guard_reason_code
_retention_reason_code_for_text = cycle_retention_reasoning.retention_reason_code_for_text
_retention_reason_code = cycle_retention_reasoning.retention_reason_code
_promotion_block_reason_code = cycle_retention_reasoning.promotion_block_reason_code


def _is_runtime_managed_artifact_path(path: str) -> bool:
    from .ops.improvement_reporting import is_runtime_managed_artifact_path

    return is_runtime_managed_artifact_path(path)


def _production_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return _reporting_production_decisions(records)


def _yield_summary_for(records: list[dict[str, object]]) -> dict[str, object]:
    return _reporting_yield_summary_for(records)


def _phase_gate_metrics_summary(report: dict[str, object]) -> dict[str, object]:
    return _reporting_phase_gate_metrics_summary(report)


def _decision_state_for_cycle_report(
    *,
    final_state: str,
    final_reason: str,
    preview_reason_code: str = "",
    decision_reason_code: str = "",
) -> dict[str, object]:
    return _reporting_decision_state_for_cycle_report(
        final_state=final_state,
        final_reason=final_reason,
        preview_reason_code=preview_reason_code,
        decision_reason_code=decision_reason_code,
    )


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
    govern_exports: bool = True,
) -> Path:
    return _persist_cycle_report(
        config=config,
        planner=planner,
        cycle_id=cycle_id,
        subsystem=subsystem,
        artifact_path=artifact_path,
        final_state=final_state,
        final_reason=final_reason,
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
        promotion_block_reason_code=_promotion_block_reason_code(
            final_reason=final_reason,
            prior_retained_guard_reason=prior_retained_guard_reason,
        ),
        protocol_match_id=protocol_match_id,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=_finalize_tolbert_runtime_summary(tolbert_runtime_summary),
        govern_exports=govern_exports,
    )


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
        runtime_config=config,
    )
    baseline_flags = dict(preview_result["baseline_flags"])
    candidate_flags = dict(preview_result["candidate_flags"])
    protocol_metrics = {
        "protocol": "autonomous",
        "protocol_match_id": str(protocol_match_id).strip(),
    }

    evaluate_record_kwargs = cycle_decision_support.tooling_evaluate_record_kwargs(
        config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        cycle_id=cycle_id,
        persist_replay_verified_tool_artifact_fn=persist_replay_verified_tool_artifact,
    )
    cycle_decision_support.append_initial_compare_record(
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        cycle_id=cycle_id,
        subsystem=subsystem,
        managed_active_artifact_path=managed_active_artifact_path,
        artifact_path=artifact_path,
        baseline=baseline,
        candidate=candidate,
        evidence=evidence,
        phase_gate_report=phase_gate_report,
        compatibility=compatibility,
        protocol_metrics=protocol_metrics,
        strategy_candidate=strategy_candidate,
        phase_gate_metrics_summary_fn=_phase_gate_metrics_summary,
        evaluate_record_kwargs=evaluate_record_kwargs,
    )
    confirmation_result = cycle_finalize_support.run_confirmation_phase(
        config=config,
        planner=planner,
        subsystem=subsystem,
        cycle_id=cycle_id,
        artifact_path=artifact_path,
        managed_active_artifact_path=managed_active_artifact_path,
        artifact_payload=artifact_payload,
        gate=gate,
        baseline_flags=baseline_flags,
        candidate_flags=candidate_flags,
        compatibility=compatibility,
        protocol_metrics=protocol_metrics,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=tolbert_runtime_summary,
        state=state,
        reason=reason,
        baseline=baseline,
        candidate=candidate,
        progress=_emit,
        retention_eval_config_fn=_retention_eval_config,
        evaluate_subsystem_metrics_with_tolbert_startup_retry_fn=_evaluate_subsystem_metrics_with_tolbert_startup_retry,
        retention_evidence_fn=retention_evidence,
        evaluate_artifact_retention_fn=evaluate_artifact_retention,
        cycle_id_safe_fn=cycle_id_safe,
    )
    state = str(confirmation_result["state"])
    reason = str(confirmation_result["reason"])
    confirmation_baseline_runs = list(confirmation_result["confirmation_baseline_runs"])
    confirmation_candidate_runs = list(confirmation_result["confirmation_candidate_runs"])

    holdout_result = cycle_finalize_support.run_holdout_phase(
        config=config,
        planner=planner,
        subsystem=subsystem,
        cycle_id=cycle_id,
        artifact_path=artifact_path,
        managed_active_artifact_path=managed_active_artifact_path,
        artifact_payload=artifact_payload,
        gate=gate,
        baseline_flags=baseline_flags,
        candidate_flags=candidate_flags,
        compatibility=compatibility,
        protocol_metrics=protocol_metrics,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=tolbert_runtime_summary,
        comparison_task_limit=comparison_task_limit,
        state=state,
        reason=reason,
        baseline=baseline,
        candidate=candidate,
        evidence=evidence,
        phase_gate_report=phase_gate_report,
        prior_retained_comparison=prior_retained_comparison,
        prior_retained_guard_reason=prior_retained_guard_reason,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
        confirmation_baseline_runs=confirmation_baseline_runs,
        confirmation_candidate_runs=confirmation_candidate_runs,
        progress=_emit,
        retention_eval_config_fn=_retention_eval_config,
        holdout_task_limit_for_retention_fn=_holdout_task_limit_for_retention,
        holdout_generated_schedule_limit_for_retention_fn=_holdout_generated_schedule_limit_for_retention,
        evaluate_subsystem_metrics_with_tolbert_startup_retry_fn=_evaluate_subsystem_metrics_with_tolbert_startup_retry,
        retention_evidence_fn=retention_evidence,
        autonomous_phase_gate_report_fn=autonomous_phase_gate_report,
        evaluate_artifact_retention_fn=evaluate_artifact_retention,
        compare_to_prior_retained_fn=compare_to_prior_retained,
        prior_retained_guard_reason_fn=prior_retained_guard_reason,
        prior_retained_guard_reason_code_fn=_prior_retained_guard_reason_code,
        phase_gate_metrics_summary_fn=_phase_gate_metrics_summary,
        cycle_id_safe_fn=cycle_id_safe,
    )
    state = str(holdout_result["state"])
    reason = str(holdout_result["reason"])
    baseline = holdout_result["baseline"]
    candidate = holdout_result["candidate"]
    evidence = dict(holdout_result["evidence"])
    phase_gate_report = dict(holdout_result["phase_gate_report"])
    prior_retained_comparison = holdout_result["prior_retained_comparison"]
    prior_retained_guard_reason = str(holdout_result["prior_retained_guard_reason"])
    prior_retained_guard_reason_code = str(holdout_result["prior_retained_guard_reason_code"])
    confirmation_baseline_runs = list(holdout_result["confirmation_baseline_runs"])
    confirmation_candidate_runs = list(holdout_result["confirmation_candidate_runs"])

    confidence_result = cycle_finalize_support.aggregate_confirmation_confidence(
        config=config,
        planner=planner,
        subsystem=subsystem,
        cycle_id=cycle_id,
        artifact_path=artifact_path,
        managed_active_artifact_path=managed_active_artifact_path,
        gate=gate,
        compatibility=compatibility,
        protocol_metrics=protocol_metrics,
        strategy_candidate=strategy_candidate,
        state=state,
        reason=reason,
        evidence=evidence,
        confirmation_baseline_runs=confirmation_baseline_runs,
        confirmation_candidate_runs=confirmation_candidate_runs,
        progress=_emit,
        confirmation_confidence_report_fn=confirmation_confidence_report,
        confirmation_confidence_failures_fn=confirmation_confidence_failures,
    )
    state = str(confidence_result["state"])
    reason = str(confidence_result["reason"])
    evidence = dict(confidence_result["evidence"])
    if state == "retain" and not bool(phase_gate_report.get("passed", False)):
        phase_gate_failures = phase_gate_report.get("failures", [])
        first_failure = ""
        if isinstance(phase_gate_failures, list) and phase_gate_failures:
            first_failure = str(phase_gate_failures[0]).strip()
        reason = first_failure or "candidate failed autonomous phase gates"
        state = "reject"
    cycle_decision_support.append_prior_retained_record(
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        cycle_id=cycle_id,
        subsystem=subsystem,
        managed_active_artifact_path=managed_active_artifact_path,
        artifact_path=artifact_path,
        prior_retained_comparison=prior_retained_comparison,
        prior_retained_guard_reason=prior_retained_guard_reason,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
        compatibility=compatibility,
        protocol_metrics=protocol_metrics,
        strategy_candidate=strategy_candidate,
        prior_retained_metrics_summary_fn=_prior_retained_metrics_summary,
    )
    decision_reason_code = _retention_reason_code(
        subsystem=subsystem,
        state=state,
        reason=reason,
        phase_gate_report=phase_gate_report,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
    )
    tolbert_runtime_summary = _finalize_tolbert_runtime_summary(tolbert_runtime_summary)
    cycle_decision_support.apply_decision_and_persist(
        config=config,
        planner=planner,
        subsystem=subsystem,
        cycle_id=cycle_id,
        artifact_path=artifact_path,
        managed_active_artifact_path=managed_active_artifact_path,
        repo_root=repo_root,
        state=state,
        reason=reason,
        baseline=baseline,
        candidate=candidate,
        evidence=evidence,
        phase_gate_report=phase_gate_report,
        prior_retained_comparison=prior_retained_comparison,
        prior_retained_guard_reason=prior_retained_guard_reason,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
        preview_reason_code=preview_reason_code,
        decision_reason_code=decision_reason_code,
        protocol_match_id=protocol_match_id,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=tolbert_runtime_summary,
        protocol_metrics=protocol_metrics,
        progress=_emit,
        retention_reason_code=decision_reason_code,
        promotion_block_reason_code_fn=_promotion_block_reason_code,
        phase_gate_metrics_summary_fn=_phase_gate_metrics_summary,
        prior_retained_metrics_summary_fn=_prior_retained_metrics_summary,
        apply_artifact_retention_decision_fn=apply_artifact_retention_decision,
        write_cycle_report_fn=_write_cycle_report,
    )
    return state, reason
