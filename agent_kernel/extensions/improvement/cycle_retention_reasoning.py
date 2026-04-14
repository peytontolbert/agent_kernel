from __future__ import annotations

from pathlib import Path

from ...extensions.strategy.subsystems import base_subsystem_for
from ...improvement_retention import (
    _generated_kind_pass_rate,
    _has_generated_kind,
    proposal_gate_failure_reason,
)


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
        tolbert_primary_signal = base_subsystem == "tolbert_model" and int(
            getattr(candidate_metrics, "tolbert_primary_episodes", 0) or 0
        ) > 0
        retrieval_influence_required = (
            candidate_metrics.trusted_retrieval_steps > 0
            or candidate_metrics.retrieval_guided_steps > 0
            or candidate_metrics.retrieval_selected_steps > 0
            or candidate_metrics.retrieval_ranked_skill_steps > 0
        )
        if retrieval_influence_required and candidate_metrics.retrieval_influenced_steps <= 0 and not tolbert_primary_signal:
            failures.append("retrieval candidate showed no retrieval influence during autonomous evaluation")
        if retrieval_influence_required and (
            candidate_metrics.retrieval_ranked_skill_steps <= 0 and candidate_metrics.retrieval_selected_steps <= 0
        ) and not tolbert_primary_signal:
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
        if not tolbert_prior_retained_selection_signal_fallback_satisfied(gate, comparison):
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


def tolbert_prior_retained_selection_signal_fallback_satisfied(
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


def prior_retained_guard_reason_code(reason: str) -> str:
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


def retention_reason_code_for_text(reason: str) -> str:
    normalized = str(reason).strip()
    if not normalized:
        return ""

    def _nested_reason_code(text: str) -> str:
        parts = str(text).rsplit(": ", 1)
        if len(parts) != 2:
            return ""
        return retention_reason_code_for_text(parts[-1])

    prior_retained_code = prior_retained_guard_reason_code(normalized)
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


def retention_reason_code(
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
        code = retention_reason_code_for_text(candidate_reason)
        if code:
            return code
    return "retention_reject_unknown"


def promotion_block_reason_code(*, final_reason: str, prior_retained_guard_reason: str = "") -> str:
    prior_code = prior_retained_guard_reason_code(prior_retained_guard_reason)
    if prior_code:
        return prior_code
    normalized = str(final_reason).strip()
    if normalized.startswith("candidate failed prior retained comparison"):
        parts = normalized.rsplit(": ", 1)
        if parts:
            return prior_retained_guard_reason_code(parts[-1])
    return ""
