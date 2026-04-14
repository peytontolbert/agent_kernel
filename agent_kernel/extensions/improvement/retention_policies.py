from __future__ import annotations

from collections.abc import Callable

from evals.metrics import EvalMetrics

from ...improvement_engine import RetentionDecisionContext, has_measurable_runtime_influence
from ...improvement_retention import _has_generated_kind, proposal_gate_failure_reason


def _generated_lane_regressed(context: RetentionDecisionContext) -> bool:
    return bool(context.candidate_metrics.generated_total) and (
        context.candidate_metrics.generated_pass_rate < context.baseline_metrics.generated_pass_rate
    )


def _failure_recovery_regressed(context: RetentionDecisionContext) -> bool:
    return context.failure_recovery_delta < 0.0


def _learning_artifact_support(context: RetentionDecisionContext, artifact_kind: str) -> int:
    summary = context.evidence.get("learning_evidence", {})
    if not isinstance(summary, dict):
        return 0
    artifact_kind_support = summary.get("artifact_kind_support", {})
    if not isinstance(artifact_kind_support, dict):
        return 0
    return int(artifact_kind_support.get(artifact_kind, 0) or 0)


def _skills_learning_support_satisfied(context: RetentionDecisionContext) -> bool:
    if context.subsystem != "skills":
        return False
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return False
    if context.candidate_metrics.average_steps > context.baseline_metrics.average_steps:
        return False
    min_support = int(context.gate.get("min_learning_support_total", 2) or 2)
    return _learning_artifact_support(context, "success_skill_candidate") >= min_support


def _common_family_and_lane_checks(
    context: RetentionDecisionContext,
    *,
    subject: str,
    require_generated_lane_non_regression_default: bool = True,
    require_failure_recovery_non_regression_default: bool = True,
    require_base_success_non_regression: bool = False,
) -> tuple[str, str] | None:
    if context.average_step_delta > float(context.gate.get("max_step_regression", 0.0)):
        return ("reject", f"{subject} increased average steps")
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", f"{subject} regressed one or more benchmark families")
    if bool(
        context.gate.get("require_generated_lane_non_regression", require_generated_lane_non_regression_default)
    ) and _generated_lane_regressed(context):
        return ("reject", f"{subject} regressed the generated-task lane")
    if bool(
        context.gate.get("require_failure_recovery_non_regression", require_failure_recovery_non_regression_default)
    ) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if _failure_recovery_regressed(context):
            return ("reject", f"{subject} regressed failure-recovery generation")
    if require_base_success_non_regression and context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", f"{subject} regressed base success")
    return None


def _evaluate_curriculum_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.pass_rate_delta < -float(context.gate.get("max_base_lane_regression", 0.0)):
        return ("reject", "base-lane pass rate regressed under the curriculum candidate")
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "curriculum candidate regressed one or more benchmark families")
    if context.generated_pass_rate_delta < float(context.gate.get("min_generated_pass_rate_delta_abs", 0.02)):
        return ("reject", "generated-task pass rate did not improve by the required margin")
    if context.generated_regressed_family_count > int(context.gate.get("max_generated_regressed_families", 0)):
        return ("reject", "curriculum candidate regressed one or more generated benchmark families")
    if bool(context.gate.get("require_failure_recovery_improvement", True)) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if context.failure_recovery_delta <= 0.0:
            return ("reject", "failure-recovery generation did not improve")
    return ("retain", "generated-task and failure-recovery performance improved without regressing the base lane")


def _evaluate_verifier_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if not bool(context.evidence.get("require_contract_strictness_satisfied", False)):
        return ("reject", "verifier proposals are not structurally stricter than the source contract")
    if bool(context.gate.get("require_candidate_family_eval", True)) and int(
        context.evidence.get("verifier_candidate_total", 0)
    ) <= 0:
        return ("reject", "verifier candidate was not evaluated on verifier_candidate tasks")
    if float(context.evidence.get("discrimination_gain", 0.0)) < float(context.gate.get("min_discrimination_gain", 0.02)):
        return ("reject", "verifier candidate did not produce the required discrimination gain")
    if float(context.evidence.get("false_failure_rate", 1.0)) > float(context.gate.get("max_false_failure_rate", 0.01)):
        return ("reject", "verifier candidate exceeded the false-failure budget")
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", "verifier candidate regressed base success")
    if context.candidate_metrics.average_steps > context.baseline_metrics.average_steps:
        return ("reject", "verifier candidate increased average steps")
    return ("retain", "verifier candidate satisfied strictness, discrimination, and false-failure gates")


def _evaluate_benchmark_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "benchmark candidate regressed one or more benchmark families")
    if float(context.evidence.get("family_discrimination_gain", 0.0)) < float(
        context.gate.get("min_pass_rate_delta_abs", 0.02)
    ):
        return ("reject", "benchmark candidate did not produce the required family discrimination gain")
    if float(context.evidence.get("false_failure_rate", 1.0)) > float(context.gate.get("max_false_failure_rate", 0.02)):
        return ("reject", "benchmark candidate exceeded the false-failure budget")
    return ("retain", "benchmark candidate increased discriminative coverage within the false-failure budget")


def _evaluate_policy_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.pass_rate_delta < float(context.gate.get("min_pass_rate_delta_abs", 0.01)):
        return ("reject", "policy candidate did not improve base pass rate by the required margin")
    common = _common_family_and_lane_checks(context, subject="policy candidate")
    if common is not None:
        return common
    return ("retain", "policy candidate improved pass rate without step or generated-lane regression")


def _evaluate_world_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    min_pass_rate_delta = float(context.gate.get("min_pass_rate_delta_abs", 0.0))
    if context.pass_rate_delta < min_pass_rate_delta and not (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps < context.baseline_metrics.average_steps
    ):
        return ("reject", "world-model candidate did not improve pass rate or reduce steps")
    if context.average_step_delta > float(context.gate.get("max_step_regression", 0.0)):
        return ("reject", "world-model candidate increased average steps")
    if int(context.evidence.get("low_confidence_episode_delta", 0)) > int(
        context.gate.get("max_low_confidence_episode_regression", 0)
    ):
        return ("reject", "world-model candidate increased low-confidence episodes")
    if int(context.evidence.get("first_step_success_delta", 0)) < int(context.gate.get("min_first_step_success_delta", 0)):
        return ("reject", "world-model candidate regressed first-step success")
    common = _common_family_and_lane_checks(context, subject="world-model candidate")
    if common is not None:
        return common
    return ("retain", "world-model candidate improved command selection quality without broader regression")


def _evaluate_state_estimation_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="state-estimation candidate",
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get("require_state_estimation_improvement", True)) and int(
        context.evidence.get("state_estimation_improvement_count", 0)
    ) <= 0:
        return ("reject", "state-estimation candidate did not improve retained transition, latent, or policy-bias controls")
    if int(context.evidence.get("no_state_progress_termination_delta", 0)) > int(
        context.gate.get("max_no_state_progress_termination_delta", 0)
    ):
        return ("reject", "state-estimation candidate increased no_state_progress terminations")
    if int(context.evidence.get("state_regression_trace_delta", 0)) > int(
        context.gate.get("max_state_regression_trace_delta", 0)
    ):
        return ("reject", "state-estimation candidate increased state_regression traces")
    if float(context.evidence.get("paired_trajectory_non_regression_rate", 0.0)) < float(
        context.gate.get("min_paired_trajectory_non_regression_rate", 0.5)
    ):
        return ("reject", "state-estimation candidate weakened paired trajectory non-regression")
    if float(context.evidence.get("regressive_recovery_rate_delta", 0.0)) < float(
        context.gate.get("min_regressive_recovery_rate_delta", 0.0)
    ):
        return ("reject", "state-estimation candidate weakened recovery after regressive transitions")
    return ("retain", "state-estimation candidate improved runtime state summarization without broader regression")


def _evaluate_control_surface_retention(
    context: RetentionDecisionContext,
    *,
    subject: str,
    improvement_key: str,
    improvement_reason: str,
    success_reason: str,
    require_generated_lane_non_regression_default: bool = True,
    require_failure_recovery_non_regression_default: bool = True,
) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject=subject,
        require_generated_lane_non_regression_default=require_generated_lane_non_regression_default,
        require_failure_recovery_non_regression_default=require_failure_recovery_non_regression_default,
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if not has_measurable_runtime_influence(context):
        return ("reject", f"{subject} did not produce a measurable runtime signal")
    if bool(context.gate.get(improvement_key, True)) and int(
        context.evidence.get(improvement_key.replace("require_", "").replace("improvement", "improvement_count"), 0)
    ) <= 0:
        return ("reject", improvement_reason)
    return ("retain", success_reason)


def _evaluate_trust_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="trust candidate",
        improvement_key="require_trust_control_improvement",
        improvement_reason="trust candidate did not tighten unattended trust controls or broaden trust coverage",
        success_reason="trust candidate strengthened unattended trust controls without broader regression",
    )


def _evaluate_universe_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    result = _evaluate_control_surface_retention(
        context,
        subject="universe candidate",
        improvement_key="require_universe_improvement",
        improvement_reason="universe candidate did not strengthen stable governance, invariant, or command-boundary controls",
        success_reason="universe candidate strengthened stable governance above the world model without broader regression",
    )
    if result[0] != "retain":
        return result
    change_scope = str(context.evidence.get("universe_change_scope", "combined")).strip()
    if change_scope == "constitution":
        if int(context.evidence.get("prior_retained_universe_cycle_count", 0)) < int(
            context.gate.get("min_prior_retained_universe_cycles", 2)
        ):
            return ("reject", "universe constitution candidate requires multiple prior retained universe wins")
        if int(context.evidence.get("cross_family_support", 0)) < int(
            context.gate.get("min_cross_family_support", 2)
        ):
            return ("reject", "universe constitution candidate requires cross-family support")
        if int(context.evidence.get("constitution_cooldown_cycles_remaining", 0)) > 0:
            return ("reject", "universe constitution candidate is still inside the constitution cooldown window")
    return result


def _evaluate_recovery_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="recovery candidate",
        improvement_key="require_recovery_control_improvement",
        improvement_reason="recovery candidate did not strengthen unattended rollback or snapshot policy",
        success_reason="recovery candidate strengthened unattended rollback policy without broader regression",
    )


def _evaluate_delegation_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="delegation candidate",
        improvement_key="require_delegation_control_improvement",
        improvement_reason="delegation candidate did not expand delegated runtime capacity",
        success_reason="delegation candidate expanded delegated runtime capacity without broader regression",
    )


def _evaluate_operator_policy_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="operator-policy candidate",
        improvement_key="require_operator_policy_improvement",
        improvement_reason="operator-policy candidate did not broaden unattended operator policy",
        success_reason="operator-policy candidate broadened unattended execution policy without broader regression",
    )


def _evaluate_transition_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="transition-model candidate",
        improvement_key="require_transition_model_improvement",
        improvement_reason="transition-model candidate did not strengthen retained bad-transition controls",
        success_reason="transition-model candidate improved retained bad-transition guidance without broader regression",
    )


def _broad_coding_retrieval_support_signal(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, object]:
    coding_families = ("project", "repository", "integration", "repo_chore")

    def _family_counts(metrics: EvalMetrics) -> tuple[dict[str, int], dict[str, int]]:
        total_counts: dict[str, int] = {}
        passed_counts: dict[str, int] = {}
        for family in coding_families:
            total = max(
                int(metrics.total_by_benchmark_family.get(family, 0) or 0),
                int(metrics.total_by_origin_benchmark_family.get(family, 0) or 0),
            )
            passed = max(
                int(metrics.passed_by_benchmark_family.get(family, 0) or 0),
                int(metrics.passed_by_origin_benchmark_family.get(family, 0) or 0),
            )
            if total > 0:
                total_counts[family] = total
                passed_counts[family] = min(total, max(0, passed))
        return total_counts, passed_counts

    baseline_total_counts, baseline_passed_counts = _family_counts(baseline_metrics)
    candidate_total_counts, candidate_passed_counts = _family_counts(candidate_metrics)
    baseline_total = sum(baseline_total_counts.values())
    candidate_total = sum(candidate_total_counts.values())
    baseline_passed = sum(baseline_passed_counts.values())
    candidate_passed = sum(candidate_passed_counts.values())
    baseline_pass_rate = 0.0 if baseline_total <= 0 else baseline_passed / float(baseline_total)
    candidate_pass_rate = 0.0 if candidate_total <= 0 else candidate_passed / float(candidate_total)
    baseline_observed_families = sorted(baseline_total_counts)
    candidate_observed_families = sorted(candidate_total_counts)
    supports_broad_coding_non_regression = (
        len(candidate_observed_families) >= min(3, max(1, len(coding_families)))
        and len(candidate_observed_families) >= len(baseline_observed_families)
        and candidate_total >= baseline_total
        and candidate_pass_rate >= baseline_pass_rate
    )
    supports_broad_coding_gain = supports_broad_coding_non_regression and (
        len(candidate_observed_families) > len(baseline_observed_families)
        or candidate_total > baseline_total
        or candidate_pass_rate > baseline_pass_rate
    )
    return {
        "coding_families": list(coding_families),
        "baseline_observed_families": baseline_observed_families,
        "candidate_observed_families": candidate_observed_families,
        "baseline_observed_family_count": len(baseline_observed_families),
        "candidate_observed_family_count": len(candidate_observed_families),
        "baseline_total": baseline_total,
        "candidate_total": candidate_total,
        "baseline_pass_rate": baseline_pass_rate,
        "candidate_pass_rate": candidate_pass_rate,
        "supports_broad_coding_non_regression": supports_broad_coding_non_regression,
        "supports_broad_coding_gain": supports_broad_coding_gain,
    }


def _tolbert_selection_signal_fallback_satisfied(context: RetentionDecisionContext) -> bool:
    if not bool(context.gate.get("allow_selection_signal_fallback", False)):
        return False
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return False
    selection_deltas = (
        int(context.evidence.get("trusted_retrieval_delta", 0) or 0),
        int(context.candidate_metrics.retrieval_selected_steps - context.baseline_metrics.retrieval_selected_steps),
        int(context.evidence.get("tolbert_primary_episodes_delta", 0) or 0),
    )
    return any(delta > 0 for delta in selection_deltas)


def _evaluate_retrieval_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "retrieval candidate regressed one or more benchmark families")
    if bool(context.gate.get("require_failure_recovery_non_regression", False)) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if _failure_recovery_regressed(context):
            return ("reject", "retrieval candidate regressed failure-recovery generation")
    discrimination_satisfied = (
        not bool(context.gate.get("require_family_discrimination", True))
        or float(context.evidence.get("family_discrimination_gain", 0.0)) > 0.0
    )
    false_failure_rate = float(context.evidence.get("false_failure_rate", 0.0))
    low_confidence_non_regression = (
        int(context.evidence.get("low_confidence_episode_delta", 0) or 0)
        <= int(context.gate.get("max_low_confidence_episode_regression", 0))
    )
    broad_coding_support = _broad_coding_retrieval_support_signal(
        context.baseline_metrics,
        context.candidate_metrics,
    )
    trusted_retrieval_delta = int(context.evidence.get("trusted_retrieval_delta", 0) or 0)
    trusted_carryover_repair_rate_delta = float(context.evidence.get("trusted_carryover_repair_rate_delta", 0.0))
    retrieval_support_gain = (
        trusted_retrieval_delta > 0
        or trusted_carryover_repair_rate_delta > 0.0
        or int(context.evidence.get("low_confidence_episode_delta", 0) or 0) < 0
        or float(context.evidence.get("family_discrimination_gain", 0.0)) > 0.0
    )
    retrieval_support_non_regression = (
        context.candidate_metrics.trusted_retrieval_steps >= context.baseline_metrics.trusted_retrieval_steps
        or context.candidate_metrics.retrieval_influenced_steps >= context.baseline_metrics.retrieval_influenced_steps
        or context.candidate_metrics.retrieval_selected_steps >= context.baseline_metrics.retrieval_selected_steps
    )
    if (
        context.pass_rate_delta >= float(context.gate.get("min_pass_rate_delta_abs", 0.02))
        and context.average_step_delta <= 0.0
        and discrimination_satisfied
    ):
        return ("retain", "retrieval candidate improved pass rate and family discrimination without increasing steps")
    if (
        context.pass_rate_delta >= float(context.gate.get("min_pass_rate_delta_abs", 0.02))
        and context.average_step_delta <= 0.0
        and retrieval_support_non_regression
        and low_confidence_non_regression
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and bool(broad_coding_support.get("supports_broad_coding_non_regression", False))
    ):
        return (
            "retain",
            "retrieval candidate improved broad coding-family support without regressing the base lane",
        )
    if (
        context.pass_rate_delta >= 0.0
        and context.average_step_delta <= 0.0
        and retrieval_support_non_regression
        and low_confidence_non_regression
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and bool(broad_coding_support.get("supports_broad_coding_gain", False))
    ):
        return (
            "retain",
            "retrieval candidate broadened coding-family support without regressing the base lane",
        )
    if (
        context.pass_rate_delta >= 0.0
        and context.average_step_delta <= 0.0
        and retrieval_support_non_regression
        and retrieval_support_gain
        and low_confidence_non_regression
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and bool(broad_coding_support.get("supports_broad_coding_non_regression", False))
    ):
        return (
            "retain",
            "retrieval candidate strengthened complementary retrieval support without regressing the base lane",
        )
    min_trusted_carryover_repair_rate = float(context.gate.get("min_trusted_carryover_repair_rate", 0.0))
    min_trusted_carryover_verified_step_delta = int(
        context.gate.get("min_trusted_carryover_verified_step_delta", 1)
    )
    trusted_carryover_repair_rate = float(context.evidence.get("trusted_carryover_repair_rate", 0.0))
    baseline_trusted_carryover_repair_rate = float(
        context.evidence.get("baseline_trusted_carryover_repair_rate", 0.0)
    )
    trusted_carryover_verified_steps = int(context.evidence.get("trusted_carryover_verified_steps", 0) or 0)
    baseline_trusted_carryover_verified_steps = int(
        context.evidence.get("baseline_trusted_carryover_verified_steps", 0) or 0
    )
    trusted_carryover_verified_step_delta = int(
        context.evidence.get("trusted_carryover_verified_step_delta", 0) or 0
    )
    candidate_artifact_sha256 = str(context.artifact_update.get("artifact_sha256", "")).strip()
    previous_artifact_sha256 = str(context.artifact_update.get("previous_artifact_sha256", "")).strip()
    zero_yield_retrieval_change = (
        candidate_artifact_sha256
        and previous_artifact_sha256
        and candidate_artifact_sha256 == previous_artifact_sha256
        and context.pass_rate_delta == 0.0
        and context.average_step_delta == 0.0
        and int(context.evidence.get("proposal_selected_steps_delta", 0) or 0) == 0
        and int(context.evidence.get("trusted_retrieval_delta", 0) or 0) == 0
        and trusted_carryover_repair_rate == baseline_trusted_carryover_repair_rate
        and trusted_carryover_verified_step_delta == 0
    )
    if zero_yield_retrieval_change:
        return ("reject", "retrieval candidate produced no material change from the retained artifact")
    if bool(context.gate.get("require_trusted_carryover_repair_improvement", False)) and (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps <= context.baseline_metrics.average_steps
        and low_confidence_non_regression
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and trusted_carryover_repair_rate >= min_trusted_carryover_repair_rate
        and trusted_carryover_verified_step_delta >= min_trusted_carryover_verified_step_delta
    ):
        return (
            "retain",
            "retrieval candidate increased verified long-horizon trusted-retrieval carryover without regressing the base lane",
        )
    if bool(context.gate.get("require_trusted_carryover_repair_improvement", False)) and (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps <= context.baseline_metrics.average_steps
        and low_confidence_non_regression
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and baseline_trusted_carryover_repair_rate >= min_trusted_carryover_repair_rate
        and trusted_carryover_repair_rate >= baseline_trusted_carryover_repair_rate
        and trusted_carryover_verified_steps >= baseline_trusted_carryover_verified_steps
    ):
        return (
            "retain",
            "retrieval candidate preserved verified long-horizon trusted-retrieval carryover without regressing the base lane",
        )
    if (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps <= context.baseline_metrics.average_steps
        and context.candidate_metrics.trusted_retrieval_steps > context.baseline_metrics.trusted_retrieval_steps
        and context.candidate_metrics.low_confidence_episodes <= context.baseline_metrics.low_confidence_episodes
        and false_failure_rate <= float(context.gate.get("max_false_failure_rate", 0.02))
        and discrimination_satisfied
    ):
        return ("retain", "retrieval candidate increased trusted retrieval usage without regressing the base lane")
    return ("reject", "retrieval candidate did not satisfy the retained retrieval gate")


def _evaluate_tolbert_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    long_horizon = context.evidence.get("long_horizon_summary", {})
    if not isinstance(long_horizon, dict):
        long_horizon = {}
    if not bool(context.evidence.get("checkpoint_exists", False)):
        return ("reject", "Tolbert model candidate did not produce a checkpoint")
    if int(context.evidence.get("cache_count", 0)) <= 0:
        return ("reject", "Tolbert model candidate did not produce a retrieval cache")
    common = _common_family_and_lane_checks(
        context,
        subject="Tolbert model candidate",
        require_generated_lane_non_regression_default=True,
        require_failure_recovery_non_regression_default=True,
    )
    if common is not None:
        return common
    if int(context.evidence.get("low_confidence_episode_delta", 0)) > int(
        context.gate.get("max_low_confidence_episode_regression", 0)
    ):
        return ("reject", "Tolbert model candidate increased low-confidence episodes")
    if float(context.evidence.get("first_step_confidence_delta", 0.0)) < float(
        context.gate.get("min_first_step_confidence_delta", 0.0)
    ):
        return ("reject", "Tolbert model candidate regressed first-step path confidence")
    if int(context.evidence.get("trusted_retrieval_delta", 0)) < int(context.gate.get("min_trusted_retrieval_delta", 0)):
        return ("reject", "Tolbert model candidate did not improve trusted retrieval usage")
    if bool(context.gate.get("require_primary_routing_signal", False)) and int(
        context.candidate_metrics.tolbert_primary_episodes
    ) < int(context.gate.get("min_primary_episodes", 0) or 0):
        return ("reject", "Tolbert model candidate never entered retained Tolbert primary routing")
    if bool(context.gate.get("require_novel_command_signal", False)) and int(
        context.candidate_metrics.proposal_selected_steps
    ) <= 0:
        if not _tolbert_selection_signal_fallback_satisfied(context):
            return ("reject", "Tolbert model candidate produced no proposal-selected commands or measurable retrieval-selection gain")
    if int(context.evidence.get("proposal_selected_steps_delta", 0)) < int(
        context.gate.get("min_proposal_selected_steps_delta", 0)
    ):
        return ("reject", "Tolbert model candidate regressed proposal-selected command usage")
    if int(context.candidate_metrics.novel_valid_command_steps) < int(
        context.gate.get("min_novel_valid_command_steps", 0)
    ):
        return ("reject", "Tolbert model candidate did not produce enough verifier-valid novel commands")
    if float(context.evidence.get("novel_valid_command_rate_delta", 0.0)) < float(
        context.gate.get("min_novel_valid_command_rate_delta", 0.0)
    ):
        return ("reject", "Tolbert model candidate regressed verifier-valid novel-command rate")
    if bool(context.gate.get("require_long_horizon_non_regression", True)) and int(
        long_horizon.get("baseline_task_count", 0) or 0
    ) + int(long_horizon.get("candidate_task_count", 0) or 0) > 0:
        if float(long_horizon.get("pass_rate_delta", 0.0) or 0.0) < 0.0:
            return ("reject", "Tolbert model candidate regressed long-horizon pass rate")
    if bool(context.gate.get("require_long_horizon_novel_command_non_regression", True)) and int(
        long_horizon.get("baseline_task_count", 0) or 0
    ) + int(long_horizon.get("candidate_task_count", 0) or 0) > 0:
        if float(long_horizon.get("novel_valid_command_rate_delta", 0.0) or 0.0) < 0.0:
            return ("reject", "Tolbert model candidate regressed long-horizon verifier-valid novel-command rate")
    long_horizon_world_feedback = long_horizon.get("world_feedback", {})
    if not isinstance(long_horizon_world_feedback, dict):
        long_horizon_world_feedback = {}
    if bool(context.gate.get("require_long_horizon_world_feedback_non_regression", True)) and int(
        long_horizon.get("baseline_world_feedback_step_count", 0) or 0
    ) + int(long_horizon.get("candidate_world_feedback_step_count", 0) or 0) > 0:
        if float(long_horizon_world_feedback.get("progress_calibration_mae_gain", 0.0) or 0.0) < 0.0:
            return ("reject", "Tolbert model candidate regressed long-horizon world-feedback calibration")
    family_gate_failure = proposal_gate_failure_reason(
        context.gate,
        context.evidence,
        subject="Tolbert model candidate",
    )
    if family_gate_failure is not None:
        return ("reject", family_gate_failure)
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", "Tolbert model candidate regressed base success")
    return ("retain", "Tolbert model candidate improved learned retrieval and novel-command behavior without broader regression")


def _evaluate_qwen_adapter_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="Qwen adapter candidate",
        require_generated_lane_non_regression_default=True,
        require_failure_recovery_non_regression_default=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get("require_support_runtime_only", True)) and not bool(
        context.evidence.get("support_runtime_only", False)
    ):
        return ("reject", "Qwen adapter candidate attempted to claim primary runtime authority")
    if bool(context.gate.get("require_teacher_generation", True)) and not bool(
        context.evidence.get("teacher_generation_enabled", False)
    ):
        return ("reject", "Qwen adapter candidate disabled teacher-generation support")
    if bool(context.gate.get("require_runtime_target_declared", True)) and not bool(
        context.evidence.get("runtime_target_declared", False)
    ):
        return ("reject", "Qwen adapter candidate did not declare a runtime target")
    if int(context.evidence.get("dataset_total_examples", 0) or 0) <= 0:
        return ("reject", "Qwen adapter candidate did not produce a training dataset")
    if not str(context.evidence.get("base_model_name", "")).strip():
        return ("reject", "Qwen adapter candidate did not declare a base model")
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", "Qwen adapter candidate regressed base success")
    if not has_measurable_runtime_influence(context):
        return ("reject", "Qwen adapter candidate did not produce a measurable runtime signal")
    if (
        context.pass_rate_delta > float(context.gate.get("min_pass_rate_delta_abs", 0.0))
        or (
            context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
            and context.candidate_metrics.average_steps <= context.baseline_metrics.average_steps
        )
    ):
        return ("retain", "Qwen adapter candidate improved or preserved the coding baseline without claiming liftoff authority")
    return ("reject", "Qwen adapter candidate did not produce a retained baseline gain")


def _evaluate_capabilities_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="capability registry candidate",
        require_generated_lane_non_regression_default=False,
        require_failure_recovery_non_regression_default=False,
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get("require_capability_surface_growth", True)):
        if (
            int(context.evidence.get("module_count_delta", 0)) <= 0
            and int(context.evidence.get("external_capability_count_delta", 0)) <= 0
            and int(context.evidence.get("improvement_surface_count_delta", 0)) <= 0
        ):
            return ("reject", "capability registry candidate did not expand capability or improvement surfaces")
    return ("retain", "capability registry candidate preserved runtime quality while expanding mutable capability surfaces")


def _evaluate_skill_or_tooling_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="candidate",
        require_generated_lane_non_regression_default=False,
        require_failure_recovery_non_regression_default=False,
    )
    if common is not None:
        return common
    if context.subsystem == "tooling" and bool(context.gate.get("require_shared_repo_bundle_coherence", True)):
        shared_repo_candidate_count = int(context.evidence.get("shared_repo_candidate_count", 0) or 0)
        shared_repo_complete_candidate_count = int(context.evidence.get("shared_repo_complete_candidate_count", 0) or 0)
        shared_repo_incomplete_integrator_count = int(
            context.evidence.get("shared_repo_incomplete_integrator_candidate_count", 0) or 0
        )
        if (
            shared_repo_candidate_count > 0
            and shared_repo_complete_candidate_count <= 0
            and shared_repo_incomplete_integrator_count > 0
        ):
            return ("reject", "tool candidate artifact surfaced only incomplete shared-repo integrator histories")
    if (
        context.candidate_metrics.pass_rate > context.baseline_metrics.pass_rate
        or (
            context.candidate_metrics.pass_rate == context.baseline_metrics.pass_rate
            and context.candidate_metrics.average_steps < context.baseline_metrics.average_steps
        )
    ):
        if context.subsystem != "tooling" or bool(context.evidence.get("replay_verified", False)):
            return ("retain", "candidate improved pass rate or preserved pass rate while reducing steps")
    if _skills_learning_support_satisfied(context):
        return ("retain", "candidate preserved benchmark performance and is backed by compiled success-skill evidence")
    if context.subsystem == "tooling" and not bool(context.evidence.get("replay_verified", False)):
        return ("reject", "tool candidate was not replay-verified before retention")
    return ("reject", "candidate did not produce a verified gain")


def _evaluate_operators_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if bool(context.gate.get("require_cross_task_support", True)) and int(context.evidence.get("support_count", 0)) < int(
        context.gate.get("min_support", 2)
    ):
        return ("reject", "operator artifact does not have the required cross-task support")
    if float(context.evidence.get("transfer_pass_rate_delta", 0.0)) < float(
        context.gate.get("min_transfer_pass_rate_delta_abs", 0.05)
    ):
        return ("reject", "operator transfer did not beat raw skill transfer by the required margin")
    return ("retain", "operator transfer beat raw skill transfer on the shared held-out lane")


RETENTION_EVALUATORS: dict[str, Callable[[RetentionDecisionContext], tuple[str, str]]] = {
    "curriculum": _evaluate_curriculum_retention,
    "verifier": _evaluate_verifier_retention,
    "benchmark": _evaluate_benchmark_retention,
    "policy": _evaluate_policy_retention,
    "world_model": _evaluate_world_model_retention,
    "state_estimation": _evaluate_state_estimation_retention,
    "universe": _evaluate_universe_retention,
    "trust": _evaluate_trust_retention,
    "recovery": _evaluate_recovery_retention,
    "delegation": _evaluate_delegation_retention,
    "operator_policy": _evaluate_operator_policy_retention,
    "transition_model": _evaluate_transition_model_retention,
    "retrieval": _evaluate_retrieval_retention,
    "tolbert_model": _evaluate_tolbert_model_retention,
    "qwen_adapter": _evaluate_qwen_adapter_retention,
    "capabilities": _evaluate_capabilities_retention,
    "skills": _evaluate_skill_or_tooling_retention,
    "tooling": _evaluate_skill_or_tooling_retention,
    "operators": _evaluate_operators_retention,
}
