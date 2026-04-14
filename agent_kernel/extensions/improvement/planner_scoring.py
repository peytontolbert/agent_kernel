from __future__ import annotations

from typing import Any

from evals.metrics import EvalMetrics

from ...improvement_engine import ImprovementExperiment, ImprovementVariant


def score_experiment(
    planner: Any,
    candidate: ImprovementExperiment,
    metrics: EvalMetrics,
    *,
    planner_controls: dict[str, object] | None = None,
    learning_candidate_summary: dict[str, dict[str, object]] | None = None,
    trust_summary: dict[str, object] | None = None,
) -> ImprovementExperiment:
    resolved_planner_controls = planner_controls if planner_controls is not None else planner._improvement_planner_controls()
    effective_subsystem = planner._base_subsystem(candidate.subsystem)
    candidate, mutation_evidence = planner._apply_improvement_planner_mutation(
        candidate,
        planner_controls=resolved_planner_controls,
    )
    history = planner.subsystem_history_summary(subsystem=candidate.subsystem)
    recent_history = planner.recent_subsystem_activity_summary(subsystem=candidate.subsystem)
    bootstrap_penalty, penalty_reasons = planner._bootstrap_penalty(
        candidate,
        history,
        planner_controls=resolved_planner_controls,
        effective_subsystem=effective_subsystem,
    )
    cold_start_penalty, cold_start_reasons = planner._cold_start_low_confidence_penalty(
        candidate,
        history,
        recent_history,
        planner_controls=resolved_planner_controls,
        effective_subsystem=effective_subsystem,
    )
    stalled_selection_penalty, stalled_selection_reasons = planner._recent_stalled_selection_penalty(
        recent_history,
        planner_controls=resolved_planner_controls,
    )
    observation_timeout_penalty, observation_timeout_reasons = planner._recent_observation_timeout_penalty(
        recent_history,
        planner_controls=resolved_planner_controls,
    )
    promotion_failure_penalty, promotion_failure_reasons = planner._recent_promotion_failure_penalty(
        recent_history,
        planner_controls=resolved_planner_controls,
    )
    benchmark_no_yield_penalty = 0.0
    benchmark_no_yield_reasons: list[str] = []
    if effective_subsystem == "benchmark":
        selected_cycles = int(recent_history.get("selected_cycles", 0) or 0)
        retained_cycles = int(recent_history.get("retained_cycles", 0) or 0)
        no_yield_cycles = int(recent_history.get("no_yield_cycles", 0) or 0)
        if selected_cycles >= 2 and retained_cycles == 0 and no_yield_cycles > 0:
            penalty_per_cycle = planner._planner_control_float(
                resolved_planner_controls,
                "benchmark_recent_no_yield_penalty_per_cycle",
                0.03,
                min_value=0.0,
                max_value=0.2,
            )
            penalty_cap = planner._planner_control_float(
                resolved_planner_controls,
                "benchmark_recent_no_yield_penalty_cap",
                0.2,
                min_value=0.0,
                max_value=0.5,
            )
            benchmark_no_yield_penalty = min(penalty_cap, no_yield_cycles * penalty_per_cycle)
            benchmark_no_yield_penalty = round(benchmark_no_yield_penalty, 4)
            if benchmark_no_yield_penalty > 0.0:
                benchmark_no_yield_reasons.append(
                    f"benchmark_recent_no_yield_penalty={benchmark_no_yield_penalty:.4f}"
                )
    memory_source_bonus, memory_source_evidence = planner._memory_source_experiment_bonus(
        effective_subsystem,
        metrics,
    )
    score_bias = planner._planner_control_subsystem_float(
        resolved_planner_controls,
        "subsystem_score_bias",
        candidate.subsystem,
        fallback_subsystem=effective_subsystem,
        default=0.0,
        min_value=-0.1,
        max_value=0.1,
    )
    learning_candidate_bonus, learning_candidate_evidence = planner._learning_candidate_experiment_bonus(
        candidate.subsystem,
        summary_by_subsystem=learning_candidate_summary,
    )
    measurement_guardrail_penalty, measurement_guardrail_reasons, measurement_guardrail_evidence = (
        planner._measurement_guardrail_penalty(
            candidate,
            metrics,
            trust_summary=trust_summary,
        )
    )
    score = round(
        max(
            0.0,
            planner._experiment_score(candidate, effective_subsystem=effective_subsystem)
            + planner._history_bonus(history)
            + planner._recent_history_bonus(recent_history)
            + memory_source_bonus
            + learning_candidate_bonus
            - bootstrap_penalty
            - cold_start_penalty
            - stalled_selection_penalty
            - observation_timeout_penalty
            - promotion_failure_penalty
            - benchmark_no_yield_penalty
            - measurement_guardrail_penalty
            + score_bias,
        ),
        4,
    )
    evidence = dict(candidate.evidence)
    evidence["base_subsystem"] = effective_subsystem
    evidence["history"] = history
    evidence["recent_history"] = recent_history
    if memory_source_evidence:
        evidence["memory_source_pressure"] = memory_source_evidence
    if learning_candidate_evidence:
        evidence["learning_candidate_pressure"] = learning_candidate_evidence
    if measurement_guardrail_evidence:
        evidence["measurement_guardrails"] = measurement_guardrail_evidence
    if mutation_evidence:
        evidence["improvement_planner_mutation"] = mutation_evidence
    selection_penalties = [
        *penalty_reasons,
        *cold_start_reasons,
        *stalled_selection_reasons,
        *observation_timeout_reasons,
        *promotion_failure_reasons,
        *benchmark_no_yield_reasons,
        *measurement_guardrail_reasons,
    ]
    if selection_penalties:
        evidence["selection_penalties"] = selection_penalties
    return ImprovementExperiment(
        subsystem=candidate.subsystem,
        reason=candidate.reason,
        priority=candidate.priority,
        expected_gain=candidate.expected_gain,
        estimated_cost=candidate.estimated_cost,
        score=score,
        evidence=evidence,
    )


def measurement_guardrail_penalty(
    planner: Any,
    candidate: ImprovementExperiment,
    metrics: EvalMetrics,
    *,
    trust_summary: dict[str, object] | None = None,
) -> tuple[float, list[str], dict[str, object]]:
    effective_subsystem = planner._base_subsystem(candidate.subsystem)
    resolved_trust_summary = trust_summary if trust_summary is not None else planner.trust_ledger_summary()
    if not resolved_trust_summary:
        return 0.0, [], {}
    false_pass_risk_rate = float(resolved_trust_summary.get("false_pass_risk_rate", 0.0) or 0.0)
    unexpected_change_report_rate = float(resolved_trust_summary.get("unexpected_change_report_rate", 0.0) or 0.0)
    distinct_family_gap = int(resolved_trust_summary.get("distinct_family_gap", 0) or 0)
    missing_required_families = (
        [
            str(value).strip()
            for value in resolved_trust_summary.get("missing_required_families", [])
            if str(value).strip()
        ]
        if isinstance(resolved_trust_summary.get("missing_required_families", []), list)
        else []
    )
    evaluator_alignment_penalty = 0.0
    benchmark_shape_penalty = 0.0
    if effective_subsystem not in {"trust", "recovery", "operating_envelope", "universe_constitution"}:
        evaluator_alignment_penalty = min(
            0.04,
            max(
                false_pass_risk_rate,
                unexpected_change_report_rate,
                0.02 if str(resolved_trust_summary.get("overall_status", "")).strip() in {"bootstrap", "restricted"} else 0.0,
            )
            + min(0.02, (distinct_family_gap * 0.005) + (len(missing_required_families) * 0.005)),
        )
    if effective_subsystem == "benchmark":
        observed_family_totals = {
            family: int(total or 0)
            for family, total in metrics.total_by_benchmark_family.items()
            if str(family).strip() and str(family).strip() != "benchmark_candidate" and int(total or 0) > 0
        }
        observed_family_total = sum(observed_family_totals.values())
        largest_family_share = (
            max(observed_family_totals.values()) / max(1, observed_family_total)
            if observed_family_totals
            else 0.0
        )
        if len(observed_family_totals) < 2:
            benchmark_shape_penalty += 0.015
        if metrics.total < 16:
            benchmark_shape_penalty += min(0.015, (16 - max(0, metrics.total)) * 0.0015)
        if largest_family_share > 0.7:
            benchmark_shape_penalty += min(0.015, (largest_family_share - 0.7) * 0.05)
        if metrics.generated_total < max(4, min(8, metrics.total)):
            benchmark_shape_penalty += 0.01
        if false_pass_risk_rate > 0.0 or unexpected_change_report_rate > 0.0:
            benchmark_shape_penalty += min(0.02, false_pass_risk_rate + unexpected_change_report_rate)
        benchmark_shape_penalty = min(0.06, benchmark_shape_penalty)
    total_penalty = round(evaluator_alignment_penalty + benchmark_shape_penalty, 4)
    if total_penalty <= 0.0:
        return 0.0, [], {}
    reasons: list[str] = []
    if evaluator_alignment_penalty > 0.0:
        reasons.append(f"evaluator_alignment_penalty={evaluator_alignment_penalty:.4f}")
    if benchmark_shape_penalty > 0.0:
        reasons.append(f"benchmark_shape_penalty={benchmark_shape_penalty:.4f}")
    return total_penalty, reasons, {
        "total_penalty": total_penalty,
        "evaluator_alignment_penalty": round(evaluator_alignment_penalty, 4),
        "benchmark_shape_penalty": round(benchmark_shape_penalty, 4),
        "false_pass_risk_rate": round(false_pass_risk_rate, 4),
        "unexpected_change_report_rate": round(unexpected_change_report_rate, 4),
        "distinct_family_gap": distinct_family_gap,
        "missing_required_families": missing_required_families,
    }


def bootstrap_penalty(
    planner: Any,
    candidate: ImprovementExperiment,
    history: dict[str, object],
    *,
    planner_controls: dict[str, object] | None = None,
    effective_subsystem: str | None = None,
) -> tuple[float, list[str]]:
    total_decisions = int(history.get("total_decisions", 0))
    retained_cycles = int(history.get("retained_cycles", 0))
    rejected_cycles = int(history.get("rejected_cycles", 0))
    if total_decisions < 2 or retained_cycles > 0 or rejected_cycles < 2:
        return 0.0, []
    evidence = candidate.evidence
    subsystem = effective_subsystem or candidate.subsystem
    multiplier = planner._planner_control_subsystem_float(
        planner_controls or {},
        "bootstrap_penalty_multiplier",
        candidate.subsystem,
        fallback_subsystem=subsystem,
        default=1.0,
        min_value=0.0,
        max_value=2.0,
    )
    if subsystem == "benchmark" and int(evidence.get("benchmark_candidate_total", 0)) == 0:
        penalty = round(0.04 * multiplier, 4)
        return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
    if subsystem == "verifier" and int(evidence.get("verifier_memory_total", 0)) == 0:
        penalty = round(0.04 * multiplier, 4)
        return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
    if subsystem == "operators" and int(evidence.get("skill_transfer_total", 0)) == 0:
        penalty = round(0.035 * multiplier, 4)
        return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
    return 0.0, []


def cold_start_low_confidence_penalty(
    planner: Any,
    candidate: ImprovementExperiment,
    history: dict[str, object],
    recent_history: dict[str, object],
    *,
    planner_controls: dict[str, object] | None = None,
    effective_subsystem: str | None = None,
) -> tuple[float, list[str]]:
    subsystem = effective_subsystem or candidate.subsystem
    if subsystem not in {"retrieval", "tolbert_model", "qwen_adapter"}:
        return 0.0, []
    evidence = candidate.evidence if isinstance(candidate.evidence, dict) else {}
    if subsystem == "qwen_adapter" and not bool(evidence.get("support_runtime_only", False)):
        return 0.0, []
    if int(history.get("total_decisions", 0) or 0) > 0:
        return 0.0, []
    if int(recent_history.get("selected_cycles", 0) or 0) > 0:
        return 0.0, []
    total = int(evidence.get("total", 0) or 0)
    low_confidence = int(evidence.get("low_confidence_episodes", 0) or 0)
    if total <= 0 or low_confidence <= 0:
        return 0.0, []
    raw_score = planner._experiment_score(candidate, effective_subsystem=subsystem)
    default_cap = 0.12 if subsystem == "retrieval" else 0.09 if subsystem == "qwen_adapter" else 0.1
    cap = planner._planner_control_subsystem_float(
        planner_controls or {},
        "cold_start_low_confidence_score_cap",
        candidate.subsystem,
        fallback_subsystem=subsystem,
        default=default_cap,
        min_value=0.0,
        max_value=0.25,
    )
    if raw_score <= cap:
        return 0.0, []
    penalty = round(raw_score - cap, 4)
    return penalty, [f"cold_start_low_confidence_penalty={penalty:.4f}"]


def recent_stalled_selection_penalty(
    planner: Any,
    recent_activity: dict[str, object],
    *,
    planner_controls: dict[str, object] | None = None,
) -> tuple[float, list[str]]:
    selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
    retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
    no_yield_cycles = int(recent_activity.get("no_yield_cycles", 0) or 0)
    recent_incomplete_cycles = int(recent_activity.get("recent_incomplete_cycles", 0) or 0)
    recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0) or 0)
    if selected_cycles < 2 or retained_cycles > 0 or no_yield_cycles <= 0:
        return 0.0, []
    resolved_planner_controls = planner_controls or {}
    penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_stalled_selection_penalty_per_cycle",
        0.015,
        min_value=0.0,
        max_value=0.05,
    )
    incomplete_bonus_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_stalled_incomplete_bonus_per_cycle",
        0.01,
        min_value=0.0,
        max_value=0.04,
    )
    repeated_phase_gate_bonus_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_stalled_phase_gate_bonus_per_cycle",
        0.005,
        min_value=0.0,
        max_value=0.03,
    )
    penalty = min(
        0.08,
        (no_yield_cycles * penalty_per_cycle)
        + (recent_incomplete_cycles * incomplete_bonus_per_cycle)
        + (recent_phase_gate_failure_cycles * repeated_phase_gate_bonus_per_cycle),
    )
    if penalty <= 0.0:
        return 0.0, []
    return round(penalty, 4), [f"recent_stalled_selection_penalty={penalty:.4f}"]


def recent_observation_timeout_penalty(
    planner: Any,
    recent_activity: dict[str, object],
    *,
    planner_controls: dict[str, object] | None = None,
) -> tuple[float, list[str]]:
    selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
    retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
    observation_timeout_cycles = int(recent_activity.get("recent_observation_timeout_cycles", 0) or 0)
    budgeted_timeout_cycles = int(recent_activity.get("recent_budgeted_observation_timeout_cycles", 0) or 0)
    repeated_timeout_budget_source_count = int(
        recent_activity.get("repeated_observation_timeout_budget_source_count", 0) or 0
    )
    if selected_cycles <= 0 or retained_cycles > 0 or observation_timeout_cycles <= 0:
        return 0.0, []
    resolved_planner_controls = planner_controls or {}
    penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_observation_timeout_penalty_per_cycle",
        0.0125,
        min_value=0.0,
        max_value=0.05,
    )
    budgeted_bonus_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_budgeted_observation_timeout_bonus_per_cycle",
        0.005,
        min_value=0.0,
        max_value=0.03,
    )
    repeated_source_bonus_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "recent_repeated_observation_timeout_source_bonus_per_cycle",
        0.003,
        min_value=0.0,
        max_value=0.02,
    )
    penalty = min(
        0.08,
        (observation_timeout_cycles * penalty_per_cycle)
        + (budgeted_timeout_cycles * budgeted_bonus_per_cycle)
        + (max(0, repeated_timeout_budget_source_count - 1) * repeated_source_bonus_per_cycle),
    )
    if penalty <= 0.0:
        return 0.0, []
    return round(penalty, 4), [f"recent_observation_timeout_penalty={penalty:.4f}"]


def recent_promotion_failure_penalty(
    planner: Any,
    recent_activity: dict[str, object],
    *,
    planner_controls: dict[str, object] | None = None,
) -> tuple[float, list[str]]:
    selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
    rejected_cycles = int(recent_activity.get("rejected_cycles", 0) or 0)
    retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
    recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0) or 0)
    rejected_pass_rate_delta = max(
        0.0,
        -float(recent_activity.get("average_rejected_pass_rate_delta", 0.0) or 0.0),
    )
    if selected_cycles <= 0 or rejected_cycles <= 0 or retained_cycles > rejected_cycles:
        return 0.0, []
    resolved_planner_controls = planner_controls or {}
    rejected_cycle_penalty = planner._planner_control_float(
        resolved_planner_controls,
        "recent_promotion_reject_penalty_per_cycle",
        0.014,
        min_value=0.0,
        max_value=0.05,
    )
    rejected_delta_weight = planner._planner_control_float(
        resolved_planner_controls,
        "recent_promotion_reject_pass_rate_delta_weight",
        0.35,
        min_value=0.0,
        max_value=1.0,
    )
    phase_gate_penalty = planner._planner_control_float(
        resolved_planner_controls,
        "recent_promotion_phase_gate_penalty_per_cycle",
        0.008,
        min_value=0.0,
        max_value=0.03,
    )
    penalty = min(
        0.1,
        (rejected_cycles * rejected_cycle_penalty)
        + (rejected_pass_rate_delta * rejected_delta_weight)
        + (recent_phase_gate_failure_cycles * phase_gate_penalty),
    )
    if penalty <= 0.0:
        return 0.0, []
    return round(penalty, 4), [f"recent_promotion_failure_penalty={penalty:.4f}"]


def score_variant(
    planner: Any,
    experiment: ImprovementExperiment,
    variant: ImprovementVariant,
    *,
    planner_controls: dict[str, object] | None = None,
) -> ImprovementVariant:
    resolved_planner_controls = planner_controls if planner_controls is not None else planner._improvement_planner_controls()
    effective_subsystem = planner._base_subsystem(variant.subsystem)
    variant, mutation_evidence = planner._apply_variant_planner_mutation(
        variant,
        planner_controls=resolved_planner_controls,
    )
    subsystem_history = planner.subsystem_history_summary(subsystem=experiment.subsystem)
    variant_history = planner.variant_history_summary(
        subsystem=experiment.subsystem,
        variant_id=variant.variant_id,
    )
    variant_recent_history = planner.recent_variant_activity_summary(
        subsystem=experiment.subsystem,
        variant_id=variant.variant_id,
    )
    score_bias = planner._planner_control_variant_float(
        resolved_planner_controls,
        "variant_score_bias",
        variant.subsystem,
        variant.variant_id,
        fallback_subsystem=effective_subsystem,
        default=0.0,
        min_value=-0.05,
        max_value=0.05,
    )
    exploration_bonus = planner._variant_exploration_bonus(
        subsystem_history=subsystem_history,
        variant_history=variant_history,
        variant_recent_history=variant_recent_history,
        planner_controls=resolved_planner_controls,
    )
    strategy_memory_adjustment, strategy_memory_evidence = planner._strategy_memory_variant_lineage_adjustment(
        dict(experiment.evidence.get("strategy_candidate", {})) if isinstance(experiment.evidence, dict) else {},
        variant,
    )
    score = round(
        max(
            0.0,
            planner._variant_score(variant)
            + (planner._history_bonus(subsystem_history) * 0.35)
            + planner._history_bonus(variant_history, variant_specific=True)
            + planner._recent_history_bonus(variant_recent_history, variant_specific=True)
            + exploration_bonus
            + strategy_memory_adjustment
            + score_bias,
        ),
        4,
    )
    controls = dict(variant.controls)
    controls["base_subsystem"] = effective_subsystem
    controls["history"] = {
        "subsystem": subsystem_history,
        "variant": variant_history,
        "recent_variant": variant_recent_history,
    }
    if mutation_evidence:
        controls["planner_mutation"] = mutation_evidence
    if exploration_bonus > 0.0:
        controls["variant_exploration_bonus"] = round(exploration_bonus, 4)
    if strategy_memory_evidence:
        controls["strategy_memory_variant_lineage"] = strategy_memory_evidence
    return ImprovementVariant(
        subsystem=variant.subsystem,
        variant_id=variant.variant_id,
        description=variant.description,
        expected_gain=variant.expected_gain,
        estimated_cost=variant.estimated_cost,
        score=score,
        controls=controls,
    )


def variant_exploration_bonus(
    planner: Any,
    *,
    subsystem_history: dict[str, object],
    variant_history: dict[str, object],
    variant_recent_history: dict[str, object],
    planner_controls: dict[str, object],
) -> float:
    if int(variant_history.get("total_decisions", 0)) > 0:
        return 0.0
    subsystem_decisions = int(subsystem_history.get("total_decisions", 0))
    if subsystem_decisions < 2:
        return 0.0
    if int(subsystem_history.get("retained_cycles", 0)) <= int(subsystem_history.get("rejected_cycles", 0)):
        return 0.0
    if str(variant_recent_history.get("last_decision_state", "")).strip():
        return 0.0
    if int(variant_recent_history.get("no_yield_cycles", 0)) > 0:
        return 0.0
    if int(variant_recent_history.get("recent_incomplete_cycles", 0)) > 0:
        return 0.0
    if int(variant_recent_history.get("recent_reconciled_failure_cycles", 0)) > 0:
        return 0.0
    if int(variant_recent_history.get("recent_regression_cycles", 0)) > 0:
        return 0.0
    return planner._planner_control_float(
        planner_controls,
        "variant_exploration_bonus",
        0.004,
        min_value=0.0,
        max_value=0.03,
    )
