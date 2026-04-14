from __future__ import annotations

import math
from typing import Any

from evals.metrics import EvalMetrics

from ...improvement_engine import ImprovementExperiment, ImprovementSearchBudget


def coding_strength_summary(metrics: EvalMetrics) -> dict[str, object]:
    coding_families = ("repository", "project", "integration")
    observed_totals: dict[str, int] = {}
    observed_passed: dict[str, int] = {}
    for family in coding_families:
        total = int(metrics.total_by_benchmark_family.get(family, 0) or 0)
        if total <= 0:
            continue
        observed_totals[family] = total
        observed_passed[family] = int(metrics.passed_by_benchmark_family.get(family, 0) or 0)
    coding_total = sum(observed_totals.values())
    coding_passed = sum(observed_passed.values())
    coding_pass_rate = 0.0 if coding_total <= 0 else float(coding_passed) / float(coding_total)
    return {
        "families": list(coding_families),
        "observed_families": sorted(observed_totals),
        "observed_family_count": len(observed_totals),
        "total": coding_total,
        "passed": coding_passed,
        "pass_rate": round(coding_pass_rate, 4),
        "overall_pass_rate": round(float(metrics.pass_rate), 4),
        "generated_pass_rate": round(float(metrics.generated_pass_rate), 4),
    }


def broad_coding_observe_diversification_signal(metrics: EvalMetrics) -> dict[str, object]:
    coding_summary = coding_strength_summary(metrics)
    broad_coding_families = ("repository", "project", "integration", "repo_chore")
    observed_families = {
        family
        for family in broad_coding_families
        if int(metrics.total_by_benchmark_family.get(family, 0) or 0) > 0
    }
    broad_coding_total = sum(
        int(metrics.total_by_benchmark_family.get(family, 0) or 0)
        for family in broad_coding_families
    )
    broad_coding_passed = sum(
        int(metrics.passed_by_benchmark_family.get(family, 0) or 0)
        for family in broad_coding_families
    )
    coding_total = max(int(coding_summary.get("total", 0) or 0), broad_coding_total)
    coding_pass_rate = (
        0.0 if broad_coding_total <= 0 else float(broad_coding_passed) / float(broad_coding_total)
    )
    generated_total = int(getattr(metrics, "generated_total", 0) or 0)
    generated_pass_rate = float(coding_summary.get("generated_pass_rate", 0.0) or 0.0)
    generated_ready = generated_total <= 0 or generated_pass_rate >= 0.95
    low_confidence = int(getattr(metrics, "low_confidence_episodes", 0) or 0)
    trusted_retrieval_steps = int(getattr(metrics, "trusted_retrieval_steps", 0) or 0)
    overall_total = max(0, int(getattr(metrics, "total", 0) or 0))
    active = (
        len(observed_families) >= 3
        and coding_total >= 4
        and coding_pass_rate >= 0.95
        and float(coding_summary.get("overall_pass_rate", 0.0) or 0.0) >= 0.95
        and generated_ready
    )
    retrieval_emergency = bool(
        overall_total > 0
        and (
            low_confidence >= max(2, int(math.ceil(float(overall_total) * 0.4)))
            or (low_confidence > 0 and trusted_retrieval_steps <= 0)
        )
    )
    return {
        "active": active,
        "retrieval_emergency": retrieval_emergency,
        "observed_families": sorted(observed_families),
        "observed_family_count": len(observed_families),
        "coding_total": coding_total,
        "coding_pass_rate": round(coding_pass_rate, 4),
        "overall_pass_rate": round(float(coding_summary.get("overall_pass_rate", 0.0) or 0.0), 4),
        "generated_total": generated_total,
        "generated_pass_rate": round(generated_pass_rate, 4),
        "generated_ready": generated_ready,
        "primary_only_broad_observe": generated_total <= 0,
        "low_confidence_episodes": low_confidence,
        "trusted_retrieval_steps": trusted_retrieval_steps,
    }


def allow_qwen_adapter_support_runtime(metrics: EvalMetrics) -> tuple[bool, dict[str, object]]:
    summary = coding_strength_summary(metrics)
    strong_coding_round = (
        int(summary.get("observed_family_count", 0) or 0) >= 2
        and int(summary.get("total", 0) or 0) >= 4
        and float(summary.get("pass_rate", 0.0) or 0.0) >= 0.8
        and float(summary.get("overall_pass_rate", 0.0) or 0.0) >= 0.8
    )
    summary["strong_coding_round"] = strong_coding_round
    return (not strong_coding_round), summary


def experiment_score(
    candidate: ImprovementExperiment,
    *,
    effective_subsystem: str | None = None,
) -> float:
    uncertainty_penalty = 0.0
    evidence = candidate.evidence
    subsystem = effective_subsystem or candidate.subsystem
    if subsystem == "retrieval":
        total = int(evidence.get("total", 0))
        low_confidence = int(evidence.get("low_confidence_episodes", 0))
        if total > 0:
            uncertainty_penalty += max(0.0, min(0.15, low_confidence / total)) * 0.1
    if subsystem == "benchmark" and int(evidence.get("benchmark_candidate_total", 0)) == 0:
        uncertainty_penalty += 0.02
    raw = (candidate.priority * candidate.expected_gain) / max(1, candidate.estimated_cost)
    return round(max(0.0, raw - uncertainty_penalty), 4)


def recommend_campaign_budget(
    planner: Any,
    metrics: EvalMetrics,
    *,
    max_width: int = 2,
) -> ImprovementSearchBudget:
    planner_controls = planner._improvement_planner_controls()
    ranked = planner.rank_experiments(metrics)
    resolved_max_width = max(1, max_width)
    if not ranked:
        return ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=0.0,
            selected_ids=[],
            reasons=["no ranked experiments were available"],
        )
    top_score = float(ranked[0].score)
    selected_ids = [ranked[0].subsystem]
    selected_surfaces = {planner._campaign_surface_key(ranked[0].subsystem)}
    reasons = [f"top subsystem {ranked[0].subsystem} score={top_score:.4f}"]
    if resolved_max_width <= 1 or len(ranked) == 1:
        return ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=top_score,
            selected_ids=selected_ids,
            reasons=reasons,
        )

    close_relative_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="campaign",
        field="close_score_relative_threshold",
        legacy_field="campaign_close_score_relative_threshold",
        default=0.9,
        min_value=0.5,
        max_value=0.99,
    )
    close_margin_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="campaign",
        field="close_score_margin_threshold",
        legacy_field="campaign_close_score_margin_threshold",
        default=0.01,
        min_value=0.0,
        max_value=0.1,
    )
    history_relative_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="campaign",
        field="history_relative_threshold",
        legacy_field="campaign_history_relative_threshold",
        default=0.8,
        min_value=0.5,
        max_value=0.99,
    )
    candidate_relative_floor = min(close_relative_threshold, history_relative_threshold)
    for candidate in ranked[1:]:
        if len(selected_ids) >= resolved_max_width:
            break
        if planner._campaign_surface_key(candidate.subsystem) in selected_surfaces:
            continue
        relative_score = 0.0 if top_score <= 0.0 else float(candidate.score) / top_score
        score_margin = top_score - float(candidate.score)
        if relative_score < candidate_relative_floor and score_margin > close_margin_threshold:
            continue
        selected_ids.append(candidate.subsystem)
        selected_surfaces.add(planner._campaign_surface_key(candidate.subsystem))
        reasons.append(
            f"added {candidate.subsystem} due to scored breadth eligibility (relative={relative_score:.3f}, margin={score_margin:.4f})"
        )
    return ImprovementSearchBudget(
        scope="campaign",
        width=max(1, len(selected_ids)),
        max_width=resolved_max_width,
        strategy="adaptive_history",
        top_score=top_score,
        selected_ids=selected_ids,
        reasons=reasons,
    )


def recommend_variant_budget(
    planner: Any,
    experiment: ImprovementExperiment,
    metrics: EvalMetrics,
    *,
    max_width: int = 2,
) -> ImprovementSearchBudget:
    planner_controls = planner._improvement_planner_controls()
    ranked = planner.rank_variants(experiment, metrics)
    resolved_max_width = max(1, max_width)
    if not ranked:
        return ImprovementSearchBudget(
            scope="variant",
            width=1,
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=0.0,
            selected_ids=[],
            reasons=[f"no ranked variants were available for subsystem={experiment.subsystem}"],
        )
    top_score = float(ranked[0].score)
    selected_ids = [ranked[0].variant_id]
    reasons = [f"top variant {ranked[0].variant_id} score={top_score:.4f}"]
    if resolved_max_width <= 1 or len(ranked) == 1:
        return ImprovementSearchBudget(
            scope="variant",
            width=1,
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=top_score,
            selected_ids=selected_ids,
            reasons=reasons,
        )

    close_relative_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="variant",
        field="close_score_relative_threshold",
        legacy_field="variant_close_score_relative_threshold",
        default=0.92,
        min_value=0.5,
        max_value=0.99,
    )
    close_margin_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="variant",
        field="close_score_margin_threshold",
        legacy_field="variant_close_score_margin_threshold",
        default=0.003,
        min_value=0.0,
        max_value=0.05,
    )
    history_relative_threshold = planner._planner_guardrail_float(
        planner_controls,
        scope="variant",
        field="history_relative_threshold",
        legacy_field="variant_history_relative_threshold",
        default=0.85,
        min_value=0.5,
        max_value=0.99,
    )
    candidate_relative_floor = min(close_relative_threshold, history_relative_threshold)
    for variant in ranked[1:]:
        if len(selected_ids) >= resolved_max_width:
            break
        relative_score = 0.0 if top_score <= 0.0 else float(variant.score) / top_score
        score_margin = top_score - float(variant.score)
        if relative_score < candidate_relative_floor and score_margin > close_margin_threshold:
            continue
        selected_ids.append(variant.variant_id)
        reasons.append(
            f"added {variant.variant_id} due to scored breadth eligibility (relative={relative_score:.3f}, margin={score_margin:.4f})"
        )
    return ImprovementSearchBudget(
        scope="variant",
        width=max(1, len(selected_ids)),
        max_width=resolved_max_width,
        strategy="adaptive_history",
        top_score=top_score,
        selected_ids=selected_ids,
        reasons=reasons,
    )
