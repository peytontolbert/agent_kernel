from __future__ import annotations

from evals.metrics import EvalMetrics


def candidate_family_failure_rate(metrics: EvalMetrics, family: str) -> float:
    total = metrics.total_by_benchmark_family.get(family, 0)
    if total == 0:
        return 1.0
    passed = metrics.passed_by_benchmark_family.get(family, 0)
    return max(0.0, min(1.0, 1.0 - (passed / total)))


def trusted_carryover_repair_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    eligible_successes = 0
    converted_repairs = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        if str(outcome.get("termination_reason", "")).strip() != "success":
            continue
        if str(outcome.get("difficulty", "")).strip() != "long_horizon":
            continue
        eligible_successes += 1
        if int(outcome.get("trusted_retrieval_carryover_verified_steps", 0) or 0) <= 0:
            continue
        converted_repairs += 1
    return converted_repairs / max(1, eligible_successes)


def trusted_carryover_verified_steps(metrics: EvalMetrics) -> int:
    if not isinstance(metrics.task_outcomes, dict):
        return 0
    verified_steps = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        verified_steps += int(outcome.get("trusted_retrieval_carryover_verified_steps", 0) or 0)
    return verified_steps


def family_discrimination_gain(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> float:
    candidate_families = {
        family
        for family in candidate_metrics.total_by_origin_benchmark_family
        if candidate_metrics.total_by_origin_benchmark_family.get(family, 0) > 0
    }
    if not candidate_families:
        return 0.0
    deltas = [
        candidate_metrics.origin_benchmark_family_pass_rate(family)
        - baseline_metrics.origin_benchmark_family_pass_rate(family)
        for family in candidate_families
    ]
    if not deltas:
        return 0.0
    mean_delta = sum(deltas) / len(deltas)
    worst_delta = min(deltas)
    if worst_delta < 0.0:
        return round(worst_delta, 4)
    return round(mean_delta, 4)


def family_pass_rate_delta_map(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> dict[str, float]:
    candidate_families = {
        family
        for family in (
            set(baseline_metrics.total_by_origin_benchmark_family)
            | set(candidate_metrics.total_by_origin_benchmark_family)
        )
        if baseline_metrics.total_by_origin_benchmark_family.get(family, 0) > 0
        or candidate_metrics.total_by_origin_benchmark_family.get(family, 0) > 0
    }
    return {
        family: round(
            candidate_metrics.origin_benchmark_family_pass_rate(family)
            - baseline_metrics.origin_benchmark_family_pass_rate(family),
            4,
        )
        for family in sorted(candidate_families)
    }


def family_regression_count(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> int:
    candidate_families = {
        family
        for family in candidate_metrics.total_by_origin_benchmark_family
        if candidate_metrics.total_by_origin_benchmark_family.get(family, 0) > 0
    }
    return sum(
        1
        for family in candidate_families
        if candidate_metrics.origin_benchmark_family_pass_rate(family)
        < baseline_metrics.origin_benchmark_family_pass_rate(family)
    )


def family_worst_delta(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> float:
    candidate_families = {
        family
        for family in candidate_metrics.total_by_origin_benchmark_family
        if candidate_metrics.total_by_origin_benchmark_family.get(family, 0) > 0
    }
    if not candidate_families:
        return 0.0
    return round(
        min(
            candidate_metrics.origin_benchmark_family_pass_rate(family)
            - baseline_metrics.origin_benchmark_family_pass_rate(family)
            for family in candidate_families
        ),
        4,
    )


__all__ = [
    "candidate_family_failure_rate",
    "family_discrimination_gain",
    "family_pass_rate_delta_map",
    "family_regression_count",
    "family_worst_delta",
    "trusted_carryover_repair_rate",
    "trusted_carryover_verified_steps",
]
