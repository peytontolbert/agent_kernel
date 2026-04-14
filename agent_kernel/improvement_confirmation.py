from __future__ import annotations

import math

from .evals.retention_stats import (
    family_conservative_bounds,
    paired_task_trace_report,
    paired_trajectory_report,
    wilson_interval,
)


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
    baseline_wilson_lower, baseline_wilson_upper = wilson_interval(baseline_passed, baseline_total, z=z)
    candidate_wilson_lower, candidate_wilson_upper = wilson_interval(candidate_passed, candidate_total, z=z)
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
    paired_task_report = paired_task_trace_report(baseline_runs, candidate_runs, z=z)
    paired_trajectory_report_payload = paired_trajectory_report(baseline_runs, candidate_runs, z=z)
    family_bounds = family_conservative_bounds(baseline_runs, candidate_runs, z=z)
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
        **paired_trajectory_report_payload,
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
