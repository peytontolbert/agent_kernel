from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import json
import math
import re

from evals.harness import compare_abstraction_transfer_modes, run_eval, scoped_eval_config

from .config import KernelConfig
from .improvement import (
    _generated_kind_pass_rate,
    _has_generated_kind,
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
    return comparison_config_for_subsystem_artifact(scoped, subsystem, artifact_path)


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

    scoped_flags = dict(flags)
    if isinstance(task_limit, int) and task_limit > 0:
        scoped_flags["task_limit"] = task_limit
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
    baseline_metrics = evaluate_subsystem_metrics(
        config=baseline_config,
        subsystem=subsystem,
        flags=scoped_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_baseline",
    )
    _emit(
        f"finalize phase=preview_prior_retained_candidate_eval subsystem={subsystem} "
        f"baseline_cycle_id={baseline_cycle_id}"
    )
    current_metrics = evaluate_subsystem_metrics(
        config=current_config,
        subsystem=subsystem,
        flags=scoped_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_candidate",
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
    progress_label_prefix: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

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
    if isinstance(task_limit, int) and task_limit > 0:
        baseline_flags["task_limit"] = task_limit
    if priority_benchmark_families:
        baseline_flags["priority_benchmark_families"] = list(priority_benchmark_families)
    if priority_benchmark_family_weights:
        baseline_flags["priority_benchmark_family_weights"] = dict(priority_benchmark_family_weights)
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
    if isinstance(task_limit, int) and task_limit > 0:
        candidate_flags["task_limit"] = task_limit
    if priority_benchmark_families:
        candidate_flags["priority_benchmark_families"] = list(priority_benchmark_families)
    if priority_benchmark_family_weights:
        candidate_flags["priority_benchmark_family_weights"] = dict(priority_benchmark_family_weights)
    _emit(f"finalize phase=preview_baseline_eval subsystem={subsystem}")
    baseline = evaluate_subsystem_metrics(
        config=baseline_config,
        subsystem=subsystem,
        flags=baseline_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_baseline",
    )
    _emit(
        f"finalize phase=preview_baseline_complete subsystem={subsystem} "
        f"baseline_pass_rate={baseline.pass_rate:.4f}"
    )
    _emit(f"finalize phase=preview_candidate_eval subsystem={subsystem}")
    candidate = evaluate_subsystem_metrics(
        config=candidate_config,
        subsystem=subsystem,
        flags=candidate_flags,
        progress_label=None if not progress_label_prefix else f"{progress_label_prefix}_candidate",
    )
    _emit(
        f"finalize phase=preview_candidate_complete subsystem={subsystem} "
        f"candidate_pass_rate={candidate.pass_rate:.4f}"
    )
    artifact_payload = None
    if artifact_path.exists():
        parsed = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            artifact_payload = effective_artifact_payload_for_retention(
                subsystem,
                parsed,
                capability_modules_path=config.capability_modules_path,
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
    return {
        "state": state,
        "reason": reason,
        "gate": gate,
        "phase_gate_report": phase_gate_report,
        "baseline": baseline,
        "candidate": candidate,
        "evidence": evidence,
        "compatibility": compatibility,
        "payload": artifact_payload,
        "prior_retained_comparison": prior_retained_comparison,
        "baseline_flags": baseline_flags,
        "candidate_flags": candidate_flags,
        "active_artifact_path": managed_active_artifact_path,
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
    generated_lane_included = bool(candidate_flags.get("include_generated", False))
    failure_recovery_lane_included = bool(candidate_flags.get("include_failure_generated", False))
    if not generated_lane_included:
        failures.append("generated-task lane was not included in autonomous cycle evaluation")
    if not failure_recovery_lane_included:
        failures.append("failure-recovery lane was not included in autonomous cycle evaluation")
    if generated_lane_included and int(candidate_metrics.generated_total) <= 0:
        failures.append("generated-task lane produced no tasks during autonomous evaluation")
    if failure_recovery_lane_included and int(candidate_metrics.generated_by_kind.get("failure_recovery", 0)) <= 0:
        failures.append("failure-recovery lane produced no generated tasks during autonomous evaluation")
    if base_subsystem_for(subsystem, capability_modules_path) in {"retrieval", "tolbert_model"}:
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
    if bool(gate.get("require_novel_command_signal", False)) and int(
        current_metrics.get("proposal_selected_steps", 0) or 0
    ) <= 0:
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
    protocol_match_id: str = "",
) -> Path:
    records = planner.load_cycle_records(config.improvement_cycles_path)
    cycle_records = [record for record in records if str(record.get("cycle_id", "")) == cycle_id]
    production_decisions = _production_decisions(records)
    summary = planner.retained_gain_summary(config.improvement_cycles_path)
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    report_path = config.improvement_reports_dir / f"cycle_report_{safe_cycle_id}.json"
    report = {
        "spec_version": "asi_v1",
        "report_kind": "improvement_cycle_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id,
        "subsystem": subsystem,
        "artifact_path": str(artifact_path),
        "candidate_artifact_path": str(artifact_update.get("candidate_artifact_path", "")),
        "active_artifact_path": str(artifact_update.get("active_artifact_path", "")),
        "artifact_kind": str(artifact_update.get("artifact_kind", "")),
        "final_state": final_state,
        "final_reason": final_reason,
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
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
                **_phase_gate_metrics_summary(phase_gate_report),
                "production_total_decisions": report["production_yield_summary"]["total_decisions"],
                "production_retained_cycles": report["production_yield_summary"]["retained_cycles"],
                "production_rejected_cycles": report["production_yield_summary"]["rejected_cycles"],
                "protocol": "autonomous",
                "protocol_match_id": str(protocol_match_id).strip(),
            },
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
    progress: Callable[[str], None] | None = None,
    protocol_match_id: str = "",
) -> tuple[str, str]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    repo_root = Path(__file__).resolve().parents[1]
    _emit(f"finalize phase=preview subsystem={subsystem}")
    preview = preview_candidate_retention(
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
        progress_label_prefix=f"{cycle_id}_{subsystem}_preview",
        progress=_emit,
    )
    managed_active_artifact_path = Path(preview["active_artifact_path"])
    baseline = preview["baseline"]
    candidate = preview["candidate"]
    evidence = dict(preview["evidence"])
    compatibility = dict(preview["compatibility"])
    artifact_payload = preview["payload"]
    state = str(preview["state"])
    reason = str(preview["reason"])
    gate = dict(preview["gate"])
    phase_gate_report = dict(preview.get("phase_gate_report", {}))
    prior_retained_comparison = preview["prior_retained_comparison"]
    _emit(
        f"finalize phase=preview_complete subsystem={subsystem} "
        f"preview_state={state} baseline_pass_rate={baseline.pass_rate:.4f} "
        f"candidate_pass_rate={candidate.pass_rate:.4f}"
    )
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
    )
    baseline_flags = dict(preview["baseline_flags"])
    candidate_flags = dict(preview["candidate_flags"])
    protocol_metrics = {
        "protocol": "autonomous",
        "protocol_match_id": str(protocol_match_id).strip(),
    }

    evaluate_record_kwargs = {}
    if base_subsystem_for(subsystem, config.capability_modules_path) == "tooling":
        replay_verified_update = persist_replay_verified_tool_artifact(
            artifact_path,
            cycle_id=cycle_id,
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
                run_eval(
                    config=baseline_config,
                    progress_label=f"{cycle_id}_{subsystem}_confirmation_baseline",
                    **baseline_flags,
                ),
                run_eval(
                    config=candidate_config,
                    progress_label=f"{cycle_id}_{subsystem}_confirmation_candidate",
                    **candidate_flags,
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
        holdout_baseline = evaluate_subsystem_metrics(
            config=holdout_baseline_config,
            subsystem=subsystem,
            flags=holdout_baseline_flags,
            progress_label=f"{cycle_id}_{subsystem}_holdout_baseline",
        )
        holdout_candidate = evaluate_subsystem_metrics(
            config=holdout_candidate_config,
            subsystem=subsystem,
            flags=holdout_candidate_flags,
            progress_label=f"{cycle_id}_{subsystem}_holdout_candidate",
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
                task_limit=None,
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
                },
                candidate_artifact_path=str(artifact_path),
                active_artifact_path=str(managed_active_artifact_path),
                compatibility=compatibility,
            ),
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
        config.tolbert_liftoff_report_path.write_text(
            json.dumps(
                {
                    "spec_version": "asi_v1",
                    "artifact_kind": "liftoff_gate_report",
                    "cycle_id": cycle_id,
                    "subsystem": subsystem,
                    "report": liftoff_report.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
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
            managed_active_artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        protocol_match_id=protocol_match_id,
    )
    _emit(f"finalize phase=done subsystem={subsystem} state={state}")
    return state, reason
