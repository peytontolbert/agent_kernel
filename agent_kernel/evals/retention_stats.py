from __future__ import annotations

import math


def wilson_interval(passed: int, total: int, *, z: float) -> tuple[float, float]:
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


def one_sided_sign_test_p_value(successes: int, trials: int) -> float:
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


def family_conservative_bounds(
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
        baseline_lower, baseline_upper = wilson_interval(baseline_passed, baseline_total, z=z)
        candidate_lower, candidate_upper = wilson_interval(candidate_passed, candidate_total, z=z)
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


def paired_task_trace_report(
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
    non_regression_lower, non_regression_upper = wilson_interval(
        paired_task_non_regression_count,
        paired_task_pair_count,
        z=z,
    )
    improvement_lower, improvement_upper = wilson_interval(
        paired_task_improvement_count,
        paired_task_pair_count,
        z=z,
    )
    non_regression_p_value = one_sided_sign_test_p_value(
        paired_task_non_regression_count,
        paired_task_pair_count,
    )
    improvement_p_value = one_sided_sign_test_p_value(
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
        **paired_trace_regression_report(baseline_runs, candidate_runs, z=z),
    }


def paired_trace_regression_report(
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
            baseline_score = task_trace_severity_score(baseline_trace)
            candidate_score = task_trace_severity_score(candidate_trace)
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
    non_regression_lower, non_regression_upper = wilson_interval(non_regression_count, pair_count, z=z)
    improvement_lower, improvement_upper = wilson_interval(improvement_count, pair_count, z=z)
    non_regression_p_value = one_sided_sign_test_p_value(non_regression_count, pair_count)
    improvement_p_value = one_sided_sign_test_p_value(improvement_count, pair_count)
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


def task_trace_severity_score(trace: dict[str, object]) -> float:
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


def paired_trajectory_report(
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
            baseline_signature = trajectory_signature(baseline_payload)
            candidate_signature = trajectory_signature(candidate_payload)
            baseline_score = trajectory_severity_score(baseline_payload)
            candidate_score = trajectory_severity_score(candidate_payload)
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
            alignment_rates.append(trajectory_alignment_rate(baseline_signature, candidate_signature))
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
    exact_lower, exact_upper = wilson_interval(exact_match_count, pair_count, z=z)
    non_regression_lower, non_regression_upper = wilson_interval(non_regression_count, pair_count, z=z)
    improvement_lower, improvement_upper = wilson_interval(improvement_count, pair_count, z=z)
    exact_match_p_value = one_sided_sign_test_p_value(exact_match_count, pair_count)
    non_regression_p_value = one_sided_sign_test_p_value(non_regression_count, pair_count)
    improvement_p_value = one_sided_sign_test_p_value(improvement_count, pair_count)
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


def trajectory_signature(payload: dict[str, object]) -> list[tuple[object, ...]]:
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


def trajectory_alignment_rate(
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


def trajectory_severity_score(payload: dict[str, object]) -> float:
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
