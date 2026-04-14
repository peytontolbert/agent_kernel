from __future__ import annotations

from evals.metrics import EvalMetrics


def proposal_gate_failure_reason(
    gate: dict[str, object],
    evidence: dict[str, object],
    *,
    subject: str,
) -> str | None:
    reasons = proposal_gate_failure_reasons_by_benchmark_family(
        gate,
        evidence,
        subject=subject,
    )
    if not reasons:
        return None
    first_family = sorted(reasons)[0]
    return reasons[first_family]


def proposal_gate_failure_reasons_by_benchmark_family(
    gate: dict[str, object],
    evidence: dict[str, object],
    *,
    subject: str,
) -> dict[str, str]:
    family_gates = gate.get("proposal_gate_by_benchmark_family", {})
    family_metrics = evidence.get("proposal_metrics_by_benchmark_family", {})
    if not isinstance(family_gates, dict) or not isinstance(family_metrics, dict):
        return {}
    reasons: dict[str, str] = {}
    for family in sorted(family_gates):
        family_gate = family_gates.get(family, {})
        metrics = family_metrics.get(family, {})
        if not isinstance(family_gate, dict) or not isinstance(metrics, dict):
            continue
        if int(metrics.get("baseline_task_count", 0) or 0) + int(metrics.get("candidate_task_count", 0) or 0) <= 0:
            continue
        if bool(family_gate.get("allow_primary_routing_signal", False)) and int(
            metrics.get("candidate_primary_episodes", 0) or 0
        ) < int(family_gate.get("min_primary_episodes", 0) or 0):
            reasons[family] = f"{subject} never entered retained Tolbert primary routing on {family} tasks"
            continue
        if bool(family_gate.get("require_novel_command_signal", False)) and int(
            metrics.get("candidate_proposal_selected_steps", 0) or 0
        ) <= 0:
            if bool(family_gate.get("allow_primary_routing_signal", False)) and int(
                metrics.get("candidate_primary_episodes", 0) or 0
            ) >= int(family_gate.get("min_primary_episodes", 0) or 0):
                continue
            reasons[family] = f"{subject} produced no proposal-selected commands on {family} tasks"
            continue
        if int(metrics.get("proposal_selected_steps_delta", 0) or 0) < int(
            family_gate.get("min_proposal_selected_steps_delta", 0) or 0
        ):
            reasons[family] = f"{subject} regressed proposal-selected command usage on {family} tasks"
            continue
        if int(metrics.get("candidate_novel_valid_command_steps", 0) or 0) < int(
            family_gate.get("min_novel_valid_command_steps", 0) or 0
        ):
            reasons[family] = f"{subject} did not produce enough verifier-valid novel commands on {family} tasks"
            continue
        if float(metrics.get("novel_valid_command_rate_delta", 0.0) or 0.0) < float(
            family_gate.get("min_novel_valid_command_rate_delta", 0.0) or 0.0
        ):
            reasons[family] = f"{subject} regressed verifier-valid novel-command rate on {family} tasks"
    return reasons


def _generated_kind_pass_rate(metrics: EvalMetrics, kind: str) -> float:
    total = int(metrics.generated_by_kind.get(kind, 0) or 0)
    if total <= 0:
        return 0.0
    passed = int(metrics.generated_passed_by_kind.get(kind, 0) or 0)
    return float(passed) / float(total)


def _has_generated_kind(metrics: EvalMetrics, kind: str) -> bool:
    return int(metrics.generated_by_kind.get(kind, 0) or 0) > 0


def _generated_family_pass_rate(metrics: EvalMetrics, family: str) -> float:
    total = metrics.generated_by_benchmark_family.get(family, 0)
    if total == 0:
        return 0.0
    return metrics.generated_passed_by_benchmark_family.get(family, 0) / total


def _generated_family_pass_rate_delta_map(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, float]:
    candidate_families = {
        family
        for family in (
            set(baseline_metrics.generated_by_benchmark_family)
            | set(candidate_metrics.generated_by_benchmark_family)
        )
        if baseline_metrics.generated_by_benchmark_family.get(family, 0) > 0
        or candidate_metrics.generated_by_benchmark_family.get(family, 0) > 0
    }
    return {
        family: round(
            _generated_family_pass_rate(candidate_metrics, family)
            - _generated_family_pass_rate(baseline_metrics, family),
            4,
        )
        for family in sorted(candidate_families)
    }


def _generated_family_regression_count(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> int:
    candidate_families = {
        family
        for family in candidate_metrics.generated_by_benchmark_family
        if candidate_metrics.generated_by_benchmark_family.get(family, 0) > 0
    }
    return sum(
        1
        for family in candidate_families
        if _generated_family_pass_rate(candidate_metrics, family)
        < _generated_family_pass_rate(baseline_metrics, family)
    )


def _generated_family_worst_delta(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> float:
    candidate_families = {
        family
        for family in candidate_metrics.generated_by_benchmark_family
        if candidate_metrics.generated_by_benchmark_family.get(family, 0) > 0
    }
    if not candidate_families:
        return 0.0
    return round(
        min(
            _generated_family_pass_rate(candidate_metrics, family)
            - _generated_family_pass_rate(baseline_metrics, family)
            for family in candidate_families
        ),
        4,
    )


def _difficulty_pass_rate_delta_map(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> dict[str, float]:
    difficulties = {
        difficulty
        for difficulty in (
            set(baseline_metrics.total_by_difficulty) | set(candidate_metrics.total_by_difficulty)
        )
        if baseline_metrics.total_by_difficulty.get(difficulty, 0) > 0
        or candidate_metrics.total_by_difficulty.get(difficulty, 0) > 0
    }
    return {
        difficulty: round(
            candidate_metrics.difficulty_pass_rate(difficulty) - baseline_metrics.difficulty_pass_rate(difficulty),
            4,
        )
        for difficulty in sorted(difficulties)
    }


def _proposal_metrics_by_difficulty(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
    payload = getattr(metrics, "proposal_metrics_by_difficulty", {})
    if isinstance(payload, dict) and payload:
        return {
            str(difficulty): dict(values)
            for difficulty, values in payload.items()
            if isinstance(values, dict)
        }
    trajectories = metrics.task_trajectories or {}
    if not isinstance(trajectories, dict):
        return {}
    summary: dict[str, dict[str, object]] = {}
    for payload in trajectories.values():
        if not isinstance(payload, dict):
            continue
        difficulty = str(payload.get("difficulty", "unknown")).strip() or "unknown"
        row = summary.setdefault(
            difficulty,
            {
                "task_count": 0,
                "success_count": 0,
                "proposal_selected_steps": 0,
                "novel_command_steps": 0,
                "novel_valid_command_steps": 0,
                "novel_valid_command_rate": 0.0,
            },
        )
        row["task_count"] = int(row.get("task_count", 0) or 0) + 1
        row["success_count"] = int(row.get("success_count", 0) or 0) + int(bool(payload.get("success", False)))
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("proposal_source", "")).strip():
                row["proposal_selected_steps"] = int(row.get("proposal_selected_steps", 0) or 0) + 1
            if bool(step.get("proposal_novel", False)):
                row["novel_command_steps"] = int(row.get("novel_command_steps", 0) or 0) + 1
                if bool(step.get("verification_passed", False)):
                    row["novel_valid_command_steps"] = int(row.get("novel_valid_command_steps", 0) or 0) + 1
    for row in summary.values():
        novel_command_steps = int(row.get("novel_command_steps", 0) or 0)
        novel_valid_steps = int(row.get("novel_valid_command_steps", 0) or 0)
        row["novel_valid_command_rate"] = (
            0.0 if novel_command_steps <= 0 else round(novel_valid_steps / novel_command_steps, 4)
        )
    return summary


def _world_feedback_by_difficulty(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
    payload = getattr(metrics, "world_feedback_by_difficulty", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(difficulty): dict(values)
        for difficulty, values in payload.items()
        if isinstance(values, dict)
    }


def _world_feedback_by_benchmark_family(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
    payload = getattr(metrics, "world_feedback_by_benchmark_family", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(family): dict(values)
        for family, values in payload.items()
        if isinstance(values, dict)
    }


def _world_feedback_delta(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    baseline_progress_mae = float(baseline.get("progress_calibration_mae", 0.0) or 0.0)
    candidate_progress_mae = float(candidate.get("progress_calibration_mae", 0.0) or 0.0)
    baseline_risk_mae = float(baseline.get("risk_calibration_mae", 0.0) or 0.0)
    candidate_risk_mae = float(candidate.get("risk_calibration_mae", 0.0) or 0.0)
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "progress_calibration_mae_gain": round(baseline_progress_mae - candidate_progress_mae, 4),
        "risk_calibration_mae_gain": round(baseline_risk_mae - candidate_risk_mae, 4),
    }


def _long_horizon_summary(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, object]:
    difficulty = "long_horizon"
    baseline_total = int(baseline_metrics.total_by_difficulty.get(difficulty, 0) or 0)
    candidate_total = int(candidate_metrics.total_by_difficulty.get(difficulty, 0) or 0)
    baseline_rate = baseline_metrics.difficulty_pass_rate(difficulty)
    candidate_rate = candidate_metrics.difficulty_pass_rate(difficulty)
    baseline_proposal = _proposal_metrics_by_difficulty(baseline_metrics).get(difficulty, {})
    candidate_proposal = _proposal_metrics_by_difficulty(candidate_metrics).get(difficulty, {})
    baseline_world = _world_feedback_by_difficulty(baseline_metrics).get(difficulty, {})
    candidate_world = _world_feedback_by_difficulty(candidate_metrics).get(difficulty, {})
    baseline_proposal_steps = int(baseline_proposal.get("proposal_selected_steps", 0) or 0)
    candidate_proposal_steps = int(candidate_proposal.get("proposal_selected_steps", 0) or 0)
    baseline_novel_valid_steps = int(baseline_proposal.get("novel_valid_command_steps", 0) or 0)
    candidate_novel_valid_steps = int(candidate_proposal.get("novel_valid_command_steps", 0) or 0)
    baseline_novel_valid_rate = float(baseline_proposal.get("novel_valid_command_rate", 0.0) or 0.0)
    candidate_novel_valid_rate = float(candidate_proposal.get("novel_valid_command_rate", 0.0) or 0.0)
    world_feedback_delta = _world_feedback_delta(
        dict(baseline_world) if isinstance(baseline_world, dict) else {},
        dict(candidate_world) if isinstance(candidate_world, dict) else {},
    )
    return {
        "difficulty": difficulty,
        "baseline_task_count": baseline_total,
        "candidate_task_count": candidate_total,
        "baseline_pass_rate": round(baseline_rate, 4),
        "candidate_pass_rate": round(candidate_rate, 4),
        "pass_rate_delta": round(candidate_rate - baseline_rate, 4),
        "baseline_proposal_selected_steps": baseline_proposal_steps,
        "candidate_proposal_selected_steps": candidate_proposal_steps,
        "proposal_selected_steps_delta": candidate_proposal_steps - baseline_proposal_steps,
        "baseline_novel_valid_command_steps": baseline_novel_valid_steps,
        "candidate_novel_valid_command_steps": candidate_novel_valid_steps,
        "novel_valid_command_steps_delta": candidate_novel_valid_steps - baseline_novel_valid_steps,
        "baseline_novel_valid_command_rate": round(baseline_novel_valid_rate, 4),
        "candidate_novel_valid_command_rate": round(candidate_novel_valid_rate, 4),
        "novel_valid_command_rate_delta": round(candidate_novel_valid_rate - baseline_novel_valid_rate, 4),
        "baseline_world_feedback_step_count": int(baseline_world.get("step_count", 0) or 0)
        if isinstance(baseline_world, dict)
        else 0,
        "candidate_world_feedback_step_count": int(candidate_world.get("step_count", 0) or 0)
        if isinstance(candidate_world, dict)
        else 0,
        "world_feedback": world_feedback_delta,
    }


def _benchmark_family_summary(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    family: str,
) -> dict[str, object]:
    family = str(family).strip()
    if not family:
        return {}
    baseline_primary_total = int(baseline_metrics.total_by_benchmark_family.get(family, 0) or 0)
    candidate_primary_total = int(candidate_metrics.total_by_benchmark_family.get(family, 0) or 0)
    baseline_generated_total = int(baseline_metrics.generated_by_benchmark_family.get(family, 0) or 0)
    candidate_generated_total = int(candidate_metrics.generated_by_benchmark_family.get(family, 0) or 0)
    baseline_primary_rate = baseline_metrics.benchmark_family_pass_rate(family)
    candidate_primary_rate = candidate_metrics.benchmark_family_pass_rate(family)
    baseline_generated_rate = _generated_family_pass_rate(baseline_metrics, family)
    candidate_generated_rate = _generated_family_pass_rate(candidate_metrics, family)
    baseline_proposal = _proposal_metrics_by_benchmark_family(baseline_metrics).get(family, {})
    candidate_proposal = _proposal_metrics_by_benchmark_family(candidate_metrics).get(family, {})
    baseline_world = _world_feedback_by_benchmark_family(baseline_metrics).get(family, {})
    candidate_world = _world_feedback_by_benchmark_family(candidate_metrics).get(family, {})
    baseline_proposal_steps = int(baseline_proposal.get("proposal_selected_steps", 0) or 0)
    candidate_proposal_steps = int(candidate_proposal.get("proposal_selected_steps", 0) or 0)
    baseline_novel_valid_steps = int(baseline_proposal.get("novel_valid_command_steps", 0) or 0)
    candidate_novel_valid_steps = int(candidate_proposal.get("novel_valid_command_steps", 0) or 0)
    baseline_novel_valid_rate = float(baseline_proposal.get("novel_valid_command_rate", 0.0) or 0.0)
    candidate_novel_valid_rate = float(candidate_proposal.get("novel_valid_command_rate", 0.0) or 0.0)
    if (
        baseline_primary_total + candidate_primary_total + baseline_generated_total + candidate_generated_total <= 0
        and not baseline_proposal
        and not candidate_proposal
        and not baseline_world
        and not candidate_world
    ):
        return {}
    world_feedback_delta = _world_feedback_delta(
        dict(baseline_world) if isinstance(baseline_world, dict) else {},
        dict(candidate_world) if isinstance(candidate_world, dict) else {},
    )
    return {
        "benchmark_family": family,
        "baseline_primary_task_count": baseline_primary_total,
        "candidate_primary_task_count": candidate_primary_total,
        "baseline_primary_pass_rate": round(baseline_primary_rate, 4),
        "candidate_primary_pass_rate": round(candidate_primary_rate, 4),
        "primary_pass_rate_delta": round(candidate_primary_rate - baseline_primary_rate, 4),
        "baseline_generated_task_count": baseline_generated_total,
        "candidate_generated_task_count": candidate_generated_total,
        "baseline_generated_pass_rate": round(baseline_generated_rate, 4),
        "candidate_generated_pass_rate": round(candidate_generated_rate, 4),
        "generated_pass_rate_delta": round(candidate_generated_rate - baseline_generated_rate, 4),
        "baseline_proposal_selected_steps": baseline_proposal_steps,
        "candidate_proposal_selected_steps": candidate_proposal_steps,
        "proposal_selected_steps_delta": candidate_proposal_steps - baseline_proposal_steps,
        "baseline_novel_valid_command_steps": baseline_novel_valid_steps,
        "candidate_novel_valid_command_steps": candidate_novel_valid_steps,
        "novel_valid_command_steps_delta": candidate_novel_valid_steps - baseline_novel_valid_steps,
        "baseline_novel_valid_command_rate": round(baseline_novel_valid_rate, 4),
        "candidate_novel_valid_command_rate": round(candidate_novel_valid_rate, 4),
        "novel_valid_command_rate_delta": round(candidate_novel_valid_rate - baseline_novel_valid_rate, 4),
        "baseline_world_feedback_step_count": int(baseline_world.get("step_count", 0) or 0)
        if isinstance(baseline_world, dict)
        else 0,
        "candidate_world_feedback_step_count": int(candidate_world.get("step_count", 0) or 0)
        if isinstance(candidate_world, dict)
        else 0,
        "world_feedback": world_feedback_delta,
    }


def _proposal_metrics_by_benchmark_family(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
    payload = getattr(metrics, "proposal_metrics_by_benchmark_family", {})
    if isinstance(payload, dict) and payload:
        return {
            str(family): dict(values)
            for family, values in payload.items()
            if isinstance(values, dict)
        }
    trajectories = metrics.task_trajectories or {}
    if not isinstance(trajectories, dict):
        return {}
    summary: dict[str, dict[str, object]] = {}
    for payload in trajectories.values():
        if not isinstance(payload, dict):
            continue
        family = str(payload.get("benchmark_family", "bounded")).strip() or "bounded"
        row = summary.setdefault(
            family,
            {
                "task_count": 0,
                "success_count": 0,
                "proposal_selected_steps": 0,
                "novel_command_steps": 0,
                "novel_valid_command_steps": 0,
                "novel_valid_command_rate": 0.0,
            },
        )
        row["task_count"] = int(row.get("task_count", 0) or 0) + 1
        row["success_count"] = int(row.get("success_count", 0) or 0) + int(bool(payload.get("success", False)))
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("proposal_source", "")).strip():
                row["proposal_selected_steps"] = int(row.get("proposal_selected_steps", 0) or 0) + 1
            if bool(step.get("proposal_novel", False)):
                row["novel_command_steps"] = int(row.get("novel_command_steps", 0) or 0) + 1
                if bool(step.get("verification_passed", False)):
                    row["novel_valid_command_steps"] = int(row.get("novel_valid_command_steps", 0) or 0) + 1
    for row in summary.values():
        novel_command_steps = int(row.get("novel_command_steps", 0) or 0)
        novel_valid_steps = int(row.get("novel_valid_command_steps", 0) or 0)
        row["novel_valid_command_rate"] = (
            0.0 if novel_command_steps <= 0 else round(novel_valid_steps / novel_command_steps, 4)
        )
    return summary


def _proposal_metrics_delta_by_benchmark_family(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, dict[str, object]]:
    baseline_summary = _proposal_metrics_by_benchmark_family(baseline_metrics)
    candidate_summary = _proposal_metrics_by_benchmark_family(candidate_metrics)
    families = sorted(set(baseline_summary) | set(candidate_summary))
    if not families:
        return {}
    delta_summary: dict[str, dict[str, object]] = {}
    for family in families:
        baseline = baseline_summary.get(family, {})
        candidate = candidate_summary.get(family, {})
        baseline_task_count = int(baseline.get("task_count", 0) or 0)
        candidate_task_count = int(candidate.get("task_count", 0) or 0)
        if baseline_task_count + candidate_task_count <= 0:
            continue
        baseline_proposal_steps = int(baseline.get("proposal_selected_steps", 0) or 0)
        candidate_proposal_steps = int(candidate.get("proposal_selected_steps", 0) or 0)
        baseline_novel_steps = int(baseline.get("novel_command_steps", 0) or 0)
        candidate_novel_steps = int(candidate.get("novel_command_steps", 0) or 0)
        baseline_valid_steps = int(baseline.get("novel_valid_command_steps", 0) or 0)
        candidate_valid_steps = int(candidate.get("novel_valid_command_steps", 0) or 0)
        baseline_valid_rate = float(baseline.get("novel_valid_command_rate", 0.0) or 0.0)
        candidate_valid_rate = float(candidate.get("novel_valid_command_rate", 0.0) or 0.0)
        delta_summary[family] = {
            "baseline_task_count": baseline_task_count,
            "candidate_task_count": candidate_task_count,
            "baseline_proposal_selected_steps": baseline_proposal_steps,
            "candidate_proposal_selected_steps": candidate_proposal_steps,
            "proposal_selected_steps_delta": candidate_proposal_steps - baseline_proposal_steps,
            "baseline_novel_command_steps": baseline_novel_steps,
            "candidate_novel_command_steps": candidate_novel_steps,
            "novel_command_steps_delta": candidate_novel_steps - baseline_novel_steps,
            "baseline_novel_valid_command_steps": baseline_valid_steps,
            "candidate_novel_valid_command_steps": candidate_valid_steps,
            "novel_valid_command_steps_delta": candidate_valid_steps - baseline_valid_steps,
            "baseline_novel_valid_command_rate": round(baseline_valid_rate, 4),
            "candidate_novel_valid_command_rate": round(candidate_valid_rate, 4),
            "novel_valid_command_rate_delta": round(candidate_valid_rate - baseline_valid_rate, 4),
        }
    return delta_summary


def _proposal_metrics_delta_by_difficulty(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, dict[str, object]]:
    baseline_summary = _proposal_metrics_by_difficulty(baseline_metrics)
    candidate_summary = _proposal_metrics_by_difficulty(candidate_metrics)
    difficulties = sorted(set(baseline_summary) | set(candidate_summary))
    if not difficulties:
        return {}
    delta_summary: dict[str, dict[str, object]] = {}
    for difficulty in difficulties:
        baseline = baseline_summary.get(difficulty, {})
        candidate = candidate_summary.get(difficulty, {})
        baseline_task_count = int(baseline.get("task_count", 0) or 0)
        candidate_task_count = int(candidate.get("task_count", 0) or 0)
        if baseline_task_count + candidate_task_count <= 0:
            continue
        baseline_proposal_steps = int(baseline.get("proposal_selected_steps", 0) or 0)
        candidate_proposal_steps = int(candidate.get("proposal_selected_steps", 0) or 0)
        baseline_novel_steps = int(baseline.get("novel_command_steps", 0) or 0)
        candidate_novel_steps = int(candidate.get("novel_command_steps", 0) or 0)
        baseline_valid_steps = int(baseline.get("novel_valid_command_steps", 0) or 0)
        candidate_valid_steps = int(candidate.get("novel_valid_command_steps", 0) or 0)
        baseline_valid_rate = float(baseline.get("novel_valid_command_rate", 0.0) or 0.0)
        candidate_valid_rate = float(candidate.get("novel_valid_command_rate", 0.0) or 0.0)
        delta_summary[difficulty] = {
            "baseline_task_count": baseline_task_count,
            "candidate_task_count": candidate_task_count,
            "baseline_proposal_selected_steps": baseline_proposal_steps,
            "candidate_proposal_selected_steps": candidate_proposal_steps,
            "proposal_selected_steps_delta": candidate_proposal_steps - baseline_proposal_steps,
            "baseline_novel_command_steps": baseline_novel_steps,
            "candidate_novel_command_steps": candidate_novel_steps,
            "novel_command_steps_delta": candidate_novel_steps - baseline_novel_steps,
            "baseline_novel_valid_command_steps": baseline_valid_steps,
            "candidate_novel_valid_command_steps": candidate_valid_steps,
            "novel_valid_command_steps_delta": candidate_valid_steps - baseline_valid_steps,
            "baseline_novel_valid_command_rate": round(baseline_valid_rate, 4),
            "candidate_novel_valid_command_rate": round(candidate_valid_rate, 4),
            "novel_valid_command_rate_delta": round(candidate_valid_rate - baseline_valid_rate, 4),
        }
    return delta_summary


def _world_feedback_delta_by_difficulty(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, dict[str, object]]:
    baseline_summary = _world_feedback_by_difficulty(baseline_metrics)
    candidate_summary = _world_feedback_by_difficulty(candidate_metrics)
    difficulties = sorted(set(baseline_summary) | set(candidate_summary))
    if not difficulties:
        return {}
    return {
        difficulty: _world_feedback_delta(
            dict(baseline_summary.get(difficulty, {})),
            dict(candidate_summary.get(difficulty, {})),
        )
        for difficulty in difficulties
    }


def _verifier_discrimination_gain(payload: dict[str, object] | None) -> float:
    if not isinstance(payload, dict):
        return 0.0
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list) or not proposals:
        return 0.0
    scores: list[float] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        evidence = proposal.get("evidence", {})
        if isinstance(evidence, dict) and (
            "discrimination_gain_estimate" in evidence or "proposal_discrimination_estimate" in evidence
        ):
            try:
                scores.append(
                    float(
                        evidence.get(
                            "proposal_discrimination_estimate",
                            evidence.get("discrimination_gain_estimate", 0.0),
                        )
                    )
                )
            except (TypeError, ValueError):
                continue
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
def evaluate_artifact_retention(*args, **kwargs):
    from .improvement import evaluate_artifact_retention as _impl

    return _impl(*args, **kwargs)


def retention_evidence(*args, **kwargs):
    from .improvement import retention_evidence as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "_generated_kind_pass_rate",
    "_has_generated_kind",
    "_generated_family_pass_rate",
    "_generated_family_pass_rate_delta_map",
    "_generated_family_regression_count",
    "_generated_family_worst_delta",
    "_difficulty_pass_rate_delta_map",
    "_proposal_metrics_by_difficulty",
    "_world_feedback_by_difficulty",
    "_world_feedback_by_benchmark_family",
    "_world_feedback_delta",
    "_long_horizon_summary",
    "_benchmark_family_summary",
    "_proposal_metrics_by_benchmark_family",
    "_proposal_metrics_delta_by_benchmark_family",
    "_proposal_metrics_delta_by_difficulty",
    "_world_feedback_delta_by_difficulty",
    "_verifier_discrimination_gain",
    "evaluate_artifact_retention",
    "proposal_gate_failure_reason",
    "proposal_gate_failure_reasons_by_benchmark_family",
    "retention_evidence",
]
