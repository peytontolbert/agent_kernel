from __future__ import annotations

from typing import Any

from evals.metrics import EvalMetrics

def memory_source_focus_summary(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
    if not isinstance(metrics, EvalMetrics):
        return {}
    overall_failure_rate = max(0.0, 1.0 - float(metrics.pass_rate))
    summary: dict[str, dict[str, object]] = {}

    def ensure_row(source: str) -> dict[str, object]:
        row = summary.get(source)
        if row is None:
            row = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "outcome_count": 0,
                "no_state_progress_steps": 0,
                "state_regression_steps": 0,
                "proposal_selected_steps": 0,
                "novel_valid_command_steps": 0,
                "completion_ratio_total": 0.0,
                "failure_signal_counts": {},
            }
            summary[source] = row
        return row

    for source, total in (metrics.total_by_memory_source or {}).items():
        normalized = str(source).strip() or "none"
        row = ensure_row(normalized)
        row["total"] = int(total or 0)
        row["passed"] = int((metrics.passed_by_memory_source or {}).get(normalized, 0) or 0)
        row["failed"] = max(0, int(row["total"]) - int(row["passed"]))

    for payload in (metrics.task_outcomes or {}).values():
        if not isinstance(payload, dict):
            continue
        normalized = str(payload.get("memory_source", "none")).strip() or "none"
        row = ensure_row(normalized)
        row["outcome_count"] = int(row["outcome_count"]) + 1
        if not bool(payload.get("success", False)):
            row["failed"] = int(row["failed"]) + 1
        row["no_state_progress_steps"] = int(row["no_state_progress_steps"]) + int(
            payload.get("no_state_progress_steps", 0) or 0
        )
        row["state_regression_steps"] = int(row["state_regression_steps"]) + int(
            payload.get("state_regression_steps", 0) or 0
        )
        row["proposal_selected_steps"] = int(row["proposal_selected_steps"]) + int(
            payload.get("proposal_selected_steps", 0) or 0
        )
        row["novel_valid_command_steps"] = int(row["novel_valid_command_steps"]) + int(
            payload.get("novel_valid_command_steps", 0) or 0
        )
        row["completion_ratio_total"] = float(row["completion_ratio_total"]) + float(
            payload.get("completion_ratio", 0.0) or 0.0
        )
        signal_counts = row.setdefault("failure_signal_counts", {})
        if isinstance(signal_counts, dict):
            for signal in payload.get("failure_signals", []) or []:
                normalized_signal = str(signal).strip()
                if not normalized_signal:
                    continue
                signal_counts[normalized_signal] = int(signal_counts.get(normalized_signal, 0) or 0) + 1

    for row in summary.values():
        total = max(int(row.get("total", 0) or 0), int(row.get("outcome_count", 0) or 0))
        passed = min(total, int(row.get("passed", 0) or 0))
        failed = min(total, max(int(row.get("failed", 0) or 0), total - passed))
        outcome_count = max(1, int(row.get("outcome_count", 0) or 0), total)
        pass_rate = 0.0 if total <= 0 else passed / total
        failure_rate = 0.0 if total <= 0 else failed / total
        completion_ratio = float(row.get("completion_ratio_total", 0.0) or 0.0) / outcome_count
        transition_pressure = min(
            1.0,
            (
                int(row.get("no_state_progress_steps", 0) or 0)
                + int(row.get("state_regression_steps", 0) or 0)
            )
            / outcome_count,
        )
        signal_counts = row.get("failure_signal_counts", {})
        command_failure_pressure = 0.0
        if isinstance(signal_counts, dict):
            command_failure_pressure = min(
                1.0,
                int(signal_counts.get("command_failure", 0) or 0) / outcome_count,
            )
        row["total"] = total
        row["passed"] = passed
        row["failed"] = failed
        row["pass_rate"] = round(pass_rate, 4)
        row["failure_rate"] = round(failure_rate, 4)
        row["failure_gap"] = round(max(0.0, failure_rate - overall_failure_rate), 4)
        row["completion_ratio"] = round(completion_ratio, 4)
        row["completion_gap"] = round(max(0.0, 1.0 - completion_ratio), 4)
        row["transition_pressure"] = round(transition_pressure, 4)
        row["command_failure_pressure"] = round(command_failure_pressure, 4)
    return summary


def memory_source_experiment_bonus(
    subsystem: str,
    metrics: EvalMetrics,
) -> tuple[float, dict[str, object]]:
    source_summary = memory_source_focus_summary(metrics)
    subsystem_sources = {
        "curriculum": ("episode", "discovered_task", "transition_pressure"),
        "tooling": ("tool",),
        "verifier": ("verifier", "verifier_candidate"),
        "skills": ("skill",),
        "operators": ("skill_transfer", "operator"),
    }
    relevant_sources = subsystem_sources.get(subsystem, ())
    if not relevant_sources:
        return 0.0, {}
    relevant_payloads: dict[str, dict[str, object]] = {}
    bonus = 0.0
    for source in relevant_sources:
        payload = source_summary.get(source)
        if not isinstance(payload, dict) or int(payload.get("total", 0) or 0) <= 0:
            continue
        relevant_payloads[source] = {
            "total": int(payload.get("total", 0) or 0),
            "pass_rate": float(payload.get("pass_rate", 0.0) or 0.0),
            "failure_gap": float(payload.get("failure_gap", 0.0) or 0.0),
            "transition_pressure": float(payload.get("transition_pressure", 0.0) or 0.0),
            "command_failure_pressure": float(payload.get("command_failure_pressure", 0.0) or 0.0),
            "completion_gap": float(payload.get("completion_gap", 0.0) or 0.0),
        }
        if subsystem == "curriculum":
            bonus += (float(payload.get("failure_gap", 0.0) or 0.0) * 0.02) + (
                float(payload.get("transition_pressure", 0.0) or 0.0) * 0.012
            )
        elif subsystem == "tooling":
            bonus += (float(payload.get("failure_gap", 0.0) or 0.0) * 0.04) + (
                float(payload.get("command_failure_pressure", 0.0) or 0.0) * 0.02
            )
        elif subsystem == "verifier":
            bonus += (float(payload.get("failure_gap", 0.0) or 0.0) * 0.035) + (
                float(payload.get("transition_pressure", 0.0) or 0.0) * 0.015
            )
        elif subsystem == "skills":
            bonus += (float(payload.get("failure_gap", 0.0) or 0.0) * 0.03) + (
                float(payload.get("completion_gap", 0.0) or 0.0) * 0.01
            )
        elif subsystem == "operators":
            bonus += (float(payload.get("failure_gap", 0.0) or 0.0) * 0.025) + (
                float(payload.get("completion_gap", 0.0) or 0.0) * 0.01
            )
    if not relevant_payloads:
        return 0.0, {}
    capped_bonus = round(min(0.04, bonus), 4)
    return capped_bonus, {
        "bonus": capped_bonus,
        "relevant_sources": relevant_payloads,
    }


def learning_candidate_summary(planner: Any) -> dict[str, dict[str, object]]:
    from ... import improvement as core

    path = planner.learning_artifacts_path
    if path is None:
        return {}
    summary: dict[str, dict[str, object]] = {}

    def ensure_row(subsystem: str) -> dict[str, object]:
        row = summary.get(subsystem)
        if row is None:
            row = {
                "candidate_count": 0,
                "support_total": 0,
                "artifact_kind_counts": {},
                "artifact_kind_support": {},
                "transition_failure_total": 0,
                "applicable_task_total": 0,
                "memory_sources": {},
            }
            summary[subsystem] = row
        return row

    for candidate in core.load_learning_candidates(path, config=planner.runtime_config):
        artifact_kind = str(candidate.get("artifact_kind", "")).strip()
        if not artifact_kind:
            continue
        try:
            support = max(1, int(candidate.get("support_count", 1) or 1))
        except (TypeError, ValueError):
            support = 1
        transition_failures = [
            str(value).strip()
            for value in candidate.get("transition_failures", [])
            if str(value).strip()
        ]
        applicable_tasks = [
            str(value).strip()
            for value in candidate.get("applicable_tasks", [])
            if str(value).strip()
        ]
        memory_source = str(candidate.get("memory_source", "")).strip()
        gap_kind = str(candidate.get("gap_kind", "")).strip()
        subsystem_targets: set[str] = set()
        if artifact_kind == "negative_command_pattern":
            subsystem_targets.add("transition_model")
        elif artifact_kind == "success_skill_candidate":
            subsystem_targets.add("skills")
        elif artifact_kind == "recovery_case":
            subsystem_targets.update({"transition_model", "curriculum"})
        elif artifact_kind == "failure_case":
            subsystem_targets.update({"verifier", "curriculum"})
        elif artifact_kind == "benchmark_gap":
            subsystem_targets.update({"benchmark", "curriculum"})
            if gap_kind in {"failure_cluster", "recovery_path", "transition_pressure"}:
                subsystem_targets.add("verifier")
            if transition_failures or gap_kind == "transition_pressure":
                subsystem_targets.add("transition_model")
        for target in subsystem_targets:
            row = ensure_row(target)
            row["candidate_count"] = int(row.get("candidate_count", 0) or 0) + 1
            row["support_total"] = int(row.get("support_total", 0) or 0) + support
            row["transition_failure_total"] = int(row.get("transition_failure_total", 0) or 0) + len(
                transition_failures
            )
            row["applicable_task_total"] = int(row.get("applicable_task_total", 0) or 0) + len(applicable_tasks)
            artifact_kind_counts = row.setdefault("artifact_kind_counts", {})
            if isinstance(artifact_kind_counts, dict):
                artifact_kind_counts[artifact_kind] = int(artifact_kind_counts.get(artifact_kind, 0) or 0) + 1
            artifact_kind_support = row.setdefault("artifact_kind_support", {})
            if isinstance(artifact_kind_support, dict):
                artifact_kind_support[artifact_kind] = int(artifact_kind_support.get(artifact_kind, 0) or 0) + support
            if memory_source:
                memory_sources = row.setdefault("memory_sources", {})
                if isinstance(memory_sources, dict):
                    memory_sources[memory_source] = int(memory_sources.get(memory_source, 0) or 0) + 1
    return summary


def learning_candidate_experiment_bonus(
    planner: Any,
    subsystem: str,
    *,
    summary_by_subsystem: dict[str, dict[str, object]] | None = None,
) -> tuple[float, dict[str, object]]:
    effective_subsystem = planner._base_subsystem(subsystem)
    resolved_summary = summary_by_subsystem if summary_by_subsystem is not None else learning_candidate_summary(planner)
    summary = resolved_summary.get(effective_subsystem)
    if not isinstance(summary, dict) or int(summary.get("candidate_count", 0) or 0) <= 0:
        return 0.0, {}
    artifact_kind_counts = (
        dict(summary.get("artifact_kind_counts", {}))
        if isinstance(summary.get("artifact_kind_counts", {}), dict)
        else {}
    )
    artifact_kind_support = (
        dict(summary.get("artifact_kind_support", {}))
        if isinstance(summary.get("artifact_kind_support", {}), dict)
        else {}
    )
    support_total = int(summary.get("support_total", 0) or 0)
    transition_failure_total = int(summary.get("transition_failure_total", 0) or 0)
    applicable_task_total = int(summary.get("applicable_task_total", 0) or 0)
    bonus = 0.0
    cap = 0.0
    if effective_subsystem == "transition_model":
        bonus = (
            int(artifact_kind_support.get("negative_command_pattern", 0) or 0) * 0.012
            + int(artifact_kind_support.get("recovery_case", 0) or 0) * 0.006
            + min(4, transition_failure_total) * 0.004
        )
        cap = 0.04
    elif effective_subsystem == "curriculum":
        bonus = (
            int(artifact_kind_support.get("recovery_case", 0) or 0) * 0.008
            + int(artifact_kind_support.get("failure_case", 0) or 0) * 0.005
            + int(artifact_kind_support.get("benchmark_gap", 0) or 0) * 0.006
        )
        cap = 0.03
    elif effective_subsystem == "verifier":
        bonus = (
            int(artifact_kind_support.get("failure_case", 0) or 0) * 0.01
            + int(artifact_kind_support.get("benchmark_gap", 0) or 0) * 0.005
        )
        cap = 0.03
    elif effective_subsystem == "benchmark":
        bonus = (
            int(artifact_kind_support.get("benchmark_gap", 0) or 0) * 0.01
            + min(6, applicable_task_total) * 0.001
        )
        cap = 0.03
    elif effective_subsystem == "skills":
        bonus = (
            int(artifact_kind_support.get("success_skill_candidate", 0) or 0) * 0.012
            + min(4, applicable_task_total) * 0.001
        )
        cap = 0.025
    if cap <= 0.0 or bonus <= 0.0:
        return 0.0, {}
    capped_bonus = round(min(cap, bonus), 4)
    return capped_bonus, {
        "bonus": capped_bonus,
        "candidate_count": int(summary.get("candidate_count", 0) or 0),
        "support_total": support_total,
        "artifact_kind_counts": artifact_kind_counts,
        "artifact_kind_support": artifact_kind_support,
        "transition_failure_total": transition_failure_total,
        "memory_sources": dict(summary.get("memory_sources", {}))
        if isinstance(summary.get("memory_sources", {}), dict)
        else {},
    }
