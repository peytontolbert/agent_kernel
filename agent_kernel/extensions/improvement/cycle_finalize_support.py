from __future__ import annotations

from pathlib import Path
from typing import Callable

from evals.harness import compare_abstraction_transfer_modes

from ...config import KernelConfig
from ...improvement import ImprovementCycleRecord, ImprovementPlanner
from ...extensions.strategy.subsystems import base_subsystem_for


def run_confirmation_phase(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    subsystem: str,
    cycle_id: str,
    artifact_path: Path,
    managed_active_artifact_path: Path,
    artifact_payload: dict[str, object] | None,
    gate: dict[str, object],
    baseline_flags: dict[str, object],
    candidate_flags: dict[str, object],
    compatibility: dict[str, object],
    protocol_metrics: dict[str, object],
    strategy_candidate: dict[str, object] | None,
    tolbert_runtime_summary: dict[str, object],
    state: str,
    reason: str,
    baseline,
    candidate,
    progress: Callable[[str], None] | None,
    retention_eval_config_fn: Callable[..., KernelConfig],
    evaluate_subsystem_metrics_with_tolbert_startup_retry_fn: Callable[..., object],
    retention_evidence_fn: Callable[..., dict[str, object]],
    evaluate_artifact_retention_fn: Callable[..., tuple[str, str]],
    cycle_id_safe_fn: Callable[[str], str],
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    confirmation_baseline_runs = [baseline]
    confirmation_candidate_runs = [candidate]
    required_confirmation_runs = max(1, int(gate.get("required_confirmation_runs", 1)))
    if state != "retain" or required_confirmation_runs <= 1:
        return {
            "state": state,
            "reason": reason,
            "confirmation_baseline_runs": confirmation_baseline_runs,
            "confirmation_candidate_runs": confirmation_candidate_runs,
        }

    baseline_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=managed_active_artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_confirmation_baseline",
    )
    candidate_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_confirmation_candidate",
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
            evaluate_subsystem_metrics_with_tolbert_startup_retry_fn(
                config=baseline_config,
                subsystem=subsystem,
                flags=baseline_flags,
                progress_label=f"{cycle_id}_{subsystem}_confirmation_baseline",
                phase_name="confirmation_baseline",
                progress=_emit,
                tolbert_runtime_summary=tolbert_runtime_summary,
            ),
            evaluate_subsystem_metrics_with_tolbert_startup_retry_fn(
                config=candidate_config,
                subsystem=subsystem,
                flags=candidate_flags,
                progress_label=f"{cycle_id}_{subsystem}_confirmation_candidate",
                phase_name="confirmation_candidate",
                progress=_emit,
                tolbert_runtime_summary=tolbert_runtime_summary,
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
        confirmation_evidence = retention_evidence_fn(
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
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                strategy_origin=str(
                    dict(strategy_candidate or {}).get(
                        "origin",
                        dict(strategy_candidate or {}).get("strategy_origin", ""),
                    )
                ).strip(),
                candidate_artifact_path=str(artifact_path),
                active_artifact_path=str(managed_active_artifact_path),
                compatibility=compatibility,
            ),
        )
        confirmation_state, confirmation_reason = evaluate_artifact_retention_fn(
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
    return {
        "state": state,
        "reason": reason,
        "confirmation_baseline_runs": confirmation_baseline_runs,
        "confirmation_candidate_runs": confirmation_candidate_runs,
    }


def run_holdout_phase(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    subsystem: str,
    cycle_id: str,
    artifact_path: Path,
    managed_active_artifact_path: Path,
    artifact_payload: dict[str, object] | None,
    gate: dict[str, object],
    baseline_flags: dict[str, object],
    candidate_flags: dict[str, object],
    compatibility: dict[str, object],
    protocol_metrics: dict[str, object],
    strategy_candidate: dict[str, object] | None,
    tolbert_runtime_summary: dict[str, object],
    comparison_task_limit: int | None,
    state: str,
    reason: str,
    baseline,
    candidate,
    evidence: dict[str, object],
    phase_gate_report: dict[str, object],
    prior_retained_comparison: dict[str, object] | None,
    prior_retained_guard_reason: str,
    prior_retained_guard_reason_code: str,
    confirmation_baseline_runs: list[object],
    confirmation_candidate_runs: list[object],
    progress: Callable[[str], None] | None,
    retention_eval_config_fn: Callable[..., KernelConfig],
    holdout_task_limit_for_retention_fn: Callable[..., int | None],
    holdout_generated_schedule_limit_for_retention_fn: Callable[..., int],
    evaluate_subsystem_metrics_with_tolbert_startup_retry_fn: Callable[..., object],
    retention_evidence_fn: Callable[..., dict[str, object]],
    autonomous_phase_gate_report_fn: Callable[..., dict[str, object]],
    evaluate_artifact_retention_fn: Callable[..., tuple[str, str]],
    compare_to_prior_retained_fn: Callable[..., dict[str, object] | None],
    prior_retained_guard_reason_fn: Callable[..., str | None],
    prior_retained_guard_reason_code_fn: Callable[[str], str],
    phase_gate_metrics_summary_fn: Callable[[dict[str, object]], dict[str, object]],
    cycle_id_safe_fn: Callable[[str], str],
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    if state != "retain" or not isinstance(comparison_task_limit, int) or comparison_task_limit <= 0:
        return {
            "state": state,
            "reason": reason,
            "baseline": baseline,
            "candidate": candidate,
            "evidence": evidence,
            "phase_gate_report": phase_gate_report,
            "prior_retained_comparison": prior_retained_comparison,
            "prior_retained_guard_reason": prior_retained_guard_reason,
            "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
            "confirmation_baseline_runs": confirmation_baseline_runs,
            "confirmation_candidate_runs": confirmation_candidate_runs,
        }

    _emit(f"finalize phase=holdout_eval subsystem={subsystem}")
    holdout_baseline_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=managed_active_artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_holdout_baseline",
    )
    holdout_candidate_config = retention_eval_config_fn(
        base_config=config,
        subsystem=subsystem,
        artifact_path=artifact_path,
        scope=f"{cycle_id_safe_fn(cycle_id)}_{subsystem}_holdout_candidate",
    )
    holdout_baseline_flags = {key: value for key, value in baseline_flags.items() if key != "task_limit"}
    holdout_candidate_flags = {key: value for key, value in candidate_flags.items() if key != "task_limit"}
    holdout_task_limit = holdout_task_limit_for_retention_fn(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
    holdout_generated_schedule_limit = holdout_generated_schedule_limit_for_retention_fn(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        capability_modules_path=config.capability_modules_path,
    )
    if isinstance(holdout_task_limit, int) and holdout_task_limit > 0:
        holdout_baseline_flags["task_limit"] = holdout_task_limit
        holdout_candidate_flags["task_limit"] = holdout_task_limit
    if holdout_generated_schedule_limit > 0:
        holdout_baseline_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
        holdout_baseline_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
        holdout_candidate_flags["max_generated_success_schedule_tasks"] = holdout_generated_schedule_limit
        holdout_candidate_flags["max_generated_failure_schedule_tasks"] = holdout_generated_schedule_limit
    holdout_baseline = evaluate_subsystem_metrics_with_tolbert_startup_retry_fn(
        config=holdout_baseline_config,
        subsystem=subsystem,
        flags=holdout_baseline_flags,
        progress_label=f"{cycle_id}_{subsystem}_holdout_baseline",
        phase_name="holdout_baseline",
        progress=_emit,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    holdout_candidate = evaluate_subsystem_metrics_with_tolbert_startup_retry_fn(
        config=holdout_candidate_config,
        subsystem=subsystem,
        flags=holdout_candidate_flags,
        progress_label=f"{cycle_id}_{subsystem}_holdout_candidate",
        phase_name="holdout_candidate",
        progress=_emit,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    holdout_evidence = retention_evidence_fn(
        subsystem,
        holdout_baseline,
        holdout_candidate,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    holdout_phase_gate_report = autonomous_phase_gate_report_fn(
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
                **phase_gate_metrics_summary_fn(holdout_phase_gate_report),
                **holdout_evidence,
            },
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            strategy_origin=str(
                dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
            ).strip(),
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            compatibility=compatibility,
        ),
    )
    holdout_state, holdout_reason = evaluate_artifact_retention_fn(
        subsystem,
        holdout_baseline,
        holdout_candidate,
        artifact_path=artifact_path,
        payload=artifact_payload,
        capability_modules_path=config.capability_modules_path,
    )
    if holdout_state == "retain" and prior_retained_comparison is not None:
        holdout_prior_retained = compare_to_prior_retained_fn(
            config=config,
            planner=planner,
            subsystem=subsystem,
            artifact_path=artifact_path,
            cycles_path=config.improvement_cycles_path,
            before_cycle_id=cycle_id,
            flags=dict(holdout_candidate_flags),
            payload=artifact_payload,
            task_limit=holdout_task_limit,
            progress_label_prefix=f"{cycle_id}_{subsystem}_holdout_prior_retained",
            progress=progress,
        )
        holdout_prior_guard_reason = prior_retained_guard_reason_fn(
            subsystem=subsystem,
            gate=gate,
            comparison=holdout_prior_retained,
            capability_modules_path=config.capability_modules_path,
        )
        if holdout_prior_guard_reason:
            prior_retained_guard_reason = str(holdout_prior_guard_reason).strip()
            prior_retained_guard_reason_code = prior_retained_guard_reason_code_fn(prior_retained_guard_reason)
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
    return {
        "state": holdout_state,
        "reason": holdout_reason,
        "baseline": holdout_baseline,
        "candidate": holdout_candidate,
        "evidence": holdout_evidence,
        "phase_gate_report": holdout_phase_gate_report,
        "prior_retained_comparison": prior_retained_comparison,
        "prior_retained_guard_reason": prior_retained_guard_reason,
        "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
        "confirmation_baseline_runs": confirmation_baseline_runs,
        "confirmation_candidate_runs": confirmation_candidate_runs,
    }


def aggregate_confirmation_confidence(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    subsystem: str,
    cycle_id: str,
    artifact_path: Path,
    managed_active_artifact_path: Path,
    gate: dict[str, object],
    compatibility: dict[str, object],
    protocol_metrics: dict[str, object],
    strategy_candidate: dict[str, object] | None,
    state: str,
    reason: str,
    evidence: dict[str, object],
    confirmation_baseline_runs: list[object],
    confirmation_candidate_runs: list[object],
    progress: Callable[[str], None] | None,
    confirmation_confidence_report_fn: Callable[..., dict[str, object]],
    confirmation_confidence_failures_fn: Callable[..., list[str]],
) -> dict[str, object]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    if state != "retain":
        return {"state": state, "reason": reason, "evidence": evidence}

    _emit(
        f"finalize phase=confidence_aggregate subsystem={subsystem} "
        f"confirmation_runs={len(confirmation_candidate_runs)}"
    )
    confirmation_report = confirmation_confidence_report_fn(
        confirmation_baseline_runs,
        confirmation_candidate_runs,
        gate=gate,
    )
    confirmation_failures = confirmation_confidence_failures_fn(confirmation_report, gate=gate)
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
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            strategy_origin=str(
                dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
            ).strip(),
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            compatibility=compatibility,
        ),
    )
    if confirmation_failures:
        state = "reject"
        reason = confirmation_failures[0]
    return {"state": state, "reason": reason, "evidence": evidence}
