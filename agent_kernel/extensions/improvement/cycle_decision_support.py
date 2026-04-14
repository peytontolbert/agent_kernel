from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...config import KernelConfig
from ...improvement import ImprovementCycleRecord, ImprovementPlanner
from ...extensions.strategy.subsystems import base_subsystem_for


def tooling_evaluate_record_kwargs(
    *,
    config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
    cycle_id: str,
    persist_replay_verified_tool_artifact_fn: Callable[..., dict[str, object]],
) -> dict[str, object]:
    if base_subsystem_for(subsystem, config.capability_modules_path) != "tooling":
        return {}
    replay_verified_update = persist_replay_verified_tool_artifact_fn(
        artifact_path,
        cycle_id=cycle_id,
        runtime_config=config,
    )
    return {
        "artifact_lifecycle_state": str(replay_verified_update["artifact_lifecycle_state"]),
        "artifact_sha256": str(replay_verified_update["artifact_sha256"]),
        "previous_artifact_sha256": str(replay_verified_update["previous_artifact_sha256"]),
        "rollback_artifact_path": str(replay_verified_update["rollback_artifact_path"]),
        "artifact_snapshot_path": str(replay_verified_update["artifact_snapshot_path"]),
    }


def append_initial_compare_record(
    *,
    planner: ImprovementPlanner,
    cycles_path: Path,
    cycle_id: str,
    subsystem: str,
    managed_active_artifact_path: Path,
    artifact_path: Path,
    baseline,
    candidate,
    evidence: dict[str, object],
    phase_gate_report: dict[str, object],
    compatibility: dict[str, object],
    protocol_metrics: dict[str, object],
    strategy_candidate: dict[str, object] | None,
    phase_gate_metrics_summary_fn: Callable[[dict[str, object]], dict[str, object]],
    evaluate_record_kwargs: dict[str, object],
) -> None:
    planner.append_cycle_record(
        cycles_path,
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
                **phase_gate_metrics_summary_fn(phase_gate_report),
                **evidence,
            },
            strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
            strategy_origin=str(
                dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
            ).strip(),
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(managed_active_artifact_path),
            compatibility=compatibility,
            **evaluate_record_kwargs,
        ),
    )


def append_prior_retained_record(
    *,
    planner: ImprovementPlanner,
    cycles_path: Path,
    cycle_id: str,
    subsystem: str,
    managed_active_artifact_path: Path,
    artifact_path: Path,
    prior_retained_comparison: dict[str, object] | None,
    prior_retained_guard_reason: str,
    prior_retained_guard_reason_code: str,
    compatibility: dict[str, object],
    protocol_metrics: dict[str, object],
    strategy_candidate: dict[str, object] | None,
    prior_retained_metrics_summary_fn: Callable[[dict[str, object] | None], dict[str, object]],
) -> None:
    if prior_retained_comparison is None:
        return
    planner.append_cycle_record(
        cycles_path,
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
                **prior_retained_metrics_summary_fn(prior_retained_comparison),
                "prior_retained_guard_reason": prior_retained_guard_reason,
                "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
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


def apply_decision_and_persist(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    subsystem: str,
    cycle_id: str,
    artifact_path: Path,
    managed_active_artifact_path: Path,
    repo_root: Path,
    state: str,
    reason: str,
    baseline,
    candidate,
    evidence: dict[str, object],
    phase_gate_report: dict[str, object],
    prior_retained_comparison: dict[str, object] | None,
    prior_retained_guard_reason: str,
    prior_retained_guard_reason_code: str,
    preview_reason_code: str,
    decision_reason_code: str,
    protocol_match_id: str,
    strategy_candidate: dict[str, object] | None,
    tolbert_runtime_summary: dict[str, object],
    protocol_metrics: dict[str, object],
    progress: Callable[[str], None] | None,
    retention_reason_code: str,
    promotion_block_reason_code_fn: Callable[..., str],
    phase_gate_metrics_summary_fn: Callable[[dict[str, object]], dict[str, object]],
    prior_retained_metrics_summary_fn: Callable[[dict[str, object] | None], dict[str, object]],
    apply_artifact_retention_decision_fn: Callable[..., dict[str, object]],
    write_cycle_report_fn: Callable[..., Path],
) -> tuple[dict[str, object], str, str, str]:
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    evidence["tolbert_runtime_summary"] = dict(tolbert_runtime_summary)
    evidence["tolbert_runtime_outcome"] = str(tolbert_runtime_summary.get("outcome", "")).strip()
    evidence["tolbert_runtime_startup_failure_count"] = int(
        tolbert_runtime_summary.get("startup_failure_count", 0) or 0
    )
    _emit(f"finalize phase=apply_decision subsystem={subsystem} state={state}")
    artifact_update = apply_artifact_retention_decision_fn(
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
        repo_root=repo_root,
    )
    for effect in list(artifact_update.get("lifecycle_effects", []) or []):
        if not isinstance(effect, dict):
            continue
        action = str(effect.get("action", "")).strip()
        if action:
            _emit(f"finalize phase={action} subsystem={subsystem}")
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="record",
                subsystem=subsystem,
                action=action or "post_apply_hook",
                artifact_path=str(effect.get("artifact_path", "")),
                artifact_kind=str(effect.get("artifact_kind", "")),
                reason=str(effect.get("reason", "")).strip(),
                metrics_summary={
                    **protocol_metrics,
                    **(
                        dict(effect.get("metrics_summary", {}))
                        if isinstance(effect.get("metrics_summary", {}), dict)
                        else {}
                    ),
                },
                strategy_candidate_id=str(dict(strategy_candidate or {}).get("strategy_candidate_id", "")).strip(),
                strategy_candidate_kind=str(dict(strategy_candidate or {}).get("strategy_candidate_kind", "")).strip(),
                strategy_origin=str(
                    dict(strategy_candidate or {}).get("origin", dict(strategy_candidate or {}).get("strategy_origin", ""))
                ).strip(),
                active_artifact_path=str(managed_active_artifact_path),
            ),
        )
    strategy_candidate_payload = dict(strategy_candidate or {})
    strategy_candidate_id = str(strategy_candidate_payload.get("strategy_candidate_id", "")).strip()
    strategy_candidate_kind = str(strategy_candidate_payload.get("strategy_candidate_kind", "")).strip()
    strategy_origin = str(
        strategy_candidate_payload.get("origin", strategy_candidate_payload.get("strategy_origin", ""))
    ).strip()
    promotion_block_reason_code = promotion_block_reason_code_fn(
        final_reason=reason,
        prior_retained_guard_reason=prior_retained_guard_reason,
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
                "preview_reason_code": preview_reason_code,
                "decision_reason_code": decision_reason_code,
                "promotion_block_reason_code": promotion_block_reason_code,
                "prior_retained_guard_reason": prior_retained_guard_reason,
                "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
                "strategy_candidate_id": strategy_candidate_id,
                "strategy_candidate_kind": strategy_candidate_kind,
                "strategy_origin": strategy_origin,
                **protocol_metrics,
                **phase_gate_metrics_summary_fn(phase_gate_report),
                **evidence,
                **prior_retained_metrics_summary_fn(prior_retained_comparison),
            },
            strategy_candidate_id=strategy_candidate_id,
            strategy_candidate_kind=strategy_candidate_kind,
            strategy_origin=strategy_origin,
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
                "preview_reason_code": preview_reason_code,
                "decision_reason_code": decision_reason_code,
                "promotion_block_reason_code": promotion_block_reason_code,
                "prior_retained_guard_reason": prior_retained_guard_reason,
                "prior_retained_guard_reason_code": prior_retained_guard_reason_code,
                "strategy_candidate_id": strategy_candidate_id,
                "strategy_candidate_kind": strategy_candidate_kind,
                "strategy_origin": strategy_origin,
                **protocol_metrics,
                **phase_gate_metrics_summary_fn(phase_gate_report),
                **evidence,
                **prior_retained_metrics_summary_fn(prior_retained_comparison),
            },
            strategy_candidate_id=strategy_candidate_id,
            strategy_candidate_kind=strategy_candidate_kind,
            strategy_origin=strategy_origin,
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
    write_cycle_report_fn(
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
        prior_retained_guard_reason=prior_retained_guard_reason,
        prior_retained_guard_reason_code=prior_retained_guard_reason_code,
        preview_reason_code=preview_reason_code,
        decision_reason_code=decision_reason_code,
        protocol_match_id=protocol_match_id,
        strategy_candidate=strategy_candidate,
        tolbert_runtime_summary=tolbert_runtime_summary,
    )
    if state == "reject" and retention_reason_code:
        _emit(
            f"finalize phase=decision_reject_reason subsystem={subsystem} "
            f"reason_code={retention_reason_code} reason={reason}"
        )
    _emit(f"finalize phase=done subsystem={subsystem} state={state}")
    return artifact_update, strategy_candidate_id, strategy_candidate_kind, strategy_origin
