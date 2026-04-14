from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from evals.metrics import EvalMetrics


@dataclass(slots=True)
class ImprovementTarget:
    subsystem: str
    reason: str
    priority: int


@dataclass(slots=True)
class ImprovementExperiment:
    subsystem: str
    reason: str
    priority: int
    expected_gain: float
    estimated_cost: int
    score: float
    evidence: dict[str, object]


@dataclass(slots=True)
class ImprovementVariant:
    subsystem: str
    variant_id: str
    description: str
    expected_gain: float
    estimated_cost: int
    score: float
    controls: dict[str, object]


@dataclass(slots=True)
class ImprovementYieldSummary:
    retained_cycles: int
    rejected_cycles: int
    total_decisions: int
    retained_by_subsystem: dict[str, int]
    rejected_by_subsystem: dict[str, int]
    average_retained_pass_rate_delta: float
    average_retained_step_delta: float
    average_rejected_pass_rate_delta: float
    average_rejected_step_delta: float


@dataclass(slots=True)
class ImprovementSearchBudget:
    scope: str
    width: int
    max_width: int
    strategy: str
    top_score: float
    selected_ids: list[str]
    reasons: list[str]


@dataclass(slots=True)
class ImprovementCycleRecord:
    cycle_id: str
    state: str
    subsystem: str
    action: str
    artifact_path: str
    artifact_kind: str
    reason: str
    metrics_summary: dict[str, object]
    candidate_artifact_path: str = ""
    active_artifact_path: str = ""
    artifact_lifecycle_state: str = ""
    artifact_sha256: str = ""
    previous_artifact_sha256: str = ""
    rollback_artifact_path: str = ""
    artifact_snapshot_path: str = ""
    selected_variant_id: str = ""
    strategy_candidate_id: str = ""
    strategy_candidate_kind: str = ""
    strategy_origin: str = ""
    prior_retained_cycle_id: str = ""
    baseline_pass_rate: float | None = None
    candidate_pass_rate: float | None = None
    baseline_average_steps: float | None = None
    candidate_average_steps: float | None = None
    phase_gate_passed: bool | None = None
    compatibility: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "spec_version": "asi_v1",
            "cycle_id": self.cycle_id,
            "state": self.state,
            "subsystem": self.subsystem,
            "action": self.action,
            "artifact_path": self.artifact_path,
            "artifact_kind": self.artifact_kind,
            "reason": self.reason,
            "metrics_summary": self.metrics_summary,
            "candidate_artifact_path": self.candidate_artifact_path,
            "active_artifact_path": self.active_artifact_path,
            "artifact_lifecycle_state": self.artifact_lifecycle_state,
            "artifact_sha256": self.artifact_sha256,
            "previous_artifact_sha256": self.previous_artifact_sha256,
            "rollback_artifact_path": self.rollback_artifact_path,
            "artifact_snapshot_path": self.artifact_snapshot_path,
            "selected_variant_id": self.selected_variant_id,
            "strategy_candidate_id": self.strategy_candidate_id,
            "strategy_candidate_kind": self.strategy_candidate_kind,
            "strategy_origin": self.strategy_origin,
            "prior_retained_cycle_id": self.prior_retained_cycle_id,
            "compatibility": self.compatibility or {},
        }
        if self.baseline_pass_rate is not None:
            payload["baseline_pass_rate"] = float(self.baseline_pass_rate)
        if self.candidate_pass_rate is not None:
            payload["candidate_pass_rate"] = float(self.candidate_pass_rate)
        if self.baseline_average_steps is not None:
            payload["baseline_average_steps"] = float(self.baseline_average_steps)
        if self.candidate_average_steps is not None:
            payload["candidate_average_steps"] = float(self.candidate_average_steps)
        if self.phase_gate_passed is not None:
            payload["phase_gate_passed"] = bool(self.phase_gate_passed)
        return payload


@dataclass(slots=True)
class RetentionDecisionContext:
    subsystem: str
    gate: dict[str, object]
    evidence: dict[str, object]
    artifact_update: dict[str, object]
    baseline_metrics: EvalMetrics
    candidate_metrics: EvalMetrics
    pass_rate_delta: float
    average_step_delta: float
    generated_pass_rate_delta: float
    regressed_family_count: int
    generated_regressed_family_count: int
    failure_recovery_delta: float


@dataclass(slots=True)
class RetentionApplyContext:
    subsystem: str
    cycle_id: str
    decision_state: str
    decision_reason: str
    candidate_artifact_path: Path
    active_artifact_path: Path
    artifact_payload: dict[str, object] | None
    baseline_metrics: EvalMetrics
    candidate_metrics: EvalMetrics
    runtime_config: Any | None = None
    repo_root: Path | None = None
    capability_modules_path: Path | None = None


def empty_strategy_memory_summary() -> dict[str, object]:
    return {
        "selected_parent_strategy_node_ids": [],
        "best_retained_strategy_node_id": "",
        "continuation_parent_node_id": "",
        "continuation_artifact_path": "",
        "continuation_workspace_ref": "",
        "continuation_branch": "",
        "parent_control_surface": {},
        "score_delta": 0.0,
        "avoid_reselection": False,
        "recent_rejects": 0,
        "recent_retains": 0,
    }


class LearningEvidenceAdapter:
    def summarize(
        self,
        *,
        subsystem: str,
        learning_artifacts_path: Path | None,
        load_learning_candidates_fn: Callable[..., list[dict[str, object]]],
        runtime_config: Any | None = None,
        capability_modules_path: Path | None = None,
        base_subsystem_fn: Callable[[str, Path | None], str] | None = None,
    ) -> dict[str, object]:
        if learning_artifacts_path is None:
            return {}
        try:
            candidates = load_learning_candidates_fn(learning_artifacts_path, config=runtime_config)
        except TypeError:
            candidates = load_learning_candidates_fn(learning_artifacts_path)
        effective_subsystem = str(subsystem).strip()
        if base_subsystem_fn is not None:
            effective_subsystem = base_subsystem_fn(effective_subsystem, capability_modules_path)
        summary = {
            "candidate_count": 0,
            "support_total": 0,
            "artifact_kind_counts": {},
            "artifact_kind_support": {},
            "transition_failure_total": 0,
            "applicable_task_total": 0,
            "actionable_candidate_count": 0,
            "memory_sources": {},
        }
        for candidate in candidates:
            artifact_kind = str(candidate.get("artifact_kind", "")).strip()
            if not artifact_kind:
                continue
            subsystem_targets = self._candidate_subsystem_targets(
                artifact_kind=artifact_kind,
                candidate=candidate,
            )
            if base_subsystem_fn is not None:
                subsystem_targets = {
                    base_subsystem_fn(target, capability_modules_path)
                    for target in subsystem_targets
                }
            if effective_subsystem not in subsystem_targets:
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
            summary["candidate_count"] = int(summary.get("candidate_count", 0) or 0) + 1
            summary["support_total"] = int(summary.get("support_total", 0) or 0) + support
            summary["transition_failure_total"] = int(summary.get("transition_failure_total", 0) or 0) + len(
                transition_failures
            )
            summary["applicable_task_total"] = int(summary.get("applicable_task_total", 0) or 0) + len(
                applicable_tasks
            )
            if applicable_tasks and memory_source:
                summary["actionable_candidate_count"] = int(
                    summary.get("actionable_candidate_count", 0) or 0
                ) + 1
            artifact_kind_counts = summary.setdefault("artifact_kind_counts", {})
            if isinstance(artifact_kind_counts, dict):
                artifact_kind_counts[artifact_kind] = int(artifact_kind_counts.get(artifact_kind, 0) or 0) + 1
            artifact_kind_support = summary.setdefault("artifact_kind_support", {})
            if isinstance(artifact_kind_support, dict):
                artifact_kind_support[artifact_kind] = int(artifact_kind_support.get(artifact_kind, 0) or 0) + support
            if memory_source:
                memory_sources = summary.setdefault("memory_sources", {})
                if isinstance(memory_sources, dict):
                    memory_sources[memory_source] = int(memory_sources.get(memory_source, 0) or 0) + 1
        if int(summary.get("candidate_count", 0) or 0) <= 0:
            return {}
        summary["has_learning_support"] = True
        summary["has_actionable_support"] = has_actionable_learning_support(summary)
        return summary

    @staticmethod
    def _candidate_subsystem_targets(
        *,
        artifact_kind: str,
        candidate: dict[str, object],
    ) -> set[str]:
        transition_failures = [
            str(value).strip()
            for value in candidate.get("transition_failures", [])
            if str(value).strip()
        ]
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
        return subsystem_targets


DEFAULT_LEARNING_EVIDENCE_ADAPTER = LearningEvidenceAdapter()


def has_actionable_learning_support(summary: dict[str, object] | None) -> bool:
    payload = summary if isinstance(summary, dict) else {}
    support_total = int(payload.get("support_total", 0) or 0)
    applicable_task_total = int(payload.get("applicable_task_total", 0) or 0)
    memory_sources = payload.get("memory_sources", {})
    if not isinstance(memory_sources, dict):
        memory_sources = {}
    memory_source_total = sum(
        max(0, int(value or 0))
        for key, value in memory_sources.items()
        if str(key).strip()
    )
    return support_total > 0 and applicable_task_total > 0 and memory_source_total > 0


def _normalized_nonzero_int_mapping(values: object) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in values.items():
        token = str(key).strip()
        if not token:
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count != 0:
            normalized[token] = count
    return normalized


def _metrics_task_surface_changed(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> bool:
    if int(candidate_metrics.total or 0) != int(baseline_metrics.total or 0):
        return True
    if int(candidate_metrics.generated_total or 0) != int(baseline_metrics.generated_total or 0):
        return True
    comparable_mappings = (
        "total_by_benchmark_family",
        "total_by_origin_benchmark_family",
        "total_by_difficulty",
        "generated_by_kind",
        "generated_by_benchmark_family",
    )
    for field in comparable_mappings:
        if _normalized_nonzero_int_mapping(getattr(candidate_metrics, field, {})) != _normalized_nonzero_int_mapping(
            getattr(baseline_metrics, field, {})
        ):
            return True
    return False


def has_measurable_runtime_influence(context: RetentionDecisionContext) -> bool:
    if context.pass_rate_delta > 0.0:
        return True
    if context.average_step_delta < 0.0:
        return True
    if context.generated_pass_rate_delta > 0.0:
        return True
    positive_float_fields = (
        "family_discrimination_gain",
        "discrimination_gain",
        "failure_recovery_pass_rate_delta",
        "first_step_confidence_delta",
        "first_step_success_delta",
        "novel_valid_command_rate_delta",
        "regressive_recovery_rate_delta",
        "trusted_carryover_repair_rate_delta",
    )
    for field in positive_float_fields:
        try:
            value = float(context.evidence.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return True
    positive_int_fields = (
        "proposal_selected_steps_delta",
        "trusted_retrieval_delta",
        "novel_command_steps_delta",
        "tolbert_primary_episodes_delta",
        "trusted_carryover_verified_step_delta",
    )
    for field in positive_int_fields:
        try:
            value = int(context.evidence.get(field, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return True
    negative_improvement_fields = (
        "low_confidence_episode_delta",
        "no_state_progress_termination_delta",
        "state_regression_trace_delta",
    )
    for field in negative_improvement_fields:
        try:
            value = int(context.evidence.get(field, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value < 0:
            return True
    learning_summary = context.evidence.get("learning_evidence", {})
    if isinstance(learning_summary, dict) and has_actionable_learning_support(learning_summary):
        return True
    if _metrics_task_surface_changed(context.baseline_metrics, context.candidate_metrics):
        return True
    return False


def rank_targets(
    *,
    metrics: EvalMetrics,
    rank_experiments_fn: Callable[[EvalMetrics], list[ImprovementExperiment]],
) -> list[ImprovementTarget]:
    return [
        ImprovementTarget(
            subsystem=experiment.subsystem,
            reason=experiment.reason,
            priority=experiment.priority,
        )
        for experiment in rank_experiments_fn(metrics)
    ]


def sort_experiments(experiments: list[ImprovementExperiment]) -> list[ImprovementExperiment]:
    return sorted(experiments, key=lambda candidate: (-candidate.score, -candidate.priority, candidate.subsystem))


def sort_variants(variants: list[ImprovementVariant]) -> list[ImprovementVariant]:
    return sorted(variants, key=lambda variant: (-variant.score, variant.variant_id))


def append_cycle_record(
    *,
    output_path: Path,
    record: ImprovementCycleRecord,
    runtime_config: Any | None = None,
    govern_exports: bool = True,
    normalize_record_fn: Callable[[ImprovementCycleRecord], ImprovementCycleRecord] | None = None,
    validate_record_fn: Callable[[ImprovementCycleRecord], None] | None = None,
    govern_export_storage_fn: Callable[..., None] | None = None,
) -> Path:
    normalized = normalize_record_fn(record) if normalize_record_fn is not None else record
    if validate_record_fn is not None:
        validate_record_fn(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalized.to_dict()
    if runtime_config is not None and runtime_config.uses_sqlite_storage():
        runtime_config.sqlite_store().append_cycle_record(
            output_path=output_path,
            payload=payload,
        )
        if runtime_config.storage_write_cycle_exports:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            if govern_exports and govern_export_storage_fn is not None:
                govern_export_storage_fn(runtime_config, preserve_paths=(output_path,))
        return output_path
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    if runtime_config is not None and govern_exports and govern_export_storage_fn is not None:
        govern_export_storage_fn(runtime_config, preserve_paths=(output_path,))
    return output_path


def rank_variants_for_experiment(
    experiment: ImprovementExperiment,
    metrics: EvalMetrics,
    *,
    default_variants_fn: Callable[..., list[dict[str, object]]],
    build_variant_fn: Callable[[str, str, str, float, int, dict[str, object]], ImprovementVariant],
    expand_variants_fn: Callable[[list[ImprovementVariant]], list[ImprovementVariant]],
    capability_modules_path: Path | None = None,
) -> list[ImprovementVariant]:
    subsystem = experiment.subsystem
    variants = [
        build_variant_fn(
            subsystem,
            str(variant["variant_id"]),
            str(variant["description"]),
            float(variant["expected_gain"]),
            int(variant["estimated_cost"]),
            dict(variant["controls"]),
        )
        for variant in default_variants_fn(
            subsystem,
            experiment,
            metrics,
            capability_modules_path=capability_modules_path,
        )
    ]
    return expand_variants_fn(variants)


def apply_artifact_retention_decision(
    *,
    artifact_path: Path,
    subsystem: str,
    cycle_id: str,
    decision_state: str,
    decision_reason: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    active_artifact_path: Path | None = None,
    capability_modules_path: Path | None = None,
    runtime_config: Any = None,
    load_json_payload_fn: Callable[[Path], dict[str, object] | Any],
    artifact_sha256_fn: Callable[[Path], str],
    snapshot_artifact_fn: Callable[[Path], Path],
    assess_artifact_compatibility_fn: Callable[..., dict[str, object]],
    update_tool_candidate_states_fn: Callable[..., None],
    stamp_tolbert_lineage_metadata_fn: Callable[..., None],
    promote_tolbert_payload_to_canonical_checkpoint_fn: Callable[..., None],
    compact_rejected_tolbert_payload_fn: Callable[[dict[str, object]], dict[str, object]],
    atomic_write_json_fn: Callable[..., None],
    cleanup_rejected_tolbert_payload_artifacts_fn: Callable[..., dict[str, object]],
    prior_active_artifact_path_fn: Callable[[dict[str, object] | Any], Path | None],
    atomic_copy_file_fn: Callable[..., None],
    synchronize_retained_universe_artifacts_fn: Callable[..., dict[str, str]],
    post_apply_hook_fn: Callable[[RetentionApplyContext], list[dict[str, object]]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    candidate_artifact_path = artifact_path
    live_artifact_path = active_artifact_path if active_artifact_path is not None else artifact_path
    staged_candidate = candidate_artifact_path != live_artifact_path
    payload = load_json_payload_fn(candidate_artifact_path)
    previous_sha256 = artifact_sha256_fn(live_artifact_path)
    rollback_snapshot_path = snapshot_artifact_fn(
        live_artifact_path,
        cycle_id=cycle_id,
        stage="pre_decision_active",
    )
    artifact_kind = str(payload.get("artifact_kind", "")) if isinstance(payload, dict) else ""
    previous_lifecycle_state = (
        str(payload.get("lifecycle_state", "")).strip() if isinstance(payload, dict) else ""
    )
    compatibility = assess_artifact_compatibility_fn(
        subsystem=subsystem,
        payload=payload,
        capability_modules_path=capability_modules_path,
    )
    lifecycle_state = "retained" if decision_state == "retain" else "rejected"
    synchronized_artifact_paths: dict[str, str] = {}
    tolbert_rejected_gc: dict[str, object] | None = None
    tolbert_rejected_output_dir = ""
    lifecycle_effects: list[dict[str, object]] = []

    if isinstance(payload, dict):
        payload["lifecycle_state"] = lifecycle_state
        payload["compatibility"] = compatibility
        payload["rollback_artifact_path"] = str(rollback_snapshot_path)
        payload["retention_decision"] = {
            "cycle_id": cycle_id,
            "state": decision_state,
            "reason": decision_reason,
            "baseline_pass_rate": baseline_metrics.pass_rate,
            "candidate_pass_rate": candidate_metrics.pass_rate,
            "baseline_average_steps": baseline_metrics.average_steps,
            "candidate_average_steps": candidate_metrics.average_steps,
            "baseline_generated_pass_rate": baseline_metrics.generated_pass_rate,
            "candidate_generated_pass_rate": candidate_metrics.generated_pass_rate,
            "previous_lifecycle_state": previous_lifecycle_state,
            "candidate_artifact_path": str(candidate_artifact_path),
            "active_artifact_path": str(live_artifact_path),
            "staged_candidate": staged_candidate,
        }
        if subsystem == "tooling":
            update_tool_candidate_states_fn(
                payload,
                decision_state=decision_state,
                lifecycle_state=lifecycle_state,
            )
        if subsystem == "tolbert_model":
            stamp_tolbert_lineage_metadata_fn(
                payload,
                decision_state=decision_state,
                cycle_id=cycle_id,
                live_artifact_path=live_artifact_path,
                parent_artifact_sha256=previous_sha256,
            )
            if decision_state == "retain":
                promote_tolbert_payload_to_canonical_checkpoint_fn(
                    payload,
                    live_artifact_path=live_artifact_path,
                    cycle_id=cycle_id,
                )
            else:
                tolbert_rejected_output_dir = str(payload.get("output_dir", "")).strip()
                payload = compact_rejected_tolbert_payload_fn(payload)

    atomic_write_json_fn(candidate_artifact_path, payload, config=runtime_config)
    if subsystem == "tolbert_model" and decision_state != "retain" and isinstance(payload, dict):
        tolbert_rejected_gc = cleanup_rejected_tolbert_payload_artifacts_fn(
            candidate_artifact_path=candidate_artifact_path,
            active_artifact_path=live_artifact_path,
            output_dir=tolbert_rejected_output_dir,
        )
        payload["rejected_storage_gc"] = tolbert_rejected_gc
        atomic_write_json_fn(candidate_artifact_path, payload, config=runtime_config)
    candidate_artifact_snapshot_path = snapshot_artifact_fn(
        candidate_artifact_path,
        cycle_id=cycle_id,
        stage=f"post_{decision_state}_candidate",
    )
    restored_live_artifact = False
    active_rollback_source = prior_active_artifact_path_fn(payload)
    active_artifact_snapshot_path = candidate_artifact_snapshot_path
    if decision_state == "retain":
        if staged_candidate:
            atomic_copy_file_fn(
                candidate_artifact_path,
                live_artifact_path,
                config=runtime_config,
            )
        synchronized_artifact_paths = synchronize_retained_universe_artifacts_fn(
            subsystem=subsystem,
            payload=payload,
            live_artifact_path=live_artifact_path,
            runtime_config=runtime_config,
        )
        active_artifact_snapshot_path = snapshot_artifact_fn(
            live_artifact_path,
            cycle_id=cycle_id,
            stage="post_retain_active",
        )
        if post_apply_hook_fn is not None:
            lifecycle_effects = list(
                post_apply_hook_fn(
                    RetentionApplyContext(
                        subsystem=subsystem,
                        cycle_id=cycle_id,
                        decision_state=decision_state,
                        decision_reason=decision_reason,
                        candidate_artifact_path=candidate_artifact_path,
                        active_artifact_path=live_artifact_path,
                        artifact_payload=payload if isinstance(payload, dict) else None,
                        baseline_metrics=baseline_metrics,
                        candidate_metrics=candidate_metrics,
                        runtime_config=runtime_config,
                        repo_root=repo_root,
                        capability_modules_path=capability_modules_path,
                    )
                )
            )
    elif not staged_candidate and active_rollback_source is not None and active_rollback_source.exists():
        atomic_copy_file_fn(
            active_rollback_source,
            live_artifact_path,
            config=runtime_config,
        )
        restored_live_artifact = True
        active_artifact_snapshot_path = candidate_artifact_snapshot_path
    current_sha256 = artifact_sha256_fn(live_artifact_path)
    return {
        "artifact_kind": artifact_kind,
        "artifact_lifecycle_state": "rejected_restored" if restored_live_artifact else lifecycle_state,
        "artifact_sha256": current_sha256,
        "previous_artifact_sha256": previous_sha256,
        "rollback_artifact_path": str(active_rollback_source or rollback_snapshot_path),
        "artifact_snapshot_path": str(active_artifact_snapshot_path),
        "candidate_artifact_path": str(candidate_artifact_path),
        "active_artifact_path": str(live_artifact_path),
        "candidate_artifact_snapshot_path": str(candidate_artifact_snapshot_path),
        "active_artifact_snapshot_path": str(active_artifact_snapshot_path),
        "compatibility": compatibility,
        "rejected_storage_gc": tolbert_rejected_gc or {},
        "synchronized_artifact_paths": synchronized_artifact_paths if decision_state == "retain" else {},
        "lifecycle_effects": lifecycle_effects,
    }


def retention_gate_for_payload(
    subsystem: str,
    payload: dict[str, object] | None,
    *,
    capability_modules_path: Path | None = None,
    base_subsystem_fn: Callable[[str, Path | None], str],
    retention_gate_fn: Callable[[str, dict[str, object] | None], dict[str, object] | Any],
) -> dict[str, object]:
    return dict(retention_gate_fn(base_subsystem_fn(subsystem, capability_modules_path), payload))


def attach_learning_evidence(
    *,
    evidence: dict[str, object],
    subsystem: str,
    learning_artifacts_path: Path | None,
    load_learning_candidates_fn: Callable[..., list[dict[str, object]]],
    runtime_config: Any | None = None,
    capability_modules_path: Path | None = None,
    base_subsystem_fn: Callable[[str, Path | None], str] | None = None,
    learning_evidence_adapter: LearningEvidenceAdapter | None = None,
) -> dict[str, object]:
    adapter = learning_evidence_adapter or DEFAULT_LEARNING_EVIDENCE_ADAPTER
    enriched = dict(evidence)
    learning_summary = adapter.summarize(
        subsystem=subsystem,
        learning_artifacts_path=learning_artifacts_path,
        load_learning_candidates_fn=load_learning_candidates_fn,
        runtime_config=runtime_config,
        capability_modules_path=capability_modules_path,
        base_subsystem_fn=base_subsystem_fn,
    )
    if not learning_summary:
        return enriched
    enriched["learning_evidence"] = learning_summary
    enriched["learning_candidate_count"] = int(learning_summary.get("candidate_count", 0) or 0)
    enriched["learning_support_total"] = int(learning_summary.get("support_total", 0) or 0)
    enriched["learning_transition_failure_total"] = int(learning_summary.get("transition_failure_total", 0) or 0)
    enriched["learning_applicable_task_total"] = int(learning_summary.get("applicable_task_total", 0) or 0)
    enriched["learning_memory_sources"] = dict(learning_summary.get("memory_sources", {}))
    enriched["learning_has_actionable_support"] = has_actionable_learning_support(learning_summary)
    return enriched


def evaluate_artifact_retention(
    *,
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    artifact_path: Path | None = None,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
    load_payload_fn: Callable[[Path], Any],
    assess_artifact_compatibility_fn: Callable[..., dict[str, object]],
    retention_gate_fn: Callable[[str, dict[str, object] | None], dict[str, object]],
    retention_evidence_fn: Callable[..., dict[str, object]],
    evaluator_fn: Callable[[RetentionDecisionContext], tuple[str, str]] | None,
) -> tuple[str, str]:
    artifact_payload = payload
    if artifact_payload is None and artifact_path is not None and artifact_path.exists():
        loaded = load_payload_fn(artifact_path)
        if isinstance(loaded, dict):
            artifact_payload = loaded
    if isinstance(artifact_payload, dict):
        compatibility = assess_artifact_compatibility_fn(subsystem=subsystem, payload=artifact_payload)
        if not bool(compatibility.get("compatible", False)):
            violations = compatibility.get("violations", [])
            violation = ""
            if isinstance(violations, list) and violations:
                violation = str(violations[0]).strip()
            return ("reject", violation or "artifact compatibility checks failed")
    gate = retention_gate_fn(subsystem, artifact_payload)
    evidence = retention_evidence_fn(
        subsystem,
        baseline_metrics,
        candidate_metrics,
        payload=artifact_payload,
        capability_modules_path=capability_modules_path,
    )
    context = RetentionDecisionContext(
        subsystem=subsystem,
        gate=gate,
        evidence=evidence,
        artifact_update=dict(artifact_payload) if isinstance(artifact_payload, dict) else {},
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        pass_rate_delta=candidate_metrics.pass_rate - baseline_metrics.pass_rate,
        average_step_delta=candidate_metrics.average_steps - baseline_metrics.average_steps,
        generated_pass_rate_delta=candidate_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
        regressed_family_count=int(evidence.get("regressed_family_count", 0)),
        generated_regressed_family_count=int(evidence.get("generated_regressed_family_count", 0)),
        failure_recovery_delta=float(evidence.get("failure_recovery_pass_rate_delta", 0.0) or 0.0),
    )
    if not has_measurable_runtime_influence(context):
        return ("reject", "candidate produced no measurable runtime influence against the retained baseline")
    if evaluator_fn is None:
        return ("reject", "unknown subsystem retention policy")
    return evaluator_fn(context)


__all__ = [
    "DEFAULT_LEARNING_EVIDENCE_ADAPTER",
    "ImprovementCycleRecord",
    "ImprovementExperiment",
    "ImprovementSearchBudget",
    "ImprovementTarget",
    "ImprovementVariant",
    "ImprovementYieldSummary",
    "LearningEvidenceAdapter",
    "RetentionDecisionContext",
    "RetentionApplyContext",
    "append_cycle_record",
    "apply_artifact_retention_decision",
    "attach_learning_evidence",
    "empty_strategy_memory_summary",
    "evaluate_artifact_retention",
    "has_actionable_learning_support",
    "has_measurable_runtime_influence",
    "rank_targets",
    "rank_variants_for_experiment",
    "retention_gate_for_payload",
    "sort_experiments",
    "sort_variants",
]
