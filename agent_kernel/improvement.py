from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.metrics import EvalMetrics

from .extensions.improvement import control_evidence
from . import improvement_evidence as evidence_helpers
from . import improvement_records as records_helpers
from . import improvement_retention as retention_helpers
from .config import KernelConfig
from .extensions.improvement.artifacts import (
    artifact_sha256 as artifacts_artifact_sha256,
    effective_artifact_payload_for_retention as artifacts_effective_artifact_payload_for_retention,
    materialize_replay_verified_tool_payload as artifacts_materialize_replay_verified_tool_payload,
    payload_with_active_artifact_context as artifacts_payload_with_active_artifact_context,
    persist_replay_verified_tool_artifact as artifacts_persist_replay_verified_tool_artifact,
    retention_gate_for_payload as artifacts_retention_gate_for_payload,
    snapshot_artifact_state as artifacts_snapshot_artifact_state,
    staged_candidate_artifact_path as artifacts_staged_candidate_artifact_path,
    stamp_artifact_experiment_variant as artifacts_stamp_artifact_experiment_variant,
    stamp_artifact_generation_context as artifacts_stamp_artifact_generation_context,
)
from .improvement_engine import (
    DEFAULT_LEARNING_EVIDENCE_ADAPTER,
    ImprovementCycleRecord,
    ImprovementExperiment,
    ImprovementSearchBudget,
    ImprovementTarget,
    ImprovementVariant,
    ImprovementYieldSummary,
    RetentionDecisionContext,
    append_cycle_record as engine_append_cycle_record,
    apply_artifact_retention_decision as engine_apply_artifact_retention_decision,
    attach_learning_evidence as engine_attach_learning_evidence,
    empty_strategy_memory_summary,
    evaluate_artifact_retention as engine_evaluate_artifact_retention,
    rank_targets as engine_rank_targets,
    rank_variants_for_experiment,
    retention_gate_for_payload as engine_retention_gate_for_payload,
    sort_experiments as engine_sort_experiments,
    sort_variants as engine_sort_variants,
)
from .extensions.improvement import artifact_compatibility as artifact_compatibility_ext
from .extensions.improvement import artifact_protocol_support as artifact_protocol_support_ext
from .extensions.improvement import artifact_runtime_support as artifact_runtime_support_ext
from .extensions.improvement import artifact_support_evidence as artifact_support_evidence_ext
from .extensions.improvement import experiment_ranking as experiment_ranking_ext
from .extensions.improvement import improvement_plugins as improvement_plugins_ext
from .extensions.improvement import improvement_support_validation as improvement_support_validation_ext
from .extensions.improvement import improvement_catalog as improvement_catalog_ext
from .extensions.improvement import improvement_common as improvement_common_ext
from .extensions.improvement import planner_artifacts as planner_artifacts_ext
from .extensions.improvement import planner_budgets as planner_budgets_ext
from .extensions.improvement import planner_controls as planner_controls_ext
from .extensions.improvement import planner_history as planner_history_ext
from .extensions.improvement import planner_learning as planner_learning_ext
from .extensions.improvement import planner_portfolio as planner_portfolio_ext
from .extensions.improvement import planner_runtime_state as planner_runtime_state_ext
from .extensions.improvement import planner_scoring as planner_scoring_ext
from .extensions.improvement import planner_strategy as planner_strategy_ext
from .extensions.improvement import prompt_improvement as prompt_improvement_ext
from .extensions.improvement import retention_evidence as retention_evidence_ext
from .extensions.improvement import retention_policies as retention_policies_ext
from .extensions.improvement import semantic_compatibility as semantic_compatibility_ext
from .extensions.improvement import transition_model_improvement as transition_model_improvement_ext
from .learning_compiler import load_learning_candidates
from .memory import EpisodeMemory
from .ops.runtime_supervision import atomic_copy_file, atomic_write_json

DEFAULT_IMPROVEMENT_PLUGIN_LAYER = improvement_plugins_ext.DEFAULT_IMPROVEMENT_PLUGIN_LAYER
DEFAULT_STRATEGY_PRIOR_STORE = improvement_plugins_ext.DEFAULT_STRATEGY_PRIOR_STORE
TaskContractCatalog = improvement_support_validation_ext.TaskContractCatalog
build_default_task_contract_catalog = improvement_support_validation_ext.build_default_task_contract_catalog
retained_improvement_planner_controls = prompt_improvement_ext.retained_improvement_planner_controls
normalized_control_mapping = improvement_common_ext.normalized_control_mapping
retained_artifact_payload = improvement_common_ext.retained_artifact_payload


def materialize_tolbert_checkpoint_from_delta(
    *,
    parent_checkpoint_path: Path,
    delta_checkpoint_path: Path,
    output_checkpoint_path: Path,
) -> Path:
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint_path,
        delta_checkpoint_path=delta_checkpoint_path,
        output_checkpoint_path=output_checkpoint_path,
    )


def resolve_tolbert_runtime_checkpoint_path(
    runtime_paths: object,
    *,
    artifact_path: Path | None = None,
) -> str | None:
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.resolve_tolbert_runtime_checkpoint_path(
        runtime_paths,
        artifact_path=artifact_path,
    )

class ImprovementPlanner:
    def __init__(
        self,
        memory_root: Path | None = None,
        cycles_path: Path | None = None,
        prompt_proposals_path: Path | None = None,
        use_prompt_proposals: bool = True,
        capability_modules_path: Path | None = None,
        trust_ledger_path: Path | None = None,
        runtime_config: KernelConfig | None = None,
    ) -> None:
        self.memory = (
            EpisodeMemory(memory_root, config=runtime_config) if memory_root is not None else None
        )
        self.cycles_path = cycles_path if cycles_path is not None else self._default_cycles_path(memory_root)
        self.prompt_proposals_path = (
            prompt_proposals_path if prompt_proposals_path is not None else self._default_prompt_proposals_path(memory_root)
        )
        self.use_prompt_proposals = bool(use_prompt_proposals)
        self.capability_modules_path = capability_modules_path
        self.trust_ledger_path = (
            trust_ledger_path if trust_ledger_path is not None else self._default_trust_ledger_path(memory_root)
        )
        self.learning_artifacts_path = (
            runtime_config.learning_artifacts_path
            if runtime_config is not None
            else self._default_learning_artifacts_path(memory_root)
        )
        self.runtime_config = runtime_config
        self._plugin_layer = DEFAULT_IMPROVEMENT_PLUGIN_LAYER
        self._strategy_prior_store = DEFAULT_STRATEGY_PRIOR_STORE

    def _base_subsystem(self, subsystem: str) -> str:
        normalized = str(subsystem).strip()
        if not normalized:
            return ""
        try:
            return self._plugin_layer.base_subsystem(normalized, self.capability_modules_path)
        except ValueError:
            return normalized

    def _campaign_surface_key(self, subsystem: str) -> str:
        base_subsystem = self._base_subsystem(subsystem)
        if base_subsystem in {"retrieval", "tolbert_model", "qwen_adapter"}:
            return "retrieval_stack"
        return base_subsystem

    def _subsystems_match(self, left: str, right: str) -> bool:
        normalized_left = str(left).strip()
        normalized_right = str(right).strip()
        if not normalized_left or not normalized_right:
            return False
        if normalized_left == normalized_right:
            return True
        universe_group = {"universe", "universe_constitution", "operating_envelope"}
        if normalized_left in universe_group and normalized_right in universe_group:
            return normalized_left == "universe" or normalized_right == "universe"
        return self._base_subsystem(normalized_left) == self._base_subsystem(normalized_right)

    _coding_strength_summary = staticmethod(planner_budgets_ext.coding_strength_summary)

    _broad_coding_observe_diversification_signal = staticmethod(
        planner_budgets_ext.broad_coding_observe_diversification_signal
    )

    _allow_qwen_adapter_support_runtime = staticmethod(
        planner_budgets_ext.allow_qwen_adapter_support_runtime
    )

    def rank_targets(self, metrics: EvalMetrics) -> list[ImprovementTarget]:
        return engine_rank_targets(metrics=metrics, rank_experiments_fn=self.rank_experiments)

    def rank_experiments(self, metrics: EvalMetrics) -> list[ImprovementExperiment]:
        return experiment_ranking_ext.rank_experiments(self, metrics)

    def choose_target(self, metrics: EvalMetrics) -> ImprovementTarget:
        return self.rank_targets(metrics)[0]

    def choose_experiment(self, metrics: EvalMetrics) -> ImprovementExperiment:
        return self.rank_experiments(metrics)[0]

    def select_campaign(
        self,
        metrics: EvalMetrics,
        *,
        max_candidates: int = 2,
        relative_score_floor: float = 0.85,
        absolute_score_margin: float = 0.02,
    ) -> list[ImprovementExperiment]:
        planner_controls = self._improvement_planner_controls()
        relative_score_floor = self._planner_guardrail_float(
            planner_controls,
            scope="campaign",
            field="relative_score_floor",
            legacy_field="campaign_relative_score_floor",
            default=relative_score_floor,
            min_value=0.5,
            max_value=0.99,
        )
        absolute_score_margin = self._planner_guardrail_float(
            planner_controls,
            scope="campaign",
            field="absolute_score_margin",
            legacy_field="campaign_absolute_score_margin",
            default=absolute_score_margin,
            min_value=0.0,
            max_value=0.2,
        )
        ranked = self.rank_experiments(metrics)
        if not ranked:
            return []
        campaign = [ranked[0]]
        selected_surfaces = {self._campaign_surface_key(ranked[0].subsystem)}
        if max_candidates <= 1:
            return campaign
        top_score = ranked[0].score
        for candidate in ranked[1:]:
            if len(campaign) >= max_candidates:
                break
            if self._campaign_surface_key(candidate.subsystem) in selected_surfaces:
                continue
            if top_score <= 0.0:
                break
            relative_score = candidate.score / top_score
            score_margin = top_score - candidate.score
            if relative_score < relative_score_floor:
                continue
            if score_margin > absolute_score_margin and candidate.priority < ranked[0].priority:
                continue
            campaign.append(candidate)
            selected_surfaces.add(self._campaign_surface_key(candidate.subsystem))
        return campaign

    select_portfolio_campaign = planner_portfolio_ext.select_portfolio_campaign

    _strategy_portfolio_choice = planner_portfolio_ext.strategy_portfolio_choice

    _strategy_candidate_options = planner_strategy_ext.strategy_candidate_options

    _strategy_signal_basis = planner_strategy_ext.strategy_signal_basis

    _build_strategy_candidate = planner_strategy_ext.build_strategy_candidate

    _strategy_semantic_hypotheses = staticmethod(planner_strategy_ext.strategy_semantic_hypotheses)

    _authored_strategy_candidate = planner_strategy_ext.authored_strategy_candidate

    _stable_strategy_fingerprint = staticmethod(planner_strategy_ext.stable_strategy_fingerprint)

    _normalized_strategy_candidate = planner_strategy_ext.normalized_strategy_candidate

    _synthesized_strategy_candidate = planner_strategy_ext.synthesized_strategy_candidate

    recent_strategy_activity_summary = planner_controls_ext.recent_strategy_activity_summary

    @staticmethod
    def _strategy_history_score_delta(
        strategy_candidate: dict[str, object] | None,
        *,
        recent_activity: dict[str, object] | None,
    ) -> float:
        strategy = strategy_candidate if isinstance(strategy_candidate, dict) else {}
        if not str(strategy.get("strategy_candidate_id", "")).strip():
            return 0.0
        recent = recent_activity if isinstance(recent_activity, dict) else {}
        retained_cycles = int(recent.get("retained_cycles", 0) or 0)
        rejected_cycles = int(recent.get("rejected_cycles", 0) or 0)
        if retained_cycles > 0:
            return round(min(0.03, 0.01 * retained_cycles), 4)
        if rejected_cycles > 0:
            return round(-min(0.03, 0.01 * rejected_cycles), 4)
        return 0.0

    _strategy_memory_summary = planner_portfolio_ext.strategy_memory_summary

    _apply_strategy_memory_control_surface = (
        planner_portfolio_ext.apply_strategy_memory_control_surface
    )

    _strategy_memory_variant_lineage_adjustment = staticmethod(
        planner_controls_ext.strategy_memory_variant_lineage_adjustment
    )

    def rank_variants(self, experiment: ImprovementExperiment, metrics: EvalMetrics) -> list[ImprovementVariant]:
        planner_controls = self._improvement_planner_controls()
        variants = self._variants_for_experiment(experiment, metrics, planner_controls=planner_controls)
        scored_variants = [self._score_variant(experiment, variant, planner_controls=planner_controls) for variant in variants]
        return engine_sort_variants(scored_variants)

    def choose_variant(self, experiment: ImprovementExperiment, metrics: EvalMetrics) -> ImprovementVariant:
        return self.rank_variants(experiment, metrics)[0]

    recommend_campaign_budget = planner_budgets_ext.recommend_campaign_budget

    recommend_variant_budget = planner_budgets_ext.recommend_variant_budget

    failure_counts = planner_runtime_state_ext.failure_counts

    def _failure_counts(self) -> dict[str, int]:
        return self.failure_counts()

    transition_failure_counts = planner_runtime_state_ext.transition_failure_counts

    transition_summary = planner_runtime_state_ext.transition_summary

    environment_violation_summary = planner_runtime_state_ext.environment_violation_summary

    universe_cycle_feedback_summary = planner_history_ext.universe_cycle_feedback_summary

    capability_surface_summary = planner_runtime_state_ext.capability_surface_summary

    trust_ledger_payload = planner_runtime_state_ext.trust_ledger_payload

    trust_ledger_summary = planner_history_ext.trust_ledger_summary

    delegation_policy_summary = planner_runtime_state_ext.delegation_policy_summary

    operator_policy_summary = planner_runtime_state_ext.operator_policy_summary

    _experiment_score = staticmethod(planner_budgets_ext.experiment_score)

    @staticmethod
    def _default_cycles_path(memory_root: Path | None) -> Path | None:
        if memory_root is None:
            return None
        return memory_root.parent / "improvement" / "cycles.jsonl"

    @staticmethod
    def _default_prompt_proposals_path(memory_root: Path | None) -> Path | None:
        if memory_root is None:
            return None
        return memory_root.parent / "prompts" / "prompt_proposals.json"

    @staticmethod
    def _default_trust_ledger_path(memory_root: Path | None) -> Path | None:
        if memory_root is None:
            return None
        return memory_root.parent / "reports" / "unattended_trust_ledger.json"

    @staticmethod
    def _default_learning_artifacts_path(memory_root: Path | None) -> Path | None:
        if memory_root is None:
            return None
        return memory_root.parent / "learning" / "run_learning_artifacts.json"

    _score_experiment = planner_scoring_ext.score_experiment

    _measurement_guardrail_penalty = planner_scoring_ext.measurement_guardrail_penalty

    _memory_source_focus_summary = staticmethod(planner_learning_ext.memory_source_focus_summary)

    _memory_source_experiment_bonus = staticmethod(
        planner_learning_ext.memory_source_experiment_bonus
    )

    _learning_candidate_summary = planner_learning_ext.learning_candidate_summary

    _learning_candidate_experiment_bonus = planner_learning_ext.learning_candidate_experiment_bonus

    @staticmethod
    def _bootstrap_penalty(
        candidate: ImprovementExperiment,
        history: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
        effective_subsystem: str | None = None,
    ) -> tuple[float, list[str]]:
        return planner_scoring_ext.bootstrap_penalty(
            ImprovementPlanner,
            candidate,
            history,
            planner_controls=planner_controls,
            effective_subsystem=effective_subsystem,
        )

    @staticmethod
    def _cold_start_low_confidence_penalty(
        candidate: ImprovementExperiment,
        history: dict[str, object],
        recent_history: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
        effective_subsystem: str | None = None,
    ) -> tuple[float, list[str]]:
        return planner_scoring_ext.cold_start_low_confidence_penalty(
            ImprovementPlanner,
            candidate,
            history,
            recent_history,
            planner_controls=planner_controls,
            effective_subsystem=effective_subsystem,
        )

    @staticmethod
    def _recent_stalled_selection_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        return planner_scoring_ext.recent_stalled_selection_penalty(
            ImprovementPlanner,
            recent_activity,
            planner_controls=planner_controls,
        )

    @staticmethod
    def _recent_observation_timeout_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        return planner_scoring_ext.recent_observation_timeout_penalty(
            ImprovementPlanner,
            recent_activity,
            planner_controls=planner_controls,
        )

    @staticmethod
    def _recent_promotion_failure_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        return planner_scoring_ext.recent_promotion_failure_penalty(
            ImprovementPlanner,
            recent_activity,
            planner_controls=planner_controls,
        )

    _score_variant = planner_scoring_ext.score_variant

    def _variants_for_experiment(
        self,
        experiment: ImprovementExperiment,
        metrics: EvalMetrics,
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> list[ImprovementVariant]:
        return rank_variants_for_experiment(
            experiment,
            metrics,
            default_variants_fn=self._plugin_layer.default_variants,
            build_variant_fn=self._variant,
            expand_variants_fn=lambda variants: self._with_variant_expansions(
                variants,
                planner_controls=planner_controls or {},
            ),
            capability_modules_path=self.capability_modules_path,
        )

    @staticmethod
    def _variant(
        subsystem: str,
        variant_id: str,
        description: str,
        expected_gain: float,
        estimated_cost: int,
        controls: dict[str, object],
    ) -> ImprovementVariant:
        score = ImprovementPlanner._variant_score_fields(expected_gain, estimated_cost)
        return ImprovementVariant(
            subsystem=subsystem,
            variant_id=variant_id,
            description=description,
            expected_gain=expected_gain,
            estimated_cost=estimated_cost,
            score=score,
            controls=controls,
        )

    @staticmethod
    def _variant_score(variant: ImprovementVariant) -> float:
        return ImprovementPlanner._variant_score_fields(variant.expected_gain, variant.estimated_cost)

    @staticmethod
    def _variant_score_fields(expected_gain: float, estimated_cost: int) -> float:
        return round(expected_gain / max(1, estimated_cost), 4)

    def append_cycle_record(
        self,
        output_path: Path,
        record: ImprovementCycleRecord,
        *,
        govern_exports: bool = True,
    ) -> Path:
        govern_export_storage = None
        if self.runtime_config is not None and govern_exports:
            from .ops.export_governance import govern_improvement_export_storage

            govern_export_storage = govern_improvement_export_storage
        return engine_append_cycle_record(
            output_path=output_path,
            record=record,
            runtime_config=self.runtime_config,
            govern_exports=govern_exports,
            normalize_record_fn=_normalized_cycle_record,
            validate_record_fn=_validate_cycle_record_consistency,
            govern_export_storage_fn=govern_export_storage,
        )

    load_cycle_records = planner_history_ext.load_cycle_records

    def cycle_history(
        self,
        output_path: Path,
        *,
        cycle_id: str | None = None,
        subsystem: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, object]]:
        return planner_history_ext.cycle_history(
            self,
            output_path,
            cycle_id=cycle_id,
            subsystem=subsystem,
            state=state,
        )

    cycle_audit_summary = planner_history_ext.cycle_audit_summary

    incomplete_cycle_summaries = planner_history_ext.incomplete_cycle_summaries

    artifact_history = planner_artifacts_ext.artifact_history

    latest_artifact_record = planner_artifacts_ext.latest_artifact_record

    latest_artifact_decision = planner_artifacts_ext.latest_artifact_decision

    artifact_rollback_metadata = planner_artifacts_ext.artifact_rollback_metadata

    prior_retained_artifact_record = planner_artifacts_ext.prior_retained_artifact_record

    rollback_artifact = planner_artifacts_ext.rollback_artifact

    _write_rollback_revalidation_receipt = planner_artifacts_ext.write_rollback_revalidation_receipt

    retained_gain_summary = planner_history_ext.retained_gain_summary

    subsystem_history_summary = planner_controls_ext.subsystem_history_summary

    variant_history_summary = planner_controls_ext.variant_history_summary

    recent_subsystem_activity_summary = planner_history_ext.recent_subsystem_activity_summary

    recent_campaign_surface_activity_summary = (
        planner_history_ext.recent_campaign_surface_activity_summary
    )

    recent_variant_activity_summary = planner_history_ext.recent_variant_activity_summary

    _decision_records = planner_runtime_state_ext.decision_records

    _load_retained_universe_payload_from_record = (
        planner_runtime_state_ext.load_retained_universe_payload_from_record
    )

    _resolve_cycles_path = planner_runtime_state_ext.resolve_cycles_path

    _improvement_planner_controls = planner_runtime_state_ext.improvement_planner_controls

    def _apply_improvement_planner_mutation(
        self,
        candidate: ImprovementExperiment,
        *,
        planner_controls: dict[str, object],
    ) -> tuple[ImprovementExperiment, dict[str, object]]:
        return planner_controls_ext.apply_improvement_planner_mutation(
            self,
            candidate,
            planner_controls=planner_controls,
        )

    _apply_variant_planner_mutation = planner_controls_ext.apply_variant_planner_mutation

    _with_variant_expansions = planner_controls_ext.with_variant_expansions

    _planner_control_float = staticmethod(planner_controls_ext.planner_control_float)

    _planner_control_variant_float = staticmethod(planner_controls_ext.planner_control_variant_float)

    _planner_guardrail_float = staticmethod(planner_controls_ext.planner_guardrail_float)

    _planner_control_subsystem_float = staticmethod(planner_controls_ext.planner_control_subsystem_float)

    def _cycle_variant_index(self, output_path: Path | None = None) -> dict[str, str]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return {}
        cycle_variants: dict[str, str] = {}
        for record in self.load_cycle_records(resolved):
            cycle_id = str(record.get("cycle_id", ""))
            if not cycle_id:
                continue
            variant_id = _record_selected_variant_id(record)
            if variant_id:
                cycle_variants[cycle_id] = variant_id
        return cycle_variants

    _decision_summary = staticmethod(planner_history_ext.decision_summary)

    _portfolio_adjusted_experiment_score = planner_portfolio_ext.portfolio_adjusted_experiment_score

    _history_bonus = staticmethod(planner_history_ext.history_bonus)

    _recent_history_bonus = staticmethod(planner_history_ext.recent_history_bonus)

    _decision_quality_score = staticmethod(planner_history_ext.decision_quality_score)

    _empty_recent_activity_summary = staticmethod(planner_history_ext.empty_recent_activity_summary)

    _variant_exploration_bonus = planner_scoring_ext.variant_exploration_bonus

    _campaign_breadth_pressure = staticmethod(planner_history_ext.campaign_breadth_pressure)

    _variant_breadth_pressure = staticmethod(planner_history_ext.variant_breadth_pressure)

    _record_has_phase_gate_failure = staticmethod(planner_history_ext.record_has_phase_gate_failure)

    _record_is_reconciled_failure = staticmethod(planner_history_ext.record_is_reconciled_failure)

    _record_has_observation_timeout = staticmethod(planner_history_ext.record_has_observation_timeout)

    _record_observation_timeout_budget_source = staticmethod(
        planner_history_ext.record_observation_timeout_budget_source
    )

    @classmethod
    def _record_has_regression_signal(cls, record: dict[str, object]) -> bool:
        return planner_runtime_state_ext.record_has_regression_signal(cls, record)

def _default_runtime_learning_artifacts_path() -> Path | None:
    try:
        return KernelConfig().learning_artifacts_path
    except Exception:
        return None


def evaluate_artifact_retention(
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    artifact_path: Path | None = None,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> tuple[str, str]:
    subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem, capability_modules_path)
    _ensure_retention_plugin_registry()
    plugin = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retention_plugin(
        subsystem,
        capability_modules_path=capability_modules_path,
    )
    return engine_evaluate_artifact_retention(
        subsystem=subsystem,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        artifact_path=artifact_path,
        payload=payload,
        capability_modules_path=capability_modules_path,
        load_payload_fn=_load_json_payload,
        assess_artifact_compatibility_fn=assess_artifact_compatibility,
        retention_gate_fn=_retention_gate,
        retention_evidence_fn=retention_evidence,
        evaluator_fn=plugin.evaluator,
    )


_generated_lane_regressed = retention_policies_ext._generated_lane_regressed
_failure_recovery_regressed = retention_policies_ext._failure_recovery_regressed
_learning_artifact_support = retention_policies_ext._learning_artifact_support
_skills_learning_support_satisfied = retention_policies_ext._skills_learning_support_satisfied
_common_family_and_lane_checks = retention_policies_ext._common_family_and_lane_checks


_evaluate_curriculum_retention = retention_policies_ext._evaluate_curriculum_retention
_evaluate_verifier_retention = retention_policies_ext._evaluate_verifier_retention
_evaluate_benchmark_retention = retention_policies_ext._evaluate_benchmark_retention
_evaluate_policy_retention = retention_policies_ext._evaluate_policy_retention
_evaluate_world_model_retention = retention_policies_ext._evaluate_world_model_retention
_evaluate_state_estimation_retention = retention_policies_ext._evaluate_state_estimation_retention
_evaluate_control_surface_retention = retention_policies_ext._evaluate_control_surface_retention
_evaluate_trust_retention = retention_policies_ext._evaluate_trust_retention
_evaluate_universe_retention = retention_policies_ext._evaluate_universe_retention
_evaluate_recovery_retention = retention_policies_ext._evaluate_recovery_retention
_evaluate_delegation_retention = retention_policies_ext._evaluate_delegation_retention
_evaluate_operator_policy_retention = retention_policies_ext._evaluate_operator_policy_retention
_evaluate_transition_model_retention = retention_policies_ext._evaluate_transition_model_retention


_broad_coding_retrieval_support_signal = retention_policies_ext._broad_coding_retrieval_support_signal
_evaluate_retrieval_retention = retention_policies_ext._evaluate_retrieval_retention
_evaluate_tolbert_model_retention = retention_policies_ext._evaluate_tolbert_model_retention
_evaluate_qwen_adapter_retention = retention_policies_ext._evaluate_qwen_adapter_retention
_evaluate_capabilities_retention = retention_policies_ext._evaluate_capabilities_retention
_evaluate_skill_or_tooling_retention = retention_policies_ext._evaluate_skill_or_tooling_retention
_evaluate_operators_retention = retention_policies_ext._evaluate_operators_retention
_RETENTION_EVALUATORS = retention_policies_ext.RETENTION_EVALUATORS

_RETENTION_PLUGIN_REGISTRY_INITIALIZED = False


def _ensure_retention_plugin_registry() -> None:
    global _RETENTION_PLUGIN_REGISTRY_INITIALIZED
    if _RETENTION_PLUGIN_REGISTRY_INITIALIZED:
        return
    for subsystem, evaluator in _RETENTION_EVALUATORS.items():
        DEFAULT_IMPROVEMENT_PLUGIN_LAYER.register_retention_plugin(
            subsystem,
            evaluator=evaluator,
        )
    _RETENTION_PLUGIN_REGISTRY_INITIALIZED = True


def retention_evidence(
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    return retention_evidence_ext.retention_evidence(
        subsystem,
        baseline_metrics,
        candidate_metrics,
        payload=payload,
        capability_modules_path=capability_modules_path,
    )


proposal_gate_failure_reason = retention_helpers.proposal_gate_failure_reason
proposal_gate_failure_reasons_by_benchmark_family = retention_helpers.proposal_gate_failure_reasons_by_benchmark_family


_tolbert_selection_signal_fallback_satisfied = retention_policies_ext._tolbert_selection_signal_fallback_satisfied


_generated_kind_pass_rate = retention_helpers._generated_kind_pass_rate
_has_generated_kind = retention_helpers._has_generated_kind
_state_estimation_latent_controls_from_payload = control_evidence.state_estimation_latent_controls_from_payload
_state_estimation_policy_controls_from_payload = control_evidence.state_estimation_policy_controls_from_payload
_state_estimation_improvement_count = control_evidence.state_estimation_improvement_count
_trajectory_has_regression = control_evidence.trajectory_has_regression
_state_regression_trace_count = control_evidence.state_regression_trace_count
_regressive_recovery_rate = control_evidence.regressive_recovery_rate
_paired_trajectory_non_regression_rate = control_evidence.paired_trajectory_non_regression_rate
_state_estimation_trajectory_score = control_evidence.state_estimation_trajectory_score
_state_estimation_evidence = control_evidence.state_estimation_evidence
_trust_controls_from_payload = control_evidence.trust_controls_from_payload
_universe_governance_from_payload = control_evidence.universe_governance_from_payload
_universe_invariants_from_payload = control_evidence.universe_invariants_from_payload
_universe_forbidden_patterns_from_payload = control_evidence.universe_forbidden_patterns_from_payload
_universe_preferred_prefixes_from_payload = control_evidence.universe_preferred_prefixes_from_payload
_universe_action_risk_controls_from_payload = control_evidence.universe_action_risk_controls_from_payload
_universe_environment_assumptions_from_payload = control_evidence.universe_environment_assumptions_from_payload
_universe_change_scope = control_evidence.universe_change_scope
_universe_cross_family_support = control_evidence.universe_cross_family_support
_universe_outcome_weighted_support = control_evidence.universe_outcome_weighted_support
_universe_improvement_count = control_evidence.universe_improvement_count
_environment_assumption_delta_count = control_evidence.environment_assumption_delta_count
_more_restrictive_environment_assumption_count = control_evidence.more_restrictive_environment_assumption_count
_universe_control_evidence = control_evidence.universe_control_evidence
_trust_control_improvement_count = control_evidence.trust_control_improvement_count
_trust_control_evidence = control_evidence.trust_control_evidence
_recovery_controls_from_payload = control_evidence.recovery_controls_from_payload
_recovery_control_improvement_count = control_evidence.recovery_control_improvement_count
_recovery_control_evidence = control_evidence.recovery_control_evidence
_delegation_controls_from_payload = control_evidence.delegation_controls_from_payload
_delegation_control_improvement_count = control_evidence.delegation_control_improvement_count
_delegation_control_evidence = control_evidence.delegation_control_evidence
_operator_policy_controls_from_payload = control_evidence.operator_policy_controls_from_payload
_operator_policy_improvement_count = control_evidence.operator_policy_improvement_count
_operator_policy_control_evidence = control_evidence.operator_policy_control_evidence
_enabled_flag_improvement_count = control_evidence.enabled_flag_improvement_count
_increased_int_control_count = control_evidence.increased_int_control_count
_expanded_sequence_control_count = control_evidence.expanded_sequence_control_count
_boolean_control_deltas = control_evidence.boolean_control_deltas
_integer_control_deltas = control_evidence.integer_control_deltas
_sequence_length_deltas = control_evidence.sequence_length_deltas
_transition_model_signatures_from_payload = transition_model_improvement_ext.retained_transition_model_signatures
_transition_model_improvement_count = control_evidence.transition_model_improvement_count
_transition_model_evidence = control_evidence.transition_model_evidence
_capability_surface_evidence = control_evidence.capability_surface_evidence


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
    runtime_config: KernelConfig | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    _ensure_retention_plugin_registry()
    plugin = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retention_plugin(
        subsystem,
        capability_modules_path=capability_modules_path,
    )
    return engine_apply_artifact_retention_decision(
        artifact_path=artifact_path,
        subsystem=subsystem,
        cycle_id=cycle_id,
        decision_state=decision_state,
        decision_reason=decision_reason,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        active_artifact_path=active_artifact_path,
        capability_modules_path=capability_modules_path,
        runtime_config=runtime_config,
        load_json_payload_fn=_load_json_payload,
        artifact_sha256_fn=artifact_sha256,
        snapshot_artifact_fn=_snapshot_artifact,
        assess_artifact_compatibility_fn=assess_artifact_compatibility,
        update_tool_candidate_states_fn=_update_tool_candidate_states,
        stamp_tolbert_lineage_metadata_fn=_stamp_tolbert_lineage_metadata,
        promote_tolbert_payload_to_canonical_checkpoint_fn=_promote_tolbert_payload_to_canonical_checkpoint,
        compact_rejected_tolbert_payload_fn=_compact_rejected_tolbert_payload,
        atomic_write_json_fn=atomic_write_json,
        cleanup_rejected_tolbert_payload_artifacts_fn=_cleanup_rejected_tolbert_payload_artifacts,
        prior_active_artifact_path_fn=_prior_active_artifact_path,
        atomic_copy_file_fn=atomic_copy_file,
        synchronize_retained_universe_artifacts_fn=_synchronize_retained_universe_artifacts,
        post_apply_hook_fn=plugin.post_apply_hook,
        repo_root=repo_root,
    )


_synchronize_retained_universe_artifacts = artifact_runtime_support_ext.synchronize_retained_universe_artifacts
_universe_sync_paths = artifact_runtime_support_ext.universe_sync_paths
_runtime_config_for_universe_sync = artifact_runtime_support_ext.runtime_config_for_universe_sync
_load_first_retained_universe_payload = artifact_runtime_support_ext.load_first_retained_universe_payload


_stamp_tolbert_lineage_metadata = artifact_runtime_support_ext.stamp_tolbert_lineage_metadata
_compact_rejected_tolbert_payload = artifact_runtime_support_ext.compact_rejected_tolbert_payload
_promote_tolbert_payload_to_canonical_checkpoint = artifact_runtime_support_ext.promote_tolbert_payload_to_canonical_checkpoint
_strip_pathlike_fields = artifact_runtime_support_ext.strip_pathlike_fields
_cleanup_rejected_tolbert_payload_artifacts = artifact_runtime_support_ext.cleanup_rejected_tolbert_payload_artifacts
_cleanup_unreferenced_tolbert_store = artifact_runtime_support_ext.cleanup_unreferenced_tolbert_store
_tolbert_reference_artifact_paths = artifact_runtime_support_ext.tolbert_reference_artifact_paths
_tolbert_store_references_from_paths = artifact_runtime_support_ext.tolbert_store_references_from_paths
_tolbert_shared_store_paths_from_payload = artifact_runtime_support_ext.tolbert_shared_store_paths_from_payload


persist_replay_verified_tool_artifact = artifacts_persist_replay_verified_tool_artifact
effective_artifact_payload_for_retention = artifacts_effective_artifact_payload_for_retention


_is_positive_int = artifact_compatibility_ext._is_positive_int
_operator_task_contract = artifact_compatibility_ext._operator_task_contract
_operator_steps = artifact_compatibility_ext._operator_steps
_operator_benchmark_families = artifact_compatibility_ext._operator_benchmark_families
_operator_support = artifact_compatibility_ext._operator_support
_tool_candidate_stage = artifact_compatibility_ext._tool_candidate_stage
_tool_candidate_lifecycle_state = artifact_compatibility_ext._tool_candidate_lifecycle_state
_artifact_contract = artifact_compatibility_ext._artifact_contract
_allowed_artifact_lifecycle_states = artifact_compatibility_ext._allowed_artifact_lifecycle_states
_artifact_validation_profile = artifact_compatibility_ext._artifact_validation_profile
_rule_error = artifact_compatibility_ext._rule_error
_string_list_has_content = artifact_compatibility_ext._string_list_has_content
_validate_profile_rule = artifact_compatibility_ext._validate_profile_rule
_validate_profile_object_section = artifact_compatibility_ext._validate_profile_object_section
_validate_profile_list_section = artifact_compatibility_ext._validate_profile_list_section
_validate_artifact_profile = artifact_compatibility_ext._validate_artifact_profile


assess_artifact_compatibility = artifact_compatibility_ext.assess_artifact_compatibility
artifact_sha256 = artifacts_artifact_sha256


_load_json_payload = artifact_protocol_support_ext.load_json_payload
_snapshot_artifact = artifact_protocol_support_ext.snapshot_artifact


snapshot_artifact_state = artifacts_snapshot_artifact_state
materialize_replay_verified_tool_payload = artifacts_materialize_replay_verified_tool_payload


_update_tool_candidate_states = artifact_protocol_support_ext.update_tool_candidate_states
_retention_gate = artifact_protocol_support_ext.retention_gate


retention_gate_for_payload = artifacts_retention_gate_for_payload


_candidate_family_failure_rate = evidence_helpers.candidate_family_failure_rate
_trusted_carryover_repair_rate = evidence_helpers.trusted_carryover_repair_rate
_trusted_carryover_verified_steps = evidence_helpers.trusted_carryover_verified_steps
_family_discrimination_gain = evidence_helpers.family_discrimination_gain
_family_pass_rate_delta_map = evidence_helpers.family_pass_rate_delta_map
_family_regression_count = evidence_helpers.family_regression_count
_family_worst_delta = evidence_helpers.family_worst_delta
_generated_family_pass_rate = retention_helpers._generated_family_pass_rate
_generated_family_pass_rate_delta_map = retention_helpers._generated_family_pass_rate_delta_map
_generated_family_regression_count = retention_helpers._generated_family_regression_count
_generated_family_worst_delta = retention_helpers._generated_family_worst_delta
_difficulty_pass_rate_delta_map = retention_helpers._difficulty_pass_rate_delta_map
_proposal_metrics_by_difficulty = retention_helpers._proposal_metrics_by_difficulty
_world_feedback_by_difficulty = retention_helpers._world_feedback_by_difficulty
_world_feedback_by_benchmark_family = retention_helpers._world_feedback_by_benchmark_family
_world_feedback_delta = retention_helpers._world_feedback_delta
_long_horizon_summary = retention_helpers._long_horizon_summary
_benchmark_family_summary = retention_helpers._benchmark_family_summary
_proposal_metrics_by_benchmark_family = retention_helpers._proposal_metrics_by_benchmark_family
_proposal_metrics_delta_by_benchmark_family = retention_helpers._proposal_metrics_delta_by_benchmark_family
_proposal_metrics_delta_by_difficulty = retention_helpers._proposal_metrics_delta_by_difficulty
_world_feedback_delta_by_difficulty = retention_helpers._world_feedback_delta_by_difficulty
_verifier_discrimination_gain = retention_helpers._verifier_discrimination_gain
_validate_cycle_record_consistency = records_helpers.validate_cycle_record_consistency
_record_metrics_summary = records_helpers.record_metrics_summary
_record_selected_variant_id = records_helpers.record_selected_variant_id
_record_strategy_candidate_id = records_helpers.record_strategy_candidate_id
_record_strategy_candidate_kind = records_helpers.record_strategy_candidate_kind
_record_strategy_origin = records_helpers.record_strategy_origin
_record_prior_retained_cycle_id = records_helpers.record_prior_retained_cycle_id
_record_float_value = records_helpers.record_float_value
_record_non_regressed_family_support = records_helpers.record_non_regressed_family_support
_record_phase_gate_passed = records_helpers.record_phase_gate_passed
_record_phase_gate_failures = records_helpers.record_phase_gate_failures
_dominant_count_label = records_helpers.dominant_count_label
_dominant_weight_label = records_helpers.dominant_weight_label
_normalized_cycle_record = records_helpers.normalized_cycle_record


_verifier_contracts_are_strict = artifact_protocol_support_ext.verifier_contracts_are_strict
_tool_candidates_have_stage = artifact_protocol_support_ext.tool_candidates_have_stage

tool_shared_repo_bundle_evidence = artifact_support_evidence_ext.tool_shared_repo_bundle_evidence
artifact_retrieval_reuse_evidence = artifact_support_evidence_ext.artifact_retrieval_reuse_evidence


_tool_candidate_shared_repo_bundle = artifact_support_evidence_ext.tool_candidate_shared_repo_bundle
_tool_candidate_retention_sort_key = artifact_support_evidence_ext.tool_candidate_retention_sort_key
_normalized_tool_candidates_for_retention = artifact_support_evidence_ext.normalized_tool_candidates_for_retention
_operator_support_count = artifact_support_evidence_ext.operator_support_count


stamp_artifact_experiment_variant = artifacts_stamp_artifact_experiment_variant
stamp_artifact_generation_context = artifacts_stamp_artifact_generation_context
payload_with_active_artifact_context = artifacts_payload_with_active_artifact_context


_prior_active_artifact_path = artifact_protocol_support_ext.prior_active_artifact_path


staged_candidate_artifact_path = artifacts_staged_candidate_artifact_path
