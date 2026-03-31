from __future__ import annotations

from dataclasses import dataclass, fields
from copy import deepcopy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from evals.metrics import EvalMetrics

from .capabilities import load_capability_modules
from .capability_improvement import capability_surface_summary
from .config import KernelConfig, current_external_task_manifests_paths
from .delegation_policy import delegation_policy_snapshot
from .improvement_common import normalized_control_mapping, retained_artifact_payload, retention_gate_preset
from .memory import EpisodeMemory
from .modeling.tolbert.delta import materialize_tolbert_checkpoint_from_delta, resolve_tolbert_runtime_checkpoint_path
from .operator_policy import operator_policy_snapshot
from .prompt_improvement import retained_improvement_planner_controls
from .state_estimation_improvement import (
    STATE_ESTIMATION_GENERATION_FOCI,
    STATE_ESTIMATION_LATENT_CONTROL_KEYS,
    STATE_ESTIMATION_POLICY_CONTROL_KEYS,
    STATE_ESTIMATION_PROPOSAL_AREAS,
    STATE_ESTIMATION_TRANSITION_CONTROL_KEYS,
    retained_state_estimation_latent_controls,
    retained_state_estimation_policy_controls,
    retained_state_estimation_transition_controls,
)
from .subsystems import (
    active_artifact_path_for_subsystem,
    base_subsystem_for,
    default_variant_definitions,
    external_planner_experiments,
)
from .task_bank import TaskBank
from .transition_model_improvement import (
    retained_transition_model_controls,
    retained_transition_model_signatures,
)
from .universe_improvement import (
    UNIVERSE_ACTION_RISK_CONTROL_KEYS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS,
    UNIVERSE_GENERATION_FOCI,
    UNIVERSE_GOVERNANCE_KEYS,
    UNIVERSE_PROPOSAL_AREAS,
    compose_universe_bundle_payloads,
    retained_universe_action_risk_controls,
    retained_universe_environment_assumptions,
    retained_universe_forbidden_command_patterns,
    retained_universe_governance,
    retained_universe_invariants,
    retained_universe_preferred_command_prefixes,
    sibling_universe_bundle_paths,
    universe_bundle_contains_path,
    universe_bundle_paths,
    write_universe_bundle_files,
)

_WORLD_MODEL_CONTROL_KEYS = {
    "expected_artifact_score_weight",
    "preserved_artifact_score_weight",
    "forbidden_artifact_penalty",
    "forbidden_cleanup_score_weight",
    "workflow_branch_target_score_weight",
    "workflow_changed_path_score_weight",
    "workflow_generated_path_score_weight",
    "workflow_report_path_score_weight",
    "workflow_preserved_path_score_weight",
    "required_tests_score_weight",
    "required_merges_score_weight",
    "long_horizon_scaffold_bonus",
    "retrieved_expected_artifact_score_weight",
    "retrieved_forbidden_artifact_penalty",
    "retrieved_preserved_artifact_score_weight",
    "retrieved_workflow_changed_path_score_weight",
    "retrieved_workflow_report_path_score_weight",
}
_WORLD_MODEL_PLANNING_CONTROL_KEYS = {
    "include_preserved_artifact_steps",
    "prefer_preserved_artifacts_first",
    "append_preservation_subgoal",
    "max_preserved_artifacts",
}
_WORLD_MODEL_PROPOSAL_AREAS = {
    "workflow_alignment",
    "conflict_avoidance",
    "preservation_bias",
    "branch_targeting",
}
_WORLD_MODEL_GENERATION_FOCI = {
    "balanced",
    "workflow_alignment",
    "preservation_bias",
    "conflict_avoidance",
}
_STATE_ESTIMATION_CONTROL_KEYS = set(STATE_ESTIMATION_TRANSITION_CONTROL_KEYS)
_STATE_ESTIMATION_LATENT_KEYS = set(STATE_ESTIMATION_LATENT_CONTROL_KEYS)
_STATE_ESTIMATION_POLICY_KEYS = set(STATE_ESTIMATION_POLICY_CONTROL_KEYS)
_STATE_ESTIMATION_PROPOSAL_AREAS = set(STATE_ESTIMATION_PROPOSAL_AREAS)
_STATE_ESTIMATION_GENERATION_FOCI = set(STATE_ESTIMATION_GENERATION_FOCI)
_RECOVERY_CONTROL_KEYS = {
    "snapshot_before_execution",
    "rollback_on_runner_exception",
    "rollback_on_failed_outcome",
    "rollback_on_safe_stop",
    "verify_post_rollback_file_count",
    "max_post_rollback_file_count",
}
_RECOVERY_PROPOSAL_AREAS = {
    "rollback_safety",
    "snapshot_coverage",
}
_RECOVERY_GENERATION_FOCI = {
    "balanced",
    "rollback_safety",
    "snapshot_coverage",
}
_DELEGATION_CONTROL_KEYS = {
    "delegated_job_max_concurrency",
    "delegated_job_max_active_per_budget_group",
    "delegated_job_max_queued_per_budget_group",
    "delegated_job_max_artifact_bytes",
    "delegated_job_max_subprocesses_per_job",
    "command_timeout_seconds",
    "llm_timeout_seconds",
    "max_steps",
}
_DELEGATION_PROPOSAL_AREAS = {
    "throughput_balance",
    "queue_elasticity",
    "worker_depth",
}
_DELEGATION_GENERATION_FOCI = {
    "balanced",
    "throughput_balance",
    "queue_elasticity",
    "worker_depth",
}
_UNIVERSE_ACTION_RISK_KEYS = set(UNIVERSE_ACTION_RISK_CONTROL_KEYS)
_UNIVERSE_ENVIRONMENT_ENUM_FIELDS = {
    str(key): set(values) for key, values in UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS.items()
}
_UNIVERSE_ENVIRONMENT_BOOL_FIELDS = set(UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS)
_UNIVERSE_GOVERNANCE_KEYS = set(UNIVERSE_GOVERNANCE_KEYS)
_UNIVERSE_PROPOSAL_AREAS = set(UNIVERSE_PROPOSAL_AREAS)
_UNIVERSE_GENERATION_FOCI = set(UNIVERSE_GENERATION_FOCI)
_OPERATOR_POLICY_CONTROL_KEYS = {
    "unattended_allowed_benchmark_families",
    "unattended_allow_git_commands",
    "unattended_allow_http_requests",
    "unattended_http_allowed_hosts",
    "unattended_http_timeout_seconds",
    "unattended_http_max_body_bytes",
    "unattended_allow_generated_path_mutations",
    "unattended_generated_path_prefixes",
}
_OPERATOR_POLICY_PROPOSAL_AREAS = {
    "family_breadth",
    "git_http_scope",
    "generated_path_scope",
}
_OPERATOR_POLICY_GENERATION_FOCI = {
    "balanced",
    "family_breadth",
    "git_http_scope",
    "generated_path_scope",
}
_TRANSITION_MODEL_CONTROL_KEYS = {
    "repeat_command_penalty",
    "regressed_path_command_penalty",
    "recovery_command_bonus",
    "progress_command_bonus",
    "max_signatures",
}
_TRANSITION_MODEL_PROPOSAL_AREAS = {
    "repeat_avoidance",
    "regression_guard",
    "recovery_bias",
}
_TRANSITION_MODEL_GENERATION_FOCI = {
    "balanced",
    "repeat_avoidance",
    "regression_guard",
    "recovery_bias",
}
_TOLBERT_MODEL_GENERATION_FOCI = {
    "balanced",
    "recovery_alignment",
    "discovered_task_adaptation",
}
_TOLBERT_MODEL_SURFACE_KEYS = {
    "encoder_surface",
    "latent_dynamics_surface",
    "decoder_surface",
    "world_model_surface",
    "retrieval_surface",
    "policy_head",
    "value_head",
    "transition_head",
    "risk_head",
    "stop_head",
    "latent_state",
    "universal_runtime",
}
_TOLBERT_RUNTIME_POLICY_KEYS = {
    "shadow_benchmark_families",
    "primary_benchmark_families",
    "min_path_confidence",
    "require_trusted_retrieval",
    "fallback_to_vllm_on_low_confidence",
    "allow_direct_command_primary",
    "allow_skill_primary",
    "primary_min_command_score",
    "use_encoder_context",
    "use_decoder_head",
    "use_value_head",
    "use_transition_head",
    "use_world_model_head",
    "use_risk_head",
    "use_stop_head",
    "use_policy_head",
    "use_latent_state",
}
_TOLBERT_DECODER_POLICY_KEYS = {
    "allow_retrieval_guidance",
    "allow_skill_commands",
    "allow_task_suggestions",
    "allow_stop_decision",
    "min_stop_completion_ratio",
    "max_task_suggestions",
}
_TOLBERT_ACTION_GENERATION_POLICY_KEYS = {
    "enabled",
    "max_candidates",
    "proposal_score_bias",
    "novel_command_bonus",
    "verifier_alignment_bonus",
    "expected_file_template_bonus",
    "cleanup_template_bonus",
    "min_family_support",
    "template_preferences",
}
_TOLBERT_ROLLOUT_POLICY_KEYS = {
    "predicted_progress_gain_weight",
    "predicted_conflict_penalty_weight",
    "predicted_preserved_bonus_weight",
    "predicted_workflow_bonus_weight",
    "latent_progress_bonus_weight",
    "latent_risk_penalty_weight",
    "recover_from_stall_bonus_weight",
    "stop_completion_weight",
    "stop_missing_expected_penalty_weight",
    "stop_forbidden_penalty_weight",
    "stop_preserved_penalty_weight",
    "stable_stop_bonus_weight",
}
_TOLBERT_LIFTOFF_GATE_KEYS = {
    "min_pass_rate_delta",
    "max_step_regression",
    "max_regressed_families",
    "require_generated_lane_non_regression",
    "require_failure_recovery_non_regression",
    "require_shadow_signal",
    "min_shadow_episodes_per_promoted_family",
    "require_family_novel_command_evidence",
    "proposal_gate_by_benchmark_family",
    "require_unsafe_ambiguous_non_regression",
    "require_hidden_side_effect_non_regression",
    "require_success_hidden_side_effect_non_regression",
    "require_trust_gate_pass",
    "require_trust_success_non_regression",
    "require_trust_unsafe_non_regression",
    "require_trust_hidden_side_effect_non_regression",
    "require_trust_success_hidden_side_effect_non_regression",
    "require_takeover_drift_eval",
    "takeover_drift_step_budget",
    "takeover_drift_wave_task_limit",
    "takeover_drift_max_waves",
    "max_takeover_drift_pass_rate_regression",
    "max_takeover_drift_unsafe_ambiguous_rate_regression",
    "max_takeover_drift_hidden_side_effect_rate_regression",
    "max_takeover_drift_trust_success_rate_regression",
    "max_takeover_drift_trust_unsafe_ambiguous_rate_regression",
}
_TOLBERT_BUILD_POLICY_KEYS = {
    "allow_kernel_autobuild",
    "allow_kernel_rebuild",
    "require_synthetic_dataset",
    "require_head_targets",
    "min_total_examples",
    "min_synthetic_examples",
    "min_policy_examples",
    "min_transition_examples",
    "min_value_examples",
    "min_stop_examples",
    "ready_total_examples",
    "ready_synthetic_examples",
    "ready_policy_examples",
    "ready_transition_examples",
    "ready_value_examples",
    "ready_stop_examples",
}


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
    baseline_metrics: EvalMetrics
    candidate_metrics: EvalMetrics
    pass_rate_delta: float
    average_step_delta: float
    generated_pass_rate_delta: float
    regressed_family_count: int
    generated_regressed_family_count: int
    failure_recovery_delta: float


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
        self.runtime_config = runtime_config

    def _base_subsystem(self, subsystem: str) -> str:
        normalized = str(subsystem).strip()
        if not normalized:
            return ""
        try:
            return base_subsystem_for(normalized, self.capability_modules_path)
        except ValueError:
            return normalized

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

    def rank_targets(self, metrics: EvalMetrics) -> list[ImprovementTarget]:
        return [
            ImprovementTarget(
                subsystem=experiment.subsystem,
                reason=experiment.reason,
                priority=experiment.priority,
            )
            for experiment in self.rank_experiments(metrics)
        ]

    def rank_experiments(self, metrics: EvalMetrics) -> list[ImprovementExperiment]:
        failure_counts = self.failure_counts()
        transition_failure_counts = self.transition_failure_counts()
        transition_summary = self.transition_summary()
        candidates: list[ImprovementExperiment] = []
        if (failure_counts or transition_failure_counts) and (
            metrics.total_by_benchmark_family.get("benchmark_candidate", 0) == 0
            or metrics.generated_pass_rate < metrics.pass_rate
        ):
            benchmark_gain_raw = max(
                0.02,
                (
                    failure_counts.get("missing_expected_file", 0)
                    + transition_failure_counts.get("no_state_progress", 0)
                    + transition_failure_counts.get("state_regression", 0)
                )
                / max(1, sum(failure_counts.values())),
            )
            # Benchmark proposal generation is useful but should not dominate selection; cap its
            # nominal gain so we still explore kernel-impacting subsystems (retrieval/policy/etc).
            benchmark_gain = round(max(0.02, min(0.05, benchmark_gain_raw)), 4)
            candidates.append(
                ImprovementExperiment(
                    subsystem="benchmark",
                    reason="failure clusters, stalled transitions, and environment patterns can be turned into benchmark proposals that are not yet fully populated or discriminative",
                    priority=5,
                    expected_gain=benchmark_gain,
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "failure_counts": failure_counts,
                        "transition_failure_counts": transition_failure_counts,
                        "benchmark_candidate_total": metrics.total_by_benchmark_family.get("benchmark_candidate", 0),
                        "generated_pass_rate": metrics.generated_pass_rate,
                        "pass_rate": metrics.pass_rate,
                        "benchmark_expected_gain_raw": round(float(benchmark_gain_raw), 4),
                        "benchmark_expected_gain_capped": benchmark_gain,
                    },
                )
            )
        if metrics.low_confidence_episodes > 0 or metrics.trusted_retrieval_steps < metrics.total // 2:
            confidence_gap = max(
                metrics.low_confidence_episodes / max(1, metrics.total),
                0.0 if metrics.total == 0 else 0.5 - (metrics.trusted_retrieval_steps / max(1, metrics.total)),
            )
            candidates.append(
                ImprovementExperiment(
                    subsystem="retrieval",
                    reason="low-confidence retrieval remains common relative to trusted retrieval usage",
                    priority=5,
                    expected_gain=round(max(0.02, confidence_gap), 4),
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                        "total": metrics.total,
                    },
                )
            )
            candidates.append(
                ImprovementExperiment(
                    subsystem="tolbert_model",
                    reason="retrieval weakness should be attacked at the learned Tolbert checkpoint, not only with retained thresholds",
                    priority=5,
                    expected_gain=round(max(0.02, confidence_gap), 4),
                    estimated_cost=4,
                    score=0.0,
                    evidence={
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                        "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                        "total": metrics.total,
                    },
                )
            )
        if metrics.total_by_memory_source.get("verifier", 0) == 0 or transition_failure_counts:
            candidates.append(
                ImprovementExperiment(
                    subsystem="verifier",
                    reason="verifier-memory lane is not yet populated enough to discriminate stalled or regressive intermediate states",
                    priority=5,
                    expected_gain=round(max(0.03, min(0.05, 0.01 * sum(transition_failure_counts.values()))), 4),
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "verifier_memory_total": metrics.total_by_memory_source.get("verifier", 0),
                        "transition_failure_counts": transition_failure_counts,
                        "transition_summary": transition_summary,
                    },
                )
            )
        if failure_counts.get("command_failure", 0) >= failure_counts.get("missing_expected_file", 0) and failure_counts:
            candidates.append(
                ImprovementExperiment(
                    subsystem="tooling",
                    reason="command failures dominate stored failures, suggesting missing reusable procedures or tools",
                    priority=4,
                    expected_gain=round(
                        max(0.01, failure_counts.get("command_failure", 0) / max(1, sum(failure_counts.values()))),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "command_failure_count": failure_counts.get("command_failure", 0),
                        "failure_counts": failure_counts,
                    },
                )
            )
        if metrics.retrieval_ranked_skill_steps < max(1, metrics.skill_selected_steps // 2):
            candidates.append(
                ImprovementExperiment(
                    subsystem="skills",
                    reason="skill usage is high but retrieval-ranked skill selection is comparatively weak",
                    priority=4,
                    expected_gain=round(
                        max(
                            0.01,
                            (metrics.skill_selected_steps - metrics.retrieval_ranked_skill_steps) / max(1, metrics.skill_selected_steps),
                        ),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "skill_selected_steps": metrics.skill_selected_steps,
                        "retrieval_ranked_skill_steps": metrics.retrieval_ranked_skill_steps,
                    },
                )
            )
        if metrics.total_by_memory_source.get("skill_transfer", 0) == 0:
            candidates.append(
                ImprovementExperiment(
                    subsystem="operators",
                    reason="cross-task operator transfer is not yet populated or measured against raw skill transfer",
                    priority=4,
                    expected_gain=0.03,
                    estimated_cost=4,
                    score=0.0,
                    evidence={
                        "skill_transfer_total": metrics.total_by_memory_source.get("skill_transfer", 0),
                        "operator_total": metrics.total_by_memory_source.get("operator", 0),
                    },
                )
            )
        if (
            (metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate)
            or transition_failure_counts.get("no_state_progress", 0) > 0
            or transition_failure_counts.get("state_regression", 0) > 0
        ):
            generated_gap = 0.0
            if metrics.generated_total:
                generated_gap = max(0.0, metrics.pass_rate - metrics.generated_pass_rate)
            candidates.append(
                ImprovementExperiment(
                    subsystem="curriculum",
                    reason="generated-task pressure should target repeated no-progress and regression transitions, not only coarse task failures",
                    priority=4,
                    expected_gain=round(
                        max(
                            0.02,
                            generated_gap,
                            min(
                                0.05,
                                (
                                    transition_failure_counts.get("no_state_progress", 0)
                                    + transition_failure_counts.get("state_regression", 0)
                                )
                                / max(1, sum(transition_failure_counts.values())),
                            ),
                        ),
                        4,
                    ),
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "pass_rate": metrics.pass_rate,
                        "generated_pass_rate": metrics.generated_pass_rate,
                        "generated_total": metrics.generated_total,
                        "generated_pass_rate_gap": round(generated_gap, 4),
                        "transition_failure_counts": transition_failure_counts,
                    },
                )
            )
        repo_world_model_total = sum(
            int(metrics.total_by_benchmark_family.get(family, 0))
            for family in ("repo_sandbox", "repo_chore", "repository", "project", "integration")
        )
        world_model_failure_signal = (
            failure_counts.get("missing_expected_file", 0)
            + failure_counts.get("command_failure", 0)
            + transition_failure_counts.get("no_state_progress", 0)
            + transition_failure_counts.get("state_regression", 0)
        )
        if repo_world_model_total > 0 and world_model_failure_signal > 0:
            candidates.append(
                ImprovementExperiment(
                    subsystem="world_model",
                    reason="repo-workflow failures and bad transitions suggest command scoring, progress estimation, and preserved-path modeling are still weak",
                    priority=3,
                    expected_gain=round(
                        max(0.01, min(0.03, world_model_failure_signal / max(1, sum(failure_counts.values())))),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "repo_world_model_total": repo_world_model_total,
                        "failure_counts": failure_counts,
                        "transition_failure_counts": transition_failure_counts,
                        "transition_summary": transition_summary,
                    },
                )
            )
        if transition_failure_counts or int(transition_summary.get("state_regression_steps", 0) or 0) > 0:
            state_estimation_signal = (
                transition_failure_counts.get("no_state_progress", 0)
                + transition_failure_counts.get("state_regression", 0)
                + int(transition_summary.get("state_regression_steps", 0) or 0)
            )
            candidates.append(
                ImprovementExperiment(
                    subsystem="state_estimation",
                    reason="state summarization should separate stalls, regressions, and recovery opportunities more explicitly before policy scoring",
                    priority=4,
                    expected_gain=round(
                        max(0.015, min(0.04, state_estimation_signal / max(1, sum(transition_failure_counts.values()) or 1))),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "transition_failure_counts": transition_failure_counts,
                        "transition_summary": transition_summary,
                    },
                )
            )
        universe_constitution_signal = (
            failure_counts.get("command_failure", 0)
            + transition_failure_counts.get("no_state_progress", 0)
            + transition_failure_counts.get("state_regression", 0)
            + metrics.low_confidence_episodes
        )
        environment_violation_summary = self.environment_violation_summary()
        universe_cycle_feedback = self.universe_cycle_feedback_summary()
        operating_envelope_signal = (
            int(environment_violation_summary.get("violation_total", 0))
            + int(environment_violation_summary.get("alignment_failure_total", 0))
        )
        broad_support_bonus = min(
            2,
            int(
                universe_cycle_feedback.get(
                    "broad_support_cycle_count",
                    universe_cycle_feedback.get("retained_cycle_count", 0),
                )
            ),
        )
        universe_constitution_signal += min(
            1,
            int(universe_cycle_feedback.get("constitution_retained_cycle_count", 0) or 0),
        )
        operating_envelope_signal += broad_support_bonus
        operating_envelope_signal += min(
            1,
            int(universe_cycle_feedback.get("operating_envelope_retained_cycle_count", 0) or 0),
        )
        if universe_constitution_signal > 0:
            candidates.append(
                ImprovementExperiment(
                    subsystem="universe_constitution",
                    reason="constitutional machine-law should tighten verifier, bounded-action, and destructive-reset rules above task-local world state",
                    priority=4,
                    expected_gain=round(
                        max(0.012, min(0.025, universe_constitution_signal / max(1, metrics.total or 1))),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "failure_counts": failure_counts,
                        "transition_failure_counts": transition_failure_counts,
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "universe_cycle_feedback": universe_cycle_feedback,
                        "total": metrics.total,
                    },
                )
            )
        if operating_envelope_signal > 0:
            candidates.append(
                ImprovementExperiment(
                    subsystem="operating_envelope",
                    reason="the retained operating envelope should calibrate to repeated environment conflicts and attested runtime mismatch",
                    priority=4,
                    expected_gain=round(
                        max(0.015, min(0.03, operating_envelope_signal / max(1, metrics.total or 1))),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "failure_counts": failure_counts,
                        "transition_failure_counts": transition_failure_counts,
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "environment_violation_summary": environment_violation_summary,
                        "universe_cycle_feedback": universe_cycle_feedback,
                        "total": metrics.total,
                    },
                )
            )
        if transition_failure_counts:
            candidates.append(
                ImprovementExperiment(
                    subsystem="transition_model",
                    reason="retained bad-transition signatures should directly penalize repeated stalled or regressive commands",
                    priority=5,
                    expected_gain=round(
                        max(
                            0.02,
                            min(
                                0.05,
                                (
                                    transition_failure_counts.get("no_state_progress", 0)
                                    + transition_failure_counts.get("state_regression", 0)
                                )
                                / max(1, sum(transition_failure_counts.values())),
                            ),
                        ),
                        4,
                    ),
                    estimated_cost=2,
                    score=0.0,
                    evidence={
                        "transition_failure_counts": transition_failure_counts,
                        "transition_summary": transition_summary,
                    },
                )
            )
        trust_summary = self.trust_ledger_summary()
        if trust_summary and (
            str(trust_summary.get("overall_status", "")).strip() in {"bootstrap", "restricted"}
            or float(trust_summary.get("unsafe_ambiguous_rate", 0.0)) > 0.0
            or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
            or int(trust_summary.get("distinct_benchmark_families", 0)) < 2
        ):
            trust_risk_signal = max(
                float(trust_summary.get("unsafe_ambiguous_rate", 0.0)),
                float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)),
                float(trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)),
            )
            candidates.append(
                ImprovementExperiment(
                    subsystem="trust",
                    reason="unattended trust gating remains restricted, under-sampled, or exposed to hidden-risk outcomes",
                    priority=4,
                    expected_gain=round(max(0.01, trust_risk_signal or 0.02), 4),
                    estimated_cost=2,
                    score=0.0,
                    evidence=trust_summary,
                )
            )
        if trust_summary and (
            float(trust_summary.get("rollback_performed_rate", 0.0)) > 0.0
            or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
        ):
            recovery_signal = max(
                float(trust_summary.get("rollback_performed_rate", 0.0)),
                float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)),
            )
            candidates.append(
                ImprovementExperiment(
                    subsystem="recovery",
                    reason="unattended runs still depend on rollback or leave hidden side-effect risk after restore paths",
                    priority=4,
                    expected_gain=round(max(0.01, recovery_signal), 4),
                    estimated_cost=2,
                    score=0.0,
                    evidence=trust_summary,
                )
            )
        delegation_summary = self.delegation_policy_summary()
        if delegation_summary and (
            int(delegation_summary.get("delegated_job_max_concurrency", 1)) <= 1
            or int(delegation_summary.get("delegated_job_max_subprocesses_per_job", 1)) <= 1
            or int(delegation_summary.get("max_steps", 5)) <= 5
        ):
            candidates.append(
                ImprovementExperiment(
                    subsystem="delegation",
                    reason="delegated execution remains throttled by shallow worker budgets or narrow queue policy",
                    priority=4,
                    expected_gain=0.02,
                    estimated_cost=2,
                    score=0.0,
                    evidence=delegation_summary,
                )
            )
        operator_policy_summary = self.operator_policy_summary()
        if operator_policy_summary and (
            len(list(operator_policy_summary.get("unattended_allowed_benchmark_families", []))) < 5
            or not bool(operator_policy_summary.get("unattended_allow_git_commands", False))
            or not bool(operator_policy_summary.get("unattended_allow_http_requests", False))
            or not bool(operator_policy_summary.get("unattended_allow_generated_path_mutations", False))
        ):
            candidates.append(
                ImprovementExperiment(
                    subsystem="operator_policy",
                    reason="unattended operator-boundary policy still limits family breadth or critical execution scopes",
                    priority=4,
                    expected_gain=0.02,
                    estimated_cost=2,
                    score=0.0,
                    evidence=operator_policy_summary,
                )
            )
        capability_summary = self.capability_surface_summary()
        if capability_summary and (
            int(capability_summary.get("enabled_module_count", 0)) == 0
            or int(capability_summary.get("improvement_surface_count", 0)) == 0
        ):
            candidates.append(
                ImprovementExperiment(
                    subsystem="capabilities",
                    reason="capability registry does not yet expose retained autonomous improvement surfaces",
                    priority=3,
                    expected_gain=0.02,
                    estimated_cost=2,
                    score=0.0,
                    evidence=capability_summary,
                )
            )
        external_subsystems = {candidate.subsystem for candidate in candidates}
        for external in external_planner_experiments(self.capability_modules_path):
            subsystem = str(external.get("subsystem", "")).strip()
            if not subsystem or subsystem in external_subsystems:
                continue
            candidates.append(
                ImprovementExperiment(
                    subsystem=subsystem,
                    reason=str(external.get("reason", "")).strip() or "module-defined improvement surface",
                    priority=int(external.get("priority", 3)),
                    expected_gain=float(external.get("expected_gain", 0.01)),
                    estimated_cost=int(external.get("estimated_cost", 2)),
                    score=0.0,
                    evidence=dict(external.get("evidence", {}))
                    if isinstance(external.get("evidence", {}), dict)
                    else {},
                )
            )
            external_subsystems.add(subsystem)
        candidates.append(
            ImprovementExperiment(
                subsystem="policy",
                reason="no sharper subsystem deficit dominates current metrics",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.0,
                evidence={"default_fallback": True},
            )
        )
        planner_controls = self._improvement_planner_controls()
        scored_candidates = [
            self._score_experiment(candidate, metrics=metrics, planner_controls=planner_controls)
            for candidate in candidates
        ]
        scored_candidates.sort(key=lambda candidate: (-candidate.score, -candidate.priority, candidate.subsystem))
        return scored_candidates

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
        selected_surfaces = {self._base_subsystem(ranked[0].subsystem)}
        if max_candidates <= 1:
            return campaign
        top_score = ranked[0].score
        for candidate in ranked[1:]:
            if len(campaign) >= max_candidates:
                break
            if self._base_subsystem(candidate.subsystem) in selected_surfaces:
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
            selected_surfaces.add(self._base_subsystem(candidate.subsystem))
        return campaign

    def select_portfolio_campaign(
        self,
        metrics: EvalMetrics,
        *,
        max_candidates: int = 2,
        relative_score_floor: float = 0.75,
        absolute_score_margin: float = 0.04,
        recent_cycle_window: int = 6,
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
        top_score = float(ranked[0].score)
        recent_activity = {
            candidate.subsystem: self.recent_subsystem_activity_summary(
                subsystem=candidate.subsystem,
                recent_cycle_window=recent_cycle_window,
            )
            for candidate in ranked
        }
        lead_recent_activity = recent_activity.get(ranked[0].subsystem, {})
        breadth_pressure = self._campaign_breadth_pressure(lead_recent_activity)
        selected: list[ImprovementExperiment] = []
        selected_surfaces: set[str] = set()
        available = list(ranked)
        while available and len(selected) < max(1, max_candidates):
            best_candidate: ImprovementExperiment | None = None
            best_sort_key: tuple[float, int, str] = (float("-inf"), -1, "")
            for candidate in available:
                if self._base_subsystem(candidate.subsystem) in selected_surfaces:
                    continue
                relative_score = 0.0 if top_score <= 0.0 else float(candidate.score) / top_score
                score_margin = top_score - float(candidate.score)
                adjusted_score, reasons = self._portfolio_adjusted_experiment_score(
                    candidate,
                    recent_activity=recent_activity.get(candidate.subsystem, {}),
                    planner_controls=planner_controls,
                )
                if breadth_pressure > 0.0:
                    reasons.append(f"campaign_breadth_pressure={breadth_pressure:.4f}")
                if selected and relative_score < relative_score_floor and score_margin > absolute_score_margin:
                    continue
                sort_key = (adjusted_score, candidate.priority, candidate.subsystem)
                if best_candidate is None or sort_key > best_sort_key:
                    evidence = dict(candidate.evidence)
                    evidence["portfolio"] = {
                        "adjusted_score": round(adjusted_score, 4),
                        "relative_score": round(relative_score, 4),
                        "score_margin": round(score_margin, 4),
                        "campaign_breadth_pressure": round(breadth_pressure, 4),
                        "recent_activity": dict(recent_activity.get(candidate.subsystem, {})),
                        "reasons": reasons,
                    }
                    best_candidate = ImprovementExperiment(
                        subsystem=candidate.subsystem,
                        reason=candidate.reason,
                        priority=candidate.priority,
                        expected_gain=candidate.expected_gain,
                        estimated_cost=candidate.estimated_cost,
                        score=candidate.score,
                        evidence=evidence,
                    )
                    best_sort_key = sort_key
            if best_candidate is None:
                break
            selected.append(best_candidate)
            selected_surfaces.add(self._base_subsystem(best_candidate.subsystem))
            available = [
                candidate
                for candidate in available
                if self._base_subsystem(candidate.subsystem) != self._base_subsystem(best_candidate.subsystem)
            ]
        if selected:
            return selected
        return [ranked[0]]

    def rank_variants(self, experiment: ImprovementExperiment, metrics: EvalMetrics) -> list[ImprovementVariant]:
        planner_controls = self._improvement_planner_controls()
        variants = self._variants_for_experiment(experiment, metrics, planner_controls=planner_controls)
        scored_variants = [self._score_variant(experiment, variant, planner_controls=planner_controls) for variant in variants]
        return sorted(scored_variants, key=lambda variant: (-variant.score, variant.variant_id))

    def choose_variant(self, experiment: ImprovementExperiment, metrics: EvalMetrics) -> ImprovementVariant:
        return self.rank_variants(experiment, metrics)[0]

    def recommend_campaign_budget(
        self,
        metrics: EvalMetrics,
        *,
        max_width: int = 2,
    ) -> ImprovementSearchBudget:
        planner_controls = self._improvement_planner_controls()
        ranked = self.rank_experiments(metrics)
        resolved_max_width = max(1, max_width)
        if not ranked:
            return ImprovementSearchBudget(
                scope="campaign",
                width=1,
                max_width=resolved_max_width,
                strategy="adaptive_history",
                top_score=0.0,
                selected_ids=[],
                reasons=["no ranked experiments were available"],
            )
        top_score = float(ranked[0].score)
        selected_ids = [ranked[0].subsystem]
        selected_surfaces = {self._base_subsystem(ranked[0].subsystem)}
        reasons = [f"top subsystem {ranked[0].subsystem} score={top_score:.4f}"]
        if resolved_max_width <= 1 or len(ranked) == 1:
            return ImprovementSearchBudget(
                scope="campaign",
                width=1,
                max_width=resolved_max_width,
                strategy="adaptive_history",
                top_score=top_score,
                selected_ids=selected_ids,
                reasons=reasons,
            )

        close_relative_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="campaign",
            field="close_score_relative_threshold",
            legacy_field="campaign_close_score_relative_threshold",
            default=0.9,
            min_value=0.5,
            max_value=0.99,
        )
        close_margin_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="campaign",
            field="close_score_margin_threshold",
            legacy_field="campaign_close_score_margin_threshold",
            default=0.01,
            min_value=0.0,
            max_value=0.1,
        )
        history_relative_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="campaign",
            field="history_relative_threshold",
            legacy_field="campaign_history_relative_threshold",
            default=0.8,
            min_value=0.5,
            max_value=0.99,
        )
        candidate_relative_floor = min(close_relative_threshold, history_relative_threshold)
        for candidate in ranked[1:]:
            if len(selected_ids) >= resolved_max_width:
                break
            if self._base_subsystem(candidate.subsystem) in selected_surfaces:
                continue
            relative_score = 0.0 if top_score <= 0.0 else float(candidate.score) / top_score
            score_margin = top_score - float(candidate.score)
            if relative_score < candidate_relative_floor and score_margin > close_margin_threshold:
                continue
            selected_ids.append(candidate.subsystem)
            selected_surfaces.add(self._base_subsystem(candidate.subsystem))
            reasons.append(
                f"added {candidate.subsystem} due to scored breadth eligibility (relative={relative_score:.3f}, margin={score_margin:.4f})"
            )
        return ImprovementSearchBudget(
            scope="campaign",
            width=max(1, len(selected_ids)),
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=top_score,
            selected_ids=selected_ids,
            reasons=reasons,
        )

    def recommend_variant_budget(
        self,
        experiment: ImprovementExperiment,
        metrics: EvalMetrics,
        *,
        max_width: int = 2,
    ) -> ImprovementSearchBudget:
        planner_controls = self._improvement_planner_controls()
        ranked = self.rank_variants(experiment, metrics)
        resolved_max_width = max(1, max_width)
        if not ranked:
            return ImprovementSearchBudget(
                scope="variant",
                width=1,
                max_width=resolved_max_width,
                strategy="adaptive_history",
                top_score=0.0,
                selected_ids=[],
                reasons=[f"no ranked variants were available for subsystem={experiment.subsystem}"],
            )
        top_score = float(ranked[0].score)
        selected_ids = [ranked[0].variant_id]
        reasons = [f"top variant {ranked[0].variant_id} score={top_score:.4f}"]
        if resolved_max_width <= 1 or len(ranked) == 1:
            return ImprovementSearchBudget(
                scope="variant",
                width=1,
                max_width=resolved_max_width,
                strategy="adaptive_history",
                top_score=top_score,
                selected_ids=selected_ids,
                reasons=reasons,
            )

        close_relative_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="variant",
            field="close_score_relative_threshold",
            legacy_field="variant_close_score_relative_threshold",
            default=0.92,
            min_value=0.5,
            max_value=0.99,
        )
        close_margin_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="variant",
            field="close_score_margin_threshold",
            legacy_field="variant_close_score_margin_threshold",
            default=0.003,
            min_value=0.0,
            max_value=0.05,
        )
        history_relative_threshold = self._planner_guardrail_float(
            planner_controls,
            scope="variant",
            field="history_relative_threshold",
            legacy_field="variant_history_relative_threshold",
            default=0.85,
            min_value=0.5,
            max_value=0.99,
        )
        candidate_relative_floor = min(close_relative_threshold, history_relative_threshold)
        for variant in ranked[1:]:
            if len(selected_ids) >= resolved_max_width:
                break
            relative_score = 0.0 if top_score <= 0.0 else float(variant.score) / top_score
            score_margin = top_score - float(variant.score)
            if relative_score < candidate_relative_floor and score_margin > close_margin_threshold:
                continue
            selected_ids.append(variant.variant_id)
            reasons.append(
                f"added {variant.variant_id} due to scored breadth eligibility (relative={relative_score:.3f}, margin={score_margin:.4f})"
            )
        return ImprovementSearchBudget(
            scope="variant",
            width=max(1, len(selected_ids)),
            max_width=resolved_max_width,
            strategy="adaptive_history",
            top_score=top_score,
            selected_ids=selected_ids,
            reasons=reasons,
        )

    def failure_counts(self) -> dict[str, int]:
        if self.memory is None:
            return {}
        counts: dict[str, int] = {}
        for document in self.memory.list_documents():
            summary = document.get("summary", {})
            for failure_type in summary.get("failure_types", []):
                label = str(failure_type)
                counts[label] = counts.get(label, 0) + 1
            for failure_type in summary.get("transition_failures", []):
                label = str(failure_type)
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _failure_counts(self) -> dict[str, int]:
        return self.failure_counts()

    def transition_failure_counts(self) -> dict[str, int]:
        if self.memory is None:
            return {}
        counts: dict[str, int] = {}
        for document in self.memory.list_documents():
            summary = document.get("summary", {})
            for failure_type in summary.get("transition_failures", []):
                label = str(failure_type).strip()
                if label:
                    counts[label] = counts.get(label, 0) + 1
        return counts

    def transition_summary(self) -> dict[str, object]:
        if self.memory is None:
            return {}
        final_completion_ratios: list[float] = []
        net_progress_deltas: list[float] = []
        state_regression_steps = 0
        state_progress_gain_steps = 0
        for document in self.memory.list_documents():
            summary = document.get("summary", {})
            try:
                final_completion_ratios.append(float(summary.get("final_completion_ratio", 0.0)))
            except (TypeError, ValueError):
                pass
            try:
                net_progress_deltas.append(float(summary.get("net_state_progress_delta", 0.0)))
            except (TypeError, ValueError):
                pass
            try:
                state_regression_steps += int(summary.get("state_regression_steps", 0))
            except (TypeError, ValueError):
                pass
            try:
                state_progress_gain_steps += int(summary.get("state_progress_gain_steps", 0))
            except (TypeError, ValueError):
                pass
        average_completion = (
            round(sum(final_completion_ratios) / len(final_completion_ratios), 4)
            if final_completion_ratios
            else 0.0
        )
        average_progress_delta = (
            round(sum(net_progress_deltas) / len(net_progress_deltas), 4)
            if net_progress_deltas
            else 0.0
        )
        return {
            "average_final_completion_ratio": average_completion,
            "average_net_state_progress_delta": average_progress_delta,
            "state_regression_steps": state_regression_steps,
            "state_progress_gain_steps": state_progress_gain_steps,
        }

    def environment_violation_summary(self) -> dict[str, object]:
        if self.memory is None:
            return {
                "violation_counts": {},
                "alignment_failure_counts": {},
                "observed_environment_modes": {},
                "violation_total": 0,
                "alignment_failure_total": 0,
            }
        violation_counts: dict[str, int] = {}
        alignment_failure_counts: dict[str, int] = {}
        observed_environment_modes: dict[str, dict[str, int]] = {
            "network_access_mode": {},
            "git_write_mode": {},
            "workspace_write_scope": {},
        }
        for document in self.memory.list_documents():
            summary = document.get("summary", {})
            for label, value in dict(summary.get("environment_violation_counts", {})).items():
                key = str(label).strip()
                if not key:
                    continue
                try:
                    violation_counts[key] = violation_counts.get(key, 0) + int(value)
                except (TypeError, ValueError):
                    continue
            for label in summary.get("environment_alignment_failures", []):
                key = str(label).strip()
                if key:
                    alignment_failure_counts[key] = alignment_failure_counts.get(key, 0) + 1
            snapshot = summary.get("environment_snapshot", {})
            if isinstance(snapshot, dict):
                for field in observed_environment_modes:
                    value = str(snapshot.get(field, "")).strip().lower()
                    if value:
                        observed_environment_modes[field][value] = observed_environment_modes[field].get(value, 0) + 1
        return {
            "violation_counts": violation_counts,
            "alignment_failure_counts": alignment_failure_counts,
            "observed_environment_modes": observed_environment_modes,
            "violation_total": sum(violation_counts.values()),
            "alignment_failure_total": sum(alignment_failure_counts.values()),
        }

    def universe_cycle_feedback_summary(
        self,
        *,
        recent_cycle_window: int = 8,
        output_path: Path | None = None,
    ) -> dict[str, object]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return {
                "retained_cycle_count": 0,
                "selected_variant_counts": {},
                "selected_variant_weights": {},
                "successful_environment_assumptions": {},
                "successful_environment_assumption_weights": {},
                "successful_action_risk_control_floor": {},
                "successful_action_risk_control_weighted_mean": {},
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
                "broad_support_cycle_count": 0,
                "constitution_retained_cycle_count": 0,
                "operating_envelope_retained_cycle_count": 0,
            }
        decision_records = [
            record
            for record in self._decision_records(resolved)
            if self._subsystems_match(str(record.get("subsystem", "")), "universe")
            and str(record.get("state", "")).strip() == "retain"
        ]
        if not decision_records:
            return {
                "retained_cycle_count": 0,
                "selected_variant_counts": {},
                "selected_variant_weights": {},
                "successful_environment_assumptions": {},
                "successful_environment_assumption_weights": {},
                "successful_action_risk_control_floor": {},
                "successful_action_risk_control_weighted_mean": {},
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
                "broad_support_cycle_count": 0,
                "constitution_retained_cycle_count": 0,
                "operating_envelope_retained_cycle_count": 0,
            }
        recent_records = decision_records[-max(1, recent_cycle_window) :]
        selected_variant_counts: dict[str, int] = {}
        selected_variant_weights: dict[str, float] = {}
        successful_environment_assumptions: dict[str, dict[str, int]] = {
            "network_access_mode": {},
            "git_write_mode": {},
            "workspace_write_scope": {},
        }
        successful_environment_assumption_weights: dict[str, dict[str, float]] = {
            "network_access_mode": {},
            "git_write_mode": {},
            "workspace_write_scope": {},
        }
        successful_action_risk_control_floor: dict[str, int] = {}
        successful_action_risk_control_weighted_sum: dict[str, float] = {}
        successful_action_risk_control_weighted_weight: dict[str, float] = {}
        pass_rate_deltas: list[float] = []
        step_deltas: list[float] = []
        broad_support_cycle_count = 0
        constitution_retained_cycle_count = 0
        operating_envelope_retained_cycle_count = 0
        for record in recent_records:
            variant_id = _record_selected_variant_id(record)
            baseline_pass_rate = _record_float_value(record, "baseline_pass_rate")
            candidate_pass_rate = _record_float_value(record, "candidate_pass_rate")
            if baseline_pass_rate is not None and candidate_pass_rate is not None:
                pass_rate_deltas.append(candidate_pass_rate - baseline_pass_rate)
            baseline_average_steps = _record_float_value(record, "baseline_average_steps")
            candidate_average_steps = _record_float_value(record, "candidate_average_steps")
            if baseline_average_steps is not None and candidate_average_steps is not None:
                step_deltas.append(candidate_average_steps - baseline_average_steps)
            non_regressed_family_support = _record_non_regressed_family_support(record)
            support_discount = 0.5 if non_regressed_family_support <= 1 else 1.0
            pass_rate_delta = 0.0
            if baseline_pass_rate is not None and candidate_pass_rate is not None:
                pass_rate_delta = candidate_pass_rate - baseline_pass_rate
            step_gain = 0.0
            if baseline_average_steps is not None and candidate_average_steps is not None:
                step_gain = baseline_average_steps - candidate_average_steps
            outcome_weight = max(0.25, 1.0 + max(0.0, pass_rate_delta) * 20.0 + max(0.0, step_gain) * 5.0)
            weighted_support = round(outcome_weight * max(1, non_regressed_family_support) * support_discount, 4)
            if non_regressed_family_support >= 2:
                broad_support_cycle_count += 1
            if variant_id:
                selected_variant_counts[variant_id] = selected_variant_counts.get(variant_id, 0) + 1
                selected_variant_weights[variant_id] = round(
                    selected_variant_weights.get(variant_id, 0.0) + weighted_support,
                    4,
                )
            payload = self._load_retained_universe_payload_from_record(record)
            artifact_kind = str(payload.get("artifact_kind", "")).strip()
            if artifact_kind == "universe_constitution":
                constitution_retained_cycle_count += 1
            elif artifact_kind == "operating_envelope":
                operating_envelope_retained_cycle_count += 1
            environment_assumptions = retained_universe_environment_assumptions(payload)
            for field, counts in successful_environment_assumptions.items():
                value = str(environment_assumptions.get(field, "")).strip().lower()
                if value:
                    counts[value] = counts.get(value, 0) + 1
                    successful_environment_assumption_weights[field][value] = round(
                        successful_environment_assumption_weights[field].get(value, 0.0) + weighted_support,
                        4,
                    )
            action_risk_controls = retained_universe_action_risk_controls(payload)
            for key, value in action_risk_controls.items():
                successful_action_risk_control_floor[key] = max(
                    int(value),
                    int(successful_action_risk_control_floor.get(key, 0)),
                )
                successful_action_risk_control_weighted_sum[key] = (
                    successful_action_risk_control_weighted_sum.get(key, 0.0)
                    + (float(value) * weighted_support)
                )
                successful_action_risk_control_weighted_weight[key] = (
                    successful_action_risk_control_weighted_weight.get(key, 0.0) + weighted_support
                )
        return {
            "retained_cycle_count": len(recent_records),
            "selected_variant_counts": selected_variant_counts,
            "selected_variant_weights": selected_variant_weights,
            "successful_environment_assumptions": {
                field: _dominant_weight_label(successful_environment_assumption_weights.get(field, counts))
                for field, counts in successful_environment_assumptions.items()
                if _dominant_weight_label(successful_environment_assumption_weights.get(field, counts))
            },
            "successful_environment_assumption_weights": successful_environment_assumption_weights,
            "successful_action_risk_control_floor": successful_action_risk_control_floor,
            "successful_action_risk_control_weighted_mean": {
                key: round(successful_action_risk_control_weighted_sum[key] / successful_action_risk_control_weighted_weight[key], 3)
                for key in successful_action_risk_control_weighted_sum
                if successful_action_risk_control_weighted_weight.get(key, 0.0) > 0.0
            },
            "average_retained_pass_rate_delta": round(sum(pass_rate_deltas) / len(pass_rate_deltas), 4)
            if pass_rate_deltas
            else 0.0,
            "average_retained_step_delta": round(sum(step_deltas) / len(step_deltas), 4)
            if step_deltas
            else 0.0,
            "broad_support_cycle_count": broad_support_cycle_count,
            "constitution_retained_cycle_count": constitution_retained_cycle_count,
            "operating_envelope_retained_cycle_count": operating_envelope_retained_cycle_count,
        }

    def capability_surface_summary(self) -> dict[str, int]:
        if self.capability_modules_path is None:
            return {"module_count": 0, "enabled_module_count": 0, "external_capability_count": 0, "improvement_surface_count": 0}
        payload = {"modules": load_capability_modules(self.capability_modules_path)}
        return capability_surface_summary(payload)

    def trust_ledger_payload(self) -> dict[str, object]:
        path = self.trust_ledger_path
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def trust_ledger_summary(self) -> dict[str, object]:
        payload = self.trust_ledger_payload()
        if not payload:
            return {}
        overall = payload.get("overall_summary", {}) if isinstance(payload.get("overall_summary", {}), dict) else {}
        gated = payload.get("gated_summary", {}) if isinstance(payload.get("gated_summary", {}), dict) else {}
        assessment = payload.get("overall_assessment", {}) if isinstance(payload.get("overall_assessment", {}), dict) else {}
        return {
            "reports_considered": int(payload.get("reports_considered", 0) or 0),
            "overall_status": str(assessment.get("status", "")).strip(),
            "overall_passed": bool(assessment.get("passed", False)),
            "success_rate": float(gated.get("success_rate", overall.get("success_rate", 0.0)) or 0.0),
            "unsafe_ambiguous_rate": float(
                gated.get("unsafe_ambiguous_rate", overall.get("unsafe_ambiguous_rate", 0.0)) or 0.0
            ),
            "hidden_side_effect_risk_rate": float(
                gated.get("hidden_side_effect_risk_rate", overall.get("hidden_side_effect_risk_rate", 0.0)) or 0.0
            ),
            "rollback_performed_rate": float(
                gated.get("rollback_performed_rate", overall.get("rollback_performed_rate", 0.0)) or 0.0
            ),
            "success_hidden_side_effect_risk_rate": float(
                gated.get(
                    "success_hidden_side_effect_risk_rate",
                    overall.get("success_hidden_side_effect_risk_rate", 0.0),
                )
                or 0.0
            ),
            "distinct_benchmark_families": int(overall.get("distinct_benchmark_families", 0) or 0),
        }

    def delegation_policy_summary(self) -> dict[str, object]:
        snapshot = delegation_policy_snapshot(self.runtime_config or KernelConfig())
        return {key: int(value) for key, value in snapshot.items()}

    def operator_policy_summary(self) -> dict[str, object]:
        snapshot = operator_policy_snapshot(self.runtime_config or KernelConfig())
        return {
            "unattended_allowed_benchmark_families": list(snapshot.get("unattended_allowed_benchmark_families", [])),
            "unattended_allow_git_commands": bool(snapshot.get("unattended_allow_git_commands", False)),
            "unattended_allow_http_requests": bool(snapshot.get("unattended_allow_http_requests", False)),
            "unattended_http_allowed_hosts": list(snapshot.get("unattended_http_allowed_hosts", [])),
            "unattended_http_timeout_seconds": int(snapshot.get("unattended_http_timeout_seconds", 1)),
            "unattended_http_max_body_bytes": int(snapshot.get("unattended_http_max_body_bytes", 1)),
            "unattended_allow_generated_path_mutations": bool(
                snapshot.get("unattended_allow_generated_path_mutations", False)
            ),
            "unattended_generated_path_prefixes": list(snapshot.get("unattended_generated_path_prefixes", [])),
        }

    @staticmethod
    def _experiment_score(
        candidate: ImprovementExperiment,
        *,
        effective_subsystem: str | None = None,
    ) -> float:
        uncertainty_penalty = 0.0
        evidence = candidate.evidence
        subsystem = effective_subsystem or candidate.subsystem
        if subsystem == "retrieval":
            total = int(evidence.get("total", 0))
            low_confidence = int(evidence.get("low_confidence_episodes", 0))
            if total > 0:
                uncertainty_penalty += max(0.0, min(0.15, low_confidence / total)) * 0.1
        if subsystem == "benchmark" and int(evidence.get("benchmark_candidate_total", 0)) == 0:
            uncertainty_penalty += 0.02
        raw = (candidate.priority * candidate.expected_gain) / max(1, candidate.estimated_cost)
        return round(max(0.0, raw - uncertainty_penalty), 4)

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

    def _score_experiment(
        self,
        candidate: ImprovementExperiment,
        metrics: EvalMetrics,
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> ImprovementExperiment:
        resolved_planner_controls = planner_controls if planner_controls is not None else self._improvement_planner_controls()
        effective_subsystem = self._base_subsystem(candidate.subsystem)
        candidate, mutation_evidence = self._apply_improvement_planner_mutation(
            candidate,
            planner_controls=resolved_planner_controls,
        )
        history = self.subsystem_history_summary(subsystem=candidate.subsystem)
        recent_history = self.recent_subsystem_activity_summary(subsystem=candidate.subsystem)
        bootstrap_penalty, penalty_reasons = self._bootstrap_penalty(
            candidate,
            history,
            planner_controls=resolved_planner_controls,
            effective_subsystem=effective_subsystem,
        )
        cold_start_penalty, cold_start_reasons = self._cold_start_low_confidence_penalty(
            candidate,
            history,
            recent_history,
            planner_controls=resolved_planner_controls,
            effective_subsystem=effective_subsystem,
        )
        stalled_selection_penalty, stalled_selection_reasons = self._recent_stalled_selection_penalty(
            recent_history,
            planner_controls=resolved_planner_controls,
        )
        observation_timeout_penalty, observation_timeout_reasons = self._recent_observation_timeout_penalty(
            recent_history,
            planner_controls=resolved_planner_controls,
        )
        promotion_failure_penalty, promotion_failure_reasons = self._recent_promotion_failure_penalty(
            recent_history,
            planner_controls=resolved_planner_controls,
        )
        benchmark_no_yield_penalty = 0.0
        benchmark_no_yield_reasons: list[str] = []
        if effective_subsystem == "benchmark":
            selected_cycles = int(recent_history.get("selected_cycles", 0) or 0)
            retained_cycles = int(recent_history.get("retained_cycles", 0) or 0)
            no_yield_cycles = int(recent_history.get("no_yield_cycles", 0) or 0)
            if selected_cycles >= 2 and retained_cycles == 0 and no_yield_cycles > 0:
                penalty_per_cycle = self._planner_control_float(
                    resolved_planner_controls,
                    "benchmark_recent_no_yield_penalty_per_cycle",
                    0.03,
                    min_value=0.0,
                    max_value=0.2,
                )
                penalty_cap = self._planner_control_float(
                    resolved_planner_controls,
                    "benchmark_recent_no_yield_penalty_cap",
                    0.2,
                    min_value=0.0,
                    max_value=0.5,
                )
                benchmark_no_yield_penalty = min(penalty_cap, no_yield_cycles * penalty_per_cycle)
                benchmark_no_yield_penalty = round(benchmark_no_yield_penalty, 4)
                if benchmark_no_yield_penalty > 0.0:
                    benchmark_no_yield_reasons.append(
                        f"benchmark_recent_no_yield_penalty={benchmark_no_yield_penalty:.4f}"
                    )
        memory_source_bonus, memory_source_evidence = self._memory_source_experiment_bonus(
            effective_subsystem,
            metrics,
        )
        score_bias = self._planner_control_subsystem_float(
            resolved_planner_controls,
            "subsystem_score_bias",
            candidate.subsystem,
            fallback_subsystem=effective_subsystem,
            default=0.0,
            min_value=-0.1,
            max_value=0.1,
        )
        score = round(
            max(
                0.0,
                self._experiment_score(candidate, effective_subsystem=effective_subsystem)
                + self._history_bonus(history)
                + self._recent_history_bonus(recent_history)
                + memory_source_bonus
                - bootstrap_penalty
                - cold_start_penalty
                - stalled_selection_penalty
                - observation_timeout_penalty
                - promotion_failure_penalty
                - benchmark_no_yield_penalty
                + score_bias,
            ),
            4,
        )
        evidence = dict(candidate.evidence)
        evidence["base_subsystem"] = effective_subsystem
        evidence["history"] = history
        evidence["recent_history"] = recent_history
        if memory_source_evidence:
            evidence["memory_source_pressure"] = memory_source_evidence
        if mutation_evidence:
            evidence["improvement_planner_mutation"] = mutation_evidence
        selection_penalties = [
            *penalty_reasons,
            *cold_start_reasons,
            *stalled_selection_reasons,
            *observation_timeout_reasons,
            *promotion_failure_reasons,
            *benchmark_no_yield_reasons,
        ]
        if selection_penalties:
            evidence["selection_penalties"] = selection_penalties
        return ImprovementExperiment(
            subsystem=candidate.subsystem,
            reason=candidate.reason,
            priority=candidate.priority,
            expected_gain=candidate.expected_gain,
            estimated_cost=candidate.estimated_cost,
            score=score,
            evidence=evidence,
        )

    @staticmethod
    def _memory_source_focus_summary(metrics: EvalMetrics) -> dict[str, dict[str, object]]:
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

        for source, row in summary.items():
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

    @classmethod
    def _memory_source_experiment_bonus(
        cls,
        subsystem: str,
        metrics: EvalMetrics,
    ) -> tuple[float, dict[str, object]]:
        source_summary = cls._memory_source_focus_summary(metrics)
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

    @staticmethod
    def _bootstrap_penalty(
        candidate: ImprovementExperiment,
        history: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
        effective_subsystem: str | None = None,
    ) -> tuple[float, list[str]]:
        total_decisions = int(history.get("total_decisions", 0))
        retained_cycles = int(history.get("retained_cycles", 0))
        rejected_cycles = int(history.get("rejected_cycles", 0))
        if total_decisions < 2 or retained_cycles > 0 or rejected_cycles < 2:
            return 0.0, []
        evidence = candidate.evidence
        subsystem = effective_subsystem or candidate.subsystem
        multiplier = ImprovementPlanner._planner_control_subsystem_float(
            planner_controls or {},
            "bootstrap_penalty_multiplier",
            candidate.subsystem,
            fallback_subsystem=subsystem,
            default=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        if subsystem == "benchmark" and int(evidence.get("benchmark_candidate_total", 0)) == 0:
            penalty = round(0.04 * multiplier, 4)
            return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
        if subsystem == "verifier" and int(evidence.get("verifier_memory_total", 0)) == 0:
            penalty = round(0.04 * multiplier, 4)
            return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
        if subsystem == "operators" and int(evidence.get("skill_transfer_total", 0)) == 0:
            penalty = round(0.035 * multiplier, 4)
            return penalty, [f"bootstrap_no_yield_penalty={penalty:.4f}"]
        return 0.0, []

    @staticmethod
    def _cold_start_low_confidence_penalty(
        candidate: ImprovementExperiment,
        history: dict[str, object],
        recent_history: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
        effective_subsystem: str | None = None,
    ) -> tuple[float, list[str]]:
        subsystem = effective_subsystem or candidate.subsystem
        if subsystem not in {"retrieval", "tolbert_model"}:
            return 0.0, []
        if int(history.get("total_decisions", 0) or 0) > 0:
            return 0.0, []
        if int(recent_history.get("selected_cycles", 0) or 0) > 0:
            return 0.0, []
        evidence = candidate.evidence if isinstance(candidate.evidence, dict) else {}
        total = int(evidence.get("total", 0) or 0)
        low_confidence = int(evidence.get("low_confidence_episodes", 0) or 0)
        if total <= 0 or low_confidence <= 0:
            return 0.0, []
        raw_score = ImprovementPlanner._experiment_score(candidate, effective_subsystem=subsystem)
        default_cap = 0.12 if subsystem == "retrieval" else 0.1
        cap = ImprovementPlanner._planner_control_subsystem_float(
            planner_controls or {},
            "cold_start_low_confidence_score_cap",
            candidate.subsystem,
            fallback_subsystem=subsystem,
            default=default_cap,
            min_value=0.0,
            max_value=0.25,
        )
        if raw_score <= cap:
            return 0.0, []
        penalty = round(raw_score - cap, 4)
        return penalty, [f"cold_start_low_confidence_penalty={penalty:.4f}"]

    @staticmethod
    def _recent_stalled_selection_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
        retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
        no_yield_cycles = int(recent_activity.get("no_yield_cycles", 0) or 0)
        recent_incomplete_cycles = int(recent_activity.get("recent_incomplete_cycles", 0) or 0)
        recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0) or 0)
        if selected_cycles < 2 or retained_cycles > 0 or no_yield_cycles <= 0:
            return 0.0, []
        resolved_planner_controls = planner_controls or {}
        penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_stalled_selection_penalty_per_cycle",
            0.015,
            min_value=0.0,
            max_value=0.05,
        )
        incomplete_bonus_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_stalled_incomplete_bonus_per_cycle",
            0.01,
            min_value=0.0,
            max_value=0.04,
        )
        repeated_phase_gate_bonus_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_stalled_phase_gate_bonus_per_cycle",
            0.005,
            min_value=0.0,
            max_value=0.03,
        )
        penalty = min(
            0.08,
            (no_yield_cycles * penalty_per_cycle)
            + (recent_incomplete_cycles * incomplete_bonus_per_cycle)
            + (recent_phase_gate_failure_cycles * repeated_phase_gate_bonus_per_cycle),
        )
        if penalty <= 0.0:
            return 0.0, []
        return round(penalty, 4), [f"recent_stalled_selection_penalty={penalty:.4f}"]

    @staticmethod
    def _recent_observation_timeout_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
        retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
        observation_timeout_cycles = int(recent_activity.get("recent_observation_timeout_cycles", 0) or 0)
        budgeted_timeout_cycles = int(recent_activity.get("recent_budgeted_observation_timeout_cycles", 0) or 0)
        repeated_timeout_budget_source_count = int(
            recent_activity.get("repeated_observation_timeout_budget_source_count", 0) or 0
        )
        if selected_cycles <= 0 or retained_cycles > 0 or observation_timeout_cycles <= 0:
            return 0.0, []
        resolved_planner_controls = planner_controls or {}
        penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_observation_timeout_penalty_per_cycle",
            0.0125,
            min_value=0.0,
            max_value=0.05,
        )
        budgeted_bonus_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_budgeted_observation_timeout_bonus_per_cycle",
            0.005,
            min_value=0.0,
            max_value=0.03,
        )
        repeated_source_bonus_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_repeated_observation_timeout_source_bonus_per_cycle",
            0.003,
            min_value=0.0,
            max_value=0.02,
        )
        penalty = min(
            0.08,
            (observation_timeout_cycles * penalty_per_cycle)
            + (budgeted_timeout_cycles * budgeted_bonus_per_cycle)
            + (max(0, repeated_timeout_budget_source_count - 1) * repeated_source_bonus_per_cycle),
        )
        if penalty <= 0.0:
            return 0.0, []
        return round(penalty, 4), [f"recent_observation_timeout_penalty={penalty:.4f}"]

    @staticmethod
    def _recent_promotion_failure_penalty(
        recent_activity: dict[str, object],
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        selected_cycles = int(recent_activity.get("selected_cycles", 0) or 0)
        rejected_cycles = int(recent_activity.get("rejected_cycles", 0) or 0)
        retained_cycles = int(recent_activity.get("retained_cycles", 0) or 0)
        recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0) or 0)
        rejected_pass_rate_delta = max(
            0.0,
            -float(recent_activity.get("average_rejected_pass_rate_delta", 0.0) or 0.0),
        )
        if selected_cycles <= 0 or rejected_cycles <= 0 or retained_cycles > rejected_cycles:
            return 0.0, []
        resolved_planner_controls = planner_controls or {}
        rejected_cycle_penalty = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_promotion_reject_penalty_per_cycle",
            0.014,
            min_value=0.0,
            max_value=0.05,
        )
        rejected_delta_weight = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_promotion_reject_pass_rate_delta_weight",
            0.35,
            min_value=0.0,
            max_value=1.0,
        )
        phase_gate_penalty = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "recent_promotion_phase_gate_penalty_per_cycle",
            0.008,
            min_value=0.0,
            max_value=0.03,
        )
        penalty = min(
            0.1,
            (rejected_cycles * rejected_cycle_penalty)
            + (rejected_pass_rate_delta * rejected_delta_weight)
            + (recent_phase_gate_failure_cycles * phase_gate_penalty),
        )
        if penalty <= 0.0:
            return 0.0, []
        return round(penalty, 4), [f"recent_promotion_failure_penalty={penalty:.4f}"]

    def _score_variant(
        self,
        experiment: ImprovementExperiment,
        variant: ImprovementVariant,
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> ImprovementVariant:
        resolved_planner_controls = planner_controls if planner_controls is not None else self._improvement_planner_controls()
        effective_subsystem = self._base_subsystem(variant.subsystem)
        variant, mutation_evidence = self._apply_variant_planner_mutation(
            variant,
            planner_controls=resolved_planner_controls,
        )
        subsystem_history = self.subsystem_history_summary(subsystem=experiment.subsystem)
        variant_history = self.variant_history_summary(
            subsystem=experiment.subsystem,
            variant_id=variant.variant_id,
        )
        variant_recent_history = self.recent_variant_activity_summary(
            subsystem=experiment.subsystem,
            variant_id=variant.variant_id,
        )
        score_bias = self._planner_control_variant_float(
            resolved_planner_controls,
            "variant_score_bias",
            variant.subsystem,
            variant.variant_id,
            fallback_subsystem=effective_subsystem,
            default=0.0,
            min_value=-0.05,
            max_value=0.05,
        )
        exploration_bonus = self._variant_exploration_bonus(
            subsystem_history=subsystem_history,
            variant_history=variant_history,
            variant_recent_history=variant_recent_history,
            planner_controls=resolved_planner_controls,
        )
        score = round(
            max(
                0.0,
                self._variant_score(variant)
                + (self._history_bonus(subsystem_history) * 0.35)
                + self._history_bonus(variant_history, variant_specific=True)
                + self._recent_history_bonus(variant_recent_history, variant_specific=True)
                + exploration_bonus
                + score_bias,
            ),
            4,
        )
        controls = dict(variant.controls)
        controls["base_subsystem"] = effective_subsystem
        controls["history"] = {
            "subsystem": subsystem_history,
            "variant": variant_history,
            "recent_variant": variant_recent_history,
        }
        if mutation_evidence:
            controls["planner_mutation"] = mutation_evidence
        if exploration_bonus > 0.0:
            controls["variant_exploration_bonus"] = round(exploration_bonus, 4)
        return ImprovementVariant(
            subsystem=variant.subsystem,
            variant_id=variant.variant_id,
            description=variant.description,
            expected_gain=variant.expected_gain,
            estimated_cost=variant.estimated_cost,
            score=score,
            controls=controls,
        )

    def _variants_for_experiment(
        self,
        experiment: ImprovementExperiment,
        metrics: EvalMetrics,
        *,
        planner_controls: dict[str, object] | None = None,
    ) -> list[ImprovementVariant]:
        subsystem = experiment.subsystem
        variants = [
            self._variant(
                subsystem,
                str(variant["variant_id"]),
                str(variant["description"]),
                float(variant["expected_gain"]),
                int(variant["estimated_cost"]),
                dict(variant["controls"]),
            )
            for variant in default_variant_definitions(
                subsystem,
                experiment,
                metrics,
                capability_modules_path=self.capability_modules_path,
            )
        ]
        return self._with_variant_expansions(variants, planner_controls=planner_controls or {})

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

    def append_cycle_record(self, output_path: Path, record: ImprovementCycleRecord) -> Path:
        record = _normalized_cycle_record(record)
        _validate_cycle_record_consistency(record)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.runtime_config is not None and self.runtime_config.uses_sqlite_storage():
            self.runtime_config.sqlite_store().append_cycle_record(
                output_path=output_path,
                payload=record.to_dict(),
            )
            if self.runtime_config.storage_write_cycle_exports:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            return output_path
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return output_path

    def load_cycle_records(self, output_path: Path) -> list[dict[str, object]]:
        if self.runtime_config is not None and self.runtime_config.uses_sqlite_storage():
            records = self.runtime_config.sqlite_store().load_cycle_records(output_path=output_path)
            if records:
                return records
        if not output_path.exists():
            return []
        records: list[dict[str, object]] = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    def cycle_history(
        self,
        output_path: Path,
        *,
        cycle_id: str | None = None,
        subsystem: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, object]]:
        records = self.load_cycle_records(output_path)
        if cycle_id is not None:
            records = [record for record in records if str(record.get("cycle_id", "")) == cycle_id]
        if subsystem is not None:
            records = [record for record in records if str(record.get("subsystem", "")) == subsystem]
        if state is not None:
            records = [record for record in records if str(record.get("state", "")) == state]
        return records

    def cycle_audit_summary(self, output_path: Path, *, cycle_id: str) -> dict[str, object] | None:
        records = self.cycle_history(output_path, cycle_id=cycle_id)
        if not records:
            return None
        latest = records[-1]
        selected_variant_id = ""
        prior_retained_cycle_id = ""
        candidate_artifact_path = ""
        active_artifact_path = ""
        decision_record: dict[str, object] | None = None
        for record in records:
            if not selected_variant_id:
                selected_variant_id = _record_selected_variant_id(record)
            if not prior_retained_cycle_id:
                prior_retained_cycle_id = _record_prior_retained_cycle_id(record)
            if not candidate_artifact_path:
                candidate_artifact_path = str(record.get("candidate_artifact_path", "")).strip()
            if not active_artifact_path:
                active_artifact_path = str(record.get("active_artifact_path", "")).strip()
            if str(record.get("state", "")).strip() in {"retain", "reject"}:
                decision_record = record
        decision_record = decision_record or latest
        return {
            "cycle_id": cycle_id,
            "subsystem": str(latest.get("subsystem", "")).strip(),
            "record_count": len(records),
            "states": [str(record.get("state", "")).strip() for record in records],
            "selected_variant_id": selected_variant_id,
            "prior_retained_cycle_id": prior_retained_cycle_id,
            "candidate_artifact_path": candidate_artifact_path,
            "active_artifact_path": active_artifact_path,
            "final_state": str(decision_record.get("state", "")).strip(),
            "final_reason": str(decision_record.get("reason", "")).strip(),
            "baseline_pass_rate": _record_float_value(decision_record, "baseline_pass_rate"),
            "candidate_pass_rate": _record_float_value(decision_record, "candidate_pass_rate"),
            "baseline_average_steps": _record_float_value(decision_record, "baseline_average_steps"),
            "candidate_average_steps": _record_float_value(decision_record, "candidate_average_steps"),
            "phase_gate_passed": _record_phase_gate_passed(decision_record),
            "artifact_kind": str(decision_record.get("artifact_kind", "")).strip(),
            "artifact_path": str(decision_record.get("artifact_path", "")).strip(),
            "artifact_lifecycle_state": str(decision_record.get("artifact_lifecycle_state", "")).strip(),
            "artifact_sha256": str(decision_record.get("artifact_sha256", "")).strip(),
            "previous_artifact_sha256": str(decision_record.get("previous_artifact_sha256", "")).strip(),
            "rollback_artifact_path": str(decision_record.get("rollback_artifact_path", "")).strip(),
            "artifact_snapshot_path": str(decision_record.get("artifact_snapshot_path", "")).strip(),
        }

    def incomplete_cycle_summaries(
        self,
        output_path: Path | None = None,
        *,
        protocol: str | None = None,
        before_cycle_id: str | None = None,
    ) -> list[dict[str, object]]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return []
        records = self.load_cycle_records(resolved)
        if not records:
            return []
        if before_cycle_id is not None:
            cutoff_index = next(
                (
                    index
                    for index, record in enumerate(records)
                    if str(record.get("cycle_id", "")) == before_cycle_id
                ),
                None,
            )
            if cutoff_index is not None:
                records = records[:cutoff_index]
        grouped: dict[str, list[dict[str, object]]] = {}
        for record in records:
            cycle_id = str(record.get("cycle_id", "")).strip()
            if not cycle_id:
                continue
            grouped.setdefault(cycle_id, []).append(record)
        summaries: list[dict[str, object]] = []
        for cycle_id, cycle_records in grouped.items():
            states = {str(record.get("state", "")).strip() for record in cycle_records}
            if states & {"retain", "reject"}:
                continue
            if not states & {"select", "generate", "evaluate"}:
                continue
            observe_record = next(
                (record for record in cycle_records if str(record.get("state", "")) == "observe"),
                None,
            )
            select_record = next(
                (record for record in cycle_records if str(record.get("state", "")) == "select"),
                None,
            )
            generate_record = next(
                (record for record in reversed(cycle_records) if str(record.get("state", "")) == "generate"),
                None,
            )
            latest_record = cycle_records[-1]
            metrics_sources = [
                record.get("metrics_summary", {})
                for record in (observe_record, select_record, generate_record, latest_record)
                if isinstance(record, dict)
            ]
            protocol_value = ""
            for metrics_summary in metrics_sources:
                if not isinstance(metrics_summary, dict):
                    continue
                token = str(metrics_summary.get("protocol", "")).strip()
                if token:
                    protocol_value = token
                    break
            if protocol is not None and protocol_value != protocol:
                continue
            metrics_summary = latest_record.get("metrics_summary", {})
            if not isinstance(metrics_summary, dict):
                metrics_summary = {}
            candidate_artifact_path = ""
            active_artifact_path = ""
            artifact_kind = str(latest_record.get("artifact_kind", "")).strip()
            artifact_path = str(latest_record.get("artifact_path", "")).strip()
            for record in reversed(cycle_records):
                candidate_value = str(record.get("candidate_artifact_path", "")).strip()
                if candidate_value and not candidate_artifact_path:
                    candidate_artifact_path = candidate_value
                active_value = str(record.get("active_artifact_path", "")).strip()
                if active_value and not active_artifact_path:
                    active_artifact_path = active_value
                artifact_value = str(record.get("artifact_path", "")).strip()
                if artifact_value and not artifact_path:
                    artifact_path = artifact_value
                kind_value = str(record.get("artifact_kind", "")).strip()
                if kind_value and not artifact_kind:
                    artifact_kind = kind_value
            summaries.append(
                {
                    "cycle_id": cycle_id,
                    "subsystem": str(latest_record.get("subsystem", "")).strip(),
                    "protocol": protocol_value,
                    "last_state": str(latest_record.get("state", "")).strip(),
                    "last_action": str(latest_record.get("action", "")).strip(),
                    "reason": str(latest_record.get("reason", "")).strip(),
                    "artifact_kind": artifact_kind,
                    "artifact_path": artifact_path,
                    "active_artifact_path": active_artifact_path,
                    "candidate_artifact_path": candidate_artifact_path,
                    "selected_variant_id": _record_selected_variant_id(latest_record)
                    or _record_selected_variant_id(select_record or {})
                    or _record_selected_variant_id(generate_record or {}),
                    "prior_retained_cycle_id": _record_prior_retained_cycle_id(latest_record)
                    or _record_prior_retained_cycle_id(generate_record or {}),
                    "selected_cycles": len(
                        {
                            str(record.get("cycle_id", ""))
                            for record in cycle_records
                            if str(record.get("state", "")) == "select"
                        }
                    ),
                    "record_count": len(cycle_records),
                }
            )
        return summaries

    def artifact_history(self, output_path: Path, artifact_path: Path | str) -> list[dict[str, object]]:
        normalized = str(artifact_path)
        return [
            record
            for record in self.load_cycle_records(output_path)
            if str(record.get("artifact_path", "")) == normalized
        ]

    def latest_artifact_record(self, output_path: Path, artifact_path: Path | str) -> dict[str, object] | None:
        history = self.artifact_history(output_path, artifact_path)
        if not history:
            return None
        return history[-1]

    def latest_artifact_decision(self, output_path: Path, artifact_path: Path | str) -> dict[str, object] | None:
        history = [
            record
            for record in self.artifact_history(output_path, artifact_path)
            if str(record.get("state", "")) in {"retain", "reject"}
        ]
        if not history:
            return None
        return history[-1]

    def artifact_rollback_metadata(self, output_path: Path, artifact_path: Path | str) -> dict[str, str]:
        latest = self.latest_artifact_record(output_path, artifact_path)
        if latest is None:
            return {}
        return {
            "artifact_sha256": str(latest.get("artifact_sha256", "")),
            "previous_artifact_sha256": str(latest.get("previous_artifact_sha256", "")),
            "rollback_artifact_path": str(latest.get("rollback_artifact_path", "")),
            "artifact_snapshot_path": str(latest.get("artifact_snapshot_path", "")),
        }

    def prior_retained_artifact_record(
        self,
        output_path: Path,
        subsystem: str,
        *,
        before_cycle_id: str | None = None,
    ) -> dict[str, object] | None:
        records = [
            record
            for record in self.load_cycle_records(output_path)
            if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
            and str(record.get("state", "")) == "retain"
        ]
        if before_cycle_id is not None:
            all_records = self.load_cycle_records(output_path)
            cutoff_index = next(
                (
                    index
                    for index, record in enumerate(all_records)
                    if str(record.get("cycle_id", "")) == before_cycle_id
                ),
                None,
            )
            if cutoff_index is not None:
                eligible_cycle_ids = {
                    str(record.get("cycle_id", ""))
                    for record in all_records[:cutoff_index]
                    if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
                    and str(record.get("state", "")) == "retain"
                }
                records = [
                    record for record in records if str(record.get("cycle_id", "")) in eligible_cycle_ids
                ]
        if not records:
            return None
        return records[-1]

    def rollback_artifact(self, output_path: Path, artifact_path: Path | str) -> Path:
        latest = self.latest_artifact_record(output_path, artifact_path)
        if latest is None:
            raise ValueError("no cycle history exists for the requested artifact")
        snapshot_path = Path(str(latest.get("rollback_artifact_path", "")))
        if not snapshot_path.exists():
            raise FileNotFoundError(f"rollback snapshot does not exist: {snapshot_path}")
        destination = Path(str(artifact_path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_path, destination)
        subsystem = str(latest.get("subsystem", "")).strip()
        if subsystem in {"universe", "universe_constitution", "operating_envelope"}:
            restored_payload = _load_json_payload(destination)
            _synchronize_retained_universe_artifacts(
                subsystem=subsystem,
                payload=restored_payload,
                live_artifact_path=destination,
                runtime_config=_runtime_config_for_universe_sync(self.runtime_config, destination),
            )
        return destination

    def retained_gain_summary(
        self,
        output_path: Path,
        *,
        subsystem: str | None = None,
    ) -> ImprovementYieldSummary:
        decisions = [
            record
            for record in self.load_cycle_records(output_path)
            if str(record.get("state", "")) in {"retain", "reject"}
        ]
        if subsystem is not None:
            decisions = [
                record
                for record in decisions
                if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
            ]

        retained = [record for record in decisions if str(record.get("state", "")) == "retain"]
        rejected = [record for record in decisions if str(record.get("state", "")) == "reject"]

        retained_by_subsystem: dict[str, int] = {}
        rejected_by_subsystem: dict[str, int] = {}
        for record in retained:
            key = str(record.get("subsystem", "unknown"))
            retained_by_subsystem[key] = retained_by_subsystem.get(key, 0) + 1
        for record in rejected:
            key = str(record.get("subsystem", "unknown"))
            rejected_by_subsystem[key] = rejected_by_subsystem.get(key, 0) + 1

        return ImprovementYieldSummary(
            retained_cycles=len(retained),
            rejected_cycles=len(rejected),
            total_decisions=len(decisions),
            retained_by_subsystem=retained_by_subsystem,
            rejected_by_subsystem=rejected_by_subsystem,
            average_retained_pass_rate_delta=_average_metric_delta(
                retained,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            average_retained_step_delta=_average_metric_delta(
                retained,
                baseline_key="baseline_average_steps",
                candidate_key="candidate_average_steps",
            ),
            average_rejected_pass_rate_delta=_average_metric_delta(
                rejected,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            average_rejected_step_delta=_average_metric_delta(
                rejected,
                baseline_key="baseline_average_steps",
                candidate_key="candidate_average_steps",
            ),
        )

    def subsystem_history_summary(
        self,
        *,
        subsystem: str,
        output_path: Path | None = None,
    ) -> dict[str, object]:
        decisions = [
            record
            for record in self._decision_records(output_path)
            if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
        ]
        return self._decision_summary(decisions)

    def variant_history_summary(
        self,
        *,
        subsystem: str,
        variant_id: str,
        output_path: Path | None = None,
    ) -> dict[str, object]:
        cycle_variants = self._cycle_variant_index(output_path)
        decisions = [
            record
            for record in self._decision_records(output_path)
            if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
            and cycle_variants.get(str(record.get("cycle_id", ""))) == variant_id
        ]
        return self._decision_summary(decisions)

    def recent_subsystem_activity_summary(
        self,
        *,
        subsystem: str,
        recent_cycle_window: int = 6,
        output_path: Path | None = None,
    ) -> dict[str, object]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return self._empty_recent_activity_summary(recent_cycle_window)
        records = self.load_cycle_records(resolved)
        if not records:
            return self._empty_recent_activity_summary(recent_cycle_window)
        recent_cycle_ids: list[str] = []
        seen_cycle_ids: set[str] = set()
        for record in reversed(records):
            cycle_id = str(record.get("cycle_id", "")).strip()
            if not cycle_id or cycle_id in seen_cycle_ids:
                continue
            seen_cycle_ids.add(cycle_id)
            recent_cycle_ids.append(cycle_id)
            if len(recent_cycle_ids) >= max(1, recent_cycle_window):
                break
        recent_cycle_id_set = set(recent_cycle_ids)
        relevant = [
            record
            for record in records
            if str(record.get("cycle_id", "")) in recent_cycle_id_set
            and self._subsystems_match(str(record.get("subsystem", "")), subsystem)
        ]
        selected_cycle_ids = {
            str(record.get("cycle_id", ""))
            for record in relevant
            if str(record.get("state", "")) == "select"
        }
        latest_by_cycle: dict[str, dict[str, object]] = {}
        for record in relevant:
            cycle_id = str(record.get("cycle_id", "")).strip()
            if cycle_id:
                latest_by_cycle[cycle_id] = record
        decision_records = [
            record for record in relevant if str(record.get("state", "")) in {"retain", "reject"}
        ]
        retained = [record for record in decision_records if str(record.get("state", "")) == "retain"]
        rejected = [record for record in decision_records if str(record.get("state", "")) == "reject"]
        last_decision = decision_records[-1] if decision_records else None
        summary = {
            "recent_cycle_window": max(1, recent_cycle_window),
            "selected_cycles": len(selected_cycle_ids),
            "retained_cycles": len(retained),
            "rejected_cycles": len(rejected),
            "last_decision_state": "" if last_decision is None else str(last_decision.get("state", "")),
            "last_cycle_id": "" if last_decision is None else str(last_decision.get("cycle_id", "")),
            "average_retained_pass_rate_delta": _average_metric_delta(
                retained,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            "average_rejected_pass_rate_delta": _average_metric_delta(
                rejected,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
        }
        total_decisions = len(decision_records)
        selected_cycles = len(selected_cycle_ids)
        summary["no_yield_cycles"] = max(0, selected_cycles - total_decisions)
        decision_cycle_ids = {str(record.get("cycle_id", "")) for record in decision_records}
        summary["recent_incomplete_cycles"] = len(
            {
                cycle_id
                for cycle_id in selected_cycle_ids
                if cycle_id not in decision_cycle_ids
                and str(latest_by_cycle.get(cycle_id, {}).get("state", "")).strip() in {"generate", "evaluate"}
            }
        )
        summary["total_decisions"] = total_decisions
        summary["retention_rate"] = round(0.0 if total_decisions == 0 else len(retained) / total_decisions, 4)
        summary["rejection_rate"] = round(0.0 if total_decisions == 0 else len(rejected) / total_decisions, 4)
        summary["recent_retention_rate"] = round(
            0.0 if selected_cycles == 0 else len(retained) / selected_cycles,
            4,
        )
        summary["recent_rejection_rate"] = round(
            0.0 if selected_cycles == 0 else len(rejected) / selected_cycles,
            4,
        )
        summary["recent_regression_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in relevant
                if self._record_has_regression_signal(record)
            }
        )
        summary["recent_phase_gate_failure_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in relevant
                if self._record_has_phase_gate_failure(record)
            }
        )
        phase_gate_failure_reasons = [
            reason
            for record in relevant
            for reason in _record_phase_gate_failures(record)
        ]
        last_phase_gate_failure_reason = phase_gate_failure_reasons[-1] if phase_gate_failure_reasons else ""
        repeated_phase_gate_reason_count = 0
        if last_phase_gate_failure_reason:
            repeated_phase_gate_reason_count = sum(
                1 for reason in phase_gate_failure_reasons if reason == last_phase_gate_failure_reason
            )
        summary["last_phase_gate_failure_reason"] = last_phase_gate_failure_reason
        summary["repeated_phase_gate_reason_count"] = repeated_phase_gate_reason_count
        summary["recent_reconciled_failure_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in decision_records
                if self._record_is_reconciled_failure(record)
            }
        )
        observation_timeout_records = [
            record for record in relevant if self._record_has_observation_timeout(record)
        ]
        summary["recent_observation_timeout_cycles"] = len(
            {str(record.get("cycle_id", "")) for record in observation_timeout_records}
        )
        timeout_budget_sources = [
            source
            for record in observation_timeout_records
            if (source := self._record_observation_timeout_budget_source(record))
        ]
        summary["recent_budgeted_observation_timeout_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in observation_timeout_records
                if self._record_observation_timeout_budget_source(record)
            }
        )
        last_observation_timeout_budget_source = timeout_budget_sources[-1] if timeout_budget_sources else ""
        repeated_observation_timeout_budget_source_count = 0
        if last_observation_timeout_budget_source:
            repeated_observation_timeout_budget_source_count = sum(
                1 for source in timeout_budget_sources if source == last_observation_timeout_budget_source
            )
        summary["last_observation_timeout_budget_source"] = last_observation_timeout_budget_source
        summary["repeated_observation_timeout_budget_source_count"] = (
            repeated_observation_timeout_budget_source_count
        )
        summary["net_retained_cycle_advantage"] = len(retained) - len(rejected)
        summary["decision_quality_score"] = self._decision_quality_score(summary)
        return summary

    def recent_variant_activity_summary(
        self,
        *,
        subsystem: str,
        variant_id: str,
        recent_cycle_window: int = 6,
        output_path: Path | None = None,
    ) -> dict[str, object]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return self._empty_recent_activity_summary(recent_cycle_window)
        cycle_variants = self._cycle_variant_index(output_path)
        activity = self.recent_subsystem_activity_summary(
            subsystem=subsystem,
            recent_cycle_window=recent_cycle_window,
            output_path=output_path,
        )
        if int(activity.get("selected_cycles", 0)) == 0 and int(activity.get("total_decisions", 0)) == 0:
            return activity
        records = self.load_cycle_records(resolved)
        relevant_cycle_ids = {
            str(record.get("cycle_id", ""))
            for record in records
            if self._subsystems_match(str(record.get("subsystem", "")), subsystem)
            and cycle_variants.get(str(record.get("cycle_id", ""))) == variant_id
        }
        if not relevant_cycle_ids:
            return self._empty_recent_activity_summary(recent_cycle_window)
        relevant = [
            record
            for record in records
            if str(record.get("cycle_id", "")) in relevant_cycle_ids
            and self._subsystems_match(str(record.get("subsystem", "")), subsystem)
        ]
        recent_cycle_ids = list(dict.fromkeys(reversed([str(record.get("cycle_id", "")) for record in relevant if str(record.get("cycle_id", "")).strip()])))
        recent_cycle_ids = list(reversed(recent_cycle_ids[: max(1, recent_cycle_window)]))
        recent_set = set(recent_cycle_ids)
        recent_relevant = [record for record in relevant if str(record.get("cycle_id", "")) in recent_set]
        selected_cycle_ids = {
            str(record.get("cycle_id", ""))
            for record in recent_relevant
            if str(record.get("state", "")) == "select"
        }
        latest_by_cycle: dict[str, dict[str, object]] = {}
        for record in recent_relevant:
            cycle_id = str(record.get("cycle_id", "")).strip()
            if cycle_id:
                latest_by_cycle[cycle_id] = record
        decision_records = [
            record for record in recent_relevant if str(record.get("state", "")) in {"retain", "reject"}
        ]
        retained = [record for record in decision_records if str(record.get("state", "")) == "retain"]
        rejected = [record for record in decision_records if str(record.get("state", "")) == "reject"]
        last_decision = decision_records[-1] if decision_records else None
        summary = {
            "recent_cycle_window": max(1, recent_cycle_window),
            "selected_cycles": len(selected_cycle_ids),
            "retained_cycles": len(retained),
            "rejected_cycles": len(rejected),
            "last_decision_state": "" if last_decision is None else str(last_decision.get("state", "")),
            "last_cycle_id": "" if last_decision is None else str(last_decision.get("cycle_id", "")),
            "average_retained_pass_rate_delta": _average_metric_delta(
                retained,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            "average_rejected_pass_rate_delta": _average_metric_delta(
                rejected,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
        }
        total_decisions = len(decision_records)
        selected_cycles = len(selected_cycle_ids)
        summary["no_yield_cycles"] = max(0, selected_cycles - total_decisions)
        decision_cycle_ids = {str(record.get("cycle_id", "")) for record in decision_records}
        summary["recent_incomplete_cycles"] = len(
            {
                cycle_id
                for cycle_id in selected_cycle_ids
                if cycle_id not in decision_cycle_ids
                and str(latest_by_cycle.get(cycle_id, {}).get("state", "")).strip() in {"generate", "evaluate"}
            }
        )
        summary["total_decisions"] = total_decisions
        summary["retention_rate"] = round(0.0 if total_decisions == 0 else len(retained) / total_decisions, 4)
        summary["rejection_rate"] = round(0.0 if total_decisions == 0 else len(rejected) / total_decisions, 4)
        summary["recent_retention_rate"] = round(
            0.0 if selected_cycles == 0 else len(retained) / selected_cycles,
            4,
        )
        summary["recent_rejection_rate"] = round(
            0.0 if selected_cycles == 0 else len(rejected) / selected_cycles,
            4,
        )
        summary["recent_regression_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in recent_relevant
                if self._record_has_regression_signal(record)
            }
        )
        summary["recent_phase_gate_failure_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in recent_relevant
                if self._record_has_phase_gate_failure(record)
            }
        )
        phase_gate_failure_reasons = [
            reason
            for record in recent_relevant
            for reason in _record_phase_gate_failures(record)
        ]
        last_phase_gate_failure_reason = phase_gate_failure_reasons[-1] if phase_gate_failure_reasons else ""
        repeated_phase_gate_reason_count = 0
        if last_phase_gate_failure_reason:
            repeated_phase_gate_reason_count = sum(
                1 for reason in phase_gate_failure_reasons if reason == last_phase_gate_failure_reason
            )
        summary["last_phase_gate_failure_reason"] = last_phase_gate_failure_reason
        summary["repeated_phase_gate_reason_count"] = repeated_phase_gate_reason_count
        summary["recent_reconciled_failure_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in decision_records
                if self._record_is_reconciled_failure(record)
            }
        )
        observation_timeout_records = [
            record for record in recent_relevant if self._record_has_observation_timeout(record)
        ]
        summary["recent_observation_timeout_cycles"] = len(
            {str(record.get("cycle_id", "")) for record in observation_timeout_records}
        )
        timeout_budget_sources = [
            source
            for record in observation_timeout_records
            if (source := self._record_observation_timeout_budget_source(record))
        ]
        summary["recent_budgeted_observation_timeout_cycles"] = len(
            {
                str(record.get("cycle_id", ""))
                for record in observation_timeout_records
                if self._record_observation_timeout_budget_source(record)
            }
        )
        last_observation_timeout_budget_source = timeout_budget_sources[-1] if timeout_budget_sources else ""
        repeated_observation_timeout_budget_source_count = 0
        if last_observation_timeout_budget_source:
            repeated_observation_timeout_budget_source_count = sum(
                1 for source in timeout_budget_sources if source == last_observation_timeout_budget_source
            )
        summary["last_observation_timeout_budget_source"] = last_observation_timeout_budget_source
        summary["repeated_observation_timeout_budget_source_count"] = (
            repeated_observation_timeout_budget_source_count
        )
        summary["net_retained_cycle_advantage"] = len(retained) - len(rejected)
        summary["decision_quality_score"] = self._decision_quality_score(summary)
        return summary

    def _decision_records(self, output_path: Path | None = None) -> list[dict[str, object]]:
        resolved = self._resolve_cycles_path(output_path)
        if resolved is None:
            return []
        return [
            record
            for record in self.load_cycle_records(resolved)
            if str(record.get("state", "")) in {"retain", "reject"}
        ]

    def _load_retained_universe_payload_from_record(self, record: dict[str, object]) -> dict[str, object]:
        for raw_path in (
            str(record.get("active_artifact_path", "")).strip(),
            str(record.get("artifact_path", "")).strip(),
            str(record.get("candidate_artifact_path", "")).strip(),
        ):
            if not raw_path:
                continue
            payload = _load_json_payload(Path(raw_path))
            artifact_kind = str(payload.get("artifact_kind", "")).strip()
            if artifact_kind not in {"universe_contract", "universe_constitution", "operating_envelope"}:
                continue
            retained = retained_artifact_payload(payload, artifact_kind=artifact_kind)
            if isinstance(retained, dict):
                return retained
        return {}

    def _resolve_cycles_path(self, output_path: Path | None = None) -> Path | None:
        if output_path is not None:
            return output_path
        return self.cycles_path

    def _improvement_planner_controls(self) -> dict[str, object]:
        if not self.use_prompt_proposals:
            return {}
        path = self.prompt_proposals_path
        if path is None or not path.exists():
            return {}
        payload = _load_json_payload(path)
        return retained_improvement_planner_controls(payload)

    def _apply_improvement_planner_mutation(
        self,
        candidate: ImprovementExperiment,
        *,
        planner_controls: dict[str, object],
    ) -> tuple[ImprovementExperiment, dict[str, object]]:
        expected_gain_multiplier = self._planner_control_subsystem_float(
            planner_controls,
            "subsystem_expected_gain_multiplier",
            candidate.subsystem,
            fallback_subsystem=self._base_subsystem(candidate.subsystem),
            default=1.0,
            min_value=0.25,
            max_value=4.0,
        )
        cost_multiplier = self._planner_control_subsystem_float(
            planner_controls,
            "subsystem_cost_multiplier",
            candidate.subsystem,
            fallback_subsystem=self._base_subsystem(candidate.subsystem),
            default=1.0,
            min_value=0.5,
            max_value=4.0,
        )
        if expected_gain_multiplier == 1.0 and cost_multiplier == 1.0:
            return candidate, {}
        expected_gain = round(max(0.0, candidate.expected_gain * expected_gain_multiplier), 4)
        estimated_cost = max(1, int(round(candidate.estimated_cost * cost_multiplier)))
        return (
            ImprovementExperiment(
                subsystem=candidate.subsystem,
                reason=candidate.reason,
                priority=candidate.priority,
                expected_gain=expected_gain,
                estimated_cost=estimated_cost,
                score=candidate.score,
                evidence=dict(candidate.evidence),
            ),
            {
                "expected_gain_multiplier": expected_gain_multiplier,
                "cost_multiplier": cost_multiplier,
            },
        )

    def _apply_variant_planner_mutation(
        self,
        variant: ImprovementVariant,
        *,
        planner_controls: dict[str, object],
    ) -> tuple[ImprovementVariant, dict[str, object]]:
        expected_gain_multiplier = self._planner_control_variant_float(
            planner_controls,
            "variant_expected_gain_multiplier",
            variant.subsystem,
            variant.variant_id,
            fallback_subsystem=self._base_subsystem(variant.subsystem),
            default=1.0,
            min_value=0.25,
            max_value=4.0,
        )
        cost_multiplier = self._planner_control_variant_float(
            planner_controls,
            "variant_cost_multiplier",
            variant.subsystem,
            variant.variant_id,
            fallback_subsystem=self._base_subsystem(variant.subsystem),
            default=1.0,
            min_value=0.5,
            max_value=4.0,
        )
        if expected_gain_multiplier == 1.0 and cost_multiplier == 1.0:
            return variant, {}
        expected_gain = round(max(0.0, variant.expected_gain * expected_gain_multiplier), 4)
        estimated_cost = max(1, int(round(variant.estimated_cost * cost_multiplier)))
        return (
            ImprovementVariant(
                subsystem=variant.subsystem,
                variant_id=variant.variant_id,
                description=variant.description,
                expected_gain=expected_gain,
                estimated_cost=estimated_cost,
                score=variant.score,
                controls=dict(variant.controls),
            ),
            {
                "expected_gain_multiplier": expected_gain_multiplier,
                "cost_multiplier": cost_multiplier,
            },
        )

    def _with_variant_expansions(
        self,
        variants: list[ImprovementVariant],
        *,
        planner_controls: dict[str, object],
    ) -> list[ImprovementVariant]:
        combined = list(variants)
        seen_variant_ids = {variant.variant_id for variant in variants}
        expansions = planner_controls.get("variant_expansions", {})
        if not isinstance(expansions, dict) or not variants:
            return combined
        subsystem = variants[0].subsystem
        effective_subsystem = self._base_subsystem(subsystem)
        for key in dict.fromkeys([subsystem, effective_subsystem]):
            entries = expansions.get(key, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                variant_id = str(entry.get("variant_id", "")).strip()
                description = str(entry.get("description", "")).strip()
                if not variant_id or not description or variant_id in seen_variant_ids:
                    continue
                try:
                    expected_gain = max(0.0, float(entry.get("expected_gain", 0.0)))
                    estimated_cost = max(1, int(entry.get("estimated_cost", 1)))
                except (TypeError, ValueError):
                    continue
                controls = entry.get("controls", {})
                combined.append(
                    self._variant(
                        subsystem,
                        variant_id,
                        description,
                        expected_gain,
                        estimated_cost,
                        dict(controls) if isinstance(controls, dict) else {},
                    )
                )
                seen_variant_ids.add(variant_id)
        return combined

    @staticmethod
    def _planner_control_float(
        planner_controls: dict[str, object],
        field: str,
        default: float,
        *,
        min_value: float,
        max_value: float,
    ) -> float:
        value = planner_controls.get(field, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, parsed))

    @staticmethod
    def _planner_control_variant_float(
        planner_controls: dict[str, object],
        field: str,
        subsystem: str,
        variant_id: str,
        *,
        fallback_subsystem: str | None = None,
        default: float,
        min_value: float,
        max_value: float,
    ) -> float:
        mapping = planner_controls.get(field, {})
        if not isinstance(mapping, dict):
            return default
        subsystem_mapping = mapping.get(subsystem, {})
        if not isinstance(subsystem_mapping, dict) and fallback_subsystem is not None:
            subsystem_mapping = mapping.get(fallback_subsystem, {})
        elif (
            isinstance(subsystem_mapping, dict)
            and variant_id not in subsystem_mapping
            and fallback_subsystem is not None
            and fallback_subsystem != subsystem
        ):
            fallback_mapping = mapping.get(fallback_subsystem, {})
            if isinstance(fallback_mapping, dict):
                subsystem_mapping = fallback_mapping
        if not isinstance(subsystem_mapping, dict):
            return default
        value = subsystem_mapping.get(variant_id, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, parsed))

    @staticmethod
    def _planner_guardrail_float(
        planner_controls: dict[str, object],
        *,
        scope: str,
        field: str,
        legacy_field: str,
        default: float,
        min_value: float,
        max_value: float,
    ) -> float:
        search_guardrails = planner_controls.get("search_guardrails", {})
        if isinstance(search_guardrails, dict):
            scope_mapping = search_guardrails.get(scope, {})
            if isinstance(scope_mapping, dict) and field in scope_mapping:
                try:
                    parsed = float(scope_mapping.get(field, default))
                except (TypeError, ValueError):
                    return default
                return max(min_value, min(max_value, parsed))
        return ImprovementPlanner._planner_control_float(
            planner_controls,
            legacy_field,
            default,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def _planner_control_subsystem_float(
        planner_controls: dict[str, object],
        field: str,
        subsystem: str,
        *,
        fallback_subsystem: str | None = None,
        default: float,
        min_value: float,
        max_value: float,
    ) -> float:
        mapping = planner_controls.get(field, {})
        if not isinstance(mapping, dict):
            return default
        value = mapping.get(subsystem, default)
        if value == default and fallback_subsystem is not None:
            value = mapping.get(fallback_subsystem, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, parsed))

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

    @staticmethod
    def _decision_summary(records: list[dict[str, object]], *, recent_decision_window: int = 3) -> dict[str, object]:
        retained = [record for record in records if str(record.get("state", "")) == "retain"]
        rejected = [record for record in records if str(record.get("state", "")) == "reject"]
        total = len(records)
        retention_rate = 0.0 if total == 0 else len(retained) / total
        rejection_rate = 0.0 if total == 0 else len(rejected) / total
        recent_records = records[-max(1, recent_decision_window) :] if records else []
        recent_retained = [record for record in recent_records if str(record.get("state", "")) == "retain"]
        recent_rejected = [record for record in recent_records if str(record.get("state", "")) == "reject"]
        last_decision = records[-1] if records else None
        summary = {
            "total_decisions": total,
            "retained_cycles": len(retained),
            "rejected_cycles": len(rejected),
            "retention_rate": round(retention_rate, 4),
            "rejection_rate": round(rejection_rate, 4),
            "average_retained_pass_rate_delta": _average_metric_delta(
                retained,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            "average_retained_step_delta": _average_metric_delta(
                retained,
                baseline_key="baseline_average_steps",
                candidate_key="candidate_average_steps",
            ),
            "average_rejected_pass_rate_delta": _average_metric_delta(
                rejected,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            "average_rejected_step_delta": _average_metric_delta(
                rejected,
                baseline_key="baseline_average_steps",
                candidate_key="candidate_average_steps",
            ),
        }
        recent_total = len(recent_records)
        summary["recent_decision_window"] = max(1, recent_decision_window)
        summary["recent_retained_cycles"] = len(recent_retained)
        summary["recent_rejected_cycles"] = len(recent_rejected)
        summary["recent_retention_rate"] = round(0.0 if recent_total == 0 else len(recent_retained) / recent_total, 4)
        summary["recent_rejection_rate"] = round(0.0 if recent_total == 0 else len(recent_rejected) / recent_total, 4)
        summary["last_decision_state"] = "" if last_decision is None else str(last_decision.get("state", ""))
        summary["last_cycle_id"] = "" if last_decision is None else str(last_decision.get("cycle_id", ""))
        summary["net_retained_cycle_advantage"] = len(retained) - len(rejected)
        summary["decision_quality_score"] = ImprovementPlanner._decision_quality_score(summary)
        return summary

    @staticmethod
    def _portfolio_adjusted_experiment_score(
        candidate: ImprovementExperiment,
        *,
        recent_activity: dict[str, object],
        planner_controls: dict[str, object] | None = None,
    ) -> tuple[float, list[str]]:
        adjusted = float(candidate.score)
        reasons = [f"base_score={candidate.score:.4f}"]
        resolved_planner_controls = planner_controls or {}
        exploration_bonus = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_exploration_bonus",
            0.01,
            min_value=0.0,
            max_value=0.05,
        )
        saturation_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_selection_saturation_penalty_per_cycle",
            0.01,
            min_value=0.0,
            max_value=0.05,
        )
        retention_multiplier = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_recent_retention_bonus_multiplier",
            1.0,
            min_value=0.0,
            max_value=2.0,
        )
        rejection_multiplier = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_recent_rejection_penalty_multiplier",
            1.0,
            min_value=0.0,
            max_value=2.0,
        )
        no_yield_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_no_yield_penalty_per_cycle",
            0.02,
            min_value=0.0,
            max_value=0.08,
        )
        regression_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_regression_penalty_per_cycle",
            0.0125,
            min_value=0.0,
            max_value=0.06,
        )
        phase_gate_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_phase_gate_penalty_per_cycle",
            0.01,
            min_value=0.0,
            max_value=0.05,
        )
        incomplete_cycle_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_incomplete_cycle_penalty_per_cycle",
            0.025,
            min_value=0.0,
            max_value=0.08,
        )
        reconciled_failure_penalty_per_cycle = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_reconciled_failure_penalty_per_cycle",
            0.03,
            min_value=0.0,
            max_value=0.1,
        )
        repeated_phase_gate_reason_penalty = ImprovementPlanner._planner_control_float(
            resolved_planner_controls,
            "portfolio_repeated_phase_gate_reason_penalty",
            0.03,
            min_value=0.0,
            max_value=0.12,
        )
        selected_cycles = int(recent_activity.get("selected_cycles", 0))
        retained_cycles = int(recent_activity.get("retained_cycles", 0))
        rejected_cycles = int(recent_activity.get("rejected_cycles", 0))
        no_yield_cycles = int(recent_activity.get("no_yield_cycles", 0))
        recent_incomplete_cycles = int(recent_activity.get("recent_incomplete_cycles", 0))
        recent_regression_cycles = int(recent_activity.get("recent_regression_cycles", 0))
        recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0))
        recent_reconciled_failure_cycles = int(recent_activity.get("recent_reconciled_failure_cycles", 0))
        repeated_phase_gate_reason_count = int(recent_activity.get("repeated_phase_gate_reason_count", 0) or 0)
        if selected_cycles == 0:
            adjusted += exploration_bonus
            reasons.append(f"exploration_bonus={exploration_bonus:.4f}")
        else:
            saturation_penalty = min(0.03, selected_cycles * saturation_penalty_per_cycle)
            adjusted -= saturation_penalty
            reasons.append(f"selection_saturation_penalty={saturation_penalty:.4f}")
        if no_yield_cycles > 0:
            no_yield_penalty = min(0.08, no_yield_cycles * no_yield_penalty_per_cycle)
            adjusted -= no_yield_penalty
            reasons.append(f"no_yield_penalty={no_yield_penalty:.4f}")
        if recent_incomplete_cycles > 0:
            incomplete_cycle_penalty = min(0.1, recent_incomplete_cycles * incomplete_cycle_penalty_per_cycle)
            adjusted -= incomplete_cycle_penalty
            reasons.append(f"recent_incomplete_cycle_penalty={incomplete_cycle_penalty:.4f}")
        retained_delta = max(0.0, float(recent_activity.get("average_retained_pass_rate_delta", 0.0)))
        if retained_cycles > 0 and retained_delta > 0.0:
            retention_bonus = min(0.015, (retained_delta * 0.1 + retained_cycles * 0.0025) * retention_multiplier)
            adjusted += retention_bonus
            reasons.append(f"recent_retention_bonus={retention_bonus:.4f}")
        rejected_delta = max(0.0, -float(recent_activity.get("average_rejected_pass_rate_delta", 0.0)))
        if rejected_cycles > 0:
            rejection_penalty = min(0.03, (rejected_cycles * 0.0125 + rejected_delta) * rejection_multiplier)
            adjusted -= rejection_penalty
            reasons.append(f"recent_rejection_penalty={rejection_penalty:.4f}")
        if recent_regression_cycles > 0:
            regression_penalty = min(0.06, recent_regression_cycles * regression_penalty_per_cycle)
            adjusted -= regression_penalty
            reasons.append(f"recent_regression_penalty={regression_penalty:.4f}")
        if recent_phase_gate_failure_cycles > 0:
            phase_gate_penalty = min(0.05, recent_phase_gate_failure_cycles * phase_gate_penalty_per_cycle)
            adjusted -= phase_gate_penalty
            reasons.append(f"recent_phase_gate_penalty={phase_gate_penalty:.4f}")
        if recent_reconciled_failure_cycles > 0:
            reconciled_failure_penalty = min(
                0.12,
                recent_reconciled_failure_cycles * reconciled_failure_penalty_per_cycle,
            )
            adjusted -= reconciled_failure_penalty
            reasons.append(f"recent_reconciled_failure_penalty={reconciled_failure_penalty:.4f}")
        if repeated_phase_gate_reason_count > 0:
            repeated_reason_penalty = min(
                0.12,
                repeated_phase_gate_reason_count * repeated_phase_gate_reason_penalty,
            )
            adjusted -= repeated_reason_penalty
            reasons.append(f"repeated_phase_gate_reason_penalty={repeated_reason_penalty:.4f}")
        observation_timeout_penalty, observation_timeout_reasons = ImprovementPlanner._recent_observation_timeout_penalty(
            recent_activity,
            planner_controls=resolved_planner_controls,
        )
        if observation_timeout_penalty > 0.0:
            adjusted -= observation_timeout_penalty
            reasons.extend(observation_timeout_reasons)
        promotion_failure_penalty, promotion_failure_reasons = ImprovementPlanner._recent_promotion_failure_penalty(
            recent_activity,
            planner_controls=resolved_planner_controls,
        )
        if promotion_failure_penalty > 0.0:
            adjusted -= promotion_failure_penalty
            reasons.extend(promotion_failure_reasons)
        decision_quality_score = float(recent_activity.get("decision_quality_score", 0.0))
        if decision_quality_score != 0.0:
            adjusted += decision_quality_score * 0.35
            reasons.append(f"decision_quality_adjustment={decision_quality_score * 0.35:.4f}")
        return round(adjusted, 4), reasons

    @staticmethod
    def _history_bonus(summary: dict[str, object], *, variant_specific: bool = False) -> float:
        total_decisions = int(summary.get("total_decisions", 0))
        if total_decisions == 0:
            return 0.0
        retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
        retained_step_gain = max(0.0, -float(summary.get("average_retained_step_delta", 0.0))) * 0.25
        rejected_pass_penalty = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
        rejected_step_penalty = max(0.0, float(summary.get("average_rejected_step_delta", 0.0))) * 0.25
        retention_signal = float(summary.get("retention_rate", 0.0)) * (0.03 if variant_specific else 0.02)
        rejection_signal = float(summary.get("rejection_rate", 0.0)) * (0.02 if variant_specific else 0.015)
        recent_retention_signal = float(summary.get("recent_retention_rate", 0.0)) * (0.02 if variant_specific else 0.015)
        recent_rejection_signal = float(summary.get("recent_rejection_rate", 0.0)) * (0.018 if variant_specific else 0.012)
        decision_quality_signal = float(summary.get("decision_quality_score", 0.0)) * (0.4 if variant_specific else 0.3)
        incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0.0)) * (
            0.016 if variant_specific else 0.012
        )
        reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0.0)) * (
            0.018 if variant_specific else 0.014
        )
        last_decision_state = str(summary.get("last_decision_state", "")).strip()
        recency_bias = 0.0
        if last_decision_state == "retain":
            recency_bias = 0.01 if variant_specific else 0.0075
        elif last_decision_state == "reject":
            recency_bias = -0.012 if variant_specific else -0.008
        bonus = (
            retained_gain
            + retained_step_gain
            + retention_signal
            + recent_retention_signal
            + decision_quality_signal
            + recency_bias
            - rejected_pass_penalty
            - rejected_step_penalty
            - rejection_signal
            - recent_rejection_signal
            - incomplete_cycle_penalty
            - reconciled_failure_penalty
        )
        return round(max(-0.1, min(0.1, bonus)), 4)

    @staticmethod
    def _recent_history_bonus(summary: dict[str, object], *, variant_specific: bool = False) -> float:
        total_decisions = int(summary.get("total_decisions", 0))
        selected_cycles = int(summary.get("selected_cycles", 0))
        if total_decisions <= 0 and selected_cycles <= 0:
            return 0.0
        retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
        rejection_penalty = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
        quality = float(summary.get("decision_quality_score", 0.0))
        no_yield_penalty = float(summary.get("no_yield_cycles", 0)) * (0.012 if variant_specific else 0.009)
        incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0)) * (
            0.014 if variant_specific else 0.011
        )
        regression_penalty = float(summary.get("recent_regression_cycles", 0)) * (
            0.01 if variant_specific else 0.008
        )
        phase_gate_penalty = float(summary.get("recent_phase_gate_failure_cycles", 0)) * (
            0.008 if variant_specific else 0.006
        )
        reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0)) * (
            0.016 if variant_specific else 0.012
        )
        scale = 0.25 if variant_specific else 0.2
        bonus = (
            ((retained_gain - rejection_penalty + quality) * scale)
            - no_yield_penalty
            - incomplete_cycle_penalty
            - regression_penalty
            - phase_gate_penalty
            - reconciled_failure_penalty
        )
        return round(max(-0.04, min(0.04, bonus)), 4)

    @staticmethod
    def _decision_quality_score(summary: dict[str, object]) -> float:
        retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
        retained_efficiency = max(0.0, -float(summary.get("average_retained_step_delta", 0.0))) * 0.25
        rejected_regression = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
        rejected_efficiency_regression = max(0.0, float(summary.get("average_rejected_step_delta", 0.0))) * 0.2
        retention_bias = float(summary.get("retention_rate", 0.0)) * 0.02
        rejection_bias = float(summary.get("rejection_rate", 0.0)) * 0.02
        recent_retention_bias = float(summary.get("recent_retention_rate", 0.0)) * 0.015
        recent_rejection_bias = float(summary.get("recent_rejection_rate", 0.0)) * 0.015
        no_yield_penalty = float(summary.get("no_yield_cycles", 0)) * 0.01
        incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0)) * 0.012
        regression_penalty = float(summary.get("recent_regression_cycles", 0)) * 0.008
        phase_gate_penalty = float(summary.get("recent_phase_gate_failure_cycles", 0)) * 0.006
        reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0)) * 0.014
        quality = (
            retained_gain
            + retained_efficiency
            + retention_bias
            + recent_retention_bias
            - rejected_regression
            - rejected_efficiency_regression
            - rejection_bias
            - recent_rejection_bias
            - no_yield_penalty
            - incomplete_cycle_penalty
            - regression_penalty
            - phase_gate_penalty
            - reconciled_failure_penalty
        )
        return round(max(-0.1, min(0.1, quality)), 4)

    @staticmethod
    def _empty_recent_activity_summary(recent_cycle_window: int) -> dict[str, object]:
        return {
            "recent_cycle_window": max(1, recent_cycle_window),
            "selected_cycles": 0,
            "retained_cycles": 0,
            "rejected_cycles": 0,
            "no_yield_cycles": 0,
            "recent_incomplete_cycles": 0,
            "recent_observation_timeout_cycles": 0,
            "recent_budgeted_observation_timeout_cycles": 0,
            "last_observation_timeout_budget_source": "",
            "repeated_observation_timeout_budget_source_count": 0,
            "last_decision_state": "",
            "last_cycle_id": "",
            "average_retained_pass_rate_delta": 0.0,
            "average_rejected_pass_rate_delta": 0.0,
            "total_decisions": 0,
            "retention_rate": 0.0,
            "rejection_rate": 0.0,
            "recent_retention_rate": 0.0,
            "recent_rejection_rate": 0.0,
            "recent_regression_cycles": 0,
            "recent_phase_gate_failure_cycles": 0,
            "repeated_phase_gate_reason_count": 0,
            "last_phase_gate_failure_reason": "",
            "recent_reconciled_failure_cycles": 0,
            "net_retained_cycle_advantage": 0,
            "decision_quality_score": 0.0,
        }

    def _variant_exploration_bonus(
        self,
        *,
        subsystem_history: dict[str, object],
        variant_history: dict[str, object],
        variant_recent_history: dict[str, object],
        planner_controls: dict[str, object],
    ) -> float:
        if int(variant_history.get("total_decisions", 0)) > 0:
            return 0.0
        subsystem_decisions = int(subsystem_history.get("total_decisions", 0))
        if subsystem_decisions < 2:
            return 0.0
        if int(subsystem_history.get("retained_cycles", 0)) <= int(subsystem_history.get("rejected_cycles", 0)):
            return 0.0
        if str(variant_recent_history.get("last_decision_state", "")).strip():
            return 0.0
        if int(variant_recent_history.get("no_yield_cycles", 0)) > 0:
            return 0.0
        if int(variant_recent_history.get("recent_incomplete_cycles", 0)) > 0:
            return 0.0
        if int(variant_recent_history.get("recent_reconciled_failure_cycles", 0)) > 0:
            return 0.0
        if int(variant_recent_history.get("recent_regression_cycles", 0)) > 0:
            return 0.0
        return self._planner_control_float(
            planner_controls,
            "variant_exploration_bonus",
            0.004,
            min_value=0.0,
            max_value=0.03,
        )

    @staticmethod
    def _campaign_breadth_pressure(summary: dict[str, object]) -> float:
        no_yield_cycles = int(summary.get("no_yield_cycles", 0) or 0)
        incomplete_cycles = int(summary.get("recent_incomplete_cycles", 0) or 0)
        reconciled_failures = int(summary.get("recent_reconciled_failure_cycles", 0) or 0)
        pressure = (
            no_yield_cycles * 0.2
            + incomplete_cycles * 0.35
            + reconciled_failures * 0.45
        )
        return round(max(0.0, min(1.0, pressure)), 4)

    @staticmethod
    def _variant_breadth_pressure(summary: dict[str, object]) -> float:
        no_yield_cycles = int(summary.get("no_yield_cycles", 0) or 0)
        incomplete_cycles = int(summary.get("recent_incomplete_cycles", 0) or 0)
        reconciled_failures = int(summary.get("recent_reconciled_failure_cycles", 0) or 0)
        pressure = (
            no_yield_cycles * 0.2
            + incomplete_cycles * 0.35
            + reconciled_failures * 0.45
        )
        return round(max(0.0, min(1.0, pressure)), 4)

    @staticmethod
    def _record_has_phase_gate_failure(record: dict[str, object]) -> bool:
        phase_gate_passed = _record_phase_gate_passed(record)
        if phase_gate_passed is False:
            return True
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            return False
        return int(metrics.get("phase_gate_failure_count", 0) or 0) > 0

    @staticmethod
    def _record_is_reconciled_failure(record: dict[str, object]) -> bool:
        if str(record.get("state", "")).strip() != "reject":
            return False
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            return False
        return bool(metrics.get("incomplete_cycle", False) or metrics.get("finalize_exception", False))

    @staticmethod
    def _record_has_observation_timeout(record: dict[str, object]) -> bool:
        if str(record.get("state", "")).strip() != "observe":
            return False
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            return False
        return bool(metrics.get("observation_timed_out", False) or metrics.get("observation_budget_exceeded", False))

    @classmethod
    def _record_observation_timeout_budget_source(cls, record: dict[str, object]) -> str:
        if not cls._record_has_observation_timeout(record):
            return ""
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            return ""
        for field in (
            "observation_current_task_timeout_budget_source",
            "current_task_timeout_budget_source",
        ):
            source = str(metrics.get(field, "")).strip()
            if source and source != "none":
                return source
        return ""

    @classmethod
    def _record_has_regression_signal(cls, record: dict[str, object]) -> bool:
        if cls._record_has_phase_gate_failure(record):
            return True
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            return False
        regression_count_fields = (
            "regressed_task_count",
            "confirmation_regressed_task_count",
            "regressed_trace_task_count",
            "confirmation_regressed_trace_task_count",
            "regressed_trajectory_task_count",
            "confirmation_regressed_trajectory_task_count",
            "regressed_family_count",
            "generated_regressed_family_count",
            "confirmation_regressed_family_conservative_count",
            "prior_retained_regressed_family_count",
            "prior_retained_generated_regressed_family_count",
        )
        for field in regression_count_fields:
            if int(metrics.get(field, 0) or 0) > 0:
                return True
        baseline_pass_rate = metrics.get("baseline_pass_rate")
        candidate_pass_rate = metrics.get("candidate_pass_rate")
        try:
            if baseline_pass_rate is not None and candidate_pass_rate is not None:
                if float(candidate_pass_rate) < float(baseline_pass_rate):
                    return True
        except (TypeError, ValueError):
            return False
        return False


def _active_artifact_payload_from_generation_context(payload: dict[str, object] | None) -> dict[str, object] | None:
    def _retained_or_legacy_runtime_payload(candidate: object) -> dict[str, object] | None:
        if not isinstance(candidate, dict):
            return None
        artifact_kind = str(candidate.get("artifact_kind", "")).strip()
        has_contract_metadata = (
            "spec_version" in candidate
            or "lifecycle_state" in candidate
            or "retention_gate" in candidate
            or "retention_decision" in candidate
        )
        if artifact_kind and has_contract_metadata:
            return retained_artifact_payload(candidate, artifact_kind=artifact_kind)
        return candidate

    if not isinstance(payload, dict):
        return None
    context = payload.get("generation_context", {})
    if not isinstance(context, dict):
        return None
    inline_payload = context.get("active_artifact_payload")
    if isinstance(inline_payload, dict):
        return _retained_or_legacy_runtime_payload(inline_payload)
    active_artifact_value = str(context.get("active_artifact_path", "")).strip()
    if not active_artifact_value:
        return None
    active_artifact_path = Path(active_artifact_value)
    if not active_artifact_path.exists():
        return None
    try:
        loaded = _load_json_payload(active_artifact_path)
    except (OSError, json.JSONDecodeError):
        return None
    return _retained_or_legacy_runtime_payload(loaded)


def evaluate_artifact_retention(
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    artifact_path: Path | None = None,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> tuple[str, str]:
    subsystem = base_subsystem_for(subsystem, capability_modules_path)
    artifact_payload = payload
    if artifact_payload is None and artifact_path is not None and artifact_path.exists():
        loaded = _load_json_payload(artifact_path)
        if isinstance(loaded, dict):
            artifact_payload = loaded
    if isinstance(artifact_payload, dict):
        compatibility = assess_artifact_compatibility(subsystem=subsystem, payload=artifact_payload)
        if not bool(compatibility.get("compatible", False)):
            violations = compatibility.get("violations", [])
            violation = ""
            if isinstance(violations, list) and violations:
                violation = str(violations[0]).strip()
            return ("reject", violation or "artifact compatibility checks failed")
    gate = _retention_gate(subsystem, artifact_payload)
    evidence = retention_evidence(
        subsystem,
        baseline_metrics,
        candidate_metrics,
        payload=artifact_payload,
    )
    context = RetentionDecisionContext(
        subsystem=subsystem,
        gate=gate,
        evidence=evidence,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        pass_rate_delta=candidate_metrics.pass_rate - baseline_metrics.pass_rate,
        average_step_delta=candidate_metrics.average_steps - baseline_metrics.average_steps,
        generated_pass_rate_delta=candidate_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
        regressed_family_count=int(evidence.get("regressed_family_count", 0)),
        generated_regressed_family_count=int(evidence.get("generated_regressed_family_count", 0)),
        failure_recovery_delta=(
            _generated_kind_pass_rate(candidate_metrics, "failure_recovery")
            - _generated_kind_pass_rate(baseline_metrics, "failure_recovery")
        ),
    )
    evaluator = _RETENTION_EVALUATORS.get(subsystem)
    if evaluator is None:
        return ("reject", "unknown subsystem retention policy")
    return evaluator(context)


def _generated_lane_regressed(context: RetentionDecisionContext) -> bool:
    return bool(context.candidate_metrics.generated_total) and (
        context.candidate_metrics.generated_pass_rate < context.baseline_metrics.generated_pass_rate
    )


def _failure_recovery_regressed(context: RetentionDecisionContext) -> bool:
    return context.failure_recovery_delta < 0.0


def _common_family_and_lane_checks(
    context: RetentionDecisionContext,
    *,
    subject: str,
    require_generated_lane_non_regression_default: bool = True,
    require_failure_recovery_non_regression_default: bool = True,
    require_base_success_non_regression: bool = False,
) -> tuple[str, str] | None:
    if context.average_step_delta > float(context.gate.get("max_step_regression", 0.0)):
        return ("reject", f"{subject} increased average steps")
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", f"{subject} regressed one or more benchmark families")
    if bool(
        context.gate.get("require_generated_lane_non_regression", require_generated_lane_non_regression_default)
    ) and _generated_lane_regressed(context):
        return ("reject", f"{subject} regressed the generated-task lane")
    if bool(
        context.gate.get("require_failure_recovery_non_regression", require_failure_recovery_non_regression_default)
    ) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if _failure_recovery_regressed(context):
            return ("reject", f"{subject} regressed failure-recovery generation")
    if require_base_success_non_regression and context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", f"{subject} regressed base success")
    return None


def _evaluate_curriculum_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.pass_rate_delta < -float(context.gate.get("max_base_lane_regression", 0.0)):
        return ("reject", "base-lane pass rate regressed under the curriculum candidate")
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "curriculum candidate regressed one or more benchmark families")
    if context.generated_pass_rate_delta < float(context.gate.get("min_generated_pass_rate_delta_abs", 0.02)):
        return ("reject", "generated-task pass rate did not improve by the required margin")
    if context.generated_regressed_family_count > int(context.gate.get("max_generated_regressed_families", 0)):
        return ("reject", "curriculum candidate regressed one or more generated benchmark families")
    if bool(context.gate.get("require_failure_recovery_improvement", True)) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if context.failure_recovery_delta <= 0.0:
            return ("reject", "failure-recovery generation did not improve")
    return ("retain", "generated-task and failure-recovery performance improved without regressing the base lane")


def _evaluate_verifier_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if not bool(context.evidence.get("require_contract_strictness_satisfied", False)):
        return ("reject", "verifier proposals are not structurally stricter than the source contract")
    if bool(context.gate.get("require_candidate_family_eval", True)) and int(
        context.evidence.get("verifier_candidate_total", 0)
    ) <= 0:
        return ("reject", "verifier candidate was not evaluated on verifier_candidate tasks")
    if float(context.evidence.get("discrimination_gain", 0.0)) < float(context.gate.get("min_discrimination_gain", 0.02)):
        return ("reject", "verifier candidate did not produce the required discrimination gain")
    if float(context.evidence.get("false_failure_rate", 1.0)) > float(context.gate.get("max_false_failure_rate", 0.01)):
        return ("reject", "verifier candidate exceeded the false-failure budget")
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", "verifier candidate regressed base success")
    if context.candidate_metrics.average_steps > context.baseline_metrics.average_steps:
        return ("reject", "verifier candidate increased average steps")
    return ("retain", "verifier candidate satisfied strictness, discrimination, and false-failure gates")


def _evaluate_benchmark_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "benchmark candidate regressed one or more benchmark families")
    if float(context.evidence.get("family_discrimination_gain", 0.0)) < float(
        context.gate.get("min_pass_rate_delta_abs", 0.02)
    ):
        return ("reject", "benchmark candidate did not produce the required family discrimination gain")
    if float(context.evidence.get("false_failure_rate", 1.0)) > float(context.gate.get("max_false_failure_rate", 0.02)):
        return ("reject", "benchmark candidate exceeded the false-failure budget")
    return ("retain", "benchmark candidate increased discriminative coverage within the false-failure budget")


def _evaluate_policy_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.pass_rate_delta < float(context.gate.get("min_pass_rate_delta_abs", 0.01)):
        return ("reject", "policy candidate did not improve base pass rate by the required margin")
    common = _common_family_and_lane_checks(context, subject="policy candidate")
    if common is not None:
        return common
    return ("retain", "policy candidate improved pass rate without step or generated-lane regression")


def _evaluate_world_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    min_pass_rate_delta = float(context.gate.get("min_pass_rate_delta_abs", 0.0))
    if context.pass_rate_delta < min_pass_rate_delta and not (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps < context.baseline_metrics.average_steps
    ):
        return ("reject", "world-model candidate did not improve pass rate or reduce steps")
    if context.average_step_delta > float(context.gate.get("max_step_regression", 0.0)):
        return ("reject", "world-model candidate increased average steps")
    if int(context.evidence.get("low_confidence_episode_delta", 0)) > int(
        context.gate.get("max_low_confidence_episode_regression", 0)
    ):
        return ("reject", "world-model candidate increased low-confidence episodes")
    if int(context.evidence.get("first_step_success_delta", 0)) < int(context.gate.get("min_first_step_success_delta", 0)):
        return ("reject", "world-model candidate regressed first-step success")
    common = _common_family_and_lane_checks(context, subject="world-model candidate")
    if common is not None:
        return common
    return ("retain", "world-model candidate improved command selection quality without broader regression")


def _evaluate_state_estimation_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="state-estimation candidate",
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get("require_state_estimation_improvement", True)) and int(
        context.evidence.get("state_estimation_improvement_count", 0)
    ) <= 0:
        return ("reject", "state-estimation candidate did not improve retained transition, latent, or policy-bias controls")
    if int(context.evidence.get("no_state_progress_termination_delta", 0)) > int(
        context.gate.get("max_no_state_progress_termination_delta", 0)
    ):
        return ("reject", "state-estimation candidate increased no_state_progress terminations")
    if int(context.evidence.get("state_regression_trace_delta", 0)) > int(
        context.gate.get("max_state_regression_trace_delta", 0)
    ):
        return ("reject", "state-estimation candidate increased state_regression traces")
    if float(context.evidence.get("paired_trajectory_non_regression_rate", 0.0)) < float(
        context.gate.get("min_paired_trajectory_non_regression_rate", 0.5)
    ):
        return ("reject", "state-estimation candidate weakened paired trajectory non-regression")
    if float(context.evidence.get("regressive_recovery_rate_delta", 0.0)) < float(
        context.gate.get("min_regressive_recovery_rate_delta", 0.0)
    ):
        return ("reject", "state-estimation candidate weakened recovery after regressive transitions")
    return ("retain", "state-estimation candidate improved runtime state summarization without broader regression")


def _evaluate_control_surface_retention(
    context: RetentionDecisionContext,
    *,
    subject: str,
    improvement_key: str,
    improvement_reason: str,
    success_reason: str,
    require_generated_lane_non_regression_default: bool = True,
    require_failure_recovery_non_regression_default: bool = True,
) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject=subject,
        require_generated_lane_non_regression_default=require_generated_lane_non_regression_default,
        require_failure_recovery_non_regression_default=require_failure_recovery_non_regression_default,
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get(improvement_key, True)) and int(context.evidence.get(improvement_key.replace("require_", "").replace("improvement", "improvement_count"), 0)) <= 0:
        return ("reject", improvement_reason)
    return ("retain", success_reason)


def _evaluate_trust_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="trust candidate",
        improvement_key="require_trust_control_improvement",
        improvement_reason="trust candidate did not tighten unattended trust controls or broaden trust coverage",
        success_reason="trust candidate strengthened unattended trust controls without broader regression",
    )


def _evaluate_universe_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    result = _evaluate_control_surface_retention(
        context,
        subject="universe candidate",
        improvement_key="require_universe_improvement",
        improvement_reason="universe candidate did not strengthen stable governance, invariant, or command-boundary controls",
        success_reason="universe candidate strengthened stable governance above the world model without broader regression",
    )
    if result[0] != "retain":
        return result
    change_scope = str(context.evidence.get("universe_change_scope", "combined")).strip()
    if change_scope == "constitution":
        if int(context.evidence.get("prior_retained_universe_cycle_count", 0)) < int(
            context.gate.get("min_prior_retained_universe_cycles", 2)
        ):
            return ("reject", "universe constitution candidate requires multiple prior retained universe wins")
        if int(context.evidence.get("cross_family_support", 0)) < int(
            context.gate.get("min_cross_family_support", 2)
        ):
            return ("reject", "universe constitution candidate requires cross-family support")
        if int(context.evidence.get("constitution_cooldown_cycles_remaining", 0)) > 0:
            return ("reject", "universe constitution candidate is still inside the constitution cooldown window")
    return result


def _evaluate_recovery_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="recovery candidate",
        improvement_key="require_recovery_control_improvement",
        improvement_reason="recovery candidate did not strengthen unattended rollback or snapshot policy",
        success_reason="recovery candidate strengthened unattended rollback policy without broader regression",
    )


def _evaluate_delegation_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="delegation candidate",
        improvement_key="require_delegation_control_improvement",
        improvement_reason="delegation candidate did not expand delegated runtime capacity",
        success_reason="delegation candidate expanded delegated runtime capacity without broader regression",
    )


def _evaluate_operator_policy_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="operator-policy candidate",
        improvement_key="require_operator_policy_improvement",
        improvement_reason="operator-policy candidate did not broaden unattended operator policy",
        success_reason="operator-policy candidate broadened unattended execution policy without broader regression",
    )


def _evaluate_transition_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    return _evaluate_control_surface_retention(
        context,
        subject="transition-model candidate",
        improvement_key="require_transition_model_improvement",
        improvement_reason="transition-model candidate did not strengthen retained bad-transition controls",
        success_reason="transition-model candidate improved retained bad-transition guidance without broader regression",
    )


def _evaluate_retrieval_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if context.regressed_family_count > int(context.gate.get("max_regressed_families", 0)):
        return ("reject", "retrieval candidate regressed one or more benchmark families")
    if bool(context.gate.get("require_failure_recovery_non_regression", False)) and (
        _has_generated_kind(context.baseline_metrics, "failure_recovery")
        or _has_generated_kind(context.candidate_metrics, "failure_recovery")
    ):
        if _failure_recovery_regressed(context):
            return ("reject", "retrieval candidate regressed failure-recovery generation")
    discrimination_satisfied = (
        not bool(context.gate.get("require_family_discrimination", True))
        or float(context.evidence.get("family_discrimination_gain", 0.0)) > 0.0
    )
    if (
        context.pass_rate_delta >= float(context.gate.get("min_pass_rate_delta_abs", 0.02))
        and context.average_step_delta <= 0.0
        and discrimination_satisfied
    ):
        return ("retain", "retrieval candidate improved pass rate and family discrimination without increasing steps")
    if (
        context.candidate_metrics.pass_rate >= context.baseline_metrics.pass_rate
        and context.candidate_metrics.average_steps <= context.baseline_metrics.average_steps
        and context.candidate_metrics.trusted_retrieval_steps > context.baseline_metrics.trusted_retrieval_steps
        and context.candidate_metrics.low_confidence_episodes <= context.baseline_metrics.low_confidence_episodes
        and float(context.evidence.get("false_failure_rate", 0.0)) <= float(context.gate.get("max_false_failure_rate", 0.02))
        and discrimination_satisfied
    ):
        return ("retain", "retrieval candidate increased trusted retrieval usage without regressing the base lane")
    return ("reject", "retrieval candidate did not satisfy the retained retrieval gate")


def _evaluate_tolbert_model_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if not bool(context.evidence.get("checkpoint_exists", False)):
        return ("reject", "Tolbert model candidate did not produce a checkpoint")
    if int(context.evidence.get("cache_count", 0)) <= 0:
        return ("reject", "Tolbert model candidate did not produce a retrieval cache")
    common = _common_family_and_lane_checks(
        context,
        subject="Tolbert model candidate",
        require_generated_lane_non_regression_default=True,
        require_failure_recovery_non_regression_default=True,
    )
    if common is not None:
        return common
    if int(context.evidence.get("low_confidence_episode_delta", 0)) > int(
        context.gate.get("max_low_confidence_episode_regression", 0)
    ):
        return ("reject", "Tolbert model candidate increased low-confidence episodes")
    if float(context.evidence.get("first_step_confidence_delta", 0.0)) < float(
        context.gate.get("min_first_step_confidence_delta", 0.0)
    ):
        return ("reject", "Tolbert model candidate regressed first-step path confidence")
    if int(context.evidence.get("trusted_retrieval_delta", 0)) < int(context.gate.get("min_trusted_retrieval_delta", 0)):
        return ("reject", "Tolbert model candidate did not improve trusted retrieval usage")
    if bool(context.gate.get("require_novel_command_signal", False)) and int(
        context.candidate_metrics.proposal_selected_steps
    ) <= 0:
        return ("reject", "Tolbert model candidate produced no proposal-selected commands")
    if int(context.evidence.get("proposal_selected_steps_delta", 0)) < int(
        context.gate.get("min_proposal_selected_steps_delta", 0)
    ):
        return ("reject", "Tolbert model candidate regressed proposal-selected command usage")
    if int(context.candidate_metrics.novel_valid_command_steps) < int(
        context.gate.get("min_novel_valid_command_steps", 0)
    ):
        return ("reject", "Tolbert model candidate did not produce enough verifier-valid novel commands")
    if float(context.evidence.get("novel_valid_command_rate_delta", 0.0)) < float(
        context.gate.get("min_novel_valid_command_rate_delta", 0.0)
    ):
        return ("reject", "Tolbert model candidate regressed verifier-valid novel-command rate")
    family_gate_failure = proposal_gate_failure_reason(
        context.gate,
        context.evidence,
        subject="Tolbert model candidate",
    )
    if family_gate_failure is not None:
        return ("reject", family_gate_failure)
    if context.candidate_metrics.pass_rate < context.baseline_metrics.pass_rate:
        return ("reject", "Tolbert model candidate regressed base success")
    return ("retain", "Tolbert model candidate improved learned retrieval and novel-command behavior without broader regression")


def _evaluate_capabilities_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="capability registry candidate",
        require_generated_lane_non_regression_default=False,
        require_failure_recovery_non_regression_default=False,
        require_base_success_non_regression=True,
    )
    if common is not None:
        return common
    if bool(context.gate.get("require_capability_surface_growth", True)):
        if (
            int(context.evidence.get("module_count_delta", 0)) <= 0
            and int(context.evidence.get("external_capability_count_delta", 0)) <= 0
            and int(context.evidence.get("improvement_surface_count_delta", 0)) <= 0
        ):
            return ("reject", "capability registry candidate did not expand capability or improvement surfaces")
    return ("retain", "capability registry candidate preserved runtime quality while expanding mutable capability surfaces")


def _evaluate_skill_or_tooling_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    common = _common_family_and_lane_checks(
        context,
        subject="candidate",
        require_generated_lane_non_regression_default=False,
        require_failure_recovery_non_regression_default=False,
    )
    if common is not None:
        return common
    if (
        context.candidate_metrics.pass_rate > context.baseline_metrics.pass_rate
        or (
            context.candidate_metrics.pass_rate == context.baseline_metrics.pass_rate
            and context.candidate_metrics.average_steps < context.baseline_metrics.average_steps
        )
    ):
        if context.subsystem != "tooling" or bool(context.evidence.get("replay_verified", False)):
            return ("retain", "candidate improved pass rate or preserved pass rate while reducing steps")
    if context.subsystem == "tooling" and not bool(context.evidence.get("replay_verified", False)):
        return ("reject", "tool candidate was not replay-verified before retention")
    return ("reject", "candidate did not produce a verified gain")


def _evaluate_operators_retention(context: RetentionDecisionContext) -> tuple[str, str]:
    if bool(context.gate.get("require_cross_task_support", True)) and int(context.evidence.get("support_count", 0)) < int(
        context.gate.get("min_support", 2)
    ):
        return ("reject", "operator artifact does not have the required cross-task support")
    if float(context.evidence.get("transfer_pass_rate_delta", 0.0)) < float(
        context.gate.get("min_transfer_pass_rate_delta_abs", 0.05)
    ):
        return ("reject", "operator transfer did not beat raw skill transfer by the required margin")
    return ("retain", "operator transfer beat raw skill transfer on the shared held-out lane")


_RETENTION_EVALUATORS: dict[str, callable] = {
    "curriculum": _evaluate_curriculum_retention,
    "verifier": _evaluate_verifier_retention,
    "benchmark": _evaluate_benchmark_retention,
    "policy": _evaluate_policy_retention,
    "world_model": _evaluate_world_model_retention,
    "state_estimation": _evaluate_state_estimation_retention,
    "universe": _evaluate_universe_retention,
    "trust": _evaluate_trust_retention,
    "recovery": _evaluate_recovery_retention,
    "delegation": _evaluate_delegation_retention,
    "operator_policy": _evaluate_operator_policy_retention,
    "transition_model": _evaluate_transition_model_retention,
    "retrieval": _evaluate_retrieval_retention,
    "tolbert_model": _evaluate_tolbert_model_retention,
    "capabilities": _evaluate_capabilities_retention,
    "skills": _evaluate_skill_or_tooling_retention,
    "tooling": _evaluate_skill_or_tooling_retention,
    "operators": _evaluate_operators_retention,
}


def retention_evidence(
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    subsystem = base_subsystem_for(subsystem, capability_modules_path)
    evidence: dict[str, object] = {
        "pass_rate_delta": candidate_metrics.pass_rate - baseline_metrics.pass_rate,
        "average_step_delta": candidate_metrics.average_steps - baseline_metrics.average_steps,
        "generated_pass_rate_delta": candidate_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
        "proposal_selected_steps_delta": (
            candidate_metrics.proposal_selected_steps - baseline_metrics.proposal_selected_steps
        ),
        "novel_command_steps_delta": (
            candidate_metrics.novel_command_steps - baseline_metrics.novel_command_steps
        ),
        "novel_valid_command_rate_delta": (
            candidate_metrics.novel_valid_command_rate - baseline_metrics.novel_valid_command_rate
        ),
        "tolbert_primary_episodes_delta": (
            candidate_metrics.tolbert_primary_episodes - baseline_metrics.tolbert_primary_episodes
        ),
        "tolbert_shadow_episodes_delta": (
            candidate_metrics.tolbert_shadow_episodes - baseline_metrics.tolbert_shadow_episodes
        ),
    }
    family_pass_rate_delta = _family_pass_rate_delta_map(baseline_metrics, candidate_metrics)
    if family_pass_rate_delta:
        evidence["family_pass_rate_delta"] = family_pass_rate_delta
        evidence["regressed_family_count"] = _family_regression_count(baseline_metrics, candidate_metrics)
        evidence["worst_family_delta"] = _family_worst_delta(baseline_metrics, candidate_metrics)
    generated_family_pass_rate_delta = _generated_family_pass_rate_delta_map(baseline_metrics, candidate_metrics)
    if generated_family_pass_rate_delta:
        evidence["generated_family_pass_rate_delta"] = generated_family_pass_rate_delta
        evidence["generated_regressed_family_count"] = _generated_family_regression_count(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["generated_worst_family_delta"] = _generated_family_worst_delta(
            baseline_metrics,
            candidate_metrics,
        )
    if _has_generated_kind(baseline_metrics, "failure_recovery") or _has_generated_kind(candidate_metrics, "failure_recovery"):
        evidence["failure_recovery_pass_rate_delta"] = _generated_kind_pass_rate(
            candidate_metrics,
            "failure_recovery",
        ) - _generated_kind_pass_rate(baseline_metrics, "failure_recovery")
    if subsystem == "retrieval":
        benchmark_candidate_total = candidate_metrics.total_by_benchmark_family.get("benchmark_candidate", 0)
        evidence["family_discrimination_gain"] = _family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["false_failure_rate"] = (
            0.0
            if benchmark_candidate_total == 0
            else _candidate_family_failure_rate(candidate_metrics, "benchmark_candidate")
        )
        evidence["trusted_retrieval_delta"] = (
            candidate_metrics.trusted_retrieval_steps - baseline_metrics.trusted_retrieval_steps
        )
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
    if subsystem == "tolbert_model":
        runtime_paths = payload.get("runtime_paths", {}) if isinstance(payload, dict) else {}
        checkpoint_path = (
            resolve_tolbert_runtime_checkpoint_path(runtime_paths)
            if isinstance(runtime_paths, dict)
            else ""
        )
        cache_paths = runtime_paths.get("cache_paths", []) if isinstance(runtime_paths, dict) else []
        evidence["family_discrimination_gain"] = _family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["trusted_retrieval_delta"] = (
            candidate_metrics.trusted_retrieval_steps - baseline_metrics.trusted_retrieval_steps
        )
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
        proposal_metrics_by_family = _proposal_metrics_delta_by_benchmark_family(
            baseline_metrics,
            candidate_metrics,
        )
        if proposal_metrics_by_family:
            evidence["proposal_metrics_by_benchmark_family"] = proposal_metrics_by_family
        evidence["first_step_confidence_delta"] = (
            candidate_metrics.average_first_step_path_confidence
            - baseline_metrics.average_first_step_path_confidence
        )
        runtime_policy = payload.get("runtime_policy", {}) if isinstance(payload, dict) else {}
        hybrid_runtime = payload.get("hybrid_runtime", {}) if isinstance(payload, dict) else {}
        if isinstance(runtime_policy, dict):
            evidence["primary_benchmark_family_count"] = len(
                [str(value).strip() for value in runtime_policy.get("primary_benchmark_families", []) if str(value).strip()]
            )
            evidence["shadow_benchmark_family_count"] = len(
                [str(value).strip() for value in runtime_policy.get("shadow_benchmark_families", []) if str(value).strip()]
            )
        if isinstance(hybrid_runtime, dict):
            evidence["primary_takeover_enabled"] = bool(hybrid_runtime.get("primary_enabled", False))
            evidence["shadow_takeover_enabled"] = bool(hybrid_runtime.get("shadow_enabled", False))
        evidence["checkpoint_exists"] = bool(checkpoint_path and Path(checkpoint_path).exists())
        evidence["cache_count"] = (
            len([str(path).strip() for path in cache_paths if str(path).strip()])
            if isinstance(cache_paths, list)
            else 0
        )
    if subsystem == "verifier":
        verifier_candidate_total = candidate_metrics.total_by_benchmark_family.get("verifier_candidate", 0)
        verifier_candidate_pass_rate = candidate_metrics.benchmark_family_pass_rate("verifier_candidate")
        false_failure_rate = _candidate_family_failure_rate(candidate_metrics, "verifier_candidate")
        evidence["verifier_candidate_total"] = verifier_candidate_total
        evidence["verifier_candidate_pass_rate"] = verifier_candidate_pass_rate
        evidence["false_failure_rate"] = false_failure_rate
        evidence["proposal_discrimination_estimate"] = _verifier_discrimination_gain(payload)
        evidence["discrimination_gain"] = max(0.0, verifier_candidate_pass_rate * (1.0 - false_failure_rate))
        evidence["require_contract_strictness_satisfied"] = _verifier_contracts_are_strict(payload)
    if subsystem == "benchmark":
        benchmark_candidate_total = candidate_metrics.total_by_benchmark_family.get("benchmark_candidate", 0)
        evidence["benchmark_candidate_total"] = benchmark_candidate_total
        evidence["family_discrimination_gain"] = _family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["false_failure_rate"] = (
            0.0
            if benchmark_candidate_total == 0
            else _candidate_family_failure_rate(candidate_metrics, "benchmark_candidate")
        )
    if subsystem == "operators":
        evidence["transfer_pass_rate"] = candidate_metrics.memory_source_pass_rate("operator")
        evidence["baseline_transfer_pass_rate"] = baseline_metrics.memory_source_pass_rate("skill_transfer")
        evidence["transfer_pass_rate_delta"] = (
            candidate_metrics.memory_source_pass_rate("operator")
            - baseline_metrics.memory_source_pass_rate("skill_transfer")
        )
        evidence["support_count"] = _operator_support_count(payload)
    if subsystem == "tooling":
        evidence["replay_verified"] = _tool_candidates_have_stage(payload, "replay_verified")
    if subsystem == "trust":
        baseline_controls = _trust_controls_from_payload(_active_artifact_payload_from_generation_context(payload))
        candidate_controls = _trust_controls_from_payload(payload)
        evidence.update(_trust_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "universe":
        baseline_payload = _active_artifact_payload_from_generation_context(payload)
        evidence.update(
            _universe_control_evidence(
                baseline_governance=_universe_governance_from_payload(baseline_payload),
                candidate_governance=_universe_governance_from_payload(payload),
                baseline_action_risk_controls=_universe_action_risk_controls_from_payload(baseline_payload),
                candidate_action_risk_controls=_universe_action_risk_controls_from_payload(payload),
                baseline_environment_assumptions=_universe_environment_assumptions_from_payload(baseline_payload),
                candidate_environment_assumptions=_universe_environment_assumptions_from_payload(payload),
                baseline_invariants=_universe_invariants_from_payload(baseline_payload),
                candidate_invariants=_universe_invariants_from_payload(payload),
                baseline_forbidden_patterns=_universe_forbidden_patterns_from_payload(baseline_payload),
                candidate_forbidden_patterns=_universe_forbidden_patterns_from_payload(payload),
                baseline_preferred_prefixes=_universe_preferred_prefixes_from_payload(baseline_payload),
                candidate_preferred_prefixes=_universe_preferred_prefixes_from_payload(payload),
            )
        )
        evidence["universe_change_scope"] = _universe_change_scope(payload)
        evidence["cross_family_support"] = _universe_cross_family_support(evidence)
        evidence["outcome_weighted_support"] = _universe_outcome_weighted_support(evidence)
        retention_context = payload.get("retention_context", {}) if isinstance(payload, dict) else {}
        if isinstance(retention_context, dict):
            for key in (
                "prior_retained_universe_cycle_count",
                "constitution_cooldown_cycles_remaining",
            ):
                if key in retention_context:
                    evidence[key] = retention_context.get(key)
    if subsystem == "recovery":
        baseline_controls = _recovery_controls_from_payload(_active_artifact_payload_from_generation_context(payload))
        candidate_controls = _recovery_controls_from_payload(payload)
        evidence.update(_recovery_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "delegation":
        baseline_controls = _delegation_controls_from_payload(_active_artifact_payload_from_generation_context(payload))
        candidate_controls = _delegation_controls_from_payload(payload)
        evidence.update(_delegation_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "operator_policy":
        baseline_controls = _operator_policy_controls_from_payload(_active_artifact_payload_from_generation_context(payload))
        candidate_controls = _operator_policy_controls_from_payload(payload)
        evidence.update(_operator_policy_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "transition_model":
        baseline_controls = _transition_model_controls_from_payload(
            _active_artifact_payload_from_generation_context(payload)
        )
        candidate_controls = _transition_model_controls_from_payload(payload)
        baseline_signatures = _transition_model_signatures_from_payload(
            _active_artifact_payload_from_generation_context(payload)
        )
        candidate_signatures = _transition_model_signatures_from_payload(payload)
        evidence.update(
            _transition_model_evidence(
                baseline_controls,
                candidate_controls,
                baseline_signatures=baseline_signatures,
                candidate_signatures=candidate_signatures,
            )
        )
    if subsystem == "capabilities":
        baseline_summary = capability_surface_summary(_active_artifact_payload_from_generation_context(payload))
        candidate_summary = capability_surface_summary(payload)
        evidence.update(_capability_surface_evidence(baseline_summary, candidate_summary))
    if subsystem == "world_model":
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
        evidence["first_step_success_delta"] = (
            candidate_metrics.first_step_successes - baseline_metrics.first_step_successes
        )
    if subsystem == "state_estimation":
        baseline_transition_controls = _state_estimation_transition_controls_from_payload(
            _active_artifact_payload_from_generation_context(payload)
        )
        candidate_transition_controls = _state_estimation_transition_controls_from_payload(payload)
        baseline_latent_controls = _state_estimation_latent_controls_from_payload(
            _active_artifact_payload_from_generation_context(payload)
        )
        candidate_latent_controls = _state_estimation_latent_controls_from_payload(payload)
        baseline_policy_controls = _state_estimation_policy_controls_from_payload(
            _active_artifact_payload_from_generation_context(payload)
        )
        candidate_policy_controls = _state_estimation_policy_controls_from_payload(payload)
        evidence.update(
            _state_estimation_evidence(
                baseline_transition_controls=baseline_transition_controls,
                candidate_transition_controls=candidate_transition_controls,
                baseline_latent_controls=baseline_latent_controls,
                candidate_latent_controls=candidate_latent_controls,
                baseline_policy_controls=baseline_policy_controls,
                candidate_policy_controls=candidate_policy_controls,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
            )
        )
    return evidence


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
        if bool(family_gate.get("require_novel_command_signal", False)) and int(
            metrics.get("candidate_proposal_selected_steps", 0) or 0
        ) <= 0:
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
    total = metrics.generated_by_kind.get(kind, 0)
    if total == 0:
        return 0.0
    return metrics.generated_passed_by_kind.get(kind, 0) / total


def _has_generated_kind(metrics: EvalMetrics, kind: str) -> bool:
    return metrics.generated_by_kind.get(kind, 0) > 0


def _state_estimation_transition_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_transition_controls(None)
    return retained_state_estimation_transition_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "controls": payload.get("controls", {}),
        }
    )


def _state_estimation_latent_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_latent_controls(None)
    return retained_state_estimation_latent_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "latent_controls": payload.get("latent_controls", {}),
        }
    )


def _state_estimation_policy_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_policy_controls(None)
    return retained_state_estimation_policy_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "policy_controls": payload.get("policy_controls", {}),
        }
    )


def _state_estimation_improvement_count(
    *,
    baseline_transition_controls: dict[str, object],
    candidate_transition_controls: dict[str, object],
    baseline_latent_controls: dict[str, object],
    candidate_latent_controls: dict[str, object],
    baseline_policy_controls: dict[str, object],
    candidate_policy_controls: dict[str, object],
) -> int:
    improvements = 0
    for key in sorted(_STATE_ESTIMATION_CONTROL_KEYS):
        if float(candidate_transition_controls.get(key, 0.0)) != float(baseline_transition_controls.get(key, 0.0)):
            improvements += 1
    for key in sorted(_STATE_ESTIMATION_LATENT_KEYS):
        if float(candidate_latent_controls.get(key, 0.0)) != float(baseline_latent_controls.get(key, 0.0)):
            improvements += 1
    for key in sorted(_STATE_ESTIMATION_POLICY_KEYS):
        if int(candidate_policy_controls.get(key, 0)) != int(baseline_policy_controls.get(key, 0)):
            improvements += 1
    return improvements


def _trajectory_has_regression(payload: dict[str, object]) -> bool:
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if int(step.get("state_regression_count", 0) or 0) > 0:
            return True
        signals = {
            str(value).strip()
            for value in step.get("failure_signals", [])
            if str(value).strip()
        }
        if "state_regression" in signals:
            return True
    return False


def _state_regression_trace_count(metrics: EvalMetrics) -> int:
    return sum(
        1
        for payload in (metrics.task_trajectories or {}).values()
        if isinstance(payload, dict) and _trajectory_has_regression(payload)
    )


def _regressive_recovery_rate(metrics: EvalMetrics) -> float:
    trajectories = [
        payload
        for payload in (metrics.task_trajectories or {}).values()
        if isinstance(payload, dict) and _trajectory_has_regression(payload)
    ]
    if not trajectories:
        return 0.0
    recovered = sum(1 for payload in trajectories if bool(payload.get("success", False)))
    return recovered / len(trajectories)


def _paired_trajectory_non_regression_rate(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> tuple[float, int]:
    baseline_trajectories = baseline_metrics.task_trajectories or {}
    candidate_trajectories = candidate_metrics.task_trajectories or {}
    if not isinstance(baseline_trajectories, dict) or not isinstance(candidate_trajectories, dict):
        return (0.0, 0)
    shared_task_ids = sorted(set(baseline_trajectories) & set(candidate_trajectories))
    if not shared_task_ids:
        return (0.0, 0)
    non_regressions = 0
    pair_count = 0
    for task_id in shared_task_ids:
        baseline_payload = baseline_trajectories.get(task_id, {})
        candidate_payload = candidate_trajectories.get(task_id, {})
        if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
            continue
        pair_count += 1
        if _state_estimation_trajectory_score(candidate_payload) <= _state_estimation_trajectory_score(baseline_payload):
            non_regressions += 1
    if pair_count <= 0:
        return (0.0, 0)
    return (non_regressions / pair_count, pair_count)


def _state_estimation_trajectory_score(payload: dict[str, object]) -> float:
    score = 0.0
    if not bool(payload.get("success", False)):
        score += 5.0
    termination_reason = str(payload.get("termination_reason", "")).strip()
    if termination_reason == "no_state_progress":
        score += 3.0
    elif termination_reason == "repeated_failed_action":
        score += 2.0
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        score += 0.5 * int(step.get("state_regression_count", 0) or 0)
        if bool(step.get("timed_out", False)):
            score += 1.0
        if int(step.get("exit_code", 0) or 0) != 0:
            score += 0.25
        signals = {
            str(value).strip()
            for value in step.get("failure_signals", [])
            if str(value).strip()
        }
        if "no_state_progress" in signals:
            score += 1.0
        if "state_regression" in signals:
            score += 1.0
    return score


def _state_estimation_evidence(
    *,
    baseline_transition_controls: dict[str, object],
    candidate_transition_controls: dict[str, object],
    baseline_latent_controls: dict[str, object],
    candidate_latent_controls: dict[str, object],
    baseline_policy_controls: dict[str, object],
    candidate_policy_controls: dict[str, object],
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, object]:
    paired_non_regression_rate, paired_trajectory_pair_count = _paired_trajectory_non_regression_rate(
        baseline_metrics,
        candidate_metrics,
    )
    return {
        "state_estimation_improvement_count": _state_estimation_improvement_count(
            baseline_transition_controls=baseline_transition_controls,
            candidate_transition_controls=candidate_transition_controls,
            baseline_latent_controls=baseline_latent_controls,
            candidate_latent_controls=candidate_latent_controls,
            baseline_policy_controls=baseline_policy_controls,
            candidate_policy_controls=candidate_policy_controls,
        ),
        "no_state_progress_termination_delta": int(
            candidate_metrics.termination_reasons.get("no_state_progress", 0)
        )
        - int(baseline_metrics.termination_reasons.get("no_state_progress", 0)),
        "state_regression_trace_delta": _state_regression_trace_count(candidate_metrics)
        - _state_regression_trace_count(baseline_metrics),
        "paired_trajectory_non_regression_rate": paired_non_regression_rate,
        "paired_trajectory_pair_count": paired_trajectory_pair_count,
        "regressive_recovery_rate_delta": _regressive_recovery_rate(candidate_metrics)
        - _regressive_recovery_rate(baseline_metrics),
        "transition_control_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_CONTROL_KEYS)
            if float(candidate_transition_controls.get(key, 0.0))
            != float(baseline_transition_controls.get(key, 0.0))
        ),
        "latent_control_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_LATENT_KEYS)
            if float(candidate_latent_controls.get(key, 0.0))
            != float(baseline_latent_controls.get(key, 0.0))
        ),
        "state_estimation_policy_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_POLICY_KEYS)
            if int(candidate_policy_controls.get(key, 0))
            != int(baseline_policy_controls.get(key, 0))
        ),
    }


def _trust_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(controls, list_fields=("required_benchmark_families",))


def _universe_governance_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_universe_governance(None)
    return retained_universe_governance(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "governance": payload.get("governance", {}),
        }
    )


def _universe_invariants_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return retained_universe_invariants(None)
    return retained_universe_invariants(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "invariants": payload.get("invariants", []),
        }
    )


def _universe_forbidden_patterns_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return retained_universe_forbidden_command_patterns(None)
    return retained_universe_forbidden_command_patterns(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "forbidden_command_patterns": payload.get("forbidden_command_patterns", []),
        }
    )


def _universe_preferred_prefixes_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return retained_universe_preferred_command_prefixes(None)
    return retained_universe_preferred_command_prefixes(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "preferred_command_prefixes": payload.get("preferred_command_prefixes", []),
        }
    )


def _universe_action_risk_controls_from_payload(payload: dict[str, object] | None) -> dict[str, int]:
    if not isinstance(payload, dict):
        return retained_universe_action_risk_controls(None)
    return retained_universe_action_risk_controls(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "action_risk_controls": payload.get("action_risk_controls", {}),
        }
    )


def _universe_environment_assumptions_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_universe_environment_assumptions(None)
    return retained_universe_environment_assumptions(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "environment_assumptions": payload.get("environment_assumptions", {}),
        }
    )


def _universe_change_scope(payload: dict[str, object] | None) -> str:
    if not isinstance(payload, dict):
        return "combined"
    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    if artifact_kind == "universe_constitution":
        return "constitution"
    if artifact_kind == "operating_envelope":
        return "operating_envelope"
    return "combined"


def _universe_cross_family_support(evidence: dict[str, object]) -> int:
    support = 0
    for key in ("family_pass_rate_delta", "generated_family_pass_rate_delta"):
        delta_map = evidence.get(key, {})
        if not isinstance(delta_map, dict):
            continue
        for value in delta_map.values():
            try:
                if float(value) >= 0.0:
                    support += 1
            except (TypeError, ValueError):
                continue
    if support > 0:
        return support
    return max(1, 1 - int(evidence.get("regressed_family_count", 0)))


def _universe_outcome_weighted_support(evidence: dict[str, object]) -> float:
    pass_gain = max(0.0, float(evidence.get("pass_rate_delta", 0.0) or 0.0))
    step_gain = max(0.0, -float(evidence.get("average_step_delta", 0.0) or 0.0))
    cross_family_support = max(1, _universe_cross_family_support(evidence))
    support_discount = 0.5 if cross_family_support <= 1 else 1.0
    return round((1.0 + pass_gain * 20.0 + step_gain * 5.0) * cross_family_support * support_discount, 4)


def _universe_improvement_count(
    *,
    baseline_governance: dict[str, object],
    candidate_governance: dict[str, object],
    baseline_action_risk_controls: dict[str, int],
    candidate_action_risk_controls: dict[str, int],
    baseline_environment_assumptions: dict[str, object],
    candidate_environment_assumptions: dict[str, object],
    baseline_invariants: list[str],
    candidate_invariants: list[str],
    baseline_forbidden_patterns: list[str],
    candidate_forbidden_patterns: list[str],
    baseline_preferred_prefixes: list[str],
    candidate_preferred_prefixes: list[str],
) -> int:
    improvements = _enabled_flag_improvement_count(
        baseline_governance,
        candidate_governance,
        keys=tuple(sorted(_UNIVERSE_GOVERNANCE_KEYS)),
    )
    improvements += _increased_int_control_count(
        baseline_action_risk_controls,
        candidate_action_risk_controls,
        keys=tuple(sorted(_UNIVERSE_ACTION_RISK_KEYS)),
    )
    improvements += _more_restrictive_environment_assumption_count(
        baseline_environment_assumptions,
        candidate_environment_assumptions,
    )
    if set(candidate_invariants) > set(baseline_invariants):
        improvements += 1
    if set(candidate_forbidden_patterns) > set(baseline_forbidden_patterns):
        improvements += 1
    if set(candidate_preferred_prefixes) > set(baseline_preferred_prefixes):
        improvements += 1
    return improvements


def _environment_assumption_delta_count(
    baseline_assumptions: dict[str, object],
    candidate_assumptions: dict[str, object],
) -> int:
    deltas = 0
    for key in sorted(_UNIVERSE_ENVIRONMENT_ENUM_FIELDS):
        if str(candidate_assumptions.get(key, "")).strip().lower() != str(baseline_assumptions.get(key, "")).strip().lower():
            deltas += 1
    for key in sorted(_UNIVERSE_ENVIRONMENT_BOOL_FIELDS):
        if bool(candidate_assumptions.get(key, False)) != bool(baseline_assumptions.get(key, False)):
            deltas += 1
    return deltas


def _more_restrictive_environment_assumption_count(
    baseline_assumptions: dict[str, object],
    candidate_assumptions: dict[str, object],
) -> int:
    improvements = 0
    for key, allowed_values in _UNIVERSE_ENVIRONMENT_ENUM_FIELDS.items():
        if key == "network_access_mode":
            order = ("blocked", "allowlist_only", "open")
        elif key == "git_write_mode":
            order = ("blocked", "operator_gated", "task_scoped")
        else:
            order = ("task_only", "generated_only", "shared_repo_gated")
        rank = {value: index for index, value in enumerate(order) if value in allowed_values}
        baseline_value = str(baseline_assumptions.get(key, order[0])).strip().lower()
        candidate_value = str(candidate_assumptions.get(key, order[0])).strip().lower()
        if baseline_value in rank and candidate_value in rank and rank[candidate_value] < rank[baseline_value]:
            improvements += 1
    for key in sorted(_UNIVERSE_ENVIRONMENT_BOOL_FIELDS):
        if not bool(baseline_assumptions.get(key, False)) and bool(candidate_assumptions.get(key, False)):
            improvements += 1
    return improvements


def _universe_control_evidence(
    *,
    baseline_governance: dict[str, object],
    candidate_governance: dict[str, object],
    baseline_action_risk_controls: dict[str, int],
    candidate_action_risk_controls: dict[str, int],
    baseline_environment_assumptions: dict[str, object],
    candidate_environment_assumptions: dict[str, object],
    baseline_invariants: list[str],
    candidate_invariants: list[str],
    baseline_forbidden_patterns: list[str],
    candidate_forbidden_patterns: list[str],
    baseline_preferred_prefixes: list[str],
    candidate_preferred_prefixes: list[str],
) -> dict[str, object]:
    return {
        "universe_improvement_count": _universe_improvement_count(
            baseline_governance=baseline_governance,
            candidate_governance=candidate_governance,
            baseline_action_risk_controls=baseline_action_risk_controls,
            candidate_action_risk_controls=candidate_action_risk_controls,
            baseline_environment_assumptions=baseline_environment_assumptions,
            candidate_environment_assumptions=candidate_environment_assumptions,
            baseline_invariants=baseline_invariants,
            candidate_invariants=candidate_invariants,
            baseline_forbidden_patterns=baseline_forbidden_patterns,
            candidate_forbidden_patterns=candidate_forbidden_patterns,
            baseline_preferred_prefixes=baseline_preferred_prefixes,
            candidate_preferred_prefixes=candidate_preferred_prefixes,
        ),
        "universe_governance_delta_count": sum(
            1
            for key in sorted(_UNIVERSE_GOVERNANCE_KEYS)
            if bool(candidate_governance.get(key, False)) != bool(baseline_governance.get(key, False))
        ),
        "universe_action_risk_delta_count": sum(
            1
            for key in sorted(_UNIVERSE_ACTION_RISK_KEYS)
            if int(candidate_action_risk_controls.get(key, 0)) != int(baseline_action_risk_controls.get(key, 0))
        ),
        "universe_environment_delta_count": _environment_assumption_delta_count(
            baseline_environment_assumptions,
            candidate_environment_assumptions,
        ),
        "universe_invariant_delta_count": len(set(candidate_invariants) - set(baseline_invariants)),
        "universe_forbidden_pattern_delta_count": len(
            set(candidate_forbidden_patterns) - set(baseline_forbidden_patterns)
        ),
        "universe_preferred_prefix_delta_count": len(
            set(candidate_preferred_prefixes) - set(baseline_preferred_prefixes)
        ),
    }


def _trust_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    improvements = 0
    if float(candidate_controls.get("min_success_rate", 0.0)) > float(baseline_controls.get("min_success_rate", 0.0)):
        improvements += 1
    if float(candidate_controls.get("max_unsafe_ambiguous_rate", 1.0)) < float(
        baseline_controls.get("max_unsafe_ambiguous_rate", 1.0)
    ):
        improvements += 1
    if float(candidate_controls.get("max_hidden_side_effect_rate", 1.0)) < float(
        baseline_controls.get("max_hidden_side_effect_rate", 1.0)
    ):
        improvements += 1
    if float(candidate_controls.get("max_success_hidden_side_effect_rate", 1.0)) < float(
        baseline_controls.get("max_success_hidden_side_effect_rate", 1.0)
    ):
        improvements += 1
    if int(candidate_controls.get("min_distinct_families", 0)) > int(baseline_controls.get("min_distinct_families", 0)):
        improvements += 1
    if int(candidate_controls.get("breadth_min_reports", 0)) > int(baseline_controls.get("breadth_min_reports", 0)):
        improvements += 1
    candidate_families = set(candidate_controls.get("required_benchmark_families", []))
    baseline_families = set(baseline_controls.get("required_benchmark_families", []))
    if candidate_families > baseline_families:
        improvements += 1
    return improvements


def _trust_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    return {
        "trust_control_improvement_count": _trust_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
        "required_family_count_delta": len(candidate_controls.get("required_benchmark_families", []))
        - len(baseline_controls.get("required_benchmark_families", [])),
        "min_success_rate_delta": float(candidate_controls.get("min_success_rate", 0.0))
        - float(baseline_controls.get("min_success_rate", 0.0)),
        "max_unsafe_ambiguous_rate_delta": float(candidate_controls.get("max_unsafe_ambiguous_rate", 0.0))
        - float(baseline_controls.get("max_unsafe_ambiguous_rate", 0.0)),
        "max_hidden_side_effect_rate_delta": float(candidate_controls.get("max_hidden_side_effect_rate", 0.0))
        - float(baseline_controls.get("max_hidden_side_effect_rate", 0.0)),
        "max_success_hidden_side_effect_rate_delta": float(
            candidate_controls.get("max_success_hidden_side_effect_rate", 0.0)
        )
        - float(baseline_controls.get("max_success_hidden_side_effect_rate", 0.0)),
    }


def _recovery_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(
        controls,
        bool_fields=(
            "snapshot_before_execution",
            "rollback_on_runner_exception",
            "rollback_on_failed_outcome",
            "rollback_on_safe_stop",
            "verify_post_rollback_file_count",
        ),
        nonnegative_int_fields=("max_post_rollback_file_count",),
    )


def _recovery_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    improvements = _enabled_flag_improvement_count(
        baseline_controls,
        candidate_controls,
        keys=(
            "snapshot_before_execution",
            "rollback_on_runner_exception",
            "rollback_on_failed_outcome",
            "rollback_on_safe_stop",
            "verify_post_rollback_file_count",
        ),
    )
    if int(candidate_controls.get("max_post_rollback_file_count", 0)) < int(
        baseline_controls.get("max_post_rollback_file_count", 0)
    ):
        improvements += 1
    return improvements


def _recovery_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    evidence = {
        "recovery_control_improvement_count": _recovery_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(
        _boolean_control_deltas(
            baseline_controls,
            candidate_controls,
            keys=(
                "snapshot_before_execution",
                "rollback_on_runner_exception",
                "rollback_on_failed_outcome",
                "rollback_on_safe_stop",
                "verify_post_rollback_file_count",
            ),
        )
    )
    evidence.update(
        _integer_control_deltas(
            baseline_controls,
            candidate_controls,
            keys=("max_post_rollback_file_count",),
        )
    )
    return evidence


def _delegation_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(controls, int_fields=tuple(sorted(_DELEGATION_CONTROL_KEYS)))


def _delegation_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    return _increased_int_control_count(
        baseline_controls,
        candidate_controls,
        keys=tuple(sorted(_DELEGATION_CONTROL_KEYS)),
    )


def _delegation_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    evidence = {
        "delegation_control_improvement_count": _delegation_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(
        _integer_control_deltas(
            baseline_controls,
            candidate_controls,
            keys=tuple(sorted(_DELEGATION_CONTROL_KEYS)),
        )
    )
    return evidence


def _operator_policy_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(
        controls,
        bool_fields=(
            "unattended_allow_git_commands",
            "unattended_allow_http_requests",
            "unattended_allow_generated_path_mutations",
        ),
        positive_int_fields=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        list_fields=("unattended_allowed_benchmark_families", "unattended_generated_path_prefixes"),
        lowercase_list_fields=("unattended_http_allowed_hosts",),
    )


def _operator_policy_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    return (
        _expanded_sequence_control_count(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allowed_benchmark_families",
                "unattended_http_allowed_hosts",
                "unattended_generated_path_prefixes",
            ),
        )
        + _enabled_flag_improvement_count(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allow_git_commands",
                "unattended_allow_http_requests",
                "unattended_allow_generated_path_mutations",
            ),
        )
        + _increased_int_control_count(
            baseline_controls,
            candidate_controls,
            keys=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        )
    )


def _operator_policy_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    evidence = {
        "operator_policy_improvement_count": _operator_policy_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(
        _sequence_length_deltas(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allowed_benchmark_families",
                "unattended_http_allowed_hosts",
                "unattended_generated_path_prefixes",
            ),
        )
    )
    rename_map = {
        "unattended_allowed_benchmark_families_count_delta": "allowed_benchmark_family_count_delta",
        "unattended_http_allowed_hosts_count_delta": "http_allowed_host_count_delta",
        "unattended_generated_path_prefixes_count_delta": "generated_path_prefix_count_delta",
    }
    for source_key, target_key in rename_map.items():
        if source_key in evidence:
            evidence[target_key] = evidence.pop(source_key)
    return evidence


def _enabled_flag_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if bool(candidate_controls.get(key, False)) and not bool(baseline_controls.get(key, False))
    )


def _increased_int_control_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if int(candidate_controls.get(key, 0)) > int(baseline_controls.get(key, 0))
    )


def _expanded_sequence_control_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if set(candidate_controls.get(key, [])) > set(baseline_controls.get(key, []))
    )


def _boolean_control_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_delta": int(bool(candidate_controls.get(key, False))) - int(bool(baseline_controls.get(key, False)))
        for key in keys
    }


def _integer_control_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_delta": int(candidate_controls.get(key, 0)) - int(baseline_controls.get(key, 0))
        for key in keys
    }


def _sequence_length_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_count_delta": len(candidate_controls.get(key, [])) - len(baseline_controls.get(key, []))
        for key in keys
    }


def _transition_model_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    return retained_transition_model_controls(payload)


def _transition_model_signatures_from_payload(payload: dict[str, object] | None) -> list[dict[str, object]]:
    return retained_transition_model_signatures(payload)


def _transition_model_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    baseline_signatures: list[dict[str, object]],
    candidate_signatures: list[dict[str, object]],
) -> int:
    improvements = 0
    for key in (
        "repeat_command_penalty",
        "regressed_path_command_penalty",
        "recovery_command_bonus",
        "progress_command_bonus",
    ):
        if int(candidate_controls.get(key, 0)) > int(baseline_controls.get(key, 0)):
            improvements += 1
    if int(candidate_controls.get("max_signatures", 0)) > int(baseline_controls.get("max_signatures", 0)):
        improvements += 1
    baseline_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in baseline_signatures
    }
    candidate_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in candidate_signatures
    }
    if candidate_signature_keys > baseline_signature_keys:
        improvements += 1
    if len(candidate_signatures) > len(baseline_signatures):
        improvements += 1
    return improvements


def _transition_model_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    baseline_signatures: list[dict[str, object]],
    candidate_signatures: list[dict[str, object]],
) -> dict[str, object]:
    baseline_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in baseline_signatures
    }
    candidate_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in candidate_signatures
    }
    return {
        "transition_model_improvement_count": _transition_model_improvement_count(
            baseline_controls,
            candidate_controls,
            baseline_signatures=baseline_signatures,
            candidate_signatures=candidate_signatures,
        ),
        "transition_signature_count": len(candidate_signatures),
        "transition_signature_count_delta": len(candidate_signatures) - len(baseline_signatures),
        "transition_signature_growth": len(candidate_signature_keys - baseline_signature_keys),
    }


def _capability_surface_evidence(
    baseline_summary: dict[str, int],
    candidate_summary: dict[str, int],
) -> dict[str, object]:
    return {
        "module_count": int(candidate_summary.get("module_count", 0)),
        "enabled_module_count": int(candidate_summary.get("enabled_module_count", 0)),
        "external_capability_count": int(candidate_summary.get("external_capability_count", 0)),
        "improvement_surface_count": int(candidate_summary.get("improvement_surface_count", 0)),
        "module_count_delta": int(candidate_summary.get("module_count", 0))
        - int(baseline_summary.get("module_count", 0)),
        "enabled_module_count_delta": int(candidate_summary.get("enabled_module_count", 0))
        - int(baseline_summary.get("enabled_module_count", 0)),
        "external_capability_count_delta": int(candidate_summary.get("external_capability_count", 0))
        - int(baseline_summary.get("external_capability_count", 0)),
        "improvement_surface_count_delta": int(candidate_summary.get("improvement_surface_count", 0))
        - int(baseline_summary.get("improvement_surface_count", 0)),
    }


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
) -> dict[str, object]:
    candidate_artifact_path = artifact_path
    live_artifact_path = active_artifact_path if active_artifact_path is not None else artifact_path
    staged_candidate = candidate_artifact_path != live_artifact_path
    payload = _load_json_payload(candidate_artifact_path)
    previous_sha256 = artifact_sha256(live_artifact_path)
    rollback_snapshot_path = _snapshot_artifact(live_artifact_path, cycle_id=cycle_id, stage="pre_decision_active")
    artifact_kind = str(payload.get("artifact_kind", "")) if isinstance(payload, dict) else ""
    previous_lifecycle_state = (
        str(payload.get("lifecycle_state", "")).strip() if isinstance(payload, dict) else ""
    )
    compatibility = assess_artifact_compatibility(
        subsystem=subsystem,
        payload=payload,
        capability_modules_path=capability_modules_path,
    )
    lifecycle_state = "retained" if decision_state == "retain" else "rejected"
    synchronized_artifact_paths: dict[str, str] = {}
    tolbert_rejected_gc: dict[str, object] | None = None
    tolbert_rejected_output_dir = ""

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
            _update_tool_candidate_states(
                payload,
                decision_state=decision_state,
                lifecycle_state=lifecycle_state,
            )
        if subsystem == "tolbert_model":
            _stamp_tolbert_lineage_metadata(
                payload,
                decision_state=decision_state,
                cycle_id=cycle_id,
                live_artifact_path=live_artifact_path,
                parent_artifact_sha256=previous_sha256,
            )
            if decision_state == "retain":
                _promote_tolbert_payload_to_canonical_checkpoint(
                    payload,
                    live_artifact_path=live_artifact_path,
                    cycle_id=cycle_id,
                )
            else:
                tolbert_rejected_output_dir = str(payload.get("output_dir", "")).strip()
                payload = _compact_rejected_tolbert_payload(payload)

    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if subsystem == "tolbert_model" and decision_state != "retain" and isinstance(payload, dict):
        tolbert_rejected_gc = _cleanup_rejected_tolbert_payload_artifacts(
            candidate_artifact_path=candidate_artifact_path,
            active_artifact_path=live_artifact_path,
            output_dir=tolbert_rejected_output_dir,
        )
        payload["rejected_storage_gc"] = tolbert_rejected_gc
        candidate_artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    candidate_artifact_snapshot_path = _snapshot_artifact(
        candidate_artifact_path,
        cycle_id=cycle_id,
        stage=f"post_{decision_state}_candidate",
    )
    restored_live_artifact = False
    active_rollback_source = _prior_active_artifact_path(payload)
    active_artifact_snapshot_path = candidate_artifact_snapshot_path
    if decision_state == "retain":
        live_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_candidate:
            shutil.copy2(candidate_artifact_path, live_artifact_path)
        synchronized_artifact_paths = _synchronize_retained_universe_artifacts(
            subsystem=subsystem,
            payload=payload,
            live_artifact_path=live_artifact_path,
            runtime_config=runtime_config,
        )
        active_artifact_snapshot_path = _snapshot_artifact(
            live_artifact_path,
            cycle_id=cycle_id,
            stage="post_retain_active",
        )
    elif not staged_candidate and active_rollback_source is not None and active_rollback_source.exists():
        shutil.copy2(active_rollback_source, live_artifact_path)
        restored_live_artifact = True
        active_artifact_snapshot_path = candidate_artifact_snapshot_path
    current_sha256 = artifact_sha256(live_artifact_path)
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
    }


def _synchronize_retained_universe_artifacts(
    *,
    subsystem: str,
    payload: dict[str, object] | None,
    live_artifact_path: Path,
    runtime_config: KernelConfig | None,
) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    if subsystem not in {"universe", "universe_constitution", "operating_envelope"}:
        return {}
    paths = _universe_sync_paths(live_artifact_path=live_artifact_path, runtime_config=runtime_config)
    if not paths:
        return {}

    constitution_path = paths["universe_constitution"]
    envelope_path = paths["operating_envelope"]
    contract_path = paths["universe"]

    if subsystem == "universe":
        bundle = compose_universe_bundle_payloads(
            constitution_payload=payload,
            operating_envelope_payload=payload,
            baseline_payload=payload,
            lifecycle_state="retained",
        )
    else:
        baseline_payload = _load_first_retained_universe_payload(contract_path, artifact_kind="universe_contract")
        current_constitution = (
            payload
            if subsystem == "universe_constitution"
            else _load_first_retained_universe_payload(
                constitution_path,
                artifact_kind="universe_constitution",
                fallback_paths=(contract_path,),
            )
        )
        current_envelope = (
            payload
            if subsystem == "operating_envelope"
            else _load_first_retained_universe_payload(
                envelope_path,
                artifact_kind="operating_envelope",
                fallback_paths=(contract_path,),
            )
        )
        bundle = compose_universe_bundle_payloads(
            constitution_payload=current_constitution,
            operating_envelope_payload=current_envelope,
            baseline_payload=baseline_payload,
            lifecycle_state="retained",
        )
    return write_universe_bundle_files(
        contract_path=contract_path,
        constitution_path=constitution_path,
        operating_envelope_path=envelope_path,
        bundle=bundle,
    )


def _universe_sync_paths(*, live_artifact_path: Path, runtime_config: KernelConfig | None) -> dict[str, Path]:
    if runtime_config is not None:
        return universe_bundle_paths(
            universe_contract_path=active_artifact_path_for_subsystem(runtime_config, "universe"),
            universe_constitution_path=active_artifact_path_for_subsystem(runtime_config, "universe_constitution"),
            operating_envelope_path=active_artifact_path_for_subsystem(runtime_config, "operating_envelope"),
        )
    return sibling_universe_bundle_paths(live_artifact_path)


def _runtime_config_for_universe_sync(
    runtime_config: KernelConfig | None,
    live_artifact_path: Path,
) -> KernelConfig | None:
    if runtime_config is None:
        return None
    configured_paths = universe_bundle_paths(
        universe_contract_path=active_artifact_path_for_subsystem(runtime_config, "universe"),
        universe_constitution_path=active_artifact_path_for_subsystem(runtime_config, "universe_constitution"),
        operating_envelope_path=active_artifact_path_for_subsystem(runtime_config, "operating_envelope"),
    )
    if universe_bundle_contains_path(configured_paths, live_artifact_path):
        return runtime_config
    return None


def _load_first_retained_universe_payload(
    artifact_path: Path,
    *,
    artifact_kind: str,
    fallback_paths: tuple[Path, ...] = (),
) -> dict[str, object] | None:
    for candidate_path in (artifact_path, *fallback_paths):
        if not candidate_path.exists():
            continue
        try:
            loaded = _load_json_payload(candidate_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        loaded_kind = str(loaded.get("artifact_kind", "")).strip() or artifact_kind
        retained = retained_artifact_payload(loaded, artifact_kind=loaded_kind)
        if retained is not None:
            return retained
    return None


def _stamp_tolbert_lineage_metadata(
    payload: dict[str, object],
    *,
    decision_state: str,
    cycle_id: str,
    live_artifact_path: Path,
    parent_artifact_sha256: str,
) -> None:
    lineage = payload.get("lineage", {})
    if not isinstance(lineage, dict):
        lineage = {}
    lineage["mode"] = str(lineage.get("mode", "canonical_parent_mutation")).strip() or "canonical_parent_mutation"
    lineage["canonical_artifact_path"] = str(live_artifact_path)
    if parent_artifact_sha256:
        lineage["parent_artifact_sha256"] = parent_artifact_sha256
        lineage["parent_artifact_path"] = str(live_artifact_path)
    lineage["checkpoint_materialization_policy"] = "evaluate_or_promote_only"
    lineage["promotion_policy"] = "canonical_replace_on_retain"
    lineage["rejected_candidate_policy"] = "metrics_and_mutation_record_only"
    if decision_state == "retain":
        lineage["promotion"] = {
            "cycle_id": cycle_id,
            "state": "promoted_to_canonical",
            "squash_strategy": "canonical_replace",
        }
    payload["lineage"] = lineage


def _compact_rejected_tolbert_payload(payload: dict[str, object]) -> dict[str, object]:
    compact = {
        "spec_version": str(payload.get("spec_version", "asi_v1")).strip() or "asi_v1",
        "artifact_kind": str(payload.get("artifact_kind", "tolbert_model_bundle")).strip() or "tolbert_model_bundle",
        "lifecycle_state": str(payload.get("lifecycle_state", "rejected")).strip() or "rejected",
        "generation_focus": str(payload.get("generation_focus", "")).strip(),
        "model_surfaces": _strip_pathlike_fields(payload.get("model_surfaces", {})),
        "runtime_policy": _strip_pathlike_fields(payload.get("runtime_policy", {})),
        "decoder_policy": _strip_pathlike_fields(payload.get("decoder_policy", {})),
        "action_generation_policy": _strip_pathlike_fields(payload.get("action_generation_policy", {})),
        "rollout_policy": _strip_pathlike_fields(payload.get("rollout_policy", {})),
        "liftoff_gate": _strip_pathlike_fields(payload.get("liftoff_gate", {})),
        "build_policy": _strip_pathlike_fields(payload.get("build_policy", {})),
        "retention_gate": _strip_pathlike_fields(payload.get("retention_gate", {})),
        "training_controls": _strip_pathlike_fields(payload.get("training_controls", {})),
        "dataset_manifest": _strip_pathlike_fields(payload.get("dataset_manifest", {})),
        "universal_dataset_manifest": _strip_pathlike_fields(payload.get("universal_dataset_manifest", {})),
        "hybrid_runtime": _strip_pathlike_fields(payload.get("hybrid_runtime", {})),
        "universal_decoder_runtime": _strip_pathlike_fields(payload.get("universal_decoder_runtime", {})),
        "universal_decoder_training_controls": _strip_pathlike_fields(
            payload.get("universal_decoder_training_controls", {})
        ),
        "proposals": _strip_pathlike_fields(payload.get("proposals", [])),
        "parameter_delta": _strip_pathlike_fields(payload.get("parameter_delta", {})),
        "compatibility": _strip_pathlike_fields(payload.get("compatibility", {})),
        "retention_decision": _strip_pathlike_fields(payload.get("retention_decision", {})),
        "rollback_artifact_path": str(payload.get("rollback_artifact_path", "")).strip(),
        "lineage": _strip_pathlike_fields(payload.get("lineage", {})),
        "generated_at": str(payload.get("generated_at", "")).strip(),
        "materialization_mode": "rejected_manifest_only",
    }
    compact["rejected_compaction"] = {
        "dropped_fields": [
            "runtime_paths",
            "output_dir",
            "job_records",
            "shared_store",
            "storage_compaction",
            "external_training_backends",
        ],
        "checkpoint_materialization_policy": "metrics_and_mutation_record_only",
    }
    return compact


def _promote_tolbert_payload_to_canonical_checkpoint(
    payload: dict[str, object],
    *,
    live_artifact_path: Path,
    cycle_id: str,
) -> None:
    runtime_paths = payload.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        return
    delta_path_value = str(runtime_paths.get("checkpoint_delta_path", "")).strip()
    parent_path_value = str(runtime_paths.get("parent_checkpoint_path", "")).strip()
    if not delta_path_value or not parent_path_value:
        return
    delta_path = Path(delta_path_value)
    parent_path = Path(parent_path_value)
    if not delta_path.exists() or not parent_path.exists():
        return
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    canonical_checkpoint_path = live_artifact_path.parent / "checkpoints" / f"tolbert_{safe_cycle_id}.pt"
    canonical_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_path,
        delta_checkpoint_path=delta_path,
        output_checkpoint_path=canonical_checkpoint_path,
    )
    runtime_paths["checkpoint_path"] = str(canonical_checkpoint_path)
    runtime_paths.pop("checkpoint_delta_path", None)
    runtime_paths.pop("parent_checkpoint_path", None)
    parameter_delta = payload.get("parameter_delta", {})
    if isinstance(parameter_delta, dict):
        parameter_delta["promotion_applied"] = True
        parameter_delta["canonical_checkpoint_path"] = str(canonical_checkpoint_path)
        parameter_delta.pop("delta_checkpoint_path", None)
        parameter_delta.pop("parent_checkpoint_path", None)
        payload["parameter_delta"] = parameter_delta
    lineage = payload.get("lineage", {})
    if isinstance(lineage, dict):
        promotion = lineage.get("promotion", {})
        if not isinstance(promotion, dict):
            promotion = {}
        promotion["canonical_checkpoint_path"] = str(canonical_checkpoint_path)
        promotion["state"] = "promoted_to_canonical"
        promotion["squash_strategy"] = "materialized_parent_plus_delta"
        lineage["promotion"] = promotion
        payload["lineage"] = lineage


def _strip_pathlike_fields(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in {
                "path",
                "paths",
                "root",
                "roots",
                "output_dir",
                "shared_store",
                "runtime_paths",
                "job_records",
                "storage_compaction",
                "external_training_backends",
            }:
                continue
            if lowered.endswith("_path") or lowered.endswith("_paths"):
                continue
            normalized[key] = _strip_pathlike_fields(item)
        return normalized
    if isinstance(value, list):
        return [_strip_pathlike_fields(item) for item in value]
    return value


def _cleanup_rejected_tolbert_payload_artifacts(
    *,
    candidate_artifact_path: Path,
    active_artifact_path: Path,
    output_dir: str,
) -> dict[str, object]:
    removed_output_dir = ""
    candidate_output_dir = Path(output_dir) if str(output_dir).strip() else None
    if candidate_output_dir is not None and candidate_output_dir.exists():
        shutil.rmtree(candidate_output_dir, ignore_errors=True)
        if not candidate_output_dir.exists():
            removed_output_dir = str(candidate_output_dir)

    store_root = active_artifact_path.parent / "store"
    removed_shared_store = _cleanup_unreferenced_tolbert_store(
        store_root=store_root,
        active_artifact_path=active_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
    )
    return {
        "removed_output_dir": removed_output_dir,
        "removed_shared_store": removed_shared_store,
    }


def _cleanup_unreferenced_tolbert_store(
    *,
    store_root: Path,
    active_artifact_path: Path,
    candidate_artifact_path: Path,
) -> list[str]:
    if not store_root.exists():
        return []
    referenced = _tolbert_store_references_from_paths(
        _tolbert_reference_artifact_paths(
            active_artifact_path=active_artifact_path,
            candidate_artifact_path=candidate_artifact_path,
        )
    )
    removed: list[str] = []
    for group_dir in sorted(path for path in store_root.iterdir() if path.is_dir()):
        for digest_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            try:
                resolved = str(digest_dir.resolve())
            except OSError:
                resolved = str(digest_dir)
            if resolved in referenced:
                continue
            shutil.rmtree(digest_dir, ignore_errors=True)
            if not digest_dir.exists():
                removed.append(str(digest_dir))
        try:
            next(group_dir.iterdir())
        except StopIteration:
            try:
                group_dir.rmdir()
            except OSError:
                pass
    try:
        next(store_root.iterdir())
    except StopIteration:
        try:
            store_root.rmdir()
        except OSError:
            pass
    return removed


def _tolbert_reference_artifact_paths(*, active_artifact_path: Path, candidate_artifact_path: Path) -> list[Path]:
    paths: list[Path] = []
    if active_artifact_path.exists():
        paths.append(active_artifact_path)
    if candidate_artifact_path.exists():
        paths.append(candidate_artifact_path)
    tolbert_candidates_root = None
    if len(candidate_artifact_path.parents) >= 2 and candidate_artifact_path.parents[1].name == "tolbert_model":
        tolbert_candidates_root = candidate_artifact_path.parents[1]
    if tolbert_candidates_root is not None and tolbert_candidates_root.exists():
        for path in sorted(tolbert_candidates_root.glob("*/*.json")):
            if path not in paths and path.exists():
                paths.append(path)
    return paths


def _tolbert_store_references_from_paths(paths: list[Path]) -> set[str]:
    referenced: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        referenced.update(_tolbert_shared_store_paths_from_payload(payload))
    return referenced


def _tolbert_shared_store_paths_from_payload(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    shared_store = payload.get("shared_store", {})
    if not isinstance(shared_store, dict):
        return set()
    entries = shared_store.get("entries", {})
    if not isinstance(entries, dict):
        return set()
    referenced: set[str] = set()
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            referenced.add(str(Path(raw_path).resolve()))
        except OSError:
            referenced.add(raw_path)
    return referenced


def persist_replay_verified_tool_artifact(artifact_path: Path, *, cycle_id: str = "manual") -> dict[str, object]:
    payload = _load_json_payload(artifact_path)
    previous_sha256 = artifact_sha256(artifact_path)
    rollback_snapshot_path = _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage="pre_replay_verified")
    replay_verified_payload = materialize_replay_verified_tool_payload(payload)
    replay_verified_payload["lifecycle_state"] = "replay_verified"
    replay_verified_payload["rollback_artifact_path"] = str(rollback_snapshot_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(replay_verified_payload, indent=2), encoding="utf-8")
    artifact_snapshot_path = _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage="post_replay_verified")
    return {
        "artifact_kind": str(replay_verified_payload.get("artifact_kind", "")),
        "artifact_lifecycle_state": "replay_verified",
        "artifact_sha256": artifact_sha256(artifact_path),
        "previous_artifact_sha256": previous_sha256,
        "rollback_artifact_path": str(rollback_snapshot_path),
        "artifact_snapshot_path": str(artifact_snapshot_path),
    }


def effective_artifact_payload_for_retention(
    subsystem: str,
    payload: Any,
    *,
    capability_modules_path: Path | None = None,
) -> Any:
    effective_subsystem = base_subsystem_for(subsystem, capability_modules_path)
    if effective_subsystem != "tooling" or not isinstance(payload, dict):
        return payload
    gate = _retention_gate(effective_subsystem, payload)
    if not bool(gate.get("require_replay_verification", True)):
        return payload
    if _tool_candidates_have_stage(payload, "replay_verified"):
        return payload
    replay_verified_payload = materialize_replay_verified_tool_payload(payload)
    replay_verified_payload["lifecycle_state"] = "replay_verified"
    return replay_verified_payload


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and int(value) > 0


def _operator_task_contract(record: dict[str, object]) -> dict[str, object]:
    task_contract = record.get("task_contract")
    if isinstance(task_contract, dict) and task_contract:
        return task_contract
    template_contract = record.get("template_contract")
    if isinstance(template_contract, dict) and template_contract:
        return template_contract
    return {}


def _operator_steps(record: dict[str, object]) -> list[str]:
    steps = record.get("steps")
    if isinstance(steps, list) and any(str(step).strip() for step in steps):
        return [str(step) for step in steps if str(step).strip()]
    template_procedure = record.get("template_procedure")
    if isinstance(template_procedure, dict):
        commands = template_procedure.get("commands", [])
        if isinstance(commands, list):
            return [str(command) for command in commands if str(command).strip()]
    return []


def _operator_benchmark_families(record: dict[str, object]) -> list[str]:
    benchmark_families = record.get("benchmark_families")
    if isinstance(benchmark_families, list) and any(str(value).strip() for value in benchmark_families):
        return [str(value) for value in benchmark_families if str(value).strip()]
    applicable_families = record.get("applicable_benchmark_families")
    if isinstance(applicable_families, list):
        return [str(value) for value in applicable_families if str(value).strip()]
    return []


def _operator_support(record: dict[str, object]) -> int:
    support = record.get("support")
    if _is_positive_int(support):
        return int(support)
    support_count = record.get("support_count")
    if _is_positive_int(support_count):
        return int(support_count)
    return 0


def _tool_candidate_stage(stage: object) -> str:
    return str(stage).strip()


def _tool_candidate_lifecycle_state(candidate: dict[str, object]) -> str:
    return str(candidate.get("lifecycle_state", "")).strip()


def _allowed_artifact_lifecycle_states(subsystem: str) -> set[str]:
    if subsystem in {
        "benchmark",
        "retrieval",
        "verifier",
        "policy",
        "universe",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "curriculum",
        "capabilities",
    }:
        return {"proposed", "retained", "rejected"}
    if subsystem == "tolbert_model":
        return {"candidate", "retained", "rejected"}
    if subsystem == "tooling":
        return {"candidate", "replay_verified", "retained", "rejected"}
    if subsystem in {"skills", "operators"}:
        return {"promoted", "retained", "rejected"}
    return set()


def assess_artifact_compatibility(
    *,
    subsystem: str,
    payload: Any,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    subsystem = base_subsystem_for(subsystem, capability_modules_path)
    checks: list[str] = []
    violations: list[str] = []
    manifest_paths = current_external_task_manifests_paths()
    try:
        bank = TaskBank(
            config=KernelConfig(),
            external_task_manifests=manifest_paths if manifest_paths else None,
        )
    except TypeError:
        bank = TaskBank()

    if not isinstance(payload, dict):
        return {
            "compatible": False,
            "checked_rules": ["artifact payload must be a JSON object"],
            "violations": ["artifact payload is not a JSON object"],
        }

    if str(payload.get("spec_version", "")).strip() != "asi_v1":
        violations.append("spec_version must be asi_v1")
    checks.append("spec_version")

    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    expected_kind = {
        "benchmark": "benchmark_candidate_set",
        "retrieval": "retrieval_policy_set",
        "tolbert_model": "tolbert_model_bundle",
        "verifier": "verifier_candidate_set",
        "policy": "prompt_proposal_set",
        "universe": ("universe_contract", "universe_constitution", "operating_envelope"),
        "world_model": "world_model_policy_set",
        "state_estimation": "state_estimation_policy_set",
        "trust": "trust_policy_set",
        "recovery": "recovery_policy_set",
        "delegation": "delegated_runtime_policy_set",
        "operator_policy": "operator_policy_set",
        "transition_model": "transition_model_policy_set",
        "curriculum": "curriculum_proposal_set",
        "capabilities": "capability_module_set",
        "tooling": "tool_candidate_set",
        "skills": "skill_set",
        "operators": "operator_class_set",
    }.get(subsystem)
    if isinstance(expected_kind, tuple):
        if artifact_kind not in expected_kind:
            violations.append(f"artifact_kind must be one of: {', '.join(expected_kind)}")
    elif expected_kind and artifact_kind != expected_kind:
        violations.append(f"artifact_kind must be {expected_kind}")
    checks.append("artifact_kind")

    lifecycle_state = str(payload.get("lifecycle_state", "")).strip()
    if not lifecycle_state:
        violations.append("artifact must contain a lifecycle_state")
    else:
        allowed_lifecycle_states = _allowed_artifact_lifecycle_states(subsystem)
        if allowed_lifecycle_states and lifecycle_state not in allowed_lifecycle_states:
            allowed_text = ", ".join(sorted(allowed_lifecycle_states))
            violations.append(f"artifact lifecycle_state must be one of: {allowed_text}")
    checks.append("lifecycle_state")

    if subsystem in {
        "benchmark",
        "retrieval",
        "verifier",
        "policy",
        "universe",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "curriculum",
        "capabilities",
    }:
        retention_gate = payload.get("retention_gate", {})
        if not isinstance(retention_gate, dict) or not retention_gate:
            violations.append("artifact must contain a retention_gate")
        checks.append("retention_gate")
    elif subsystem in {"skills", "operators"}:
        retention_gate = payload.get("retention_gate", {})
        if not isinstance(retention_gate, dict) or not retention_gate:
            violations.append("artifact must contain a retention_gate")
        checks.append("retention_gate")

    if subsystem in {
        "benchmark",
        "retrieval",
        "verifier",
        "policy",
        "universe",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "curriculum",
    }:
        proposals = payload.get("proposals", [])
        if not isinstance(proposals, list) or not proposals:
            violations.append("artifact must contain a non-empty proposals list")
        checks.append("proposals")
    if subsystem == "tolbert_model":
        proposals = payload.get("proposals", [])
        runtime_paths = payload.get("runtime_paths", {})
        dataset_manifest = payload.get("dataset_manifest", {})
        if not isinstance(proposals, list) or not proposals:
            violations.append("Tolbert model artifact must contain a non-empty proposals list")
        if not isinstance(runtime_paths, dict):
            violations.append("Tolbert model artifact must contain runtime_paths")
        if not isinstance(dataset_manifest, dict) or int(dataset_manifest.get("total_examples", 0)) <= 0:
            violations.append("Tolbert model artifact must contain a non-empty dataset_manifest")
        checks.append("tolbert_model_surface")
    if subsystem == "benchmark":
        proposals = payload.get("proposals", [])
        for proposal in proposals:
            if not isinstance(proposal, dict):
                violations.append("every benchmark proposal must be an object")
                continue
            if not str(proposal.get("proposal_id", "")).strip():
                violations.append("every benchmark proposal must contain a proposal_id")
            if not str(proposal.get("source_task_id", "")).strip():
                violations.append("every benchmark proposal must contain a source_task_id")
            if not str(proposal.get("benchmark_family", "")).strip():
                violations.append("every benchmark proposal must contain a benchmark_family")
            if not str(proposal.get("kind", "")).strip():
                violations.append("every benchmark proposal must contain a kind")
            if not str(proposal.get("prompt", "")).strip():
                violations.append("every benchmark proposal must contain a prompt")
            if not (
                isinstance(proposal.get("failure_types"), list)
                or isinstance(proposal.get("transition_failures"), list)
                or proposal.get("command_count") is not None
            ):
                violations.append(
                    "every benchmark proposal must contain discriminative source details such as failure_types, transition_failures, or command_count"
                )
        checks.append("benchmark_proposals")
    if subsystem == "tooling":
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            violations.append("artifact must contain a non-empty candidates list")
        checks.append("candidates")
    if subsystem == "skills":
        skills = payload.get("skills", [])
        if not isinstance(skills, list) or not skills:
            violations.append("artifact must contain a non-empty skills list")
        else:
            for skill in skills:
                if not isinstance(skill, dict):
                    violations.append("every skill must be an object")
                    continue
                if not str(skill.get("skill_id", "")).strip():
                    violations.append("every skill must contain a skill_id")
                if not str(skill.get("source_task_id", "")).strip():
                    violations.append("every skill must contain a source_task_id")
                if not str(skill.get("benchmark_family", "")).strip():
                    violations.append("every skill must contain a benchmark_family")
                quality = skill.get("quality")
                if isinstance(quality, bool) or not isinstance(quality, (int, float)):
                    violations.append("every skill must contain a numeric quality")
                if not isinstance(skill.get("procedure"), dict):
                    violations.append("every skill must contain a procedure object")
                if not isinstance(skill.get("task_contract"), dict):
                    violations.append("every skill must contain a task_contract object")
                if not isinstance(skill.get("verifier"), dict):
                    violations.append("every skill must contain a verifier object")
        checks.append("skills")
        checks.append("skill_contracts")
    if subsystem == "operators":
        operators = payload.get("operators", [])
        if not isinstance(operators, list) or not operators:
            violations.append("artifact must contain a non-empty operators list")
        checks.append("operators")
    if subsystem == "capabilities":
        modules = payload.get("modules", [])
        if not isinstance(modules, list) or not modules:
            violations.append("artifact must contain a non-empty modules list")
        checks.append("modules")

    if subsystem == "verifier":
        proposals = payload.get("proposals", [])
        for proposal in proposals:
            if not isinstance(proposal, dict):
                violations.append("every verifier proposal must be an object")
                continue
            if not str(proposal.get("proposal_id", "")).strip():
                violations.append("every verifier proposal must contain a proposal_id")
            if not str(proposal.get("source_task_id", "")).strip():
                violations.append("every verifier proposal must contain a source_task_id")
            if not str(proposal.get("benchmark_family", "")).strip():
                violations.append("every verifier proposal must contain a benchmark_family")
        if any(not isinstance(proposal.get("contract"), dict) for proposal in proposals if isinstance(proposal, dict)):
            violations.append("every verifier proposal must contain a contract object")
        checks.append("verifier_contracts")
    if subsystem == "policy":
        proposals = payload.get("proposals", [])
        for proposal in proposals:
            if not isinstance(proposal, dict):
                violations.append("every prompt proposal must be an object")
                continue
            if not str(proposal.get("area", "")).strip():
                violations.append("every prompt proposal must contain an area")
            if not _is_positive_int(proposal.get("priority", 0)):
                violations.append("every prompt proposal must contain a positive integer priority")
            if not str(proposal.get("reason", "")).strip():
                violations.append("every prompt proposal must contain a reason")
            if not str(proposal.get("suggestion", "")).strip():
                violations.append("every prompt proposal must contain a suggestion")
        checks.append("prompt_proposals")
    if subsystem == "curriculum":
        proposals = payload.get("proposals", [])
        for proposal in proposals:
            if not isinstance(proposal, dict):
                violations.append("every curriculum proposal must be an object")
                continue
            if not str(proposal.get("area", "")).strip():
                violations.append("every curriculum proposal must contain an area")
            if not _is_positive_int(proposal.get("priority", 0)):
                violations.append("every curriculum proposal must contain a positive integer priority")
            if not str(proposal.get("reason", "")).strip():
                violations.append("every curriculum proposal must contain a reason")
            if not str(proposal.get("suggestion", "")).strip():
                violations.append("every curriculum proposal must contain a suggestion")
        checks.append("curriculum_proposals")
    if subsystem == "retrieval":
        proposals = payload.get("proposals", [])
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict):
            violations.append("retrieval artifact must contain an overrides object")
        for proposal in proposals:
            if not isinstance(proposal, dict):
                violations.append("every retrieval proposal must be an object")
                continue
            if not str(proposal.get("proposal_id", "")).strip():
                violations.append("every retrieval proposal must contain a proposal_id")
            if not str(proposal.get("area", "")).strip():
                violations.append("every retrieval proposal must contain an area")
            if not str(proposal.get("reason", "")).strip():
                violations.append("every retrieval proposal must contain a reason")
        if any(not isinstance(proposal.get("overrides"), dict) for proposal in proposals if isinstance(proposal, dict)):
            violations.append("every retrieval proposal must contain an overrides object")
        checks.append("retrieval_overrides")
    if subsystem == "tolbert_model":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _TOLBERT_MODEL_GENERATION_FOCI:
            violations.append("generation_focus must be a supported tolbert_model focus")
        training_controls = payload.get("training_controls", {})
        if not isinstance(training_controls, dict) or not training_controls:
            violations.append("Tolbert model artifact must contain training_controls")
        model_surfaces = payload.get("model_surfaces", {})
        if not isinstance(model_surfaces, dict) or not model_surfaces:
            violations.append("Tolbert model artifact must contain model_surfaces")
        else:
            for key, value in model_surfaces.items():
                if key not in _TOLBERT_MODEL_SURFACE_KEYS:
                    violations.append(f"Tolbert model surface is unsupported: {key}")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert model surface {key} must be boolean")
        runtime_policy = payload.get("runtime_policy", {})
        if not isinstance(runtime_policy, dict) or not runtime_policy:
            violations.append("Tolbert model artifact must contain runtime_policy")
        else:
            for key, value in runtime_policy.items():
                if key not in _TOLBERT_RUNTIME_POLICY_KEYS:
                    violations.append(f"Tolbert runtime policy is unsupported: {key}")
                    continue
                if key in {"shadow_benchmark_families", "primary_benchmark_families"}:
                    if not isinstance(value, list):
                        violations.append(f"Tolbert runtime policy {key} must be a list")
                elif key in {"min_path_confidence"}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert runtime policy {key} must be numeric")
                elif key in {"primary_min_command_score"}:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert runtime policy {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert runtime policy {key} must be boolean")
        decoder_policy = payload.get("decoder_policy", {})
        if not isinstance(decoder_policy, dict) or not decoder_policy:
            violations.append("Tolbert model artifact must contain decoder_policy")
        else:
            for key, value in decoder_policy.items():
                if key not in _TOLBERT_DECODER_POLICY_KEYS:
                    violations.append(f"Tolbert decoder policy is unsupported: {key}")
                    continue
                if key == "min_stop_completion_ratio":
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert decoder policy {key} must be numeric")
                elif key == "max_task_suggestions":
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert decoder policy {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert decoder policy {key} must be boolean")
        action_generation_policy = payload.get("action_generation_policy", {})
        if action_generation_policy:
            if not isinstance(action_generation_policy, dict):
                violations.append("Tolbert action generation policy must be an object")
            else:
                for key, value in action_generation_policy.items():
                    if key not in _TOLBERT_ACTION_GENERATION_POLICY_KEYS:
                        violations.append(f"Tolbert action generation policy is unsupported: {key}")
                        continue
                    if key == "template_preferences":
                        if not isinstance(value, dict):
                            violations.append("Tolbert action generation policy template_preferences must be an object")
                            continue
                        for family, items in value.items():
                            if not str(family).strip():
                                violations.append("Tolbert action generation policy family keys must be non-empty")
                            if not isinstance(items, list):
                                violations.append(
                                    "Tolbert action generation policy template_preferences values must be lists"
                                )
                                continue
                            for item in items:
                                if not isinstance(item, dict):
                                    violations.append(
                                        "Tolbert action generation policy template preference entries must be objects"
                                    )
                                    continue
                                if not str(item.get("template_kind", "")).strip():
                                    violations.append(
                                        "Tolbert action generation policy template preference must include template_kind"
                                    )
                                support = item.get("support", 0)
                                if isinstance(support, bool) or not isinstance(support, int):
                                    violations.append(
                                        "Tolbert action generation policy template preference support must be an integer"
                                    )
                                pass_rate = item.get("pass_rate", 0.0)
                                if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
                                    violations.append(
                                        "Tolbert action generation policy template preference pass_rate must be numeric"
                                    )
                    elif key in {"max_candidates", "min_family_support"}:
                        if isinstance(value, bool) or not isinstance(value, int):
                            violations.append(f"Tolbert action generation policy {key} must be an integer")
                    elif key == "enabled":
                        if not isinstance(value, bool):
                            violations.append(f"Tolbert action generation policy {key} must be boolean")
                    elif isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert action generation policy {key} must be numeric")
        rollout_policy = payload.get("rollout_policy", {})
        if not isinstance(rollout_policy, dict) or not rollout_policy:
            violations.append("Tolbert model artifact must contain rollout_policy")
        else:
            for key, value in rollout_policy.items():
                if key not in _TOLBERT_ROLLOUT_POLICY_KEYS:
                    violations.append(f"Tolbert rollout policy is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(f"Tolbert rollout policy {key} must be numeric")
        liftoff_gate = payload.get("liftoff_gate", {})
        if not isinstance(liftoff_gate, dict) or not liftoff_gate:
            violations.append("Tolbert model artifact must contain liftoff_gate")
        else:
            for key, value in liftoff_gate.items():
                if key not in _TOLBERT_LIFTOFF_GATE_KEYS:
                    violations.append(f"Tolbert liftoff gate is unsupported: {key}")
                    continue
                if key == "proposal_gate_by_benchmark_family":
                    if not isinstance(value, dict):
                        violations.append("Tolbert liftoff gate proposal_gate_by_benchmark_family must be an object")
                        continue
                    for family, family_gate in value.items():
                        if not str(family).strip():
                            violations.append(
                                "Tolbert liftoff gate proposal_gate_by_benchmark_family keys must be non-empty"
                            )
                        if not isinstance(family_gate, dict):
                            violations.append(
                                "Tolbert liftoff gate proposal_gate_by_benchmark_family values must be objects"
                            )
                            continue
                        for family_key, family_value in family_gate.items():
                            if family_key not in {
                                "require_novel_command_signal",
                                "min_proposal_selected_steps_delta",
                                "min_novel_valid_command_steps",
                                "min_novel_valid_command_rate_delta",
                            }:
                                violations.append(
                                    f"Tolbert liftoff family proposal gate is unsupported: {family_key}"
                                )
                                continue
                            if family_key == "require_novel_command_signal":
                                if not isinstance(family_value, bool):
                                    violations.append(
                                        "Tolbert liftoff family proposal gate require_novel_command_signal must be boolean"
                                    )
                            elif family_key == "min_novel_valid_command_rate_delta":
                                if isinstance(family_value, bool) or not isinstance(family_value, (int, float)):
                                    violations.append(
                                        "Tolbert liftoff family proposal gate min_novel_valid_command_rate_delta must be numeric"
                                    )
                            elif isinstance(family_value, bool) or not isinstance(family_value, int):
                                violations.append(
                                    f"Tolbert liftoff family proposal gate {family_key} must be an integer"
                                )
                elif key in {
                    "min_pass_rate_delta",
                    "max_step_regression",
                    "max_takeover_drift_pass_rate_regression",
                    "max_takeover_drift_unsafe_ambiguous_rate_regression",
                    "max_takeover_drift_hidden_side_effect_rate_regression",
                    "max_takeover_drift_trust_success_rate_regression",
                    "max_takeover_drift_trust_unsafe_ambiguous_rate_regression",
                }:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert liftoff gate {key} must be numeric")
                elif key in {
                    "max_regressed_families",
                    "min_shadow_episodes_per_promoted_family",
                    "takeover_drift_step_budget",
                    "takeover_drift_wave_task_limit",
                    "takeover_drift_max_waves",
                }:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert liftoff gate {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert liftoff gate {key} must be boolean")
        build_policy = payload.get("build_policy", {})
        if not isinstance(build_policy, dict) or not build_policy:
            violations.append("Tolbert model artifact must contain build_policy")
        else:
            for key, value in build_policy.items():
                if key not in _TOLBERT_BUILD_POLICY_KEYS:
                    violations.append(f"Tolbert build policy is unsupported: {key}")
                    continue
                if key in {
                    "allow_kernel_autobuild",
                    "allow_kernel_rebuild",
                    "require_synthetic_dataset",
                }:
                    if not isinstance(value, bool):
                        violations.append(f"Tolbert build policy {key} must be boolean")
                else:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert build policy {key} must be an integer")
        runtime_paths = payload.get("runtime_paths", {})
        if isinstance(runtime_paths, dict):
            for key in (
                "config_path",
                "checkpoint_path",
                "nodes_path",
                "label_map_path",
                "source_spans_paths",
                "cache_paths",
            ):
                if not runtime_paths.get(key):
                    violations.append(f"Tolbert model runtime_paths must include {key}")
        checks.append("tolbert_model_surface")
        checks.append("tolbert_model_runtime_paths")
    if subsystem == "universe":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _UNIVERSE_GENERATION_FOCI:
            violations.append("generation_focus must be a supported universe focus")
        artifact_kind = str(payload.get("artifact_kind", "")).strip()
        control_schema = str(payload.get("control_schema", "")).strip()
        if artifact_kind == "universe_constitution":
            if control_schema != "universe_constitution_v1":
                violations.append("universe constitution artifacts must declare control_schema universe_constitution_v1")
        elif artifact_kind == "operating_envelope":
            if control_schema != "operating_envelope_v1":
                violations.append("operating envelope artifacts must declare control_schema operating_envelope_v1")
        elif control_schema != "universe_contract_v1":
            violations.append("universe artifacts must declare control_schema universe_contract_v1")
        governance = payload.get("governance", {})
        if artifact_kind != "operating_envelope" and (not isinstance(governance, dict) or not governance):
            violations.append("universe artifact must contain a non-empty governance object")
        elif isinstance(governance, dict) and governance:
            for key, value in governance.items():
                if key not in _UNIVERSE_GOVERNANCE_KEYS:
                    violations.append(f"universe governance control is unsupported: {key}")
                    continue
                if not isinstance(value, bool):
                    violations.append(f"universe governance control {key} must be boolean")
        action_risk_controls = payload.get("action_risk_controls", {})
        if artifact_kind != "universe_constitution" and (
            not isinstance(action_risk_controls, dict) or not action_risk_controls
        ):
            violations.append("universe artifact must contain a non-empty action_risk_controls object")
        elif isinstance(action_risk_controls, dict) and action_risk_controls:
            for key, value in action_risk_controls.items():
                if key not in _UNIVERSE_ACTION_RISK_KEYS:
                    violations.append(f"universe action risk control is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
                    violations.append(f"universe action risk control {key} must be a positive integer")
        environment_assumptions = payload.get("environment_assumptions", {})
        if artifact_kind != "universe_constitution" and (
            not isinstance(environment_assumptions, dict) or not environment_assumptions
        ):
            violations.append("universe artifact must contain a non-empty environment_assumptions object")
        elif isinstance(environment_assumptions, dict) and environment_assumptions:
            for key, allowed_values in _UNIVERSE_ENVIRONMENT_ENUM_FIELDS.items():
                value = str(environment_assumptions.get(key, "")).strip().lower()
                if value not in allowed_values:
                    violations.append(f"universe environment assumption {key} must be one of {sorted(allowed_values)}")
            for key in sorted(_UNIVERSE_ENVIRONMENT_BOOL_FIELDS):
                if key not in environment_assumptions or not isinstance(environment_assumptions.get(key), bool):
                    violations.append(f"universe environment assumption {key} must be boolean")
        invariants = payload.get("invariants", [])
        if artifact_kind != "operating_envelope" and (
            not isinstance(invariants, list) or not [str(item).strip() for item in invariants if str(item).strip()]
        ):
            violations.append("universe artifact must contain a non-empty invariants list")
        forbidden_patterns = payload.get("forbidden_command_patterns", [])
        if artifact_kind != "operating_envelope" and (
            not isinstance(forbidden_patterns, list)
            or not [str(item).strip() for item in forbidden_patterns if str(item).strip()]
        ):
            violations.append("universe artifact must contain a non-empty forbidden_command_patterns list")
        preferred_prefixes = payload.get("preferred_command_prefixes", [])
        if artifact_kind != "operating_envelope" and (
            not isinstance(preferred_prefixes, list)
            or not [str(item).strip() for item in preferred_prefixes if str(item).strip()]
        ):
            violations.append("universe artifact must contain a non-empty preferred_command_prefixes list")
        checks.append("universe_controls")
    if subsystem == "world_model":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _WORLD_MODEL_GENERATION_FOCI:
            violations.append("generation_focus must be a supported world_model focus")
        if str(payload.get("control_schema", "")).strip() != "world_model_behavior_controls_v1":
            violations.append("world_model artifacts must declare control_schema world_model_behavior_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("world_model artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _WORLD_MODEL_CONTROL_KEYS:
                    violations.append(f"world_model control is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(f"world_model control {key} must be numeric")
                    continue
                if float(value) < 0.0:
                    violations.append(f"world_model control {key} must be non-negative")
        planning_controls = payload.get("planning_controls", {})
        if not isinstance(planning_controls, dict) or not planning_controls:
            violations.append("world_model artifact must contain a non-empty planning_controls object")
        else:
            for key, value in planning_controls.items():
                if key not in _WORLD_MODEL_PLANNING_CONTROL_KEYS:
                    violations.append(f"world_model planning control is unsupported: {key}")
                    continue
                if key == "max_preserved_artifacts":
                    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
                        violations.append(
                            "world_model planning control max_preserved_artifacts must be a non-negative integer"
                        )
                elif not isinstance(value, bool):
                    violations.append(f"world_model planning control {key} must be boolean")
        checks.append("world_model_controls")
    if subsystem == "state_estimation":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _STATE_ESTIMATION_GENERATION_FOCI:
            violations.append("generation_focus must be a supported state_estimation focus")
        if str(payload.get("control_schema", "")).strip() != "state_estimation_controls_v1":
            violations.append("state_estimation artifacts must declare control_schema state_estimation_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("state_estimation artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _STATE_ESTIMATION_CONTROL_KEYS:
                    violations.append(f"state_estimation control is unsupported: {key}")
                    continue
                if key in {"min_state_change_score_for_progress", "regression_path_budget"}:
                    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
                        violations.append(f"state_estimation control {key} must be a non-negative integer")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(f"state_estimation control {key} must be numeric")
                elif key in {"regression_severity_weight", "progress_recovery_credit"} and float(value) < 0.0:
                    violations.append(f"state_estimation control {key} must be non-negative")
        latent_controls = payload.get("latent_controls", {})
        if not isinstance(latent_controls, dict) or not latent_controls:
            violations.append("state_estimation artifact must contain a non-empty latent_controls object")
        else:
            for key, value in latent_controls.items():
                if key not in _STATE_ESTIMATION_LATENT_KEYS:
                    violations.append(f"state_estimation latent control is unsupported: {key}")
                    continue
                if key in {"regressive_regression_count", "blocked_forbidden_count", "active_path_budget"}:
                    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                        violations.append(f"state_estimation latent control {key} must be a positive integer")
                elif key == "advancing_completion_ratio":
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"state_estimation latent control {key} must be numeric")
                    elif not 0.0 <= float(value) <= 1.0:
                        violations.append(f"state_estimation latent control {key} must stay within [0.0, 1.0]")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(f"state_estimation latent control {key} must be numeric")
        policy_controls = payload.get("policy_controls", {})
        if not isinstance(policy_controls, dict) or not policy_controls:
            violations.append("state_estimation artifact must contain a non-empty policy_controls object")
        else:
            for key, value in policy_controls.items():
                if key not in _STATE_ESTIMATION_POLICY_KEYS:
                    violations.append(f"state_estimation policy control is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
                    violations.append(f"state_estimation policy control {key} must be a non-negative integer")
        transition_summary = payload.get("transition_summary", {})
        if not isinstance(transition_summary, dict):
            violations.append("state_estimation artifact must contain a transition_summary object")
        checks.append("state_estimation_controls")
    if subsystem == "trust":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in {"balanced", "safety", "breadth", "stability"}:
            violations.append("generation_focus must be a supported trust focus")
        if str(payload.get("control_schema", "")).strip() != "unattended_trust_controls_v1":
            violations.append("trust artifacts must declare control_schema unattended_trust_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("trust artifact must contain a non-empty controls object")
        else:
            if "required_benchmark_families" in controls:
                families = controls.get("required_benchmark_families")
                if not isinstance(families, list) or not [str(value).strip() for value in families if str(value).strip()]:
                    violations.append("trust control required_benchmark_families must be a non-empty list")
            for key in (
                "recent_report_limit",
                "bootstrap_min_reports",
                "breadth_min_reports",
                "min_distinct_families",
            ):
                if key in controls and (isinstance(controls[key], bool) or not isinstance(controls[key], int) or int(controls[key]) < 0):
                    violations.append(f"trust control {key} must be a non-negative integer")
            for key in (
                "min_success_rate",
                "max_unsafe_ambiguous_rate",
                "max_hidden_side_effect_rate",
                "max_success_hidden_side_effect_rate",
            ):
                if key in controls:
                    value = controls[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"trust control {key} must be numeric")
                    elif not 0.0 <= float(value) <= 1.0:
                        violations.append(f"trust control {key} must stay within [0.0, 1.0]")
        checks.append("trust_controls")
    if subsystem == "recovery":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _RECOVERY_GENERATION_FOCI:
            violations.append("generation_focus must be a supported recovery focus")
        if str(payload.get("control_schema", "")).strip() != "workspace_recovery_controls_v1":
            violations.append("recovery artifacts must declare control_schema workspace_recovery_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("recovery artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _RECOVERY_CONTROL_KEYS:
                    violations.append(f"recovery control is unsupported: {key}")
                    continue
                if key == "max_post_rollback_file_count":
                    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
                        violations.append("recovery control max_post_rollback_file_count must be a non-negative integer")
                    continue
                if not isinstance(value, bool):
                    violations.append(f"recovery control {key} must be boolean")
        checks.append("recovery_controls")
    if subsystem == "delegation":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _DELEGATION_GENERATION_FOCI:
            violations.append("generation_focus must be a supported delegation focus")
        if str(payload.get("control_schema", "")).strip() != "delegated_resource_controls_v1":
            violations.append("delegation artifacts must declare control_schema delegated_resource_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("delegation artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _DELEGATION_CONTROL_KEYS:
                    violations.append(f"delegation control is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
                    violations.append(f"delegation control {key} must be a non-negative integer")
        checks.append("delegation_controls")
    if subsystem == "operator_policy":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _OPERATOR_POLICY_GENERATION_FOCI:
            violations.append("generation_focus must be a supported operator_policy focus")
        if str(payload.get("control_schema", "")).strip() != "unattended_operator_controls_v1":
            violations.append("operator_policy artifacts must declare control_schema unattended_operator_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("operator_policy artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _OPERATOR_POLICY_CONTROL_KEYS:
                    violations.append(f"operator_policy control is unsupported: {key}")
                    continue
                if key in {"unattended_allow_git_commands", "unattended_allow_http_requests", "unattended_allow_generated_path_mutations"}:
                    if not isinstance(value, bool):
                        violations.append(f"operator_policy control {key} must be boolean")
                elif key in {"unattended_http_timeout_seconds", "unattended_http_max_body_bytes"}:
                    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                        violations.append(f"operator_policy control {key} must be a positive integer")
                elif key in {"unattended_allowed_benchmark_families", "unattended_http_allowed_hosts", "unattended_generated_path_prefixes"}:
                    if not isinstance(value, list) or not [str(item).strip() for item in value if str(item).strip()]:
                        violations.append(f"operator_policy control {key} must be a non-empty list")
        checks.append("operator_policy_controls")
    if subsystem == "transition_model":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _TRANSITION_MODEL_GENERATION_FOCI:
            violations.append("generation_focus must be a supported transition_model focus")
        if str(payload.get("control_schema", "")).strip() != "transition_model_controls_v1":
            violations.append("transition_model artifacts must declare control_schema transition_model_controls_v1")
        controls = payload.get("controls", {})
        if not isinstance(controls, dict) or not controls:
            violations.append("transition_model artifact must contain a non-empty controls object")
        else:
            for key, value in controls.items():
                if key not in _TRANSITION_MODEL_CONTROL_KEYS:
                    violations.append(f"transition_model control is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                    violations.append(f"transition_model control {key} must be a positive integer")
        signatures = payload.get("signatures", [])
        if not isinstance(signatures, list):
            violations.append("transition_model artifact must contain a signatures list")
        else:
            for signature in signatures:
                if not isinstance(signature, dict):
                    violations.append("every transition_model signature must be an object")
                    continue
                signal = str(signature.get("signal", "")).strip()
                if signal not in {"no_state_progress", "state_regression"}:
                    violations.append("transition_model signature must contain a supported signal")
                if not str(signature.get("command", "")).strip():
                    violations.append("transition_model signature must contain a non-empty command")
                command_pattern = signature.get("command_pattern", "")
                if command_pattern not in {"", None} and not str(command_pattern).strip():
                    violations.append("transition_model signature command_pattern must be a non-empty string when present")
                support = signature.get("support", 0)
                if isinstance(support, bool) or not isinstance(support, int) or int(support) <= 0:
                    violations.append("transition_model signature support must be a positive integer")
                regressions = signature.get("regressions", [])
                if regressions is not None and not isinstance(regressions, list):
                    violations.append("transition_model signature regressions must be a list")
        checks.append("transition_model_controls")
    if subsystem == "tooling":
        candidates = payload.get("candidates", [])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                violations.append("every tool candidate must be an object")
                continue
            if str(candidate.get("spec_version", "")).strip() not in {"", "asi_v1"}:
                violations.append("every tool candidate spec_version must be asi_v1 when present")
            if not str(candidate.get("tool_id", "")).strip():
                violations.append("every tool candidate must contain a tool_id")
            if str(candidate.get("kind", "")).strip() != "local_shell_procedure":
                violations.append("every tool candidate must use kind local_shell_procedure")
            candidate_lifecycle_state = _tool_candidate_lifecycle_state(candidate)
            if not candidate_lifecycle_state:
                violations.append("every tool candidate must contain a lifecycle_state")
            promotion_stage = _tool_candidate_stage(candidate.get("promotion_stage", ""))
            if not promotion_stage:
                violations.append("every tool candidate must contain a promotion_stage")
            else:
                allowed_stage_states = {
                    "candidate_procedure": "candidate",
                    "replay_verified": "replay_verified",
                    "promoted_tool": "retained",
                    "rejected": "rejected",
                }
                expected_lifecycle_state = allowed_stage_states.get(promotion_stage, "")
                if not expected_lifecycle_state:
                    violations.append("every tool candidate promotion_stage must be candidate_procedure, replay_verified, promoted_tool, or rejected")
                elif candidate_lifecycle_state and candidate_lifecycle_state != expected_lifecycle_state:
                    violations.append(
                        f"tool candidate lifecycle_state must be {expected_lifecycle_state} when promotion_stage is {promotion_stage}"
                    )
            if not str(candidate.get("source_task_id", "")).strip():
                violations.append("every tool candidate must contain a source_task_id")
            if not str(candidate.get("benchmark_family", "")).strip():
                violations.append("every tool candidate must contain a benchmark_family")
            quality = candidate.get("quality")
            if isinstance(quality, bool) or not isinstance(quality, (int, float)):
                violations.append("every tool candidate must contain a numeric quality")
            if not str(candidate.get("script_name", "")).strip():
                violations.append("every tool candidate must contain a script_name")
            if not str(candidate.get("script_body", "")).strip():
                violations.append("every tool candidate must contain a script_body")
            if not isinstance(candidate.get("procedure"), dict):
                violations.append("every tool candidate must contain a procedure object")
            if not isinstance(candidate.get("task_contract"), dict):
                violations.append("every tool candidate must contain a task_contract object")
            if not isinstance(candidate.get("verifier"), dict):
                violations.append("every tool candidate must contain a verifier object")
        expected_top_level_stage = {
            "candidate": "candidate_procedure",
            "replay_verified": "replay_verified",
            "retained": "promoted_tool",
            "rejected": "rejected",
        }.get(lifecycle_state, "")
        if expected_top_level_stage and isinstance(candidates, list) and candidates and not _tool_candidates_have_stage(
            payload,
            expected_top_level_stage,
        ):
            violations.append(
                f"tool artifact lifecycle_state {lifecycle_state} requires all candidates to be in promotion_stage {expected_top_level_stage}"
            )
        checks.append("tool_ids")
        checks.append("tool_candidate_contracts")
    if subsystem == "operators":
        operators = payload.get("operators", [])
        for operator in operators:
            if not isinstance(operator, dict):
                violations.append("every operator must be an object")
                continue
            if not str(operator.get("operator_id", "")).strip():
                violations.append("every operator must contain an operator_id")
            if _operator_support(operator) <= 0:
                violations.append("every operator must contain a positive support value")
            if not _operator_benchmark_families(operator):
                violations.append("every operator must contain benchmark_families")
            if not _operator_steps(operator):
                violations.append("every operator must contain non-empty steps")
            if not _operator_task_contract(operator):
                violations.append("every operator must contain a task_contract")
        checks.append("operator_ids")
        checks.append("operator_contracts")
    if subsystem == "capabilities":
        config_field_names = {entry.name for entry in fields(KernelConfig)}
        modules = payload.get("modules", [])
        for module in modules:
            if not isinstance(module, dict):
                violations.append("every capability module must be an object")
                continue
            if not str(module.get("module_id", "")).strip():
                violations.append("every capability module must contain a module_id")
            capabilities = module.get("capabilities", [])
            if not isinstance(capabilities, list):
                violations.append("every capability module must contain a capabilities list")
            settings = module.get("settings", {})
            if settings is not None and not isinstance(settings, dict):
                violations.append("every capability module settings entry must be an object")
                continue
            if not isinstance(settings, dict):
                continue
            improvement_subsystems = settings.get("improvement_subsystems", [])
            if improvement_subsystems is None:
                continue
            if not isinstance(improvement_subsystems, list):
                violations.append("capability module improvement_subsystems must be a list")
                continue
            for surface in improvement_subsystems:
                if not isinstance(surface, dict):
                    violations.append("every capability improvement surface must be an object")
                    continue
                if not str(surface.get("subsystem_id", "")).strip():
                    violations.append("every capability improvement surface must contain a subsystem_id")
                base_subsystem = str(surface.get("base_subsystem", "")).strip()
                base_artifact_kinds = {
                    "benchmark": "benchmark_candidate_set",
                    "retrieval": "retrieval_policy_set",
                    "tolbert_model": "tolbert_model_bundle",
                    "verifier": "verifier_candidate_set",
                    "policy": "prompt_proposal_set",
                    "universe": "universe_contract",
                    "world_model": "world_model_policy_set",
                    "state_estimation": "state_estimation_policy_set",
                    "trust": "trust_policy_set",
                    "recovery": "recovery_policy_set",
                    "delegation": "delegated_runtime_policy_set",
                    "operator_policy": "operator_policy_set",
                    "transition_model": "transition_model_policy_set",
                    "curriculum": "curriculum_proposal_set",
                    "tooling": "tool_candidate_set",
                    "skills": "skill_set",
                    "operators": "operator_class_set",
                    "capabilities": "capability_module_set",
                }
                if base_subsystem not in {
                    "benchmark",
                    "retrieval",
                    "tolbert_model",
                    "verifier",
                    "policy",
                    "universe",
                    "world_model",
                    "state_estimation",
                    "trust",
                    "recovery",
                    "delegation",
                    "operator_policy",
                    "transition_model",
                    "curriculum",
                    "tooling",
                    "skills",
                    "operators",
                    "capabilities",
                }:
                    violations.append("capability improvement surfaces must declare a supported base_subsystem")
                artifact_path_attr = str(surface.get("artifact_path_attr", "")).strip()
                if artifact_path_attr and artifact_path_attr not in config_field_names:
                    violations.append("capability improvement surfaces must declare a valid config artifact_path_attr")
                proposal_toggle_attr = str(surface.get("proposal_toggle_attr", "")).strip()
                if proposal_toggle_attr and proposal_toggle_attr not in config_field_names:
                    violations.append("capability improvement surfaces must declare a valid config proposal_toggle_attr")
                generator_kind = str(surface.get("generator_kind", "")).strip()
                if generator_kind and generator_kind != base_subsystem:
                    violations.append("capability improvement surfaces must use the base_subsystem generator_kind")
                artifact_kind_override = str(surface.get("artifact_kind", "")).strip()
                expected_artifact_kind = base_artifact_kinds.get(base_subsystem, "")
                if artifact_kind_override and artifact_kind_override != expected_artifact_kind:
                    violations.append("capability improvement surfaces must use the base_subsystem artifact_kind")
        checks.append("capability_modules")

    semantic_violations = _semantic_compatibility_violations(
        subsystem=subsystem,
        payload=payload,
        bank=bank,
    )
    if semantic_violations:
        violations.extend(semantic_violations)
    checks.append("semantic_source_contract")

    return {
        "compatible": not violations,
        "checked_rules": checks,
        "violations": violations,
    }


def artifact_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_artifact(artifact_path: Path, *, cycle_id: str, stage: str) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    history_root = artifact_path.parent / ".artifact_history"
    history_root.mkdir(parents=True, exist_ok=True)
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "_", stage).strip("._") or "snapshot"
    suffix = artifact_path.suffix or ".json"
    snapshot_path = history_root / f"{artifact_path.stem}.{safe_cycle_id}.{safe_stage}{suffix}"
    if artifact_path.exists():
        shutil.copy2(artifact_path, snapshot_path)
    else:
        snapshot_path.write_text("{}", encoding="utf-8")
    return snapshot_path


def snapshot_artifact_state(artifact_path: Path, *, cycle_id: str, stage: str) -> Path:
    return _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage=stage)


def materialize_replay_verified_tool_payload(payload: Any) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("tool artifact payload must be a JSON object")
    clone = deepcopy(payload)
    candidates = clone.get("candidates", [])
    if not isinstance(candidates, list):
        return clone
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate["promotion_stage"] = "replay_verified"
        candidate["lifecycle_state"] = "replay_verified"
    return clone


def _update_tool_candidate_states(
    payload: dict[str, object],
    *,
    decision_state: str,
    lifecycle_state: str,
) -> None:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if decision_state == "retain":
            candidate["promotion_stage"] = "promoted_tool"
            candidate["lifecycle_state"] = "retained"
        else:
            candidate["promotion_stage"] = "rejected"
            candidate["lifecycle_state"] = "rejected"


def _retention_gate(subsystem: str, payload: dict[str, object] | None) -> dict[str, object]:
    subsystem = base_subsystem_for(subsystem)
    if isinstance(payload, dict):
        retention_gate = payload.get("retention_gate", {})
        if isinstance(retention_gate, dict):
            return retention_gate
    defaults: dict[str, dict[str, object]] = {
        "benchmark": retention_gate_preset("benchmark"),
        "curriculum": retention_gate_preset("curriculum"),
        "verifier": retention_gate_preset("verifier"),
        "policy": retention_gate_preset("policy"),
        "universe": retention_gate_preset("universe"),
        "world_model": retention_gate_preset("world_model"),
        "state_estimation": retention_gate_preset("state_estimation"),
        "trust": retention_gate_preset("trust"),
        "recovery": retention_gate_preset("recovery"),
        "delegation": retention_gate_preset("delegation"),
        "operator_policy": retention_gate_preset("operator_policy"),
        "transition_model": retention_gate_preset("transition_model"),
        "capabilities": retention_gate_preset("capabilities"),
        "retrieval": retention_gate_preset("retrieval"),
        "tolbert_model": retention_gate_preset("tolbert_model"),
        "skills": {
            "require_non_regression": True,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "tooling": {
            "require_replay_verification": True,
            "require_future_task_gain": True,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "operators": {
            "min_transfer_pass_rate_delta_abs": 0.05,
            "require_cross_task_support": True,
            "min_support": 2,
        },
    }
    return defaults.get(subsystem, {})


def retention_gate_for_payload(
    subsystem: str,
    payload: dict[str, object] | None,
    *,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    return dict(_retention_gate(base_subsystem_for(subsystem, capability_modules_path), payload))


def _candidate_family_failure_rate(metrics: EvalMetrics, family: str) -> float:
    total = metrics.total_by_benchmark_family.get(family, 0)
    if total == 0:
        return 1.0
    passed = metrics.passed_by_benchmark_family.get(family, 0)
    return max(0.0, min(1.0, 1.0 - (passed / total)))


def _family_discrimination_gain(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> float:
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


def _family_pass_rate_delta_map(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> dict[str, float]:
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


def _family_regression_count(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> int:
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


def _family_worst_delta(baseline_metrics: EvalMetrics, candidate_metrics: EvalMetrics) -> float:
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


def _validate_cycle_record_consistency(record: ImprovementCycleRecord) -> None:
    expected_kinds = {
        "benchmark": "benchmark_candidate_set",
        "retrieval": "retrieval_policy_set",
        "tolbert_model": "tolbert_model_bundle",
        "verifier": "verifier_candidate_set",
        "policy": "prompt_proposal_set",
        "universe": "universe_contract",
        "universe_constitution": "universe_constitution",
        "operating_envelope": "operating_envelope",
        "world_model": "world_model_policy_set",
        "state_estimation": "state_estimation_policy_set",
        "trust": "trust_policy_set",
        "recovery": "recovery_policy_set",
        "delegation": "delegated_runtime_policy_set",
        "transition_model": "transition_model_policy_set",
        "curriculum": "curriculum_proposal_set",
        "capabilities": "capability_module_set",
        "tooling": "tool_candidate_set",
        "skills": "skill_set",
        "operators": "operator_class_set",
    }
    if record.state != "generate":
        return
    expected_kind = expected_kinds.get(record.subsystem)
    if expected_kind and record.artifact_kind and record.artifact_kind != expected_kind:
        raise ValueError(
            f"generate record artifact kind {record.artifact_kind!r} does not match subsystem "
            f"{record.subsystem!r} expected kind {expected_kind!r}"
        )


def _record_metrics_summary(record: dict[str, object] | ImprovementCycleRecord) -> dict[str, object]:
    metrics_summary = record.metrics_summary if isinstance(record, ImprovementCycleRecord) else record.get("metrics_summary", {})
    return metrics_summary if isinstance(metrics_summary, dict) else {}


def _record_selected_variant_id(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        if str(record.selected_variant_id).strip():
            return str(record.selected_variant_id).strip()
    else:
        direct = str(record.get("selected_variant_id", "")).strip()
        if direct:
            return direct
    metrics_summary = _record_metrics_summary(record)
    direct = str(metrics_summary.get("selected_variant_id", "")).strip()
    if direct:
        return direct
    selected_variant = metrics_summary.get("selected_variant", {})
    if isinstance(selected_variant, dict):
        return str(selected_variant.get("variant_id", "")).strip()
    return ""


def _record_prior_retained_cycle_id(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        if str(record.prior_retained_cycle_id).strip():
            return str(record.prior_retained_cycle_id).strip()
    else:
        direct = str(record.get("prior_retained_cycle_id", "")).strip()
        if direct:
            return direct
    metrics_summary = _record_metrics_summary(record)
    return str(metrics_summary.get("prior_retained_cycle_id", "")).strip()


def _record_float_value(record: dict[str, object] | ImprovementCycleRecord, key: str) -> float | None:
    if isinstance(record, ImprovementCycleRecord):
        direct = getattr(record, key)
        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError):
                return None
    else:
        if key in record and record.get(key) is not None:
            try:
                return float(record.get(key))
            except (TypeError, ValueError):
                return None
    metrics_summary = _record_metrics_summary(record)
    if key in metrics_summary and metrics_summary.get(key) is not None:
        try:
            return float(metrics_summary.get(key))
        except (TypeError, ValueError):
            return None
    return None


def _record_non_regressed_family_support(record: dict[str, object] | ImprovementCycleRecord) -> int:
    metrics_summary = _record_metrics_summary(record)
    support = 0
    for key in ("family_pass_rate_delta", "generated_family_pass_rate_delta"):
        delta_map = metrics_summary.get(key, {})
        if not isinstance(delta_map, dict):
            continue
        for value in delta_map.values():
            try:
                if float(value) >= 0.0:
                    support += 1
            except (TypeError, ValueError):
                continue
    if support > 0:
        return support
    regressed = 0
    if "regressed_family_count" in metrics_summary:
        try:
            regressed = int(metrics_summary.get("regressed_family_count", 0))
        except (TypeError, ValueError):
            regressed = 0
    return max(1, 1 - regressed)


def _record_phase_gate_passed(record: dict[str, object] | ImprovementCycleRecord) -> bool | None:
    if isinstance(record, ImprovementCycleRecord):
        if record.phase_gate_passed is not None:
            return bool(record.phase_gate_passed)
    else:
        if "phase_gate_passed" in record:
            return bool(record.get("phase_gate_passed"))
    metrics_summary = _record_metrics_summary(record)
    if "phase_gate_passed" in metrics_summary:
        return bool(metrics_summary.get("phase_gate_passed"))
    return None


def _record_phase_gate_failures(record: dict[str, object] | ImprovementCycleRecord) -> list[str]:
    metrics_summary = _record_metrics_summary(record)
    failures = metrics_summary.get("phase_gate_failures", [])
    if not isinstance(failures, list):
        return []
    return [str(failure).strip() for failure in failures if str(failure).strip()]


def _dominant_count_label(counts: dict[str, int]) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    ranked = sorted(
        (
            (int(value), str(label).strip())
            for label, value in counts.items()
            if str(label).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0:
        return ""
    return ranked[0][1]


def _dominant_weight_label(counts: dict[str, object]) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    ranked = sorted(
        (
            (float(value), str(label).strip())
            for label, value in counts.items()
            if str(label).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0.0:
        return ""
    return ranked[0][1]


def _normalized_cycle_record(record: ImprovementCycleRecord) -> ImprovementCycleRecord:
    return ImprovementCycleRecord(
        cycle_id=record.cycle_id,
        state=record.state,
        subsystem=record.subsystem,
        action=record.action,
        artifact_path=record.artifact_path,
        artifact_kind=record.artifact_kind,
        reason=record.reason,
        metrics_summary=record.metrics_summary,
        candidate_artifact_path=record.candidate_artifact_path,
        active_artifact_path=record.active_artifact_path,
        artifact_lifecycle_state=record.artifact_lifecycle_state,
        artifact_sha256=record.artifact_sha256,
        previous_artifact_sha256=record.previous_artifact_sha256,
        rollback_artifact_path=record.rollback_artifact_path,
        artifact_snapshot_path=record.artifact_snapshot_path,
        selected_variant_id=record.selected_variant_id or _record_selected_variant_id(record),
        prior_retained_cycle_id=record.prior_retained_cycle_id or _record_prior_retained_cycle_id(record),
        baseline_pass_rate=record.baseline_pass_rate
        if record.baseline_pass_rate is not None
        else _record_float_value(record, "baseline_pass_rate"),
        candidate_pass_rate=record.candidate_pass_rate
        if record.candidate_pass_rate is not None
        else _record_float_value(record, "candidate_pass_rate"),
        baseline_average_steps=record.baseline_average_steps
        if record.baseline_average_steps is not None
        else _record_float_value(record, "baseline_average_steps"),
        candidate_average_steps=record.candidate_average_steps
        if record.candidate_average_steps is not None
        else _record_float_value(record, "candidate_average_steps"),
        phase_gate_passed=record.phase_gate_passed
        if record.phase_gate_passed is not None
        else _record_phase_gate_passed(record),
        compatibility=record.compatibility,
    )


def _verifier_contracts_are_strict(payload: dict[str, object] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list) or not proposals:
        return False
    return all(
        isinstance(proposal, dict)
        and isinstance(proposal.get("evidence", {}), dict)
        and bool(proposal.get("evidence", {}).get("strict_contract", False))
        for proposal in proposals
    )


def _tool_candidates_have_stage(payload: dict[str, object] | None, stage: str) -> bool:
    if not isinstance(payload, dict):
        return False
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return False
    return all(
        isinstance(candidate, dict) and str(candidate.get("promotion_stage", "")).strip() == stage
        for candidate in candidates
    )


def _operator_support_count(payload: dict[str, object] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    operators = payload.get("operators", [])
    if not isinstance(operators, list) or not operators:
        return 0
    counts = [
        len([str(value) for value in operator.get("source_task_ids", []) if str(value).strip()])
        for operator in operators
        if isinstance(operator, dict)
    ]
    return min(counts) if counts else 0


def stamp_artifact_experiment_variant(artifact_path: Path, variant: ImprovementVariant) -> None:
    if not artifact_path.exists():
        return
    payload = _load_json_payload(artifact_path)
    if not isinstance(payload, dict):
        return
    payload["experiment_variant"] = {
        "subsystem": variant.subsystem,
        "variant_id": variant.variant_id,
        "description": variant.description,
        "expected_gain": variant.expected_gain,
        "estimated_cost": variant.estimated_cost,
        "score": variant.score,
        "controls": dict(variant.controls),
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stamp_artifact_generation_context(
    artifact_path: Path,
    *,
    cycle_id: str,
    active_artifact_path: Path | None = None,
    candidate_artifact_path: Path | None = None,
    prior_active_artifact_path: Path | None = None,
    prior_retained_cycle_id: str | None = None,
    prior_retained_artifact_snapshot_path: Path | None = None,
) -> None:
    if not artifact_path.exists():
        return
    payload = _load_json_payload(artifact_path)
    if not isinstance(payload, dict):
        return
    context: dict[str, object] = {
        "cycle_id": cycle_id,
    }
    if active_artifact_path is not None:
        context["active_artifact_path"] = str(active_artifact_path)
    if candidate_artifact_path is not None:
        context["candidate_artifact_path"] = str(candidate_artifact_path)
    if prior_active_artifact_path is not None:
        context["prior_active_artifact_path"] = str(prior_active_artifact_path)
    if prior_retained_cycle_id:
        context["prior_retained_cycle_id"] = prior_retained_cycle_id
    if prior_retained_artifact_snapshot_path is not None:
        context["prior_retained_artifact_snapshot_path"] = str(prior_retained_artifact_snapshot_path)
    payload["generation_context"] = context
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def payload_with_active_artifact_context(
    payload: dict[str, object] | None,
    *,
    active_artifact_path: Path | None = None,
    active_artifact_payload: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    rebound = deepcopy(payload)
    context = rebound.get("generation_context", {})
    if not isinstance(context, dict):
        context = {}
    else:
        context = deepcopy(context)
    if active_artifact_path is not None:
        context["active_artifact_path"] = str(active_artifact_path)
    if isinstance(active_artifact_payload, dict):
        context["active_artifact_payload"] = deepcopy(active_artifact_payload)
    rebound["generation_context"] = context
    return rebound


def _prior_active_artifact_path(payload: Any) -> Path | None:
    if not isinstance(payload, dict):
        return None
    context = payload.get("generation_context", {})
    if not isinstance(context, dict):
        return None
    value = str(context.get("prior_active_artifact_path", "")).strip()
    if not value:
        return None
    return Path(value)


def staged_candidate_artifact_path(
    active_artifact_path: Path,
    *,
    candidates_root: Path,
    subsystem: str,
    cycle_id: str,
) -> Path:
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    target_dir = candidates_root / subsystem / safe_cycle_id
    return target_dir / active_artifact_path.name


def _semantic_compatibility_violations(
    *,
    subsystem: str,
    payload: dict[str, object],
    bank: TaskBank,
) -> list[str]:
    violations: list[str] = []
    if subsystem == "verifier":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                source_task_id = str(proposal.get("source_task_id", "")).strip()
                contract = proposal.get("contract", {})
                if not source_task_id or not isinstance(contract, dict):
                    continue
                try:
                    source_task = bank.get(source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for verifier proposal: {source_task_id}")
                    continue
                if not _is_stricter_contract(source_task, contract):
                    violations.append(
                        f"verifier proposal for {source_task_id} does not strengthen the source contract"
                    )
        return violations

    if subsystem == "retrieval":
        proposals = payload.get("proposals", [])
        allowed = {
            "tolbert_branch_results",
            "tolbert_global_results",
            "tolbert_top_branches",
            "tolbert_ancestor_branch_levels",
            "tolbert_max_spans_per_source",
            "tolbert_context_max_chunks",
            "tolbert_confidence_threshold",
            "tolbert_branch_confidence_margin",
            "tolbert_low_confidence_widen_threshold",
            "tolbert_low_confidence_branch_multiplier",
            "tolbert_low_confidence_global_multiplier",
            "tolbert_deterministic_command_confidence",
            "tolbert_first_step_direct_command_confidence",
            "tolbert_direct_command_min_score",
            "tolbert_skill_ranking_min_confidence",
            "tolbert_distractor_penalty",
        }
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                if not str(proposal.get("area", "")).strip():
                    violations.append("retrieval proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("retrieval proposal must contain a non-empty reason")
                overrides = proposal.get("overrides", {})
                if not isinstance(overrides, dict) or not overrides:
                    violations.append("retrieval proposal must contain non-empty overrides")
                    continue
                for key, value in overrides.items():
                    if key not in allowed:
                        violations.append(f"retrieval proposal contains unsupported override: {key}")
                        continue
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        violations.append(f"retrieval override {key} must be numeric")
                        continue
                    if "confidence" in key and not 0.0 <= numeric_value <= 1.0:
                        violations.append(f"retrieval override {key} must stay within [0.0, 1.0]")
                    if key in {
                        "tolbert_branch_results",
                        "tolbert_global_results",
                        "tolbert_top_branches",
                        "tolbert_ancestor_branch_levels",
                        "tolbert_max_spans_per_source",
                        "tolbert_context_max_chunks",
                        "tolbert_direct_command_min_score",
                    } and numeric_value < 0:
                        violations.append(f"retrieval override {key} must be non-negative")
        return violations

    if subsystem == "policy":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every prompt proposal must be an object")
                    continue
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("prompt proposal priority must be a positive integer")
                if not str(proposal.get("area", "")).strip():
                    violations.append("prompt proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("prompt proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("prompt proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "curriculum":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every curriculum proposal must be an object")
                    continue
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("curriculum proposal priority must be a positive integer")
                if not str(proposal.get("area", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "world_model":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every world_model proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _WORLD_MODEL_PROPOSAL_AREAS:
                    violations.append("world_model proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("world_model proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("world_model proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("world_model proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "state_estimation":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every state_estimation proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _STATE_ESTIMATION_PROPOSAL_AREAS:
                    violations.append("state_estimation proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("state_estimation proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("state_estimation proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("state_estimation proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "universe":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every universe proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _UNIVERSE_PROPOSAL_AREAS:
                    violations.append("universe proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("universe proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("universe proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("universe proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "trust":
        proposals = payload.get("proposals", [])
        allowed_areas = {"safety", "breadth", "stability"}
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every trust proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in allowed_areas:
                    violations.append("trust proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("trust proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("trust proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("trust proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "recovery":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every recovery proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _RECOVERY_PROPOSAL_AREAS:
                    violations.append("recovery proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("recovery proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("recovery proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("recovery proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "delegation":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every delegation proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _DELEGATION_PROPOSAL_AREAS:
                    violations.append("delegation proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("delegation proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("delegation proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("delegation proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "transition_model":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every transition_model proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _TRANSITION_MODEL_PROPOSAL_AREAS:
                    violations.append("transition_model proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("transition_model proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("transition_model proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("transition_model proposal must contain a non-empty suggestion")
        return violations

    if subsystem in {"skills", "tooling"}:
        collection_key = "skills" if subsystem == "skills" else "candidates"
        records = payload.get(collection_key, [])
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source_task_id = str(record.get("source_task_id", record.get("task_id", ""))).strip()
                if not source_task_id:
                    continue
                try:
                    source_task = bank.get(source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for {subsystem} artifact: {source_task_id}")
                    continue
                task_contract = record.get("task_contract", {})
                if isinstance(task_contract, dict) and task_contract:
                    expected_files = set(str(path) for path in source_task.expected_files)
                    contract_expected_files = set(
                        str(path) for path in task_contract.get("expected_files", source_task.expected_files)
                    )
                    if expected_files and contract_expected_files != expected_files:
                        violations.append(
                            f"{subsystem} artifact for {source_task_id} changes expected_files away from the source contract"
                        )
                procedure = record.get("procedure", {})
                commands = procedure.get("commands", []) if isinstance(procedure, dict) else []
                if not isinstance(commands, list) or not commands:
                    violations.append(f"{subsystem} artifact for {source_task_id} has no procedure commands")
                    continue
                normalized_commands = " ".join(str(command) for command in commands)
                expected_file_match = any(path and path in normalized_commands for path in source_task.expected_files)
                success_command_match = source_task.success_command and any(
                    token in normalized_commands
                    for token in _significant_tokens(source_task.success_command)
                )
                if source_task.expected_files and not expected_file_match and not success_command_match:
                    violations.append(
                        f"{subsystem} artifact for {source_task_id} is not obviously aligned with source artifacts or verifier intent"
                    )
        return violations

    if subsystem == "operators":
        records = payload.get("operators", [])
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source_task_ids = [str(value) for value in record.get("source_task_ids", []) if str(value).strip()]
                if len(source_task_ids) < 2:
                    violations.append("operator artifact must aggregate at least two source tasks")
                    continue
                commands = _operator_steps(record)
                if not isinstance(commands, list) or not commands:
                    violations.append("operator artifact must contain a non-empty step sequence")
                template_contract = _operator_task_contract(record)
                if not isinstance(template_contract, dict) or not template_contract:
                    violations.append("operator artifact must contain a task contract")
                for source_task_id in source_task_ids:
                    try:
                        source_task = bank.get(source_task_id)
                    except KeyError:
                        violations.append(f"unknown source task for operators artifact: {source_task_id}")
                        continue
                    if isinstance(template_contract, dict):
                        contract_expected_files = set(
                            str(path) for path in template_contract.get("expected_files", source_task.expected_files)
                        )
                        if source_task.expected_files and not contract_expected_files:
                            violations.append(
                                f"operator artifact for {source_task_id} is missing expected file anchors"
                            )
        return violations

    if subsystem == "benchmark":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                source_task_id = str(proposal.get("source_task_id", "")).strip()
                prompt = str(proposal.get("prompt", "")).strip().lower()
                if not source_task_id:
                    continue
                try:
                    source_task = bank.get(source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for benchmark proposal: {source_task_id}")
                    continue
                source_terms = _significant_tokens(source_task.prompt)
                if source_terms and not any(term in prompt for term in source_terms[:5]):
                    violations.append(
                        f"benchmark proposal for {source_task_id} is not semantically anchored to the source task prompt"
                    )
        return violations

    return violations


def _is_stricter_contract(source_task, contract: dict[str, object]) -> bool:
    expected_files = set(str(path) for path in source_task.expected_files)
    source_forbidden = set(str(path) for path in source_task.forbidden_files)
    source_expected_contents = dict(source_task.expected_file_contents)
    source_forbidden_output = set(str(value) for value in source_task.forbidden_output_substrings)

    contract_expected = set(str(path) for path in contract.get("expected_files", source_task.expected_files))
    contract_forbidden = set(str(path) for path in contract.get("forbidden_files", source_task.forbidden_files))
    contract_expected_contents = {
        str(path): str(content)
        for path, content in dict(contract.get("expected_file_contents", source_task.expected_file_contents)).items()
    }
    contract_forbidden_output = set(
        str(value) for value in contract.get("forbidden_output_substrings", source_task.forbidden_output_substrings)
    )

    if not expected_files.issubset(contract_expected):
        return False
    if not source_forbidden.issubset(contract_forbidden):
        return False
    if source_expected_contents.items() - contract_expected_contents.items():
        return False
    if not source_forbidden_output.issubset(contract_forbidden_output):
        return False

    strictly_stronger = (
        contract_forbidden != source_forbidden
        or contract_expected_contents != source_expected_contents
        or contract_forbidden_output != source_forbidden_output
    )
    return strictly_stronger


def _significant_tokens(text: str) -> list[str]:
    tokens = []
    for token in str(text).lower().replace("/", " ").replace(".", " ").replace("'", " ").replace('"', " ").split():
        cleaned = token.strip()
        if len(cleaned) < 4:
            continue
        if cleaned in {"create", "containing", "string", "under", "same", "task", "with", "from"}:
            continue
        if cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def _average_metric_delta(
    records: list[dict[str, object]],
    *,
    baseline_key: str,
    candidate_key: str,
) -> float:
    deltas: list[float] = []
    for record in records:
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            continue
        baseline = metrics_summary.get(baseline_key)
        candidate = metrics_summary.get(candidate_key)
        if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
            deltas.append(float(candidate) - float(baseline))
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)
