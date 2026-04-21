from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalMetrics:
    total: int
    passed: int
    average_steps: float = 0.0
    average_success_steps: float = 0.0
    unsafe_ambiguous_episodes: int = 0
    hidden_side_effect_risk_episodes: int = 0
    success_hidden_side_effect_risk_episodes: int = 0
    total_by_capability: dict[str, int] = field(default_factory=dict)
    passed_by_capability: dict[str, int] = field(default_factory=dict)
    total_by_difficulty: dict[str, int] = field(default_factory=dict)
    passed_by_difficulty: dict[str, int] = field(default_factory=dict)
    total_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    passed_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    total_by_memory_source: dict[str, int] = field(default_factory=dict)
    passed_by_memory_source: dict[str, int] = field(default_factory=dict)
    total_by_origin_benchmark_family: dict[str, int] = field(default_factory=dict)
    passed_by_origin_benchmark_family: dict[str, int] = field(default_factory=dict)
    termination_reasons: dict[str, int] = field(default_factory=dict)
    skill_selected_steps: int = 0
    episodes_with_skill_use: int = 0
    available_but_unused_skill_steps: int = 0
    excess_available_skill_slots: int = 0
    average_available_skills: float = 0.0
    average_retrieval_candidates: float = 0.0
    average_retrieval_evidence: float = 0.0
    retrieval_guided_steps: int = 0
    retrieval_selected_steps: int = 0
    retrieval_influenced_steps: int = 0
    retrieval_ranked_skill_steps: int = 0
    proposal_selected_steps: int = 0
    novel_command_steps: int = 0
    novel_valid_command_steps: int = 0
    trusted_retrieval_steps: int = 0
    low_confidence_steps: int = 0
    first_step_successes: int = 0
    average_retrieval_direct_candidates: float = 0.0
    average_first_step_path_confidence: float = 0.0
    success_first_step_path_confidence: float = 0.0
    failed_first_step_path_confidence: float = 0.0
    low_confidence_episodes: int = 0
    low_confidence_passed: int = 0
    memory_documents: int = 0
    reusable_skills: int = 0
    generated_total: int = 0
    generated_passed: int = 0
    generated_by_kind: dict[str, int] = field(default_factory=dict)
    generated_passed_by_kind: dict[str, int] = field(default_factory=dict)
    generated_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    generated_passed_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    decision_source_counts: dict[str, int] = field(default_factory=dict)
    tolbert_route_mode_counts: dict[str, int] = field(default_factory=dict)
    tolbert_shadow_episodes: int = 0
    tolbert_primary_episodes: int = 0
    tolbert_shadow_episodes_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    tolbert_primary_episodes_by_benchmark_family: dict[str, int] = field(default_factory=dict)
    proposal_metrics_by_benchmark_family: dict[str, dict[str, object]] = field(default_factory=dict)
    proposal_metrics_by_difficulty: dict[str, dict[str, object]] = field(default_factory=dict)
    world_feedback_summary: dict[str, object] = field(default_factory=dict)
    world_feedback_by_benchmark_family: dict[str, dict[str, object]] = field(default_factory=dict)
    world_feedback_by_difficulty: dict[str, dict[str, object]] = field(default_factory=dict)
    long_horizon_persistence_summary: dict[str, object] = field(default_factory=dict)
    contract_clean_failure_recovery_summary: dict[str, object] = field(default_factory=dict)
    contract_clean_failure_recovery_by_origin_benchmark_family: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    transfer_alignment_summary: dict[str, object] = field(default_factory=dict)
    task_outcomes: dict[str, dict[str, object]] = field(default_factory=dict)
    generated_task_summaries: dict[str, dict[str, object]] = field(default_factory=dict)
    task_trajectories: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.total == 0 else self.passed / self.total

    @property
    def unsafe_ambiguous_rate(self) -> float:
        return 0.0 if self.total == 0 else self.unsafe_ambiguous_episodes / self.total

    @property
    def hidden_side_effect_risk_rate(self) -> float:
        return 0.0 if self.total == 0 else self.hidden_side_effect_risk_episodes / self.total

    @property
    def success_hidden_side_effect_risk_rate(self) -> float:
        return 0.0 if self.total == 0 else self.success_hidden_side_effect_risk_episodes / self.total

    def capability_pass_rate(self, capability: str) -> float:
        total = self.total_by_capability.get(capability, 0)
        if total == 0:
            return 0.0
        return self.passed_by_capability.get(capability, 0) / total

    def difficulty_pass_rate(self, difficulty: str) -> float:
        total = self.total_by_difficulty.get(difficulty, 0)
        if total == 0:
            return 0.0
        return self.passed_by_difficulty.get(difficulty, 0) / total

    def benchmark_family_pass_rate(self, family: str) -> float:
        total = self.total_by_benchmark_family.get(family, 0)
        if total == 0:
            return 0.0
        return self.passed_by_benchmark_family.get(family, 0) / total

    def memory_source_pass_rate(self, source: str) -> float:
        total = self.total_by_memory_source.get(source, 0)
        if total == 0:
            return 0.0
        return self.passed_by_memory_source.get(source, 0) / total

    def origin_benchmark_family_pass_rate(self, family: str) -> float:
        total = self.total_by_origin_benchmark_family.get(family, 0)
        if total == 0:
            return 0.0
        return self.passed_by_origin_benchmark_family.get(family, 0) / total

    @property
    def skill_use_rate(self) -> float:
        return 0.0 if self.total == 0 else self.episodes_with_skill_use / self.total

    @property
    def retrieval_guidance_rate(self) -> float:
        total_steps = self.average_steps * self.total
        if total_steps <= 0:
            return 0.0
        return self.retrieval_guided_steps / total_steps

    @property
    def novel_valid_command_rate(self) -> float:
        if self.novel_command_steps == 0:
            return 0.0
        return self.novel_valid_command_steps / self.novel_command_steps

    @property
    def generated_pass_rate(self) -> float:
        return 0.0 if self.generated_total == 0 else self.generated_passed / self.generated_total

    @property
    def low_confidence_episode_pass_rate(self) -> float:
        if self.low_confidence_episodes == 0:
            return 0.0
        return self.low_confidence_passed / self.low_confidence_episodes


@dataclass(slots=True)
class SkillComparison:
    with_skills: EvalMetrics
    without_skills: EvalMetrics
    pass_rate_delta: float
    average_steps_delta: float
    capability_pass_rate_delta: dict[str, float] = field(default_factory=dict)
    benchmark_family_pass_rate_delta: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TolbertComparison:
    with_tolbert: EvalMetrics
    without_tolbert: EvalMetrics
    pass_rate_delta: float
    average_steps_delta: float
    capability_pass_rate_delta: dict[str, float] = field(default_factory=dict)
    benchmark_family_pass_rate_delta: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TolbertModeComparison:
    mode_metrics: dict[str, EvalMetrics] = field(default_factory=dict)


@dataclass(slots=True)
class AbstractionComparison:
    operator_metrics: EvalMetrics
    raw_skill_metrics: EvalMetrics
    pass_rate_delta: float
    average_steps_delta: float
    transfer_pass_rate_delta: float
