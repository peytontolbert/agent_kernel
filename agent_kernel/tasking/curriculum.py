from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

from ..config import KernelConfig, current_external_task_manifests_paths
from ..extensions.improvement.improvement_common import retained_artifact_payload
from ..memory import EpisodeMemory
from ..schemas import EpisodeRecord, TaskSpec
from ..extensions.strategy.semantic_hub import semantic_query_terms
from ..tasking.task_budget import uplifted_task_max_steps
from .curriculum_catalog import load_curriculum_metadata_catalog, render_curriculum_template
from .task_bank import TaskBank, annotate_light_supervision_contract


_CURRICULUM_METADATA = load_curriculum_metadata_catalog()
_LONG_HORIZON_TARGET_FAMILY_RULES = tuple(
    rule
    for rule in _CURRICULUM_METADATA.get("long_horizon_target_family_rules", [])
    if isinstance(rule, dict)
)
_LONG_HORIZON_VARIANT_RULES = {
    str(family): dict(rule)
    for family, rule in dict(_CURRICULUM_METADATA.get("long_horizon_variants", {})).items()
    if isinstance(family, str) and isinstance(rule, dict)
}
_LINEAGE_BRANCH_KIND_RULES = {
    str(branch_kind): dict(rule)
    for branch_kind, rule in dict(_CURRICULUM_METADATA.get("lineage_branch_kind_rules", {})).items()
    if isinstance(branch_kind, str) and isinstance(rule, dict)
}
_LATE_WAVE_ROTATION_FAMILIES = (
    "validation",
    "governance",
    "oversight",
    "assurance",
    "adjudication",
)
_CODING_FRONTIER_FAMILIES = (
    "repo_sandbox",
    "repository",
    "integration",
    "tooling",
    "project",
    "workflow",
)
_LATE_WAVE_PRIOR_RECENCY_HALF_LIFE_SECONDS = 6.0 * 60.0 * 60.0


class CurriculumEngine:
    def __init__(self, memory_root: Path | None = None, config: KernelConfig | None = None) -> None:
        self.memory = EpisodeMemory(memory_root) if memory_root is not None else None
        self.config = config or KernelConfig()
        self._curriculum_controls_cache: dict[str, object] | None = None

    def generate_followup_task(self, episode: EpisodeRecord) -> TaskSpec:
        if episode.success:
            task = self.generate_adjacent_task(episode)
        else:
            task = self.generate_failure_driven_task(episode)
        return annotate_light_supervision_contract(self._apply_curriculum_controls(self._with_curriculum_hint(task)))

    @staticmethod
    def _budgeted_task(task: TaskSpec) -> TaskSpec:
        task.max_steps = uplifted_task_max_steps(
            task.max_steps,
            metadata=task.metadata,
            suggested_commands=task.suggested_commands,
        )
        return task

    def schedule_generated_seed_episodes(
        self,
        episodes: list[EpisodeRecord],
        *,
        curriculum_kind: str,
    ) -> list[EpisodeRecord]:
        controls = self._curriculum_controls()
        success_only = curriculum_kind == "adjacent_success"
        preferred_family = str(controls.get("preferred_benchmark_family", "")).strip()
        shared_repo_bundle_state = (
            self._adjacent_success_shared_repo_bundle_state(episodes)
            if success_only
            else {}
        )
        ranked: list[tuple[int, str, EpisodeRecord]] = []
        for episode in episodes:
            family = self._episode_benchmark_family(episode)
            score = self._generated_seed_base_priority(
                episode,
                preferred_family=preferred_family,
            )
            if success_only:
                if episode.success:
                    score += 2
                score += self._long_horizon_seed_priority(episode)
                score += self._late_wave_seed_coverage_expansion_priority(episode)
                score += self._coding_frontier_seed_priority(episode)
                score += self._adjacent_success_shared_repo_bundle_priority(
                    episode,
                    bundle_state=shared_repo_bundle_state,
                )
            else:
                score += self._failure_recovery_seed_priority(episode)
            ranked.append((score, episode.task_id, episode))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        limit = self._max_generated_seed_tasks(curriculum_kind)
        selected = [episode for _, _, episode in ranked]
        if success_only:
            selected = self._diversify_adjacent_success_seeds(selected)
            selected = self._prioritize_adjacent_success_shared_repo_bundles(selected)
        if success_only and limit > 0:
            selected = self._select_adjacent_success_seed_set(selected, limit=limit)
        elif limit > 0:
            selected = self._select_failure_recovery_seed_set(selected, limit=limit)
        return selected

    def _generated_seed_base_priority(
        self,
        episode: EpisodeRecord,
        *,
        preferred_family: str,
    ) -> int:
        score = 0
        family = self._episode_benchmark_family(episode)
        if preferred_family and family == preferred_family:
            score += 4
        score += self._light_supervision_seed_priority(episode)
        return score

    @staticmethod
    def _episode_light_supervision_candidate(episode: EpisodeRecord) -> bool:
        metadata = dict(episode.task_metadata) if isinstance(episode.task_metadata, dict) else {}
        return bool(metadata.get("light_supervision_candidate", False))

    @staticmethod
    def _episode_light_supervision_contract_kind(episode: EpisodeRecord) -> str:
        metadata = dict(episode.task_metadata) if isinstance(episode.task_metadata, dict) else {}
        return str(metadata.get("light_supervision_contract_kind", "")).strip()

    def _light_supervision_seed_priority(self, episode: EpisodeRecord) -> int:
        if not self._episode_light_supervision_candidate(episode):
            return 0
        contract_kind = self._episode_light_supervision_contract_kind(episode)
        bonus = 2
        if contract_kind in {"semantic_verifier", "workspace_acceptance"}:
            bonus += 1
        return bonus

    @staticmethod
    def _episode_curriculum_kind(episode: EpisodeRecord) -> str:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        return str(metadata.get("curriculum_kind", "")).strip()

    @classmethod
    def _episode_is_contract_clean_primary_failure(cls, episode: EpisodeRecord) -> bool:
        if episode.success:
            return False
        if cls._episode_curriculum_kind(episode):
            return False
        return cls._episode_light_supervision_candidate(episode)

    @classmethod
    def _contract_clean_failure_recovery_step_floor(
        cls,
        episode: EpisodeRecord,
        *,
        benchmark_family: str,
        failure_types: list[str],
    ) -> int:
        normalized_family = str(benchmark_family or "").strip().lower()
        failure_type_set = {str(value).strip() for value in failure_types if str(value).strip()}
        command_steps = len([step for step in episode.steps if step.action == "code_execute"])
        step_floor = 8
        if normalized_family in _CODING_FRONTIER_FAMILIES:
            step_floor += 2
        if "missing_expected_file" in failure_type_set or "no_state_progress" in failure_type_set:
            step_floor += 1
        if str(episode.termination_reason).strip() in {"repeated_failed_action", "no_state_progress"}:
            step_floor += 1
        if command_steps > 1:
            step_floor += min(3, command_steps - 1)
        return min(16, max(8, step_floor))

    @classmethod
    def _contract_clean_failure_recovery_metadata(
        cls,
        episode: EpisodeRecord,
        *,
        benchmark_family: str,
        failure_types: list[str],
    ) -> dict[str, object]:
        if not cls._episode_is_contract_clean_primary_failure(episode):
            return {}
        step_floor = cls._contract_clean_failure_recovery_step_floor(
            episode,
            benchmark_family=benchmark_family,
            failure_types=failure_types,
        )
        contract_kind = cls._episode_light_supervision_contract_kind(episode)
        return {
            "contract_clean_failure_recovery_origin": True,
            "contract_clean_failure_recovery_origin_task": episode.task_id,
            "contract_clean_failure_recovery_origin_family": benchmark_family,
            "contract_clean_failure_recovery_origin_contract_kind": contract_kind,
            "contract_clean_failure_recovery_step_floor": step_floor,
            "failure_recovery_depth_profile": "contract_clean_primary",
            "deep_failure_recovery_budget": True,
            "budget_step_floor": step_floor,
            "step_floor": step_floor,
        }

    def _failure_recovery_seed_priority(self, episode: EpisodeRecord) -> int:
        score = 0
        if not episode.success:
            score += 3
        failure_types = self._failure_types(episode)
        score += min(2, len(failure_types))
        if str(episode.termination_reason).strip() == "repeated_failed_action":
            score += 1
        family = self._episode_benchmark_family(episode)
        if family in _CODING_FRONTIER_FAMILIES:
            score += 1
        priority_families = self._curriculum_control_family_list("frontier_priority_families")
        retention_priority_families = set(
            self._curriculum_control_family_list("frontier_retention_priority_families")
        )
        if family in priority_families:
            score += 2 + max(0, len(priority_families) - priority_families.index(family) - 1)
        if family in retention_priority_families:
            score += 2
        score -= self._failure_recovery_replay_penalty(episode)
        return score

    def _failure_recovery_replay_penalty(self, episode: EpisodeRecord) -> int:
        metadata = self._episode_curriculum_metadata(episode)
        penalty = 0
        memory_source = str(metadata.get("memory_source", "none")).strip().lower() or "none"
        if memory_source not in {"", "none"}:
            penalty += 2
        task_origin = str(metadata.get("task_origin", "")).strip().lower()
        if task_origin in {
            "episode_replay",
            "skill_replay",
            "skill_transfer",
            "operator_replay",
            "tool_replay",
            "verifier_replay",
            "discovered_task",
            "transition_pressure",
            "benchmark_candidate",
            "verifier_candidate",
        }:
            penalty += 2
        difficulty = str(metadata.get("difficulty", "")).strip().lower()
        if difficulty == "retrieval":
            penalty += 1
        if bool(metadata.get("requires_retrieval", False)) and str(metadata.get("source_task", "")).strip():
            penalty += 2
        return penalty

    def _failure_recovery_seed_group(self, episode: EpisodeRecord) -> str:
        benchmark_family = self._episode_benchmark_family(episode)
        failure_types = self._failure_types(episode)
        latest_command = self._latest_command(episode)
        failure_pattern = self._failure_pattern(episode, latest_command)
        if failure_pattern == "workspace_prefixed_path":
            return "failure_path_recovery"
        strategy_family = str(self._curriculum_controls().get("recovery_strategy_family", "")).strip()
        strategy_template_id = self._strategy_biased_failure_template_id(
            strategy_family,
            benchmark_family=benchmark_family,
            failure_types=failure_types,
            latest_command=latest_command,
        )
        if strategy_template_id:
            return strategy_template_id
        family_template_id = self._failure_family_template_id(benchmark_family)
        if family_template_id:
            return family_template_id
        failure_type_set = {str(value).strip() for value in failure_types if str(value).strip()}
        if "missing_expected_file" in failure_type_set:
            return "failure_file_recovery"
        if "command_failure" in failure_type_set and latest_command:
            return "failure_avoidance_recovery"
        return "failure_safe_retry"

    def _failure_recovery_seed_incremental_bonus(
        self,
        episode: EpisodeRecord,
        *,
        preferred_family: str,
        selected_groups: set[str],
        selected_families: set[str],
        selected_contract_kinds: set[str],
    ) -> int:
        bonus = 0
        group = self._failure_recovery_seed_group(episode)
        family = self._episode_benchmark_family(episode)
        non_preferred_family = bool(preferred_family and family and family != preferred_family)
        if group and group not in selected_groups:
            bonus += 1 if non_preferred_family else 4
        if family and family not in selected_families:
            bonus += 0 if non_preferred_family else 2
        contract_kind = self._episode_light_supervision_contract_kind(episode)
        if contract_kind and contract_kind not in selected_contract_kinds:
            bonus += 1
        return bonus

    def _select_failure_recovery_seed_set(
        self,
        episodes: list[EpisodeRecord],
        *,
        limit: int,
    ) -> list[EpisodeRecord]:
        if limit <= 0 or not episodes:
            return []
        preferred_family = str(self._curriculum_controls().get("preferred_benchmark_family", "")).strip()
        remaining = list(episodes)
        selected: list[EpisodeRecord] = []
        selected_groups: set[str] = set()
        selected_families: set[str] = set()
        selected_contract_kinds: set[str] = set()
        selected_repo_semantic_clusters: set[str] = set()
        while remaining and len(selected) < limit:
            best_index = 0
            best_score: tuple[int, int, int, int, int, int] | None = None
            for index, episode in enumerate(remaining):
                incremental_bonus = self._failure_recovery_seed_incremental_bonus(
                    episode,
                    preferred_family=preferred_family,
                    selected_groups=selected_groups,
                    selected_families=selected_families,
                    selected_contract_kinds=selected_contract_kinds,
                )
                semantic_cluster_gain = self._repo_semantic_cluster_gain(
                    episode,
                    selected_clusters=selected_repo_semantic_clusters,
                )
                replay_penalty = self._failure_recovery_replay_penalty(episode)
                score = (
                    self._generated_seed_base_priority(
                        episode,
                        preferred_family=preferred_family,
                    )
                    + self._failure_recovery_seed_priority(episode)
                    + incremental_bonus
                    + semantic_cluster_gain,
                    self._generated_seed_base_priority(
                        episode,
                        preferred_family=preferred_family,
                    )
                    + self._failure_recovery_seed_priority(episode),
                    self._failure_recovery_seed_priority(episode) + incremental_bonus,
                    semantic_cluster_gain,
                    -replay_penalty,
                    len(self._failure_types(episode)),
                )
                if best_score is None or score > best_score or (
                    score == best_score and str(episode.task_id) < str(remaining[best_index].task_id)
                ):
                    best_index = index
                    best_score = score
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            group = self._failure_recovery_seed_group(chosen)
            if group:
                selected_groups.add(group)
            family = self._episode_benchmark_family(chosen)
            if family:
                selected_families.add(family)
            contract_kind = self._episode_light_supervision_contract_kind(chosen)
            if contract_kind:
                selected_contract_kinds.add(contract_kind)
            selected_repo_semantic_clusters.update(self._episode_repo_semantic_clusters(chosen))
        return selected

    def _late_wave_seed_coverage_expansion_priority(self, episode: EpisodeRecord) -> int:
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return 0
        source_family = self._episode_benchmark_family(episode)
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return 0
        target_family = self._adjacent_success_target_benchmark_family(episode)
        if target_family == source_family:
            return 0
        signal = self._lineage_scheduler_signal(
            episode,
            source_family=source_family,
            target_family=target_family,
        )
        if not bool(signal["branch_outward"]):
            return 0
        coverage_gap = int(signal["rotation_coverage_gap"])
        source_family_count = int(signal["source_family_count"])
        target_family_count = int(signal["target_family_count"])
        return (
            coverage_gap * 3
            + max(0, source_family_count - target_family_count)
            + min(2, int(signal["lineage_depth"]) // 6)
        )

    def _late_wave_seed_set_incremental_bonus(
        self,
        episode: EpisodeRecord,
        *,
        selected_families: set[str],
        selected_branch_kinds: set[str],
        selected_stage_family_keys: set[str],
    ) -> int:
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return 0
        source_family = self._episode_benchmark_family(episode)
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return 0
        target_family = self._adjacent_success_target_benchmark_family(episode)
        branch_kind = self._adjacent_success_lineage_branch_kind(
            episode,
            target_family,
            self._long_horizon_adjacent_variant(episode, target_family),
        )
        bonus = 0
        if target_family and target_family not in selected_families:
            bonus += 6
        if branch_kind and branch_kind not in selected_branch_kinds:
            bonus += 3
        for stage_family_key in self._late_wave_seed_stage_family_keys(episode):
            if stage_family_key and stage_family_key not in selected_stage_family_keys:
                bonus += 6
        return bonus

    @staticmethod
    def _lineage_phase_bucket(depth: int) -> str:
        if depth >= 13:
            return "late"
        if depth >= 8:
            return "mid"
        return "early"

    def _late_wave_seed_stage_family_keys(self, episode: EpisodeRecord) -> set[str]:
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return set()
        source_family = self._episode_benchmark_family(episode)
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return set()
        target_family = self._adjacent_success_target_benchmark_family(episode)
        if not target_family:
            return set()
        lineage_depth = max(0, self._episode_lineage_depth(episode))
        current_phase = self._lineage_phase_bucket(lineage_depth)
        next_phase = self._lineage_phase_bucket(lineage_depth + 1)
        keys = {f"{current_phase}:{target_family}"}
        signal = self._lineage_scheduler_signal(
            episode,
            source_family=source_family,
            target_family=target_family,
        )
        if next_phase != current_phase and (
            bool(signal["branch_outward"]) or target_family != source_family
        ):
            keys.add(f"{next_phase}:{target_family}")
        return keys

    def _adjacent_success_seed_cost_units(self, episode: EpisodeRecord) -> int:
        metadata = self._episode_curriculum_metadata(episode)
        observed_runtime_seconds = float(metadata.get("observed_runtime_seconds", 0.0) or 0.0)
        observed_runtime_prior_seconds = float(metadata.get("observed_runtime_prior_seconds", 0.0) or 0.0)
        observed_runtime_prior_count = int(metadata.get("observed_runtime_prior_count", 0) or 0)
        observed_runtime_family_branch_prior_seconds = float(
            metadata.get("observed_runtime_family_branch_prior_seconds", 0.0) or 0.0
        )
        observed_runtime_family_branch_prior_count = int(
            metadata.get("observed_runtime_family_branch_prior_count", 0) or 0
        )
        observed_runtime_family_prior_seconds = float(
            metadata.get("observed_runtime_family_prior_seconds", 0.0) or 0.0
        )
        observed_runtime_family_prior_count = int(
            metadata.get("observed_runtime_family_prior_count", 0) or 0
        )
        observed_runtime_late_wave_branch_prior_seconds = float(
            metadata.get("observed_runtime_late_wave_branch_prior_seconds", 0.0) or 0.0
        )
        observed_runtime_late_wave_branch_prior_count = int(
            metadata.get("observed_runtime_late_wave_branch_prior_count", 0) or 0
        )
        observed_runtime_late_wave_phase_prior_seconds = float(
            metadata.get("observed_runtime_late_wave_phase_prior_seconds", 0.0) or 0.0
        )
        observed_runtime_late_wave_phase_prior_count = int(
            metadata.get("observed_runtime_late_wave_phase_prior_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_seconds = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_seconds", 0.0) or 0.0
        )
        observed_runtime_late_wave_phase_state_prior_count = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_support_count = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_support_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_dispersion_count = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_dispersion_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_directional_dispersion_count = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_directional_dispersion_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_phase_transition_count = float(
            metadata.get("observed_runtime_late_wave_phase_state_prior_phase_transition_count", 0) or 0
        )
        observed_runtime_late_wave_phase_state_prior_recency_weighted = bool(
            metadata.get("observed_runtime_late_wave_phase_state_prior_is_recency_weighted", False)
        )
        effective_runtime_seconds = 0.0
        weighted_total = 0.0
        weighted_count = 0.0
        if observed_runtime_seconds > 0.0:
            weighted_total += observed_runtime_seconds
            weighted_count += 1.0
        if observed_runtime_prior_seconds > 0.0 and observed_runtime_prior_count > 0:
            prior_weight = float(min(4, observed_runtime_prior_count))
            weighted_total += observed_runtime_prior_seconds * prior_weight
            weighted_count += prior_weight
        if observed_runtime_family_branch_prior_seconds > 0.0 and observed_runtime_family_branch_prior_count > 0:
            family_branch_weight = float(min(3, observed_runtime_family_branch_prior_count))
            weighted_total += observed_runtime_family_branch_prior_seconds * family_branch_weight
            weighted_count += family_branch_weight
        if (
            observed_runtime_late_wave_phase_state_prior_seconds > 0.0
            and observed_runtime_late_wave_phase_state_prior_count > 0
        ):
            late_wave_phase_state_weight = float(min(3, observed_runtime_late_wave_phase_state_prior_count))
            if observed_runtime_late_wave_phase_state_prior_recency_weighted:
                late_wave_phase_state_weight += 0.5
                late_wave_phase_state_weight += min(1.0, observed_runtime_late_wave_phase_state_prior_support_count * 0.25)
                late_wave_phase_state_weight += min(
                    0.75,
                    max(0.0, observed_runtime_late_wave_phase_state_prior_dispersion_count - 1.0) * 0.25,
                )
                late_wave_phase_state_weight += min(
                    1.0,
                    max(0.0, observed_runtime_late_wave_phase_state_prior_directional_dispersion_count) * 0.35,
                )
                late_wave_phase_state_weight += min(
                    1.25,
                    max(0.0, observed_runtime_late_wave_phase_state_prior_phase_transition_count) * 0.4,
                )
            weighted_total += observed_runtime_late_wave_phase_state_prior_seconds * late_wave_phase_state_weight
            weighted_count += late_wave_phase_state_weight
        if observed_runtime_late_wave_phase_prior_seconds > 0.0 and observed_runtime_late_wave_phase_prior_count > 0:
            late_wave_phase_weight = float(min(2, observed_runtime_late_wave_phase_prior_count))
            weighted_total += observed_runtime_late_wave_phase_prior_seconds * late_wave_phase_weight
            weighted_count += late_wave_phase_weight
        if observed_runtime_late_wave_branch_prior_seconds > 0.0 and observed_runtime_late_wave_branch_prior_count > 0:
            late_wave_branch_weight = float(min(1, observed_runtime_late_wave_branch_prior_count))
            weighted_total += observed_runtime_late_wave_branch_prior_seconds * late_wave_branch_weight
            weighted_count += late_wave_branch_weight
        if observed_runtime_family_prior_seconds > 0.0 and observed_runtime_family_prior_count > 0:
            family_weight = float(min(2, observed_runtime_family_prior_count))
            weighted_total += observed_runtime_family_prior_seconds * family_weight
            weighted_count += family_weight
        if weighted_count > 0.0:
            effective_runtime_seconds = weighted_total / weighted_count
        if effective_runtime_seconds > 0.0:
            return max(1, min(8, int((effective_runtime_seconds + 1.9999) // 2.0)))
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return 1
        source_family = self._episode_benchmark_family(episode)
        target_family = self._adjacent_success_target_benchmark_family(episode)
        variant = self._long_horizon_adjacent_variant(episode, target_family or source_family)
        branch_kind = self._adjacent_success_lineage_branch_kind(
            episode,
            target_family or source_family,
            variant,
        )
        step_count = int(episode.task_metadata.get("long_horizon_step_count", 0) or len(episode.steps) or 0)
        cost = 1
        cost += min(3, max(0, step_count) // 4)
        if target_family in {"integration", "repo_chore", "validation", "governance", "oversight", "assurance", "adjudication"}:
            cost += 1
        if (
            branch_kind in {"audit", "crosscheck"}
            and target_family in {"integration", "repo_chore", "validation", "governance", "oversight", "assurance", "adjudication"}
        ):
            cost += 1
        return max(1, cost)

    def _adjacent_success_seed_expected_value(self, episode: EpisodeRecord) -> float:
        metadata = self._episode_curriculum_metadata(episode)
        local_value = float(
            self._late_wave_seed_coverage_expansion_priority(episode)
            + self._long_horizon_seed_priority(episode)
        )
        exact_count = int(metadata.get("observed_outcome_prior_count", 0) or 0)
        exact_success_rate = float(metadata.get("observed_success_prior_rate", 0.0) or 0.0)
        exact_timeout_rate = float(metadata.get("observed_timeout_prior_rate", 0.0) or 0.0)
        exact_budget_exceeded_rate = float(metadata.get("observed_budget_exceeded_prior_rate", 0.0) or 0.0)
        family_branch_count = int(metadata.get("observed_outcome_family_branch_prior_count", 0) or 0)
        family_branch_success_rate = float(
            metadata.get("observed_success_family_branch_prior_rate", 0.0) or 0.0
        )
        family_branch_timeout_rate = float(
            metadata.get("observed_timeout_family_branch_prior_rate", 0.0) or 0.0
        )
        family_branch_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_family_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_count = int(metadata.get("observed_outcome_late_wave_branch_prior_count", 0) or 0)
        late_wave_branch_success_rate = float(
            metadata.get("observed_success_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_count = int(metadata.get("observed_outcome_late_wave_phase_prior_count", 0) or 0)
        late_wave_phase_success_rate = float(
            metadata.get("observed_success_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_count", 0) or 0
        )
        late_wave_phase_state_support_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_support_count", 0) or 0
        )
        late_wave_phase_state_dispersion_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_dispersion_count", 0) or 0
        )
        late_wave_phase_state_directional_dispersion_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_directional_dispersion_count", 0) or 0
        )
        late_wave_phase_state_phase_transition_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_phase_transition_count", 0) or 0
        )
        late_wave_phase_state_success_rate = float(
            metadata.get("observed_success_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_recency_weighted = bool(
            metadata.get("observed_outcome_late_wave_phase_state_prior_is_recency_weighted", False)
        )
        family_count = int(metadata.get("observed_outcome_family_prior_count", 0) or 0)
        family_success_rate = float(metadata.get("observed_success_family_prior_rate", 0.0) or 0.0)
        family_timeout_rate = float(metadata.get("observed_timeout_family_prior_rate", 0.0) or 0.0)
        family_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_family_prior_rate", 0.0) or 0.0
        )
        weighted_count = 0.0
        success_total = 0.0
        timeout_total = 0.0
        budget_total = 0.0
        if exact_count > 0:
            exact_weight = float(min(4, exact_count))
            weighted_count += exact_weight
            success_total += exact_success_rate * exact_weight
            timeout_total += exact_timeout_rate * exact_weight
            budget_total += exact_budget_exceeded_rate * exact_weight
        if family_branch_count > 0:
            family_branch_weight = float(min(3, family_branch_count))
            weighted_count += family_branch_weight
            success_total += family_branch_success_rate * family_branch_weight
            timeout_total += family_branch_timeout_rate * family_branch_weight
            budget_total += family_branch_budget_exceeded_rate * family_branch_weight
        if late_wave_phase_state_count > 0:
            late_wave_phase_state_weight = float(min(3, late_wave_phase_state_count))
            if late_wave_phase_state_recency_weighted:
                late_wave_phase_state_weight += 0.5
                late_wave_phase_state_weight += min(1.0, late_wave_phase_state_support_count * 0.25)
                late_wave_phase_state_weight += min(
                    0.75,
                    max(0.0, late_wave_phase_state_dispersion_count - 1.0) * 0.25,
                )
                late_wave_phase_state_weight += min(
                    1.0,
                    max(0.0, late_wave_phase_state_directional_dispersion_count) * 0.35,
                )
                late_wave_phase_state_weight += min(
                    1.25,
                    max(0.0, late_wave_phase_state_phase_transition_count) * 0.4,
                )
            weighted_count += late_wave_phase_state_weight
            success_total += late_wave_phase_state_success_rate * late_wave_phase_state_weight
            timeout_total += late_wave_phase_state_timeout_rate * late_wave_phase_state_weight
            budget_total += late_wave_phase_state_budget_exceeded_rate * late_wave_phase_state_weight
        if late_wave_phase_count > 0:
            late_wave_phase_weight = float(min(2, late_wave_phase_count))
            weighted_count += late_wave_phase_weight
            success_total += late_wave_phase_success_rate * late_wave_phase_weight
            timeout_total += late_wave_phase_timeout_rate * late_wave_phase_weight
            budget_total += late_wave_phase_budget_exceeded_rate * late_wave_phase_weight
        if late_wave_branch_count > 0:
            late_wave_branch_weight = float(min(1, late_wave_branch_count))
            weighted_count += late_wave_branch_weight
            success_total += late_wave_branch_success_rate * late_wave_branch_weight
            timeout_total += late_wave_branch_timeout_rate * late_wave_branch_weight
            budget_total += late_wave_branch_budget_exceeded_rate * late_wave_branch_weight
        if family_count > 0:
            family_weight = float(min(2, family_count))
            weighted_count += family_weight
            success_total += family_success_rate * family_weight
            timeout_total += family_timeout_rate * family_weight
            budget_total += family_budget_exceeded_rate * family_weight
        if weighted_count <= 0.0:
            return local_value
        success_rate = success_total / weighted_count
        timeout_rate = timeout_total / weighted_count
        budget_exceeded_rate = budget_total / weighted_count
        prior_weight = min(4.0, weighted_count)
        success_floor = 0.25
        adjusted_success = max(success_floor, success_rate)
        expansion_bonus = 0.0
        if late_wave_phase_state_recency_weighted and late_wave_phase_state_support_count > 0:
            adjusted_success = min(
                1.0,
                adjusted_success + min(0.08, late_wave_phase_state_support_count * 0.02),
            )
            adjusted_success = min(
                1.0,
                adjusted_success + min(0.06, max(0.0, late_wave_phase_state_dispersion_count - 1.0) * 0.03),
            )
            adjusted_success = min(
                1.0,
                adjusted_success + min(0.08, late_wave_phase_state_directional_dispersion_count * 0.04),
            )
            adjusted_success = min(
                1.0,
                adjusted_success + min(0.1, late_wave_phase_state_phase_transition_count * 0.05),
            )
            expansion_bonus += min(1.0, late_wave_phase_state_directional_dispersion_count * 0.35)
            expansion_bonus += min(1.25, late_wave_phase_state_phase_transition_count * 0.45)
        risk_penalty = min(0.75, (timeout_rate * 0.75) + (budget_exceeded_rate * 0.5))
        return max(
            0.0,
            ((local_value + expansion_bonus) * ((adjusted_success * prior_weight) + 1.0) / float(prior_weight + 1))
            - (local_value * risk_penalty),
        )

    def _adjacent_success_seed_projected_completion_mix(self, episode: EpisodeRecord) -> float:
        metadata = self._episode_curriculum_metadata(episode)
        exact_count = int(metadata.get("observed_outcome_prior_count", 0) or 0)
        exact_success_rate = float(metadata.get("observed_success_prior_rate", 0.0) or 0.0)
        exact_timeout_rate = float(metadata.get("observed_timeout_prior_rate", 0.0) or 0.0)
        exact_budget_exceeded_rate = float(metadata.get("observed_budget_exceeded_prior_rate", 0.0) or 0.0)
        family_branch_count = int(metadata.get("observed_outcome_family_branch_prior_count", 0) or 0)
        family_branch_success_rate = float(
            metadata.get("observed_success_family_branch_prior_rate", 0.0) or 0.0
        )
        family_branch_timeout_rate = float(
            metadata.get("observed_timeout_family_branch_prior_rate", 0.0) or 0.0
        )
        family_branch_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_family_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_count = int(metadata.get("observed_outcome_late_wave_branch_prior_count", 0) or 0)
        late_wave_branch_success_rate = float(
            metadata.get("observed_success_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_branch_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_branch_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_count = int(metadata.get("observed_outcome_late_wave_phase_prior_count", 0) or 0)
        late_wave_phase_success_rate = float(
            metadata.get("observed_success_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_phase_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_count", 0) or 0
        )
        late_wave_phase_state_support_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_support_count", 0) or 0
        )
        late_wave_phase_state_dispersion_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_dispersion_count", 0) or 0
        )
        late_wave_phase_state_directional_dispersion_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_directional_dispersion_count", 0) or 0
        )
        late_wave_phase_state_phase_transition_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_phase_transition_count", 0) or 0
        )
        late_wave_phase_state_success_rate = float(
            metadata.get("observed_success_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_timeout_rate = float(
            metadata.get("observed_timeout_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_late_wave_phase_state_prior_rate", 0.0) or 0.0
        )
        late_wave_phase_state_recency_weighted = bool(
            metadata.get("observed_outcome_late_wave_phase_state_prior_is_recency_weighted", False)
        )
        family_count = int(metadata.get("observed_outcome_family_prior_count", 0) or 0)
        family_success_rate = float(metadata.get("observed_success_family_prior_rate", 0.0) or 0.0)
        family_timeout_rate = float(metadata.get("observed_timeout_family_prior_rate", 0.0) or 0.0)
        family_budget_exceeded_rate = float(
            metadata.get("observed_budget_exceeded_family_prior_rate", 0.0) or 0.0
        )

        weighted_count = 0.0
        success_total = 0.0
        timeout_total = 0.0
        budget_total = 0.0
        if exact_count > 0:
            exact_weight = float(min(4, exact_count))
            weighted_count += exact_weight
            success_total += exact_success_rate * exact_weight
            timeout_total += exact_timeout_rate * exact_weight
            budget_total += exact_budget_exceeded_rate * exact_weight
        if family_branch_count > 0:
            family_branch_weight = float(min(3, family_branch_count))
            weighted_count += family_branch_weight
            success_total += family_branch_success_rate * family_branch_weight
            timeout_total += family_branch_timeout_rate * family_branch_weight
            budget_total += family_branch_budget_exceeded_rate * family_branch_weight
        if late_wave_phase_state_count > 0:
            late_wave_phase_state_weight = float(min(3, late_wave_phase_state_count))
            if late_wave_phase_state_recency_weighted:
                late_wave_phase_state_weight += 0.5
                late_wave_phase_state_weight += min(1.0, late_wave_phase_state_support_count * 0.25)
                late_wave_phase_state_weight += min(
                    0.75,
                    max(0.0, late_wave_phase_state_dispersion_count - 1.0) * 0.25,
                )
                late_wave_phase_state_weight += min(
                    1.0,
                    max(0.0, late_wave_phase_state_directional_dispersion_count) * 0.35,
                )
                late_wave_phase_state_weight += min(
                    1.25,
                    max(0.0, late_wave_phase_state_phase_transition_count) * 0.4,
                )
            weighted_count += late_wave_phase_state_weight
            success_total += late_wave_phase_state_success_rate * late_wave_phase_state_weight
            timeout_total += late_wave_phase_state_timeout_rate * late_wave_phase_state_weight
            budget_total += late_wave_phase_state_budget_exceeded_rate * late_wave_phase_state_weight
        if late_wave_phase_count > 0:
            late_wave_phase_weight = float(min(2, late_wave_phase_count))
            weighted_count += late_wave_phase_weight
            success_total += late_wave_phase_success_rate * late_wave_phase_weight
            timeout_total += late_wave_phase_timeout_rate * late_wave_phase_weight
            budget_total += late_wave_phase_budget_exceeded_rate * late_wave_phase_weight
        if late_wave_branch_count > 0:
            late_wave_branch_weight = float(min(1, late_wave_branch_count))
            weighted_count += late_wave_branch_weight
            success_total += late_wave_branch_success_rate * late_wave_branch_weight
            timeout_total += late_wave_branch_timeout_rate * late_wave_branch_weight
            budget_total += late_wave_branch_budget_exceeded_rate * late_wave_branch_weight
        if family_count > 0:
            family_weight = float(min(2, family_count))
            weighted_count += family_weight
            success_total += family_success_rate * family_weight
            timeout_total += family_timeout_rate * family_weight
            budget_total += family_budget_exceeded_rate * family_weight
        if weighted_count <= 0.0:
            return 1.0
        success_rate = success_total / weighted_count
        timeout_rate = timeout_total / weighted_count
        budget_exceeded_rate = budget_total / weighted_count
        completion_mix = success_rate - (timeout_rate * 0.75) - (budget_exceeded_rate * 0.5)
        if late_wave_phase_state_recency_weighted and late_wave_phase_state_support_count > 0:
            completion_mix += min(0.08, late_wave_phase_state_support_count * 0.015)
            completion_mix += min(0.06, max(0.0, late_wave_phase_state_dispersion_count - 1.0) * 0.025)
            completion_mix += min(0.08, late_wave_phase_state_directional_dispersion_count * 0.03)
            completion_mix += min(0.1, late_wave_phase_state_phase_transition_count * 0.04)
        return max(0.0, completion_mix)

    def _adjacent_success_seed_projected_completion_freshness(
        self,
        episode: EpisodeRecord,
        *,
        reference_now: float | None = None,
    ) -> float:
        metadata = self._episode_curriculum_metadata(episode)
        recorded_at = float(metadata.get("observed_recorded_at", 0.0) or 0.0)
        recency_weighted = bool(
            metadata.get("observed_outcome_late_wave_phase_state_prior_is_recency_weighted", False)
        )
        support_count = float(
            metadata.get("observed_outcome_late_wave_phase_state_prior_support_count", 0.0) or 0.0
        )
        if recorded_at <= 0.0:
            return 1.0 + min(0.2, support_count * 0.02) if recency_weighted else 1.0
        effective_now = float(reference_now if reference_now is not None else time.time())
        effective_now = max(effective_now, recorded_at)
        age_seconds = max(0.0, effective_now - recorded_at)
        recency_weight = 0.5 ** (age_seconds / _LATE_WAVE_PRIOR_RECENCY_HALF_LIFE_SECONDS)
        freshness = max(0.25, recency_weight)
        if recency_weighted:
            freshness += 0.1
            freshness += min(0.25, support_count * 0.025)
        return freshness

    def _adjacent_success_seed_budget_units(self, limit: int) -> int:
        controls = self._curriculum_controls()
        explicit = controls.get("max_generated_adjacent_cost_units")
        try:
            explicit_budget = int(explicit)
        except (TypeError, ValueError):
            explicit_budget = 0
        if explicit_budget > 0:
            return explicit_budget
        return max(1, limit * 3)

    def _coding_frontier_seed_priority(self, episode: EpisodeRecord) -> int:
        target_family = self._adjacent_success_target_benchmark_family(episode)
        source_family = self._episode_benchmark_family(episode)
        priority_families = self._curriculum_control_family_list("frontier_priority_families")
        missing_families = set(self._curriculum_control_family_list("frontier_missing_families"))
        retention_priority_families = set(
            self._curriculum_control_family_list("frontier_retention_priority_families")
        )
        generalization_priority_families = set(
            self._curriculum_control_family_list("frontier_generalization_priority_families")
        )
        retained_gain_families = set(self._curriculum_control_family_list("frontier_retained_gain_families"))
        promotion_risk_families = set(self._curriculum_control_family_list("frontier_promotion_risk_families"))
        failure_motif_priority_pairs = self._curriculum_control_family_signal_pairs(
            "frontier_failure_motif_priority_pairs"
        )
        repo_setting_priority_pairs = self._curriculum_control_family_signal_pairs(
            "frontier_repo_setting_priority_pairs"
        )
        priority_bonus = self._curriculum_positive_int_control("frontier_priority_family_bonus", default=0)
        missing_bonus = self._curriculum_positive_int_control("frontier_missing_family_bonus", default=0)
        retention_bonus = self._curriculum_positive_int_control("frontier_retention_priority_bonus", default=0)
        generalization_bonus = self._curriculum_positive_int_control("frontier_generalization_bonus", default=0)
        outward_branch_bonus = self._curriculum_positive_int_control("frontier_outward_branch_bonus", default=0)
        lineage_breadth_bonus = self._curriculum_positive_int_control("frontier_lineage_breadth_bonus", default=0)
        failure_motif_bonus = self._curriculum_positive_int_control("frontier_failure_motif_bonus", default=0)
        repo_setting_bonus = self._curriculum_positive_int_control("frontier_repo_setting_bonus", default=0)
        harder_bonus = self._curriculum_positive_int_control("frontier_harder_task_bonus", default=0)
        retained_gain_bonus = self._curriculum_positive_int_control("frontier_retained_gain_bonus", default=0)
        promotion_risk_penalty = self._curriculum_positive_int_control("frontier_promotion_risk_penalty", default=0)
        min_lineage_depth = self._curriculum_nonnegative_int_control("frontier_min_lineage_depth", default=0)
        retained_family_delta = self._curriculum_control_float_map("frontier_retained_family_delta")
        promotion_risk_family_delta = self._curriculum_control_float_map("frontier_promotion_risk_family_delta")

        score = 0
        if target_family in priority_families:
            score += max(1, priority_bonus)
            score += max(0, len(priority_families) - priority_families.index(target_family) - 1)
        elif source_family in priority_families:
            score += max(1, priority_bonus // 2)
        if target_family in missing_families:
            score += max(1, missing_bonus)
        if target_family in retention_priority_families:
            score += max(1, retention_bonus)
        if target_family in generalization_priority_families:
            score += max(1, generalization_bonus)
        retained_gain = float(retained_family_delta.get(target_family, 0.0) or 0.0)
        if target_family in retained_gain_families or retained_gain > 0.0:
            score += max(1, retained_gain_bonus)
            score += min(3, max(0, int(round(retained_gain * 20.0))))
        if any(family == target_family for family, _ in failure_motif_priority_pairs):
            score += max(1, failure_motif_bonus)
        episode_repo_setting_signatures = set(self._episode_repo_setting_signatures(episode))
        if any(
            family == target_family and signature in episode_repo_setting_signatures
            for family, signature in repo_setting_priority_pairs
        ):
            score += max(1, repo_setting_bonus)
        if (
            source_family in _CODING_FRONTIER_FAMILIES
            and target_family in _CODING_FRONTIER_FAMILIES
            and source_family != target_family
        ):
            score += max(0, outward_branch_bonus)
            lineage_breadth = self._coding_frontier_lineage_breadth(episode)
            if target_family in generalization_priority_families and lineage_breadth > 1:
                score += min(max(0, lineage_breadth_bonus), lineage_breadth - 1)
        if target_family in _CODING_FRONTIER_FAMILIES:
            metadata = self._episode_curriculum_metadata(episode)
            lineage_depth = self._episode_lineage_depth(episode)
            long_horizon_step_count = int(metadata.get("long_horizon_step_count", 0) or 0)
            difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip().lower()
            if difficulty == "long_horizon":
                score += max(0, harder_bonus)
                if lineage_depth >= min_lineage_depth:
                    score += 1
                if long_horizon_step_count >= 8:
                    score += 1
                promotion_risk = float(promotion_risk_family_delta.get(target_family, 0.0) or 0.0)
                if target_family in promotion_risk_families or promotion_risk > 0.0:
                    score -= max(1, promotion_risk_penalty)
                    score -= min(3, max(0, int(round(promotion_risk * 20.0))))
        return score

    def _coding_frontier_lineage_breadth(self, episode: EpisodeRecord) -> int:
        families = {
            family
            for family in self._episode_lineage_families(episode)
            if family in _CODING_FRONTIER_FAMILIES
        }
        source_family = self._episode_benchmark_family(episode)
        if source_family in _CODING_FRONTIER_FAMILIES:
            families.add(source_family)
        return len(families)

    def _episode_repo_setting_signatures(self, episode: EpisodeRecord) -> list[str]:
        metadata = self._episode_curriculum_metadata(episode)
        difficulty = str(
            metadata.get("difficulty", metadata.get("task_difficulty", ""))
        ).strip().lower()
        lineage_families = self._episode_lineage_families(episode)
        surfaces = [
            value
            for value in [
                self._episode_long_horizon_surface(episode),
                *self._episode_lineage_surfaces(episode),
            ]
            if value
        ]
        branch_kinds = self._episode_lineage_branch_kinds(episode)
        benchmark_family = self._episode_benchmark_family(episode)
        signatures: list[str] = []
        if difficulty == "long_horizon":
            signatures.append("long_horizon")
        if benchmark_family == "repo_sandbox" or "repo_sandbox" in lineage_families:
            signatures.append("repo_sandbox")
        if any("shared_repo" in surface for surface in surfaces):
            signatures.append("shared_repo")
        if any("worker" in surface for surface in surfaces):
            signatures.append("worker_handoff")
        if any("integrator" in surface for surface in surfaces):
            signatures.append("integrator_handoff")
        if any("validation" in surface for surface in surfaces):
            signatures.append("validation_lane")
        if "cleanup" in branch_kinds or any("cleanup" in surface for surface in surfaces):
            signatures.append("cleanup_lane")
        if "audit" in branch_kinds or any("audit" in surface for surface in surfaces):
            signatures.append("audit_lane")
        return signatures

    def _select_adjacent_success_seed_set(
        self,
        episodes: list[EpisodeRecord],
        *,
        limit: int,
    ) -> list[EpisodeRecord]:
        if limit <= 0:
            return []
        if not episodes:
            return []
        remaining = list(episodes)
        chosen: list[EpisodeRecord] = []
        selected_families: set[str] = set()
        selected_branch_kinds: set[str] = set()
        selected_stage_family_keys: set[str] = set()
        selected_groups: set[str] = set()
        selected_repo_semantic_clusters: set[str] = set()
        selected_shared_repo_worker_branches: dict[str, set[str]] = {}
        remaining_budget_units = self._adjacent_success_seed_budget_units(limit)
        bundle_state = self._adjacent_success_shared_repo_bundle_state(episodes)
        recorded_ats = [
            float(self._episode_curriculum_metadata(episode).get("observed_recorded_at", 0.0) or 0.0)
            for episode in episodes
        ]
        batch_reference_now = max([time.time(), *recorded_ats])
        while remaining and len(chosen) < limit:
            best_index = 0
            best_score: tuple[float, float, float, float, int, float, float, str] | None = None
            for index, episode in enumerate(remaining):
                cost_units = self._adjacent_success_seed_cost_units(episode)
                if cost_units > remaining_budget_units and chosen:
                    continue
                episode_stage_family_keys = {
                    stage_family_key
                    for stage_family_key in self._late_wave_seed_stage_family_keys(episode)
                    if stage_family_key
                }
                stage_family_gain = len(episode_stage_family_keys - selected_stage_family_keys)
                stage_family_gain_density = float(stage_family_gain) / float(max(1, cost_units))
                seed_group = self._adjacent_success_seed_group(episode)
                incremental_bonus = self._late_wave_seed_set_incremental_bonus(
                    episode,
                    selected_families=selected_families,
                    selected_branch_kinds=selected_branch_kinds,
                    selected_stage_family_keys=selected_stage_family_keys,
                )
                total_value = (
                    (4.0 if seed_group not in selected_groups else -4.0)
                    + float(incremental_bonus)
                    + float(
                        self._repo_semantic_cluster_gain(
                            episode,
                            selected_clusters=selected_repo_semantic_clusters,
                        )
                    )
                    + float(
                        self._adjacent_success_shared_repo_incremental_bonus(
                            episode,
                            bundle_state=bundle_state,
                            selected_worker_branches=selected_shared_repo_worker_branches,
                        )
                    )
                    + float(self._light_supervision_seed_priority(episode))
                    + self._adjacent_success_seed_expected_value(episode)
                    + float(self._coding_frontier_seed_priority(episode))
                )
                completion_mix = self._adjacent_success_seed_projected_completion_mix(episode)
                completion_freshness = self._adjacent_success_seed_projected_completion_freshness(
                    episode,
                    reference_now=batch_reference_now,
                )
                value_density = float(total_value) / float(max(1, cost_units))
                completion_mix_density = float(completion_mix) / float(max(1, cost_units))
                projected_stage_gain = float(stage_family_gain)
                projected_total_value = float(total_value)
                projected_completion_mix = float(completion_mix)
                projected_fresh_completion_mix = float(completion_mix * completion_freshness)
                remaining_slots_after_pick = limit - len(chosen) - 1
                remaining_budget_after_pick = max(0, remaining_budget_units - cost_units)
                if remaining_slots_after_pick > 0 and remaining_budget_after_pick > 0:
                    future_stage_gain, future_total_value, future_completion_mix, future_fresh_completion_mix = self._adjacent_success_seed_set_future_value(
                        [candidate for candidate_position, candidate in enumerate(remaining) if candidate_position != index],
                        selected_families=selected_families | (
                            {self._adjacent_success_target_benchmark_family(episode)}
                            if self._adjacent_success_target_benchmark_family(episode)
                            else set()
                        ),
                        selected_branch_kinds=selected_branch_kinds | (
                            {
                                self._adjacent_success_lineage_branch_kind(
                                    episode,
                                    self._adjacent_success_target_benchmark_family(episode)
                                    or self._episode_benchmark_family(episode),
                                    self._long_horizon_adjacent_variant(
                                        episode,
                                        self._adjacent_success_target_benchmark_family(episode)
                                        or self._episode_benchmark_family(episode),
                                    ),
                                )
                            }
                        ),
                        selected_stage_family_keys=selected_stage_family_keys | episode_stage_family_keys,
                        selected_groups=selected_groups | {seed_group},
                        selected_shared_repo_worker_branches=self._selected_shared_repo_worker_branches_after_pick(
                            selected_shared_repo_worker_branches,
                            episode,
                        ),
                        bundle_state=bundle_state,
                        remaining_budget_units=remaining_budget_after_pick,
                        remaining_slots=remaining_slots_after_pick,
                        reference_now=batch_reference_now,
                    )
                    projected_stage_gain += future_stage_gain
                    projected_total_value += future_total_value
                    projected_completion_mix += future_completion_mix
                    projected_fresh_completion_mix += future_fresh_completion_mix
                projected_stage_gain_density = projected_stage_gain / float(max(1, cost_units))
                projected_total_value_density = projected_total_value / float(max(1, cost_units))
                projected_completion_mix_density = projected_completion_mix / float(max(1, cost_units))
                projected_fresh_completion_mix_density = projected_fresh_completion_mix / float(max(1, cost_units))
                score = (
                    projected_fresh_completion_mix_density,
                    projected_completion_mix_density,
                    projected_stage_gain_density,
                    projected_fresh_completion_mix,
                    int(projected_stage_gain),
                    projected_total_value_density,
                    completion_mix_density + value_density,
                    episode.task_id,
                )
                if best_score is None or score[0] > best_score[0] or (
                    score[0] == best_score[0] and score[1] > best_score[1]
                ) or (
                    score[0] == best_score[0] and score[1] == best_score[1] and score[2] > best_score[2]
                ) or (
                    score[0] == best_score[0]
                    and score[1] == best_score[1]
                    and score[2] == best_score[2]
                    and score[3] > best_score[3]
                ) or (
                    score[0] == best_score[0]
                    and score[1] == best_score[1]
                    and score[2] == best_score[2]
                    and score[3] == best_score[3]
                    and score[4] > best_score[4]
                ) or (
                    score[0] == best_score[0]
                    and score[1] == best_score[1]
                    and score[2] == best_score[2]
                    and score[3] == best_score[3]
                    and score[4] == best_score[4]
                    and score[5] > best_score[5]
                ) or (
                    score[0] == best_score[0]
                    and score[1] == best_score[1]
                    and score[2] == best_score[2]
                    and score[3] == best_score[3]
                    and score[4] == best_score[4]
                    and score[5] == best_score[5]
                    and score[6] > best_score[6]
                ):
                    best_index = index
                    best_score = score
            if best_score is None:
                break
            episode = remaining.pop(best_index)
            chosen.append(episode)
            remaining_budget_units = max(
                0,
                remaining_budget_units - self._adjacent_success_seed_cost_units(episode),
            )
            target_family = self._adjacent_success_target_benchmark_family(episode)
            if target_family:
                selected_families.add(target_family)
            selected_stage_family_keys.update(self._late_wave_seed_stage_family_keys(episode))
            branch_kind = self._adjacent_success_lineage_branch_kind(
                episode,
                target_family or self._episode_benchmark_family(episode),
                self._long_horizon_adjacent_variant(
                    episode,
                    target_family or self._episode_benchmark_family(episode),
                ),
            )
            if branch_kind:
                selected_branch_kinds.add(branch_kind)
            selected_groups.add(self._adjacent_success_seed_group(episode))
            selected_repo_semantic_clusters.update(self._episode_repo_semantic_clusters(episode))
            selected_shared_repo_worker_branches = self._selected_shared_repo_worker_branches_after_pick(
                selected_shared_repo_worker_branches,
                episode,
            )
        return chosen

    def _adjacent_success_seed_set_future_value(
        self,
        episodes: list[EpisodeRecord],
        *,
        selected_families: set[str],
        selected_branch_kinds: set[str],
        selected_stage_family_keys: set[str],
        selected_groups: set[str],
        selected_shared_repo_worker_branches: dict[str, set[str]],
        bundle_state: dict[str, dict[str, set[str] | list[str]]],
        remaining_budget_units: int,
        remaining_slots: int,
        reference_now: float,
    ) -> tuple[float, float, float, float]:
        future_stage_gain = 0.0
        future_total_value = 0.0
        future_completion_mix = 0.0
        future_fresh_completion_mix = 0.0
        future_remaining = list(episodes)
        future_selected_families = set(selected_families)
        future_selected_branch_kinds = set(selected_branch_kinds)
        future_selected_stage_family_keys = set(selected_stage_family_keys)
        future_selected_groups = set(selected_groups)
        future_selected_shared_repo_worker_branches = {
            repo_id: set(branches)
            for repo_id, branches in dict(selected_shared_repo_worker_branches).items()
        }
        budget_left = max(0, remaining_budget_units)
        slots_left = max(0, remaining_slots)
        while future_remaining and slots_left > 0 and budget_left > 0:
            future_best_index = -1
            future_best_score: tuple[float, float, float, int, float] | None = None
            for index, episode in enumerate(future_remaining):
                cost_units = self._adjacent_success_seed_cost_units(episode)
                if cost_units > budget_left:
                    continue
                episode_stage_family_keys = {
                    stage_family_key
                    for stage_family_key in self._late_wave_seed_stage_family_keys(episode)
                    if stage_family_key
                }
                stage_family_gain = len(episode_stage_family_keys - future_selected_stage_family_keys)
                seed_group = self._adjacent_success_seed_group(episode)
                incremental_bonus = self._late_wave_seed_set_incremental_bonus(
                    episode,
                    selected_families=future_selected_families,
                    selected_branch_kinds=future_selected_branch_kinds,
                    selected_stage_family_keys=future_selected_stage_family_keys,
                )
                total_value = (
                    (4.0 if seed_group not in future_selected_groups else -4.0)
                    + float(incremental_bonus)
                    + float(
                        self._adjacent_success_shared_repo_incremental_bonus(
                            episode,
                            bundle_state=bundle_state,
                            selected_worker_branches=future_selected_shared_repo_worker_branches,
                        )
                    )
                    + float(self._light_supervision_seed_priority(episode))
                    + self._adjacent_success_seed_expected_value(episode)
                )
                completion_mix = self._adjacent_success_seed_projected_completion_mix(episode)
                fresh_completion_mix = completion_mix * self._adjacent_success_seed_projected_completion_freshness(
                    episode,
                    reference_now=reference_now,
                )
                score = (
                    float(fresh_completion_mix) / float(max(1, cost_units)),
                    float(completion_mix) / float(max(1, cost_units)),
                    float(stage_family_gain) / float(max(1, cost_units)),
                    fresh_completion_mix,
                    stage_family_gain,
                    float(total_value) / float(max(1, cost_units)),
                )
                if future_best_score is None or score[0] > future_best_score[0] or (
                    score[0] == future_best_score[0] and score[1] > future_best_score[1]
                ) or (
                    score[0] == future_best_score[0] and score[1] == future_best_score[1] and score[2] > future_best_score[2]
                ) or (
                    score[0] == future_best_score[0]
                    and score[1] == future_best_score[1]
                    and score[2] == future_best_score[2]
                    and score[3] > future_best_score[3]
                ) or (
                    score[0] == future_best_score[0]
                    and score[1] == future_best_score[1]
                    and score[2] == future_best_score[2]
                    and score[3] == future_best_score[3]
                    and score[4] > future_best_score[4]
                ):
                    future_best_index = index
                    future_best_score = score
            if future_best_index < 0:
                break
            episode = future_remaining.pop(future_best_index)
            cost_units = self._adjacent_success_seed_cost_units(episode)
            episode_stage_family_keys = {
                stage_family_key
                for stage_family_key in self._late_wave_seed_stage_family_keys(episode)
                if stage_family_key
            }
            future_stage_gain += float(len(episode_stage_family_keys - future_selected_stage_family_keys))
            seed_group = self._adjacent_success_seed_group(episode)
            incremental_bonus = self._late_wave_seed_set_incremental_bonus(
                episode,
                selected_families=future_selected_families,
                selected_branch_kinds=future_selected_branch_kinds,
                selected_stage_family_keys=future_selected_stage_family_keys,
            )
            future_total_value += float(
                (4.0 if seed_group not in future_selected_groups else -4.0)
                + float(incremental_bonus)
                + self._adjacent_success_shared_repo_incremental_bonus(
                    episode,
                    bundle_state=bundle_state,
                    selected_worker_branches=future_selected_shared_repo_worker_branches,
                )
                + self._adjacent_success_seed_expected_value(episode)
            )
            completion_mix = float(self._adjacent_success_seed_projected_completion_mix(episode))
            future_completion_mix += completion_mix
            future_fresh_completion_mix += completion_mix * self._adjacent_success_seed_projected_completion_freshness(
                episode,
                reference_now=reference_now,
            )
            budget_left = max(0, budget_left - cost_units)
            slots_left -= 1
            target_family = self._adjacent_success_target_benchmark_family(episode)
            if target_family:
                future_selected_families.add(target_family)
            future_selected_stage_family_keys.update(episode_stage_family_keys)
            future_selected_groups.add(seed_group)
            branch_kind = self._adjacent_success_lineage_branch_kind(
                episode,
                target_family or self._episode_benchmark_family(episode),
                self._long_horizon_adjacent_variant(
                    episode,
                    target_family or self._episode_benchmark_family(episode),
                ),
            )
            if branch_kind:
                future_selected_branch_kinds.add(branch_kind)
            future_selected_shared_repo_worker_branches = self._selected_shared_repo_worker_branches_after_pick(
                future_selected_shared_repo_worker_branches,
                episode,
            )
        return future_stage_gain, future_total_value, future_completion_mix, future_fresh_completion_mix

    def generate_adjacent_task(self, episode: EpisodeRecord) -> TaskSpec:
        retrieval_context = self._retrieve_context(episode, success_only=True)
        parent_benchmark_family = self._episode_benchmark_family(episode)
        parent_origin_benchmark_family = self._episode_origin_benchmark_family(episode) or parent_benchmark_family
        metadata = {
            "parent_task": episode.task_id,
            "curriculum_kind": "adjacent_success",
            "origin_benchmark_family": parent_benchmark_family,
            "parent_origin_benchmark_family": parent_origin_benchmark_family,
            **retrieval_context["metadata"],
        }
        benchmark_family = str(
            retrieval_context["metadata"].get("benchmark_family", self._episode_benchmark_family(episode) or "bounded")
        )
        long_horizon_task = self._generate_long_horizon_adjacent_task(
            episode,
            retrieval_context=retrieval_context,
            metadata=metadata,
            benchmark_family=benchmark_family,
        )
        if long_horizon_task is not None:
            return long_horizon_task
        if benchmark_family == "project":
            return self._render_adjacent_catalog_task(
                "adjacent_project_handoff",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if benchmark_family == "repository":
            return self._render_adjacent_catalog_task(
                "adjacent_repository_handoff",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if benchmark_family == "tooling":
            return self._render_adjacent_catalog_task(
                "adjacent_tool_handoff",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if benchmark_family == "integration":
            return self._render_adjacent_catalog_task(
                "adjacent_integration_handoff",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if benchmark_family == "repo_chore":
            return self._render_adjacent_catalog_task(
                "adjacent_repo_chore_handoff",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if benchmark_family == "workflow":
            return self._render_adjacent_catalog_task(
                "adjacent_workflow_audit",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        if episode.task_id == "hello_task":
            return self._render_adjacent_catalog_task(
                "adjacent_hello_followup",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
            )
        return self._render_adjacent_catalog_task(
            "adjacent_generic_summary",
            episode=episode,
            metadata=metadata,
            retrieval_context=retrieval_context,
        )

    def generate_failure_driven_task(self, episode: EpisodeRecord) -> TaskSpec:
        failure_types = self._failure_types(episode)
        latest_command = self._latest_command(episode)
        retrieval_context = self._retrieve_context(episode, success_only=False)
        metadata = {
            "parent_task": episode.task_id,
            "source_task": str(episode.task_metadata.get("source_task", episode.task_id)),
            "curriculum_kind": "failure_recovery",
            "failure_types": failure_types,
            "failure_pattern": self._failure_pattern(episode, latest_command),
            "failed_command": latest_command,
            **retrieval_context["metadata"],
        }
        benchmark_family = str(
            retrieval_context["metadata"].get("benchmark_family", self._episode_benchmark_family(episode) or "bounded")
        )
        metadata.update(
            self._contract_clean_failure_recovery_metadata(
                episode,
                benchmark_family=benchmark_family,
                failure_types=failure_types,
            )
        )
        recovery_strategy_family = str(self._curriculum_controls().get("recovery_strategy_family", "")).strip()
        if recovery_strategy_family:
            metadata["recovery_strategy_family"] = recovery_strategy_family
        prompt_suffix = ""
        if retrieval_context["summary"]:
            prompt_suffix = f" Use the validated pattern from {retrieval_context['summary']}."
        if retrieval_context["avoidance_note"]:
            prompt_suffix += f" Avoid {retrieval_context['avoidance_note']}."
        if latest_command:
            prompt_suffix += f" Do not repeat the failed command shape {latest_command!r}."

        failure_pattern = str(metadata["failure_pattern"])

        if failure_pattern == "workspace_prefixed_path":
            return self._render_failure_catalog_task(
                "failure_path_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        strategy_template_id = self._strategy_biased_failure_template_id(
            recovery_strategy_family,
            benchmark_family=benchmark_family,
            failure_types=failure_types,
            latest_command=latest_command,
        )
        if strategy_template_id:
            return self._render_failure_catalog_task(
                strategy_template_id,
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "project":
            return self._render_failure_catalog_task(
                "failure_project_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "repository":
            return self._render_failure_catalog_task(
                "failure_repository_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "tooling":
            return self._render_failure_catalog_task(
                "failure_tool_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "integration":
            return self._render_failure_catalog_task(
                "failure_integration_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "workflow":
            return self._render_failure_catalog_task(
                "failure_workflow_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if "missing_expected_file" in failure_types:
            return self._render_failure_catalog_task(
                "failure_file_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        if "command_failure" in failure_types and latest_command:
            return self._render_failure_catalog_task(
                "failure_avoidance_recovery",
                episode=episode,
                metadata=metadata,
                retrieval_context=retrieval_context,
                latest_command=latest_command,
                prompt_suffix=prompt_suffix,
            )
        return self._render_failure_catalog_task(
            "failure_safe_retry",
            episode=episode,
            metadata=metadata,
            retrieval_context=retrieval_context,
            latest_command=latest_command,
            prompt_suffix=prompt_suffix,
        )

    @staticmethod
    def _failure_family_template_id(benchmark_family: str) -> str:
        normalized = str(benchmark_family or "").strip()
        if normalized == "project":
            return "failure_project_recovery"
        if normalized == "repository":
            return "failure_repository_recovery"
        if normalized == "tooling":
            return "failure_tool_recovery"
        if normalized == "integration":
            return "failure_integration_recovery"
        if normalized == "workflow":
            return "failure_workflow_recovery"
        return ""

    @classmethod
    def _strategy_biased_failure_template_id(
        cls,
        strategy_family: str,
        *,
        benchmark_family: str,
        failure_types: list[str],
        latest_command: str,
    ) -> str:
        normalized = str(strategy_family or "").strip()
        family_template_id = cls._failure_family_template_id(benchmark_family)
        failure_type_set = {str(value).strip() for value in list(failure_types or []) if str(value).strip()}
        if normalized in {
            "rollback_validation",
            "restore_verification",
            "snapshot_integrity",
            "workspace_restore_verification",
        }:
            return family_template_id
        if normalized in {
            "snapshot_coverage",
            "verifier_crosscheck",
            "post_success_replay",
        }:
            if "missing_expected_file" in failure_type_set:
                return "failure_file_recovery"
            return family_template_id
        if normalized in {
            "mutation_residue_scan",
            "unexpected_change_audit",
        }:
            if "command_failure" in failure_type_set and str(latest_command).strip():
                return "failure_avoidance_recovery"
            return family_template_id
        return ""

    def _generate_long_horizon_adjacent_task(
        self,
        episode: EpisodeRecord,
        *,
        retrieval_context: dict[str, object],
        metadata: dict[str, object],
        benchmark_family: str,
    ) -> TaskSpec | None:
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return None
        long_horizon_surface = self._episode_long_horizon_surface(episode)
        variant = self._long_horizon_adjacent_variant(episode, benchmark_family)
        prompt_suffix = self._curriculum_prompt_suffix(retrieval_context)
        prompt_suffix = f"{prompt_suffix}{self._long_horizon_lineage_prompt_suffix(episode, benchmark_family)}"
        if benchmark_family == "project":
            shared_repo_task = self._generate_long_horizon_shared_repo_project_adjacent_task(
                episode,
                retrieval_context=retrieval_context,
                metadata=metadata,
                long_horizon_surface=long_horizon_surface,
            )
            if shared_repo_task is not None:
                return shared_repo_task
            template_id = "project_validation_bundle" if variant == "project_validation_bundle" else "project_release_bundle"
            return self._render_curriculum_catalog_task(
                template_id,
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "repository":
            template_id = "repository_validation_bundle" if variant == "repository_validation_bundle" else "repository_release_bundle"
            metadata_overrides: dict[str, object] = {
                "long_horizon_coding_surface": variant,
                "long_horizon_variant": variant,
                **self._next_lineage_metadata(
                    episode,
                    benchmark_family=benchmark_family,
                    variant=variant,
                ),
            }
            if long_horizon_surface in {"shared_repo_synthetic_worker", "shared_repo_integrator"}:
                metadata_overrides["parent_long_horizon_coding_surface"] = long_horizon_surface
            if long_horizon_surface == "shared_repo_synthetic_worker":
                metadata_overrides["synthetic_worker"] = True
            return self._render_curriculum_catalog_task(
                template_id,
                episode=episode,
                metadata=metadata,
                metadata_overrides=metadata_overrides,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "workflow":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "tooling":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "integration":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "repo_chore":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "validation":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "governance":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "oversight":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "assurance":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        if benchmark_family == "adjudication":
            return self._render_curriculum_catalog_task(
                variant,
                episode=episode,
                metadata=metadata,
                metadata_overrides={
                    "long_horizon_coding_surface": variant,
                    "long_horizon_variant": variant,
                    **self._next_lineage_metadata(
                        episode,
                        benchmark_family=benchmark_family,
                        variant=variant,
                    ),
                },
                prompt_suffix=prompt_suffix,
            )
        return None

    def _generate_long_horizon_shared_repo_project_adjacent_task(
        self,
        episode: EpisodeRecord,
        *,
        retrieval_context: dict[str, object],
        metadata: dict[str, object],
        long_horizon_surface: str,
    ) -> TaskSpec | None:
        prompt_suffix = self._curriculum_prompt_suffix(retrieval_context)
        if long_horizon_surface == "shared_repo_synthetic_worker":
            return self._render_curriculum_catalog_task(
                "shared_repo_worker_bundle",
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        if long_horizon_surface == "shared_repo_integrator":
            return self._render_curriculum_catalog_task(
                "shared_repo_integrator_bundle",
                episode=episode,
                metadata=metadata,
                prompt_suffix=prompt_suffix,
            )
        return None

    def _render_curriculum_catalog_task(
        self,
        template_id: str,
        *,
        episode: EpisodeRecord,
        metadata: dict[str, object],
        metadata_overrides: dict[str, object] | None = None,
        replacements: dict[str, object] | None = None,
        payload_overrides: dict[str, object] | None = None,
        prompt_suffix: str = "",
    ) -> TaskSpec:
        merged_metadata = dict(metadata)
        if metadata_overrides:
            merged_metadata.update(metadata_overrides)
        return annotate_light_supervision_contract(
            self._budgeted_task(
                render_curriculum_template(
                    template_id,
                    replacements={"task_id": episode.task_id, **(replacements or {})},
                    metadata_overrides=merged_metadata,
                    payload_overrides=payload_overrides,
                    prompt_suffix=prompt_suffix,
                )
            )
        )

    @staticmethod
    def _curriculum_prompt_suffix(retrieval_context: dict[str, object]) -> str:
        summary = str(retrieval_context.get("summary", "")).strip()
        if not summary:
            return ""
        return f" Reuse the validated pattern from {summary}."

    @staticmethod
    def _long_horizon_lineage_prompt_suffix(episode: EpisodeRecord, benchmark_family: str) -> str:
        if benchmark_family not in {"validation", "governance", "oversight"}:
            return ""
        lineage_surfaces = CurriculumEngine._episode_lineage_surfaces(episode)
        integrator_markers = {
            "shared_repo_integrator_bundle",
            "repository_integrator_bundle",
            "workflow_integrator_validation_bundle",
            "tooling_integrator_validation_bundle",
            "integration_integrator_validation_bundle",
            "validation_integrator_cleanup_gate_bundle",
            "validation_integrator_audit_gate_bundle",
            "governance_integrator_cleanup_review_bundle",
            "governance_integrator_audit_review_bundle",
        }
        worker_markers = {
            "shared_repo_worker_bundle",
            "repository_worker_bundle",
            "workflow_worker_validation_bundle",
            "tooling_worker_validation_bundle",
            "integration_worker_validation_bundle",
        }
        if any(surface in integrator_markers for surface in lineage_surfaces):
            return (
                " Continue the integrator handoff lineage: preserve merged-branch evidence,"
                " cross-repo coordination artifacts, and explicit handoff ownership."
            )
        if any(surface in worker_markers for surface in lineage_surfaces):
            return (
                " Continue the worker handoff lineage: preserve worker-owned scope,"
                " claim or proof artifacts, and explicit changed-path accountability."
            )
        return ""

    def _render_adjacent_catalog_task(
        self,
        template_id: str,
        *,
        episode: EpisodeRecord,
        metadata: dict[str, object],
        retrieval_context: dict[str, object],
    ) -> TaskSpec:
        task = self._render_curriculum_catalog_task(
            template_id,
            episode=episode,
            metadata=metadata,
            prompt_suffix=self._curriculum_prompt_suffix(retrieval_context),
        )
        return self._budgeted_task(
            replace(
                task,
                suggested_commands=self._merged_commands(
                    task.suggested_commands,
                    retrieval_context["successful_commands"],
                ),
            )
        )

    def _render_failure_catalog_task(
        self,
        template_id: str,
        *,
        episode: EpisodeRecord,
        metadata: dict[str, object],
        retrieval_context: dict[str, object],
        latest_command: str,
        prompt_suffix: str,
    ) -> TaskSpec:
        task = self._render_curriculum_catalog_task(
            template_id,
            episode=episode,
            metadata=metadata,
            replacements={"failed_command": latest_command},
            prompt_suffix=prompt_suffix,
        )
        expected_files = list(dict.fromkeys([*task.expected_files, *task.expected_file_contents.keys()]))
        return self._budgeted_task(
            replace(
                task,
                suggested_commands=self._failure_recovery_commands(
                    task.suggested_commands,
                    retrieval_context["successful_commands"],
                    expected_files=expected_files,
                    failed_command=latest_command,
                ),
            )
        )

    def _retrieve_context(self, episode: EpisodeRecord, *, success_only: bool) -> dict[str, object]:
        if success_only:
            fast_context = self._adjacent_success_context(episode)
            if fast_context is not None:
                return fast_context
        documents = self._memory_documents()
        if not documents:
            return {
                "successful_commands": [],
                "summary": "",
                "avoidance_note": "",
                "metadata": {
                    "reference_task_ids": [],
                    "reference_commands": [],
                    "retrieved_failure_types": [],
                    "retrieved_transition_failures": [],
                },
            }

        target_failure_types = set(self._failure_types(episode))
        target_semantic_terms = set(self._episode_context_terms(episode))
        episode_repo_semantics = set(self._episode_repo_semantic_clusters(episode))
        episode_lineage_families = set(self._episode_lineage_families(episode))
        episode_lineage_surfaces = set(self._episode_lineage_surfaces(episode))
        controls = self._curriculum_controls()
        preferred_family = str(controls.get("preferred_benchmark_family", "")).strip()
        family_only = bool(controls.get("failure_reference_family_only", False)) and not success_only
        episode_family = str(episode.task_metadata.get("benchmark_family", "")).strip()
        ranked: list[tuple[int, int, str, dict, list[str]]] = []
        for document in documents:
            task_id = str(document.get("task_id", ""))
            if task_id == episode.task_id:
                continue
            document_family = str(
                document.get("task_metadata", {}).get(
                    "benchmark_family",
                    document.get("metadata", {}).get("benchmark_family", ""),
                )
            ).strip()
            if family_only and episode_family and document_family and document_family != episode_family:
                continue
            score = 0
            if document.get("success"):
                score += 4
            if preferred_family and document_family == preferred_family:
                score += 3
            if episode_family and document_family == episode_family:
                score += 2
            if task_id.startswith(f"{episode.task_id}_") or episode.task_id.startswith(f"{task_id}_"):
                score += 3
            summary = document.get("summary", {})
            doc_failure_types = set(summary.get("failure_types", [])) | set(summary.get("transition_failures", []))
            if target_failure_types and doc_failure_types.intersection(target_failure_types):
                score += 3
            document_semantic_terms = set(self._document_context_terms(document))
            semantic_matches = sorted(target_semantic_terms.intersection(document_semantic_terms))
            if semantic_matches:
                score += min(8, len(semantic_matches))
            document_repo_semantics = set(self._document_repo_semantic_clusters(document))
            repo_semantic_matches = episode_repo_semantics.intersection(document_repo_semantics)
            if repo_semantic_matches:
                score += min(6, 2 * len(repo_semantic_matches))
            document_lineage_families = set(self._document_metadata_list(document, "lineage_families"))
            if episode_lineage_families and document_lineage_families.intersection(episode_lineage_families):
                score += 3
            document_lineage_surfaces = set(self._document_metadata_list(document, "lineage_surfaces"))
            if episode_lineage_surfaces and document_lineage_surfaces.intersection(episode_lineage_surfaces):
                score += 3
            if score:
                ranked.append((score, len(semantic_matches), task_id, document, semantic_matches))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        reference_limit = self._context_reference_limit(success_only=success_only)
        selected = [
            document
            for _, _, _, document, _ in ranked
            if not success_only or document.get("success")
        ][:reference_limit]
        reference_scores = {
            str(task_id): score
            for score, _, task_id, document, _ in ranked
            if document in selected and str(task_id).strip()
        }
        semantic_context_matches = {
            str(task_id): matches[:10]
            for _, _, task_id, document, matches in ranked
            if document in selected and matches
        }

        successful_commands: list[str] = []
        retrieved_failure_types: set[str] = set()
        retrieved_transition_failures: set[str] = set()
        for document in selected:
            summary = document.get("summary", {})
            retrieved_failure_types.update(summary.get("failure_types", []))
            retrieved_transition_failures.update(summary.get("transition_failures", []))
            for command in self._reference_commands_for_document(document):
                command_text = str(command).strip()
                if command_text and command_text not in successful_commands:
                    successful_commands.append(command_text)

        summary_label = ", ".join(str(document.get("task_id", "")) for document in selected[:2])
        avoidance_note = ", ".join(
            sorted(target_failure_types.intersection(retrieved_failure_types | retrieved_transition_failures))
        )
        return {
            "successful_commands": successful_commands[: self._success_reference_limit()],
            "summary": summary_label,
            "avoidance_note": avoidance_note,
            "metadata": {
                "reference_task_ids": [str(document.get("task_id", "")) for document in selected],
                "reference_commands": successful_commands[: self._success_reference_limit()],
                "retrieved_failure_types": sorted(retrieved_failure_types),
                "retrieved_transition_failures": sorted(retrieved_transition_failures),
                "benchmark_family": self._benchmark_family(episode.task_id, selected),
                "context_reference_limit": reference_limit,
                "reference_scores": reference_scores,
                "semantic_context_terms": sorted(target_semantic_terms)[:24],
                "semantic_context_matches": semantic_context_matches,
            },
        }

    def _adjacent_success_context(self, episode: EpisodeRecord) -> dict[str, object] | None:
        if not episode.success:
            return None
        benchmark_family = self._adjacent_success_target_benchmark_family(episode)
        reference_commands = self._reference_commands_for_episode(episode)
        if benchmark_family == "bounded" and not reference_commands:
            return None
        return {
            "successful_commands": reference_commands[: self._success_reference_limit()],
            "summary": episode.task_id,
            "avoidance_note": "",
            "metadata": {
                "reference_task_ids": [episode.task_id],
                "reference_commands": reference_commands[: self._success_reference_limit()],
                "retrieved_failure_types": [],
                "retrieved_transition_failures": [],
                "benchmark_family": benchmark_family,
            },
        }

    @staticmethod
    def _episode_origin_benchmark_family(episode: EpisodeRecord) -> str:
        origin = str(episode.task_metadata.get("origin_benchmark_family", "")).strip().lower()
        if origin:
            return origin
        contract = episode.task_contract if isinstance(episode.task_contract, dict) else {}
        contract_metadata = contract.get("metadata", {}) if isinstance(contract, dict) else {}
        if isinstance(contract_metadata, dict):
            origin = str(contract_metadata.get("origin_benchmark_family", "")).strip().lower()
            if origin:
                return origin
        return ""

    @staticmethod
    def _episode_curriculum_metadata(episode: EpisodeRecord) -> dict[str, object]:
        metadata = dict(episode.task_metadata)
        contract = episode.task_contract if isinstance(episode.task_contract, dict) else {}
        contract_metadata = contract.get("metadata", {}) if isinstance(contract, dict) else {}
        if isinstance(contract_metadata, dict):
            for key, value in contract_metadata.items():
                metadata.setdefault(str(key), value)
        return metadata

    @staticmethod
    def _episode_repo_semantic_clusters(episode: EpisodeRecord) -> list[str]:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        values = metadata.get("repo_semantics", [])
        if not isinstance(values, list):
            values = []
        normalized = [str(value).strip().lower() for value in values if str(value).strip()]
        if not normalized:
            benchmark_family = str(metadata.get("benchmark_family", "bounded")).strip().lower() or "bounded"
            normalized = [benchmark_family]
        deduped: list[str] = []
        for value in normalized:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @classmethod
    def _repo_semantic_cluster_gain(cls, episode: EpisodeRecord, *, selected_clusters: set[str]) -> int:
        clusters = {
            value
            for value in cls._episode_repo_semantic_clusters(episode)
            if str(value).strip()
        }
        return min(3, len(clusters - set(selected_clusters)))

    @staticmethod
    def _episode_lineage_families(episode: EpisodeRecord) -> list[str]:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        values = metadata.get("lineage_families", [])
        if not isinstance(values, list):
            return []
        return [str(value).strip().lower() for value in values if str(value).strip()]

    @staticmethod
    def _episode_lineage_surfaces(episode: EpisodeRecord) -> list[str]:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        values = metadata.get("lineage_surfaces", [])
        if not isinstance(values, list):
            return []
        return [str(value).strip().lower() for value in values if str(value).strip()]

    @staticmethod
    def _episode_lineage_branch_kinds(episode: EpisodeRecord) -> list[str]:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        values = metadata.get("lineage_branch_kinds", [])
        if not isinstance(values, list):
            return []
        return [str(value).strip().lower() for value in values if str(value).strip()]

    def _episode_uses_long_horizon_adjacent_curriculum(self, episode: EpisodeRecord) -> bool:
        metadata = self._episode_curriculum_metadata(episode)
        difficulty = str(
            metadata.get("difficulty", metadata.get("task_difficulty", ""))
        ).strip().lower()
        if difficulty == "long_horizon":
            return True
        benchmark_family = str(metadata.get("benchmark_family", self._episode_benchmark_family(episode))).strip().lower()
        if benchmark_family != "repo_sandbox":
            return False
        return difficulty in {
            "git_worker_branch",
            "git_parallel_merge",
            "git_conflict_resolution",
            "git_workflow",
            "git_test_repair",
        }

    def _episode_long_horizon_surface(self, episode: EpisodeRecord) -> str:
        metadata = self._episode_curriculum_metadata(episode)
        surface = str(metadata.get("long_horizon_coding_surface", "")).strip().lower()
        if surface:
            return surface
        benchmark_family = str(metadata.get("benchmark_family", self._episode_benchmark_family(episode))).strip().lower()
        if benchmark_family != "repo_sandbox":
            return ""
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        if str(workflow_guard.get("worker_branch", "")).strip():
            return "shared_repo_synthetic_worker"
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_merged_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if required_merged_branches or int(metadata.get("shared_repo_order", 0) or 0) > 0:
            return "shared_repo_integrator"
        if "review" in episode.task_id.lower():
            return "repository_validation_bundle"
        return "repository_release_bundle"

    def _adjacent_success_target_benchmark_family(self, episode: EpisodeRecord) -> str:
        benchmark_family = self._episode_benchmark_family(episode)
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return benchmark_family
        long_horizon_surface = self._episode_long_horizon_surface(episode)
        origin_benchmark_family = self._episode_origin_benchmark_family(episode)
        for rule in _LONG_HORIZON_TARGET_FAMILY_RULES:
            if str(rule.get("source_family", "")).strip() != benchmark_family:
                continue
            required_surfaces = {
                str(value).strip()
                for value in rule.get("required_surfaces", [])
                if str(value).strip()
            }
            if required_surfaces and long_horizon_surface not in required_surfaces:
                continue
            allowed_origin_families = {
                str(value).strip()
                for value in rule.get("allowed_origin_families", [])
                if str(value).strip()
            }
            if allowed_origin_families and origin_benchmark_family not in allowed_origin_families:
                continue
            target_family = str(rule.get("target_family", "")).strip()
            if target_family:
                if not self._lineage_should_branch_outward(
                    episode,
                    source_family=benchmark_family,
                    target_family=target_family,
                ):
                    continue
                return target_family
        late_wave_rotation_target_family = self._late_wave_rotation_target_family(
            episode,
            source_family=benchmark_family,
        )
        if late_wave_rotation_target_family:
            return late_wave_rotation_target_family
        return benchmark_family

    def _long_horizon_adjacent_variant(self, episode: EpisodeRecord, benchmark_family: str) -> str:
        long_horizon_surface = self._episode_long_horizon_surface(episode)
        task_id = episode.task_id.lower()
        persisted_branch_kinds = self._episode_lineage_branch_kinds(episode)
        persisted_branch_kind = persisted_branch_kinds[-1] if persisted_branch_kinds else ""
        lineage_surfaces = list(reversed(self._episode_lineage_surfaces(episode)))
        family_rules = _LONG_HORIZON_VARIANT_RULES.get(benchmark_family, {})
        surface_variants = {
            str(surface): str(variant)
            for surface, variant in dict(family_rules.get("surface_variants", {})).items()
            if str(surface).strip() and str(variant).strip()
        }
        if long_horizon_surface in surface_variants:
            return surface_variants[long_horizon_surface]
        lineage_surface_branch_variants: dict[str, dict[str, str]] = {}
        for surface, branch_variants in dict(family_rules.get("lineage_surface_branch_variants", {})).items():
            surface_name = str(surface).strip()
            if not surface_name or not isinstance(branch_variants, dict):
                continue
            normalized_branch_variants = {
                str(branch_kind).strip(): str(variant).strip()
                for branch_kind, variant in branch_variants.items()
                if str(branch_kind).strip() and str(variant).strip()
            }
            if normalized_branch_variants:
                lineage_surface_branch_variants[surface_name] = normalized_branch_variants
        if persisted_branch_kind:
            for surface in lineage_surfaces:
                branch_variants = lineage_surface_branch_variants.get(surface)
                if branch_variants and persisted_branch_kind in branch_variants:
                    return branch_variants[persisted_branch_kind]
        lineage_surface_variants = {
            str(surface): str(variant)
            for surface, variant in dict(family_rules.get("lineage_surface_variants", {})).items()
            if str(surface).strip() and str(variant).strip()
        }
        for surface in lineage_surfaces:
            if surface in lineage_surface_variants:
                return lineage_surface_variants[surface]
        branch_kind_variants = {
            str(branch_kind): str(variant)
            for branch_kind, variant in dict(family_rules.get("branch_kind_variants", {})).items()
            if str(branch_kind).strip() and str(variant).strip()
        }
        if persisted_branch_kind in branch_kind_variants:
            return branch_kind_variants[persisted_branch_kind]
        for rule in family_rules.get("keyword_variants", []):
            if not isinstance(rule, dict):
                continue
            keywords = [str(value).strip().lower() for value in rule.get("keywords", []) if str(value).strip()]
            variant = str(rule.get("variant", "")).strip()
            if variant and any(keyword in task_id for keyword in keywords):
                return variant
        return str(family_rules.get("release_variant", benchmark_family)).strip() or benchmark_family

    def _adjacent_success_lineage_branch_kind(self, episode: EpisodeRecord, benchmark_family: str, variant: str) -> str:
        task_id = episode.task_id.lower()
        variant_lower = variant.lower()
        source_family = self._episode_benchmark_family(episode)
        if source_family == "project" and self._episode_long_horizon_surface(episode) == "shared_repo_integrator":
            return "project_release"
        for branch_kind, rule in _LINEAGE_BRANCH_KIND_RULES.items():
            task_keywords = [str(value).strip().lower() for value in rule.get("task_keywords", []) if str(value).strip()]
            variant_keywords = [
                str(value).strip().lower()
                for value in rule.get("variant_keywords", [])
                if str(value).strip()
            ]
            if any(keyword in task_id for keyword in task_keywords):
                return branch_kind
            if any(keyword in variant_lower for keyword in variant_keywords):
                return branch_kind
        if variant_lower in {"project_validation_bundle", "repository_validation_bundle"}:
            return "crosscheck"
        persisted = self._episode_lineage_branch_kinds(episode)
        if persisted:
            return persisted[-1]
        if source_family in {"project", "repository", "workflow", "tooling", "integration"}:
            return f"{source_family}_release"
        return f"{benchmark_family}_release"

    def _next_lineage_metadata(
        self,
        episode: EpisodeRecord,
        *,
        benchmark_family: str,
        variant: str,
    ) -> dict[str, object]:
        lineage_families = self._episode_lineage_families(episode)
        if not lineage_families:
            parent_family = self._episode_benchmark_family(episode)
            if parent_family:
                lineage_families = [parent_family]
        lineage_surfaces = self._episode_lineage_surfaces(episode)
        if not lineage_surfaces:
            parent_surface = self._episode_long_horizon_surface(episode)
            if parent_surface:
                lineage_surfaces = [parent_surface]
        lineage_branch_kinds = self._episode_lineage_branch_kinds(episode)
        if not lineage_branch_kinds:
            parent_family = self._episode_benchmark_family(episode) or benchmark_family
            parent_surface = self._episode_long_horizon_surface(episode) or variant
            lineage_branch_kinds = [
                self._adjacent_success_lineage_branch_kind(episode, parent_family, parent_surface)
            ]
        next_branch_kind = self._adjacent_success_lineage_branch_kind(episode, benchmark_family, variant)
        if not lineage_families or lineage_families[-1] != benchmark_family:
            lineage_families = [*lineage_families, benchmark_family]
        if not lineage_surfaces or lineage_surfaces[-1] != variant:
            lineage_surfaces = [*lineage_surfaces, variant]
        if not lineage_branch_kinds or lineage_branch_kinds[-1] != next_branch_kind:
            lineage_branch_kinds = [*lineage_branch_kinds, next_branch_kind]
        return {
            "lineage_families": lineage_families,
            "lineage_surfaces": lineage_surfaces,
            "lineage_branch_kinds": lineage_branch_kinds,
            "lineage_depth": len(lineage_families),
            "lineage_branch_kind": next_branch_kind,
        }

    def _episode_lineage_depth(self, episode: EpisodeRecord) -> int:
        lineage_families = self._episode_lineage_families(episode)
        if lineage_families:
            return len(lineage_families)
        parent_family = self._episode_benchmark_family(episode)
        return 1 if parent_family else 0

    def _episode_recent_family_repetition(self, episode: EpisodeRecord, family: str) -> int:
        lineage_families = self._episode_lineage_families(episode)
        if not lineage_families:
            return 1 if self._episode_benchmark_family(episode) == family else 0
        repetition = 0
        for value in reversed(lineage_families):
            if value != family:
                break
            repetition += 1
        if repetition == 0 and self._episode_benchmark_family(episode) == family:
            return 1
        return repetition

    def _episode_recent_branch_saturation(self, episode: EpisodeRecord, branch_kind: str) -> int:
        if not branch_kind:
            return 0
        branch_kinds = self._episode_lineage_branch_kinds(episode)
        if not branch_kinds:
            inferred = self._adjacent_success_lineage_branch_kind(
                episode,
                self._episode_benchmark_family(episode),
                self._episode_long_horizon_surface(episode),
            )
            return 1 if inferred == branch_kind else 0
        saturation = 0
        for value in reversed(branch_kinds):
            if value != branch_kind:
                break
            saturation += 1
        return saturation

    def _episode_recent_late_wave_families(self, episode: EpisodeRecord) -> list[str]:
        lineage_families = [
            family
            for family in self._episode_lineage_families(episode)
            if family in _LATE_WAVE_ROTATION_FAMILIES
        ]
        if not lineage_families:
            benchmark_family = self._episode_benchmark_family(episode)
            if benchmark_family in _LATE_WAVE_ROTATION_FAMILIES:
                return [benchmark_family]
            return []
        window = max(len(_LATE_WAVE_ROTATION_FAMILIES) * 2, len(_LATE_WAVE_ROTATION_FAMILIES))
        return lineage_families[-window:]

    def _late_wave_rotation_family_counts(self, episode: EpisodeRecord) -> dict[str, int]:
        counts = {family: 0 for family in _LATE_WAVE_ROTATION_FAMILIES}
        for family in self._episode_recent_late_wave_families(episode):
            counts[family] += 1
        return counts

    def _late_wave_rotation_candidate_families(self, source_family: str) -> list[str]:
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return []
        source_index = _LATE_WAVE_ROTATION_FAMILIES.index(source_family)
        ordered: list[str] = []
        for offset in range(1, len(_LATE_WAVE_ROTATION_FAMILIES)):
            ordered.append(
                _LATE_WAVE_ROTATION_FAMILIES[
                    (source_index + offset) % len(_LATE_WAVE_ROTATION_FAMILIES)
                ]
            )
        return ordered

    def _lineage_scheduler_signal(
        self,
        episode: EpisodeRecord,
        *,
        source_family: str,
        target_family: str,
    ) -> dict[str, int | bool]:
        parent_surface = self._episode_long_horizon_surface(episode)
        branch_kind = self._adjacent_success_lineage_branch_kind(
            episode,
            source_family,
            parent_surface or source_family,
        )
        lineage_depth = self._episode_lineage_depth(episode)
        recent_family_repetition = self._episode_recent_family_repetition(episode, source_family)
        branch_saturation = self._episode_recent_branch_saturation(episode, branch_kind)
        is_late_wave = lineage_depth >= 10
        late_wave_counts = self._late_wave_rotation_family_counts(episode)
        source_family_count = int(late_wave_counts.get(source_family, 0))
        target_family_count = int(late_wave_counts.get(target_family, 0))
        coverage_gap = sum(1 for count in late_wave_counts.values() if count == 0)
        branch_outward = False
        if target_family != source_family and is_late_wave:
            if coverage_gap > 0 and target_family_count == 0:
                branch_outward = recent_family_repetition >= 1 or branch_saturation >= 2
            elif target_family_count < source_family_count:
                branch_outward = (
                    recent_family_repetition >= 2
                    or branch_saturation >= 3
                    or lineage_depth >= 12
                )
            elif (
                source_family == _LATE_WAVE_ROTATION_FAMILIES[-1]
                and target_family == _LATE_WAVE_ROTATION_FAMILIES[0]
            ):
                branch_outward = recent_family_repetition >= 1 or branch_saturation >= 1
        return {
            "lineage_depth": lineage_depth,
            "recent_family_repetition": recent_family_repetition,
            "branch_saturation": branch_saturation,
            "is_late_wave": is_late_wave,
            "branch_outward": branch_outward,
            "rotation_coverage_gap": coverage_gap,
            "source_family_count": source_family_count,
            "target_family_count": target_family_count,
        }

    def _lineage_should_branch_outward(
        self,
        episode: EpisodeRecord,
        *,
        source_family: str,
        target_family: str,
    ) -> bool:
        if target_family == source_family:
            return True
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return True
        signal = self._lineage_scheduler_signal(
            episode,
            source_family=source_family,
            target_family=target_family,
        )
        if not bool(signal["is_late_wave"]):
            return source_family in {"validation", "governance", "oversight"}
        adaptive_target_family = self._late_wave_rotation_target_family(
            episode,
            source_family=source_family,
        )
        if not adaptive_target_family:
            return False
        return adaptive_target_family == target_family

    def _late_wave_rotation_target_family(
        self,
        episode: EpisodeRecord,
        *,
        source_family: str,
    ) -> str:
        if source_family not in _LATE_WAVE_ROTATION_FAMILIES:
            return ""
        late_wave_counts = self._late_wave_rotation_family_counts(episode)
        candidate_families = self._late_wave_rotation_candidate_families(source_family)
        if not candidate_families:
            return ""
        next_family = min(
            candidate_families,
            key=lambda family: (int(late_wave_counts.get(family, 0)), candidate_families.index(family)),
        )
        signal = self._lineage_scheduler_signal(
            episode,
            source_family=source_family,
            target_family=next_family,
        )
        if not bool(signal["branch_outward"]):
            return ""
        return next_family

    def _long_horizon_seed_priority(self, episode: EpisodeRecord) -> int:
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return 0
        long_horizon_surface = self._episode_long_horizon_surface(episode)
        priority = 0
        if long_horizon_surface == "shared_repo_integrator":
            priority += 3
        elif long_horizon_surface == "shared_repo_synthetic_worker":
            priority += 2
        step_count = int(episode.task_metadata.get("long_horizon_step_count", 0) or len(episode.steps) or 0)
        priority += min(2, max(0, step_count) // 4)
        target_family = self._adjacent_success_target_benchmark_family(episode)
        signal = self._lineage_scheduler_signal(
            episode,
            source_family=self._episode_benchmark_family(episode),
            target_family=target_family,
        )
        if bool(signal["branch_outward"]):
            priority += 3
            priority += min(2, int(signal["rotation_coverage_gap"]))
        elif bool(signal["is_late_wave"]) and int(signal["branch_saturation"]) >= 3:
            priority -= 1
        return priority

    def _adjacent_success_seed_group(self, episode: EpisodeRecord) -> str:
        benchmark_family = self._adjacent_success_target_benchmark_family(episode)
        if not self._episode_uses_long_horizon_adjacent_curriculum(episode):
            return benchmark_family
        variant = self._long_horizon_adjacent_variant(episode, benchmark_family)
        long_horizon_surface = self._episode_long_horizon_surface(episode)
        origin_benchmark_family = self._episode_origin_benchmark_family(episode)
        lineage_branch_kind = self._adjacent_success_lineage_branch_kind(
            episode,
            benchmark_family,
            variant,
        )
        repo_id = self._episode_shared_repo_id(episode)
        if long_horizon_surface == "shared_repo_synthetic_worker":
            worker_branch = self._episode_shared_repo_worker_branch(episode)
            if repo_id and worker_branch:
                return (
                    f"{benchmark_family}:{variant}:{origin_benchmark_family}:{lineage_branch_kind}:"
                    f"{repo_id}:{worker_branch}"
                )
        if long_horizon_surface == "shared_repo_integrator" and repo_id and benchmark_family == "project":
            return f"{benchmark_family}:{variant}:{origin_benchmark_family}:{lineage_branch_kind}:{repo_id}"
        return f"{benchmark_family}:{variant}:{origin_benchmark_family}:{lineage_branch_kind}"

    def _diversify_adjacent_success_seeds(self, episodes: list[EpisodeRecord]) -> list[EpisodeRecord]:
        remaining = list(episodes)
        diversified: list[EpisodeRecord] = []
        while remaining:
            seen_groups: set[str] = set()
            next_round: list[EpisodeRecord] = []
            for episode in remaining:
                group = self._adjacent_success_seed_group(episode)
                if group in seen_groups:
                    next_round.append(episode)
                    continue
                diversified.append(episode)
                seen_groups.add(group)
            remaining = next_round
        return diversified

    def _prioritize_adjacent_success_shared_repo_bundles(
        self,
        episodes: list[EpisodeRecord],
    ) -> list[EpisodeRecord]:
        if len(episodes) <= 1:
            return list(episodes)
        bundle_state = self._adjacent_success_shared_repo_bundle_state(episodes)
        if not bundle_state:
            return list(episodes)
        ordered: list[EpisodeRecord] = []
        used_task_ids: set[str] = set()
        for episode in episodes:
            if episode.task_id in used_task_ids:
                continue
            repo_id = self._episode_shared_repo_id(episode)
            surface = self._episode_long_horizon_surface(episode)
            if not repo_id or surface not in {"shared_repo_synthetic_worker", "shared_repo_integrator"}:
                ordered.append(episode)
                used_task_ids.add(episode.task_id)
                continue
            state = bundle_state.get(repo_id, {})
            required_branches = {
                str(branch).strip()
                for branch in state.get("required_branches", set())
                if str(branch).strip()
            }
            required_branch_order = [
                str(branch).strip()
                for branch in state.get("required_branch_order", [])
                if str(branch).strip()
            ]
            worker_episodes = [
                candidate
                for candidate in episodes
                if candidate.task_id not in used_task_ids
                and self._episode_shared_repo_id(candidate) == repo_id
                and self._episode_long_horizon_surface(candidate) == "shared_repo_synthetic_worker"
            ]
            integrator_episodes = [
                candidate
                for candidate in episodes
                if candidate.task_id not in used_task_ids
                and self._episode_shared_repo_id(candidate) == repo_id
                and self._episode_long_horizon_surface(candidate) == "shared_repo_integrator"
            ]
            worker_episodes.sort(
                key=lambda candidate: (
                    self._shared_repo_worker_branch_index(
                        self._episode_shared_repo_worker_branch(candidate),
                        required_branch_order=required_branch_order,
                    ),
                    candidate.task_id,
                )
            )
            bundle_complete = bool(required_branches) and required_branches.issubset(
                {
                    self._episode_shared_repo_worker_branch(candidate)
                    for candidate in worker_episodes
                    if self._episode_shared_repo_worker_branch(candidate)
                }
            )
            if bundle_complete:
                for worker_episode in worker_episodes:
                    if worker_episode.task_id in used_task_ids:
                        continue
                    ordered.append(worker_episode)
                    used_task_ids.add(worker_episode.task_id)
                for integrator_episode in integrator_episodes:
                    if integrator_episode.task_id in used_task_ids:
                        continue
                    ordered.append(integrator_episode)
                    used_task_ids.add(integrator_episode.task_id)
                continue
            if surface == "shared_repo_integrator":
                continue
            ordered.append(episode)
            used_task_ids.add(episode.task_id)
        for episode in episodes:
            if episode.task_id in used_task_ids:
                continue
            ordered.append(episode)
            used_task_ids.add(episode.task_id)
        return ordered

    def _adjacent_success_shared_repo_bundle_state(
        self,
        episodes: list[EpisodeRecord],
    ) -> dict[str, dict[str, set[str] | list[str]]]:
        state: dict[str, dict[str, set[str] | list[str]]] = {}
        for episode in episodes:
            repo_id = self._episode_shared_repo_id(episode)
            if not repo_id:
                continue
            entry = state.setdefault(
                repo_id,
                {
                    "worker_branches": set(),
                    "required_branches": set(),
                    "required_branch_order": [],
                },
            )
            worker_branch = self._episode_shared_repo_worker_branch(episode)
            if worker_branch:
                worker_branches = entry.setdefault("worker_branches", set())
                if isinstance(worker_branches, set):
                    worker_branches.add(worker_branch)
            required_branches = entry.setdefault("required_branches", set())
            if isinstance(required_branches, set):
                required_branches.update(self._episode_shared_repo_required_merged_branches(episode))
            required_branch_order = entry.setdefault("required_branch_order", [])
            if isinstance(required_branch_order, list):
                for branch in self._episode_shared_repo_required_merged_branches(episode):
                    if branch and branch not in required_branch_order:
                        required_branch_order.append(branch)
        return state

    def _adjacent_success_shared_repo_bundle_priority(
        self,
        episode: EpisodeRecord,
        *,
        bundle_state: dict[str, dict[str, set[str] | list[str]]],
    ) -> int:
        repo_id = self._episode_shared_repo_id(episode)
        if not repo_id:
            return 0
        state = bundle_state.get(repo_id, {})
        required_branches = {
            str(branch).strip()
            for branch in state.get("required_branches", set())
            if str(branch).strip()
        }
        worker_branches = {
            str(branch).strip()
            for branch in state.get("worker_branches", set())
            if str(branch).strip()
        }
        surface = self._episode_long_horizon_surface(episode)
        bundle_complete = bool(required_branches) and required_branches.issubset(worker_branches)
        if surface == "shared_repo_synthetic_worker":
            worker_branch = self._episode_shared_repo_worker_branch(episode)
            if worker_branch and worker_branch in required_branches:
                return 5 if bundle_complete else 4
            return 2
        if surface == "shared_repo_integrator":
            if required_branches and not bundle_complete:
                return -4
            return 1 if bundle_complete else 0
        return 0

    def _adjacent_success_shared_repo_incremental_bonus(
        self,
        episode: EpisodeRecord,
        *,
        bundle_state: dict[str, dict[str, set[str] | list[str]]],
        selected_worker_branches: dict[str, set[str]],
    ) -> int:
        repo_id = self._episode_shared_repo_id(episode)
        if not repo_id:
            return 0
        state = bundle_state.get(repo_id, {})
        required_branches = {
            str(branch).strip()
            for branch in state.get("required_branches", set())
            if str(branch).strip()
        }
        surface = self._episode_long_horizon_surface(episode)
        selected_required_branches = {
            str(branch).strip()
            for branch in selected_worker_branches.get(repo_id, set())
            if str(branch).strip()
        }
        if surface == "shared_repo_synthetic_worker":
            worker_branch = self._episode_shared_repo_worker_branch(episode)
            if worker_branch and worker_branch in required_branches and worker_branch not in selected_required_branches:
                return 6
            return 0
        if surface == "shared_repo_integrator" and not required_branches:
            return 4
        if surface == "shared_repo_integrator" and required_branches:
            if required_branches.issubset(selected_required_branches):
                return 12
            if selected_required_branches.intersection(required_branches):
                return -4
            return -8
        return 0

    def _selected_shared_repo_worker_branches_after_pick(
        self,
        selected_worker_branches: dict[str, set[str]],
        episode: EpisodeRecord,
    ) -> dict[str, set[str]]:
        updated = {repo_id: set(branches) for repo_id, branches in dict(selected_worker_branches).items()}
        repo_id = self._episode_shared_repo_id(episode)
        worker_branch = self._episode_shared_repo_worker_branch(episode)
        if repo_id and worker_branch and self._episode_long_horizon_surface(episode) == "shared_repo_synthetic_worker":
            updated.setdefault(repo_id, set()).add(worker_branch)
        return updated

    def _episode_shared_repo_id(self, episode: EpisodeRecord) -> str:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        repo_id = str(workflow_guard.get("shared_repo_id", "")).strip()
        if repo_id:
            return repo_id
        if (
            self._episode_benchmark_family(episode) == "project"
            and self._episode_long_horizon_surface(episode) in {"shared_repo_synthetic_worker", "shared_repo_integrator"}
        ):
            return "repo_sandbox_parallel_merge"
        return ""

    @staticmethod
    def _episode_shared_repo_worker_branch(episode: EpisodeRecord) -> str:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        return str(workflow_guard.get("worker_branch", "")).strip()

    @staticmethod
    def _episode_shared_repo_required_merged_branches(episode: EpisodeRecord) -> list[str]:
        metadata = CurriculumEngine._episode_curriculum_metadata(episode)
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        return [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]

    @staticmethod
    def _shared_repo_worker_branch_index(
        worker_branch: str,
        *,
        required_branch_order: list[str],
    ) -> int:
        if worker_branch and worker_branch in required_branch_order:
            return required_branch_order.index(worker_branch)
        return len(required_branch_order)

    @staticmethod
    def _reference_commands_for_document(document: dict[str, object]) -> list[str]:
        if not bool(document.get("success", False)):
            return []
        fragments = document.get("fragments", [])
        commands: list[str] = []
        if isinstance(fragments, list):
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("kind") != "command" or not fragment.get("passed", False):
                    continue
                command = str(fragment.get("command", "")).strip()
                if command and command not in commands:
                    commands.append(command)
        if commands:
            return commands
        summary = document.get("summary", {})
        if not isinstance(summary, dict):
            return []
        return [str(command).strip() for command in summary.get("executed_commands", []) if str(command).strip()]

    @classmethod
    def _reference_commands_for_episode(cls, episode: EpisodeRecord) -> list[str]:
        commands: list[str] = []
        for step in episode.steps:
            if step.action != "code_execute":
                continue
            command = str(step.content).strip()
            if not command:
                continue
            verification = step.verification if isinstance(step.verification, dict) else {}
            if bool(verification.get("passed", False)) and command not in commands:
                commands.append(command)
        return commands

    def _memory_documents(self) -> list[dict]:
        if self.memory is None:
            return []
        return self.memory.list_documents()

    @classmethod
    def _episode_context_terms(cls, episode: EpisodeRecord) -> list[str]:
        contract = episode.task_contract if isinstance(episode.task_contract, dict) else {}
        metadata = cls._episode_curriculum_metadata(episode)
        latest_commands = [
            step.content
            for step in episode.steps
            if step.action == "code_execute" and str(step.content).strip()
        ][-3:]
        failure_signals: list[str] = []
        for step in episode.steps:
            failure_signals.extend(str(value).strip() for value in step.failure_signals if str(value).strip())
        return semantic_query_terms(
            episode.task_id,
            episode.prompt,
            metadata,
            contract,
            cls._failure_types(episode),
            failure_signals,
            latest_commands,
        )

    @classmethod
    def _document_context_terms(cls, document: dict[str, object]) -> list[str]:
        return semantic_query_terms(
            document.get("task_id", ""),
            document.get("prompt", ""),
            document.get("task_metadata", {}),
            document.get("metadata", {}),
            document.get("task_contract", {}),
            document.get("summary", {}),
            document.get("fragments", []),
        )

    @classmethod
    def _document_repo_semantic_clusters(cls, document: dict[str, object]) -> list[str]:
        metadata = document.get("task_metadata", {})
        if not isinstance(metadata, dict):
            metadata = document.get("metadata", {}) if isinstance(document.get("metadata", {}), dict) else {}
        values = metadata.get("repo_semantics", []) if isinstance(metadata, dict) else []
        if not isinstance(values, list):
            values = []
        normalized = [str(value).strip().lower() for value in values if str(value).strip()]
        if not normalized:
            family = str(
                metadata.get("benchmark_family", "bounded") if isinstance(metadata, dict) else "bounded"
            ).strip().lower() or "bounded"
            normalized = [family]
        deduped: list[str] = []
        for value in normalized:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _document_metadata_list(document: dict[str, object], field: str) -> list[str]:
        metadata = document.get("task_metadata", {})
        if not isinstance(metadata, dict):
            metadata = document.get("metadata", {}) if isinstance(document.get("metadata", {}), dict) else {}
        values = metadata.get(field, []) if isinstance(metadata, dict) else []
        if not isinstance(values, list):
            return []
        return [str(value).strip().lower() for value in values if str(value).strip()]

    @staticmethod
    def _merged_commands(primary: list[str], fallback: list[str]) -> list[str]:
        merged: list[str] = []
        for command in [*primary, *fallback]:
            command_text = str(command).strip()
            if command_text and command_text not in merged:
                merged.append(command_text)
        return merged

    @classmethod
    def _failure_recovery_commands(
        cls,
        fallback: list[str],
        retrieved: list[str],
        *,
        expected_files: list[str],
        failed_command: str,
    ) -> list[str]:
        anchors = cls._command_anchors(expected_files)
        normalized_failed = " ".join(str(failed_command).strip().split())
        aligned_retrieved: list[str] = []
        for command in retrieved:
            command_text = str(command).strip()
            normalized = " ".join(command_text.split())
            if not command_text or normalized == normalized_failed:
                continue
            if anchors and not any(anchor in command_text for anchor in anchors):
                continue
            if command_text not in aligned_retrieved:
                aligned_retrieved.append(command_text)
        return cls._merged_commands(fallback, aligned_retrieved)

    @staticmethod
    def _command_anchors(expected_files: list[str]) -> list[str]:
        anchors: list[str] = []
        for path in expected_files:
            normalized = str(path).strip()
            if not normalized:
                continue
            parts = [part for part in Path(normalized).parts if part not in {".", ""}]
            for part in parts:
                if part not in anchors:
                    anchors.append(part)
        return anchors

    def _with_curriculum_hint(self, task: TaskSpec) -> TaskSpec:
        if not self.config.use_curriculum_proposals:
            return task
        path = self.config.curriculum_proposals_path
        if not path.exists():
            return task
        payload = json.loads(path.read_text(encoding="utf-8"))
        retained = retained_artifact_payload(payload, artifact_kind="curriculum_proposal_set")
        if retained is None:
            return task
        proposals = retained.get("proposals", [])
        if not isinstance(proposals, list):
            return task
        family = str(task.metadata.get("benchmark_family", "bounded"))
        hint = ""
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            suggestion = str(proposal.get("suggestion", "")).strip()
            reason = str(proposal.get("reason", "")).strip()
            if family in reason or proposal.get("area") in {"failure_recovery", "benchmark_family"}:
                hint = suggestion
                break
        if not hint:
            return task
        metadata = dict(task.metadata)
        metadata["curriculum_proposal_hint"] = hint
        return replace(task, prompt=f"{task.prompt} Curriculum guidance: {hint}", metadata=metadata)

    def _apply_curriculum_controls(self, task: TaskSpec) -> TaskSpec:
        controls = self._curriculum_controls()
        if not controls:
            return task
        suggested_commands = [str(command).strip() for command in task.suggested_commands if str(command).strip()]
        if str(task.metadata.get("curriculum_kind", "")).strip() != "failure_recovery":
            capped = self._cap_commands(
                suggested_commands,
                self._adjacent_reference_limit(),
            )
            if capped == task.suggested_commands:
                return task
            metadata = dict(task.metadata)
            metadata["curriculum_behavior_controls"] = dict(controls)
            return replace(task, suggested_commands=capped, metadata=metadata)
        min_anchor_matches = self._failure_recovery_anchor_min_matches()
        filtered_commands: list[str] = []
        if min_anchor_matches > 1:
            anchors = self._command_anchors([*task.expected_files, *task.expected_file_contents.keys()])
            for index, command_text in enumerate(suggested_commands):
                anchor_matches = sum(1 for anchor in anchors if anchor in command_text)
                if index == 0 or anchor_matches >= min_anchor_matches:
                    filtered_commands.append(command_text)
        else:
            filtered_commands = list(suggested_commands)
        if not filtered_commands:
            filtered_commands = list(suggested_commands)
        filtered_commands = self._cap_commands(
            filtered_commands,
            self._failure_recovery_command_cap(),
        )
        metadata = dict(task.metadata)
        metadata["curriculum_behavior_controls"] = dict(controls)
        return replace(task, suggested_commands=filtered_commands, metadata=metadata)

    def _curriculum_controls(self) -> dict[str, object]:
        if self._curriculum_controls_cache is not None:
            return self._curriculum_controls_cache
        if not self.config.use_curriculum_proposals:
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        path = self.config.curriculum_proposals_path
        if not path.exists():
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        payload = json.loads(path.read_text(encoding="utf-8"))
        retained = retained_artifact_payload(payload, artifact_kind="curriculum_proposal_set")
        if retained is None:
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        controls = retained.get("controls", {})
        self._curriculum_controls_cache = dict(controls) if isinstance(controls, dict) else {}
        return self._curriculum_controls_cache

    def _curriculum_control_family_list(self, field: str) -> list[str]:
        values = self._curriculum_controls().get(field, [])
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        for value in values:
            text = str(value).strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _curriculum_control_family_signal_pairs(self, field: str) -> list[tuple[str, str]]:
        values = self._curriculum_controls().get(field, [])
        if not isinstance(values, list):
            return []
        normalized: list[tuple[str, str]] = []
        for value in values:
            text = str(value).strip().lower()
            family, separator, signal = text.partition(":")
            if not separator or not family or not signal:
                continue
            pair = (family, signal)
            if pair not in normalized:
                normalized.append(pair)
        return normalized

    def _curriculum_control_float_map(self, field: str) -> dict[str, float]:
        values = self._curriculum_controls().get(field, {})
        if not isinstance(values, dict):
            return {}
        normalized: dict[str, float] = {}
        for key, value in values.items():
            label = str(key).strip().lower()
            if not label:
                continue
            try:
                normalized[label] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    def _curriculum_positive_int_control(self, field: str, *, default: int) -> int:
        value = self._curriculum_controls().get(field, default)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    def _curriculum_nonnegative_int_control(self, field: str, *, default: int) -> int:
        value = self._curriculum_controls().get(field, default)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    def _max_generated_seed_tasks(self, curriculum_kind: str) -> int:
        field = "max_generated_failure_recovery_tasks" if curriculum_kind == "failure_recovery" else "max_generated_adjacent_tasks"
        value = self._curriculum_controls().get(field, 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _success_reference_limit(self) -> int:
        value = self._curriculum_controls().get("success_reference_limit", 3)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 3

    def _context_reference_limit(self, *, success_only: bool) -> int:
        controls = self._curriculum_controls()
        field = "success_context_reference_limit" if success_only else "failure_context_reference_limit"
        value = controls.get(field, controls.get("context_reference_limit", 6))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 6

    def _adjacent_reference_limit(self) -> int:
        value = self._curriculum_controls().get("adjacent_reference_limit", self._success_reference_limit())
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return self._success_reference_limit()

    def _failure_recovery_anchor_min_matches(self) -> int:
        value = self._curriculum_controls().get("failure_recovery_anchor_min_matches", 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _failure_recovery_command_cap(self) -> int:
        value = self._curriculum_controls().get("failure_recovery_command_cap", 4)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 4

    @staticmethod
    def _episode_benchmark_family(episode: EpisodeRecord) -> str:
        return str(episode.task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"

    @staticmethod
    def _cap_commands(commands: list[str], limit: int) -> list[str]:
        limited: list[str] = []
        for command in commands:
            command_text = str(command).strip()
            if command_text and command_text not in limited:
                limited.append(command_text)
            if len(limited) >= max(1, limit):
                break
        return limited

    @staticmethod
    def _failure_types(episode: EpisodeRecord) -> list[str]:
        failure_types: set[str] = set()
        for step in episode.steps:
            for reason in step.verification.get("reasons", []):
                lowered = reason.lower()
                if "missing expected file" in lowered:
                    failure_types.add("missing_expected_file")
                elif "forbidden file present" in lowered:
                    failure_types.add("forbidden_file_present")
                elif "unexpected file content" in lowered:
                    failure_types.add("unexpected_file_content")
                elif "forbidden output present" in lowered:
                    failure_types.add("forbidden_output_present")
                elif "exit code" in lowered:
                    failure_types.add("command_failure")
                elif "timed out" in lowered:
                    failure_types.add("timeout")
                elif "repeated failed action" in lowered:
                    failure_types.add("repeated_failed_action")
            for signal in step.failure_signals:
                normalized = str(signal).strip()
                if normalized:
                    failure_types.add(normalized)
        if episode.termination_reason:
            failure_types.add(episode.termination_reason)
        return sorted(failure_types)

    @staticmethod
    def _latest_command(episode: EpisodeRecord) -> str:
        for step in reversed(episode.steps):
            if step.action == "code_execute" and step.content:
                return step.content
        return ""

    @staticmethod
    def _failure_pattern(episode: EpisodeRecord, latest_command: str) -> str:
        workspace_name = Path(episode.workspace).name.strip()
        if workspace_name and f"{workspace_name}/" in latest_command:
            return "workspace_prefixed_path"
        return "generic_recovery"

    @staticmethod
    def _benchmark_family(task_id: str, documents: list[dict]) -> str:
        for document in documents:
            task_metadata = document.get("task_metadata", {})
            family = str(task_metadata.get("benchmark_family", "")).strip()
            if family:
                return family
            metadata = document.get("metadata", {})
            family = str(metadata.get("benchmark_family", "")).strip()
            if family:
                return family
        try:
            try:
                manifest_paths = current_external_task_manifests_paths()
                bank = TaskBank(
                    config=KernelConfig(),
                    external_task_manifests=manifest_paths if manifest_paths else None,
                )
            except TypeError:
                bank = TaskBank()
            return str(bank.get(task_id).metadata.get("benchmark_family", "bounded"))
        except KeyError:
            return "bounded"
