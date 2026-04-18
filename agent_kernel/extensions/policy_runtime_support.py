from __future__ import annotations

import json
from pathlib import Path

from ..config import KernelConfig
from ..extensions.improvement.improvement_common import artifact_payload_in_lifecycle_states, retained_artifact_payload
from ..extensions.improvement.policy_improvement import (
    dedupe_prompt_adjustments,
    retained_policy_controls,
    retained_role_directives,
    retained_tolbert_decoder_policy_overrides,
    retained_tolbert_hybrid_scoring_policy_overrides,
    retained_tolbert_rollout_policy_overrides,
    retained_tolbert_runtime_policy_overrides,
)
from ..extensions.improvement.retrieval_improvement import retained_retrieval_overrides
from ..extensions.improvement.state_estimation_improvement import (
    retained_state_estimation_payload,
    retained_state_estimation_policy_controls,
)
from ..extensions.improvement.transition_model_improvement import (
    retained_transition_model_controls,
    retained_transition_model_signatures,
)
from ..extensions.policy_command_utils import canonicalize_command as _canonicalize_command
from ..extensions.runtime_modeling_adapter import (
    load_model_artifact,
    retained_tolbert_action_generation_policy,
    retained_tolbert_active_decoder_runtime,
    retained_tolbert_decoder_policy,
    retained_tolbert_hybrid_runtime,
    retained_tolbert_model_surfaces,
    retained_tolbert_rollout_policy,
    retained_tolbert_runtime_policy,
    retained_tolbert_universal_decoder_runtime,
    score_hybrid_candidates,
)
from ..resource_registry import runtime_resource_registry
from ..resource_types import PROMPT_RESOURCE_DECISION, PROMPT_RESOURCE_SYSTEM, subsystem_resource_id


class SkillLibrary:
    def __init__(self, skills: list[dict[str, object]], min_quality: float = 0.0) -> None:
        self.min_quality = min_quality
        self.skills = sorted(
            [skill for skill in skills if self._skill_quality(skill) >= self.min_quality],
            key=lambda skill: (
                -self._skill_quality(skill),
                len(self._commands_for_skill(skill)),
                str(skill.get("skill_id", "")),
            ),
        )

    @classmethod
    def from_path(cls, path: Path, *, min_quality: float = 0.0) -> "SkillLibrary":
        if not path.exists():
            return cls([], min_quality=min_quality)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if artifact_payload_in_lifecycle_states(
                payload,
                artifact_kind="skill_set",
                allowed_states={"promoted", "retained"},
            ) is None:
                return cls([], min_quality=min_quality)
        skills = payload.get("skills", payload) if isinstance(payload, dict) else payload
        return cls(skills, min_quality=min_quality)

    def summarize_for_task(self, task_id: str, source_task_id: str = "") -> list[dict[str, object]]:
        return [
            self._to_summary(skill, task_id)
            for skill in self.matching_skills(task_id, source_task_id=source_task_id)[:3]
        ]

    def matching_skills(self, task_id: str, *, source_task_id: str = "") -> list[dict[str, object]]:
        task_ids = {task_id}
        if source_task_id:
            task_ids.add(source_task_id)
        return [skill for skill in self.skills if any(self._skill_matches_task(skill, candidate) for candidate in task_ids)]

    def best_skill_match(
        self,
        *,
        task_id: str,
        preferred_task_ids: list[str] | None = None,
        recommended_commands: list[str] | None = None,
        blocked_commands: list[str] | None = None,
    ) -> dict[str, object] | None:
        preferred = preferred_task_ids or []
        recommended = {self._normalize_command(command) for command in recommended_commands or []}
        blocked = {self._normalize_command(command) for command in blocked_commands or []}

        matches = self.matching_skills(task_id)
        if not matches and preferred:
            for preferred_task_id in preferred:
                matches = self.matching_skills(preferred_task_id)
                if matches:
                    break
        ranked = sorted(
            matches,
            key=lambda skill: (
                -self._skill_context_rank(
                    skill,
                    preferred_task_ids=preferred,
                    recommended_commands=recommended,
                    blocked_commands=blocked,
                ),
                -self._skill_quality(skill),
                len(self._commands_for_skill(skill)),
                str(skill.get("skill_id", "")),
            ),
        )
        return ranked[0] if ranked else None

    @staticmethod
    def _commands_for_skill(skill: dict[str, object]) -> list[str]:
        commands = skill.get("procedure", {}).get("commands", [])
        if commands:
            return [str(command) for command in commands]
        return [str(command) for command in skill.get("commands", [])]

    @staticmethod
    def _skill_matches_task(skill: dict[str, object], task_id: str) -> bool:
        applicable_tasks = [str(value) for value in skill.get("applicable_tasks", [])]
        if applicable_tasks:
            return task_id in applicable_tasks
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", "")))
        return source_task_id == task_id

    @staticmethod
    def _skill_quality(skill: dict[str, object]) -> float:
        try:
            return float(skill.get("quality", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @classmethod
    def _skill_context_rank(
        cls,
        skill: dict[str, object],
        *,
        preferred_task_ids: list[str],
        recommended_commands: set[str],
        blocked_commands: set[str],
    ) -> int:
        rank = 0
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", "")))
        if source_task_id in preferred_task_ids:
            rank += 8
        commands = cls._commands_for_skill(skill)
        if commands:
            first_command = cls._normalize_command(commands[0])
            if first_command in recommended_commands:
                rank += 10
            if first_command in blocked_commands:
                rank -= 10
        return rank

    @classmethod
    def _normalize_command(cls, command: str) -> str:
        return _canonicalize_command(command)

    @classmethod
    def _to_summary(cls, skill: dict[str, object], task_id: str) -> dict[str, object]:
        return {
            "skill_id": skill.get("skill_id", f"skill:{skill.get('task_id', task_id)}:primary"),
            "kind": skill.get("kind", "command_sequence"),
            "source_task_id": skill.get("source_task_id", skill.get("task_id")),
            "command_count": len(cls._commands_for_skill(skill)),
            "quality": cls._skill_quality(skill),
            "known_failure_types": skill.get(
                "known_failure_types",
                skill.get("failure_types", []),
            ),
        }


class PolicyRuntimeSupport:
    def __init__(self, *, config: KernelConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self._resource_registry = runtime_resource_registry(config, repo_root=repo_root)
        self._policy_controls_cache: dict[str, object] | None = None
        self._role_directives_cache: dict[str, str] | None = None
        self._state_estimation_policy_controls_cache: dict[str, object] | None = None
        self._transition_model_controls_cache: dict[str, object] | None = None
        self._transition_model_signatures_cache: list[dict[str, object]] | None = None
        self._tolbert_model_payload_cache: dict[str, object] | None = None
        self._tolbert_runtime_policy_cache: dict[str, object] | None = None
        self._tolbert_model_surfaces_cache: dict[str, object] | None = None
        self._tolbert_decoder_policy_cache: dict[str, object] | None = None
        self._tolbert_action_generation_policy_cache: dict[str, object] | None = None
        self._tolbert_rollout_policy_cache: dict[str, object] | None = None
        self._tolbert_hybrid_runtime_cache: dict[str, object] | None = None
        self._tolbert_universal_decoder_runtime_cache: dict[str, object] | None = None
        self._tolbert_active_decoder_runtime_cache: dict[str, object] | None = None
        self._prompt_policy_payload_cache: dict[str, object] | None = None
        self.last_hybrid_runtime_error: str = ""

    def prompt_template(self, name: str) -> str:
        resource_id = {
            "system": PROMPT_RESOURCE_SYSTEM,
            "decision": PROMPT_RESOURCE_DECISION,
        }.get(str(name).strip().lower())
        if not resource_id:
            raise ValueError(f"unsupported prompt template: {name}")
        return self._resource_registry.load_text(resource_id)

    def retrieval_overrides(self) -> dict[str, object]:
        if not self.config.use_retrieval_proposals:
            return {}
        path = self.config.retrieval_proposals_path
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return retained_retrieval_overrides(payload)

    def policy_controls(self) -> dict[str, object]:
        if self._policy_controls_cache is not None:
            return self._policy_controls_cache
        self._policy_controls_cache = retained_policy_controls(self.prompt_policy_payload())
        return self._policy_controls_cache

    def tolbert_model_payload(self) -> dict[str, object]:
        if self._tolbert_model_payload_cache is not None:
            return self._tolbert_model_payload_cache
        if not self.config.use_tolbert_model_artifacts:
            self._tolbert_model_payload_cache = {}
            return self._tolbert_model_payload_cache
        self._tolbert_model_payload_cache = load_model_artifact(self.config.tolbert_model_artifact_path)
        return self._tolbert_model_payload_cache

    def tolbert_runtime_policy(self) -> dict[str, object]:
        if self._tolbert_runtime_policy_cache is not None:
            return self._tolbert_runtime_policy_cache
        normalized = retained_tolbert_runtime_policy(self.tolbert_model_payload())
        overrides = retained_tolbert_runtime_policy_overrides(self.prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        retrieval_overrides = self.retrieval_overrides()
        if retrieval_overrides:
            try:
                deterministic_confidence = float(
                    retrieval_overrides.get(
                        "tolbert_deterministic_command_confidence",
                        normalized.get("min_path_confidence", 0.75),
                    )
                )
            except (TypeError, ValueError):
                deterministic_confidence = float(normalized.get("min_path_confidence", 0.75) or 0.75)
            try:
                direct_command_min_score = int(retrieval_overrides.get("tolbert_direct_command_min_score", 0))
            except (TypeError, ValueError):
                direct_command_min_score = 0
            normalized["min_path_confidence"] = max(
                float(normalized.get("min_path_confidence", 0.75) or 0.75),
                deterministic_confidence,
            )
            if direct_command_min_score > 0:
                normalized["require_trusted_retrieval"] = True
            normalized["primary_min_command_score"] = max(
                int(normalized.get("primary_min_command_score", 2) or 2),
                direct_command_min_score,
            )
        self._tolbert_runtime_policy_cache = normalized
        return self._tolbert_runtime_policy_cache

    def tolbert_model_surfaces(self) -> dict[str, object]:
        if self._tolbert_model_surfaces_cache is not None:
            return self._tolbert_model_surfaces_cache
        self._tolbert_model_surfaces_cache = retained_tolbert_model_surfaces(self.tolbert_model_payload())
        return self._tolbert_model_surfaces_cache

    def tolbert_decoder_policy(self) -> dict[str, object]:
        if self._tolbert_decoder_policy_cache is not None:
            return self._tolbert_decoder_policy_cache
        normalized = retained_tolbert_decoder_policy(self.tolbert_model_payload())
        overrides = retained_tolbert_decoder_policy_overrides(self.prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        self._tolbert_decoder_policy_cache = normalized
        return self._tolbert_decoder_policy_cache

    def tolbert_action_generation_policy(self) -> dict[str, object]:
        if self._tolbert_action_generation_policy_cache is not None:
            return self._tolbert_action_generation_policy_cache
        self._tolbert_action_generation_policy_cache = retained_tolbert_action_generation_policy(
            self.tolbert_model_payload()
        )
        return self._tolbert_action_generation_policy_cache

    def tolbert_rollout_policy(self) -> dict[str, object]:
        if self._tolbert_rollout_policy_cache is not None:
            return self._tolbert_rollout_policy_cache
        normalized = retained_tolbert_rollout_policy(self.tolbert_model_payload())
        overrides = retained_tolbert_rollout_policy_overrides(self.prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        self._tolbert_rollout_policy_cache = normalized
        return self._tolbert_rollout_policy_cache

    def tolbert_hybrid_runtime(self) -> dict[str, object]:
        if self._tolbert_hybrid_runtime_cache is not None:
            return self._tolbert_hybrid_runtime_cache
        normalized = retained_tolbert_hybrid_runtime(self.tolbert_model_payload())
        scoring_overrides = retained_tolbert_hybrid_scoring_policy_overrides(self.prompt_policy_payload())
        if scoring_overrides:
            scoring_policy = normalized.get("scoring_policy", {})
            merged = dict(scoring_policy) if isinstance(scoring_policy, dict) else {}
            merged.update(scoring_overrides)
            normalized["scoring_policy"] = merged
        self._tolbert_hybrid_runtime_cache = normalized
        return self._tolbert_hybrid_runtime_cache

    def tolbert_universal_decoder_runtime(self) -> dict[str, object]:
        if self._tolbert_universal_decoder_runtime_cache is not None:
            return self._tolbert_universal_decoder_runtime_cache
        self._tolbert_universal_decoder_runtime_cache = retained_tolbert_universal_decoder_runtime(
            self.tolbert_model_payload()
        )
        return self._tolbert_universal_decoder_runtime_cache

    def tolbert_active_decoder_runtime(self) -> dict[str, object]:
        if self._tolbert_active_decoder_runtime_cache is not None:
            return self._tolbert_active_decoder_runtime_cache
        universal_runtime = self.tolbert_universal_decoder_runtime()
        if bool(universal_runtime.get("materialized", False)) and str(
            universal_runtime.get("bundle_manifest_path", "")
        ).strip():
            normalized = dict(self.tolbert_hybrid_runtime())
            normalized.update(universal_runtime)
            normalized["runtime_key"] = "universal_decoder_runtime"
            normalized["runtime_role"] = "universal_decoder_runtime"
            normalized["supports_prompt_completion_surface"] = bool(
                universal_runtime.get("supports_prompt_completion_surface", True)
            )
            normalized["supports_state_conditioned_generation"] = bool(
                universal_runtime.get("supports_state_conditioned_generation", True)
            )
            self._tolbert_active_decoder_runtime_cache = normalized
            return self._tolbert_active_decoder_runtime_cache
        normalized = retained_tolbert_active_decoder_runtime(self.tolbert_model_payload())
        if self._tolbert_hybrid_runtime_cache is not None:
            normalized.update(self._tolbert_hybrid_runtime_cache)
        normalized["runtime_key"] = "hybrid_runtime"
        normalized["runtime_role"] = "hybrid_runtime"
        normalized["supports_prompt_completion_surface"] = True
        normalized["supports_state_conditioned_generation"] = bool(normalized.get("supports_decoder_surface", False))
        self._tolbert_active_decoder_runtime_cache = normalized
        return self._tolbert_active_decoder_runtime_cache

    def hybrid_scored_candidates(
        self,
        *,
        state,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        runtime = self.tolbert_hybrid_runtime()
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        if not manifest_raw:
            self.last_hybrid_runtime_error = "missing bundle_manifest_path"
            raise RuntimeError("hybrid runtime is enabled but bundle_manifest_path is missing")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = self.repo_root / manifest_path
        if not manifest_path.exists():
            self.last_hybrid_runtime_error = f"bundle manifest does not exist: {manifest_path}"
            raise RuntimeError(f"hybrid runtime bundle manifest does not exist: {manifest_path}")
        self.last_hybrid_runtime_error = ""
        try:
            return score_hybrid_candidates(
                state=state,
                candidates=candidates,
                bundle_manifest_path=manifest_path,
                device=str(runtime.get("preferred_device", "cpu")).strip() or "cpu",
                scoring_policy=runtime.get("scoring_policy", {}),
            )
        except Exception as exc:
            self.last_hybrid_runtime_error = str(exc).strip() or exc.__class__.__name__
            raise RuntimeError(
                f"hybrid runtime scoring failed for {manifest_path}: {self.last_hybrid_runtime_error}"
            ) from exc

    def role_directive_overrides(self) -> dict[str, str]:
        if self._role_directives_cache is not None:
            return self._role_directives_cache
        self._role_directives_cache = retained_role_directives(self.prompt_policy_payload())
        return self._role_directives_cache

    def prompt_policy_payload(self) -> dict[str, object]:
        if self._prompt_policy_payload_cache is not None:
            return self._prompt_policy_payload_cache
        if not self.config.use_prompt_proposals:
            self._prompt_policy_payload_cache = {}
            return self._prompt_policy_payload_cache
        payload = self._resource_registry.load_json(subsystem_resource_id("policy"))
        self._prompt_policy_payload_cache = payload if isinstance(payload, dict) else {}
        return self._prompt_policy_payload_cache

    def transition_model_controls(self) -> dict[str, object]:
        if self._transition_model_controls_cache is not None:
            return self._transition_model_controls_cache
        if not self.config.use_transition_model_proposals:
            self._transition_model_controls_cache = {}
            return self._transition_model_controls_cache
        path = self.config.transition_model_proposals_path
        if not path.exists():
            self._transition_model_controls_cache = {}
            return self._transition_model_controls_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._transition_model_controls_cache = {}
            return self._transition_model_controls_cache
        self._transition_model_controls_cache = retained_transition_model_controls(payload)
        return self._transition_model_controls_cache

    def state_estimation_policy_controls(self) -> dict[str, object]:
        if self._state_estimation_policy_controls_cache is not None:
            return dict(self._state_estimation_policy_controls_cache)
        payload = retained_state_estimation_payload(self.config)
        self._state_estimation_policy_controls_cache = retained_state_estimation_policy_controls(payload)
        return dict(self._state_estimation_policy_controls_cache)

    def transition_model_signatures(self) -> list[dict[str, object]]:
        if self._transition_model_signatures_cache is not None:
            return self._transition_model_signatures_cache
        if not self.config.use_transition_model_proposals:
            self._transition_model_signatures_cache = []
            return self._transition_model_signatures_cache
        path = self.config.transition_model_proposals_path
        if not path.exists():
            self._transition_model_signatures_cache = []
            return self._transition_model_signatures_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._transition_model_signatures_cache = []
            return self._transition_model_signatures_cache
        self._transition_model_signatures_cache = retained_transition_model_signatures(payload)
        return self._transition_model_signatures_cache

    def policy_control_float(self, field: str) -> float:
        value = self.policy_controls().get(field, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def policy_control_int(self, field: str) -> int:
        value = self.policy_controls().get(field, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def transition_model_control_int(self, field: str, default: int) -> int:
        value = self.transition_model_controls().get(field, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def active_prompt_adjustments(self) -> list[dict[str, object]]:
        if not self.config.use_prompt_proposals:
            return []
        path = self.config.prompt_proposals_path
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        retained = retained_artifact_payload(payload, artifact_kind="prompt_proposal_set")
        if retained is None:
            return []
        proposals = retained.get("proposals", [])
        if not isinstance(proposals, list):
            return []
        deduped = dedupe_prompt_adjustments([proposal for proposal in proposals if isinstance(proposal, dict)])
        return deduped[:3]
