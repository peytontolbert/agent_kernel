from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Protocol

from .actions import ALLOWED_ACTIONS, CODE_EXECUTE
from .config import KernelConfig
from .context_budget import ContextBudgeter
from .improvement_common import artifact_payload_in_lifecycle_states, retained_artifact_payload
from .llm import LLMClient, coerce_action_decision
from .modeling.artifacts import (
    load_model_artifact,
    retained_tolbert_action_generation_policy,
    retained_tolbert_decoder_policy,
    retained_tolbert_hybrid_runtime,
    retained_tolbert_model_surfaces,
    retained_tolbert_rollout_policy,
    retained_tolbert_runtime_policy,
)
from .modeling.policy.decoder import decode_action_generation_candidates, decode_bounded_action_candidates
from .modeling.policy.runtime import choose_tolbert_route
from .modeling.tolbert.runtime import score_hybrid_candidates
from .modeling.world.latent_state import latent_command_bias
from .policy_improvement import (
    retained_policy_controls,
    retained_role_directives,
    retained_tolbert_decoder_policy_overrides,
    retained_tolbert_hybrid_scoring_policy_overrides,
    retained_tolbert_rollout_policy_overrides,
    retained_tolbert_runtime_policy_overrides,
)
from .retrieval_improvement import retained_retrieval_overrides
from .schemas import ActionDecision
from .state import AgentState
from .state_estimation_improvement import (
    retained_state_estimation_payload,
    retained_state_estimation_policy_controls,
    state_estimation_policy_bias,
)
from .transition_model_improvement import (
    retained_transition_model_controls,
    retained_transition_model_signatures,
    transition_model_command_pattern,
)
from .universe_model import UniverseModel
from .world_model import WorldModel


class Policy:
    def decide(self, state: AgentState) -> ActionDecision:
        raise NotImplementedError

    def set_decision_progress_callback(self, callback) -> None:
        del callback


class ContextCompilationError(RuntimeError):
    """Raised when the context provider cannot produce a valid context packet."""


class ContextProvider(Protocol):
    def compile(self, state: AgentState) -> object:
        ...

    def close(self) -> None:
        ...


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


class LLMDecisionPolicy(Policy):
    def __init__(
        self,
        client: LLMClient,
        context_provider: ContextProvider | None = None,
        skill_library: SkillLibrary | None = None,
        config: KernelConfig | None = None,
    ) -> None:
        self.client = client
        self.context_provider = context_provider
        self.skill_library = skill_library or SkillLibrary([])
        self.config = config or KernelConfig()
        self.context_budgeter = ContextBudgeter(self.config)
        self.universe_model = UniverseModel(config=self.config)
        self.world_model = WorldModel(config=self.config)
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        self.system_prompt = prompts_dir.joinpath("system.md").read_text(encoding="utf-8")
        self.decision_prompt = prompts_dir.joinpath("decision.md").read_text(encoding="utf-8")
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
        self._last_hybrid_runtime_error: str = ""
        self._prompt_policy_payload_cache: dict[str, object] | None = None
        self._decision_progress_callback = None

    def close(self) -> None:
        if self.context_provider is None:
            return
        close = getattr(self.context_provider, "close", None)
        if callable(close):
            close()

    def set_decision_progress_callback(self, callback) -> None:
        self._decision_progress_callback = callback

    def _emit_decision_progress(self, stage: str, **payload: object) -> None:
        if self._decision_progress_callback is None:
            return
        event = {"step_stage": str(stage).strip()}
        event.update(payload)
        self._decision_progress_callback(event)

    def decide(self, state: AgentState) -> ActionDecision:
        if self.context_provider is not None:
            self._emit_decision_progress("context_compile")
            set_progress_callback = getattr(self.context_provider, "set_progress_callback", None)
            if callable(set_progress_callback):
                set_progress_callback(self._decision_progress_callback)
            try:
                state.context_packet = self.context_provider.compile(state)
            except Exception as exc:
                raise ContextCompilationError(f"context packet compilation failed: {exc}") from exc
            finally:
                if callable(set_progress_callback):
                    set_progress_callback(None)
            self._emit_decision_progress("context_ready")
        if not state.universe_summary:
            self._emit_decision_progress("universe_summary")
            state.universe_summary = self.universe_model.summarize(
                state.task,
                world_model_summary=state.world_model_summary,
            )
        tolbert_mode = self._tolbert_mode()
        source_task_id = str(state.task.metadata.get("source_task", "")).strip()
        retrieval_guidance = self._retrieval_guidance(state)
        preferred_task_ids = self._preferred_task_ids(state) if self._tolbert_skill_ranking_active(state) else []
        if source_task_id and source_task_id not in preferred_task_ids:
            preferred_task_ids = [source_task_id, *preferred_task_ids]
        blocked_commands = self._blocked_commands(state)
        recommended_commands = (
            retrieval_guidance.get("recommended_commands", [])
            if self._tolbert_skill_ranking_active(state)
            else []
        )
        retrieval_has_signal = self._has_retrieval_signal(state)
        top_skill = self.skill_library.best_skill_match(
            task_id=state.task.task_id,
            preferred_task_ids=preferred_task_ids if self._tolbert_skill_ranking_enabled() else [],
            recommended_commands=recommended_commands if self._tolbert_skill_ranking_enabled() else [],
            blocked_commands=blocked_commands,
        )
        ranked_skills = self._ranked_skill_summaries(state, top_skill)
        state.available_skills = ranked_skills
        tolbert_route = self._tolbert_route_decision(state)
        tolbert_shadow = self._tolbert_shadow_decision(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            route_mode=tolbert_route.mode,
        )
        if tolbert_route.mode == "primary":
            primary_decision = self._tolbert_primary_decision(
                state,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
            )
            if primary_decision is not None:
                return self._apply_tolbert_shadow(primary_decision, tolbert_shadow, tolbert_route.mode)

        role = self._normalized_role(state)
        deterministic_decision = self._deterministic_role_decision(
            state,
            role=role,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
        )
        if deterministic_decision is not None:
            return self._apply_tolbert_shadow(deterministic_decision, tolbert_shadow, tolbert_route.mode)

        followup_skill_decision = self._followup_skill_decision(
            state,
            role=role,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
        )
        if followup_skill_decision is not None:
            return self._apply_tolbert_shadow(followup_skill_decision, tolbert_shadow, tolbert_route.mode)

        self._emit_decision_progress("payload_build")
        payload = self.context_budgeter.build_payload(
            state=state,
            task_payload=asdict(state.task),
            history_payload=[asdict(step) for step in state.history],
            llm_context_packet=self._llm_context_packet(state),
            retrieval_plan=self._retrieval_plan(state),
            transition_preview=self._transition_preview(state),
            available_skills=state.available_skills,
            prompt_adjustments=self._active_prompt_adjustments(),
            allowed_actions=sorted(ALLOWED_ACTIONS),
            graph_summary=self._compact_graph_summary(state.graph_summary),
            universe_summary=self._compact_universe_summary(state.universe_summary),
            world_model_summary=self._compact_world_model_summary(state.world_model_summary),
            plan=self._compact_plan(state.plan),
            active_subgoal=self._truncate_text(state.active_subgoal),
        )
        system_prompt = self._role_system_prompt(role)
        decision_prompt = self._role_decision_prompt(role)
        self._emit_decision_progress("llm_request")
        raw = self.client.create_decision(
            system_prompt=system_prompt,
            decision_prompt=decision_prompt,
            state_payload=payload,
        )
        self._emit_decision_progress("llm_response")
        normalized = coerce_action_decision(raw)
        action = normalized["action"]
        if action not in ALLOWED_ACTIONS:
            normalized = {
                "thought": f"Unsupported action from model: {action}",
                "action": "respond",
                "content": "Stopping because the model returned an unsupported action.",
                "done": True,
            }
        content = normalized["content"]
        if normalized["action"] == CODE_EXECUTE:
            content = _normalize_command_for_workspace(
                content,
                state.task.workspace_subdir,
            )
        matched_span_id = None
        if normalized["action"] == CODE_EXECUTE:
            matched_span_id = self._matching_retrieval_span_id(
                content,
                retrieval_guidance.get("recommended_command_spans", []),
            )
        return self._apply_tolbert_shadow(ActionDecision(
            thought=normalized["thought"],
            action=normalized["action"],
            content=content,
            done=normalized["done"],
            retrieval_influenced=(
                matched_span_id is not None
                or (retrieval_has_signal and self._tolbert_influence_enabled())
            ),
            selected_retrieval_span_id=matched_span_id,
        ), tolbert_shadow, tolbert_route.mode)

    def _deterministic_role_decision(
        self,
        state: AgentState,
        *,
        role: str,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
    ) -> ActionDecision | None:
        if role == "planner":
            decision = self._planner_direct_decision(
                state,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
                tolbert_mode=tolbert_mode,
                retrieval_has_signal=retrieval_has_signal,
            )
            if decision is not None:
                return decision
        elif role == "critic":
            decision = self._critic_direct_decision(
                state,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
                tolbert_mode=tolbert_mode,
                retrieval_has_signal=retrieval_has_signal,
            )
            if decision is not None:
                return decision
        return self._executor_direct_decision(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
        )

    def _planner_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
    ) -> ActionDecision | None:
        decision = self._tolbert_direct_decision(
            state,
            top_skill=top_skill,
            recommended_commands=retrieval_guidance.get("recommended_command_spans", []),
            blocked_commands=blocked_commands,
        )
        if decision is None:
            return None
        if self._command_control_score(state, decision.content) <= 0 and state.active_subgoal:
            return None
        return decision

    def _critic_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
    ) -> ActionDecision | None:
        del top_skill, retrieval_guidance, tolbert_mode, retrieval_has_signal
        if state.context_packet is not None and self._has_retrieval_signal(state) and state.history and state.consecutive_failures > 0:
            return ActionDecision(
                thought="Pause unsafe repetition and hand control back to synthesis.",
                action="respond",
                content="No safe deterministic command remains.",
                done=True,
            )
        return None

    def _executor_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
    ) -> ActionDecision | None:
        del tolbert_mode, retrieval_has_signal
        return self._tolbert_direct_decision(
            state,
            top_skill=top_skill,
            recommended_commands=retrieval_guidance.get("recommended_command_spans", []),
            blocked_commands=blocked_commands,
        )

    def _followup_skill_decision(
        self,
        state: AgentState,
        *,
        role: str,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        tolbert_mode: str,
        retrieval_has_signal: bool,
    ) -> ActionDecision | None:
        if role == "critic":
            return None
        require_active_skill_ranking = (
            not state.history
            and self._policy_control_float("skill_ranking_confidence_boost") > 0.0
            and not self._tolbert_skill_ranking_active(state)
        )
        if not self._skill_is_safe_for_task(state, top_skill):
            top_skill = None
        if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
            skill_commands = self.skill_library._commands_for_skill(top_skill)
            if skill_commands and self._command_control_score(state, skill_commands[0]) < 0:
                top_skill = None
        if require_active_skill_ranking:
            top_skill = None
        if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
            skill_commands = self.skill_library._commands_for_skill(top_skill)
            if skill_commands and not self._first_step_command_covers_required_artifacts(state, skill_commands[0]):
                top_skill = None
        if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
            matched_span_id = self._matching_retrieval_span_id(
                self.skill_library._commands_for_skill(top_skill)[0],
                retrieval_guidance.get("recommended_command_spans", []),
            )
            return self._skill_action_decision(
                state,
                top_skill,
                thought_prefix="Use Tolbert-ranked skill" if self._tolbert_skill_ranking_enabled() else "Use skill",
                retrieval_influenced=retrieval_has_signal and self._tolbert_influence_enabled(),
                retrieval_ranked_skill=retrieval_has_signal and self._tolbert_skill_ranking_enabled(),
                selected_retrieval_span_id=matched_span_id,
            )
        return None

    def _skill_is_safe_for_task(self, state: AgentState, skill: dict[str, object] | None) -> bool:
        if skill is None:
            return False
        curriculum_kind = str(state.task.metadata.get("curriculum_kind", "")).strip()
        if curriculum_kind != "failure_recovery":
            return True
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
        source_task = str(state.task.metadata.get("source_task", "")).strip()
        if source_task_id and source_task_id in {state.task.task_id, source_task}:
            return True
        skill_family = str(skill.get("benchmark_family", "")).strip()
        task_family = str(state.task.metadata.get("benchmark_family", "")).strip()
        if skill_family and task_family and skill_family != task_family:
            return False
        commands = self.skill_library._commands_for_skill(skill)
        if not commands:
            return False
        first_command = commands[0]
        anchors = self._task_command_anchors(state)
        if not anchors:
            return False
        return any(anchor in first_command for anchor in anchors)

    @staticmethod
    def _task_command_anchors(state: AgentState) -> list[str]:
        anchors: list[str] = []
        for path in [*state.task.expected_files, *state.task.expected_file_contents.keys()]:
            normalized = str(path).strip()
            if not normalized:
                continue
            for part in Path(normalized).parts:
                if part not in {"", "."} and part not in anchors:
                    anchors.append(part)
        return anchors

    def _ranked_skill_summaries(
        self,
        state: AgentState,
        top_skill: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        task_matches = self.skill_library.matching_skills(state.task.task_id)
        if not task_matches:
            source_task_id = str(state.task.metadata.get("source_task", "")).strip()
            task_matches = self.skill_library.matching_skills(
                state.task.task_id,
                source_task_id=source_task_id,
            )
        summaries = [self.skill_library._to_summary(skill, state.task.task_id) for skill in task_matches]
        if top_skill is None:
            return summaries[:3]
        top_skill_id = str(top_skill.get("skill_id", ""))
        summaries.sort(key=lambda skill: (0 if str(skill.get("skill_id", "")) == top_skill_id else 1))
        return summaries[:3]

    def _tolbert_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        recommended_commands: list[dict[str, str]],
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        if state.context_packet is None or not self._tolbert_direct_command_enabled():
            return None
        trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False))
        confidence = float(state.context_packet.control.get("path_confidence", 0.0))
        normalized_blocked = {SkillLibrary._normalize_command(command) for command in blocked_commands}
        ranked_commands = self._rank_direct_retrieval_commands(
            state,
            recommended_commands,
            blocked_commands=normalized_blocked,
        )
        state.retrieval_direct_candidates = ranked_commands
        for entry in ranked_commands:
            command = str(entry.get("command", ""))
            if self._can_execute_direct_retrieval_command(
                state,
                top_skill=top_skill,
                command=command,
                control_score=int(entry.get("control_score", 0)),
                confidence=confidence,
                trust_retrieval=trust_retrieval,
            ):
                return ActionDecision(
                    thought="Use Tolbert retrieval-guided command.",
                    action=CODE_EXECUTE,
                    content=_normalize_command_for_workspace(command, state.task.workspace_subdir),
                    done=False,
                    selected_retrieval_span_id=str(entry.get("span_id", "")).strip() or None,
                    retrieval_influenced=True,
                )
        if (
            top_skill is not None
            and self._tolbert_skill_ranking_enabled()
            and confidence >= self._retrieval_float("tolbert_deterministic_command_confidence")
        ):
            matched_span_id = self._matching_retrieval_span_id(
                self.skill_library._commands_for_skill(top_skill)[0],
                recommended_commands,
            )
            return self._skill_action_decision(
                state,
                top_skill,
                thought_prefix="Use Tolbert-validated skill",
                retrieval_influenced=True,
                retrieval_ranked_skill=True,
                selected_retrieval_span_id=matched_span_id,
            )
        return None

    def _rank_direct_retrieval_commands(
        self,
        state: AgentState,
        recommended_commands: list[dict[str, str]],
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, str]]:
        verifier = state.task
        ranked: list[tuple[int, dict[str, str]]] = []
        for entry in recommended_commands:
            command = str(entry.get("command", "")).strip()
            normalized = SkillLibrary._normalize_command(command)
            if not normalized or normalized in blocked_commands:
                continue
            score = 0
            if any(path and path in command for path in verifier.expected_files):
                score += 4
            if any(path and path in command for path in verifier.expected_file_contents):
                score += 4
            if any(path and path in command for path in verifier.forbidden_files):
                score -= 6
            if verifier.workspace_subdir and f"{verifier.workspace_subdir}/" in command:
                score -= 3
            if "printf " in command:
                score += 1
            control_score = self._command_control_score(state, command)
            score += control_score
            ranked.append(
                (
                    score,
                    {
                        "span_id": str(entry.get("span_id", "")),
                        "command": command,
                        "control_score": control_score,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]["span_id"], item[1]["command"]))
        return [entry for _, entry in ranked]

    def _can_execute_direct_retrieval_command(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        command: str,
        control_score: int,
        confidence: float,
        trust_retrieval: bool,
    ) -> bool:
        if control_score < self._retrieval_float("tolbert_direct_command_min_score"):
            return False
        if state.history:
            return trust_retrieval or confidence >= self._retrieval_float("tolbert_deterministic_command_confidence")
        if not self._first_step_command_covers_required_artifacts(state, command):
            return False
        if not trust_retrieval:
            return False
        if confidence < max(
            self._retrieval_float("tolbert_deterministic_command_confidence"),
            self._retrieval_float("tolbert_first_step_direct_command_confidence"),
            self._retrieval_float("tolbert_first_step_direct_command_confidence")
            + self._policy_control_float("direct_command_confidence_boost"),
        ):
            return False
        if top_skill is not None:
            skill_commands = self.skill_library._commands_for_skill(top_skill)
            if skill_commands:
                skill_score = self._command_control_score(state, skill_commands[0])
                required_artifact_bias = self._policy_control_int("required_artifact_first_step_bias")
                if skill_score > control_score + max(0, required_artifact_bias):
                    return False
        return True

    def _command_control_score(self, state: AgentState, command: str) -> int:
        score = self.universe_model.score_command(state.universe_summary, command)
        score += self.world_model.score_command(state.world_model_summary, command)
        score += self._transition_model_command_score(state, command)
        if self._tolbert_model_surfaces().get("latent_state", False):
            score += latent_command_bias(state.latent_state_summary, command)
        score += state_estimation_policy_bias(
            state.latent_state_summary,
            command,
            self._state_estimation_policy_controls(),
        )
        verifier_alignment_bias = self._policy_control_int("verifier_alignment_bias")
        role = str(state.current_role or "executor")
        active_subgoal = str(state.active_subgoal or "")
        if verifier_alignment_bias:
            if any(path and path in command for path in state.task.expected_files):
                score += verifier_alignment_bias
            if any(path and path in command for path in state.task.expected_file_contents):
                score += verifier_alignment_bias
            if any(path and path in command for path in state.task.forbidden_files):
                score -= verifier_alignment_bias * 2
        if role == "planner":
            planner_subgoal_command_bias = self._policy_control_int("planner_subgoal_command_bias")
            if active_subgoal and any(token in command for token in active_subgoal.split()):
                score += max(2, planner_subgoal_command_bias)
            if "mkdir -p " in command or "cp " in command:
                score += 1
        elif role == "critic":
            critic_repeat_failure_bias = self._policy_control_int("critic_repeat_failure_bias")
            for step in state.history:
                if not step.verification.get("passed", False) and step.content and step.content in command:
                    score -= max(4, critic_repeat_failure_bias * 2)
            if "rm -rf" in command:
                score -= 4
        elif role == "executor":
            if "printf " in command or "> " in command:
                score += 1
        return score

    def _tolbert_primary_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        runtime_policy = self._tolbert_runtime_policy()
        candidates = decode_action_generation_candidates(
            state=state,
            world_model=self.world_model,
            skill_library=self.skill_library,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            proposal_policy=self._tolbert_action_generation_policy(),
            rollout_policy=self._tolbert_rollout_policy(),
            command_score_fn=self._command_control_score,
            normalize_command_fn=_normalize_command_for_workspace,
            canonicalize_command_fn=_canonicalize_command,
        )
        candidates.extend(
            decode_bounded_action_candidates(
                state=state,
                world_model=self.world_model,
                skill_library=self.skill_library,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
                decoder_policy=self._tolbert_decoder_policy(),
                rollout_policy=self._tolbert_rollout_policy(),
                command_score_fn=self._command_control_score,
                match_span_fn=self._matching_retrieval_span_id,
                normalize_command_fn=_normalize_command_for_workspace,
                canonicalize_command_fn=_canonicalize_command,
            )
        )
        candidates = sorted(candidates, key=lambda item: (-item.score, item.action, item.content))
        if not candidates:
            return None
        hybrid_runtime = self._tolbert_hybrid_runtime()
        best = candidates[0]
        if bool(hybrid_runtime.get("primary_enabled", False)):
            candidate_lookup = {
                f"{candidate.action}:{candidate.content}": candidate
                for candidate in candidates
            }
            scored = self._hybrid_scored_candidates(
                state,
                [
                    {
                        "action": candidate.action,
                        "command": candidate.content,
                        "content": candidate.content,
                        "score": float(candidate.score),
                        "reason": candidate.reason,
                        "selected_skill_id": candidate.selected_skill_id,
                        "span_id": candidate.selected_retrieval_span_id,
                        "retrieval_influenced": candidate.retrieval_influenced,
                        "retrieval_ranked_skill": candidate.retrieval_ranked_skill,
                        "proposal_source": candidate.proposal_source,
                        "proposal_novel": candidate.proposal_novel,
                    }
                    for candidate in candidates
                ],
            )
            if scored:
                selected_key = (
                    f"{str(scored[0].get('action', 'code_execute')).strip()}:"
                    f"{str(scored[0].get('content', scored[0].get('command', ''))).strip()}"
                )
                best = candidate_lookup.get(selected_key, best)
                best.proposal_metadata = {
                    **dict(best.proposal_metadata or {}),
                    **self._hybrid_candidate_metadata(scored[0]),
                }
        if best.action == CODE_EXECUTE and float(best.score) < float(runtime_policy.get("primary_min_command_score", 2)):
            return None
        return ActionDecision(
            thought=best.thought,
            action=best.action,
            content=best.content,
            done=best.action != CODE_EXECUTE,
            selected_skill_id=best.selected_skill_id,
            selected_retrieval_span_id=best.selected_retrieval_span_id,
            retrieval_influenced=best.retrieval_influenced,
            retrieval_ranked_skill=best.retrieval_ranked_skill,
            proposal_source=best.proposal_source,
            proposal_novel=best.proposal_novel,
            proposal_metadata=dict(best.proposal_metadata or {}),
            decision_source=(
                "tolbert_hybrid_proposal_decoder"
                if best.proposal_source and bool(hybrid_runtime.get("primary_enabled", False))
                else "tolbert_proposal_decoder"
                if best.proposal_source
                else "tolbert_hybrid_decoder"
                if bool(hybrid_runtime.get("primary_enabled", False))
                else "tolbert_decoder"
            ),
            tolbert_route_mode="primary",
        )

    def _tolbert_shadow_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        route_mode: str,
    ) -> dict[str, object]:
        if route_mode not in {"shadow", "primary"}:
            return {}
        candidates = self._tolbert_ranked_candidates(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
        )
        if not candidates:
            return {}
        hybrid_runtime = self._tolbert_hybrid_runtime()
        if bool(hybrid_runtime.get("shadow_enabled", False)):
            try:
                scored = self._hybrid_scored_candidates(state, candidates)
            except RuntimeError as exc:
                return {
                    "mode": route_mode,
                    "command": "",
                    "score": 0.0,
                    "reason": f"hybrid_runtime_error:{str(exc)}",
                    "span_id": "",
                    "model_family": str(hybrid_runtime.get("model_family", "")).strip(),
                }
            if scored:
                best = scored[0]
                return {
                    "mode": route_mode,
                    "command": str(best.get("command", "")),
                    "score": float(best.get("hybrid_total_score", best.get("score", 0.0))),
                    "reason": f"hybrid_runtime:{str(best.get('reason', '')).strip() or 'candidate'}",
                    "span_id": str(best.get("span_id", "")).strip(),
                    "model_family": str(best.get("hybrid_model_family", "")).strip(),
                    "hybrid_metadata": self._hybrid_candidate_metadata(best),
                }
        best = candidates[0]
        return {
            "mode": route_mode,
            "command": str(best["command"]),
            "score": int(best["score"]),
            "reason": str(best["reason"]),
            "span_id": str(best.get("span_id", "")).strip(),
        }

    @staticmethod
    def _hybrid_candidate_metadata(candidate: dict[str, object]) -> dict[str, object]:
        return {
            "hybrid_total_score": float(candidate.get("hybrid_total_score", candidate.get("score", 0.0)) or 0.0),
            "hybrid_learned_score": float(candidate.get("hybrid_learned_score", 0.0) or 0.0),
            "hybrid_policy_score": float(candidate.get("hybrid_policy_score", 0.0) or 0.0),
            "hybrid_value_score": float(candidate.get("hybrid_value_score", 0.0) or 0.0),
            "hybrid_risk_score": float(candidate.get("hybrid_risk_score", 0.0) or 0.0),
            "hybrid_stop_score": float(candidate.get("hybrid_stop_score", 0.0) or 0.0),
            "hybrid_transition_progress": float(candidate.get("hybrid_transition_progress", 0.0) or 0.0),
            "hybrid_transition_regression": float(candidate.get("hybrid_transition_regression", 0.0) or 0.0),
            "hybrid_world_progress_score": float(candidate.get("hybrid_world_progress_score", 0.0) or 0.0),
            "hybrid_world_risk_score": float(candidate.get("hybrid_world_risk_score", 0.0) or 0.0),
            "hybrid_decoder_world_progress_score": float(
                candidate.get("hybrid_decoder_world_progress_score", 0.0) or 0.0
            ),
            "hybrid_decoder_world_risk_score": float(
                candidate.get("hybrid_decoder_world_risk_score", 0.0) or 0.0
            ),
            "hybrid_decoder_world_entropy_mean": float(
                candidate.get("hybrid_decoder_world_entropy_mean", 0.0) or 0.0
            ),
            "hybrid_world_prior_backend": str(candidate.get("hybrid_world_prior_backend", "")).strip(),
            "hybrid_world_prior_top_state": int(candidate.get("hybrid_world_prior_top_state", -1) or -1),
            "hybrid_world_prior_top_probability": float(
                candidate.get("hybrid_world_prior_top_probability", 0.0) or 0.0
            ),
            "hybrid_world_transition_family": str(
                candidate.get("hybrid_world_transition_family", "")
            ).strip(),
            "hybrid_world_transition_bandwidth": int(
                candidate.get("hybrid_world_transition_bandwidth", 0) or 0
            ),
            "hybrid_world_transition_gate": float(candidate.get("hybrid_world_transition_gate", 0.0) or 0.0),
            "hybrid_world_final_entropy_mean": float(
                candidate.get("hybrid_world_final_entropy_mean", 0.0) or 0.0
            ),
            "hybrid_ssm_last_state_norm_mean": float(
                candidate.get("hybrid_ssm_last_state_norm_mean", 0.0) or 0.0
            ),
            "hybrid_ssm_pooled_state_norm_mean": float(
                candidate.get("hybrid_ssm_pooled_state_norm_mean", 0.0) or 0.0
            ),
            "hybrid_model_family": str(candidate.get("hybrid_model_family", "")).strip(),
            "hybrid_reason": str(candidate.get("reason", "")).strip(),
        }

    def _tolbert_ranked_candidates(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
    ) -> list[dict[str, object]]:
        runtime_policy = self._tolbert_runtime_policy()
        allow_direct = bool(runtime_policy.get("allow_direct_command_primary", True))
        allow_skill = bool(runtime_policy.get("allow_skill_primary", True))
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        candidates: list[dict[str, object]] = []
        seen: set[str] = set()
        if allow_direct:
            for item in retrieval_guidance.get("recommended_command_spans", []):
                command = _normalize_command_for_workspace(str(item.get("command", "")), state.task.workspace_subdir)
                normalized = _canonicalize_command(command)
                if not normalized or normalized in seen or normalized in blocked:
                    continue
                seen.add(normalized)
                candidates.append(
                    {
                        "command": command,
                        "reason": "retrieval guidance",
                        "span_id": str(item.get("span_id", "")).strip(),
                        "retrieval_influenced": True,
                        "retrieval_ranked_skill": False,
                    }
                )
        if allow_skill and top_skill is not None:
            for command in self.skill_library._commands_for_skill(top_skill)[:1]:
                command = _normalize_command_for_workspace(command, state.task.workspace_subdir)
                normalized = _canonicalize_command(command)
                if not normalized or normalized in seen or normalized in blocked:
                    continue
                seen.add(normalized)
                candidates.append(
                    {
                        "command": command,
                        "reason": "retained skill",
                        "span_id": self._matching_retrieval_span_id(
                            command,
                            retrieval_guidance.get("recommended_command_spans", []),
                        ),
                        "retrieval_influenced": False,
                        "retrieval_ranked_skill": True,
                    }
                )
        for command in state.task.suggested_commands[:3]:
            command = _normalize_command_for_workspace(command, state.task.workspace_subdir)
            normalized = _canonicalize_command(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "command": command,
                    "reason": "task suggestion",
                    "span_id": self._matching_retrieval_span_id(
                        command,
                        retrieval_guidance.get("recommended_command_spans", []),
                    ),
                    "retrieval_influenced": False,
                    "retrieval_ranked_skill": False,
                }
            )
        for candidate in candidates:
            candidate["score"] = self._command_control_score(state, str(candidate["command"]))
        return sorted(candidates, key=lambda item: (-int(item["score"]), str(item["command"])))

    def _tolbert_route_decision(self, state: AgentState):
        path_confidence = self._path_confidence(state)
        trust_retrieval = self._trust_retrieval(state)
        benchmark_family = str(state.task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        route = choose_tolbert_route(
            runtime_policy=self._tolbert_runtime_policy(),
            benchmark_family=benchmark_family,
            path_confidence=path_confidence,
            trust_retrieval=trust_retrieval,
        )
        if route.mode == "disabled" and not bool(
            self._tolbert_runtime_policy().get("fallback_to_vllm_on_low_confidence", True)
        ):
            return route
        return route

    @staticmethod
    def _path_confidence(state: AgentState) -> float:
        if state.context_packet is None:
            return 0.0
        try:
            return float(state.context_packet.control.get("path_confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _trust_retrieval(state: AgentState) -> bool:
        if state.context_packet is None:
            return False
        return bool(state.context_packet.control.get("trust_retrieval", False))

    def _apply_tolbert_shadow(
        self,
        decision: ActionDecision,
        shadow_decision: dict[str, object],
        route_mode: str,
    ) -> ActionDecision:
        decision.shadow_decision = dict(shadow_decision)
        if not decision.tolbert_route_mode:
            decision.tolbert_route_mode = route_mode if route_mode in {"shadow", "primary"} else ""
        if not decision.decision_source:
            decision.decision_source = "llm"
        return decision

    def _transition_model_command_score(self, state: AgentState, command: str) -> int:
        if not self.config.use_transition_model_proposals:
            return 0
        normalized = _canonicalize_command(command)
        if not normalized:
            return 0
        normalized_pattern = transition_model_command_pattern(normalized) or normalized
        controls = self._transition_model_controls()
        if not controls:
            return 0
        score = 0
        latest_transition = state.latest_state_transition if isinstance(state.latest_state_transition, dict) else {}
        regressed_paths = {
            str(path).strip()
            for path in latest_transition.get("regressions", [])
            if str(path).strip()
        }
        last_command = ""
        if state.history:
            last_command = _canonicalize_command(str(state.history[-1].content))
        last_pattern = transition_model_command_pattern(last_command) if last_command else ""
        cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
        for signature in self._transition_model_signatures():
            signature_command = _canonicalize_command(str(signature.get("command", "")))
            if not signature_command:
                continue
            signature_pattern = (
                transition_model_command_pattern(str(signature.get("command_pattern", "")))
                or transition_model_command_pattern(signature_command)
                or signature_command
            )
            support = max(1, int(signature.get("support", 1)))
            signal = str(signature.get("signal", "")).strip()
            signature_regressions = {
                str(path).strip()
                for path in signature.get("regressions", [])
                if str(path).strip()
            }
            if normalized == signature_command:
                penalty = self._transition_model_control_int("repeat_command_penalty", 4) + min(3, support - 1)
                if signal == "state_regression":
                    penalty += self._transition_model_control_int("regressed_path_command_penalty", 3)
                score -= penalty
                continue
            if normalized_pattern == signature_pattern:
                penalty = max(1, self._transition_model_control_int("repeat_command_penalty", 4) - 2) + min(
                    2, support - 1
                )
                if signal == "state_regression":
                    penalty += max(1, self._transition_model_control_int("regressed_path_command_penalty", 3) - 1)
                score -= penalty
                continue
            if signature_regressions and any(path in normalized for path in signature_regressions):
                score -= self._transition_model_control_int("regressed_path_command_penalty", 3)
            if regressed_paths and signature_regressions and regressed_paths.intersection(signature_regressions):
                if any(path in normalized for path in regressed_paths):
                    score += self._transition_model_control_int("recovery_command_bonus", 2)
                    if cleanup_command:
                        score += 1
        if latest_transition.get("no_progress", False):
            if last_command and normalized == last_command:
                score -= self._transition_model_control_int("repeat_command_penalty", 4)
            elif last_pattern and normalized_pattern == last_pattern:
                score -= max(1, self._transition_model_control_int("repeat_command_penalty", 4) // 2)
            elif normalized != last_command:
                score += self._transition_model_control_int("progress_command_bonus", 2)
        return score

    def _first_step_command_covers_required_artifacts(self, state: AgentState, command: str) -> bool:
        if state.history:
            return True
        required_paths = self._required_first_step_artifacts(state)
        if not required_paths:
            return True
        normalized = SkillLibrary._normalize_command(command)
        if all(path in normalized for path in required_paths):
            return True
        required_artifact_bias = self._policy_control_int("required_artifact_first_step_bias")
        if required_artifact_bias <= 0:
            return False
        matched_paths = sum(1 for path in required_paths if path in normalized)
        return matched_paths >= max(1, len(required_paths) - required_artifact_bias)

    @staticmethod
    def _required_first_step_artifacts(state: AgentState) -> list[str]:
        setup_commands = [str(command) for command in state.task.setup_commands]
        required: list[str] = []
        for path in state.task.expected_files:
            normalized = str(path).strip()
            if not normalized:
                continue
            if any(normalized in command for command in setup_commands):
                continue
            required.append(normalized)
        return required

    def _transition_preview(self, state: AgentState) -> dict[str, object]:
        candidates: list[str] = []
        retrieval_guidance = self._retrieval_guidance(state)
        for item in retrieval_guidance.get("recommended_command_spans", [])[:2]:
            command = str(item.get("command", "")).strip()
            if command and command not in candidates:
                candidates.append(command)
        for command in state.task.suggested_commands[:2]:
            command = str(command).strip()
            if command and command not in candidates:
                candidates.append(command)
        previews = []
        for command in candidates[:3]:
            effect = self.world_model.simulate_command_effect(state.world_model_summary, command)
            governance = self.universe_model.simulate_command_governance(state.universe_summary, command)
            previews.append({"command": command, **effect, "governance": governance})
        return {
            "candidates": previews,
            "universe_id": state.universe_summary.get("universe_id", ""),
            "completion_ratio": state.world_model_summary.get("completion_ratio", 0.0),
            "missing_expected_artifacts": list(state.world_model_summary.get("missing_expected_artifacts", []))[:4],
            "present_forbidden_artifacts": list(state.world_model_summary.get("present_forbidden_artifacts", []))[:4],
            "changed_preserved_artifacts": list(state.world_model_summary.get("changed_preserved_artifacts", []))[:4],
        }

    def _tolbert_mode(self) -> str:
        mode = str(self.config.tolbert_mode or "full").strip().lower()
        if not self.config.use_tolbert_context:
            return "disabled"
        if mode not in {"full", "path_only", "retrieval_only", "deterministic_command", "skill_ranking", "disabled"}:
            return "full"
        return mode

    def _tolbert_influence_enabled(self) -> bool:
        return self._tolbert_mode() != "disabled"

    def _tolbert_direct_command_enabled(self) -> bool:
        return self._tolbert_mode() in {"full", "deterministic_command"}

    def _tolbert_skill_ranking_enabled(self) -> bool:
        return self._tolbert_mode() in {"full", "skill_ranking"}

    def _tolbert_skill_ranking_active(self, state: AgentState) -> bool:
        if not self._tolbert_skill_ranking_enabled() or state.context_packet is None:
            return False
        control = state.context_packet.control
        if bool(control.get("trust_retrieval", False)):
            return True
        confidence = float(control.get("path_confidence", 0.0))
        return confidence >= (
            self._retrieval_float("tolbert_skill_ranking_min_confidence")
            + self._policy_control_float("skill_ranking_confidence_boost")
        )

    def _retrieval_float(self, field: str) -> float:
        value = self._retrieval_overrides().get(field, getattr(self.config, field))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(getattr(self.config, field))

    def _retrieval_overrides(self) -> dict[str, object]:
        if not self.config.use_retrieval_proposals:
            return {}
        path = self.config.retrieval_proposals_path
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return retained_retrieval_overrides(payload)

    def _policy_controls(self) -> dict[str, object]:
        if self._policy_controls_cache is not None:
            return self._policy_controls_cache
        self._policy_controls_cache = retained_policy_controls(self._prompt_policy_payload())
        return self._policy_controls_cache

    def _tolbert_model_payload(self) -> dict[str, object]:
        if self._tolbert_model_payload_cache is not None:
            return self._tolbert_model_payload_cache
        if not self.config.use_tolbert_model_artifacts:
            self._tolbert_model_payload_cache = {}
            return self._tolbert_model_payload_cache
        self._tolbert_model_payload_cache = load_model_artifact(self.config.tolbert_model_artifact_path)
        return self._tolbert_model_payload_cache

    def _tolbert_runtime_policy(self) -> dict[str, object]:
        if self._tolbert_runtime_policy_cache is not None:
            return self._tolbert_runtime_policy_cache
        normalized = retained_tolbert_runtime_policy(self._tolbert_model_payload())
        overrides = retained_tolbert_runtime_policy_overrides(self._prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        self._tolbert_runtime_policy_cache = normalized
        return self._tolbert_runtime_policy_cache

    def _tolbert_model_surfaces(self) -> dict[str, object]:
        if self._tolbert_model_surfaces_cache is not None:
            return self._tolbert_model_surfaces_cache
        self._tolbert_model_surfaces_cache = retained_tolbert_model_surfaces(self._tolbert_model_payload())
        return self._tolbert_model_surfaces_cache

    def _tolbert_decoder_policy(self) -> dict[str, object]:
        if self._tolbert_decoder_policy_cache is not None:
            return self._tolbert_decoder_policy_cache
        normalized = retained_tolbert_decoder_policy(self._tolbert_model_payload())
        overrides = retained_tolbert_decoder_policy_overrides(self._prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        self._tolbert_decoder_policy_cache = normalized
        return self._tolbert_decoder_policy_cache

    def _tolbert_action_generation_policy(self) -> dict[str, object]:
        if self._tolbert_action_generation_policy_cache is not None:
            return self._tolbert_action_generation_policy_cache
        self._tolbert_action_generation_policy_cache = retained_tolbert_action_generation_policy(
            self._tolbert_model_payload()
        )
        return self._tolbert_action_generation_policy_cache

    def _tolbert_rollout_policy(self) -> dict[str, object]:
        if self._tolbert_rollout_policy_cache is not None:
            return self._tolbert_rollout_policy_cache
        normalized = retained_tolbert_rollout_policy(self._tolbert_model_payload())
        overrides = retained_tolbert_rollout_policy_overrides(self._prompt_policy_payload())
        if overrides:
            normalized.update(overrides)
        self._tolbert_rollout_policy_cache = normalized
        return self._tolbert_rollout_policy_cache

    def _tolbert_hybrid_runtime(self) -> dict[str, object]:
        if self._tolbert_hybrid_runtime_cache is not None:
            return self._tolbert_hybrid_runtime_cache
        normalized = retained_tolbert_hybrid_runtime(self._tolbert_model_payload())
        scoring_overrides = retained_tolbert_hybrid_scoring_policy_overrides(self._prompt_policy_payload())
        if scoring_overrides:
            scoring_policy = normalized.get("scoring_policy", {})
            merged = dict(scoring_policy) if isinstance(scoring_policy, dict) else {}
            merged.update(scoring_overrides)
            normalized["scoring_policy"] = merged
        self._tolbert_hybrid_runtime_cache = normalized
        return self._tolbert_hybrid_runtime_cache

    def _hybrid_scored_candidates(
        self,
        state: AgentState,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        runtime = self._tolbert_hybrid_runtime()
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        if not manifest_raw:
            self._last_hybrid_runtime_error = "missing bundle_manifest_path"
            raise RuntimeError("hybrid runtime is enabled but bundle_manifest_path is missing")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = Path(__file__).resolve().parents[1] / manifest_path
        if not manifest_path.exists():
            self._last_hybrid_runtime_error = f"bundle manifest does not exist: {manifest_path}"
            raise RuntimeError(f"hybrid runtime bundle manifest does not exist: {manifest_path}")
        self._last_hybrid_runtime_error = ""
        try:
            return score_hybrid_candidates(
                state=state,
                candidates=candidates,
                bundle_manifest_path=manifest_path,
                device=str(runtime.get("preferred_device", "cpu")).strip() or "cpu",
                scoring_policy=runtime.get("scoring_policy", {}),
            )
        except Exception as exc:
            self._last_hybrid_runtime_error = str(exc).strip() or exc.__class__.__name__
            raise RuntimeError(
                f"hybrid runtime scoring failed for {manifest_path}: {self._last_hybrid_runtime_error}"
            ) from exc

    def _role_directive_overrides(self) -> dict[str, str]:
        if self._role_directives_cache is not None:
            return self._role_directives_cache
        self._role_directives_cache = retained_role_directives(self._prompt_policy_payload())
        return self._role_directives_cache

    def _prompt_policy_payload(self) -> dict[str, object]:
        if self._prompt_policy_payload_cache is not None:
            return self._prompt_policy_payload_cache
        if not self.config.use_prompt_proposals:
            self._prompt_policy_payload_cache = {}
            return self._prompt_policy_payload_cache
        path = self.config.prompt_proposals_path
        if not path.exists():
            self._prompt_policy_payload_cache = {}
            return self._prompt_policy_payload_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._prompt_policy_payload_cache = {}
            return self._prompt_policy_payload_cache
        self._prompt_policy_payload_cache = payload if isinstance(payload, dict) else {}
        return self._prompt_policy_payload_cache

    def _transition_model_controls(self) -> dict[str, object]:
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

    def _state_estimation_policy_controls(self) -> dict[str, object]:
        if self._state_estimation_policy_controls_cache is not None:
            return dict(self._state_estimation_policy_controls_cache)
        payload = retained_state_estimation_payload(self.config)
        self._state_estimation_policy_controls_cache = retained_state_estimation_policy_controls(payload)
        return dict(self._state_estimation_policy_controls_cache)

    def _transition_model_signatures(self) -> list[dict[str, object]]:
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

    def _policy_control_float(self, field: str) -> float:
        value = self._policy_controls().get(field, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _policy_control_int(self, field: str) -> int:
        value = self._policy_controls().get(field, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _transition_model_control_int(self, field: str, default: int) -> int:
        value = self._transition_model_controls().get(field, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _skill_action_decision(
        self,
        state: AgentState,
        skill: dict[str, object],
        *,
        thought_prefix: str,
        retrieval_influenced: bool = False,
        retrieval_ranked_skill: bool = False,
        selected_retrieval_span_id: str | None = None,
    ) -> ActionDecision:
        skill_id = str(skill.get("skill_id", f"skill:{state.task.task_id}:primary"))
        commands = self.skill_library._commands_for_skill(skill)
        normalized_command = _normalize_command_for_workspace(commands[0], state.task.workspace_subdir)
        return ActionDecision(
            thought=f"{thought_prefix} {skill_id}.",
            action=CODE_EXECUTE,
            content=normalized_command,
            done=False,
            selected_skill_id=skill_id,
            selected_retrieval_span_id=selected_retrieval_span_id,
            retrieval_influenced=retrieval_influenced,
            retrieval_ranked_skill=retrieval_ranked_skill,
        )

    @staticmethod
    def _matching_retrieval_span_id(
        command: str,
        recommended_command_spans: list[dict[str, str]],
    ) -> str | None:
        normalized = SkillLibrary._normalize_command(command)
        for entry in recommended_command_spans:
            candidate = SkillLibrary._normalize_command(str(entry.get("command", "")))
            if candidate and candidate == normalized:
                span_id = str(entry.get("span_id", "")).strip()
                return span_id or None
        return None

    @staticmethod
    def _retrieval_guidance(state: AgentState) -> dict[str, list[str]]:
        if state.context_packet is None:
            return {
                "recommended_commands": [],
                "recommended_command_spans": [],
                "avoidance_notes": [],
                "evidence": [],
            }
        return state.context_packet.control.get("retrieval_guidance", {})

    @staticmethod
    def _preferred_task_ids(state: AgentState) -> list[str]:
        if state.context_packet is None:
            return []
        task_ids: list[str] = []
        for item in (
            state.context_packet.retrieval.get("branch_scoped", [])
            + state.context_packet.retrieval.get("fallback_scoped", [])
            + state.context_packet.retrieval.get("global", [])
        ):
            metadata = item.get("metadata") or {}
            task_id = str(metadata.get("task_id", "")).strip()
            if task_id and task_id not in task_ids:
                task_ids.append(task_id)
        return task_ids

    @staticmethod
    def _blocked_commands(state: AgentState) -> list[str]:
        blocked: list[str] = []
        if state.context_packet is not None:
            for note in state.context_packet.control.get("retrieval_guidance", {}).get("avoidance_notes", []):
                match = re.search(r"avoid repeating '([^']+)'", str(note))
                if match:
                    blocked.append(match.group(1))
        for step in state.history:
            if not step.verification.get("passed", False) and step.content:
                blocked.append(step.content)
        return blocked

    @staticmethod
    def _has_retrieval_signal(state: AgentState) -> bool:
        if state.context_packet is None:
            return False
        guidance = state.context_packet.control.get("retrieval_guidance", {})
        if guidance.get("recommended_command_spans") or guidance.get("recommended_commands"):
            return True
        if guidance.get("avoidance_notes") or guidance.get("evidence"):
            return True
        for bucket in ("branch_scoped", "fallback_scoped", "global"):
            if state.context_packet.retrieval.get(bucket):
                return True
        return False

    @staticmethod
    def _retrieval_plan(state: AgentState) -> dict[str, object]:
        if state.context_packet is None:
            return {
                "strategy": "synthesize",
                "trust_retrieval": False,
                "path_confidence": 0.0,
                "level_focus": "",
            }
        control = state.context_packet.control
        trust_retrieval = bool(control.get("trust_retrieval", False))
        path_confidence = float(control.get("path_confidence", 0.0))
        if trust_retrieval:
            strategy = "minimal_adaptation"
        elif path_confidence > 0.0:
            strategy = "guided_synthesis"
        else:
            strategy = "synthesize"
        return {
            "strategy": strategy,
            "trust_retrieval": trust_retrieval,
            "path_confidence": path_confidence,
            "level_focus": str(control.get("level_focus", "")),
            "acting_role": str(state.current_role or "executor"),
        }

    @staticmethod
    def _llm_context_packet(state: AgentState) -> dict[str, object] | None:
        if state.context_packet is None:
            return None
        control = state.context_packet.control
        retrieval = state.context_packet.retrieval
        max_chunks = int(control.get("context_chunk_budget", {}).get("max_chunks", 8) or 8)
        compact_control = {
            "mode": control.get("mode"),
            "level_focus": control.get("level_focus"),
            "path_confidence": control.get("path_confidence"),
            "trust_retrieval": control.get("trust_retrieval"),
            "selected_branch_level": control.get("selected_branch_level"),
            "branch_candidates": control.get("branch_candidates", [])[:3],
            "context_chunk_budget": control.get("context_chunk_budget", {}),
            "selected_context_chunks": control.get("selected_context_chunks", [])[:max_chunks],
            "retrieval_guidance": control.get("retrieval_guidance", {}),
        }
        compact_retrieval = {
            bucket: [
                {
                    "span_id": str(item.get("span_id", "")),
                    "span_type": str(item.get("span_type", "")),
                    "score": float(item.get("score", 0.0)),
                    "metadata": dict(item.get("metadata", {})),
                }
                for item in retrieval.get(bucket, [])[:3]
            ]
            for bucket in ("branch_scoped", "fallback_scoped", "global")
        }
        return {
            "task": dict(state.context_packet.task),
            "control": compact_control,
            "tolbert": dict(state.context_packet.tolbert),
            "retrieval": compact_retrieval,
            "verifier_contract": dict(state.context_packet.verifier_contract),
        }

    def _compact_plan(self, plan: list[str]) -> list[str]:
        max_items = max(1, self.config.llm_plan_max_items)
        return [self._truncate_text(item) for item in plan[:max_items] if str(item).strip()]

    def _compact_graph_summary(self, graph_summary: dict[str, object]) -> dict[str, object]:
        if not graph_summary:
            return {}
        summary: dict[str, object] = {}
        if "document_count" in graph_summary:
            summary["document_count"] = graph_summary["document_count"]
        if "benchmark_families" in graph_summary:
            families = graph_summary.get("benchmark_families", {})
            if isinstance(families, dict):
                summary["benchmark_families"] = dict(list(families.items())[:6])
        if "failure_types" in graph_summary:
            failures = graph_summary.get("failure_types", {})
            if isinstance(failures, dict):
                summary["failure_types"] = dict(list(failures.items())[:6])
        elif "failure_type_counts" in graph_summary:
            failures = graph_summary.get("failure_type_counts", {})
            if isinstance(failures, dict):
                summary["failure_types"] = dict(list(failures.items())[:6])
        if "related_tasks" in graph_summary:
            related = graph_summary.get("related_tasks", [])
            if isinstance(related, list):
                summary["related_tasks"] = [self._truncate_text(str(item), limit=80) for item in related[:6]]
        elif "neighbors" in graph_summary:
            related = graph_summary.get("neighbors", [])
            if isinstance(related, list):
                summary["related_tasks"] = [self._truncate_text(str(item), limit=80) for item in related[:6]]
        return summary

    def _compact_world_model_summary(self, world_model_summary: dict[str, object]) -> dict[str, object]:
        if not world_model_summary:
            return {}
        summary: dict[str, object] = {}
        for key in ("benchmark_family", "horizon", "completion_ratio"):
            if key in world_model_summary:
                summary[key] = world_model_summary[key]
        for key in ("expected_artifacts", "forbidden_artifacts", "preserved_artifacts"):
            values = world_model_summary.get(key, [])
            if isinstance(values, list):
                summary[key] = [self._truncate_text(str(item), limit=80) for item in values[:6]]
        for key in (
            "missing_expected_artifacts",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "intact_preserved_artifacts",
            "updated_workflow_paths",
            "updated_generated_paths",
            "updated_report_paths",
        ):
            values = world_model_summary.get(key, [])
            if isinstance(values, list) and values:
                summary[key] = [self._truncate_text(str(item), limit=80) for item in values[:6]]
        return summary

    def _compact_universe_summary(self, universe_summary: dict[str, object]) -> dict[str, object]:
        if not universe_summary:
            return {}
        summary: dict[str, object] = {}
        for key in ("universe_id", "stability", "governance_mode"):
            if key in universe_summary:
                summary[key] = universe_summary[key]
        for key in ("requires_verification", "requires_bounded_steps", "prefer_reversible_actions"):
            if key in universe_summary:
                summary[key] = bool(universe_summary[key])
        for key in ("invariants", "forbidden_command_patterns", "preferred_command_prefixes"):
            values = universe_summary.get(key, [])
            if isinstance(values, list) and values:
                summary[key] = [self._truncate_text(str(item), limit=80) for item in values[:6]]
        action_risk_controls = universe_summary.get("action_risk_controls", {})
        if isinstance(action_risk_controls, dict) and action_risk_controls:
            summary["action_risk_controls"] = {
                str(key): int(value)
                for key, value in action_risk_controls.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        for key in (
            "environment_assumptions",
            "environment_alignment",
            "envelope_alignment",
            "constitutional_compliance",
            "runtime_attestation",
            "plan_risk_summary",
        ):
            value = universe_summary.get(key, {})
            if isinstance(value, dict) and value:
                summary[key] = {
                    str(item_key): item_value
                    for item_key, item_value in list(value.items())[:8]
                }
        autonomy_scope = universe_summary.get("autonomy_scope", {})
        if isinstance(autonomy_scope, dict) and autonomy_scope:
            summary["autonomy_scope"] = {
                str(key): self._truncate_text(str(value), limit=80)
                for key, value in list(autonomy_scope.items())[:8]
            }
        return summary

    def _truncate_text(self, text: object, *, limit: int | None = None) -> str:
        value = " ".join(str(text).split())
        max_chars = max(32, limit or self.config.llm_summary_max_chars)
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."

    def _role_system_prompt(self, role: str) -> str:
        directive = self._role_directive(role)
        return f"{self.system_prompt}\n{directive}"

    def _role_decision_prompt(self, role: str) -> str:
        directive = self._role_directive(role)
        return f"{directive}\n{self.decision_prompt}"

    def _active_prompt_adjustments(self) -> list[dict[str, object]]:
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
        return [proposal for proposal in proposals[:3] if isinstance(proposal, dict)]

    def _role_directive(self, role: str) -> str:
        normalized = str(role or "executor").strip().lower()
        if normalized == "planner":
            base = "Active role: planner. Prefer clarifying the next verifier-relevant subgoal and commands that establish expected artifacts."
        elif normalized == "critic":
            base = "Active role: critic. Prefer avoiding repeated failures, forbidden artifacts, and brittle commands before suggesting execution."
        else:
            base = "Active role: executor. Prefer the most direct verifier-relevant command that completes the current subgoal."
        override = self._role_directive_overrides().get(normalized, "")
        if not override:
            return base
        return f"{base} {override}"

    @staticmethod
    def _normalized_role(state: AgentState) -> str:
        return str(state.current_role or "executor").strip().lower() or "executor"


def _normalize_command_for_workspace(command: str, workspace_subdir: str) -> str:
    normalized = command.strip()
    workspace_name = workspace_subdir.strip().strip("/")
    if not workspace_name:
        return normalized

    mkdir_prefix = f"mkdir -p {workspace_name} && "
    if normalized.startswith(mkdir_prefix):
        normalized = normalized[len(mkdir_prefix):].strip()

    normalized = re.sub(rf"(?<![\w./-]){re.escape(workspace_name)}/", "", normalized)
    return normalized


def _canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    # Retrieval templates can encode newlines literally inside quoted strings,
    # while model outputs often use the shell-escaped `\\n` form. Normalize both
    # spellings so retrieval grounding and metrics attach to the same command.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())
