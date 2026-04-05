from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shlex
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
from .modeling.policy.decoder import (
    _render_structured_edit_command,
    decode_action_generation_candidates,
    decode_bounded_action_candidates,
)
from .modeling.policy.runtime import choose_tolbert_route
from .modeling.tolbert.runtime import score_hybrid_candidates
from .modeling.world.latent_state import latent_command_bias
from .policy_improvement import (
    dedupe_prompt_adjustments,
    retained_policy_controls,
    retained_role_directives,
    retained_tolbert_decoder_policy_overrides,
    retained_tolbert_hybrid_scoring_policy_overrides,
    retained_tolbert_rollout_policy_overrides,
    retained_tolbert_runtime_policy_overrides,
)
from .retrieval_improvement import retained_retrieval_overrides
from .schemas import ActionDecision
from .state import AgentState, _software_work_objective_phase, _software_work_phase_rank
from .state_estimation_improvement import (
    retained_state_estimation_payload,
    retained_state_estimation_policy_controls,
    state_estimation_policy_bias,
)
from .syntax_motor import summarize_python_edit_step
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

    def fallback_decision(self, state: AgentState, *, failure_origin: str = "") -> ActionDecision | None:
        del state, failure_origin
        return None


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

    def fallback_decision(self, state: AgentState, *, failure_origin: str = "") -> ActionDecision | None:
        if str(failure_origin).strip() != "inference_failure":
            return None
        source_task_id = str(state.task.metadata.get("source_task", "")).strip()
        preferred_task_ids = [source_task_id] if source_task_id else []
        retrieval_guidance = self._retrieval_guidance(state)
        blocked_commands = self._blocked_commands(state)
        top_skill = self.skill_library.best_skill_match(
            task_id=state.task.task_id,
            preferred_task_ids=preferred_task_ids,
            recommended_commands=list(retrieval_guidance.get("recommended_commands", [])),
            blocked_commands=blocked_commands,
        )
        state.available_skills = self._ranked_skill_summaries(state, top_skill)
        fallback = self._best_deterministic_fallback_decision(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
        )
        if fallback is None:
            return None
        tolbert_route = self._tolbert_route_decision(state)
        tolbert_shadow = self._tolbert_shadow_decision(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            route_mode=tolbert_route.mode,
        )
        return self._apply_tolbert_shadow(fallback, tolbert_shadow, tolbert_route.mode)

    def _emit_decision_progress(self, stage: str, **payload: object) -> None:
        if self._decision_progress_callback is None:
            return
        event = {"step_stage": str(stage).strip()}
        event.update(payload)
        self._decision_progress_callback(event)

    def decide(self, state: AgentState) -> ActionDecision:
        pre_context_adjacent_success = self._pre_context_adjacent_success_direct_decision(state)
        if pre_context_adjacent_success is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_adjacent_success)
        pre_context_synthetic_edit = self._pre_context_synthetic_edit_plan_direct_decision(state)
        if pre_context_synthetic_edit is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_synthetic_edit)
        pre_context_shared_repo_integrator = self._pre_context_shared_repo_integrator_direct_decision(state)
        if pre_context_shared_repo_integrator is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_shared_repo_integrator)
        pre_context_git_repo_review = self._pre_context_git_repo_review_direct_decision(state)
        if pre_context_git_repo_review is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_git_repo_review)
        pre_context_plan_progress = self._pre_context_plan_progress_direct_decision(state)
        if pre_context_plan_progress is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_plan_progress)
        pre_context_trusted_retrieval_carryover = self._pre_context_trusted_retrieval_carryover_decision(state)
        if pre_context_trusted_retrieval_carryover is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_trusted_retrieval_carryover)
        pre_context_recovery_exhaustion = self._pre_context_recovery_exhaustion_decision(state)
        if pre_context_recovery_exhaustion is not None:
            return self._apply_pre_context_tolbert_route(state, pre_context_recovery_exhaustion)
        if self.context_provider is not None:
            reused_packet = self._reusable_context_packet(state)
            if reused_packet is not None:
                state.context_packet = reused_packet
                self._emit_decision_progress("context_reuse")
            else:
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
                self._stamp_context_reuse_signature(state)
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
        synthetic_edit_priority = self._synthetic_edit_plan_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if synthetic_edit_priority is not None:
            return synthetic_edit_priority
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
        planner_recovery_rewrite_brief = self._planner_recovery_rewrite_brief(
            state,
            blocked_commands=blocked_commands,
        )
        tolbert_shadow = self._tolbert_shadow_decision(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            route_mode=tolbert_route.mode,
        )
        if tolbert_route.mode == "primary" and not planner_recovery_rewrite_brief:
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
            history_payload=[
                asdict(step)
                for step in state.history[-max(1, int(self.config.payload_history_step_window)) :]
            ],
            history_archive=dict(state.history_archive),
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
        if planner_recovery_rewrite_brief:
            payload["planner_recovery_brief"] = planner_recovery_rewrite_brief
        software_work_phase_gate_brief = self._software_work_phase_gate_brief(state)
        if software_work_phase_gate_brief:
            payload["software_work_phase_gate_brief"] = software_work_phase_gate_brief
        campaign_contract_brief = self._campaign_contract_brief(state)
        if campaign_contract_brief:
            payload["campaign_contract_brief"] = campaign_contract_brief
        if state.planner_recovery_artifact:
            payload["planner_recovery_artifact"] = dict(state.planner_recovery_artifact)
        system_prompt = self._role_system_prompt(role)
        decision_prompt = self._role_decision_prompt(role)
        if planner_recovery_rewrite_brief:
            decision_prompt = (
                f"{decision_prompt}\n"
                "Recovery rewrite directive: "
                f"{planner_recovery_rewrite_brief}"
            )
        if software_work_phase_gate_brief:
            decision_prompt = (
                f"{decision_prompt}\n"
                "Software-work phase gate: "
                f"{software_work_phase_gate_brief}"
            )
        if campaign_contract_brief:
            decision_prompt = (
                f"{decision_prompt}\n"
                "Long-horizon campaign contract: "
                f"{campaign_contract_brief}"
            )
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
        integrator_segment_decision = self._shared_repo_integrator_segment_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if integrator_segment_decision is not None:
            return integrator_segment_decision
        git_repo_review_decision = self._git_repo_review_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if git_repo_review_decision is not None:
            return git_repo_review_decision
        adjacent_success_decision = self._adjacent_success_direct_decision(state)
        if adjacent_success_decision is not None:
            return adjacent_success_decision
        synthetic_edit_plan_decision = self._synthetic_edit_plan_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if synthetic_edit_plan_decision is not None:
            return synthetic_edit_plan_decision
        if self._planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
            return None
        plan_progress_decision = self._plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if plan_progress_decision is not None:
            return plan_progress_decision
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
        integrator_segment_decision = self._shared_repo_integrator_segment_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if integrator_segment_decision is not None:
            return integrator_segment_decision
        git_repo_review_decision = self._git_repo_review_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if git_repo_review_decision is not None:
            return git_repo_review_decision
        horizon = str(
            state.world_model_summary.get(
                "horizon",
                state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
            )
        ).strip()
        if state.active_subgoal_diagnosis() or (state.active_subgoal and horizon == "long_horizon"):
            plan_progress_decision = self._plan_progress_direct_decision(
                state,
                blocked_commands=blocked_commands,
            )
            if plan_progress_decision is not None:
                return plan_progress_decision
        if state.context_packet is not None and self._has_retrieval_signal(state) and state.history and state.consecutive_failures > 0:
            return ActionDecision(
                thought="Pause unsafe repetition and hand control back to synthesis.",
                action="respond",
                content="No safe deterministic command remains.",
                done=True,
            )
        if self._recovery_contract_exhausted(state, blocked_commands=blocked_commands):
            return ActionDecision(
                thought="No safe task-contract recovery command remains.",
                action="respond",
                content="No safe deterministic recovery command remains.",
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
        integrator_segment_decision = self._shared_repo_integrator_segment_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if integrator_segment_decision is not None:
            return integrator_segment_decision
        git_repo_review_decision = self._git_repo_review_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if git_repo_review_decision is not None:
            return git_repo_review_decision
        adjacent_success_decision = self._adjacent_success_direct_decision(state)
        if adjacent_success_decision is not None:
            return adjacent_success_decision
        synthetic_edit_plan_decision = self._synthetic_edit_plan_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if synthetic_edit_plan_decision is not None:
            return synthetic_edit_plan_decision
        plan_progress_decision = self._plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if plan_progress_decision is not None:
            return plan_progress_decision
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
        if role == "planner" and self._planner_recovery_rewrite_required(
            state,
            blocked_commands=self._blocked_commands(state),
        ):
            return None
        require_active_skill_ranking = (
            not state.history
            and retrieval_has_signal
            and self._tolbert_skill_ranking_enabled()
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

    def _adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if state.history:
            return None
        if str(state.task.metadata.get("curriculum_kind", "")).strip() != "adjacent_success":
            return None
        for command in state.task.suggested_commands[:2]:
            normalized = str(command).strip()
            if not normalized:
                continue
            if self._command_control_score(state, normalized) < 0:
                continue
            if not self._first_step_command_covers_required_artifacts(state, normalized):
                continue
            return ActionDecision(
                thought="Use adjacent-success task contract command.",
                action=CODE_EXECUTE,
                content=_normalize_command_for_workspace(normalized, state.task.workspace_subdir),
                done=False,
                decision_source="adjacent_success_direct",
                retrieval_influenced=False,
            )
        return None

    def _synthetic_edit_plan_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        edit_plan = state.task.metadata.get("synthetic_edit_plan", [])
        if not isinstance(edit_plan, list) or not edit_plan:
            return None
        if not self._synthetic_edit_plan_direct_active(state):
            return None
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        executed_commands = state.all_executed_command_signatures()
        ranked_steps: list[tuple[tuple[int, int, int, int], int, dict[str, object] | None, dict[str, object]]] = []
        for index, step in enumerate(edit_plan):
            if not isinstance(step, dict):
                continue
            syntax_motor = summarize_python_edit_step(
                step,
                expected_file_contents=state.task.expected_file_contents,
            )
            ranked_steps.append(
                (
                    self._synthetic_edit_step_priority_key(step, syntax_motor=syntax_motor),
                    index,
                    syntax_motor,
                    step,
                )
            )
        for _, _, syntax_motor, step in sorted(ranked_steps, key=lambda item: (item[0], -item[1]), reverse=True):
            command = self._render_synthetic_edit_step_command(step)
            if not command:
                continue
            normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked or canonical in executed_commands:
                continue
            if self._command_control_score(state, normalized) < 0:
                continue
            if not state.history and not self._first_step_command_covers_required_artifacts(state, normalized):
                continue
            edit_kind = str(step.get("edit_kind", "rewrite")).strip() or "rewrite"
            return ActionDecision(
                thought="Advance the task using the next synthetic edit-plan step.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="synthetic_edit_plan_direct",
                proposal_source=f"structured_edit:{edit_kind}",
                proposal_novel=False,
                proposal_metadata={
                    "edit_kind": edit_kind,
                    "edit_source": "synthetic_edit_plan",
                    "path": str(step.get("path", "")).strip(),
                    "edit_score": int(step.get("edit_score", 0) or 0),
                    "syntax_motor": syntax_motor,
                    "target_symbol_fqn": (
                        str(syntax_motor.get("edited_symbol_fqn", "")).strip()
                        if isinstance(syntax_motor, dict)
                        else ""
                    ),
                    "enclosing_symbol_qualname": (
                        str(syntax_motor.get("enclosing_symbol_qualname", "")).strip()
                        if isinstance(syntax_motor, dict)
                        else ""
                    ),
                    "import_change_risk": bool(syntax_motor.get("import_change_risk", False))
                    if isinstance(syntax_motor, dict)
                    else False,
                    "signature_change_risk": bool(syntax_motor.get("signature_change_risk", False))
                    if isinstance(syntax_motor, dict)
                    else False,
                    "call_targets": list(syntax_motor.get("call_targets_after", []))[:8]
                    if isinstance(syntax_motor, dict)
                    else [],
                },
                retrieval_influenced=False,
            )
        return None

    @staticmethod
    def _synthetic_edit_step_priority_key(
        step: dict[str, object],
        *,
        syntax_motor: dict[str, object] | None,
    ) -> tuple[int, int, int, int]:
        edit_score = int(step.get("edit_score", 0) or 0)
        targeted_symbol_bonus = 1 if isinstance(syntax_motor, dict) and syntax_motor.get("edited_symbol_fqn") else 0
        import_risk_penalty = -1 if isinstance(syntax_motor, dict) and syntax_motor.get("import_change_risk") else 0
        signature_risk_penalty = -1 if isinstance(syntax_motor, dict) and syntax_motor.get("signature_change_risk") else 0
        return (
            edit_score + (targeted_symbol_bonus * 4) + (import_risk_penalty * 3) + (signature_risk_penalty * 2),
            targeted_symbol_bonus,
            -1 if import_risk_penalty else 0,
            -1 if signature_risk_penalty else 0,
        )

    def _shared_repo_integrator_segment_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if bool(metadata.get("synthetic_worker", False)):
            return None
        if not str(metadata.get("workflow_guard", {})).strip() and not verifier:
            return None
        if int(metadata.get("shared_repo_order", 0) or 0) <= 0:
            return None
        required_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if not required_branches:
            return None
        if not state.task.suggested_commands:
            return None
        primary_command = str(state.task.suggested_commands[0]).strip()
        if " && " not in primary_command:
            return None
        segments = self._shared_repo_integrator_grouped_segments(primary_command)
        if len(segments) < 2:
            return None
        required_branch_segment = self._shared_repo_required_branch_segment(
            state,
            segments=segments,
            required_branches=required_branches,
        )
        if required_branch_segment is not None:
            normalized = _normalize_command_for_workspace(
                required_branch_segment,
                state.task.workspace_subdir,
            )
            return ActionDecision(
                thought="Advance the shared-repo integrator by accepting the next unresolved required branch.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="shared_repo_integrator_segment_direct",
                retrieval_influenced=False,
            )
        sequential_segment = self._shared_repo_integrator_next_segment(state, segments)
        if sequential_segment is not None:
            normalized = _normalize_command_for_workspace(
                sequential_segment,
                state.task.workspace_subdir,
            )
            control_score = self._command_control_score(state, normalized)
            adjusted_control_score = control_score + max(0, self._software_work_phase_gate_command_score(state, normalized))
            if adjusted_control_score >= -8:
                return ActionDecision(
                    thought="Continue the shared-repo integrator through its next workflow segment.",
                    action=CODE_EXECUTE,
                    content=normalized,
                    done=False,
                    decision_source="shared_repo_integrator_segment_direct",
                    retrieval_influenced=False,
                )
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        executed = state.all_successful_command_signatures()
        ranked_segments: list[tuple[tuple[int, int, str], str]] = []
        for segment in segments:
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in blocked or canonical in executed:
                continue
            control_score = self._command_control_score(state, normalized)
            adjusted_control_score = control_score + max(0, self._software_work_phase_gate_command_score(state, normalized))
            phase_rank = self._shared_repo_integrator_segment_phase_rank(normalized)
            if adjusted_control_score < -4:
                continue
            ranked_segments.append(((phase_rank, -adjusted_control_score, normalized), normalized))
        if not ranked_segments:
            return None
        ranked_segments.sort(key=lambda item: item[0])
        normalized = ranked_segments[0][1]
        return ActionDecision(
            thought="Advance the shared-repo integrator one workflow segment at a time.",
            action=CODE_EXECUTE,
            content=normalized,
            done=False,
            decision_source="shared_repo_integrator_segment_direct",
            retrieval_influenced=False,
        )

    @staticmethod
    def _shared_repo_required_branch_segment(
        state: AgentState,
        *,
        segments: list[str],
        required_branches: list[str],
    ) -> str | None:
        if not segments or not required_branches:
            return None
        unresolved = {branch for branch in required_branches if branch}
        if state.history:
            unresolved = set()
            verification = state.history[-1].verification if isinstance(state.history[-1].verification, dict) else {}
            for reason in verification.get("reasons", []):
                text = str(reason).strip()
                if not text.startswith("required worker branch not accepted into "):
                    continue
                branch = text.rsplit(": ", 1)[-1].strip()
                if branch:
                    unresolved.add(branch)
        if not unresolved:
            return None
        successful = state.all_successful_command_signatures()
        for segment in segments:
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if canonical in successful:
                continue
            if "git merge" not in normalized:
                continue
            if any(branch in normalized for branch in unresolved):
                return segment
        return None

    @staticmethod
    def _shared_repo_integrator_next_segment(
        state: AgentState,
        segments: list[str],
    ) -> str | None:
        if not segments:
            return None
        successful = state.all_successful_command_signatures()
        for segment in segments:
            canonical = _canonicalize_command(
                _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            )
            if canonical not in successful:
                return segment
        return None

    @staticmethod
    def _shared_repo_integrator_segment_phase_rank(command: str) -> int:
        normalized = str(command).strip()
        if normalized.startswith("git merge "):
            return 0
        if "reports/" in normalized or normalized.startswith("mkdir -p reports"):
            return 3
        if normalized.startswith("printf ") or normalized.startswith("scripts/") or normalized.startswith("./scripts/"):
            return 1
        if normalized.startswith("tests/") or " tests/" in normalized:
            return 2
        if normalized.startswith("git add ") or normalized.startswith("git commit "):
            if "reports/" in normalized:
                return 3
            return 4
        if normalized.startswith("git diff ") or normalized.startswith("cat "):
            return 5
        if normalized.startswith("git branch "):
            return 6
        return 7

    @classmethod
    def _shared_repo_integrator_grouped_segments(cls, command: str) -> list[str]:
        raw_segments = [segment.strip() for segment in str(command).split(" && ") if segment.strip()]
        if len(raw_segments) < 2:
            return raw_segments
        grouped: list[str] = []
        buffer: list[str] = []
        buffer_phase: int | None = None
        for segment in raw_segments:
            phase = cls._shared_repo_integrator_segment_phase_rank(segment)
            starts_new_group = False
            if phase == 0:
                starts_new_group = True
            elif buffer_phase is None:
                starts_new_group = True
            elif phase != buffer_phase:
                if {phase, buffer_phase} <= {3, 4}:
                    starts_new_group = False
                elif (
                    phase == 4
                    and buffer_phase == 1
                    and buffer
                    and str(segment).strip().startswith("git add ")
                ):
                    starts_new_group = False
                else:
                    starts_new_group = True
            elif phase in {3, 4, 5, 6}:
                starts_new_group = False
            if starts_new_group and buffer:
                grouped.append(" && ".join(buffer))
                buffer = []
                buffer_phase = None
            buffer.append(segment)
            buffer_phase = phase if buffer_phase is None else max(buffer_phase, phase)
        if buffer:
            grouped.append(" && ".join(buffer))
        return grouped

    @staticmethod
    def _synthetic_edit_plan_direct_active(state: AgentState) -> bool:
        metadata = dict(state.task.metadata)
        if bool(metadata.get("synthetic_worker", False)):
            return True
        difficulty = str(
            metadata.get("difficulty", metadata.get("task_difficulty", ""))
        ).strip().lower()
        if difficulty == "long_horizon":
            return True
        if str(state.world_model_summary.get("horizon", "")).strip() == "long_horizon":
            return True
        return bool(state.active_subgoal) or bool(state.history)

    @staticmethod
    def _render_synthetic_edit_step_command(step: dict[str, object]) -> str:
        command = _render_structured_edit_command(step)
        if command:
            return command
        path = str(step.get("path", "")).strip()
        target_content = step.get("target_content")
        if not path or target_content is None:
            return ""
        write_commands: list[str] = []
        parent = Path(path).parent
        if str(parent) not in {"", "."}:
            write_commands.append(f"mkdir -p {shlex.quote(str(parent))}")
        write_commands.append(f"printf %s {shlex.quote(str(target_content))} > {shlex.quote(path)}")
        return " && ".join(write_commands)

    def _plan_progress_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        ranked = self._rank_plan_progress_candidates(
            state,
            blocked_commands=blocked_commands,
        )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        command = ranked[0][2]
        return ActionDecision(
            thought="Use task-contract command to advance the current subgoal.",
            action=CODE_EXECUTE,
            content=_normalize_command_for_workspace(command, state.task.workspace_subdir),
            done=False,
            decision_source="plan_progress_direct",
            retrieval_influenced=False,
        )

    def _rank_plan_progress_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> list[tuple[int, int, str]]:
        if not state.history:
            return []
        if not state.task.suggested_commands:
            return []
        active_subgoal = str(state.active_subgoal or "").strip()
        if (
            not active_subgoal
            and state.consecutive_no_progress_steps <= 0
            and state.consecutive_failures <= 0
            and str(state.world_model_summary.get("horizon", "")).strip() != "long_horizon"
        ):
            return []
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        failed_commands = state.all_failed_command_signatures()
        ranked: list[tuple[int, int, str]] = []
        phase_gate = state.software_work_phase_gate_state()
        gate_objectives = [
            str(item).strip()
            for item in phase_gate.get("gate_objectives", [])
            if str(item).strip()
        ] if isinstance(phase_gate, dict) else []
        for command in state.task.suggested_commands[:5]:
            normalized = str(command).strip()
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked:
                continue
            if gate_objectives and not self._command_matches_any_software_work_objective(normalized, gate_objectives):
                continue
            if canonical in failed_commands and (state.consecutive_failures > 0 or state.consecutive_no_progress_steps > 0):
                continue
            control_score = self._command_control_score(state, normalized)
            alignment_score = self._subgoal_alignment_score(state, normalized)
            alignment_score += self._software_work_phase_gate_command_score(state, normalized)
            if active_subgoal and alignment_score <= 0:
                continue
            combined_score = control_score + alignment_score
            if combined_score <= 0:
                continue
            retry_penalty = self._recovery_loop_penalty(state, normalized)
            ranked.append((combined_score, -retry_penalty, normalized))
        return ranked

    def _recovery_contract_exhausted(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        if self._rank_plan_progress_candidates(state, blocked_commands=blocked_commands):
            return False
        if not state.history or not state.task.suggested_commands:
            return False
        active_subgoal = str(state.active_subgoal or "").strip()
        horizon = str(state.world_model_summary.get("horizon", "")).strip()
        if not active_subgoal and state.consecutive_failures <= 0 and state.consecutive_no_progress_steps <= 0 and horizon != "long_horizon":
            return False
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        failed_commands = state.all_failed_command_signatures()
        last_command = _canonicalize_command(str(state.last_action_signature).partition(":")[2])
        recovery_seen = False
        for command in state.task.suggested_commands[:5]:
            normalized = str(command).strip()
            canonical = _canonicalize_command(normalized)
            if not canonical:
                continue
            recovery_seen = True
            if canonical in normalized_blocked:
                continue
            if canonical in failed_commands:
                continue
            if canonical == last_command and state.repeated_action_count > 1:
                continue
            return False
        return recovery_seen

    def _pre_context_adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if state.history:
            return None
        if str(state.task.metadata.get("curriculum_kind", "")).strip() != "adjacent_success":
            return None
        forbidden = {
            str(path).strip()
            for path in state.task.forbidden_files
            if str(path).strip()
        }
        for command in state.task.suggested_commands[:2]:
            normalized = str(command).strip()
            if not normalized:
                continue
            if forbidden and any(path in normalized for path in forbidden):
                continue
            if not self._first_step_command_covers_required_artifacts(state, normalized):
                continue
            return ActionDecision(
                thought="Use adjacent-success task contract command before context compilation.",
                action=CODE_EXECUTE,
                content=_normalize_command_for_workspace(normalized, state.task.workspace_subdir),
                done=False,
                decision_source="adjacent_success_direct",
                retrieval_influenced=False,
            )
        return None

    def _pre_context_synthetic_edit_plan_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self._synthetic_edit_plan_direct_decision(state, blocked_commands=[])

    def _pre_context_shared_repo_integrator_direct_decision(self, state: AgentState) -> ActionDecision | None:
        decision = self._shared_repo_integrator_segment_direct_decision(state, blocked_commands=[])
        if decision is None:
            return None
        if state.history:
            return decision
        return decision if self._shared_repo_integrator_first_segment_safe_without_context(state, decision.content) else None

    def _git_repo_review_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        if state.history:
            return None
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        if str(metadata.get("curriculum_kind", "")).strip() == "adjacent_success":
            return None
        if int(metadata.get("shared_repo_order", 0) or 0) > 0:
            return None
        if bool(metadata.get("synthetic_worker", False)):
            return None
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if str(verifier.get("kind", "")).strip() != "git_repo_review":
            return None
        if verifier.get("required_merged_branches"):
            return None
        if not state.task.suggested_commands:
            return None
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        for command in state.task.suggested_commands[:2]:
            normalized = _normalize_command_for_workspace(
                str(command).strip(),
                state.task.workspace_subdir,
            )
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked:
                continue
            if self._command_control_score(state, normalized) < 0:
                continue
            if not self._git_repo_review_first_step_safe_without_context(state, normalized):
                continue
            return ActionDecision(
                thought="Use the git repo review task contract command.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="git_repo_review_direct",
                retrieval_influenced=False,
            )
        return None

    def _pre_context_git_repo_review_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self._git_repo_review_direct_decision(state, blocked_commands=[])

    def _pre_context_plan_progress_direct_decision(self, state: AgentState) -> ActionDecision | None:
        role = self._normalized_role(state)
        if role not in {"planner", "critic"}:
            return None
        blocked_commands = sorted(state.all_failed_command_signatures())
        if role == "planner" and self._planner_recovery_rewrite_required(
            state,
            blocked_commands=blocked_commands,
        ):
            return None
        return self._plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    def _pre_context_trusted_retrieval_carryover_decision(self, state: AgentState) -> ActionDecision | None:
        role = self._normalized_role(state)
        if role not in {"planner", "critic"}:
            return None
        if not self._trusted_retrieval_carryover_active(state):
            return None
        candidates = self._trusted_retrieval_carryover_candidates(
            state,
            blocked_commands=state.all_failed_command_signatures(),
        )
        if not candidates:
            return None
        best = candidates[0]
        command = str(best.get("command", "")).strip()
        if not command or int(best.get("control_score", 0) or 0) <= 0:
            return None
        return ActionDecision(
            thought=(
                "Continue a previously trusted retrieval-backed repair sequence."
                if bool(best.get("procedure", False))
                else (
                    "Reuse a previously trusted retrieval-backed write pattern."
                    if bool(best.get("generated", False))
                    else "Reuse a previously trusted retrieval-backed repair command."
                )
            ),
            action=CODE_EXECUTE,
            content=command,
            done=False,
            selected_retrieval_span_id=str(best.get("span_id", "")).strip() or None,
            retrieval_influenced=True,
            decision_source="trusted_retrieval_carryover_direct",
        )

    def _pre_context_recovery_exhaustion_decision(self, state: AgentState) -> ActionDecision | None:
        if self._normalized_role(state) != "critic":
            return None
        blocked_commands = sorted(state.all_failed_command_signatures())
        if not self._recovery_contract_exhausted(state, blocked_commands=blocked_commands):
            return None
        return ActionDecision(
            thought="No safe task-contract recovery command remains.",
            action="respond",
            content="No safe deterministic recovery command remains.",
            done=True,
        )

    def _planner_recovery_rewrite_required(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if artifact:
            if str(artifact.get("kind", "")).strip() != "planner_recovery_rewrite":
                return False
            if str(artifact.get("source_subgoal", "")).strip() != str(state.active_subgoal).strip():
                return False
            return True
        if self._normalized_role(state) != "planner":
            return False
        diagnosis = state.active_subgoal_diagnosis()
        if str(diagnosis.get("source_role", "")).strip().lower() != "critic":
            return False
        return self._recovery_contract_exhausted(state, blocked_commands=blocked_commands)

    def _planner_recovery_rewrite_brief(self, state: AgentState, *, blocked_commands: list[str]) -> str:
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if artifact and self._planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
            rewritten_subgoal = str(artifact.get("rewritten_subgoal", "")).strip()
            next_stage_objective = str(artifact.get("next_stage_objective", "")).strip()
            summary = str(artifact.get("summary", "")).strip()
            contract_outline = [
                str(item).strip()
                for item in artifact.get("contract_outline", [])
                if str(item).strip()
            ]
            stale_commands = [
                str(item).strip()
                for item in artifact.get("stale_commands", [])
                if str(item).strip()
            ]
            related_objectives = [
                str(item).strip()
                for item in artifact.get("related_objectives", [])
                if str(item).strip()
            ]
            parts = []
            if rewritten_subgoal:
                parts.append(f"Critic has exhausted the bounded repair set. New planner objective: {rewritten_subgoal}.")
            if next_stage_objective:
                parts.append(f"Current staged recovery objective: {next_stage_objective}.")
            if summary:
                parts.append(f"Diagnosis: {summary}.")
            if related_objectives:
                parts.append(f"Related verifier obligations: {', '.join(related_objectives[:4])}.")
            if stale_commands:
                parts.append(f"Do not reuse stale repair commands: {', '.join(stale_commands[:3])}.")
            if contract_outline:
                parts.append(f"Rewrite outline: {'; '.join(contract_outline[:3])}.")
            return " ".join(parts).strip()
        if not self._planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
            return ""
        diagnosis = state.active_subgoal_diagnosis()
        active_subgoal = str(state.active_subgoal or "").strip() or "the current verifier-facing subgoal"
        diagnosis_summary = str(diagnosis.get("summary", "")).strip()
        path = str(diagnosis.get("path", "")).strip()
        failed_candidates = [
            str(command).strip()
            for command in state.task.suggested_commands[:5]
            if _canonicalize_command(str(command).strip()) in state.all_failed_command_signatures()
        ]
        exhausted_surface = ", ".join(failed_candidates[:3]) if failed_candidates else "current task-contract repair commands"
        location_clause = f" around {path}" if path else ""
        summary_clause = f" Critic diagnosis: {diagnosis_summary}." if diagnosis_summary else ""
        return (
            f"Critic has exhausted the bounded recovery set for {active_subgoal}{location_clause}.{summary_clause} "
            f"Do not reuse or lightly re-rank the stale repair surface ({exhausted_surface}). "
            "Synthesize a fresh verifier-relevant subgoal or rewrite the recovery contract before choosing the next command."
        )

    def _software_work_phase_gate_brief(self, state: AgentState) -> str:
        gate_state = state.software_work_phase_gate_state()
        if not isinstance(gate_state, dict) or not gate_state:
            return ""
        gate_phase = str(gate_state.get("gate_phase", "")).strip()
        gate_reason = str(gate_state.get("gate_reason", "")).strip()
        gate_objectives = [
            str(item).strip()
            for item in gate_state.get("gate_objectives", [])
            if str(item).strip()
        ]
        blocked_phases = [
            str(item).strip()
            for item in gate_state.get("blocked_phases", [])
            if str(item).strip()
        ]
        parts: list[str] = []
        if gate_phase:
            parts.append(f"Current workflow gate phase: {gate_phase}.")
        if gate_reason:
            parts.append(gate_reason)
        if gate_objectives:
            parts.append(f"Resolve these obligations first: {', '.join(gate_objectives[:3])}.")
        if blocked_phases:
            parts.append(f"Do not advance into later phases yet: {', '.join(blocked_phases)}.")
        return " ".join(parts).strip()

    def _campaign_contract_brief(self, state: AgentState) -> str:
        contract = state.campaign_contract_state()
        if not isinstance(contract, dict) or not contract:
            return ""
        current_objective = str(contract.get("current_objective", "")).strip()
        anchor_objectives = [
            str(item).strip()
            for item in contract.get("anchor_objectives", [])
            if str(item).strip()
        ]
        regressed_objectives = [
            str(item).strip()
            for item in contract.get("regressed_objectives", [])
            if str(item).strip()
        ]
        stalled_objectives = [
            str(item).strip()
            for item in contract.get("stalled_objectives", [])
            if str(item).strip()
        ]
        parts: list[str] = []
        if current_objective:
            parts.append(f"Current campaign objective: {current_objective}.")
        if anchor_objectives:
            parts.append(f"Unresolved campaign obligations: {', '.join(anchor_objectives[:3])}.")
        if regressed_objectives:
            parts.append(f"Regressed obligations to restore first: {', '.join(regressed_objectives[:2])}.")
        elif stalled_objectives:
            parts.append(f"Stalled obligations still anchoring the run: {', '.join(stalled_objectives[:2])}.")
        return " ".join(parts).strip()

    def _apply_pre_context_tolbert_route(self, state: AgentState, decision: ActionDecision) -> ActionDecision:
        route = self._tolbert_route_decision(state)
        return self._apply_tolbert_shadow(decision, {}, route.mode)

    def _reusable_context_packet(self, state: AgentState):
        packet = state.context_packet
        if packet is None or not self._context_packet_reuse_enabled(state):
            return None
        control = packet.control if isinstance(packet.control, dict) else {}
        expected = self._context_reuse_signature(state)
        if str(control.get("context_reuse_signature", "")).strip() != expected:
            return None
        return packet

    def _stamp_context_reuse_signature(self, state: AgentState) -> None:
        packet = state.context_packet
        if packet is None or not self._context_packet_reuse_enabled(state):
            return
        control = dict(packet.control) if isinstance(packet.control, dict) else {}
        control["context_reuse_signature"] = self._context_reuse_signature(state)
        packet.control = control

    @staticmethod
    def _context_packet_reuse_enabled(state: AgentState) -> bool:
        if not state.history:
            return False
        horizon = str(
            state.world_model_summary.get(
                "horizon",
                state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
            )
        ).strip()
        return horizon == "long_horizon"

    def _context_reuse_signature(self, state: AgentState) -> str:
        history_window = [
            {
                "index": int(step.index),
                "action": str(step.action),
                "content": str(step.content),
                "selected_retrieval_span_id": str(step.selected_retrieval_span_id or ""),
                "retrieval_influenced": bool(step.retrieval_influenced),
                "trust_retrieval": bool(step.trust_retrieval),
                "retrieval_command_match": bool(step.retrieval_command_match),
                "verification": dict(step.verification or {}),
                "command_result": dict(step.command_result or {}) if step.command_result else None,
            }
            for step in state.history[-3:]
        ]
        payload = {
            "task": asdict(state.task),
            "recent_workspace_summary": str(state.recent_workspace_summary),
            "world_model_summary": dict(state.world_model_summary or {}),
            "history_window": history_window,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return digest

    def _shared_repo_integrator_first_segment_safe_without_context(self, state: AgentState, command: str) -> bool:
        normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
        adjusted_control_score = self._command_control_score(state, normalized) + max(
            0,
            self._software_work_phase_gate_command_score(state, normalized),
        )
        if adjusted_control_score < 0:
            return False
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_branches = {
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        }
        if required_branches and any(branch in normalized for branch in required_branches):
            return True
        prioritized_paths = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if not prioritized_paths:
            prioritized_paths = [
                str(path).strip()
                for path in workflow_guard.get("claimed_paths", [])
                if str(path).strip()
            ]
        if prioritized_paths and any(path in normalized for path in prioritized_paths):
            return True
        return self._first_step_command_covers_required_artifacts(state, normalized)

    def _git_repo_review_first_step_safe_without_context(self, state: AgentState, command: str) -> bool:
        normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if str(verifier.get("kind", "")).strip() != "git_repo_review":
            return False
        expected_branch = str(verifier.get("expected_branch", "")).strip()
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        worker_branch = str(workflow_guard.get("worker_branch", "")).strip()
        if expected_branch and expected_branch != worker_branch and expected_branch not in normalized:
            return False
        prioritized_paths = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if not prioritized_paths:
            prioritized_paths = [
                str(path).strip()
                for path in workflow_guard.get("claimed_paths", [])
                if str(path).strip()
            ]
        if prioritized_paths:
            matched_paths = sum(1 for path in prioritized_paths if path in normalized)
            required_artifact_bias = max(0, self._policy_control_int("required_artifact_first_step_bias"))
            minimum_matches = max(1, len(prioritized_paths) - required_artifact_bias)
            if matched_paths < minimum_matches:
                return False
        required_test_commands: list[str] = []
        for test_command in verifier.get("test_commands", []):
            if not isinstance(test_command, dict):
                continue
            argv = test_command.get("argv", [])
            if not isinstance(argv, list) or not argv:
                continue
            entry = str(argv[0]).strip()
            if entry:
                required_test_commands.append(entry)
        if required_test_commands and not all(entry in normalized for entry in required_test_commands):
            return False
        return True

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
                span_id=str(entry.get("span_id", "")).strip(),
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

    def _screened_tolbert_retrieval_guidance(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
    ) -> dict[str, list[str]]:
        spans = retrieval_guidance.get("recommended_command_spans", [])
        if state.context_packet is None or not isinstance(spans, list) or not spans:
            return retrieval_guidance
        trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False))
        confidence = float(state.context_packet.control.get("path_confidence", 0.0))
        normalized_blocked = {SkillLibrary._normalize_command(command) for command in blocked_commands}
        filtered_spans: list[dict[str, str]] = []
        filtered_commands: list[str] = []
        seen: set[str] = set()
        for entry in spans:
            command = str(entry.get("command", "")).strip()
            normalized = SkillLibrary._normalize_command(command)
            if not normalized or normalized in normalized_blocked or normalized in seen:
                continue
            control_score = self._command_control_score(state, command)
            if not self._can_execute_direct_retrieval_command(
                state,
                top_skill=top_skill,
                command=command,
                span_id=str(entry.get("span_id", "")).strip(),
                control_score=control_score,
                confidence=confidence,
                trust_retrieval=trust_retrieval,
            ):
                continue
            seen.add(normalized)
            filtered_spans.append(
                {
                    "span_id": str(entry.get("span_id", "")).strip(),
                    "command": command,
                }
            )
            filtered_commands.append(command)
        screened = dict(retrieval_guidance)
        screened["recommended_commands"] = filtered_commands
        screened["recommended_command_spans"] = filtered_spans
        return screened

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
            score += self._trusted_retrieval_carryover_match_bonus(state, command)
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
        span_id: str,
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
        if (
            confidence < max(
                self._retrieval_float("tolbert_deterministic_command_confidence"),
                self._retrieval_float("tolbert_first_step_direct_command_confidence"),
                self._retrieval_float("tolbert_first_step_direct_command_confidence")
                + self._policy_control_float("direct_command_confidence_boost"),
            )
            and not self._trusted_workflow_guidance_bypasses_first_step_confidence(
                state,
                command=command,
                span_id=span_id,
            )
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

    def _trusted_workflow_guidance_bypasses_first_step_confidence(
        self,
        state: AgentState,
        *,
        command: str,
        span_id: str,
    ) -> bool:
        if state.history:
            return False
        normalized_span_id = str(span_id).strip()
        if not normalized_span_id:
            return False
        if not (
            normalized_span_id.startswith("learning:success_skill:")
            or normalized_span_id.startswith("learning:recovery_case:")
            or normalized_span_id.startswith("procedure:")
            or normalized_span_id.startswith("tool:")
        ):
            return False
        workflow_guard = state.task.metadata.get("workflow_guard", {}) or {}
        if not isinstance(workflow_guard, dict):
            return False
        claimed_paths = [
            str(path).strip()
            for path in workflow_guard.get("claimed_paths", [])
            if str(path).strip()
        ]
        if not claimed_paths:
            return False
        return self._first_step_guarded_command_coverage(state, command, claimed_paths)

    @staticmethod
    def _first_step_guarded_command_coverage(
        state: AgentState,
        command: str,
        claimed_paths: list[str],
    ) -> bool:
        desired_paths = {
            str(path).strip()
            for path in (
                *claimed_paths,
                *state.task.expected_files,
                *state.task.expected_file_contents.keys(),
            )
            if str(path).strip()
        }
        covered_paths = {path for path in desired_paths if path in command}
        if len(covered_paths) >= 2:
            return True
        expected_outputs = {
            str(path).strip()
            for path in (
                *state.task.expected_files,
                *state.task.expected_file_contents.keys(),
            )
            if str(path).strip()
        }
        return bool(covered_paths and covered_paths & expected_outputs and "git " in command)

    def _command_control_score(self, state: AgentState, command: str) -> int:
        score = self.universe_model.score_command(state.universe_summary, command)
        score += self.world_model.score_command(state.world_model_summary, command)
        score += self._transition_model_command_score(state, command)
        score += self._graph_memory_environment_command_score(state, command)
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
        active_diagnosis = state.active_subgoal_diagnosis()
        if verifier_alignment_bias:
            if any(path and path in command for path in state.task.expected_files):
                score += verifier_alignment_bias
            if any(path and path in command for path in state.task.expected_file_contents):
                score += verifier_alignment_bias
            if any(path and path in command for path in state.task.forbidden_files):
                score -= verifier_alignment_bias * 2
        score += self._active_subgoal_diagnosis_command_score(
            state,
            command,
            role=role,
            active_subgoal=active_subgoal,
            diagnosis=active_diagnosis,
        )
        score += self._planner_recovery_stage_command_score(
            state,
            command,
            role=role,
        )
        score += self._campaign_contract_command_score(
            state,
            command,
            role=role,
        )
        if role == "planner":
            planner_subgoal_command_bias = self._policy_control_int("planner_subgoal_command_bias")
            if active_subgoal and any(token in command for token in active_subgoal.split()):
                score += max(2, planner_subgoal_command_bias)
            if "mkdir -p " in command or "cp " in command:
                score += 1
            score -= self._recovery_loop_penalty(state, command)
        elif role == "critic":
            critic_repeat_failure_bias = self._policy_control_int("critic_repeat_failure_bias")
            score -= max(
                self._recovery_loop_penalty(state, command),
                self._failed_command_attempts(state, command) * max(4, critic_repeat_failure_bias * 2),
            )
            if "rm -rf" in command:
                score -= 4
        elif role == "executor":
            if "printf " in command or "> " in command:
                score += 1
        return score

    def _campaign_contract_command_score(
        self,
        state: AgentState,
        command: str,
        *,
        role: str,
    ) -> int:
        if role not in {"planner", "critic"}:
            return 0
        contract = state.campaign_contract_state()
        if not isinstance(contract, dict) or not contract:
            return 0
        anchors = [
            str(item).strip()
            for item in contract.get("anchor_objectives", [])
            if str(item).strip()
        ]
        if not anchors:
            return 0
        regressed = {
            str(item).strip()
            for item in contract.get("regressed_objectives", [])
            if str(item).strip()
        }
        stalled = {
            str(item).strip()
            for item in contract.get("stalled_objectives", [])
            if str(item).strip()
        }
        for index, objective in enumerate(anchors[:4], start=1):
            if not self._command_matches_software_work_objective(command, objective):
                continue
            bonus = max(2, 8 - (index * 2))
            if objective in regressed:
                bonus += 2
            elif objective in stalled:
                bonus += 1
            if index == 1 and role == "planner":
                bonus += 1
            return bonus
        governance = self.universe_model.simulate_command_governance(state.universe_summary, command)
        action_categories = {
            str(item).strip()
            for item in governance.get("action_categories", [])
            if str(item).strip()
        }
        verification_aligned = bool(governance.get("verification_aligned", False))
        required_paths = {
            str(item).strip().lower()
            for item in contract.get("required_paths", [])
            if str(item).strip()
        }
        normalized_command = str(command).strip().lower()
        if required_paths and any(path in normalized_command for path in required_paths):
            return 2
        if "read_only_discovery" in action_categories or verification_aligned:
            return 0
        drift_pressure = max(0, int(contract.get("drift_pressure", 0) or 0))
        if drift_pressure <= 0:
            return 0
        return -min(8, 2 + drift_pressure)

    def _graph_memory_environment_command_score(self, state: AgentState, command: str) -> int:
        graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
        universe_summary = state.universe_summary if isinstance(state.universe_summary, dict) else {}
        if not graph_summary or not universe_summary:
            return 0
        governance = self.universe_model.simulate_command_governance(universe_summary, command)
        action_categories = {
            str(item).strip()
            for item in governance.get("action_categories", [])
            if str(item).strip()
        }
        risk_flags = {
            str(item).strip()
            for item in governance.get("risk_flags", [])
            if str(item).strip()
        }
        verification_aligned = bool(governance.get("verification_aligned", False))
        novelty = self._historical_environment_novelty(graph_summary, universe_summary)
        alignment_failures = self._graph_environment_alignment_failures(graph_summary)
        score = 0

        if novelty > 0:
            if "read_only_discovery" in action_categories:
                score += novelty * 2
            if verification_aligned:
                score += novelty
            score -= novelty * sum(
                1
                for label in risk_flags
                if label
                in {
                    "destructive_mutation",
                    "git_mutation",
                    "network_fetch",
                    "remote_execution",
                    "workspace_scope_escape",
                    "git_write_conflict",
                    "network_access_conflict",
                    "path_scope_conflict",
                }
            )

        if alignment_failures.get("network_access_aligned", 0) > 0 and (
            "network_fetch" in risk_flags or "network_access_conflict" in risk_flags
        ):
            score -= min(6, alignment_failures["network_access_aligned"] + novelty)
        if alignment_failures.get("git_write_aligned", 0) > 0 and (
            "git_mutation" in risk_flags or "git_write_conflict" in risk_flags
        ):
            score -= min(6, alignment_failures["git_write_aligned"] + novelty)
        if alignment_failures.get("workspace_scope_aligned", 0) > 0 and (
            "workspace_scope_escape" in risk_flags or "path_scope_conflict" in risk_flags
        ):
            score -= min(6, alignment_failures["workspace_scope_aligned"] + novelty)

        if (novelty > 0 or any(alignment_failures.values())) and "read_only_discovery" in action_categories:
            score += 1
        if any(alignment_failures.values()) and verification_aligned:
            score += 1
        return score

    def _planner_recovery_stage_command_score(
        self,
        state: AgentState,
        command: str,
        *,
        role: str,
    ) -> int:
        if role not in {"planner", "critic"}:
            return 0
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if str(artifact.get("kind", "")).strip() != "planner_recovery_rewrite":
            return 0
        ranked = artifact.get("ranked_objectives", [])
        if not isinstance(ranked, list):
            ranked = []
        best = 0
        for index, item in enumerate(ranked[:3], start=1):
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective", "")).strip()
            if not objective or not self._command_matches_planner_recovery_objective(command, objective):
                continue
            base = max(1, 7 - (index * 2))
            if index == 1 and role == "planner":
                base += 2
            best = max(best, base)
        return best

    def _software_work_phase_gate_command_score(self, state: AgentState, command: str) -> int:
        gate_state = state.software_work_phase_gate_state()
        if not isinstance(gate_state, dict) or not gate_state:
            return 0
        gate_phase = str(gate_state.get("gate_phase", "")).strip()
        gate_objectives = [
            str(item).strip()
            for item in gate_state.get("gate_objectives", [])
            if str(item).strip()
        ]
        if self._command_matches_any_software_work_objective(command, gate_objectives):
            return 10
        candidate_phase = self._software_work_command_phase(state, command)
        if gate_phase and candidate_phase and _software_work_phase_rank(candidate_phase) > _software_work_phase_rank(gate_phase):
            return -12
        return -4 if gate_objectives else 0

    def _command_matches_any_software_work_objective(self, command: str, objectives: list[str]) -> bool:
        return any(
            self._command_matches_software_work_objective(command, objective)
            for objective in objectives
        )

    @classmethod
    def _command_matches_software_work_objective(cls, command: str, objective: str) -> bool:
        normalized_objective = str(objective).strip().lower()
        if not normalized_objective:
            return False
        if cls._command_matches_planner_recovery_objective(command, normalized_objective):
            return True
        normalized_command = str(command).strip().lower()
        for prefix in (
            "apply planned edit ",
            "complete implementation for ",
            "revise implementation for ",
            "materialize expected artifact ",
            "remove forbidden artifact ",
            "preserve required artifact ",
        ):
            if normalized_objective.startswith(prefix):
                target = normalized_objective.removeprefix(prefix).strip()
                return bool(target) and target in normalized_command
        return False

    @classmethod
    def _software_work_command_phase(cls, state: AgentState, command: str) -> str:
        normalized_command = str(command).strip().lower()
        if not normalized_command:
            return ""
        for objective in state.software_work_plan_update()[:6]:
            if cls._command_matches_software_work_objective(normalized_command, objective):
                return _software_work_objective_phase(objective)
        if any(token in normalized_command for token in ("pytest", "unittest", "nose", "tox", "smoke", "test")):
            return "test"
        if any(token in normalized_command for token in ("git merge", "git cherry-pick", "git rebase", "codegen", "generate")):
            return "migration"
        if any(token in normalized_command for token in ("report", "summary", "postmortem", "fix", "repair")):
            return "follow_up_fix"
        return "implementation"

    @staticmethod
    def _command_matches_planner_recovery_objective(command: str, objective: str) -> bool:
        normalized_command = str(command).strip().lower()
        normalized_objective = str(objective).strip().lower()
        if not normalized_command or not normalized_objective:
            return False
        for prefix in (
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
            "accept required branch ",
            "prepare workflow branch ",
            "run workflow test ",
        ):
            if normalized_objective.startswith(prefix):
                target = normalized_objective.removeprefix(prefix).strip()
                if not target:
                    return False
                if prefix == "accept required branch ":
                    return "git merge" in normalized_command and target in normalized_command
                if prefix == "prepare workflow branch ":
                    return target in normalized_command and any(
                        token in normalized_command for token in ("git checkout", "git switch", "git branch")
                    )
                if prefix == "run workflow test ":
                    objective_tokens = {token for token in target.split() if len(token) > 2}
                    return (
                        ("test" in normalized_command or "pytest" in normalized_command)
                        and bool(objective_tokens.intersection(set(normalized_command.split())))
                    )
                return target in normalized_command
        return any(token in normalized_command for token in normalized_objective.split() if len(token) > 3)

    @staticmethod
    def _failed_command_attempts(state: AgentState, command: str) -> int:
        canonical = _canonicalize_command(command)
        if not canonical:
            return 0
        return sum(
            1
            for step in state.history
            if not step.verification.get("passed", False)
            and canonical == _canonicalize_command(str(step.content))
        )

    @classmethod
    def _recovery_loop_penalty(cls, state: AgentState, command: str) -> int:
        canonical = _canonicalize_command(command)
        if not canonical:
            return 0
        penalty = 0
        failed_attempts = cls._failed_command_attempts(state, command)
        if failed_attempts:
            penalty += failed_attempts * 3
        if canonical == _canonicalize_command(str(state.last_action_signature).partition(":")[2]):
            penalty += max(1, int(state.repeated_action_count) - 1) * 2
        if state.consecutive_no_progress_steps > 0 and canonical in state.all_executed_command_signatures():
            penalty += 1
        return penalty

    def _active_subgoal_diagnosis_command_score(
        self,
        state: AgentState,
        command: str,
        *,
        role: str,
        active_subgoal: str,
        diagnosis: dict[str, object],
    ) -> int:
        if role not in {"planner", "critic"} or not diagnosis:
            return 0
        path = str(diagnosis.get("path", "")).strip()
        if not path:
            return 0
        normalized_command = str(command).strip().lower()
        normalized_path = path.lower()
        signals = {
            str(signal).strip()
            for signal in diagnosis.get("signals", [])
            if str(signal).strip()
        }
        touches_path = normalized_path in normalized_command
        score = 0
        if touches_path:
            score += 3
            if signals.intersection({"state_regression", "no_state_progress"}):
                score += 3
            if signals.intersection({"command_failure", "command_timeout", "inference_failure", "retrieval_failure"}):
                score += 2
            score += self._active_subgoal_repair_shape_bonus(
                active_subgoal=active_subgoal,
                command=normalized_command,
                path=normalized_path,
            )
        elif role == "critic" and signals.intersection({"state_regression", "command_failure", "command_timeout"}):
            score -= 2
        failed_commands = state.all_failed_command_signatures()
        canonical = _canonicalize_command(command)
        if role == "critic" and canonical and canonical in failed_commands:
            score -= max(3, self._policy_control_int("critic_repeat_failure_bias") * 2)
        return score

    @staticmethod
    def _active_subgoal_repair_shape_bonus(
        *,
        active_subgoal: str,
        command: str,
        path: str,
    ) -> int:
        del path
        normalized_goal = str(active_subgoal).strip().lower()
        if not normalized_goal:
            return 0
        if normalized_goal.startswith("remove forbidden artifact "):
            if command.startswith("rm ") or command.startswith("unlink ") or " rm " in command:
                return 3
            return 1
        if normalized_goal.startswith("materialize expected artifact "):
            if (
                "printf " in command
                or " > " in command
                or command.startswith("touch ")
                or command.startswith("cp ")
                or command.startswith("mkdir -p ")
            ):
                return 3
            return 1
        if normalized_goal.startswith("preserve required artifact "):
            if (
                command.startswith("cat ")
                or command.startswith("sed -n ")
                or command.startswith("git diff ")
                or command.startswith("diff ")
                or " pytest" in command
                or command.startswith("pytest ")
            ):
                return 2
        return 0

    def _tolbert_primary_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        runtime_policy = self._tolbert_runtime_policy()
        screened_retrieval_guidance = self._screened_tolbert_retrieval_guidance(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
        )
        candidates = decode_action_generation_candidates(
            state=state,
            world_model=self.world_model,
            skill_library=self.skill_library,
            top_skill=top_skill,
            retrieval_guidance=screened_retrieval_guidance,
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
            retrieval_guidance=screened_retrieval_guidance,
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
        screened_retrieval_guidance = self._screened_tolbert_retrieval_guidance(
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
        )
        candidates = self._tolbert_ranked_candidates(
            state,
            top_skill=top_skill,
            retrieval_guidance=screened_retrieval_guidance,
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
            "hybrid_recovery_stage_alignment": float(
                candidate.get("hybrid_recovery_stage_alignment", 0.0) or 0.0
            ),
            "hybrid_recovery_stage_rank": int(candidate.get("hybrid_recovery_stage_rank", -1) or -1),
            "hybrid_recovery_stage_objective": str(
                candidate.get("hybrid_recovery_stage_objective", "")
            ).strip(),
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
        for item in self._trusted_retrieval_carryover_candidates(state, blocked_commands=blocked):
            command = str(item.get("command", "")).strip()
            normalized = _canonicalize_command(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "command": command,
                    "reason": "trusted retrieval carryover",
                    "span_id": str(item.get("span_id", "")).strip(),
                    "retrieval_influenced": True,
                    "retrieval_ranked_skill": False,
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
            command = str(candidate["command"])
            candidate["score"] = (
                self._command_control_score(state, command)
                + self._trusted_retrieval_carryover_match_bonus(state, command)
            )
        return sorted(candidates, key=lambda item: (-int(item["score"]), str(item["command"])))

    def _best_deterministic_fallback_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        candidates = self._tolbert_ranked_candidates(
            state,
            top_skill=top_skill if self._skill_is_safe_for_task(state, top_skill) else None,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
        )
        for candidate in candidates:
            command = str(candidate.get("command", "")).strip()
            if not command:
                continue
            if int(candidate.get("score", 0)) <= 0:
                continue
            if not self._first_step_command_covers_required_artifacts(state, command):
                continue
            return ActionDecision(
                thought="Use deterministic fallback after inference failure.",
                action=CODE_EXECUTE,
                content=command,
                done=False,
                selected_retrieval_span_id=str(candidate.get("span_id", "")).strip() or None,
                retrieval_influenced=bool(candidate.get("retrieval_influenced", False)),
                retrieval_ranked_skill=bool(candidate.get("retrieval_ranked_skill", False)),
                decision_source="deterministic_fallback",
            )
        return None

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
        task_benchmark_family = str(state.task.metadata.get("benchmark_family", "")).strip()
        task_difficulty = str(
            state.task.metadata.get("difficulty", state.task.metadata.get("task_difficulty", ""))
        ).strip()
        task_horizon = str(state.world_model_summary.get("horizon", "")).strip()
        if not task_difficulty and task_horizon:
            task_difficulty = task_horizon
        is_long_horizon = task_difficulty == "long_horizon" or task_horizon == "long_horizon"
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
        base_repeat_penalty = self._transition_model_control_int("repeat_command_penalty", 4)
        base_progress_bonus = self._transition_model_control_int("progress_command_bonus", 2)
        long_horizon_repeat_penalty = self._transition_model_control_int("long_horizon_repeat_command_penalty", 1)
        long_horizon_progress_bonus = self._transition_model_control_int("long_horizon_progress_command_bonus", 1)
        last_command = ""
        if state.history:
            last_command = _canonicalize_command(str(state.history[-1].content))
        last_pattern = transition_model_command_pattern(last_command) if last_command else ""
        cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
        for signature in self._transition_model_signatures():
            signature_command = _canonicalize_command(str(signature.get("command", "")))
            if not signature_command:
                continue
            signature_family = str(signature.get("benchmark_family", "")).strip()
            if signature_family and task_benchmark_family and signature_family != task_benchmark_family:
                continue
            signature_difficulty = str(signature.get("difficulty", "")).strip()
            if signature_difficulty and task_difficulty and signature_difficulty != task_difficulty:
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
            repeat_penalty = base_repeat_penalty
            progress_bonus = base_progress_bonus
            if is_long_horizon:
                repeat_penalty += long_horizon_repeat_penalty
                progress_bonus += long_horizon_progress_bonus
            if normalized == signature_command:
                penalty = repeat_penalty + min(3, support - 1)
                if signal == "state_regression":
                    penalty += self._transition_model_control_int("regressed_path_command_penalty", 3)
                score -= penalty
                continue
            if normalized_pattern == signature_pattern:
                penalty = max(1, repeat_penalty - 2) + min(
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
            latest_repeat_penalty = base_repeat_penalty + (long_horizon_repeat_penalty if is_long_horizon else 0)
            latest_progress_bonus = base_progress_bonus + (long_horizon_progress_bonus if is_long_horizon else 0)
            if last_command and normalized == last_command:
                score -= latest_repeat_penalty
            elif last_pattern and normalized_pattern == last_pattern:
                score -= max(1, latest_repeat_penalty // 2)
            elif normalized != last_command:
                score += latest_progress_bonus
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

    def _subgoal_alignment_score(self, state: AgentState, command: str) -> int:
        active_subgoal = str(state.active_subgoal or "").strip().lower()
        normalized_command = str(command).strip().lower()
        score = 0
        if active_subgoal:
            for token in active_subgoal.split():
                cleaned = token.strip(".,:;()[]{}\"'")
                if len(cleaned) < 3:
                    continue
                if cleaned in normalized_command:
                    score += 2
        diagnosis = state.active_subgoal_diagnosis()
        path = str(diagnosis.get("path", "")).strip().lower()
        if path and path in normalized_command:
            score += 4
            signals = {
                str(signal).strip()
                for signal in diagnosis.get("signals", [])
                if str(signal).strip()
            }
            if signals.intersection({"state_regression", "command_failure", "command_timeout", "no_state_progress"}):
                score += 2
        return score

    @staticmethod
    def _required_first_step_artifacts(state: AgentState) -> list[str]:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        if bool(metadata.get("synthetic_worker", False)):
            verifier = metadata.get("semantic_verifier", {})
            verifier = dict(verifier) if isinstance(verifier, dict) else {}
            workflow_guard = metadata.get("workflow_guard", {})
            workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
            prioritized = [
                str(path).strip()
                for path in verifier.get("expected_changed_paths", [])
                if str(path).strip()
            ]
            if not prioritized:
                prioritized = [
                    str(path).strip()
                    for path in workflow_guard.get("claimed_paths", [])
                    if str(path).strip()
                ]
            if prioritized:
                return prioritized
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
        retrieval_overrides = self._retrieval_overrides()
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
                normalized["fallback_to_vllm_on_low_confidence"] = True
            normalized["primary_min_command_score"] = max(
                int(normalized.get("primary_min_command_score", 2) or 2),
                direct_command_min_score,
            )
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
                command = LLMDecisionPolicy._avoidance_note_command(note)
                if command:
                    blocked.append(command)
        blocked.extend(sorted(state.all_failed_command_signatures()))
        return blocked

    @staticmethod
    def _avoidance_note_command(note: object) -> str | None:
        text = str(note).strip()
        prefix = "avoid repeating "
        if not text.startswith(prefix):
            return None
        literal = text[len(prefix) :].split(" when ", 1)[0].strip()
        if not literal:
            return None
        try:
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            match = re.match(r"""(['"])(.*)\1$""", literal)
            if match is None:
                return None
            value = match.group(2)
        return str(value).strip() or None

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
        for key in (
            "retrieval_backed_successes",
            "retrieval_influenced_successes",
            "trusted_retrieval_successes",
        ):
            if key in graph_summary:
                summary[key] = graph_summary[key]
        observed_modes = graph_summary.get("observed_environment_modes", {})
        if isinstance(observed_modes, dict):
            compact_modes: dict[str, object] = {}
            for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
                value = self._dominant_graph_environment_mode(graph_summary, field)
                if value:
                    compact_modes[field] = value
            if compact_modes:
                summary["observed_environment_modes"] = compact_modes
        alignment_failures = graph_summary.get("environment_alignment_failures", {})
        if isinstance(alignment_failures, dict) and alignment_failures:
            summary["environment_alignment_failures"] = {
                str(key): int(value)
                for key, value in list(
                    sorted(
                        (
                            (str(key).strip(), self._safe_int(value, 0))
                            for key, value in alignment_failures.items()
                            if str(key).strip()
                        ),
                        key=lambda item: (-item[1], item[0]),
                    )
                )[:3]
                if value > 0
            }
        trusted_commands = graph_summary.get("trusted_retrieval_command_counts", {})
        if isinstance(trusted_commands, dict):
            summary["trusted_retrieval_command_counts"] = dict(list(trusted_commands.items())[:4])
        trusted_procedures = graph_summary.get("trusted_retrieval_procedures", {})
        if isinstance(trusted_procedures, list):
            summary["trusted_retrieval_procedures"] = [
                {
                    "commands": [str(value) for value in dict(item).get("commands", [])[:4] if str(value).strip()],
                    "count": self._safe_int(dict(item).get("count", 0), 0),
                }
                for item in trusted_procedures[:2]
                if isinstance(item, dict)
            ]
        return summary

    @classmethod
    def _historical_environment_novelty(
        cls,
        graph_summary: dict[str, object],
        universe_summary: dict[str, object],
    ) -> int:
        snapshot = universe_summary.get("environment_snapshot", {})
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        novelty = 0
        for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
            current = str(snapshot.get(field, "")).strip().lower()
            dominant = cls._dominant_graph_environment_mode(graph_summary, field)
            if current and dominant and current != dominant:
                novelty += 1
        return novelty

    @staticmethod
    def _graph_environment_alignment_failures(graph_summary: dict[str, object]) -> dict[str, int]:
        failures = graph_summary.get("environment_alignment_failures", {})
        if not isinstance(failures, dict):
            return {}
        return {
            str(key).strip(): value
            for key, value in (
                (str(key).strip(), LLMDecisionPolicy._safe_int(raw_value, 0))
                for key, raw_value in failures.items()
            )
            if key and value > 0
        }

    @staticmethod
    def _dominant_graph_environment_mode(graph_summary: dict[str, object], field: str) -> str:
        observed_modes = graph_summary.get("observed_environment_modes", {})
        if not isinstance(observed_modes, dict):
            return ""
        values = observed_modes.get(field, {})
        if not isinstance(values, dict):
            return ""
        ranked = sorted(
            (
                (str(mode).strip().lower(), LLMDecisionPolicy._safe_int(count, 0))
                for mode, count in values.items()
                if str(mode).strip()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[0][0] if ranked and ranked[0][1] > 0 else ""

    @staticmethod
    def _trusted_retrieval_carryover_active(state: AgentState) -> bool:
        if not state.history:
            return False
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip().lower()
        horizon = str(state.world_model_summary.get("horizon", "")).strip().lower()
        if difficulty == "long_horizon" or horizon == "long_horizon":
            return True
        return bool(state.active_subgoal or state.subgoal_diagnoses or state.consecutive_failures > 0)

    def _trusted_retrieval_carryover_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        trusted_commands = state.graph_summary.get("trusted_retrieval_command_counts", {})
        if not isinstance(trusted_commands, dict):
            return []
        candidates: list[tuple[tuple[int, int, str], dict[str, object]]] = []
        seen: set[str] = set()
        for command, raw_count in trusted_commands.items():
            command_text = _normalize_command_for_workspace(str(command).strip(), state.task.workspace_subdir)
            canonical = _canonicalize_command(command_text)
            if not canonical or canonical in blocked_commands or canonical in seen:
                continue
            count = self._safe_int(raw_count, 0)
            if count <= 0:
                continue
            if not self._command_matches_current_repair_surface(state, command_text):
                continue
            control_score = self._command_control_score(state, command_text)
            total_score = control_score + self._trusted_retrieval_carryover_match_bonus(state, command_text)
            total_score += min(4, count * 2)
            candidates.append(
                (
                    (-total_score, -count, command_text),
                    {
                        "command": command_text,
                        "count": count,
                        "control_score": control_score,
                        "total_score": total_score,
                        "span_id": self._trusted_retrieval_carryover_span_id(command_text),
                        "generated": False,
                    },
                )
            )
            seen.add(canonical)
        generated_materialize = self._trusted_retrieval_carryover_materialize_candidate(
            state,
            trusted_commands=trusted_commands,
            blocked_commands=blocked_commands,
        )
        if generated_materialize is not None:
            canonical = _canonicalize_command(str(generated_materialize.get("command", "")).strip())
            if canonical and canonical not in seen and canonical not in blocked_commands:
                candidates.append(
                    (
                        (
                            -int(generated_materialize.get("total_score", 0) or 0),
                            -int(generated_materialize.get("count", 0) or 0),
                            str(generated_materialize.get("command", "")),
                        ),
                        generated_materialize,
                    )
                )
        for procedure_candidate in self._trusted_retrieval_carryover_procedure_candidates(
            state,
            blocked_commands=blocked_commands,
        ):
            canonical = _canonicalize_command(str(procedure_candidate.get("command", "")).strip())
            if canonical and canonical not in seen and canonical not in blocked_commands:
                candidates.append(
                    (
                        (
                            -int(procedure_candidate.get("total_score", 0) or 0),
                            -int(procedure_candidate.get("count", 0) or 0),
                            str(procedure_candidate.get("command", "")),
                        ),
                        procedure_candidate,
                    )
                )
                seen.add(canonical)
        candidates.sort(key=lambda item: item[0])
        return [payload for _, payload in candidates]

    def _trusted_retrieval_carryover_match_bonus(self, state: AgentState, command: str) -> int:
        trusted_commands = state.graph_summary.get("trusted_retrieval_command_counts", {})
        if not isinstance(trusted_commands, dict):
            return 0
        canonical = _canonicalize_command(command)
        if not canonical:
            return 0
        for candidate, raw_count in trusted_commands.items():
            if canonical != _canonicalize_command(str(candidate)):
                continue
            return min(4, self._safe_int(raw_count, 0) * 2)
        return 0

    def _trusted_retrieval_carryover_materialize_candidate(
        self,
        state: AgentState,
        *,
        trusted_commands: dict[str, object],
        blocked_commands: set[str],
    ) -> dict[str, object] | None:
        target = self._trusted_retrieval_materialize_target(state)
        if target is None:
            return None
        path, target_content = target
        if self._trusted_retrieval_materialize_requires_structured_edit(state, path):
            return None
        source_command = ""
        count = 0
        for command, raw_count in sorted(
            trusted_commands.items(),
            key=lambda item: (-self._safe_int(item[1], 0), str(item[0])),
        ):
            normalized_command = _normalize_command_for_workspace(str(command).strip(), state.task.workspace_subdir)
            if not self._trusted_retrieval_supports_materialize_write(normalized_command):
                continue
            candidate_count = self._safe_int(raw_count, 0)
            if candidate_count <= 0:
                continue
            source_command = normalized_command
            count = candidate_count
            break
        if not source_command or count <= 0:
            return None
        command = self._trusted_retrieval_materialize_write_command(path, target_content)
        canonical = _canonicalize_command(command)
        if not canonical or canonical in blocked_commands:
            return None
        if not self._command_matches_current_repair_surface(state, command):
            return None
        control_score = self._command_control_score(state, command)
        total_score = control_score + min(3, count * 2) + 2
        return {
            "command": command,
            "count": count,
            "control_score": control_score,
            "total_score": total_score,
            "span_id": self._trusted_retrieval_materialize_span_id(source_command, path),
            "generated": True,
            "source_command": source_command,
        }

    def _trusted_retrieval_carryover_procedure_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        procedures = state.graph_summary.get("trusted_retrieval_procedures", {})
        if not isinstance(procedures, list):
            return []
        recent_commands = self._recent_successful_command_suffix(state)
        if not recent_commands:
            return []
        candidates: list[tuple[tuple[int, int, str], dict[str, object]]] = []
        for item in procedures[:4]:
            if not isinstance(item, dict):
                continue
            commands = [
                _normalize_command_for_workspace(str(value).strip(), state.task.workspace_subdir)
                for value in item.get("commands", [])
                if str(value).strip()
            ]
            if len(commands) < 2:
                continue
            matched_prefix = self._trusted_retrieval_procedure_prefix_match(recent_commands, commands)
            if matched_prefix <= 0 or matched_prefix >= len(commands):
                continue
            next_command = commands[matched_prefix]
            canonical = _canonicalize_command(next_command)
            if not canonical or canonical in blocked_commands:
                continue
            control_score = self._command_control_score(state, next_command)
            if control_score <= 0 and self._verification_or_report_command(next_command):
                control_score = 1
            if control_score <= 0:
                continue
            if not (
                self._command_matches_current_repair_surface(state, next_command)
                or self._verification_or_report_command(next_command)
            ):
                continue
            count = self._safe_int(item.get("count", 0), 0)
            total_score = control_score + min(6, matched_prefix * 2) + min(4, count * 2)
            candidates.append(
                (
                    (-total_score, -count, next_command),
                    {
                        "command": next_command,
                        "count": count,
                        "control_score": control_score,
                        "total_score": total_score,
                        "span_id": self._trusted_retrieval_procedure_span_id(commands),
                        "generated": False,
                        "procedure": True,
                        "matched_prefix": matched_prefix,
                    },
                )
            )
        candidates.sort(key=lambda item: item[0])
        return [payload for _, payload in candidates]

    def _trusted_retrieval_materialize_target(self, state: AgentState) -> tuple[str, str] | None:
        path = ""
        active_subgoal = str(state.active_subgoal or "").strip()
        if active_subgoal.lower().startswith("materialize expected artifact "):
            path = active_subgoal[len("materialize expected artifact ") :].strip()
        if not path:
            diagnosis = state.active_subgoal_diagnosis()
            diagnosis_path = str(diagnosis.get("path", "")).strip()
            if diagnosis_path and diagnosis_path in state.task.expected_file_contents:
                path = diagnosis_path
        if not path or path not in state.task.expected_file_contents:
            return None
        unsatisfied = {
            str(item).strip()
            for item in state.world_model_summary.get("unsatisfied_expected_contents", [])
            if str(item).strip()
        }
        missing = {
            str(item).strip()
            for item in state.world_model_summary.get("missing_expected_artifacts", [])
            if str(item).strip()
        }
        if (unsatisfied or missing) and path not in unsatisfied and path not in missing:
            return None
        return path, str(state.task.expected_file_contents.get(path, ""))

    @staticmethod
    def _trusted_retrieval_materialize_requires_structured_edit(state: AgentState, path: str) -> bool:
        previews = state.world_model_summary.get("workspace_file_previews", {})
        return isinstance(previews, dict) and isinstance(previews.get(path), dict)

    @staticmethod
    def _trusted_retrieval_supports_materialize_write(command: str) -> bool:
        normalized = str(command).strip()
        return "printf " in normalized and " > " in normalized and " >> " not in normalized

    @staticmethod
    def _trusted_retrieval_materialize_write_command(path: str, target_content: str) -> str:
        commands: list[str] = []
        parent = Path(path).parent
        if str(parent) not in {"", "."}:
            commands.append(f"mkdir -p {shlex.quote(str(parent))}")
        commands.append(f"printf %s {shlex.quote(str(target_content))} > {shlex.quote(path)}")
        return " && ".join(commands)

    @staticmethod
    def _trusted_retrieval_materialize_span_id(source_command: str, path: str) -> str:
        digest = hashlib.sha1(f"{source_command}|{path}".encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:materialize:{digest}"

    def _command_matches_current_repair_surface(self, state: AgentState, command: str) -> bool:
        normalized = str(command).strip().lower()
        if not normalized:
            return False
        diagnosis = state.active_subgoal_diagnosis()
        path = str(diagnosis.get("path", "")).strip().lower()
        if path and path in normalized:
            return True
        if self._subgoal_alignment_score(state, command) > 0:
            return True
        for value in (
            list(state.task.expected_files)
            + list(state.task.expected_file_contents.keys())
            + list(state.task.forbidden_files)
        ):
            candidate_path = str(value).strip().lower()
            if candidate_path and candidate_path in normalized:
                return True
        return False

    @staticmethod
    def _recent_successful_command_suffix(state: AgentState) -> list[str]:
        commands: list[str] = []
        for step in state.history[-6:]:
            if str(getattr(step, "action", "")).strip() != CODE_EXECUTE:
                continue
            command = str(getattr(step, "content", "")).strip()
            if not command:
                continue
            command_result = getattr(step, "command_result", {})
            if isinstance(command_result, dict) and not bool(command_result.get("timed_out", False)):
                try:
                    if int(command_result.get("exit_code", 1)) == 0:
                        commands.append(_canonicalize_command(command))
                        continue
                except (TypeError, ValueError):
                    pass
            verification = getattr(step, "verification", {})
            if isinstance(verification, dict) and bool(verification.get("passed", False)):
                commands.append(_canonicalize_command(command))
        return [command for command in commands if command]

    @staticmethod
    def _trusted_retrieval_procedure_prefix_match(
        recent_commands: list[str],
        procedure_commands: list[str],
    ) -> int:
        canonical_procedure = [_canonicalize_command(command) for command in procedure_commands]
        canonical_procedure = [command for command in canonical_procedure if command]
        if len(canonical_procedure) < 2:
            return 0
        max_prefix = min(len(recent_commands), len(canonical_procedure) - 1)
        for prefix_len in range(max_prefix, 0, -1):
            if recent_commands[-prefix_len:] == canonical_procedure[:prefix_len]:
                return prefix_len
        return 0

    @staticmethod
    def _verification_or_report_command(command: str) -> bool:
        normalized = str(command).strip().lower()
        if not normalized:
            return False
        return any(
            token in normalized
            for token in ("pytest", "unittest", "tox", "nose", " test", "report", "summary", "status")
        )

    @staticmethod
    def _trusted_retrieval_carryover_span_id(command: str) -> str:
        digest = hashlib.sha1(str(command).encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:{digest}"

    @staticmethod
    def _trusted_retrieval_procedure_span_id(commands: list[str]) -> str:
        digest = hashlib.sha1("||".join(commands).encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:procedure:{digest}"

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

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
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "intact_preserved_artifacts",
            "workflow_report_paths",
            "workflow_generated_paths",
            "workflow_required_merges",
            "workflow_branch_targets",
            "workflow_required_tests",
            "updated_workflow_paths",
            "updated_generated_paths",
            "updated_report_paths",
        ):
            values = world_model_summary.get(key, [])
            if isinstance(values, list) and values:
                summary[key] = [self._truncate_text(str(item), limit=80) for item in values[:6]]
        previews = world_model_summary.get("workspace_file_previews", {})
        if isinstance(previews, dict) and previews:
            summary["workspace_file_previews"] = {
                str(path): self._compact_workspace_preview(preview)
                for path, preview in list(previews.items())[:4]
                if isinstance(preview, dict) and str(path).strip()
            }
        return summary

    def _compact_workspace_preview(self, preview: dict[str, object]) -> dict[str, object]:
        windows = preview.get("edit_windows", [])
        if isinstance(windows, list) and windows:
            compact_windows = [
                {
                    "content": self._truncate_preview_content(window.get("content", "")),
                    "truncated": bool(window.get("truncated", False)),
                    "line_start": int(window.get("line_start", 1) or 1),
                    "line_end": int(window.get("line_end", 1) or 1),
                }
                for window in windows[:3]
                if isinstance(window, dict)
            ]
            if compact_windows:
                compact = dict(compact_windows[0])
                compact["edit_windows"] = compact_windows
                return compact
        return {
            "content": self._truncate_preview_content(preview.get("content", "")),
            "truncated": bool(preview.get("truncated", False)),
        }

    def _truncate_preview_content(self, value: object) -> str:
        content = str(value)
        return content[:157] + "..." if len(content) > 160 else content

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
        deduped = dedupe_prompt_adjustments([proposal for proposal in proposals if isinstance(proposal, dict)])
        return deduped[:3]

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
