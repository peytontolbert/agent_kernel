from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .actions import ALLOWED_ACTIONS, CODE_EXECUTE
from .config import KernelConfig
from .extensions.context_budget import ContextBudgeter
from .extensions.policy_command_utils import (
    canonicalize_command as _canonicalize_command,
    normalize_command_for_workspace as _normalize_command_for_workspace,
)
from .extensions import (
    policy_decision_support,
    policy_context_support,
    policy_prompt_support,
    policy_scoring_support,
    policy_summary_support,
    policy_tolbert_support,
    policy_workflow_support,
)
from .extensions.policy_runtime_support import PolicyRuntimeSupport, SkillLibrary
from .extensions.policy_workflow_adapter import PolicyWorkflowAdapter
from .llm import LLMClient, coerce_action_decision
from .extensions.runtime_modeling_adapter import (
    choose_tolbert_route,
    score_hybrid_candidates,
)
from .schemas import ActionDecision
from .state import AgentState
from .universe_model import UniverseModel
from .world_model import WorldModel


class Policy:
    def decide(self, state: AgentState) -> ActionDecision:
        raise NotImplementedError

    def set_decision_progress_callback(self, callback) -> None:
        del callback

    def fallback_decision(
        self,
        state: AgentState,
        *,
        failure_origin: str = "",
        error_text: str = "",
    ) -> ActionDecision | None:
        del state, failure_origin, error_text
        return None


class ContextCompilationError(RuntimeError):
    """Raised when the context provider cannot produce a valid context packet."""


class ContextProvider(Protocol):
    def compile(self, state: AgentState) -> object:
        ...

    def close(self) -> None:
        ...


class LLMDecisionPolicy(Policy):
    def __init__(
        self,
        client: LLMClient,
        context_provider: ContextProvider | None = None,
        skill_library: SkillLibrary | None = None,
        config: KernelConfig | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.client = client
        self.context_provider = context_provider
        self.skill_library = skill_library or SkillLibrary([])
        self.config = config or KernelConfig()
        self.context_budgeter = ContextBudgeter(self.config)
        self.universe_model = UniverseModel(config=self.config) if self.config.use_universe_model else None
        self.world_model = WorldModel(config=self.config)
        resolved_repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
        self.runtime_support = PolicyRuntimeSupport(config=self.config, repo_root=resolved_repo_root)
        self.system_prompt = self.runtime_support.prompt_template("system")
        self.decision_prompt = self.runtime_support.prompt_template("decision")
        self.workflow_adapter = PolicyWorkflowAdapter(self)
        self._decision_progress_callback = None

    def close(self) -> None:
        if self.context_provider is None:
            return
        close = getattr(self.context_provider, "close", None)
        if callable(close):
            close()

    def set_decision_progress_callback(self, callback) -> None:
        self._decision_progress_callback = callback

    @staticmethod
    def _is_retryable_tolbert_startup_failure(error_text: str) -> bool:
        normalized = str(error_text).strip().lower()
        if not normalized:
            return False
        return (
            "tolbert service failed to become ready" in normalized
            or "tolbert service exited before startup ready" in normalized
        )

    def fallback_decision(
        self,
        state: AgentState,
        *,
        failure_origin: str = "",
        error_text: str = "",
    ) -> ActionDecision | None:
        if self._require_live_llm_coding_control(state):
            return None
        normalized_failure_origin = str(failure_origin).strip()
        allow_fallback = normalized_failure_origin == "inference_failure"
        allow_progressive_first_step = False
        if (
            not allow_fallback
            and normalized_failure_origin == "retrieval_failure"
            and bool(self.config.use_tolbert_context)
            and self._is_retryable_tolbert_startup_failure(error_text)
        ):
            allow_fallback = True
            allow_progressive_first_step = True
        if not allow_fallback:
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
            allow_partial_first_step=allow_progressive_first_step,
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
        return self._apply_tolbert_shadow(fallback, tolbert_shadow, tolbert_route.mode, state=state)

    def _require_live_llm_coding_control(self, state: AgentState) -> bool:
        del state
        return bool(self.config.asi_coding_require_live_llm)

    def _emit_decision_progress(self, stage: str, **payload: object) -> None:
        if self._decision_progress_callback is None:
            return
        event = {"step_stage": str(stage).strip()}
        event.update(payload)
        self._decision_progress_callback(event)

    def decide(self, state: AgentState) -> ActionDecision:
        require_live_llm_coding_control = self._require_live_llm_coding_control(state)
        state.retrieval_direct_candidates = []
        context_compile_warning: dict[str, object] | None = None
        if not require_live_llm_coding_control:
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
        if not require_live_llm_coding_control:
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
                    error_text = str(exc)
                    if (
                        require_live_llm_coding_control
                        and bool(self.config.use_tolbert_context)
                        and self._is_retryable_tolbert_startup_failure(error_text)
                    ):
                        state.context_packet = None
                        context_compile_warning = {
                            "status": "degraded",
                            "reason": "tolbert_startup_failure",
                            "failure_origin": "retrieval_failure",
                            "error_type": exc.__class__.__name__,
                            "message": self._truncate_text(error_text, limit=240),
                        }
                        self._emit_decision_progress(
                            "context_degraded",
                            failure_origin="retrieval_failure",
                            degrade_reason="tolbert_startup_failure",
                            retryable=True,
                        )
                    else:
                        raise ContextCompilationError(f"context packet compilation failed: {exc}") from exc
                finally:
                    if callable(set_progress_callback):
                        set_progress_callback(None)
                if state.context_packet is not None:
                    self._stamp_context_reuse_signature(state)
                    self._emit_decision_progress("context_ready")
        if self.universe_model is not None and not state.universe_summary:
            self._emit_decision_progress("universe_summary")
            state.universe_summary = self.universe_model.summarize(
                state.task,
                world_model_summary=state.world_model_summary,
            )
        tolbert_mode = self._tolbert_mode()
        source_task_id = str(state.task.metadata.get("source_task", "")).strip()
        retrieval_guidance = self._retrieval_guidance(state)
        self._emit_decision_progress(
            "memory_retrieved",
            retrieval_candidate_count=len(list(retrieval_guidance.get("recommended_commands", []) or [])),
            retrieval_evidence_count=len(list(retrieval_guidance.get("evidence", []) or [])),
        )
        preferred_task_ids = self._preferred_task_ids(state) if self._tolbert_skill_ranking_active(state) else []
        if source_task_id and source_task_id not in preferred_task_ids:
            preferred_task_ids = [source_task_id, *preferred_task_ids]
        blocked_commands = self._blocked_commands(state)
        if not require_live_llm_coding_control:
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
        if not require_live_llm_coding_control and tolbert_route.mode == "primary" and not planner_recovery_rewrite_brief:
            primary_decision = self._tolbert_primary_decision(
                state,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
            )
            if primary_decision is not None:
                return self._apply_tolbert_shadow(primary_decision, tolbert_shadow, tolbert_route.mode, state=state)

        role = self._normalized_role(state)
        if not require_live_llm_coding_control:
            deterministic_decision = self._deterministic_role_decision(
                state,
                role=role,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                blocked_commands=blocked_commands,
                tolbert_mode=tolbert_mode,
                retrieval_has_signal=retrieval_has_signal,
                tolbert_route_mode=tolbert_route.mode,
            )
            if deterministic_decision is not None:
                return self._apply_tolbert_shadow(deterministic_decision, tolbert_shadow, tolbert_route.mode, state=state)

            followup_skill_decision = self._followup_skill_decision(
                state,
                role=role,
                top_skill=top_skill,
                retrieval_guidance=retrieval_guidance,
                tolbert_mode=tolbert_mode,
                retrieval_has_signal=retrieval_has_signal,
            )
            if followup_skill_decision is not None:
                return self._apply_tolbert_shadow(followup_skill_decision, tolbert_shadow, tolbert_route.mode, state=state)

        self._emit_decision_progress("plan_candidates")
        transition_preview = self._transition_preview(state)
        self._emit_decision_progress(
            "transition_simulated",
            preview_candidate_count=len(list(transition_preview.get("candidates", []) or [])),
        )
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
            transition_preview=transition_preview,
            available_skills=state.available_skills,
            prompt_adjustments=self._active_prompt_adjustments(),
            allowed_actions=sorted(ALLOWED_ACTIONS),
            graph_summary=self._compact_graph_summary(state.graph_summary),
            universe_summary=self._compact_universe_summary(state.universe_summary),
            world_model_summary=self._compact_world_model_summary(state.world_model_summary),
            plan=self._compact_plan(state.plan),
            active_subgoal=self._truncate_text(state.active_subgoal),
        )
        if context_compile_warning is not None:
            payload["context_compile_warning"] = dict(context_compile_warning)
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
        raw_decision_source = str(raw.get("decision_source", "")).strip()
        raw_proposal_metadata = raw.get("proposal_metadata", {})
        if not isinstance(raw_proposal_metadata, dict):
            raw_proposal_metadata = {}
        if context_compile_warning is not None:
            raw_proposal_metadata = {
                **raw_proposal_metadata,
                "context_compile_degraded": dict(context_compile_warning),
            }
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
            if self._is_prohibited_null_command(state, content):
                guarded_fallback = None
                if not require_live_llm_coding_control:
                    guarded_fallback = self._best_deterministic_fallback_decision(
                        state,
                        top_skill=top_skill,
                        retrieval_guidance=retrieval_guidance,
                        blocked_commands=blocked_commands,
                    )
                if guarded_fallback is not None and not self._is_prohibited_null_command(
                    state,
                    guarded_fallback.content,
                ):
                    return self._apply_tolbert_shadow(guarded_fallback, tolbert_shadow, tolbert_route.mode, state=state)
                return self._apply_tolbert_shadow(
                    ActionDecision(
                        thought="Reject null failing command proposed by the model.",
                        action="respond",
                        content="Stopping because the model proposed a prohibited null failing command.",
                        done=True,
                        decision_source="llm_guardrail",
                    ),
                    tolbert_shadow,
                    tolbert_route.mode,
                    state=state,
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
            decision_source=raw_decision_source or "llm",
            proposal_metadata=dict(raw_proposal_metadata),
        ), tolbert_shadow, tolbert_route.mode, state=state)

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
        tolbert_route_mode: str,
    ) -> ActionDecision | None:
        return policy_decision_support.deterministic_role_decision(
            self,
            state,
            role=role,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
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
        tolbert_route_mode: str,
    ) -> ActionDecision | None:
        return policy_decision_support.planner_direct_decision(
            self,
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
        )

    def _critic_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
        tolbert_route_mode: str,
    ) -> ActionDecision | None:
        return policy_decision_support.critic_direct_decision(
            self,
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
        )

    def _executor_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        retrieval_guidance: dict[str, list[str]],
        blocked_commands: list[str],
        tolbert_mode: str,
        retrieval_has_signal: bool,
        tolbert_route_mode: str,
    ) -> ActionDecision | None:
        return policy_decision_support.executor_direct_decision(
            self,
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
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
        return policy_decision_support.followup_skill_decision(
            self,
            state,
            role=role,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
        )

    def _adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.adjacent_success_direct_decision(state)

    def _synthetic_edit_plan_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        return self.workflow_adapter.synthetic_edit_plan_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    @staticmethod
    def _synthetic_edit_step_priority_key(
        step: dict[str, object],
        *,
        syntax_motor: dict[str, object] | None,
    ) -> tuple[int, int, int, int]:
        return self.workflow_adapter.synthetic_edit_step_priority_key(
            step,
            syntax_motor=syntax_motor,
        )

    def _shared_repo_integrator_segment_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        return self.workflow_adapter.shared_repo_integrator_segment_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    _shared_repo_unresolved_required_branches = staticmethod(
        PolicyWorkflowAdapter.shared_repo_unresolved_required_branches
    )

    def _shared_repo_integrator_premerge_segment(
        self,
        state: AgentState,
        *,
        segments: list[str],
        required_branches: list[str],
        blocked_commands: list[str],
    ) -> str | None:
        return self.workflow_adapter.shared_repo_integrator_premerge_segment(
            state,
            segments=segments,
            required_branches=required_branches,
            blocked_commands=blocked_commands,
        )

    _shared_repo_required_branch_segment = staticmethod(
        PolicyWorkflowAdapter.shared_repo_required_branch_segment
    )

    _shared_repo_integrator_next_segment = staticmethod(
        PolicyWorkflowAdapter.shared_repo_integrator_next_segment
    )

    _shared_repo_integrator_segment_phase_rank = staticmethod(
        PolicyWorkflowAdapter.shared_repo_integrator_segment_phase_rank
    )

    _shared_repo_integrator_grouped_segments = staticmethod(
        PolicyWorkflowAdapter.shared_repo_integrator_grouped_segments
    )

    _synthetic_edit_plan_direct_active = staticmethod(
        PolicyWorkflowAdapter.synthetic_edit_plan_direct_active
    )

    _render_synthetic_edit_step_command = staticmethod(
        PolicyWorkflowAdapter.render_synthetic_edit_step_command
    )

    def _plan_progress_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        return self.workflow_adapter.plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    def _rank_plan_progress_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> list[tuple[int, int, str]]:
        return self.workflow_adapter.rank_plan_progress_candidates(
            state,
            blocked_commands=blocked_commands,
        )

    def _recovery_contract_exhausted(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        return self.workflow_adapter.recovery_contract_exhausted(
            state,
            blocked_commands=blocked_commands,
        )

    def _pre_context_adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_adjacent_success_direct_decision(state)

    def _pre_context_synthetic_edit_plan_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_synthetic_edit_plan_direct_decision(state)

    def _pre_context_shared_repo_integrator_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_shared_repo_integrator_direct_decision(state)

    def _git_repo_review_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        return self.workflow_adapter.git_repo_review_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    def _pre_context_git_repo_review_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_git_repo_review_direct_decision(state)

    def _pre_context_plan_progress_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_plan_progress_direct_decision(state)

    def _pre_context_trusted_retrieval_carryover_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_trusted_retrieval_carryover_decision(state)

    def _pre_context_recovery_exhaustion_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_recovery_exhaustion_decision(state)

    def _planner_recovery_rewrite_required(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        return self.workflow_adapter.planner_recovery_rewrite_required(
            state,
            blocked_commands=blocked_commands,
        )

    def _planner_recovery_rewrite_brief(self, state: AgentState, *, blocked_commands: list[str]) -> str:
        return self.workflow_adapter.planner_recovery_rewrite_brief(
            state,
            blocked_commands=blocked_commands,
        )

    def _software_work_phase_gate_brief(self, state: AgentState) -> str:
        return self.workflow_adapter.software_work_phase_gate_brief(state)

    def _campaign_contract_brief(self, state: AgentState) -> str:
        return self.workflow_adapter.campaign_contract_brief(state)

    def _apply_pre_context_tolbert_route(self, state: AgentState, decision: ActionDecision) -> ActionDecision:
        route = self._tolbert_route_decision(state)
        return self._apply_tolbert_shadow(decision, {}, route.mode, state=state)

    _reusable_context_packet = policy_context_support.reusable_context_packet

    _stamp_context_reuse_signature = policy_context_support.stamp_context_reuse_signature

    _context_packet_reuse_enabled = staticmethod(policy_context_support.context_packet_reuse_enabled)

    _context_reuse_signature = policy_context_support.context_reuse_signature

    def _shared_repo_integrator_first_segment_safe_without_context(self, state: AgentState, command: str) -> bool:
        return self.workflow_adapter.shared_repo_integrator_first_segment_safe_without_context(
            state,
            command,
        )

    def _git_repo_review_first_step_safe_without_context(self, state: AgentState, command: str) -> bool:
        return self.workflow_adapter.git_repo_review_first_step_safe_without_context(
            state,
            command,
        )

    _skill_is_safe_for_task = policy_context_support.skill_is_safe_for_task

    _task_command_anchors = staticmethod(policy_context_support.task_command_anchors)

    _ranked_skill_summaries = policy_context_support.ranked_skill_summaries

    def _tolbert_direct_decision(
        self,
        state: AgentState,
        *,
        top_skill: dict[str, object] | None,
        recommended_commands: list[dict[str, str]],
        blocked_commands: list[str],
        route_mode: str,
    ) -> ActionDecision | None:
        return policy_tolbert_support.tolbert_direct_decision(
            self,
            state,
            top_skill=top_skill,
            recommended_commands=recommended_commands,
            blocked_commands=blocked_commands,
            route_mode=route_mode,
        )

    _screened_tolbert_retrieval_guidance = policy_tolbert_support.screened_tolbert_retrieval_guidance

    _rank_direct_retrieval_commands = policy_tolbert_support.rank_direct_retrieval_commands

    _can_execute_direct_retrieval_command = policy_tolbert_support.can_execute_direct_retrieval_command

    _trusted_workflow_guidance_bypasses_first_step_confidence = (
        policy_prompt_support.trusted_workflow_guidance_bypasses_first_step_confidence
    )

    @staticmethod
    def _first_step_guarded_command_coverage(
        state: AgentState,
        command: str,
        claimed_paths: list[str],
    ) -> bool:
        return policy_prompt_support.first_step_guarded_command_coverage(
            None,
            state,
            command,
            claimed_paths,
        )

    _command_control_score = policy_scoring_support.command_control_score

    @staticmethod
    def _is_literal_null_failing_command(command: str) -> bool:
        normalized = str(command).strip()
        return normalized in {"false", "/bin/false"}

    def _command_explicitly_required_for_task(self, state: AgentState, command: str) -> bool:
        normalized = str(command).strip()
        if not normalized:
            return False
        if bool(state.task.metadata.get("allow_literal_failure_commands", False)):
            return True
        allowed_commands = [
            *list(state.task.setup_commands),
            *list(state.task.suggested_commands),
        ]
        success_command = str(state.task.success_command).strip()
        if success_command:
            allowed_commands.append(success_command)
        normalized_allowed = {_canonicalize_command(candidate) for candidate in allowed_commands if str(candidate).strip()}
        return _canonicalize_command(normalized) in normalized_allowed

    def _is_prohibited_null_command(self, state: AgentState, command: str) -> bool:
        return self._is_literal_null_failing_command(command) and not self._command_explicitly_required_for_task(
            state,
            command,
        )

    _campaign_contract_command_score = policy_scoring_support.campaign_contract_command_score

    _graph_memory_environment_command_score = policy_scoring_support.graph_memory_environment_command_score

    _graph_memory_failure_signal_command_score = (
        policy_scoring_support.graph_memory_failure_signal_command_score
    )

    _planner_recovery_stage_command_score = policy_workflow_support.planner_recovery_stage_command_score

    def _software_work_phase_gate_command_score(self, state: AgentState, command: str) -> int:
        return policy_workflow_support.software_work_phase_gate_command_score(command, state)

    _command_matches_any_software_work_objective = staticmethod(
        policy_workflow_support.command_matches_any_software_work_objective
    )

    _command_matches_software_work_objective = staticmethod(
        policy_workflow_support.command_matches_software_work_objective
    )

    _software_work_command_phase = staticmethod(policy_workflow_support.software_work_command_phase)

    _command_matches_planner_recovery_objective = staticmethod(
        policy_workflow_support.command_matches_planner_recovery_objective
    )

    _failed_command_attempts = staticmethod(policy_workflow_support.failed_command_attempts)

    _recovery_loop_penalty = staticmethod(policy_workflow_support.recovery_loop_penalty)

    _active_subgoal_diagnosis_command_score = (
        policy_scoring_support.active_subgoal_diagnosis_command_score
    )

    _active_subgoal_repair_shape_bonus = staticmethod(
        policy_workflow_support.active_subgoal_repair_shape_bonus
    )

    _tolbert_primary_decision = policy_tolbert_support.tolbert_primary_decision

    _tolbert_shadow_decision = policy_tolbert_support.tolbert_shadow_decision

    _hybrid_candidate_metadata = staticmethod(policy_tolbert_support.hybrid_candidate_metadata)

    _tolbert_ranked_candidates = policy_tolbert_support.tolbert_ranked_candidates

    _best_deterministic_fallback_decision = policy_decision_support.best_deterministic_fallback_decision

    def _tolbert_route_decision(self, state: AgentState):
        path_confidence = self._path_confidence(state)
        trust_retrieval = self._trust_retrieval(state)
        benchmark_family = str(state.task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        return choose_tolbert_route(
            runtime_policy=self._tolbert_runtime_policy(),
            benchmark_family=benchmark_family,
            path_confidence=path_confidence,
            trust_retrieval=trust_retrieval,
        )

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
        *,
        state: AgentState | None = None,
    ) -> ActionDecision:
        if (
            state is not None
            and decision.action == CODE_EXECUTE
            and self._is_prohibited_null_command(state, decision.content)
        ):
            decision = ActionDecision(
                thought="Reject prohibited null failing command before execution.",
                action="respond",
                content="Stopping because policy selected a prohibited null failing command.",
                done=True,
                decision_source="command_guardrail",
            )
        decision.shadow_decision = dict(shadow_decision)
        if not decision.tolbert_route_mode:
            decision.tolbert_route_mode = route_mode if route_mode in {"shadow", "primary"} else ""
        if not decision.decision_source:
            decision.decision_source = "llm"
        return decision

    _transition_model_command_score = policy_scoring_support.transition_model_command_score

    _first_step_command_covers_required_artifacts = (
        policy_decision_support.first_step_command_covers_required_artifacts
    )

    _subgoal_alignment_score = staticmethod(policy_decision_support.subgoal_alignment_score)

    _required_first_step_artifacts = staticmethod(policy_decision_support.required_first_step_artifacts)

    _transition_preview = policy_decision_support.transition_preview

    def _tolbert_mode(self) -> str:
        mode = str(self.config.tolbert_mode or "full").strip().lower()
        if not self.config.use_tolbert_context:
            return "disabled"
        if mode not in {"full", "path_only", "retrieval_only", "deterministic_command", "skill_ranking", "disabled"}:
            return "full"
        return mode

    def _universe_command_score(self, state: AgentState, command: str) -> int:
        if self.universe_model is None:
            return 0
        return self.universe_model.score_command(state.universe_summary, command)

    def _simulate_command_governance(
        self,
        universe_summary: dict[str, object],
        command: str,
    ) -> dict[str, object]:
        if self.universe_model is None:
            return {}
        return self.universe_model.simulate_command_governance(universe_summary, command)

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
        return self.runtime_support.retrieval_overrides()

    def _policy_controls(self) -> dict[str, object]:
        return self.runtime_support.policy_controls()

    def _tolbert_model_payload(self) -> dict[str, object]:
        return self.runtime_support.tolbert_model_payload()

    def _tolbert_runtime_policy(self) -> dict[str, object]:
        return self.runtime_support.tolbert_runtime_policy()

    def _tolbert_model_surfaces(self) -> dict[str, object]:
        return self.runtime_support.tolbert_model_surfaces()

    def _tolbert_decoder_policy(self) -> dict[str, object]:
        return self.runtime_support.tolbert_decoder_policy()

    def _tolbert_action_generation_policy(self) -> dict[str, object]:
        return self.runtime_support.tolbert_action_generation_policy()

    def _tolbert_rollout_policy(self) -> dict[str, object]:
        return self.runtime_support.tolbert_rollout_policy()

    def _tolbert_hybrid_runtime(self) -> dict[str, object]:
        return self.runtime_support.tolbert_hybrid_runtime()

    def _tolbert_active_decoder_runtime(self) -> dict[str, object]:
        return self.runtime_support.tolbert_active_decoder_runtime()

    def _hybrid_scored_candidates(
        self,
        state: AgentState,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        runtime = self._tolbert_hybrid_runtime()
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        if not manifest_raw:
            self.runtime_support.last_hybrid_runtime_error = "missing bundle_manifest_path"
            raise RuntimeError("hybrid runtime is enabled but bundle_manifest_path is missing")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = self.runtime_support.repo_root / manifest_path
        if not manifest_path.exists():
            self.runtime_support.last_hybrid_runtime_error = f"bundle manifest does not exist: {manifest_path}"
            raise RuntimeError(f"hybrid runtime bundle manifest does not exist: {manifest_path}")
        self.runtime_support.last_hybrid_runtime_error = ""
        try:
            return score_hybrid_candidates(
                state=state,
                candidates=candidates,
                bundle_manifest_path=manifest_path,
                device=str(runtime.get("preferred_device", "cpu")).strip() or "cpu",
                scoring_policy=runtime.get("scoring_policy", {}),
            )
        except Exception as exc:
            self.runtime_support.last_hybrid_runtime_error = str(exc).strip() or exc.__class__.__name__
            raise RuntimeError(
                f"hybrid runtime scoring failed for {manifest_path}: {self.runtime_support.last_hybrid_runtime_error}"
            ) from exc

    def _role_directive_overrides(self) -> dict[str, str]:
        return self.runtime_support.role_directive_overrides()

    def _prompt_policy_payload(self) -> dict[str, object]:
        return self.runtime_support.prompt_policy_payload()

    def _transition_model_controls(self) -> dict[str, object]:
        return self.runtime_support.transition_model_controls()

    def _state_estimation_policy_controls(self) -> dict[str, object]:
        return self.runtime_support.state_estimation_policy_controls()

    def _transition_model_signatures(self) -> list[dict[str, object]]:
        return self.runtime_support.transition_model_signatures()

    def _policy_control_float(self, field: str) -> float:
        return self.runtime_support.policy_control_float(field)

    def _policy_control_int(self, field: str) -> int:
        return self.runtime_support.policy_control_int(field)

    def _transition_model_control_int(self, field: str, default: int) -> int:
        return self.runtime_support.transition_model_control_int(field, default)

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
            decision_source="skill_direct",
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

    _retrieval_guidance = staticmethod(policy_context_support.retrieval_guidance)

    _preferred_task_ids = staticmethod(policy_context_support.preferred_task_ids)

    @staticmethod
    def _blocked_commands(state: AgentState) -> list[str]:
        return policy_context_support.blocked_commands(
            state,
            avoidance_note_command_fn=LLMDecisionPolicy._avoidance_note_command,
        )

    _avoidance_note_command = staticmethod(policy_context_support.avoidance_note_command)

    _has_retrieval_signal = staticmethod(policy_context_support.has_retrieval_signal)

    _retrieval_plan = staticmethod(policy_context_support.retrieval_plan)

    _llm_context_packet = staticmethod(policy_context_support.llm_context_packet)

    _compact_plan = policy_context_support.compact_plan

    _compact_graph_summary = policy_summary_support.compact_graph_summary

    @classmethod
    def _historical_environment_novelty(
        cls,
        graph_summary: dict[str, object],
        universe_summary: dict[str, object],
    ) -> int:
        return policy_summary_support.historical_environment_novelty(
            cls,
            graph_summary,
            universe_summary,
        )

    @staticmethod
    def _graph_environment_alignment_failures(graph_summary: dict[str, object]) -> dict[str, int]:
        return policy_summary_support.graph_environment_alignment_failures(
            LLMDecisionPolicy,
            graph_summary,
        )

    @staticmethod
    def _dominant_graph_environment_mode(graph_summary: dict[str, object], field: str) -> str:
        return policy_summary_support.dominant_graph_environment_mode(
            LLMDecisionPolicy,
            graph_summary,
            field,
        )

    _trusted_retrieval_carryover_active = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_carryover_active
    )

    def _trusted_retrieval_carryover_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        return self.workflow_adapter.trusted_retrieval_carryover_candidates(
            state,
            blocked_commands=blocked_commands,
        )

    def _trusted_retrieval_carryover_match_bonus(self, state: AgentState, command: str) -> int:
        return self.workflow_adapter.trusted_retrieval_carryover_match_bonus(state, command)

    def _trusted_retrieval_carryover_materialize_candidate(
        self,
        state: AgentState,
        *,
        trusted_commands: dict[str, object],
        blocked_commands: set[str],
    ) -> dict[str, object] | None:
        return self.workflow_adapter.trusted_retrieval_carryover_materialize_candidate(
            state,
            trusted_commands=trusted_commands,
            blocked_commands=blocked_commands,
        )

    def _trusted_retrieval_carryover_procedure_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        return self.workflow_adapter.trusted_retrieval_carryover_procedure_candidates(
            state,
            blocked_commands=blocked_commands,
        )

    def _trusted_retrieval_materialize_target(self, state: AgentState) -> tuple[str, str] | None:
        return self.workflow_adapter.trusted_retrieval_materialize_target(state)

    _trusted_retrieval_materialize_requires_structured_edit = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_materialize_requires_structured_edit
    )

    _trusted_retrieval_supports_materialize_write = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_supports_materialize_write
    )

    _trusted_retrieval_materialize_write_command = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_materialize_write_command
    )

    _trusted_retrieval_materialize_span_id = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_materialize_span_id
    )

    def _command_matches_current_repair_surface(self, state: AgentState, command: str) -> bool:
        return self.workflow_adapter.command_matches_current_repair_surface(state, command)

    _recent_successful_command_suffix = staticmethod(
        PolicyWorkflowAdapter.recent_successful_command_suffix
    )

    _trusted_retrieval_procedure_prefix_match = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_procedure_prefix_match
    )

    _verification_or_report_command = staticmethod(PolicyWorkflowAdapter.verification_or_report_command)

    _trusted_retrieval_carryover_span_id = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_carryover_span_id
    )

    _trusted_retrieval_procedure_span_id = staticmethod(
        PolicyWorkflowAdapter.trusted_retrieval_procedure_span_id
    )

    _safe_int = staticmethod(policy_prompt_support.safe_int)

    _compact_world_model_summary = policy_summary_support.compact_world_model_summary

    _compact_workspace_preview = policy_summary_support.compact_workspace_preview

    _truncate_preview_content = staticmethod(policy_summary_support.truncate_preview_content)

    _compact_universe_summary = policy_summary_support.compact_universe_summary

    _truncate_text = policy_prompt_support.truncate_text

    _role_system_prompt = policy_prompt_support.role_system_prompt

    _role_decision_prompt = policy_prompt_support.role_decision_prompt

    def _active_prompt_adjustments(self) -> list[dict[str, object]]:
        return self.runtime_support.active_prompt_adjustments()

    _role_directive = policy_prompt_support.role_directive

    _normalized_role = staticmethod(policy_prompt_support.normalized_role)
