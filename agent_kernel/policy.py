from __future__ import annotations

import ast
from dataclasses import asdict
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Protocol

from .actions import ALLOWED_ACTIONS, CODE_EXECUTE
from .config import KernelConfig
from .extensions.context_budget import ContextBudgeter
from .extensions.policy_command_utils import (
    canonicalize_command as _canonicalize_command,
    normalize_command_for_workspace as _normalize_command_for_workspace,
)
from .extensions import (
    artifact_repair_contracts,
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
from .verifier import (
    _is_python_test_path,
    _python_executable_ast_changed,
    _python_init_generator_names,
    _python_init_return_value_names,
    _python_local_load_before_assignment_names,
    _unused_new_python_parameters,
)
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
    def _is_retryable_decision_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        module_name = exc.__class__.__module__.lower()
        return (
            isinstance(exc, (TimeoutError, OSError))
            or "vllm" in message
            or "ollama" in message
            or "connection refused" in message
            or "timed out" in message
            or "llm request" in message
            or "parseable json" in message
            or "invalid json" in message
            or "json decision" in message
            or "vllm" in module_name
            or "ollama" in module_name
        )

    def _create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, object],
    ) -> dict[str, object]:
        attempts = 2
        for attempt in range(attempts):
            try:
                return self.client.create_decision(
                    system_prompt=system_prompt,
                    decision_prompt=decision_prompt,
                    state_payload=state_payload,
                )
            except Exception as exc:
                if attempt + 1 >= attempts or not self._is_retryable_decision_exception(exc):
                    raise
        raise RuntimeError("unreachable decision retry state")

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
        normalized_failure_origin = str(failure_origin).strip()
        if normalized_failure_origin == "inference_failure":
            artifact_fallback = self._artifact_inference_failure_source_context_fallback(state)
            if artifact_fallback is not None:
                return artifact_fallback
        if self._require_live_llm_coding_control(state):
            return None
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
            pre_context_workspace_contract = self._pre_context_workspace_contract_direct_decision(state)
            if pre_context_workspace_contract is not None:
                return self._apply_pre_context_tolbert_route(state, pre_context_workspace_contract)
            pre_context_shared_repo_worker = self._pre_context_shared_repo_worker_direct_decision(state)
            if pre_context_shared_repo_worker is not None:
                return self._apply_pre_context_tolbert_route(state, pre_context_shared_repo_worker)
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
                    retryable_tolbert_startup_failure = (
                        bool(self.config.use_tolbert_context)
                        and self._is_retryable_tolbert_startup_failure(error_text)
                    )
                    if retryable_tolbert_startup_failure:
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
                        if not require_live_llm_coding_control:
                            fallback = self.fallback_decision(
                                state,
                                failure_origin="retrieval_failure",
                                error_text=error_text,
                            )
                            if fallback is not None:
                                fallback.proposal_metadata = dict(getattr(fallback, "proposal_metadata", {}) or {})
                                fallback.proposal_metadata["context_compile_degraded"] = dict(
                                    context_compile_warning
                                )
                                return fallback
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

        artifact_source_lines_followup = self._artifact_source_lines_followup_direct_decision(state)
        if artifact_source_lines_followup is not None:
            return self._apply_tolbert_shadow(
                artifact_source_lines_followup,
                tolbert_shadow,
                tolbert_route.mode,
                state=state,
            )

        artifact_suggested_command = self._artifact_suggested_builder_command_direct_decision(state)
        if artifact_suggested_command is not None:
            return self._apply_tolbert_shadow(
                artifact_suggested_command,
                tolbert_shadow,
                tolbert_route.mode,
                state=state,
            )

        artifact_repair = self._artifact_builder_from_diff_direct_decision(state)
        if artifact_repair is not None:
            return self._apply_tolbert_shadow(artifact_repair, tolbert_shadow, tolbert_route.mode, state=state)

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
        exact_verifier_repair_brief = self._exact_verifier_repair_brief(state)
        if exact_verifier_repair_brief:
            payload["exact_verifier_repair_brief"] = exact_verifier_repair_brief
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
        if exact_verifier_repair_brief:
            decision_prompt = (
                f"{decision_prompt}\n"
                "Exact verifier repair directive: "
                f"{exact_verifier_repair_brief}"
            )
        self._emit_decision_progress("llm_request")
        raw = self._create_decision(
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
        if normalized["action"] != CODE_EXECUTE:
            artifact_repair_continue = self._artifact_repair_continue_decision(
                state=state,
                system_prompt=system_prompt,
                decision_prompt=decision_prompt,
                payload=payload,
                proposed_content=content,
                context_compile_warning=context_compile_warning,
            )
            if artifact_repair_continue is not None:
                return self._apply_tolbert_shadow(
                    artifact_repair_continue,
                    tolbert_shadow,
                    tolbert_route.mode,
                    state=state,
                )
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
            artifact_materialization_retry = self._artifact_materialization_retry_decision(
                state=state,
                system_prompt=system_prompt,
                decision_prompt=decision_prompt,
                payload=payload,
                proposed_content=content,
                context_compile_warning=context_compile_warning,
            )
            if artifact_materialization_retry is not None:
                return self._apply_tolbert_shadow(
                    artifact_materialization_retry,
                    tolbert_shadow,
                    tolbert_route.mode,
                    state=state,
                )
            artifact_identifier_retry = self._artifact_identifier_retry_decision(
                state=state,
                system_prompt=system_prompt,
                decision_prompt=decision_prompt,
                payload=payload,
                proposed_content=content,
                context_compile_warning=context_compile_warning,
            )
            if artifact_identifier_retry is not None:
                return self._apply_tolbert_shadow(
                    artifact_identifier_retry,
                    tolbert_shadow,
                    tolbert_route.mode,
                    state=state,
                )
            artifact_semantic_retry = self._artifact_semantic_retry_decision(
                state=state,
                system_prompt=system_prompt,
                decision_prompt=decision_prompt,
                payload=payload,
                proposed_content=content,
                context_compile_warning=context_compile_warning,
            )
            if artifact_semantic_retry is not None:
                return self._apply_tolbert_shadow(
                    artifact_semantic_retry,
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

    def _artifact_materialization_retry_decision(
        self,
        *,
        state: AgentState,
        system_prompt: str,
        decision_prompt: str,
        payload: dict[str, object],
        proposed_content: str,
        context_compile_warning: dict[str, object] | None,
    ) -> ActionDecision | None:
        if not self._artifact_repair_context(state):
            return None
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        materializes = self._artifact_materializes(state, proposed_content)
        guard_rejection_reason = self._artifact_builder_guard_rejection_reason(state, proposed_content)
        if materializes and not guard_rejection_reason:
            return None
        source_context_available = self._artifact_source_context_available(state)
        if self._artifact_source_inspection_allowed_after_history(state, proposed_content):
            return None
        if not source_context_available and self._artifact_prior_source_inspection_count(state) < 1:
            return None
        if not self._artifact_non_materializing_repair_command(state, proposed_content):
            return None
        rejected_retry_command = ""
        rejected_retry_reason = guard_rejection_reason or self._artifact_materialization_rejection_reason(
            state,
            proposed_content,
        )
        if rejected_retry_reason == "artifact_path_as_source":
            artifact_repair_context = self._artifact_retry_context_payload(
                state,
                artifact_path=artifact_path,
                builder_command=artifact_repair_contracts.preferred_builder_command(contract),
            )
            remapped_command = self._artifact_remap_builder_source_path_command(
                state,
                proposed_content,
                artifact_path=artifact_path,
                builder_command=artifact_repair_contracts.preferred_builder_command(contract),
                artifact_repair_context=artifact_repair_context,
            )
            if remapped_command:
                return ActionDecision(
                    thought="Remap an artifact builder command that used the output artifact as its source path.",
                    action=CODE_EXECUTE,
                    content=remapped_command,
                    done=False,
                    decision_source="artifact_builder_source_path_remap_direct",
                    proposal_metadata={
                        "artifact_builder_source_path_remap_direct": True,
                        "rejected_command": proposed_content,
                    },
                )
        if rejected_retry_reason == "broad_replacement":
            narrowed_command = self._artifact_narrow_broad_builder_command(
                state,
                proposed_content,
                artifact_path=artifact_path,
            )
            if narrowed_command:
                return ActionDecision(
                    thought="Narrow an over-broad artifact builder command to a focused executable edit.",
                    action=CODE_EXECUTE,
                    content=narrowed_command,
                    done=False,
                    decision_source="artifact_broad_builder_narrow_direct",
                    proposal_metadata={
                        "artifact_broad_builder_narrow_direct": True,
                        "rejected_command": proposed_content,
                    },
                )
        if rejected_retry_reason == "escaped_newline_replacement":
            split_command = self._artifact_split_escaped_newline_builder_command(
                state,
                proposed_content,
                artifact_path=artifact_path,
            )
            if split_command:
                return ActionDecision(
                    thought="Split an escaped-newline artifact builder replacement into explicit replacement lines.",
                    action=CODE_EXECUTE,
                    content=split_command,
                    done=False,
                    decision_source="artifact_escaped_newline_split_direct",
                    proposal_metadata={
                        "artifact_escaped_newline_split_direct": True,
                        "rejected_command": proposed_content,
                    },
                )
            if self._artifact_broad_builder_last_resort_allowed(state, proposed_content):
                return ActionDecision(
                    thought="Use a bounded broad artifact builder command after focused narrowing failed.",
                    action=CODE_EXECUTE,
                    content=proposed_content,
                    done=False,
                    decision_source="artifact_broad_builder_last_resort_direct",
                    proposal_metadata={
                        "artifact_broad_builder_last_resort_direct": True,
                        "rejected_command": proposed_content,
                    },
                )
        if rejected_retry_reason in {"definition_header_removal", "invalid_python_replacement"}:
            signature_command = self._artifact_expand_partial_header_signature_builder_command(
                state,
                proposed_content,
                artifact_path=artifact_path,
            )
            if signature_command:
                return ActionDecision(
                    thought="Expand a partial multiline Python signature edit to the full signature range.",
                    action=CODE_EXECUTE,
                    content=signature_command,
                    done=False,
                    decision_source="artifact_header_signature_expand_direct",
                    proposal_metadata={
                        "artifact_header_signature_expand_direct": True,
                        "rejected_command": proposed_content,
                    },
                )
        artifact_repair_context = self._artifact_retry_context_payload(
            state,
            artifact_path=artifact_path,
            builder_command=artifact_repair_contracts.preferred_builder_command(contract),
        )
        artifact_repair_context_prompt = self._artifact_retry_context_prompt(artifact_repair_context)
        task_context = str(getattr(state.task, "prompt", "")).strip()
        if len(task_context) > 2400:
            task_context = f"{task_context[:2400]}\n...[task prompt truncated for artifact retry]..."
        for retry_attempt in range(1, 6):
            retry_payload = dict(payload)
            active_subgoal = str(state.active_subgoal or "materialize expected artifact").strip()
            retry_payload["subgoal_diagnoses"] = {
                active_subgoal: {
                    "summary": f"Required artifact `{artifact_path}` is missing.",
                    "signals": ["verifier_failure"],
                    "path": artifact_path,
                    "source_role": "artifact_materialization_guard",
                    "repair_instruction": (
                        f"Create or overwrite `{artifact_path}` now with one allowed builder command. "
                        "Source context is already available in artifact_repair_context; do not inspect files."
                    ),
                }
            }
            guard_rejected_command = rejected_retry_command or proposed_content
            retry_payload["artifact_materialization_guard"] = {
                "attempt": retry_attempt,
                "rejected_command": guard_rejected_command,
                "rejected_reason": rejected_retry_reason,
                "reason": self._artifact_materialization_retry_reason_text(
                    rejected_retry_reason,
                    artifact_path=artifact_path,
                ),
                "source_context_available": source_context_available,
                "required_command_shape": artifact_repair_contracts.required_command_shape(contract),
                "artifact_path_is_output_only": artifact_path,
                "forbidden_command_patterns": [
                    f"--path {artifact_path}",
                    "--replace-line 1234",
                    "new_code",
                    "new_code_here",
                    "print(\"test\")",
                    "assert True",
                    "assert False",
                ],
            }
            if rejected_retry_reason == "source_identical_noop":
                source_identical_details = self._artifact_source_identical_operation_details(
                    state,
                    guard_rejected_command,
                )
                if source_identical_details:
                    retry_payload["artifact_materialization_guard"][
                        "source_identical_noop_operations"
                    ] = source_identical_details
            if rejected_retry_reason == "invalid_python_replacement":
                invalid_python_details = self._artifact_invalid_python_replacement_details(
                    state,
                    guard_rejected_command,
                )
                if invalid_python_details:
                    retry_payload["artifact_materialization_guard"][
                        "invalid_python_replacement_operations"
                    ] = invalid_python_details
            if rejected_retry_reason in {
                "line_out_of_source_range",
                "placeholder_replacement",
                "comment_only_replacement",
                "unknown_source_path",
                "artifact_path_as_source",
                "duplicate_adjacent_line",
                "python_ast_noop",
                "unused_signature_parameter",
                "invalid_init_return_value",
                "invalid_init_generator",
                "local_use_before_assignment",
            }:
                guard_details = self._artifact_builder_guard_rejection_details(
                    state,
                    rejected_retry_reason,
                    guard_rejected_command,
                )
                if guard_details:
                    retry_payload["artifact_materialization_guard"][
                        "artifact_guard_rejection_operations"
                    ] = guard_details
            if artifact_repair_context:
                retry_payload["artifact_repair_context"] = dict(artifact_repair_context)
            builder_command = artifact_repair_contracts.preferred_builder_command(contract)
            retry_prompt = (
                "Artifact materialization guard: reject the previous command. "
                "Source context is already available in the payload; do not inspect files or ask for more context. "
                f"Return exactly one command that creates or overwrites `{artifact_path}` using {builder_command}, "
                "with an allowed candidate path, exact source_lines line numbers, at least one --with replacement, "
                f"and a final redirect to {artifact_path}. Do not return cat, sed, head, tail, ls, find, git, grep, "
                "python/python3 -c, shell pipes, another source-inspection command, or a final/respond action. "
                f"Rejected reason: {rejected_retry_reason}. "
                "If the reason is source_inspection, use the source_context/source_lines already in the payload. "
                "If the reason is broad_replacement, choose one to three executable body lines inside the target "
                "function instead of replacing a whole function or high-value window. If the reason is "
                "definition_header_removal, preserve every original def/class header and edit only the function "
                "body. If the reason is invalid_python_replacement, replace the whole syntactic statement or "
                "continuation block around the target line with valid Python at the original indentation; do not "
                "edit only one line of a multi-line string, parenthesized expression, function call, or message "
                "construction. Review invalid_python_replacement_operations in the payload; when it includes a "
                "suggested_statement_range, use that full range with --replace-lines and one --with per output line, "
                "not embedded \\n escapes inside one --with value. If the suggested_statement_range is broad, keep "
                "the edit narrow by replacing a complete small expression/argument line with the same syntactic kind; "
                "never insert def/class/return/import/if/for/while/try/with into the middle of an existing expression "
                "or continuation line. If the reason is escaped_newline_replacement, split the intended multiline output "
                "into separate --with arguments or choose a single valid source line; do not include literal \\n characters "
                "inside one --with value. If the reason is source_identical_noop, change behavior rather than copying existing "
                "source text; review source_identical_noop_operations in the payload and do not repeat that "
                "line/span/replacement. When source_identical_noop_operations includes suggested_statement_range, "
                "prefer that complete executable statement with --replace-lines and a replacement that is textually "
                "different from the existing source. If the reason is duplicate_adjacent_line, do not copy an "
                "existing neighboring executable line into the replacement; preserve loop/order structure and choose "
                "a replacement that changes behavior without duplicating the previous or next source line. "
                "If the reason is python_ast_noop, do not change only comments, whitespace, docstrings, annotations, "
                "or other AST-equivalent text; alter executable behavior. If the reason is unused_signature_parameter, "
                "do not add parameters unless the function body reads them before assigning to them. If the reason is "
                "invalid_init_return_value, never add a value-returning return statement to __init__; assign state "
                "or return without a value instead. If the reason is invalid_init_generator, never add yield or "
                "yield from inside __init__; materialize iterable behavior in a normal method, helper, or assigned "
                "state instead. If the reason is local_use_before_assignment, ensure every local variable is assigned "
                "before its first read, or use a distinct already-bound name from the source context. "
                "If the reason is artifact_path_as_source or "
                "unknown_source_path, start from one of the concrete command_skeletons in artifact_repair_context "
                "and replace only the --with text; do not invent a different --path. The builder --path must be an "
                "allowed source file, never the output artifact path itself. If the reason is line_out_of_source_range, "
                "choose exact line numbers from source_lines_excerpt or edit_windows; never use arbitrary issue "
                "numbers, placeholder line numbers, or unverified anchors. If the reason is "
                "line_outside_anchor_preview, choose one of the valid_line_numbers_preview anchors exactly; do not "
                "switch to unrelated line numbers from the broader source file. "
                "If the reason is placeholder_replacement or comment_only_replacement, replace real executable "
                "source with a concrete behavior-changing line; do not use new_code, print('test'), assert True, "
                "assert False, comments-only edits, or generic stubs."
            )
            if task_context:
                retry_prompt += (
                    "\nTask context for choosing the semantic replacement follows. Use it only to choose the "
                    f"behavioral edit; the required command shape above is authoritative.\n{task_context}"
                )
            if artifact_repair_context_prompt:
                retry_prompt = f"{retry_prompt}\n{artifact_repair_context_prompt}"
            if retry_attempt > 1:
                retry_prompt += (
                    " Escalated materialization attempt: do not reuse the rejected command, do not copy the exact "
                    "existing source listed in the guard payload, and do not replace a whole function/class window."
                )
            if retry_attempt >= 4:
                retry_prompt += (
                    " This is the final materialization attempt: the only acceptable response is a code_execute "
                    f"{builder_command} --path command that writes {artifact_path} with a syntactically valid, "
                    "behavior-changing edit."
                )
            raw = self._create_decision(
                system_prompt=system_prompt,
                decision_prompt=retry_prompt,
                state_payload=retry_payload,
            )
            normalized = coerce_action_decision(raw)
            if normalized["action"] != CODE_EXECUTE:
                rejected_retry_command = str(normalized["content"]).strip()
                rejected_retry_reason = "non_code_response"
                continue
            content = _normalize_command_for_workspace(
                normalized["content"],
                state.task.workspace_subdir,
            )
            if self._artifact_source_inspection_allowed_after_history(state, content):
                raw_proposal_metadata = raw.get("proposal_metadata", {})
                if not isinstance(raw_proposal_metadata, dict):
                    raw_proposal_metadata = {}
                if context_compile_warning is not None:
                    raw_proposal_metadata = {
                        **raw_proposal_metadata,
                        "context_compile_degraded": dict(context_compile_warning),
                    }
                return ActionDecision(
                    thought=str(normalized["thought"]).strip()
                    or "Read line-numbered source before materializing the required artifact.",
                    action=CODE_EXECUTE,
                    content=content,
                    done=bool(normalized["done"]),
                    decision_source=str(raw.get("decision_source", "")).strip()
                    or "llm_artifact_source_inspection_followup",
                    proposal_metadata={
                        **dict(raw_proposal_metadata),
                        "artifact_source_inspection_followup": True,
                        "artifact_materialization_retry_attempt": retry_attempt,
                        "rejected_command": proposed_content,
                    },
                )
            if not self._artifact_materializes(state, content):
                rejected_retry_command = content
                rejected_retry_reason = self._artifact_materialization_rejection_reason(state, content)
                if rejected_retry_reason == "artifact_path_as_source":
                    remapped_command = self._artifact_remap_builder_source_path_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                    )
                    if remapped_command:
                        return ActionDecision(
                            thought="Remap a retry builder command that used the output artifact as its source path.",
                            action=CODE_EXECUTE,
                            content=remapped_command,
                            done=False,
                            decision_source="artifact_builder_source_path_remap_direct",
                            proposal_metadata={
                                "artifact_builder_source_path_remap_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if self._artifact_anchorable_rejection_reason(rejected_retry_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=rejected_retry_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                continue
            if self._artifact_builder_outside_retry_anchor_preview(state, content, artifact_repair_context):
                rejected_retry_command = content
                rejected_retry_reason = "line_outside_anchor_preview"
                anchored_retry = self._artifact_anchor_replacement_retry_decision(
                    state=state,
                    system_prompt=system_prompt,
                    decision_prompt=retry_prompt,
                    payload=retry_payload,
                    artifact_path=artifact_path,
                    builder_command=builder_command,
                    artifact_repair_context=artifact_repair_context,
                    rejected_command=content,
                    rejected_reason=rejected_retry_reason,
                    retry_attempt=retry_attempt,
                    context_compile_warning=context_compile_warning,
                )
                if anchored_retry is not None:
                    return anchored_retry
                continue
            guard_rejection_reason = self._artifact_builder_guard_rejection_reason(state, content)
            if guard_rejection_reason:
                if guard_rejection_reason == "artifact_path_as_source":
                    remapped_command = self._artifact_remap_builder_source_path_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                    )
                    if remapped_command:
                        return ActionDecision(
                            thought="Remap a guarded retry builder command that used the output artifact as its source path.",
                            action=CODE_EXECUTE,
                            content=remapped_command,
                            done=False,
                            decision_source="artifact_builder_source_path_remap_direct",
                            proposal_metadata={
                                "artifact_builder_source_path_remap_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if guard_rejection_reason == "escaped_newline_replacement":
                    split_command = self._artifact_split_escaped_newline_builder_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                    )
                    if split_command:
                        return ActionDecision(
                            thought="Split an escaped-newline retry builder replacement into explicit replacement lines.",
                            action=CODE_EXECUTE,
                            content=split_command,
                            done=False,
                            decision_source="artifact_escaped_newline_split_direct",
                            proposal_metadata={
                                "artifact_escaped_newline_split_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if guard_rejection_reason == "broad_replacement":
                    narrowed_command = self._artifact_narrow_broad_builder_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                    )
                    if narrowed_command:
                        return ActionDecision(
                            thought="Narrow an over-broad retry builder command to a focused executable edit.",
                            action=CODE_EXECUTE,
                            content=narrowed_command,
                            done=False,
                            decision_source="artifact_broad_builder_narrow_direct",
                            proposal_metadata={
                                "artifact_broad_builder_narrow_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                    if self._artifact_broad_builder_last_resort_allowed(state, content):
                        return ActionDecision(
                            thought="Use a bounded broad retry builder command after focused narrowing failed.",
                            action=CODE_EXECUTE,
                            content=content,
                            done=False,
                            decision_source="artifact_broad_builder_last_resort_direct",
                            proposal_metadata={
                                "artifact_broad_builder_last_resort_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if guard_rejection_reason in {"definition_header_removal", "invalid_python_replacement"}:
                    signature_command = self._artifact_expand_partial_header_signature_builder_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                    )
                    if signature_command:
                        return ActionDecision(
                            thought="Expand a retry command's partial multiline Python signature edit.",
                            action=CODE_EXECUTE,
                            content=signature_command,
                            done=False,
                            decision_source="artifact_header_signature_expand_direct",
                            proposal_metadata={
                                "artifact_header_signature_expand_direct": True,
                                "artifact_materialization_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if self._artifact_anchorable_rejection_reason(guard_rejection_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=guard_rejection_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                rejected_retry_command = content
                rejected_retry_reason = guard_rejection_reason
                continue
            raw_proposal_metadata = raw.get("proposal_metadata", {})
            if not isinstance(raw_proposal_metadata, dict):
                raw_proposal_metadata = {}
            if context_compile_warning is not None:
                raw_proposal_metadata = {
                    **raw_proposal_metadata,
                    "context_compile_degraded": dict(context_compile_warning),
                }
            return ActionDecision(
                thought=str(normalized["thought"]).strip()
                or f"Materialize {artifact_path} after verifier repair feedback.",
                action=CODE_EXECUTE,
                content=content,
                done=bool(normalized["done"]),
                decision_source=str(raw.get("decision_source", "")).strip() or "llm_artifact_materialization_retry",
                proposal_metadata={
                    **dict(raw_proposal_metadata),
                    "artifact_materialization_retry": True,
                    "artifact_materialization_retry_attempt": retry_attempt,
                    "rejected_command": proposed_content,
                },
            )
        terminal_metadata: dict[str, object] = {
            "artifact_materialization_retry": True,
            "artifact_materialization_retry_attempts": 5,
            "rejected_command": proposed_content,
            "retry_command": rejected_retry_command,
            "retry_rejected_reason": rejected_retry_reason,
        }
        if rejected_retry_reason == "source_identical_noop":
            source_identical_details = self._artifact_source_identical_operation_details(
                state,
                rejected_retry_command,
            )
            if source_identical_details:
                terminal_metadata["source_identical_noop_operations"] = source_identical_details
        if rejected_retry_reason == "invalid_python_replacement":
            invalid_python_details = self._artifact_invalid_python_replacement_details(
                state,
                rejected_retry_command,
            )
            if invalid_python_details:
                terminal_metadata["invalid_python_replacement_operations"] = invalid_python_details
        if rejected_retry_reason in {
            "line_out_of_source_range",
            "placeholder_replacement",
            "comment_only_replacement",
            "unknown_source_path",
            "duplicate_adjacent_line",
            "python_ast_noop",
            "unused_signature_parameter",
            "invalid_init_return_value",
            "invalid_init_generator",
            "local_use_before_assignment",
        }:
            guard_details = self._artifact_builder_guard_rejection_details(
                state,
                rejected_retry_reason,
                rejected_retry_command,
            )
            if guard_details:
                terminal_metadata["artifact_guard_rejection_operations"] = guard_details
        if artifact_repair_context:
            terminal_metadata["artifact_repair_context"] = dict(artifact_repair_context)
        return ActionDecision(
            thought="Reject repeated non-materializing artifact repair command.",
            action="respond",
            content="Stopping because the model did not produce a materialization command after verifier repair feedback.",
            done=True,
            decision_source="artifact_materialization_guard",
            proposal_metadata=terminal_metadata,
        )

    def _artifact_repair_continue_decision(
        self,
        *,
        state: AgentState,
        system_prompt: str,
        decision_prompt: str,
        payload: dict[str, object],
        proposed_content: str,
        context_compile_warning: dict[str, object] | None,
    ) -> ActionDecision | None:
        if not self._artifact_latest_repairable_failure(state):
            return None
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        builder_command = artifact_repair_contracts.preferred_builder_command(contract)
        rejected_retry_command = str(proposed_content).strip()
        artifact_repair_context = self._artifact_retry_context_payload(
            state,
            artifact_path=artifact_path,
            builder_command=builder_command,
        )
        artifact_repair_context_prompt = self._artifact_retry_context_prompt(artifact_repair_context)
        rejected_retry_reason = "respond_before_artifact_materialized"
        for retry_attempt in range(1, 4):
            retry_payload = dict(payload)
            retry_payload["artifact_repair_continue_guard"] = {
                "attempt": retry_attempt,
                "rejected_response": rejected_retry_command,
                "rejected_reason": rejected_retry_reason,
                "reason": (
                    "the task has repairable artifact verifier feedback; "
                    "respond/done is not allowed until the required artifact passes"
                ),
                "required_action": "code_execute",
            }
            if artifact_repair_context:
                retry_payload["artifact_repair_context"] = dict(artifact_repair_context)
            retry_prompt = (
                f"{decision_prompt}\n"
                "Artifact repair continue guard: do not stop or respond while the required artifact has repairable "
                f"verifier feedback. Return exactly one code_execute command that overwrites {artifact_path} with "
                f"{builder_command} --path using exact source line anchors. If the previous rejection involved "
                "invalid Python, replace the complete syntactic statement/block instead of a single continuation "
                "line. Do not inspect source first, do not repeat previous failed commands, and do not return a "
                "final answer. The builder --path must be an allowed source file, never the output artifact path "
                "itself. Use line numbers that appear in source_lines or edit_windows, and do not use placeholder "
                "or comments-only replacements."
            )
            if artifact_repair_context_prompt:
                retry_prompt = f"{retry_prompt}\n{artifact_repair_context_prompt}"
            if retry_attempt > 1:
                retry_prompt += (
                    f" This is the final continue attempt: return only a code_execute {builder_command} command "
                    f"that writes {artifact_path}."
                )
            raw = self._create_decision(
                system_prompt=system_prompt,
                decision_prompt=retry_prompt,
                state_payload=retry_payload,
            )
            normalized = coerce_action_decision(raw)
            if normalized["action"] != CODE_EXECUTE:
                rejected_retry_command = str(normalized["content"]).strip()
                rejected_retry_reason = "non_code_response"
                continue
            content = _normalize_command_for_workspace(
                normalized["content"],
                state.task.workspace_subdir,
            )
            if not self._artifact_materializes(state, content):
                rejected_retry_command = content
                rejected_retry_reason = self._artifact_materialization_rejection_reason(state, content)
                if rejected_retry_reason == "artifact_path_as_source":
                    remapped_command = self._artifact_remap_builder_source_path_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                    )
                    if remapped_command:
                        return ActionDecision(
                            thought="Remap a continue-repair builder command that used the output artifact as its source path.",
                            action=CODE_EXECUTE,
                            content=remapped_command,
                            done=False,
                            decision_source="artifact_builder_source_path_remap_direct",
                            proposal_metadata={
                                "artifact_builder_source_path_remap_direct": True,
                                "artifact_repair_continue_retry": True,
                                "artifact_repair_continue_attempt": retry_attempt,
                                "rejected_response": proposed_content,
                                "rejected_command": content,
                            },
                        )
                if self._artifact_anchorable_rejection_reason(rejected_retry_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=rejected_retry_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                        retry_metadata={
                            "artifact_repair_continue_retry": True,
                            "artifact_repair_continue_attempt": retry_attempt,
                            "rejected_response": proposed_content,
                        },
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                continue
            if self._artifact_builder_outside_retry_anchor_preview(state, content, artifact_repair_context):
                rejected_retry_command = content
                rejected_retry_reason = "line_outside_anchor_preview"
                anchored_retry = self._artifact_anchor_replacement_retry_decision(
                    state=state,
                    system_prompt=system_prompt,
                    decision_prompt=retry_prompt,
                    payload=retry_payload,
                    artifact_path=artifact_path,
                    builder_command=builder_command,
                    artifact_repair_context=artifact_repair_context,
                    rejected_command=content,
                    rejected_reason=rejected_retry_reason,
                    retry_attempt=retry_attempt,
                    context_compile_warning=context_compile_warning,
                    retry_metadata={
                        "artifact_repair_continue_retry": True,
                        "artifact_repair_continue_attempt": retry_attempt,
                        "rejected_response": proposed_content,
                    },
                )
                if anchored_retry is not None:
                    return anchored_retry
                continue
            guard_rejection_reason = self._artifact_builder_guard_rejection_reason(state, content)
            if guard_rejection_reason:
                if guard_rejection_reason == "artifact_path_as_source":
                    remapped_command = self._artifact_remap_builder_source_path_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                    )
                    if remapped_command:
                        return ActionDecision(
                            thought="Remap a guarded continue-repair builder command that used the output artifact as its source path.",
                            action=CODE_EXECUTE,
                            content=remapped_command,
                            done=False,
                            decision_source="artifact_builder_source_path_remap_direct",
                            proposal_metadata={
                                "artifact_builder_source_path_remap_direct": True,
                                "artifact_repair_continue_retry": True,
                                "artifact_repair_continue_attempt": retry_attempt,
                                "rejected_response": proposed_content,
                                "rejected_command": content,
                            },
                        )
                if guard_rejection_reason == "broad_replacement":
                    narrowed_command = self._artifact_narrow_broad_builder_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                    )
                    if narrowed_command:
                        return ActionDecision(
                            thought="Narrow an over-broad continue-repair builder command to a focused executable edit.",
                            action=CODE_EXECUTE,
                            content=narrowed_command,
                            done=False,
                            decision_source="artifact_broad_builder_narrow_direct",
                            proposal_metadata={
                                "artifact_broad_builder_narrow_direct": True,
                                "artifact_repair_continue_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                    if self._artifact_broad_builder_last_resort_allowed(state, content):
                        return ActionDecision(
                            thought="Use a bounded broad continue-repair builder command after focused narrowing failed.",
                            action=CODE_EXECUTE,
                            content=content,
                            done=False,
                            decision_source="artifact_broad_builder_last_resort_direct",
                            proposal_metadata={
                                "artifact_broad_builder_last_resort_direct": True,
                                "artifact_repair_continue_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if self._artifact_anchorable_rejection_reason(guard_rejection_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=guard_rejection_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                        retry_metadata={
                            "artifact_repair_continue_retry": True,
                            "artifact_repair_continue_attempt": retry_attempt,
                            "rejected_response": proposed_content,
                        },
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                rejected_retry_command = content
                rejected_retry_reason = guard_rejection_reason
                continue
            if self._artifact_noop_repair_context(state) and self._artifact_noop_retry_needed(state, content):
                rejected_retry_command = content
                rejected_retry_reason = "noop_repair"
                continue
            raw_proposal_metadata = raw.get("proposal_metadata", {})
            if not isinstance(raw_proposal_metadata, dict):
                raw_proposal_metadata = {}
            if context_compile_warning is not None:
                raw_proposal_metadata = {
                    **raw_proposal_metadata,
                    "context_compile_degraded": dict(context_compile_warning),
                }
            return ActionDecision(
                thought=str(normalized["thought"]).strip() or "Continue artifact repair instead of stopping.",
                action=CODE_EXECUTE,
                content=content,
                done=bool(normalized["done"]),
                decision_source=str(raw.get("decision_source", "")).strip() or "llm_artifact_repair_continue",
                proposal_metadata={
                    **dict(raw_proposal_metadata),
                    "artifact_repair_continue_retry": True,
                    "artifact_repair_continue_attempt": retry_attempt,
                    "rejected_response": proposed_content,
                },
            )
        return ActionDecision(
            thought="Reject early stop while required artifact still has repairable verifier feedback.",
            action="respond",
            content="Stopping because the model did not produce an artifact materialization command after continue feedback.",
            done=True,
            decision_source="artifact_repair_continue_guard",
            proposal_metadata={
                "artifact_repair_continue_retry": True,
                "artifact_repair_continue_attempts": 3,
                "rejected_response": proposed_content,
                "retry_command": rejected_retry_command,
                "retry_rejected_reason": rejected_retry_reason,
            },
        )

    def _artifact_identifier_retry_decision(
        self,
        *,
        state: AgentState,
        system_prompt: str,
        decision_prompt: str,
        payload: dict[str, object],
        proposed_content: str,
        context_compile_warning: dict[str, object] | None,
    ) -> ActionDecision | None:
        identifier = self._artifact_latest_required_identifier_failure(state)
        if not identifier:
            return None
        if self._artifact_command_references_identifier(proposed_content, identifier):
            return None
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        builder_command = artifact_repair_contracts.preferred_builder_command(contract)
        rejected_retry_command = str(proposed_content).strip()
        for retry_attempt in range(1, 3):
            retry_payload = dict(payload)
            retry_payload["artifact_required_identifier_guard"] = {
                "attempt": retry_attempt,
                "required_identifier": identifier,
                "rejected_command": rejected_retry_command,
                "reason": "the previous artifact did not reference the required identifier",
            }
            retry_prompt = (
                f"{decision_prompt}\n"
                f"Artifact required identifier guard: the verifier requires `{artifact_path}` to reference `{identifier}`. "
                "Reject the previous command because it did not include that identifier. Return exactly one "
                f"{builder_command} --path command that overwrites {artifact_path} and includes the required identifier "
                "inside the replacement source text. Do not use generic signature-only or **kwargs changes that "
                "ignore the named identifier."
            )
            if retry_attempt > 1:
                retry_prompt += (
                    f" This is the final identifier repair attempt: return only a {builder_command} command "
                    f"that writes {artifact_path} and includes the required identifier."
                )
            raw = self._create_decision(
                system_prompt=system_prompt,
                decision_prompt=retry_prompt,
                state_payload=retry_payload,
            )
            normalized = coerce_action_decision(raw)
            if normalized["action"] != CODE_EXECUTE:
                rejected_retry_command = str(normalized["content"]).strip()
                continue
            content = _normalize_command_for_workspace(
                normalized["content"],
                state.task.workspace_subdir,
            )
            if not self._artifact_materializes(state, content):
                rejected_retry_command = content
                continue
            if self._artifact_builder_guard_rejection_reason(state, content):
                rejected_retry_command = content
                continue
            if not self._artifact_command_references_identifier(content, identifier):
                rejected_retry_command = content
                continue
            raw_proposal_metadata = raw.get("proposal_metadata", {})
            if not isinstance(raw_proposal_metadata, dict):
                raw_proposal_metadata = {}
            if context_compile_warning is not None:
                raw_proposal_metadata = {
                    **raw_proposal_metadata,
                    "context_compile_degraded": dict(context_compile_warning),
                }
            return ActionDecision(
                thought=str(normalized["thought"]).strip() or f"Repair artifact to reference {identifier}.",
                action=CODE_EXECUTE,
                content=content,
                done=bool(normalized["done"]),
                decision_source=str(raw.get("decision_source", "")).strip()
                or "llm_artifact_required_identifier_retry",
                proposal_metadata={
                    **dict(raw_proposal_metadata),
                    "artifact_required_identifier_retry": True,
                    "artifact_required_identifier_retry_attempt": retry_attempt,
                    "required_identifier": identifier,
                    "rejected_command": proposed_content,
                },
            )
        return ActionDecision(
            thought="Reject artifact repair command that omits the required identifier.",
            action="respond",
            content=f"Stopping because the model did not produce a {artifact_path} command referencing {identifier}.",
            done=True,
            decision_source="artifact_required_identifier_guard",
            proposal_metadata={
                "artifact_required_identifier_retry": True,
                "artifact_required_identifier_retry_attempts": 2,
                "required_identifier": identifier,
                "rejected_command": proposed_content,
                "retry_command": rejected_retry_command,
            },
        )

    def _artifact_semantic_retry_decision(
        self,
        *,
        state: AgentState,
        system_prompt: str,
        decision_prompt: str,
        payload: dict[str, object],
        proposed_content: str,
        context_compile_warning: dict[str, object] | None,
    ) -> ActionDecision | None:
        if not self._artifact_noop_repair_context(state):
            return None
        if not self._artifact_noop_retry_needed(state, proposed_content):
            return None
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        builder_command = artifact_repair_contracts.preferred_builder_command(contract)
        artifact_repair_context = self._artifact_retry_context_payload(
            state,
            artifact_path=artifact_path,
            builder_command=builder_command,
        )
        artifact_repair_context_prompt = self._artifact_retry_context_prompt(artifact_repair_context)
        rejected_retry_command = ""
        rejected_retry_reason = "noop_repair"
        for retry_attempt in range(1, 3):
            retry_payload = dict(payload)
            retry_payload["artifact_semantic_repair_guard"] = {
                "attempt": retry_attempt,
                "rejected_command": rejected_retry_command or proposed_content,
                "rejected_reason": rejected_retry_reason,
                "reason": (
                    "the previous artifact repair had no meaningful content or executable AST change; "
                    "the next artifact must change executable behavior"
                ),
                "required_command_shape": artifact_repair_contracts.required_command_shape(contract),
            }
            retry_prompt = (
                f"{decision_prompt}\n"
                "Artifact semantic repair guard: reject the previous no-op or signature-only repair command. "
                f"Return exactly one command that overwrites {artifact_path} with {builder_command}. The edit must change "
                "executable behavior inside an existing function/method body or expression, not only docstrings, "
                "comments, imports, decorators, def/class lines, whitespace, or repeated source text. Do not repeat "
                f"any previous failed {builder_command} command and do not inspect source first. The builder --path "
                "must be an allowed source file, never the output artifact path itself. Use line numbers that appear "
                "in source_lines or edit_windows, and do not use placeholder or comments-only replacements."
            )
            if artifact_repair_context:
                retry_payload["artifact_repair_context"] = dict(artifact_repair_context)
            if artifact_repair_context_prompt:
                retry_prompt = f"{retry_prompt}\n{artifact_repair_context_prompt}"
            if retry_attempt > 1:
                retry_prompt += (
                    " This is the final semantic repair attempt: return only a behavior-changing "
                    f"{builder_command} --path command that writes {artifact_path}."
                )
            raw = self._create_decision(
                system_prompt=system_prompt,
                decision_prompt=retry_prompt,
                state_payload=retry_payload,
            )
            normalized = coerce_action_decision(raw)
            if normalized["action"] != CODE_EXECUTE:
                rejected_retry_command = str(normalized["content"]).strip()
                rejected_retry_reason = "non_code_response"
                continue
            content = _normalize_command_for_workspace(
                normalized["content"],
                state.task.workspace_subdir,
            )
            if not self._artifact_materializes(state, content):
                rejected_retry_command = content
                rejected_retry_reason = self._artifact_materialization_rejection_reason(state, content)
                if self._artifact_anchorable_rejection_reason(rejected_retry_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=rejected_retry_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                        retry_metadata={
                            "artifact_semantic_repair_retry": True,
                            "artifact_semantic_repair_retry_attempt": retry_attempt,
                            "rejected_command": proposed_content,
                        },
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                continue
            if self._artifact_builder_outside_retry_anchor_preview(state, content, artifact_repair_context):
                rejected_retry_command = content
                rejected_retry_reason = "line_outside_anchor_preview"
                anchored_retry = self._artifact_anchor_replacement_retry_decision(
                    state=state,
                    system_prompt=system_prompt,
                    decision_prompt=retry_prompt,
                    payload=retry_payload,
                    artifact_path=artifact_path,
                    builder_command=builder_command,
                    artifact_repair_context=artifact_repair_context,
                    rejected_command=content,
                    rejected_reason=rejected_retry_reason,
                    retry_attempt=retry_attempt,
                    context_compile_warning=context_compile_warning,
                    retry_metadata={
                        "artifact_semantic_repair_retry": True,
                        "artifact_semantic_repair_retry_attempt": retry_attempt,
                        "rejected_command": proposed_content,
                    },
                )
                if anchored_retry is not None:
                    return anchored_retry
                continue
            guard_rejection_reason = self._artifact_builder_guard_rejection_reason(state, content)
            if guard_rejection_reason:
                if guard_rejection_reason == "broad_replacement":
                    narrowed_command = self._artifact_narrow_broad_builder_command(
                        state,
                        content,
                        artifact_path=artifact_path,
                    )
                    if narrowed_command:
                        return ActionDecision(
                            thought="Narrow an over-broad semantic-repair builder command to a focused executable edit.",
                            action=CODE_EXECUTE,
                            content=narrowed_command,
                            done=False,
                            decision_source="artifact_broad_builder_narrow_direct",
                            proposal_metadata={
                                "artifact_broad_builder_narrow_direct": True,
                                "artifact_semantic_repair_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                    if self._artifact_broad_builder_last_resort_allowed(state, content):
                        return ActionDecision(
                            thought="Use a bounded broad semantic-repair builder command after focused narrowing failed.",
                            action=CODE_EXECUTE,
                            content=content,
                            done=False,
                            decision_source="artifact_broad_builder_last_resort_direct",
                            proposal_metadata={
                                "artifact_broad_builder_last_resort_direct": True,
                                "artifact_semantic_repair_retry_attempt": retry_attempt,
                                "rejected_command": content,
                            },
                        )
                if self._artifact_anchorable_rejection_reason(guard_rejection_reason):
                    anchored_retry = self._artifact_anchor_replacement_retry_decision(
                        state=state,
                        system_prompt=system_prompt,
                        decision_prompt=retry_prompt,
                        payload=retry_payload,
                        artifact_path=artifact_path,
                        builder_command=builder_command,
                        artifact_repair_context=artifact_repair_context,
                        rejected_command=content,
                        rejected_reason=guard_rejection_reason,
                        retry_attempt=retry_attempt,
                        context_compile_warning=context_compile_warning,
                        retry_metadata={
                            "artifact_semantic_repair_retry": True,
                            "artifact_semantic_repair_retry_attempt": retry_attempt,
                            "rejected_command": proposed_content,
                        },
                    )
                    if anchored_retry is not None:
                        return anchored_retry
                rejected_retry_command = content
                rejected_retry_reason = guard_rejection_reason
                continue
            if self._artifact_noop_retry_needed(state, content):
                rejected_retry_command = content
                rejected_retry_reason = "noop_repair"
                continue
            raw_proposal_metadata = raw.get("proposal_metadata", {})
            if not isinstance(raw_proposal_metadata, dict):
                raw_proposal_metadata = {}
            if context_compile_warning is not None:
                raw_proposal_metadata = {
                    **raw_proposal_metadata,
                    "context_compile_degraded": dict(context_compile_warning),
                }
            return ActionDecision(
                thought=str(normalized["thought"]).strip()
                or "Create a behavior-changing artifact repair after no-op verifier feedback.",
                action=CODE_EXECUTE,
                content=content,
                done=bool(normalized["done"]),
                decision_source=str(raw.get("decision_source", "")).strip() or "llm_artifact_semantic_repair",
                proposal_metadata={
                    **dict(raw_proposal_metadata),
                    "artifact_semantic_repair_retry": True,
                    "artifact_semantic_repair_retry_attempt": retry_attempt,
                    "rejected_command": proposed_content,
                },
            )
        return ActionDecision(
            thought="Reject repeated no-op artifact repair command.",
            action="respond",
            content=f"Stopping because the model did not produce a behavior-changing {artifact_path} repair command.",
            done=True,
            decision_source="artifact_semantic_repair_guard",
            proposal_metadata={
                "artifact_semantic_repair_retry": True,
                "artifact_semantic_repair_retry_attempts": 2,
                "rejected_command": proposed_content,
                "retry_command": rejected_retry_command,
                "retry_rejected_reason": rejected_retry_reason,
            },
        )

    @staticmethod
    def _artifact_materializes(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=True)
        if spec is not None and LLMDecisionPolicy._artifact_builder_uses_artifact_as_source(state, command):
            return False
        return spec is not None and bool(spec.get("path")) and bool(spec.get("operations"))

    @classmethod
    def _artifact_materialization_rejection_reason(cls, state: AgentState, command: str) -> str:
        normalized = str(command).strip()
        if not normalized:
            return "empty_command"
        guard_rejection_reason = cls._artifact_builder_guard_rejection_reason(state, normalized)
        if guard_rejection_reason:
            return guard_rejection_reason
        if normalized.startswith(("cat ", "sed ", "head ", "tail ", "ls ", "find ", "grep ")):
            return "source_inspection"
        if any(token in normalized for token in (" | ", " && ", " || ", " > /tmp/", " < ")):
            return "unsupported_shell_or_source_inspection"
        if cls._artifact_builder_command_prefix(state, normalized):
            return "malformed_or_non_materializing_patch_builder"
        return "non_materializing_command"

    @classmethod
    def _artifact_builder_guard_rejection_reason(cls, state: AgentState, command: str) -> str:
        normalized = str(command).strip()
        if not cls._artifact_builder_command_prefix(state, normalized):
            return ""
        if cls._artifact_builder_uses_artifact_as_source(state, normalized):
            return "artifact_path_as_source"
        if cls._artifact_builder_uses_unknown_source_path(state, normalized):
            return "unknown_source_path"
        if cls._artifact_builder_line_out_of_known_source_range(state, normalized):
            return "line_out_of_source_range"
        if cls._artifact_builder_has_placeholder_replacement(state, normalized):
            return "placeholder_replacement"
        if cls._artifact_builder_has_comment_only_replacement(state, normalized):
            return "comment_only_replacement"
        if cls._artifact_builder_replaces_existing_source(state, normalized):
            return "source_identical_noop"
        if cls._artifact_builder_introduces_adjacent_duplicate_line(state, normalized):
            return "duplicate_adjacent_line"
        if cls._artifact_builder_broad_replacement(state, normalized):
            return "broad_replacement"
        if cls._artifact_builder_removes_definition_header(state, normalized):
            return "definition_header_removal"
        if cls._artifact_builder_has_escaped_newline_replacement(state, normalized):
            return "escaped_newline_replacement"
        if cls._artifact_builder_would_produce_invalid_python(state, normalized):
            return "invalid_python_replacement"
        if cls._artifact_builder_python_ast_unchanged(state, normalized):
            return "python_ast_noop"
        if cls._artifact_builder_unused_python_parameter_names(state, normalized):
            return "unused_signature_parameter"
        if cls._artifact_builder_invalid_init_return_names(state, normalized):
            return "invalid_init_return_value"
        if cls._artifact_builder_invalid_init_generator_names(state, normalized):
            return "invalid_init_generator"
        if cls._artifact_builder_local_load_before_assignment_names(state, normalized):
            return "local_use_before_assignment"
        return ""

    @staticmethod
    def _artifact_materialization_retry_reason_text(reason: str, *, artifact_path: str) -> str:
        normalized = str(reason).strip() or "non_materializing_command"
        return (
            f"{artifact_path} repair is active and source context is already available; "
            f"the next command must materialize a non-noop focused artifact ({normalized})"
        )

    @staticmethod
    def _artifact_anchorable_rejection_reason(reason: str) -> bool:
        return str(reason).strip() in {
            "artifact_path_as_source",
            "line_out_of_source_range",
            "line_outside_anchor_preview",
            "placeholder_replacement",
            "comment_only_replacement",
            "unknown_source_path",
            "source_inspection",
            "invalid_python_replacement",
            "source_identical_noop",
            "python_ast_noop",
            "invalid_init_generator",
            "local_use_before_assignment",
        }

    def _artifact_anchor_replacement_retry_decision(
        self,
        *,
        state: AgentState,
        system_prompt: str,
        decision_prompt: str,
        payload: dict[str, object],
        artifact_path: str,
        builder_command: str,
        artifact_repair_context: dict[str, object],
        rejected_command: str,
        rejected_reason: str,
        retry_attempt: int,
        context_compile_warning: dict[str, object] | None,
        retry_metadata: dict[str, object] | None = None,
    ) -> ActionDecision | None:
        path = str(artifact_repair_context.get("preferred_source_path", "")).strip()
        raw_line_numbers = artifact_repair_context.get("valid_line_numbers_preview", [])
        valid_line_numbers = [
            int(value)
            for value in raw_line_numbers
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        ]
        if not path or not valid_line_numbers:
            return None
        command = self._artifact_builder_response_to_anchor_command(
            state,
            rejected_command,
            path=path,
            valid_line_numbers=valid_line_numbers,
            artifact_path=artifact_path,
            builder_command=builder_command,
        )
        if command:
            return ActionDecision(
                thought="Remap a rejected artifact builder replacement onto a validated source anchor.",
                action=CODE_EXECUTE,
                content=command,
                done=False,
                decision_source="artifact_anchor_replacement_direct",
                proposal_metadata={
                    "artifact_anchor_replacement_retry": True,
                    "artifact_anchor_replacement_attempt": 0,
                    "artifact_materialization_retry_attempt": retry_attempt,
                    "rejected_command": rejected_command,
                    "rejected_reason": rejected_reason,
                    **(retry_metadata if isinstance(retry_metadata, dict) else {}),
                },
            )
        previous_anchor_response = ""
        task_context = str(getattr(state.task, "prompt", "")).strip()
        if len(task_context) > 1600:
            task_context = f"{task_context[:1600]}\n...[task prompt truncated for artifact anchor repair]..."
        for anchor_attempt in range(1, 3):
            anchor_payload = dict(payload)
            anchor_payload["artifact_anchor_repair_guard"] = {
                "attempt": retry_attempt,
                "anchor_attempt": anchor_attempt,
                "fixed_path": path,
                "valid_line_numbers": valid_line_numbers[:32],
                "rejected_command": rejected_command,
                "rejected_reason": rejected_reason,
                "previous_anchor_response": previous_anchor_response,
                "required_response_content": "<line_number> || <replacement source line>",
            "source_lines_excerpt": str(artifact_repair_context.get("source_lines_excerpt", ""))[:2400],
            "edit_windows": str(artifact_repair_context.get("edit_windows", ""))[:2400],
        }
            anchor_prompt = (
                "Constrained artifact anchor repair: the previous artifact command had an invalid path, invalid "
                "line anchor, placeholder replacement, comments-only replacement, or repeated source inspection "
                "after source context was already available, or it would produce invalid Python. Do not return prose. "
                "Source context is already available in artifact_anchor_repair_guard; do not inspect files. "
                "Prefer one code_execute action whose content is exactly `<line_number> || <replacement source line>`. "
                f"The line_number must be one of: {', '.join(str(value) for value in valid_line_numbers[:32])}. "
                f"The replacement line will be inserted by the kernel into `{builder_command} --path {path} "
                f"--replace-line <line_number> --with ... > {artifact_path}`. The replacement must be a concrete "
                "behavior-changing Python source line at the same indentation, not new_code, not print('test'), not "
                "assert True/False, not a comment, and not a def/class/decorator line. If you return a builder "
                "command anyway, the kernel will ignore its path and line number and reuse only a valid replacement "
                "line from it."
            )
            if task_context:
                anchor_prompt += (
                    "\nTask context for choosing the semantic replacement follows. Use it only to choose the "
                    f"behavioral edit; the constrained response shape above is authoritative.\n{task_context}"
                )
            if previous_anchor_response:
                anchor_prompt += (
                    f" Previous anchor response was invalid and ignored: {previous_anchor_response[:500]}. "
                    "Now return only `<line_number> || <replacement source line>`."
                )
            raw = self._create_decision(
                system_prompt=system_prompt,
                decision_prompt=anchor_prompt,
                state_payload=anchor_payload,
            )
            normalized = coerce_action_decision(raw)
            if normalized["action"] != CODE_EXECUTE:
                previous_anchor_response = str(normalized["content"]).strip()
                continue
            content = str(normalized["content"]).strip()
            direct_command = _normalize_command_for_workspace(content, state.task.workspace_subdir)
            if self._artifact_materializes(state, direct_command) and not self._artifact_builder_guard_rejection_reason(
                state,
                direct_command,
            ):
                command = direct_command
            else:
                command = self._artifact_anchor_replacement_content_to_command(
                    state,
                    content,
                    path=path,
                    valid_line_numbers=valid_line_numbers,
                    artifact_path=artifact_path,
                    builder_command=builder_command,
                )
            if not command:
                command = self._artifact_builder_response_to_anchor_command(
                    state,
                    content,
                    path=path,
                    valid_line_numbers=valid_line_numbers,
                    artifact_path=artifact_path,
                    builder_command=builder_command,
                )
            if not command:
                previous_anchor_response = content
                continue
            raw_proposal_metadata = raw.get("proposal_metadata", {})
            if not isinstance(raw_proposal_metadata, dict):
                raw_proposal_metadata = {}
            if context_compile_warning is not None:
                raw_proposal_metadata = {
                    **raw_proposal_metadata,
                    "context_compile_degraded": dict(context_compile_warning),
                }
            extra_retry_metadata = retry_metadata if isinstance(retry_metadata, dict) else {}
            return ActionDecision(
                thought=str(normalized["thought"]).strip()
                or "Construct an anchored artifact builder command from a constrained replacement line.",
                action=CODE_EXECUTE,
                content=command,
                done=False,
                decision_source="artifact_anchor_replacement_direct",
                proposal_metadata={
                    **dict(raw_proposal_metadata),
                    "artifact_anchor_replacement_retry": True,
                    "artifact_anchor_replacement_attempt": anchor_attempt,
                    "artifact_materialization_retry_attempt": retry_attempt,
                    "rejected_command": rejected_command,
                    "rejected_reason": rejected_reason,
                    **extra_retry_metadata,
                },
            )
        return None

    def _artifact_builder_outside_retry_anchor_preview(
        self,
        state: AgentState,
        command: str,
        artifact_repair_context: dict[str, object],
    ) -> bool:
        if not isinstance(artifact_repair_context, dict) or not artifact_repair_context:
            return False
        raw_valid_line_numbers = artifact_repair_context.get("valid_line_numbers_preview", [])
        if not isinstance(raw_valid_line_numbers, list) or not raw_valid_line_numbers:
            return False
        valid_line_numbers = {
            int(value)
            for value in raw_valid_line_numbers
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        }
        if not valid_line_numbers:
            return False
        preferred_path = str(artifact_repair_context.get("preferred_source_path", "")).strip().strip("/")
        if not preferred_path:
            return False
        spec = self._artifact_builder_command_spec(state, command, require_redirect=True)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if path != preferred_path:
            return False
        for start_line, end_line, _replacement_lines in spec.get("operations", []):
            line_span = range(int(start_line), int(end_line) + 1)
            if any(line_number not in valid_line_numbers for line_number in line_span):
                return True
        return False

    def _artifact_anchor_replacement_content_to_command(
        self,
        state: AgentState,
        content: str,
        *,
        path: str,
        valid_line_numbers: list[int],
        artifact_path: str,
        builder_command: str,
    ) -> str:
        if "||" not in str(content):
            return ""
        raw_line_number, raw_replacement = str(content).split("||", 1)
        delimiter_left_padding = raw_line_number.endswith(" ")
        raw_line_number = raw_line_number.strip()
        if not raw_line_number.isdigit():
            return ""
        line_number = int(raw_line_number)
        if line_number not in set(valid_line_numbers):
            return ""
        replacement_line = raw_replacement.splitlines()[0].rstrip("\n")
        if delimiter_left_padding and replacement_line.startswith(" "):
            replacement_line = replacement_line[1:]
        candidate = (
            f"{builder_command} --path {shlex.quote(path)} --replace-line {line_number} "
            f"--with {shlex.quote(replacement_line)} > {artifact_path}"
        )
        if self._artifact_builder_guard_rejection_reason(state, candidate):
            return ""
        return candidate

    def _artifact_builder_response_to_anchor_command(
        self,
        state: AgentState,
        content: str,
        *,
        path: str,
        valid_line_numbers: list[int],
        artifact_path: str,
        builder_command: str,
    ) -> str:
        spec = self._artifact_builder_command_spec(state, content, require_redirect=False)
        if spec is None:
            return ""
        preferred_lines: list[int] = []
        for start_line, _end_line, _replacement_lines in spec.get("operations", []):
            if int(start_line) in valid_line_numbers and int(start_line) not in preferred_lines:
                preferred_lines.append(int(start_line))
        for line_number in valid_line_numbers:
            if line_number not in preferred_lines:
                preferred_lines.append(line_number)
        for _start_line, _end_line, replacement_lines in spec.get("operations", []):
            for raw_replacement in self._artifact_expanded_replacement_lines(replacement_lines):
                replacement_line = str(raw_replacement).splitlines()[0].rstrip("\n")
                stripped = replacement_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped in {"new_code", "new_code_here", "changed"}:
                    continue
                if stripped in {"assert True", "assert False"}:
                    continue
                if re.match(r"print\((['\"])(test|hello)\1\)", stripped, re.IGNORECASE):
                    continue
                for line_number in preferred_lines:
                    candidate = (
                        f"{builder_command} --path {shlex.quote(path)} --replace-line {line_number} "
                        f"--with {shlex.quote(replacement_line)} > {artifact_path}"
                    )
                    if not self._artifact_builder_guard_rejection_reason(state, candidate):
                        return candidate
        return ""

    def _artifact_remap_builder_source_path_command(
        self,
        state: AgentState,
        command: str,
        *,
        artifact_path: str,
        builder_command: str,
        artifact_repair_context: dict[str, object],
    ) -> str:
        spec = self._artifact_builder_command_spec(state, command, require_redirect=True)
        if spec is None:
            return ""
        source_path = str(spec.get("path") or "").strip().strip("/")
        if source_path != str(artifact_path).strip().strip("/"):
            return ""
        target_path = str(artifact_repair_context.get("preferred_source_path", "")).strip().strip("/")
        if not target_path:
            source_paths = [
                str(path).strip().strip("/")
                for path in artifact_repair_context.get("allowed_source_paths", [])
                if str(path).strip()
            ]
            if len(source_paths) != 1:
                return ""
            target_path = source_paths[0]
        valid_line_numbers = {
            int(value)
            for value in artifact_repair_context.get("valid_line_numbers_preview", [])
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        }
        operations = list(spec.get("operations", []))
        if valid_line_numbers:
            for start_line, end_line, _replacement_lines in operations:
                if any(line_number not in valid_line_numbers for line_number in range(int(start_line), int(end_line) + 1)):
                    return ""
        candidate = self._artifact_build_builder_command(
            builder_command=builder_command,
            path=target_path,
            operations=operations,
            artifact_path=artifact_path,
        )
        if not candidate:
            return ""
        if self._artifact_builder_outside_retry_anchor_preview(state, candidate, artifact_repair_context):
            return ""
        if self._artifact_builder_guard_rejection_reason(state, candidate):
            return ""
        return candidate

    @staticmethod
    def _artifact_build_builder_command(
        *,
        builder_command: str,
        path: str,
        operations: list[object],
        artifact_path: str,
    ) -> str:
        if not path or not operations:
            return ""
        parts = [str(builder_command or "patch_builder").strip() or "patch_builder", "--path", shlex.quote(path)]
        for raw_operation in operations:
            try:
                start_line, end_line, replacement_lines = raw_operation
            except (TypeError, ValueError):
                return ""
            expanded_replacements = LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines)
            if not expanded_replacements:
                return ""
            start = int(start_line)
            end = int(end_line)
            if start <= 0 or end < start:
                return ""
            if start == end and len(expanded_replacements) == 1:
                parts.extend(["--replace-line", str(start)])
            else:
                parts.extend(["--replace-lines", str(start), str(end)])
            for replacement in expanded_replacements:
                parts.extend(["--with", shlex.quote(str(replacement).rstrip("\n"))])
        parts.extend([">", shlex.quote(artifact_path)])
        return " ".join(parts)

    def _artifact_retry_context_payload(
        self,
        state: AgentState,
        *,
        artifact_path: str,
        builder_command: str,
    ) -> dict[str, object]:
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        if not isinstance(setup_file_contents, dict):
            setup_file_contents = {}
        source_context = self._artifact_source_context_by_path(state)
        source_paths = [path for path in source_context.keys() if path][:8]
        last_source_path = self._artifact_last_builder_source_path(state) or self._artifact_last_inspected_source_path(state)
        if not last_source_path and source_paths:
            last_source_path = source_paths[0]
        preferred_source_path = last_source_path or (source_paths[0] if source_paths else "")
        source_lines_path = f"source_lines/{last_source_path}.lines" if last_source_path else ""
        edit_windows = str(
            metadata.get("artifact_executable_edit_windows")
            or metadata.get("swe_executable_edit_windows")
            or ""
        ).strip()
        discarded_edit_window_line_numbers: set[int] = set()
        if edit_windows and preferred_source_path:
            edit_windows, discarded_edit_window_line_numbers = self._artifact_verified_edit_windows(
                state,
                preferred_source_path,
                edit_windows,
            )
        prompt_line_numbers = self._artifact_line_numbers_from_prompt(str(getattr(state.task, "prompt", "")))
        edit_window_line_numbers = self._artifact_line_numbers_from_text(edit_windows)
        if source_lines_path and source_lines_path not in setup_file_contents:
            source_lines_path = self._artifact_first_setup_source_lines_path(setup_file_contents)
        source_lines_excerpt = ""
        if source_lines_path:
            source_lines_text = str(setup_file_contents.get(source_lines_path, ""))
            source_lines_excerpt = self._artifact_source_lines_excerpt(
                source_lines_text,
                target_line_numbers=[*prompt_line_numbers, *edit_window_line_numbers],
            )
        else:
            source_lines_text = ""
        excerpt_line_numbers = self._artifact_line_numbers_from_text(source_lines_excerpt)
        candidate_line_numbers: list[int] = []
        seen_line_numbers: set[int] = set()
        for line_number in [*prompt_line_numbers, *edit_window_line_numbers, *excerpt_line_numbers]:
            if line_number in discarded_edit_window_line_numbers:
                continue
            if line_number in seen_line_numbers:
                continue
            seen_line_numbers.add(line_number)
            candidate_line_numbers.append(line_number)
        if not candidate_line_numbers and preferred_source_path:
            source_text = self._artifact_source_text_for_path(state, preferred_source_path)
            source_line_count = len(source_text.splitlines()) if source_text else 0
            candidate_line_numbers = list(range(1, min(source_line_count, 16) + 1))
        replacement_anchor_line_numbers = (
            self._artifact_replacement_anchor_line_numbers(
                state,
                preferred_source_path,
                candidate_line_numbers,
            )
            if preferred_source_path and candidate_line_numbers
            else []
        )
        valid_line_numbers = replacement_anchor_line_numbers or candidate_line_numbers
        if source_lines_text and valid_line_numbers:
            prioritized_source_lines_excerpt = self._artifact_source_lines_excerpt(
                source_lines_text,
                target_line_numbers=valid_line_numbers[:12],
            )
            if prioritized_source_lines_excerpt:
                source_lines_excerpt = prioritized_source_lines_excerpt
        command_skeletons: list[str] = []
        if preferred_source_path and valid_line_numbers:
            quoted_source_path = shlex.quote(preferred_source_path)
            for line_number in valid_line_numbers[:6]:
                command_skeletons.append(
                    f"{builder_command} --path {quoted_source_path} --replace-line {line_number} "
                    f"--with '<full replacement source line>' > {artifact_path}"
                )
        payload: dict[str, object] = {
            "artifact_path": artifact_path,
            "builder_command": builder_command,
            "required_command_shape": artifact_repair_contracts.required_command_shape(
                artifact_repair_contracts.contract_from_state(state)
            ),
            "allowed_source_paths": source_paths,
            "forbidden_source_paths": [artifact_path],
        }
        if preferred_source_path:
            payload["preferred_source_path"] = preferred_source_path
        if valid_line_numbers:
            payload["valid_line_numbers_preview"] = valid_line_numbers[:32]
        if prompt_line_numbers:
            payload["prompt_line_numbers_preview"] = prompt_line_numbers[:16]
        if discarded_edit_window_line_numbers:
            payload["discarded_edit_window_line_numbers"] = sorted(discarded_edit_window_line_numbers)[:16]
        if replacement_anchor_line_numbers:
            payload["anchor_line_policy"] = "replacement_safe_source_lines"
        if command_skeletons:
            payload["command_skeletons"] = command_skeletons
        if last_source_path:
            payload["last_source_path"] = last_source_path
        if source_lines_path:
            payload["source_lines_path"] = source_lines_path
        if edit_windows:
            payload["edit_windows"] = edit_windows[:2400]
        if source_lines_excerpt:
            payload["source_lines_excerpt"] = source_lines_excerpt[:2400]
        return payload

    @staticmethod
    def _artifact_line_numbers_from_prompt(prompt: str) -> list[int]:
        numbers: list[int] = []
        seen: set[int] = set()

        def add_line(value: int) -> None:
            if value <= 0 or value in seen:
                return
            seen.add(value)
            numbers.append(value)

        def add_range(start: int, end: int) -> None:
            if start <= 0 or end <= 0:
                return
            if end < start:
                start, end = end, start
            if end - start <= 16:
                for value in range(start, end + 1):
                    add_line(value)
                return
            add_line(start)
            add_line(end)

        text = str(prompt or "")
        for match in re.finditer(r"#L(\d+)(?:-L?(\d+))?", text, flags=re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            add_range(start, end)
        for match in re.finditer(r"\bL(\d+)(?:-L?(\d+))?\b", text, flags=re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            add_range(start, end)
        for match in re.finditer(r"\blines?\s+(\d+)(?:\s*[-:]\s*(\d+))?", text, flags=re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            add_range(start, end)
        return numbers[:64]

    @classmethod
    def _artifact_replacement_anchor_line_numbers(
        cls,
        state: AgentState,
        path: str,
        line_numbers: list[int],
    ) -> list[int]:
        if not line_numbers:
            return []
        source_by_number = cls._artifact_source_lines_by_number(state, path)
        if not source_by_number:
            return []
        selected: list[int] = []
        seen: set[int] = set()
        for raw_line_number in line_numbers:
            line_number = int(raw_line_number)
            if line_number in seen:
                continue
            seen.add(line_number)
            source_line = source_by_number.get(line_number)
            if source_line is None:
                continue
            if cls._artifact_replacement_anchor_line_safe(source_line):
                selected.append(line_number)
        return selected

    @staticmethod
    def _artifact_replacement_anchor_line_safe(source_line: str) -> bool:
        stripped = str(source_line).strip()
        if not stripped:
            return False
        if stripped.startswith(("#", "@")):
            return False
        if stripped.startswith(("def ", "async def ", "class ")):
            return False
        if stripped.startswith(('"""', "'''", '"', "'", ")", "]", "}", ",")):
            return False
        if stripped.startswith(("import ", "from ")):
            return False
        if stripped in {"pass", "break", "continue"}:
            return False
        return True

    @staticmethod
    def _artifact_retry_context_prompt(context: dict[str, object]) -> str:
        if not context:
            return ""
        lines = [
            "Artifact repair context:",
            f"- artifact_path: {context.get('artifact_path', '')}",
            f"- required_command_shape: {context.get('required_command_shape', '')}",
        ]
        allowed_source_paths = context.get("allowed_source_paths", [])
        if isinstance(allowed_source_paths, list) and allowed_source_paths:
            lines.append("- allowed_source_paths: " + ", ".join(str(path) for path in allowed_source_paths[:8]))
        forbidden_source_paths = context.get("forbidden_source_paths", [])
        if isinstance(forbidden_source_paths, list) and forbidden_source_paths:
            lines.append("- forbidden_source_paths: " + ", ".join(str(path) for path in forbidden_source_paths[:8]))
        preferred_source_path = str(context.get("preferred_source_path", "")).strip()
        if preferred_source_path:
            lines.append(f"- preferred_source_path: {preferred_source_path}")
        valid_line_numbers = context.get("valid_line_numbers_preview", [])
        if isinstance(valid_line_numbers, list) and valid_line_numbers:
            lines.append("- valid_line_numbers_preview: " + ", ".join(str(number) for number in valid_line_numbers[:16]))
        prompt_line_numbers = context.get("prompt_line_numbers_preview", [])
        if isinstance(prompt_line_numbers, list) and prompt_line_numbers:
            lines.append("- prompt_line_numbers_preview: " + ", ".join(str(number) for number in prompt_line_numbers[:16]))
        anchor_line_policy = str(context.get("anchor_line_policy", "")).strip()
        if anchor_line_policy:
            lines.append(f"- anchor_line_policy: {anchor_line_policy}")
        command_skeletons = context.get("command_skeletons", [])
        if isinstance(command_skeletons, list) and command_skeletons:
            lines.append("command_skeletons:\n" + "\n".join(str(command) for command in command_skeletons[:6]))
        source_lines_path = str(context.get("source_lines_path", "")).strip()
        if source_lines_path:
            lines.append(f"- source_lines_path: {source_lines_path}")
        edit_windows = str(context.get("edit_windows", "")).strip()
        if edit_windows:
            lines.append("edit_windows:\n" + edit_windows)
        source_lines_excerpt = str(context.get("source_lines_excerpt", "")).strip()
        if source_lines_excerpt:
            lines.append("source_lines_excerpt:\n" + source_lines_excerpt)
        return "\n".join(lines)

    @staticmethod
    def _artifact_first_setup_source_lines_path(setup_file_contents: dict[object, object]) -> str:
        for raw_path, content in setup_file_contents.items():
            path = str(raw_path).strip().strip("/")
            if path.startswith("source_lines/") and path.endswith(".lines") and isinstance(content, str):
                return path
        return ""

    @classmethod
    def _artifact_verified_edit_windows(cls, state: AgentState, path: str, edit_windows: str) -> tuple[str, set[int]]:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path:
            return str(edit_windows or ""), set()
        source_by_number = cls._artifact_source_lines_by_number(state, normalized_path)
        if not source_by_number:
            return str(edit_windows or ""), set()
        filtered_lines: list[str] = []
        active_path = ""
        conflicted_lines: set[int] = set()
        for raw_line in str(edit_windows or "").splitlines():
            header = re.match(r"\s*###\s+([^:]+)::", raw_line)
            if header:
                active_path = str(header.group(1)).strip().strip("/")
                filtered_lines.append(raw_line)
                continue
            line_match = re.match(r"^(\s*)(\d+):\s?(.*)$", raw_line)
            if not line_match:
                filtered_lines.append(raw_line)
                continue
            line_number = int(line_match.group(2))
            window_source = str(line_match.group(3)).rstrip()
            if active_path and active_path != normalized_path:
                filtered_lines.append(raw_line)
                continue
            canonical_source = source_by_number.get(line_number)
            if canonical_source is None:
                conflicted_lines.add(line_number)
                continue
            if cls._artifact_source_line_equivalent(window_source, canonical_source):
                filtered_lines.append(raw_line)
            else:
                conflicted_lines.add(line_number)
        if conflicted_lines:
            filtered_lines.append(
                "# discarded "
                f"{len(conflicted_lines)} edit-window anchors that conflicted with canonical source_lines"
            )
        return "\n".join(filtered_lines).strip(), conflicted_lines

    @classmethod
    def _artifact_verified_edit_windows_text(cls, state: AgentState, path: str, edit_windows: str) -> str:
        verified_text, _discarded_line_numbers = cls._artifact_verified_edit_windows(state, path, edit_windows)
        return verified_text

    @staticmethod
    def _artifact_source_line_equivalent(left: str, right: str) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"\s+", " ", str(value).strip())

        return normalize(left) == normalize(right)

    @staticmethod
    def _artifact_line_numbers_from_text(text: str) -> list[int]:
        numbers: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"(?m)^\s*(\d+):", str(text)):
            value = int(match.group(1))
            if value not in seen:
                seen.add(value)
                numbers.append(value)
        for match in re.finditer(r"\blines\s+(\d+)-(\d+)", str(text)):
            start = int(match.group(1))
            end = int(match.group(2))
            for value in (start, end):
                if value not in seen:
                    seen.add(value)
                    numbers.append(value)
        return numbers[:32]

    @staticmethod
    def _artifact_source_lines_excerpt(source_lines_text: str, *, target_line_numbers: list[int]) -> str:
        lines = str(source_lines_text or "").splitlines()
        if not lines:
            return ""
        selected_indexes: set[int] = set()
        ordered_indexes: list[int] = []
        target_numbers: list[int] = []
        seen_targets: set[int] = set()
        for value in target_line_numbers:
            target = int(value)
            if target <= 0 or target in seen_targets:
                continue
            seen_targets.add(target)
            target_numbers.append(target)
        if target_numbers:
            for target in target_numbers:
                for index, line in enumerate(lines):
                    match = re.match(r"\s*(\d+):", line)
                    if not match:
                        continue
                    line_number = int(match.group(1))
                    if abs(line_number - target) <= 4 and index not in selected_indexes:
                        selected_indexes.add(index)
                        ordered_indexes.append(index)
        if not selected_indexes:
            ordered_indexes = list(range(min(60, len(lines))))
        excerpt_lines = [lines[index] for index in ordered_indexes]
        excerpt = "\n".join(excerpt_lines)
        if len(excerpt) <= 2400:
            return excerpt
        return excerpt[:2397].rstrip() + "..."

    @classmethod
    def _artifact_noop_retry_needed(cls, state: AgentState, command: str) -> bool:
        normalized = str(command).strip()
        if not normalized:
            return True
        if cls._artifact_non_materializing_repair_command(state, normalized) and not cls._artifact_materializes(
            state,
            normalized,
        ):
            return True
        prior_noop_commands = cls._artifact_prior_noop_failure_commands(state)
        if normalized in prior_noop_commands:
            return True
        if cls._artifact_builder_repeats_prior_noop_operation(state, normalized):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_low_semantic_signal(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_broad_replacement(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_removes_definition_header(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_uses_artifact_as_source(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_uses_unknown_source_path(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(
            state,
            normalized,
        ) and cls._artifact_builder_line_out_of_known_source_range(state, normalized):
            return True
        if cls._artifact_builder_command_prefix(
            state,
            normalized,
        ) and cls._artifact_builder_has_placeholder_replacement(state, normalized):
            return True
        if cls._artifact_builder_command_prefix(
            state,
            normalized,
        ) and cls._artifact_builder_has_comment_only_replacement(state, normalized):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_has_escaped_newline_replacement(
            state,
            normalized,
        ):
            return True
        if cls._artifact_builder_command_prefix(state, normalized) and cls._artifact_builder_replaces_existing_source(
            state,
            normalized,
        ):
            return True
        return False

    @staticmethod
    def _artifact_noop_repair_context(state: AgentState) -> bool:
        if not LLMDecisionPolicy._artifact_repair_context(state):
            return False
        for step in reversed(state.history):
            verification = step.verification if isinstance(step.verification, dict) else {}
            reasons = verification.get("reasons", [])
            reason_text = artifact_repair_contracts.reason_text(reasons)
            if artifact_repair_contracts.is_noop_repair_reason_text(reason_text):
                return True
        return False

    @staticmethod
    def _artifact_latest_repairable_failure(state: AgentState) -> bool:
        if not LLMDecisionPolicy._artifact_repair_context(state):
            return False
        if not any(step.action == CODE_EXECUTE for step in state.history):
            return True
        if not any(
            step.action == CODE_EXECUTE and LLMDecisionPolicy._artifact_materializes(state, str(step.content))
            for step in state.history
        ):
            return True
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        if LLMDecisionPolicy._artifact_diagnosis_has_repairable_failure(
            state,
            artifact_path=artifact_path,
        ):
            return True
        for step in reversed(state.history):
            if step.action != CODE_EXECUTE:
                continue
            verification = step.verification if isinstance(step.verification, dict) else {}
            reasons = verification.get("reasons", [])
            reason_text = artifact_repair_contracts.reason_text(reasons)
            return artifact_repair_contracts.is_repairable_reason_text(reason_text, artifact_path=artifact_path)
        return False

    @staticmethod
    def _artifact_diagnosis_has_repairable_failure(state: AgentState, *, artifact_path: str) -> bool:
        diagnoses = state.subgoal_diagnoses if isinstance(state.subgoal_diagnoses, dict) else {}
        if not diagnoses:
            return False
        ordered: list[object] = []
        active = str(state.active_subgoal).strip()
        if active and active in diagnoses:
            ordered.append(diagnoses[active])
        ordered.extend(value for key, value in diagnoses.items() if key != active)
        for diagnosis in ordered:
            if not isinstance(diagnosis, dict):
                continue
            text_parts = [
                str(diagnosis.get("summary", "")),
                str(diagnosis.get("repair_instruction", "")),
            ]
            reason_text = "\n".join(part for part in text_parts if part.strip())
            if artifact_repair_contracts.is_repairable_reason_text(reason_text, artifact_path=artifact_path):
                return True
            signals = diagnosis.get("signals", [])
            normalized_signals = {
                str(signal).strip()
                for signal in signals
                if str(signal).strip()
            } if isinstance(signals, list) else set()
            if str(diagnosis.get("path", "")).strip() == artifact_path and "verifier_failure" in normalized_signals:
                return True
        return False

    @staticmethod
    def _artifact_latest_required_identifier_failure(state: AgentState) -> str:
        if not LLMDecisionPolicy._artifact_repair_context(state):
            return ""
        for step in reversed(state.history):
            if step.action != CODE_EXECUTE:
                continue
            verification = step.verification if isinstance(step.verification, dict) else {}
            reasons = verification.get("reasons", [])
            return artifact_repair_contracts.required_identifier_from_reasons(reasons)
        return ""

    @staticmethod
    def _artifact_command_references_identifier(command: str, identifier: str) -> bool:
        normalized_identifier = str(identifier).strip()
        if not normalized_identifier:
            return True
        return normalized_identifier in str(command)

    @staticmethod
    def _artifact_prior_noop_failure_commands(state: AgentState) -> set[str]:
        commands: set[str] = set()
        for step in state.history:
            if step.action != CODE_EXECUTE:
                continue
            verification = step.verification if isinstance(step.verification, dict) else {}
            reasons = verification.get("reasons", [])
            reason_text = artifact_repair_contracts.reason_text(reasons)
            if artifact_repair_contracts.is_noop_repair_reason_text(reason_text):
                content = str(step.content).strip()
                if content:
                    commands.add(content)
        return commands

    @staticmethod
    def _artifact_prior_noop_failure_operation_signatures(state: AgentState) -> set[tuple[str, int, int, tuple[str, ...]]]:
        signatures: set[tuple[str, int, int, tuple[str, ...]]] = set()
        for step in state.history:
            if step.action != CODE_EXECUTE:
                continue
            verification = step.verification if isinstance(step.verification, dict) else {}
            reasons = verification.get("reasons", [])
            reason_text = artifact_repair_contracts.reason_text(reasons)
            if not artifact_repair_contracts.is_noop_repair_reason_text(reason_text):
                continue
            spec = LLMDecisionPolicy._artifact_builder_command_spec(state, str(step.content), require_redirect=False)
            if spec is None:
                continue
            path = str(spec.get("path") or "")
            for start_line, end_line, replacement_lines in spec.get("operations", []):
                normalized_replacement = tuple(str(line).strip() for line in replacement_lines)
                signatures.add((path, int(start_line), int(end_line), normalized_replacement))
        return signatures

    @staticmethod
    def _artifact_builder_repeats_prior_noop_operation(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        prior_signatures = LLMDecisionPolicy._artifact_prior_noop_failure_operation_signatures(state)
        if not prior_signatures:
            return False
        path = str(spec.get("path") or "")
        proposed: set[tuple[str, int, int, tuple[str, ...]]] = set()
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            proposed.add((path, int(start_line), int(end_line), tuple(str(line).strip() for line in replacement_lines)))
        return bool(proposed) and proposed.issubset(prior_signatures)

    @staticmethod
    def _artifact_builder_command_spec(
        state: AgentState,
        command: str,
        *,
        require_redirect: bool,
    ) -> dict[str, object] | None:
        contract = artifact_repair_contracts.contract_from_state(state)
        return artifact_repair_contracts.parse_builder_command(
            command,
            artifact_path=contract.artifact_path if contract is not None else "patch.diff",
            builder_commands=(
                contract.builder_commands
                if contract is not None
                else artifact_repair_contracts.PATCH_BUILDER_COMMANDS
            ),
            require_redirect=require_redirect,
        )

    @staticmethod
    def _artifact_builder_with_payloads(state: AgentState, command: str) -> list[str]:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return []
        payloads: list[str] = []
        for _start_line, _end_line, replacement_lines in spec.get("operations", []):
            payloads.extend(str(line) for line in replacement_lines)
        return payloads

    @classmethod
    def _artifact_builder_low_semantic_signal(cls, state: AgentState, command: str) -> bool:
        payloads = [
            payload.strip()
            for payload in cls._artifact_builder_with_payloads(state, command)
            if payload.strip()
        ]
        if not payloads:
            return True
        low_signal_prefixes = (
            "def ",
            "async def ",
            "class ",
            "@",
            "import ",
            "from ",
            "#",
            '"""',
            "'''",
        )
        return all(payload.startswith(low_signal_prefixes) for payload in payloads)

    @staticmethod
    def _artifact_expanded_replacement_lines(replacement_lines: object) -> list[str]:
        expanded: list[str] = []
        for raw_line in list(replacement_lines or []):
            line = str(raw_line).rstrip("\n")
            if "\n" in line:
                expanded.extend(line.splitlines())
            else:
                expanded.append(line)
        return expanded

    @staticmethod
    def _artifact_builder_broad_replacement(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            source_span = int(end_line) - int(start_line) + 1
            expanded_replacement_lines = LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines)
            replacement_span = len(expanded_replacement_lines)
            first_replacement = ""
            for line in expanded_replacement_lines:
                first_replacement = str(line).strip()
                if first_replacement:
                    break
            if source_span >= 20 or replacement_span >= 20:
                return True
            if source_span >= 8 and first_replacement.startswith(("def ", "async def ", "class ")):
                return True
        return False

    @staticmethod
    def _artifact_narrow_broad_builder_command(
        state: AgentState,
        command: str,
        *,
        artifact_path: str,
    ) -> str:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return ""
        builder = str(spec.get("builder") or "patch_builder").strip() or "patch_builder"
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return ""
        source_by_number = LLMDecisionPolicy._artifact_source_lines_by_number(state, path)
        if not source_by_number:
            return ""
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            source_span = int(end_line) - int(start_line) + 1
            replacement_values = LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines)
            if source_span < 8 and len(replacement_values) < 8:
                continue
            for offset, replacement_line in enumerate(replacement_values):
                line_number = int(start_line) + offset
                original_line = source_by_number.get(line_number)
                if original_line is None:
                    continue
                stripped = replacement_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith(("def ", "async def ", "class ", "@")):
                    continue
                if path.endswith(".py") and not LLMDecisionPolicy._artifact_python_line_shape_compatible(
                    str(original_line),
                    replacement_line,
                ):
                    continue
                if replacement_line.rstrip() == str(original_line).rstrip():
                    continue
                candidate = (
                    f"{builder} --path {shlex.quote(path)} --replace-line {line_number} "
                    f"--with {shlex.quote(replacement_line)} > {artifact_path}"
                )
                if not LLMDecisionPolicy._artifact_builder_guard_rejection_reason(state, candidate):
                    return candidate
        return ""

    @staticmethod
    def _artifact_python_line_shape_compatible(original_line: str, replacement_line: str) -> bool:
        original = str(original_line).strip()
        replacement = str(replacement_line).strip()
        if not original or not replacement:
            return False
        string_prefixes = ('"""', "'''", '"', "'")
        if replacement.startswith(string_prefixes):
            return False
        if original.startswith(string_prefixes):
            return False

        def keyword_shape(value: str) -> str:
            for prefix, shape in (
                ("async def ", "async def"),
                ("def ", "def"),
                ("class ", "class"),
                ("if ", "if"),
                ("elif ", "elif"),
                ("else:", "else"),
                ("for ", "for"),
                ("while ", "while"),
                ("try:", "try"),
                ("except ", "except"),
                ("finally:", "finally"),
                ("with ", "with"),
                ("return ", "return"),
                ("raise ", "raise"),
                ("yield ", "yield"),
                ("assert ", "assert"),
                ("import ", "import"),
                ("from ", "from"),
            ):
                if value.startswith(prefix):
                    return shape
            return ""

        original_shape = keyword_shape(original)
        replacement_shape = keyword_shape(replacement)
        if original_shape or replacement_shape:
            return original_shape == replacement_shape
        if ("=" in original) != ("=" in replacement):
            return False
        return True

    @staticmethod
    def _artifact_python_header_signature_range(
        state: AgentState,
        path: str,
        *,
        start_line: int,
    ) -> dict[str, object]:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path.endswith(".py"):
            return {}
        source_text = LLMDecisionPolicy._artifact_python_source_text_for_path(state, normalized_path)
        if not source_text or LLMDecisionPolicy._artifact_python_compile_error(source_text, normalized_path):
            return {}
        source_lines = source_text.splitlines()
        if start_line < 1 or start_line > len(source_lines):
            return {}
        first_line = source_lines[start_line - 1]
        first_stripped = first_line.strip()
        if not first_stripped.startswith(("def ", "async def ", "class ")):
            return {}
        bracket_depth = 0
        for index in range(start_line - 1, min(len(source_lines), start_line + 40)):
            line = source_lines[index]
            for char in line:
                if char in "([{":
                    bracket_depth += 1
                elif char in ")]}":
                    bracket_depth = max(0, bracket_depth - 1)
            if line.strip().endswith(":") and bracket_depth == 0:
                return {
                    "start_line": int(start_line),
                    "end_line": index + 1,
                    "source": "\n".join(source_lines[start_line - 1 : index + 1]),
                }
        return {}

    @staticmethod
    def _artifact_expand_partial_header_signature_builder_command(
        state: AgentState,
        command: str,
        *,
        artifact_path: str,
    ) -> str:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return ""
        operations = list(spec.get("operations", []))
        if len(operations) != 1:
            return ""
        start_line, end_line, replacement_lines = operations[0]
        builder = str(spec.get("builder") or "patch_builder").strip() or "patch_builder"
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return ""
        replacement_values = LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines)
        if len(replacement_values) != 1:
            return ""
        replacement = str(replacement_values[0]).rstrip("\n")
        if not replacement.strip().startswith(("def ", "async def ", "class ")):
            return ""
        if not replacement.strip().endswith(":"):
            return ""
        signature_range = LLMDecisionPolicy._artifact_python_header_signature_range(
            state,
            path,
            start_line=int(start_line),
        )
        signature_end = int(signature_range.get("end_line", 0) or 0)
        if signature_end <= int(end_line):
            return ""
        candidate = (
            f"{builder} --path {shlex.quote(path)} --replace-lines {int(start_line)} {signature_end} "
            f"--with {shlex.quote(replacement)} > {artifact_path}"
        )
        if LLMDecisionPolicy._artifact_builder_guard_rejection_reason(state, candidate):
            return ""
        return candidate

    @staticmethod
    def _artifact_split_escaped_newline_builder_command(
        state: AgentState,
        command: str,
        *,
        artifact_path: str,
    ) -> str:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=True)
        if spec is None:
            return ""
        operations = list(spec.get("operations", []))
        if len(operations) != 1:
            return ""
        start_line, _end_line, replacement_lines = operations[0]
        split_replacements: list[str] = []
        saw_escaped_newline = False
        for raw_line in replacement_lines:
            line = str(raw_line).rstrip("\n")
            if "\\n" in line:
                saw_escaped_newline = True
                split_replacements.extend(line.split("\\n"))
            else:
                split_replacements.append(line)
        if not saw_escaped_newline or not split_replacements:
            return ""
        builder = str(spec.get("builder") or "patch_builder").strip() or "patch_builder"
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return ""
        start = int(start_line)
        end = start + len(split_replacements) - 1
        candidate = (
            f"{builder} --path {shlex.quote(path)} --replace-lines {start} {end} "
            + " ".join(f"--with {shlex.quote(line)}" for line in split_replacements)
            + f" > {artifact_path}"
        )
        if LLMDecisionPolicy._artifact_builder_guard_rejection_reason(state, candidate):
            return ""
        return candidate

    @staticmethod
    def _artifact_broad_builder_last_resort_allowed(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=True)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return False
        if LLMDecisionPolicy._artifact_builder_uses_artifact_as_source(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_uses_unknown_source_path(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_line_out_of_known_source_range(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_has_placeholder_replacement(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_has_comment_only_replacement(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_replaces_existing_source(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_removes_definition_header(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_has_escaped_newline_replacement(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_would_produce_invalid_python(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_python_ast_unchanged(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_unused_python_parameter_names(state, command):
            return False
        if LLMDecisionPolicy._artifact_builder_invalid_init_return_names(state, command):
            return False
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            source_span = int(end_line) - int(start_line) + 1
            replacement_span = len(LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines))
            if source_span > 80 or replacement_span > 80:
                return False
        return True

    @staticmethod
    def _artifact_builder_removes_definition_header(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return False
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, path)
        if not source_text:
            return False
        source_lines = source_text.splitlines()
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                continue
            original_headers = [
                line.strip()
                for line in source_lines[start_line - 1 : end_line]
                if line.strip().startswith(("def ", "async def ", "class "))
            ]
            replacement_header_set = {
                str(line).strip()
                for line in LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines)
                if str(line).strip().startswith(("def ", "async def ", "class "))
            }
            replacement_line_count = len(LLMDecisionPolicy._artifact_expanded_replacement_lines(replacement_lines))
            source_span = int(end_line) - int(start_line) + 1
            if replacement_header_set and not original_headers:
                return True
            if replacement_header_set and original_headers and source_span == 1 and replacement_line_count > 1:
                return True
            original_header_kinds = {
                "async def"
                if header.startswith("async def ")
                else "def"
                if header.startswith("def ")
                else "class"
                for header in original_headers
            }
            replacement_header_kinds = {
                "async def"
                if header.startswith("async def ")
                else "def"
                if header.startswith("def ")
                else "class"
                for header in replacement_header_set
            }
            if original_headers and not original_header_kinds.issubset(replacement_header_kinds):
                return True
            original_header_names = {
                name
                for header in original_headers
                for name in [LLMDecisionPolicy._artifact_definition_header_name(header)]
                if name
            }
            replacement_header_names = {
                name
                for header in replacement_header_set
                for name in [LLMDecisionPolicy._artifact_definition_header_name(header)]
                if name
            }
            if original_header_names and not original_header_names.issubset(replacement_header_names):
                return True
        return False

    @staticmethod
    def _artifact_definition_header_name(header: str) -> str:
        match = re.match(r"\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", str(header))
        return match.group(1) if match else ""

    @staticmethod
    def _artifact_builder_command_prefix(state: AgentState, command: str) -> bool:
        normalized = str(command).strip()
        if not normalized:
            return False
        contract = artifact_repair_contracts.contract_from_state(state)
        builder_commands = (
            contract.builder_commands if contract is not None else artifact_repair_contracts.PATCH_BUILDER_COMMANDS
        )
        return any(normalized.startswith(f"{builder} ") for builder in builder_commands)

    @staticmethod
    def _artifact_builder_uses_artifact_as_source(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        return str(spec.get("path") or "").strip().strip("/") == str(artifact_path).strip().strip("/")

    @staticmethod
    def _artifact_builder_uses_unknown_source_path(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return False
        known_paths = {str(candidate).strip().strip("/") for candidate in LLMDecisionPolicy._artifact_source_context_by_path(state)}
        if not known_paths:
            return False
        return path not in known_paths

    @staticmethod
    def _artifact_builder_line_out_of_known_source_range(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return False
        known_line_numbers = LLMDecisionPolicy._artifact_source_line_numbers_for_path(state, path)
        if known_line_numbers:
            minimum = min(known_line_numbers)
            maximum = max(known_line_numbers)
            for start_line, end_line, _replacement_lines in spec.get("operations", []):
                if start_line < minimum or end_line < start_line or end_line > maximum:
                    return True
            return False
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, path)
        if not source_text:
            return False
        line_count = len(source_text.splitlines())
        for start_line, end_line, _replacement_lines in spec.get("operations", []):
            if start_line < 1 or end_line < start_line or end_line > line_count:
                return True
        return False

    @staticmethod
    def _artifact_builder_has_placeholder_replacement(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        placeholder_literals = {
            "...",
            "changed",
            "new_code",
            "new_code_here",
            "replacement",
            "replacement_code",
            "todo",
            "your_code_here",
        }
        for _start_line, _end_line, replacement_lines in spec.get("operations", []):
            stripped_lines = [str(line).strip() for line in replacement_lines if str(line).strip()]
            if not stripped_lines:
                continue
            lowered = [line.lower().strip("'\"") for line in stripped_lines]
            if all(line in placeholder_literals for line in lowered):
                return True
            if all(re.match(r"print\((['\"])(test|hello)\1\)", line, re.IGNORECASE) for line in stripped_lines):
                return True
            if all(line in {"assert True", "assert False"} for line in stripped_lines):
                return True
        return False

    @staticmethod
    def _artifact_builder_has_comment_only_replacement(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        for _start_line, _end_line, replacement_lines in spec.get("operations", []):
            stripped_lines = [str(line).strip() for line in replacement_lines if str(line).strip()]
            if stripped_lines and all(line.startswith("#") for line in stripped_lines):
                return True
        return False

    @staticmethod
    def _artifact_builder_guard_rejection_details(
        state: AgentState,
        reason: str,
        command: str,
    ) -> list[dict[str, object]]:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return []
        path = str(spec.get("path") or "").strip().strip("/")
        known_line_numbers = LLMDecisionPolicy._artifact_source_line_numbers_for_path(state, path) if path else []
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, path) if path else ""
        line_count = len(source_text.splitlines()) if source_text else 0
        details: list[dict[str, object]] = []
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            replacement_preview = "\n".join(str(line).rstrip("\n") for line in replacement_lines)
            detail: dict[str, object] = {
                "reason": str(reason),
                "path": path,
                "start_line": int(start_line),
                "end_line": int(end_line),
                "replacement_preview": replacement_preview[:800],
            }
            if line_count:
                detail["known_source_line_count"] = line_count
            if known_line_numbers:
                detail["known_source_line_min"] = min(known_line_numbers)
                detail["known_source_line_max"] = max(known_line_numbers)
            if reason == "python_ast_noop":
                detail["semantic_issue"] = "python AST unchanged after ignoring docstrings/comments"
            elif reason == "unused_signature_parameter":
                unused_params = LLMDecisionPolicy._artifact_builder_unused_python_parameter_names(state, command)
                if unused_params:
                    detail["unused_parameters"] = unused_params[:8]
            elif reason == "invalid_init_return_value":
                invalid_inits = LLMDecisionPolicy._artifact_builder_invalid_init_return_names(state, command)
                if invalid_inits:
                    detail["invalid_init_returns"] = invalid_inits[:8]
            elif reason == "invalid_init_generator":
                invalid_generators = LLMDecisionPolicy._artifact_builder_invalid_init_generator_names(state, command)
                if invalid_generators:
                    detail["invalid_init_generators"] = invalid_generators[:8]
            elif reason == "local_use_before_assignment":
                unbound_locals = LLMDecisionPolicy._artifact_builder_local_load_before_assignment_names(state, command)
                if unbound_locals:
                    detail["local_use_before_assignment"] = unbound_locals[:8]
            details.append(detail)
        if not details and path:
            details.append({"reason": str(reason), "path": path, "known_source_line_count": line_count})
        return details

    @staticmethod
    def _artifact_builder_has_escaped_newline_replacement(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        for _start_line, _end_line, replacement_lines in spec.get("operations", []):
            if any("\\n" in str(line) for line in replacement_lines):
                return True
        return False

    @staticmethod
    def _artifact_non_materializing_repair_command(state: AgentState, command: str) -> bool:
        normalized = str(command).strip()
        if not normalized:
            return False
        if LLMDecisionPolicy._artifact_builder_command_prefix(state, normalized):
            return True
        return artifact_repair_contracts.command_is_source_inspection(normalized)

    @staticmethod
    def _artifact_source_context_available(state: AgentState) -> bool:
        return bool(LLMDecisionPolicy._artifact_source_context_by_path(state))

    @staticmethod
    def _artifact_source_context_by_path(state: AgentState) -> dict[str, str]:
        contract = artifact_repair_contracts.contract_from_state(state)
        if contract is None:
            return artifact_repair_contracts.source_context_by_path(state.task)
        return dict(contract.source_context_by_path or {})

    @staticmethod
    def _artifact_source_text_for_path(state: AgentState, path: str) -> str:
        candidates = LLMDecisionPolicy._artifact_source_text_candidates_for_path(state, path)
        return candidates[0] if candidates else ""

    @staticmethod
    def _artifact_python_source_text_for_path(state: AgentState, path: str) -> str:
        base_source = LLMDecisionPolicy._artifact_swe_base_source_text(state, path)
        if base_source:
            if not LLMDecisionPolicy._artifact_python_compile_error(
                base_source,
                str(path).strip().strip("/"),
            ):
                return base_source
        candidates = LLMDecisionPolicy._artifact_source_text_candidates_for_path(state, path)
        if not candidates:
            return ""
        if not str(path).strip().endswith(".py"):
            return candidates[0]
        for candidate in candidates:
            if LLMDecisionPolicy._artifact_python_compile_error(candidate, str(path).strip().strip("/")):
                continue
            return candidate
        return candidates[0]

    @staticmethod
    def _artifact_python_compile_error(source_text: str, filename: str) -> str:
        try:
            compile(source_text, filename or "<artifact-source>", "exec")
        except SyntaxError as exc:
            location = f" line {exc.lineno}" if exc.lineno else ""
            return f"{exc.msg}{location}".strip()
        return ""

    @staticmethod
    def _artifact_source_text_candidates_for_path(state: AgentState, path: str) -> list[str]:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path:
            return []
        candidates: list[str] = []

        def add_candidate(value: object) -> None:
            if not isinstance(value, str) or not value:
                return
            if value not in candidates:
                candidates.append(value)

        add_candidate(LLMDecisionPolicy._artifact_source_context_by_path(state).get(normalized_path, ""))
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        if not isinstance(setup_file_contents, dict):
            return candidates
        add_candidate(setup_file_contents.get(normalized_path))
        line_numbered = setup_file_contents.get(f"source_lines/{normalized_path}.lines")
        if isinstance(line_numbered, str):
            extracted: list[str] = []
            for raw_line in line_numbered.splitlines():
                match = re.match(r"\s*\d+:\s?(.*)$", raw_line)
                extracted.append(match.group(1) if match else raw_line)
            add_candidate("\n".join(extracted) + ("\n" if extracted else ""))
        return candidates

    @staticmethod
    def _artifact_swe_repo_cache_path(repo_cache_root: str, repo: str) -> Path | None:
        root = Path(repo_cache_root)
        candidates = [
            root / repo,
            root / repo.replace("/", "__"),
            root / repo.split("/")[-1],
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @classmethod
    def _artifact_swe_base_source_text(cls, state: AgentState, path: str) -> str:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path.endswith(".py"):
            return ""
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        verifier = metadata.get("semantic_verifier", {})
        if not isinstance(verifier, dict):
            return ""
        if verifier.get("kind") != "swe_patch_apply_check":
            return ""
        repo = str(verifier.get("repo", "")).strip()
        base_commit = str(verifier.get("base_commit", "")).strip()
        repo_cache_root = str(verifier.get("repo_cache_root", "")).strip()
        if not repo or not base_commit or not repo_cache_root:
            return ""
        repo_path = cls._artifact_swe_repo_cache_path(repo_cache_root, repo)
        if repo_path is None:
            return ""
        result = subprocess.run(
            ["git", "-C", str(repo_path), "show", f"{base_commit}:{normalized_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout

    @staticmethod
    def _artifact_source_line_numbers_for_path(state: AgentState, path: str) -> list[int]:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path:
            return []
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        numbers: list[int] = []
        seen: set[int] = set()
        if isinstance(setup_file_contents, dict):
            line_numbered = setup_file_contents.get(f"source_lines/{normalized_path}.lines")
            if isinstance(line_numbered, str):
                for raw_line in line_numbered.splitlines():
                    match = re.match(r"\s*(\d+):", raw_line)
                    if not match:
                        continue
                    value = int(match.group(1))
                    if value in seen:
                        continue
                    seen.add(value)
                    numbers.append(value)
        edit_windows = str(
            metadata.get("artifact_executable_edit_windows")
            or metadata.get("swe_executable_edit_windows")
            or ""
        )
        active_section_matches = not normalized_path
        for raw_line in edit_windows.splitlines():
            header = re.match(r"\s*###\s+([^:]+)::", raw_line)
            if header:
                active_section_matches = str(header.group(1)).strip().strip("/") == normalized_path
                if active_section_matches:
                    range_match = re.search(r"\blines\s+(\d+)\s*-\s*(\d+)\b", raw_line)
                    if range_match:
                        for raw_value in range_match.groups():
                            value = int(raw_value)
                            if value in seen:
                                continue
                            seen.add(value)
                            numbers.append(value)
                continue
            if not active_section_matches:
                continue
            match = re.match(r"\s*(\d+):", raw_line)
            if not match:
                continue
            value = int(match.group(1))
            if value in seen:
                continue
            seen.add(value)
            numbers.append(value)
        return numbers

    @staticmethod
    def _artifact_source_lines_by_number(state: AgentState, path: str) -> dict[int, str]:
        normalized_path = str(path).strip().strip("/")
        if not normalized_path:
            return {}
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        if isinstance(setup_file_contents, dict):
            line_numbered = setup_file_contents.get(f"source_lines/{normalized_path}.lines")
            if isinstance(line_numbered, str):
                by_number: dict[int, str] = {}
                for raw_line in line_numbered.splitlines():
                    match = re.match(r"\s*(\d+):\s?(.*)$", raw_line)
                    if not match:
                        continue
                    by_number[int(match.group(1))] = match.group(2)
                if by_number:
                    return by_number
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, normalized_path)
        if not source_text:
            return {}
        return {index: line for index, line in enumerate(source_text.splitlines(), start=1)}

    @staticmethod
    def _format_replacement_like_builder(
        source_lines: list[str],
        *,
        start_line: int,
        replacement_lines: list[str],
    ) -> list[str]:
        formatted: list[str] = []
        for offset, replacement in enumerate(replacement_lines):
            line = replacement if str(replacement).endswith("\n") else f"{replacement}\n"
            original_index = start_line - 1 + offset
            if original_index < len(source_lines) and line.strip() and not line[:1].isspace():
                original_indent = ""
                for char in source_lines[original_index]:
                    if char not in {" ", "\t"}:
                        break
                    original_indent += char
                if original_indent:
                    line = f"{original_indent}{line}"
            elif (
                original_index < len(source_lines)
                and line.strip()
                and line.startswith(" ")
                and not line.startswith("  ")
            ):
                original_indent = ""
                for char in source_lines[original_index]:
                    if char not in {" ", "\t"}:
                        break
                    original_indent += char
                if len(original_indent) >= 2:
                    line = f"{original_indent}{line.lstrip()}"
            formatted.append(line)
        return formatted

    @staticmethod
    def _artifact_builder_replaces_existing_source(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return False
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, path)
        if not source_text:
            return False
        source_lines = source_text.splitlines(keepends=True)
        operations = list(spec.get("operations", []))
        if not operations:
            return False
        for start_line, end_line, replacement_lines in operations:
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                return False
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines,
                start_line=start_line,
                replacement_lines=list(replacement_lines),
            )
            original = source_lines[start_line - 1 : end_line]
            if [line.rstrip("\n") for line in formatted] != [line.rstrip("\n") for line in original]:
                return False
        return True

    @staticmethod
    def _artifact_duplicate_line_guard_candidate(line: str) -> bool:
        stripped = str(line).strip()
        if len(stripped) < 8:
            return False
        if stripped.startswith(("#", "@", "\"", "'", ")", "]", "}")):
            return False
        if stripped in {"pass", "break", "continue"}:
            return False
        return True

    @staticmethod
    def _artifact_builder_introduces_adjacent_duplicate_line(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path.endswith(".py"):
            return False
        source_text = LLMDecisionPolicy._artifact_python_source_text_for_path(state, path)
        if not source_text:
            return False
        source_lines = source_text.splitlines(keepends=True)
        if not source_lines:
            return False
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                continue
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines,
                start_line=int(start_line),
                replacement_lines=list(replacement_lines),
            )
            formatted_stripped = [
                str(line).strip()
                for line in formatted
                if LLMDecisionPolicy._artifact_duplicate_line_guard_candidate(str(line))
            ]
            if not formatted_stripped:
                continue
            neighbors: set[str] = set()
            if int(start_line) > 1:
                before = source_lines[int(start_line) - 2].strip()
                if LLMDecisionPolicy._artifact_duplicate_line_guard_candidate(before):
                    neighbors.add(before)
            if int(end_line) < len(source_lines):
                after = source_lines[int(end_line)].strip()
                if LLMDecisionPolicy._artifact_duplicate_line_guard_candidate(after):
                    neighbors.add(after)
            if any(line in neighbors for line in formatted_stripped):
                return True
            for left, right in zip(formatted_stripped, formatted_stripped[1:]):
                if left == right:
                    return True
        return False

    @staticmethod
    def _artifact_builder_python_source_pair(
        state: AgentState,
        command: str,
    ) -> tuple[str, str, str]:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return "", "", ""
        path = str(spec.get("path") or "").strip().strip("/")
        if not path.endswith(".py"):
            return "", "", ""
        source_text = LLMDecisionPolicy._artifact_python_source_text_for_path(state, path)
        if not source_text:
            return "", "", ""
        if LLMDecisionPolicy._artifact_python_compile_error(source_text, path):
            return "", "", ""
        source_lines = source_text.splitlines(keepends=True)
        updated_lines = list(source_lines)
        operations = sorted(list(spec.get("operations", [])), key=lambda item: int(item[0]), reverse=True)
        for start_line, end_line, replacement_lines in operations:
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                return "", "", ""
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines,
                start_line=int(start_line),
                replacement_lines=list(replacement_lines),
            )
            updated_lines[int(start_line) - 1 : int(end_line)] = formatted
        updated_source = "".join(updated_lines)
        if LLMDecisionPolicy._artifact_python_compile_error(updated_source, path):
            return "", "", ""
        return path, source_text, updated_source

    @staticmethod
    def _artifact_builder_python_ast_unchanged(state: AgentState, command: str) -> bool:
        path, before_source, after_source = LLMDecisionPolicy._artifact_builder_python_source_pair(state, command)
        if not path or before_source == after_source:
            return False
        return not _python_executable_ast_changed(before_source, after_source)

    @staticmethod
    def _artifact_builder_unused_python_parameter_names(state: AgentState, command: str) -> list[str]:
        path, before_source, after_source = LLMDecisionPolicy._artifact_builder_python_source_pair(state, command)
        if not path or _is_python_test_path(path):
            return []
        return _unused_new_python_parameters(before_source, after_source)

    @staticmethod
    def _artifact_builder_invalid_init_return_names(state: AgentState, command: str) -> list[str]:
        path, before_source, after_source = LLMDecisionPolicy._artifact_builder_python_source_pair(state, command)
        if not path or _is_python_test_path(path):
            return []
        before_invalid = set(_python_init_return_value_names(before_source))
        after_invalid = set(_python_init_return_value_names(after_source))
        return sorted(after_invalid - before_invalid)

    @staticmethod
    def _artifact_builder_invalid_init_generator_names(state: AgentState, command: str) -> list[str]:
        path, before_source, after_source = LLMDecisionPolicy._artifact_builder_python_source_pair(state, command)
        if not path or _is_python_test_path(path):
            return []
        before_invalid = set(_python_init_generator_names(before_source))
        after_invalid = set(_python_init_generator_names(after_source))
        return sorted(after_invalid - before_invalid)

    @staticmethod
    def _artifact_builder_local_load_before_assignment_names(state: AgentState, command: str) -> list[str]:
        path, before_source, after_source = LLMDecisionPolicy._artifact_builder_python_source_pair(state, command)
        if not path or _is_python_test_path(path):
            return []
        before_invalid = set(_python_local_load_before_assignment_names(before_source))
        after_invalid = set(_python_local_load_before_assignment_names(after_source))
        return sorted(after_invalid - before_invalid)

    @staticmethod
    def _artifact_source_identical_operation_details(state: AgentState, command: str) -> list[dict[str, object]]:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return []
        path = str(spec.get("path") or "").strip().strip("/")
        if not path:
            return []
        source_text = LLMDecisionPolicy._artifact_source_text_for_path(state, path)
        if not source_text:
            return []
        source_lines = source_text.splitlines(keepends=True)
        details: list[dict[str, object]] = []
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                continue
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines,
                start_line=start_line,
                replacement_lines=list(replacement_lines),
            )
            original = source_lines[start_line - 1 : end_line]
            if [line.rstrip("\n") for line in formatted] != [line.rstrip("\n") for line in original]:
                continue
            existing_source = "\n".join(line.rstrip("\n") for line in original)
            rejected_replacement = "\n".join(line.rstrip("\n") for line in formatted)
            detail: dict[str, object] = {
                "path": path,
                "start_line": int(start_line),
                "end_line": int(end_line),
                "existing_source": existing_source[:800],
                "rejected_replacement": rejected_replacement[:800],
            }
            if path.endswith(".py"):
                source_text_plain = LLMDecisionPolicy._artifact_python_source_text_for_path(state, path) or "".join(
                    source_lines
                )
                statement_range = LLMDecisionPolicy._artifact_python_statement_range(
                    source_text_plain,
                    target_line=int(start_line),
                )
                if statement_range:
                    suggested_start = int(statement_range.get("start_line", start_line) or start_line)
                    suggested_end = int(statement_range.get("end_line", end_line) or end_line)
                    plain_lines = [line.rstrip("\n") for line in source_lines]
                    detail["suggested_statement_range"] = statement_range
                    detail["suggested_existing_source"] = "\n".join(
                        plain_lines[suggested_start - 1 : suggested_end]
                    )[:1200]
                    context_start = max(1, suggested_start - 3)
                    context_end = min(len(plain_lines), suggested_end + 3)
                    detail["nearby_source_context"] = "\n".join(
                        f"{line_no}: {plain_lines[line_no - 1]}"
                        for line_no in range(context_start, context_end + 1)
                    )[:1600]
            details.append(detail)
        return details

    @staticmethod
    def _artifact_python_statement_range(source_text: str, *, target_line: int) -> dict[str, object]:
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return {}
        best: tuple[int, int, str] | None = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start <= target_line <= end:
                candidate = (start, end, node.__class__.__name__)
                if best is None or (candidate[1] - candidate[0], candidate[0]) < (best[1] - best[0], best[0]):
                    best = candidate
        if best is None:
            return {}
        return {
            "start_line": best[0],
            "end_line": best[1],
            "node_type": best[2],
        }

    @staticmethod
    def _artifact_invalid_python_replacement_details(state: AgentState, command: str) -> list[dict[str, object]]:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return []
        path = str(spec.get("path") or "").strip().strip("/")
        if not path.endswith(".py"):
            return []
        source_text = LLMDecisionPolicy._artifact_python_source_text_for_path(state, path)
        if not source_text:
            return []
        if LLMDecisionPolicy._artifact_python_compile_error(source_text, path):
            return []
        source_lines = source_text.splitlines()
        source_lines_keepends = source_text.splitlines(keepends=True)
        updated_lines = list(source_lines_keepends)
        operations = sorted(list(spec.get("operations", [])), key=lambda item: int(item[0]), reverse=True)
        for start_line, end_line, replacement_lines in operations:
            if start_line < 1 or end_line < start_line or end_line > len(source_lines_keepends):
                return []
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines_keepends,
                start_line=start_line,
                replacement_lines=list(replacement_lines),
            )
            updated_lines[start_line - 1 : end_line] = formatted
        syntax_error = LLMDecisionPolicy._artifact_python_compile_error("".join(updated_lines), path)
        if not syntax_error:
            return []
        details: list[dict[str, object]] = []
        for start_line, end_line, replacement_lines in spec.get("operations", []):
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                continue
            statement_range = LLMDecisionPolicy._artifact_python_statement_range(
                source_text,
                target_line=int(start_line),
            )
            suggested_start = int(statement_range.get("start_line", start_line) or start_line)
            suggested_end = int(statement_range.get("end_line", end_line) or end_line)
            existing_source = "\n".join(source_lines[suggested_start - 1 : suggested_end])
            attempted_existing_source = "\n".join(source_lines[int(start_line) - 1 : int(end_line)])
            replacement_preview = "\n".join(str(line).rstrip("\n") for line in replacement_lines)
            first_replacement = ""
            for line in replacement_lines:
                first_replacement = str(line).strip()
                if first_replacement:
                    break
            first_existing = source_lines[int(start_line) - 1].strip() if source_lines else ""
            statement_keyword_prefixes = (
                "def ",
                "async def ",
                "class ",
                "return ",
                "if ",
                "for ",
                "while ",
                "try:",
                "with ",
                "import ",
                "from ",
            )
            detail: dict[str, object] = {
                "path": path,
                "attempted_start_line": int(start_line),
                "attempted_end_line": int(end_line),
                "attempted_existing_source": attempted_existing_source[:800],
                "syntax_error": syntax_error[:400],
                "replacement_line_count": len(list(replacement_lines)),
                "replacement_preview": replacement_preview[:800],
            }
            if statement_range:
                detail["suggested_statement_range"] = statement_range
                detail["suggested_existing_source"] = existing_source[:1200]
                statement_span = suggested_end - suggested_start + 1
                if statement_span > 8:
                    detail["large_enclosing_statement"] = True
                    detail["recommended_repair_shape"] = (
                        "Use a narrow syntactically valid expression/argument replacement at the original "
                        "indentation, or choose a smaller complete statement inside the same edit window; do "
                        "not replace the whole broad enclosing statement unless necessary."
                    )
            if (
                first_replacement.startswith(statement_keyword_prefixes)
                and first_existing
                and not first_existing.startswith(statement_keyword_prefixes)
            ):
                detail["replacement_syntax_kind_mismatch"] = {
                    "existing_line": first_existing[:400],
                    "replacement_first_line": first_replacement[:400],
                    "repair_hint": (
                        "Replacement starts with a statement header/keyword but the target line is an "
                        "expression or continuation line; replace it with the same syntactic kind or choose "
                        "a complete statement range."
                    ),
                }
            details.append(detail)
        return details

    @staticmethod
    def _artifact_builder_would_produce_invalid_python(state: AgentState, command: str) -> bool:
        spec = LLMDecisionPolicy._artifact_builder_command_spec(state, command, require_redirect=False)
        if spec is None:
            return False
        path = str(spec.get("path") or "").strip().strip("/")
        if not path.endswith(".py"):
            return False
        source_text = LLMDecisionPolicy._artifact_python_source_text_for_path(state, path)
        if not source_text:
            return False
        if LLMDecisionPolicy._artifact_python_compile_error(source_text, path):
            return False
        source_lines = source_text.splitlines(keepends=True)
        updated_lines = list(source_lines)
        operations = sorted(list(spec.get("operations", [])), key=lambda item: int(item[0]), reverse=True)
        for start_line, end_line, replacement_lines in operations:
            if start_line < 1 or end_line < start_line or end_line > len(source_lines):
                return False
            formatted = LLMDecisionPolicy._format_replacement_like_builder(
                source_lines,
                start_line=start_line,
                replacement_lines=list(replacement_lines),
            )
            updated_lines[start_line - 1 : end_line] = formatted
        if LLMDecisionPolicy._artifact_python_compile_error("".join(updated_lines), path):
            return True
        return False

    def _artifact_prior_source_inspection_count(self, state: AgentState) -> int:
        count = 0
        for step in state.history:
            if step.action != CODE_EXECUTE:
                continue
            content = str(step.content).strip()
            if self._artifact_non_materializing_repair_command(state, content) and not self._artifact_materializes(
                state,
                content,
            ):
                count += 1
        return count

    def _artifact_prior_source_lines_inspection_count(self, state: AgentState) -> int:
        count = 0
        for step in state.history:
            if step.action != CODE_EXECUTE:
                continue
            if self._artifact_source_lines_inspection_command(str(step.content)):
                count += 1
        return count

    def _artifact_source_inspection_allowed_after_history(self, state: AgentState, command: str) -> bool:
        normalized = str(command).strip()
        if not artifact_repair_contracts.command_is_source_inspection(normalized):
            return False
        if self._artifact_materializes(state, normalized):
            return False
        prior_source_inspections = self._artifact_prior_source_inspection_count(state)
        if prior_source_inspections < 1:
            return True
        if (
            prior_source_inspections == 1
            and self._artifact_source_lines_inspection_command(normalized)
            and self._artifact_prior_source_lines_inspection_count(state) < 1
        ):
            return True
        return False

    @staticmethod
    def _artifact_source_lines_inspection_command(command: str) -> bool:
        normalized = str(command).strip()
        return (
            artifact_repair_contracts.command_is_source_inspection(normalized)
            and "source_lines/" in normalized
            and ".lines" in normalized
        )

    def _artifact_source_lines_followup_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if not self._artifact_latest_repairable_failure(state):
            return None
        if self._artifact_prior_source_lines_inspection_count(state) > 0:
            return None
        source_path = self._artifact_last_builder_source_path(state) or self._artifact_last_inspected_source_path(state)
        if not source_path:
            return None
        source_lines_path = f"source_lines/{source_path}.lines"
        if not self._artifact_setup_file_available(state, source_lines_path):
            return None
        reason = (
            "after a no-op artifact repair"
            if self._artifact_noop_repair_context(state)
            else "after source inspection with missing-artifact verifier feedback"
        )
        return ActionDecision(
            thought=f"Read line-numbered source {reason} so the next edit has exact anchors.",
            action=CODE_EXECUTE,
            content=f"cat {shlex.quote(source_lines_path)}",
            done=False,
            decision_source="artifact_source_lines_followup_direct",
            proposal_metadata={
                "artifact_source_lines_followup": True,
                "source_path": source_path,
                "source_lines_path": source_lines_path,
            },
        )

    def _artifact_last_inspected_source_path(self, state: AgentState) -> str:
        known_sources = set(self._artifact_source_context_by_path(state).keys())
        for step in reversed(state.history):
            if step.action != CODE_EXECUTE:
                continue
            command = str(step.content).strip()
            if self._artifact_source_lines_inspection_command(command):
                continue
            if not artifact_repair_contracts.command_is_source_inspection(command):
                continue
            path = self._artifact_source_path_from_inspection_command(command)
            if path and (not known_sources or path in known_sources):
                return path
        return ""

    @staticmethod
    def _artifact_source_path_from_inspection_command(command: str) -> str:
        try:
            parts = shlex.split(str(command).strip())
        except ValueError:
            return ""
        if not parts:
            return ""
        executable = parts[0]
        if executable not in artifact_repair_contracts.SOURCE_INSPECTION_EXECUTABLES:
            return ""
        for token in reversed(parts[1:]):
            candidate = str(token).strip().strip("'\"").strip("/")
            if not candidate or candidate.startswith("-"):
                continue
            if candidate.replace(",", "").replace("-", "").isdigit():
                continue
            if candidate.startswith("source_lines/") and candidate.endswith(".lines"):
                return candidate.removeprefix("source_lines/").removesuffix(".lines").strip("/")
            if candidate.startswith("source_context/"):
                return candidate.removeprefix("source_context/").strip("/")
            if candidate in {".", "/"}:
                continue
            return candidate
        return ""

    @classmethod
    def _artifact_last_builder_source_path(cls, state: AgentState) -> str:
        for step in reversed(state.history):
            if step.action != CODE_EXECUTE:
                continue
            spec = cls._artifact_builder_command_spec(state, str(step.content), require_redirect=False)
            if spec is None:
                continue
            path = str(spec.get("path", "")).strip().strip("/")
            if path:
                return path
        return ""

    @staticmethod
    def _artifact_setup_file_available(state: AgentState, path: str) -> bool:
        normalized = str(path).strip().strip("/")
        if not normalized:
            return False
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        if isinstance(setup_file_contents, dict) and normalized in {
            str(key).strip().strip("/") for key in setup_file_contents.keys()
        }:
            return True
        workflow_guard = metadata.get("workflow_guard", {})
        if isinstance(workflow_guard, dict):
            managed_paths = workflow_guard.get("managed_paths", [])
            if isinstance(managed_paths, list) and normalized in {
                str(item).strip().strip("/") for item in managed_paths
            }:
                return True
        return False

    @staticmethod
    def _artifact_repair_context(state: AgentState) -> bool:
        contract = artifact_repair_contracts.contract_from_state(state)
        if contract is None:
            return False
        artifact_path = contract.artifact_path
        task = state.task
        if artifact_path in task.expected_files:
            return True
        return any(
            str(diagnosis.get("path", "")).strip() == artifact_path
            and (
                str(diagnosis.get("repair_instruction", "")).strip()
                or "verifier_failure" in [str(signal) for signal in diagnosis.get("signals", [])]
                or str(diagnosis.get("source_role", "")).strip() == "verifier"
            )
            for diagnosis in state.subgoal_diagnoses.values()
            if isinstance(diagnosis, dict)
        )

    @staticmethod
    def _artifact_inference_failure_source_context_fallback(state: AgentState) -> ActionDecision | None:
        if not LLMDecisionPolicy._artifact_latest_repairable_failure(state):
            return None
        prior_inference_fallbacks = sum(
            1
            for step in state.history
            if str(step.decision_source).strip() == "artifact_inference_failure_source_context_fallback"
        )
        if prior_inference_fallbacks >= 1:
            return None
        attempted = {
            str(step.content).strip()
            for step in state.history
            if step.action == CODE_EXECUTE and str(step.content).strip()
        }
        metadata = state.task.metadata if isinstance(state.task.metadata, dict) else {}
        setup_file_contents = metadata.get("setup_file_contents", {})
        if not isinstance(setup_file_contents, dict):
            return None

        candidate_paths: list[str] = []
        for raw_path, content in setup_file_contents.items():
            if not isinstance(content, str):
                continue
            path = str(raw_path).strip().strip("/")
            if path.startswith("source_lines/") and path.endswith(".lines"):
                candidate_paths.append(path)
        for raw_path, content in setup_file_contents.items():
            if not isinstance(content, str):
                continue
            path = str(raw_path).strip().strip("/")
            if path.startswith("source_context/"):
                candidate_paths.append(path)

        seen: set[str] = set()
        for path in candidate_paths:
            if not path or path in seen or any(part == ".." for part in Path(path).parts):
                continue
            seen.add(path)
            command = f"cat {shlex.quote(path)}"
            if command in attempted:
                continue
            return ActionDecision(
                thought=(
                    "Recover from an inference-format failure during artifact repair by reading "
                    "unattempted source context before asking the live model again."
                ),
                action=CODE_EXECUTE,
                content=command,
                done=False,
                decision_source="artifact_inference_failure_source_context_fallback",
                proposal_metadata={
                    "artifact_inference_failure_fallback": True,
                    "fallback_path": path,
                },
            )
        return None

    def _artifact_suggested_builder_command_direct_decision(self, state: AgentState) -> ActionDecision | None:
        task = state.task
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        contract = artifact_repair_contracts.contract_from_state(state)
        if contract is None:
            return None
        commands = metadata.get("artifact_suggested_builder_commands", [])
        if not isinstance(commands, list) or not commands:
            # Legacy benchmark adapters still emit this key. Keep it as an
            # adapter input, then validate through the generic artifact contract.
            commands = metadata.get("swe_suggested_patch_commands", [])
        if not isinstance(commands, list):
            return None
        attempted = {
            str(step.content).strip()
            for step in state.history
            if step.action == CODE_EXECUTE and str(step.content).strip()
        }
        for raw_command in commands:
            command = str(raw_command).strip()
            if not command or command in attempted:
                continue
            if "\n" in command or "\r" in command:
                continue
            spec = self._artifact_builder_command_spec(state, command, require_redirect=True)
            if spec is None or not spec.get("path") or not spec.get("operations"):
                continue
            if self._artifact_builder_replaces_existing_source(state, command):
                continue
            if self._artifact_builder_broad_replacement(state, command):
                continue
            if self._artifact_builder_removes_definition_header(state, command):
                continue
            if self._artifact_builder_would_produce_invalid_python(state, command):
                continue
            return ActionDecision(
                thought=(
                    "Execute the source-grounded suggested artifact repair command before asking "
                    "the model to choose a broader edit window."
                ),
                action=CODE_EXECUTE,
                content=command,
                done=False,
                decision_source="artifact_suggested_builder_command_direct",
                proposal_metadata={"artifact_suggested_builder_command": True},
            )
        return None

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

    def _artifact_builder_from_diff_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if not self._artifact_repair_context(state):
            return None
        # Replaying an already failed artifact tends to preserve invalid syntax
        # or semantic no-ops. Keep repair model-guided via explicit builder
        # commands instead of auto-replaying a stale artifact file.
        return None

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

    def _shared_repo_worker_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        return self.workflow_adapter.shared_repo_worker_direct_decision(
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

    def _pre_context_shared_repo_worker_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_shared_repo_worker_direct_decision(state)

    def _pre_context_workspace_contract_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workflow_adapter.pre_context_workspace_contract_direct_decision(state)

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

    @staticmethod
    def _exact_verifier_repair_brief(state: AgentState) -> str:
        contract = artifact_repair_contracts.contract_from_state(state)
        artifact_path = contract.artifact_path if contract is not None else "patch.diff"
        builder_command = artifact_repair_contracts.preferred_builder_command(contract)
        diagnoses: list[tuple[str, dict[str, object]]] = []
        active = str(state.active_subgoal).strip()
        if active:
            diagnosis = state.diagnosis_for_subgoal(active)
            if diagnosis:
                diagnoses.append((active, diagnosis))
        for goal, diagnosis in state.subgoal_diagnoses.items():
            normalized = str(goal).strip()
            if not normalized or normalized == active or not isinstance(diagnosis, dict):
                continue
            diagnoses.append((normalized, dict(diagnosis)))
        repair_items: list[str] = []
        for goal, diagnosis in diagnoses:
            path = str(diagnosis.get("path", "")).strip()
            repair_instruction = str(diagnosis.get("repair_instruction", "")).strip()
            if repair_instruction:
                if path == artifact_path:
                    summary = str(diagnosis.get("summary", "")).strip()
                    semantic_repair = ""
                    if "SWE patch removes existing Python definitions" in summary:
                        semantic_repair = (
                            f"; do not run {builder_command} --from-diff because it will preserve the invalid broad hunk; "
                            "do not use broad --replace-lines over a whole function/class or high-value window; "
                            "choose the smallest behavior-changing hunk that preserves every existing def/class line"
                        )
                    elif "SWE patch python syntax check failed" in summary:
                        semantic_repair = (
                            f"; do not run {builder_command} --from-diff because it will preserve the same invalid Python; "
                            "replace the exact bad line or narrow block with syntactically valid source at the same indentation level; "
                            "do not use single-space indentation in Python blocks; do not add indented statements at module top level "
                            "unless they remain inside an existing block"
                        )
                    elif "SWE patch leaves invalid __init__ return values" in summary:
                        semantic_repair = (
                            f"; do not run {builder_command} --from-diff because it will preserve the invalid __init__ return; "
                            "never add or keep a value-returning return statement inside __init__; choose the smallest "
                            "behavior-changing assignment/body edit or use a bare return only when early exit is required"
                        )
                    elif "SWE patch leaves invalid __init__ generators" in summary:
                        semantic_repair = (
                            f"; do not run {builder_command} --from-diff because it will preserve the generator __init__; "
                            "never add yield or yield from inside __init__; keep __init__ as ordinary object initialization "
                            "and move iterable behavior into a normal method/helper or assigned state"
                        )
                    elif "SWE patch introduces local use before assignment" in summary:
                        semantic_repair = (
                            f"; do not run {builder_command} --from-diff because it will preserve the unbound local read; "
                            "ensure every local variable is assigned before its first read and preserve the existing "
                            "initialization order with a smallest-scope statement edit"
                        )
                    repair_instruction = (
                        f"{repair_instruction}; next command must create or overwrite {artifact_path} directly "
                        f"with one {builder_command} command using exact line numbers from source_lines or the high-value "
                        f"edit window; prefer --replace-line/--replace-lines and redirect to {artifact_path}; do not hand-write "
                        "unified diff hunk headers with cat/printf; do not run cat, ls, find, git, python/python3 -c, "
                        "or source inspection first"
                        f"{semantic_repair}"
                    )
                repair_items.append(repair_instruction)
                if len(repair_items) >= 3:
                    break
                continue
            if not path or diagnosis.get("expected_content") is None:
                continue
            expected = str(diagnosis.get("expected_content", ""))
            preview = str(diagnosis.get("expected_content_preview", "")).strip()
            if not preview:
                preview = json.dumps(expected)
            repair_items.append(
                f"rewrite {path} to exactly {preview}; do not paraphrase, pretty-print, add headings, or change spacing"
            )
            if len(repair_items) >= 3:
                break
        if not repair_items:
            return ""
        return " | ".join(repair_items)

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
            if bool(getattr(self.config, "use_research_library_context", False)):
                return "deterministic_command"
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
