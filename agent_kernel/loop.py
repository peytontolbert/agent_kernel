from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib import error as url_error

from .actions import CODE_EXECUTE
from .config import KernelConfig
from .episode_store import episode_storage_metadata
from .learning_compiler import persist_episode_learning_candidates
from .llm import MockLLMClient, OllamaClient, TolbertFallbackClient, VLLMClient
from .memory import EpisodeMemory, GraphMemory
from .modeling.artifacts import load_model_artifact, retained_tolbert_hybrid_runtime
from .modeling.tolbert.runtime import infer_hybrid_world_signal
from .modeling.world.latent_state import build_latent_state_summary
from .state_estimation_improvement import (
    retained_state_estimation_latent_controls,
    retained_state_estimation_payload,
    summarize_state_transition,
)
from .syntax_motor import summarize_python_edit_step
from .multi_agent import RoleCoordinator
from .policy import LLMDecisionPolicy, Policy, SkillLibrary
from .runtime_supervision import atomic_write_json
from .prompt_improvement import retained_planner_controls
from .sandbox import Sandbox
from .schemas import EpisodeRecord, StepRecord, TaskSpec
from .shared_repo import (
    materialize_shared_repo_workspace,
    prepare_runtime_task,
    publish_shared_repo_branch,
    uses_shared_repo,
)
from .state import AgentState
from .tolbert import MockTolbertContextCompiler, TolbertContextCompiler
from .universe_model import UniverseModel
from .verifier import Verifier
from .world_model import WorldModel
from .kernel_catalog import kernel_catalog_string_set


_FRONTIER_STEP_FLOOR_FAMILIES = frozenset(kernel_catalog_string_set("runtime_defaults", "frontier_step_floor_families"))


class AgentKernel:
    def __init__(
        self,
        config: KernelConfig | None = None,
        policy: Policy | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.config = config or KernelConfig()
        self.config.ensure_directories()
        self.policy = policy or self._build_default_policy()
        self.verifier = verifier or Verifier()
        self.sandbox = Sandbox(self.config.command_timeout_seconds, config=self.config)
        self.memory = EpisodeMemory(self.config.trajectories_root, config=self.config)
        self.graph_memory = GraphMemory(self.memory)
        self.universe_model = UniverseModel(config=self.config)
        self.world_model = WorldModel(config=self.config)
        self.role_coordinator = RoleCoordinator()
        self._tolbert_model_payload_cache: dict[str, object] | None = None

    def _build_default_policy(self) -> Policy:
        repo_root = Path(__file__).resolve().parents[1]
        context_provider = None
        if self.config.use_tolbert_context:
            if self.config.provider == "mock":
                context_provider = MockTolbertContextCompiler(config=self.config, repo_root=repo_root)
            else:
                context_provider = TolbertContextCompiler(config=self.config, repo_root=repo_root)
        skill_library = (
            SkillLibrary.from_path(
                repo_root / self.config.skills_path,
                min_quality=self.config.min_skill_quality,
            )
            if self.config.use_skills
            else SkillLibrary([])
        )
        if self.config.provider == "ollama":
            client = OllamaClient(
                host=self.config.ollama_host,
                model_name=self.config.model_name,
                timeout_seconds=self.config.llm_timeout_seconds,
                retry_attempts=self.config.llm_retry_attempts,
                retry_backoff_seconds=self.config.llm_retry_backoff_seconds,
            )
            return LLMDecisionPolicy(
                client,
                context_provider=context_provider,
                skill_library=skill_library,
                config=self.config,
            )
        if self.config.provider == "vllm":
            client = VLLMClient(
                host=self.config.vllm_host,
                model_name=self.config.model_name,
                timeout_seconds=self.config.llm_timeout_seconds,
                retry_attempts=self.config.llm_retry_attempts,
                retry_backoff_seconds=self.config.llm_retry_backoff_seconds,
                api_key=self.config.vllm_api_key,
            )
            return LLMDecisionPolicy(
                client,
                context_provider=context_provider,
                skill_library=skill_library,
                config=self.config,
            )
        if self.config.provider == "mock":
            return LLMDecisionPolicy(
                MockLLMClient(),
                context_provider=context_provider,
                skill_library=skill_library,
                config=self.config,
            )
        if self.config.provider == "tolbert":
            return LLMDecisionPolicy(
                TolbertFallbackClient(),
                context_provider=context_provider,
                skill_library=skill_library,
                config=self.config,
            )
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    @staticmethod
    def _should_persist_learning_candidates(task: TaskSpec) -> bool:
        return str(task.metadata.get("curriculum_kind", "")).strip() != "adjacent_success"

    def _resolved_task_step_limit(self, task: TaskSpec) -> int:
        requested_steps = max(1, int(task.max_steps))
        explicit_floor = self._task_step_floor_override(task)
        frontier_floor = self._frontier_task_step_floor(task)
        long_horizon_floor = self._long_horizon_runtime_step_floor(task)
        requested_steps = max(requested_steps, explicit_floor, frontier_floor, long_horizon_floor)
        return min(requested_steps, self.config.max_task_steps_hard_cap)

    @staticmethod
    def _task_step_floor_override(task: TaskSpec) -> int:
        raw_value = task.metadata.get("step_floor")
        try:
            step_floor = int(raw_value)
        except (TypeError, ValueError):
            return 0
        return max(0, step_floor)

    def _frontier_task_step_floor(self, task: TaskSpec) -> int:
        benchmark_family = str(task.metadata.get("benchmark_family", "")).strip()
        difficulty = str(task.metadata.get("difficulty", "")).strip()
        horizon = str(task.metadata.get("horizon", "")).strip()
        if (
            benchmark_family in _FRONTIER_STEP_FLOOR_FAMILIES
            or difficulty == "long_horizon"
            or horizon == "long_horizon"
        ):
            return max(1, int(self.config.frontier_task_step_floor))
        return 0

    @staticmethod
    def _long_horizon_runtime_step_floor(task: TaskSpec) -> int:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        difficulty = str(metadata.get("difficulty", "")).strip()
        horizon = str(metadata.get("horizon", "")).strip()
        curriculum_shape = str(metadata.get("curriculum_shape", "")).strip()
        if (
            difficulty != "long_horizon"
            and horizon != "long_horizon"
            and curriculum_shape != "long_horizon_structured_edit"
        ):
            return 0
        try:
            planned_steps = int(metadata.get("long_horizon_step_count", 0) or 0)
        except (TypeError, ValueError):
            planned_steps = 0
        synthetic_edit_plan = metadata.get("synthetic_edit_plan", [])
        if planned_steps <= 0 and isinstance(synthetic_edit_plan, list):
            planned_steps = len(synthetic_edit_plan)
        planned_steps = max(planned_steps, int(task.max_steps or 0), 1)
        lineage_depth = max(0, str(task.task_id or "").count("_adjacent"))
        return max(
            planned_steps + 3,
            planned_steps * 2,
            12 + (lineage_depth * 4),
        )

    def run_task(
        self,
        task: TaskSpec,
        clean_workspace: bool = True,
        checkpoint_path: Path | None = None,
        resume: bool = False,
        runtime_overrides: dict[str, object] | None = None,
        job_id: str = "",
        progress_callback=None,
    ) -> EpisodeRecord:
        task = prepare_runtime_task(task, runtime_overrides=runtime_overrides, job_id=job_id) if uses_shared_repo(
            task,
            runtime_overrides=runtime_overrides,
        ) else deepcopy(task)
        workspace = self.config.workspace_root / task.workspace_subdir
        checkpoint = None
        setup_history: list[dict[str, object]] = []
        if checkpoint_path is not None and resume and checkpoint_path.exists():
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint.get("status") == "completed":
                return self._episode_from_payload(checkpoint["episode"])
            workspace = Path(str(checkpoint.get("workspace", workspace)))
            setup_history = self._setup_history_from_checkpoint(checkpoint)
            state = self._state_from_checkpoint(task, checkpoint)
            state.termination_reason = ""
            clean_workspace = False
        else:
            if uses_shared_repo(task, runtime_overrides=runtime_overrides):
                workspace = materialize_shared_repo_workspace(
                    task,
                    config=self.config,
                    runtime_overrides=runtime_overrides,
                    job_id=job_id,
                    resume=resume and checkpoint_path is not None and checkpoint_path.exists(),
                )
                self._ensure_parallel_worker_branches(
                    task,
                    workspace=workspace,
                    runtime_overrides=runtime_overrides,
                    job_id=job_id,
                )
                clean_workspace = False
            if clean_workspace and workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            state = AgentState(task=task)
            if checkpoint_path is not None:
                self._write_checkpoint(
                    checkpoint_path,
                    task=task,
                    workspace=workspace,
                    state=state,
                    success=False,
                    status="setup_in_progress" if task.setup_commands else "in_progress",
                    termination_reason="setup_pending" if task.setup_commands else "max_steps_reached",
                    setup_history=setup_history,
                    phase="setup" if task.setup_commands else "execute",
                )
        setup_resume_index = self._completed_setup_command_count(task, setup_history)
        if setup_resume_index < len(task.setup_commands):
            for command_index, command in enumerate(task.setup_commands[setup_resume_index:], start=setup_resume_index):
                result = self.sandbox.run(command, workspace, task=task)
                entry = {
                    "index": command_index,
                    "command": command,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                if len(setup_history) > command_index:
                    setup_history[command_index] = entry
                else:
                    setup_history.append(entry)
                if checkpoint_path is not None:
                    self._write_checkpoint(
                        checkpoint_path,
                        task=task,
                        workspace=workspace,
                        state=state,
                        success=False,
                        status="setup_in_progress",
                        termination_reason="setup_pending",
                        setup_history=setup_history,
                        phase="setup",
                    )
                if result.exit_code != 0 or result.timed_out:
                    return self._setup_failure_episode(
                        task=task,
                        workspace=workspace,
                        checkpoint_path=checkpoint_path,
                        state=state,
                        setup_history=setup_history,
                    )
        if self.config.use_graph_memory and not state.graph_summary:
            state.graph_summary = self.graph_memory.summarize(task.task_id)
        if not state.universe_summary:
            state.universe_summary = self.universe_model.summarize(
                task,
                world_model_summary=state.world_model_summary,
                workspace=workspace,
                planned_commands=task.setup_commands + task.suggested_commands + ([task.success_command] if task.success_command else []),
            )
        state_estimation_payload = retained_state_estimation_payload(self.config)
        latent_controls = retained_state_estimation_latent_controls(state_estimation_payload)
        if self.config.use_world_model:
            if not state.workspace_snapshot:
                state.workspace_snapshot = self.world_model.capture_workspace_snapshot(task, workspace)
            if not state.world_model_summary:
                state.world_model_summary = self.world_model.summarize(
                    task,
                    state.graph_summary,
                    workspace=workspace,
                    workspace_snapshot=state.workspace_snapshot,
                )
                state.universe_summary = self.universe_model.summarize(
                    task,
                    world_model_summary=state.world_model_summary,
                    workspace=workspace,
                    planned_commands=task.setup_commands + task.suggested_commands + ([task.success_command] if task.success_command else []),
                )
                state.recent_workspace_summary = self.world_model.describe_progress(state.world_model_summary)
            learned_world_signal = self._infer_learned_world_signal(state)
            state.latent_state_summary = build_latent_state_summary(
                world_model_summary=state.world_model_summary,
                latest_transition=state.latest_state_transition,
                task_metadata=task.metadata,
                recent_history=[asdict(step) for step in state.history[-3:]],
                context_control=state.context_packet.control if state.context_packet is not None else {},
                latent_controls=latent_controls,
                learned_world_signal=learned_world_signal,
            )
        if self.config.use_planner and not state.initial_plan:
            state.plan = self._build_plan(task)
            state.initial_plan = list(state.plan)
            state.active_subgoal = state.plan[0] if state.plan else ""
        if checkpoint_path is not None:
            self._write_checkpoint(
                checkpoint_path,
                task=task,
                workspace=workspace,
                state=state,
                success=False,
                status="in_progress",
                termination_reason="max_steps_reached",
                setup_history=setup_history,
                phase="execute",
            )
        success = False
        termination_reason = "max_steps_reached"
        if checkpoint is not None:
            success = bool(checkpoint.get("success", False))
            termination_reason = str(checkpoint.get("termination_reason", "max_steps_reached"))

        effective_step_limit = self._resolved_task_step_limit(task)
        remaining_steps = max(0, effective_step_limit - state.completed_step_count())
        for _ in range(remaining_steps):
            if self.config.use_role_specialization:
                state.current_role = self._resolve_role_before_decision(state)
            if self.config.use_planner and state.current_role == "planner":
                self._refresh_planner_subgoals(state)
            step_index = state.next_step_index()
            step_active_subgoal = state.active_subgoal
            step_role = state.current_role
            previous_world_model_summary = dict(state.world_model_summary)
            failure_signals: list[str] = []
            failure_origin = ""
            step_started_at = time.monotonic()
            decision_progress_callback = lambda payload: self._emit_progress_callback(
                progress_callback,
                {
                    "event": "decision_progress",
                    "step_index": step_index,
                    "completed_steps": state.completed_step_count(),
                    "active_subgoal": step_active_subgoal,
                    "acting_role": step_role,
                    **dict(payload),
                },
            )
            self._emit_progress_callback(
                progress_callback,
                {
                    "event": "step_start",
                    "step_index": step_index,
                    "step_stage": "decision_pending",
                    "completed_steps": state.completed_step_count(),
                    "active_subgoal": step_active_subgoal,
                    "acting_role": step_role,
                },
            )
            self.policy.set_decision_progress_callback(decision_progress_callback)
            try:
                decision = self.policy.decide(state)
            except Exception as exc:
                failure_origin = self.classify_policy_failure(exc)
                failure_signals = [failure_origin]
                fallback_decision = self.policy.fallback_decision(state, failure_origin=failure_origin)
                if fallback_decision is not None:
                    fallback_decision.proposal_metadata = {
                        **dict(fallback_decision.proposal_metadata or {}),
                        "fallback_failure_origin": failure_origin,
                        "fallback_error_type": exc.__class__.__name__,
                    }
                    decision = fallback_decision
                else:
                    decision = self.policy_error_decision(exc, failure_origin=failure_origin)
            finally:
                self.policy.set_decision_progress_callback(None)
            self._emit_progress_callback(
                progress_callback,
                {
                    "event": "decision_ready",
                    "step_index": step_index,
                    "step_stage": "decision_ready",
                    "completed_steps": state.completed_step_count(),
                    "active_subgoal": step_active_subgoal,
                    "acting_role": step_role,
                    "decision_action": str(decision.action),
                    "decision_done": bool(decision.done),
                    "decision_source": str(decision.decision_source),
                },
            )
            command_result = None
            verification = {"passed": False, "reasons": ["no command executed"]}
            structured_edit_before_content: str | None = None
            runtime_proposal_metadata = dict(decision.proposal_metadata or {})

            if decision.action == CODE_EXECUTE:
                proposal_source = str(decision.proposal_source).strip()
                if proposal_source.startswith("structured_edit:"):
                    proposal_metadata = dict(runtime_proposal_metadata)
                    structured_edit_path = str(proposal_metadata.get("path", "")).strip()
                    if structured_edit_path:
                        try:
                            structured_edit_before_content = (workspace / structured_edit_path).read_text(
                                encoding="utf-8"
                            )
                        except (FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError):
                            structured_edit_before_content = None
                command_result = self.sandbox.run(decision.content, workspace, task=task)
                verified = self.verifier.verify(task, workspace, command_result)
                verification = {
                    "passed": verified.passed,
                    "reasons": verified.reasons,
                }
                success = verified.passed
            elif decision.done:
                reasons = ["policy terminated"]
                if failure_origin:
                    reasons.append(f"policy failure origin: {failure_origin}")
                verification = {"passed": success, "reasons": reasons}

            transition: dict[str, object] = {}
            command_governance: dict[str, object] = {}
            if self.config.use_world_model:
                state.world_model_summary = self.world_model.summarize(
                    task,
                    state.graph_summary,
                    workspace=workspace,
                    workspace_snapshot=state.workspace_snapshot,
                )
                state.universe_summary = self.universe_model.summarize(
                    task,
                    world_model_summary=state.world_model_summary,
                )
                raw_transition = self.world_model.describe_transition(
                    previous_world_model_summary,
                    state.world_model_summary,
                )
                transition = summarize_state_transition(
                    raw_transition,
                    payload=state_estimation_payload,
                )
                if self._structured_edit_mutated_path(
                    workspace,
                    decision,
                    before_content=structured_edit_before_content,
                    command_result=command_result,
                ):
                    transition["progress_delta"] = max(float(transition.get("progress_delta", 0.0)), 0.001)
                    transition["no_progress"] = False
                    transition["state_change_score"] = max(int(transition.get("state_change_score", 0) or 0), 1)
                syntax_motor_progress = self._structured_edit_syntax_progress(
                    task,
                    workspace=workspace,
                    decision=decision,
                    before_content=structured_edit_before_content,
                    command_result=command_result,
                )
                if syntax_motor_progress:
                    runtime_proposal_metadata["syntax_motor_progress"] = dict(syntax_motor_progress)
                    runtime_proposal_metadata["syntax_motor"] = dict(
                        syntax_motor_progress.get("syntax_motor", {})
                    )
                    transition["syntax_motor_progress"] = dict(syntax_motor_progress)
                    if bool(syntax_motor_progress.get("strong_progress", False)):
                        transition["progress_delta"] = max(float(transition.get("progress_delta", 0.0)), 0.025)
                        transition["no_progress"] = False
                        transition["state_change_score"] = max(int(transition.get("state_change_score", 0) or 0), 2)
                state.recent_workspace_summary = self.world_model.describe_progress(
                    state.world_model_summary,
                    command=decision.content if decision.action == CODE_EXECUTE else "",
                    step_index=step_index,
                )
                if command_result is not None and bool(transition.get("no_progress", False)):
                    failure_signals.append("no_state_progress")
                if command_result is not None and list(transition.get("regressions", [])):
                    failure_signals.append("state_regression")
                learned_world_signal = self._infer_learned_world_signal(state)
                state.latent_state_summary = build_latent_state_summary(
                    world_model_summary=state.world_model_summary,
                    latest_transition=transition,
                    task_metadata=task.metadata,
                    recent_history=[asdict(step) for step in state.history[-3:]],
                    context_control=state.context_packet.control if state.context_packet is not None else {},
                    latent_controls=latent_controls,
                    learned_world_signal=learned_world_signal,
                )
            if decision.action == CODE_EXECUTE and decision.content:
                command_governance = self.universe_model.simulate_command_governance(
                    state.universe_summary,
                    decision.content,
                )
            step_software_work_objective = state.current_software_work_objective() if self.config.use_planner else ""
            if self.config.use_planner:
                state.refresh_plan_progress(state.world_model_summary)
            state.update_after_step(
                decision=decision,
                command_result=command_result,
                verification_passed=verification["passed"],
                step_index=step_index,
                progress_delta=float(transition.get("progress_delta", 0.0)),
                state_regressed=bool(transition.get("regressions", [])),
                state_transition=transition,
                software_work_objective=step_software_work_objective,
            )
            if step_role == "critic":
                self._attach_critic_subgoal_diagnoses(
                    state,
                    step_index=step_index,
                    step_active_subgoal=step_active_subgoal,
                    failure_signals=failure_signals,
                    failure_origin=failure_origin,
                    command_result=command_result,
                )
            if not verification["passed"]:
                self._attach_verifier_subgoal_diagnoses(
                    state,
                    step_index=step_index,
                    verification_reasons=verification["reasons"],
                )
                if self.config.use_planner:
                    self._promote_prioritized_subgoals(
                        state,
                        prioritized=self._verifier_hotspot_subgoals(verification["reasons"]),
                    )
            if self.config.use_planner:
                self._refresh_planner_recovery_artifact(state)
            step_record = StepRecord(
                index=step_index,
                thought=decision.thought,
                action=decision.action,
                content=decision.content,
                selected_skill_id=decision.selected_skill_id,
                command_result=asdict(command_result) if command_result else None,
                verification=verification,
                available_skill_count=self._available_skill_count(state),
                retrieval_candidate_count=self._retrieval_candidate_count(state),
                retrieval_evidence_count=self._retrieval_evidence_count(state),
                retrieval_command_match=self._retrieval_command_match(state, decision.content),
                selected_retrieval_span_id=decision.selected_retrieval_span_id,
                retrieval_influenced=decision.retrieval_influenced,
                retrieval_ranked_skill=decision.retrieval_ranked_skill,
                path_confidence=self._path_confidence(state),
                trust_retrieval=self._trust_retrieval(state),
                retrieval_direct_candidate_count=len(state.retrieval_direct_candidates),
                active_subgoal=step_active_subgoal,
                subgoal_diagnoses=dict(state.subgoal_diagnoses),
                acting_role=step_role,
                world_model_horizon=str(state.world_model_summary.get("horizon", "")),
                state_progress_delta=float(transition.get("progress_delta", 0.0)),
                state_regression_count=len(list(transition.get("regressions", []))),
                state_transition=dict(transition),
                failure_signals=failure_signals,
                failure_origin=failure_origin,
                command_governance=dict(command_governance),
                runtime_attestation=dict(state.universe_summary.get("runtime_attestation", {})),
                decision_source=decision.decision_source,
                tolbert_route_mode=decision.tolbert_route_mode,
                proposal_source=decision.proposal_source,
                proposal_novel=decision.proposal_novel,
                proposal_metadata=dict(runtime_proposal_metadata),
                shadow_decision=dict(decision.shadow_decision),
                latent_state_summary=dict(state.latent_state_summary),
            )
            state.history.append(step_record)
            state.compact_history(
                max_recent_steps=self.config.runtime_history_step_window,
                summary_char_limit=self.config.history_archive_summary_max_chars,
            )
            self._emit_progress_callback(
                progress_callback,
                {
                    "event": "step_complete",
                    "step_index": step_index,
                    "step_stage": "step_complete",
                    "completed_steps": state.completed_step_count(),
                    "active_subgoal": step_active_subgoal,
                    "acting_role": step_role,
                    "decision_action": str(decision.action),
                    "decision_done": bool(decision.done),
                    "decision_source": str(decision.decision_source),
                    "verification_passed": bool(verification["passed"]),
                    "termination_reason": str(termination_reason),
                    "step_elapsed_seconds": round(time.monotonic() - step_started_at, 4),
                },
            )

            if success:
                termination_reason = "success"
            elif decision.done:
                termination_reason = "policy_terminated"
            elif state.should_stop_for_stuckness():
                if state.termination_reason == "repeated_failed_action":
                    verification["reasons"].append("repeated failed action detected")
                elif state.termination_reason == "no_state_progress":
                    verification["reasons"].append("no state progress detected")
                termination_reason = state.termination_reason
            elif self.config.use_role_specialization:
                state.current_role = self._resolve_role_after_step(
                    state,
                    verification_passed=verification["passed"],
                )

            if checkpoint_path is not None:
                self._write_checkpoint(
                    checkpoint_path,
                    task=task,
                    workspace=workspace,
                    state=state,
                    success=success,
                    status="completed" if success or decision.done or state.termination_reason else "in_progress",
                    termination_reason=termination_reason,
                    setup_history=setup_history,
                    phase="execute",
                )

            if success or decision.done or state.termination_reason:
                break

        episode = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(workspace),
            success=success,
            steps=state.history,
            task_metadata=dict(task.metadata),
            task_contract=self._task_contract_payload(task),
            plan=list(state.initial_plan),
            graph_summary=dict(state.graph_summary),
            universe_summary=dict(state.universe_summary),
            world_model_summary=dict(state.world_model_summary),
            history_archive=dict(state.history_archive),
            termination_reason=termination_reason,
        )
        episode_path = None
        if self.config.persist_episode_memory:
            episode_path = self.memory.save(episode)
        if self._should_persist_learning_candidates(task):
            persist_episode_learning_candidates(
                episode,
                config=self.config,
                episode_storage=(
                    episode_storage_metadata(self.config.trajectories_root, episode_path)
                    if episode_path is not None
                    else None
                ),
            )
        if episode.success and uses_shared_repo(task, runtime_overrides=runtime_overrides):
            publish_shared_repo_branch(
                task,
                config=self.config,
                runtime_overrides=runtime_overrides,
                job_id=job_id,
            )
        if checkpoint_path is not None:
            self._write_checkpoint(
                checkpoint_path,
                task=task,
                workspace=workspace,
                state=state,
                success=success,
                status="completed",
                termination_reason=termination_reason,
                episode=episode,
                setup_history=setup_history,
                phase="execute",
            )
        return episode

    def _ensure_parallel_worker_branches(
        self,
        task: TaskSpec,
        *,
        workspace: Path,
        runtime_overrides: dict[str, object] | None,
        job_id: str,
    ) -> None:
        metadata = dict(getattr(task, "metadata", {}) or {})
        if bool(metadata.get("synthetic_worker", False)):
            return
        if int(metadata.get("shared_repo_order", 0) or 0) <= 0:
            return
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if not required_branches:
            return
        branch_output = self._git_output(workspace, "branch", "--format=%(refname:short)")
        existing_branches = (
            {line.strip() for line in branch_output.splitlines() if line.strip()}
            if branch_output is not None
            else set()
        )
        missing_branches = [branch for branch in required_branches if branch not in existing_branches]
        if not missing_branches:
            return
        from .task_bank import TaskBank

        bank = TaskBank(config=self.config)
        for worker in bank.parallel_worker_tasks(task.task_id):
            worker_guard = dict(worker.metadata.get("workflow_guard", {}) or {})
            worker_branch = str(worker_guard.get("worker_branch", "")).strip()
            if worker_branch and worker_branch not in missing_branches:
                continue
            episode = self.run_task(
                worker,
                clean_workspace=False,
                runtime_overrides=runtime_overrides,
                job_id=job_id,
            )
            if episode.success and worker_branch:
                self._git_fetch_branch(workspace, worker_branch)

    @staticmethod
    def _git_output(workspace: Path, *args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    @staticmethod
    def _git_fetch_branch(workspace: Path, branch: str) -> bool:
        normalized = str(branch).strip()
        if not normalized:
            return False
        try:
            completed = subprocess.run(
                ["git", "fetch", "origin", f"{normalized}:{normalized}"],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    @staticmethod
    def _emit_progress_callback(progress_callback, payload: dict[str, object]) -> None:
        if progress_callback is None:
            return
        progress_callback(dict(payload))

    def _tolbert_model_payload(self) -> dict[str, object]:
        if self._tolbert_model_payload_cache is not None:
            return self._tolbert_model_payload_cache
        self._tolbert_model_payload_cache = load_model_artifact(self.config.tolbert_model_artifact_path)
        return self._tolbert_model_payload_cache

    def _infer_learned_world_signal(self, state: AgentState) -> dict[str, object] | None:
        if not bool(self.config.use_tolbert_model_artifacts):
            return None
        runtime = retained_tolbert_hybrid_runtime(self._tolbert_model_payload())
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        if not manifest_raw or not bool(runtime.get("supports_world_model_surface", False)):
            return None
        manifest_path = Path(manifest_raw)
        if not manifest_path.exists():
            return None
        try:
            return infer_hybrid_world_signal(
                state=state,
                bundle_manifest_path=manifest_path,
                device=str(runtime.get("preferred_device", self.config.tolbert_device)).strip() or self.config.tolbert_device,
                scoring_policy=runtime.get("scoring_policy", {}),
            )
        except Exception:
            return None

    @staticmethod
    def _structured_edit_mutated_path(
        workspace: Path,
        decision,
        *,
        before_content: str | None,
        command_result,
    ) -> bool:
        if command_result is None:
            return False
        if int(command_result.exit_code) != 0 or bool(command_result.timed_out):
            return False
        proposal_source = str(getattr(decision, "proposal_source", "")).strip()
        if not proposal_source.startswith("structured_edit:"):
            return False
        proposal_metadata = dict(getattr(decision, "proposal_metadata", {}) or {})
        path = str(proposal_metadata.get("path", "")).strip()
        if not path:
            return False
        target = workspace / path
        try:
            after_content = target.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError):
            return False
        return before_content != after_content

    @staticmethod
    def _structured_edit_syntax_progress(
        task: TaskSpec,
        *,
        workspace: Path,
        decision,
        before_content: str | None,
        command_result,
    ) -> dict[str, object]:
        if command_result is None:
            return {}
        if int(command_result.exit_code) != 0 or bool(command_result.timed_out):
            return {}
        proposal_source = str(getattr(decision, "proposal_source", "")).strip()
        if not proposal_source.startswith("structured_edit:"):
            return {}
        proposal_metadata = dict(getattr(decision, "proposal_metadata", {}) or {})
        path = str(proposal_metadata.get("path", "")).strip()
        if not path.endswith(".py"):
            return {}
        step = dict(proposal_metadata)
        if before_content is not None and not str(step.get("baseline_content", "")).strip():
            step["baseline_content"] = before_content
        syntax_motor = summarize_python_edit_step(
            step,
            workspace=workspace,
            expected_file_contents=dict(getattr(task, "expected_file_contents", {}) or {}),
        )
        if not isinstance(syntax_motor, dict):
            return {}
        baseline_symbol = syntax_motor.get("baseline_target_symbol", {})
        target_symbol = syntax_motor.get("target_target_symbol", {})
        baseline_fqn = str(baseline_symbol.get("fqn", "")).strip() if isinstance(baseline_symbol, dict) else ""
        target_fqn = str(target_symbol.get("fqn", "")).strip() if isinstance(target_symbol, dict) else ""
        edited_symbol_fqn = str(syntax_motor.get("edited_symbol_fqn", "")).strip()
        symbol_aligned = bool(
            edited_symbol_fqn
            and (not baseline_fqn or baseline_fqn == edited_symbol_fqn)
            and (not target_fqn or target_fqn == edited_symbol_fqn)
        )
        syntax_safe = bool(syntax_motor.get("syntax_ok", False))
        import_change_risk = bool(syntax_motor.get("import_change_risk", False))
        signature_change_risk = bool(syntax_motor.get("signature_change_risk", False))
        strong_progress = bool(symbol_aligned and syntax_safe and not import_change_risk and not signature_change_risk)
        return {
            "symbol_aligned": symbol_aligned,
            "syntax_safe": syntax_safe,
            "import_change_risk": import_change_risk,
            "signature_change_risk": signature_change_risk,
            "strong_progress": strong_progress,
            "edited_symbol_fqn": edited_symbol_fqn,
            "call_targets_after": list(syntax_motor.get("call_targets_after", []))[:8],
            "syntax_motor": dict(syntax_motor),
        }

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()

    def _build_plan(self, task: TaskSpec) -> list[str]:
        planner_controls = {
            **self.world_model.retained_planning_controls(),
            **self._planner_controls(),
        }
        world_model_summary = self.world_model.summarize(task)
        plan: list[str] = []
        prefer_expected_artifacts_first = bool(planner_controls.get("prefer_expected_artifacts_first", True))
        if bool(planner_controls.get("prepend_verifier_contract_check", False)):
            plan.append("check verifier contract before terminating")
        plan.extend(self._workflow_plan_steps(task, planner_controls=planner_controls))
        expected_steps = [f"materialize expected artifact {path}" for path in task.expected_files]
        forbidden_steps = [f"remove forbidden artifact {path}" for path in task.forbidden_files]
        if prefer_expected_artifacts_first:
            plan.extend(expected_steps)
            plan.extend(forbidden_steps)
        else:
            plan.extend(forbidden_steps)
            plan.extend(expected_steps)
        if not plan:
            plan.append("satisfy verifier contract")
        if bool(planner_controls.get("append_preservation_subgoal", False)) and world_model_summary.get("preserved_artifacts", []):
            plan.append("verify preserved artifacts remain unchanged before termination")
        if bool(planner_controls.get("append_validation_subgoal", True)):
            plan.append("validate expected artifacts and forbidden artifacts before termination")
        try:
            max_initial_subgoals = max(1, int(planner_controls.get("max_initial_subgoals", 5)))
        except (TypeError, ValueError):
            max_initial_subgoals = 5
        deduped: list[str] = []
        for item in plan:
            normalized = str(item).strip()
            if not normalized or normalized in deduped:
                continue
            deduped.append(normalized)
        if len(deduped) <= max_initial_subgoals:
            return deduped
        validation_step = "validate expected artifacts and forbidden artifacts before termination"
        preservation_verification_step = "verify preserved artifacts remain unchanged before termination"
        preservation_steps = [
            item
            for item in deduped
            if item.startswith("preserve required artifact ")
        ]
        must_keep: list[str] = []
        if validation_step in deduped and bool(planner_controls.get("append_validation_subgoal", True)):
            must_keep.append(validation_step)
        if preservation_verification_step in deduped and bool(planner_controls.get("append_preservation_subgoal", False)):
            must_keep.append(preservation_verification_step)
        for item in preservation_steps:
            if item not in must_keep:
                must_keep.append(item)
        trimmed = [item for item in deduped if item not in must_keep]
        budget = max(0, max_initial_subgoals - len(must_keep))
        selected = trimmed[:budget]
        ordered = [item for item in deduped if item in {*selected, *must_keep}]
        if ordered:
            return ordered[:max_initial_subgoals]
        return deduped[:max_initial_subgoals]

    def _refresh_planner_subgoals(self, state: AgentState) -> None:
        state.refresh_plan_progress(state.world_model_summary)
        verifier_hotspots = self._verifier_hotspot_subgoals(self._latest_failed_verification_reasons(state))
        learned_hotspots = self._learned_world_hotspot_entries(state)
        hotspot_subgoals = [*verifier_hotspots, *[str(entry.get("subgoal", "")).strip() for entry in learned_hotspots]]
        if not hotspot_subgoals:
            return
        learned_priority_by_goal = {
            str(entry.get("subgoal", "")).strip(): int(entry.get("priority", 0) or 0)
            for entry in learned_hotspots
            if str(entry.get("subgoal", "")).strip()
        }
        hotspot_subgoals = sorted(
            hotspot_subgoals,
            key=lambda goal: (
                -max(
                    self._subgoal_diagnosis_priority(state.diagnosis_for_subgoal(goal)),
                    learned_priority_by_goal.get(goal, 0),
                ),
                hotspot_subgoals.index(goal),
            ),
        )
        self._promote_prioritized_subgoals(state, prioritized=hotspot_subgoals)
        self._refresh_planner_recovery_artifact(state)

    def _resolve_role_before_decision(self, state: AgentState) -> str:
        base_role = self.role_coordinator.role_before_decision(state)
        return self._maybe_override_long_horizon_recovery_role(state, base_role)

    def _resolve_role_after_step(self, state: AgentState, *, verification_passed: bool) -> str:
        base_role = self.role_coordinator.role_after_step(
            state,
            verification_passed=verification_passed,
        )
        return self._maybe_override_long_horizon_recovery_role(state, base_role)

    def _maybe_override_long_horizon_recovery_role(self, state: AgentState, role: str) -> str:
        normalized_role = str(role or "executor").strip() or "executor"
        if state.world_horizon() != "long_horizon":
            return normalized_role
        learned_progress, learned_risk = state.learned_world_progress_and_risk()
        if bool(state.latest_state_transition.get("regressed", False)):
            return self._higher_priority_role(normalized_role, "critic")
        learned_hotspots = self._learned_world_hotspot_entries(state)
        if not learned_hotspots:
            return normalized_role
        recovery_pressure = state.long_horizon_recovery_pressure()
        if recovery_pressure >= 2:
            return self._higher_priority_role(normalized_role, "critic")
        if recovery_pressure >= 1:
            return self._higher_priority_role(normalized_role, "planner")
        if learned_risk >= 0.55 and learned_risk > learned_progress:
            return self._higher_priority_role(normalized_role, "planner")
        return normalized_role

    @staticmethod
    def _higher_priority_role(current_role: str, promoted_role: str) -> str:
        order = {"executor": 0, "planner": 1, "critic": 2}
        current = str(current_role or "executor").strip() or "executor"
        promoted = str(promoted_role or current).strip() or current
        if order.get(promoted, 0) > order.get(current, 0):
            return promoted
        return current

    @staticmethod
    def _promote_prioritized_subgoals(state: AgentState, *, prioritized: list[str]) -> None:
        prioritized_goals = [str(goal).strip() for goal in prioritized if str(goal).strip()]
        if not prioritized_goals:
            return
        remaining = [goal for goal in state.plan if str(goal).strip()]
        reordered: list[str] = []
        for goal in [*prioritized_goals, *remaining]:
            normalized = str(goal).strip()
            if not normalized or normalized in reordered:
                continue
            reordered.append(normalized)
        state.plan = reordered
        state.active_subgoal = reordered[0] if reordered else ""

    def _refresh_planner_recovery_artifact(self, state: AgentState) -> None:
        if state.world_horizon() != "long_horizon":
            state.planner_recovery_artifact = {}
            return
        active_subgoal = str(state.active_subgoal).strip()
        diagnosis = state.active_subgoal_diagnosis()
        if not active_subgoal or not isinstance(diagnosis, dict):
            state.planner_recovery_artifact = {}
            return
        if str(diagnosis.get("source_role", "")).strip().lower() != "critic":
            state.planner_recovery_artifact = {}
            return
        if not self._planner_recovery_surface_exhausted(state, active_subgoal):
            state.planner_recovery_artifact = {}
            return
        state.planner_recovery_artifact = self._build_planner_recovery_artifact(
            state,
            active_subgoal=active_subgoal,
            diagnosis=diagnosis,
        )

    def _planner_recovery_surface_exhausted(self, state: AgentState, goal: str) -> bool:
        if not state.history or not state.task.suggested_commands:
            return False
        if state.consecutive_failures <= 0 and state.consecutive_no_progress_steps <= 0:
            return False
        failed_commands = state.all_failed_command_signatures()
        last_command = str(state.last_action_signature).partition(":")[2]
        last_canonical = self._normalize_command(last_command)
        seen_recovery_candidate = False
        for command in state.task.suggested_commands[:5]:
            normalized = str(command).strip()
            canonical = self._normalize_command(normalized)
            if not canonical or not self._command_matches_subgoal_surface(state, goal=goal, command=normalized):
                continue
            seen_recovery_candidate = True
            if canonical in failed_commands:
                continue
            if canonical == last_canonical and state.repeated_action_count > 1:
                continue
            return False
        return seen_recovery_candidate

    def _build_planner_recovery_artifact(
        self,
        state: AgentState,
        *,
        active_subgoal: str,
        diagnosis: dict[str, object],
    ) -> dict[str, object]:
        path = str(diagnosis.get("path", "")).strip() or self._subgoal_path(active_subgoal)
        related_objectives = self._planner_recovery_related_objectives(
            state,
            primary_subgoal=active_subgoal,
        )
        ranked_objectives = self._planner_recovery_ranked_objectives(
            state,
            primary_subgoal=active_subgoal,
            related_objectives=related_objectives,
            diagnosis=diagnosis,
        )
        staged_plan_update = [
            str(item.get("objective", "")).strip()
            for item in ranked_objectives
            if isinstance(item, dict) and str(item.get("objective", "")).strip()
        ]
        rewritten_subgoal = self._planner_rewrite_subgoal(
            active_subgoal,
            path=path,
            related_objectives=related_objectives,
        )
        failed_commands = state.all_failed_command_signatures()
        stale_commands = [
            str(command).strip()
            for command in state.task.suggested_commands[:5]
            if self._command_matches_subgoal_surface(state, goal=active_subgoal, command=str(command).strip())
            and self._normalize_command(str(command).strip()) in failed_commands
        ]
        focus_paths = self._planner_recovery_focus_paths(state, primary_path=path)
        return {
            "kind": "planner_recovery_rewrite",
            "source_subgoal": active_subgoal,
            "rewritten_subgoal": rewritten_subgoal,
            "next_stage_objective": staged_plan_update[0] if staged_plan_update else rewritten_subgoal,
            "staged_plan_update": staged_plan_update[:4],
            "objective_kind": (
                "workflow_verifier_recovery"
                if any(str(item).startswith(("update workflow path ", "write workflow report ", "regenerate generated artifact ", "accept required branch ", "run workflow test ", "prepare workflow branch ")) for item in related_objectives)
                or active_subgoal.startswith(("update workflow path ", "write workflow report ", "regenerate generated artifact ", "accept required branch ", "run workflow test ", "prepare workflow branch "))
                else "artifact_recovery"
            ),
            "summary": str(diagnosis.get("summary", "")).strip(),
            "signals": [
                str(signal).strip()
                for signal in diagnosis.get("signals", [])
                if str(signal).strip()
            ][:4],
            "focus_path": path,
            "focus_paths": focus_paths,
            "related_objectives": related_objectives,
            "ranked_objectives": ranked_objectives[:4],
            "stale_commands": stale_commands[:3],
            "contract_outline": self._planner_recovery_contract_outline(
                active_subgoal,
                path=path,
                focus_paths=focus_paths,
                related_objectives=staged_plan_update or related_objectives,
            ),
            "source_role": "critic",
            "updated_step_index": int(diagnosis.get("updated_step_index", state.next_step_index()) or state.next_step_index()),
        }

    @staticmethod
    def _planner_rewrite_subgoal(goal: str, *, path: str, related_objectives: list[str]) -> str:
        workflow_objectives = [
            objective
            for objective in related_objectives
            if objective.startswith(
                (
                    "update workflow path ",
                    "write workflow report ",
                    "regenerate generated artifact ",
                    "accept required branch ",
                    "run workflow test ",
                    "prepare workflow branch ",
                )
            )
        ]
        if workflow_objectives:
            scope = []
            for objective in [goal, *workflow_objectives]:
                normalized = str(objective).strip()
                if normalized and normalized not in scope:
                    scope.append(normalized)
            return (
                "restore verifier-visible workflow state across "
                + ", ".join(scope[:4])
            )
        if str(goal).startswith("materialize expected artifact "):
            return f"reframe verifier-visible recovery for expected artifact {path or goal}"
        if str(goal).startswith("remove forbidden artifact "):
            return f"reframe verifier-visible recovery for forbidden artifact {path or goal}"
        if str(goal).startswith("update workflow path "):
            return f"reframe workflow recovery contract for {path or goal}"
        if str(goal).startswith("write workflow report "):
            return f"reframe report recovery contract for {path or goal}"
        return f"reframe verifier-visible recovery for {goal}"

    def _planner_recovery_related_objectives(self, state: AgentState, *, primary_subgoal: str) -> list[str]:
        related: list[str] = []
        summary = dict(state.world_model_summary or {})
        updated_workflow_paths = {
            str(path).strip() for path in summary.get("updated_workflow_paths", []) if str(path).strip()
        }
        updated_generated_paths = {
            str(path).strip() for path in summary.get("updated_generated_paths", []) if str(path).strip()
        }
        updated_report_paths = {
            str(path).strip() for path in summary.get("updated_report_paths", []) if str(path).strip()
        }
        for path in summary.get("workflow_expected_changed_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in updated_workflow_paths:
                related.append(f"update workflow path {normalized}")
        for path in summary.get("workflow_generated_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in updated_generated_paths:
                related.append(f"regenerate generated artifact {normalized}")
        for path in summary.get("workflow_report_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in updated_report_paths:
                related.append(f"write workflow report {normalized}")
        for branch in summary.get("workflow_required_merges", []):
            normalized = str(branch).strip()
            if normalized:
                related.append(f"accept required branch {normalized}")
        for branch in summary.get("workflow_branch_targets", []):
            normalized = str(branch).strip()
            if normalized:
                related.append(f"prepare workflow branch {normalized}")
        for label in summary.get("workflow_required_tests", []):
            normalized = str(label).strip()
            if normalized:
                related.append(f"run workflow test {normalized}")
        ordered: list[str] = []
        for objective in [primary_subgoal, *related]:
            normalized = str(objective).strip()
            if not normalized or normalized == primary_subgoal or normalized in ordered:
                continue
            ordered.append(normalized)
        return ordered[:6]

    def _planner_recovery_ranked_objectives(
        self,
        state: AgentState,
        *,
        primary_subgoal: str,
        related_objectives: list[str],
        diagnosis: dict[str, object],
    ) -> list[dict[str, object]]:
        ranked: list[dict[str, object]] = []
        diagnosis_path = str(diagnosis.get("path", "")).strip()
        for objective in related_objectives:
            normalized = str(objective).strip()
            if not normalized:
                continue
            kind = self._planner_recovery_objective_kind(normalized)
            target = self._planner_recovery_objective_target(normalized)
            attempt_pressure = self._planner_recovery_objective_attempt_pressure(state, normalized)
            satisfied = self._planner_recovery_objective_satisfied(state, normalized)
            score = self._planner_recovery_objective_base_score(kind)
            reasons: list[str] = []
            if target and target == diagnosis_path:
                score += 24
                reasons.append("matches the diagnosed hotspot")
            if self._planner_recovery_objective_blocks_verifier(state, normalized):
                score += 18
                reasons.append("still blocks verifier-visible progress")
            if satisfied:
                score -= 60
                reasons.append("already has matching execution evidence")
            else:
                reasons.append("still lacks matching execution evidence")
            if attempt_pressure > 0:
                score -= min(30, attempt_pressure * 8)
                reasons.append(f"attempt pressure={attempt_pressure}")
            if kind == "workflow_path" and normalized != primary_subgoal:
                score += 8
                reasons.append("expands beyond the exhausted hotspot")
            ranked.append(
                {
                    "objective": normalized,
                    "kind": kind,
                    "target": target,
                    "score": score,
                    "status": "satisfied" if satisfied else ("attempted" if attempt_pressure > 0 else "pending"),
                    "reason": "; ".join(reasons[:3]),
                }
            )
        ranked.sort(
            key=lambda item: (
                -int(item.get("score", 0) or 0),
                str(item.get("status", "")) != "pending",
                str(item.get("objective", "")),
            )
        )
        return ranked

    @staticmethod
    def _planner_recovery_objective_kind(objective: str) -> str:
        normalized = str(objective).strip()
        if normalized.startswith("update workflow path "):
            return "workflow_path"
        if normalized.startswith("regenerate generated artifact "):
            return "generated_artifact"
        if normalized.startswith("write workflow report "):
            return "workflow_report"
        if normalized.startswith("accept required branch "):
            return "required_merge"
        if normalized.startswith("prepare workflow branch "):
            return "branch_target"
        if normalized.startswith("run workflow test "):
            return "workflow_test"
        return "other"

    @classmethod
    def _planner_recovery_objective_target(cls, objective: str) -> str:
        normalized = str(objective).strip()
        for prefix in (
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
            "accept required branch ",
            "prepare workflow branch ",
            "run workflow test ",
        ):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix).strip()
        return ""

    @staticmethod
    def _planner_recovery_objective_base_score(kind: str) -> int:
        return {
            "workflow_path": 120,
            "generated_artifact": 105,
            "workflow_report": 95,
            "required_merge": 85,
            "branch_target": 75,
            "workflow_test": 65,
        }.get(str(kind).strip(), 50)

    def _planner_recovery_objective_blocks_verifier(self, state: AgentState, objective: str) -> bool:
        summary = dict(state.world_model_summary or {})
        kind = self._planner_recovery_objective_kind(objective)
        target = self._planner_recovery_objective_target(objective)
        if not target:
            return False
        if kind == "workflow_path":
            unresolved = {
                str(item).strip()
                for item in (
                    list(summary.get("missing_expected_artifacts", []))
                    + list(summary.get("unsatisfied_expected_contents", []))
                )
                if str(item).strip()
            }
            return target in unresolved
        if kind == "generated_artifact":
            return target not in {
                str(item).strip() for item in summary.get("updated_generated_paths", []) if str(item).strip()
            }
        if kind == "workflow_report":
            return target not in {
                str(item).strip() for item in summary.get("updated_report_paths", []) if str(item).strip()
            }
        if kind == "required_merge":
            return True
        if kind == "branch_target":
            return True
        if kind == "workflow_test":
            return True
        return False

    def _planner_recovery_objective_attempt_pressure(self, state: AgentState, objective: str) -> int:
        total = 0
        for step in state.history[-6:]:
            if self._planner_recovery_command_aligns_objective(str(step.content), objective):
                total += 2 if not step.verification.get("passed", False) else 1
        return total

    def _planner_recovery_objective_satisfied(self, state: AgentState, objective: str) -> bool:
        kind = self._planner_recovery_objective_kind(objective)
        target = self._planner_recovery_objective_target(objective)
        summary = dict(state.world_model_summary or {})
        if not target:
            return False
        if kind == "workflow_path":
            return target in {
                str(item).strip() for item in summary.get("updated_workflow_paths", []) if str(item).strip()
            }
        if kind == "generated_artifact":
            return target in {
                str(item).strip() for item in summary.get("updated_generated_paths", []) if str(item).strip()
            }
        if kind == "workflow_report":
            return target in {
                str(item).strip() for item in summary.get("updated_report_paths", []) if str(item).strip()
            }
        recent_successes = [
            str(step.content)
            for step in state.history[-6:]
            if step.verification.get("passed", False)
        ]
        return any(self._planner_recovery_command_aligns_objective(command, objective) for command in recent_successes)

    def _planner_recovery_command_aligns_objective(self, command: str, objective: str) -> bool:
        normalized_command = str(command).strip().lower()
        target = self._planner_recovery_objective_target(objective).lower()
        kind = self._planner_recovery_objective_kind(objective)
        if not normalized_command or not target:
            return False
        if kind in {"workflow_path", "generated_artifact", "workflow_report"}:
            return target in normalized_command
        if kind == "required_merge":
            return "git merge" in normalized_command and target in normalized_command
        if kind == "branch_target":
            return target in normalized_command and any(
                token in normalized_command for token in ("git checkout", "git switch", "git branch")
            )
        if kind == "workflow_test":
            objective_tokens = {token for token in target.split() if len(token) > 2}
            return (
                ("test" in normalized_command or "pytest" in normalized_command)
                and bool(objective_tokens.intersection(set(normalized_command.split())))
            )
        return False

    def _planner_recovery_focus_paths(self, state: AgentState, *, primary_path: str) -> list[str]:
        focus_paths: list[str] = []
        for candidate in (
            primary_path,
            *[str(item).strip() for item in state.world_model_summary.get("missing_expected_artifacts", [])[:3]],
            *[str(item).strip() for item in state.world_model_summary.get("unsatisfied_expected_contents", [])[:3]],
            *[str(item).strip() for item in state.world_model_summary.get("present_forbidden_artifacts", [])[:3]],
            *[str(item).strip() for item in state.world_model_summary.get("updated_workflow_paths", [])[:2]],
        ):
            normalized = str(candidate).strip()
            if normalized and normalized not in focus_paths:
                focus_paths.append(normalized)
        return focus_paths[:4]

    @staticmethod
    def _planner_recovery_contract_outline(
        goal: str,
        *,
        path: str,
        focus_paths: list[str],
        related_objectives: list[str],
    ) -> list[str]:
        primary = path or str(goal).strip()
        outline = [
            f"inspect current repo/workspace state around {primary}",
            f"define the next verifier-visible milestone for {primary}",
        ]
        if related_objectives:
            outline.append(
                "sequence related verifier obligations: " + ", ".join(related_objectives[:3])
            )
        outline.append("choose a new command path outside the exhausted task-contract repair set")
        if focus_paths and focus_paths[0] != primary:
            outline.insert(1, f"reconcile related verifier hotspots: {', '.join(focus_paths[:3])}")
        return outline[:4]

    def _command_matches_subgoal_surface(self, state: AgentState, *, goal: str, command: str) -> bool:
        normalized_goal = str(goal).strip()
        normalized_command = str(command).strip().lower()
        if not normalized_goal or not normalized_command:
            return False
        path = self._subgoal_path(normalized_goal).lower()
        if path and path in normalized_command:
            return True
        if normalized_goal.startswith("materialize expected artifact "):
            return any(str(item).strip().lower() in normalized_command for item in state.task.expected_files)
        if normalized_goal.startswith("remove forbidden artifact "):
            return any(str(item).strip().lower() in normalized_command for item in state.task.forbidden_files)
        if normalized_goal.startswith("update workflow path "):
            verifier = state.task.metadata.get("semantic_verifier", {})
            verifier = verifier if isinstance(verifier, dict) else {}
            workflow_paths = [
                *verifier.get("expected_changed_paths", []),
                *state.world_model_summary.get("updated_workflow_paths", []),
            ]
            return any(str(item).strip().lower() in normalized_command for item in workflow_paths)
        if normalized_goal.startswith("write workflow report "):
            verifier = state.task.metadata.get("semantic_verifier", {})
            verifier = verifier if isinstance(verifier, dict) else {}
            report_rules = verifier.get("report_rules", [])
            report_paths = [
                str(rule.get("path", "")).strip()
                for rule in report_rules
                if isinstance(rule, dict)
            ]
            return any(str(item).strip().lower() in normalized_command for item in report_paths)
        return False

    def _attach_critic_subgoal_diagnoses(
        self,
        state: AgentState,
        *,
        step_index: int,
        step_active_subgoal: str,
        failure_signals: list[str],
        failure_origin: str,
        command_result,
    ) -> None:
        ordered_signals = self._ordered_failure_signals(
            failure_signals=failure_signals,
            failure_origin=failure_origin,
            command_result=command_result,
        )
        if not ordered_signals:
            return
        for goal in self._diagnosis_candidate_subgoals(state, step_active_subgoal=step_active_subgoal):
            diagnosis = self._build_subgoal_failure_diagnosis(
                state,
                goal=goal,
                step_index=step_index,
                ordered_signals=ordered_signals,
                command_result=command_result,
                step_active_subgoal=step_active_subgoal,
            )
            if not diagnosis:
                continue
            existing = state.diagnosis_for_subgoal(goal)
            merged_signals: list[str] = []
            for signal in [*existing.get("signals", []), *diagnosis.get("signals", [])]:
                normalized = str(signal).strip()
                if normalized and normalized not in merged_signals:
                    merged_signals.append(normalized)
            diagnosis["signals"] = merged_signals
            state.subgoal_diagnoses[str(goal).strip()] = diagnosis

    def _attach_verifier_subgoal_diagnoses(
        self,
        state: AgentState,
        *,
        step_index: int,
        verification_reasons: list[object],
    ) -> None:
        for entry in self._verifier_failure_entries(verification_reasons):
            goal = str(entry.get("subgoal", "")).strip()
            if not goal:
                continue
            existing = state.diagnosis_for_subgoal(goal)
            summary_parts: list[str] = []
            for candidate in [existing.get("summary", ""), entry.get("summary", "")]:
                normalized = str(candidate).strip()
                if normalized and normalized not in summary_parts:
                    summary_parts.append(normalized)
            signals: list[str] = []
            for signal in [*existing.get("signals", []), "verifier_failure"]:
                normalized = str(signal).strip()
                if normalized and normalized not in signals:
                    signals.append(normalized)
            path = str(existing.get("path", "")).strip() or str(entry.get("path", "")).strip() or self._subgoal_path(goal)
            state.subgoal_diagnoses[goal] = {
                "summary": "; ".join(summary_parts[:2]),
                "signals": signals,
                "path": path,
                "source_role": "verifier",
                "updated_step_index": step_index,
            }

    def _diagnosis_candidate_subgoals(self, state: AgentState, *, step_active_subgoal: str) -> list[str]:
        candidates: list[str] = []
        for goal in [step_active_subgoal, state.active_subgoal, *self._learned_world_hotspot_subgoals(state), *state.plan]:
            normalized = str(goal).strip()
            if not normalized or normalized in candidates:
                continue
            candidates.append(normalized)
        return candidates[:6]

    def _build_subgoal_failure_diagnosis(
        self,
        state: AgentState,
        *,
        goal: str,
        step_index: int,
        ordered_signals: list[str],
        command_result,
        step_active_subgoal: str,
    ) -> dict[str, object]:
        normalized_goal = str(goal).strip()
        if not normalized_goal:
            return {}
        path = self._subgoal_path(normalized_goal)
        regressed_paths = {
            str(item).strip() for item in state.latest_state_transition.get("regressions", []) if str(item).strip()
        }
        goal_signals: list[str] = []
        for signal in ordered_signals:
            normalized_signal = str(signal).strip()
            if not normalized_signal:
                continue
            if normalized_signal in {"command_failure", "command_timeout", "inference_failure", "retrieval_failure"}:
                if normalized_goal != str(step_active_subgoal).strip():
                    continue
            if normalized_signal == "state_regression":
                if regressed_paths and path and path not in regressed_paths:
                    continue
            if normalized_signal not in goal_signals:
                goal_signals.append(normalized_signal)
        evidence: list[str] = []
        if path:
            if normalized_goal.startswith("remove forbidden artifact "):
                present = {str(item).strip() for item in state.world_model_summary.get("present_forbidden_artifacts", [])}
                if path in present:
                    evidence.append(f"{path} is still present")
            elif normalized_goal.startswith("materialize expected artifact "):
                missing = {str(item).strip() for item in state.world_model_summary.get("missing_expected_artifacts", [])}
                unsatisfied = {
                    str(item).strip() for item in state.world_model_summary.get("unsatisfied_expected_contents", [])
                }
                if path in missing:
                    evidence.append(f"{path} is still missing")
                if path in unsatisfied:
                    evidence.append(f"{path} content is still unsatisfied")
            elif normalized_goal.startswith("preserve required artifact "):
                changed = {str(item).strip() for item in state.world_model_summary.get("changed_preserved_artifacts", [])}
                if path in changed:
                    evidence.append(f"{path} changed unexpectedly")
        if not evidence and path:
            evidence.append(f"{path} remains on the critical path")
        summary = self._subgoal_diagnosis_summary(goal_signals, command_result=command_result)
        if evidence and summary:
            summary = f"{'; '.join(evidence)}; {summary}"
        elif evidence:
            summary = "; ".join(evidence)
        return {
            "summary": summary,
            "signals": list(goal_signals),
            "path": path,
            "source_role": "critic",
            "updated_step_index": step_index,
        }

    def _latest_failed_verification_reasons(self, state: AgentState) -> list[str]:
        if not state.history:
            return []
        verification = state.history[-1].verification if isinstance(state.history[-1].verification, dict) else {}
        if bool(verification.get("passed", False)):
            return []
        return [str(reason).strip() for reason in verification.get("reasons", []) if str(reason).strip()]

    def _verifier_hotspot_subgoals(self, verification_reasons: list[object]) -> list[str]:
        hotspots: list[str] = []
        for entry in self._verifier_failure_entries(verification_reasons):
            goal = str(entry.get("subgoal", "")).strip()
            if goal and goal not in hotspots:
                hotspots.append(goal)
        return hotspots

    def _verifier_failure_entries(self, verification_reasons: list[object]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for reason in verification_reasons:
            text = str(reason).strip()
            if not text or text == "verification passed":
                continue
            subgoal = ""
            path = ""
            if text.startswith("missing expected file: "):
                path = text.removeprefix("missing expected file: ").strip()
                subgoal = f"materialize expected artifact {path}"
            elif text.startswith("missing expected file content target: "):
                path = text.removeprefix("missing expected file content target: ").strip()
                subgoal = f"materialize expected artifact {path}"
            elif text.startswith("unexpected file content: "):
                path = text.removeprefix("unexpected file content: ").strip()
                subgoal = f"materialize expected artifact {path}"
            elif text.startswith("forbidden file present: "):
                path = text.removeprefix("forbidden file present: ").strip()
                subgoal = f"remove forbidden artifact {path}"
            elif text.startswith("git diff missing expected path: "):
                path = text.removeprefix("git diff missing expected path: ").strip()
                subgoal = f"update workflow path {path}"
            elif text.startswith("git diff unexpectedly changed preserved path: "):
                path = text.removeprefix("git diff unexpectedly changed preserved path: ").strip()
                subgoal = f"preserve required artifact {path}"
            elif text.startswith("semantic report missing: "):
                path = text.removeprefix("semantic report missing: ").strip()
                subgoal = f"write workflow report {path}"
            elif text.startswith("semantic report missing phrase "):
                path = text.rsplit(": ", 1)[-1].strip()
                subgoal = f"write workflow report {path}"
            elif text.startswith("semantic report does not cover "):
                path = text.rsplit(": ", 1)[-1].strip()
                subgoal = f"write workflow report {path}"
            elif text.startswith("generated artifact missing: "):
                path = text.removeprefix("generated artifact missing: ").strip()
                subgoal = f"regenerate generated artifact {path}"
            elif text.startswith("generated artifact not recorded in git diff: "):
                path = text.removeprefix("generated artifact not recorded in git diff: ").strip()
                subgoal = f"regenerate generated artifact {path}"
            elif text.startswith("git conflict remains unresolved: "):
                path = text.removeprefix("git conflict remains unresolved: ").strip()
                subgoal = f"update workflow path {path}"
            elif text.startswith("conflict markers still present after merge resolution: "):
                path = text.removeprefix("conflict markers still present after merge resolution: ").strip()
                subgoal = f"update workflow path {path}"
            elif text.startswith("required worker branch not accepted into "):
                branch = text.rsplit(": ", 1)[-1].strip()
                path = branch
                subgoal = f"accept required branch {branch}"
            else:
                match = re.fullmatch(r"(.+?) exited with code \d+", text)
                if match is not None:
                    label = str(match.group(1)).strip()
                    if label:
                        path = label
                        subgoal = f"run workflow test {label}"
            if subgoal:
                entries.append({"subgoal": subgoal, "summary": text, "path": path})
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            key = (entry["subgoal"], entry["summary"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    def _learned_world_hotspot_entries(self, state: AgentState) -> list[dict[str, object]]:
        return self.world_model.prioritized_long_horizon_hotspots(
            state.task,
            state.world_model_summary,
            latest_transition=state.latest_state_transition,
            latent_state_summary=state.latent_state_summary,
            active_subgoal=state.active_subgoal,
        )

    def _learned_world_hotspot_subgoals(self, state: AgentState) -> list[str]:
        hotspot_subgoals: list[str] = []
        for entry in self._learned_world_hotspot_entries(state):
            subgoal = str(entry.get("subgoal", "")).strip()
            if subgoal and subgoal not in hotspot_subgoals:
                hotspot_subgoals.append(subgoal)
        return hotspot_subgoals

    @staticmethod
    def _ordered_failure_signals(
        *,
        failure_signals: list[str],
        failure_origin: str,
        command_result,
    ) -> list[str]:
        ordered: list[str] = []

        def append(signal: object) -> None:
            normalized = str(signal).strip()
            if normalized and normalized not in ordered:
                ordered.append(normalized)

        for signal in failure_signals:
            append(signal)
        append(failure_origin)
        if command_result is not None:
            if bool(getattr(command_result, "timed_out", False)):
                append("command_timeout")
            elif int(getattr(command_result, "exit_code", 0)) != 0:
                append("command_failure")
        return ordered

    @staticmethod
    def _subgoal_path(goal: str) -> str:
        normalized = str(goal).strip()
        for prefix in (
            "prepare workflow branch ",
            "accept required branch ",
            "remove forbidden artifact ",
            "materialize expected artifact ",
            "preserve required artifact ",
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
        ):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix).strip()
        return ""

    @staticmethod
    def _subgoal_diagnosis_summary(ordered_signals: list[str], *, command_result) -> str:
        parts: list[str] = []
        for signal in ordered_signals:
            if signal == "state_regression":
                parts.append("recent step regressed workspace state")
            elif signal == "no_state_progress":
                parts.append("recent step produced no state progress")
            elif signal == "command_timeout":
                parts.append("recent command timed out")
            elif signal == "command_failure":
                exit_code = int(getattr(command_result, "exit_code", 1)) if command_result is not None else 1
                parts.append(f"recent command exited {exit_code}")
            elif signal == "inference_failure":
                parts.append("recent policy inference failed")
            elif signal == "retrieval_failure":
                parts.append("recent retrieval guidance failed")
            elif signal:
                parts.append(signal.replace("_", " "))
        deduped: list[str] = []
        for part in parts:
            normalized = str(part).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return "; ".join(deduped[:3])

    @staticmethod
    def _subgoal_diagnosis_priority(diagnosis: dict[str, object]) -> int:
        signals = diagnosis.get("signals", []) if isinstance(diagnosis, dict) else []
        weights = {
            "state_regression": 5,
            "verifier_failure": 4,
            "command_timeout": 4,
            "command_failure": 3,
            "inference_failure": 3,
            "retrieval_failure": 3,
            "no_state_progress": 2,
        }
        return max((weights.get(str(signal).strip(), 1) for signal in signals), default=0)

    @staticmethod
    def _float_value(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _workflow_plan_steps(task: TaskSpec, *, planner_controls: dict[str, object] | None = None) -> list[str]:
        metadata = dict(task.metadata)
        verifier = metadata.get("semantic_verifier", {})
        guard = metadata.get("workflow_guard", {})
        contract = dict(verifier) if isinstance(verifier, dict) else {}
        workflow_guard = dict(guard) if isinstance(guard, dict) else {}
        controls = planner_controls or {}
        steps: list[str] = []
        for branch in (
            str(contract.get("expected_branch", "")).strip(),
            str(workflow_guard.get("worker_branch", "")).strip(),
            str(workflow_guard.get("target_branch", "")).strip(),
        ):
            if branch:
                steps.append(f"prepare workflow branch {branch}")
        for branch in contract.get("required_merged_branches", []):
            normalized = str(branch).strip()
            if normalized:
                steps.append(f"accept required branch {normalized}")
        preserved_steps: list[str] = []
        for path in contract.get("preserved_paths", []):
            normalized = str(path).strip()
            if normalized:
                preserved_steps.append(f"preserve required artifact {normalized}")
        if bool(controls.get("include_preserved_artifact_steps", True)):
            try:
                max_preserved = max(0, int(controls.get("max_preserved_artifacts", len(preserved_steps))))
            except (TypeError, ValueError):
                max_preserved = len(preserved_steps)
            preserved_steps = preserved_steps[:max_preserved]
            if bool(controls.get("prefer_preserved_artifacts_first", False)):
                steps.extend(preserved_steps)
        for path in contract.get("expected_changed_paths", []):
            normalized = str(path).strip()
            if normalized:
                steps.append(f"update workflow path {normalized}")
        for path in contract.get("generated_paths", []):
            normalized = str(path).strip()
            if normalized:
                steps.append(f"regenerate generated artifact {normalized}")
        if preserved_steps and not bool(controls.get("prefer_preserved_artifacts_first", False)):
            steps.extend(preserved_steps)
        for rule in contract.get("test_commands", []):
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label", "")).strip() or "workflow test command"
            steps.append(f"run workflow test {label}")
        for rule in contract.get("report_rules", []):
            if not isinstance(rule, dict):
                continue
            path = str(rule.get("path", "")).strip()
            if path:
                steps.append(f"write workflow report {path}")
        return steps

    def _setup_failure_episode(
        self,
        *,
        task: TaskSpec,
        workspace: Path,
        checkpoint_path: Path | None,
        state: AgentState,
        setup_history: list[dict[str, object]],
    ) -> EpisodeRecord:
        episode = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(workspace),
            success=False,
            steps=[],
            task_metadata=dict(task.metadata),
            task_contract=self._task_contract_payload(task),
            plan=[],
            graph_summary={},
            universe_summary={},
            world_model_summary={},
            termination_reason="setup_failed",
        )
        episode_path = None
        if self.config.persist_episode_memory:
            episode_path = self.memory.save(episode)
        persist_episode_learning_candidates(
            episode,
            config=self.config,
            episode_storage=(
                episode_storage_metadata(self.config.trajectories_root, episode_path)
                if episode_path is not None
                else None
            ),
        )
        if checkpoint_path is not None:
            self._write_checkpoint(
                checkpoint_path,
                task=task,
                workspace=workspace,
                state=state,
                success=False,
                status="completed",
                termination_reason="setup_failed",
                episode=episode,
                setup_history=setup_history,
                phase="execute",
            )
        return episode

    @staticmethod
    def _task_contract_payload(task: TaskSpec) -> dict[str, object]:
        metadata = dict(task.metadata)
        return {
            "prompt": task.prompt,
            "workspace_subdir": task.workspace_subdir,
            "setup_commands": list(task.setup_commands),
            "success_command": task.success_command,
            "suggested_commands": list(task.suggested_commands),
            "expected_files": list(task.expected_files),
            "expected_output_substrings": list(task.expected_output_substrings),
            "forbidden_files": list(task.forbidden_files),
            "forbidden_output_substrings": list(task.forbidden_output_substrings),
            "expected_file_contents": dict(task.expected_file_contents),
            "max_steps": task.max_steps,
            "metadata": metadata,
            "synthetic_edit_plan": [
                dict(step)
                for step in metadata.get("synthetic_edit_plan", [])
                if isinstance(step, dict)
            ],
            "synthetic_edit_candidates": [
                dict(step)
                for step in metadata.get("synthetic_edit_candidates", [])
                if isinstance(step, dict)
            ],
        }

    def _planner_controls(self) -> dict[str, object]:
        if not self.config.use_prompt_proposals:
            return {}
        path = self.config.prompt_proposals_path
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return retained_planner_controls(payload)

    @staticmethod
    def _available_skill_count(state: AgentState) -> int:
        return len(state.available_skills)

    @staticmethod
    def _retrieval_candidate_count(state: AgentState) -> int:
        if state.context_packet is None:
            return 0
        retrieval = state.context_packet.retrieval
        return len(retrieval.get("branch_scoped", [])) + len(retrieval.get("global", []))

    @staticmethod
    def _retrieval_evidence_count(state: AgentState) -> int:
        if state.context_packet is None:
            return 0
        guidance = state.context_packet.control.get("retrieval_guidance", {})
        return len(guidance.get("evidence", []))

    @staticmethod
    def _retrieval_command_match(state: AgentState, command: str) -> bool:
        if state.context_packet is None or not command.strip():
            return False
        guidance = state.context_packet.control.get("retrieval_guidance", {})
        recommended_commands = guidance.get("recommended_commands", [])
        normalized_command = AgentKernel._normalize_command(command)
        return any(
            AgentKernel._normalize_command(candidate) == normalized_command
            for candidate in recommended_commands
        )

    @staticmethod
    def _path_confidence(state: AgentState) -> float:
        if state.context_packet is None:
            return 0.0
        return float(state.context_packet.control.get("path_confidence", 0.0))

    @staticmethod
    def _trust_retrieval(state: AgentState) -> bool:
        if state.context_packet is None:
            return False
        return bool(state.context_packet.control.get("trust_retrieval", False))

    @staticmethod
    def _normalize_command(command: str) -> str:
        return SkillLibrary._normalize_command(command)

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    @staticmethod
    def _state_from_checkpoint(task: TaskSpec, payload: dict[str, object]) -> AgentState:
        state = AgentState(task=task)
        state.history = [
            StepRecord(**step)
            for step in payload.get("history", [])
            if isinstance(step, dict)
        ]
        history_archive = payload.get("history_archive", {})
        if isinstance(history_archive, dict):
            state.history_archive = dict(history_archive)
        state.recent_workspace_summary = str(payload.get("recent_workspace_summary", ""))
        graph_summary = payload.get("graph_summary", {})
        if isinstance(graph_summary, dict):
            state.graph_summary = dict(graph_summary)
        universe_summary = payload.get("universe_summary", {})
        if isinstance(universe_summary, dict):
            state.universe_summary = dict(universe_summary)
        world_model_summary = payload.get("world_model_summary", {})
        if isinstance(world_model_summary, dict):
            state.world_model_summary = dict(world_model_summary)
        workspace_snapshot = payload.get("workspace_snapshot", {})
        if isinstance(workspace_snapshot, dict):
            state.workspace_snapshot = {str(key): str(value) for key, value in workspace_snapshot.items()}
        latest_state_transition = payload.get("latest_state_transition", {})
        if isinstance(latest_state_transition, dict):
            state.latest_state_transition = dict(latest_state_transition)
        latent_state_summary = payload.get("latent_state_summary", {})
        if isinstance(latent_state_summary, dict):
            state.latent_state_summary = dict(latent_state_summary)
        state.plan = [str(item) for item in payload.get("plan", [])]
        state.initial_plan = [str(item) for item in payload.get("initial_plan", [])]
        state.active_subgoal = str(payload.get("active_subgoal", ""))
        subgoal_diagnoses = payload.get("subgoal_diagnoses", {})
        if isinstance(subgoal_diagnoses, dict):
            state.subgoal_diagnoses = {
                str(goal): dict(diagnosis)
                for goal, diagnosis in subgoal_diagnoses.items()
                if str(goal).strip() and isinstance(diagnosis, dict)
            }
        planner_recovery_artifact = payload.get("planner_recovery_artifact", {})
        if isinstance(planner_recovery_artifact, dict):
            state.planner_recovery_artifact = dict(planner_recovery_artifact)
        software_work_stage_state = payload.get("software_work_stage_state", {})
        if isinstance(software_work_stage_state, dict):
            state.software_work_stage_state = dict(software_work_stage_state)
        state.current_role = str(payload.get("current_role", "executor"))
        state.consecutive_failures = int(payload.get("consecutive_failures", 0))
        state.repeated_action_count = int(payload.get("repeated_action_count", 0))
        state.consecutive_no_progress_steps = int(payload.get("consecutive_no_progress_steps", 0))
        state.last_action_signature = str(payload.get("last_action_signature", ""))
        state.termination_reason = str(payload.get("termination_reason", ""))
        return state

    @staticmethod
    def _setup_history_from_checkpoint(payload: dict[str, object]) -> list[dict[str, object]]:
        return [dict(entry) for entry in payload.get("setup_history", []) if isinstance(entry, dict)]

    @staticmethod
    def _completed_setup_command_count(task: TaskSpec, setup_history: list[dict[str, object]]) -> int:
        completed = 0
        for index, command in enumerate(task.setup_commands):
            if index >= len(setup_history):
                break
            entry = setup_history[index]
            if (
                str(entry.get("command", "")).strip() == command
                and int(entry.get("exit_code", 1)) == 0
                and not bool(entry.get("timed_out", False))
            ):
                completed += 1
                continue
            break
        return completed

    @staticmethod
    def _episode_from_payload(payload: dict[str, object]) -> EpisodeRecord:
        return EpisodeRecord(
            task_id=str(payload.get("task_id", "")),
            prompt=str(payload.get("prompt", "")),
            workspace=str(payload.get("workspace", "")),
            success=bool(payload.get("success", False)),
            steps=[
                StepRecord(**step)
                for step in payload.get("steps", [])
                if isinstance(step, dict)
            ],
            task_metadata=dict(payload.get("task_metadata", {}))
            if isinstance(payload.get("task_metadata", {}), dict)
            else {},
            task_contract=dict(payload.get("task_contract", {}))
            if isinstance(payload.get("task_contract", {}), dict)
            else {},
            plan=[str(item) for item in payload.get("plan", [])],
            graph_summary=dict(payload.get("graph_summary", {}))
            if isinstance(payload.get("graph_summary", {}), dict)
            else {},
            universe_summary=dict(payload.get("universe_summary", {}))
            if isinstance(payload.get("universe_summary", {}), dict)
            else {},
            world_model_summary=dict(payload.get("world_model_summary", {}))
            if isinstance(payload.get("world_model_summary", {}), dict)
            else {},
            history_archive=dict(payload.get("history_archive", {}))
            if isinstance(payload.get("history_archive", {}), dict)
            else {},
            termination_reason=str(payload.get("termination_reason", "")),
        )

    def _write_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        task: TaskSpec,
        workspace: Path,
        state: AgentState,
        success: bool,
        status: str,
        termination_reason: str,
        episode: EpisodeRecord | None = None,
        setup_history: list[dict[str, object]] | None = None,
        phase: str = "execute",
    ) -> Path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_recent_window = max(1, int(self.config.checkpoint_history_step_window))
        checkpoint_history = state.history[-checkpoint_recent_window:]
        checkpoint_archive = dict(state.history_archive)
        omitted_recent_steps = state.history[:-checkpoint_recent_window]
        if omitted_recent_steps:
            archived_state = AgentState(task=task)
            archived_state.history_archive = checkpoint_archive
            archived_state._append_history_archive(
                omitted_recent_steps,
                summary_char_limit=self.config.history_archive_summary_max_chars,
            )
            checkpoint_archive = dict(archived_state.history_archive)
        payload: dict[str, object] = {
            "task_id": task.task_id,
            "workspace": str(workspace),
            "status": status,
            "success": success,
            "termination_reason": termination_reason,
            "phase": phase,
            "recent_workspace_summary": state.recent_workspace_summary,
            "graph_summary": dict(state.graph_summary),
            "universe_summary": dict(state.universe_summary),
            "world_model_summary": dict(state.world_model_summary),
            "workspace_snapshot": dict(state.workspace_snapshot),
            "latest_state_transition": dict(state.latest_state_transition),
            "latent_state_summary": dict(state.latent_state_summary),
            "plan": list(state.plan),
            "initial_plan": list(state.initial_plan),
            "active_subgoal": state.active_subgoal,
            "subgoal_diagnoses": dict(state.subgoal_diagnoses),
            "planner_recovery_artifact": dict(state.planner_recovery_artifact),
            "software_work_stage_state": dict(state.software_work_stage_state),
            "current_role": state.current_role,
            "consecutive_failures": state.consecutive_failures,
            "repeated_action_count": state.repeated_action_count,
            "consecutive_no_progress_steps": state.consecutive_no_progress_steps,
            "last_action_signature": state.last_action_signature,
            "history": [asdict(step) for step in checkpoint_history],
            "history_archive": checkpoint_archive,
            "task_contract": {
                "prompt": task.prompt,
                "workspace_subdir": task.workspace_subdir,
                "setup_commands": list(task.setup_commands),
                "success_command": task.success_command,
                "suggested_commands": list(task.suggested_commands),
                "expected_files": list(task.expected_files),
                "expected_output_substrings": list(task.expected_output_substrings),
                "forbidden_files": list(task.forbidden_files),
                "forbidden_output_substrings": list(task.forbidden_output_substrings),
                "expected_file_contents": dict(task.expected_file_contents),
                "max_steps": task.max_steps,
                "metadata": dict(task.metadata),
            },
            "setup_history": [dict(entry) for entry in (setup_history or [])],
        }
        if episode is not None:
            payload["episode"] = episode.to_dict()
        atomic_write_json(checkpoint_path, payload, config=self.config)
        return checkpoint_path

    @staticmethod
    def classify_policy_failure(exc: Exception) -> str:
        message = str(exc).lower()
        module_name = exc.__class__.__module__.lower()
        if (
            isinstance(exc, (TimeoutError, url_error.URLError, OSError))
            or "vllm" in message
            or "ollama" in message
            or "connection refused" in message
            or "timed out" in message
            or "llm request" in message
        ):
            return "inference_failure"
        if (
            "tolbert" in message
            or "retrieval" in message
            or "context packet" in message
            or "tolbert" in module_name
        ):
            return "retrieval_failure"
        if "verify" in message or "verifier" in message:
            return "verifier_failure"
        return "controller_failure"

    @staticmethod
    def policy_error_decision(exc: Exception, *, failure_origin: str = ""):
        from .actions import RESPOND
        from .schemas import ActionDecision

        suffix = f" [{failure_origin}]" if failure_origin else ""
        return ActionDecision(
            thought=f"Policy error: {exc.__class__.__name__}",
            action=RESPOND,
            content=f"{exc}{suffix}",
            done=True,
        )
