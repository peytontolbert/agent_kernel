from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
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
from .multi_agent import RoleCoordinator
from .policy import LLMDecisionPolicy, Policy, SkillLibrary
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

        remaining_steps = max(0, min(task.max_steps, self.config.max_steps) - len(state.history))
        for _ in range(remaining_steps):
            if self.config.use_role_specialization:
                state.current_role = self.role_coordinator.role_before_decision(state)
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
                    "completed_steps": len(state.history),
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
                    "completed_steps": len(state.history),
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
                decision = self.policy_error_decision(exc, failure_origin=failure_origin)
            finally:
                self.policy.set_decision_progress_callback(None)
            self._emit_progress_callback(
                progress_callback,
                {
                    "event": "decision_ready",
                    "step_index": step_index,
                    "step_stage": "decision_ready",
                    "completed_steps": len(state.history),
                    "active_subgoal": step_active_subgoal,
                    "acting_role": step_role,
                    "decision_action": str(decision.action),
                    "decision_done": bool(decision.done),
                    "decision_source": str(decision.decision_source),
                },
            )
            command_result = None
            verification = {"passed": False, "reasons": ["no command executed"]}

            if decision.action == CODE_EXECUTE:
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
            )
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
                proposal_metadata=dict(decision.proposal_metadata),
                shadow_decision=dict(decision.shadow_decision),
                latent_state_summary=dict(state.latent_state_summary),
            )
            state.history.append(step_record)
            self._emit_progress_callback(
                progress_callback,
                {
                    "event": "step_complete",
                    "step_index": step_index,
                    "step_stage": "step_complete",
                    "completed_steps": len(state.history),
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
                state.current_role = self.role_coordinator.role_after_step(
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
            termination_reason=termination_reason,
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
            termination_reason=str(payload.get("termination_reason", "")),
        )

    @staticmethod
    def _write_checkpoint(
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
            "current_role": state.current_role,
            "consecutive_failures": state.consecutive_failures,
            "repeated_action_count": state.repeated_action_count,
            "consecutive_no_progress_steps": state.consecutive_no_progress_steps,
            "last_action_signature": state.last_action_signature,
            "history": [asdict(step) for step in state.history],
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
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
