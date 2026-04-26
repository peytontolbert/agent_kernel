from __future__ import annotations

from pathlib import Path
import subprocess
from urllib import error as url_error

from .config import KernelConfig
from .llm import HybridDecoderClient, MockLLMClient, ModelStackClient, OllamaClient, VLLMClient
from .memory import EpisodeMemory, GraphMemory
from .extensions.planner_recovery import (
    build_planner_recovery_artifact,
    command_matches_subgoal_surface,
    planner_recovery_command_aligns_objective,
    planner_recovery_contract_outline,
    planner_recovery_focus_paths,
    planner_recovery_objective_attempt_pressure,
    planner_recovery_objective_base_score,
    planner_recovery_objective_blocks_verifier,
    planner_recovery_objective_kind,
    planner_recovery_objective_satisfied,
    planner_recovery_objective_target,
    planner_recovery_ranked_objectives,
    planner_recovery_related_objectives,
    planner_recovery_surface_exhausted,
    planner_rewrite_subgoal,
    refresh_planner_recovery_artifact,
    subgoal_path,
)
from .extensions.syntax_motor import summarize_python_edit_step
from .extensions.multi_agent import RoleCoordinator
from .policy import LLMDecisionPolicy, Policy, SkillLibrary
from .ops.loop_checkpointing import (
    completed_setup_command_count,
    episode_from_payload,
    load_checkpoint,
    setup_history_from_checkpoint,
    state_from_checkpoint,
    task_contract_payload,
    write_checkpoint,
)
from .ops.loop_progress import emit_progress_callback
from .ops import loop_diagnostics_support, loop_planning_support, loop_run_support
from .learning_compiler import persist_episode_learning_candidates
from .ops.episode_store import episode_storage_metadata
from .ops.loop_runtime_support import (
    build_default_policy,
    persist_episode_outputs,
)
from .sandbox import Sandbox
from .schemas import (
    EpisodeRecord,
    TaskSpec,
)
from .extensions.runtime_modeling_adapter import (
    build_context_provider,
    infer_hybrid_world_signal,
    load_model_artifact,
    retained_tolbert_hybrid_runtime,
)
from .state import AgentState
from .universe_model import UniverseModel
from .verifier import Verifier
from .world_model import WorldModel
from .extensions.strategy.kernel_catalog import kernel_catalog_string_set


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
        self.universe_model = UniverseModel(config=self.config) if self.config.use_universe_model else None
        self.world_model = WorldModel(config=self.config)
        self.role_coordinator = RoleCoordinator()
        self._tolbert_model_payload_cache: dict[str, object] | None = None

    def _build_default_policy(self) -> Policy:
        repo_root = Path(__file__).resolve().parents[1]
        return build_default_policy(
            config=self.config,
            repo_root=repo_root,
            skill_library_cls=SkillLibrary,
            llm_decision_policy_cls=LLMDecisionPolicy,
            context_provider_factory=build_context_provider,
            ollama_client_cls=OllamaClient,
            vllm_client_cls=VLLMClient,
            model_stack_client_cls=ModelStackClient,
            mock_client_factory=MockLLMClient,
            hybrid_client_factory=HybridDecoderClient,
        )

    @staticmethod
    def _should_persist_learning_candidates(task: TaskSpec, *, config: KernelConfig | None = None) -> bool:
        if config is not None and not bool(config.persist_learning_candidates):
            return False
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
        (
            task,
            workspace,
            checkpoint,
            setup_history,
            state,
            clean_workspace,
            early_episode,
        ) = loop_run_support.resume_or_initialize_run(
            self,
            task=task,
            clean_workspace=clean_workspace,
            checkpoint_path=checkpoint_path,
            resume=resume,
            runtime_overrides=runtime_overrides,
            job_id=job_id,
        )
        del clean_workspace
        if early_episode is not None:
            return early_episode
        setup_failure = loop_run_support.execute_setup_commands(
            self,
            task=task,
            workspace=workspace,
            checkpoint_path=checkpoint_path,
            state=state,
            setup_history=setup_history,
        )
        if setup_failure is not None:
            return setup_failure
        state_estimation_payload, latent_controls = loop_run_support.bootstrap_runtime_state(
            self,
            task=task,
            workspace=workspace,
            state=state,
            progress_callback=progress_callback,
        )
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
            success, termination_reason, should_break = loop_run_support.execute_step(
                self,
                task=task,
                workspace=workspace,
                checkpoint_path=checkpoint_path,
                state=state,
                setup_history=setup_history,
                progress_callback=progress_callback,
                state_estimation_payload=state_estimation_payload,
                latent_controls=latent_controls,
                success=success,
                termination_reason=termination_reason,
            )
            if should_break:
                break

        return loop_run_support.finalize_episode(
            self,
            task=task,
            workspace=workspace,
            checkpoint_path=checkpoint_path,
            state=state,
            success=success,
            termination_reason=termination_reason,
            setup_history=setup_history,
            runtime_overrides=runtime_overrides,
            job_id=job_id,
        )

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
        from .tasking.task_bank import TaskBank

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
        emit_progress_callback(progress_callback, payload)

    @staticmethod
    def _learned_world_progress_payload(latent_state_summary: dict[str, object]) -> dict[str, object]:
        return loop_diagnostics_support.learned_world_progress_payload(latent_state_summary)

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
        return loop_planning_support.build_plan(self, task)

    def _refresh_planner_subgoals(self, state: AgentState) -> None:
        verifier_hotspots = self._verifier_hotspot_subgoals(self._latest_failed_verification_reasons(state))
        learned_hotspots = self._learned_world_hotspot_entries(state)
        hotspot_subgoals = [*verifier_hotspots, *[str(entry.get("subgoal", "")).strip() for entry in learned_hotspots]]
        state.refresh_plan_progress(
            state.world_model_summary,
            expand_long_horizon=bool(hotspot_subgoals),
        )
        if not hotspot_subgoals:
            self._refresh_planner_recovery_artifact(state)
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
        graph_role_override = self._graph_memory_recovery_role(state)
        if graph_role_override:
            normalized_role = self._higher_priority_role(normalized_role, graph_role_override)
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
    def _graph_memory_recovery_role(state: AgentState) -> str:
        graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
        if not graph_summary:
            return ""
        failure_signals = graph_summary.get("failure_signals", {})
        failure_signals = failure_signals if isinstance(failure_signals, dict) else {}
        alignment_failures = graph_summary.get("environment_alignment_failures", {})
        alignment_failures = alignment_failures if isinstance(alignment_failures, dict) else {}
        no_progress = int(failure_signals.get("no_state_progress", 0) or 0)
        regressions = int(failure_signals.get("state_regression", 0) or 0)
        environment_pressure = sum(int(value or 0) for value in alignment_failures.values() if not isinstance(value, bool))
        if regressions + environment_pressure >= 3:
            return "critic"
        if no_progress + regressions + environment_pressure >= 3:
            return "planner"
        return ""

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
        refresh_planner_recovery_artifact(
            state,
            normalize_command_fn=self._normalize_command,
        )

    def _planner_recovery_surface_exhausted(self, state: AgentState, goal: str) -> bool:
        return planner_recovery_surface_exhausted(
            state,
            goal,
            normalize_command_fn=self._normalize_command,
        )

    def _build_planner_recovery_artifact(
        self,
        state: AgentState,
        *,
        active_subgoal: str,
        diagnosis: dict[str, object],
    ) -> dict[str, object]:
        return build_planner_recovery_artifact(
            state,
            active_subgoal=active_subgoal,
            diagnosis=diagnosis,
            normalize_command_fn=self._normalize_command,
        )

    @staticmethod
    def _planner_rewrite_subgoal(goal: str, *, path: str, related_objectives: list[str]) -> str:
        return planner_rewrite_subgoal(goal, path=path, related_objectives=related_objectives)

    def _planner_recovery_related_objectives(self, state: AgentState, *, primary_subgoal: str) -> list[str]:
        return planner_recovery_related_objectives(state, primary_subgoal=primary_subgoal)

    def _planner_recovery_ranked_objectives(
        self,
        state: AgentState,
        *,
        primary_subgoal: str,
        related_objectives: list[str],
        diagnosis: dict[str, object],
    ) -> list[dict[str, object]]:
        return planner_recovery_ranked_objectives(
            state,
            primary_subgoal=primary_subgoal,
            related_objectives=related_objectives,
            diagnosis=diagnosis,
        )

    @staticmethod
    def _planner_recovery_objective_kind(objective: str) -> str:
        return planner_recovery_objective_kind(objective)

    @classmethod
    def _planner_recovery_objective_target(cls, objective: str) -> str:
        del cls
        return planner_recovery_objective_target(objective)

    @staticmethod
    def _planner_recovery_objective_base_score(kind: str) -> int:
        return planner_recovery_objective_base_score(kind)

    def _planner_recovery_objective_blocks_verifier(self, state: AgentState, objective: str) -> bool:
        return planner_recovery_objective_blocks_verifier(state, objective)

    def _planner_recovery_objective_attempt_pressure(self, state: AgentState, objective: str) -> int:
        return planner_recovery_objective_attempt_pressure(state, objective)

    def _planner_recovery_objective_satisfied(self, state: AgentState, objective: str) -> bool:
        return planner_recovery_objective_satisfied(state, objective)

    def _planner_recovery_command_aligns_objective(self, command: str, objective: str) -> bool:
        return planner_recovery_command_aligns_objective(command, objective)

    def _planner_recovery_focus_paths(self, state: AgentState, *, primary_path: str) -> list[str]:
        return planner_recovery_focus_paths(state, primary_path=primary_path)

    @staticmethod
    def _planner_recovery_contract_outline(
        goal: str,
        *,
        path: str,
        focus_paths: list[str],
        related_objectives: list[str],
    ) -> list[str]:
        return planner_recovery_contract_outline(
            goal,
            path=path,
            focus_paths=focus_paths,
            related_objectives=related_objectives,
        )

    def _command_matches_subgoal_surface(self, state: AgentState, *, goal: str, command: str) -> bool:
        return command_matches_subgoal_surface(state, goal=goal, command=command)

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
        loop_diagnostics_support.attach_critic_subgoal_diagnoses(
            self,
            state,
            step_index=step_index,
            step_active_subgoal=step_active_subgoal,
            failure_signals=failure_signals,
            failure_origin=failure_origin,
            command_result=command_result,
        )

    def _attach_verifier_subgoal_diagnoses(
        self,
        state: AgentState,
        *,
        step_index: int,
        verification_reasons: list[object],
    ) -> None:
        loop_diagnostics_support.attach_verifier_subgoal_diagnoses(
            self,
            state,
            step_index=step_index,
            verification_reasons=verification_reasons,
        )

    def _diagnosis_candidate_subgoals(self, state: AgentState, *, step_active_subgoal: str) -> list[str]:
        return loop_diagnostics_support.diagnosis_candidate_subgoals(
            self,
            state,
            step_active_subgoal=step_active_subgoal,
        )

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
        return loop_diagnostics_support.build_subgoal_failure_diagnosis(
            self,
            state,
            goal=goal,
            step_index=step_index,
            ordered_signals=ordered_signals,
            command_result=command_result,
            step_active_subgoal=step_active_subgoal,
        )

    def _latest_failed_verification_reasons(self, state: AgentState) -> list[str]:
        if not state.history:
            return []
        verification = state.history[-1].verification if isinstance(state.history[-1].verification, dict) else {}
        if bool(verification.get("passed", False)):
            return []
        return [str(reason).strip() for reason in verification.get("reasons", []) if str(reason).strip()]

    def _verifier_hotspot_subgoals(self, verification_reasons: list[object]) -> list[str]:
        return loop_diagnostics_support.verifier_hotspot_subgoals(self, verification_reasons)

    def _verifier_failure_entries(self, verification_reasons: list[object]) -> list[dict[str, str]]:
        return loop_diagnostics_support.verifier_failure_entries(verification_reasons)

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
        return loop_diagnostics_support.ordered_failure_signals(
            failure_signals=failure_signals,
            failure_origin=failure_origin,
            command_result=command_result,
        )

    @staticmethod
    def _subgoal_path(goal: str) -> str:
        return subgoal_path(goal)

    @staticmethod
    def _subgoal_diagnosis_summary(ordered_signals: list[str], *, command_result) -> str:
        return loop_diagnostics_support.subgoal_diagnosis_summary(
            ordered_signals,
            command_result=command_result,
        )

    @staticmethod
    def _subgoal_diagnosis_priority(diagnosis: dict[str, object]) -> int:
        return loop_diagnostics_support.subgoal_diagnosis_priority(diagnosis)

    @staticmethod
    def _float_value(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _workflow_plan_steps(task: TaskSpec, *, planner_controls: dict[str, object] | None = None) -> list[str]:
        return loop_planning_support.workflow_plan_steps(task, planner_controls=planner_controls)

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
        episode_path = persist_episode_outputs(
            task=task,
            episode=episode,
            config=self.config,
            memory=self.memory,
            should_persist_learning_candidates_fn=self._should_persist_learning_candidates,
            persist_episode_learning_candidates_fn=persist_episode_learning_candidates,
            episode_storage_metadata_fn=episode_storage_metadata,
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
        return task_contract_payload(task)

    def _planner_controls(self) -> dict[str, object]:
        return loop_planning_support.planner_controls(self)

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
        return load_checkpoint(checkpoint_path)

    @staticmethod
    def _state_from_checkpoint(task: TaskSpec, payload: dict[str, object]) -> AgentState:
        return state_from_checkpoint(task, payload)

    @staticmethod
    def _setup_history_from_checkpoint(payload: dict[str, object]) -> list[dict[str, object]]:
        return setup_history_from_checkpoint(payload)

    @staticmethod
    def _completed_setup_command_count(task: TaskSpec, setup_history: list[dict[str, object]]) -> int:
        return completed_setup_command_count(task, setup_history)

    @staticmethod
    def _episode_from_payload(payload: dict[str, object]) -> EpisodeRecord:
        return episode_from_payload(payload)

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
        return write_checkpoint(
            checkpoint_path,
            task=task,
            workspace=workspace,
            state=state,
            success=success,
            status=status,
            termination_reason=termination_reason,
            config=self.config,
            episode=episode,
            setup_history=setup_history,
            phase=phase,
        )

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
