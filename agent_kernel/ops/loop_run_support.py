from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from ..actions import CODE_EXECUTE
from ..config import KernelConfig
from ..extensions.improvement.state_estimation_improvement import (
    retained_state_estimation_latent_controls,
    retained_state_estimation_payload,
    summarize_state_transition,
)
from ..extensions.runtime_modeling_adapter import (
    build_latent_state_summary,
)
from ..learning_compiler import persist_episode_learning_candidates
from ..ops.episode_store import episode_storage_metadata
from ..ops.loop_progress import decision_progress_event, progress_event
from ..ops.loop_runtime_support import (
    materialize_workspace_for_new_run,
    maybe_publish_shared_repo_branch,
    persist_episode_outputs,
    prepare_task_for_run,
)
from ..ops.shared_repo import (
    materialize_shared_repo_workspace,
    prepare_runtime_task,
    publish_shared_repo_branch,
    uses_shared_repo,
)
from ..schemas import (
    CommandResult,
    EpisodeRecord,
    StepRecord,
    TaskSpec,
    VerificationResult,
    classify_verification_reason,
    episode_success_criteria,
)
from ..state import AgentState, _canonicalize_command


def verification_payload(
    *,
    passed: bool,
    reasons: list[str],
    command_result: CommandResult | None = None,
    outcome_label: str = "",
    controllability: str = "agent",
    extra_failure_codes: list[str] | None = None,
) -> dict[str, object]:
    failure_codes = [
        code
        for code in (classify_verification_reason(reason) for reason in reasons)
        if code
    ]
    for code in extra_failure_codes or []:
        normalized_code = str(code).strip()
        if normalized_code and normalized_code not in failure_codes:
            failure_codes.append(normalized_code)
    return VerificationResult(
        passed=passed,
        reasons=list(reasons),
        command_result=command_result,
        process_score=1.0 if passed else 0.0,
        outcome_label=outcome_label.strip() or ("success" if passed else (failure_codes[0] if failure_codes else "failure")),
        outcome_confidence=1.0,
        controllability=controllability,
        failure_codes=failure_codes,
    ).to_payload()


def _shared_repo_integrator_segment_made_progress(
    state: AgentState,
    *,
    decision_source: str,
    command: str,
    command_result: CommandResult | None,
) -> bool:
    if decision_source != "shared_repo_integrator_segment_direct":
        return False
    if command_result is None:
        return False
    if int(command_result.exit_code) != 0 or bool(command_result.timed_out):
        return False
    canonical = _canonicalize_command(command)
    if not canonical:
        return False
    return canonical not in state.all_successful_command_signatures()


def resume_or_initialize_run(
    kernel: Any,
    *,
    task: TaskSpec,
    clean_workspace: bool,
    checkpoint_path: Path | None,
    resume: bool,
    runtime_overrides: dict[str, object] | None,
    job_id: str,
) -> tuple[TaskSpec, Path, dict[str, object] | None, list[dict[str, object]], AgentState, bool, EpisodeRecord | None]:
    task = prepare_task_for_run(
        task,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
        uses_shared_repo_fn=uses_shared_repo,
        prepare_runtime_task_fn=prepare_runtime_task,
    )
    workspace = kernel.config.workspace_root / task.workspace_subdir
    checkpoint = None
    setup_history: list[dict[str, object]] = []
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        checkpoint = kernel._load_checkpoint(checkpoint_path)
        if checkpoint.get("status") == "completed":
            return task, workspace, checkpoint, setup_history, AgentState(task=task), clean_workspace, kernel._episode_from_payload(checkpoint["episode"])
        workspace = Path(str(checkpoint.get("workspace", workspace)))
        setup_history = kernel._setup_history_from_checkpoint(checkpoint)
        state = kernel._state_from_checkpoint(task, checkpoint)
        state.termination_reason = ""
        clean_workspace = False
    else:
        workspace, clean_workspace = materialize_workspace_for_new_run(
            task,
            config=kernel.config,
            runtime_overrides=runtime_overrides,
            job_id=job_id,
            resume=resume,
            checkpoint_path=checkpoint_path,
            clean_workspace=clean_workspace,
            uses_shared_repo_fn=uses_shared_repo,
            materialize_shared_repo_workspace_fn=materialize_shared_repo_workspace,
            ensure_parallel_worker_branches_fn=kernel._ensure_parallel_worker_branches,
        )
        state = AgentState(task=task)
        if checkpoint_path is not None:
            kernel._write_checkpoint(
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
    return task, workspace, checkpoint, setup_history, state, clean_workspace, None


def execute_setup_commands(
    kernel: Any,
    *,
    task: TaskSpec,
    workspace: Path,
    checkpoint_path: Path | None,
    state: AgentState,
    setup_history: list[dict[str, object]],
) -> EpisodeRecord | None:
    setup_resume_index = kernel._completed_setup_command_count(task, setup_history)
    if setup_resume_index >= len(task.setup_commands):
        return None
    for command_index, command in enumerate(task.setup_commands[setup_resume_index:], start=setup_resume_index):
        result = kernel.sandbox.run(command, workspace, task=task)
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
            kernel._write_checkpoint(
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
            return kernel._setup_failure_episode(
                task=task,
                workspace=workspace,
                checkpoint_path=checkpoint_path,
                state=state,
                setup_history=setup_history,
            )
    return None


def bootstrap_runtime_state(
    kernel: Any,
    *,
    task: TaskSpec,
    workspace: Path,
    state: AgentState,
    progress_callback,
) -> tuple[dict[str, object], dict[str, object]]:
    if kernel.config.use_graph_memory and not state.graph_summary:
        state.graph_summary = kernel.graph_memory.summarize(task.task_id)
        recalled_semantic_episodes = kernel.graph_memory.recall(
            task_id=task.task_id,
            benchmark_family=str(task.metadata.get("benchmark_family", "")).strip(),
            changed_paths=_task_semantic_recall_paths(task),
            require_success=True,
            limit=6,
        )
        recalled_semantic_prototypes = kernel.graph_memory.prototype_recall(
            task_id=task.task_id,
            benchmark_family=str(task.metadata.get("benchmark_family", "")).strip(),
            changed_paths=_task_semantic_recall_paths(task),
            require_success=True,
            limit=4,
        )
        if recalled_semantic_episodes:
            state.graph_summary["semantic_episodes"] = recalled_semantic_episodes
        if recalled_semantic_prototypes:
            state.graph_summary["semantic_prototypes"] = recalled_semantic_prototypes
        kernel._emit_progress_callback(
            progress_callback,
            progress_event(
                "memory_retrieved",
                step_stage="memory_retrieved",
                step_subphase="graph_memory",
                completed_steps=state.completed_step_count(),
            ),
        )
    if kernel.universe_model is not None and not state.universe_summary:
        state.universe_summary = kernel.universe_model.summarize(
            task,
            world_model_summary=state.world_model_summary,
            workspace=workspace,
            planned_commands=task.setup_commands + task.suggested_commands + ([task.success_command] if task.success_command else []),
        )
    state_estimation_payload = retained_state_estimation_payload(kernel.config)
    latent_controls = retained_state_estimation_latent_controls(state_estimation_payload)
    if kernel.config.use_world_model:
        if not state.workspace_snapshot:
            state.workspace_snapshot = kernel.world_model.capture_workspace_snapshot(task, workspace)
        if not state.world_model_summary:
            state.world_model_summary = kernel.world_model.summarize(
                task,
                state.graph_summary,
                workspace=workspace,
                workspace_snapshot=state.workspace_snapshot,
            )
            kernel._emit_progress_callback(
                progress_callback,
                progress_event(
                    "state_estimated",
                    step_stage="state_estimated",
                    step_subphase="world_model_initial",
                    completed_steps=state.completed_step_count(),
                ),
            )
            if kernel.universe_model is not None:
                state.universe_summary = kernel.universe_model.summarize(
                    task,
                    world_model_summary=state.world_model_summary,
                    workspace=workspace,
                    planned_commands=task.setup_commands + task.suggested_commands + ([task.success_command] if task.success_command else []),
                )
            state.recent_workspace_summary = kernel.world_model.describe_progress(state.world_model_summary)
        learned_world_signal = kernel._infer_learned_world_signal(state)
        state.latent_state_summary = build_latent_state_summary(
            world_model_summary=state.world_model_summary,
            latest_transition=state.latest_state_transition,
            task_metadata=task.metadata,
            recent_history=[asdict(step) for step in state.history[-3:]],
            context_control=state.context_packet.control if state.context_packet is not None else {},
            latent_controls=latent_controls,
            learned_world_signal=learned_world_signal,
        )
    if kernel.config.use_planner and not state.initial_plan:
        state.plan = kernel._build_plan(task)
        state.initial_plan = list(state.plan)
        state.active_subgoal = state.plan[0] if state.plan else ""
    return state_estimation_payload, latent_controls


def _task_semantic_recall_paths(task: TaskSpec) -> list[str]:
    paths: list[str] = []
    for value in [*task.expected_files, *list(task.expected_file_contents.keys())]:
        normalized = str(value).strip()
        if normalized and normalized not in paths:
            paths.append(normalized)
    verifier = task.metadata.get("semantic_verifier", {})
    verifier = dict(verifier) if isinstance(verifier, dict) else {}
    for value in verifier.get("expected_changed_paths", []):
        normalized = str(value).strip()
        if normalized and normalized not in paths:
            paths.append(normalized)
    for rule in verifier.get("report_rules", []):
        if not isinstance(rule, dict):
            continue
        normalized = str(rule.get("path", "")).strip()
        if normalized and normalized not in paths:
            paths.append(normalized)
    return paths


def summarize_transition(
    kernel: Any,
    *,
    task: TaskSpec,
    workspace: Path,
    state: AgentState,
    previous_world_model_summary: dict[str, object],
    decision,
    command_result,
    structured_edit_before_content: str | None,
    state_estimation_payload: dict[str, object],
    latent_controls: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    transition: dict[str, object] = {}
    runtime_proposal_metadata_additions: dict[str, object] = {}
    if not kernel.config.use_world_model:
        return transition, runtime_proposal_metadata_additions
    state.world_model_summary = kernel.world_model.summarize(
        task,
        state.graph_summary,
        workspace=workspace,
        workspace_snapshot=state.workspace_snapshot,
    )
    if kernel.universe_model is not None:
        state.universe_summary = kernel.universe_model.summarize(
            task,
            world_model_summary=state.world_model_summary,
        )
    raw_transition = kernel.world_model.describe_transition(
        previous_world_model_summary,
        state.world_model_summary,
    )
    transition = summarize_state_transition(
        raw_transition,
        payload=state_estimation_payload,
    )
    if kernel._structured_edit_mutated_path(
        workspace,
        decision,
        before_content=structured_edit_before_content,
        command_result=command_result,
    ):
        transition["progress_delta"] = max(float(transition.get("progress_delta", 0.0)), 0.001)
        transition["no_progress"] = False
        transition["state_change_score"] = max(int(transition.get("state_change_score", 0) or 0), 1)
    syntax_motor_progress = kernel._structured_edit_syntax_progress(
        task,
        workspace=workspace,
        decision=decision,
        before_content=structured_edit_before_content,
        command_result=command_result,
    )
    if syntax_motor_progress:
        runtime_proposal_metadata_additions["syntax_motor_progress"] = dict(syntax_motor_progress)
        runtime_proposal_metadata_additions["syntax_motor"] = dict(
            syntax_motor_progress.get("syntax_motor", {})
        )
        transition["syntax_motor_progress"] = dict(syntax_motor_progress)
        if bool(syntax_motor_progress.get("strong_progress", False)):
            transition["progress_delta"] = max(float(transition.get("progress_delta", 0.0)), 0.025)
            transition["no_progress"] = False
            transition["state_change_score"] = max(int(transition.get("state_change_score", 0) or 0), 2)
    state.recent_workspace_summary = kernel.world_model.describe_progress(
        state.world_model_summary,
        command=decision.content if decision.action == "code_execute" else "",
        step_index=state.next_step_index(),
    )
    learned_world_signal = kernel._infer_learned_world_signal(state)
    state.latent_state_summary = build_latent_state_summary(
        world_model_summary=state.world_model_summary,
        latest_transition=transition,
        task_metadata=task.metadata,
        recent_history=[asdict(step) for step in state.history[-3:]],
        context_control=state.context_packet.control if state.context_packet is not None else {},
        latent_controls=latent_controls,
        learned_world_signal=learned_world_signal,
    )
    return transition, runtime_proposal_metadata_additions


def execute_step(
    kernel: Any,
    *,
    task: TaskSpec,
    workspace: Path,
    checkpoint_path: Path | None,
    state: AgentState,
    setup_history: list[dict[str, object]],
    progress_callback,
    state_estimation_payload: dict[str, object],
    latent_controls: dict[str, object],
    success: bool,
    termination_reason: str,
) -> tuple[bool, str, bool]:
    if kernel.config.use_role_specialization:
        state.current_role = kernel._resolve_role_before_decision(state)
    if kernel.config.use_planner and state.current_role == "planner":
        kernel._refresh_planner_subgoals(state)
    step_index = state.next_step_index()
    step_active_subgoal = state.active_subgoal
    step_role = state.current_role
    previous_world_model_summary = dict(state.world_model_summary)
    failure_signals: list[str] = []
    failure_origin = ""
    step_started_at = time.monotonic()
    decision_progress_callback = lambda payload: kernel._emit_progress_callback(
        progress_callback,
        decision_progress_event(
            {
                "completed_steps": state.completed_step_count(),
                "acting_role": step_role,
                **dict(payload),
            },
            step_index=step_index,
            active_subgoal=step_active_subgoal,
        ),
    )
    kernel._emit_progress_callback(
        progress_callback,
        progress_event(
            "step_start",
            step_index=step_index,
            step_stage="decision_pending",
            completed_steps=state.completed_step_count(),
            active_subgoal=step_active_subgoal,
            extra={"acting_role": step_role},
        ),
    )
    kernel.policy.set_decision_progress_callback(decision_progress_callback)
    try:
        decision = kernel.policy.decide(state)
    except Exception as exc:
        failure_origin = kernel.classify_policy_failure(exc)
        failure_signals = [failure_origin]
        fallback_decision = kernel.policy.fallback_decision(
            state,
            failure_origin=failure_origin,
            error_text=str(exc),
        )
        if fallback_decision is not None:
            fallback_decision.proposal_metadata = {
                **dict(fallback_decision.proposal_metadata or {}),
                "fallback_failure_origin": failure_origin,
                "fallback_error_type": exc.__class__.__name__,
            }
            decision = fallback_decision
        else:
            decision = kernel.policy_error_decision(exc, failure_origin=failure_origin)
    finally:
        kernel.policy.set_decision_progress_callback(None)
    kernel._emit_progress_callback(
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
    verification = verification_payload(
        passed=False,
        reasons=["no command executed"],
        outcome_label="no_command_executed",
    )
    command_governance: dict[str, object] = {}
    structured_edit_before_content: str | None = None
    runtime_proposal_metadata = dict(decision.proposal_metadata or {})

    if decision.action == CODE_EXECUTE:
        if kernel.universe_model is not None and decision.content:
            command_governance = kernel.universe_model.should_block_command(
                state.universe_summary,
                decision.content,
            )
            if bool(command_governance.get("blocked", False)):
                block_message = str(command_governance.get("block_message", "")).strip()
                block_reason = str(command_governance.get("block_reason", "")).strip()
                command_result = CommandResult(
                    command=decision.content,
                    exit_code=126,
                    stdout="",
                    stderr=block_message or "universe governance rejected command",
                )
                verification = verification_payload(
                    passed=False,
                    reasons=[block_message or "universe governance rejected command"],
                    command_result=command_result,
                    outcome_label="governance_rejected",
                    controllability="policy",
                    extra_failure_codes=["governance_rejected", block_reason] if block_reason else ["governance_rejected"],
                )
                runtime_proposal_metadata["governance_block"] = {
                    "reason": block_reason,
                    "reasons": list(command_governance.get("block_reasons", []) or []),
                    "message": block_message,
                }
                failure_signals.append("governance_rejected")
                if block_reason:
                    failure_signals.append(block_reason)
                kernel._emit_progress_callback(
                    progress_callback,
                    {
                        "event": "governance_rejected",
                        "step_index": step_index,
                        "step_stage": "governance_rejected",
                        "completed_steps": state.completed_step_count(),
                        "active_subgoal": step_active_subgoal,
                        "acting_role": step_role,
                        "governance_block_reason": block_reason,
                    },
                )
        proposal_source = str(decision.proposal_source).strip()
        if command_result is None and proposal_source.startswith("structured_edit:"):
            proposal_metadata = dict(runtime_proposal_metadata)
            structured_edit_path = str(proposal_metadata.get("path", "")).strip()
            if structured_edit_path:
                try:
                    structured_edit_before_content = (workspace / structured_edit_path).read_text(
                        encoding="utf-8"
                    )
                except (FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError):
                    structured_edit_before_content = None
        if command_result is None:
            command_result = kernel.sandbox.run(decision.content, workspace, task=task)
            verified = kernel.verifier.verify(task, workspace, command_result)
            verification = verified.to_payload()
            success = verified.passed
    elif decision.done:
        reasons = ["policy terminated"]
        if failure_origin:
            reasons.append(f"policy failure origin: {failure_origin}")
        verification = verification_payload(
            passed=success,
            reasons=reasons,
            outcome_label="policy_terminated",
            controllability="policy",
        )

    transition: dict[str, object] = {}
    if kernel.config.use_world_model:
        kernel._emit_progress_callback(
            progress_callback,
            {
                "event": "world_model_updated",
                "step_index": step_index,
                "step_stage": "world_model_updated",
                "completed_steps": state.completed_step_count(),
                "active_subgoal": step_active_subgoal,
                "acting_role": step_role,
            },
        )
        transition, runtime_proposal_metadata_additions = summarize_transition(
            kernel,
            task=task,
            workspace=workspace,
            state=state,
            previous_world_model_summary=previous_world_model_summary,
            decision=decision,
            command_result=command_result,
            structured_edit_before_content=structured_edit_before_content,
            state_estimation_payload=state_estimation_payload,
            latent_controls=latent_controls,
        )
        runtime_proposal_metadata.update(runtime_proposal_metadata_additions)
        syntax_motor_progress = runtime_proposal_metadata_additions.get("syntax_motor_progress", {})
        if not syntax_motor_progress:
            syntax_motor_progress = kernel._structured_edit_syntax_progress(
                task,
                workspace=workspace,
                decision=decision,
                before_content=structured_edit_before_content,
                command_result=command_result,
            )
        if command_result is not None and bool(transition.get("no_progress", False)):
            failure_signals.append("no_state_progress")
        if command_result is not None and list(transition.get("regressions", [])):
            failure_signals.append("state_regression")
    if _shared_repo_integrator_segment_made_progress(
        state,
        decision_source=str(decision.decision_source),
        command=str(decision.content),
        command_result=command_result,
    ):
        transition["progress_delta"] = max(float(transition.get("progress_delta", 0.0)), 0.001)
        transition["no_progress"] = False
        transition["state_change_score"] = max(int(transition.get("state_change_score", 0) or 0), 1)
        failure_signals = [signal for signal in failure_signals if signal != "no_state_progress"]
    if (
        kernel.universe_model is not None
        and decision.action == CODE_EXECUTE
        and decision.content
        and not command_governance
    ):
        command_governance = kernel.universe_model.simulate_command_governance(
            state.universe_summary,
            decision.content,
        )
    step_software_work_objective = state.current_software_work_objective() if kernel.config.use_planner else ""
    if kernel.config.use_planner:
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
    kernel._emit_progress_callback(
        progress_callback,
        {
            "event": "memory_update_written",
            "step_index": step_index,
            "step_stage": "memory_update_written",
            "completed_steps": state.completed_step_count(),
            "active_subgoal": step_active_subgoal,
            "acting_role": step_role,
            "verification_passed": bool(verification["passed"]),
            "verification_outcome_label": str(verification.get("outcome_label", "")).strip(),
            "verification_process_score": float(verification.get("process_score", 0.0) or 0.0),
            "verification_failure_codes": list(verification.get("failure_codes", []) or []),
            **kernel._learned_world_progress_payload(state.latent_state_summary),
        },
    )
    if step_role == "critic":
        kernel._attach_critic_subgoal_diagnoses(
            state,
            step_index=step_index,
            step_active_subgoal=step_active_subgoal,
            failure_signals=failure_signals,
            failure_origin=failure_origin,
            command_result=command_result,
        )
        kernel._emit_progress_callback(
            progress_callback,
            {
                "event": "critique_reflected",
                "step_index": step_index,
                "step_stage": "critique_reflected",
                "completed_steps": state.completed_step_count(),
                "active_subgoal": step_active_subgoal,
                "acting_role": step_role,
            },
        )
    if not verification["passed"]:
        kernel._attach_verifier_subgoal_diagnoses(
            state,
            step_index=step_index,
            verification_reasons=verification["reasons"],
        )
        kernel._emit_progress_callback(
            progress_callback,
            {
                "event": "verification_result",
                "step_index": step_index,
                "step_stage": "verification_result",
                "completed_steps": state.completed_step_count(),
                "active_subgoal": step_active_subgoal,
                "acting_role": step_role,
                "verification_passed": False,
                "verification_outcome_label": str(verification.get("outcome_label", "")).strip(),
                "verification_process_score": float(verification.get("process_score", 0.0) or 0.0),
                "verification_failure_codes": list(verification.get("failure_codes", []) or []),
            },
        )
        if kernel.config.use_planner:
            kernel._promote_prioritized_subgoals(
                state,
                prioritized=kernel._verifier_hotspot_subgoals(verification["reasons"]),
            )
    if kernel.config.use_planner:
        kernel._refresh_planner_recovery_artifact(state)
    step_record = StepRecord(
        index=step_index,
        thought=decision.thought,
        action=decision.action,
        content=decision.content,
        selected_skill_id=decision.selected_skill_id,
        command_result=asdict(command_result) if command_result else None,
        verification=verification,
        available_skill_count=kernel._available_skill_count(state),
        retrieval_candidate_count=kernel._retrieval_candidate_count(state),
        retrieval_evidence_count=kernel._retrieval_evidence_count(state),
        retrieval_command_match=kernel._retrieval_command_match(state, decision.content),
        selected_retrieval_span_id=decision.selected_retrieval_span_id,
        retrieval_influenced=decision.retrieval_influenced,
        retrieval_ranked_skill=decision.retrieval_ranked_skill,
        path_confidence=kernel._path_confidence(state),
        trust_retrieval=kernel._trust_retrieval(state),
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
        max_recent_steps=kernel.config.runtime_history_step_window,
        summary_char_limit=kernel.config.history_archive_summary_max_chars,
    )
    kernel._emit_progress_callback(
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
        if _defer_stuckness_for_exact_verifier_repair(state, step_index=step_index):
            termination_reason = ""
        elif state.termination_reason == "repeated_failed_action":
            verification["reasons"].append("repeated failed action detected")
        elif state.termination_reason == "no_state_progress":
            verification["reasons"].append("no state progress detected")
        if state.termination_reason:
            termination_reason = state.termination_reason
    elif kernel.config.use_role_specialization:
        state.current_role = kernel._resolve_role_after_step(
            state,
            verification_passed=verification["passed"],
        )

    if checkpoint_path is not None:
        kernel._write_checkpoint(
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

    should_break = bool(success or decision.done or state.termination_reason)
    return success, termination_reason, should_break


def _defer_stuckness_for_exact_verifier_repair(state: AgentState, *, step_index: int) -> bool:
    termination_reason = str(state.termination_reason).strip()
    if termination_reason not in {"repeated_failed_action", "no_state_progress"}:
        return False
    repair_signature = ""
    for goal, diagnosis in state.subgoal_diagnoses.items():
        if not isinstance(diagnosis, dict):
            continue
        if int(diagnosis.get("updated_step_index", -1) or -1) != int(step_index):
            continue
        path = str(diagnosis.get("path", "")).strip()
        expected_content = diagnosis.get("expected_content")
        repair_instruction = str(diagnosis.get("repair_instruction", "")).strip()
        if not path:
            continue
        summary = str(diagnosis.get("summary", "")).lower()
        exact_content_repair = expected_content is not None and "unexpected file content" in summary
        swe_patch_repair = path == "patch.diff" and bool(repair_instruction)
        if not exact_content_repair and not swe_patch_repair:
            continue
        repair_signature = f"{termination_reason}|{goal}|{path}"
        break
    if not repair_signature:
        return False
    deferred_signatures = state.history_archive.get("exact_verifier_repair_stuckness_deferrals", [])
    if not isinstance(deferred_signatures, list):
        deferred_signatures = []
    seen = {str(value).strip() for value in deferred_signatures if str(value).strip()}
    if repair_signature in seen:
        return False
    seen.add(repair_signature)
    state.history_archive["exact_verifier_repair_stuckness_deferrals"] = sorted(seen)
    state.termination_reason = ""
    return True


def finalize_episode(
    kernel: Any,
    *,
    task: TaskSpec,
    workspace: Path,
    checkpoint_path: Path | None,
    state: AgentState,
    success: bool,
    termination_reason: str,
    setup_history: list[dict[str, object]],
    runtime_overrides: dict[str, object] | None,
    job_id: str,
) -> EpisodeRecord:
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=success,
        steps=state.history,
        task_metadata=dict(task.metadata),
        task_contract=kernel._task_contract_payload(task),
        plan=list(state.initial_plan),
        graph_summary=dict(state.graph_summary),
        universe_summary=dict(state.universe_summary),
        world_model_summary=dict(state.world_model_summary),
        history_archive=dict(state.history_archive),
        termination_reason=termination_reason,
    )
    success_contract = episode_success_criteria(episode)
    episode.success = bool(success_contract["verifier_aligned_task_success"])
    episode_path = persist_episode_outputs(
        task=task,
        episode=episode,
        config=kernel.config,
        memory=kernel.memory,
        should_persist_learning_candidates_fn=kernel._should_persist_learning_candidates,
        persist_episode_learning_candidates_fn=persist_episode_learning_candidates,
        episode_storage_metadata_fn=episode_storage_metadata,
    )
    maybe_publish_shared_repo_branch(
        episode=episode,
        task=task,
        config=kernel.config,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
        uses_shared_repo_fn=uses_shared_repo,
        publish_shared_repo_branch_fn=publish_shared_repo_branch,
    )
    if checkpoint_path is not None:
        kernel._write_checkpoint(
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


def setup_failure_episode(
    kernel: Any,
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
        task_contract=kernel._task_contract_payload(task),
        plan=[],
        graph_summary={},
        universe_summary={},
        world_model_summary={},
        termination_reason="setup_failed",
    )
    episode_path = persist_episode_outputs(
        task=task,
        episode=episode,
        config=kernel.config,
        memory=kernel.memory,
        should_persist_learning_candidates_fn=kernel._should_persist_learning_candidates,
        persist_episode_learning_candidates_fn=persist_episode_learning_candidates,
        episode_storage_metadata_fn=episode_storage_metadata,
    )
    del episode_path
    if checkpoint_path is not None:
        kernel._write_checkpoint(
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
