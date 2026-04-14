from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from ..config import KernelConfig
from ..ops.runtime_supervision import atomic_write_json
from ..schemas import EpisodeRecord, StepRecord, TaskSpec
from ..state import AgentState


def task_contract_payload(task: TaskSpec) -> dict[str, object]:
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


def load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def state_from_checkpoint(task: TaskSpec, payload: dict[str, object]) -> AgentState:
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


def setup_history_from_checkpoint(payload: dict[str, object]) -> list[dict[str, object]]:
    return [dict(entry) for entry in payload.get("setup_history", []) if isinstance(entry, dict)]


def completed_setup_command_count(task: TaskSpec, setup_history: list[dict[str, object]]) -> int:
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


def episode_from_payload(payload: dict[str, object]) -> EpisodeRecord:
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


def write_checkpoint(
    checkpoint_path: Path,
    *,
    task: TaskSpec,
    workspace: Path,
    state: AgentState,
    success: bool,
    status: str,
    termination_reason: str,
    config: KernelConfig,
    episode: EpisodeRecord | None = None,
    setup_history: list[dict[str, object]] | None = None,
    phase: str = "execute",
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_recent_window = max(1, int(config.checkpoint_history_step_window))
    checkpoint_history = state.history[-checkpoint_recent_window:]
    checkpoint_archive = dict(state.history_archive)
    omitted_recent_steps = state.history[:-checkpoint_recent_window]
    if omitted_recent_steps:
        archived_state = AgentState(task=task)
        archived_state.history_archive = checkpoint_archive
        archived_state._append_history_archive(
            omitted_recent_steps,
            summary_char_limit=config.history_archive_summary_max_chars,
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
    atomic_write_json(checkpoint_path, payload, config=config)
    return checkpoint_path


__all__ = [
    "completed_setup_command_count",
    "episode_from_payload",
    "load_checkpoint",
    "setup_history_from_checkpoint",
    "state_from_checkpoint",
    "task_contract_payload",
    "write_checkpoint",
]
