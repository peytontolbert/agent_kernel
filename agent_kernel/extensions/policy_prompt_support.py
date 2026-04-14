from __future__ import annotations

from typing import Any

from ..state import AgentState


def trusted_workflow_guidance_bypasses_first_step_confidence(
    policy: Any,
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
    return first_step_guarded_command_coverage(policy, state, command, claimed_paths)


def first_step_guarded_command_coverage(
    policy: Any,
    state: AgentState,
    command: str,
    claimed_paths: list[str],
) -> bool:
    del policy
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


def safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def truncate_text(policy: Any, text: object, *, limit: int | None = None) -> str:
    value = " ".join(str(text).split())
    max_chars = max(32, limit or policy.config.llm_summary_max_chars)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def role_system_prompt(policy: Any, role: str) -> str:
    directive = role_directive(policy, role)
    return f"{policy.system_prompt}\n{directive}"


def role_decision_prompt(policy: Any, role: str) -> str:
    directive = role_directive(policy, role)
    return f"{directive}\n{policy.decision_prompt}"


def role_directive(policy: Any, role: str) -> str:
    normalized = str(role or "executor").strip().lower()
    if normalized == "planner":
        base = "Active role: planner. Prefer clarifying the next verifier-relevant subgoal and commands that establish expected artifacts."
    elif normalized == "critic":
        base = "Active role: critic. Prefer avoiding repeated failures, forbidden artifacts, and brittle commands before suggesting execution."
    else:
        base = "Active role: executor. Prefer the most direct verifier-relevant command that completes the current subgoal."
    override = policy._role_directive_overrides().get(normalized, "")
    if not override:
        return base
    return f"{base} {override}"


def normalized_role(state: AgentState) -> str:
    return str(state.current_role or "executor").strip().lower() or "executor"
