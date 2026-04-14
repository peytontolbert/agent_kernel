from __future__ import annotations


def emit_progress_callback(progress_callback, payload: dict[str, object]) -> None:
    if progress_callback is None:
        return
    progress_callback(dict(payload))


def progress_event(
    event: str,
    *,
    step_stage: str = "",
    step_subphase: str = "",
    completed_steps: int | None = None,
    step_index: int | None = None,
    active_subgoal: str = "",
    current_role: str = "",
    command: str = "",
    verification_passed: bool | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"event": event}
    if step_stage:
        payload["step_stage"] = step_stage
    if step_subphase:
        payload["step_subphase"] = step_subphase
    if completed_steps is not None:
        payload["completed_steps"] = int(completed_steps)
    if step_index is not None:
        payload["step_index"] = int(step_index)
    if active_subgoal:
        payload["active_subgoal"] = active_subgoal
    if current_role:
        payload["current_role"] = current_role
    if command:
        payload["command"] = command
    if verification_passed is not None:
        payload["verification_passed"] = bool(verification_passed)
    if extra:
        payload.update(dict(extra))
    return payload


def decision_progress_event(
    payload: dict[str, object],
    *,
    step_index: int,
    active_subgoal: str,
) -> dict[str, object]:
    merged = dict(payload)
    merged.setdefault("event", "decision_progress")
    merged.setdefault("step_index", step_index)
    if active_subgoal:
        merged.setdefault("active_subgoal", active_subgoal)
    return merged


__all__ = [
    "decision_progress_event",
    "emit_progress_callback",
    "progress_event",
]
