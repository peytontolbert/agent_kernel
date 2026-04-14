from __future__ import annotations

from typing import Any

_SOFTWARE_WORK_PHASES = ("implementation", "migration", "test", "follow_up_fix")


def software_work_phase_rank(phase: str) -> int:
    normalized = str(phase).strip()
    try:
        return _SOFTWARE_WORK_PHASES.index(normalized)
    except ValueError:
        return len(_SOFTWARE_WORK_PHASES)


def software_work_objective_phase(
    objective: str,
    *,
    test_attempted: bool = False,
    regression_signaled: bool = False,
) -> str:
    normalized = str(objective).strip().lower()
    if not normalized:
        return "implementation"
    if normalized.startswith("run workflow test "):
        return "test"
    if normalized.startswith(("accept required branch ", "prepare workflow branch ", "regenerate generated artifact ")):
        return "migration"
    if normalized.startswith("write workflow report "):
        return "follow_up_fix"
    if normalized.startswith(("revise implementation for ", "remove forbidden artifact ", "preserve required artifact ")):
        if test_attempted or regression_signaled:
            return "follow_up_fix"
        return "implementation"
    if normalized.startswith(
        (
            "materialize expected artifact ",
            "apply planned edit ",
            "complete implementation for ",
            "update workflow path ",
        )
    ):
        return "implementation"
    if regression_signaled and any(
        token in normalized for token in ("fix", "repair", "revise", "restore", "cleanup", "follow-up")
    ):
        return "follow_up_fix"
    return "implementation"


def refresh_plan_progress(
    state: Any,
    world_model_summary: dict[str, object],
    *,
    expand_long_horizon: bool = True,
) -> None:
    remaining = [goal for goal in state.plan if not subgoal_satisfied(state, goal, world_model_summary)]
    if state.world_horizon() == "long_horizon" and expand_long_horizon:
        remaining = reconcile_long_horizon_plan(state, remaining, world_model_summary)
    state.plan = remaining
    state.active_subgoal = remaining[0] if remaining else ""
    state.subgoal_diagnoses = {
        goal: dict(diagnosis)
        for goal, diagnosis in state.subgoal_diagnoses.items()
        if goal in remaining and isinstance(diagnosis, dict)
    }
    refresh_planner_recovery_progress(state, remaining, world_model_summary)


def software_work_plan_update(state: Any) -> list[str]:
    if state.world_horizon() != "long_horizon":
        return []
    base_objectives = software_work_base_objectives(state)
    stage_state = software_work_stage_overview(state)
    phase_gate = software_work_phase_gate_state(state)
    gate_active = bool(phase_gate.get("active", False))
    gate_phase = str(phase_gate.get("gate_phase", "")).strip()
    gate_status = str(phase_gate.get("gate_status", "")).strip()
    gate_kind = str(phase_gate.get("gate_kind", "")).strip()
    gate_objectives = {
        str(item).strip()
        for item in phase_gate.get("gate_objectives", [])
        if str(item).strip()
    }
    objective_states = stage_state.get("objective_states", {})
    objective_states = objective_states if isinstance(objective_states, dict) else {}
    attempt_counts = stage_state.get("attempt_counts", {})
    attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}
    scored: list[tuple[int, int, str]] = []
    for index, item in enumerate(base_objectives):
        status = str(objective_states.get(item, "pending")).strip() or "pending"
        attempts = max(0, int(attempt_counts.get(item, 0) or 0))
        score = 100 - (index * 5)
        if item == state.active_subgoal.strip():
            score += -12 if status in {"stalled", "regressed"} else 12
        if status == "completed":
            score -= 1000
        elif status == "advanced":
            score += 18
        elif status == "pending":
            score += 8
        elif status == "stalled":
            score -= 18 + (min(3, attempts) * 4)
        elif status == "regressed":
            score -= 26 + (min(3, attempts) * 5)
        if gate_active:
            objective_phase = software_work_objective_phase(item)
            if item in gate_objectives:
                if gate_kind == "merge_acceptance" or gate_status != "stalled":
                    score += 36 - (index * 2)
                else:
                    score -= 8
            elif (
                gate_phase
                and software_work_phase_rank(objective_phase) > software_work_phase_rank(gate_phase)
                and (gate_kind == "merge_acceptance" or gate_status != "stalled")
            ):
                score -= 52
        scored.append((score, index, item))
    ordered: list[str] = []
    for _, _, item in sorted(scored, key=lambda entry: (-entry[0], entry[1], entry[2])):
        if item and item not in ordered:
            ordered.append(item)
    return ordered[:6]


def campaign_contract_state(state: Any) -> dict[str, object]:
    if state.world_horizon() != "long_horizon":
        return {}
    stage_state = software_work_stage_overview(state)
    phase_gate = software_work_phase_gate_state(state)
    objective_states = stage_state.get("objective_states", {})
    objective_states = objective_states if isinstance(objective_states, dict) else {}
    recent_outcomes = stage_state.get("recent_outcomes", [])
    recent_outcomes = recent_outcomes if isinstance(recent_outcomes, list) else []
    attempt_counts = stage_state.get("attempt_counts", {})
    attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}

    regressed_objectives: list[str] = []
    stalled_objectives: list[str] = []
    for objective in software_work_plan_update(state)[:8]:
        normalized = str(objective).strip()
        if not normalized or subgoal_satisfied(state, normalized, state.world_model_summary):
            continue
        status = str(objective_states.get(normalized, "pending")).strip() or "pending"
        if status == "regressed" and normalized not in regressed_objectives:
            regressed_objectives.append(normalized)
        elif status == "stalled" and normalized not in stalled_objectives:
            stalled_objectives.append(normalized)
    for item in recent_outcomes[-6:]:
        if not isinstance(item, dict):
            continue
        objective = str(item.get("objective", "")).strip()
        status = str(item.get("status", "")).strip()
        if not objective or subgoal_satisfied(state, objective, state.world_model_summary):
            continue
        if status == "regressed" and objective not in regressed_objectives:
            regressed_objectives.append(objective)
        elif status == "stalled" and objective not in stalled_objectives:
            stalled_objectives.append(objective)

    current_objective = current_software_work_objective(state)
    gate_objectives = [
        str(item).strip()
        for item in phase_gate.get("gate_objectives", [])
        if str(item).strip() and not subgoal_satisfied(state, str(item).strip(), state.world_model_summary)
    ] if isinstance(phase_gate, dict) else []

    anchor_objectives: list[str] = []
    for objective in [
        *gate_objectives,
        *regressed_objectives,
        current_objective,
        *stalled_objectives,
        *software_work_plan_update(state),
    ]:
        normalized = str(objective).strip()
        if not normalized or normalized in anchor_objectives:
            continue
        if subgoal_satisfied(state, normalized, state.world_model_summary):
            continue
        anchor_objectives.append(normalized)

    recent_regressions = [
        str(item).strip()
        for item in state.latest_state_transition.get("regressions", [])
        if str(item).strip()
    ]
    required_paths = campaign_contract_required_paths(state, anchor_objectives)
    drift_signals: list[str] = []
    if state.consecutive_failures > 0:
        drift_signals.append("failure_pressure")
    if state.consecutive_no_progress_steps > 0:
        drift_signals.append("no_progress")
    if state.repeated_action_count > 1:
        drift_signals.append("repeat_pressure")
    if regressed_objectives:
        drift_signals.append("regressed_obligation")
    if stalled_objectives:
        drift_signals.append("stalled_obligation")
    if recent_regressions:
        drift_signals.append("state_regression")
    drift_pressure = (
        min(3, int(state.consecutive_failures > 0) + max(0, state.repeated_action_count - 1))
        + min(3, int(state.consecutive_no_progress_steps))
        + min(2, len(regressed_objectives))
        + min(1, len(recent_regressions))
    )
    return {
        "current_objective": current_objective,
        "anchor_objectives": anchor_objectives[:6],
        "regressed_objectives": regressed_objectives[:4],
        "stalled_objectives": stalled_objectives[:4],
        "required_paths": required_paths[:6],
        "phase_gate_active": bool(phase_gate.get("active", False)) if isinstance(phase_gate, dict) else False,
        "gate_phase": str(phase_gate.get("gate_phase", "")).strip() if isinstance(phase_gate, dict) else "",
        "recent_regressions": recent_regressions[:4],
        "drift_signals": drift_signals,
        "drift_pressure": drift_pressure,
        "attempt_counts": {
            objective: max(0, int(attempt_counts.get(objective, 0) or 0))
            for objective in anchor_objectives[:6]
        },
    }


def software_work_phase_gate_state(state: Any) -> dict[str, object]:
    if state.world_horizon() != "long_horizon":
        return {}
    phase_state = software_work_phase_state(state)
    if not isinstance(phase_state, dict) or not phase_state:
        return {}
    current_phase = str(phase_state.get("current_phase", "")).strip()
    current_phase_status = str(phase_state.get("current_phase_status", "")).strip()
    if not current_phase or current_phase_status in {"", "absent", "completed", "handoff_ready"}:
        return {}
    gate_objectives: list[str] = []
    for objective in software_work_base_objectives(state):
        normalized = str(objective).strip()
        if not normalized:
            continue
        if software_work_objective_phase(normalized) != current_phase:
            continue
        if subgoal_satisfied(state, normalized, state.world_model_summary):
            continue
        gate_objectives.append(normalized)
    if not gate_objectives:
        return {}
    prioritized: list[str] = []
    for prefix in (
        "accept required branch ",
        "prepare workflow branch ",
        "regenerate generated artifact ",
        "update workflow path ",
    ):
        for objective in gate_objectives:
            if objective.startswith(prefix) and objective not in prioritized:
                prioritized.append(objective)
    for objective in gate_objectives:
        if objective not in prioritized:
            prioritized.append(objective)
    blocked_phases = [
        phase
        for phase in _SOFTWARE_WORK_PHASES
        if software_work_phase_rank(phase) > software_work_phase_rank(current_phase)
    ]
    gate_kind = "merge_acceptance" if any(
        objective.startswith("accept required branch ")
        for objective in prioritized
    ) else "phase_progression"
    gate_reason = (
        "Required branch acceptance remains unresolved before later workflow phases."
        if gate_kind == "merge_acceptance"
        else f"{current_phase} phase objectives remain unresolved before later workflow phases."
    )
    return {
        "active": True,
        "gate_kind": gate_kind,
        "gate_phase": current_phase,
        "gate_status": current_phase_status,
        "gate_reason": gate_reason,
        "gate_objectives": prioritized[:4],
        "blocked_phases": blocked_phases,
    }


def current_software_work_objective(state: Any) -> str:
    if state.world_horizon() != "long_horizon":
        return ""
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    next_stage = str(artifact.get("next_stage_objective", "")).strip()
    if next_stage and not subgoal_satisfied(state, next_stage, state.world_model_summary):
        return next_stage
    if state.active_subgoal.strip():
        return state.active_subgoal.strip()
    objectives = software_work_base_objectives(state)
    return objectives[0] if objectives else ""


def software_work_stage_overview(state: Any) -> dict[str, object]:
    state_map = dict(state.software_work_stage_state) if isinstance(state.software_work_stage_state, dict) else {}
    objective_states = state_map.get("objective_states", {})
    attempt_counts = state_map.get("attempt_counts", {})
    recent_outcomes = state_map.get("recent_outcomes", [])
    return {
        "current_objective": str(state_map.get("current_objective", "")).strip(),
        "last_status": str(state_map.get("last_status", "")).strip(),
        "objective_states": {
            str(key): str(value).strip()
            for key, value in objective_states.items()
            if str(key).strip() and str(value).strip()
        }
        if isinstance(objective_states, dict)
        else {},
        "attempt_counts": {
            str(key): int(value or 0)
            for key, value in attempt_counts.items()
            if str(key).strip()
        }
        if isinstance(attempt_counts, dict)
        else {},
        "recent_outcomes": [
            dict(item)
            for item in recent_outcomes[-6:]
            if isinstance(item, dict)
        ]
        if isinstance(recent_outcomes, list)
        else [],
    }


def software_work_phase_state(state: Any) -> dict[str, object]:
    if state.world_horizon() != "long_horizon":
        return {}
    overview = software_work_stage_overview(state)
    objective_states = overview.get("objective_states", {})
    objective_states = objective_states if isinstance(objective_states, dict) else {}
    attempt_counts = overview.get("attempt_counts", {})
    attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}
    recent_outcomes = overview.get("recent_outcomes", [])
    recent_outcomes = recent_outcomes if isinstance(recent_outcomes, list) else []
    test_attempted = any(
        software_work_objective_phase(str(item.get("objective", ""))) == "test"
        for item in recent_outcomes
        if isinstance(item, dict)
    )
    regression_signaled = any(
        str(item.get("status", "")).strip() == "regressed"
        for item in recent_outcomes
        if isinstance(item, dict)
    ) or bool(state.latest_state_transition.get("regressed", False))
    phase_objectives: dict[str, list[str]] = {phase: [] for phase in _SOFTWARE_WORK_PHASES}
    for objective in software_work_base_objectives(state):
        phase = software_work_objective_phase(
            objective,
            test_attempted=test_attempted,
            regression_signaled=regression_signaled,
        )
        if objective and objective not in phase_objectives[phase]:
            phase_objectives[phase].append(objective)
    phase_states: dict[str, dict[str, object]] = {}
    for phase in _SOFTWARE_WORK_PHASES:
        objectives = phase_objectives[phase]
        status_counts = {"pending": 0, "advanced": 0, "stalled": 0, "regressed": 0, "completed": 0}
        phase_attempts = 0
        for objective in objectives:
            status = str(objective_states.get(objective, "pending")).strip() or "pending"
            if status not in status_counts:
                status = "pending"
            status_counts[status] += 1
            phase_attempts += max(0, int(attempt_counts.get(objective, 0) or 0))
        objective_count = len(objectives)
        if objective_count == 0:
            phase_status = "absent"
        elif status_counts["completed"] >= objective_count:
            phase_status = "completed"
        elif status_counts["completed"] + status_counts["advanced"] >= objective_count:
            phase_status = "handoff_ready"
        elif status_counts["regressed"] > 0:
            phase_status = "regressed"
        elif status_counts["stalled"] > 0:
            phase_status = "stalled"
        elif status_counts["advanced"] > 0:
            phase_status = "advanced"
        else:
            phase_status = "pending"
        phase_states[phase] = {
            "status": phase_status,
            "objective_count": objective_count,
            "attempt_count": phase_attempts,
            "objectives": objectives[:3],
            **status_counts,
        }
    current_phase = ""
    for phase in _SOFTWARE_WORK_PHASES:
        status = str(phase_states[phase].get("status", "")).strip()
        if status and status not in {"absent", "completed"}:
            current_phase = phase
            break
    next_phase = ""
    if current_phase:
        for phase in _SOFTWARE_WORK_PHASES[software_work_phase_rank(current_phase) + 1 :]:
            status = str(phase_states[phase].get("status", "")).strip()
            if status and status not in {"absent", "completed"}:
                next_phase = phase
                break
    current_phase_status = str(phase_states.get(current_phase, {}).get("status", "")).strip() if current_phase else ""
    suggested_phase = current_phase
    if current_phase_status == "handoff_ready" and next_phase:
        suggested_phase = next_phase
    return {
        "current_phase": current_phase,
        "current_phase_status": current_phase_status,
        "next_phase": next_phase,
        "suggested_phase": suggested_phase,
        "handoff_ready": bool(current_phase and current_phase_status == "handoff_ready" and next_phase),
        "phase_states": phase_states,
    }


def software_work_base_objectives(state: Any) -> list[str]:
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    staged = artifact.get("staged_plan_update", [])
    prioritized = [str(item).strip() for item in staged if str(item).strip()] if isinstance(staged, list) else []
    derived: list[str] = []
    if state.active_subgoal.strip():
        derived.append(state.active_subgoal.strip())
    derived.extend(str(item).strip() for item in state.plan if str(item).strip())
    derived.extend(pending_synthetic_edit_objectives(state))
    derived.extend(pending_world_software_objectives(state))
    ordered: list[str] = []
    for item in [*prioritized, *derived]:
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def reconcile_long_horizon_plan(
    state: Any,
    remaining: list[str],
    world_model_summary: dict[str, object],
) -> list[str]:
    concrete_remaining = [
        goal
        for goal in remaining
        if not is_generic_contract_subgoal(goal)
    ]
    generic_remaining = [
        goal
        for goal in remaining
        if is_generic_contract_subgoal(goal)
    ]
    gate_state = software_work_phase_gate_state(state)
    gate_objectives = [
        str(item).strip()
        for item in gate_state.get("gate_objectives", [])
        if (
            str(item).strip()
            and not is_generic_contract_subgoal(str(item).strip())
            and not subgoal_satisfied(state, str(item).strip(), world_model_summary)
        )
    ] if isinstance(gate_state, dict) else []
    stage_overview = software_work_stage_overview(state)
    objective_states = stage_overview.get("objective_states", {})
    objective_states = objective_states if isinstance(objective_states, dict) else {}
    stalled_or_regressed = [
        objective
        for objective, status in objective_states.items()
        if str(objective).strip()
        and not is_generic_contract_subgoal(str(objective).strip())
        and status in {"stalled", "regressed"}
        and not subgoal_satisfied(state, str(objective).strip(), world_model_summary)
    ]
    reconciled: list[str] = []
    for objective in [
        *gate_objectives,
        *stalled_or_regressed,
        *artifact_stage_objectives(state, world_model_summary),
        *pending_world_software_objectives(state),
        *concrete_remaining,
        *generic_remaining,
    ]:
        normalized = str(objective).strip()
        if (
            not normalized
            or normalized in reconciled
            or subgoal_satisfied(state, normalized, world_model_summary)
            or not is_plan_trackable_subgoal(normalized)
        ):
            continue
        reconciled.append(normalized)
    return reconciled


def artifact_stage_objectives(state: Any, world_model_summary: dict[str, object]) -> list[str]:
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    staged = artifact.get("staged_plan_update", [])
    if not isinstance(staged, list):
        return []
    return [
        str(item).strip()
        for item in staged
        if str(item).strip() and not subgoal_satisfied(state, str(item).strip(), world_model_summary)
    ]


def refresh_planner_recovery_progress(
    state: Any,
    remaining: list[str],
    world_model_summary: dict[str, object],
) -> None:
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    if not artifact:
        state.planner_recovery_artifact = {}
        return
    staged = artifact.get("staged_plan_update", [])
    unresolved_staged = [
        str(item).strip()
        for item in staged
        if str(item).strip() and not subgoal_satisfied(state, str(item).strip(), world_model_summary)
    ] if isinstance(staged, list) else []
    source_subgoal = str(artifact.get("source_subgoal", "")).strip()
    if not remaining and not unresolved_staged:
        state.planner_recovery_artifact = {}
        return
    if source_subgoal and source_subgoal not in remaining and unresolved_staged:
        artifact["source_subgoal"] = unresolved_staged[0]
    elif source_subgoal and source_subgoal not in remaining:
        state.planner_recovery_artifact = {}
        return
    artifact["staged_plan_update"] = unresolved_staged[:4]
    next_stage = str(artifact.get("next_stage_objective", "")).strip()
    if next_stage and subgoal_satisfied(state, next_stage, world_model_summary):
        next_stage = ""
    if not next_stage and unresolved_staged:
        next_stage = unresolved_staged[0]
    if next_stage:
        artifact["next_stage_objective"] = next_stage
    else:
        artifact.pop("next_stage_objective", None)
    state.planner_recovery_artifact = artifact


def record_software_work_outcome(
    state: Any,
    *,
    objective: str,
    step_index: int,
    command: str,
    verification_passed: bool,
    progress_delta: float,
    state_regressed: bool,
    state_transition: dict[str, object],
) -> None:
    normalized = str(objective).strip()
    if state.world_horizon() != "long_horizon" or not normalized:
        return
    status = "stalled"
    if verification_passed or subgoal_satisfied(state, normalized, state.world_model_summary):
        status = "completed"
    elif state_regressed or bool(state_transition.get("regressed", False)) or list(state_transition.get("regressions", [])):
        status = "regressed"
    elif float(progress_delta or 0.0) > 0.0:
        status = "advanced"
    state_map = dict(state.software_work_stage_state) if isinstance(state.software_work_stage_state, dict) else {}
    objective_states = dict(state_map.get("objective_states", {})) if isinstance(state_map.get("objective_states", {}), dict) else {}
    attempt_counts = dict(state_map.get("attempt_counts", {})) if isinstance(state_map.get("attempt_counts", {}), dict) else {}
    recent_outcomes = [
        dict(item)
        for item in state_map.get("recent_outcomes", [])
        if isinstance(item, dict)
    ]
    objective_states[normalized] = status
    attempt_counts[normalized] = int(attempt_counts.get(normalized, 0) or 0) + 1
    recent_outcomes.append(
        {
            "objective": normalized,
            "status": status,
            "step_index": int(step_index),
            "command": str(command).strip(),
            "progress_delta": round(float(progress_delta or 0.0), 4),
            "regressed": bool(state_regressed or state_transition.get("regressed", False)),
        }
    )
    state.software_work_stage_state = {
        "current_objective": normalized,
        "last_status": status,
        "objective_states": objective_states,
        "attempt_counts": attempt_counts,
        "recent_outcomes": recent_outcomes[-8:],
    }


def pending_synthetic_edit_objectives(state: Any) -> list[str]:
    metadata = dict(getattr(state.task, "metadata", {}) or {})
    edit_plan = metadata.get("synthetic_edit_plan", [])
    if not isinstance(edit_plan, list):
        return []
    executed = state.all_executed_command_signatures()
    pending: list[str] = []
    for step in edit_plan:
        if not isinstance(step, dict):
            continue
        path = str(step.get("path", "")).strip()
        if not path:
            continue
        if any(path.lower() in signature.lower() for signature in executed):
            continue
        pending.append(f"apply planned edit {path}")
    return pending


def pending_world_software_objectives(state: Any) -> list[str]:
    summary = dict(state.world_model_summary or {})
    pending: list[str] = []
    for path in summary.get("missing_expected_artifacts", []):
        normalized = str(path).strip()
        if normalized:
            pending.append(f"complete implementation for {normalized}")
    for path in summary.get("unsatisfied_expected_contents", []):
        normalized = str(path).strip()
        if normalized:
            pending.append(f"revise implementation for {normalized}")
    updated_reports = {str(item).strip() for item in summary.get("updated_report_paths", []) if str(item).strip()}
    for path in summary.get("workflow_report_paths", []):
        normalized = str(path).strip()
        if normalized and normalized not in updated_reports:
            pending.append(f"write workflow report {normalized}")
    updated_generated = {str(item).strip() for item in summary.get("updated_generated_paths", []) if str(item).strip()}
    for path in summary.get("workflow_generated_paths", []):
        normalized = str(path).strip()
        if normalized and normalized not in updated_generated:
            pending.append(f"regenerate generated artifact {normalized}")
    for branch in summary.get("workflow_required_merges", []):
        normalized = str(branch).strip()
        if normalized:
            pending.append(f"accept required branch {normalized}")
    for branch in summary.get("workflow_branch_targets", []):
        normalized = str(branch).strip()
        if normalized:
            pending.append(f"prepare workflow branch {normalized}")
    for label in summary.get("workflow_required_tests", []):
        normalized = str(label).strip()
        if normalized:
            pending.append(f"run workflow test {normalized}")
    return pending


def campaign_contract_required_paths(state: Any, objectives: list[str]) -> list[str]:
    required: list[str] = []
    for objective in objectives:
        normalized = str(objective).strip()
        if not normalized:
            continue
        target = ""
        for prefix in (
            "apply planned edit ",
            "complete implementation for ",
            "revise implementation for ",
            "materialize expected artifact ",
            "remove forbidden artifact ",
            "preserve required artifact ",
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
            "accept required branch ",
            "prepare workflow branch ",
            "run workflow test ",
        ):
            if normalized.startswith(prefix):
                target = normalized.removeprefix(prefix).strip()
                break
        if target and target not in required:
            required.append(target)
    for key in (
        "missing_expected_artifacts",
        "unsatisfied_expected_contents",
        "present_forbidden_artifacts",
        "changed_preserved_artifacts",
        "workflow_report_paths",
        "workflow_generated_paths",
        "workflow_required_merges",
        "workflow_branch_targets",
        "workflow_required_tests",
    ):
        for value in state.world_model_summary.get(key, []):
            normalized = str(value).strip()
            if normalized and normalized not in required:
                required.append(normalized)
    return required


def is_plan_trackable_subgoal(goal: str) -> bool:
    normalized = str(goal).strip()
    if not normalized:
        return False
    return normalized.startswith(
        (
            "prepare workflow branch ",
            "accept required branch ",
            "materialize expected artifact ",
            "complete implementation for ",
            "revise implementation for ",
            "remove forbidden artifact ",
            "preserve required artifact ",
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
            "run workflow test ",
            "satisfy verifier contract",
            "check verifier contract before terminating",
            "validate expected artifacts and forbidden artifacts before termination",
            "verify preserved artifacts remain unchanged before termination",
        )
    )


def is_generic_contract_subgoal(goal: str) -> bool:
    return str(goal).strip() in {
        "satisfy verifier contract",
        "check verifier contract before terminating",
        "validate expected artifacts and forbidden artifacts before termination",
        "verify preserved artifacts remain unchanged before termination",
    }


def subgoal_satisfied(state: Any, goal: str, world_model_summary: dict[str, object]) -> bool:
    normalized = str(goal).strip()
    if not normalized:
        return True
    if normalized in {
        "satisfy verifier contract",
        "check verifier contract before terminating",
    }:
        return verifier_contract_satisfied(state, world_model_summary)
    if normalized == "validate expected artifacts and forbidden artifacts before termination":
        return validation_contract_satisfied(world_model_summary)
    if normalized == "verify preserved artifacts remain unchanged before termination":
        return preservation_contract_satisfied(world_model_summary)
    if normalized.startswith("prepare workflow branch "):
        branch = normalized.removeprefix("prepare workflow branch ").strip()
        return recent_command_mentions(state, branch, prefixes=("git switch", "git checkout"))
    if normalized.startswith("accept required branch "):
        branch = normalized.removeprefix("accept required branch ").strip()
        return recent_command_mentions(state, branch, prefixes=("git merge", "git cherry-pick", "git rebase"))
    if normalized.startswith(
        (
            "materialize expected artifact ",
            "complete implementation for ",
            "revise implementation for ",
        )
    ):
        path = normalized
        for prefix in (
            "materialize expected artifact ",
            "complete implementation for ",
            "revise implementation for ",
        ):
            if path.startswith(prefix):
                path = path.removeprefix(prefix).strip()
                break
        existing = {str(item) for item in world_model_summary.get("existing_expected_artifacts", [])}
        unsatisfied_contents = {str(item) for item in world_model_summary.get("unsatisfied_expected_contents", [])}
        if path not in existing:
            return False
        return path not in unsatisfied_contents
    if normalized.startswith("remove forbidden artifact "):
        path = normalized.removeprefix("remove forbidden artifact ").strip()
        present = {str(item) for item in world_model_summary.get("present_forbidden_artifacts", [])}
        return path not in present
    if normalized.startswith("preserve required artifact "):
        path = normalized.removeprefix("preserve required artifact ").strip()
        intact = {str(item) for item in world_model_summary.get("intact_preserved_artifacts", [])}
        return path in intact
    if normalized.startswith("update workflow path "):
        path = normalized.removeprefix("update workflow path ").strip()
        updated = {str(item) for item in world_model_summary.get("updated_workflow_paths", [])}
        return path in updated
    if normalized.startswith("regenerate generated artifact "):
        path = normalized.removeprefix("regenerate generated artifact ").strip()
        updated = {str(item) for item in world_model_summary.get("updated_generated_paths", [])}
        return path in updated
    if normalized.startswith("write workflow report "):
        path = normalized.removeprefix("write workflow report ").strip()
        updated = {str(item) for item in world_model_summary.get("updated_report_paths", [])}
        return path in updated
    if normalized.startswith("run workflow test "):
        label = normalized.removeprefix("run workflow test ").strip()
        return workflow_test_satisfied(state, label)
    return False


def validation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
    return not any(
        list(world_model_summary.get(key, []))
        for key in (
            "missing_expected_artifacts",
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
        )
    )


def preservation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
    return not any(
        list(world_model_summary.get(key, []))
        for key in (
            "changed_preserved_artifacts",
            "missing_preserved_artifacts",
        )
    )


def verifier_contract_satisfied(state: Any, world_model_summary: dict[str, object]) -> bool:
    if not validation_contract_satisfied(world_model_summary):
        return False
    if not preservation_contract_satisfied(world_model_summary):
        return False
    updated_workflow_paths = {
        str(item).strip() for item in world_model_summary.get("updated_workflow_paths", []) if str(item).strip()
    }
    updated_generated_paths = {
        str(item).strip() for item in world_model_summary.get("updated_generated_paths", []) if str(item).strip()
    }
    updated_report_paths = {
        str(item).strip() for item in world_model_summary.get("updated_report_paths", []) if str(item).strip()
    }
    pending_workflow_paths = [
        str(item).strip()
        for item in world_model_summary.get("workflow_expected_changed_paths", [])
        if str(item).strip() and str(item).strip() not in updated_workflow_paths
    ]
    pending_generated_paths = [
        str(item).strip()
        for item in world_model_summary.get("workflow_generated_paths", [])
        if str(item).strip() and str(item).strip() not in updated_generated_paths
    ]
    pending_report_paths = [
        str(item).strip()
        for item in world_model_summary.get("workflow_report_paths", [])
        if str(item).strip() and str(item).strip() not in updated_report_paths
    ]
    pending_merges = [
        str(item).strip()
        for item in world_model_summary.get("workflow_required_merges", [])
        if str(item).strip() and not subgoal_satisfied(state, f"accept required branch {str(item).strip()}", world_model_summary)
    ]
    pending_branches = [
        str(item).strip()
        for item in world_model_summary.get("workflow_branch_targets", [])
        if str(item).strip() and not subgoal_satisfied(state, f"prepare workflow branch {str(item).strip()}", world_model_summary)
    ]
    pending_tests = [
        str(item).strip()
        for item in world_model_summary.get("workflow_required_tests", [])
        if str(item).strip() and not subgoal_satisfied(state, f"run workflow test {str(item).strip()}", world_model_summary)
    ]
    return not any(
        (
            pending_workflow_paths,
            pending_generated_paths,
            pending_report_paths,
            pending_merges,
            pending_branches,
            pending_tests,
        )
    )


def workflow_test_satisfied(state: Any, label: str) -> bool:
    normalized_label = str(label).strip().lower()
    if not normalized_label:
        return False
    verifier = state.task.metadata.get("semantic_verifier", {})
    verifier = verifier if isinstance(verifier, dict) else {}
    tokens = workflow_test_match_tokens(normalized_label, verifier)
    for step in reversed(state.history):
        if not bool(step.verification.get("passed", False)):
            continue
        command = " ".join(
            value.strip().lower()
            for value in (
                str(step.content),
                str(step.command_result.get("command", "")) if isinstance(step.command_result, dict) else "",
            )
            if value.strip()
        )
        if not command:
            continue
        if normalized_label in command:
            return True
        if tokens and all(token in command for token in tokens):
            return True
    return False


def workflow_test_match_tokens(label: str, verifier: dict[str, object]) -> list[str]:
    for rule in verifier.get("test_commands", []):
        if not isinstance(rule, dict):
            continue
        candidate_label = str(rule.get("label", "")).strip().lower()
        if candidate_label != label:
            continue
        argv = rule.get("argv", [])
        if not isinstance(argv, list):
            return []
        tokens = [
            normalized
            for value in argv
            if (normalized := str(value).strip().lower()) and not normalized.startswith("-")
        ]
        return tokens
    return []


def recent_command_mentions(state: Any, token: str, *, prefixes: tuple[str, ...] = ()) -> bool:
    normalized_token = str(token).strip().lower()
    if not normalized_token:
        return False
    normalized_prefixes = tuple(str(prefix).strip().lower() for prefix in prefixes if str(prefix).strip())
    for step in reversed(state.history[-6:]):
        command = str(step.content).strip().lower()
        if not command or normalized_token not in command:
            continue
        if normalized_prefixes and not any(command.startswith(prefix) for prefix in normalized_prefixes):
            continue
        return True
    return False
