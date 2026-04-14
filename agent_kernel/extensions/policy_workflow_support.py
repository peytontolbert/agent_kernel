from __future__ import annotations

from ..extensions.policy_command_utils import canonicalize_command as _canonicalize_command
from ..state import _software_work_objective_phase, _software_work_phase_rank


def planner_recovery_stage_command_score(policy, state, command: str, *, role: str) -> int:
    if role not in {"planner", "critic"}:
        return 0
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    if str(artifact.get("kind", "")).strip() != "planner_recovery_rewrite":
        return 0
    ranked = artifact.get("ranked_objectives", [])
    if not isinstance(ranked, list):
        ranked = []
    best = 0
    for index, item in enumerate(ranked[:3], start=1):
        if not isinstance(item, dict):
            continue
        objective = str(item.get("objective", "")).strip()
        if not objective or not command_matches_planner_recovery_objective(command, objective):
            continue
        base = max(1, 7 - (index * 2))
        if index == 1 and role == "planner":
            base += 2
        best = max(best, base)
    return best


def software_work_phase_gate_command_score(command: str, state) -> int:
    gate_state = state.software_work_phase_gate_state()
    if not isinstance(gate_state, dict) or not gate_state:
        return 0
    gate_phase = str(gate_state.get("gate_phase", "")).strip()
    gate_objectives = [
        str(item).strip()
        for item in gate_state.get("gate_objectives", [])
        if str(item).strip()
    ]
    if command_matches_any_software_work_objective(command, gate_objectives):
        return 10
    candidate_phase = software_work_command_phase(state, command)
    if gate_phase and candidate_phase and _software_work_phase_rank(candidate_phase) > _software_work_phase_rank(gate_phase):
        return -12
    return -4 if gate_objectives else 0


def command_matches_any_software_work_objective(command: str, objectives: list[str]) -> bool:
    return any(
        command_matches_software_work_objective(command, objective)
        for objective in objectives
    )


def command_matches_software_work_objective(command: str, objective: str) -> bool:
    normalized_objective = str(objective).strip().lower()
    if not normalized_objective:
        return False
    if command_matches_planner_recovery_objective(command, normalized_objective):
        return True
    normalized_command = str(command).strip().lower()
    for prefix in (
        "apply planned edit ",
        "complete implementation for ",
        "revise implementation for ",
        "materialize expected artifact ",
        "remove forbidden artifact ",
        "preserve required artifact ",
    ):
        if normalized_objective.startswith(prefix):
            target = normalized_objective.removeprefix(prefix).strip()
            return bool(target) and target in normalized_command
    return False


def software_work_command_phase(state, command: str) -> str:
    normalized_command = str(command).strip().lower()
    if not normalized_command:
        return ""
    for objective in state.software_work_plan_update()[:6]:
        if command_matches_software_work_objective(normalized_command, objective):
            return _software_work_objective_phase(objective)
    if any(token in normalized_command for token in ("pytest", "unittest", "nose", "tox", "smoke", "test")):
        return "test"
    if any(token in normalized_command for token in ("git merge", "git cherry-pick", "git rebase", "codegen", "generate")):
        return "migration"
    if any(token in normalized_command for token in ("report", "summary", "postmortem", "fix", "repair")):
        return "follow_up_fix"
    return "implementation"


def command_matches_planner_recovery_objective(command: str, objective: str) -> bool:
    normalized_command = str(command).strip().lower()
    normalized_objective = str(objective).strip().lower()
    if not normalized_command or not normalized_objective:
        return False
    for prefix in (
        "update workflow path ",
        "regenerate generated artifact ",
        "write workflow report ",
        "accept required branch ",
        "prepare workflow branch ",
        "run workflow test ",
    ):
        if normalized_objective.startswith(prefix):
            target = normalized_objective.removeprefix(prefix).strip()
            if not target:
                return False
            if prefix == "accept required branch ":
                return "git merge" in normalized_command and target in normalized_command
            if prefix == "prepare workflow branch ":
                return target in normalized_command and any(
                    token in normalized_command for token in ("git checkout", "git switch", "git branch")
                )
            if prefix == "run workflow test ":
                objective_tokens = {token for token in target.split() if len(token) > 2}
                return (
                    ("test" in normalized_command or "pytest" in normalized_command)
                    and bool(objective_tokens.intersection(set(normalized_command.split())))
                )
            return target in normalized_command
    return any(token in normalized_command for token in normalized_objective.split() if len(token) > 3)


def failed_command_attempts(state, command: str) -> int:
    canonical = _canonicalize_command(command)
    if not canonical:
        return 0
    return sum(
        1
        for step in state.history
        if not step.verification.get("passed", False)
        and canonical == _canonicalize_command(str(step.content))
    )


def recovery_loop_penalty(state, command: str) -> int:
    canonical = _canonicalize_command(command)
    if not canonical:
        return 0
    penalty = 0
    failed_attempts = failed_command_attempts(state, command)
    if failed_attempts:
        penalty += failed_attempts * 3
    if canonical == _canonicalize_command(str(state.last_action_signature).partition(":")[2]):
        penalty += max(1, int(state.repeated_action_count) - 1) * 2
    if state.consecutive_no_progress_steps > 0 and canonical in state.all_executed_command_signatures():
        penalty += 1
    return penalty


def active_subgoal_repair_shape_bonus(*, active_subgoal: str, command: str, path: str) -> int:
    del path
    normalized_goal = str(active_subgoal).strip().lower()
    if not normalized_goal:
        return 0
    if normalized_goal.startswith("remove forbidden artifact "):
        if command.startswith("rm ") or command.startswith("unlink ") or " rm " in command:
            return 3
        return 1
    if normalized_goal.startswith("materialize expected artifact "):
        if (
            "printf " in command
            or " > " in command
            or command.startswith("touch ")
            or command.startswith("cp ")
            or command.startswith("mkdir -p ")
        ):
            return 3
        return 1
    if normalized_goal.startswith("preserve required artifact "):
        if (
            command.startswith("cat ")
            or command.startswith("sed -n ")
            or command.startswith("git diff ")
            or command.startswith("diff ")
            or " pytest" in command
            or command.startswith("pytest ")
        ):
            return 2
    return 0
