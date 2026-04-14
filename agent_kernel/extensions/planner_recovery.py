from __future__ import annotations

from ..state import AgentState


def refresh_planner_recovery_artifact(
    state: AgentState,
    *,
    normalize_command_fn,
) -> None:
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
    if not planner_recovery_surface_exhausted(
        state,
        active_subgoal,
        normalize_command_fn=normalize_command_fn,
    ):
        state.planner_recovery_artifact = {}
        return
    state.planner_recovery_artifact = build_planner_recovery_artifact(
        state,
        active_subgoal=active_subgoal,
        diagnosis=diagnosis,
        normalize_command_fn=normalize_command_fn,
    )


def planner_recovery_surface_exhausted(
    state: AgentState,
    goal: str,
    *,
    normalize_command_fn,
) -> bool:
    if not state.history or not state.task.suggested_commands:
        return False
    if state.consecutive_failures <= 0 and state.consecutive_no_progress_steps <= 0:
        return False
    failed_commands = state.all_failed_command_signatures()
    last_command = str(state.last_action_signature).partition(":")[2]
    last_canonical = normalize_command_fn(last_command)
    seen_recovery_candidate = False
    for command in state.task.suggested_commands[:5]:
        normalized = str(command).strip()
        canonical = normalize_command_fn(normalized)
        if not canonical or not command_matches_subgoal_surface(state, goal=goal, command=normalized):
            continue
        seen_recovery_candidate = True
        if canonical in failed_commands:
            continue
        if canonical == last_canonical and state.repeated_action_count > 1:
            continue
        return False
    return seen_recovery_candidate


def build_planner_recovery_artifact(
    state: AgentState,
    *,
    active_subgoal: str,
    diagnosis: dict[str, object],
    normalize_command_fn,
) -> dict[str, object]:
    path = str(diagnosis.get("path", "")).strip() or subgoal_path(active_subgoal)
    related_objectives = planner_recovery_related_objectives(
        state,
        primary_subgoal=active_subgoal,
    )
    ranked_objectives = planner_recovery_ranked_objectives(
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
    rewritten_subgoal = planner_rewrite_subgoal(
        active_subgoal,
        path=path,
        related_objectives=related_objectives,
    )
    failed_commands = state.all_failed_command_signatures()
    stale_commands = [
        str(command).strip()
        for command in state.task.suggested_commands[:5]
        if command_matches_subgoal_surface(state, goal=active_subgoal, command=str(command).strip())
        and normalize_command_fn(str(command).strip()) in failed_commands
    ]
    focus_paths = planner_recovery_focus_paths(state, primary_path=path)
    return {
        "kind": "planner_recovery_rewrite",
        "source_subgoal": active_subgoal,
        "rewritten_subgoal": rewritten_subgoal,
        "next_stage_objective": staged_plan_update[0] if staged_plan_update else rewritten_subgoal,
        "staged_plan_update": staged_plan_update[:4],
        "objective_kind": (
            "workflow_verifier_recovery"
            if any(
                str(item).startswith(
                    (
                        "update workflow path ",
                        "write workflow report ",
                        "regenerate generated artifact ",
                        "accept required branch ",
                        "run workflow test ",
                        "prepare workflow branch ",
                    )
                )
                for item in related_objectives
            )
            or active_subgoal.startswith(
                (
                    "update workflow path ",
                    "write workflow report ",
                    "regenerate generated artifact ",
                    "accept required branch ",
                    "run workflow test ",
                    "prepare workflow branch ",
                )
            )
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
        "contract_outline": planner_recovery_contract_outline(
            active_subgoal,
            path=path,
            focus_paths=focus_paths,
            related_objectives=staged_plan_update or related_objectives,
        ),
        "source_role": "critic",
        "updated_step_index": int(diagnosis.get("updated_step_index", state.next_step_index()) or state.next_step_index()),
    }


def planner_rewrite_subgoal(goal: str, *, path: str, related_objectives: list[str]) -> str:
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
        return "restore verifier-visible workflow state across " + ", ".join(scope[:4])
    if str(goal).startswith("materialize expected artifact "):
        return f"reframe verifier-visible recovery for expected artifact {path or goal}"
    if str(goal).startswith("remove forbidden artifact "):
        return f"reframe verifier-visible recovery for forbidden artifact {path or goal}"
    if str(goal).startswith("update workflow path "):
        return f"reframe workflow recovery contract for {path or goal}"
    if str(goal).startswith("write workflow report "):
        return f"reframe report recovery contract for {path or goal}"
    return f"reframe verifier-visible recovery for {goal}"


def planner_recovery_related_objectives(state: AgentState, *, primary_subgoal: str) -> list[str]:
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


def planner_recovery_ranked_objectives(
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
        kind = planner_recovery_objective_kind(normalized)
        target = planner_recovery_objective_target(normalized)
        attempt_pressure = planner_recovery_objective_attempt_pressure(state, normalized)
        satisfied = planner_recovery_objective_satisfied(state, normalized)
        score = planner_recovery_objective_base_score(kind)
        reasons: list[str] = []
        if target and target == diagnosis_path:
            score += 24
            reasons.append("matches the diagnosed hotspot")
        if planner_recovery_objective_blocks_verifier(state, normalized):
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


def planner_recovery_objective_kind(objective: str) -> str:
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


def planner_recovery_objective_target(objective: str) -> str:
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


def planner_recovery_objective_base_score(kind: str) -> int:
    return {
        "workflow_path": 120,
        "generated_artifact": 105,
        "workflow_report": 95,
        "required_merge": 85,
        "branch_target": 75,
        "workflow_test": 65,
    }.get(str(kind).strip(), 50)


def planner_recovery_objective_blocks_verifier(state: AgentState, objective: str) -> bool:
    summary = dict(state.world_model_summary or {})
    kind = planner_recovery_objective_kind(objective)
    target = planner_recovery_objective_target(objective)
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
    if kind in {"required_merge", "branch_target", "workflow_test"}:
        return True
    return False


def planner_recovery_objective_attempt_pressure(state: AgentState, objective: str) -> int:
    total = 0
    for step in state.history[-6:]:
        if planner_recovery_command_aligns_objective(str(step.content), objective):
            total += 2 if not step.verification.get("passed", False) else 1
    return total


def planner_recovery_objective_satisfied(state: AgentState, objective: str) -> bool:
    kind = planner_recovery_objective_kind(objective)
    target = planner_recovery_objective_target(objective)
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
    return any(planner_recovery_command_aligns_objective(command, objective) for command in recent_successes)


def planner_recovery_command_aligns_objective(command: str, objective: str) -> bool:
    normalized_command = str(command).strip().lower()
    target = planner_recovery_objective_target(objective).lower()
    kind = planner_recovery_objective_kind(objective)
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


def planner_recovery_focus_paths(state: AgentState, *, primary_path: str) -> list[str]:
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


def planner_recovery_contract_outline(
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
        outline.append("sequence related verifier obligations: " + ", ".join(related_objectives[:3]))
    outline.append("choose a new command path outside the exhausted task-contract repair set")
    if focus_paths and focus_paths[0] != primary:
        outline.insert(1, f"reconcile related verifier hotspots: {', '.join(focus_paths[:3])}")
    return outline[:4]


def command_matches_subgoal_surface(state: AgentState, *, goal: str, command: str) -> bool:
    normalized_goal = str(goal).strip()
    normalized_command = str(command).strip().lower()
    if not normalized_goal or not normalized_command:
        return False
    path = subgoal_path(normalized_goal).lower()
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


def subgoal_path(goal: str) -> str:
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


__all__ = [
    "build_planner_recovery_artifact",
    "command_matches_subgoal_surface",
    "planner_recovery_command_aligns_objective",
    "planner_recovery_contract_outline",
    "planner_recovery_focus_paths",
    "planner_recovery_objective_attempt_pressure",
    "planner_recovery_objective_base_score",
    "planner_recovery_objective_blocks_verifier",
    "planner_recovery_objective_kind",
    "planner_recovery_objective_satisfied",
    "planner_recovery_objective_target",
    "planner_recovery_ranked_objectives",
    "planner_recovery_related_objectives",
    "planner_recovery_surface_exhausted",
    "planner_rewrite_subgoal",
    "refresh_planner_recovery_artifact",
    "subgoal_path",
]
