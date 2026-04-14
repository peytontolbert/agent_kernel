from __future__ import annotations

import json
from typing import Any

from ..extensions.improvement.prompt_improvement import retained_planner_controls
from ..schemas import TaskSpec


def build_plan(kernel: Any, task: TaskSpec) -> list[str]:
    planner_controls = {
        **kernel.world_model.retained_planning_controls(),
        **kernel._planner_controls(),
    }
    world_model_summary = kernel.world_model.summarize(task)
    plan: list[str] = []
    prefer_expected_artifacts_first = bool(planner_controls.get("prefer_expected_artifacts_first", True))
    if bool(planner_controls.get("prepend_verifier_contract_check", False)):
        plan.append("check verifier contract before terminating")
    plan.extend(kernel._workflow_plan_steps(task, planner_controls=planner_controls))
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


def workflow_plan_steps(task: TaskSpec, *, planner_controls: dict[str, object] | None = None) -> list[str]:
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


def planner_controls(kernel: Any) -> dict[str, object]:
    if not kernel.config.use_prompt_proposals:
        return {}
    path = kernel.config.prompt_proposals_path
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return retained_planner_controls(payload)
