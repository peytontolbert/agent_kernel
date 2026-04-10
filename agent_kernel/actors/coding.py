"""Coding actor contracts for unattended kernel execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from agent_kernel.schemas import EpisodeRecord, TaskSpec, episode_success_criteria


class CodingActorMode(StrEnum):
    """How tightly the kernel constrains actor execution."""

    REMOTE_CONTROLLED = "remote_controlled"
    SUPERVISED = "supervised"
    SELF_CONTROLLED = "self_controlled"


class CodingActorOutcome(StrEnum):
    """Terminal outcome for a bounded coding actor attempt."""

    COMPLETED = "completed"
    RETAINED = "retained"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class CodingActorPolicy:
    """Kernel-level policy surface for a coding actor."""

    actor_id: str = "coding"
    mode: CodingActorMode = CodingActorMode.SUPERVISED
    max_steps: int = 12
    max_commands: int = 8
    max_runtime_minutes: int = 20
    require_verifier: bool = True
    require_workspace_scope: bool = True
    allow_remote_control_takeover: bool = True
    support_retrieval_context: bool = True
    support_patch_application: bool = True
    support_test_execution: bool = True
    supported_families: tuple[str, ...] = ("project", "repository", "integration", "repo_chore")


@dataclass(frozen=True)
class CodingActorPlan:
    """Bounded action plan that the kernel can hand to a coding actor."""

    objective: str
    workspace_scope: tuple[str, ...]
    expected_files: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ("workspace_fs", "workspace_exec")
    preferred_families: tuple[str, ...] = ()
    retrieval_required: bool = False
    verifier_required: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodingActorResult:
    """Execution result surfaced back to unattended policy and trust layers."""

    actor_id: str
    mode: CodingActorMode
    outcome: CodingActorOutcome
    steps_executed: int
    commands_executed: int
    runtime_minutes: float
    files_touched: tuple[str, ...] = ()
    tests_run: tuple[str, ...] = ()
    verifier_passed: bool = False
    retained_gain_hint: bool = False
    decision_credit_hint: bool = False
    retrieval_support_used: bool = False
    remote_control_takeover_used: bool = False
    family: str = ""
    notes: tuple[str, ...] = ()


def default_coding_actor_policy() -> CodingActorPolicy:
    """Return the default coding actor contract for unattended runs."""

    return CodingActorPolicy()


def coding_actor_plan_for_task(task: TaskSpec) -> CodingActorPlan:
    """Build a bounded coding actor plan from a kernel task."""

    metadata = dict(task.metadata or {})
    workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
    required_capabilities = tuple(
        str(value).strip()
        for value in workflow_guard.get("required_capabilities", [])
        if str(value).strip()
    ) or ("workspace_fs", "workspace_exec")
    preferred_families = tuple(
        family
        for family in (
            str(metadata.get("decision_yield_family", "")).strip(),
            str(metadata.get("benchmark_family", "")).strip(),
        )
        if family
    )
    notes: list[str] = []
    if bool(metadata.get("light_supervision_candidate", False)):
        notes.append("light_supervision_candidate")
    if str(workflow_guard.get("shared_repo_id", "")).strip():
        notes.append("shared_repo_scoped")
    return CodingActorPlan(
        objective=task.prompt,
        workspace_scope=(task.workspace_subdir,),
        expected_files=tuple(str(path).strip() for path in task.expected_files if str(path).strip()),
        required_capabilities=required_capabilities,
        preferred_families=preferred_families,
        retrieval_required=bool(
            str(metadata.get("memory_source", "")).strip() in {"episode_replay", "skill_transfer", "operator_replay"}
        ),
        verifier_required=True,
        notes=tuple(notes),
    )


def coding_actor_applicable(task: TaskSpec, *, policy: CodingActorPolicy | None = None) -> bool:
    """Return whether the coding actor should own this task contract."""

    active_policy = policy or default_coding_actor_policy()
    metadata = dict(task.metadata or {})
    if bool(metadata.get("light_supervision_candidate", False)):
        return True
    family = str(metadata.get("benchmark_family", "")).strip()
    return family in set(active_policy.supported_families)


def coding_actor_result_from_episode(
    *,
    policy: CodingActorPolicy,
    task: TaskSpec,
    episode: EpisodeRecord,
) -> CodingActorResult:
    """Summarize an executed episode as coding-actor output."""

    metadata = dict(task.metadata or {})
    tests_run = tuple(
        str(step.content).strip()
        for step in episode.steps
        if str(step.action).strip() == "shell"
        and any(token in str(step.content).strip() for token in ("pytest", "python -m pytest", "uv run pytest"))
    )
    success_contract = episode_success_criteria(episode)
    verifier_passed = bool(success_contract["terminal_verifier_passed"])
    task_success = bool(success_contract["verifier_aligned_task_success"])
    return CodingActorResult(
        actor_id=policy.actor_id,
        mode=policy.mode,
        outcome=CodingActorOutcome.COMPLETED if task_success else CodingActorOutcome.REJECTED,
        steps_executed=len(episode.steps),
        commands_executed=sum(1 for step in episode.steps if str(step.action).strip() == "shell"),
        runtime_minutes=max(0.0, float(metadata.get("observed_runtime_seconds", 0.0) or 0.0) / 60.0),
        files_touched=tuple(str(path).strip() for path in task.expected_files if str(path).strip()),
        tests_run=tests_run,
        verifier_passed=verifier_passed,
        retained_gain_hint=False,
        decision_credit_hint=task_success,
        retrieval_support_used=any(bool(step.retrieval_influenced) for step in episode.steps),
        remote_control_takeover_used=False,
        family=str(metadata.get("benchmark_family", "bounded")).strip() or "bounded",
        notes=tuple(
            note
            for note in (
                "task_success" if task_success else "task_failure",
                str(episode.termination_reason or "").strip(),
            )
            if note
        ),
    )


def coding_actor_kernel_summary(
    *,
    policy: CodingActorPolicy,
    plan: CodingActorPlan,
    result: CodingActorResult | None = None,
) -> dict[str, object]:
    """Serialize coding actor state into a compact kernel-facing summary."""

    supported_families = [family for family in policy.supported_families if family]
    scope_size = len(tuple(path for path in plan.workspace_scope if path))
    summary: dict[str, object] = {
        "actor_type": "coding",
        "actor_id": policy.actor_id,
        "mode": policy.mode.value,
        "objective": plan.objective,
        "workspace_scope": list(plan.workspace_scope),
        "workspace_scope_size": scope_size,
        "expected_files": list(plan.expected_files),
        "required_capabilities": list(plan.required_capabilities),
        "preferred_families": list(plan.preferred_families),
        "supported_families": supported_families,
        "retrieval_required": bool(plan.retrieval_required),
        "verifier_required": bool(plan.verifier_required and policy.require_verifier),
        "action_budget": {
            "max_steps": int(policy.max_steps),
            "max_commands": int(policy.max_commands),
            "max_runtime_minutes": int(policy.max_runtime_minutes),
        },
        "control_contract": {
            "allow_remote_control_takeover": bool(policy.allow_remote_control_takeover),
            "require_workspace_scope": bool(policy.require_workspace_scope),
            "support_retrieval_context": bool(policy.support_retrieval_context),
            "support_patch_application": bool(policy.support_patch_application),
            "support_test_execution": bool(policy.support_test_execution),
        },
    }
    if result is None:
        summary["status"] = "planned"
        return summary
    result_payload = asdict(result)
    result_payload["mode"] = result.mode.value
    result_payload["outcome"] = result.outcome.value
    for key in ("files_touched", "tests_run", "notes"):
        if isinstance(result_payload.get(key), tuple):
            result_payload[key] = list(result_payload.get(key, ()))
    summary["status"] = "completed"
    summary["result"] = result_payload
    summary["decision_ready"] = bool(result.decision_credit_hint or result.retained_gain_hint)
    summary["retained_gain_hint"] = bool(result.retained_gain_hint)
    summary["decision_credit_hint"] = bool(result.decision_credit_hint)
    return summary


def coding_actor_episode_summary(
    *,
    policy: CodingActorPolicy,
    task: TaskSpec,
    episode: EpisodeRecord,
) -> dict[str, Any]:
    """Attach a coding-actor plan/result summary to an executed episode."""

    return coding_actor_kernel_summary(
        policy=policy,
        plan=coding_actor_plan_for_task(task),
        result=coding_actor_result_from_episode(policy=policy, task=task, episode=episode),
    )
