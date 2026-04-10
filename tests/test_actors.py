from agent_kernel import (
    coding_actor_applicable,
    coding_actor_episode_summary,
    coding_actor_plan_for_task,
    CodingActorMode,
    CodingActorOutcome,
    CodingActorPlan,
    CodingActorPolicy,
    CodingActorResult,
    coding_actor_kernel_summary,
    default_coding_actor_policy,
)
from agent_kernel.schemas import EpisodeRecord, StepRecord, TaskSpec


def test_default_coding_actor_policy_exposes_supervised_coding_contract():
    policy = default_coding_actor_policy()

    assert policy.actor_id == "coding"
    assert policy.mode is CodingActorMode.SUPERVISED
    assert policy.require_verifier is True
    assert policy.allow_remote_control_takeover is True
    assert "project" in policy.supported_families
    assert "repo_chore" in policy.supported_families


def test_coding_actor_kernel_summary_serializes_plan_budget_and_support_contract():
    policy = CodingActorPolicy(max_steps=16, max_commands=10, max_runtime_minutes=30)
    plan = CodingActorPlan(
        objective="Patch retained coding regressions",
        workspace_scope=("agent_kernel/improvement.py", "agent_kernel/cycle_runner.py"),
        expected_files=("agent_kernel/improvement.py",),
        preferred_families=("project", "integration"),
        retrieval_required=True,
    )

    summary = coding_actor_kernel_summary(policy=policy, plan=plan)

    assert summary["actor_type"] == "coding"
    assert summary["status"] == "planned"
    assert summary["workspace_scope_size"] == 2
    assert summary["retrieval_required"] is True
    assert summary["action_budget"] == {
        "max_steps": 16,
        "max_commands": 10,
        "max_runtime_minutes": 30,
    }
    assert summary["control_contract"]["allow_remote_control_takeover"] is True


def test_coding_actor_kernel_summary_marks_decision_ready_result():
    policy = default_coding_actor_policy()
    plan = CodingActorPlan(
        objective="Apply bounded coding fix",
        workspace_scope=("agent_kernel/actors/coding.py",),
    )
    result = CodingActorResult(
        actor_id="coding",
        mode=CodingActorMode.SELF_CONTROLLED,
        outcome=CodingActorOutcome.RETAINED,
        steps_executed=6,
        commands_executed=4,
        runtime_minutes=7.5,
        files_touched=("agent_kernel/actors/coding.py",),
        tests_run=("pytest -q tests/test_actors.py",),
        verifier_passed=True,
        retained_gain_hint=True,
        decision_credit_hint=True,
        retrieval_support_used=True,
        family="project",
    )

    summary = coding_actor_kernel_summary(policy=policy, plan=plan, result=result)

    assert summary["status"] == "completed"
    assert summary["decision_ready"] is True
    assert summary["retained_gain_hint"] is True
    assert summary["decision_credit_hint"] is True
    assert summary["result"]["mode"] == "self_controlled"
    assert summary["result"]["outcome"] == "retained"


def test_coding_actor_helpers_attach_to_light_supervision_task():
    task = TaskSpec(
        task_id="repo_cleanup_review_task",
        prompt="Clean repo status",
        workspace_subdir="repo_cleanup_review_task",
        expected_files=["reports/status.txt"],
        metadata={
            "benchmark_family": "repo_chore",
            "light_supervision_candidate": True,
            "workflow_guard": {"required_capabilities": ["workspace_fs", "workspace_exec"]},
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=task.workspace_subdir,
        success=True,
        steps=[
            StepRecord(
                index=1,
                thought="inspect",
                action="shell",
                content="pytest -q tests/test_actors.py",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": True},
                retrieval_influenced=True,
            )
        ],
        termination_reason="completed",
    )

    assert coding_actor_applicable(task) is True
    plan = coding_actor_plan_for_task(task)
    summary = coding_actor_episode_summary(
        policy=default_coding_actor_policy(),
        task=task,
        episode=episode,
    )

    assert plan.workspace_scope == ("repo_cleanup_review_task",)
    assert summary["actor_type"] == "coding"
    assert summary["decision_ready"] is True
    assert summary["result"]["tests_run"] == ["pytest -q tests/test_actors.py"]


def test_coding_actor_result_uses_terminal_verifier_success_not_all_steps_verified():
    policy = default_coding_actor_policy()
    task = TaskSpec(
        task_id="integration_fix_task",
        prompt="Repair integration task",
        workspace_subdir="integration_fix_task",
        metadata={"benchmark_family": "integration", "observed_runtime_seconds": 120.0},
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=task.workspace_subdir,
        success=True,
        steps=[
            StepRecord(
                index=1,
                thought="first attempt",
                action="shell",
                content="pytest -q tests/test_one.py",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False},
            ),
            StepRecord(
                index=2,
                thought="recovery attempt",
                action="shell",
                content="pytest -q tests/test_two.py",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": True},
            ),
        ],
        termination_reason="success",
    )

    summary = coding_actor_episode_summary(policy=policy, task=task, episode=episode)

    assert summary["decision_ready"] is True
    assert summary["result"]["outcome"] == "completed"
    assert summary["result"]["verifier_passed"] is True
