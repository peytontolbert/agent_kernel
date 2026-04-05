import json
from io import StringIO
import os
from pathlib import Path
import sys
import threading
import time

import evals.harness as harness_module
from agent_kernel.config import KernelConfig
from agent_kernel.schemas import EpisodeRecord, StepRecord, TaskSpec
from agent_kernel.task_bank import TaskBank
from agent_kernel.trust import build_unattended_trust_ledger
from evals.harness import (
    _limit_tasks_for_compare,
    compare_abstraction_transfer_modes,
    compare_skill_modes,
    compare_tolbert_feature_modes,
    compare_tolbert_modes,
    run_eval,
    scoped_eval_config,
)
from evals.metrics import EvalMetrics


def _write_hello_skill(config: KernelConfig) -> None:
    config.skills_path.parent.mkdir(parents=True, exist_ok=True)
    config.skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "kind": "command_sequence",
                        "source_task_id": "hello_task",
                        "applicable_tasks": ["hello_task"],
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "known_failure_types": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_operator_classes(config: KernelConfig) -> None:
    config.operator_classes_path.parent.mkdir(parents=True, exist_ok=True)
    config.operator_classes_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_class_set",
                "lifecycle_state": "retained",
                "operators": [
                    {
                        "operator_id": "operator:file_write:bounded",
                        "operator_kind": "single_emit",
                        "source_task_ids": ["hello_task"],
                        "applicable_capabilities": ["file_write"],
                        "applicable_benchmark_families": ["bounded"],
                        "template_procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "template_contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_success_episode(config: KernelConfig) -> None:
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    config.trajectories_root.joinpath("hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing the string hello agent kernel.",
                "workspace": str(config.workspace_root / "hello_task"),
                "success": True,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing the string hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                },
                "summary": {
                    "executed_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "failure_types": [],
                },
                "fragments": [
                    {
                        "kind": "command",
                        "command": "printf 'hello agent kernel\\n' > hello.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
                "termination_reason": "success",
            }
        ),
        encoding="utf-8",
    )


def _write_failed_episode(config: KernelConfig) -> None:
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    config.trajectories_root.joinpath("hello_task_failed.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing the string hello agent kernel.",
                "workspace": str(config.workspace_root / "hello_task_failed"),
                "success": False,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing the string hello agent kernel.",
                    "workspace_subdir": "hello_task_failed",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                },
                "summary": {
                    "executed_commands": ["false"],
                    "failure_types": ["command_failure"],
                    "transition_failures": ["no_state_progress"],
                },
                "fragments": [
                    {
                        "kind": "command",
                        "command": "false",
                        "passed": False,
                    }
                ],
                "steps": [],
                "termination_reason": "step_limit",
            }
        ),
        encoding="utf-8",
    )


def test_eval_reports_capability_breakdown(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    metrics = run_eval(config=config)

    assert metrics.passed == metrics.total
    assert metrics.average_steps >= 1.0
    assert metrics.average_success_steps >= 1.0
    assert metrics.total_by_capability["file_write"] >= 2
    assert metrics.passed_by_capability["file_write"] == metrics.total_by_capability["file_write"]
    assert metrics.total_by_difficulty["seed"] >= 2
    assert metrics.passed_by_difficulty["seed"] == metrics.total_by_difficulty["seed"]
    assert metrics.termination_reasons["success"] == metrics.total
    assert metrics.skill_selected_steps >= 1
    assert metrics.episodes_with_skill_use >= 1
    assert metrics.average_available_skills > 0.0
    assert metrics.memory_documents == metrics.total
    assert metrics.reusable_skills >= 1
    assert metrics.retrieval_selected_steps == 0
    assert metrics.retrieval_influenced_steps == 0
    assert metrics.retrieval_ranked_skill_steps == 0
    assert metrics.average_first_step_path_confidence == 0.0
    assert metrics.low_confidence_episodes == 0
    assert metrics.generated_total == 0


def test_run_tasks_with_progress_flushes_callbacks_before_unattended_report_write(monkeypatch, tmp_path):
    task = TaskSpec(
        task_id="report_order_task",
        prompt="order",
        workspace_subdir="report_order_task",
        metadata={"benchmark_family": "repository"},
    )
    events: list[str] = []

    class FakeKernel:
        def run_task(self, task, progress_callback=None):
            del progress_callback
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

    monkeypatch.setattr(
        harness_module,
        "write_unattended_task_report",
        lambda **kwargs: events.append("report"),
    )

    results = harness_module._run_tasks_with_progress(
        [task],
        FakeKernel(),
        progress_label=None,
        report_config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            run_reports_dir=tmp_path / "reports",
        ),
        on_task_complete=lambda *_args: events.append("task_complete"),
        on_result=lambda *_args: events.append("result"),
    )

    assert len(results) == 1
    assert events == ["task_complete", "result", "report"]


def test_run_eval_can_write_unattended_reports_for_trust(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="repository_shadow_task",
                    prompt="Create hello.txt.",
                    workspace_subdir="repository_shadow_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "repository", "capability": "file_write"},
                )
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, progress_callback=None):
            del progress_callback
            workspace = tmp_path / "workspace" / task.workspace_subdir
            workspace.mkdir(parents=True, exist_ok=True)
            workspace.joinpath("hello.txt").write_text("hello\n", encoding="utf-8")
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(workspace),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="write file",
                        action="code_execute",
                        content="printf 'hello\\n' > hello.txt",
                        selected_skill_id=None,
                        command_result=None,
                        verification={"passed": True, "reasons": []},
                    )
                ],
                termination_reason="success",
                task_metadata=dict(task.metadata),
            )

        def close(self):
            return None

    monkeypatch.setattr(harness_module, "TaskBank", FakeTaskBank)
    monkeypatch.setattr(harness_module, "AgentKernel", FakeKernel)

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust_ledger.json",
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_min_distinct_families=1,
    )

    metrics = run_eval(
        config=config,
        task_limit=1,
        priority_benchmark_families=["repository"],
        write_unattended_reports=True,
    )

    reports = sorted(config.run_reports_dir.glob("*.json"))
    assert metrics.total == 1
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["report_kind"] == "unattended_task_report"
    assert payload["benchmark_family"] == "repository"
    ledger = build_unattended_trust_ledger(config)
    assert ledger["reports_considered"] == 1
    assert ledger["overall_assessment"]["status"] == "trusted"


def test_run_eval_reports_world_feedback_calibration(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="world_feedback_task",
                    prompt="wf",
                    workspace_subdir="world_feedback_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "long_horizon"},
                )
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[
                        StepRecord(
                            index=1,
                            thought="wf",
                            action="code_execute",
                            content="printf 'hello\\n' > hello.txt",
                        selected_skill_id=None,
                        command_result=None,
                            verification={"passed": True, "reasons": []},
                            active_subgoal="materialize expected artifact report.txt",
                            acting_role="executor",
                            world_model_horizon="long_horizon",
                            state_progress_delta=0.6,
                            state_regression_count=0,
                            state_transition={"no_progress": False},
                        proposal_metadata={
                            "hybrid_total_score": 4.0,
                            "hybrid_world_progress_score": 0.8,
                            "hybrid_world_risk_score": 0.2,
                            "hybrid_decoder_world_progress_score": 0.9,
                            "hybrid_decoder_world_risk_score": 0.1,
                            "hybrid_trusted_retrieval_alignment": 0.7,
                            "hybrid_graph_environment_alignment": 0.5,
                            "hybrid_transfer_novelty": 1.0,
                        },
                        latent_state_summary={
                            "learned_world_state": {
                                "source": "tolbert_hybrid_runtime",
                                "model_family": "tolbert_ssm_v1",
                                "progress_signal": 0.85,
                                "risk_signal": 0.1,
                            }
                        },
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
        )
    )

    assert metrics.world_feedback_summary["step_count"] == 1
    assert metrics.world_feedback_summary["progress_calibration_mae"] >= 0.0
    assert metrics.world_feedback_by_benchmark_family["workflow"]["step_count"] == 1
    assert metrics.world_feedback_by_difficulty["long_horizon"]["step_count"] == 1
    assert metrics.long_horizon_persistence_summary["productive_long_horizon_step_rate"] == 1.0
    assert metrics.transfer_alignment_summary["transfer_step_count"] == 1
    assert metrics.transfer_alignment_summary["graph_environment_alignment_mean"] == 0.5
    assert metrics.proposal_metrics_by_difficulty["long_horizon"]["task_count"] == 1
    step_payload = metrics.task_trajectories["world_feedback_task"]["steps"][0]
    assert step_payload["world_model_horizon"] == "long_horizon"
    assert step_payload["active_subgoal"] == "materialize expected artifact report.txt"
    assert step_payload["acting_role"] == "executor"
    step_feedback = metrics.task_trajectories["world_feedback_task"]["steps"][0]["world_feedback"]
    assert step_feedback["progress_signal"] == 0.9
    assert step_feedback["trusted_retrieval_alignment"] == 0.7
    assert step_feedback["graph_environment_alignment"] == 0.5
    assert step_feedback["transfer_novelty"] == 1.0
    assert step_feedback["observed_progress"] == 1.0


def test_run_eval_task_limit_prioritizes_requested_benchmark_families(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    metrics = run_eval(
        config=config,
        task_limit=3,
        priority_benchmark_families=["project", "repository", "integration"],
    )

    assert metrics.total == 3
    assert set(metrics.total_by_benchmark_family) == {"project", "repository", "integration"}


def test_run_eval_task_limit_allocates_more_budget_to_higher_weight_priority_families(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            tasks = []
            for family in ("project", "repository", "integration", "tooling"):
                for index in range(3):
                    tasks.append(
                        TaskSpec(
                            task_id=f"{family}_{index}",
                            prompt=family,
                            workspace_subdir=f"{family}_{index}",
                            expected_files=["out.txt"],
                            metadata={"benchmark_family": family},
                        )
                    )
            return tasks

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    unweighted = run_eval(
        config=config,
        task_limit=5,
        priority_benchmark_families=["project", "repository", "integration"],
    )
    weighted = run_eval(
        config=config,
        task_limit=5,
        priority_benchmark_families=["project", "repository", "integration"],
        priority_benchmark_family_weights={"project": 5.0, "repository": 3.0, "integration": 1.0},
    )

    assert unweighted.total == 5
    assert weighted.total == 5
    assert weighted.total_by_benchmark_family["project"] > unweighted.total_by_benchmark_family["project"]
    assert weighted.total_by_benchmark_family["integration"] < unweighted.total_by_benchmark_family["integration"]


def test_limit_tasks_for_compare_can_prefer_low_cost_tasks_within_family():
    costly_bounded = TaskSpec(
        task_id="z_costly_bounded",
        prompt="Create a costly bounded artifact after several steps.",
        workspace_subdir="z_costly_bounded",
        suggested_commands=["step1", "step2", "step3"],
        success_command="true",
        max_steps=9,
        metadata={"benchmark_family": "bounded", "difficulty": "bounded"},
    )
    cheap_seed = TaskSpec(
        task_id="a_cheap_seed",
        prompt="Write ok.txt.",
        workspace_subdir="a_cheap_seed",
        suggested_commands=["printf 'ok\\n' > ok.txt"],
        success_command="true",
        max_steps=5,
        metadata={"benchmark_family": "bounded", "difficulty": "seed"},
    )
    retrieval_tail = TaskSpec(
        task_id="m_retrieval_tail",
        prompt="Reuse a prior pattern.",
        workspace_subdir="m_retrieval_tail",
        suggested_commands=[],
        success_command="true",
        max_steps=5,
        metadata={
            "benchmark_family": "bounded",
            "difficulty": "retrieval",
            "requires_retrieval": True,
        },
    )

    selected = _limit_tasks_for_compare(
        [costly_bounded, retrieval_tail, cheap_seed],
        1,
        priority_families=["bounded"],
        prefer_low_cost_tasks=True,
    )

    assert [task.task_id for task in selected] == ["a_cheap_seed"]


def test_limit_tasks_for_compare_prefers_light_supervision_contract_tasks_before_retrieval_tails():
    retrieval_tail = TaskSpec(
        task_id="project_retrieval_tail",
        prompt="Reuse a prior project pattern.",
        workspace_subdir="project_retrieval_tail",
        success_command="true",
        expected_files=["out.txt"],
        metadata={
            "benchmark_family": "project",
            "difficulty": "retrieval",
            "requires_retrieval": True,
            "source_task": "deployment_manifest_task",
            "light_supervision_candidate": False,
        },
    )
    primary = TaskSpec(
        task_id="project_primary",
        prompt="Write a verifier-clean project artifact.",
        workspace_subdir="project_primary",
        success_command="test -f out.txt",
        expected_files=["out.txt"],
        metadata={
            "benchmark_family": "project",
            "difficulty": "bounded",
            "light_supervision_candidate": True,
        },
    )

    selected = _limit_tasks_for_compare(
        [retrieval_tail, primary],
        1,
        priority_families=["project"],
    )

    assert [task.task_id for task in selected] == ["project_primary"]


def test_limit_tasks_for_compare_prefers_executable_project_tasks_over_retrieval_companions():
    tasks = [
        TaskSpec(
            task_id="deployment_manifest_retrieval_task",
            prompt="retrieval companion",
            workspace_subdir="deployment_manifest_retrieval_task",
            max_steps=12,
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "requires_retrieval": True,
                "source_task": "deployment_manifest_task",
            },
        ),
        TaskSpec(
            task_id="deployment_manifest_task",
            prompt="deployment manifest",
            workspace_subdir="deployment_manifest_task",
            max_steps=12,
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        ),
        TaskSpec(
            task_id="project_release_cutover_retrieval_task",
            prompt="retrieval companion",
            workspace_subdir="project_release_cutover_retrieval_task",
            max_steps=18,
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "requires_retrieval": True,
                "source_task": "project_release_cutover_task",
            },
        ),
        TaskSpec(
            task_id="project_release_cutover_task",
            prompt="project release cutover",
            workspace_subdir="project_release_cutover_task",
            max_steps=18,
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        ),
        TaskSpec(
            task_id="release_packet_retrieval_task",
            prompt="retrieval companion",
            workspace_subdir="release_packet_retrieval_task",
            max_steps=12,
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "requires_retrieval": True,
                "source_task": "release_packet_task",
            },
        ),
        TaskSpec(
            task_id="release_packet_task",
            prompt="release packet",
            workspace_subdir="release_packet_task",
            max_steps=12,
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        ),
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        3,
        priority_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert {task.task_id for task in selected} == {
        "deployment_manifest_task",
        "release_packet_task",
        "project_release_cutover_task",
    }
    assert all(not bool(task.metadata.get("requires_retrieval", False)) for task in selected)


def test_limit_tasks_for_compare_prefers_executable_required_integration_tasks_over_retrieval_companions():
    tasks = [
        TaskSpec(
            task_id="incident_matrix_retrieval_task",
            prompt="retrieval companion",
            workspace_subdir="incident_matrix_retrieval_task",
            max_steps=5,
            metadata={
                "benchmark_family": "integration",
                "difficulty": "retrieval",
                "requires_retrieval": True,
                "source_task": "incident_matrix_task",
            },
        ),
        TaskSpec(
            task_id="incident_matrix_task",
            prompt="incident matrix",
            workspace_subdir="incident_matrix_task",
            max_steps=5,
            metadata={"benchmark_family": "integration", "difficulty": "multi_system"},
        ),
        TaskSpec(
            task_id="queue_failover_retrieval_task",
            prompt="retrieval companion",
            workspace_subdir="queue_failover_retrieval_task",
            max_steps=5,
            metadata={
                "benchmark_family": "integration",
                "difficulty": "multi_system",
                "requires_retrieval": True,
                "source_task": "queue_failover_task",
            },
        ),
        TaskSpec(
            task_id="queue_failover_task",
            prompt="queue failover",
            workspace_subdir="queue_failover_task",
            max_steps=5,
            metadata={"benchmark_family": "integration", "difficulty": "multi_system"},
        ),
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        2,
        priority_families=["integration"],
        prefer_low_cost_tasks=True,
        required_executable_families=["integration"],
    )

    assert {task.task_id for task in selected} == {
        "incident_matrix_task",
        "queue_failover_task",
    }
    assert all(not bool(task.metadata.get("requires_retrieval", False)) for task in selected)


def test_limit_tasks_for_compare_uses_five_low_cost_integration_primaries_before_retrieval_or_long_horizon():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "integration"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        5,
        priority_families=["integration"],
        prefer_low_cost_tasks=True,
        required_executable_families=["integration"],
    )

    assert [task.task_id for task in selected] == [
        "incident_matrix_task",
        "service_mesh_task",
        "queue_failover_task",
        "bridge_handoff_task",
        "replica_cutover_task",
    ]
    assert all(not bool(task.metadata.get("requires_retrieval", False)) for task in selected)
    assert "integration_failover_drill_task" not in {task.task_id for task in selected}


def test_limit_tasks_for_compare_uses_five_low_cost_repository_primaries_before_retrieval_or_long_horizon():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "repository"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        5,
        priority_families=["repository"],
        prefer_low_cost_tasks=True,
        required_executable_families=["repository"],
    )

    assert [task.task_id for task in selected] == [
        "repository_guardrail_sync_task",
        "repository_audit_packet_task",
        "service_release_task",
        "schema_alignment_task",
        "repo_sync_matrix_task",
    ]
    assert all(not bool(task.metadata.get("requires_retrieval", False)) for task in selected)
    assert "repository_migration_wave_task" not in {task.task_id for task in selected}


def test_limit_tasks_for_compare_uses_five_low_cost_repo_chore_primaries_before_retrieval():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "repo_chore"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        5,
        priority_families=["repo_chore"],
        prefer_low_cost_tasks=True,
        required_executable_families=["repo_chore"],
    )

    assert [task.task_id for task in selected] == [
        "repo_notice_review_task",
        "repo_packet_review_task",
        "repo_guardrail_review_task",
        "repo_cleanup_review_task",
        "repo_patch_review_task",
    ]
    assert all(not bool(task.metadata.get("requires_retrieval", False)) for task in selected)
    assert {
        "repo_cleanup_review_retrieval_task",
        "repo_patch_review_retrieval_task",
    }.isdisjoint({task.task_id for task in selected})


def test_limit_tasks_for_compare_uses_cleaner_repo_sandbox_primaries_before_known_unstable_roots():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "repo_sandbox"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        8,
        priority_families=["repo_sandbox"],
        prefer_low_cost_tasks=True,
        required_executable_families=["repo_sandbox"],
    )

    assert [task.task_id for task in selected] == [
        "git_release_train_worker_api_task",
        "git_release_train_worker_docs_task",
        "git_release_train_worker_ops_task",
        "git_release_train_conflict_worker_docs_task",
        "git_release_train_conflict_worker_api_task",
        "git_release_train_conflict_worker_ops_task",
        "git_parallel_merge_acceptance_task",
        "git_repo_test_repair_task",
    ]
    assert {
        "git_conflict_worker_status_task",
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
        "git_generated_conflict_resolution_task",
    }.isdisjoint({task.task_id for task in selected})


def test_limit_tasks_for_compare_widens_repo_sandbox_before_acceptance_tail():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "repo_sandbox"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        10,
        priority_families=["repo_sandbox"],
        prefer_low_cost_tasks=True,
        required_executable_families=["repo_sandbox"],
    )

    selected_ids = {task.task_id for task in selected}
    assert "git_conflict_worker_status_task" in selected_ids
    assert "git_release_train_acceptance_task" not in selected_ids
    assert "git_release_train_conflict_acceptance_task" not in selected_ids


def test_limit_tasks_for_compare_prefers_clean_repo_sandbox_retrieval_before_unstable_bridge_tail():
    tasks = [
        task
        for task in TaskBank().list()
        if str(task.metadata.get("benchmark_family", "")) == "repo_sandbox"
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        12,
        priority_families=["repo_sandbox"],
        prefer_low_cost_tasks=True,
        required_executable_families=["repo_sandbox"],
    )

    selected_ids = [task.task_id for task in selected]
    assert "git_repo_test_repair_retrieval_task" in selected_ids
    assert "git_repo_status_review_retrieval_task" in selected_ids
    assert "git_parallel_worker_api_task" not in selected_ids
    assert "git_parallel_worker_docs_task" not in selected_ids
    assert "git_generated_conflict_resolution_task" not in selected_ids
    assert "git_release_train_acceptance_task" not in selected_ids
    assert "git_release_train_conflict_acceptance_task" not in selected_ids


def test_limit_tasks_for_compare_keeps_prioritized_families_at_front_of_round_robin():
    tasks = [
        TaskSpec(
            task_id="bounded_a",
            prompt="bounded",
            workspace_subdir="bounded_a",
            success_command="true",
            metadata={"benchmark_family": "bounded"},
        ),
        TaskSpec(
            task_id="project_a",
            prompt="project",
            workspace_subdir="project_a",
            success_command="true",
            metadata={"benchmark_family": "project"},
        ),
        TaskSpec(
            task_id="repository_a",
            prompt="repository",
            workspace_subdir="repository_a",
            success_command="true",
            metadata={"benchmark_family": "repository"},
        ),
        TaskSpec(
            task_id="integration_a",
            prompt="integration",
            workspace_subdir="integration_a",
            success_command="true",
            metadata={"benchmark_family": "integration"},
        ),
    ]

    selected = _limit_tasks_for_compare(
        tasks,
        3,
        priority_families=["project", "repository"],
    )

    assert [task.metadata["benchmark_family"] for task in selected] == [
        "project",
        "repository",
        "bounded",
    ]


def test_run_eval_defaults_bounded_sampling_to_real_world_families(monkeypatch, tmp_path):
    seen = {}

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="hello_task",
                    prompt="hello",
                    workspace_subdir="hello_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "micro", "difficulty": "seed"},
                ),
                TaskSpec(
                    task_id="workflow_ready",
                    prompt="workflow",
                    workspace_subdir="workflow_ready",
                    expected_files=["out.txt"],
                    metadata={"benchmark_family": "workflow", "difficulty": "bounded"},
                ),
                TaskSpec(
                    task_id="repo_branch_fix",
                    prompt="repo sandbox",
                    workspace_subdir="repo_branch_fix",
                    expected_files=["reports/fix.txt"],
                    metadata={"benchmark_family": "repo_sandbox", "difficulty": "git_test_repair"},
                ),
                TaskSpec(
                    task_id="service_patch",
                    prompt="repository",
                    workspace_subdir="service_patch",
                    expected_files=["src/service.py"],
                    metadata={"benchmark_family": "repository", "difficulty": "cross_component"},
                ),
                TaskSpec(
                    task_id="tool_sync",
                    prompt="tooling",
                    workspace_subdir="tool_sync",
                    expected_files=["tool/report.txt"],
                    metadata={"benchmark_family": "tooling", "difficulty": "cross_tool"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            seen.setdefault("task_ids", []).append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
        ),
        task_limit=3,
    )

    assert metrics.total == 3
    assert seen["task_ids"] == ["repo_branch_fix", "service_patch", "tool_sync"]
    assert "micro" not in metrics.total_by_benchmark_family


def test_run_eval_writes_partial_progress_snapshot(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="workflow_ready",
                    prompt="workflow",
                    workspace_subdir="workflow_ready",
                    expected_files=["out.txt"],
                    metadata={"benchmark_family": "workflow", "memory_source": "episode"},
                ),
                TaskSpec(
                    task_id="repository_retry",
                    prompt="repository",
                    workspace_subdir="repository_retry",
                    expected_files=["out.txt"],
                    metadata={"benchmark_family": "repository", "memory_source": "skill"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            success = task.task_id == "workflow_ready"
            signals = [] if success else ["no_state_progress"]
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=success,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="code_execute",
                        content="true" if success else "false",
                        selected_skill_id=None,
                        selected_retrieval_span_id=(
                            "learning:success_skill:workflow_ready" if success else None
                        ),
                        command_result=None,
                        proposal_source="retrieval" if success else "",
                        decision_source=(
                            "trusted_retrieval_carryover_direct" if success else ""
                        ),
                        retrieval_influenced=success,
                        trust_retrieval=success,
                        path_confidence=0.2 if not success else 0.9,
                        verification={"passed": success, "reasons": []},
                        failure_signals=signals,
                        state_regression_count=0 if success else 1,
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success" if success else "step_limit",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    progress_path = tmp_path / "partial_progress.json"
    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
        ),
        progress_snapshot_path=progress_path,
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert metrics.total == 2
    assert payload["artifact_kind"] == "eval_partial_progress"
    assert payload["phase"] == "complete"
    assert payload["completed_primary_tasks"] == 2
    assert payload["primary_passed"] == 1
    assert payload["scheduled_task_order"] == ["repository_retry", "workflow_ready"]
    assert payload["scheduled_task_summaries"]["repository_retry"]["benchmark_family"] == "repository"
    assert payload["observed_benchmark_families"] == ["repository", "workflow"]
    assert payload["last_completed_task_id"] == "workflow_ready"
    assert payload["passed_by_benchmark_family"]["workflow"] == 1
    assert payload["completed_task_summaries"]["repository_retry"]["termination_reason"] == "step_limit"
    assert payload["retrieval_selected_steps"] == 1
    assert payload["retrieval_influenced_steps"] == 1
    assert payload["trusted_retrieval_steps"] == 1
    assert payload["selected_retrieval_span_ids"] == ["learning:success_skill:workflow_ready"]
    assert payload["last_selected_retrieval_span_id"] == "learning:success_skill:workflow_ready"
    assert payload["retrieval_influenced_task_ids"] == ["workflow_ready"]
    assert (
        payload["completed_task_summaries"]["workflow_ready"]["last_selected_retrieval_span_id"]
        == "learning:success_skill:workflow_ready"
    )
    assert payload["completed_task_summaries"]["workflow_ready"]["trusted_retrieval_carryover_steps"] == 1
    assert payload["completed_task_summaries"]["workflow_ready"]["trusted_retrieval_carryover_verified_steps"] == 1
    assert metrics.task_outcomes["workflow_ready"]["trusted_retrieval_carryover_verified_steps"] == 1


def test_run_eval_progress_snapshot_tracks_inflight_current_task(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="slow_task",
                    prompt="slow",
                    workspace_subdir="slow_task",
                    expected_files=["out.txt"],
                    metadata={"benchmark_family": "bounded", "memory_source": "none"},
                )
            ]

    started = threading.Event()
    release = threading.Event()

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True, progress_callback=None):
            del clean_workspace
            started.set()
            assert callable(progress_callback)
            progress_callback(
                {
                    "event": "step_start",
                    "step_index": 1,
                    "step_stage": "decision_pending",
                    "completed_steps": 0,
                }
            )
            release.wait(timeout=2.0)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="code_execute",
                        content="true",
                        selected_skill_id=None,
                        command_result=None,
                        proposal_source="",
                        retrieval_influenced=False,
                        trust_retrieval=False,
                        path_confidence=0.9,
                        verification={"passed": True, "reasons": []},
                        failure_signals=[],
                        state_regression_count=0,
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    progress_path = tmp_path / "partial_progress_live.json"
    metrics_holder: dict[str, object] = {}

    def _run_eval_in_thread():
        metrics_holder["metrics"] = run_eval(
            config=KernelConfig(
                provider="mock",
                use_tolbert_context=False,
                workspace_root=tmp_path / "workspace",
                trajectories_root=tmp_path / "trajectories",
            ),
            progress_snapshot_path=progress_path,
        )

    thread = threading.Thread(target=_run_eval_in_thread)
    thread.start()
    assert started.wait(timeout=1.0)

    snapshot = {}
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if progress_path.exists():
            snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                snapshot.get("current_task_id") == "slow_task"
                and float(snapshot.get("current_task_elapsed_seconds", 0.0) or 0.0) > 0.0
                and int(snapshot.get("current_task_step_index", 0) or 0) == 1
            ):
                break
        time.sleep(0.02)

    assert snapshot.get("current_task_id") == "slow_task"
    assert snapshot.get("current_task_phase") == "primary"
    assert snapshot.get("current_task_benchmark_family") == "bounded"
    assert snapshot.get("current_task_memory_source") == "none"
    assert float(snapshot.get("current_task_elapsed_seconds", 0.0) or 0.0) > 0.0
    assert snapshot.get("current_task_started_at")
    assert snapshot.get("current_task_step_index") == 1
    assert snapshot.get("current_task_step_stage") == "decision_pending"
    assert snapshot.get("current_task_completed_steps") == 0

    release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    final_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert final_payload["current_task_id"] == ""
    assert metrics_holder["metrics"].total == 1


def test_run_eval_progress_snapshot_caps_current_task_timeline(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="timeline_task",
                    prompt="timeline",
                    workspace_subdir="timeline_task",
                    expected_files=["out.txt"],
                    metadata={"benchmark_family": "bounded", "memory_source": "none"},
                )
            ]

    started = threading.Event()
    release = threading.Event()

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True, progress_callback=None):
            del clean_workspace
            assert callable(progress_callback)
            for index in range(1, 90):
                if index == 1:
                    started.set()
                progress_callback(
                    {
                        "event": "step_start",
                        "step_index": index,
                        "step_stage": "decision_pending",
                        "step_subphase": "tolbert_query",
                        "completed_steps": 0,
                        "step_elapsed_seconds": 0.001 * index,
                        "step_budget_seconds": 6.0,
                    }
                )
                time.sleep(0.001)
            release.wait(timeout=2.0)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="code_execute",
                        content="true",
                        selected_skill_id=None,
                        command_result=None,
                        proposal_source="",
                        retrieval_influenced=False,
                        trust_retrieval=False,
                        path_confidence=0.9,
                        verification={"passed": True, "reasons": []},
                        failure_signals=[],
                        state_regression_count=0,
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    progress_path = tmp_path / "partial_progress_timeline.json"
    metrics_holder: dict[str, object] = {}

    def _run_eval_in_thread():
        metrics_holder["metrics"] = run_eval(
            config=KernelConfig(
                provider="mock",
                use_tolbert_context=False,
                workspace_root=tmp_path / "workspace",
                trajectories_root=tmp_path / "trajectories",
            ),
            progress_snapshot_path=progress_path,
        )

    thread = threading.Thread(target=_run_eval_in_thread)
    thread.start()
    assert started.wait(timeout=1.0)

    snapshot = {}
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if progress_path.exists():
            snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
            if snapshot.get("current_task_id") == "timeline_task":
                timeline = snapshot.get("current_task_progress_timeline", [])
                if isinstance(timeline, list) and len(timeline) >= 8:
                    break
        time.sleep(0.02)

    timeline_snapshot = list(snapshot.get("current_task_progress_timeline", []))
    assert isinstance(timeline_snapshot, list)
    assert timeline_snapshot
    assert len(timeline_snapshot) <= 64

    try:
        assert snapshot.get("current_task_step_subphase") == "tolbert_query"
        assert timeline_snapshot[-1].get("step_index") == int(snapshot.get("current_task_step_index", 0) or 0)
    finally:
        release.set()
    thread.join(timeout=8.0)
    assert not thread.is_alive()
    assert metrics_holder["metrics"].total == 1


def test_run_eval_context_compile_heartbeat_keeps_step_elapsed_advancing(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="stalled_compile_task",
                    prompt="stalled compile",
                    workspace_subdir="stalled_compile_task",
                    metadata={"benchmark_family": "bounded", "memory_source": "none"},
                )
            ]

    started = threading.Event()

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True, progress_callback=None):
            del clean_workspace
            assert callable(progress_callback)
            progress_callback(
                {
                    "event": "context_compile_start",
                    "step_stage": "context_compile",
                    "step_subphase": "tolbert_query",
                    "step_index": 1,
                    "completed_steps": 0,
                    "step_elapsed_seconds": 0.0,
                    "step_budget_seconds": 3.0,
                }
            )
            started.set()
            time.sleep(0.7)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="respond",
                        content="complete",
                        selected_skill_id=None,
                        command_result=None,
                        proposal_source="",
                        retrieval_influenced=False,
                        trust_retrieval=False,
                        path_confidence=0.9,
                        verification={"passed": True, "reasons": []},
                        failure_signals=[],
                        state_regression_count=0,
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.1)

    progress_path = tmp_path / "partial_progress_stall.json"
    metrics_holder: dict[str, object] = {}

    def _run_eval_in_thread():
        metrics_holder["metrics"] = run_eval(
            config=KernelConfig(
                provider="mock",
                use_tolbert_context=True,
                workspace_root=tmp_path / "workspace",
                trajectories_root=tmp_path / "trajectories",
            ),
            progress_snapshot_path=progress_path,
        )

    thread = threading.Thread(target=_run_eval_in_thread)
    thread.start()
    assert started.wait(timeout=1.0)

    snapshot = {}
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if progress_path.exists():
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                payload.get("current_task_id") == "stalled_compile_task"
                and payload.get("current_task_step_stage") == "context_compile"
                and float(payload.get("current_task_step_elapsed_seconds", 0.0) or 0.0) > 0.2
            ):
                snapshot = payload
                break
        time.sleep(0.02)

    thread.join(timeout=8.0)
    assert not thread.is_alive()
    assert metrics_holder["metrics"].total == 1
    assert snapshot.get("current_task_step_stage") == "context_compile"
    assert float(snapshot.get("current_task_step_elapsed_seconds", 0.0) or 0.0) > 0.2


def test_run_eval_includes_discovered_tasks_from_failed_episode_memory(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_failed_episode(config)

    metrics = run_eval(config=config, include_discovered_tasks=True)

    assert metrics.total_by_memory_source["discovered_task"] == 1
    assert metrics.passed_by_memory_source["discovered_task"] == 1
    assert metrics.total_by_benchmark_family["discovered_task"] == 1
    assert metrics.total_by_memory_source["transition_pressure"] == 1
    assert metrics.passed_by_memory_source["transition_pressure"] == 1
    assert metrics.total_by_benchmark_family["transition_pressure"] == 1


def test_run_eval_skips_git_tasks_when_git_policy_disabled(monkeypatch, tmp_path):
    seen = {}

    class FakeTaskBank:
        def list(self):
            return [
                TaskSpec(
                    task_id="git_repo_status_review_task",
                    prompt="git repo sandbox",
                    workspace_subdir="git_repo_status_review_task",
                    expected_files=["reports/test_report.txt"],
                    metadata={"benchmark_family": "repo_sandbox", "requires_git": True},
                ),
                TaskSpec(
                    task_id="hello_task",
                    prompt="hello",
                    workspace_subdir="hello_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "micro", "capability": "file_write"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            seen.setdefault("task_ids", []).append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
            unattended_allow_git_commands=False,
        )
    )

    assert metrics.total == 1
    assert seen["task_ids"] == ["hello_task"]


def test_run_eval_skips_tasks_when_required_capability_is_missing(monkeypatch, tmp_path):
    seen = {}

    class FakeTaskBank:
        def list(self):
            return [
                TaskSpec(
                    task_id="github_issue_triage",
                    prompt="triage issue",
                    workspace_subdir="github_issue_triage",
                    expected_files=["reports/issue.txt"],
                    metadata={
                        "benchmark_family": "tooling",
                        "workflow_guard": {"required_capabilities": ["github_read"]},
                    },
                ),
                TaskSpec(
                    task_id="hello_task",
                    prompt="hello",
                    workspace_subdir="hello_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "micro", "capability": "file_write"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            seen.setdefault("task_ids", []).append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
            capability_modules_path=tmp_path / "config" / "capabilities.json",
        )
    )

    assert metrics.total == 1
    assert seen["task_ids"] == ["hello_task"]


def test_run_eval_skips_generated_path_tasks_when_generated_policy_disabled(monkeypatch, tmp_path):
    seen = {}

    class FakeTaskBank:
        def list(self):
            return [
                TaskSpec(
                    task_id="git_generated_conflict_resolution_task",
                    prompt="generated repo sandbox",
                    workspace_subdir="git_generated_conflict_resolution_task",
                    expected_files=["dist/status_bundle.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "requires_git": True,
                        "workflow_guard": {"touches_generated_paths": True},
                    },
                ),
                TaskSpec(
                    task_id="hello_task",
                    prompt="hello",
                    workspace_subdir="hello_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "micro", "capability": "file_write"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            seen.setdefault("task_ids", []).append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
            unattended_allow_git_commands=True,
            unattended_allow_generated_path_mutations=False,
        )
    )

    assert metrics.total == 1
    assert seen["task_ids"] == ["hello_task"]


def test_run_eval_uses_retained_operator_policy_for_generated_path_tasks(monkeypatch, tmp_path):
    seen = {}
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "unattended_allowed_benchmark_families": ["micro"],
                    "unattended_allow_git_commands": False,
                    "unattended_allow_http_requests": False,
                    "unattended_http_allowed_hosts": [],
                    "unattended_http_timeout_seconds": 10,
                    "unattended_http_max_body_bytes": 65536,
                    "unattended_allow_generated_path_mutations": True,
                    "unattended_generated_path_prefixes": ["dist"],
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeTaskBank:
        def list(self):
            return [
                TaskSpec(
                    task_id="generated_bundle_task",
                    prompt="generated repo sandbox",
                    workspace_subdir="generated_bundle_task",
                    expected_files=["dist/status_bundle.txt"],
                    metadata={
                        "benchmark_family": "micro",
                        "workflow_guard": {"touches_generated_paths": True},
                    },
                ),
                TaskSpec(
                    task_id="hello_task",
                    prompt="hello",
                    workspace_subdir="hello_task",
                    expected_files=["hello.txt"],
                    metadata={"benchmark_family": "micro", "capability": "file_write"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            seen.setdefault("task_ids", []).append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
            unattended_allow_generated_path_mutations=False,
            operator_policy_proposals_path=operator_policy_path,
        )
    )

    assert metrics.total == 2
    assert seen["task_ids"] == ["generated_bundle_task", "hello_task"]


def test_run_eval_closes_kernel(monkeypatch, tmp_path):
    closed = False

    class FakeTaskBank:
        def list(self):
            return [
                TaskSpec(
                    task_id="cleanup_task",
                    prompt="write a file",
                    workspace_subdir="cleanup_task",
                    success_command="true",
                )
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del task, clean_workspace
            return EpisodeRecord(
                task_id="cleanup_task",
                prompt="write a file",
                workspace=str(tmp_path / "workspace" / "cleanup_task"),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="done",
                        action="code_execute",
                        content="true",
                        selected_skill_id=None,
                        command_result=None,
                        verification={"passed": True, "reasons": ["ok"]},
                    )
                ],
                task_metadata={"benchmark_family": "micro", "capability": "file_write"},
                task_contract={"metadata": {"benchmark_family": "micro", "capability": "file_write"}},
                termination_reason="success",
            )

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    metrics = run_eval(config=config)

    assert metrics.total == 1
    assert closed is True


def test_eval_can_include_episode_and_skill_memory_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)
    _write_success_episode(config)

    metrics = run_eval(
        config=config,
        include_episode_memory=True,
        include_skill_memory=True,
    )

    assert metrics.passed == metrics.total
    assert metrics.total_by_benchmark_family["episode_memory"] == 1
    assert metrics.passed_by_benchmark_family["episode_memory"] == 1
    assert metrics.total_by_benchmark_family["skill_memory"] == 1
    assert metrics.passed_by_benchmark_family["skill_memory"] == 1
    assert metrics.total_by_memory_source["episode"] == 1
    assert metrics.passed_by_memory_source["episode"] == 1
    assert metrics.total_by_memory_source["skill"] == 1
    assert metrics.passed_by_memory_source["skill"] == 1
    assert metrics.total_by_origin_benchmark_family["micro"] == 2
    assert metrics.passed_by_origin_benchmark_family["micro"] == 2


def test_eval_can_include_skill_transfer_and_operator_memory(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
    )
    _write_hello_skill(config)
    _write_operator_classes(config)

    operator_metrics = run_eval(config=config, include_operator_memory=True)
    skill_transfer_metrics = run_eval(config=config, include_skill_transfer=True)

    assert operator_metrics.total_by_memory_source["operator"] == 1
    assert operator_metrics.passed_by_memory_source["operator"] == 1
    assert skill_transfer_metrics.total_by_memory_source["skill_transfer"] == 1


def test_eval_can_include_verifier_memory_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_success_episode(config)

    metrics = run_eval(
        config=config,
        include_verifier_memory=True,
    )

    assert metrics.passed == metrics.total
    assert metrics.total_by_benchmark_family["verifier_memory"] == 1
    assert metrics.passed_by_benchmark_family["verifier_memory"] == 1
    assert metrics.total_by_memory_source["verifier"] == 1
    assert metrics.passed_by_memory_source["verifier"] == 1


def test_eval_can_include_tool_memory_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
    )
    config.tool_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    config.tool_candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "mkdir -p gateway",
                                "printf 'routes synced\\n' > gateway/routes.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "Prepare integration workspace.",
                            "workspace_subdir": "service_mesh_task",
                            "setup_commands": [],
                            "success_command": "test -f gateway/routes.txt && grep -q '^routes synced$' gateway/routes.txt",
                            "suggested_commands": [],
                            "expected_files": ["gateway/routes.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"gateway/routes.txt": "routes synced\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "integration", "capability": "integration_environment", "difficulty": "multi_system"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = run_eval(config=config, include_tool_memory=True)

    assert metrics.passed == metrics.total
    assert metrics.total_by_benchmark_family["tool_memory"] == 1
    assert metrics.passed_by_benchmark_family["tool_memory"] == 1
    assert metrics.total_by_memory_source["tool"] == 1
    assert metrics.passed_by_memory_source["tool"] == 1


def test_eval_can_include_benchmark_candidate_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
    )
    config.benchmark_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark_candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "benchmark_candidate_set",
                "lifecycle_state": "proposed",
                "proposals": [
                    {
                        "proposal_id": "benchmark:hello_task:failure_cluster",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "kind": "failure_cluster",
                        "prompt": "Create hello.txt containing the string hello agent kernel.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = run_eval(config=config, include_benchmark_candidates=True)

    assert metrics.total_by_benchmark_family["benchmark_candidate"] == 1
    assert metrics.passed_by_benchmark_family["benchmark_candidate"] == 1


def test_eval_can_include_verifier_candidate_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
    )
    config.verifier_contracts_path.parent.mkdir(parents=True, exist_ok=True)
    config.verifier_contracts_path.write_text(
        json.dumps(
            {
                "artifact_kind": "verifier_candidate_set",
                "lifecycle_state": "proposed",
                "proposals": [
                    {
                        "proposal_id": "verifier:hello_task:strict",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "forbidden_output_substrings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = run_eval(config=config, include_verifier_candidates=True)

    assert metrics.total_by_benchmark_family["verifier_candidate"] == 1
    assert metrics.passed_by_benchmark_family["verifier_candidate"] == 1


def test_eval_can_disable_skill_usage(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_skills=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    metrics = run_eval(config=config)

    assert metrics.passed == metrics.total
    assert metrics.skill_selected_steps == 0
    assert metrics.episodes_with_skill_use == 0
    assert metrics.average_available_skills == 0.0


def test_compare_skill_modes_reports_deltas(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    comparison = compare_skill_modes(config=config)

    assert comparison.with_skills.skill_selected_steps >= 1
    assert comparison.without_skills.skill_selected_steps == 0
    assert comparison.with_skills.average_available_skills > comparison.without_skills.average_available_skills
    assert comparison.average_steps_delta <= 0.0
    assert "file_write" in comparison.capability_pass_rate_delta
    assert "workflow" in comparison.benchmark_family_pass_rate_delta


def test_compare_abstraction_transfer_modes_reports_transfer_delta(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
    )
    _write_hello_skill(config)
    _write_operator_classes(config)

    comparison = compare_abstraction_transfer_modes(config=config)

    assert comparison.operator_metrics.total_by_memory_source["operator"] == 1
    assert comparison.raw_skill_metrics.total_by_memory_source["skill_transfer"] == 1


def test_compare_tolbert_modes_reports_deltas(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    comparison = compare_tolbert_modes(config=config)

    assert comparison.with_tolbert.total == comparison.without_tolbert.total
    assert "file_write" in comparison.capability_pass_rate_delta
    assert "workflow" in comparison.benchmark_family_pass_rate_delta


def test_compare_tolbert_feature_modes_reports_all_modes(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    comparison = compare_tolbert_feature_modes(config=config)

    assert set(comparison.mode_metrics) == {
        "path_only",
        "retrieval_only",
        "deterministic_command",
        "skill_ranking",
        "full",
    }
    for metrics in comparison.mode_metrics.values():
        assert metrics.total >= 1


def test_compare_tolbert_feature_modes_limits_real_provider_task_count(tmp_path, monkeypatch):
    config = KernelConfig(
        provider="ollama",
        use_tolbert_context=True,
        compare_feature_max_tasks=7,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)
    _write_success_episode(config)
    task_limits = []

    def fake_run_eval(*, config, include_discovered_tasks=False, include_episode_memory=False, include_skill_memory=False, include_skill_transfer=False, include_operator_memory=False, include_tool_memory=False, include_verifier_memory=False, include_benchmark_candidates=False, include_verifier_candidates=False, include_generated=False, include_failure_generated=False, task_limit=None, progress_label=None):
        del config, include_discovered_tasks, include_episode_memory, include_skill_memory, include_skill_transfer, include_operator_memory, include_tool_memory, include_verifier_memory, include_benchmark_candidates, include_verifier_candidates, include_generated, include_failure_generated, progress_label
        task_limits.append(task_limit)
        return EvalMetrics(total=task_limit or 0, passed=task_limit or 0)

    monkeypatch.setattr("evals.harness.run_eval", fake_run_eval)

    comparison = compare_tolbert_feature_modes(
        config=config,
        include_episode_memory=True,
        include_skill_memory=True,
        include_verifier_memory=True,
    )

    assert task_limits == [7, 7, 7, 7, 7]
    for metrics in comparison.mode_metrics.values():
        assert metrics.total <= 7

def test_compare_modes_isolate_runtime_roots(tmp_path):
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    skill_comparison = compare_skill_modes(config=config)
    tolbert_comparison = compare_tolbert_modes(config=config)

    assert skill_comparison.with_skills.total == skill_comparison.without_skills.total
    assert tolbert_comparison.with_tolbert.total == tolbert_comparison.without_tolbert.total
    assert (tmp_path / "workspace" / "with_skills").exists()
    assert (tmp_path / "workspace" / "without_skills").exists()
    assert (tmp_path / "workspace" / "with_tolbert").exists()
    assert (tmp_path / "workspace" / "without_tolbert").exists()


def test_compare_modes_scope_mutable_artifact_paths(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        world_model_proposals_path=tmp_path / "world_model" / "world_model_proposals.json",
        trust_proposals_path=tmp_path / "trust" / "trust_proposals.json",
        recovery_proposals_path=tmp_path / "recovery" / "recovery_proposals.json",
        delegation_proposals_path=tmp_path / "delegation" / "delegation_proposals.json",
        operator_policy_proposals_path=tmp_path / "operator_policy" / "operator_policy_proposals.json",
        transition_model_proposals_path=tmp_path / "transition_model" / "transition_model_proposals.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        capability_modules_path=tmp_path / "config" / "capabilities.json",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.retrieval_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.prompt_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.world_model_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.trust_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.recovery_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.delegation_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.operator_policy_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.transition_model_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.curriculum_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.capability_modules_path.parent.mkdir(parents=True, exist_ok=True)
    config.delegated_job_queue_path.parent.mkdir(parents=True, exist_ok=True)
    config.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.unattended_workspace_snapshot_root.mkdir(parents=True, exist_ok=True)
    config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.retrieval_proposals_path.write_text("{}", encoding="utf-8")
    config.retrieval_asset_bundle_path.write_text("{}", encoding="utf-8")
    config.prompt_proposals_path.write_text("{}", encoding="utf-8")
    config.world_model_proposals_path.write_text("{}", encoding="utf-8")
    config.trust_proposals_path.write_text("{}", encoding="utf-8")
    config.recovery_proposals_path.write_text("{}", encoding="utf-8")
    config.delegation_proposals_path.write_text("{}", encoding="utf-8")
    config.operator_policy_proposals_path.write_text("{}", encoding="utf-8")
    config.transition_model_proposals_path.write_text("{}", encoding="utf-8")
    config.curriculum_proposals_path.write_text("{}", encoding="utf-8")
    config.capability_modules_path.write_text(json.dumps({"modules": [{"module_id": "github", "enabled": True}]}), encoding="utf-8")
    config.delegated_job_queue_path.write_text(json.dumps({"jobs": [{"job_id": "job-1"}]}), encoding="utf-8")
    config.delegated_job_runtime_state_path.write_text(json.dumps({"active": [{"job_id": "job-1"}]}), encoding="utf-8")
    (config.run_checkpoints_dir / "resume.json").write_text(json.dumps({"checkpoint": True}), encoding="utf-8")
    (config.unattended_workspace_snapshot_root / "hello_task").mkdir(parents=True, exist_ok=True)
    (config.unattended_workspace_snapshot_root / "hello_task" / "state.txt").write_text("snapshot", encoding="utf-8")
    config.unattended_trust_ledger_path.write_text(json.dumps({"ledger_kind": "unattended_trust_ledger"}), encoding="utf-8")
    seen_paths = []

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        seen_paths.append(
            (
                config.retrieval_proposals_path,
                config.retrieval_asset_bundle_path,
                config.prompt_proposals_path,
                config.world_model_proposals_path,
                config.trust_proposals_path,
                config.recovery_proposals_path,
                config.delegation_proposals_path,
                config.operator_policy_proposals_path,
                config.transition_model_proposals_path,
                config.curriculum_proposals_path,
                config.improvement_cycles_path,
                config.capability_modules_path,
                config.delegated_job_queue_path,
                config.delegated_job_runtime_state_path,
                config.run_checkpoints_dir,
                config.unattended_workspace_snapshot_root,
                config.unattended_trust_ledger_path,
            )
        )
        assert json.loads(config.capability_modules_path.read_text(encoding="utf-8"))["modules"][0]["module_id"] == "github"
        assert json.loads(config.delegated_job_queue_path.read_text(encoding="utf-8"))["jobs"][0]["job_id"] == "job-1"
        assert json.loads(config.delegated_job_runtime_state_path.read_text(encoding="utf-8"))["active"][0]["job_id"] == "job-1"
        assert json.loads(config.world_model_proposals_path.read_text(encoding="utf-8")) == {}
        assert json.loads(config.trust_proposals_path.read_text(encoding="utf-8")) == {}
        assert json.loads(config.recovery_proposals_path.read_text(encoding="utf-8")) == {}
        assert json.loads(config.delegation_proposals_path.read_text(encoding="utf-8")) == {}
        assert json.loads(config.operator_policy_proposals_path.read_text(encoding="utf-8")) == {}
        assert json.loads(config.transition_model_proposals_path.read_text(encoding="utf-8")) == {}
        assert (config.run_checkpoints_dir / "resume.json").exists()
        assert (config.unattended_workspace_snapshot_root / "hello_task" / "state.txt").read_text(encoding="utf-8") == "snapshot"
        assert json.loads(config.unattended_trust_ledger_path.read_text(encoding="utf-8"))["ledger_kind"] == "unattended_trust_ledger"
        return EvalMetrics(total=1, passed=1)

    monkeypatch.setattr("evals.harness.run_eval", fake_run_eval)

    compare_skill_modes(config=config)

    assert len(seen_paths) == 2
    assert seen_paths[0] != seen_paths[1]
    for path_group in seen_paths:
        for path in path_group:
            assert str(tmp_path) in str(path)
            assert path.parent.exists()
    with_skills_snapshot_root = seen_paths[0][15]
    without_skills_snapshot_root = seen_paths[1][15]
    assert not (with_skills_snapshot_root / "with_skills").exists()
    assert not (without_skills_snapshot_root / "without_skills").exists()


def test_scoped_eval_config_skips_existing_scoped_tolbert_dataset_dirs(tmp_path):
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    config.tolbert_supervised_datasets_dir.mkdir(parents=True, exist_ok=True)
    (config.tolbert_supervised_datasets_dir / "dataset.jsonl").write_text("{}", encoding="utf-8")
    (config.tolbert_supervised_datasets_dir / "with_skills").mkdir(parents=True, exist_ok=True)
    (config.tolbert_supervised_datasets_dir / "with_skills" / "stale.txt").write_text("stale", encoding="utf-8")
    (config.tolbert_supervised_datasets_dir / "tolbert_full").mkdir(parents=True, exist_ok=True)
    (config.tolbert_supervised_datasets_dir / "tolbert_full" / "stale.txt").write_text("stale", encoding="utf-8")

    scoped = scoped_eval_config(config, "with_skills")

    assert scoped.tolbert_supervised_datasets_dir == config.tolbert_supervised_datasets_dir
    assert (scoped.tolbert_supervised_datasets_dir / "dataset.jsonl").exists()
    assert (scoped.tolbert_supervised_datasets_dir / "with_skills" / "stale.txt").exists()
    assert (scoped.tolbert_supervised_datasets_dir / "tolbert_full" / "stale.txt").exists()


def test_scoped_eval_config_skips_nested_scoped_snapshot_dirs(tmp_path):
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
    )
    run_root = config.unattended_workspace_snapshot_root / "legacy_run"
    (run_root / "state.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "state.txt").write_text("snapshot", encoding="utf-8")
    (run_root / "generated_success" / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "generated_success" / "stale.txt").write_text("stale", encoding="utf-8")
    (run_root / "generated_failure_seed" / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "generated_failure_seed" / "stale.txt").write_text("stale", encoding="utf-8")
    nested_run_root = config.unattended_workspace_snapshot_root / "archive" / "legacy_nested_run"
    (nested_run_root / "kept.txt").parent.mkdir(parents=True, exist_ok=True)
    (nested_run_root / "kept.txt").write_text("nested", encoding="utf-8")
    (nested_run_root / "generated_success" / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (nested_run_root / "generated_success" / "stale.txt").write_text("stale", encoding="utf-8")

    scoped = scoped_eval_config(config, "generated_success")

    copied_run_root = scoped.unattended_workspace_snapshot_root / "legacy_run"
    copied_nested_run_root = scoped.unattended_workspace_snapshot_root / "archive" / "legacy_nested_run"
    assert (copied_run_root / "state.txt").read_text(encoding="utf-8") == "snapshot"
    assert not (copied_run_root / "generated_success").exists()
    assert not (copied_run_root / "generated_failure_seed").exists()
    assert (copied_nested_run_root / "kept.txt").read_text(encoding="utf-8") == "nested"
    assert not (copied_nested_run_root / "generated_success").exists()


def test_compare_skill_modes_cleans_scoped_checkpoints_and_snapshots(tmp_path, monkeypatch):
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
    )

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        config.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config.unattended_workspace_snapshot_root.mkdir(parents=True, exist_ok=True)
        (config.run_checkpoints_dir / "resume.json").write_text("{}", encoding="utf-8")
        (config.unattended_workspace_snapshot_root / "hello_task").mkdir(parents=True, exist_ok=True)
        return EvalMetrics(total=1, passed=1)

    monkeypatch.setattr("evals.harness.run_eval", fake_run_eval)

    compare_skill_modes(config=config)

    assert not (tmp_path / "checkpoints" / "with_skills").exists()
    assert not (tmp_path / "checkpoints" / "without_skills").exists()
    assert not (tmp_path / "recovery" / "workspaces" / "with_skills").exists()
    assert not (tmp_path / "recovery" / "workspaces" / "without_skills").exists()


def test_eval_can_include_generated_curriculum_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    metrics = run_eval(config=config, include_generated=True)

    assert metrics.passed == metrics.total
    assert metrics.generated_total == metrics.total
    assert metrics.generated_passed == metrics.generated_total
    assert metrics.generated_by_kind["adjacent_success"] == metrics.generated_total
    assert metrics.generated_passed_by_kind["adjacent_success"] == metrics.generated_total


def test_eval_can_include_failure_generated_curriculum_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    metrics = run_eval(
        config=config,
        include_generated=True,
        include_failure_generated=True,
    )

    assert metrics.passed == metrics.total
    assert metrics.generated_total == metrics.total * 2
    assert metrics.generated_passed == metrics.generated_total
    assert metrics.generated_by_kind["adjacent_success"] == metrics.total
    assert metrics.generated_by_kind["failure_recovery"] == metrics.total
    assert metrics.generated_passed_by_kind["adjacent_success"] == metrics.total
    assert metrics.generated_passed_by_kind["failure_recovery"] == metrics.total


def test_run_eval_surfaces_contract_clean_failure_recovery_eval_slice(monkeypatch, tmp_path):
    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="repository_primary",
                    prompt="repository primary",
                    workspace_subdir="repository_primary",
                    success_command="test -f out.txt",
                    expected_files=["out.txt"],
                    metadata={
                        "benchmark_family": "repository",
                        "difficulty": "long_horizon",
                        "light_supervision_candidate": True,
                    },
                )
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config
            self._forced_failure = policy is not None

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            generated_failure = str(task.metadata.get("curriculum_kind", "")).strip() == "failure_recovery"
            success = generated_failure or not self._forced_failure
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=success,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="code_execute",
                        content="true" if success else "false",
                        selected_skill_id=None,
                        command_result=None,
                        verification={"passed": success, "reasons": []},
                        failure_signals=[] if success else ["no_state_progress"],
                        state_regression_count=0 if success else 1,
                        world_model_horizon="long_horizon",
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success" if success else "repeated_failed_action",
            )

        def close(self):
            return None

    def fake_schedule(self, episodes, *, curriculum_kind):
        del self, curriculum_kind
        return list(episodes[:1])

    def fake_generate(self, episode):
        del self, episode
        return TaskSpec(
            task_id="repository_contract_clean_recovery",
            prompt="repair repository state",
            workspace_subdir="repository_contract_clean_recovery",
            success_command="test -f out.txt",
            expected_files=["out.txt"],
            max_steps=14,
            metadata={
                "benchmark_family": "repository",
                "difficulty": "long_horizon",
                "curriculum_kind": "failure_recovery",
                "contract_clean_failure_recovery_origin": True,
                "contract_clean_failure_recovery_origin_family": "repository",
                "contract_clean_failure_recovery_step_floor": 12,
            },
        )

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness.CurriculumEngine.schedule_generated_seed_episodes", fake_schedule)
    monkeypatch.setattr("evals.harness.CurriculumEngine.generate_followup_task", fake_generate)

    progress_path = tmp_path / "partial_progress_recovery.json"
    metrics = run_eval(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
        ),
        include_generated=False,
        include_failure_generated=True,
        progress_snapshot_path=progress_path,
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert metrics.generated_passed == 1
    assert metrics.generated_by_kind["failure_recovery"] == 1
    assert metrics.contract_clean_failure_recovery_summary == {
        "task_count": 1,
        "success_count": 1,
        "clean_success_count": 1,
        "long_horizon_task_count": 1,
        "long_horizon_success_count": 1,
        "total_steps": 1,
        "max_step_floor": 12,
        "pass_rate": 1.0,
        "clean_success_rate": 1.0,
        "long_horizon_pass_rate": 1.0,
        "average_steps": 1.0,
        "average_step_floor": 12.0,
        "distinct_origin_benchmark_families": 1,
    }
    assert metrics.contract_clean_failure_recovery_by_origin_benchmark_family["repository"]["task_count"] == 1
    assert metrics.contract_clean_failure_recovery_by_origin_benchmark_family["repository"]["long_horizon_pass_rate"] == 1.0
    assert payload["contract_clean_failure_recovery_summary"]["task_count"] == 1
    assert payload["contract_clean_failure_recovery_summary"]["long_horizon_pass_rate"] == 1.0
    assert payload["contract_clean_failure_recovery_by_origin_benchmark_family"]["repository"]["average_step_floor"] == 12.0


def test_eval_isolates_generated_failure_episodes_from_primary_trajectory_store(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    metrics = run_eval(
        config=config,
        include_generated=True,
        include_failure_generated=True,
    )
    primary_episode = json.loads(config.trajectories_root.joinpath("hello_task.json").read_text(encoding="utf-8"))

    assert metrics.generated_total == metrics.total * 2
    assert primary_episode["termination_reason"] == "success"
    assert primary_episode["steps"][0]["content"] != "false"
    assert (tmp_path / "trajectories" / "generated_failure_seed").exists()
    assert (tmp_path / "trajectories" / "generated_failure").exists()


def test_eval_uses_curriculum_seed_scheduler_for_generated_tasks(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)

    def fake_schedule(self, episodes, *, curriculum_kind):
        del self, curriculum_kind
        return list(episodes[:1])

    monkeypatch.setattr("evals.harness.CurriculumEngine.schedule_generated_seed_episodes", fake_schedule)

    metrics = run_eval(
        config=config,
        include_generated=True,
        include_failure_generated=True,
    )

    assert metrics.generated_total == 2
    assert metrics.generated_by_kind["adjacent_success"] == 1
    assert metrics.generated_by_kind["failure_recovery"] == 1


def test_eval_can_reuse_primary_seed_documents_for_generated_success_without_primary_rerun(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    config.trajectories_root.joinpath("repo_success.json").write_text(
        json.dumps(
            {
                "task_id": "repo_success",
                "prompt": "Create repo handoff.",
                "workspace": str(tmp_path / "workspace" / "repo_success"),
                "success": True,
                "task_metadata": {"benchmark_family": "repository"},
                "task_contract": {"metadata": {"benchmark_family": "repository"}},
                "summary": {"executed_commands": ["mkdir -p repo && printf 'ok\\n' > repo/status.txt"]},
                "termination_reason": "success",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return []

    class FakeKernel:
        run_count = 0

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            FakeKernel.run_count += 1
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(config.trajectories_root),
    )

    assert FakeKernel.run_count == 1
    assert metrics.total == 0
    assert metrics.generated_total == 1
    assert metrics.generated_by_kind["adjacent_success"] == 1


def test_eval_generated_success_followup_skips_primary_bootstrap_components(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    config.trajectories_root.joinpath("repo_success.json").write_text(
        json.dumps(
            {
                "task_id": "repo_success",
                "prompt": "Create repo handoff.",
                "workspace": str(tmp_path / "workspace" / "repo_success"),
                "success": True,
                "task_metadata": {"benchmark_family": "repository"},
                "task_contract": {"metadata": {"benchmark_family": "repository"}},
                "termination_reason": "success",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    class ExplodingTaskBank:
        def __init__(self, config=None):
            raise AssertionError(f"generated-only followup should not initialize TaskBank: {config}")

    class FakeKernel:
        init_configs = []

        def __init__(self, config=None, policy=None):
            del policy
            FakeKernel.init_configs.append(config)

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", ExplodingTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(config.trajectories_root),
    )

    assert metrics.total == 0
    assert metrics.generated_total == 1
    assert len(FakeKernel.init_configs) == 1
    assert FakeKernel.init_configs[0].trajectories_root.name == "generated_success"


def test_eval_generated_success_can_skip_historical_seed_fallback(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)

    def explode_seed_load(_root):
        raise AssertionError("historical generated-success seed fallback should be disabled")

    monkeypatch.setattr("evals.harness._load_generated_success_seed_episodes", explode_seed_load)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(config.trajectories_root),
        allow_generated_success_seed_fallback=False,
    )

    assert metrics.total == 0
    assert metrics.generated_total == 0
    assert metrics.generated_passed == 0


def test_eval_generated_success_can_cap_schedule_materialization(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)

    seed_episodes = [
        EpisodeRecord(
            task_id="seed_a",
            prompt="a",
            workspace=str(tmp_path / "workspace" / "seed_a"),
            success=True,
            steps=[],
            task_metadata={"benchmark_family": "repository"},
            task_contract={"metadata": {"benchmark_family": "repository"}},
            termination_reason="success",
        ),
        EpisodeRecord(
            task_id="seed_b",
            prompt="b",
            workspace=str(tmp_path / "workspace" / "seed_b"),
            success=True,
            steps=[],
            task_metadata={"benchmark_family": "project"},
            task_contract={"metadata": {"benchmark_family": "project"}},
            termination_reason="success",
        ),
    ]

    class FakeEngine:
        generated_from: list[str] = []

        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, episodes, *, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(episodes)

        def generate_followup_task(self, episode):
            FakeEngine.generated_from.append(episode.task_id)
            return TaskSpec(
                task_id=f"{episode.task_id}_adjacent",
                prompt=f"follow up {episode.task_id}",
                workspace_subdir=f"{episode.task_id}_adjacent",
                metadata=dict(episode.task_metadata),
            )

    class FakeKernel:
        run_task_ids: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            FakeKernel.run_task_ids.append(task.task_id)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr(
        "evals.harness._load_generated_success_seed_episodes",
        lambda root, workspace_root=None: list(seed_episodes),
    )

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        allow_generated_success_seed_fallback=False,
        max_generated_success_schedule_tasks=1,
        generated_success_seed_documents_path=str(config.trajectories_root),
    )

    assert metrics.total == 0
    assert metrics.generated_total == 0
    assert FakeEngine.generated_from == []
    assert FakeKernel.run_task_ids == []

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(config.trajectories_root),
        max_generated_success_schedule_tasks=1,
    )

    assert metrics.generated_total == 1
    assert FakeEngine.generated_from == ["seed_a"]
    assert FakeKernel.run_task_ids == ["seed_a_adjacent"]


def test_eval_generated_success_seed_fallback_can_filter_to_workspace_root(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_a",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)

    seed_docs = [
        {
            "task_id": "scope_repo_success",
            "prompt": "repo",
            "workspace": str(config.workspace_root / "scope_repo_success"),
            "success": True,
            "task_metadata": {"benchmark_family": "repository"},
            "task_contract": {"metadata": {"benchmark_family": "repository"}},
            "termination_reason": "success",
        },
        {
            "task_id": "other_scope_success",
            "prompt": "other",
            "workspace": str(tmp_path / "workspace" / "scope_b" / "other_scope_success"),
            "success": True,
            "task_metadata": {"benchmark_family": "project"},
            "task_contract": {"metadata": {"benchmark_family": "project"}},
            "termination_reason": "success",
        },
    ]

    class FakeEngine:
        generated_from: list[str] = []

        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, episodes, *, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(episodes)

        def generate_followup_task(self, episode):
            FakeEngine.generated_from.append(episode.task_id)
            return TaskSpec(
                task_id=f"{episode.task_id}_adjacent",
                prompt="follow up",
                workspace_subdir=f"{episode.task_id}_adjacent",
                metadata=dict(episode.task_metadata),
            )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness.iter_episode_documents", lambda root: list(seed_docs))

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(config.trajectories_root),
        generated_success_seed_workspace_root=str(config.workspace_root),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    assert metrics.generated_total == 1
    assert FakeEngine.generated_from == ["scope_repo_success"]


def test_eval_generated_success_seed_bundle_file_can_filter_to_workspace_root(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_a",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps({"episodes": []}),
        encoding="utf-8",
    )
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "scope_repo_success",
                        "prompt": "repo",
                        "workspace": str(config.workspace_root / "scope_repo_success"),
                        "success": True,
                        "task_metadata": {"benchmark_family": "repository"},
                        "task_contract": {"metadata": {"benchmark_family": "repository"}},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "other_scope_success",
                        "prompt": "other",
                        "workspace": str(tmp_path / "workspace" / "scope_b" / "other_scope_success"),
                        "success": True,
                        "task_metadata": {"benchmark_family": "project"},
                        "task_contract": {"metadata": {"benchmark_family": "project"}},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeEngine:
        generated_from: list[str] = []

        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, episodes, *, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(episodes)

        def generate_followup_task(self, episode):
            FakeEngine.generated_from.append(episode.task_id)
            return TaskSpec(
                task_id=f"{episode.task_id}_adjacent",
                prompt="follow up",
                workspace_subdir=f"{episode.task_id}_adjacent",
                metadata=dict(episode.task_metadata),
            )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_workspace_root=str(config.workspace_root),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    assert metrics.generated_total == 1
    assert FakeEngine.generated_from == ["scope_repo_success"]


def test_eval_generated_success_seed_bundle_file_can_filter_to_relative_workspace_root(monkeypatch, tmp_path):
    workspace_root = tmp_path / "workspace" / "scope_a"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=workspace_root,
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "scope_repo_success",
                        "prompt": "repo",
                        "workspace": str(workspace_root / "scope_repo_success"),
                        "success": True,
                        "task_metadata": {"benchmark_family": "repository"},
                        "task_contract": {"metadata": {"benchmark_family": "repository"}},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "other_scope_success",
                        "prompt": "other",
                        "workspace": str(tmp_path / "workspace" / "scope_b" / "other_scope_success"),
                        "success": True,
                        "task_metadata": {"benchmark_family": "project"},
                        "task_contract": {"metadata": {"benchmark_family": "project"}},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeEngine:
        generated_from: list[str] = []

        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, episodes, *, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(episodes)

        def generate_followup_task(self, episode):
            FakeEngine.generated_from.append(episode.task_id)
            return TaskSpec(
                task_id=f"{episode.task_id}_adjacent",
                prompt="follow up",
                workspace_subdir=f"{episode.task_id}_adjacent",
                metadata=dict(episode.task_metadata),
            )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    relative_workspace_root = os.path.relpath(workspace_root, Path.cwd())
    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_workspace_root=str(relative_workspace_root),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    assert metrics.generated_total == 1
    assert FakeEngine.generated_from == ["scope_repo_success"]


def test_eval_generated_success_seed_bundle_file_falls_back_when_workspace_filter_misses(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_a",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "scope_repo_success",
                        "prompt": "repo",
                        "workspace": str(config.workspace_root / "scope_repo_success"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "project",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "shared_repo_integrator",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "project",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "shared_repo_integrator",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeEngine:
        generated_from: list[str] = []

        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, episodes, *, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(episodes)

        def generate_followup_task(self, episode):
            FakeEngine.generated_from.append(episode.task_id)
            return TaskSpec(
                task_id=f"{episode.task_id}_adjacent",
                prompt="follow up",
                workspace_subdir=f"{episode.task_id}_adjacent",
                metadata=dict(episode.task_metadata),
            )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_workspace_root=str(tmp_path / "workspace" / "scope_other"),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    assert metrics.generated_total == 1
    assert FakeEngine.generated_from == ["scope_repo_success"]


def test_eval_generated_success_keeps_long_horizon_surface_diversity(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_a",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "worker_success",
                        "prompt": "worker",
                        "workspace": str(config.workspace_root / "worker_success"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "project",
                            "origin_benchmark_family": "repo_sandbox",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "shared_repo_synthetic_worker",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "project",
                                "origin_benchmark_family": "repo_sandbox",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "shared_repo_synthetic_worker",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "integrator_success",
                        "prompt": "integrator",
                        "workspace": str(config.workspace_root / "integrator_success"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "project",
                            "origin_benchmark_family": "repo_sandbox",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "shared_repo_integrator",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "project",
                                "origin_benchmark_family": "repo_sandbox",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "shared_repo_integrator",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "project_success",
                        "prompt": "project",
                        "workspace": str(config.workspace_root / "project_success"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "project",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "project_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "project",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "project_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        run_task_surfaces: list[str] = []
        run_task_ids: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            FakeKernel.run_task_ids.append(task.task_id)
            metadata = dict(task.metadata)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
    )

    assert metrics.generated_total == 3
    assert metrics.generated_by_benchmark_family == {"repository": 2, "project": 1}
    assert FakeKernel.run_task_surfaces == [
        "repository_worker_bundle",
        "repository_integrator_bundle",
        "project_release_bundle",
    ]
    assert FakeKernel.run_task_ids == [
        "worker_success_repository_adjacent",
        "integrator_success_repository_adjacent",
        "project_success_project_adjacent",
    ]


def test_eval_generated_success_promotes_raw_repo_sandbox_seeds_to_long_horizon_repository_followups(
    monkeypatch, tmp_path
):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_repo_sandbox",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_repo_sandbox_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_parallel_worker_api_task",
                        "prompt": "worker",
                        "workspace": str(config.workspace_root / "git_parallel_worker_api_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_worker_branch",
                            "workflow_guard": {
                                "requires_git": True,
                                "shared_repo_id": "repo_sandbox_parallel_merge",
                                "target_branch": "main",
                                "worker_branch": "worker/api-status",
                                "claimed_paths": ["src/api_status.txt"],
                            },
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_worker_branch",
                                "workflow_guard": {
                                    "requires_git": True,
                                    "shared_repo_id": "repo_sandbox_parallel_merge",
                                    "target_branch": "main",
                                    "worker_branch": "worker/api-status",
                                    "claimed_paths": ["src/api_status.txt"],
                                },
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task",
                        "prompt": "integrator",
                        "workspace": str(config.workspace_root / "git_parallel_merge_acceptance_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_parallel_merge",
                            "shared_repo_order": 1,
                            "workflow_guard": {
                                "requires_git": True,
                                "shared_repo_id": "repo_sandbox_parallel_merge",
                                "target_branch": "main",
                            },
                            "semantic_verifier": {
                                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                            },
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_parallel_merge",
                                "shared_repo_order": 1,
                                "workflow_guard": {
                                    "requires_git": True,
                                    "shared_repo_id": "repo_sandbox_parallel_merge",
                                    "target_branch": "main",
                                },
                                "semantic_verifier": {
                                    "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                                },
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_repo_status_review_task",
                        "prompt": "review",
                        "workspace": str(config.workspace_root / "git_repo_status_review_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_workflow",
                            "workflow_guard": {"requires_git": True},
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_workflow",
                                "workflow_guard": {"requires_git": True},
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        run_task_surfaces: list[str] = []
        run_task_ids: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            FakeKernel.run_task_ids.append(task.task_id)
            metadata = dict(task.metadata)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
    )

    assert metrics.generated_total == 3
    assert metrics.generated_by_benchmark_family == {"repository": 3}
    assert FakeKernel.run_task_surfaces == [
        "repository_worker_bundle",
        "repository_integrator_bundle",
        "repository_validation_bundle",
    ]
    assert FakeKernel.run_task_ids == [
        "git_parallel_worker_api_task_repository_adjacent",
        "git_parallel_merge_acceptance_task_repository_adjacent",
        "git_repo_status_review_task_repository_adjacent",
    ]


def test_eval_generated_success_prefers_complete_shared_repo_worker_bundle_before_integrator(
    monkeypatch, tmp_path
):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace" / "scope_repo_sandbox",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_repo_bundle_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_parallel_worker_api_task",
                        "prompt": "worker api",
                        "workspace": str(config.workspace_root / "git_parallel_worker_api_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_worker_branch",
                            "workflow_guard": {
                                "requires_git": True,
                                "shared_repo_id": "repo_sandbox_parallel_merge",
                                "target_branch": "main",
                                "worker_branch": "worker/api-status",
                            },
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_worker_branch",
                                "workflow_guard": {
                                    "requires_git": True,
                                    "shared_repo_id": "repo_sandbox_parallel_merge",
                                    "target_branch": "main",
                                    "worker_branch": "worker/api-status",
                                },
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_worker_docs_task",
                        "prompt": "worker docs",
                        "workspace": str(config.workspace_root / "git_parallel_worker_docs_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_worker_branch",
                            "workflow_guard": {
                                "requires_git": True,
                                "shared_repo_id": "repo_sandbox_parallel_merge",
                                "target_branch": "main",
                                "worker_branch": "worker/docs-status",
                            },
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_worker_branch",
                                "workflow_guard": {
                                    "requires_git": True,
                                    "shared_repo_id": "repo_sandbox_parallel_merge",
                                    "target_branch": "main",
                                    "worker_branch": "worker/docs-status",
                                },
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task",
                        "prompt": "integrator",
                        "workspace": str(config.workspace_root / "git_parallel_merge_acceptance_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_parallel_merge",
                            "shared_repo_order": 1,
                            "workflow_guard": {
                                "requires_git": True,
                                "shared_repo_id": "repo_sandbox_parallel_merge",
                                "target_branch": "main",
                            },
                            "semantic_verifier": {
                                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                            },
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_parallel_merge",
                                "shared_repo_order": 1,
                                "workflow_guard": {
                                    "requires_git": True,
                                    "shared_repo_id": "repo_sandbox_parallel_merge",
                                    "target_branch": "main",
                                },
                                "semantic_verifier": {
                                    "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                                },
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_repo_status_review_task",
                        "prompt": "review",
                        "workspace": str(config.workspace_root / "git_repo_status_review_task"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_sandbox",
                            "difficulty": "git_workflow",
                            "workflow_guard": {"requires_git": True},
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_sandbox",
                                "difficulty": "git_workflow",
                                "workflow_guard": {"requires_git": True},
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            FakeKernel.run_task_ids.append(task.task_id)
            metadata = dict(task.metadata)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        max_generated_success_schedule_tasks=3,
    )

    assert metrics.generated_total == 3
    assert FakeKernel.run_task_surfaces == [
        "repository_worker_bundle",
        "repository_worker_bundle",
        "repository_integrator_bundle",
    ]
    assert FakeKernel.run_task_ids == [
        "git_parallel_worker_api_task_repository_adjacent",
        "git_parallel_worker_docs_task_repository_adjacent",
        "git_parallel_merge_acceptance_task_repository_adjacent",
    ]


def test_eval_can_emit_current_cycle_generated_success_seed_bundle(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    seed_output_path = tmp_path / "reports" / "generated_success_seeds.json"

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="repo_seed",
                    prompt="repo",
                    workspace_subdir="repo_seed",
                    metadata={"benchmark_family": "repository"},
                ),
                TaskSpec(
                    task_id="project_fail",
                    prompt="project",
                    workspace_subdir="project_fail",
                    metadata={"benchmark_family": "project"},
                ),
            ]

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            success = task.task_id == "repo_seed"
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=success,
                steps=[
                    StepRecord(
                        index=1,
                        thought="run",
                        action="code_execute",
                        content=f"printf '{task.task_id}\\n' > status.txt",
                        selected_skill_id=None,
                        command_result=None,
                        verification={"passed": success, "reasons": []},
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success" if success else "policy_terminated",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_generated=False,
        generated_success_seed_output_path=str(seed_output_path),
    )

    assert metrics.total == 2
    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert [episode["task_id"] for episode in payload["episodes"]] == ["repo_seed"]
    assert payload["episodes"][0]["summary"]["executed_commands"] == ["printf 'repo_seed\\n' > status.txt"]


def test_eval_generated_success_can_emit_next_wave_seed_bundle_from_generated_results(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "repository_seed",
                        "prompt": "repo",
                        "workspace": str(config.workspace_root / "repository_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repository",
                            "origin_benchmark_family": "project",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "repository_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repository",
                                "origin_benchmark_family": "project",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "repository_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert [episode["task_id"] for episode in payload["episodes"]] == ["repository_seed_workflow_adjacent"]
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "workflow"


def test_eval_generated_success_seed_bundle_persists_observed_runtime(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed",
                        "prompt": "workflow",
                        "workspace": str(config.workspace_root / "workflow_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        time.sleep(0.02)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata=dict(task.metadata),
            task_contract={"metadata": dict(task.metadata)},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert float(episode["task_metadata"]["observed_runtime_seconds"]) > 0.0
    assert episode["task_metadata"]["observed_runtime_phase"] == "generated_success"
    assert float(episode["summary"]["observed_runtime_seconds"]) > 0.0
    assert episode["summary"]["observed_runtime_phase"] == "generated_success"


def test_eval_generated_success_flushes_completion_snapshot_before_seed_bundle_persistence(
    monkeypatch,
    tmp_path,
):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    seed_bundle_path = tmp_path / "reports" / "generated_success_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    progress_path = tmp_path / "reports" / "generated_success_progress.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "shared_repo_seed",
                        "prompt": "seed",
                        "workspace": str(config.workspace_root / "shared_repo_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "project",
                            "long_horizon_coding_surface": "shared_repo_integrator",
                        },
                        "task_contract": {"metadata": {"benchmark_family": "project"}},
                        "summary": {},
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeEngine:
        def __init__(self, memory_root=None, config=None):
            del memory_root, config

        def schedule_generated_seed_episodes(self, results, curriculum_kind):
            assert curriculum_kind == "adjacent_success"
            return list(results)

        def generate_followup_task(self, result):
            del result
            return TaskSpec(
                task_id="git_parallel_merge_acceptance_task__worker__worker_docs-status_repository_adjacent",
                prompt="followup",
                workspace_subdir="followup_task",
                metadata={"benchmark_family": "repository", "curriculum_kind": "adjacent_success"},
            )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True, progress_callback=None):
            del clean_workspace
            assert callable(progress_callback)
            progress_callback(
                {
                    "event": "step_complete",
                    "step_index": 1,
                    "step_stage": "step_complete",
                    "completed_steps": 1,
                    "decision_action": "code_execute",
                    "verification_passed": True,
                    "step_elapsed_seconds": 0.01,
                }
            )
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[
                    StepRecord(
                        index=1,
                        thought="step",
                        action="code_execute",
                        content="true",
                        selected_skill_id=None,
                        command_result=None,
                        verification={"passed": True, "reasons": []},
                    )
                ],
                task_metadata=dict(task.metadata),
                task_contract={"metadata": dict(task.metadata)},
                termination_reason="success",
            )

        def close(self):
            return None

    bundle_write_started = threading.Event()
    release_bundle_write = threading.Event()
    bundle_write_calls = {"count": 0}

    def fake_write_success_seed_bundle(output_path, **kwargs):
        del output_path, kwargs
        bundle_write_calls["count"] += 1
        if bundle_write_calls["count"] == 1:
            bundle_write_started.set()
            release_bundle_write.wait(timeout=2.0)

    monkeypatch.setattr("evals.harness.CurriculumEngine", FakeEngine)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._write_success_seed_bundle", fake_write_success_seed_bundle)

    metrics_holder: dict[str, object] = {}

    def _run_eval_in_thread():
        metrics_holder["metrics"] = run_eval(
            config=config,
            include_primary_tasks=False,
            include_generated=True,
            generated_success_seed_documents_path=str(seed_bundle_path),
            generated_success_seed_output_path=str(seed_output_path),
            allow_generated_success_seed_fallback=True,
            max_generated_success_schedule_tasks=1,
            progress_snapshot_path=progress_path,
        )

    thread = threading.Thread(target=_run_eval_in_thread)
    thread.start()
    assert bundle_write_started.wait(timeout=1.0)

    snapshot = {}
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if progress_path.exists():
            snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
            if snapshot.get("completed_generated_tasks") == 1:
                break
        time.sleep(0.02)

    assert snapshot.get("phase") == "generated_success"
    assert snapshot.get("completed_generated_tasks") == 1
    assert snapshot.get("generated_passed") == 1
    assert (
        snapshot.get("last_completed_generated_task_id")
        == "git_parallel_merge_acceptance_task__worker__worker_docs-status_repository_adjacent"
    )
    assert snapshot.get("last_completed_generated_benchmark_family") == "repository"
    assert snapshot.get("current_task_id") == ""

    release_bundle_write.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert metrics_holder["metrics"].generated_total == 1
    assert metrics_holder["metrics"].generated_passed == 1


def test_eval_generated_success_seed_bundle_aggregates_runtime_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed_prior_a",
                        "prompt": "workflow prior a",
                        "workspace": str(config.workspace_root / "workflow_seed_prior_a"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                            "observed_runtime_seconds": 2.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                                "observed_runtime_seconds": 2.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 2.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "workflow_seed_prior_b",
                        "prompt": "workflow prior b",
                        "workspace": str(config.workspace_root / "workflow_seed_prior_b"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                            "observed_runtime_seconds": 4.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                                "observed_runtime_seconds": 4.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 4.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 5.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 5.0}},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert float(episode["task_metadata"]["observed_runtime_prior_seconds"]) >= 3.0
    assert int(episode["task_metadata"]["observed_runtime_prior_count"]) >= 3
    runtime_priors = payload["runtime_priors"]
    assert any(float(row["mean_seconds"]) >= 3.0 for row in runtime_priors.values())


def test_eval_generated_success_seed_bundle_aggregates_outcome_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed_prior_a",
                        "prompt": "workflow prior a",
                        "workspace": str(config.workspace_root / "workflow_seed_prior_a"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                            "observed_runtime_seconds": 2.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                                "observed_runtime_seconds": 2.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 2.0},
                        "termination_reason": "success",
                    }
                ],
                "outcome_priors": {
                    "workflow:workflow_release_bundle:workflow_release": {
                        "count": 4.0,
                        "success_rate": 0.5,
                        "timeout_rate": 0.25,
                        "budget_exceeded_rate": 0.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=False,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 3.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 3.0}},
            termination_reason="time_budget_exceeded",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert payload["episodes"] == []
    outcome_priors = payload["outcome_priors"]
    assert any(float(row["success_rate"]) < 0.7 for row in outcome_priors.values())
    assert any(float(row["budget_exceeded_rate"]) > 0.0 for row in outcome_priors.values())


def test_eval_generated_success_seed_bundle_aggregates_family_policy_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed_prior_a",
                        "prompt": "workflow prior a",
                        "workspace": str(config.workspace_root / "workflow_seed_prior_a"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                            "lineage_branch_kind": "cleanup",
                            "observed_runtime_seconds": 2.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                                "lineage_branch_kind": "cleanup",
                                "observed_runtime_seconds": 2.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 2.0},
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 3.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 3.0}},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert float(episode["task_metadata"]["observed_runtime_family_prior_seconds"]) >= 2.0
    assert int(episode["task_metadata"]["observed_runtime_family_prior_count"]) >= 2
    assert float(episode["task_metadata"]["observed_success_family_branch_prior_rate"]) > 0.5
    assert int(episode["task_metadata"]["observed_outcome_family_prior_count"]) >= 2
    emitted_family = str(episode["task_metadata"]["benchmark_family"])
    assert emitted_family in payload["runtime_family_priors"]
    assert any(str(key).startswith(f"{emitted_family}:") for key in payload["runtime_family_branch_priors"])
    assert emitted_family in payload["outcome_family_priors"]
    assert any(str(key).startswith(f"{emitted_family}:") for key in payload["outcome_family_branch_priors"])


def test_eval_generated_success_seed_bundle_aggregates_late_wave_branch_policy_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave9_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave10_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "oversight_cleanup_seed",
                        "prompt": "oversight cleanup",
                        "workspace": str(config.workspace_root / "oversight_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "oversight",
                            "origin_benchmark_family": "governance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                            "lineage_branch_kind": "cleanup",
                            "observed_runtime_seconds": 2.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "oversight",
                                "origin_benchmark_family": "governance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                                "lineage_branch_kind": "cleanup",
                                "observed_runtime_seconds": 2.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 2.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "adjudication_cleanup_seed",
                        "prompt": "adjudication cleanup",
                        "workspace": str(config.workspace_root / "adjudication_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "adjudication",
                            "origin_benchmark_family": "assurance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                            "lineage_branch_kind": "cleanup",
                            "observed_runtime_seconds": 4.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "adjudication",
                                "origin_benchmark_family": "assurance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                                "lineage_branch_kind": "cleanup",
                                "observed_runtime_seconds": 4.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 4.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 3.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 3.0}},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert float(episode["task_metadata"]["observed_runtime_late_wave_branch_prior_seconds"]) >= 2.5
    assert int(episode["task_metadata"]["observed_runtime_late_wave_branch_prior_count"]) >= 3
    assert float(episode["task_metadata"]["observed_success_late_wave_branch_prior_rate"]) > 0.5
    assert int(episode["task_metadata"]["observed_outcome_late_wave_branch_prior_count"]) >= 3
    assert "cleanup" in payload["runtime_late_wave_branch_priors"]
    assert "cleanup" in payload["outcome_late_wave_branch_priors"]


def test_eval_generated_success_seed_bundle_aggregates_late_wave_phase_policy_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave10_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "assurance_mid_cleanup_seed",
                        "prompt": "assurance mid cleanup",
                        "workspace": str(config.workspace_root / "assurance_mid_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 11,
                            "observed_runtime_seconds": 3.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "assurance",
                                "origin_benchmark_family": "oversight",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                                "lineage_branch_kind": "cleanup",
                                "lineage_depth": 11,
                                "observed_runtime_seconds": 3.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 3.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "adjudication_late_cleanup_seed",
                        "prompt": "adjudication late cleanup",
                        "workspace": str(config.workspace_root / "adjudication_late_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "adjudication",
                            "origin_benchmark_family": "assurance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 5.0,
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "adjudication",
                                "origin_benchmark_family": "assurance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                                "lineage_branch_kind": "cleanup",
                                "lineage_depth": 15,
                                "observed_runtime_seconds": 5.0,
                            }
                        },
                        "summary": {"observed_runtime_seconds": 5.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 4.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 4.0}},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert float(episode["task_metadata"]["observed_runtime_late_wave_phase_prior_seconds"]) >= 3.0
    assert int(episode["task_metadata"]["observed_runtime_late_wave_phase_prior_count"]) >= 2
    assert float(episode["task_metadata"]["observed_success_late_wave_phase_prior_rate"]) > 0.5
    assert int(episode["task_metadata"]["observed_outcome_late_wave_phase_prior_count"]) >= 2
    assert episode["task_metadata"]["observed_lineage_phase"] in {"early", "mid", "late"}
    assert any(str(key).startswith("cleanup:") for key in payload["runtime_late_wave_phase_priors"])
    assert any(str(key).startswith("cleanup:") for key in payload["outcome_late_wave_phase_priors"])


def test_eval_generated_success_seed_bundle_aggregates_late_wave_phase_state_policy_priors(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave10_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "assurance_late_cleanup_productive_seed",
                        "prompt": "assurance late cleanup productive",
                        "workspace": str(config.workspace_root / "assurance_late_cleanup_productive_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 4.0,
                            "observed_scheduler_state": "productive",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "assurance",
                                "origin_benchmark_family": "oversight",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                                "lineage_branch_kind": "cleanup",
                                "lineage_depth": 15,
                                "observed_runtime_seconds": 4.0,
                                "observed_scheduler_state": "productive",
                            }
                        },
                        "summary": {"observed_runtime_seconds": 4.0, "observed_scheduler_state": "productive"},
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        report_config=None,
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress, report_config
        assert phase == "generated_success"
        task = tasks[0]
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata={**dict(task.metadata), "observed_runtime_seconds": 5.0},
            task_contract={"metadata": {**dict(task.metadata), "observed_runtime_seconds": 5.0}},
            termination_reason="success",
        )
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        return [result]

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert episode["task_metadata"]["observed_scheduler_state"] == "productive"
    assert float(episode["task_metadata"]["observed_runtime_late_wave_phase_state_prior_seconds"]) >= 4.0
    assert int(episode["task_metadata"]["observed_runtime_late_wave_phase_state_prior_count"]) >= 1
    assert float(episode["task_metadata"]["observed_success_late_wave_phase_state_prior_rate"]) >= 1.0
    assert int(episode["task_metadata"]["observed_outcome_late_wave_phase_state_prior_count"]) >= 1
    assert any(str(key).endswith(":productive") for key in payload["runtime_late_wave_phase_state_priors"])
    assert any(str(key).endswith(":productive") for key in payload["outcome_late_wave_phase_state_priors"])


def test_eval_generated_success_seed_bundle_recency_weights_late_wave_phase_state_priors(monkeypatch, tmp_path):
    from evals import harness as eval_harness

    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "old_stalled_seed",
                        "prompt": "old stalled",
                        "workspace": str(tmp_path / "workspace" / "old_stalled_seed"),
                        "success": False,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 8.0,
                            "observed_scheduler_state": "stalled",
                            "observed_recorded_at": 1000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 1000.0}},
                        "summary": {"observed_runtime_seconds": 8.0, "observed_scheduler_state": "stalled", "observed_recorded_at": 1000.0},
                        "termination_reason": "time_budget_exceeded",
                    },
                    {
                        "task_id": "fresh_productive_seed",
                        "prompt": "fresh productive",
                        "workspace": str(tmp_path / "workspace" / "fresh_productive_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 2.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 100000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 100000.0}},
                        "summary": {"observed_runtime_seconds": 2.0, "observed_scheduler_state": "productive", "observed_recorded_at": 100000.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evals.harness.time.time", lambda: 100000.0)
    eval_harness._write_success_seed_bundle(
        str(seed_output_path),
        primary_tasks=[],
        primary_results=[],
        generated_tasks=[],
        generated_results=[],
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    productive = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":productive"))
    stalled = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":stalled"))
    assert float(productive["count"]) > float(stalled["count"])


def test_eval_generated_success_seed_bundle_tracks_phase_state_recency_support_count(monkeypatch, tmp_path):
    from evals import harness as eval_harness

    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "fresh_productive_seed_a",
                        "prompt": "fresh productive a",
                        "workspace": str(tmp_path / "workspace" / "fresh_productive_seed_a"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 2.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 100000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 100000.0}},
                        "summary": {"observed_runtime_seconds": 2.0, "observed_scheduler_state": "productive", "observed_recorded_at": 100000.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "fresh_productive_seed_b",
                        "prompt": "fresh productive b",
                        "workspace": str(tmp_path / "workspace" / "fresh_productive_seed_b"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "observed_runtime_seconds": 3.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 99990.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 99990.0}},
                        "summary": {"observed_runtime_seconds": 3.0, "observed_scheduler_state": "productive", "observed_recorded_at": 99990.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evals.harness.time.time", lambda: 100000.0)
    eval_harness._write_success_seed_bundle(
        str(seed_output_path),
        primary_tasks=[],
        primary_results=[],
        generated_tasks=[],
        generated_results=[],
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    productive = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":productive"))
    assert float(productive["support_count"]) >= 2.0


def test_eval_generated_success_seed_bundle_tracks_phase_state_dispersion_count(monkeypatch, tmp_path):
    from evals import harness as eval_harness

    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "assurance_cleanup_seed",
                        "prompt": "assurance cleanup",
                        "workspace": str(tmp_path / "workspace" / "assurance_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["validation", "governance", "oversight", "assurance"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 2.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 100000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 100000.0}},
                        "summary": {"observed_runtime_seconds": 2.0, "observed_scheduler_state": "productive", "observed_recorded_at": 100000.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "adjudication_cleanup_seed",
                        "prompt": "adjudication cleanup",
                        "workspace": str(tmp_path / "workspace" / "adjudication_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "adjudication",
                            "origin_benchmark_family": "assurance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["governance", "oversight", "assurance", "adjudication"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 3.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 99990.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 99990.0}},
                        "summary": {"observed_runtime_seconds": 3.0, "observed_scheduler_state": "productive", "observed_recorded_at": 99990.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evals.harness.time.time", lambda: 100000.0)
    eval_harness._write_success_seed_bundle(
        str(seed_output_path),
        primary_tasks=[],
        primary_results=[],
        generated_tasks=[],
        generated_results=[],
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    productive = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":productive"))
    assert float(productive["dispersion_count"]) >= 2.0


def test_eval_generated_success_seed_bundle_tracks_phase_state_directional_dispersion_count(monkeypatch, tmp_path):
    from evals import harness as eval_harness

    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "assurance_cleanup_seed",
                        "prompt": "assurance cleanup",
                        "workspace": str(tmp_path / "workspace" / "assurance_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["validation", "governance", "oversight", "assurance"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 2.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 100000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 100000.0}},
                        "summary": {"observed_runtime_seconds": 2.0, "observed_scheduler_state": "productive", "observed_recorded_at": 100000.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "adjudication_cleanup_seed",
                        "prompt": "adjudication cleanup",
                        "workspace": str(tmp_path / "workspace" / "adjudication_cleanup_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "adjudication",
                            "origin_benchmark_family": "assurance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["governance", "oversight", "assurance", "adjudication"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 3.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 99990.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 99990.0}},
                        "summary": {"observed_runtime_seconds": 3.0, "observed_scheduler_state": "productive", "observed_recorded_at": 99990.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "assurance_cleanup_seed_lateral",
                        "prompt": "assurance cleanup lateral",
                        "workspace": str(tmp_path / "workspace" / "assurance_cleanup_seed_lateral"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["validation", "governance", "oversight", "assurance"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 2.5,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 99980.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 99980.0}},
                        "summary": {"observed_runtime_seconds": 2.5, "observed_scheduler_state": "productive", "observed_recorded_at": 99980.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evals.harness.time.time", lambda: 100000.0)
    eval_harness._write_success_seed_bundle(
        str(seed_output_path),
        primary_tasks=[],
        primary_results=[],
        generated_tasks=[],
        generated_results=[],
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    productive = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":productive"))
    assert float(productive["dispersion_count"]) >= 2.0
    assert float(productive["directional_dispersion_count"]) >= 2.0


def test_eval_generated_success_seed_bundle_tracks_phase_state_phase_transition_count(monkeypatch, tmp_path):
    from evals import harness as eval_harness

    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_output_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "assurance_mid_to_late_seed",
                        "prompt": "assurance mid to late",
                        "workspace": str(tmp_path / "workspace" / "assurance_mid_to_late_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 14,
                            "lineage_families": ["validation", "governance", "oversight", "assurance"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 2.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 100000.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 100000.0}},
                        "summary": {"observed_runtime_seconds": 2.0, "observed_scheduler_state": "productive", "observed_recorded_at": 100000.0},
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "adjudication_late_seed",
                        "prompt": "adjudication late",
                        "workspace": str(tmp_path / "workspace" / "adjudication_late_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "adjudication",
                            "origin_benchmark_family": "assurance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
                            "lineage_branch_kind": "cleanup",
                            "lineage_depth": 15,
                            "lineage_families": ["governance", "oversight", "assurance", "adjudication"],
                            "lineage_branch_kinds": ["cleanup", "cleanup", "cleanup", "cleanup"],
                            "observed_runtime_seconds": 3.0,
                            "observed_scheduler_state": "productive",
                            "observed_recorded_at": 99990.0,
                        },
                        "task_contract": {"metadata": {"observed_recorded_at": 99990.0}},
                        "summary": {"observed_runtime_seconds": 3.0, "observed_scheduler_state": "productive", "observed_recorded_at": 99990.0},
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evals.harness.time.time", lambda: 100000.0)
    eval_harness._write_success_seed_bundle(
        str(seed_output_path),
        primary_tasks=[],
        primary_results=[],
        generated_tasks=[],
        generated_results=[],
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    productive = next(v for k, v in payload["outcome_late_wave_phase_state_priors"].items() if str(k).endswith(":productive"))
    assert float(productive["phase_transition_count"]) >= 1.0


def test_eval_generated_success_can_emit_third_wave_seed_bundle_from_generated_results(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed",
                        "prompt": "workflow",
                        "workspace": str(config.workspace_root / "workflow_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert [episode["task_id"] for episode in payload["episodes"]] == ["workflow_seed_tooling_adjacent"]
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "tooling"


def test_eval_generated_success_persists_seed_bundle_before_generated_timeout(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave2_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "workflow_seed",
                        "prompt": "workflow",
                        "workspace": str(config.workspace_root / "workflow_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "workflow",
                            "origin_benchmark_family": "repository",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "workflow_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "workflow",
                                "origin_benchmark_family": "repository",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "workflow_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def close(self):
            return None

    def fake_run_tasks_with_progress(
        tasks,
        kernel,
        *,
        progress_label,
        phase="",
        on_result=None,
        on_task_start=None,
        on_task_progress=None,
        on_task_complete=None,
    ):
        del kernel, progress_label, on_task_progress
        if phase != "generated_success":
            return []
        assert len(tasks) == 1
        task = tasks[0]
        result = EpisodeRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            workspace=str(tmp_path / "workspace" / task.workspace_subdir),
            success=True,
            steps=[],
            task_metadata=dict(task.metadata),
            task_contract={"metadata": dict(task.metadata)},
            termination_reason="success",
        )
        if on_task_start is not None:
            on_task_start(task, 1, 1)
        if on_task_complete is not None:
            on_task_complete(task, result, 1, 1)
        if on_result is not None:
            on_result(task, result, 1, 1)
        raise RuntimeError("simulated generated timeout after first completion")

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)
    monkeypatch.setattr("evals.harness._run_tasks_with_progress", fake_run_tasks_with_progress)

    try:
        run_eval(
            config=config,
            include_primary_tasks=False,
            include_generated=True,
            generated_success_seed_documents_path=str(seed_bundle_path),
            generated_success_seed_output_path=str(seed_output_path),
            allow_generated_success_seed_fallback=True,
            max_generated_success_schedule_tasks=1,
        )
    except RuntimeError as exc:
        assert "simulated generated timeout" in str(exc)
    else:
        raise AssertionError("expected generated-success timeout simulation")

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert [episode["task_id"] for episode in payload["episodes"]] == ["workflow_seed_tooling_adjacent"]
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "tooling"


def test_eval_generated_success_can_emit_fourth_wave_seed_bundle_from_generated_results(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave3_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave4_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "tooling_seed",
                        "prompt": "tooling",
                        "workspace": str(config.workspace_root / "tooling_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "tooling",
                            "origin_benchmark_family": "workflow",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "tooling_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "tooling",
                                "origin_benchmark_family": "workflow",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "tooling_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert [episode["task_id"] for episode in payload["episodes"]] == ["tooling_seed_integration_adjacent"]
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "integration"


def test_eval_generated_success_can_emit_fifth_wave_seed_bundle_from_generated_results(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave4_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave5_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "integration_seed",
                        "prompt": "integration",
                        "workspace": str(config.workspace_root / "integration_seed"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "integration",
                            "origin_benchmark_family": "tooling",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "integration_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "integration",
                                "origin_benchmark_family": "tooling",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "integration_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert [episode["task_id"] for episode in payload["episodes"]] == ["integration_seed_repo_chore_adjacent"]
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "repo_chore"


def test_eval_generated_success_diversifies_repo_chore_terminal_variants(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave4_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave5_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
                        "prompt": "integration cleanup",
                        "workspace": str(config.workspace_root / "integration_seed_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "integration",
                            "origin_benchmark_family": "tooling",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "integration_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "integration",
                                "origin_benchmark_family": "tooling",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "integration_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
                        "prompt": "integration audit",
                        "workspace": str(config.workspace_root / "integration_seed_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "integration",
                            "origin_benchmark_family": "tooling",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "integration_release_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "integration",
                                "origin_benchmark_family": "tooling",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "integration_release_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "repo_chore_cleanup_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "repo_chore_audit_bundle"


def test_eval_generated_success_bridges_repo_chore_into_validation(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave5_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave6_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
                        "prompt": "repo chore cleanup",
                        "workspace": str(config.workspace_root / "repo_chore_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_chore",
                            "origin_benchmark_family": "integration",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "repo_chore_cleanup_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_chore",
                                "origin_benchmark_family": "integration",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "repo_chore_cleanup_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
                        "prompt": "repo chore audit",
                        "workspace": str(config.workspace_root / "repo_chore_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "repo_chore",
                            "origin_benchmark_family": "integration",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "repo_chore_audit_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "repo_chore",
                                "origin_benchmark_family": "integration",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "repo_chore_audit_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "validation"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "validation"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "validation_cleanup_gate_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "validation_audit_gate_bundle"


def test_eval_generated_success_bridges_validation_into_governance(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave6_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave7_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
                        "prompt": "validation cleanup",
                        "workspace": str(config.workspace_root / "validation_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "validation",
                            "origin_benchmark_family": "repo_chore",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "validation_cleanup_gate_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "validation",
                                "origin_benchmark_family": "repo_chore",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "validation_cleanup_gate_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
                        "prompt": "validation audit",
                        "workspace": str(config.workspace_root / "validation_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "validation",
                            "origin_benchmark_family": "repo_chore",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "validation_audit_gate_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "validation",
                                "origin_benchmark_family": "repo_chore",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "validation_audit_gate_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "governance"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "governance"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "governance_cleanup_review_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "governance_audit_review_bundle"


def test_eval_generated_success_bridges_validation_integrator_into_governance(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave6_integrator_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave7_integrator_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
                        "prompt": "validation integrator cleanup",
                        "workspace": str(config.workspace_root / "validation_integrator_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "validation",
                            "origin_benchmark_family": "repo_chore",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "validation_integrator_cleanup_gate_bundle",
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "shared_repo_integrator_bundle",
                                "repository_integrator_bundle",
                                "workflow_release_bundle",
                                "tooling_release_bundle",
                                "integration_release_bundle",
                                "repo_chore_cleanup_bundle",
                                "validation_integrator_cleanup_gate_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "project_release",
                                "workflow_release",
                                "tooling_release",
                                "integration_release",
                                "cleanup",
                                "cleanup",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "validation",
                                "origin_benchmark_family": "repo_chore",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "validation_integrator_cleanup_gate_bundle",
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "shared_repo_integrator_bundle",
                                    "repository_integrator_bundle",
                                    "workflow_release_bundle",
                                    "tooling_release_bundle",
                                    "integration_release_bundle",
                                    "repo_chore_cleanup_bundle",
                                    "validation_integrator_cleanup_gate_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "project_release",
                                    "workflow_release",
                                    "tooling_release",
                                    "integration_release",
                                    "cleanup",
                                    "cleanup",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
                        "prompt": "validation integrator audit",
                        "workspace": str(config.workspace_root / "validation_integrator_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "validation",
                            "origin_benchmark_family": "repo_chore",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "validation_integrator_audit_gate_bundle",
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "shared_repo_integrator_bundle",
                                "repository_integrator_bundle",
                                "workflow_release_bundle",
                                "tooling_release_bundle",
                                "integration_release_bundle",
                                "repo_chore_audit_bundle",
                                "validation_integrator_audit_gate_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "project_release",
                                "workflow_release",
                                "tooling_release",
                                "integration_release",
                                "audit",
                                "audit",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "validation",
                                "origin_benchmark_family": "repo_chore",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "validation_integrator_audit_gate_bundle",
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "shared_repo_integrator_bundle",
                                    "repository_integrator_bundle",
                                    "workflow_release_bundle",
                                    "tooling_release_bundle",
                                    "integration_release_bundle",
                                    "repo_chore_audit_bundle",
                                    "validation_integrator_audit_gate_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "project_release",
                                    "workflow_release",
                                    "tooling_release",
                                    "integration_release",
                                    "audit",
                                    "audit",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "governance"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "governance"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "governance_integrator_cleanup_review_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "governance_integrator_audit_review_bundle"


def test_eval_generated_success_bridges_governance_into_oversight(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave7_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave8_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
                        "prompt": "governance cleanup",
                        "workspace": str(config.workspace_root / "governance_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "governance",
                            "origin_benchmark_family": "validation",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "governance_cleanup_review_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "governance",
                                "origin_benchmark_family": "validation",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "governance_cleanup_review_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
                        "prompt": "governance audit",
                        "workspace": str(config.workspace_root / "governance_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "governance",
                            "origin_benchmark_family": "validation",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "governance_audit_review_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "governance",
                                "origin_benchmark_family": "validation",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "governance_audit_review_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "oversight"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "oversight"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "oversight_cleanup_crosscheck_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "oversight_audit_crosscheck_bundle"


def test_eval_generated_success_bridges_governance_integrator_into_oversight(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave7_integrator_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave8_integrator_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
                        "prompt": "governance integrator cleanup",
                        "workspace": str(config.workspace_root / "governance_integrator_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "governance",
                            "origin_benchmark_family": "validation",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "governance_integrator_cleanup_review_bundle",
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "shared_repo_integrator_bundle",
                                "repository_integrator_bundle",
                                "workflow_release_bundle",
                                "tooling_release_bundle",
                                "integration_release_bundle",
                                "repo_chore_cleanup_bundle",
                                "validation_integrator_cleanup_gate_bundle",
                                "governance_integrator_cleanup_review_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "project_release",
                                "workflow_release",
                                "tooling_release",
                                "integration_release",
                                "cleanup",
                                "cleanup",
                                "cleanup",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "governance",
                                "origin_benchmark_family": "validation",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "governance_integrator_cleanup_review_bundle",
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "shared_repo_integrator_bundle",
                                    "repository_integrator_bundle",
                                    "workflow_release_bundle",
                                    "tooling_release_bundle",
                                    "integration_release_bundle",
                                    "repo_chore_cleanup_bundle",
                                    "validation_integrator_cleanup_gate_bundle",
                                    "governance_integrator_cleanup_review_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "project_release",
                                    "workflow_release",
                                    "tooling_release",
                                    "integration_release",
                                    "cleanup",
                                    "cleanup",
                                    "cleanup",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
                        "prompt": "governance integrator audit",
                        "workspace": str(config.workspace_root / "governance_integrator_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "governance",
                            "origin_benchmark_family": "validation",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "governance_integrator_audit_review_bundle",
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "shared_repo_integrator_bundle",
                                "repository_integrator_bundle",
                                "workflow_release_bundle",
                                "tooling_release_bundle",
                                "integration_release_bundle",
                                "repo_chore_audit_bundle",
                                "validation_integrator_audit_gate_bundle",
                                "governance_integrator_audit_review_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "project_release",
                                "workflow_release",
                                "tooling_release",
                                "integration_release",
                                "audit",
                                "audit",
                                "audit",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "governance",
                                "origin_benchmark_family": "validation",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "governance_integrator_audit_review_bundle",
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "shared_repo_integrator_bundle",
                                    "repository_integrator_bundle",
                                    "workflow_release_bundle",
                                    "tooling_release_bundle",
                                    "integration_release_bundle",
                                    "repo_chore_audit_bundle",
                                    "validation_integrator_audit_gate_bundle",
                                    "governance_integrator_audit_review_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "project_release",
                                    "workflow_release",
                                    "tooling_release",
                                    "integration_release",
                                    "audit",
                                    "audit",
                                    "audit",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "oversight"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "oversight"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "oversight_integrator_cleanup_crosscheck_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "oversight_integrator_audit_crosscheck_bundle"


def test_eval_generated_success_bridges_oversight_into_assurance(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave8_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave9_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
                        "prompt": "oversight cleanup",
                        "workspace": str(config.workspace_root / "oversight_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "oversight",
                            "origin_benchmark_family": "governance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "oversight",
                                "origin_benchmark_family": "governance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                    {
                        "task_id": "git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
                        "prompt": "oversight audit",
                        "workspace": str(config.workspace_root / "oversight_audit"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "oversight",
                            "origin_benchmark_family": "governance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "oversight_audit_crosscheck_bundle",
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "oversight",
                                "origin_benchmark_family": "governance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "oversight_audit_crosscheck_bundle",
                            }
                        },
                        "termination_reason": "success",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=2,
        task_limit=2,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 2
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "assurance"
    assert payload["episodes"][1]["task_metadata"]["benchmark_family"] == "assurance"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "assurance_cleanup_cert_bundle"
    assert payload["episodes"][1]["task_metadata"]["long_horizon_coding_surface"] == "assurance_audit_cert_bundle"


def test_eval_generated_success_seed_bundle_persists_lineage_metadata(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave8_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave9_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "cleanup_seed",
                        "prompt": "oversight cleanup",
                        "workspace": str(config.workspace_root / "oversight_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "oversight",
                            "origin_benchmark_family": "governance",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                            "lineage_families": ["project", "repository", "governance", "oversight"],
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "repository_release_bundle",
                                "governance_cleanup_review_bundle",
                                "oversight_cleanup_crosscheck_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "cleanup",
                                "cleanup",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "oversight",
                                "origin_benchmark_family": "governance",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
                                "lineage_families": ["project", "repository", "governance", "oversight"],
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "repository_release_bundle",
                                    "governance_cleanup_review_bundle",
                                    "oversight_cleanup_crosscheck_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "cleanup",
                                    "cleanup",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
        task_limit=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert episode["task_metadata"]["lineage_families"][-1] == "assurance"
    assert episode["task_metadata"]["lineage_surfaces"][-1] == "assurance_cleanup_cert_bundle"
    assert episode["task_metadata"]["lineage_branch_kinds"][-1] == "cleanup"
    assert episode["task_contract"]["metadata"]["lineage_families"][-1] == "assurance"


def test_eval_generated_success_bridges_assurance_into_adjudication(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    seed_bundle_path = tmp_path / "reports" / "generated_success_wave10_seeds.json"
    seed_output_path = tmp_path / "reports" / "generated_success_wave11_seeds.json"
    seed_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": "generic_assurance_cleanup_tail",
                        "prompt": "assurance cleanup",
                        "workspace": str(config.workspace_root / "assurance_cleanup"),
                        "success": True,
                        "task_metadata": {
                            "benchmark_family": "assurance",
                            "origin_benchmark_family": "oversight",
                            "difficulty": "long_horizon",
                            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                            "lineage_families": [
                                "project",
                                "repository",
                                "workflow",
                                "tooling",
                                "integration",
                                "repo_chore",
                                "validation",
                                "governance",
                                "oversight",
                                "assurance",
                            ],
                            "lineage_surfaces": [
                                "project_release_bundle",
                                "repository_release_bundle",
                                "workflow_release_bundle",
                                "tooling_release_bundle",
                                "integration_release_bundle",
                                "repo_chore_cleanup_bundle",
                                "validation_cleanup_gate_bundle",
                                "governance_cleanup_review_bundle",
                                "oversight_cleanup_crosscheck_bundle",
                                "assurance_cleanup_cert_bundle",
                            ],
                            "lineage_branch_kinds": [
                                "project_release",
                                "project_release",
                                "workflow_release",
                                "tooling_release",
                                "integration_release",
                                "cleanup",
                                "cleanup",
                                "cleanup",
                                "cleanup",
                                "cleanup",
                            ],
                        },
                        "task_contract": {
                            "metadata": {
                                "benchmark_family": "assurance",
                                "origin_benchmark_family": "oversight",
                                "difficulty": "long_horizon",
                                "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
                                "lineage_families": [
                                    "project",
                                    "repository",
                                    "workflow",
                                    "tooling",
                                    "integration",
                                    "repo_chore",
                                    "validation",
                                    "governance",
                                    "oversight",
                                    "assurance",
                                ],
                                "lineage_surfaces": [
                                    "project_release_bundle",
                                    "repository_release_bundle",
                                    "workflow_release_bundle",
                                    "tooling_release_bundle",
                                    "integration_release_bundle",
                                    "repo_chore_cleanup_bundle",
                                    "validation_cleanup_gate_bundle",
                                    "governance_cleanup_review_bundle",
                                    "oversight_cleanup_crosscheck_bundle",
                                    "assurance_cleanup_cert_bundle",
                                ],
                                "lineage_branch_kinds": [
                                    "project_release",
                                    "project_release",
                                    "workflow_release",
                                    "tooling_release",
                                    "integration_release",
                                    "cleanup",
                                    "cleanup",
                                    "cleanup",
                                    "cleanup",
                                    "cleanup",
                                ],
                            }
                        },
                        "termination_reason": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeKernel:
        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        include_primary_tasks=False,
        include_generated=True,
        generated_success_seed_documents_path=str(seed_bundle_path),
        generated_success_seed_output_path=str(seed_output_path),
        allow_generated_success_seed_fallback=True,
        max_generated_success_schedule_tasks=1,
        task_limit=1,
    )

    payload = json.loads(seed_output_path.read_text(encoding="utf-8"))
    assert metrics.generated_total == 1
    assert payload["episodes"][0]["task_metadata"]["benchmark_family"] == "adjudication"
    assert payload["episodes"][0]["task_metadata"]["long_horizon_coding_surface"] == "adjudication_cleanup_ruling_bundle"
    assert payload["episodes"][0]["task_metadata"]["lineage_depth"] == 11


def test_run_eval_surfaces_shared_repo_synthetic_workers_for_project_priority(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="plain_project_task",
                    prompt="plain project",
                    workspace_subdir="plain_project_task",
                    metadata={"benchmark_family": "project"},
                ),
                TaskSpec(
                    task_id="shared_repo_integrator",
                    prompt="shared repo integrator",
                    workspace_subdir="shared_repo_integrator",
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/a"],
                        },
                    },
                ),
            ]

        def parallel_worker_tasks(self, task_id, *, target_worker_count=None):
            del target_worker_count
            if task_id != "shared_repo_integrator":
                return []
            return [
                TaskSpec(
                    task_id="shared_repo_worker",
                    prompt="shared repo worker",
                    workspace_subdir="shared_repo_worker",
                    suggested_commands=["sed -i '1s#pending#ready#' src/status.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [
                            {
                                "edit_kind": "line_replace",
                                "path": "src/status.txt",
                                "line_number": 1,
                                "old_text": "pending",
                                "new_text": "ready",
                                "edit_score": 42.0,
                            }
                        ],
                    },
                )
            ]

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_families: list[str] = []
        run_task_origins: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            FakeKernel.run_task_ids.append(task.task_id)
            FakeKernel.run_task_families.append(str(metadata.get("benchmark_family", "")))
            FakeKernel.run_task_origins.append(str(metadata.get("origin_benchmark_family", "")))
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        task_limit=1,
        priority_benchmark_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert metrics.total == 1
    assert FakeKernel.run_task_ids == ["shared_repo_worker"]
    assert FakeKernel.run_task_families == ["project"]
    assert FakeKernel.run_task_origins == ["repo_sandbox"]
    assert FakeKernel.run_task_surfaces == ["shared_repo_synthetic_worker"]


def test_run_eval_surfaces_shared_repo_integrators_for_project_priority(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="shared_repo_integrator",
                    prompt="shared repo integrator",
                    workspace_subdir="shared_repo_integrator",
                    suggested_commands=[
                        "git merge --no-ff worker/a -m 'merge worker/a' && tests/test_a.sh && mkdir -p reports && printf 'passed\\n' > reports/test_report.txt"
                    ],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "shared_repo_order": 1,
                        "workflow_guard": {"shared_repo_id": "repo-a", "target_branch": "main"},
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/a"],
                            "expected_changed_paths": ["src/a.txt", "reports/test_report.txt"],
                        },
                    },
                )
            ]

        def parallel_worker_tasks(self, task_id, *, target_worker_count=None):
            del task_id, target_worker_count
            return []

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            FakeKernel.run_task_ids.append(task.task_id)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        task_limit=1,
        priority_benchmark_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert metrics.total == 1
    assert FakeKernel.run_task_ids == ["shared_repo_integrator"]
    assert FakeKernel.run_task_surfaces == ["shared_repo_integrator"]


def test_run_eval_reserves_shared_repo_integrator_coverage_for_project_priority(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="shared_repo_integrator",
                    prompt="shared repo integrator",
                    workspace_subdir="shared_repo_integrator",
                    suggested_commands=[
                        "git merge --no-ff worker/a -m 'merge worker/a' && git merge --no-ff worker/b -m 'merge worker/b'"
                    ],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "shared_repo_order": 1,
                        "workflow_guard": {"shared_repo_id": "repo-a", "target_branch": "main"},
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/a", "worker/b"],
                            "expected_changed_paths": ["src/a.txt", "src/b.txt", "reports/test_report.txt"],
                        },
                    },
                ),
            ]

        def parallel_worker_tasks(self, task_id, *, target_worker_count=None):
            del target_worker_count
            if task_id != "shared_repo_integrator":
                return []
            return [
                TaskSpec(
                    task_id="shared_repo_worker_a",
                    prompt="shared repo worker a",
                    workspace_subdir="shared_repo_worker_a",
                    suggested_commands=["sed -i '1s#pending#ready#' src/a.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [
                            {
                                "edit_kind": "line_replace",
                                "path": "src/a.txt",
                                "line_number": 1,
                                "old_text": "pending",
                                "new_text": "ready",
                                "edit_score": 40.0,
                            }
                        ],
                        "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/a"},
                    },
                ),
                TaskSpec(
                    task_id="shared_repo_worker_b",
                    prompt="shared repo worker b",
                    workspace_subdir="shared_repo_worker_b",
                    suggested_commands=["sed -i '1s#pending#ready#' src/b.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [
                            {
                                "edit_kind": "line_replace",
                                "path": "src/b.txt",
                                "line_number": 1,
                                "old_text": "pending",
                                "new_text": "ready",
                                "edit_score": 39.0,
                            }
                        ],
                        "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/b"},
                    },
                ),
                TaskSpec(
                    task_id="shared_repo_worker_c",
                    prompt="shared repo worker c",
                    workspace_subdir="shared_repo_worker_c",
                    suggested_commands=["sed -i '1s#pending#ready#' src/c.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [
                            {
                                "edit_kind": "line_replace",
                                "path": "src/c.txt",
                                "line_number": 1,
                                "old_text": "pending",
                                "new_text": "ready",
                                "edit_score": 38.0,
                            }
                        ],
                        "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/c"},
                    },
                ),
            ]

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            FakeKernel.run_task_ids.append(task.task_id)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        task_limit=3,
        priority_benchmark_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert metrics.total == 3
    assert "shared_repo_integrator" in FakeKernel.run_task_ids
    assert FakeKernel.run_task_ids == [
        "shared_repo_worker_a",
        "shared_repo_worker_b",
        "shared_repo_integrator",
    ]
    assert FakeKernel.run_task_surfaces.count("shared_repo_integrator") == 1
    assert FakeKernel.run_task_surfaces.count("shared_repo_synthetic_worker") == 2


def test_run_eval_keeps_incomplete_shared_repo_bundle_on_workers_until_all_required_branches_fit(
    monkeypatch, tmp_path
):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="shared_repo_integrator",
                    prompt="shared repo integrator",
                    workspace_subdir="shared_repo_integrator",
                    suggested_commands=[
                        "git merge --no-ff worker/a -m 'merge worker/a' && git merge --no-ff worker/b -m 'merge worker/b'"
                    ],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "shared_repo_order": 1,
                        "workflow_guard": {"shared_repo_id": "repo-a", "target_branch": "main"},
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/a", "worker/b"],
                            "expected_changed_paths": ["src/a.txt", "src/b.txt", "reports/test_report.txt"],
                        },
                    },
                ),
            ]

        def parallel_worker_tasks(self, task_id, *, target_worker_count=None):
            del target_worker_count
            if task_id != "shared_repo_integrator":
                return []
            return [
                TaskSpec(
                    task_id="shared_repo_worker_a",
                    prompt="shared repo worker a",
                    workspace_subdir="shared_repo_worker_a",
                    suggested_commands=["sed -i '1s#pending#ready#' src/a.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [{"path": "src/a.txt"}],
                        "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/a"},
                    },
                ),
                TaskSpec(
                    task_id="shared_repo_worker_b",
                    prompt="shared repo worker b",
                    workspace_subdir="shared_repo_worker_b",
                    suggested_commands=["sed -i '1s#pending#ready#' src/b.txt"],
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "synthetic_worker": True,
                        "synthetic_edit_plan": [{"path": "src/b.txt"}],
                        "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/b"},
                    },
                ),
            ]

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            FakeKernel.run_task_ids.append(task.task_id)
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        task_limit=2,
        priority_benchmark_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert metrics.total == 2
    assert FakeKernel.run_task_ids == ["shared_repo_worker_a", "shared_repo_worker_b"]
    assert FakeKernel.run_task_surfaces == [
        "shared_repo_synthetic_worker",
        "shared_repo_synthetic_worker",
    ]


def test_run_eval_keeps_shared_repo_bundle_coherent_for_project_priority(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    class FakeTaskBank:
        def __init__(self, config=None):
            del config

        def list(self):
            return [
                TaskSpec(
                    task_id="shared_repo_integrator_a",
                    prompt="shared repo integrator a",
                    workspace_subdir="shared_repo_integrator_a",
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "shared_repo_order": 1,
                        "workflow_guard": {"shared_repo_id": "repo-a", "target_branch": "main"},
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/a", "worker/b"],
                            "expected_changed_paths": ["src/a.txt", "src/b.txt"],
                        },
                    },
                ),
                TaskSpec(
                    task_id="shared_repo_integrator_b",
                    prompt="shared repo integrator b",
                    workspace_subdir="shared_repo_integrator_b",
                    metadata={
                        "benchmark_family": "repo_sandbox",
                        "shared_repo_order": 1,
                        "workflow_guard": {"shared_repo_id": "repo-b", "target_branch": "main"},
                        "semantic_verifier": {
                            "required_merged_branches": ["worker/c"],
                            "expected_changed_paths": ["src/c.txt"],
                        },
                    },
                ),
            ]

        def parallel_worker_tasks(self, task_id, *, target_worker_count=None):
            del target_worker_count
            if task_id == "shared_repo_integrator_a":
                return [
                    TaskSpec(
                        task_id="shared_repo_worker_a",
                        prompt="shared repo worker a",
                        workspace_subdir="shared_repo_worker_a",
                        metadata={
                            "benchmark_family": "repo_sandbox",
                            "synthetic_worker": True,
                            "synthetic_edit_plan": [{"path": "src/a.txt"}],
                            "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/a"},
                        },
                    ),
                    TaskSpec(
                        task_id="shared_repo_worker_b",
                        prompt="shared repo worker b",
                        workspace_subdir="shared_repo_worker_b",
                        metadata={
                            "benchmark_family": "repo_sandbox",
                            "synthetic_worker": True,
                            "synthetic_edit_plan": [{"path": "src/b.txt"}],
                            "workflow_guard": {"shared_repo_id": "repo-a", "worker_branch": "worker/b"},
                        },
                    ),
                ]
            if task_id == "shared_repo_integrator_b":
                return [
                    TaskSpec(
                        task_id="shared_repo_worker_c",
                        prompt="shared repo worker c",
                        workspace_subdir="shared_repo_worker_c",
                        metadata={
                            "benchmark_family": "repo_sandbox",
                            "synthetic_worker": True,
                            "synthetic_edit_plan": [{"path": "src/c.txt"}],
                            "workflow_guard": {"shared_repo_id": "repo-b", "worker_branch": "worker/c"},
                        },
                    ),
                ]
            return []

    class FakeKernel:
        run_task_ids: list[str] = []
        run_task_repo_ids: list[str] = []
        run_task_surfaces: list[str] = []

        def __init__(self, config=None, policy=None):
            del config, policy

        def run_task(self, task, clean_workspace=True):
            del clean_workspace
            metadata = dict(task.metadata)
            workflow_guard = dict(metadata.get("workflow_guard", {}) or {})
            FakeKernel.run_task_ids.append(task.task_id)
            FakeKernel.run_task_repo_ids.append(str(workflow_guard.get("shared_repo_id", "")))
            FakeKernel.run_task_surfaces.append(str(metadata.get("long_horizon_coding_surface", "")))
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(tmp_path / "workspace" / task.workspace_subdir),
                success=True,
                steps=[],
                task_metadata=metadata,
                task_contract={"metadata": metadata},
                termination_reason="success",
            )

        def close(self):
            return None

    monkeypatch.setattr("evals.harness.TaskBank", FakeTaskBank)
    monkeypatch.setattr("evals.harness.AgentKernel", FakeKernel)

    metrics = run_eval(
        config=config,
        task_limit=3,
        priority_benchmark_families=["project"],
        prefer_low_cost_tasks=True,
    )

    assert metrics.total == 3
    assert FakeKernel.run_task_ids == [
        "shared_repo_worker_a",
        "shared_repo_worker_b",
        "shared_repo_integrator_a",
    ]
    assert FakeKernel.run_task_repo_ids == ["repo-a", "repo-a", "repo-a"]
    assert FakeKernel.run_task_surfaces.count("shared_repo_integrator") == 1
    assert FakeKernel.run_task_surfaces.count("shared_repo_synthetic_worker") == 2


def test_eval_emits_progress_for_generated_curriculum_phases(tmp_path, monkeypatch):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    _write_hello_skill(config)
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    metrics = run_eval(
        config=config,
        include_generated=True,
        include_failure_generated=True,
        progress_label="generated-progress",
        task_limit=1,
    )

    output = stream.getvalue()
    assert metrics.generated_total == metrics.total * 2
    assert "[eval:generated-progress] phase=generated_success_schedule" in output
    assert "[eval:generated-progress] phase=generated_success total=" in output
    assert "[eval:generated-progress] phase=generated_success task 1/" in output
    assert "family=" in output
    assert "[eval:generated-progress] phase=generated_failure_seed total=" in output
    assert "[eval:generated-progress] phase=generated_failure_seed task 1/" in output
    assert "[eval:generated-progress] phase=generated_failure total=" in output
    assert "[eval:generated-progress] phase=generated_failure task 1/" in output
    assert "[eval:generated-progress] phase=metrics_finalize start" in output
    assert "[eval:generated-progress] phase=metrics_finalize complete" in output
