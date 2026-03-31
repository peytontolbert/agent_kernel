import json
from io import StringIO
import sys
import threading
import time

from agent_kernel.config import KernelConfig
from agent_kernel.schemas import EpisodeRecord, StepRecord, TaskSpec
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
                    metadata={"benchmark_family": "workflow", "capability": "file_write"},
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
                        state_progress_delta=0.6,
                        state_regression_count=0,
                        state_transition={"no_progress": False},
                        proposal_metadata={
                            "hybrid_total_score": 4.0,
                            "hybrid_world_progress_score": 0.8,
                            "hybrid_world_risk_score": 0.2,
                            "hybrid_decoder_world_progress_score": 0.9,
                            "hybrid_decoder_world_risk_score": 0.1,
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
    step_feedback = metrics.task_trajectories["world_feedback_task"]["steps"][0]["world_feedback"]
    assert step_feedback["progress_signal"] == 0.9
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
                        command_result=None,
                        proposal_source="retrieval" if success else "",
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
