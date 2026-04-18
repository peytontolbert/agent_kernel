import hashlib
import json

from agent_kernel.config import KernelConfig
from agent_kernel.ops.loop_run_support import _task_semantic_recall_paths
from agent_kernel.tasking.curriculum import CurriculumEngine, EpisodeRecord
from agent_kernel.llm import MockLLMClient
from agent_kernel.loop import AgentKernel
from agent_kernel.policy import LLMDecisionPolicy, Policy
from agent_kernel.schemas import ActionDecision, CommandResult, ContextPacket, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.universe_model import UniverseModel
from agent_kernel.world_model import WorldModel
import pytest


def test_kernel_solves_seed_task(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    episode = AgentKernel(config=config).run_task(TaskBank().get("hello_task"))

    assert episode.success is True
    assert (config.workspace_root / "hello_task" / "hello.txt").exists()
    assert (config.trajectories_root / "hello_task.json").exists()
    assert episode.plan
    assert "document_count" in episode.graph_summary
    assert episode.universe_summary["stability"] == "stable"
    assert episode.universe_summary["governance_mode"] == "bounded_autonomous"
    assert episode.world_model_summary["expected_artifacts"] == ["hello.txt"]
    assert episode.world_model_summary["horizon"] == "bounded"
    assert episode.steps[0].acting_role in {"planner", "executor"}
    assert episode.steps[0].active_subgoal
    assert episode.steps[0].world_model_horizon == "bounded"


def test_kernel_accepts_tolbert_as_provider_name(tmp_path):
    config = KernelConfig(
        provider="tolbert",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("hello_task"))

    assert episode.success is True


def test_kernel_accepts_hybrid_as_provider_name(tmp_path):
    config = KernelConfig(
        provider="hybrid",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("hello_task"))

    assert episode.success is True


def test_kernel_solves_seed_task_with_executable_floor_only(tmp_path, monkeypatch):
    config = KernelConfig.executable_floor(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    seen: dict[str, bool] = {"called": False}

    def fail_learning_compile(*args, **kwargs):
        del args, kwargs
        seen["called"] = True
        raise AssertionError("executable floor should not compile learning candidates")

    monkeypatch.setattr("agent_kernel.loop.persist_episode_learning_candidates", fail_learning_compile)

    kernel = AgentKernel(config=config)
    try:
        episode = kernel.run_task(TaskBank().get("hello_task"))
    finally:
        kernel.close()

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert (config.workspace_root / "hello_task" / "hello.txt").exists()
    assert (config.trajectories_root / "hello_task.json").exists()
    assert episode.plan == []
    assert episode.graph_summary == {}
    assert episode.universe_summary == {}
    assert episode.world_model_summary == {}
    assert episode.steps[0].acting_role == "executor"
    assert episode.steps[0].active_subgoal == ""
    assert episode.steps[0].world_model_horizon == ""
    assert kernel.universe_model is None
    assert seen["called"] is False


def test_kernel_emits_structured_verification_failure_payload(tmp_path):
    class AlwaysFailPolicy(Policy):
        def decide(self, state):
            del state
            return ActionDecision(
                action="code_execute",
                content="false",
                thought="trigger verification failure",
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    events: list[dict[str, object]] = []
    task = TaskSpec(
        task_id="structured_verification_failure_task",
        prompt="Create hello.txt containing hello agent kernel.",
        workspace_subdir="structured_verification_failure_task",
        expected_files=["hello.txt"],
        max_steps=1,
    )

    episode = AgentKernel(config=config, policy=AlwaysFailPolicy()).run_task(
        task,
        progress_callback=lambda event: events.append(dict(event)),
    )

    verification = episode.steps[0].verification
    assert verification["passed"] is False
    assert verification["outcome_label"] == "command_failure"
    assert "command_failure" in verification["failure_codes"]
    assert "missing_expected_file" in verification["failure_codes"]
    verification_event = next(event for event in events if event.get("event") == "verification_result")
    assert verification_event["verification_outcome_label"] == "command_failure"
    assert "missing_expected_file" in verification_event["verification_failure_codes"]


def test_kernel_pre_execution_governance_gate_blocks_before_sandbox_run(tmp_path, monkeypatch):
    class UnsafePolicy(Policy):
        def decide(self, state):
            del state
            return ActionDecision(
                action="code_execute",
                content="git reset --hard HEAD",
                thought="unsafe reset",
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    events: list[dict[str, object]] = []
    sandbox_called = {"value": False}

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        sandbox_called["value"] = True
        raise AssertionError("sandbox.run should not execute when universe governance blocks the command")

    kernel = AgentKernel(config=config, policy=UnsafePolicy())
    monkeypatch.setattr(kernel.sandbox, "run", fail_if_called)

    episode = kernel.run_task(
        TaskSpec(
            task_id="governance_pre_execution_block_task",
            prompt="Do not reset the repository.",
            workspace_subdir="governance_pre_execution_block_task",
            expected_files=["hello.txt"],
            max_steps=1,
        ),
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert sandbox_called["value"] is False
    assert episode.success is False
    assert episode.steps[0].command_result["exit_code"] == 126
    assert episode.steps[0].verification["outcome_label"] == "governance_rejected"
    assert "governance_rejected" in episode.steps[0].verification["failure_codes"]
    assert "forbidden_pattern" in episode.steps[0].verification["failure_codes"]
    assert episode.steps[0].command_governance["blocked"] is True
    assert episode.steps[0].command_governance["block_reason"] == "forbidden_pattern"
    assert any(event.get("event") == "governance_rejected" for event in events)


def test_kernel_records_learned_world_signal_from_retained_tolbert_runtime(tmp_path, monkeypatch):
    bundle_manifest_path = tmp_path / "tolbert_bundle_manifest.json"
    bundle_manifest_path.write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "tolbert_model_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "bundle_manifest_path": str(bundle_manifest_path),
                    "preferred_device": "cpu",
                    "supports_world_model_surface": True,
                    "scoring_policy": {"long_horizon_progress_bonus_weight": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_tolbert_model_artifacts=True,
        tolbert_device="cpu",
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    captured: dict[str, object] = {}
    events: list[dict[str, object]] = []

    def fake_infer_hybrid_world_signal(*, state, bundle_manifest_path, device="cpu", scoring_policy=None):
        captured["task_id"] = state.task.task_id
        captured["bundle_manifest_path"] = bundle_manifest_path
        captured["device"] = device
        captured["scoring_policy"] = dict(scoring_policy or {})
        return {
            "source": "tolbert_hybrid_runtime",
            "progress_signal": 0.91,
            "risk_signal": 0.18,
            "controller_belief": {"recover": 0.2, "continue": 0.7, "stop": 0.1},
            "controller_mode": "continue",
            "controller_mode_probability": 0.7,
            "controller_expected_world_belief": [0.65, 0.25, 0.1],
            "controller_expected_world_top_states": [0, 1, 2],
            "controller_expected_world_top_state_probs": [0.65, 0.25, 0.1],
            "world_prior_backend": "profile_conditioned",
            "world_prior_top_state": 1,
            "world_prior_top_probability": 0.73,
            "world_transition_family": "banded",
            "world_transition_bandwidth": 2,
            "world_transition_gate": 0.61,
            "world_profile_horizons": [1, 2, 4],
        }

    monkeypatch.setattr("agent_kernel.loop.infer_hybrid_world_signal", fake_infer_hybrid_world_signal)

    episode = AgentKernel(config=config).run_task(
        TaskBank().get("hello_task"),
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert episode.success is True
    assert captured["task_id"] == "hello_task"
    assert captured["bundle_manifest_path"] == bundle_manifest_path
    assert captured["device"] == "cpu"
    assert captured["scoring_policy"]["long_horizon_progress_bonus_weight"] == 0.2
    assert episode.steps[0].latent_state_summary["learned_world_state"]["source"] == "tolbert_hybrid_runtime"
    assert episode.steps[0].latent_state_summary["learned_world_state"]["controller_mode"] == "continue"
    assert episode.steps[0].latent_state_summary["learned_world_state"]["controller_belief"]["continue"] == 0.7
    assert episode.steps[0].latent_state_summary["learned_world_state"]["controller_expected_world_belief"] == [
        0.65,
        0.25,
        0.1,
    ]
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_prior_backend"] == "profile_conditioned"
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_prior_top_state"] == 1
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_prior_top_probability"] == 0.73
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_transition_family"] == "banded"
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_transition_bandwidth"] == 2
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_transition_gate"] == 0.61
    assert episode.steps[0].latent_state_summary["learned_world_state"]["world_profile_horizons"] == [1, 2, 4]
    memory_event = next(event for event in events if event.get("event") == "memory_update_written")
    assert memory_event["learned_world_source"] == "tolbert_hybrid_runtime"
    assert memory_event["learned_world_controller_mode"] == "continue"
    assert memory_event["learned_world_controller_mode_probability"] == 0.7
    assert memory_event["learned_world_controller_belief"]["continue"] == 0.7
    assert memory_event["learned_world_expected_top_state"] == 0
    assert memory_event["learned_world_expected_top_state_probability"] == 0.65


def test_world_model_uses_semantic_verifier_preserved_paths(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    summary = kernel.world_model.summarize(TaskBank().get("git_repo_status_review_task"))

    assert "docs/context.md" in summary["preserved_artifacts"]
    assert "tests/check_status.sh" in summary["preserved_artifacts"]
    assert "src/app.py" not in summary["preserved_artifacts"]
    assert "docs/context.md" in summary["workflow_preserved_paths"]


def test_world_model_summarize_tracks_dynamic_workspace_progress(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    task = TaskBank().get("cleanup_task")
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "temp.txt").write_text("temporary\n", encoding="utf-8")
    snapshot = kernel.world_model.capture_workspace_snapshot(task, workspace)

    initial = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=snapshot,
    )

    assert initial["missing_expected_artifacts"] == ["status.txt"]
    assert initial["present_forbidden_artifacts"] == ["temp.txt"]
    assert initial["completion_ratio"] < 1.0

    (workspace / "status.txt").write_text("cleaned\n", encoding="utf-8")
    (workspace / "temp.txt").unlink()

    updated = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=snapshot,
    )

    assert updated["existing_expected_artifacts"] == ["status.txt"]
    assert updated["present_forbidden_artifacts"] == []
    assert updated["satisfied_expected_contents"] == ["status.txt"]
    assert updated["completion_ratio"] == 1.0


def test_world_model_summarize_includes_workspace_file_previews(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    task = TaskBank().get("rewrite_task")
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "note.txt").write_text("todo\n", encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    assert summary["workspace_file_previews"]["note.txt"]["content"] == "todo\n"
    assert summary["workspace_file_previews"]["note.txt"]["truncated"] is False
    assert summary["unsatisfied_expected_contents"] == ["note.txt"]


def test_world_model_preview_prioritizes_unsatisfied_expected_content_under_cap(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    expected_file_contents = {
        "a1.txt": "done 1\n",
        "a2.txt": "done 2\n",
        "a3.txt": "done 3\n",
        "a4.txt": "done 4\n",
        "a5.txt": "done 5\n",
        "a6.txt": "done 6\n",
        "z_target.txt": "done target\n",
    }
    task = TaskSpec(
        task_id="preview_priority_task",
        prompt="Repair the target file content without losing preview signal.",
        workspace_subdir="preview_priority_task",
        expected_files=list(expected_file_contents),
        expected_file_contents=expected_file_contents,
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    for path, content in expected_file_contents.items():
        payload = "todo target\n" if path == "z_target.txt" else content
        (workspace / path).write_text(payload, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    assert summary["unsatisfied_expected_contents"] == ["z_target.txt"]
    assert "z_target.txt" in summary["workspace_file_previews"]
    assert summary["workspace_file_previews"]["z_target.txt"]["content"] == "todo target\n"
    assert len(summary["workspace_file_previews"]) == 6


def test_world_model_preview_expands_priority_file_exact_budget(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current = "x" * 900 + "\n"
    task = TaskSpec(
        task_id="preview_priority_budget_task",
        prompt="Repair the large target file without losing exact preview coverage.",
        workspace_subdir="preview_priority_budget_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": current.replace("x\n", "y\n")},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    assert summary["unsatisfied_expected_contents"] == ["large_target.txt"]
    assert summary["workspace_file_previews"]["large_target.txt"]["content"] == current
    assert summary["workspace_file_previews"]["large_target.txt"]["truncated"] is False


def test_world_model_preview_expands_exact_budget_across_first_four_priority_files(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    large_payload = "x" * 900 + "\n"
    expected_file_contents = {
        "priority_1.txt": large_payload.replace("x\n", "a\n"),
        "priority_2.txt": large_payload.replace("x\n", "b\n"),
        "priority_3.txt": large_payload.replace("x\n", "c\n"),
        "priority_4.txt": large_payload.replace("x\n", "d\n"),
        "overflow.txt": large_payload.replace("x\n", "e\n"),
    }
    task = TaskSpec(
        task_id="preview_multi_priority_budget_task",
        prompt="Repair several large target files without losing exact preview coverage.",
        workspace_subdir="preview_multi_priority_budget_task",
        expected_files=list(expected_file_contents),
        expected_file_contents=expected_file_contents,
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    for path in expected_file_contents:
        (workspace / path).write_text(large_payload, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    previews = summary["workspace_file_previews"]
    assert previews["priority_1.txt"]["content"] == large_payload
    assert previews["priority_2.txt"]["content"] == large_payload
    assert previews["priority_3.txt"]["content"] == large_payload
    assert previews["priority_4.txt"]["content"] == large_payload
    assert previews["priority_1.txt"]["truncated"] is False
    assert previews["priority_2.txt"]["truncated"] is False
    assert previews["priority_3.txt"]["truncated"] is False
    assert previews["priority_4.txt"]["truncated"] is False
    assert previews["overflow.txt"]["truncated"] is True


def test_world_model_truncated_preview_exposes_safe_edit_window_metadata(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    line_a = ("a" * 550) + "\n"
    line_b = ("b" * 550) + "\n"
    line_c = ("c" * 550) + "\n"
    current = line_a + line_b + line_c
    task = TaskSpec(
        task_id="preview_window_metadata_task",
        prompt="Repair a large file while preserving a safe edit window.",
        workspace_subdir="preview_window_metadata_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": current.replace("b", "x", 1)},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    assert preview["truncated"] is True
    assert preview["content"].startswith(line_a + line_b)
    assert preview["content"] == preview["edit_content"]
    assert preview["edit_content"] == line_a + line_b
    assert preview["line_start"] == 1
    assert preview["line_end"] == 2
    assert preview["target_line_start"] == 1
    assert preview["target_line_end"] == 2
    assert preview["line_delta"] == 0
    assert preview["omitted_prefix_sha1"] == hashlib.sha1("".encode("utf-8")).hexdigest()
    assert preview["omitted_suffix_sha1"] == hashlib.sha1(line_c.encode("utf-8")).hexdigest()
    assert preview["omitted_sha1"] == hashlib.sha1(line_c.encode("utf-8")).hexdigest()


def test_world_model_targets_mid_file_preview_window_for_large_unsatisfied_file(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current_lines = [f"line-{index:02d} {'a' * 60}\n" for index in range(1, 31)]
    target_lines = list(current_lines)
    target_lines[14] = f"line-15 {'b' * 60}\n"
    current = "".join(current_lines)
    target = "".join(target_lines)
    task = TaskSpec(
        task_id="mid_file_preview_window_task",
        prompt="Repair a large file with a centered preview window.",
        workspace_subdir="mid_file_preview_window_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": target},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    assert preview["truncated"] is True
    assert preview["line_start"] > 1
    assert preview["line_start"] <= 15 <= preview["line_end"]
    assert preview["target_line_start"] > 1
    assert preview["target_line_start"] <= 15 <= preview["target_line_end"]
    assert preview["line_delta"] == 0
    assert "line-15 " in preview["content"]
    assert preview["content"] == preview["edit_content"]
    assert preview["omitted_prefix_sha1"] == hashlib.sha1(
        "".join(current_lines[: preview["line_start"] - 1]).encode("utf-8")
    ).hexdigest()
    assert preview["omitted_suffix_sha1"] == hashlib.sha1(
        "".join(current_lines[preview["line_end"] :]).encode("utf-8")
    ).hexdigest()


def test_world_model_exposes_multiple_retained_preview_windows_for_distant_edits(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current_lines = [f"line-{index:02d} {'a' * 60}\n" for index in range(1, 61)]
    target_lines = list(current_lines)
    target_lines[9] = f"line-10 {'b' * 60}\n"
    target_lines[49] = f"line-50 {'c' * 60}\n"
    task = TaskSpec(
        task_id="multi_window_preview_task",
        prompt="Repair two distant localized edits without losing either retained preview window.",
        workspace_subdir="multi_window_preview_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": "".join(target_lines)},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text("".join(current_lines), encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    assert preview["truncated"] is True
    assert len(windows) == 2
    assert windows[0]["line_start"] <= 10 <= windows[0]["line_end"]
    assert windows[1]["line_start"] <= 50 <= windows[1]["line_end"]
    assert "line-10 " in windows[0]["content"]
    assert "line-50 " in windows[1]["content"]


def test_world_model_exposes_multiple_edit_windows_for_distant_large_file_edits(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current_lines = [f"line-{index:02d} {'a' * 60}\n" for index in range(1, 31)]
    target_lines = list(current_lines)
    target_lines[4] = f"line-05 {'b' * 60}\n"
    target_lines[24] = f"line-25 {'c' * 60}\n"
    current = "".join(current_lines)
    target = "".join(target_lines)
    task = TaskSpec(
        task_id="multi_window_preview_task",
        prompt="Repair two distant edits without collapsing to one preview window.",
        workspace_subdir="multi_window_preview_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": target},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    assert len(windows) >= 2
    assert windows[0]["line_start"] <= 5 <= windows[0]["line_end"]
    assert windows[1]["line_start"] <= 25 <= windows[1]["line_end"]
    assert windows[0]["target_edit_content"] != windows[0]["edit_content"]
    assert windows[1]["target_edit_content"] != windows[1]["edit_content"]
    assert windows[0]["line_delta"] == 0
    assert windows[1]["line_delta"] == 0


def test_world_model_exposes_hidden_gap_bridge_metadata_for_consecutive_retained_windows(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    unchanged_prefix = "prefix=ok\n"
    changed_left = "alpha=" + ("a" * 1590) + "\n"
    hidden_gap = "keep=" + ("k" * 40) + "\n"
    changed_right = "gamma=" + ("c" * 1570) + "\n"
    unchanged_suffix = "suffix=ok\n"
    current = "".join([unchanged_prefix, changed_left, hidden_gap, changed_right, unchanged_suffix])
    target = "".join(
        [
            unchanged_prefix,
            changed_left.replace("a", "x", 1),
            hidden_gap,
            changed_right.replace("c", "z", 1),
            unchanged_suffix,
        ]
    )
    task = TaskSpec(
        task_id="hidden_gap_bridge_preview_task",
        prompt="Emit explicit hidden-gap proof when consecutive retained windows cannot be merged into one preview.",
        workspace_subdir="hidden_gap_bridge_preview_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": target},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    bridges = preview["bridged_edit_windows"]
    bridge_runs = preview["bridged_edit_window_runs"]
    assert len(windows) == 2
    assert len(bridges) == 1
    assert len(bridge_runs) == 1
    bridge = bridges[0]
    bridge_run = bridge_runs[0]
    assert bridge["bridge_window_indices"] == [0, 1]
    assert bridge["explicit_hidden_gap_current_proof"] is True
    assert bridge["hidden_gap_current_line_count"] == 1
    assert bridge["hidden_gap_target_line_count"] == 1
    assert bridge["hidden_gap_current_content"] == hidden_gap
    assert bridge["hidden_gap_target_content"] == hidden_gap
    assert bridge["line_start"] == windows[0]["line_start"]
    assert bridge["line_end"] == windows[1]["line_end"]
    assert bridge["target_line_start"] == windows[0]["target_line_start"]
    assert bridge["target_line_end"] == windows[1]["target_line_end"]
    assert bridge_run["bridge_window_indices"] == [0, 1]
    assert bridge_run["hidden_gap_current_line_count"] == 1
    assert bridge_run["hidden_gap_target_line_count"] == 1
    assert bridge_run["line_start"] == windows[0]["line_start"]
    assert bridge_run["line_end"] == windows[1]["line_end"]
    assert bridge_run["target_line_start"] == windows[0]["target_line_start"]
    assert bridge_run["target_line_end"] == windows[1]["target_line_end"]
    assert len(bridge_run["bridge_segments"]) == 1
    assert bridge_run["bridge_segments"][0]["bridge_window_indices"] == [0, 1]
    assert bridge_run["bridge_segments"][0]["hidden_gap_current_content"] == hidden_gap
    assert bridge_run["bridge_segments"][0]["hidden_gap_target_content"] == hidden_gap


def test_world_model_compacts_consecutive_hidden_gap_bridges_into_one_maximal_run(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    unchanged_prefix = "prefix=ok\n"
    changed_left = "alpha=" + ("a" * 1590) + "\n"
    hidden_gap_left = "keep_left\n"
    changed_middle = "beta=" + ("b" * 1590) + "\n"
    hidden_gap_right = "keep_right\n"
    changed_right = "gamma=" + ("c" * 1590) + "\n"
    unchanged_suffix = "suffix=ok\n"
    current = "".join(
        [
            unchanged_prefix,
            changed_left,
            hidden_gap_left,
            changed_middle,
            hidden_gap_right,
            changed_right,
            unchanged_suffix,
        ]
    )
    target = "".join(
        [
            unchanged_prefix,
            changed_left.replace("a", "x", 1),
            hidden_gap_left,
            changed_middle.replace("b", "y", 1),
            hidden_gap_right,
            changed_right.replace("c", "z", 1),
            unchanged_suffix,
        ]
    )
    task = TaskSpec(
        task_id="hidden_gap_bridge_run_preview_task",
        prompt="Compact consecutive explicit hidden-gap bridges into one maximal run payload.",
        workspace_subdir="hidden_gap_bridge_run_preview_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": target},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    bridges = preview["bridged_edit_windows"]
    bridge_runs = preview["bridged_edit_window_runs"]
    assert len(windows) == 3
    assert len(bridges) == 2
    assert len(bridge_runs) == 1
    bridge_run = bridge_runs[0]
    assert bridge_run["bridge_window_indices"] == [0, 1, 2]
    assert bridge_run["line_start"] == windows[0]["line_start"]
    assert bridge_run["line_end"] == windows[2]["line_end"]
    assert bridge_run["target_line_start"] == windows[0]["target_line_start"]
    assert bridge_run["target_line_end"] == windows[2]["target_line_end"]
    assert bridge_run["hidden_gap_current_line_count"] == 2
    assert bridge_run["hidden_gap_target_line_count"] == 2
    assert bridge_run["line_delta"] == 0
    assert bridge_run["explicit_hidden_gap_current_proof"] is True
    assert len(bridge_run["bridge_segments"]) == 2
    assert bridge_run["bridge_segments"][0]["bridge_window_indices"] == [0, 1]
    assert bridge_run["bridge_segments"][1]["bridge_window_indices"] == [1, 2]
    assert bridge_run["bridge_segments"][0]["hidden_gap_target_content"] == hidden_gap_left
    assert bridge_run["bridge_segments"][1]["hidden_gap_target_content"] == hidden_gap_right


def test_world_model_emits_exact_edit_window_proofs_for_omitted_windows():
    windows = [
        (0, 1, 0, 1),
        (2, 3, 2, 3),
        (4, 5, 4, 5),
        (6, 7, 6, 7),
    ]

    proofs = WorldModel._exact_targeted_preview_proof_windows(
        windows=windows,
        retained_window_count=3,
    )

    assert len(proofs) == 1
    proof = proofs[0]
    assert proof["window_index"] == 3
    assert proof["explicit_current_span_proof"] is True
    assert proof["line_start"] == 7
    assert proof["line_end"] == 7
    assert proof["target_line_start"] == 7
    assert proof["target_line_end"] == 7
    assert proof["current_line_count"] == 1
    assert proof["target_line_count"] == 1
    assert proof["line_delta"] == 0


def test_world_model_exposes_current_only_hidden_gap_bridge_when_target_gap_payload_exceeds_budget():
    current_lines = [
        "prefix=ok\n",
        "alpha=1\n",
        "old_gap\n",
        "beta=1\n",
        "suffix=ok\n",
    ]
    inserted_target_lines = [f"add_{index:03d}=z\n" for index in range(1, 121)]
    expected_lines = [
        "prefix=ok\n",
        "alpha=2\n",
        *inserted_target_lines,
        "beta=2\n",
        "suffix=ok\n",
    ]
    windows = [
        (1, 2, 1, 2),
        (3, 4, 122, 123),
    ]

    bridges = WorldModel._bridged_targeted_preview_windows(
        current_lines=current_lines,
        expected_lines=expected_lines,
        windows=windows,
        max_bytes=1024,
        max_chars=256,
    )
    bridge_runs = WorldModel._bridged_targeted_preview_window_runs(bridged_windows=bridges)

    assert len(bridges) == 1
    assert len(bridge_runs) == 1
    bridge = bridges[0]
    bridge_run = bridge_runs[0]
    assert bridge["bridge_window_indices"] == [0, 1]
    assert bridge["explicit_hidden_gap_current_proof"] is True
    assert bridge["hidden_gap_current_content"] == "old_gap\n"
    assert bridge["hidden_gap_target_content"] == ""
    assert bridge["hidden_gap_target_from_expected_content"] is True
    assert bridge["hidden_gap_current_line_count"] == 1
    assert bridge["hidden_gap_target_line_count"] == len(inserted_target_lines)
    assert bridge_run["hidden_gap_target_from_expected_content"] is True
    assert bridge_run["bridge_segments"][0]["hidden_gap_target_from_expected_content"] is True


def test_world_model_exposes_proof_only_hidden_gap_bridge_when_current_gap_payload_exceeds_budget():
    current_gap_lines = [f"old_gap_{index:03d}=x\n" for index in range(1, 41)]
    target_gap_lines = [f"new_gap_{index:03d}=y\n" for index in range(1, 41)]
    current_lines = [
        "prefix=ok\n",
        "alpha=1\n",
        *current_gap_lines,
        "omega=1\n",
        "suffix=ok\n",
    ]
    expected_lines = [
        "prefix=ok\n",
        "alpha=2\n",
        *target_gap_lines,
        "omega=2\n",
        "suffix=ok\n",
    ]
    windows = [
        (1, 2, 1, 2),
        (42, 43, 42, 43),
    ]

    bridges = WorldModel._bridged_targeted_preview_windows(
        current_lines=current_lines,
        expected_lines=expected_lines,
        windows=windows,
        max_bytes=1024,
        max_chars=256,
    )
    bridge_runs = WorldModel._bridged_targeted_preview_window_runs(bridged_windows=bridges)

    assert len(bridges) == 1
    assert len(bridge_runs) == 1
    bridge = bridges[0]
    bridge_run = bridge_runs[0]
    assert bridge["bridge_window_indices"] == [0, 1]
    assert bridge["explicit_hidden_gap_current_proof"] is True
    assert bridge["hidden_gap_current_content"] == ""
    assert bridge["hidden_gap_target_content"] == ""
    assert bridge["hidden_gap_current_from_line_span_proof"] is True
    assert bridge["hidden_gap_target_from_expected_content"] is True
    assert bridge["hidden_gap_current_line_count"] == len(current_gap_lines)
    assert bridge["hidden_gap_target_line_count"] == len(target_gap_lines)
    assert bridge_run["hidden_gap_current_from_line_span_proof"] is True
    assert bridge_run["hidden_gap_target_from_expected_content"] is True
    assert bridge_run["bridge_segments"][0]["hidden_gap_current_from_line_span_proof"] is True
    assert bridge_run["bridge_segments"][0]["hidden_gap_target_from_expected_content"] is True


def test_world_model_extends_hidden_gap_bridge_runs_across_omitted_exact_proof_windows(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    unchanged_prefix = "prefix=ok\n"
    hidden_gap_a = "keep_a\n"
    hidden_gap_b = "keep_b\n"
    hidden_gap_c = "keep_c\n"
    changed_lines = [
        "alpha=" + ("a" * 1590) + "\n",
        "beta=" + ("b" * 1590) + "\n",
        "gamma=" + ("c" * 1590) + "\n",
        "delta=" + ("d" * 1590) + "\n",
    ]
    current = "".join(
        [
            unchanged_prefix,
            changed_lines[0],
            hidden_gap_a,
            changed_lines[1],
            hidden_gap_b,
            changed_lines[2],
            hidden_gap_c,
            changed_lines[3],
            "suffix=ok\n",
        ]
    )
    target = "".join(
        [
            unchanged_prefix,
            changed_lines[0].replace("a", "x", 1),
            hidden_gap_a,
            changed_lines[1].replace("b", "y", 1),
            hidden_gap_b,
            changed_lines[2].replace("c", "z", 1),
            hidden_gap_c,
            changed_lines[3].replace("d", "w", 1),
            "suffix=ok\n",
        ]
    )
    task = TaskSpec(
        task_id="hidden_gap_bridge_with_omitted_proof_task",
        prompt="Keep bridge-run proof alive across an omitted exact diff window.",
        workspace_subdir="hidden_gap_bridge_with_omitted_proof_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": target},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text(current, encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    exact_proofs = preview["exact_edit_window_proofs"]
    bridges = preview["bridged_edit_windows"]
    bridge_runs = preview["bridged_edit_window_runs"]
    assert len(windows) == 3
    assert [window["window_index"] for window in windows] == [0, 1, 2]
    assert len(exact_proofs) == 1
    assert exact_proofs[0]["window_index"] == 3
    assert len(bridges) == 3
    assert [bridge["bridge_window_indices"] for bridge in bridges] == [[0, 1], [1, 2], [2, 3]]
    assert len(bridge_runs) == 1
    bridge_run = bridge_runs[0]
    assert bridge_run["bridge_window_indices"] == [0, 1, 2, 3]
    assert len(bridge_run["bridge_segments"]) == 3
    assert bridge_run["bridge_segments"][2]["bridge_window_indices"] == [2, 3]
    assert bridge_run["bridge_segments"][2]["hidden_gap_current_content"] == hidden_gap_c
    assert bridge_run["bridge_segments"][2]["hidden_gap_target_content"] == hidden_gap_c
    proof_regions = preview["hidden_gap_current_proof_regions"]
    assert len(proof_regions) == 1
    proof_region = proof_regions[0]
    assert proof_region["window_indices"] == [0, 1, 2, 3]
    assert proof_region["current_proof_span_count"] == 3
    assert [
        (span["current_line_start"], span["current_line_end"])
        for span in proof_region["current_proof_spans"]
    ] == [(3, 3), (5, 5), (7, 7)]


def test_world_model_marks_partial_hidden_gap_current_proof_regions():
    proof_regions = WorldModel._hidden_gap_current_proof_regions(
        bridged_runs=[
            {
                "bridge_window_indices": [0, 1, 2, 3],
                "line_start": 1,
                "line_end": 7,
                "target_line_start": 1,
                "target_line_end": 7,
                "truncated": True,
                "explicit_hidden_gap_current_proof": False,
                "hidden_gap_current_from_line_span_proof": False,
                "hidden_gap_target_from_expected_content": True,
                "bridge_segments": [
                    {
                        "hidden_gap_current_line_start": 2,
                        "hidden_gap_current_line_end": 2,
                        "hidden_gap_target_line_start": 2,
                        "hidden_gap_target_line_end": 2,
                        "hidden_gap_current_content": "",
                        "hidden_gap_target_content": "keep_a\n",
                        "hidden_gap_current_from_line_span_proof": True,
                        "hidden_gap_target_from_expected_content": True,
                    },
                    {
                        "hidden_gap_current_line_start": 4,
                        "hidden_gap_current_line_end": 4,
                        "hidden_gap_target_line_start": 4,
                        "hidden_gap_target_line_end": 4,
                        "hidden_gap_current_content": "",
                        "hidden_gap_target_content": "keep_b\n",
                        "hidden_gap_current_from_line_span_proof": False,
                        "hidden_gap_target_from_expected_content": True,
                    },
                    {
                        "hidden_gap_current_line_start": 6,
                        "hidden_gap_current_line_end": 6,
                        "hidden_gap_target_line_start": 6,
                        "hidden_gap_target_line_end": 6,
                        "hidden_gap_current_content": "",
                        "hidden_gap_target_content": "keep_c\n",
                        "hidden_gap_current_from_line_span_proof": True,
                        "hidden_gap_target_from_expected_content": True,
                    },
                ],
            }
        ]
    )

    assert len(proof_regions) == 1
    proof_region = proof_regions[0]
    assert proof_region["current_proof_complete"] is False
    assert proof_region["current_proof_partial_coverage"] is True
    assert proof_region["current_proof_covered_line_count"] == 2
    assert proof_region["current_proof_missing_line_count"] == 1
    assert proof_region["current_proof_missing_span_count"] == 1
    assert proof_region["current_proof_opaque_span_count"] == 1
    assert proof_region["current_proof_opaque_spans"] == [
        {
            "current_line_start": 4,
            "current_line_end": 4,
            "target_line_start": 4,
            "target_line_end": 4,
            "reason": "missing_current_proof",
        }
    ]


def test_world_model_builds_sparse_current_proof_region_across_nonbridge_joins():
    windows = [
        (0, 1, 0, 1),
        (1, 2, 1, 2),
        (3, 4, 3, 4),
        (4, 5, 4, 5),
        (6, 7, 6, 7),
    ]
    bridges = [
        {
            "bridge_window_indices": [1, 2],
            "line_start": 2,
            "line_end": 4,
            "target_line_start": 2,
            "target_line_end": 4,
            "hidden_gap_current_line_start": 3,
            "hidden_gap_current_line_end": 3,
            "hidden_gap_target_line_start": 3,
            "hidden_gap_target_line_end": 3,
            "hidden_gap_current_content": "keep_a\n",
            "hidden_gap_target_content": "keep_a\n",
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "line_delta": 0,
            "explicit_hidden_gap_current_proof": True,
        },
        {
            "bridge_window_indices": [3, 4],
            "line_start": 5,
            "line_end": 7,
            "target_line_start": 5,
            "target_line_end": 7,
            "hidden_gap_current_line_start": 6,
            "hidden_gap_current_line_end": 6,
            "hidden_gap_target_line_start": 6,
            "hidden_gap_target_line_end": 6,
            "hidden_gap_current_content": "keep_b\n",
            "hidden_gap_target_content": "keep_b\n",
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "line_delta": 0,
            "explicit_hidden_gap_current_proof": True,
        },
    ]

    proof_regions = WorldModel._sparse_hidden_gap_current_proof_regions(
        windows=windows,
        bridged_windows=bridges,
    )

    assert len(proof_regions) == 1
    proof_region = proof_regions[0]
    assert proof_region["window_indices"] == [0, 1, 2, 3, 4]
    assert proof_region["line_start"] == 1
    assert proof_region["line_end"] == 7
    assert proof_region["current_proof_span_count"] == 2
    assert proof_region["current_proof_complete"] is True
    assert proof_region["current_proof_partial_coverage"] is False
    assert proof_region["current_proof_covered_line_count"] == 2
    assert proof_region["current_proof_missing_line_count"] == 0
    assert proof_region["current_proof_opaque_span_count"] == 0
    assert proof_region["current_proof_opaque_spans"] == []


def test_world_model_builds_sparse_partial_current_proof_region_with_opaque_subspan():
    windows = [
        (0, 1, 0, 1),
        (2, 3, 2, 3),
        (4, 5, 4, 5),
        (6, 7, 6, 7),
    ]
    bridges = [
        {
            "bridge_window_indices": [0, 1],
            "line_start": 1,
            "line_end": 3,
            "target_line_start": 1,
            "target_line_end": 3,
            "hidden_gap_current_line_start": 2,
            "hidden_gap_current_line_end": 2,
            "hidden_gap_target_line_start": 2,
            "hidden_gap_target_line_end": 2,
            "hidden_gap_current_content": "keep_a\n",
            "hidden_gap_target_content": "keep_a\n",
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "line_delta": 0,
            "explicit_hidden_gap_current_proof": True,
        },
        {
            "bridge_window_indices": [2, 3],
            "line_start": 5,
            "line_end": 7,
            "target_line_start": 5,
            "target_line_end": 7,
            "hidden_gap_current_line_start": 6,
            "hidden_gap_current_line_end": 6,
            "hidden_gap_target_line_start": 6,
            "hidden_gap_target_line_end": 6,
            "hidden_gap_current_content": "keep_b\n",
            "hidden_gap_target_content": "keep_b\n",
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "line_delta": 0,
            "explicit_hidden_gap_current_proof": True,
        },
    ]

    proof_regions = WorldModel._sparse_hidden_gap_current_proof_regions(
        windows=windows,
        bridged_windows=bridges,
    )

    assert len(proof_regions) == 1
    proof_region = proof_regions[0]
    assert proof_region["window_indices"] == [0, 1, 2, 3]
    assert proof_region["current_proof_complete"] is False
    assert proof_region["current_proof_partial_coverage"] is True
    assert proof_region["current_proof_covered_line_count"] == 2
    assert proof_region["current_proof_missing_line_count"] == 1
    assert proof_region["current_proof_opaque_span_count"] == 1
    assert proof_region["current_proof_opaque_spans"] == [
        {
            "current_line_start": 4,
            "current_line_end": 4,
            "target_line_start": 4,
            "target_line_end": 4,
            "reason": "no_adjacent_pair_bridge",
        }
    ]


def test_world_model_marks_partial_retained_window_coverage_when_change_clusters_exceed_cap(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current_lines = [f"line-{index:02d} {'a' * 60}\n" for index in range(1, 41)]
    target_lines = list(current_lines)
    target_lines[4] = f"line-05 {'b' * 60}\n"
    target_lines[14] = f"line-15 {'c' * 60}\n"
    target_lines[24] = f"line-25 {'d' * 60}\n"
    target_lines[34] = f"line-35 {'e' * 60}\n"
    task = TaskSpec(
        task_id="partial_window_coverage_task",
        prompt="Expose when only a subset of distant change windows can be retained.",
        workspace_subdir="partial_window_coverage_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": "".join(target_lines)},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text("".join(current_lines), encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    windows = preview["edit_windows"]
    assert len(windows) == 3
    assert preview["retained_edit_window_count"] == 3
    assert preview["total_edit_window_count"] == 4
    assert preview["partial_window_coverage"] is True
    assert windows[0]["retained_edit_window_count"] == 3
    assert windows[0]["total_edit_window_count"] == 4
    assert windows[0]["partial_window_coverage"] is True


def test_world_model_exposes_target_span_and_line_delta_for_insert_window(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    current_lines = [f"line-{index:02d} {'a' * 60}\n" for index in range(1, 31)]
    target_lines = list(current_lines)
    target_lines.insert(14, f"line-15b {'b' * 60}\n")
    task = TaskSpec(
        task_id="insert_preview_window_task",
        prompt="Expose target span metadata for an inserted retained window.",
        workspace_subdir="insert_preview_window_task",
        expected_files=["large_target.txt"],
        expected_file_contents={"large_target.txt": "".join(target_lines)},
    )
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "large_target.txt").write_text("".join(current_lines), encoding="utf-8")

    summary = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=kernel.world_model.capture_workspace_snapshot(task, workspace),
    )

    preview = summary["workspace_file_previews"]["large_target.txt"]
    assert preview["truncated"] is True
    assert preview["line_start"] <= 15 <= preview["line_end"]
    assert preview["target_line_start"] <= 15 <= preview["target_line_end"]
    current_visible_lines = preview["line_end"] - preview["line_start"] + 1
    target_visible_lines = preview["target_line_end"] - preview["target_line_start"] + 1
    assert preview["line_delta"] == target_visible_lines - current_visible_lines


def test_kernel_build_plan_applies_retained_planner_controls(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planner_controls": {
                    "prepend_verifier_contract_check": True,
                    "append_validation_subgoal": True,
                    "prefer_expected_artifacts_first": True,
                    "max_initial_subgoals": 3,
                },
                "role_directives": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        prompt_proposals_path=prompt_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(
        TaskSpec(
            task_id="plan_task",
            prompt="create hello.txt and avoid bad.txt",
            workspace_subdir="plan_task",
            expected_files=["hello.txt"],
            forbidden_files=["bad.txt"],
        )
    )

    assert plan[0] == "check verifier contract before terminating"
    assert plan[-1] == "validate expected artifacts and forbidden artifacts before termination"
    assert len(plan) == 3


def test_kernel_build_plan_includes_preserved_artifact_steps(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_status_review_task"))

    assert any(step == "preserve required artifact docs/context.md" for step in plan)
    assert any(step == "preserve required artifact tests/check_status.sh" for step in plan)


def test_kernel_advances_active_subgoal_from_workspace_state(tmp_path):
    class SubgoalPolicy(Policy):
        def decide(self, state):
            subgoal = state.active_subgoal
            if subgoal == "materialize expected artifact status.txt":
                return ActionDecision(
                    thought="create expected artifact",
                    action="code_execute",
                    content="printf 'cleaned\\n' > status.txt",
                )
            if subgoal == "remove forbidden artifact temp.txt":
                return ActionDecision(
                    thought="remove forbidden artifact",
                    action="code_execute",
                    content="rm -f temp.txt",
                )
            return ActionDecision(
                thought="validate",
                action="code_execute",
                content="test -f status.txt && test ! -f temp.txt",
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=4,
    )
    task = TaskBank().get("cleanup_task")
    episode = AgentKernel(config=config, policy=SubgoalPolicy()).run_task(task)

    assert episode.success is True
    assert len(episode.steps) == 2
    assert episode.steps[0].active_subgoal == "materialize expected artifact status.txt"
    assert episode.steps[1].active_subgoal == "remove forbidden artifact temp.txt"


def test_kernel_refreshes_planner_subgoals_from_learned_world_hotspots(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.plan = [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
        "validate expected artifacts and forbidden artifacts before termination",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "changed_preserved_artifacts": [],
        "preserved_artifacts": [],
        "existing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latent_state_summary = {
        "active_paths": ["temp.txt", "status.txt"],
        "learned_world_state": {
            "progress_signal": 0.22,
            "risk_signal": 0.89,
            "world_risk_score": 0.84,
        },
    }

    kernel._refresh_planner_subgoals(state)

    assert state.active_subgoal == "remove forbidden artifact temp.txt"
    assert state.plan[:2] == [
        "remove forbidden artifact temp.txt",
        "materialize expected artifact status.txt",
    ]


def test_kernel_keeps_existing_planner_subgoal_when_learned_world_risk_is_low(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.plan = [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "changed_preserved_artifacts": [],
        "preserved_artifacts": [],
        "existing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latent_state_summary = {
        "active_paths": ["temp.txt", "status.txt"],
        "learned_world_state": {
            "progress_signal": 0.76,
            "risk_signal": 0.34,
        },
    }

    kernel._refresh_planner_subgoals(state)

    assert state.active_subgoal == "materialize expected artifact status.txt"
    assert state.plan == [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
    ]


def test_kernel_refreshes_planner_subgoals_from_pending_workflow_hotspots_without_active_paths(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.plan = [
        "prepare workflow branch fix/release-ready",
        "update workflow path src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "run workflow test release test script",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_generated_paths": [],
        "updated_workflow_paths": [],
        "updated_report_paths": [],
        "updated_generated_paths": [],
        "existing_expected_artifacts": [],
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "missing_preserved_artifacts": [],
    }
    state.latest_state_transition = {
        "no_progress": True,
        "regressions": [],
    }
    state.latent_state_summary = {
        "active_paths": [],
        "learned_world_state": {
            "progress_signal": 0.11,
            "risk_signal": 0.4,
        },
    }

    kernel._refresh_planner_subgoals(state)

    assert state.active_subgoal == "update workflow path src/release_state.txt"
    assert state.plan[:3] == [
        "update workflow path src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "prepare workflow branch fix/release-ready",
    ]


def test_kernel_promotes_executor_to_planner_for_moderate_risk_long_horizon_hotspots(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.current_role = "executor"
    state.history = [
        StepRecord(
            index=1,
            thought="stalled on workflow repair",
            action="code_execute",
            content="git status --short",
            selected_skill_id=None,
            command_result={
                "command": "git status --short",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["release path still pending"]},
        )
    ]
    state.consecutive_no_progress_steps = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_generated_paths": [],
        "updated_workflow_paths": [],
        "updated_report_paths": [],
        "updated_generated_paths": [],
        "existing_expected_artifacts": [],
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "missing_preserved_artifacts": [],
    }
    state.latest_state_transition = {
        "no_progress": True,
        "regressions": [],
    }
    state.latent_state_summary = {
        "active_paths": [],
        "learned_world_state": {
            "progress_signal": 0.12,
            "risk_signal": 0.4,
        },
    }

    assert kernel._resolve_role_before_decision(state) == "planner"
    assert kernel._resolve_role_after_step(state, verification_passed=False) == "critic"


def test_kernel_uses_graph_memory_pressure_to_promote_long_horizon_recovery_role(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.current_role = "executor"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "updated_workflow_paths": [],
        "updated_report_paths": [],
        "updated_generated_paths": [],
        "existing_expected_artifacts": [],
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "missing_preserved_artifacts": [],
    }
    state.graph_summary = {
        "failure_signals": {
            "no_state_progress": 2,
            "state_regression": 1,
        },
        "environment_alignment_failures": {
            "git_write_aligned": 2,
        },
    }
    state.latent_state_summary = {
        "active_paths": [],
        "learned_world_state": {
            "progress_signal": 0.2,
            "risk_signal": 0.1,
        },
    }

    assert kernel._resolve_role_before_decision(state) == "critic"


def test_kernel_promotes_long_horizon_hotspot_recovery_to_critic_under_repeated_pressure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.current_role = "planner"
    state.history = [
        StepRecord(
            index=1,
            thought="first failed workflow repair",
            action="code_execute",
            content="python scripts/release_check.py",
            selected_skill_id=None,
            command_result={
                "command": "python scripts/release_check.py",
                "exit_code": 1,
                "stdout": "",
                "stderr": "failed",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["release path still pending"]},
        )
    ]
    state.consecutive_failures = 2
    state.repeated_action_count = 2
    state.consecutive_no_progress_steps = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "workflow_report_paths": [],
        "workflow_generated_paths": [],
        "updated_workflow_paths": [],
        "updated_report_paths": [],
        "updated_generated_paths": [],
        "existing_expected_artifacts": [],
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "missing_preserved_artifacts": [],
    }
    state.latest_state_transition = {
        "no_progress": True,
        "regressions": [],
    }
    state.latent_state_summary = {
        "active_paths": [],
        "learned_world_state": {
            "progress_signal": 0.09,
            "risk_signal": 0.42,
        },
    }

    assert kernel._resolve_role_before_decision(state) == "critic"
    assert kernel._resolve_role_after_step(state, verification_passed=False) == "critic"


def test_kernel_critic_diagnoses_guide_planner_hotspot_refresh(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.plan = [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
        "validate expected artifacts and forbidden artifacts before termination",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "changed_preserved_artifacts": [],
        "preserved_artifacts": [],
        "existing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latest_state_transition = {
        "regressions": ["temp.txt"],
        "no_progress": True,
    }
    state.latent_state_summary = {
        "active_paths": ["status.txt", "temp.txt"],
        "learned_world_state": {
            "progress_signal": 0.12,
            "risk_signal": 0.93,
            "world_risk_score": 0.88,
        },
    }

    kernel._attach_critic_subgoal_diagnoses(
        state,
        step_index=3,
        step_active_subgoal="materialize expected artifact status.txt",
        failure_signals=["state_regression", "no_state_progress"],
        failure_origin="",
        command_result=CommandResult(
            command="printf 'status\\n' > status.txt",
            exit_code=1,
            stdout="",
            stderr="permission denied",
            timed_out=False,
        ),
    )

    remove_diagnosis = state.subgoal_diagnoses["remove forbidden artifact temp.txt"]
    materialize_diagnosis = state.subgoal_diagnoses["materialize expected artifact status.txt"]

    assert remove_diagnosis["source_role"] == "critic"
    assert "temp.txt is still present" in remove_diagnosis["summary"]
    assert "recent step regressed workspace state" in remove_diagnosis["summary"]
    assert "state_regression" in remove_diagnosis["signals"]
    assert "command_failure" not in remove_diagnosis["signals"]
    assert "status.txt is still missing" in materialize_diagnosis["summary"]
    assert "recent command exited 1" in materialize_diagnosis["summary"]
    assert "command_failure" in materialize_diagnosis["signals"]
    assert "state_regression" not in materialize_diagnosis["signals"]

    kernel._refresh_planner_subgoals(state)

    assert state.active_subgoal == "remove forbidden artifact temp.txt"
    assert state.plan[:2] == [
        "remove forbidden artifact temp.txt",
        "materialize expected artifact status.txt",
    ]


def test_kernel_refreshes_planner_subgoals_from_verifier_failures(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.plan = [
        "prepare workflow branch fix/release-ready",
        "accept required branch worker/release-ready",
        "update workflow path src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "run workflow test release test script",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "updated_workflow_paths": [],
        "updated_report_paths": [],
        "existing_expected_artifacts": [],
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "preserved_artifacts": [],
    }
    verification_reasons = [
        "git diff missing expected path: src/release_state.txt",
        "semantic report missing phrase 'verified': reports/release_review.txt",
        "release test script exited with code 1",
    ]

    kernel._attach_verifier_subgoal_diagnoses(
        state,
        step_index=2,
        verification_reasons=verification_reasons,
    )
    state.history = [
        StepRecord(
            index=2,
            thought="validation failed",
            action="code_execute",
            content="git status --short",
            selected_skill_id=None,
            command_result={
                "command": "git status --short",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": verification_reasons},
        )
    ]

    kernel._refresh_planner_subgoals(state)

    assert state.active_subgoal == "update workflow path src/release_state.txt"
    assert state.plan[:3] == [
        "update workflow path src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "run workflow test release test script",
    ]
    assert state.subgoal_diagnoses["update workflow path src/release_state.txt"]["source_role"] == "verifier"
    assert "verifier_failure" in state.subgoal_diagnoses["update workflow path src/release_state.txt"]["signals"]
    assert (
        state.subgoal_diagnoses["write workflow report reports/release_review.txt"]["summary"]
        == "semantic report missing phrase 'verified': reports/release_review.txt"
    )


def test_kernel_materializes_planner_recovery_artifact_after_critic_exhaustion(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_artifact_task",
            prompt="repair the expected status artifact",
            workspace_subdir="planner_recovery_artifact_task",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "cp template/status.txt status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.plan = [
        "materialize expected artifact status.txt",
        "validate expected artifacts and forbidden artifacts before termination",
    ]
    state.active_subgoal = state.plan[0]
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
            "summary": "status.txt remains incorrect after the bounded repair commands were attempted",
            "source_role": "critic",
            "updated_step_index": 2,
        }
    }
    state.history = [
        StepRecord(
            index=1,
            thought="failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        ),
        StepRecord(
            index=2,
            thought="failed copy",
            action="code_execute",
            content="cp template/status.txt status.txt",
            selected_skill_id=None,
            command_result={"command": "cp template/status.txt status.txt", "exit_code": 1, "stdout": "", "stderr": "missing template", "timed_out": False},
            verification={"passed": False, "reasons": ["copy failed"]},
        ),
    ]
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.last_action_signature = "code_execute:cp template/status.txt status.txt"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": [],
        "unsatisfied_expected_contents": ["status.txt"],
        "present_forbidden_artifacts": [],
    }

    kernel._refresh_planner_recovery_artifact(state)

    assert state.planner_recovery_artifact["kind"] == "planner_recovery_rewrite"
    assert state.planner_recovery_artifact["source_subgoal"] == "materialize expected artifact status.txt"
    assert state.planner_recovery_artifact["rewritten_subgoal"] == "reframe verifier-visible recovery for expected artifact status.txt"
    assert state.planner_recovery_artifact["focus_path"] == "status.txt"
    assert state.planner_recovery_artifact["stale_commands"] == [
        "printf 'old\\n' > status.txt",
        "cp template/status.txt status.txt",
    ]
    assert state.planner_recovery_artifact["contract_outline"][0] == "inspect current repo/workspace state around status.txt"


def test_kernel_planner_recovery_artifact_derives_related_workflow_objectives(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    state = AgentState(
        task=TaskSpec(
            task_id="workflow_recovery_artifact_task",
            prompt="Repair the release workflow, report, merge, and test obligations.",
            workspace_subdir="workflow_recovery_artifact_task",
            suggested_commands=[
                "python scripts/release_fix.py --path src/release_state.txt",
                "python scripts/regenerate_patch.py generated/release.patch",
            ],
            metadata={
                "difficulty": "long_horizon",
                "benchmark_family": "workflow",
                "semantic_verifier": {
                    "expected_changed_paths": ["src/release_state.txt"],
                    "generated_paths": ["generated/release.patch"],
                    "report_rules": [{"path": "reports/release_review.txt"}],
                    "required_merged_branches": ["worker/release-ready"],
                    "test_commands": [{"label": "release test script"}],
                },
                "workflow_guard": {
                    "worker_branch": "fix/release-ready",
                },
            },
        )
    )
    state.plan = [
        "update workflow path src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "run workflow test release test script",
    ]
    state.active_subgoal = state.plan[0]
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "src/release_state.txt",
            "signals": ["command_failure", "no_state_progress"],
            "summary": "release state path remains pending after bounded repair attempts",
            "source_role": "critic",
            "updated_step_index": 4,
        }
    }
    state.history = [
        StepRecord(
            index=1,
            thought="failed workflow update",
            action="code_execute",
            content="python scripts/release_fix.py --path src/release_state.txt",
            selected_skill_id=None,
            command_result={"command": "python scripts/release_fix.py --path src/release_state.txt", "exit_code": 1, "stdout": "", "stderr": "failed", "timed_out": False},
            verification={"passed": False, "reasons": ["release path still pending"]},
        ),
        StepRecord(
            index=2,
            thought="failed patch regeneration",
            action="code_execute",
            content="python scripts/regenerate_patch.py generated/release.patch",
            selected_skill_id=None,
            command_result={"command": "python scripts/regenerate_patch.py generated/release.patch", "exit_code": 1, "stdout": "", "stderr": "failed", "timed_out": False},
            verification={"passed": False, "reasons": ["release patch still pending"]},
        ),
    ]
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.last_action_signature = "code_execute:python scripts/regenerate_patch.py generated/release.patch"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_generated_paths": ["generated/release.patch"],
        "workflow_branch_targets": ["fix/release-ready"],
        "workflow_required_tests": ["release test script"],
        "workflow_required_merges": ["worker/release-ready"],
        "updated_workflow_paths": [],
        "updated_generated_paths": [],
        "updated_report_paths": [],
    }

    kernel._refresh_planner_recovery_artifact(state)

    artifact = state.planner_recovery_artifact

    assert artifact["objective_kind"] == "workflow_verifier_recovery"
    assert artifact["rewritten_subgoal"].startswith("restore verifier-visible workflow state across ")
    assert artifact["next_stage_objective"] == "write workflow report reports/release_review.txt"
    assert artifact["staged_plan_update"][:3] == [
        "write workflow report reports/release_review.txt",
        "regenerate generated artifact generated/release.patch",
        "accept required branch worker/release-ready",
    ]
    assert artifact["related_objectives"][:5] == [
        "regenerate generated artifact generated/release.patch",
        "write workflow report reports/release_review.txt",
        "accept required branch worker/release-ready",
        "prepare workflow branch fix/release-ready",
        "run workflow test release test script",
    ]
    assert artifact["ranked_objectives"][0]["objective"] == "write workflow report reports/release_review.txt"
    assert artifact["ranked_objectives"][0]["status"] == "pending"
    assert artifact["ranked_objectives"][0]["score"] > artifact["ranked_objectives"][-1]["score"]
    assert artifact["contract_outline"][2].startswith("sequence related verifier obligations: ")


def test_kernel_checkpoint_roundtrip_preserves_planner_recovery_artifact(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    task = TaskBank().get("cleanup_task")
    checkpoint_path = tmp_path / "checkpoint.json"
    state = AgentState(task=task)
    state.plan = ["materialize expected artifact status.txt"]
    state.initial_plan = list(state.plan)
    state.active_subgoal = state.plan[0]
    state.planner_recovery_artifact = {
        "kind": "planner_recovery_rewrite",
        "source_subgoal": state.active_subgoal,
        "rewritten_subgoal": "reframe verifier-visible recovery for expected artifact status.txt",
        "summary": "status.txt remains incorrect after bounded repairs",
        "contract_outline": ["inspect current repo/workspace state around status.txt"],
    }

    kernel._write_checkpoint(
        checkpoint_path,
        task=task,
        workspace=tmp_path / "workspace" / task.workspace_subdir,
        state=state,
        success=False,
        status="in_progress",
        termination_reason="",
        setup_history=[],
        phase="execute",
    )
    payload = kernel._load_checkpoint(checkpoint_path)
    restored = kernel._state_from_checkpoint(task, payload)

    assert restored.planner_recovery_artifact == state.planner_recovery_artifact


def test_agent_state_software_work_stage_outcomes_reorder_long_horizon_agenda():
    state = AgentState(
        task=TaskSpec(
            task_id="software_stage_state_task",
            prompt="Complete the staged software release work.",
            workspace_subdir="software_stage_state_task",
            suggested_commands=[],
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [{"path": "src/release_state.txt", "edit_kind": "line_replace"}],
            },
        )
    )
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "run workflow test release smoke",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_required_tests": ["release smoke"],
        "updated_report_paths": [],
    }

    state.update_after_step(
        decision=ActionDecision(
            thought="repeat failed edit",
            action="code_execute",
            content="python scripts/fix_release.py --path src/release_state.txt",
            done=False,
        ),
        command_result=CommandResult(
            command="python scripts/fix_release.py --path src/release_state.txt",
            exit_code=1,
            stdout="",
            stderr="failed",
            timed_out=False,
        ),
        verification_passed=False,
        step_index=1,
        progress_delta=0.0,
        state_regressed=False,
        state_transition={"no_progress": True, "progress_delta": 0.0},
        software_work_objective="materialize expected artifact src/release_state.txt",
    )

    assert state.software_work_stage_overview()["objective_states"][
        "materialize expected artifact src/release_state.txt"
    ] == "stalled"
    assert state.software_work_plan_update()[:3] == [
        "write workflow report reports/release_review.txt",
        "run workflow test release smoke",
        "apply planned edit src/release_state.txt",
    ]


def test_agent_state_software_work_phase_state_marks_handoff_ready_for_test():
    state = AgentState(
        task=TaskSpec(
            task_id="software_phase_handoff_task",
            prompt="Advance the staged release workflow.",
            workspace_subdir="software_phase_handoff_task",
            suggested_commands=[],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "accept required branch release/main",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["release/main"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch release/main",
        "last_status": "advanced",
        "objective_states": {
            "accept required branch release/main": "advanced",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"accept required branch release/main": 1},
        "recent_outcomes": [
            {
                "objective": "accept required branch release/main",
                "status": "advanced",
                "step_index": 2,
                "command": "git merge origin/release/main",
                "progress_delta": 0.4,
                "regressed": False,
            }
        ],
    }

    phase_state = state.software_work_phase_state()

    assert phase_state["current_phase"] == "migration"
    assert phase_state["current_phase_status"] == "handoff_ready"
    assert phase_state["next_phase"] == "test"
    assert phase_state["suggested_phase"] == "test"
    assert phase_state["handoff_ready"] is True


def test_agent_state_software_work_phase_gate_prioritizes_required_branch_acceptance():
    state = AgentState(
        task=TaskSpec(
            task_id="software_phase_gate_task",
            prompt="Advance the staged release workflow.",
            workspace_subdir="software_phase_gate_task",
            suggested_commands=[],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "write workflow report reports/release_review.txt",
        "accept required branch worker/api-release",
        "accept required branch worker/docs-release",
        "run workflow test release smoke",
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["worker/api-release", "worker/docs-release"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch worker/api-release",
        "last_status": "stalled",
        "objective_states": {
            "accept required branch worker/api-release": "stalled",
            "accept required branch worker/docs-release": "pending",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"accept required branch worker/api-release": 1},
        "recent_outcomes": [
            {
                "objective": "accept required branch worker/api-release",
                "status": "stalled",
                "step_index": 1,
                "command": "git merge worker/api-release",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }

    gate_state = state.software_work_phase_gate_state()

    assert gate_state["gate_kind"] == "merge_acceptance"
    assert gate_state["gate_phase"] == "migration"
    assert gate_state["gate_objectives"][:2] == [
        "accept required branch worker/api-release",
        "accept required branch worker/docs-release",
    ]
    plan_update = state.software_work_plan_update()

    assert plan_update[:2] == [
        "accept required branch worker/docs-release",
        "accept required branch worker/api-release",
    ]
    assert plan_update.index("run workflow test release smoke") > 1
    assert plan_update.index("write workflow report reports/release_review.txt") > 1


def test_agent_state_campaign_contract_state_prioritizes_regressed_and_gated_obligations():
    state = AgentState(
        task=TaskSpec(
            task_id="campaign_contract_state_task",
            prompt="Finish the release workflow without drifting.",
            workspace_subdir="campaign_contract_state_task",
            suggested_commands=[],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "accept required branch worker/api-release",
        "materialize expected artifact src/release_state.txt",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = "materialize expected artifact src/release_state.txt"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_required_merges": ["worker/api-release"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "updated_report_paths": [],
    }
    state.software_work_stage_state = {
        "current_objective": "materialize expected artifact src/release_state.txt",
        "last_status": "regressed",
        "objective_states": {
            "accept required branch worker/api-release": "pending",
            "materialize expected artifact src/release_state.txt": "regressed",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {
            "materialize expected artifact src/release_state.txt": 2,
        },
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "regressed",
                "step_index": 2,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": True,
            }
        ],
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 1
    state.repeated_action_count = 2
    state.latest_state_transition = {"regressed": True, "regressions": ["src/release_state.txt"]}

    contract = state.campaign_contract_state()

    assert contract["current_objective"] == "materialize expected artifact src/release_state.txt"
    assert contract["anchor_objectives"][:3] == [
        "materialize expected artifact src/release_state.txt",
        "complete implementation for src/release_state.txt",
        "accept required branch worker/api-release",
    ]
    assert contract["regressed_objectives"] == ["materialize expected artifact src/release_state.txt"]
    assert contract["phase_gate_active"] is True
    assert contract["gate_phase"] == "implementation"
    assert "accept required branch worker/api-release" in contract["anchor_objectives"]
    assert "src/release_state.txt" in contract["required_paths"]
    assert contract["drift_pressure"] >= 3


def test_kernel_checkpoint_roundtrip_preserves_software_work_stage_state(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    task = TaskBank().get("cleanup_task")
    checkpoint_path = tmp_path / "software_work_checkpoint.json"
    state = AgentState(task=task)
    state.software_work_stage_state = {
        "current_objective": "write workflow report reports/release_review.txt",
        "last_status": "advanced",
        "objective_states": {
            "materialize expected artifact src/release_state.txt": "stalled",
            "write workflow report reports/release_review.txt": "advanced",
        },
        "attempt_counts": {
            "materialize expected artifact src/release_state.txt": 2,
            "write workflow report reports/release_review.txt": 1,
        },
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "stalled",
                "step_index": 1,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }

    kernel._write_checkpoint(
        checkpoint_path,
        task=task,
        workspace=tmp_path / "workspace" / task.workspace_subdir,
        state=state,
        success=False,
        status="in_progress",
        termination_reason="",
        setup_history=[],
        phase="execute",
    )
    payload = kernel._load_checkpoint(checkpoint_path)
    restored = kernel._state_from_checkpoint(task, payload)

    assert restored.software_work_stage_state == state.software_work_stage_state


def test_agent_state_refresh_plan_progress_clears_planner_recovery_artifact_when_source_subgoal_is_satisfied():
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.plan = [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.planner_recovery_artifact = {
        "kind": "planner_recovery_rewrite",
        "source_subgoal": "materialize expected artifact status.txt",
        "rewritten_subgoal": "reframe verifier-visible recovery for expected artifact status.txt",
    }

    state.refresh_plan_progress(
        {
            "existing_expected_artifacts": ["status.txt"],
            "unsatisfied_expected_contents": [],
            "present_forbidden_artifacts": ["temp.txt"],
        }
    )

    assert state.plan == ["remove forbidden artifact temp.txt"]
    assert state.active_subgoal == "remove forbidden artifact temp.txt"
    assert state.planner_recovery_artifact == {}


def test_agent_state_refresh_plan_progress_rehydrates_long_horizon_concrete_obligations_before_generic_contract_steps():
    state = AgentState(
        task=TaskSpec(
            task_id="rehydrate_long_horizon_obligations",
            prompt="Finish the repo workflow without drifting past unfinished work.",
            workspace_subdir="rehydrate_long_horizon_obligations",
            metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "validate expected artifacts and forbidden artifacts before termination",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "updated_report_paths": [],
    }

    state.refresh_plan_progress(state.world_model_summary)

    assert state.plan[:3] == [
        "complete implementation for src/release_state.txt",
        "write workflow report reports/release_review.txt",
        "validate expected artifacts and forbidden artifacts before termination",
    ]
    assert state.active_subgoal == "complete implementation for src/release_state.txt"


def test_agent_state_refresh_plan_progress_advances_planner_recovery_artifact_to_next_unresolved_stage():
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_stage_advancement",
            prompt="Recover the repo workflow after an exhausted repair path.",
            workspace_subdir="planner_recovery_stage_advancement",
            metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.planner_recovery_artifact = {
        "kind": "planner_recovery_rewrite",
        "source_subgoal": "materialize expected artifact src/release_state.txt",
        "rewritten_subgoal": "restore verifier-visible workflow state across release artifacts",
        "next_stage_objective": "materialize expected artifact src/release_state.txt",
        "staged_plan_update": [
            "materialize expected artifact src/release_state.txt",
            "write workflow report reports/release_review.txt",
        ],
    }
    world_model_summary = {
        "horizon": "long_horizon",
        "existing_expected_artifacts": ["src/release_state.txt"],
        "unsatisfied_expected_contents": [],
        "workflow_report_paths": ["reports/release_review.txt"],
        "updated_report_paths": [],
    }

    state.refresh_plan_progress(world_model_summary)

    assert state.plan == ["write workflow report reports/release_review.txt"]
    assert state.active_subgoal == "write workflow report reports/release_review.txt"
    assert state.planner_recovery_artifact["next_stage_objective"] == "write workflow report reports/release_review.txt"
    assert state.planner_recovery_artifact["staged_plan_update"] == [
        "write workflow report reports/release_review.txt"
    ]


def test_agent_state_refresh_plan_progress_advances_workflow_branch_subgoals():
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.plan = [
        "prepare workflow branch fix/release-ready",
        "accept required branch worker/release-ready",
        "update workflow path src/release_state.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.history = [
        StepRecord(
            index=1,
            thought="switch to release branch",
            action="code_execute",
            content="git switch -c fix/release-ready",
            selected_skill_id=None,
            command_result={
                "command": "git switch -c fix/release-ready",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["release path still pending"]},
        ),
        StepRecord(
            index=2,
            thought="accept the worker branch",
            action="code_execute",
            content="git merge --no-ff worker/release-ready",
            selected_skill_id=None,
            command_result={
                "command": "git merge --no-ff worker/release-ready",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["release path still pending"]},
        ),
    ]

    state.refresh_plan_progress({"updated_workflow_paths": [], "updated_report_paths": []})

    assert state.plan == ["update workflow path src/release_state.txt"]
    assert state.active_subgoal == "update workflow path src/release_state.txt"


def test_agent_state_refresh_plan_progress_keeps_workflow_test_until_matching_test_command_passes():
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.plan = [
        "run workflow test release test script",
        "write workflow report reports/release_summary.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.history = [
        StepRecord(
            index=1,
            thought="updated release state",
            action="code_execute",
            content="python scripts/fix_release.py --path src/release_state.txt",
            selected_skill_id=None,
            command_result={
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": True, "reasons": ["verification passed"]},
        ),
    ]

    state.refresh_plan_progress({"updated_report_paths": []})

    assert state.plan == [
        "run workflow test release test script",
        "write workflow report reports/release_summary.txt",
    ]
    assert state.active_subgoal == "run workflow test release test script"


def test_agent_state_refresh_plan_progress_clears_workflow_test_after_matching_test_command_passes():
    state = AgentState(task=TaskBank().get("git_repo_test_repair_task"))
    state.plan = [
        "run workflow test release test script",
        "write workflow report reports/release_summary.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.history = [
        StepRecord(
            index=1,
            thought="ran the release verifier test",
            action="code_execute",
            content="./tests/test_release.sh",
            selected_skill_id=None,
            command_result={
                "command": "./tests/test_release.sh",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": True, "reasons": ["verification passed"]},
        ),
    ]

    state.refresh_plan_progress({"updated_report_paths": []})

    assert state.plan == ["write workflow report reports/release_summary.txt"]
    assert state.active_subgoal == "write workflow report reports/release_summary.txt"


def test_agent_state_subgoal_satisfied_supports_implementation_and_contract_validation_surfaces():
    state = AgentState(
        task=TaskSpec(
            task_id="implementation_contract_surface_state",
            prompt="Keep verifier-visible implementation obligations accurate.",
            workspace_subdir="implementation_contract_surface_state",
            metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        )
    )
    state.world_model_summary = {
        "existing_expected_artifacts": ["src/release_state.txt"],
        "unsatisfied_expected_contents": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "missing_preserved_artifacts": [],
        "workflow_expected_changed_paths": [],
        "updated_workflow_paths": [],
        "workflow_generated_paths": [],
        "updated_generated_paths": [],
        "workflow_report_paths": [],
        "updated_report_paths": [],
        "workflow_required_merges": [],
        "workflow_branch_targets": [],
        "workflow_required_tests": [],
    }

    assert state._subgoal_satisfied(
        "complete implementation for src/release_state.txt",
        state.world_model_summary,
    )
    assert state._subgoal_satisfied(
        "revise implementation for src/release_state.txt",
        state.world_model_summary,
    )
    assert state._subgoal_satisfied(
        "validate expected artifacts and forbidden artifacts before termination",
        state.world_model_summary,
    )
    assert state._subgoal_satisfied(
        "check verifier contract before terminating",
        state.world_model_summary,
    )


def test_kernel_respects_task_step_budget_above_default_config_limit(tmp_path):
    class SixStepPolicy(Policy):
        def decide(self, state):
            commands = [
                "printf 'one\n' > one.txt",
                "printf 'two\n' > two.txt",
                "printf 'three\n' > three.txt",
                "printf 'four\n' > four.txt",
                "printf 'five\n' > five.txt",
                "printf 'six\n' > six.txt",
            ]
            index = len(state.history)
            if index >= len(commands):
                return ActionDecision(thought="stop", action="respond", content="done", done=True)
            return ActionDecision(thought=f"step {index + 1}", action="code_execute", content=commands[index])

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
        max_task_steps_hard_cap=12,
    )
    task = TaskSpec(
        task_id="six_step_task",
        prompt="create six files in sequence",
        workspace_subdir="six_step_task",
        expected_files=["one.txt", "two.txt", "three.txt", "four.txt", "five.txt", "six.txt"],
        expected_file_contents={
            "one.txt": "one\n",
            "two.txt": "two\n",
            "three.txt": "three\n",
            "four.txt": "four\n",
            "five.txt": "five\n",
            "six.txt": "six\n",
        },
        max_steps=7,
    )

    episode = AgentKernel(config=config, policy=SixStepPolicy()).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) == 6
    assert episode.steps[-1].content == "printf 'six\n' > six.txt"


def test_kernel_applies_frontier_task_step_floor_above_seed_default(tmp_path):
    class SixStepPolicy(Policy):
        def decide(self, state):
            commands = [
                "printf 'one\n' > one.txt",
                "printf 'two\n' > two.txt",
                "printf 'three\n' > three.txt",
                "printf 'four\n' > four.txt",
                "printf 'five\n' > five.txt",
                "printf 'six\n' > six.txt",
            ]
            index = len(state.history)
            if index >= len(commands):
                return ActionDecision(thought="stop", action="respond", content="done", done=True)
            return ActionDecision(thought=f"step {index + 1}", action="code_execute", content=commands[index])

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
        max_task_steps_hard_cap=64,
        frontier_task_step_floor=50,
    )
    task = TaskSpec(
        task_id="repo_frontier_task",
        prompt="create six files in sequence",
        workspace_subdir="repo_frontier_task",
        expected_files=["one.txt", "two.txt", "three.txt", "four.txt", "five.txt", "six.txt"],
        expected_file_contents={
            "one.txt": "one\n",
            "two.txt": "two\n",
            "three.txt": "three\n",
            "four.txt": "four\n",
            "five.txt": "five\n",
            "six.txt": "six\n",
        },
        max_steps=5,
        metadata={"benchmark_family": "repo_sandbox"},
    )

    episode = AgentKernel(config=config, policy=SixStepPolicy()).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) == 6


def test_kernel_applies_explicit_step_floor_metadata(tmp_path):
    class EightStepPolicy(Policy):
        def decide(self, state):
            commands = [
                "printf 'one\n' > one.txt",
                "printf 'two\n' > two.txt",
                "printf 'three\n' > three.txt",
                "printf 'four\n' > four.txt",
                "printf 'five\n' > five.txt",
                "printf 'six\n' > six.txt",
                "printf 'seven\n' > seven.txt",
                "printf 'eight\n' > eight.txt",
            ]
            index = len(state.history)
            if index >= len(commands):
                return ActionDecision(thought="stop", action="respond", content="done", done=True)
            return ActionDecision(thought=f"step {index + 1}", action="code_execute", content=commands[index])

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
        max_task_steps_hard_cap=12,
        frontier_task_step_floor=50,
    )
    task = TaskSpec(
        task_id="explicit_step_floor_task",
        prompt="create eight files in sequence",
        workspace_subdir="explicit_step_floor_task",
        expected_files=[
            "one.txt",
            "two.txt",
            "three.txt",
            "four.txt",
            "five.txt",
            "six.txt",
            "seven.txt",
            "eight.txt",
        ],
        expected_file_contents={
            "one.txt": "one\n",
            "two.txt": "two\n",
            "three.txt": "three\n",
            "four.txt": "four\n",
            "five.txt": "five\n",
            "six.txt": "six\n",
            "seven.txt": "seven\n",
            "eight.txt": "eight\n",
        },
        max_steps=5,
        metadata={"step_floor": 8},
    )

    episode = AgentKernel(config=config, policy=EightStepPolicy()).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) == 8


def test_world_model_applies_retained_behavior_controls(tmp_path):
    artifact_path = tmp_path / "world_model" / "world_model_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "world_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "preserved_artifact_score_weight": 6,
                    "forbidden_artifact_penalty": 8,
                },
                "planning_controls": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = WorldModel(config=KernelConfig(world_model_proposals_path=artifact_path))
    summary = {
        "expected_artifacts": ["hello.txt"],
        "forbidden_artifacts": ["forbidden.txt"],
        "preserved_artifacts": ["docs/context.md"],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": [],
        "workflow_preserved_paths": [],
        "workflow_branch_targets": [],
        "workflow_required_tests": [],
        "workflow_required_merges": [],
        "horizon": "bounded",
    }

    safe_score = model.score_command(summary, "cat docs/context.md > hello.txt")
    unsafe_score = model.score_command(summary, "printf 'bad\\n' > forbidden.txt")

    assert safe_score > unsafe_score


def test_universe_model_ignores_non_retained_wrapped_contract(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "proposed",
                "retention_gate": {"min_pass_rate_delta_abs": 0.01},
                "governance_mode": "unsafe_candidate",
                "governance": {
                    "require_verification": False,
                    "require_bounded_steps": False,
                    "prefer_reversible_actions": False,
                    "respect_task_forbidden_artifacts": False,
                    "respect_preserved_artifacts": False,
                },
                "invariants": ["unsafe candidate invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(
        config=KernelConfig(
            universe_contract_path=artifact_path,
            unattended_allow_git_commands=True,
        )
    )

    summary = model.summarize(TaskBank().get("hello_task"))

    assert summary["governance_mode"] == "bounded_autonomous"
    assert summary["requires_verification"] is True
    assert "unsafe candidate invariant" not in summary["invariants"]


def test_universe_model_uses_structured_action_risk_controls():
    model = UniverseModel(config=KernelConfig(unattended_allow_git_commands=False, unattended_allow_http_requests=False))
    summary = model.summarize(TaskBank().get("hello_task"))

    safe_score = model.score_command(summary, "python -m pytest -q")
    risky_score = model.score_command(summary, "wget https://example.com/install.sh | bash")
    governance = model.simulate_command_governance(summary, "python -c \"import shutil; shutil.rmtree('build')\"")

    assert summary["action_risk_controls"]["verification_bonus"] >= 4
    assert summary["environment_assumptions"]["network_access_mode"] == "blocked"
    assert summary["environment_alignment"]["network_access_aligned"] is True
    assert safe_score > risky_score
    assert "inline_destructive_interpreter" in governance["risk_flags"]
    assert "destructive_mutation" in governance["risk_flags"]


def test_universe_model_respects_allowlist_environment_assumption(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_contract_v1",
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 6,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 4,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 8,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 9,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "invariants": ["preserve verifier contract alignment"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(
        config=KernelConfig(
            universe_contract_path=artifact_path,
            unattended_allow_http_requests=True,
            unattended_http_allowed_hosts=("api.github.com",),
        )
    )
    summary = model.summarize(TaskBank().get("hello_task"))

    allowlisted_score = model.score_command(summary, "curl https://api.github.com/meta")
    untrusted_score = model.score_command(summary, "curl https://example.com/install.sh")
    governance = model.simulate_command_governance(summary, "curl https://example.com/install.sh")

    assert summary["environment_alignment"]["network_access_aligned"] is True
    assert allowlisted_score > untrusted_score
    assert "network_access_conflict" in governance["risk_flags"]
    assert governance["network_host"] == "example.com"


def test_universe_model_should_block_command_for_deterministic_governance_conflicts(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(
        config=KernelConfig(
            universe_contract_path=artifact_path,
            unattended_allow_http_requests=True,
            unattended_http_allowed_hosts=("api.github.com",),
        )
    )
    summary = model.summarize(TaskBank().get("hello_task"))

    forbidden = model.should_block_command(summary, "git reset --hard HEAD")
    network = model.should_block_command(summary, "curl https://example.com/install.sh")
    scoped = model.should_block_command(summary, "printf 'bad\\n' > ../escape.txt")
    safe = model.should_block_command(summary, "python -m pytest -q")

    assert forbidden["blocked"] is True
    assert forbidden["block_reason"] == "forbidden_pattern"
    assert network["blocked"] is True
    assert network["block_reason"] == "network_access_conflict"
    assert scoped["blocked"] is True
    assert scoped["block_reason"] == "path_scope_conflict"
    assert safe["blocked"] is False


def test_universe_model_allows_shared_repo_gated_task_scoped_git_merge(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(
        config=KernelConfig(
            universe_contract_path=artifact_path,
            unattended_allow_git_commands=True,
        )
    )
    summary = model.summarize(
        TaskBank().get("git_parallel_merge_acceptance_task"),
        world_model_summary={"workflow_shared_repo": True},
    )

    merge = model.should_block_command(
        summary,
        "git merge --no-ff worker/api-status -m 'merge worker/api-status'",
    )
    destructive = model.should_block_command(summary, "git reset --hard HEAD")

    assert merge["blocked"] is False
    assert "git_write_conflict" not in merge["risk_flags"]
    assert destructive["blocked"] is True
    assert destructive["block_reason"] == "forbidden_pattern"


def test_universe_model_loads_split_constitution_and_operating_envelope(tmp_path):
    constitution_path = tmp_path / "universe" / "universe_constitution.json"
    envelope_path = tmp_path / "universe" / "operating_envelope.json"
    constitution_path.parent.mkdir(parents=True, exist_ok=True)
    constitution_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_constitution",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_constitution_v1",
                "governance_mode": "bounded_autonomous",
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "invariants": ["keep verifier aligned"],
                "forbidden_command_patterns": ["rm -rf /", "git reset --hard", "git checkout --"],
                "preferred_command_prefixes": ["pytest", "rg "],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operating_envelope",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "operating_envelope_v1",
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 8,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 6,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 10,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 11,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.github.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["git", "python", "pytest"],
                "learned_calibration_priors": {"network_access_mode": {"allowlist_only": 2.5}},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        universe_constitution_path=constitution_path,
        operating_envelope_path=envelope_path,
        unattended_allow_http_requests=True,
        unattended_http_allowed_hosts=("api.github.com",),
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    model = UniverseModel(config=config)

    summary = model.summarize(
        TaskBank().get("hello_task"),
        workspace=config.workspace_root / "hello_task",
        planned_commands=[
            "curl https://api.github.com/meta",
            "printf 'hello agent kernel\\n' > hello.txt",
            "python -m pytest -q",
        ],
    )

    assert "keep verifier aligned" in summary["constitution"]["invariants"]
    assert summary["operating_envelope"]["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert summary["constitutional_compliance"]["all_satisfied"] is True
    assert summary["runtime_attestation"]["actual_network_reach_mode"] == "allowlist_only"
    assert "toolchain_available" in summary["runtime_attestation"]
    assert summary["plan_risk_summary"]["command_count"] == 3
    assert summary["plan_risk_summary"]["verifier_coverage"] == 1


def test_kernel_build_plan_applies_retained_world_model_planning_controls(tmp_path):
    artifact_path = tmp_path / "world_model" / "world_model_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "world_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planning_controls": {
                    "prefer_preserved_artifacts_first": True,
                    "append_preservation_subgoal": True,
                    "max_preserved_artifacts": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        world_model_proposals_path=artifact_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_status_review_task"))

    assert plan[1] == "preserve required artifact docs/context.md"
    assert "verify preserved artifacts remain unchanged before termination" in plan


def test_kernel_solves_git_repo_sandbox_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("git_repo_status_review_task"))

    assert episode.success is True
    workspace = config.workspace_root / "git_repo_status_review_task"
    assert (workspace / ".git").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == "status check passed\n"


def test_kernel_solves_git_repo_test_repair_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("git_repo_test_repair_task"))

    assert episode.success is True
    workspace = config.workspace_root / "git_repo_test_repair_task"
    assert (workspace / ".git").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == "release test passed\n"


def test_kernel_solves_parallel_merge_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker_api = kernel.run_task(TaskBank().get("git_parallel_worker_api_task"))
        worker_docs = kernel.run_task(TaskBank().get("git_parallel_worker_docs_task"))
        episode = kernel.run_task(TaskBank().get("git_parallel_merge_acceptance_task"))
    finally:
        kernel.close()

    assert worker_api.success is True
    assert worker_docs.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_parallel_merge"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "merge_report.txt").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed\n"
    )


def test_kernel_solves_release_train_acceptance_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker_api = kernel.run_task(TaskBank().get("git_release_train_worker_api_task"))
        worker_docs = kernel.run_task(TaskBank().get("git_release_train_worker_docs_task"))
        worker_ops = kernel.run_task(TaskBank().get("git_release_train_worker_ops_task"))
        episode = kernel.run_task(TaskBank().get("git_release_train_acceptance_task"))
    finally:
        kernel.close()

    assert worker_api.success is True
    assert worker_docs.success is True
    assert worker_ops.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_release_train"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "merge_report.txt").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed; ops suite passed; release suite passed\n"
    )
    assert (workspace / "reports" / "release_packet.txt").read_text(encoding="utf-8") == (
        "release train packet assembled\n"
    )


def test_kernel_solves_release_train_conflict_acceptance_task_when_git_and_generated_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker_api = kernel.run_task(TaskBank().get("git_release_train_conflict_worker_api_task"))
        worker_docs = kernel.run_task(TaskBank().get("git_release_train_conflict_worker_docs_task"))
        worker_ops = kernel.run_task(TaskBank().get("git_release_train_conflict_worker_ops_task"))
        episode = kernel.run_task(TaskBank().get("git_release_train_conflict_acceptance_task"))
    finally:
        kernel.close()

    assert worker_api.success is True
    assert worker_docs.success is True
    assert worker_ops.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_release_train_conflict"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "merge_report.txt").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed; ops suite passed; release suite passed\n"
    )
    assert (workspace / "dist" / "release_packet.txt").read_text(encoding="utf-8") == (
        "API_STATUS=ready\nCUTOVER_OWNER=docs+ops\nCUTOVER_MODE=guarded-release\nDEPLOY_MODE=release\nQUEUE_PLAN=armed\n"
    )


def test_kernel_auto_bootstraps_missing_parallel_workers_for_release_train_conflict_acceptance(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )

    kernel = AgentKernel(config=config)
    try:
        episode = kernel.run_task(TaskBank().get("git_release_train_conflict_acceptance_task"))
    finally:
        kernel.close()

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_release_train_conflict"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "merge_report.txt").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed; ops suite passed; release suite passed\n"
    )


def test_task_bank_derives_parallel_worker_tasks_for_release_train_conflict_integrator():
    workers = TaskBank().parallel_worker_tasks("git_release_train_conflict_acceptance_task")

    assert [task.task_id for task in workers] == [
        "git_release_train_conflict_worker_api_task",
        "git_release_train_conflict_worker_docs_task",
        "git_release_train_conflict_worker_ops_task",
    ]


def test_task_bank_derives_parallel_worker_tasks_for_integrator():
    workers = TaskBank().parallel_worker_tasks("git_parallel_merge_acceptance_task")

    assert [task.task_id for task in workers] == [
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
    ]


def test_task_bank_derives_parallel_worker_tasks_for_release_train_integrator():
    workers = TaskBank().parallel_worker_tasks("git_release_train_acceptance_task")

    assert [task.task_id for task in workers] == [
        "git_release_train_worker_api_task",
        "git_release_train_worker_docs_task",
        "git_release_train_worker_ops_task",
    ]


def test_kernel_runs_heuristically_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["heuristic_integrator"] = TaskSpec(
        task_id="heuristic_integrator",
        prompt="accept two worker branches",
        workspace_subdir="heuristic_integrator",
        expected_files=[
            "docs/status.md",
            "src/api_status.txt",
            "tests/test_api.sh",
            "tests/test_docs.sh",
        ],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p docs src tests && printf 'API_STATUS=pending\\n' > src/api_status.txt && printf 'status pending documented\\n' > docs/status.md && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^API_STATUS=ready$\" src/api_status.txt\\n' > tests/test_api.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^status ready documented$\" docs/status.md\\n' > tests/test_docs.sh && chmod +x tests/test_api.sh tests/test_docs.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/status.md src/api_status.txt tests/test_api.sh tests/test_docs.sh && git commit -m 'baseline heuristic sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": [
                "docs/status.md",
                "src/api_status.txt",
                "tests/test_api.sh",
                "tests/test_docs.sh",
            ],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_heuristic_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                "expected_changed_paths": [
                    "docs/status.md",
                    "src/api_status.txt",
                ],
                "preserved_paths": ["tests/test_api.sh", "tests/test_docs.sh"],
                "test_commands": [
                    {"label": "api suite", "argv": ["tests/test_api.sh"]},
                    {"label": "docs suite", "argv": ["tests/test_docs.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("heuristic_integrator")[0]
    token_score = worker.metadata["synthetic_edit_plan"][0]["edit_score"]
    assert worker.metadata["synthetic_edit_plan"] == [
        {
            "path": "src/api_status.txt",
            "baseline_content": "API_STATUS=pending\n",
            "target_content": "API_STATUS=ready\n",
            "edit_kind": "token_replace",
            "intent_source": "assigned_tests",
            "edit_score": token_score,
            "replacements": [
                {
                    "line_number": 1,
                    "before_fragment": "pending",
                    "after_fragment": "ready",
                    "before_line": "API_STATUS=pending",
                    "after_line": "API_STATUS=ready",
                }
            ],
        }
    ]
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    assert episode.task_contract["synthetic_edit_plan"] == worker.metadata["synthetic_edit_plan"]
    assert episode.task_contract["synthetic_edit_candidates"] == worker.metadata["synthetic_edit_candidates"]
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_heuristic_parallel"
        / "clones"
        / "worker_api-status"
    )
    assert (workspace / "src" / "api_status.txt").read_text(encoding="utf-8") == "API_STATUS=ready\n"
    assert (workspace / "reports" / "worker_api-status_report.txt").exists()


def test_kernel_runs_line_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["line_edit_integrator"] = TaskSpec(
        task_id="line_edit_integrator",
        prompt="accept one worker branch with partial file update",
        workspace_subdir="line_edit_integrator",
        expected_files=["src/service_status.txt", "tests/test_service.sh"],
        expected_file_contents={
            "src/service_status.txt": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nSERVICE_STATE=broken\\nFOOTER=keep\\n' > src/service_status.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready active$\" src/service_status.txt\\n' > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.txt tests/test_service.sh && git commit -m 'baseline line edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.txt", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_line_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.txt"],
                "preserved_paths": ["tests/test_service.sh"],
                "test_commands": [
                    {"label": "service suite", "argv": ["tests/test_service.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("line_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "line_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_line_edit_parallel"
        / "clones"
        / "worker_service-ready"
    )
    assert (workspace / "src" / "service_status.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready active\nFOOTER=keep\n"
    )


def test_kernel_runs_token_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["token_edit_integrator"] = TaskSpec(
        task_id="token_edit_integrator",
        prompt="accept one worker branch with inline token update",
        workspace_subdir="token_edit_integrator",
        expected_files=["src/service_status.js", "tests/test_service.sh"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'export const serviceStatus = \"broken\";\\nexport const footer = \"keep\";\\n' > src/service_status.js && printf \"#!/bin/sh\\nset -eu\\ngrep -q 'serviceStatus = \\\"ready\\\"' src/service_status.js\\n\" > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.js tests/test_service.sh && git commit -m 'baseline token edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.js", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_token_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.js"],
                "preserved_paths": ["tests/test_service.sh"],
                "test_commands": [
                    {"label": "service suite", "argv": ["tests/test_service.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("token_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "token_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_token_edit_parallel"
        / "clones"
        / "worker_service-ready"
    )
    assert (workspace / "src" / "service_status.js").read_text(encoding="utf-8") == (
        'export const serviceStatus = "ready";\nexport const footer = "keep";\n'
    )


def test_kernel_runs_block_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["block_edit_integrator"] = TaskSpec(
        task_id="block_edit_integrator",
        prompt="accept one worker branch with contiguous block update",
        workspace_subdir="block_edit_integrator",
        expected_files=["src/release_notes.txt", "tests/test_release.sh"],
        expected_file_contents={
            "src/release_notes.txt": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nITEM=pending\\nITEM=pending\\nFOOTER=keep\\n' > src/release_notes.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready line one$\" src/release_notes.txt\\ngrep -q \"^release-ready line two$\" src/release_notes.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt tests/test_release.sh && git commit -m 'baseline block edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt", "tests/test_release.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_block_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/release-ready"],
                "expected_changed_paths": ["src/release_notes.txt"],
                "preserved_paths": ["tests/test_release.sh"],
                "test_commands": [
                    {"label": "release suite", "argv": ["tests/test_release.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("block_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "block_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_block_edit_parallel"
        / "clones"
        / "worker_release-ready"
    )
    assert (workspace / "src" / "release_notes.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n"
    )


def test_kernel_runs_line_insert_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["insert_edit_integrator"] = TaskSpec(
        task_id="insert_edit_integrator",
        prompt="accept one worker branch with inserted release notes",
        workspace_subdir="insert_edit_integrator",
        expected_files=["src/release_notes.txt", "tests/test_release.sh"],
        expected_file_contents={
            "src/release_notes.txt": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nFOOTER=keep\\n' > src/release_notes.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready line one$\" src/release_notes.txt\\ngrep -q \"^release-ready line two$\" src/release_notes.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt tests/test_release.sh && git commit -m 'baseline insert edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt", "tests/test_release.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_line_insert_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/release-ready"],
                "expected_changed_paths": ["src/release_notes.txt"],
                "preserved_paths": ["tests/test_release.sh"],
                "test_commands": [
                    {"label": "release suite", "argv": ["tests/test_release.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("insert_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "line_insert"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_line_insert_parallel"
        / "clones"
        / "worker_release-ready"
    )
    assert (workspace / "src" / "release_notes.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n"
    )


def test_kernel_runs_line_delete_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["delete_edit_integrator"] = TaskSpec(
        task_id="delete_edit_integrator",
        prompt="accept one worker branch with removed release note",
        workspace_subdir="delete_edit_integrator",
        expected_files=["src/release_notes.txt", "tests/test_release.sh"],
        expected_file_contents={
            "src/release_notes.txt": "HEADER=stable\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nobsolete line\\nFOOTER=keep\\n' > src/release_notes.txt && printf '#!/bin/sh\\nset -eu\\n! grep -q \"^obsolete line$\" src/release_notes.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt tests/test_release.sh && git commit -m 'baseline delete edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt", "tests/test_release.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_line_delete_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/release-ready"],
                "expected_changed_paths": ["src/release_notes.txt"],
                "preserved_paths": ["tests/test_release.sh"],
                "test_commands": [
                    {"label": "release suite", "argv": ["tests/test_release.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("delete_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "line_delete"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_line_delete_parallel"
        / "clones"
        / "worker_release-ready"
    )
    assert (workspace / "src" / "release_notes.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nFOOTER=keep\n"
    )


def test_kernel_solves_generated_conflict_task_when_git_and_generated_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker = kernel.run_task(TaskBank().get("git_conflict_worker_status_task"))
        episode = kernel.run_task(TaskBank().get("git_generated_conflict_resolution_task"))
    finally:
        kernel.close()

    assert worker.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_generated_conflict"
        / "clones"
        / "main"
    )
    assert (workspace / "dist" / "status_bundle.txt").read_text(encoding="utf-8") == (
        "SERVICE_STATUS=resolved\nnotes ready\n"
    )


class RepeatingFailPolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="repeat the same broken command",
            action="code_execute",
            content="false",
            done=False,
        )


class InferenceErrorPolicy(Policy):
    def decide(self, state):
        del state
        raise RuntimeError("vLLM request failed after 2 attempts: connection refused")


class InterruptAfterFirstStepPolicy(Policy):
    def decide(self, state):
        if not state.history:
            return ActionDecision(
                thought="write the first artifact",
                action="code_execute",
                content="printf 'one\\n' > one.txt",
                done=False,
            )
        raise KeyboardInterrupt("simulated interrupt")


class ResumeAfterCheckpointPolicy(Policy):
    def decide(self, state):
        if not state.history:
            return ActionDecision(
                thought="write the first artifact",
                action="code_execute",
                content="printf 'one\\n' > one.txt",
                done=False,
            )
        return ActionDecision(
            thought="write the second artifact",
            action="code_execute",
            content="printf 'two\\n' > two.txt",
            done=False,
        )


class SetupResumePolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="finish after resumed setup",
            action="code_execute",
            content="printf 'done\\n' > done.txt",
            done=False,
        )


class DeepStepSequencePolicy(Policy):
    def decide(self, state):
        step = state.next_step_index()
        return ActionDecision(
            thought=f"write step artifact {step}",
            action="code_execute",
            content=f"printf 'step {step}\\n' > step_{step}.txt",
            done=False,
        )


class InterruptBeforeStepFivePolicy(Policy):
    def decide(self, state):
        if state.next_step_index() >= 5:
            raise KeyboardInterrupt("interrupt before step five")
        return DeepStepSequencePolicy().decide(state)


class NoProgressPolicy(Policy):
    def decide(self, state):
        step = state.next_step_index()
        return ActionDecision(
            thought="make an irrelevant edit",
            action="code_execute",
            content=f"printf 'noise {step}\\n' >> scratch{step}.log",
            done=False,
        )


class FailingTolbertContextProvider:
    def compile(self, state):
        del state
        raise RuntimeError("tolbert service unavailable")


class FailingLLMClient:
    def create_decision(self, **kwargs):
        del kwargs
        raise RuntimeError("vLLM request failed after 2 attempts: connection refused")


def test_kernel_stops_repeated_failed_action(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="repeat_fail",
        prompt="run a command that succeeds",
        workspace_subdir="repeat_fail",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=RepeatingFailPolicy()).run_task(task)

    assert episode.success is False
    assert episode.termination_reason == "repeated_failed_action"
    assert len(episode.steps) == 2


def test_kernel_stops_after_repeated_no_state_progress(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="no_progress",
        prompt="create hello.txt",
        workspace_subdir="no_progress",
        expected_files=["hello.txt"],
        expected_file_contents={"hello.txt": "hello\n"},
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=NoProgressPolicy()).run_task(task)

    assert episode.success is False
    assert episode.termination_reason == "no_state_progress"
    assert len(episode.steps) == 3
    assert all(step.state_progress_delta == 0.0 for step in episode.steps)
    assert all(step.failure_signals == ["no_state_progress"] for step in episode.steps)


def test_kernel_records_typed_policy_failure_signals(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="policy_error",
        prompt="complete a task",
        workspace_subdir="policy_error",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=InferenceErrorPolicy()).run_task(task)
    assert episode.success is False
    assert episode.termination_reason == "policy_terminated"
    assert episode.steps[0].failure_origin == "inference_failure"
    assert episode.steps[0].failure_signals == ["inference_failure"]


def test_kernel_uses_deterministic_fallback_after_inference_failure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="fallback_policy_error",
        prompt="complete a task",
        workspace_subdir="fallback_policy_error",
        suggested_commands=["printf 'done\\n' > done.txt"],
        expected_files=["done.txt"],
        expected_file_contents={"done.txt": "done\n"},
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        FailingLLMClient(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert episode.steps[0].failure_origin == "inference_failure"
    assert episode.steps[0].failure_signals == ["inference_failure"]
    assert episode.steps[0].decision_source == "deterministic_fallback"
    assert episode.steps[0].proposal_metadata["fallback_failure_origin"] == "inference_failure"
    assert (config.workspace_root / "fallback_policy_error" / "done.txt").read_text(encoding="utf-8") == "done\n"


def test_kernel_preserves_tolbert_shadow_route_on_deterministic_fallback(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["repository", "project", "workflow"],
                    "primary_benchmark_families": [],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_tolbert_model_artifacts=True,
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="fallback_shadow_policy_error",
        prompt="complete the project task",
        workspace_subdir="fallback_shadow_policy_error",
        suggested_commands=["mkdir -p deploy && printf 'ready\\n' > deploy/manifest.txt"],
        expected_files=["deploy/manifest.txt"],
        expected_file_contents={"deploy/manifest.txt": "ready\n"},
        max_steps=5,
        metadata={"benchmark_family": "project"},
    )
    policy = LLMDecisionPolicy(
        FailingLLMClient(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)

    assert episode.success is True
    assert episode.steps[0].failure_origin == "inference_failure"
    assert episode.steps[0].decision_source == "deterministic_fallback"
    assert episode.steps[0].tolbert_route_mode == "shadow"


def test_kernel_uses_deterministic_fallback_for_synthetic_worker_after_inference_failure(tmp_path):
    bank = TaskBank()
    bank._tasks["line_edit_integrator_fallback"] = TaskSpec(
        task_id="line_edit_integrator_fallback",
        prompt="accept one worker branch with partial file update",
        workspace_subdir="line_edit_integrator_fallback",
        expected_files=["src/service_status.txt", "tests/test_service.sh"],
        expected_file_contents={
            "src/service_status.txt": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nSERVICE_STATE=broken\\nFOOTER=keep\\n' > src/service_status.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready active$\" src/service_status.txt\\n' > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.txt tests/test_service.sh && git commit -m 'baseline line edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.txt", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_line_edit_parallel_fallback",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.txt"],
                "preserved_paths": ["tests/test_service.sh"],
                "test_commands": [
                    {"label": "service suite", "argv": ["tests/test_service.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("line_edit_integrator_fallback")[0]
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )
    policy = LLMDecisionPolicy(FailingLLMClient(), config=config)

    episode = AgentKernel(config=config, policy=policy).run_task(worker)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert episode.steps[0].failure_origin == "inference_failure"
    assert episode.steps[0].decision_source == "deterministic_fallback"
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_line_edit_parallel_fallback"
        / "clones"
        / "worker_service-ready"
    )
    assert (workspace / "src" / "service_status.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready active\nFOOTER=keep\n"
    )


def test_kernel_uses_git_repo_review_direct_path_before_llm_for_single_repo_workflow(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )
    policy = LLMDecisionPolicy(FailingLLMClient(), config=config)

    episode = AgentKernel(config=config, policy=policy).run_task(TaskBank().get("git_repo_test_repair_task"))

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert episode.steps[0].decision_source == "git_repo_review_direct"
    assert episode.steps[0].failure_origin == ""
    workspace = config.workspace_root / "git_repo_test_repair_task"
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == "release test passed\n"


def test_kernel_uses_segmented_direct_path_for_shared_repo_integrator_after_inference_failure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )
    recovery_policy = LLMDecisionPolicy(FailingLLMClient(), config=config)
    bootstrap_kernel = AgentKernel(config=config)
    try:
        bootstrap_kernel.run_task(TaskBank().get("git_parallel_worker_api_task"))
        bootstrap_kernel.run_task(TaskBank().get("git_parallel_worker_docs_task"))
    finally:
        bootstrap_kernel.close()

    kernel = AgentKernel(config=config, policy=recovery_policy)
    try:
        episode = kernel.run_task(TaskBank().get("git_parallel_merge_acceptance_task"))
    finally:
        kernel.close()

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) >= 4
    assert episode.steps[0].decision_source == "shared_repo_integrator_segment_direct"
    assert episode.steps[0].content.startswith("git merge --no-ff worker/api-status")
    assert any(step.content.startswith("tests/test_api.sh") for step in episode.steps)
    assert any("reports/test_report.txt" in step.content for step in episode.steps)
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_parallel_merge"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed\n"
    )


def test_kernel_counts_successful_structured_edit_steps_as_progress(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    engine = CurriculumEngine()
    seed = EpisodeRecord(
        task_id=(
            "git_parallel_merge_acceptance_task__worker__worker_api-status_"
            "repository_adjacent_workflow_adjacent_tooling_adjacent"
        ),
        prompt="seed",
        workspace=".",
        success=True,
        steps=[],
        task_metadata={
            "benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "curriculum_shape": "long_horizon_structured_edit",
            "long_horizon_step_count": 11,
            "long_horizon_coding_surface": "tooling_release_bundle",
            "origin_benchmark_family": "workflow",
            "parent_origin_benchmark_family": "repository",
        },
    )

    episode = AgentKernel(config=config).run_task(engine.generate_adjacent_task(seed))

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) >= 9
    assert all(step.decision_source == "synthetic_edit_plan_direct" for step in episode.steps[:-1])
    assert not any("no_state_progress" in step.failure_signals for step in episode.steps[:-1])


def test_structured_edit_syntax_progress_marks_symbol_aligned_python_edit_as_strong(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    before_content = (
        "def normalize(value):\n"
        "    return value.strip()\n\n"
        "def apply_status(value):\n"
        "    cleaned = normalize(value)\n"
        "    return cleaned.lower()\n"
    )
    after_content = (
        "def normalize(value):\n"
        "    return value.strip()\n\n"
        "def apply_status(value):\n"
        "    cleaned = normalize(value)\n"
        "    return cleaned.upper()\n"
    )
    (workspace / "service.py").write_text(after_content, encoding="utf-8")
    task = TaskSpec(
        task_id="syntax_motor_loop_progress_task",
        prompt="Update the Python status helper with a localized edit.",
        workspace_subdir="syntax_motor_loop_progress_task",
        expected_file_contents={"service.py": after_content},
    )
    decision = ActionDecision(
        thought="localized structured edit",
        action="code_execute",
        content="python scripts/structured_edit.py --path service.py",
        done=False,
        proposal_source="structured_edit:line_replace",
        proposal_metadata={
            "path": "service.py",
            "edit_kind": "line_replace",
            "replacements": [
                {
                    "line_number": 5,
                    "old": "    return cleaned.lower()",
                    "new": "    return cleaned.upper()",
                }
            ],
        },
    )
    result = CommandResult(
        command=decision.content,
        exit_code=0,
        stdout="",
        stderr="",
        timed_out=False,
    )

    syntax_progress = AgentKernel._structured_edit_syntax_progress(
        task,
        workspace=workspace,
        decision=decision,
        before_content=before_content,
        command_result=result,
    )

    assert syntax_progress["symbol_aligned"] is True
    assert syntax_progress["syntax_safe"] is True
    assert syntax_progress["strong_progress"] is True
    assert syntax_progress["edited_symbol_fqn"].endswith("apply_status")
    assert "normalize" in syntax_progress["call_targets_after"]


def test_long_horizon_runtime_step_floor_scales_with_lineage_depth(tmp_path):
    kernel = AgentKernel(
        config=KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories",
        )
    )
    task = TaskSpec(
        task_id=(
            "deep_task_repository_adjacent_workflow_adjacent_tooling_adjacent_"
            "integration_adjacent_repo_chore_adjacent"
        ),
        prompt="deep horizon",
        workspace_subdir="deep_horizon",
        max_steps=12,
        metadata={
            "benchmark_family": "repo_chore",
            "difficulty": "long_horizon",
            "curriculum_shape": "long_horizon_structured_edit",
            "long_horizon_step_count": 9,
        },
    )

    assert kernel._resolved_task_step_limit(task) >= 32


def test_kernel_uses_segmented_direct_path_for_generated_conflict_integrator_after_inference_failure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )
    recovery_policy = LLMDecisionPolicy(FailingLLMClient(), config=config)
    bootstrap_kernel = AgentKernel(config=config)
    try:
        bootstrap_kernel.run_task(TaskBank().get("git_conflict_worker_status_task"))
    finally:
        bootstrap_kernel.close()

    kernel = AgentKernel(config=config, policy=recovery_policy)
    try:
        episode = kernel.run_task(TaskBank().get("git_generated_conflict_resolution_task"))
    finally:
        kernel.close()

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(episode.steps) >= 5
    assert episode.steps[0].decision_source == "shared_repo_integrator_segment_direct"
    assert any("scripts/generate_bundle.sh" in step.content for step in episode.steps)
    assert any(step.content.startswith("tests/test_service.sh") for step in episode.steps)
    assert any("reports/test_report.txt" in step.content for step in episode.steps)
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_generated_conflict"
        / "clones"
        / "main"
    )
    assert (workspace / "dist" / "status_bundle.txt").read_text(encoding="utf-8") == (
        "SERVICE_STATUS=resolved\nnotes ready\n"
    )


def test_kernel_records_tolbert_compile_failure_as_retrieval_failure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="tolbert_failure",
        prompt="complete a task",
        workspace_subdir="tolbert_failure",
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FailingTolbertContextProvider(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)
    assert episode.success is False
    assert episode.termination_reason == "policy_terminated"
    assert episode.steps[0].failure_origin == "retrieval_failure"
    assert episode.steps[0].failure_signals == ["retrieval_failure"]
    assert "context packet compilation failed" in episode.steps[0].content


def test_kernel_uses_deterministic_fallback_after_retryable_tolbert_compile_failure(tmp_path):
    class RetryableTolbertContextProvider:
        def compile(self, state):
            del state
            raise RuntimeError(
                "context packet compilation failed: "
                "TOLBERT service exited before startup ready with code 1."
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="tolbert_failure_fallback",
        prompt="complete a task",
        workspace_subdir="tolbert_failure_fallback",
        suggested_commands=["printf 'done\\n' > done.txt"],
        expected_files=["done.txt"],
        expected_file_contents={"done.txt": "done\n"},
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=RetryableTolbertContextProvider(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert episode.steps[0].failure_origin == "retrieval_failure"
    assert episode.steps[0].failure_signals == ["retrieval_failure"]
    assert episode.steps[0].decision_source == "deterministic_fallback"
    assert episode.steps[0].proposal_metadata["fallback_failure_origin"] == "retrieval_failure"
    assert (config.workspace_root / "tolbert_failure_fallback" / "done.txt").read_text(encoding="utf-8") == "done\n"


def test_kernel_strict_live_llm_mode_continues_without_tolbert_context_after_retryable_compile_failure(tmp_path):
    class RetryableTolbertContextProvider:
        def compile(self, state):
            del state
            raise RuntimeError(
                "TOLBERT service exited before startup ready with code 1. "
                "checkpoint mismatch"
            )

    class LiveLLMClient:
        def create_decision(self, **kwargs):
            del kwargs
            return {
                "thought": "continue without Tolbert context",
                "action": "code_execute",
                "content": "printf 'done\\n' > done.txt",
                "done": False,
            }

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        asi_coding_require_live_llm=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="tolbert_failure_live_llm",
        prompt="complete a task",
        workspace_subdir="tolbert_failure_live_llm",
        expected_files=["done.txt"],
        expected_file_contents={"done.txt": "done\n"},
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        LiveLLMClient(),
        context_provider=RetryableTolbertContextProvider(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert episode.steps[0].decision_source == "llm"
    assert episode.steps[0].failure_origin == ""
    assert episode.steps[0].failure_signals == []
    assert episode.steps[0].proposal_metadata["context_compile_degraded"]["reason"] == "tolbert_startup_failure"
    assert (config.workspace_root / "tolbert_failure_live_llm" / "done.txt").read_text(encoding="utf-8") == "done\n"


def test_kernel_uses_progressive_deterministic_fallback_for_long_horizon_retryable_tolbert_failure(tmp_path):
    class RetryableTolbertContextProvider:
        def compile(self, state):
            del state
            raise RuntimeError(
                "context packet compilation failed: "
                "TOLBERT service exited before startup ready with code 1."
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=20,
    )
    task = TaskSpec(
        task_id="tolbert_failure_long_horizon_fallback",
        prompt="complete a long-horizon task",
        workspace_subdir="tolbert_failure_long_horizon_fallback",
        suggested_commands=[
            "mkdir -p reports audit",
            "rm -f drafts/obsolete.txt",
            "printf 'freeze locked\\nowner handoff\\nvalidation queued\\n' > plan/timeline.txt",
            "printf 'smoke checklist complete\\nrollback checklist complete\\n' > plan/checklist.txt",
            "printf 'cutover packet assembled\\n' > reports/packet.txt",
            "printf 'signoff captured\\n' > reports/signoff.txt",
            "printf 'project cutover recorded\\n' > audit/summary.txt",
        ],
        expected_files=[
            "notes/brief.txt",
            "docs/charter.md",
            "plan/timeline.txt",
            "plan/checklist.txt",
            "reports/packet.txt",
            "reports/signoff.txt",
            "audit/summary.txt",
        ],
        forbidden_files=["drafts/obsolete.txt"],
        expected_file_contents={
            "notes/brief.txt": "project brief\n",
            "docs/charter.md": "project charter\n",
            "plan/timeline.txt": "freeze locked\nowner handoff\nvalidation queued\n",
            "plan/checklist.txt": "smoke checklist complete\nrollback checklist complete\n",
            "reports/packet.txt": "cutover packet assembled\n",
            "reports/signoff.txt": "signoff captured\n",
            "audit/summary.txt": "project cutover recorded\n",
        },
        max_steps=20,
    )
    workspace = config.workspace_root / task.workspace_subdir
    (workspace / "notes").mkdir(parents=True, exist_ok=True)
    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    (workspace / "plan").mkdir(parents=True, exist_ok=True)
    (workspace / "drafts").mkdir(parents=True, exist_ok=True)
    (workspace / "notes" / "brief.txt").write_text("project brief\n", encoding="utf-8")
    (workspace / "docs" / "charter.md").write_text("project charter\n", encoding="utf-8")
    (workspace / "drafts" / "obsolete.txt").write_text("stale\n", encoding="utf-8")

    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=RetryableTolbertContextProvider(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task, clean_workspace=False)

    assert episode.success is True
    assert episode.steps[0].decision_source == "deterministic_fallback"
    assert episode.steps[0].action == "code_execute"
    assert episode.steps[0].content != "false"
    assert episode.steps[0].proposal_metadata["fallback_failure_origin"] == "retrieval_failure"
    assert (workspace / "reports" / "packet.txt").read_text(encoding="utf-8") == "cutover packet assembled\n"
    assert (workspace / "audit" / "summary.txt").read_text(encoding="utf-8") == "project cutover recorded\n"
    assert not (workspace / "drafts" / "obsolete.txt").exists()


def test_kernel_skips_learning_persistence_for_adjacent_success_followup(tmp_path, monkeypatch):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="adjacent_success_task",
        prompt="complete adjacent success",
        workspace_subdir="adjacent_success_task",
        suggested_commands=["printf 'done\\n' > done.txt"],
        expected_files=["done.txt"],
        expected_file_contents={"done.txt": "done\n"},
        metadata={"curriculum_kind": "adjacent_success", "benchmark_family": "repository"},
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        FailingLLMClient(),
        config=config,
    )

    def explode_persist(*args, **kwargs):
        raise AssertionError("adjacent_success followup should not persist learning candidates")

    monkeypatch.setattr("agent_kernel.loop.persist_episode_learning_candidates", explode_persist)

    episode = AgentKernel(config=config, policy=policy).run_task(task)

    assert episode.success is True
    assert episode.termination_reason == "success"


def test_kernel_can_resume_from_checkpoint_after_interrupt(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="resume_task",
        prompt="create one.txt and two.txt",
        workspace_subdir="resume_task",
        expected_files=["one.txt", "two.txt"],
        expected_file_contents={"one.txt": "one\n", "two.txt": "two\n"},
        max_steps=5,
    )
    checkpoint_path = tmp_path / "checkpoints" / "resume_task.json"

    with pytest.raises(KeyboardInterrupt):
        AgentKernel(config=config, policy=InterruptAfterFirstStepPolicy()).run_task(
            task,
            checkpoint_path=checkpoint_path,
        )

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "in_progress"
    assert len(checkpoint_payload["history"]) == 1
    assert "workspace_snapshot" in checkpoint_payload
    assert "latest_state_transition" in checkpoint_payload

    resumed = AgentKernel(config=config, policy=ResumeAfterCheckpointPolicy()).run_task(
        task,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed.success is True
    assert len(resumed.steps) == 2
    assert resumed.steps[0].content == "printf 'one\\n' > one.txt"
    assert resumed.steps[1].content == "printf 'two\\n' > two.txt"
    assert (config.workspace_root / "resume_task" / "one.txt").read_text(encoding="utf-8") == "one\n"
    assert (config.workspace_root / "resume_task" / "two.txt").read_text(encoding="utf-8") == "two\n"


def test_kernel_can_resume_after_setup_phase_interrupt(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="resume_setup_task",
        prompt="prepare seed.txt and prep.txt before creating done.txt",
        workspace_subdir="resume_setup_task",
        setup_commands=[
            "printf 'seed\\n' > seed.txt",
            "printf 'prep\\n' > prep.txt",
        ],
        expected_files=["seed.txt", "prep.txt", "done.txt"],
        expected_file_contents={
            "seed.txt": "seed\n",
            "prep.txt": "prep\n",
            "done.txt": "done\n",
        },
        max_steps=5,
    )
    checkpoint_path = tmp_path / "checkpoints" / "resume_setup_task.json"
    kernel = AgentKernel(config=config, policy=SetupResumePolicy())
    real_run = kernel.sandbox.run
    calls = {"count": 0}

    def interrupt_during_second_setup(command, cwd, *, task=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt("setup interrupted")
        return real_run(command, cwd, task=task)

    kernel.sandbox.run = interrupt_during_second_setup  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt):
        kernel.run_task(task, checkpoint_path=checkpoint_path)

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "setup_in_progress"
    assert checkpoint_payload["phase"] == "setup"
    assert len(checkpoint_payload["setup_history"]) == 1

    resumed = AgentKernel(config=config, policy=SetupResumePolicy()).run_task(
        task,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed.success is True
    workspace = config.workspace_root / "resume_setup_task"
    assert (workspace / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert (workspace / "prep.txt").read_text(encoding="utf-8") == "prep\n"
    assert (workspace / "done.txt").read_text(encoding="utf-8") == "done\n"


def test_kernel_compacts_runtime_history_for_deep_step_tasks(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        runtime_history_step_window=2,
        checkpoint_history_step_window=2,
        payload_history_step_window=2,
        history_archive_summary_max_chars=256,
        max_steps=10,
    )
    expected = {f"step_{index}.txt": f"step {index}\n" for index in range(1, 7)}
    task = TaskSpec(
        task_id="deep_step_compaction_task",
        prompt="Write six step artifacts in sequence.",
        workspace_subdir="deep_step_compaction_task",
        expected_files=list(expected),
        expected_file_contents=expected,
        max_steps=10,
        metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
    )

    episode = AgentKernel(config=config, policy=DeepStepSequencePolicy()).run_task(task)

    assert episode.success is True
    assert len(episode.steps) == 2
    assert episode.steps[0].index == 5
    assert episode.steps[1].index == 6
    assert episode.history_archive["archived_step_count"] == 4
    assert episode.history_archive["recent_archived_summaries"]


def test_kernel_resumes_from_compacted_checkpoint(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        runtime_history_step_window=2,
        checkpoint_history_step_window=2,
        payload_history_step_window=2,
        history_archive_summary_max_chars=256,
        max_steps=10,
    )
    expected = {f"step_{index}.txt": f"step {index}\n" for index in range(1, 7)}
    task = TaskSpec(
        task_id="deep_step_resume_task",
        prompt="Write six step artifacts in sequence.",
        workspace_subdir="deep_step_resume_task",
        expected_files=list(expected),
        expected_file_contents=expected,
        max_steps=10,
        metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
    )
    checkpoint_path = tmp_path / "checkpoints" / "deep_step_resume_task.json"

    with pytest.raises(KeyboardInterrupt):
        AgentKernel(config=config, policy=InterruptBeforeStepFivePolicy()).run_task(
            task,
            checkpoint_path=checkpoint_path,
        )

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "in_progress"
    assert len(checkpoint_payload["history"]) == 2
    assert checkpoint_payload["history"][0]["index"] == 3
    assert checkpoint_payload["history_archive"]["archived_step_count"] == 2

    resumed = AgentKernel(config=config, policy=DeepStepSequencePolicy()).run_task(
        task,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed.success is True
    assert resumed.steps[0].index == 5
    assert resumed.steps[1].index == 6
    assert resumed.history_archive["archived_step_count"] == 4


def test_kernel_build_plan_includes_repo_workflow_steps(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planner_controls": {
                    "prepend_verifier_contract_check": True,
                    "append_validation_subgoal": True,
                    "prefer_expected_artifacts_first": True,
                    "max_initial_subgoals": 10,
                },
                "role_directives": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        prompt_proposals_path=prompt_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_test_repair_task"))

    assert "prepare workflow branch fix/release-ready" in plan
    assert "update workflow path src/release_state.txt" in plan
    assert "run workflow test release test script" in plan


def test_graph_memory_summarizes_prior_documents(tmp_path):
    from agent_kernel.memory import EpisodeMemory
    from agent_kernel.schemas import EpisodeRecord, StepRecord

    memory = EpisodeMemory(tmp_path / "trajectories")
    memory.save(
        EpisodeRecord(
            task_id="hello_task",
            prompt="Create hello.txt containing hello agent kernel.",
            workspace=str(tmp_path / "workspace" / "hello_task"),
            success=True,
            steps=[
                StepRecord(
                    index=1,
                    thought="write file",
                    action="code_execute",
                    content="printf 'hello agent kernel\\n' > hello.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
            task_metadata={"benchmark_family": "micro"},
        )
    )

    summary = memory.graph_summary("hello_task_followup")

    assert summary["document_count"] == 1
    assert summary["benchmark_families"]["micro"] == 1
    assert summary["failure_types"] == {"other": 1}
    assert summary["related_tasks"] == ["hello_task"]


def test_task_semantic_recall_paths_collects_expected_and_semantic_verifier_paths():
    task = TaskSpec(
        task_id="release_followup",
        prompt="Repair release outputs and keep docs stable.",
        workspace_subdir="release_followup",
        expected_files=["reports/release_review.txt"],
        expected_file_contents={"src/release_state.txt": "ready\n"},
        metadata={
            "benchmark_family": "repository",
            "semantic_verifier": {
                "expected_changed_paths": ["src/release_state.txt", "generated/release.patch"],
                "report_rules": [
                    {"path": "reports/release_review.txt", "must_mention": ["ready"]},
                    {"path": "reports/summary.txt", "must_mention": ["stable"]},
                ],
            }
        },
    )

    assert _task_semantic_recall_paths(task) == [
        "reports/release_review.txt",
        "src/release_state.txt",
        "generated/release.patch",
        "reports/summary.txt",
    ]


def test_role_specialization_reaches_critic_on_failure(tmp_path):
    from agent_kernel.schemas import TaskSpec

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="critic_fail",
        prompt="run a command that succeeds",
        workspace_subdir="critic_fail",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=RepeatingFailPolicy()).run_task(task)

    assert any(step.acting_role == "critic" for step in episode.steps[1:])


def test_kernel_builds_vllm_provider(monkeypatch, tmp_path):
    captured = {}

    class FakeVLLMClient:
        def __init__(self, host, model_name, timeout_seconds, retry_attempts, retry_backoff_seconds, api_key):
            captured["host"] = host
            captured["model_name"] = model_name
            captured["timeout_seconds"] = timeout_seconds
            captured["retry_attempts"] = retry_attempts
            captured["retry_backoff_seconds"] = retry_backoff_seconds
            captured["api_key"] = api_key

    monkeypatch.setattr("agent_kernel.loop.VLLMClient", FakeVLLMClient)
    config = KernelConfig(
        provider="vllm",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        vllm_host="http://127.0.0.1:8000",
        vllm_api_key="secret",
    )

    kernel = AgentKernel(config=config)

    assert kernel.policy.client.__class__ is FakeVLLMClient
    assert captured["host"] == "http://127.0.0.1:8000"
    assert captured["model_name"] == config.model_name
    assert captured["api_key"] == "secret"


def test_kernel_matches_retrieval_command_when_guidance_uses_literal_newline():
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-03-16T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "retrieval_guidance": {
                "recommended_commands": ["printf 'hello agent kernel\n' > hello.txt"],
                "recommended_command_spans": [
                    {
                        "span_id": "task:hello:literal-newline",
                        "command": "printf 'hello agent kernel\n' > hello.txt",
                    }
                ],
                "avoidance_notes": [],
                "evidence": [],
            }
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    assert AgentKernel._retrieval_command_match(
        state,
        "printf 'hello agent kernel\\n' > hello.txt",
    )
