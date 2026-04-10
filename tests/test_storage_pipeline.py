from __future__ import annotations

import json
import os
from pathlib import Path

import agent_kernel.export_governance as export_governance
from agent_kernel.config import KernelConfig
from agent_kernel.export_governance import govern_improvement_export_storage
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner
from agent_kernel.job_queue import DelegatedJobQueue
from agent_kernel.learning_compiler import load_learning_candidates, persist_episode_learning_candidates
from agent_kernel.memory import EpisodeMemory
from agent_kernel.modeling.training.hybrid_dataset import materialize_hybrid_training_dataset
from agent_kernel.modeling.training.universal_dataset import collect_universal_decoder_examples
from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig
from agent_kernel.runtime_supervision import append_jsonl, atomic_write_json
from agent_kernel.schemas import EpisodeRecord, StepRecord


def _storage_config(tmp_path: Path, monkeypatch) -> KernelConfig:
    config = KernelConfig(
        storage_backend="sqlite",
        runtime_database_path=tmp_path / "var" / "runtime.sqlite3",
        storage_write_episode_exports=False,
        storage_write_learning_exports=False,
        storage_write_cycle_exports=False,
        storage_write_job_state_exports=False,
        trajectories_root=tmp_path / "trajectories" / "episodes",
        learning_artifacts_path=tmp_path / "trajectories" / "learning" / "run_learning_artifacts.json",
        improvement_cycles_path=tmp_path / "trajectories" / "improvement" / "cycles.jsonl",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        run_reports_dir=tmp_path / "trajectories" / "reports",
    )
    config.ensure_directories()
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("AGENT_KERNEL_RUNTIME_DATABASE_PATH", str(config.runtime_database_path))
    monkeypatch.setenv("AGENT_KERNEL_TRAJECTORIES_ROOT", str(config.trajectories_root))
    monkeypatch.setenv("AGENT_KERNEL_LEARNING_ARTIFACTS_PATH", str(config.learning_artifacts_path))
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", str(config.improvement_cycles_path))
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH", str(config.delegated_job_queue_path))
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH", str(config.delegated_job_runtime_state_path))
    return config


def _episode(task_id: str = "hello_task", *, success: bool = True) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task_id,
        prompt="create hello output",
        workspace=f"workspace/{task_id}",
        success=success,
        steps=[
            StepRecord(
                index=1,
                thought="run the command",
                action="code_execute",
                content="printf 'hello' > hello.txt",
                selected_skill_id=None,
                command_result={"exit_code": 0},
                verification={"passed": success, "reasons": ["verification passed"] if success else ["policy terminated"]},
                failure_signals=[] if success else ["no_state_progress"],
            )
        ],
        task_metadata={"benchmark_family": "bounded"},
        task_contract={"prompt": "create hello output"},
        universe_summary={"environment_snapshot": {"network_access_mode": "deny"}},
        world_model_summary={"completion_ratio": 1.0 if success else 0.0},
        termination_reason="success" if success else "policy_terminated",
    )


def test_episode_and_learning_candidates_persist_to_sqlite_without_exports(tmp_path: Path, monkeypatch) -> None:
    config = _storage_config(tmp_path, monkeypatch)
    memory = EpisodeMemory(config.trajectories_root, config=config)
    episode = _episode()

    episode_path = memory.save(episode)
    persist_episode_learning_candidates(episode, config=config, episode_storage={"relative_path": episode_path.name, "phase": "primary"})

    assert not episode_path.exists()
    assert not config.learning_artifacts_path.exists()

    loaded = memory.load(episode.task_id)
    assert loaded["task_id"] == episode.task_id
    assert loaded["episode_storage"]["relative_path"] == f"{episode.task_id}.json"

    candidates = load_learning_candidates(config.learning_artifacts_path, config=config)
    assert any(candidate["artifact_kind"] == "success_skill_candidate" for candidate in candidates)


def test_universal_dataset_reads_trajectories_from_sqlite_store(tmp_path: Path, monkeypatch) -> None:
    config = _storage_config(tmp_path, monkeypatch)
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "agent_kernel").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
    EpisodeMemory(config.trajectories_root, config=config).save(_episode())

    examples = collect_universal_decoder_examples(config=config, repo_root=repo_root)

    assert any(example["source_type"] == "trajectory_step" for example in examples)
    assert any("create hello output" in example["prompt"] for example in examples)


def test_sqlite_episode_storage_preserves_attempts_and_aggregates_by_task_id(tmp_path: Path, monkeypatch) -> None:
    config = _storage_config(tmp_path, monkeypatch)
    memory = EpisodeMemory(config.trajectories_root, config=config)

    primary_success = _episode("shared_task", success=True)
    primary_success.workspace = "workspace/shared_task"
    failed_retry = _episode("shared_task", success=False)
    failed_retry.workspace = "workspace/generated_failure_seed/shared_task"

    memory.save(primary_success)
    memory.save(failed_retry)

    attempts = config.sqlite_store().load_episode_attempt_documents("shared_task")
    assert len(attempts) == 2
    assert {payload["workspace"] for payload in attempts} == {
        "workspace/shared_task",
        "workspace/generated_failure_seed/shared_task",
    }
    assert {payload["episode_storage"]["phase"] for payload in attempts} == {
        "primary",
        "generated_failure_seed",
    }

    aggregated = memory.load("shared_task")
    assert aggregated["success"] is True
    assert aggregated["workspace"] == "workspace/shared_task"
    assert aggregated["attempt_aggregation"]["attempt_count"] == 2
    assert aggregated["attempt_aggregation"]["successful_attempts"] == 1
    assert aggregated["attempt_aggregation"]["attempts_by_phase"] == {
        "generated_failure_seed": 1,
        "primary": 1,
    }


def test_hybrid_dataset_removes_stale_shards_by_default(tmp_path: Path, monkeypatch) -> None:
    config = _storage_config(tmp_path, monkeypatch)
    episodes = config.trajectories_root
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "difficulty": "seed"},
                "steps": [
                    {
                        "action": "code_execute",
                        "content": "printf 'hello\n' > hello.txt",
                        "verification": {"passed": True},
                        "state_progress_delta": 0.5,
                        "state_regression_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dataset" / "hybrid_training_dataset.jsonl"
    stale_shards = output_path.parent / f"{output_path.stem}_shards"
    stale_shards.mkdir(parents=True, exist_ok=True)
    (stale_shards / "stale.jsonl").write_text('{"stale": true}\n', encoding="utf-8")

    manifest = materialize_hybrid_training_dataset(
        trajectories_root=episodes,
        output_path=output_path,
        config=HybridTolbertSSMConfig(sequence_length=4),
    )

    assert manifest["shard_paths"] == []
    assert output_path.exists()
    assert not stale_shards.exists()


def test_improvement_cycles_and_job_queue_round_trip_through_sqlite(tmp_path: Path, monkeypatch) -> None:
    config = _storage_config(tmp_path, monkeypatch)
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        runtime_config=config,
    )
    record = ImprovementCycleRecord(
        cycle_id="cycle:test:1",
        state="observe",
        subsystem="retrieval",
        action="run_eval",
        artifact_path="",
        artifact_kind="eval_metrics",
        reason="test record",
        metrics_summary={"scope_id": "scope_test", "scoped_run": True},
    )
    planner.append_cycle_record(config.improvement_cycles_path, record)

    loaded_records = planner.load_cycle_records(config.improvement_cycles_path)
    assert [item["cycle_id"] for item in loaded_records] == ["cycle:test:1"]
    assert not config.improvement_cycles_path.exists()

    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queued = queue.enqueue(task_id="hello_task", notes="sqlite-backed queue")
    loaded_jobs = queue.list_jobs()

    assert [job.job_id for job in loaded_jobs] == [queued.job_id]
    stored_jobs = config.sqlite_store().load_job_queue(queue_path=config.delegated_job_queue_path)
    assert stored_jobs[0]["job_id"] == queued.job_id


def _file_export_config(tmp_path: Path, monkeypatch) -> KernelConfig:
    config = KernelConfig(
        storage_backend="json",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        learning_artifacts_path=tmp_path / "trajectories" / "learning" / "run_learning_artifacts.json",
        improvement_cycles_path=tmp_path / "trajectories" / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "trajectories" / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "trajectories" / "improvement" / "reports",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        unattended_trust_ledger_path=tmp_path / "trajectories" / "reports" / "unattended_trust_ledger.json",
        storage_write_cycle_exports=True,
        storage_keep_cycle_export_files=1,
        storage_keep_report_export_files=1,
        storage_keep_candidate_export_dirs=1,
        storage_keep_run_report_files=1,
        storage_keep_run_checkpoint_files=1,
        storage_keep_namespace_candidate_dirs=1,
        storage_max_cycle_export_records=2,
        storage_max_report_history_records=2,
    )
    config.ensure_directories()
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_BACKEND", "json")
    monkeypatch.setenv("AGENT_KERNEL_TRAJECTORIES_ROOT", str(config.trajectories_root))
    monkeypatch.setenv("AGENT_KERNEL_LEARNING_ARTIFACTS_PATH", str(config.learning_artifacts_path))
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", str(config.improvement_cycles_path))
    monkeypatch.setenv("AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT", str(config.candidate_artifacts_root))
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", str(config.improvement_reports_dir))
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH", str(config.delegated_job_queue_path))
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH", str(config.delegated_job_runtime_state_path))
    monkeypatch.setenv("AGENT_KERNEL_RUN_REPORTS_DIR", str(config.run_reports_dir))
    monkeypatch.setenv("AGENT_KERNEL_RUN_CHECKPOINTS_DIR", str(config.run_checkpoints_dir))
    monkeypatch.setenv("AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH", str(config.unattended_trust_ledger_path))
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_CYCLE_EXPORT_FILES", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_REPORT_EXPORT_FILES", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_CANDIDATE_EXPORT_DIRS", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_RUN_REPORT_FILES", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_RUN_CHECKPOINT_FILES", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_NAMESPACE_CANDIDATE_DIRS", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_MAX_CYCLE_EXPORT_RECORDS", "2")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_MAX_REPORT_HISTORY_RECORDS", "2")
    return config


def test_cycle_export_compaction_and_file_pruning(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    old_cycles = config.improvement_cycles_path.parent / "cycles_old.jsonl"
    old_cycles.write_text('{"cycle_id":"old"}\n', encoding="utf-8")
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        runtime_config=config,
    )
    for idx in range(3):
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=f"cycle:test:{idx}",
                state="observe",
                subsystem="retrieval",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason="test record",
                metrics_summary={},
            ),
        )

    lines = config.improvement_cycles_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cycle_id"] == "cycle:test:1"
    assert json.loads(lines[1])["cycle_id"] == "cycle:test:2"
    assert not old_cycles.exists()


def test_report_history_compaction_and_sticky_report_preservation(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    sticky = config.improvement_reports_dir / "supervised_parallel_frontier.json"
    atomic_write_json(sticky, {"frontier": []})
    old_report = config.improvement_reports_dir / "campaign_report_old.json"
    atomic_write_json(old_report, {"old": True})
    new_report = config.improvement_reports_dir / "campaign_report_new.json"
    atomic_write_json(new_report, {"new": True})

    history_path = config.improvement_reports_dir / "supervisor_loop_history.jsonl"
    append_jsonl(history_path, {"round": 1})
    append_jsonl(history_path, {"round": 2})
    append_jsonl(history_path, {"round": 3})

    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 2
    assert json.loads(history_lines[0])["round"] == 2
    assert json.loads(history_lines[1])["round"] == 3
    assert sticky.exists()
    assert not old_report.exists()
    assert new_report.exists()


def test_atomic_write_json_uses_explicit_config_instead_of_env_roots(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", str(tmp_path / "wrong" / "reports"))
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", str(tmp_path / "wrong" / "improvement" / "cycles.jsonl"))
    monkeypatch.setenv("AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT", str(tmp_path / "wrong" / "improvement" / "candidates"))

    old_report = config.improvement_reports_dir / "campaign_report_old.json"
    new_report = config.improvement_reports_dir / "campaign_report_new.json"

    atomic_write_json(old_report, {"old": True}, config=config)
    atomic_write_json(new_report, {"new": True}, config=config)

    assert not old_report.exists()
    assert new_report.exists()


def test_append_jsonl_uses_explicit_config_for_history_compaction(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", str(tmp_path / "wrong" / "reports"))
    monkeypatch.setenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", str(tmp_path / "wrong" / "improvement" / "cycles.jsonl"))
    monkeypatch.setenv("AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT", str(tmp_path / "wrong" / "improvement" / "candidates"))

    history_path = config.improvement_reports_dir / "supervisor_loop_history.jsonl"
    append_jsonl(history_path, {"round": 1}, config=config)
    append_jsonl(history_path, {"round": 2}, config=config)
    append_jsonl(history_path, {"round": 3}, config=config)

    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 2
    assert json.loads(history_lines[0])["round"] == 2
    assert json.loads(history_lines[1])["round"] == 3


def test_run_report_and_checkpoint_pruning_preserves_trust_ledger(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    config.unattended_trust_ledger_path.write_text('{"ledger_kind":"unattended_trust_ledger"}', encoding="utf-8")

    old_report = config.run_reports_dir / "task_report_old.json"
    new_report = config.run_reports_dir / "task_report_new.json"
    old_checkpoint = config.run_checkpoints_dir / "old_checkpoint.json"
    new_checkpoint = config.run_checkpoints_dir / "new_checkpoint.json"

    atomic_write_json(old_report, {"task": "old"}, config=config)
    atomic_write_json(old_checkpoint, {"checkpoint": "old"}, config=config)
    atomic_write_json(new_report, {"task": "new"}, config=config)
    atomic_write_json(new_checkpoint, {"checkpoint": "new"}, config=config)

    assert not old_report.exists()
    assert new_report.exists()
    assert not old_checkpoint.exists()
    assert new_checkpoint.exists()
    assert config.unattended_trust_ledger_path.exists()


def test_job_queue_prunes_old_terminal_records_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    config = KernelConfig(
        storage_backend="json",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        storage_keep_terminal_job_records=1,
        storage_prune_terminal_job_artifacts=True,
    )
    config.ensure_directories()
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_BACKEND", "json")
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH", str(config.delegated_job_queue_path))
    monkeypatch.setenv("AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH", str(config.delegated_job_runtime_state_path))
    monkeypatch.setenv("AGENT_KERNEL_RUN_REPORTS_DIR", str(config.run_reports_dir))
    monkeypatch.setenv("AGENT_KERNEL_RUN_CHECKPOINTS_DIR", str(config.run_checkpoints_dir))
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_KEEP_TERMINAL_JOB_RECORDS", "1")
    monkeypatch.setenv("AGENT_KERNEL_STORAGE_PRUNE_TERMINAL_JOB_ARTIFACTS", "1")

    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    first = queue.enqueue(task_id="first_task")
    first_checkpoint = config.run_checkpoints_dir / "first.json"
    first_report = config.run_reports_dir / "first.json"
    first_checkpoint.write_text("{}", encoding="utf-8")
    first_report.write_text("{}", encoding="utf-8")
    queue.finalize(
        first.job_id,
        state="completed",
        checkpoint_path=first_checkpoint,
        report_path=first_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    second = queue.enqueue(task_id="second_task")
    second_checkpoint = config.run_checkpoints_dir / "second.json"
    second_report = config.run_reports_dir / "second.json"
    second_checkpoint.write_text("{}", encoding="utf-8")
    second_report.write_text("{}", encoding="utf-8")
    queue.finalize(
        second.job_id,
        state="completed",
        checkpoint_path=second_checkpoint,
        report_path=second_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    jobs = queue.list_jobs()

    assert [job.job_id for job in jobs] == [second.job_id]
    assert queue.get(first.job_id) is None
    assert not first_checkpoint.exists()
    assert not first_report.exists()
    assert second_checkpoint.exists()
    assert second_report.exists()


def test_report_history_compaction_uses_atomic_rewrite(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    history_path = config.improvement_reports_dir / "supervisor_loop_history.jsonl"
    history_path.write_text('{"round": 1}\n{"round": 2}\n{"round": 3}\n', encoding="utf-8")

    captured: dict[str, object] = {}
    original = export_governance._atomic_write_text

    def wrapped(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        captured["path"] = path
        captured["content"] = content
        original(path, content, encoding=encoding)

    monkeypatch.setattr(export_governance, "_atomic_write_text", wrapped)

    govern_improvement_export_storage(config)

    assert captured["path"] == history_path
    assert '"round": 2' in str(captured["content"])
    assert '"round": 3' in str(captured["content"])


def test_candidate_export_governance_prunes_scope_and_namespace_dirs(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    scope_old = config.candidate_artifacts_root / "scope_old"
    scope_new = config.candidate_artifacts_root / "scope_new"
    for idx, path in enumerate((scope_old, scope_new), start=1):
        (path / "tooling").mkdir(parents=True, exist_ok=True)
        (path / "tooling" / "artifact.json").write_text("{}", encoding="utf-8")
        path.touch()
        os.utime(path, (100 + idx, 100 + idx))

    policy_old = config.candidate_artifacts_root / "policy" / "cycle_policy_old"
    policy_new = config.candidate_artifacts_root / "policy" / "cycle_policy_new"
    for idx, path in enumerate((policy_old, policy_new), start=1):
        path.mkdir(parents=True, exist_ok=True)
        (path / "artifact.json").write_text("{}", encoding="utf-8")
        os.utime(path, (200 + idx, 200 + idx))

    govern_improvement_export_storage(config, preserve_paths=(scope_new / "tooling" / "artifact.json",))

    assert not scope_old.exists()
    assert scope_new.exists()
    assert not policy_old.exists()
    assert policy_new.exists()


def test_candidate_export_governance_keeps_scope_with_newer_nested_artifact(tmp_path: Path, monkeypatch) -> None:
    config = _file_export_config(tmp_path, monkeypatch)
    scope_old = config.candidate_artifacts_root / "scope_old"
    scope_new = config.candidate_artifacts_root / "scope_new"
    old_artifact = scope_old / "tooling" / "artifact.json"
    new_artifact = scope_new / "tooling" / "artifact.json"
    old_artifact.parent.mkdir(parents=True, exist_ok=True)
    new_artifact.parent.mkdir(parents=True, exist_ok=True)
    old_artifact.write_text("{}", encoding="utf-8")
    new_artifact.write_text("{}", encoding="utf-8")
    os.utime(old_artifact, (100, 100))
    os.utime(new_artifact, (200, 200))

    govern_improvement_export_storage(config)

    assert not scope_old.exists()
    assert scope_new.exists()
