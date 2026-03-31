from __future__ import annotations

import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner
from agent_kernel.job_queue import DelegatedJobQueue
from agent_kernel.learning_compiler import load_learning_candidates, persist_episode_learning_candidates
from agent_kernel.memory import EpisodeMemory
from agent_kernel.modeling.training.universal_dataset import collect_universal_decoder_examples
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
