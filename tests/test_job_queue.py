import importlib.util
import argparse
import json
from pathlib import Path
from io import StringIO
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.ops.job_queue import (
    DelegatedJobQueue,
    DelegatedRuntimeController,
    delegated_job_progress_path,
    drain_delegated_jobs,
    enqueue_with_parallel_worker_decomposition,
    run_next_delegated_job,
)
from agent_kernel.schemas import EpisodeRecord, TaskSpec
from agent_kernel.tasking.task_bank import TaskBank


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_successful_kernel(monkeypatch):
    from agent_kernel.ops import job_queue as job_queue_module
    from agent_kernel.ops.shared_repo import prepare_runtime_task

    class SuccessfulKernel:
        def __init__(self, config):
            self.config = config

        def run_task(
            self,
            task,
            checkpoint_path=None,
            resume=False,
            runtime_overrides=None,
            job_id=None,
            progress_callback=None,
        ):
            del resume
            runtime_task = prepare_runtime_task(
                task,
                runtime_overrides=dict(runtime_overrides or {}),
                job_id=job_id,
            )
            workspace = self.config.workspace_root / runtime_task.workspace_subdir
            workspace.mkdir(parents=True, exist_ok=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "step_start",
                        "step_index": 1,
                        "step_stage": "decision_pending",
                    }
                )
            expected_contents = dict(getattr(runtime_task, "expected_file_contents", {}))
            for relative_path, content in expected_contents.items():
                target = workspace / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content), encoding="utf-8")
            for relative_path in getattr(runtime_task, "expected_files", []):
                target = workspace / relative_path
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("ok\n", encoding="utf-8")
            episode = EpisodeRecord(
                task_id=runtime_task.task_id,
                prompt=runtime_task.prompt,
                workspace=str(workspace),
                success=True,
                steps=[],
                task_metadata=dict(getattr(runtime_task, "metadata", {})),
                task_contract={
                    "prompt": runtime_task.prompt,
                    "workspace_subdir": runtime_task.workspace_subdir,
                    "setup_commands": list(getattr(runtime_task, "setup_commands", [])),
                    "success_command": getattr(runtime_task, "success_command", ""),
                    "suggested_commands": list(getattr(runtime_task, "suggested_commands", [])),
                    "expected_files": list(getattr(runtime_task, "expected_files", [])),
                    "expected_output_substrings": list(getattr(runtime_task, "expected_output_substrings", [])),
                    "forbidden_files": list(getattr(runtime_task, "forbidden_files", [])),
                    "forbidden_output_substrings": list(getattr(runtime_task, "forbidden_output_substrings", [])),
                    "expected_file_contents": dict(getattr(runtime_task, "expected_file_contents", {})),
                    "max_steps": int(getattr(runtime_task, "max_steps", 1)),
                    "metadata": dict(getattr(runtime_task, "metadata", {})),
                },
                termination_reason="verification_passed",
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "success": True,
                            "termination_reason": episode.termination_reason,
                            "workspace": episode.workspace,
                            "episode": {
                                "task_id": episode.task_id,
                                "prompt": episode.prompt,
                                "workspace": episode.workspace,
                                "success": episode.success,
                                "steps": [],
                                "task_metadata": episode.task_metadata,
                                "task_contract": episode.task_contract,
                                "termination_reason": episode.termination_reason,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "step_complete",
                        "step_index": 1,
                        "step_stage": "step_complete",
                        "verification_passed": True,
                    }
            )
            return episode

        def close(self):
            return None

    monkeypatch.setattr(job_queue_module, "AgentKernel", SuccessfulKernel)


def test_runtime_overrides_parse_asi_coding_live_llm_flag():
    module = _load_script_module("run_job_queue.py")

    overrides = module._runtime_overrides_from_args(
        argparse.Namespace(
            provider=None,
            model=None,
            parallel_worker_count=None,
            shared_repo_id=None,
            worker_branch=None,
            target_branch=None,
            claim_path=None,
            use_tolbert_context=None,
            use_skills=None,
            use_graph_memory=None,
            use_world_model=None,
            use_planner=None,
            use_role_specialization=None,
            use_prompt_proposals=None,
            use_curriculum_proposals=None,
            use_retrieval_proposals=None,
            use_state_estimation_proposals=None,
            use_trust_proposals=None,
            use_recovery_proposals=None,
            use_delegation_proposals=None,
            use_operator_policy_proposals=None,
            use_transition_model_proposals=None,
            asi_coding_require_live_llm="1",
            allow_git_commands=None,
            allow_http_requests=None,
            allow_generated_path_mutations=None,
        )
    )

    assert overrides["asi_coding_require_live_llm"] is True


def test_queue_claim_next_respects_deadline_then_priority(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "jobs" / "queue.json")
    expired = queue.enqueue(task_id="hello_task", priority=10, deadline_at="2000-01-01T00:00:00+00:00")
    low = queue.enqueue(task_id="math_task", priority=1)
    high = queue.enqueue(task_id="nested_file_task", priority=5)

    claimed = queue.claim_next()

    assert claimed is not None
    assert claimed.job_id == high.job_id
    assert queue.get(expired.job_id).state == "expired"
    assert queue.get(low.job_id).state == "queued"
    assert queue.get(high.job_id).state == "in_progress"


def test_queue_does_not_duplicate_claim_live_in_progress_job_by_default(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "jobs" / "queue.json")
    first = queue.enqueue(task_id="hello_task", priority=5)
    second = queue.enqueue(task_id="math_task", priority=1)

    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.job_id == first.job_id

    duplicate = queue.claim(first.job_id)
    next_claimed = queue.claim_next()

    assert duplicate is None
    assert next_claimed is not None
    assert next_claimed.job_id == second.job_id
    updated = queue.get(first.job_id)
    assert updated is not None
    assert updated.attempt_count == 1


def test_queue_list_jobs_tolerates_invalid_json_snapshot(tmp_path):
    queue_path = tmp_path / "jobs" / "queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("{", encoding="utf-8")

    queue = DelegatedJobQueue(queue_path)

    assert queue.list_jobs() == []


def test_queue_cancel_marks_pending_and_resumable_jobs(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "jobs" / "queue.json")
    resumed = queue.enqueue(task_id="math_task", priority=5)
    queued = queue.enqueue(task_id="hello_task", priority=1)
    queue.claim_next()
    cancel_requested = queue.cancel(resumed.job_id)
    cancelled = queue.cancel(queued.job_id)

    assert cancelled.state == "cancelled"
    assert cancel_requested.state == "cancel_requested"


def test_run_next_delegated_job_completes_mock_job(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        sandbox_command_containment_mode="disabled",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    enqueued = queue.enqueue(task_id="hello_task")

    job = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        enforce_preflight=False,
    )

    assert job is not None
    assert job.job_id == enqueued.job_id
    assert job.state == "completed"
    assert job.outcome == "success"
    assert Path(job.report_path).exists()
    checkpoint_payload = json.loads(Path(job.checkpoint_path).read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "completed"


def test_run_next_delegated_job_completes_worker_command_job(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        sandbox_command_containment_mode="disabled",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(
        task_id="worker_command_job",
        runtime_overrides={
            "task_payload": TaskSpec(
                task_id="worker_command_job",
                prompt="Create a file through the delegated worker command runner.",
                workspace_subdir="worker_command_job",
                success_command="test -f produced.txt && grep -q ok produced.txt",
                max_steps=1,
                metadata={"benchmark_family": "tooling", "capability": "python"},
            ).to_dict(),
            "worker_command": "printf 'ok\\n' > produced.txt",
        },
    )

    job = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        enforce_preflight=False,
    )

    assert job is not None
    assert job.state == "completed"
    assert job.outcome == "success"
    report_payload = json.loads(Path(job.report_path).read_text(encoding="utf-8"))
    assert report_payload["outcome"] == "success"
    assert (config.workspace_root / "worker_command_job" / "produced.txt").exists()


def test_run_next_delegated_job_persists_worker_command_diagnostics_in_checkpoint(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(
        task_id="worker_command_failure_job",
        runtime_overrides={
            "task_payload": TaskSpec(
                task_id="worker_command_failure_job",
                prompt="Fail through the delegated worker command runner.",
                workspace_subdir="worker_command_failure_job",
                success_command="test -f produced.txt",
                max_steps=1,
                metadata={"benchmark_family": "tooling", "capability": "python"},
            ).to_dict(),
            "worker_command": "printf 'worker stdout\\n'; printf 'worker stderr\\n' >&2; exit 7",
        },
    )

    job = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        enforce_preflight=False,
    )

    assert job is not None
    assert job.state == "safe_stop"
    assert job.outcome == "safe_stop"
    checkpoint_payload = json.loads(Path(job.checkpoint_path).read_text(encoding="utf-8"))
    assert checkpoint_payload["termination_reason"] == "worker_command_failed"
    assert checkpoint_payload["exit_code"] == 7
    assert checkpoint_payload["timed_out"] is False
    assert "exit code was 7" in checkpoint_payload["verification_reasons"]
    assert checkpoint_payload["verification"]["passed"] is False
    assert checkpoint_payload["verification"]["outcome_label"] == "command_failure"
    assert "command_failure" in checkpoint_payload["verification"]["failure_codes"]
    assert checkpoint_payload["command_result"]["exit_code"] == 7
    assert checkpoint_payload["task_contract_summary"]["benchmark_family"] == "tooling"
    assert "worker stdout" in checkpoint_payload["stdout_summary"]["tail"]
    assert "worker stderr" in checkpoint_payload["stderr_summary"]["tail"]
    assert checkpoint_payload["stdout_summary"]["line_count"] == 1
    assert checkpoint_payload["stderr_summary"]["line_count"] == 1


def test_run_next_delegated_job_persists_success_command_failure_diagnostics_in_checkpoint(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(
        task_id="worker_command_success_contract_failure_job",
        runtime_overrides={
            "task_payload": TaskSpec(
                task_id="worker_command_success_contract_failure_job",
                prompt="Exit cleanly but fail the success contract.",
                workspace_subdir="worker_command_success_contract_failure_job",
                success_command="test -f produced.txt",
                max_steps=1,
                metadata={"benchmark_family": "repository", "capability": "shell"},
            ).to_dict(),
            "worker_command": "printf 'command ok\\n'",
        },
    )

    job = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        enforce_preflight=False,
    )

    assert job is not None
    assert job.state == "safe_stop"
    checkpoint_payload = json.loads(Path(job.checkpoint_path).read_text(encoding="utf-8"))
    assert checkpoint_payload["verification"]["passed"] is False
    assert checkpoint_payload["verification"]["outcome_label"] == "success_command_failed"
    assert "success_command_failed" in checkpoint_payload["verification"]["failure_codes"]
    assert checkpoint_payload["success_command_result"]["command"] == "test -f produced.txt"
    assert checkpoint_payload["success_command_result"]["exit_code"] == 1
    assert checkpoint_payload["task_contract_summary"]["benchmark_family"] == "repository"


def test_run_next_delegated_job_writes_report_on_runner_exception(monkeypatch, tmp_path):
    from agent_kernel.ops import job_queue as job_queue_module

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    enqueued = queue.enqueue(task_id="hello_task")

    class FailingKernel:
        def __init__(self, config):
            del config

        def run_task(self, task, checkpoint_path=None, resume=False, **kwargs):
            del task
            del checkpoint_path
            del resume
            del kwargs
            raise RuntimeError("runner exploded")

        def close(self):
            return None

    monkeypatch.setattr(job_queue_module, "AgentKernel", FailingKernel)

    job = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert job is not None
    assert job.job_id == enqueued.job_id
    assert job.state == "failed"
    payload = json.loads(Path(job.report_path).read_text(encoding="utf-8"))
    assert payload["outcome"] == "unsafe_ambiguous"
    assert payload["outcome_reasons"] == ["runner_exception"]
    assert payload["termination_reason"] == "RuntimeError: runner exploded"


def test_run_next_delegated_job_safe_stops_when_acceptance_verifier_fails(monkeypatch, tmp_path):
    from agent_kernel.ops import job_queue as job_queue_module
    from agent_kernel.schemas import EpisodeRecord

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(task_id="hello_task")

    class PassingKernel:
        def __init__(self, config):
            self.config = config

        def run_task(self, task, checkpoint_path=None, resume=False, **kwargs):
            del checkpoint_path
            del resume
            del kwargs
            workspace = self.config.workspace_root / task.workspace_subdir
            workspace.mkdir(parents=True, exist_ok=True)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(workspace),
                success=True,
                steps=[],
                termination_reason="verification_passed",
            )

        def close(self):
            return None

    def fake_write_report(**kwargs):
        report_path = kwargs["report_path"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "unattended_task_report",
                    "acceptance_packet": {
                        "target_branch": "main",
                        "expected_branch": "worker/api-status",
                        "verifier_result": {"passed": False},
                    },
                }
            ),
            encoding="utf-8",
        )
        return report_path

    monkeypatch.setattr(job_queue_module, "AgentKernel", PassingKernel)
    monkeypatch.setattr(job_queue_module, "write_unattended_task_report", fake_write_report)

    job = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert job is not None
    assert job.state == "safe_stop"
    assert job.outcome == "safe_stop"
    assert "acceptance_verifier_failed" in job.outcome_reasons
    assert "acceptance packet verifier did not pass" in job.last_error


def test_run_next_delegated_job_ignores_non_promotion_acceptance_packets(monkeypatch, tmp_path):
    from agent_kernel.ops import job_queue as job_queue_module
    from agent_kernel.schemas import EpisodeRecord

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(task_id="hello_task")

    class PassingKernel:
        def __init__(self, config):
            self.config = config

        def run_task(self, task, checkpoint_path=None, resume=False, **kwargs):
            del checkpoint_path
            del resume
            del kwargs
            workspace = self.config.workspace_root / task.workspace_subdir
            workspace.mkdir(parents=True, exist_ok=True)
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(workspace),
                success=True,
                steps=[],
                termination_reason="verification_passed",
            )

        def close(self):
            return None

    def fake_write_report(**kwargs):
        report_path = kwargs["report_path"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "unattended_task_report",
                    "acceptance_packet": {
                        "synthetic_worker": False,
                        "expected_branch": "",
                        "target_branch": "",
                        "required_merged_branches": [],
                        "diff_base_ref": "",
                        "selected_edits": [],
                        "candidate_edit_sets": [],
                        "tests": [],
                        "report_rules": [],
                        "verifier_result": {
                            "passed": False,
                            "outcome": "safe_stop",
                            "reasons": ["worker_command_failed"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return report_path

    monkeypatch.setattr(job_queue_module, "AgentKernel", PassingKernel)
    monkeypatch.setattr(job_queue_module, "write_unattended_task_report", fake_write_report)

    job = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert job is not None
    assert job.state == "completed"
    assert job.outcome == "success"
    assert "acceptance_verifier_failed" not in job.outcome_reasons
    assert job.last_error == ""


def test_drain_stops_after_interrupted_job(monkeypatch, tmp_path):
    from agent_kernel.ops import job_queue as job_queue_module

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(task_id="hello_task")
    queue.enqueue(task_id="math_task")

    class InterruptingKernel:
        def __init__(self, config):
            del config

        def run_task(self, task, checkpoint_path=None, resume=False, **kwargs):
            del task
            del checkpoint_path
            del resume
            del kwargs
            raise KeyboardInterrupt("stop here")

        def close(self):
            return None

    monkeypatch.setattr(job_queue_module, "AgentKernel", InterruptingKernel)
    drained = drain_delegated_jobs(queue, limit=0, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert len(drained) == 1
    assert drained[0].state == "in_progress"
    remaining = queue.list_jobs()
    assert remaining[0].state == "in_progress"
    assert remaining[1].state == "queued"


def test_run_next_delegated_job_resumes_interrupted_job_without_active_lease(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    resumed = queue.enqueue(task_id="hello_task", priority=5)
    queued = queue.enqueue(task_id="math_task", priority=1)
    queue.mark_interrupted(
        resumed.job_id,
        checkpoint_path=config.run_checkpoints_dir / "hello.json",
        report_path=config.run_reports_dir / "hello.json",
        error="runner interrupted",
    )

    claimed = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert claimed is not None
    assert claimed.job_id == resumed.job_id
    assert claimed.state == "completed"
    assert queue.get(queued.job_id).state == "queued"


def test_run_next_delegated_job_reaps_stale_lease_and_resumes_orphaned_job(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    job = queue.enqueue(task_id="hello_task", priority=5)
    claimed = queue.claim(job.job_id)
    assert claimed is not None
    checkpoint_path = config.run_checkpoints_dir / "hello.json"
    report_path = config.run_reports_dir / "hello.json"
    queue.set_paths(job.job_id, checkpoint_path=checkpoint_path, report_path=report_path)
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path)
    task = TaskBank(config=config).get("hello_task")
    lease, reason = controller.acquire(
        job=claimed,
        task=task,
        config=config,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
    )
    assert lease is not None
    assert reason is None

    runtime_payload = json.loads(config.delegated_job_runtime_state_path.read_text(encoding="utf-8"))
    runtime_payload["active_leases"][0]["runner_pid"] = 999_999_999
    config.delegated_job_runtime_state_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

    resumed = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert resumed is not None
    assert resumed.job_id == job.job_id
    assert resumed.state == "completed"
    assert resumed.outcome == "success"
    final_job = queue.get(job.job_id)
    assert final_job is not None
    assert final_job.attempt_count == 2
    final_runtime = json.loads(config.delegated_job_runtime_state_path.read_text(encoding="utf-8"))
    assert final_runtime["active_leases"] == []
    assert any(entry["event"] == "lease_reaped" for entry in final_runtime["history"])


def test_run_next_delegated_job_skips_live_leased_in_progress_job(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    live = queue.enqueue(task_id="hello_task", priority=5)
    queued = queue.enqueue(task_id="math_task", priority=1)
    live = queue.claim(live.job_id)
    assert live is not None
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    task = TaskBank(config=config).get("hello_task")
    lease, reason = controller.acquire(
        job=live,
        task=task,
        config=config,
        checkpoint_path=config.run_checkpoints_dir / "live.json",
        report_path=config.run_reports_dir / "live.json",
    )

    claimed = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        runtime_controller=controller,
    )

    assert lease is not None
    assert reason is None
    assert claimed is not None
    assert claimed.job_id == queued.job_id
    assert claimed.state == "completed"
    assert queue.get(live.job_id).state == "in_progress"


def test_run_job_queue_enqueue_cli(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "enqueue", "--task-id", "hello_task", "--priority", "3"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue().strip()
    assert "task_id=hello_task" in output
    assert "priority=3" in output


def test_run_job_queue_enqueue_cli_records_shared_repo_claims(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_queue.py",
            "enqueue",
            "--task-id",
            "git_parallel_merge_acceptance_task",
            "--shared-repo-id",
            "repo-a",
            "--worker-branch",
            "worker/api-status",
            "--target-branch",
            "main",
            "--claim-path",
            "src/api_status.txt",
            "--claim-path",
            "reports/merge_report.txt",
        ],
    )

    module.main()

    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    job = queue.list_jobs()[0]
    assert job.runtime_overrides["shared_repo_id"] == "repo-a"
    assert job.runtime_overrides["worker_branch"] == "worker/api-status"
    assert job.runtime_overrides["target_branch"] == "main"
    assert job.runtime_overrides["claimed_paths"] == [
        "src/api_status.txt",
        "reports/merge_report.txt",
    ]


def test_queue_enqueue_infers_family_budget_group_from_task_payload(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")

    job = queue.enqueue(
        task_id="synthetic_tooling_job",
        runtime_overrides={
            "task_payload": TaskSpec(
                task_id="synthetic_tooling_job",
                prompt="run a synthetic tooling worker",
                workspace_subdir="synthetic_tooling_job",
                success_command="true",
                max_steps=1,
                metadata={"benchmark_family": "tooling", "capability": "python"},
            ).to_dict(),
        },
    )

    assert job.budget_group == "family_tooling"


def test_run_job_queue_enqueue_cli_infers_family_budget_group_for_default_budget(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_job_queue.py", "enqueue", "--task-id", "git_parallel_merge_acceptance_task"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue().strip()
    assert "budget_group=family_repo_sandbox" in output
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    assert queue.list_jobs()[0].budget_group == "family_repo_sandbox"


def test_run_job_queue_enqueue_manifest_cli_enqueues_external_workstream_json(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    manifest_path = tmp_path / "workstream.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "external_cli_project_task",
                        "prompt": "Create project.txt.",
                        "workspace_subdir": "external_cli_project_task",
                        "suggested_commands": ["printf 'project ready\\n' > project.txt"],
                        "success_command": "test -f project.txt",
                        "expected_files": ["project.txt"],
                        "metadata": {"benchmark_family": "project", "capability": "cli_workstream"},
                    },
                    {
                        "task_id": "external_cli_repository_task",
                        "prompt": "Create repository.txt.",
                        "workspace_subdir": "external_cli_repository_task",
                        "suggested_commands": ["printf 'repository ready\\n' > repository.txt"],
                        "success_command": "test -f repository.txt",
                        "expected_files": ["repository.txt"],
                        "metadata": {"benchmark_family": "repository", "capability": "cli_workstream"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_queue.py",
            "enqueue-manifest",
            "--manifest-path",
            str(manifest_path),
            "--priority-start",
            "9",
            "--json",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["selected_task_count"] == 2
    assert payload["enqueued_job_count"] == 2
    assert [job["task_id"] for job in payload["enqueued_jobs"]] == [
        "external_cli_project_task",
        "external_cli_repository_task",
    ]
    assert [job["priority"] for job in payload["enqueued_jobs"]] == [9, 8]
    assert [job["budget_group"] for job in payload["enqueued_jobs"]] == [
        "family_project",
        "family_repository",
    ]
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    jobs = queue.list_jobs()
    assert [job.task_id for job in jobs] == ["external_cli_project_task", "external_cli_repository_task"]
    assert [job.budget_group for job in jobs] == ["family_project", "family_repository"]
    assert jobs[0].runtime_overrides["task_payload"]["metadata"]["task_origin"] == "external_manifest"
    assert jobs[0].runtime_overrides["task_payload"]["metadata"]["external_manifest_path"] == str(manifest_path)


def test_run_job_queue_enqueue_manifest_cli_honors_max_queued_budget_override(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    manifest_path = tmp_path / "workstream.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "external_cli_patch_task_1",
                        "prompt": "Create one.txt.",
                        "workspace_subdir": "external_cli_patch_task_1",
                        "suggested_commands": ["printf 'one\\n' > one.txt"],
                        "success_command": "test -f one.txt",
                        "expected_files": ["one.txt"],
                        "metadata": {"benchmark_family": "patch", "capability": "cli_workstream"},
                    },
                    {
                        "task_id": "external_cli_patch_task_2",
                        "prompt": "Create two.txt.",
                        "workspace_subdir": "external_cli_patch_task_2",
                        "suggested_commands": ["printf 'two\\n' > two.txt"],
                        "success_command": "test -f two.txt",
                        "expected_files": ["two.txt"],
                        "metadata": {"benchmark_family": "patch", "capability": "cli_workstream"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            delegated_job_max_queued_per_budget_group=1,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_queue.py",
            "enqueue-manifest",
            "--manifest-path",
            str(manifest_path),
            "--budget-group",
            "same_patch_budget",
            "--max-queued-per-budget-group",
            "2",
            "--json",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["enqueued_job_count"] == 2
    assert [job["budget_group"] for job in payload["enqueued_jobs"]] == [
        "same_patch_budget",
        "same_patch_budget",
    ]


def test_a7_unfamiliar_transfer_manifest_loads_as_held_out_external_tasks():
    manifest_path = Path(__file__).resolve().parents[1] / "config" / "a7_unfamiliar_transfer_manifest.json"
    bank = TaskBank(external_task_manifests=(str(manifest_path),))
    frontier_tasks = [
        task
        for task in bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(frontier_tasks) == 5
    assert {task.metadata["benchmark_family"] for task in frontier_tasks} == {
        "project",
        "repository",
        "integration",
        "repo_sandbox",
        "repo_chore",
    }
    assert all(task.metadata["task_origin"] == "external_manifest" for task in frontier_tasks)
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer" for task in frontier_tasks)
    assert all(task.expected_file_contents for task in frontier_tasks)


def test_a7_unfamiliar_transfer_rotation_two_is_non_identical_and_loadable():
    repo_root = Path(__file__).resolve().parents[1]
    first_manifest = repo_root / "config" / "a7_unfamiliar_transfer_manifest.json"
    second_manifest = repo_root / "config" / "a7_unfamiliar_transfer_manifest_r2.json"
    first_bank = TaskBank(external_task_manifests=(str(first_manifest),))
    second_bank = TaskBank(external_task_manifests=(str(second_manifest),))
    first_frontier = {
        task.task_id: task
        for task in first_bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    }
    second_frontier = [
        task
        for task in second_bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(second_frontier) == 5
    assert not set(first_frontier).intersection({task.task_id for task in second_frontier})
    assert {task.metadata["benchmark_family"] for task in second_frontier} == {
        "project",
        "repository",
        "integration",
        "repo_sandbox",
        "repo_chore",
    }
    assert {task.metadata["frontier_slice"] for task in second_frontier} == {
        "release_control_novelty",
        "schema_topology_novelty",
        "cross_service_recovery_novelty",
        "patch_queue_novelty",
        "retention_exception_novelty",
    }
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer" for task in second_frontier)
    assert all(task.expected_file_contents for task in second_frontier)


def test_a7_unfamiliar_transfer_rotation_three_is_hintless_and_loadable():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "config" / "a7_unfamiliar_transfer_manifest_r3_hard.json"
    bank = TaskBank(external_task_manifests=(str(manifest_path),))
    frontier_tasks = [
        task
        for task in bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(frontier_tasks) == 5
    assert {task.metadata["benchmark_family"] for task in frontier_tasks} == {
        "project",
        "repository",
        "integration",
        "repo_sandbox",
        "repo_chore",
    }
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer_hard" for task in frontier_tasks)
    assert all(task.metadata["suggested_commands_absent"] is True for task in frontier_tasks)
    assert all(not task.suggested_commands for task in frontier_tasks)
    assert all(task.expected_file_contents for task in frontier_tasks)
    assert {
        task.metadata["frontier_slice"]
        for task in frontier_tasks
    } == {
        "hintless_delivery_novelty",
        "hintless_api_migration_novelty",
        "hintless_failback_novelty",
        "hintless_sandbox_freeze_novelty",
        "hintless_legal_hold_novelty",
    }


def test_a7_unfamiliar_transfer_rotation_four_is_larger_hintless_scale():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "config" / "a7_unfamiliar_transfer_manifest_r4_hard_scale.json"
    bank = TaskBank(external_task_manifests=(str(manifest_path),))
    frontier_tasks = [
        task
        for task in bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(frontier_tasks) == 10
    family_counts = {}
    for task in frontier_tasks:
        family_counts[task.metadata["benchmark_family"]] = family_counts.get(task.metadata["benchmark_family"], 0) + 1
    assert family_counts == {
        "project": 2,
        "repository": 2,
        "integration": 2,
        "repo_sandbox": 2,
        "repo_chore": 2,
    }
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer_hard" for task in frontier_tasks)
    assert all(task.metadata["suggested_commands_absent"] is True for task in frontier_tasks)
    assert all(not task.suggested_commands for task in frontier_tasks)
    assert all(task.expected_file_contents for task in frontier_tasks)
    assert len({task.metadata["frontier_slice"] for task in frontier_tasks}) == 10
    assert not {
        "hintless_delivery_novelty",
        "hintless_api_migration_novelty",
        "hintless_failback_novelty",
        "hintless_sandbox_freeze_novelty",
        "hintless_legal_hold_novelty",
    }.intersection({task.metadata["frontier_slice"] for task in frontier_tasks})


def test_a7_unfamiliar_transfer_rotation_five_is_stateful_repair():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "config" / "a7_unfamiliar_transfer_manifest_r5_stateful_repair.json"
    bank = TaskBank(external_task_manifests=(str(manifest_path),))
    frontier_tasks = [
        task
        for task in bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(frontier_tasks) == 5
    assert {task.metadata["benchmark_family"] for task in frontier_tasks} == {
        "project",
        "repository",
        "integration",
        "repo_sandbox",
        "repo_chore",
    }
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer_stateful_hard" for task in frontier_tasks)
    assert all(task.metadata["suggested_commands_absent"] is True for task in frontier_tasks)
    assert all(task.setup_commands for task in frontier_tasks)
    assert all(not task.suggested_commands for task in frontier_tasks)
    assert all(task.expected_file_contents for task in frontier_tasks)
    assert all(task.forbidden_files for task in frontier_tasks)
    assert len({task.metadata["frontier_slice"] for task in frontier_tasks}) == 5


def test_a7_unfamiliar_transfer_rotation_six_is_diagnostic_synthesis():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "config" / "a7_unfamiliar_transfer_manifest_r6_diagnostic_synthesis.json"
    bank = TaskBank(external_task_manifests=(str(manifest_path),))
    frontier_tasks = [
        task
        for task in bank.list()
        if bool(task.metadata.get("held_out_frontier_task", False))
    ]

    assert len(frontier_tasks) == 5
    assert {task.metadata["benchmark_family"] for task in frontier_tasks} == {
        "project",
        "repository",
        "integration",
        "repo_sandbox",
        "repo_chore",
    }
    assert all(task.metadata["capability"] == "unfamiliar_environment_transfer_diagnostic_hard" for task in frontier_tasks)
    assert all("diagnostic_synthesis" in task.metadata["novelty_axes"] for task in frontier_tasks)
    assert all(task.metadata["suggested_commands_absent"] is True for task in frontier_tasks)
    assert all(task.setup_commands for task in frontier_tasks)
    assert all(not task.suggested_commands for task in frontier_tasks)
    assert all(task.expected_file_contents for task in frontier_tasks)
    assert all(task.metadata.get("workflow_guard", {}).get("managed_paths") for task in frontier_tasks)
    assert len({task.metadata["frontier_slice"] for task in frontier_tasks}) == 5


def test_enqueue_with_parallel_worker_decomposition_expands_integrator(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")

    jobs = enqueue_with_parallel_worker_decomposition(
        queue,
        bank=TaskBank(),
        task_id="git_parallel_merge_acceptance_task",
        priority=2,
        budget_group="campaign-a",
        notes="auto plan",
    )

    assert [job.task_id for job in jobs] == [
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
        "git_parallel_merge_acceptance_task",
    ]
    assert [job.priority for job in jobs] == [3, 3, 2]
    assert "[auto_worker_for:git_parallel_merge_acceptance_task]" in jobs[0].notes
    assert "[auto_integrator]" in jobs[-1].notes
    assert jobs[-1].runtime_overrides["dependency_job_ids"] == [jobs[0].job_id, jobs[1].job_id]
    assert jobs[-1].runtime_overrides["required_worker_branches"] == [
        "worker/api-status",
        "worker/docs-status",
    ]


def test_project_parallel_release_task_decomposes_into_project_family_workers(tmp_path):
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")

    jobs = enqueue_with_parallel_worker_decomposition(
        queue,
        bank=TaskBank(),
        task_id="project_parallel_release_task",
        priority=5,
    )

    assert [job.task_id for job in jobs] == [
        "project_parallel_release_task__worker__worker_api-plan",
        "project_parallel_release_task__worker__worker_ops-cutover",
        "project_parallel_release_task",
    ]
    assert [job.budget_group for job in jobs] == ["family_project", "family_project", "family_project"]
    assert jobs[0].runtime_overrides["task_payload"]["metadata"]["benchmark_family"] == "project"
    assert jobs[1].runtime_overrides["task_payload"]["metadata"]["benchmark_family"] == "project"
    assert jobs[-1].runtime_overrides["dependency_job_ids"] == [jobs[0].job_id, jobs[1].job_id]


def test_enqueue_with_parallel_worker_decomposition_expands_to_target_parallel_worker_count(tmp_path):
    bank = TaskBank()
    bank._tasks["five_way_integrator"] = TaskSpec(
        task_id="five_way_integrator",
        prompt="accept five worker branches",
        workspace_subdir="five_way_integrator",
        expected_files=[
            "src/api_status.txt",
            "docs/status.md",
            "config/app.env",
            "scripts/deploy.sh",
            "notes/release.txt",
        ],
        expected_file_contents={
            "src/api_status.txt": "API_STATUS=ready\n",
            "docs/status.md": "status ready documented\n",
            "config/app.env": "MODE=prod\n",
            "scripts/deploy.sh": "#!/bin/sh\necho deploy ready\n",
            "notes/release.txt": "release ready\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src docs config scripts notes && "
                "printf 'API_STATUS=pending\\n' > src/api_status.txt && "
                "printf 'status pending documented\\n' > docs/status.md && "
                "printf 'MODE=dev\\n' > config/app.env && "
                "printf '#!/bin/sh\\necho deploy pending\\n' > scripts/deploy.sh && "
                "printf 'release pending\\n' > notes/release.txt && "
                "git init && git checkout -b main && git config user.email agent@example.com && "
                "git config user.name 'Agent Kernel' && "
                "git add src/api_status.txt docs/status.md config/app.env scripts/deploy.sh notes/release.txt && "
                "git commit -m 'baseline five way sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": [
                "src/api_status.txt",
                "docs/status.md",
                "config/app.env",
                "scripts/deploy.sh",
                "notes/release.txt",
            ],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-five-way",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                "expected_changed_paths": [
                    "src/api_status.txt",
                    "docs/status.md",
                    "config/app.env",
                    "scripts/deploy.sh",
                    "notes/release.txt",
                ],
            },
        },
    )
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")

    jobs = enqueue_with_parallel_worker_decomposition(
        queue,
        bank=bank,
        task_id="five_way_integrator",
        priority=2,
        budget_group="campaign-a",
        runtime_overrides={"parallel_worker_count": 5},
    )

    assert len(jobs) == 6
    worker_branches = [
        job.runtime_overrides["task_payload"]["metadata"]["workflow_guard"]["worker_branch"]
        for job in jobs[:-1]
    ]
    assert worker_branches == jobs[-1].runtime_overrides["required_worker_branches"]
    assert len(worker_branches) == 5
    assert len(set(worker_branches)) == 5
    assert worker_branches[:2] == ["worker/api-status", "worker/docs-status"]
    assert jobs[-1].runtime_overrides["dependency_job_ids"] == [job.job_id for job in jobs[:-1]]


def test_run_job_queue_enqueue_cli_decomposes_parallel_workers(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_queue.py",
            "enqueue",
            "--task-id",
            "git_parallel_merge_acceptance_task",
            "--priority",
            "2",
            "--decompose-workers",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    assert "task_id=git_parallel_worker_api_task" in lines[0]
    assert "task_id=git_parallel_worker_docs_task" in lines[1]
    assert "task_id=git_parallel_merge_acceptance_task" in lines[2]


def test_run_next_delegated_job_defers_integrator_until_worker_dependencies_complete(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
        unattended_allow_git_commands=True,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    worker = queue.enqueue(task_id="git_parallel_worker_api_task", priority=1)
    integrator = queue.enqueue(
        task_id="git_parallel_merge_acceptance_task",
        priority=5,
        runtime_overrides={
            "dependency_job_ids": [worker.job_id],
            "required_worker_branches": ["worker/api-status"],
        },
    )

    claimed = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert claimed is not None
    assert claimed.job_id == worker.job_id
    deferred = queue.get(integrator.job_id)
    assert deferred is not None
    assert deferred.state == "queued"
    assert deferred.attempt_count == 0
    assert deferred.last_error == ""
    assert deferred.scheduler_blocked_count == 0
    assert deferred.scheduler_blocked_open is False


def test_run_next_delegated_job_defers_preflight_blocked_job_without_attempt(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        sandbox_command_containment_mode="disabled",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        unattended_allow_git_commands=False,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    blocked = queue.enqueue(task_id="git_parallel_worker_api_task", priority=5)
    runnable = queue.enqueue(task_id="hello_task", priority=1)

    claimed = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert claimed is not None
    assert claimed.job_id == runnable.job_id
    deferred = queue.get(blocked.job_id)
    assert deferred is not None
    assert deferred.state == "queued"
    assert deferred.attempt_count == 0
    assert deferred.last_error.startswith("preflight_blocked:operator_policy")


def test_run_next_delegated_job_writes_progress_sidecar(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queued = queue.enqueue(task_id="hello_task", priority=1)

    claimed = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert claimed is not None
    progress_payload = json.loads(delegated_job_progress_path(config, queued).read_text(encoding="utf-8"))
    assert progress_payload["job_id"] == queued.job_id
    assert progress_payload["task_id"] == "hello_task"
    assert progress_payload["event"] == "delegated_job_finished"
    assert progress_payload["terminal_state"] == "completed"
    assert progress_payload["outcome"] == "success"


def test_run_next_delegated_job_marks_blocked_job_ready_before_selection(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    job = queue.enqueue(task_id="hello_task", priority=1)
    queue.record_scheduler_decision(job.job_id, decision="deferred:dependency_blocked", detail="dependency_waiting:x")

    claimed = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert claimed is not None
    updated = queue.get(job.job_id)
    assert updated is not None
    assert updated.scheduler_blocked_open is False
    assert updated.scheduler_unblock_count == 1
    assert updated.scheduler_last_unblocked_at
    assert any(
        entry.get("event") == "scheduler_decision"
        and str(entry.get("detail", "")).startswith("ready:runnable|")
        for entry in updated.history
    )


def test_run_next_delegated_job_applies_bounded_fairness_boost(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    fresh = queue.enqueue(task_id="hello_task", priority=3)
    aged = queue.enqueue(task_id="math_task", priority=1)
    for _ in range(3):
        queue.record_scheduler_decision(aged.job_id, decision="deferred:dependency_blocked", detail="dependency_waiting:x")

    claimed = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert claimed is not None
    assert claimed.job_id == aged.job_id
    updated = queue.get(aged.job_id)
    assert updated is not None
    assert updated.scheduler_selected_count == 1
    assert updated.scheduler_unblock_count == 1


def test_run_next_delegated_job_rotates_budget_group_when_consecutive_limit_hit(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_consecutive_selections_per_budget_group=1,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    hot = queue.enqueue(task_id="hello_task", priority=5, budget_group="campaign-a")
    cool = queue.enqueue(task_id="math_task", priority=1, budget_group="campaign-b")
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    controller.record_budget_group_selection("campaign-a")

    claimed = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        runtime_controller=controller,
    )

    assert claimed is not None
    assert claimed.job_id == cool.job_id
    blocked = queue.get(hot.job_id)
    assert blocked is not None
    assert any(
        entry.get("event") == "scheduler_decision"
        and "deferred:anti_starvation_budget_group|" in str(entry.get("detail", ""))
        for entry in blocked.history
    )
    scheduler = controller.scheduler_state()
    assert scheduler["last_selected_budget_group"] == "campaign-b"
    assert scheduler["consecutive_budget_group_selections"] == 1


def test_run_next_delegated_job_allows_same_budget_group_when_no_other_runnable_exists(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_consecutive_selections_per_budget_group=1,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    hot = queue.enqueue(task_id="hello_task", priority=5, budget_group="campaign-a")
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    controller.record_budget_group_selection("campaign-a")

    claimed = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        runtime_controller=controller,
    )

    assert claimed is not None
    assert claimed.job_id == hot.job_id
    assert claimed.state == "completed"
    scheduler = controller.scheduler_state()
    assert scheduler["last_selected_budget_group"] == "campaign-a"
    assert scheduler["consecutive_budget_group_selections"] == 2


def test_task_bank_synthesizes_parallel_workers_without_hand_authored_tasks():
    bank = TaskBank()
    bank._tasks["synthetic_integrator"] = TaskSpec(
        task_id="synthetic_integrator",
        prompt="accept two synthetic worker branches",
        workspace_subdir="synthetic_integrator",
        suggested_commands=["git merge --no-ff worker/a -m 'merge worker/a' && git merge --no-ff worker/b -m 'merge worker/b'"],
        expected_files=["src/a.txt", "docs/b.txt"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src docs && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && printf 'base\\n' > src/a.txt && printf 'base\\n' > docs/b.txt && git add src/a.txt docs/b.txt && git commit -m 'baseline'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/a.txt", "docs/b.txt"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-synth",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/a", "worker/b"],
                "expected_changed_paths": ["src/a.txt", "docs/b.txt"],
            },
            "parallel_workers": [
                {
                    "worker_branch": "worker/a",
                    "prompt": "update src/a.txt",
                    "suggested_commands": ["printf 'A\\n' > src/a.txt && git add src/a.txt && git commit -m 'worker a'"],
                    "expected_changed_paths": ["src/a.txt"],
                    "expected_file_contents": {"src/a.txt": "A\n"},
                },
                {
                    "worker_branch": "worker/b",
                    "prompt": "update docs/b.txt",
                    "suggested_commands": ["printf 'B\\n' > docs/b.txt && git add docs/b.txt && git commit -m 'worker b'"],
                    "expected_changed_paths": ["docs/b.txt"],
                    "expected_file_contents": {"docs/b.txt": "B\n"},
                },
            ],
        },
    )

    workers = bank.parallel_worker_tasks("synthetic_integrator")

    assert [task.task_id for task in workers] == [
        "synthetic_integrator__worker__worker_a",
        "synthetic_integrator__worker__worker_b",
    ]
    assert workers[0].metadata["synthetic_worker"] is True
    assert workers[0].metadata["workflow_guard"]["worker_branch"] == "worker/a"


def test_task_bank_heuristically_synthesizes_parallel_workers_from_integrator_contract():
    bank = TaskBank()
    bank._tasks["heuristic_integrator"] = TaskSpec(
        task_id="heuristic_integrator",
        prompt="accept two worker branches",
        workspace_subdir="heuristic_integrator",
        expected_files=[
            "docs/status.md",
            "src/api_status.txt",
            "reports/merge_report.txt",
            "reports/test_report.txt",
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
                "shared_repo_id": "repo-synth",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                "expected_changed_paths": [
                    "docs/status.md",
                    "reports/merge_report.txt",
                    "reports/test_report.txt",
                    "src/api_status.txt",
                ],
                "preserved_paths": ["tests/test_api.sh", "tests/test_docs.sh"],
                "test_commands": [
                    {"label": "api suite", "argv": ["tests/test_api.sh"]},
                    {"label": "docs suite", "argv": ["tests/test_docs.sh"]},
                ],
                "report_rules": [
                    {"path": "reports/merge_report.txt"},
                    {"path": "reports/test_report.txt"},
                ],
            },
        },
    )

    workers = bank.parallel_worker_tasks("heuristic_integrator")

    assert [task.task_id for task in workers] == [
        "heuristic_integrator__worker__worker_api-status",
        "heuristic_integrator__worker__worker_docs-status",
    ]
    assert workers[0].metadata["workflow_guard"]["claimed_paths"] == [
        "src/api_status.txt",
        "reports/worker_api-status_report.txt",
    ]
    assert workers[1].metadata["workflow_guard"]["claimed_paths"] == [
        "docs/status.md",
        "reports/worker_docs-status_report.txt",
    ]
    assert workers[0].metadata["semantic_verifier"]["test_commands"] == [
        {"label": "api suite", "argv": ["tests/test_api.sh"]}
    ]
    assert workers[1].metadata["semantic_verifier"]["test_commands"] == [
        {"label": "docs suite", "argv": ["tests/test_docs.sh"]}
    ]
    assert workers[0].suggested_commands
    assert "sed -i '1s#pending#ready#'" in workers[0].suggested_commands[0]
    token_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
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
    assert workers[0].metadata["synthetic_edit_candidates"] == [
        {
            "path": "src/api_status.txt",
            "selected_kind": "token_replace",
            "selected_score": token_score,
            "selected": workers[0].metadata["synthetic_edit_plan"][0],
            "candidates": workers[0].metadata["synthetic_edit_candidates"][0]["candidates"],
        }
    ]
    assert [candidate["edit_kind"] for candidate in workers[0].metadata["synthetic_edit_candidates"][0]["candidates"]] == [
        "token_replace",
        "line_replace",
        "rewrite",
    ]
    docs_score = workers[1].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[1].metadata["synthetic_edit_plan"] == [
        {
            "path": "docs/status.md",
            "baseline_content": "status pending documented\n",
            "target_content": "status ready documented\n",
            "edit_kind": "token_replace",
            "intent_source": "assigned_tests",
            "edit_score": docs_score,
            "replacements": [
                {
                    "line_number": 1,
                    "before_fragment": "pending",
                    "after_fragment": "ready",
                    "before_line": "status pending documented",
                    "after_line": "status ready documented",
                }
            ],
        }
    ]
    assert workers[0].metadata["semantic_verifier"]["report_rules"] == [
        {
            "path": "reports/worker_api-status_report.txt",
            "must_mention": ["updated", "worker/api-status", "api", "suite"],
            "covers": ["src/api_status.txt", "tests/test_api.sh"],
        }
    ]
    assert "reports/worker_api-status_report.txt" in workers[0].expected_files
    assert "worker/api-status" in workers[0].suggested_commands[0]


def test_task_bank_falls_back_to_branch_intent_for_worker_edit_plan():
    bank = TaskBank()
    bank._tasks["branch_intent_integrator"] = TaskSpec(
        task_id="branch_intent_integrator",
        prompt="accept one worker branch",
        workspace_subdir="branch_intent_integrator",
        expected_files=["src/service_state.txt"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src && printf 'SERVICE_STATE=broken\\n' > src/service_state.txt && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_state.txt && git commit -m 'baseline branch intent sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_state.txt"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-branch-intent",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_state.txt"],
            },
        },
    )

    workers = bank.parallel_worker_tasks("branch_intent_integrator")

    assert len(workers) == 1
    token_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/service_state.txt",
            "baseline_content": "SERVICE_STATE=broken\n",
            "target_content": "SERVICE_STATE=ready\n",
            "edit_kind": "token_replace",
            "intent_source": "branch_intent",
            "edit_score": token_score,
            "replacements": [
                {
                    "line_number": 1,
                    "before_fragment": "broken",
                    "after_fragment": "ready",
                    "before_line": "SERVICE_STATE=broken",
                    "after_line": "SERVICE_STATE=ready",
                }
            ],
        }
    ]
    assert "sed -i '1s#broken#ready#'" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_line_replace_edit_plan_for_partial_file_update():
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
                "shared_repo_id": "repo-line-edit",
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

    workers = bank.parallel_worker_tasks("line_edit_integrator")

    assert len(workers) == 1
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/service_status.txt",
            "baseline_content": "HEADER=stable\nSERVICE_STATE=broken\nFOOTER=keep\n",
            "target_content": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n",
            "edit_kind": "line_replace",
            "intent_source": "expected_file_contents",
            "edit_score": workers[0].metadata["synthetic_edit_plan"][0]["edit_score"],
            "replacements": [
                {
                    "line_number": 2,
                    "before_line": "SERVICE_STATE=broken",
                    "after_line": "release-ready active",
                }
            ],
        }
    ]
    assert "sed -i '2s#^SERVICE_STATE=broken$#release-ready active#'" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_line_replace_edit_plan_with_duplicate_source_lines():
    bank = TaskBank()
    bank._tasks["line_edit_duplicate_integrator"] = TaskSpec(
        task_id="line_edit_duplicate_integrator",
        prompt="accept one worker branch with duplicate source lines",
        workspace_subdir="line_edit_duplicate_integrator",
        expected_files=["src/release_notes.txt", "tests/test_release.sh"],
        expected_file_contents={
            "src/release_notes.txt": "ITEM=pending\nrelease-ready active\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'ITEM=pending\\nITEM=pending\\nFOOTER=keep\\n' > src/release_notes.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready active$\" src/release_notes.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt tests/test_release.sh && git commit -m 'baseline duplicate line edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt", "tests/test_release.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-line-duplicate",
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

    workers = bank.parallel_worker_tasks("line_edit_duplicate_integrator")

    assert len(workers) == 1
    assert workers[0].metadata["synthetic_edit_plan"][0]["edit_kind"] == "line_replace"
    assert workers[0].metadata["synthetic_edit_plan"][0]["replacements"] == [
        {
            "line_number": 2,
            "before_line": "ITEM=pending",
            "after_line": "release-ready active",
        }
    ]
    assert "sed -i '2s#^ITEM=pending$#release-ready active#'" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_token_replace_edit_plan_for_inline_change():
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
                "shared_repo_id": "repo-token-edit",
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

    workers = bank.parallel_worker_tasks("token_edit_integrator")

    assert len(workers) == 1
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/service_status.js",
            "baseline_content": 'export const serviceStatus = "broken";\nexport const footer = "keep";\n',
            "target_content": 'export const serviceStatus = "ready";\nexport const footer = "keep";\n',
            "edit_kind": "token_replace",
            "intent_source": "assigned_tests",
            "edit_score": workers[0].metadata["synthetic_edit_plan"][0]["edit_score"],
            "replacements": [
                {
                    "line_number": 1,
                    "before_fragment": "broken",
                    "after_fragment": "ready",
                    "before_line": 'export const serviceStatus = "broken";',
                    "after_line": 'export const serviceStatus = "ready";',
                }
            ],
        }
    ]
    assert "sed -i '1s#broken#ready#' src/service_status.js" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_token_replace_edit_plan_with_duplicate_file_tokens():
    bank = TaskBank()
    bank._tasks["token_edit_duplicate_integrator"] = TaskSpec(
        task_id="token_edit_duplicate_integrator",
        prompt="accept one worker branch with duplicate file tokens",
        workspace_subdir="token_edit_duplicate_integrator",
        expected_files=["src/service_status.js", "tests/test_service.sh"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'export const serviceStatus = \"broken\";\\nexport const backupStatus = \"broken\";\\n' > src/service_status.js && printf \"#!/bin/sh\\nset -eu\\ngrep -q 'serviceStatus = \\\"ready\\\"' src/service_status.js\\ngrep -q 'backupStatus = \\\"broken\\\"' src/service_status.js\\n\" > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.js tests/test_service.sh && git commit -m 'baseline duplicate token edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.js", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-token-duplicate",
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

    workers = bank.parallel_worker_tasks("token_edit_duplicate_integrator")

    assert len(workers) == 1
    assert workers[0].metadata["synthetic_edit_plan"][0]["edit_kind"] == "token_replace"
    assert workers[0].metadata["synthetic_edit_plan"][0]["replacements"] == [
        {
            "line_number": 1,
            "before_fragment": "broken",
            "after_fragment": "ready",
            "before_line": 'export const serviceStatus = "broken";',
            "after_line": 'export const serviceStatus = "ready";',
        }
    ]
    assert "sed -i '1s#broken#ready#' src/service_status.js" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_block_replace_edit_plan_for_contiguous_multiline_change():
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
                "shared_repo_id": "repo-block-edit",
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

    workers = bank.parallel_worker_tasks("block_edit_integrator")

    assert len(workers) == 1
    block_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/release_notes.txt",
            "baseline_content": "HEADER=stable\nITEM=pending\nITEM=pending\nFOOTER=keep\n",
            "target_content": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n",
            "edit_kind": "block_replace",
            "intent_source": "expected_file_contents",
            "edit_score": block_score,
            "replacement": {
                "start_line": 2,
                "end_line": 3,
                "before_lines": ["ITEM=pending", "ITEM=pending"],
                "after_lines": ["release-ready line one", "release-ready line two"],
            },
        }
    ]
    assert workers[0].metadata["synthetic_edit_candidates"] == [
        {
            "path": "src/release_notes.txt",
            "selected_kind": "block_replace",
            "selected_score": block_score,
            "selected": workers[0].metadata["synthetic_edit_plan"][0],
            "candidates": workers[0].metadata["synthetic_edit_candidates"][0]["candidates"],
        }
    ]
    assert [candidate["edit_kind"] for candidate in workers[0].metadata["synthetic_edit_candidates"][0]["candidates"]] == [
        "block_replace",
        "rewrite",
    ]
    assert block_score > 0
    assert "sed -i '2,3c\\" in workers[0].suggested_commands[0]
    assert "release-ready line one" in workers[0].suggested_commands[0]
    assert "release-ready line two" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_line_insert_edit_plan_for_contiguous_insert():
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
                "shared_repo_id": "repo-insert-edit",
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

    workers = bank.parallel_worker_tasks("insert_edit_integrator")

    assert len(workers) == 1
    insert_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/release_notes.txt",
            "baseline_content": "HEADER=stable\nFOOTER=keep\n",
            "target_content": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n",
            "edit_kind": "line_insert",
            "intent_source": "expected_file_contents",
            "edit_score": insert_score,
            "insertion": {
                "line_number": 2,
                "mode": "before",
                "after_lines": ["release-ready line one", "release-ready line two"],
            },
        }
    ]
    assert workers[0].metadata["synthetic_edit_candidates"] == [
        {
            "path": "src/release_notes.txt",
            "selected_kind": "line_insert",
            "selected_score": insert_score,
            "selected": workers[0].metadata["synthetic_edit_plan"][0],
            "candidates": workers[0].metadata["synthetic_edit_candidates"][0]["candidates"],
        }
    ]
    assert workers[0].metadata["synthetic_edit_candidates"][0]["candidates"][0]["edit_kind"] == "line_insert"
    assert any(
        candidate["edit_kind"] == "rewrite"
        for candidate in workers[0].metadata["synthetic_edit_candidates"][0]["candidates"]
    )
    assert "sed -i '2i\\" in workers[0].suggested_commands[0]
    assert "release-ready line one" in workers[0].suggested_commands[0]
    assert "release-ready line two" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_line_delete_edit_plan_for_contiguous_delete():
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
                "shared_repo_id": "repo-delete-edit",
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

    workers = bank.parallel_worker_tasks("delete_edit_integrator")

    assert len(workers) == 1
    delete_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/release_notes.txt",
            "baseline_content": "HEADER=stable\nobsolete line\nFOOTER=keep\n",
            "target_content": "HEADER=stable\nFOOTER=keep\n",
            "edit_kind": "line_delete",
            "intent_source": "expected_file_contents",
            "edit_score": delete_score,
            "deletion": {
                "start_line": 2,
                "end_line": 2,
                "before_lines": ["obsolete line"],
            },
        }
    ]
    assert workers[0].metadata["synthetic_edit_candidates"] == [
        {
            "path": "src/release_notes.txt",
            "selected_kind": "line_delete",
            "selected_score": delete_score,
            "selected": workers[0].metadata["synthetic_edit_plan"][0],
            "candidates": workers[0].metadata["synthetic_edit_candidates"][0]["candidates"],
        }
    ]
    assert [candidate["edit_kind"] for candidate in workers[0].metadata["synthetic_edit_candidates"][0]["candidates"]] == [
        "line_delete",
        "rewrite",
    ]
    assert "sed -i '2,2d' src/release_notes.txt" in workers[0].suggested_commands[0]


def test_task_bank_synthesizes_rewrite_when_scoped_edits_are_not_valid():
    bank = TaskBank()
    bank._tasks["rewrite_integrator"] = TaskSpec(
        task_id="rewrite_integrator",
        prompt="accept one worker branch with full rewrite",
        workspace_subdir="rewrite_integrator",
        expected_files=["src/payload.json"],
        expected_file_contents={
            "src/payload.json": '{"release":"ready","notes":["one","two"]}\n',
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git commit --allow-empty -m 'baseline rewrite sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": [],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-rewrite",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/rewrite-ready"],
                "expected_changed_paths": ["src/payload.json"],
            },
        },
    )

    workers = bank.parallel_worker_tasks("rewrite_integrator")

    assert len(workers) == 1
    rewrite_score = workers[0].metadata["synthetic_edit_plan"][0]["edit_score"]
    assert workers[0].metadata["synthetic_edit_plan"] == [
        {
            "path": "src/payload.json",
            "baseline_content": "",
            "target_content": '{"release":"ready","notes":["one","two"]}\n',
            "edit_kind": "rewrite",
            "intent_source": "expected_file_contents",
            "edit_score": rewrite_score,
        }
    ]
    assert workers[0].metadata["synthetic_edit_plan"][0]["edit_score"] > 100
    assert "printf %s '{\"release\":\"ready\",\"notes\":[\"one\",\"two\"]}" in workers[0].suggested_commands[0]


def test_task_bank_edit_scores_are_recorded_for_selected_candidates():
    bank = TaskBank()
    token_worker = bank.parallel_worker_tasks("token_edit_integrator")[0] if "token_edit_integrator" in bank._tasks else None
    if token_worker is None:
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
                "workflow_guard": {"requires_git": True, "shared_repo_id": "repo-token-edit", "target_branch": "main"},
                "semantic_verifier": {
                    "kind": "git_repo_review",
                    "expected_branch": "main",
                    "required_merged_branches": ["worker/service-ready"],
                    "expected_changed_paths": ["src/service_status.js"],
                    "preserved_paths": ["tests/test_service.sh"],
                    "test_commands": [{"label": "service suite", "argv": ["tests/test_service.sh"]}],
                },
            },
        )
        token_worker = bank.parallel_worker_tasks("token_edit_integrator")[0]
    token_score = token_worker.metadata["synthetic_edit_plan"][0]["edit_score"]

    bank._tasks["line_edit_integrator_score"] = TaskSpec(
        task_id="line_edit_integrator_score",
        prompt="accept one worker branch with line update",
        workspace_subdir="line_edit_integrator_score",
        expected_files=["src/service_status.txt"],
        expected_file_contents={"src/service_status.txt": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n"},
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src && printf 'HEADER=stable\\nSERVICE_STATE=broken\\nFOOTER=keep\\n' > src/service_status.txt && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.txt && git commit -m 'baseline line score sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.txt"],
            "workflow_guard": {"requires_git": True, "shared_repo_id": "repo-line-edit-score", "target_branch": "main"},
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.txt"],
            },
        },
    )
    line_score = bank.parallel_worker_tasks("line_edit_integrator_score")[0].metadata["synthetic_edit_plan"][0]["edit_score"]

    bank._tasks["block_edit_integrator_score"] = TaskSpec(
        task_id="block_edit_integrator_score",
        prompt="accept one worker branch with block update",
        workspace_subdir="block_edit_integrator_score",
        expected_files=["src/release_notes.txt"],
        expected_file_contents={"src/release_notes.txt": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n"},
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src && printf 'HEADER=stable\\nITEM=pending\\nITEM=pending\\nFOOTER=keep\\n' > src/release_notes.txt && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt && git commit -m 'baseline block score sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt"],
            "workflow_guard": {"requires_git": True, "shared_repo_id": "repo-block-edit-score", "target_branch": "main"},
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/release-ready"],
                "expected_changed_paths": ["src/release_notes.txt"],
            },
        },
    )
    block_score = bank.parallel_worker_tasks("block_edit_integrator_score")[0].metadata["synthetic_edit_plan"][0]["edit_score"]

    bank._tasks["rewrite_integrator_score"] = TaskSpec(
        task_id="rewrite_integrator_score",
        prompt="accept one worker branch with rewrite",
        workspace_subdir="rewrite_integrator_score",
        expected_files=["src/payload.json"],
        expected_file_contents={"src/payload.json": '{"release":"ready","notes":["one","two"]}\n'},
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git commit --allow-empty -m 'baseline rewrite score sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": [],
            "workflow_guard": {"requires_git": True, "shared_repo_id": "repo-rewrite-score", "target_branch": "main"},
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/rewrite-ready"],
                "expected_changed_paths": ["src/payload.json"],
            },
        },
    )
    rewrite_score = bank.parallel_worker_tasks("rewrite_integrator_score")[0].metadata["synthetic_edit_plan"][0]["edit_score"]

    assert token_score > 0
    assert line_score > 0
    assert block_score > 0
    assert rewrite_score > 0
    assert rewrite_score > token_score
    assert rewrite_score > line_score


def test_enqueue_with_parallel_worker_decomposition_embeds_synthetic_task_payload(tmp_path):
    bank = TaskBank()
    bank._tasks["synthetic_integrator"] = TaskSpec(
        task_id="synthetic_integrator",
        prompt="accept a synthetic worker branch",
        workspace_subdir="synthetic_integrator",
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo-synth",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/a"],
                "expected_changed_paths": ["src/a.txt"],
            },
            "parallel_workers": [
                {
                    "worker_branch": "worker/a",
                    "prompt": "update src/a.txt",
                    "suggested_commands": ["printf 'A\\n' > src/a.txt && git add src/a.txt && git commit -m 'worker a'"],
                    "expected_changed_paths": ["src/a.txt"],
                }
            ],
        },
    )
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")

    jobs = enqueue_with_parallel_worker_decomposition(
        queue,
        bank=bank,
        task_id="synthetic_integrator",
        priority=2,
        budget_group="campaign-a",
    )

    assert jobs[0].runtime_overrides["task_payload"]["task_id"] == "synthetic_integrator__worker__worker_a"
    assert jobs[1].runtime_overrides["dependency_job_ids"] == [jobs[0].job_id]


def test_runtime_controller_blocks_colliding_workspace_and_reports_policy(tmp_path):
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
        delegated_job_max_artifact_bytes=1024,
    )
    config.ensure_directories()
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    first = queue.enqueue(task_id="hello_task")
    second = queue.enqueue(task_id="hello_task")
    task = type("Task", (), {"workspace_subdir": "hello_task"})()

    lease, reason = controller.acquire(
        job=first,
        task=task,
        config=config,
        checkpoint_path=tmp_path / "checkpoints" / "hello.json",
        report_path=tmp_path / "reports" / "hello.json",
    )
    blocked_lease, blocked_reason = controller.acquire(
        job=second,
        task=task,
        config=config,
        checkpoint_path=tmp_path / "checkpoints" / "hello-2.json",
        report_path=tmp_path / "reports" / "hello-2.json",
    )
    snapshot = controller.snapshot(config=config)

    assert lease is not None
    assert reason is None
    assert blocked_lease is None
    assert blocked_reason is not None
    assert blocked_reason.startswith("lease_collision:workspace:")
    assert snapshot["policy"]["max_concurrent_jobs"] == 2
    assert snapshot["policy"]["max_artifact_bytes_per_job"] == 1024
    assert len(snapshot["active_leases"]) == 1


def test_runtime_controller_snapshot_applies_retained_delegation_policy(tmp_path):
    proposal_path = tmp_path / "delegation" / "delegation_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "artifact_kind": "delegated_runtime_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "delegated_job_max_concurrency": 3,
                    "delegated_job_max_active_per_budget_group": 2,
                    "delegated_job_max_queued_per_budget_group": 4,
                    "delegated_job_max_artifact_bytes": 2048,
                    "delegated_job_max_subprocesses_per_job": 2,
                    "command_timeout_seconds": 40,
                    "llm_timeout_seconds": 40,
                    "max_steps": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=1,
        delegated_job_max_artifact_bytes=1024,
        delegation_proposals_path=proposal_path,
    )
    config.ensure_directories()
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")

    snapshot = controller.snapshot(config=config)

    assert snapshot["policy"]["max_concurrent_jobs"] == 3
    assert snapshot["policy"]["max_artifact_bytes_per_job"] == 2048
    assert snapshot["policy"]["max_subprocesses_per_job"] == 2
    assert snapshot["policy"]["max_steps"] == 7
    assert snapshot["policy"]["frontier_task_step_floor"] == config.frontier_task_step_floor


def test_runtime_controller_snapshot_tolerates_invalid_json_state(tmp_path):
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_state_path.write_text("{", encoding="utf-8")
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        delegated_job_runtime_state_path=runtime_state_path,
    )
    controller = DelegatedRuntimeController(runtime_state_path, runner_id="runner-a")

    snapshot = controller.snapshot(config=config)

    assert snapshot["active_leases"] == []
    assert snapshot["history"] == []


def test_runtime_controller_blocks_colliding_shared_repo_claims_and_allows_disjoint_paths(tmp_path):
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=3,
    )
    config.ensure_directories()
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    first = queue.enqueue(
        task_id="git_parallel_worker_api_task",
        runtime_overrides={
            "shared_repo_id": "repo-a",
            "target_branch": "main",
            "worker_branch": "worker/api-status",
            "claimed_paths": ["src/api_status.txt"],
        },
    )
    second = queue.enqueue(
        task_id="git_parallel_worker_docs_task",
        runtime_overrides={
            "shared_repo_id": "repo-a",
            "target_branch": "main",
            "worker_branch": "worker/docs-status",
            "claimed_paths": ["docs/status.md"],
        },
    )
    third = queue.enqueue(
        task_id="git_parallel_worker_api_task",
        runtime_overrides={
            "shared_repo_id": "repo-a",
            "target_branch": "main",
            "worker_branch": "worker/api-followup",
            "claimed_paths": ["src"],
        },
    )
    from agent_kernel.tasking.task_bank import TaskBank

    bank = TaskBank()
    first_task = bank.get("git_parallel_worker_api_task")
    second_task = bank.get("git_parallel_worker_docs_task")
    third_task = bank.get("git_parallel_worker_api_task")

    first_lease, first_reason = controller.acquire(
        job=first,
        task=first_task,
        config=config,
        checkpoint_path=tmp_path / "checkpoints" / "first.json",
        report_path=tmp_path / "reports" / "first.json",
    )
    second_lease, second_reason = controller.acquire(
        job=second,
        task=second_task,
        config=config,
        checkpoint_path=tmp_path / "checkpoints" / "second.json",
        report_path=tmp_path / "reports" / "second.json",
    )
    blocked_lease, blocked_reason = controller.acquire(
        job=third,
        task=third_task,
        config=config,
        checkpoint_path=tmp_path / "checkpoints" / "third.json",
        report_path=tmp_path / "reports" / "third.json",
    )
    snapshot = controller.snapshot(config=config)

    assert first_lease is not None
    assert first_reason is None
    assert second_lease is not None
    assert second_reason is None
    assert blocked_lease is None
    assert blocked_reason == "lease_collision:claimed_path:repo-a:src"
    assert len(snapshot["active_leases"]) == 2
    assert snapshot["active_leases"][0]["shared_repo_id"] == "repo-a"


def test_run_next_delegated_jobs_complete_shared_repo_worker_then_integrator_flow(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
        unattended_allow_git_commands=True,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    worker_api = queue.enqueue(task_id="git_parallel_worker_api_task", priority=3)
    worker_docs = queue.enqueue(task_id="git_parallel_worker_docs_task", priority=3)
    integrator = queue.enqueue(task_id="git_parallel_merge_acceptance_task", priority=1)

    first = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])
    second = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])
    third = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert first is not None and first.job_id == worker_api.job_id and first.state == "completed"
    assert second is not None and second.job_id == worker_docs.job_id and second.state == "completed"
    assert third is not None and third.job_id == integrator.job_id and third.state == "completed"
    main_workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_parallel_merge"
        / "clones"
        / "main"
    )
    assert (main_workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed\n"
    )


def test_materialize_shared_repo_workspace_uses_absolute_git_paths(tmp_path):
    from agent_kernel.ops.shared_repo import materialize_shared_repo_workspace

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        unattended_allow_git_commands=True,
    )
    config.ensure_directories()
    task = TaskBank().parallel_worker_tasks("git_generated_conflict_resolution_task")[0]

    workspace = materialize_shared_repo_workspace(task, config=config)

    assert workspace.exists()
    assert (workspace / ".git").exists()
    assert (workspace / "src" / "shared_status.txt").exists()


def test_runtime_controller_blocks_budget_group_overcommit_and_reports_counts(tmp_path):
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
        delegated_job_max_active_per_budget_group=1,
    )
    config.ensure_directories()
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    first = queue.enqueue(task_id="hello_task", budget_group="campaign-a")
    second = queue.enqueue(task_id="math_task", budget_group="campaign-a")

    first_lease, first_reason = controller.acquire(
        job=first,
        task=type("Task", (), {"workspace_subdir": "hello_task"})(),
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "hello.json",
        report_path=tmp_path / "trajectories" / "reports" / "hello.json",
    )
    blocked_lease, blocked_reason = controller.acquire(
        job=second,
        task=type("Task", (), {"workspace_subdir": "math_task"})(),
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "math.json",
        report_path=tmp_path / "trajectories" / "reports" / "math.json",
    )
    snapshot = controller.snapshot(config=config)

    assert first_lease is not None
    assert first_reason is None
    assert blocked_lease is None
    assert blocked_reason == "resource_limit:budget_group:campaign-a"
    assert snapshot["policy"]["max_active_jobs_per_budget_group"] == 1
    assert snapshot["budget_groups"]["campaign-a"] == 1


def test_run_next_delegated_job_defers_when_runtime_slot_unavailable(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=1,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    job = queue.enqueue(task_id="hello_task")
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    task = type("Task", (), {"workspace_subdir": "other_task"})()
    acquired, denied = controller.acquire(
        job=job,
        task=task,
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "occupied.json",
        report_path=tmp_path / "trajectories" / "reports" / "occupied.json",
    )

    deferred = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        runtime_controller=DelegatedRuntimeController(
            config.delegated_job_runtime_state_path,
            runner_id="runner-b",
        ),
    )

    assert acquired is not None
    assert denied is None
    assert deferred is not None
    assert deferred.state == "queued"
    assert deferred.last_error == "resource_limit:max_concurrent_jobs"


def test_run_next_delegated_job_skips_budget_blocked_group_and_runs_next_available(monkeypatch, tmp_path):
    _install_successful_kernel(monkeypatch)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
        delegated_job_max_active_per_budget_group=1,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    blocked = queue.enqueue(task_id="hello_task", priority=5, budget_group="campaign-a")
    runnable = queue.enqueue(task_id="math_task", priority=1, budget_group="campaign-b")
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    acquired, denied = controller.acquire(
        job=blocked,
        task=type("Task", (), {"workspace_subdir": "occupied_task"})(),
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "occupied.json",
        report_path=tmp_path / "trajectories" / "reports" / "occupied.json",
    )

    finished = run_next_delegated_job(
        queue,
        base_config=config,
        repo_root=Path(__file__).resolve().parents[1],
        runtime_controller=DelegatedRuntimeController(
            config.delegated_job_runtime_state_path,
            runner_id="runner-b",
        ),
    )

    assert acquired is not None
    assert denied is None
    assert finished is not None
    assert finished.job_id == runnable.job_id
    assert finished.state == "completed"
    blocked_state = queue.get(blocked.job_id)
    assert blocked_state is not None
    assert blocked_state.state == "queued"
    assert blocked_state.last_error == "resource_limit:budget_group:campaign-a"


def test_run_next_delegated_job_marks_safe_stop_when_artifact_budget_exceeded(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_artifact_bytes=1,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(task_id="hello_task")

    job = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert job is not None
    assert job.state == "safe_stop"
    assert job.outcome == "safe_stop"
    assert "artifact_budget_exceeded" in job.outcome_reasons
    assert "exceeded limit" in job.last_error


def test_run_next_delegated_job_restores_workspace_after_non_success(monkeypatch, tmp_path):
    from agent_kernel.ops import job_queue as job_queue_module
    from agent_kernel.schemas import EpisodeRecord, StepRecord

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories" / "episodes",
        run_reports_dir=tmp_path / "trajectories" / "reports",
        run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
        delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "trajectories" / "recovery" / "workspaces",
        unattended_rollback_on_failure=True,
    )
    config.ensure_directories()
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    queue.enqueue(task_id="hello_task")

    class FailingKernel:
        def __init__(self, config):
            self.config = config

        def run_task(self, task, checkpoint_path=None, resume=False, **kwargs):
            del checkpoint_path
            del resume
            del kwargs
            workspace = self.config.workspace_root / task.workspace_subdir
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "draft.txt").write_text("partial\n", encoding="utf-8")
            return EpisodeRecord(
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=str(workspace),
                success=False,
                steps=[
                    StepRecord(
                        index=1,
                        thought="write an unmanaged file",
                        action="code_execute",
                        content="printf 'partial\\n' > draft.txt",
                        selected_skill_id=None,
                        command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                        verification={"passed": False, "reasons": ["missing expected file: hello.txt"]},
                    )
                ],
                termination_reason="max_steps_reached",
            )

        def close(self):
            return None

    monkeypatch.setattr(job_queue_module, "AgentKernel", FailingKernel)

    job = run_next_delegated_job(queue, base_config=config, repo_root=Path(__file__).resolve().parents[1])

    assert job is not None
    assert job.state == "safe_stop"
    workspace = config.workspace_root / "hello_task"
    assert list(workspace.rglob("*")) == []
    payload = json.loads(Path(job.report_path).read_text(encoding="utf-8"))
    assert payload["side_effects"]["hidden_side_effect_risk"] is True
    assert payload["recovery"]["rollback_performed"] is True
    assert payload["recovery"]["post_rollback_file_count"] == 0


def test_run_job_queue_enqueue_cli_applies_budget_group_limit(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            delegated_job_max_queued_per_budget_group=1,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_job_queue.py", "enqueue", "--task-id", "hello_task", "--budget-group", "campaign-a"],
    )
    module.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_job_queue.py", "enqueue", "--task-id", "math_task", "--budget-group", "campaign-a"],
    )

    try:
        module.main()
        raised = False
    except ValueError as exc:
        raised = True
        assert str(exc) == "queue_budget_exceeded:campaign-a:1"

    assert raised is True


def test_run_job_queue_status_cli(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            delegated_job_max_concurrency=2,
            delegated_job_max_active_per_budget_group=1,
            delegated_job_max_queued_per_budget_group=3,
            delegated_job_max_artifact_bytes=2048,
            delegated_job_max_consecutive_selections_per_budget_group=2,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="hello_task", budget_group="campaign-a")
    queue.record_scheduler_decision(job.job_id, decision="selected:runnable_job", detail="priority=0")
    controller = DelegatedRuntimeController(runtime_state_path, runner_id="runner-a")
    controller.acquire(
        job=queue.list_jobs()[0],
        task=type("Task", (), {"workspace_subdir": "hello_task"})(),
        config=KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            delegated_job_runtime_state_path=runtime_state_path,
            delegated_job_max_concurrency=2,
            delegated_job_max_active_per_budget_group=1,
            delegated_job_max_queued_per_budget_group=3,
            delegated_job_max_artifact_bytes=2048,
            delegated_job_max_consecutive_selections_per_budget_group=2,
        ),
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "hello.json",
        report_path=tmp_path / "trajectories" / "reports" / "hello.json",
    )
    controller.record_budget_group_selection("campaign-a")
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "policy max_concurrency=2" in output
    assert "max_active_per_budget_group=1" in output
    assert "max_queued_per_budget_group=3" in output
    assert "max_consecutive_selections_per_budget_group=2" in output
    assert "operator_policy allow_git_commands=0" in output
    assert "allow_http_requests=0" in output
    assert "http_allowed_hosts=" in output
    assert "capability_policy enabled_builtin=workspace_exec,workspace_fs" in output
    assert "rollback_on_failure=1" in output
    assert "trust_policy enforce=1" in output
    assert "trust_status status=bootstrap" in output
    assert "trust_family family=repo_chore required=1 status=absent reports=0" in output
    assert "queue_status total_jobs=1 runnable_jobs=1 runnable_workers=0 blocked_jobs=0 promotable_jobs=0" in output
    assert "queue_decisions selected:runnable_job=1" in output
    assert "queue_family family=bounded total_jobs=1 runnable_jobs=1 blocked_jobs=0 promotable_jobs=0 worker_jobs=0 integrator_jobs=0 budget_groups=campaign-a" in output
    assert "queue_fairness blocked_open_jobs=0 scheduler_selected_total=1 scheduler_blocked_total=0 scheduler_unblock_total=0 oldest_blocked_at=-" in output
    assert "scheduler_streak budget_group=campaign-a consecutive=1" in output
    assert "role_closeout ready=0 mode=active_leases total_jobs=1 completed_success_jobs=0 unfinished_jobs=1 terminal_non_success_jobs=0 blocked_open_jobs=0 active_leases=1 trust_status=bootstrap operator_steering_required=1" in output
    assert "active_roles total=1 worker_leases=0 integrator_leases=0 other_leases=1 families=bounded:1 budget_groups=campaign-a:1 shared_repos=-" in output
    assert "next_runnable " in output
    assert "task_id=hello_task" in output
    assert "active_leases=1" in output
    assert "budget_groups campaign-a=1" in output
    assert "lease job_id=" in output
    assert "budget_group=campaign-a" in output


def test_run_job_queue_status_json_includes_queue_trust_and_capability_details(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "adapter_kind": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            capability_modules_path=modules_path,
            delegated_job_max_concurrency=2,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="hello_task", budget_group="campaign-a")
    queue.record_scheduler_decision(job.job_id, decision="selected:runnable_job", detail="priority=0")
    controller = DelegatedRuntimeController(runtime_state_path, runner_id="runner-a")
    controller.acquire(
        job=queue.list_jobs()[0],
        task=type("Task", (), {"workspace_subdir": "hello_task"})(),
        config=KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            delegated_job_runtime_state_path=runtime_state_path,
            delegated_job_max_concurrency=2,
        ),
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "hello.json",
        report_path=tmp_path / "trajectories" / "reports" / "hello.json",
    )
    controller.record_budget_group_selection("campaign-a")
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status", "--json"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["policy"]["max_concurrent_jobs"] == 2
    assert payload["capability_policy"]["modules"][0]["module_id"] == "github"
    assert payload["trust"]["overall_assessment"]["status"] == "bootstrap"
    assert payload["queue"]["totals"]["runnable_jobs"] == 1
    assert payload["queue"]["state_counts"]["queued"] == 1
    assert payload["queue"]["benchmark_families"]["bounded"]["total_jobs"] == 1
    assert payload["queue"]["benchmark_families"]["bounded"]["budget_groups"] == ["campaign-a"]
    assert payload["queue"]["decision_counts"]["selected:runnable_job"] == 1
    assert payload["queue"]["scheduler_streak"]["budget_group"] == "campaign-a"
    assert payload["queue"]["next_runnable"]["job"]["task_id"] == "hello_task"
    assert payload["queue"]["active_lease_roles"]["total"] == 1
    assert payload["queue"]["active_lease_roles"]["other_jobs"] == 1
    assert payload["queue"]["active_lease_roles"]["benchmark_families"] == {"bounded": 1}
    assert payload["queue"]["active_lease_roles"]["budget_groups"] == {"campaign-a": 1}
    assert payload["queue"]["role_closeout"]["closeout_ready"] is False
    assert payload["queue"]["role_closeout"]["closeout_mode"] == "active_leases"
    assert payload["queue"]["role_closeout"]["operator_steering_required"] is True
    assert payload["queue"]["role_closeout"]["active_leases"] == 1
    assert payload["queue"]["role_closeout"]["trust_status"] == "bootstrap"
    assert payload["queue"]["active_leases"][0]["budget_group"] == "campaign-a"


def test_run_job_queue_status_json_reports_trusted_role_closeout(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            delegated_job_max_concurrency=2,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_unattended_trust_ledger",
        lambda config: {
            "gated_summary": {
                "total": 5,
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
            },
            "overall_assessment": {"status": "trusted", "passed": True},
            "policy": {"required_benchmark_families": []},
            "family_summaries": {},
            "family_assessments": {},
        },
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="hello_task", budget_group="campaign-a")
    queue.finalize(
        job.job_id,
        state="completed",
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "hello.json",
        report_path=tmp_path / "trajectories" / "reports" / "hello.json",
        outcome="success",
        outcome_reasons=[],
    )
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status", "--json"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    role_closeout = payload["queue"]["role_closeout"]
    assert role_closeout["closeout_ready"] is True
    assert role_closeout["closeout_mode"] == "queue_empty_trusted"
    assert role_closeout["operator_steering_required"] is False
    assert role_closeout["total_jobs"] == 1
    assert role_closeout["completed_success_jobs"] == 1
    assert role_closeout["unfinished_jobs"] == 0
    assert role_closeout["terminal_non_success_jobs"] == 0
    assert role_closeout["blocked_open_jobs"] == 0
    assert role_closeout["active_leases"] == 0
    assert role_closeout["trust_status"] == "trusted"


def test_run_job_queue_status_json_reports_active_integrator_and_worker_leases(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            delegated_job_max_concurrency=2,
            delegated_job_max_active_per_budget_group=2,
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    worker = queue.enqueue(task_id="git_parallel_worker_api_task", budget_group="family_repo_sandbox")
    integrator = queue.enqueue(task_id="git_release_train_acceptance_task", budget_group="family_repo_sandbox")
    controller = DelegatedRuntimeController(runtime_state_path, runner_id="runner-a")
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        sandbox_command_containment_mode="disabled",
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=runtime_state_path,
        delegated_job_max_concurrency=2,
        delegated_job_max_active_per_budget_group=2,
        unattended_allow_git_commands=True,
    )
    bank = TaskBank()
    controller.acquire(
        job=queue.get(worker.job_id),
        task=bank.get("git_parallel_worker_api_task"),
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "worker.json",
        report_path=tmp_path / "trajectories" / "reports" / "worker.json",
    )
    controller.acquire(
        job=queue.get(integrator.job_id),
        task=bank.get("git_release_train_acceptance_task"),
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "integrator.json",
        report_path=tmp_path / "trajectories" / "reports" / "integrator.json",
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status", "--json"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["queue"]["active_lease_roles"]["total"] == 2
    assert payload["queue"]["active_lease_roles"]["worker_jobs"] == 1
    assert payload["queue"]["active_lease_roles"]["integrator_jobs"] == 1
    assert payload["queue"]["active_lease_roles"]["other_jobs"] == 0
    assert payload["queue"]["active_lease_roles"]["benchmark_families"] == {"repo_sandbox": 2}
    assert payload["queue"]["active_lease_roles"]["budget_groups"] == {"family_repo_sandbox": 2}
    assert payload["queue"]["active_lease_roles"]["shared_repo_ids"] == {
        "repo_sandbox_parallel_merge": 1,
        "repo_sandbox_release_train": 1,
    }


def test_run_job_queue_status_uses_retained_operator_policy(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    runtime_state_path = tmp_path / "trajectories" / "jobs" / "runtime_state.json"
    runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_state_path.write_text(json.dumps({"active": []}), encoding="utf-8")
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "unattended_allowed_benchmark_families": ["micro", "repo_chore"],
                    "unattended_allow_git_commands": True,
                    "unattended_allow_http_requests": True,
                    "unattended_http_allowed_hosts": ["example.com"],
                    "unattended_http_timeout_seconds": 15,
                    "unattended_http_max_body_bytes": 131072,
                    "unattended_allow_generated_path_mutations": True,
                    "unattended_generated_path_prefixes": ["build"],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            delegated_job_runtime_state_path=runtime_state_path,
            operator_policy_proposals_path=operator_policy_path,
            unattended_allow_git_commands=False,
            unattended_allow_http_requests=False,
            unattended_allow_generated_path_mutations=False,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "operator_policy allow_git_commands=1" in output
    assert "allow_http_requests=1" in output
    assert "allow_generated_path_mutations=1" in output
    assert "allowed_benchmark_families=micro,repo_chore" in output


def test_run_job_queue_status_reports_capability_module_validity(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "adapter_kind": "github",
                        "enabled": True,
                        "settings": {
                            "repo_scopes": ["openai/agentkernel"],
                        },
                    },
                    {
                        "module_id": "twitter",
                        "adapter_kind": "twitter",
                        "enabled": True,
                        "capabilities": ["twitter_post"],
                        "settings": {
                            "read_only": True,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=tmp_path / "trajectories" / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            capability_modules_path=modules_path,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "status"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "capability_module module_id=github adapter_kind=github enabled=1 valid=1 capabilities=github_read" in output
    assert "capability_module module_id=twitter adapter_kind=twitter enabled=1 valid=0 capabilities=twitter_post" in output
    assert "issues=twitter_post declares read_only with access_tier=write;twitter_post declares read_only with write_tier=content_post;twitter_post requires repo_scopes or account_scopes" in output


def test_run_job_queue_list_cli_surfaces_acceptance_summary(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    report_path = reports_dir / "hello.json"
    report_path.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "synthetic_worker": True,
                    "target_branch": "main",
                    "expected_branch": "worker/api-status",
                    "required_merged_branches": ["worker/api-status"],
                    "tests": [{"command": "tests/test_api.sh"}],
                    "selected_edits": [{"path": "src/api_status.txt"}],
                    "candidate_edit_sets": [{"path": "src/api_status.txt"}],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="hello_task")
    queue.finalize(
        job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "hello.json",
        report_path=report_path,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "list"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "acceptance_verifier_passed=1" in output
    assert "synthetic_worker=1" in output
    assert "target_branch=main" in output
    assert "expected_branch=worker/api-status" in output
    assert "merged_branches=worker/api-status" in output
    assert "tests=1" in output
    assert "selected_edits=1" in output
    assert "candidate_sets=1" in output


def test_run_job_queue_list_cli_filters_ready_to_accept(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    accepted_report = reports_dir / "accepted.json"
    accepted_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    accepted = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    pending = queue.enqueue(task_id="hello_task")
    queue.finalize(
        accepted.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "accepted.json",
        report_path=accepted_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "list", "--ready-to-accept-only", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "task_id=git_parallel_merge_acceptance_task" in output
    assert "task_id=hello_task" not in output
    del pending


def test_run_job_queue_promotable_cli_excludes_synthetic_workers(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    integrator_report = reports_dir / "integrator.json"
    integrator_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    worker_report = reports_dir / "worker.json"
    worker_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "synthetic_worker": True,
                    "expected_branch": "worker/api-status",
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    integrator = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    worker = queue.enqueue(task_id="git_parallel_worker_api_task")
    queue.finalize(
        integrator.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "integrator.json",
        report_path=integrator_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.finalize(
        worker.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "worker.json",
        report_path=worker_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "promotable"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "task_id=git_parallel_merge_acceptance_task" in output
    assert "task_id=git_parallel_worker_api_task" not in output


def test_run_job_queue_promotable_cli_excludes_worker_jobs_without_synthetic_flag(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    integrator_report = reports_dir / "integrator.json"
    integrator_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    worker_report = reports_dir / "worker.json"
    worker_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "expected_branch": "worker/api-status",
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    integrator = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    worker = queue.enqueue(task_id="git_parallel_worker_api_task")
    queue.finalize(
        integrator.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "integrator.json",
        report_path=integrator_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.finalize(
        worker.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "worker.json",
        report_path=worker_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "promotable"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "task_id=git_parallel_merge_acceptance_task" in output
    assert "task_id=git_parallel_worker_api_task" not in output


def test_run_job_queue_acceptance_review_cli_isolates_unpromoted_integrators(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    merge_report = reports_dir / "merge_integrator.json"
    merge_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    release_report = reports_dir / "release_integrator.json"
    release_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": [
                        "worker/api-cutover",
                        "worker/docs-cutover",
                        "worker/ops-cutover",
                    ],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    worker_report = reports_dir / "worker.json"
    worker_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "synthetic_worker": True,
                    "expected_branch": "worker/api-status",
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    unpromoted = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    promoted = queue.enqueue(task_id="git_release_train_acceptance_task")
    worker = queue.enqueue(task_id="git_parallel_worker_api_task")
    queue.finalize(
        unpromoted.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "merge_integrator.json",
        report_path=merge_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.finalize(
        promoted.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "release_integrator.json",
        report_path=release_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.finalize(
        worker.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "worker.json",
        report_path=worker_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.promote(promoted.job_id, detail="accepted by operator")

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "acceptance-review"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "acceptance_review total_jobs=1 promotable_jobs=1 promoted_jobs=0" in output
    assert "task_id=git_parallel_merge_acceptance_task" in output
    assert "shared_repo_id=repo_sandbox_parallel_merge" in output
    assert "task_id=git_release_train_acceptance_task" not in output
    assert "task_id=git_parallel_worker_api_task" not in output


def test_run_job_queue_acceptance_review_json_includes_rollup_and_promoted_jobs(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    merge_report = reports_dir / "merge_integrator.json"
    merge_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    release_report = reports_dir / "release_integrator.json"
    release_report.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": [
                        "worker/api-cutover",
                        "worker/docs-cutover",
                        "worker/ops-cutover",
                    ],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    merge_job = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    release_job = queue.enqueue(task_id="git_release_train_acceptance_task")
    queue.finalize(
        merge_job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "merge_integrator.json",
        report_path=merge_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.finalize(
        release_job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "release_integrator.json",
        report_path=release_report,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )
    queue.promote(release_job.job_id, detail="accepted by operator")

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_job_queue.py", "acceptance-review", "--include-promoted", "1", "--json"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["review"]["total_jobs"] == 2
    assert payload["review"]["promotable_jobs"] == 1
    assert payload["review"]["promoted_jobs"] == 1
    assert payload["review"]["benchmark_families"] == {"repo_sandbox": 2}
    assert payload["review"]["shared_repo_ids"] == {
        "repo_sandbox_parallel_merge": 1,
        "repo_sandbox_release_train": 1,
    }
    assert payload["review"]["target_branches"] == {"main": 2}
    assert [job["job"]["job"]["task_id"] for job in payload["jobs"]] == [
        "git_parallel_merge_acceptance_task",
        "git_release_train_acceptance_task",
    ]
    assert payload["jobs"][0]["shared_repo_id"] == "repo_sandbox_parallel_merge"
    assert payload["jobs"][1]["job"]["readiness"]["promoted"] is True


def test_run_job_queue_inspect_cli_surfaces_acceptance_trust_and_capabilities(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    report_path = reports_dir / "integrator.json"
    report_path.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "benchmark_family": "repo_sandbox",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "tests": [{"label": "api", "command": "tests/test_api.sh"}],
                    "selected_edits": [{"path": "src/api_status.txt", "kind": "line_replace", "edit_score": 1}],
                    "candidate_edit_sets": [{"path": "src/api_status.txt"}],
                    "verifier_result": {"passed": True},
                },
                "capability_usage": {
                    "required_capabilities": ["git_commands"],
                    "used_capabilities": ["git_commands", "workspace_exec", "workspace_fs"],
                    "external_capabilities_used": [],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
            unattended_trust_bootstrap_min_reports=10,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    queue.finalize(
        job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "integrator.json",
        report_path=report_path,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "inspect", "--job-id", job.job_id])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert f"job job_id={job.job_id} state=completed task_id=git_parallel_merge_acceptance_task outcome=success" in output
    assert "acceptance verifier_passed=1 synthetic_worker=0 target_branch=main" in output
    assert "trust_family family=repo_sandbox status=bootstrap reports=1" in output
    assert "capability_usage required=git_commands used=git_commands,workspace_exec,workspace_fs external=-" in output
    assert "test label=api command=tests/test_api.sh" in output
    assert "selected_edit path=src/api_status.txt kind=line_replace score=1" in output


def test_run_job_queue_inspect_json_surfaces_acceptance_trust_and_capabilities(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    report_path = reports_dir / "integrator.json"
    report_path.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "benchmark_family": "repo_sandbox",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "tests": [{"label": "api", "command": "tests/test_api.sh"}],
                    "selected_edits": [{"path": "src/api_status.txt", "kind": "line_replace", "edit_score": 1}],
                    "candidate_edit_sets": [{"path": "src/api_status.txt"}],
                    "verifier_result": {"passed": True},
                },
                "capability_usage": {
                    "required_capabilities": ["git_commands"],
                    "used_capabilities": ["git_commands", "workspace_exec", "workspace_fs"],
                    "external_capabilities_used": [],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
            unattended_trust_bootstrap_min_reports=10,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    queue.finalize(
        job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "integrator.json",
        report_path=report_path,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "inspect", "--job-id", job.job_id, "--json"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["job"]["job"]["job_id"] == job.job_id
    assert payload["job"]["job"]["task_id"] == "git_parallel_merge_acceptance_task"
    assert payload["job"]["acceptance"]["verifier_passed"] == 1
    assert payload["trust_family"]["family"] == "repo_sandbox"
    assert payload["capability_usage"]["required_capabilities"] == ["git_commands"]
    assert payload["tests"][0]["label"] == "api"
    assert payload["selected_edits"][0]["path"] == "src/api_status.txt"


def test_run_job_queue_promote_cli_marks_job_promoted_and_hides_from_promotable(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"
    reports_dir = tmp_path / "trajectories" / "reports"
    checkpoints_dir = tmp_path / "trajectories" / "checkpoints"
    reports_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    report_path = reports_dir / "accepted.json"
    report_path.write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "acceptance_packet": {
                    "target_branch": "main",
                    "required_merged_branches": ["worker/api-status"],
                    "verifier_result": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=reports_dir,
            run_checkpoints_dir=checkpoints_dir,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    job = queue.enqueue(task_id="git_parallel_merge_acceptance_task")
    queue.finalize(
        job.job_id,
        state="completed",
        checkpoint_path=checkpoints_dir / "accepted.json",
        report_path=report_path,
        outcome="success",
        outcome_reasons=["verification_passed"],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_job_queue.py", "promote", "--job-id", job.job_id, "--detail", "accepted by operator"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    module.main()

    output = stream.getvalue()
    assert f"job_id={job.job_id} promoted=1" in output
    promoted = DelegatedJobQueue(queue_path).get(job.job_id)
    assert promoted is not None
    assert promoted.promoted_at
    assert promoted.promotion_detail == "accepted by operator"

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "promotable"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    module.main()

    assert f"job_id={job.job_id}" not in stream.getvalue()


def test_run_job_queue_list_cli_shows_blockers(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_trust_bootstrap_min_reports=100,
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    worker = queue.enqueue(task_id="git_parallel_worker_api_task", priority=1)
    queue.enqueue(
        task_id="git_parallel_merge_acceptance_task",
        priority=2,
        runtime_overrides={
            "dependency_job_ids": [worker.job_id],
            "required_worker_branches": ["worker/api-status"],
        },
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "list", "--blocked-only", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "task_id=git_parallel_merge_acceptance_task" in output
    assert "blocked=1" in output
    assert "blocked_by=dependency" in output


def test_run_job_queue_next_runnable_cli_prefers_worker_over_blocked_integrator(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    worker = queue.enqueue(task_id="git_parallel_worker_api_task", priority=1)
    queue.enqueue(
        task_id="git_parallel_merge_acceptance_task",
        priority=5,
        runtime_overrides={
            "dependency_job_ids": [worker.job_id],
            "required_worker_branches": ["worker/api-status"],
        },
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "next-runnable"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "task_id=git_parallel_worker_api_task" in output
    assert "runnable=1" in output
    assert "worker_job=1" in output


def test_run_job_queue_list_cli_orders_runnable_worker_before_blocked_integrator(monkeypatch, tmp_path):
    module = _load_script_module("run_job_queue.py")
    queue_path = tmp_path / "trajectories" / "jobs" / "queue.json"

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            model_name="mock-model",
            use_tolbert_context=False,
            sandbox_command_containment_mode="disabled",
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
            unattended_allow_git_commands=True,
        ),
    )
    queue = DelegatedJobQueue(queue_path)
    worker = queue.enqueue(task_id="git_parallel_worker_api_task", priority=1)
    queue.enqueue(
        task_id="git_parallel_merge_acceptance_task",
        priority=5,
        runtime_overrides={
            "dependency_job_ids": [worker.job_id],
            "required_worker_branches": ["worker/api-status"],
        },
    )

    monkeypatch.setattr(sys, "argv", ["run_job_queue.py", "list", "--show-blockers", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert "task_id=git_parallel_worker_api_task" in lines[0]
    assert "runnable=1" in lines[0]
    assert "worker_job=1" in lines[0]
    assert any("task_id=git_parallel_merge_acceptance_task" in line and "blocked=1" in line for line in lines[1:])
