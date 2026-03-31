from agent_kernel.config import KernelConfig
from agent_kernel.job_queue import DelegatedJobQueue, DelegatedRuntimeController
from agent_kernel.schemas import TaskSpec
from agent_kernel.shared_repo import prepare_runtime_task, shared_repo_claim


def test_shared_repo_claim_derives_claimed_paths_from_semantic_verifier():
    task = TaskSpec(
        task_id="derived_worker",
        prompt="update one worker-owned path",
        workspace_subdir="derived_worker",
        expected_files=["src/api_status.txt", "docs/status.md"],
        metadata={
            "workflow_guard": {
                "shared_repo_id": "repo-a",
                "target_branch": "main",
                "worker_branch": "worker/api-status",
            },
            "semantic_verifier": {
                "expected_changed_paths": ["src/api_status.txt"],
                "report_rules": [{"path": "reports/worker_report.txt"}],
            },
        },
    )

    claim = shared_repo_claim(task)

    assert claim["claimed_paths"] == ("reports/worker_report.txt", "src/api_status.txt")
    assert claim["claim_source"] == "derived"


def test_prepare_runtime_task_injects_derived_claimed_paths():
    task = TaskSpec(
        task_id="derived_worker",
        prompt="update one worker-owned path",
        workspace_subdir="derived_worker",
        expected_files=["src/api_status.txt", "docs/status.md"],
        metadata={
            "workflow_guard": {
                "shared_repo_id": "repo-a",
                "target_branch": "main",
                "worker_branch": "worker/api-status",
            },
            "semantic_verifier": {
                "expected_changed_paths": ["src/api_status.txt"],
            },
        },
    )

    prepared = prepare_runtime_task(task, job_id="job-1")
    workflow_guard = prepared.metadata["workflow_guard"]

    assert workflow_guard["claimed_paths"] == ["src/api_status.txt"]
    assert workflow_guard["claimed_paths_source"] == "derived"


def test_prepare_runtime_task_overlays_required_worker_branches():
    task = TaskSpec(
        task_id="integrator",
        prompt="accept worker branches",
        workspace_subdir="integrator",
        metadata={
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/a"],
            },
        },
    )

    prepared = prepare_runtime_task(
        task,
        runtime_overrides={"required_worker_branches": ["worker/a", "worker/b", "worker/c"]},
        job_id="job-2",
    )

    assert prepared.metadata["semantic_verifier"]["required_merged_branches"] == [
        "worker/a",
        "worker/b",
        "worker/c",
    ]


def test_runtime_controller_blocks_colliding_derived_shared_repo_claims(tmp_path):
    config = KernelConfig(
        provider="mock",
        model_name="mock-model",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        delegated_job_runtime_state_path=tmp_path / "trajectories" / "jobs" / "runtime_state.json",
        delegated_job_max_concurrency=2,
    )
    config.ensure_directories()
    controller = DelegatedRuntimeController(config.delegated_job_runtime_state_path, runner_id="runner-a")
    queue = DelegatedJobQueue(tmp_path / "trajectories" / "jobs" / "queue.json")
    first_job = queue.enqueue(task_id="derived_worker_a", runtime_overrides={"shared_repo_id": "repo-a"})
    second_job = queue.enqueue(task_id="derived_worker_b", runtime_overrides={"shared_repo_id": "repo-a"})
    first_task = TaskSpec(
        task_id="derived_worker_a",
        prompt="worker a",
        workspace_subdir="derived_worker_a",
        expected_files=["src/api_status.txt"],
        metadata={
            "workflow_guard": {"target_branch": "main", "worker_branch": "worker/api-status"},
            "semantic_verifier": {"expected_changed_paths": ["src/api_status.txt"]},
        },
    )
    second_task = TaskSpec(
        task_id="derived_worker_b",
        prompt="worker b",
        workspace_subdir="derived_worker_b",
        expected_files=["src/api_status.txt"],
        metadata={
            "workflow_guard": {"target_branch": "main", "worker_branch": "worker/api-copy"},
            "semantic_verifier": {"expected_changed_paths": ["src/api_status.txt"]},
        },
    )

    first_lease, first_error = controller.acquire(
        job=first_job,
        task=first_task,
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "a.json",
        report_path=tmp_path / "trajectories" / "reports" / "a.json",
    )
    second_lease, second_error = controller.acquire(
        job=second_job,
        task=second_task,
        config=config,
        checkpoint_path=tmp_path / "trajectories" / "checkpoints" / "b.json",
        report_path=tmp_path / "trajectories" / "reports" / "b.json",
    )

    assert first_lease is not None
    assert first_error is None
    assert second_lease is None
    assert second_error == "lease_collision:claimed_path:repo-a:src/api_status.txt"


def test_shared_repo_claim_derives_claimed_paths_from_synthetic_edit_metadata():
    task = TaskSpec(
        task_id="synthetic_worker",
        prompt="update one worker-owned path",
        workspace_subdir="synthetic_worker",
        expected_files=["src/api_status.txt", "docs/status.md"],
        metadata={
            "workflow_guard": {
                "shared_repo_id": "repo-a",
                "target_branch": "main",
                "worker_branch": "worker/api-status",
            },
            "synthetic_edit_plan": [
                {"path": "src/api_status.txt", "edit_kind": "token_replace"},
            ],
            "synthetic_edit_candidates": [
                {
                    "path": "src/api_status.txt",
                    "selected": {"path": "src/api_status.txt"},
                    "candidates": [{"path": "src/api_status.txt"}],
                }
            ],
        },
    )

    claim = shared_repo_claim(task)

    assert claim["claimed_paths"] == ("src/api_status.txt",)
    assert claim["claim_source"] == "derived"
