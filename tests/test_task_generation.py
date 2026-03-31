import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from evals.harness import run_eval
from agent_kernel.task_bank import TaskBank


def test_task_bank_returns_isolated_copies():
    bank = TaskBank()
    task_a = bank.get("hello_task")
    task_b = bank.get("hello_task")

    task_a.suggested_commands.append("echo mutated")

    assert "echo mutated" not in task_b.suggested_commands


def test_task_bank_exposes_multiple_capabilities():
    tasks = TaskBank().list()

    assert len(tasks) >= 48
    assert {task.metadata["capability"] for task in tasks} >= {
        "file_write",
        "nested_filesystem",
        "filesystem_mutation",
        "file_edit",
        "cleanup",
        "workflow_environment",
        "project_environment",
        "repo_environment",
        "tool_environment",
        "integration_environment",
        "retrieval_bootstrap",
        "retrieval_dependent",
    }
    assert {task.metadata["benchmark_family"] for task in tasks} >= {
        "micro",
        "workflow",
        "project",
        "repository",
        "repo_chore",
        "tooling",
        "integration",
        "bounded",
    }
    retrieval_task = TaskBank().get("handshake_retrieval_task")
    assert retrieval_task.metadata["requires_retrieval"] is True
    assert TaskBank().get("status_phrase_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("status_phrase_fallback_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("archive_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("avoidance_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("bundle_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("bundle_legacy_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("checkpoint_blue_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("checkpoint_green_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("release_bundle_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("config_sync_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("deployment_manifest_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("report_rollup_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("release_packet_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("service_release_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("schema_alignment_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("repo_sync_matrix_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("repo_patch_review_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("repo_cleanup_review_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("git_repo_test_repair_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("api_contract_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("cli_exchange_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("service_mesh_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("incident_matrix_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("queue_failover_retrieval_task").metadata["requires_retrieval"] is True


def test_task_bank_loads_external_manifest_tasks(tmp_path):
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "external_manifest_task",
                        "prompt": "Create external.txt containing external ready.",
                        "workspace_subdir": "external_manifest_task",
                        "suggested_commands": ["printf 'external ready\\n' > external.txt"],
                        "success_command": "test -f external.txt && grep -q '^external ready$' external.txt",
                        "expected_files": ["external.txt"],
                        "expected_file_contents": {"external.txt": "external ready\n"},
                        "metadata": {"benchmark_family": "external_lab", "capability": "external_flow"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bank = TaskBank(config=KernelConfig(external_task_manifests_paths=(str(manifest_path),)))
    task = bank.get("external_manifest_task")

    assert task.metadata["task_origin"] == "external_manifest"
    assert task.metadata["external_manifest_path"] == str(manifest_path)
    assert task.metadata["benchmark_family"] == "external_lab"


def test_task_bank_loads_external_manifest_tasks_from_directory_and_glob(tmp_path):
    manifests_dir = tmp_path / "task_manifests"
    manifests_dir.mkdir()
    directory_manifest = manifests_dir / "directory_tasks.json"
    directory_manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "directory_manifest_task",
                        "prompt": "Create directory.txt containing directory ready.",
                        "workspace_subdir": "directory_manifest_task",
                        "suggested_commands": ["printf 'directory ready\\n' > directory.txt"],
                        "success_command": "test -f directory.txt && grep -q '^directory ready$' directory.txt",
                        "expected_files": ["directory.txt"],
                        "expected_file_contents": {"directory.txt": "directory ready\n"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    glob_manifest = manifests_dir / "glob_tasks.jsonl"
    glob_manifest.write_text(
        json.dumps(
            {
                "task_id": "glob_manifest_task",
                "prompt": "Create glob.txt containing glob ready.",
                "workspace_subdir": "glob_manifest_task",
                "suggested_commands": ["printf 'glob ready\\n' > glob.txt"],
                "success_command": "test -f glob.txt && grep -q '^glob ready$' glob.txt",
                "expected_files": ["glob.txt"],
                "expected_file_contents": {"glob.txt": "glob ready\n"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bank = TaskBank(
        config=KernelConfig(
            external_task_manifests_paths=(str(manifests_dir), str(manifests_dir / "*.jsonl")),
        )
    )

    assert bank.get("directory_manifest_task").metadata["external_manifest_path"] == str(directory_manifest)
    assert bank.get("glob_manifest_task").metadata["external_manifest_path"] == str(glob_manifest)


def test_mock_eval_solves_expanded_task_bank(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    metrics = run_eval(config=config)

    assert metrics.passed == metrics.total
    assert metrics.total == 30
    assert metrics.total_by_benchmark_family["workflow"] == 2
    assert metrics.passed_by_benchmark_family["workflow"] == 2
    assert metrics.total_by_benchmark_family["project"] == 3
    assert metrics.passed_by_benchmark_family["project"] == 3
    assert metrics.total_by_benchmark_family["repository"] == 3
    assert metrics.passed_by_benchmark_family["repository"] == 3
    assert metrics.total_by_benchmark_family["repo_chore"] == 2
    assert metrics.passed_by_benchmark_family["repo_chore"] == 2
    assert metrics.total_by_benchmark_family["tooling"] == 2
    assert metrics.passed_by_benchmark_family["tooling"] == 2
    assert metrics.total_by_benchmark_family["integration"] == 3
    assert metrics.passed_by_benchmark_family["integration"] == 3
    assert metrics.total_by_difficulty["long_horizon"] == 3
    assert metrics.passed_by_difficulty["long_horizon"] == 3
    assert metrics.total_by_difficulty["cross_component"] == 3
    assert metrics.passed_by_difficulty["cross_component"] == 3
    assert metrics.total_by_difficulty["cross_tool"] == 2
    assert metrics.passed_by_difficulty["cross_tool"] == 2
    assert metrics.total_by_difficulty["multi_system"] == 3
    assert metrics.passed_by_difficulty["multi_system"] == 3


def test_mock_eval_includes_external_manifest_tasks(tmp_path):
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "external_manifest_eval_task",
                        "prompt": "Create external.txt containing external ready.",
                        "workspace_subdir": "external_manifest_eval_task",
                        "suggested_commands": ["printf 'external ready\\n' > external.txt"],
                        "success_command": "test -f external.txt && grep -q '^external ready$' external.txt",
                        "expected_files": ["external.txt"],
                        "expected_file_contents": {"external.txt": "external ready\n"},
                        "metadata": {"benchmark_family": "external_lab", "capability": "external_flow"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        external_task_manifests_paths=(str(manifest_path),),
    )

    metrics = run_eval(config=config)

    assert metrics.total_by_benchmark_family["external_lab"] == 1
    assert metrics.passed_by_benchmark_family["external_lab"] == 1


def test_bootstrap_skill_fixture_covers_retrieval_bootstrap_tasks():
    fixture_path = Path("var/skills/bootstrap_command_skills.json")
    skills = json.loads(fixture_path.read_text(encoding="utf-8"))
    source_tasks = {skill["source_task_id"] for skill in skills}

    assert {
        "hello_task",
        "math_task",
        "handshake_seed_task",
        "status_phrase_seed_task",
        "status_phrase_fallback_seed_task",
        "archive_command_seed_task",
        "avoidance_seed_task",
        "bundle_primary_seed_task",
        "bundle_legacy_seed_task",
        "checkpoint_blue_seed_task",
        "checkpoint_green_seed_task",
        "deployment_manifest_task",
        "report_rollup_task",
        "release_packet_task",
        "service_release_task",
        "schema_alignment_task",
        "repo_sync_matrix_task",
        "repo_patch_review_task",
        "repo_cleanup_review_task",
        "api_contract_task",
        "cli_exchange_task",
        "service_mesh_task",
        "incident_matrix_task",
        "queue_failover_task",
    } <= source_tasks
