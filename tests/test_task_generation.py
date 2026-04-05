import json
from pathlib import Path

import pytest

from agent_kernel.config import KernelConfig
from evals.harness import run_eval
from agent_kernel import task_bank as task_bank_module
from agent_kernel.task_bank import TaskBank


def test_task_bank_loads_default_tasks_from_bundled_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "bundled_tasks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "bundled_only_task",
                        "prompt": "Create bundled.txt containing bundled ready.",
                        "workspace_subdir": "bundled_only_task",
                        "suggested_commands": ["printf 'bundled ready\n' > bundled.txt"],
                        "success_command": "test -f bundled.txt && grep -q '^bundled ready$' bundled.txt",
                        "expected_files": ["bundled.txt"],
                        "expected_file_contents": {"bundled.txt": "bundled ready\n"},
                        "metadata": {"benchmark_family": "bundled_lab", "capability": "bundled_flow", "difficulty": "seed"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(task_bank_module, "_BUILTIN_TASK_MANIFEST_PATH", manifest_path)
    bank = task_bank_module.TaskBank(external_task_manifests=())

    assert bank.get("bundled_only_task").metadata["benchmark_family"] == "bundled_lab"
    with pytest.raises(KeyError):
        bank.get("hello_task")


def test_task_bank_loads_synthesis_rules_from_bundled_dataset(monkeypatch, tmp_path):
    rules_path = tmp_path / "synthesis_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "worker_prompt_template": "branch={branch} owned={owned}",
                "worker_report": {
                    "path_template": "logs/{worker_name}.txt",
                    "base_must_mention": ["logged"],
                },
                "branch_intent": {
                    "preferred_states": {"fixed": "ready"},
                    "direct_replacements": ["pending"],
                    "todo_replacement": "done",
                    "draft_replacement": "final",
                },
                "edit_kind_order": ["rewrite", "token_replace"],
                "edit_scores": {
                    "token_replace": {"base": 10, "per_replacement": 5, "char_weight": 1},
                    "rewrite": {"base": 120, "char_weight": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(task_bank_module, "_TASK_BANK_SYNTHESIS_RULES_PATH", rules_path)
    task_bank_module._task_bank_synthesis_rules.cache_clear()

    try:
        assert task_bank_module._worker_prompt("worker/fixed", ["src/status.txt"]) == (
            "branch=worker/fixed owned=src/status.txt"
        )
        assert task_bank_module._derive_worker_report_rules(
            "worker/fixed",
            ["src/status.txt"],
            [],
        ) == [
            {
                "path": "logs/worker_fixed.txt",
                "must_mention": ["logged", "worker/fixed"],
                "covers": ["src/status.txt"],
            }
        ]
        assert task_bank_module._preferred_branch_state("worker/fixed", "src/status.txt") == "ready"
        assert task_bank_module._edit_kind_rank("rewrite") == 0
    finally:
        task_bank_module._task_bank_synthesis_rules.cache_clear()


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
    assert TaskBank().get("hello_task").metadata["light_supervision_candidate"] is True
    assert TaskBank().get("hello_task").metadata["light_supervision_contract_kind"] == "workspace_acceptance"
    assert TaskBank().get("hello_task").metadata["decision_yield_family"] == "bounded"
    assert TaskBank().get("hello_task").metadata["decision_yield_contract_candidate"] is True
    retrieval_task = TaskBank().get("handshake_retrieval_task")
    assert retrieval_task.metadata["requires_retrieval"] is True
    assert retrieval_task.metadata["light_supervision_candidate"] is False
    assert retrieval_task.metadata["decision_yield_family"] == "bounded"
    assert retrieval_task.metadata["decision_yield_contract_candidate"] is False
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
    assert TaskBank().get("project_release_cutover_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("repository_migration_wave_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("tooling_release_contract_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("integration_failover_drill_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("git_release_train_acceptance_retrieval_task").metadata["requires_retrieval"] is True
    assert TaskBank().get("git_release_train_conflict_acceptance_retrieval_task").metadata["requires_retrieval"] is True


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


def test_task_bank_preserves_nested_repo_sandbox_metadata_from_bundled_manifest():
    task = TaskBank().get("git_generated_conflict_resolution_task")

    assert task.metadata["shared_repo_bootstrap_fixture_dir"] == "repo_sandbox_generated_conflict"
    assert "scripts/generate_bundle.sh" in task.metadata["shared_repo_bootstrap_executable_paths"]
    assert task.metadata["workflow_guard"]["shared_repo_id"] == "repo_sandbox_generated_conflict"
    assert task.metadata["semantic_verifier"]["required_merged_branches"] == ["worker/status-refresh"]
    assert "dist/status_bundle.txt" in task.metadata["semantic_verifier"]["generated_paths"]


def test_task_bank_preserves_release_train_conflict_repo_metadata_from_bundled_manifest():
    task = TaskBank().get("git_release_train_conflict_acceptance_task")

    assert task.metadata["shared_repo_bootstrap_fixture_dir"] == "repo_sandbox_release_train_conflict"
    assert "scripts/build_release_packet.sh" in task.metadata["shared_repo_bootstrap_executable_paths"]
    assert task.metadata["workflow_guard"]["shared_repo_id"] == "repo_sandbox_release_train_conflict"
    assert task.metadata["workflow_guard"]["touches_generated_paths"] is True
    assert task.metadata["semantic_verifier"]["required_merged_branches"] == [
        "worker/api-release",
        "worker/docs-release",
        "worker/ops-release",
    ]
    assert task.metadata["semantic_verifier"]["resolved_conflict_paths"] == ["docs/runbook.md"]
    assert "dist/release_packet.txt" in task.metadata["semantic_verifier"]["generated_paths"]


def test_task_bank_frontier_includes_harder_bundled_long_horizon_tasks():
    bank = TaskBank()

    assert bank.get("project_release_cutover_task").max_steps >= 18
    assert bank.get("repository_migration_wave_task").max_steps >= 22
    assert bank.get("tooling_release_contract_task").max_steps >= 20
    assert bank.get("integration_failover_drill_task").max_steps >= 24
    assert bank.get("git_release_train_acceptance_task").max_steps >= 28
    assert bank.get("git_release_train_conflict_acceptance_task").max_steps >= 32


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
    assert metrics.total == 34
    assert metrics.total_by_benchmark_family["workflow"] == 2
    assert metrics.passed_by_benchmark_family["workflow"] == 2
    assert metrics.total_by_benchmark_family["project"] == 4
    assert metrics.passed_by_benchmark_family["project"] == 4
    assert metrics.total_by_benchmark_family["repository"] == 4
    assert metrics.passed_by_benchmark_family["repository"] == 4
    assert metrics.total_by_benchmark_family["repo_chore"] == 2
    assert metrics.passed_by_benchmark_family["repo_chore"] == 2
    assert metrics.total_by_benchmark_family["tooling"] == 3
    assert metrics.passed_by_benchmark_family["tooling"] == 3
    assert metrics.total_by_benchmark_family["integration"] == 4
    assert metrics.passed_by_benchmark_family["integration"] == 4
    assert metrics.total_by_difficulty["long_horizon"] == 4
    assert metrics.passed_by_difficulty["long_horizon"] == 4
    assert metrics.total_by_difficulty["cross_component"] == 4
    assert metrics.passed_by_difficulty["cross_component"] == 4
    assert metrics.total_by_difficulty["cross_tool"] == 3
    assert metrics.passed_by_difficulty["cross_tool"] == 3
    assert metrics.total_by_difficulty["multi_system"] == 4
    assert metrics.passed_by_difficulty["multi_system"] == 4


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


def test_task_bank_exposes_five_low_cost_integration_primaries_for_trust_bootstrap():
    bank = TaskBank()
    integration_primaries = [
        task.task_id
        for task in bank.list()
        if str(task.metadata.get("benchmark_family", "")) == "integration"
        and not bool(task.metadata.get("requires_retrieval", False))
    ]

    assert {
        "incident_matrix_task",
        "service_mesh_task",
        "queue_failover_task",
        "replica_cutover_task",
        "bridge_handoff_task",
    } <= set(integration_primaries)


def test_task_bank_exposes_five_low_cost_repository_primaries_for_trust_bootstrap():
    bank = TaskBank()
    repository_primaries = [
        task.task_id
        for task in bank.list()
        if str(task.metadata.get("benchmark_family", "")) == "repository"
        and not bool(task.metadata.get("requires_retrieval", False))
    ]

    assert {
        "service_release_task",
        "schema_alignment_task",
        "repo_sync_matrix_task",
        "repository_guardrail_sync_task",
        "repository_audit_packet_task",
    } <= set(repository_primaries)


def test_task_bank_exposes_five_low_cost_repo_chore_primaries_for_trust_bootstrap():
    bank = TaskBank()
    repo_chore_primaries = [
        task.task_id
        for task in bank.list()
        if str(task.metadata.get("benchmark_family", "")) == "repo_chore"
        and not bool(task.metadata.get("requires_retrieval", False))
    ]

    assert {
        "repo_notice_review_task",
        "repo_guardrail_review_task",
        "repo_packet_review_task",
        "repo_cleanup_review_task",
        "repo_patch_review_task",
    } <= set(repo_chore_primaries)
