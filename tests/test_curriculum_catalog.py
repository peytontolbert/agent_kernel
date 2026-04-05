from pathlib import Path

from agent_kernel.curriculum import CurriculumEngine
from agent_kernel.curriculum_catalog import (
    load_curriculum_metadata_catalog,
    load_curriculum_template_catalog,
    render_curriculum_template,
)
from agent_kernel.schemas import EpisodeRecord


def test_curriculum_catalog_exposes_validation_long_horizon_templates():
    catalog = load_curriculum_template_catalog()

    for template_id, family, min_plan_size in (
        ("workflow_validation_bundle", "workflow", 10),
        ("workflow_worker_validation_bundle", "workflow", 10),
        ("workflow_integrator_validation_bundle", "workflow", 10),
        ("tooling_validation_bundle", "tooling", 10),
        ("tooling_worker_validation_bundle", "tooling", 10),
        ("tooling_integrator_validation_bundle", "tooling", 10),
        ("integration_validation_bundle", "integration", 10),
        ("integration_worker_validation_bundle", "integration", 10),
        ("integration_integrator_validation_bundle", "integration", 10),
        ("validation_release_bundle", "validation", 11),
        ("validation_cleanup_gate_bundle", "validation", 12),
        ("validation_audit_gate_bundle", "validation", 12),
        ("validation_integrator_cleanup_gate_bundle", "validation", 14),
        ("validation_integrator_audit_gate_bundle", "validation", 14),
        ("governance_integrator_cleanup_review_bundle", "governance", 14),
        ("governance_integrator_audit_review_bundle", "governance", 14),
        ("oversight_integrator_cleanup_crosscheck_bundle", "oversight", 14),
        ("oversight_integrator_audit_crosscheck_bundle", "oversight", 14),
    ):
        template = catalog[template_id]
        metadata = template["metadata"]
        synthetic_edit_plan = metadata.get(
            "synthetic_edit_plan",
            template.get("payload", {}).get("synthetic_edit_plan", []),
        )

        assert metadata["benchmark_family"] == family
        assert metadata["difficulty"] == "long_horizon"
        assert metadata["long_horizon_variant"] == template_id
        assert metadata["long_horizon_coding_surface"] == template_id
        assert len(synthetic_edit_plan) >= min_plan_size
        if family == "workflow":
            assert template["task_id_template"] == "{task_id}_workflow_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_workflow_adjacent"
        elif family == "tooling":
            assert template["task_id_template"] == "{task_id}_tooling_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_tooling_adjacent"
        elif family == "integration":
            assert template["task_id_template"] == "{task_id}_integration_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_integration_adjacent"
        elif family == "validation":
            assert template["task_id_template"] == "{task_id}_validation_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_validation_adjacent"
        elif family == "governance":
            assert template["task_id_template"] == "{task_id}_governance_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_governance_adjacent"
        else:
            assert template["task_id_template"] == "{task_id}_oversight_adjacent"
            assert template["workspace_subdir_template"] == "{task_id}_oversight_adjacent"


def test_curriculum_catalog_exposes_long_horizon_metadata_catalogs():
    metadata = load_curriculum_metadata_catalog()

    assert metadata["long_horizon_target_family_rules"][0] == {
        "source_family": "repo_sandbox",
        "target_family": "repository",
    }
    assert metadata["long_horizon_variants"]["project"]["surface_variants"] == {
        "shared_repo_integrator": "shared_repo_integrator_bundle",
        "shared_repo_synthetic_worker": "shared_repo_worker_bundle",
    }
    assert metadata["long_horizon_variants"]["workflow"]["keyword_variants"][0] == {
        "keywords": ["review", "validation", "checkpoint"],
        "variant": "workflow_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["workflow"]["lineage_surface_variants"] == {
        "repository_integrator_bundle": "workflow_integrator_validation_bundle",
        "repository_validation_bundle": "workflow_validation_bundle",
        "repository_worker_bundle": "workflow_worker_validation_bundle",
        "shared_repo_integrator_bundle": "workflow_integrator_validation_bundle",
        "shared_repo_worker_bundle": "workflow_worker_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["tooling"]["keyword_variants"][0] == {
        "keywords": ["review", "contract", "validation"],
        "variant": "tooling_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["tooling"]["lineage_surface_variants"] == {
        "repository_integrator_bundle": "tooling_integrator_validation_bundle",
        "repository_validation_bundle": "tooling_validation_bundle",
        "repository_worker_bundle": "tooling_worker_validation_bundle",
        "shared_repo_integrator_bundle": "tooling_integrator_validation_bundle",
        "shared_repo_worker_bundle": "tooling_worker_validation_bundle",
        "workflow_integrator_validation_bundle": "tooling_integrator_validation_bundle",
        "workflow_worker_validation_bundle": "tooling_worker_validation_bundle",
        "workflow_validation_bundle": "tooling_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["integration"]["keyword_variants"][0] == {
        "keywords": ["review", "validation", "drill", "failover"],
        "variant": "integration_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["integration"]["lineage_surface_variants"] == {
        "repository_integrator_bundle": "integration_integrator_validation_bundle",
        "repository_validation_bundle": "integration_validation_bundle",
        "repository_worker_bundle": "integration_worker_validation_bundle",
        "shared_repo_integrator_bundle": "integration_integrator_validation_bundle",
        "shared_repo_worker_bundle": "integration_worker_validation_bundle",
        "tooling_integrator_validation_bundle": "integration_integrator_validation_bundle",
        "tooling_worker_validation_bundle": "integration_worker_validation_bundle",
        "tooling_validation_bundle": "integration_validation_bundle",
        "workflow_integrator_validation_bundle": "integration_integrator_validation_bundle",
        "workflow_worker_validation_bundle": "integration_worker_validation_bundle",
        "workflow_validation_bundle": "integration_validation_bundle",
    }
    assert metadata["long_horizon_variants"]["repo_chore"]["lineage_surface_variants"] == {
        "integration_integrator_validation_bundle": "repo_chore_audit_bundle",
        "integration_worker_validation_bundle": "repo_chore_cleanup_bundle",
        "repository_integrator_bundle": "repo_chore_audit_bundle",
        "repository_worker_bundle": "repo_chore_cleanup_bundle",
        "shared_repo_integrator_bundle": "repo_chore_audit_bundle",
        "shared_repo_worker_bundle": "repo_chore_cleanup_bundle",
    }
    assert metadata["long_horizon_variants"]["validation"]["lineage_surface_branch_variants"] == {
        "shared_repo_integrator_bundle": {
            "audit": "validation_integrator_audit_gate_bundle",
            "cleanup": "validation_integrator_cleanup_gate_bundle",
        },
        "repository_integrator_bundle": {
            "audit": "validation_integrator_audit_gate_bundle",
            "cleanup": "validation_integrator_cleanup_gate_bundle",
        },
    }
    assert metadata["long_horizon_variants"]["governance"]["lineage_surface_variants"] == {
        "validation_integrator_audit_gate_bundle": "governance_integrator_audit_review_bundle",
        "validation_integrator_cleanup_gate_bundle": "governance_integrator_cleanup_review_bundle",
    }
    assert metadata["long_horizon_variants"]["oversight"]["lineage_surface_variants"] == {
        "governance_integrator_audit_review_bundle": "oversight_integrator_audit_crosscheck_bundle",
        "governance_integrator_cleanup_review_bundle": "oversight_integrator_cleanup_crosscheck_bundle",
    }
    assert metadata["lineage_branch_kind_rules"]["crosscheck"]["variant_keywords"] == [
        "crosscheck",
        "cert",
    ]


def test_render_curriculum_template_renders_validation_release_bundle():
    task = render_curriculum_template(
        "validation_release_bundle",
        replacements={"task_id": "repo_tail"},
        metadata_overrides={"parent_task": "repo_tail_repo_chore_adjacent"},
        prompt_suffix=" Reuse the validated pattern from retained evidence.",
    )

    assert task.task_id == "repo_tail_validation_adjacent"
    assert task.workspace_subdir == "repo_tail_validation_adjacent"
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["long_horizon_coding_surface"] == "validation_release_bundle"
    assert task.metadata["parent_task"] == "repo_tail_repo_chore_adjacent"
    assert task.metadata["synthetic_edit_plan"][0]["path"] == "validation/gate/summary.md"
    assert "staged validation gate for repo_tail" in task.prompt
    assert task.prompt.endswith("Reuse the validated pattern from retained evidence.")
    assert task.expected_file_contents["validation/gate/packet.txt"].startswith("validation packet\nrepo_tail")
    assert "repo_tail validation gate ready" in task.success_command


def test_curriculum_generates_validation_cleanup_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore cleanup bundle.",
        workspace="workspace/repo_chore_cleanup_seed",
        success=True,
        task_metadata={
            "benchmark_family": "repo_chore",
            "origin_benchmark_family": "integration",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "repo_chore_cleanup_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "integration"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "validation_cleanup_gate_bundle"
    assert task.metadata["long_horizon_variant"] == "validation_cleanup_gate_bundle"
    assert "validation/cleanup/packet.txt" in task.expected_files
    assert "validation/reports/sweep.txt" in task.expected_files


def test_curriculum_generates_validation_audit_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore audit bundle.",
        workspace="workspace/repo_chore_audit_seed",
        success=True,
        task_metadata={
            "benchmark_family": "repo_chore",
            "origin_benchmark_family": "integration",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "repo_chore_audit_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "integration"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "validation_audit_gate_bundle"
    assert task.metadata["long_horizon_variant"] == "validation_audit_gate_bundle"
    assert "validation/audit/packet.txt" in task.expected_files
    assert "validation/reports/findings.txt" in task.expected_files
