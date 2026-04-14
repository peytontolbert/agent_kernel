from pathlib import Path
import json

from agent_kernel.tasking.curriculum import CurriculumEngine
from agent_kernel.tasking.curriculum_catalog import render_curriculum_template
from agent_kernel.loop import AgentKernel
from agent_kernel.memory import EpisodeMemory
from agent_kernel.policy import Policy
from agent_kernel.schemas import EpisodeRecord, StepRecord
from agent_kernel.config import KernelConfig
from agent_kernel.schemas import ActionDecision
from agent_kernel.tasking.task_bank import TaskBank


def test_curriculum_generates_adjacent_task():
    episode = EpisodeRecord(
        task_id="hello_task",
        prompt="Create hello.txt containing hello agent kernel.",
        workspace="workspace/hello_task",
        success=True,
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine().generate_adjacent_task(episode)

    assert task.task_id == "hello_task_followup"
    assert task.metadata["parent_task"] == "hello_task"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["light_supervision_candidate"] is True
    assert task.metadata["light_supervision_contract_kind"] == "workspace_acceptance"


def test_task_bank_uplifts_frontier_task_contract_budgets():
    bank = TaskBank()

    assert bank.get("deployment_manifest_task").max_steps >= 12
    assert bank.get("project_release_cutover_task").max_steps >= 18
    assert bank.get("service_release_task").max_steps >= 14
    assert bank.get("repository_migration_wave_task").max_steps >= 22
    assert bank.get("tooling_release_contract_task").max_steps >= 20
    assert bank.get("integration_failover_drill_task").max_steps >= 24
    assert bank.get("git_release_train_acceptance_task").max_steps >= 28
    assert bank.get("git_release_train_conflict_acceptance_task").max_steps >= 32
    assert bank.get("git_parallel_merge_acceptance_task").max_steps >= 20
    assert bank.get("git_generated_conflict_resolution_task").max_steps >= 20


def test_task_bank_infers_repo_semantics_for_builtin_tasks():
    bank = TaskBank()

    project_task = bank.get("deployment_manifest_task")
    shared_repo_task = bank.get("git_parallel_merge_acceptance_task")

    assert "project" in project_task.metadata["repo_semantics"]
    assert "validation" in project_task.metadata["repo_semantics"]
    assert project_task.metadata["workflow_shape"] in {"multi_artifact_workspace", "command_driven", "single_workspace"}
    assert project_task.metadata["contract_shape"] == "workspace_acceptance"
    assert "shared_repo" in shared_repo_task.metadata["repo_semantics"]
    assert "integration" in shared_repo_task.metadata["repo_semantics"]
    assert shared_repo_task.metadata["workflow_shape"] == "shared_repo_parallel"


def test_curriculum_failure_recovery_selection_prefers_new_repo_semantic_clusters():
    engine = CurriculumEngine()
    alpha = EpisodeRecord(
        task_id="alpha_project_seed",
        prompt="repair project release packet",
        workspace="workspace/alpha_project_seed",
        success=False,
        termination_reason="command_failure",
        task_metadata={
            "benchmark_family": "project",
            "repo_semantics": ["project", "validation"],
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        steps=[],
    )
    beta = EpisodeRecord(
        task_id="beta_repository_seed",
        prompt="repair repo validation packet",
        workspace="workspace/beta_repository_seed",
        success=False,
        termination_reason="command_failure",
        task_metadata={
            "benchmark_family": "project",
            "repo_semantics": ["repository", "integration"],
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        steps=[],
    )
    gamma = EpisodeRecord(
        task_id="gamma_project_seed",
        prompt="repair project checklist",
        workspace="workspace/gamma_project_seed",
        success=False,
        termination_reason="command_failure",
        task_metadata={
            "benchmark_family": "project",
            "repo_semantics": ["project", "validation"],
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        steps=[],
    )

    selected = engine._select_failure_recovery_seed_set([alpha, beta, gamma], limit=2)

    assert [episode.task_id for episode in selected] == ["alpha_project_seed", "beta_repository_seed"]


def test_curriculum_templates_surface_synthetic_edit_plan_in_metadata():
    task = render_curriculum_template(
        "adjudication_cleanup_ruling_bundle",
        replacements={"task_id": "synthetic_probe"},
    )

    assert len(task.metadata["synthetic_edit_plan"]) >= 8


def test_curriculum_generates_workflow_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="release_bundle_task",
            prompt="Prepare release bundle.",
            workspace="workspace/release_bundle_task",
            success=True,
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="release_bundle_task",
        prompt="Prepare release bundle.",
        workspace="workspace/release_bundle_task",
        success=True,
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "release_bundle_task_workflow_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "workflow"
    assert "audit/summary.txt" in task.expected_files


def test_curriculum_generates_project_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="deployment_manifest_task",
            prompt="Prepare deployment workspace.",
            workspace="workspace/deployment_manifest_task",
            success=True,
            task_metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=True,
        task_metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "deployment_manifest_task_project_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "project"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_step_count"] == 9
    assert task.metadata["long_horizon_coding_surface"] == "project_release_bundle"
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "project/plan/summary.md",
        "project/plan/summary.md",
        "project/plan/summary.md",
        "project/plan/timeline.txt",
        "project/reports/checklist.txt",
        "project/reports/checklist.txt",
        "project/reports/packet.txt",
        "project/reports/verify.txt",
        "project/reports/checklist.txt",
    ]
    assert [step["edit_kind"] for step in task.metadata["synthetic_edit_plan"]] == [
        "line_replace",
        "line_replace",
        "line_replace",
        "line_replace",
        "line_replace",
        "line_replace",
        "block_replace",
        "line_replace",
        "line_insert",
    ]
    assert task.suggested_commands == []
    assert "project/source.txt" in task.expected_files
    assert "project/reports/packet.txt" in task.expected_files
    assert "project/source.txt" in task.expected_file_contents
    assert task.max_steps == 12


def test_curriculum_generates_shared_repo_worker_project_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task__worker__worker_docs-status",
        prompt="Prepare shared-repo worker patch.",
        workspace="workspace/git_parallel_merge_acceptance_task__worker__worker_docs-status",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "shared_repo_synthetic_worker",
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_worker_adjacent")
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "project"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_step_count"] == 9
    assert task.metadata["long_horizon_coding_surface"] == "shared_repo_worker_bundle"
    assert task.metadata["parent_long_horizon_coding_surface"] == "shared_repo_synthetic_worker"
    assert task.metadata["synthetic_worker"] is True
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "shared_repo/worker/claim.txt",
        "shared_repo/worker/claim.txt",
        "shared_repo/worker/summary.md",
        "shared_repo/worker/summary.md",
        "shared_repo/worker/checklist.txt",
        "shared_repo/worker/checklist.txt",
        "shared_repo/worker/packet.txt",
        "shared_repo/worker/verify.txt",
        "shared_repo/worker/checklist.txt",
    ]
    assert "shared_repo/worker/packet.txt" in task.expected_files
    assert task.max_steps == 12


def test_curriculum_retargets_shared_repo_worker_adjacent_to_repository_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task__worker__worker_docs-status",
        prompt="Prepare shared-repo worker patch.",
        workspace="workspace/git_parallel_merge_acceptance_task__worker__worker_docs-status",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "origin_benchmark_family": "repo_sandbox",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "shared_repo_synthetic_worker",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repo_sandbox"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repository_adjacent")
    assert task.metadata["benchmark_family"] == "repository"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repository_worker_bundle"
    assert task.metadata["long_horizon_variant"] == "repository_worker_bundle"
    assert task.metadata["parent_long_horizon_coding_surface"] == "shared_repo_synthetic_worker"
    assert "repo/reports/packet.txt" in task.expected_files


def test_curriculum_retargets_raw_repo_sandbox_worker_adjacent_to_repository_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task",
        prompt="Prepare worker branch update.",
        workspace="workspace/git_parallel_worker_api_task",
        success=True,
        task_metadata={
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
        task_contract={
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
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "git_parallel_worker_api_task_repository_adjacent"
    assert task.metadata["benchmark_family"] == "repository"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_coding_surface"] == "repository_worker_bundle"
    assert task.metadata["long_horizon_variant"] == "repository_worker_bundle"
    assert task.metadata["parent_long_horizon_coding_surface"] == "shared_repo_synthetic_worker"
    assert task.metadata["synthetic_worker"] is True
    assert "repo/reports/packet.txt" in task.expected_files


def test_curriculum_generates_shared_repo_integrator_project_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task",
        prompt="Prepare shared-repo integrator packet.",
        workspace="workspace/git_generated_conflict_resolution_task",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "shared_repo_integrator",
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integrator_adjacent")
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "project"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_step_count"] == 10
    assert task.metadata["long_horizon_coding_surface"] == "shared_repo_integrator_bundle"
    assert task.metadata["parent_long_horizon_coding_surface"] == "shared_repo_integrator"
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "shared_repo/integration/summary.md",
        "shared_repo/integration/summary.md",
        "shared_repo/integration/merge_plan.txt",
        "shared_repo/integration/checklist.txt",
        "shared_repo/integration/checklist.txt",
        "shared_repo/integration/checklist.txt",
        "shared_repo/integration/tests.txt",
        "shared_repo/integration/packet.txt",
        "shared_repo/integration/report.txt",
        "shared_repo/integration/finalize.txt",
    ]
    assert "shared_repo/integration/packet.txt" in task.expected_files
    assert task.max_steps == 14


def test_curriculum_retargets_shared_repo_integrator_adjacent_to_repository_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task",
        prompt="Prepare shared-repo integrator packet.",
        workspace="workspace/git_generated_conflict_resolution_task",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "origin_benchmark_family": "repo_sandbox",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "shared_repo_integrator",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repo_sandbox"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repository_adjacent")
    assert task.metadata["benchmark_family"] == "repository"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repository_integrator_bundle"
    assert task.metadata["long_horizon_variant"] == "repository_integrator_bundle"
    assert task.metadata["parent_long_horizon_coding_surface"] == "shared_repo_integrator"
    assert "repo/reports/packet.txt" in task.expected_files


def test_curriculum_retargets_raw_repo_sandbox_review_adjacent_to_repository_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_repo_status_review_task",
        prompt="Prepare status review packet.",
        workspace="workspace/git_repo_status_review_task",
        success=True,
        task_metadata={
            "benchmark_family": "repo_sandbox",
            "difficulty": "git_workflow",
            "workflow_guard": {"requires_git": True},
        },
        task_contract={
            "metadata": {
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_workflow",
                "workflow_guard": {"requires_git": True},
            }
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "git_repo_status_review_task_repository_adjacent"
    assert task.metadata["benchmark_family"] == "repository"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_coding_surface"] == "repository_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "repository_validation_bundle"
    assert "repo/validation/matrix.txt" in task.expected_files


def test_curriculum_generates_project_validation_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="report_rollup_task",
        prompt="Prepare project validation rollup.",
        workspace="workspace/report_rollup_task",
        success=True,
        task_metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "report_rollup_task_project_adjacent"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["long_horizon_coding_surface"] == "project_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "project_validation_bundle"
    assert task.metadata["long_horizon_step_count"] == 10
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "project/validation/summary.md",
        "project/validation/summary.md",
        "project/validation/summary.md",
        "project/validation/risk_register.txt",
        "project/validation/risk_register.txt",
        "project/reports/rollout.txt",
        "project/reports/rollout.txt",
        "project/reports/manifest.txt",
        "project/reports/checkpoint.txt",
        "project/validation/risk_register.txt",
    ]
    assert "project/reports/rollout.txt" in task.expected_files
    assert task.max_steps == 13


def test_curriculum_generates_long_horizon_repository_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="service_release_task",
            prompt="Prepare service release repo slice.",
            workspace="workspace/service_release_task",
            success=True,
            task_metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=True,
        task_metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "service_release_task_repository_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "repository"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["difficulty"] == "long_horizon"
    assert task.metadata["long_horizon_step_count"] == 9
    assert task.metadata["long_horizon_coding_surface"] == "repository_release_bundle"
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "repo/release/notes.md",
        "repo/release/notes.md",
        "repo/release/notes.md",
        "repo/release/plan.txt",
        "repo/reports/checklist.txt",
        "repo/reports/checklist.txt",
        "repo/reports/packet.txt",
        "repo/reports/audit.txt",
        "repo/reports/checklist.txt",
    ]
    assert task.max_steps == 12


def test_curriculum_generates_repository_validation_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="repo_sync_matrix_task",
        prompt="Prepare repository validation sync matrix.",
        workspace="workspace/repo_sync_matrix_task",
        success=True,
        task_metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "repo_sync_matrix_task_repository_adjacent"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["long_horizon_coding_surface"] == "repository_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "repository_validation_bundle"
    assert task.metadata["long_horizon_step_count"] == 10
    assert [step["path"] for step in task.metadata["synthetic_edit_plan"]] == [
        "repo/validation/summary.md",
        "repo/validation/summary.md",
        "repo/validation/summary.md",
        "repo/validation/matrix.txt",
        "repo/validation/matrix.txt",
        "repo/reports/packet.txt",
        "repo/reports/packet.txt",
        "repo/reports/audit.txt",
        "repo/reports/checkpoint.txt",
        "repo/validation/matrix.txt",
    ]
    assert "repo/reports/packet.txt" in task.expected_files
    assert task.max_steps == 13


def test_curriculum_generates_repository_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="service_release_task",
            prompt="Prepare service release repo slice.",
            workspace="workspace/service_release_task",
            success=True,
            task_metadata={"benchmark_family": "repository"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=True,
        task_metadata={"benchmark_family": "repository"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "service_release_task_repository_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "repository"
    assert "repo/check.txt" in task.expected_files
    assert task.max_steps >= 14


def test_curriculum_generates_tool_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="api_contract_task",
            prompt="Prepare API contract bundle.",
            workspace="workspace/api_contract_task",
            success=True,
            task_metadata={"benchmark_family": "tooling"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="api_contract_task",
        prompt="Prepare API contract bundle.",
        workspace="workspace/api_contract_task",
        success=True,
        task_metadata={"benchmark_family": "tooling"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "api_contract_task_tool_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "tooling"
    assert "tool/check.txt" in task.expected_files
    assert task.max_steps >= 14


def test_curriculum_generates_integration_adjacent_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="service_mesh_task",
            prompt="Prepare integration mesh.",
            workspace="workspace/service_mesh_task",
            success=True,
            task_metadata={"benchmark_family": "integration"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="service_mesh_task",
        prompt="Prepare integration mesh.",
        workspace="workspace/service_mesh_task",
        success=True,
        task_metadata={"benchmark_family": "integration"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "service_mesh_task_integration_adjacent"
    assert task.metadata["benchmark_family"] == "integration"
    assert "integration/check.txt" in task.expected_files
    assert task.max_steps >= 16


def test_curriculum_generates_workflow_validation_adjacent_task(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="release_review_task",
        prompt="Prepare workflow validation review.",
        workspace="workspace/release_review_task",
        success=True,
        task_metadata={"benchmark_family": "workflow", "difficulty": "long_horizon"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "release_review_task_workflow_adjacent"
    assert task.metadata["curriculum_shape"] == "long_horizon_structured_edit"
    assert task.metadata["long_horizon_coding_surface"] == "workflow_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "workflow_validation_bundle"
    assert task.metadata["long_horizon_step_count"] == 10
    assert "workflow/reports/packet.txt" in task.expected_files
    assert task.max_steps == 13


def test_curriculum_generates_failure_recovery_task():
    episode = EpisodeRecord(
        task_id="broken_task",
        prompt="Create out.txt containing ok.",
        workspace="workspace/broken_task",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="run a broken command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: out.txt"],
                },
            )
        ],
    )

    task = CurriculumEngine().generate_followup_task(episode)

    assert task.task_id == "broken_task_file_recovery"
    assert task.metadata["curriculum_kind"] == "failure_recovery"
    assert "missing_expected_file" in task.metadata["failure_types"]
    assert task.metadata["failed_command"] == "false"


def test_curriculum_contract_clean_primary_failure_gets_deeper_recovery_budget():
    episode = EpisodeRecord(
        task_id="contract_clean_failure_task",
        prompt="Create recovery.txt containing fixed.",
        workspace="workspace/contract_clean_failure_task",
        success=False,
        termination_reason="repeated_failed_action",
        task_metadata={
            "benchmark_family": "repository",
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        steps=[
            StepRecord(
                index=1,
                thought="first failed command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: recovery.txt"],
                },
            ),
            StepRecord(
                index=2,
                thought="second failed command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: recovery.txt"],
                },
            ),
        ],
    )

    task = CurriculumEngine().generate_followup_task(episode)

    assert task.metadata["contract_clean_failure_recovery_origin"] is True
    assert task.metadata["contract_clean_failure_recovery_origin_task"] == "contract_clean_failure_task"
    assert task.metadata["contract_clean_failure_recovery_origin_family"] == "repository"
    assert task.metadata["contract_clean_failure_recovery_origin_contract_kind"] == "workspace_acceptance"
    assert task.metadata["failure_recovery_depth_profile"] == "contract_clean_primary"
    assert task.metadata["deep_failure_recovery_budget"] is True
    assert int(task.metadata["budget_step_floor"]) >= 12
    assert int(task.metadata["step_floor"]) == int(task.metadata["budget_step_floor"])
    assert int(task.metadata["contract_clean_failure_recovery_step_floor"]) == int(task.metadata["budget_step_floor"])
    assert task.max_steps >= int(task.metadata["budget_step_floor"])


def test_curriculum_strategy_biases_repository_recovery_over_file_recovery():
    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=False,
        task_metadata={"benchmark_family": "repository"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad repo command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: repo/recovery.txt"],
                },
            )
        ],
    )

    engine = CurriculumEngine()
    engine._curriculum_controls_cache = {"recovery_strategy_family": "rollback_validation"}
    task = engine.generate_followup_task(episode)

    assert task.task_id == "service_release_task_repository_recovery"
    assert task.metadata["recovery_strategy_family"] == "rollback_validation"


def test_curriculum_strategy_biases_snapshot_coverage_to_file_recovery():
    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=False,
        task_metadata={"benchmark_family": "repository"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad repo command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: repo/recovery.txt"],
                },
            )
        ],
    )

    engine = CurriculumEngine()
    engine._curriculum_controls_cache = {"recovery_strategy_family": "snapshot_coverage"}
    task = engine.generate_followup_task(episode)

    assert task.task_id == "service_release_task_file_recovery"
    assert task.metadata["recovery_strategy_family"] == "snapshot_coverage"


def test_curriculum_uses_memory_backed_reference_commands(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="seed_success_task",
            prompt="Create seeded.txt containing seeded.",
            workspace="workspace/seed_success_task",
            success=True,
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="use known good command",
                    action="code_execute",
                    content="printf 'seeded\\n' > seeded.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
        )
    )

    episode = EpisodeRecord(
        task_id="seed_success_task_variant",
        prompt="Create followup output.",
        workspace="workspace/seed_success_task_variant",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.metadata["reference_task_ids"] == ["seed_success_task"]
    assert "printf 'seeded\\n' > seeded.txt" in task.metadata["reference_commands"]
    assert "printf 'seeded\\n' > seeded.txt" not in task.suggested_commands


def test_adjacent_success_uses_parent_episode_fast_path_without_memory_scan(tmp_path: Path, monkeypatch):
    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=True,
        task_metadata={"benchmark_family": "repository"},
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="use known good repository handoff command",
                action="code_execute",
                content="mkdir -p repo && printf 'repo verified\\n' > repo/status.txt",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": True, "reasons": ["verification passed"]},
            )
        ],
    )

    def explode_memory_scan(self):
        raise AssertionError("adjacent-success fast path should not scan episode memory")

    monkeypatch.setattr(CurriculumEngine, "_memory_documents", explode_memory_scan)

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.metadata["reference_task_ids"] == ["service_release_task"]
    assert task.metadata["benchmark_family"] == "repository"
    assert "mkdir -p repo && printf 'repo verified\\n' > repo/status.txt" in task.metadata["reference_commands"]
    assert task.task_id == "service_release_task_repository_adjacent"


def test_curriculum_records_retrieved_failure_patterns(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="prior_failure_task",
            prompt="Recover safely.",
            workspace="workspace/prior_failure_task",
            success=False,
            termination_reason="repeated_failed_action",
            steps=[
                StepRecord(
                    index=1,
                    thought="bad command",
                    action="code_execute",
                    content="false",
                    selected_skill_id=None,
                    command_result=None,
                    verification={
                        "passed": False,
                        "reasons": ["exit code was 1", "missing expected file: out.txt"],
                    },
                )
            ],
        )
    )

    episode = EpisodeRecord(
        task_id="broken_task",
        prompt="Create out.txt containing ok.",
        workspace="workspace/broken_task",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="run a broken command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: out.txt"],
                },
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert "missing_expected_file" in task.metadata["retrieved_failure_types"]
    assert "Avoid" in task.prompt


def test_curriculum_retrieves_more_than_three_semantic_failure_references(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    for index in range(5):
        memory.save(
            EpisodeRecord(
                task_id=f"repo_regression_reference_{index}",
                prompt="Recover repository state regression by recreating out.txt.",
                workspace=f"workspace/repo_regression_reference_{index}",
                success=False,
                task_metadata={
                    "benchmark_family": "repository",
                    "repo_semantics": ["stateful_repo", "validation"],
                    "lineage_families": ["repository"],
                },
                termination_reason="no_state_progress",
                steps=[
                    StepRecord(
                        index=1,
                        thought="stalled command",
                        action="code_execute",
                        content="false",
                        selected_skill_id=None,
                        command_result=None,
                        verification={
                            "passed": False,
                            "reasons": ["missing expected file: out.txt", "no state progress detected"],
                        },
                        failure_signals=["no_state_progress", "state_regression"],
                    )
                ],
            )
        )

    episode = EpisodeRecord(
        task_id="repo_regression_target",
        prompt="Recover repository state regression and create out.txt.",
        workspace="workspace/repo_regression_target",
        success=False,
        task_metadata={
            "benchmark_family": "repository",
            "repo_semantics": ["stateful_repo", "validation"],
            "lineage_families": ["repository"],
        },
        termination_reason="no_state_progress",
        steps=[
            StepRecord(
                index=1,
                thought="bad command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["missing expected file: out.txt", "no state progress detected"],
                },
                failure_signals=["no_state_progress", "state_regression"],
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.metadata["context_reference_limit"] == 6
    assert len(task.metadata["reference_task_ids"]) == 5
    assert all(task_id.startswith("repo_regression_reference_") for task_id in task.metadata["reference_task_ids"])
    assert task.metadata["semantic_context_matches"]


def test_curriculum_records_retrieved_transition_failures(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="prior_transition_task",
            prompt="Recover from stalled state.",
            workspace="workspace/prior_transition_task",
            success=False,
            termination_reason="no_state_progress",
            steps=[
                StepRecord(
                    index=1,
                    thought="irrelevant edit",
                    action="code_execute",
                    content="printf 'noise\\n' >> scratch.log",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": False, "reasons": ["no state progress detected"]},
                    failure_signals=["no_state_progress"],
                )
            ],
        )
    )

    episode = EpisodeRecord(
        task_id="broken_transition_task",
        prompt="Create out.txt containing ok.",
        workspace="workspace/broken_transition_task",
        success=False,
        termination_reason="no_state_progress",
        steps=[
            StepRecord(
                index=1,
                thought="stalled command",
                action="code_execute",
                content="printf 'noise\\n' >> scratch.log",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["no state progress detected"]},
                failure_signals=["no_state_progress"],
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert "no_state_progress" in task.metadata["retrieved_transition_failures"]
    assert "Avoid no_state_progress" in task.prompt


class ForcedFailurePolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="force failure",
            action="code_execute",
            content="false",
            done=False,
        )


def test_failure_path_selfplay_generates_recovery_followup(tmp_path: Path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )
    failing_kernel = AgentKernel(config=config, policy=ForcedFailurePolicy())
    normal_kernel = AgentKernel(config=config)

    seed = failing_kernel.run_task(TaskBank().get("hello_task"))
    followup = CurriculumEngine(memory_root=config.trajectories_root).generate_followup_task(seed)
    result = normal_kernel.run_task(followup)

    assert seed.success is False
    assert followup.metadata["curriculum_kind"] == "failure_recovery"
    assert followup.metadata["parent_task"] == "hello_task"
    assert result.success is True


def test_curriculum_generates_workspace_path_recovery_task():
    episode = EpisodeRecord(
        task_id="rewrite_task",
        prompt="Overwrite note.txt so it contains only the string done.",
        workspace="/tmp/agentkernel_verify/workspace/rewrite_task",
        success=False,
        termination_reason="policy_terminated",
        steps=[
            StepRecord(
                index=1,
                thought="write into nested workspace path",
                action="code_execute",
                content="printf 'done\\n' > rewrite_task/note.txt",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 2", "missing expected file: note.txt"],
                },
            )
        ],
    )

    task = CurriculumEngine().generate_followup_task(episode)

    assert task.task_id == "rewrite_task_path_recovery"
    assert task.metadata["curriculum_kind"] == "failure_recovery"
    assert task.metadata["failure_pattern"] == "workspace_prefixed_path"
    assert "rewrite_task/resolved.txt" in task.forbidden_files


def test_curriculum_generates_command_avoidance_recovery_task():
    episode = EpisodeRecord(
        task_id="broken_retry_task",
        prompt="Recover from repeated false command.",
        workspace="/tmp/agentkernel_verify/workspace/broken_retry_task",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1"],
                },
            )
        ],
    )

    task = CurriculumEngine().generate_followup_task(episode)

    assert task.task_id == "broken_retry_task_avoidance_recovery"
    assert task.metadata["curriculum_kind"] == "failure_recovery"
    assert task.metadata["failure_pattern"] == "generic_recovery"
    assert "false" in task.prompt


def test_curriculum_generates_workflow_recovery_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="release_bundle_task",
            prompt="Prepare release bundle.",
            workspace="workspace/release_bundle_task",
            success=True,
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="release_bundle_task",
        prompt="Prepare release bundle.",
        workspace="workspace/release_bundle_task",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad workflow command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.task_id == "release_bundle_task_workflow_recovery"
    assert task.metadata["benchmark_family"] == "workflow"
    assert "recovery/summary.txt" in task.expected_files


def test_curriculum_generates_project_recovery_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="deployment_manifest_task",
            prompt="Prepare deployment workspace.",
            workspace="workspace/deployment_manifest_task",
            success=True,
            task_metadata={"benchmark_family": "project"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=False,
        task_metadata={"benchmark_family": "project"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.task_id == "deployment_manifest_task_project_recovery"
    assert task.metadata["benchmark_family"] == "project"
    assert task.metadata["source_task"] == "deployment_manifest_task"
    assert "project/recovery.txt" in task.expected_files
    assert task.suggested_commands[0].startswith("mkdir -p project")


def test_curriculum_generates_repository_recovery_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="service_release_task",
            prompt="Prepare service release repo slice.",
            workspace="workspace/service_release_task",
            success=True,
            task_metadata={"benchmark_family": "repository"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="service_release_task",
        prompt="Prepare service release repo slice.",
        workspace="workspace/service_release_task",
        success=False,
        task_metadata={"benchmark_family": "repository"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad repo command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.task_id == "service_release_task_repository_recovery"
    assert task.metadata["benchmark_family"] == "repository"
    assert "repo/recovery.txt" in task.expected_files


def test_curriculum_generates_tool_recovery_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="api_contract_task",
            prompt="Prepare API contract bundle.",
            workspace="workspace/api_contract_task",
            success=True,
            task_metadata={"benchmark_family": "tooling"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="api_contract_task",
        prompt="Prepare API contract bundle.",
        workspace="workspace/api_contract_task",
        success=False,
        task_metadata={"benchmark_family": "tooling"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad tool command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.task_id == "api_contract_task_tool_recovery"
    assert task.metadata["benchmark_family"] == "tooling"


def test_curriculum_failure_recovery_filters_unrelated_retrieved_commands(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="hello_task",
            prompt="Create hello.txt containing hello agent kernel.",
            workspace="workspace/hello_task",
            success=True,
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="hello command",
                    action="code_execute",
                    content="printf 'hello agent kernel\\n' > hello.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
        )
    )
    memory.save(
        EpisodeRecord(
            task_id="deployment_manifest_task",
            prompt="Prepare deployment workspace.",
            workspace="workspace/deployment_manifest_task",
            success=True,
            task_metadata={"benchmark_family": "project"},
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="project command",
                    action="code_execute",
                    content="mkdir -p project && printf 'deployment recovered\\n' > project/recovery.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
        )
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=False,
        task_metadata={"benchmark_family": "project"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert task.suggested_commands[0].startswith("mkdir -p project")
    assert all("hello.txt" not in command for command in task.suggested_commands)


def test_curriculum_failure_recovery_does_not_reuse_failed_document_commands(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="deployment_manifest_task_failed_prior",
            prompt="Prepare deployment workspace.",
            workspace="workspace/deployment_manifest_task_failed_prior",
            success=False,
            task_metadata={"benchmark_family": "project"},
            termination_reason="repeated_failed_action",
            steps=[
                StepRecord(
                    index=1,
                    thought="bad project command",
                    action="code_execute",
                    content="mkdir -p project && printf 'broken\\n' > project/recovery.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": False, "reasons": ["exit code was 1"]},
                )
            ],
        )
    )
    memory.save(
        EpisodeRecord(
            task_id="deployment_manifest_task_success_prior",
            prompt="Prepare deployment workspace.",
            workspace="workspace/deployment_manifest_task_success_prior",
            success=True,
            task_metadata={"benchmark_family": "project"},
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="good project command",
                    action="code_execute",
                    content="mkdir -p project && printf 'deployment recovered\\n' > project/recovery.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
        )
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=False,
        task_metadata={"benchmark_family": "project"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_followup_task(episode)

    assert "mkdir -p project && printf 'deployment recovered\\n' > project/recovery.txt" in task.metadata["reference_commands"]
    assert "mkdir -p project && printf 'broken\\n' > project/recovery.txt" not in task.metadata["reference_commands"]


def test_curriculum_generates_integration_recovery_task(tmp_path: Path):
    memory = EpisodeMemory(tmp_path / "episodes")
    memory.save(
        EpisodeRecord(
            task_id="incident_matrix_task",
            prompt="Prepare incident integration bundle.",
            workspace="workspace/incident_matrix_task",
            success=True,
            task_metadata={"benchmark_family": "integration"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="incident_matrix_task",
        prompt="Prepare incident integration bundle.",
        workspace="workspace/incident_matrix_task",
        success=False,
        task_metadata={"benchmark_family": "integration"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad integration command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_failure_driven_task(episode)

    assert task.task_id == "incident_matrix_task_integration_recovery"
    assert task.metadata["benchmark_family"] == "integration"
    assert "integration/check.txt" in task.expected_files
    assert "integration/recovery.txt" in task.expected_files


def test_curriculum_applies_retained_behavior_controls(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    (episodes_root / "deployment_manifest_task_reference.json").write_text(
        json.dumps(
            {
                "task_id": "deployment_manifest_task_reference",
                "prompt": "Prepare deployment workspace.",
                "workspace": "workspace/deployment_manifest_task_reference",
                "success": True,
                "task_metadata": {"benchmark_family": "project"},
                "termination_reason": "success",
                "summary": {"failure_types": []},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "mkdir -p project && printf 'deployment recovered\\n' > project/recovery.txt && printf 'verified\\n' > project/status.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    (episodes_root / "other_family_reference.json").write_text(
        json.dumps(
            {
                "task_id": "other_family_reference",
                "prompt": "Prepare workflow workspace.",
                "workspace": "workspace/other_family_reference",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "termination_reason": "success",
                "summary": {"failure_types": []},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "mkdir -p recovery && printf 'workflow recovered\\n' > recovery/status.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    curriculum_path = tmp_path / "curriculum" / "curriculum_proposals.json"
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.write_text(
        json.dumps(
            {
                "artifact_kind": "curriculum_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "failure_reference_family_only": True,
                    "success_reference_limit": 1,
                    "failure_recovery_anchor_min_matches": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=False,
        task_metadata={"benchmark_family": "project"},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )

    task = CurriculumEngine(
        memory_root=episodes_root,
        config=KernelConfig(curriculum_proposals_path=curriculum_path),
    ).generate_followup_task(episode)

    assert task.metadata["reference_task_ids"] == ["deployment_manifest_task_reference"]
    assert len(task.metadata["reference_commands"]) == 1
    assert task.metadata["curriculum_behavior_controls"]["failure_recovery_anchor_min_matches"] == 2
    assert all("project/" in command for command in task.suggested_commands if "mkdir -p" in command)


def test_generated_eval_tracks_generated_benchmark_family(tmp_path: Path):
    from evals.harness import run_eval

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    metrics = run_eval(config=config, include_generated=True, include_failure_generated=True)

    assert sum(metrics.generated_by_benchmark_family.values()) == metrics.generated_total
    assert sum(metrics.generated_passed_by_benchmark_family.values()) == metrics.generated_total
    assert "tooling" in metrics.generated_by_benchmark_family


def test_run_eval_applies_recovery_strategy_family_to_failure_curriculum(tmp_path: Path, monkeypatch):
    from evals import harness

    captured: dict[str, object] = {}
    original = harness.CurriculumEngine.generate_failure_driven_task

    def wrapped(self, episode):
        captured.update(dict(self._curriculum_controls() or {}))
        return original(self, episode)

    monkeypatch.setattr(harness.CurriculumEngine, "generate_failure_driven_task", wrapped)

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )

    harness.run_eval(
        config=config,
        include_failure_generated=True,
        task_limit=1,
        recovery_variant_strategy_family="rollback_validation",
    )

    assert captured["recovery_strategy_family"] == "rollback_validation"
    assert captured["preferred_benchmark_family"] == "repository"
    assert captured["failure_reference_family_only"] is True


def test_curriculum_controls_cap_failure_recovery_commands(tmp_path: Path):
    curriculum_path = tmp_path / "curriculum" / "curriculum_proposals.json"
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.write_text(
        json.dumps(
            {
                "artifact_kind": "curriculum_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "failure_recovery_anchor_min_matches": 1,
                    "failure_recovery_command_cap": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    episode = EpisodeRecord(
        task_id="broken_task",
        prompt="Create out.txt containing ok.",
        workspace="workspace/broken_task",
        success=False,
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="run a broken command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: out.txt"],
                },
            )
        ],
    )

    task = CurriculumEngine(config=KernelConfig(curriculum_proposals_path=curriculum_path)).generate_followup_task(episode)

    assert len(task.suggested_commands) == 2
    assert task.metadata["curriculum_behavior_controls"]["failure_recovery_command_cap"] == 2


def test_curriculum_seed_scheduler_respects_limits_and_preferred_family(tmp_path: Path):
    curriculum_path = tmp_path / "curriculum" / "curriculum_proposals.json"
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.write_text(
        json.dumps(
            {
                "artifact_kind": "curriculum_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "preferred_benchmark_family": "project",
                    "max_generated_adjacent_tasks": 1,
                    "max_generated_failure_recovery_tasks": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    engine = CurriculumEngine(config=KernelConfig(curriculum_proposals_path=curriculum_path))
    success_episodes = [
        EpisodeRecord(
            task_id="workflow_success",
            prompt="workflow",
            workspace="workspace/workflow_success",
            success=True,
            task_metadata={"benchmark_family": "workflow"},
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="project_success",
            prompt="project",
            workspace="workspace/project_success",
            success=True,
            task_metadata={"benchmark_family": "project"},
            termination_reason="success",
            steps=[],
        ),
    ]
    failure_episodes = [
        EpisodeRecord(
            task_id="workflow_failure",
            prompt="workflow",
            workspace="workspace/workflow_failure",
            success=False,
            task_metadata={"benchmark_family": "workflow"},
            termination_reason="repeated_failed_action",
            steps=[
                StepRecord(
                    index=1,
                    thought="bad command",
                    action="code_execute",
                    content="false",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": False, "reasons": ["exit code was 1"]},
                )
            ],
        ),
        EpisodeRecord(
            task_id="project_failure",
            prompt="project",
            workspace="workspace/project_failure",
            success=False,
            task_metadata={"benchmark_family": "project"},
            termination_reason="repeated_failed_action",
            steps=[
                StepRecord(
                    index=1,
                    thought="bad command",
                    action="code_execute",
                    content="false",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": False, "reasons": ["exit code was 1"]},
                )
            ],
        ),
        EpisodeRecord(
            task_id="project_failure_two",
            prompt="project",
            workspace="workspace/project_failure_two",
            success=False,
            task_metadata={"benchmark_family": "project"},
            termination_reason="repeated_failed_action",
            steps=[
                StepRecord(
                    index=1,
                    thought="bad command",
                    action="code_execute",
                    content="false",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": False, "reasons": ["exit code was 1"]},
                )
            ],
        ),
    ]

    scheduled_success = engine.schedule_generated_seed_episodes(success_episodes, curriculum_kind="adjacent_success")
    scheduled_failure = engine.schedule_generated_seed_episodes(failure_episodes, curriculum_kind="failure_recovery")

    assert [episode.task_id for episode in scheduled_success] == ["project_success"]
    assert [episode.task_id for episode in scheduled_failure] == ["project_failure", "project_failure_two"]


def test_curriculum_seed_scheduler_prioritizes_coding_frontier_gaps(tmp_path: Path, monkeypatch):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 1,
        "frontier_priority_families": ["repository", "workflow", "project"],
        "frontier_missing_families": ["repository"],
        "frontier_retention_priority_families": [],
        "frontier_priority_family_bonus": 2,
        "frontier_missing_family_bonus": 4,
        "frontier_retention_priority_bonus": 0,
        "frontier_harder_task_bonus": 0,
        "frontier_min_lineage_depth": 0,
    }

    project_seed = EpisodeRecord(
        task_id="project_seed",
        prompt="project",
        workspace="workspace/project_seed",
        success=True,
        task_metadata={"benchmark_family": "project"},
        termination_reason="success",
        steps=[],
    )
    repository_seed = EpisodeRecord(
        task_id="repository_seed",
        prompt="repository",
        workspace="workspace/repository_seed",
        success=True,
        task_metadata={"benchmark_family": "repository"},
        termination_reason="success",
        steps=[],
    )

    monkeypatch.setattr(engine, "_select_adjacent_success_seed_set", lambda episodes, limit: episodes[:limit])

    scheduled = engine.schedule_generated_seed_episodes(
        [project_seed, repository_seed],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["repository_seed"]


def test_curriculum_seed_scheduler_prefers_light_supervision_candidate_primaries(tmp_path: Path, monkeypatch):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 1}

    retrieval_tail_seed = EpisodeRecord(
        task_id="a_retrieval_tail_seed",
        prompt="project",
        workspace="workspace/a_retrieval_tail_seed",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "light_supervision_candidate": False,
        },
        termination_reason="success",
        steps=[],
    )
    contract_clean_seed = EpisodeRecord(
        task_id="z_contract_clean_seed",
        prompt="project",
        workspace="workspace/z_contract_clean_seed",
        success=True,
        task_metadata={
            "benchmark_family": "project",
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        termination_reason="success",
        steps=[],
    )

    monkeypatch.setattr(engine, "_select_adjacent_success_seed_set", lambda episodes, limit: episodes[:limit])

    scheduled = engine.schedule_generated_seed_episodes(
        [retrieval_tail_seed, contract_clean_seed],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["z_contract_clean_seed"]


def test_curriculum_failure_seed_scheduler_prefers_light_supervision_candidate_primaries(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_failure_recovery_tasks": 1}

    retrieval_tail_failure = EpisodeRecord(
        task_id="a_retrieval_tail_failure",
        prompt="project retrieval tail",
        workspace="workspace/a_retrieval_tail_failure",
        success=False,
        task_metadata={
            "benchmark_family": "project",
            "light_supervision_candidate": False,
            "memory_source": "episode",
            "task_origin": "episode_replay",
            "requires_retrieval": True,
            "source_task": "seed_project_task",
        },
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad replayed command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
    )
    contract_clean_failure = EpisodeRecord(
        task_id="z_contract_clean_failure",
        prompt="project contract clean",
        workspace="workspace/z_contract_clean_failure",
        success=False,
        task_metadata={
            "benchmark_family": "project",
            "light_supervision_candidate": True,
            "light_supervision_contract_kind": "workspace_acceptance",
        },
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad direct command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: project/recovery.txt"],
                },
            )
        ],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [retrieval_tail_failure, contract_clean_failure],
        curriculum_kind="failure_recovery",
    )

    assert [episode.task_id for episode in scheduled] == ["z_contract_clean_failure"]


def test_curriculum_failure_seed_scheduler_diversifies_recovery_groups(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_failure_recovery_tasks": 2}

    project_file_failure = EpisodeRecord(
        task_id="a_project_file_failure",
        prompt="project missing file",
        workspace="workspace/a_project_file_failure",
        success=False,
        task_metadata={"benchmark_family": "project", "light_supervision_candidate": True},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: project/recovery.txt"],
                },
            )
        ],
    )
    repository_file_failure = EpisodeRecord(
        task_id="b_repository_file_failure",
        prompt="repository missing file",
        workspace="workspace/b_repository_file_failure",
        success=False,
        task_metadata={"benchmark_family": "repository", "light_supervision_candidate": True},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad repository command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: repo/recovery.txt"],
                },
            )
        ],
    )
    repeated_project_file_failure = EpisodeRecord(
        task_id="c_project_file_failure_two",
        prompt="project missing file again",
        workspace="workspace/c_project_file_failure_two",
        success=False,
        task_metadata={"benchmark_family": "project", "light_supervision_candidate": True},
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="bad project command again",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result=None,
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: project/recovery.txt"],
                },
            )
        ],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [project_file_failure, repository_file_failure, repeated_project_file_failure],
        curriculum_kind="failure_recovery",
    )

    assert [episode.task_id for episode in scheduled] == [
        "a_project_file_failure",
        "b_repository_file_failure",
    ]


def test_curriculum_failure_seed_scheduler_prioritizes_frontier_family_yield_targets(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_failure_recovery_tasks": 1,
        "frontier_priority_families": ["repository", "project"],
        "frontier_retention_priority_families": ["repository"],
    }

    project_failure = EpisodeRecord(
        task_id="project_failure",
        prompt="project failure",
        workspace="workspace/project_failure",
        success=False,
        task_metadata={"benchmark_family": "project", "light_supervision_candidate": True},
        termination_reason="repeated_failed_action",
        steps=[],
    )
    repository_failure = EpisodeRecord(
        task_id="repository_failure",
        prompt="repository failure",
        workspace="workspace/repository_failure",
        success=False,
        task_metadata={"benchmark_family": "repository", "light_supervision_candidate": True},
        termination_reason="repeated_failed_action",
        steps=[],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [project_failure, repository_failure],
        curriculum_kind="failure_recovery",
    )

    assert [episode.task_id for episode in scheduled] == ["repository_failure"]


def test_curriculum_seed_scheduler_compounds_retained_winner_and_avoids_risky_escalation(
    tmp_path: Path,
    monkeypatch,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 1,
        "frontier_priority_families": ["tooling", "repository"],
        "frontier_missing_families": [],
        "frontier_retention_priority_families": [],
        "frontier_generalization_priority_families": [],
        "frontier_retained_gain_families": ["tooling"],
        "frontier_promotion_risk_families": ["repository"],
        "frontier_retained_family_delta": {"tooling": 0.12},
        "frontier_promotion_risk_family_delta": {"repository": 0.11},
        "frontier_priority_family_bonus": 1,
        "frontier_missing_family_bonus": 0,
        "frontier_retention_priority_bonus": 0,
        "frontier_generalization_bonus": 0,
        "frontier_outward_branch_bonus": 0,
        "frontier_lineage_breadth_bonus": 0,
        "frontier_failure_motif_bonus": 0,
        "frontier_repo_setting_bonus": 0,
        "frontier_harder_task_bonus": 1,
        "frontier_retained_gain_bonus": 2,
        "frontier_promotion_risk_penalty": 2,
        "frontier_min_lineage_depth": 0,
    }

    retained_winner = EpisodeRecord(
        task_id="retained_winner",
        prompt="tooling",
        workspace="workspace/retained_winner",
        success=True,
        task_metadata={"benchmark_family": "tooling", "difficulty": "long_horizon", "long_horizon_step_count": 8},
        termination_reason="success",
        steps=[],
    )
    risky_family = EpisodeRecord(
        task_id="risky_family",
        prompt="repository",
        workspace="workspace/risky_family",
        success=True,
        task_metadata={"benchmark_family": "repository", "difficulty": "long_horizon", "long_horizon_step_count": 8},
        termination_reason="success",
        steps=[],
    )

    monkeypatch.setattr(engine, "_select_adjacent_success_seed_set", lambda episodes, limit: episodes[:limit])
    monkeypatch.setattr(engine, "_adjacent_success_target_benchmark_family", lambda episode: episode.task_metadata["benchmark_family"])

    scheduled = engine.schedule_generated_seed_episodes(
        [risky_family, retained_winner],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["retained_winner"]


def test_curriculum_seed_scheduler_prioritizes_generalization_branching(tmp_path: Path, monkeypatch):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 1,
        "frontier_priority_families": ["workflow", "project"],
        "frontier_missing_families": [],
        "frontier_retention_priority_families": [],
        "frontier_generalization_priority_families": ["workflow"],
        "frontier_priority_family_bonus": 1,
        "frontier_missing_family_bonus": 0,
        "frontier_retention_priority_bonus": 0,
        "frontier_generalization_bonus": 4,
        "frontier_outward_branch_bonus": 2,
        "frontier_lineage_breadth_bonus": 1,
        "frontier_harder_task_bonus": 0,
        "frontier_min_lineage_depth": 0,
    }

    project_seed = EpisodeRecord(
        task_id="project_seed",
        prompt="project",
        workspace="workspace/project_seed",
        success=True,
        task_metadata={"benchmark_family": "project"},
        termination_reason="success",
        steps=[],
    )
    repository_to_workflow_seed = EpisodeRecord(
        task_id="repository_seed",
        prompt="repository",
        workspace="workspace/repository_seed",
        success=True,
        task_metadata={
            "benchmark_family": "repository",
            "difficulty": "long_horizon",
            "origin_benchmark_family": "project",
            "lineage_families": ["repo_sandbox", "repository"],
        },
        termination_reason="success",
        steps=[],
    )

    monkeypatch.setattr(engine, "_select_adjacent_success_seed_set", lambda episodes, limit: episodes[:limit])

    scheduled = engine.schedule_generated_seed_episodes(
        [project_seed, repository_to_workflow_seed],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["repository_seed"]


def test_curriculum_seed_scheduler_prioritizes_repo_setting_signature_pressure(tmp_path: Path, monkeypatch):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 1,
        "frontier_priority_families": ["workflow"],
        "frontier_missing_families": [],
        "frontier_retention_priority_families": [],
        "frontier_generalization_priority_families": [],
        "frontier_repo_setting_priority_pairs": ["workflow:worker_handoff"],
        "frontier_priority_family_bonus": 1,
        "frontier_missing_family_bonus": 0,
        "frontier_retention_priority_bonus": 0,
        "frontier_generalization_bonus": 0,
        "frontier_repo_setting_bonus": 4,
        "frontier_harder_task_bonus": 0,
        "frontier_min_lineage_depth": 0,
    }

    generic_repository_seed = EpisodeRecord(
        task_id="generic_repository_seed",
        prompt="repository",
        workspace="workspace/generic_repository_seed",
        success=True,
        task_metadata={
            "benchmark_family": "repository",
            "difficulty": "long_horizon",
            "origin_benchmark_family": "project",
            "lineage_families": ["repo_sandbox", "repository"],
            "lineage_surfaces": ["repository_release_bundle"],
        },
        termination_reason="success",
        steps=[],
    )
    worker_repository_seed = EpisodeRecord(
        task_id="worker_repository_seed",
        prompt="repository worker",
        workspace="workspace/worker_repository_seed",
        success=True,
        task_metadata={
            "benchmark_family": "repository",
            "difficulty": "long_horizon",
            "origin_benchmark_family": "project",
            "lineage_families": ["repo_sandbox", "repository"],
            "lineage_surfaces": ["repository_worker_bundle"],
        },
        termination_reason="success",
        steps=[],
    )

    monkeypatch.setattr(engine, "_select_adjacent_success_seed_set", lambda episodes, limit: episodes[:limit])

    scheduled = engine.schedule_generated_seed_episodes(
        [generic_repository_seed, worker_repository_seed],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["worker_repository_seed"]


def test_curriculum_schedule_generated_seed_episodes_diversifies_long_horizon_groups(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 3}

    episodes = [
        EpisodeRecord(
            task_id="git_generated_conflict_resolution_task",
            prompt="integrator",
            workspace="workspace/git_generated_conflict_resolution_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "long_horizon_coding_surface": "shared_repo_integrator",
                "long_horizon_step_count": 10,
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_parallel_merge_acceptance_task",
            prompt="integrator 2",
            workspace="workspace/git_parallel_merge_acceptance_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "long_horizon_coding_surface": "shared_repo_integrator",
                "long_horizon_step_count": 10,
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="deployment_manifest_task",
            prompt="release",
            workspace="workspace/deployment_manifest_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "long_horizon_step_count": 9,
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="report_rollup_task",
            prompt="validation",
            workspace="workspace/report_rollup_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "long_horizon_step_count": 10,
            },
            termination_reason="success",
            steps=[],
        ),
    ]

    scheduled = engine.schedule_generated_seed_episodes(episodes, curriculum_kind="adjacent_success")

    assert [episode.task_id for episode in scheduled] == [
        "git_generated_conflict_resolution_task",
        "deployment_manifest_task",
        "report_rollup_task",
    ]
    assert [engine._adjacent_success_seed_group(episode) for episode in scheduled] == [
        "project:shared_repo_integrator_bundle::project_release:repo_sandbox_parallel_merge",
        "project:project_release_bundle::project_release",
        "project:project_validation_bundle::crosscheck",
    ]


def test_curriculum_schedule_generated_seed_episodes_uses_repository_group_for_shared_repo_origins(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 2}

    episodes = [
        EpisodeRecord(
            task_id="git_generated_conflict_resolution_task",
            prompt="integrator",
            workspace="workspace/git_generated_conflict_resolution_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "origin_benchmark_family": "repo_sandbox",
                "difficulty": "long_horizon",
                "long_horizon_coding_surface": "shared_repo_integrator",
                "long_horizon_step_count": 10,
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="deployment_manifest_task",
            prompt="release",
            workspace="workspace/deployment_manifest_task",
            success=True,
            task_metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "long_horizon_step_count": 9,
            },
            termination_reason="success",
            steps=[],
        ),
    ]

    scheduled = engine.schedule_generated_seed_episodes(episodes, curriculum_kind="adjacent_success")

    assert [engine._adjacent_success_seed_group(episode) for episode in scheduled] == [
        "repository:repository_integrator_bundle:repo_sandbox:project_release",
        "project:project_release_bundle::project_release",
    ]


def test_curriculum_retargets_workflow_adjacent_to_tooling_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
        prompt="Advance workflow audit bundle.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "workflow",
            "origin_benchmark_family": "repository",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "workflow_release_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repository"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_tooling_adjacent")
    assert task.metadata["benchmark_family"] == "tooling"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "tooling_release_bundle"
    assert task.metadata["long_horizon_variant"] == "tooling_release_bundle"
    assert "tooling/release/packet.txt" in task.expected_files


def test_curriculum_retargets_workflow_review_adjacent_to_tooling_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="repo_status_review_task_repository_adjacent_workflow_adjacent",
        prompt="Advance workflow validation bundle.",
        workspace="workspace/repo_status_review_task_repository_adjacent_workflow_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "workflow",
            "origin_benchmark_family": "repository",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "workflow_validation_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repository"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_tooling_adjacent")
    assert task.metadata["benchmark_family"] == "tooling"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "tooling_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "tooling_validation_bundle"
    assert "tooling/validation/matrix.txt" in task.expected_files


def test_curriculum_retargets_tooling_adjacent_to_integration_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling release bundle.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_release_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_release_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_release_bundle"
    assert "integration/handoff/packet.txt" in task.expected_files


def test_curriculum_retargets_tooling_contract_adjacent_to_integration_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="api_contract_review_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling validation bundle.",
        workspace="workspace/api_contract_review_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_validation_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_validation_bundle"
    assert "integration/validation/matrix.txt" in task.expected_files


def test_curriculum_retargets_workflow_release_with_repo_lineage_to_tooling_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="repo_status_review_task_repository_adjacent_workflow_adjacent",
        prompt="Advance workflow release bundle with repository validation lineage.",
        workspace="workspace/repo_status_review_task_repository_adjacent_workflow_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "workflow",
            "origin_benchmark_family": "repository",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "workflow_release_bundle",
            "lineage_surfaces": [
                "repository_validation_bundle",
                "workflow_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repository_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "repository"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_tooling_adjacent")
    assert task.metadata["benchmark_family"] == "tooling"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "tooling_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "tooling_validation_bundle"
    assert "tooling/validation/matrix.txt" in task.expected_files


def test_curriculum_retargets_workflow_release_with_worker_lineage_to_tooling_worker_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent",
        prompt="Advance workflow release bundle with worker lineage.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "workflow",
            "origin_benchmark_family": "repository",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "workflow_release_bundle",
            "lineage_surfaces": [
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "repository"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_tooling_adjacent")
    assert task.metadata["benchmark_family"] == "tooling"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "tooling_worker_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "tooling_worker_validation_bundle"
    assert "tooling/worker/claim.txt" in task.expected_files


def test_curriculum_retargets_workflow_release_with_integrator_lineage_to_tooling_integrator_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent",
        prompt="Advance workflow release bundle with integrator lineage.",
        workspace="workspace/git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "workflow",
            "origin_benchmark_family": "repository",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "workflow_release_bundle",
            "lineage_surfaces": [
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "repository"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_tooling_adjacent")
    assert task.metadata["benchmark_family"] == "tooling"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "tooling_integrator_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "tooling_integrator_validation_bundle"
    assert "tooling/integrator/merge_plan.txt" in task.expected_files


def test_curriculum_retargets_integration_adjacent_to_repo_chore_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration release bundle.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_release_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_cleanup_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_cleanup_bundle"
    assert "upkeep/cleanup/packet.txt" in task.expected_files


def test_curriculum_retargets_tooling_release_with_repo_lineage_to_integration_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling release bundle with repo-sandbox lineage.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_release_bundle",
            "lineage_surfaces": [
                "shared_repo_synthetic_worker",
                "repository_validation_bundle",
                "workflow_release_bundle",
                "tooling_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_validation_bundle"
    assert "integration/validation/matrix.txt" in task.expected_files


def test_curriculum_retargets_tooling_release_with_worker_lineage_to_integration_worker_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling release bundle with worker lineage.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_release_bundle",
            "lineage_surfaces": [
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_worker_validation_bundle",
                "tooling_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_worker_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_worker_validation_bundle"
    assert "integration/worker/checklist.txt" in task.expected_files


def test_curriculum_retargets_tooling_release_with_integrator_lineage_to_integration_integrator_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling release bundle with integrator lineage.",
        workspace="workspace/git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_release_bundle",
            "lineage_surfaces": [
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_integrator_validation_bundle",
                "tooling_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_integrator_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_integrator_validation_bundle"
    assert "integration/integrator/checklist.txt" in task.expected_files


def test_curriculum_retargets_tooling_worker_validation_adjacent_to_integration_worker_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling worker validation bundle.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_worker_validation_bundle",
            "lineage_surfaces": [
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_worker_validation_bundle",
                "tooling_worker_validation_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_worker_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_worker_validation_bundle"
    assert "integration/worker/checklist.txt" in task.expected_files


def test_curriculum_retargets_tooling_integrator_validation_adjacent_to_integration_integrator_validation_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        prompt="Advance tooling integrator validation bundle.",
        workspace="workspace/git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "tooling",
            "origin_benchmark_family": "workflow",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "tooling_integrator_validation_bundle",
            "lineage_surfaces": [
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_integrator_validation_bundle",
                "tooling_integrator_validation_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "workflow"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_integration_adjacent")
    assert task.metadata["benchmark_family"] == "integration"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "integration_integrator_validation_bundle"
    assert task.metadata["long_horizon_variant"] == "integration_integrator_validation_bundle"
    assert "integration/integrator/checklist.txt" in task.expected_files


def test_curriculum_retargets_integration_release_with_worker_lineage_to_repo_chore_cleanup_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration release bundle with worker lineage.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_release_bundle",
            "lineage_surfaces": [
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_release_bundle",
                "tooling_release_bundle",
                "integration_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_cleanup_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_cleanup_bundle"
    assert task.metadata["lineage_branch_kind"] == "cleanup"
    assert task.metadata["lineage_branch_kinds"][-1] == "cleanup"
    assert "upkeep/cleanup/packet.txt" in task.expected_files


def test_curriculum_retargets_integration_worker_validation_adjacent_to_repo_chore_cleanup_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration worker validation bundle.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_worker_validation_bundle",
            "lineage_surfaces": [
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_worker_validation_bundle",
                "tooling_worker_validation_bundle",
                "integration_worker_validation_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_cleanup_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_cleanup_bundle"
    assert task.metadata["lineage_branch_kind"] == "cleanup"
    assert task.metadata["lineage_branch_kinds"][-1] == "cleanup"
    assert "upkeep/cleanup/packet.txt" in task.expected_files


def test_curriculum_retargets_integration_release_with_integrator_lineage_to_repo_chore_audit_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration release bundle with integrator lineage.",
        workspace="workspace/git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_release_bundle",
            "lineage_surfaces": [
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_release_bundle",
                "tooling_release_bundle",
                "integration_release_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_audit_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_audit_bundle"
    assert task.metadata["lineage_branch_kind"] == "audit"
    assert task.metadata["lineage_branch_kinds"][-1] == "audit"
    assert "upkeep/audit/packet.txt" in task.expected_files


def test_curriculum_retargets_integration_integrator_validation_adjacent_to_repo_chore_audit_family(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration integrator validation bundle.",
        workspace="workspace/git_parallel_merge_packet_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_integrator_validation_bundle",
            "lineage_surfaces": [
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_integrator_validation_bundle",
                "tooling_integrator_validation_bundle",
                "integration_integrator_validation_bundle",
            ],
            "lineage_branch_kinds": [
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
                "repo_sandbox_release",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_audit_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_audit_bundle"
    assert task.metadata["lineage_branch_kind"] == "audit"
    assert task.metadata["lineage_branch_kinds"][-1] == "audit"
    assert "upkeep/audit/packet.txt" in task.expected_files


def test_curriculum_retargets_merge_acceptance_integration_adjacent_to_repo_chore_audit_bundle(
    tmp_path: Path,
):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        prompt="Advance integration release bundle.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "integration",
            "origin_benchmark_family": "tooling",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "integration_release_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "tooling"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_repo_chore_adjacent")
    assert task.metadata["benchmark_family"] == "repo_chore"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "repo_chore_audit_bundle"
    assert task.metadata["long_horizon_variant"] == "repo_chore_audit_bundle"
    assert "upkeep/audit/packet.txt" in task.expected_files


def test_curriculum_retargets_repo_chore_cleanup_adjacent_to_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore cleanup bundle.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
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


def test_curriculum_preserves_integrator_cleanup_lineage_into_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore cleanup bundle.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "repo_chore",
            "origin_benchmark_family": "integration",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "repo_chore_cleanup_bundle",
            "lineage_surfaces": [
                "project_release_bundle",
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_release_bundle",
                "tooling_release_bundle",
                "integration_release_bundle",
                "repo_chore_cleanup_bundle",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "cleanup",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "integration"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "validation_integrator_cleanup_gate_bundle"
    assert task.metadata["long_horizon_variant"] == "validation_integrator_cleanup_gate_bundle"
    assert "mirrors/repo_alpha/conflict_status.txt" in task.expected_files
    assert "validation/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_retargets_repo_chore_audit_adjacent_to_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore audit bundle.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
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


def test_curriculum_preserves_integrator_audit_lineage_into_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore audit bundle.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "repo_chore",
            "origin_benchmark_family": "integration",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "repo_chore_audit_bundle",
            "lineage_surfaces": [
                "project_release_bundle",
                "shared_repo_integrator_bundle",
                "repository_integrator_bundle",
                "workflow_release_bundle",
                "tooling_release_bundle",
                "integration_release_bundle",
                "repo_chore_audit_bundle",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "audit",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "integration"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "validation_integrator_audit_gate_bundle"
    assert task.metadata["long_horizon_variant"] == "validation_integrator_audit_gate_bundle"
    assert "mirrors/repo_alpha/acceptance_status.txt" in task.expected_files
    assert "validation/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_preserves_worker_lineage_prompt_into_validation_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        prompt="Advance repo chore cleanup bundle.",
        workspace="workspace/git_parallel_worker_api_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "repo_chore",
            "origin_benchmark_family": "integration",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "repo_chore_cleanup_bundle",
            "lineage_surfaces": [
                "project_release_bundle",
                "shared_repo_worker_bundle",
                "repository_worker_bundle",
                "workflow_worker_validation_bundle",
                "tooling_worker_validation_bundle",
                "integration_worker_validation_bundle",
                "repo_chore_cleanup_bundle",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "cleanup",
            ],
        },
        task_contract={"metadata": {"origin_benchmark_family": "integration"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert "Continue the worker handoff lineage:" in task.prompt


def test_curriculum_retargets_validation_cleanup_adjacent_to_governance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        prompt="Advance validation cleanup gate.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "validation",
            "origin_benchmark_family": "repo_chore",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "validation_cleanup_gate_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repo_chore"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "governance_cleanup_review_bundle"
    assert task.metadata["long_horizon_variant"] == "governance_cleanup_review_bundle"
    assert "governance/cleanup/packet.txt" in task.expected_files


def test_curriculum_preserves_integrator_cleanup_lineage_into_governance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        prompt="Advance validation cleanup gate.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        success=True,
        task_metadata={
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
        task_contract={"metadata": {"origin_benchmark_family": "repo_chore"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "governance_integrator_cleanup_review_bundle"
    assert task.metadata["long_horizon_variant"] == "governance_integrator_cleanup_review_bundle"
    assert "mirrors/repo_alpha/board_status.txt" in task.expected_files
    assert "governance/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_retargets_validation_audit_adjacent_to_governance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        prompt="Advance validation audit gate.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "validation",
            "origin_benchmark_family": "repo_chore",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "validation_audit_gate_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "repo_chore"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "governance_audit_review_bundle"
    assert task.metadata["long_horizon_variant"] == "governance_audit_review_bundle"
    assert "governance/audit/packet.txt" in task.expected_files


def test_curriculum_preserves_integrator_audit_lineage_into_governance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        prompt="Advance validation audit gate.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent",
        success=True,
        task_metadata={
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
        task_contract={"metadata": {"origin_benchmark_family": "repo_chore"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "governance_integrator_audit_review_bundle"
    assert task.metadata["long_horizon_variant"] == "governance_integrator_audit_review_bundle"
    assert "mirrors/repo_alpha/board_status.txt" in task.expected_files
    assert "governance/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_retargets_governance_cleanup_adjacent_to_oversight_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        prompt="Advance governance cleanup review.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "governance",
            "origin_benchmark_family": "validation",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "governance_cleanup_review_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "validation"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_oversight_adjacent")
    assert task.metadata["benchmark_family"] == "oversight"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "oversight_cleanup_crosscheck_bundle"
    assert task.metadata["long_horizon_variant"] == "oversight_cleanup_crosscheck_bundle"
    assert "oversight/cleanup/packet.txt" in task.expected_files


def test_curriculum_preserves_integrator_cleanup_lineage_into_oversight_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        prompt="Advance governance cleanup review.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        success=True,
        task_metadata={
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
        task_contract={"metadata": {"origin_benchmark_family": "validation"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_oversight_adjacent")
    assert task.metadata["benchmark_family"] == "oversight"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "oversight_integrator_cleanup_crosscheck_bundle"
    assert task.metadata["long_horizon_variant"] == "oversight_integrator_cleanup_crosscheck_bundle"
    assert "mirrors/repo_alpha/crosscheck_status.txt" in task.expected_files
    assert "oversight/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_retargets_governance_audit_adjacent_to_oversight_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        prompt="Advance governance audit review.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "governance",
            "origin_benchmark_family": "validation",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "governance_audit_review_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "validation"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_oversight_adjacent")
    assert task.metadata["benchmark_family"] == "oversight"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "oversight_audit_crosscheck_bundle"
    assert task.metadata["long_horizon_variant"] == "oversight_audit_crosscheck_bundle"
    assert "oversight/audit/packet.txt" in task.expected_files


def test_curriculum_preserves_integrator_audit_lineage_into_oversight_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        prompt="Advance governance audit review.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent",
        success=True,
        task_metadata={
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
        task_contract={"metadata": {"origin_benchmark_family": "validation"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_oversight_adjacent")
    assert task.metadata["benchmark_family"] == "oversight"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "oversight_integrator_audit_crosscheck_bundle"
    assert task.metadata["long_horizon_variant"] == "oversight_integrator_audit_crosscheck_bundle"
    assert "mirrors/repo_alpha/crosscheck_status.txt" in task.expected_files
    assert "oversight/integrator/packet.txt" in task.expected_files
    assert "Continue the integrator handoff lineage:" in task.prompt


def test_curriculum_retargets_oversight_cleanup_adjacent_to_assurance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
        prompt="Advance oversight cleanup crosscheck.",
        workspace="workspace/git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "oversight",
            "origin_benchmark_family": "governance",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "governance"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_assurance_adjacent")
    assert task.metadata["benchmark_family"] == "assurance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "assurance_cleanup_cert_bundle"
    assert task.metadata["long_horizon_variant"] == "assurance_cleanup_cert_bundle"
    assert "assurance/cleanup/packet.txt" in task.expected_files


def test_curriculum_retargets_oversight_audit_adjacent_to_assurance_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
        prompt="Advance oversight audit crosscheck.",
        workspace="workspace/git_parallel_merge_acceptance_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent_validation_adjacent_governance_adjacent_oversight_adjacent",
        success=True,
        task_metadata={
            "benchmark_family": "oversight",
            "origin_benchmark_family": "governance",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "oversight_audit_crosscheck_bundle",
        },
        task_contract={"metadata": {"origin_benchmark_family": "governance"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_assurance_adjacent")
    assert task.metadata["benchmark_family"] == "assurance"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["long_horizon_coding_surface"] == "assurance_audit_cert_bundle"
    assert task.metadata["long_horizon_variant"] == "assurance_audit_cert_bundle"
    assert "assurance/audit/packet.txt" in task.expected_files


def test_curriculum_seed_group_includes_origin_and_lineage_branch_kind(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    cleanup = EpisodeRecord(
        task_id="cleanup_seed",
        prompt="cleanup",
        workspace="workspace/cleanup_seed",
        success=True,
        task_metadata={
            "benchmark_family": "oversight",
            "origin_benchmark_family": "governance",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
        },
        termination_reason="success",
        steps=[],
    )
    audit = EpisodeRecord(
        task_id="audit_seed",
        prompt="audit",
        workspace="workspace/audit_seed",
        success=True,
        task_metadata={
            "benchmark_family": "oversight",
            "origin_benchmark_family": "governance",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "oversight_audit_crosscheck_bundle",
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_group(cleanup) == (
        "assurance:assurance_cleanup_cert_bundle:governance:cleanup"
    )
    assert engine._adjacent_success_seed_group(audit) == (
        "assurance:assurance_audit_cert_bundle:governance:audit"
    )


def test_curriculum_uses_persisted_lineage_metadata_instead_of_task_id_patterns(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="generic_tail_seed",
        prompt="generic",
        workspace="workspace/generic_tail_seed",
        success=True,
        task_metadata={
            "benchmark_family": "oversight",
            "origin_benchmark_family": "governance",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "oversight_cleanup_crosscheck_bundle",
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
            ],
        },
        task_contract={
            "metadata": {
                "origin_benchmark_family": "governance",
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
                ],
            }
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.metadata["benchmark_family"] == "assurance"
    assert task.metadata["long_horizon_coding_surface"] == "assurance_cleanup_cert_bundle"
    assert task.metadata["lineage_branch_kind"] == "cleanup"


def test_curriculum_branches_assurance_to_adjudication_when_late_wave_signal_is_saturated(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="generic_assurance_tail_seed",
        prompt="generic assurance",
        workspace="workspace/generic_assurance_tail_seed",
        success=True,
        task_metadata={
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
        task_contract={"metadata": {"origin_benchmark_family": "oversight"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_adjudication_adjacent")
    assert task.metadata["benchmark_family"] == "adjudication"
    assert task.metadata["long_horizon_coding_surface"] == "adjudication_cleanup_ruling_bundle"
    assert task.metadata["lineage_depth"] == 11


def test_curriculum_keeps_assurance_when_late_wave_signal_is_not_saturated(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="generic_assurance_seed",
        prompt="generic assurance",
        workspace="workspace/generic_assurance_seed",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_families": ["oversight", "assurance"],
            "lineage_surfaces": ["oversight_cleanup_crosscheck_bundle", "assurance_cleanup_cert_bundle"],
            "lineage_branch_kinds": ["cleanup", "cleanup"],
        },
        task_contract={"metadata": {"origin_benchmark_family": "oversight"}},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_assurance_adjacent")
    assert task.metadata["benchmark_family"] == "assurance"
    assert task.metadata["long_horizon_coding_surface"] == "assurance_cleanup_cert_bundle"


def test_curriculum_rotates_adjudication_to_validation_after_saturated_late_wave(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="generic_adjudication_tail_seed",
        prompt="generic adjudication",
        workspace="workspace/generic_adjudication_tail_seed",
        success=True,
        task_metadata={
            "benchmark_family": "adjudication",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "adjudication_cleanup_ruling_bundle",
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
                "adjudication",
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
                "adjudication_cleanup_ruling_bundle",
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
                "cleanup",
            ],
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_validation_adjacent")
    assert task.metadata["benchmark_family"] == "validation"
    assert task.metadata["long_horizon_coding_surface"] == "validation_cleanup_gate_bundle"
    assert task.metadata["lineage_depth"] == 12


def test_curriculum_rotates_validation_to_governance_from_late_wave_cycle_without_origin_match(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="cycled_validation_tail_seed",
        prompt="cycled validation",
        workspace="workspace/cycled_validation_tail_seed",
        success=True,
        task_metadata={
            "benchmark_family": "validation",
            "origin_benchmark_family": "adjudication",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "validation_cleanup_gate_bundle",
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
                "adjudication",
                "validation",
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
                "adjudication_cleanup_ruling_bundle",
                "validation_cleanup_gate_bundle",
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
                "cleanup",
                "cleanup",
            ],
        },
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["long_horizon_coding_surface"] == "governance_cleanup_review_bundle"
    assert task.metadata["lineage_depth"] == 13


def test_curriculum_adapts_late_wave_rotation_to_undercovered_family(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="assurance_tail_with_governance_gap",
        prompt="adaptive assurance",
        workspace="workspace/assurance_tail_with_governance_gap",
        success=True,
        task_metadata={
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
                "oversight",
                "assurance",
                "adjudication",
                "validation",
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
                "oversight_cleanup_crosscheck_bundle",
                "assurance_cleanup_cert_bundle",
                "adjudication_cleanup_ruling_bundle",
                "validation_cleanup_gate_bundle",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        steps=[],
        termination_reason="success",
    )

    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    signal = engine._lineage_scheduler_signal(
        episode,
        source_family="assurance",
        target_family="governance",
    )
    task = engine.generate_adjacent_task(episode)

    assert signal["branch_outward"] is True
    assert signal["rotation_coverage_gap"] == 1
    assert signal["target_family_count"] == 0
    assert task.task_id.endswith("_governance_adjacent")
    assert task.metadata["benchmark_family"] == "governance"
    assert task.metadata["long_horizon_coding_surface"] == "governance_cleanup_review_bundle"


def test_curriculum_keeps_late_wave_family_when_rotation_coverage_is_balanced(tmp_path: Path):
    episode = EpisodeRecord(
        task_id="balanced_assurance_tail",
        prompt="balanced assurance",
        workspace="workspace/balanced_assurance_tail",
        success=True,
        task_metadata={
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
                "adjudication",
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
                "adjudication_cleanup_ruling_bundle",
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
                "cleanup",
            ],
        },
        steps=[],
        termination_reason="success",
    )

    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    signal = engine._lineage_scheduler_signal(
        episode,
        source_family="assurance",
        target_family="adjudication",
    )
    task = engine.generate_adjacent_task(episode)

    assert signal["branch_outward"] is False
    assert signal["rotation_coverage_gap"] == 0
    assert task.task_id.endswith("_assurance_adjacent")
    assert task.metadata["benchmark_family"] == "assurance"
    assert task.metadata["long_horizon_coding_surface"] == "assurance_cleanup_cert_bundle"


def test_curriculum_seed_priority_prefers_saturated_late_wave_outward_branch(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    outward = EpisodeRecord(
        task_id="late_wave_assurance",
        prompt="late wave",
        workspace="workspace/late_wave_assurance",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_audit_cert_bundle",
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
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
            ],
        },
        steps=[],
        termination_reason="success",
    )
    early = EpisodeRecord(
        task_id="early_assurance",
        prompt="early",
        workspace="workspace/early_assurance",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_audit_cert_bundle",
            "lineage_families": ["oversight", "assurance"],
            "lineage_branch_kinds": ["audit", "audit"],
        },
        steps=[],
        termination_reason="success",
    )

    assert engine._long_horizon_seed_priority(outward) > engine._long_horizon_seed_priority(early)


def test_curriculum_schedule_generated_seed_episodes_keeps_complete_shared_repo_bundle_together(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 3}

    episodes = [
        EpisodeRecord(
            task_id="git_parallel_worker_api_task",
            prompt="worker api",
            workspace="workspace/git_parallel_worker_api_task",
            success=True,
            task_metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_worker_branch",
                "workflow_guard": {
                    "requires_git": True,
                    "shared_repo_id": "repo_sandbox_parallel_merge",
                    "target_branch": "main",
                    "worker_branch": "worker/api-status",
                },
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_parallel_worker_docs_task",
            prompt="worker docs",
            workspace="workspace/git_parallel_worker_docs_task",
            success=True,
            task_metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_worker_branch",
                "workflow_guard": {
                    "requires_git": True,
                    "shared_repo_id": "repo_sandbox_parallel_merge",
                    "target_branch": "main",
                    "worker_branch": "worker/docs-status",
                },
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_parallel_merge_acceptance_task",
            prompt="integrator",
            workspace="workspace/git_parallel_merge_acceptance_task",
            success=True,
            task_metadata={
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
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_repo_status_review_task",
            prompt="review",
            workspace="workspace/git_repo_status_review_task",
            success=True,
            task_metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_workflow",
                "workflow_guard": {"requires_git": True},
            },
            termination_reason="success",
            steps=[],
        ),
    ]

    scheduled = engine.schedule_generated_seed_episodes(episodes, curriculum_kind="adjacent_success")

    assert [episode.task_id for episode in scheduled] == [
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
        "git_parallel_merge_acceptance_task",
    ]


def test_curriculum_schedule_generated_seed_episodes_demotes_incomplete_shared_repo_integrator(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 2}

    episodes = [
        EpisodeRecord(
            task_id="git_parallel_worker_api_task",
            prompt="worker api",
            workspace="workspace/git_parallel_worker_api_task",
            success=True,
            task_metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_worker_branch",
                "workflow_guard": {
                    "requires_git": True,
                    "shared_repo_id": "repo_sandbox_parallel_merge",
                    "target_branch": "main",
                    "worker_branch": "worker/api-status",
                },
            },
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_parallel_merge_acceptance_task",
            prompt="integrator",
            workspace="workspace/git_parallel_merge_acceptance_task",
            success=True,
            task_metadata={
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
            termination_reason="success",
            steps=[],
        ),
        EpisodeRecord(
            task_id="git_repo_status_review_task",
            prompt="review",
            workspace="workspace/git_repo_status_review_task",
            success=True,
            task_metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "git_workflow",
                "workflow_guard": {"requires_git": True},
            },
            termination_reason="success",
            steps=[],
        ),
    ]

    scheduled = engine.schedule_generated_seed_episodes(episodes, curriculum_kind="adjacent_success")

    assert [episode.task_id for episode in scheduled] == [
        "git_parallel_worker_api_task",
        "git_repo_status_review_task",
    ]


def test_curriculum_schedule_generated_seed_episodes_prefers_coverage_expanding_late_wave_tail(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 1}

    saturated_but_deeper = EpisodeRecord(
        task_id="late_wave_assurance_balanced",
        prompt="balanced",
        workspace="workspace/late_wave_assurance_balanced",
        success=True,
        task_metadata={
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
                "adjudication",
                "assurance",
                "validation",
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
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    coverage_expanding = EpisodeRecord(
        task_id="late_wave_assurance_gapfill",
        prompt="gapfill",
        workspace="workspace/late_wave_assurance_gapfill",
        success=True,
        task_metadata={
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
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [saturated_but_deeper, coverage_expanding],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == ["late_wave_assurance_gapfill"]
    assert (
        engine._late_wave_seed_coverage_expansion_priority(coverage_expanding)
        > engine._late_wave_seed_coverage_expansion_priority(saturated_but_deeper)
    )


def test_curriculum_schedule_generated_seed_episodes_scores_late_wave_seed_sets_globally(tmp_path: Path, monkeypatch):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 2}

    same_branch_a = EpisodeRecord(
        task_id="assurance_gapfill_cleanup_a",
        prompt="cleanup a",
        workspace="workspace/assurance_gapfill_cleanup_a",
        success=True,
        task_metadata={
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
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    same_branch_b = EpisodeRecord(
        task_id="assurance_gapfill_cleanup_b",
        prompt="cleanup b",
        workspace="workspace/assurance_gapfill_cleanup_b",
        success=True,
        task_metadata={
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
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    diverse_family = EpisodeRecord(
        task_id="adjudication_gapfill_audit",
        prompt="audit gapfill",
        workspace="workspace/adjudication_gapfill_audit",
        success=True,
        task_metadata={
            "benchmark_family": "adjudication",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "adjudication_audit_ruling_bundle",
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
                "adjudication",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    # Keep this legacy regression focused on set novelty rather than reliability.
    monkeypatch.setattr(engine, "_adjacent_success_seed_projected_completion_mix", lambda episode: 1.0)
    stage_keys = {
        "assurance_gapfill_cleanup_a": {"late:adjudication"},
        "assurance_gapfill_cleanup_b": {"late:adjudication"},
        "adjudication_gapfill_audit": {"late:validation"},
    }
    expected_values = {
        "assurance_gapfill_cleanup_a": 5.1,
        "assurance_gapfill_cleanup_b": 5.0,
        "adjudication_gapfill_audit": 4.9,
    }
    branch_kinds = {
        "assurance_gapfill_cleanup_a": "cleanup",
        "assurance_gapfill_cleanup_b": "cleanup",
        "adjudication_gapfill_audit": "audit",
    }
    target_families = {
        "assurance_gapfill_cleanup_a": "adjudication",
        "assurance_gapfill_cleanup_b": "adjudication",
        "adjudication_gapfill_audit": "validation",
    }
    monkeypatch.setattr(engine, "_late_wave_seed_stage_family_keys", lambda episode: stage_keys[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_expected_value", lambda episode: expected_values[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_cost_units", lambda episode: 2)
    monkeypatch.setattr(
        engine,
        "_adjacent_success_target_benchmark_family",
        lambda episode: target_families[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_lineage_branch_kind",
        lambda episode, benchmark_family, variant: branch_kinds[episode.task_id],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [same_branch_a, same_branch_b, diverse_family],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == [
        "assurance_gapfill_cleanup_a",
        "adjudication_gapfill_audit",
    ]


def test_curriculum_adjacent_success_seed_set_prefers_new_stage_family_coverage_across_batch(
    tmp_path: Path,
    monkeypatch,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {"max_generated_adjacent_tasks": 2}

    same_stage_cleanup = EpisodeRecord(
        task_id="same_stage_cleanup",
        prompt="same stage cleanup",
        workspace="workspace/same_stage_cleanup",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
                "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
                "observed_outcome_late_wave_phase_state_prior_phase_transition_count": 0.0,
                "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
                "observed_success_late_wave_phase_state_prior_rate": 0.6,
                "observed_timeout_late_wave_phase_state_prior_rate": 0.2,
                "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
                "observed_runtime_late_wave_phase_state_prior_seconds": 5.0,
                "observed_runtime_late_wave_phase_state_prior_count": 2.0,
                "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
                "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
                "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
                "observed_runtime_late_wave_phase_state_prior_phase_transition_count": 0.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    late_transition_cleanup = EpisodeRecord(
        task_id="late_transition_cleanup",
        prompt="late transition cleanup",
        workspace="workspace/late_transition_cleanup",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 12,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_phase_transition_count": 1.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_phase_transition_count": 1.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    late_audit = EpisodeRecord(
        task_id="late_audit",
        prompt="late audit",
        workspace="workspace/late_audit",
        success=True,
        task_metadata={
            "benchmark_family": "adjudication",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "adjudication_audit_ruling_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "adjudication",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "audit", "audit", "audit", "audit", "audit",
                "audit", "audit", "audit", "audit", "audit",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_phase_transition_count": 0.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_phase_transition_count": 0.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    stage_keys = {
        "same_stage_cleanup": {"late:adjudication"},
        "late_transition_cleanup": {"mid:adjudication", "late:adjudication"},
        "late_audit": {"late:validation"},
    }
    expected_values = {
        "same_stage_cleanup": 6.2,
        "late_transition_cleanup": 6.0,
        "late_audit": 4.9,
    }
    branch_kinds = {
        "same_stage_cleanup": "cleanup",
        "late_transition_cleanup": "cleanup",
        "late_audit": "audit",
    }
    target_families = {
        "same_stage_cleanup": "adjudication",
        "late_transition_cleanup": "adjudication",
        "late_audit": "validation",
    }

    monkeypatch.setattr(
        engine,
        "_late_wave_seed_stage_family_keys",
        lambda episode: stage_keys[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_seed_expected_value",
        lambda episode: expected_values[episode.task_id],
    )
    monkeypatch.setattr(engine, "_adjacent_success_seed_cost_units", lambda episode: 2)
    monkeypatch.setattr(
        engine,
        "_adjacent_success_target_benchmark_family",
        lambda episode: target_families[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_lineage_branch_kind",
        lambda episode, benchmark_family, variant: branch_kinds[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_episode_uses_long_horizon_adjacent_curriculum",
        lambda episode: True,
    )
    monkeypatch.setattr(
        engine,
        "_episode_benchmark_family",
        lambda episode: episode.task_metadata.get("benchmark_family", ""),
    )
    monkeypatch.setattr(
        engine,
        "_long_horizon_adjacent_variant",
        lambda episode, benchmark_family: str(episode.task_metadata.get("long_horizon_coding_surface", "")),
    )

    scheduled = engine._select_adjacent_success_seed_set([same_stage_cleanup, late_transition_cleanup, late_audit], limit=2)

    assert {episode.task_id for episode in scheduled} == {
        "late_transition_cleanup",
        "late_audit",
    }


def test_curriculum_adjacent_success_seed_set_prefers_stage_family_coverage_per_budget(
    tmp_path: Path,
    monkeypatch,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 2,
        "max_generated_adjacent_cost_units": 4,
    }

    expensive_dual = EpisodeRecord(
        task_id="expensive_dual",
        prompt="expensive dual",
        workspace="workspace/expensive_dual",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    cheap_cleanup = EpisodeRecord(
        task_id="cheap_cleanup",
        prompt="cheap cleanup",
        workspace="workspace/cheap_cleanup",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    cheap_audit = EpisodeRecord(
        task_id="cheap_audit",
        prompt="cheap audit",
        workspace="workspace/cheap_audit",
        success=True,
        task_metadata={"benchmark_family": "adjudication", "long_horizon_coding_surface": "adjudication_audit_ruling_bundle"},
        termination_reason="success",
        steps=[],
    )

    stage_keys = {
        "expensive_dual": {"mid:adjudication", "late:adjudication"},
        "cheap_cleanup": {"late:adjudication"},
        "cheap_audit": {"late:validation"},
    }
    expected_values = {
        "expensive_dual": 7.0,
        "cheap_cleanup": 5.2,
        "cheap_audit": 5.1,
    }
    costs = {
        "expensive_dual": 4,
        "cheap_cleanup": 2,
        "cheap_audit": 2,
    }
    branch_kinds = {
        "expensive_dual": "cleanup",
        "cheap_cleanup": "cleanup",
        "cheap_audit": "audit",
    }
    target_families = {
        "expensive_dual": "adjudication",
        "cheap_cleanup": "adjudication",
        "cheap_audit": "validation",
    }

    monkeypatch.setattr(engine, "_late_wave_seed_stage_family_keys", lambda episode: stage_keys[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_expected_value", lambda episode: expected_values[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_cost_units", lambda episode: costs[episode.task_id])
    monkeypatch.setattr(
        engine,
        "_adjacent_success_target_benchmark_family",
        lambda episode: target_families[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_lineage_branch_kind",
        lambda episode, benchmark_family, variant: branch_kinds[episode.task_id],
    )
    monkeypatch.setattr(engine, "_episode_uses_long_horizon_adjacent_curriculum", lambda episode: True)
    monkeypatch.setattr(
        engine,
        "_episode_benchmark_family",
        lambda episode: str(episode.task_metadata.get("benchmark_family", "")),
    )
    monkeypatch.setattr(
        engine,
        "_long_horizon_adjacent_variant",
        lambda episode, benchmark_family: str(episode.task_metadata.get("long_horizon_coding_surface", "")),
    )

    scheduled = engine._select_adjacent_success_seed_set(
        [expensive_dual, cheap_cleanup, cheap_audit],
        limit=2,
    )

    assert {episode.task_id for episode in scheduled} == {
        "cheap_cleanup",
        "cheap_audit",
    }


def test_curriculum_adjacent_success_seed_set_prefers_reliable_completion_mix_per_budget(
    tmp_path: Path,
    monkeypatch,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 2,
        "max_generated_adjacent_cost_units": 4,
    }

    risky_gapfill = EpisodeRecord(
        task_id="risky_gapfill",
        prompt="risky gapfill",
        workspace="workspace/risky_gapfill",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    safe_gapfill_a = EpisodeRecord(
        task_id="safe_gapfill_a",
        prompt="safe gapfill a",
        workspace="workspace/safe_gapfill_a",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    safe_gapfill_b = EpisodeRecord(
        task_id="safe_gapfill_b",
        prompt="safe gapfill b",
        workspace="workspace/safe_gapfill_b",
        success=True,
        task_metadata={"benchmark_family": "adjudication", "long_horizon_coding_surface": "adjudication_audit_ruling_bundle"},
        termination_reason="success",
        steps=[],
    )

    stage_keys = {
        "risky_gapfill": {"late:validation", "late:governance"},
        "safe_gapfill_a": {"late:validation"},
        "safe_gapfill_b": {"late:governance"},
    }
    expected_values = {
        "risky_gapfill": 8.0,
        "safe_gapfill_a": 5.0,
        "safe_gapfill_b": 4.9,
    }
    projected_completion_mix = {
        "risky_gapfill": 0.4,
        "safe_gapfill_a": 1.2,
        "safe_gapfill_b": 1.1,
    }
    costs = {
        "risky_gapfill": 4,
        "safe_gapfill_a": 2,
        "safe_gapfill_b": 2,
    }
    branch_kinds = {
        "risky_gapfill": "cleanup",
        "safe_gapfill_a": "cleanup",
        "safe_gapfill_b": "audit",
    }
    target_families = {
        "risky_gapfill": "validation",
        "safe_gapfill_a": "validation",
        "safe_gapfill_b": "governance",
    }

    monkeypatch.setattr(engine, "_late_wave_seed_stage_family_keys", lambda episode: stage_keys[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_expected_value", lambda episode: expected_values[episode.task_id])
    monkeypatch.setattr(
        engine,
        "_adjacent_success_seed_projected_completion_mix",
        lambda episode: projected_completion_mix[episode.task_id],
    )
    monkeypatch.setattr(engine, "_adjacent_success_seed_cost_units", lambda episode: costs[episode.task_id])
    monkeypatch.setattr(
        engine,
        "_adjacent_success_target_benchmark_family",
        lambda episode: target_families[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_lineage_branch_kind",
        lambda episode, benchmark_family, variant: branch_kinds[episode.task_id],
    )
    monkeypatch.setattr(engine, "_episode_uses_long_horizon_adjacent_curriculum", lambda episode: True)
    monkeypatch.setattr(
        engine,
        "_episode_benchmark_family",
        lambda episode: str(episode.task_metadata.get("benchmark_family", "")),
    )
    monkeypatch.setattr(
        engine,
        "_long_horizon_adjacent_variant",
        lambda episode, benchmark_family: str(episode.task_metadata.get("long_horizon_coding_surface", "")),
    )

    scheduled = engine._select_adjacent_success_seed_set(
        [risky_gapfill, safe_gapfill_a, safe_gapfill_b],
        limit=2,
    )

    assert {episode.task_id for episode in scheduled} == {
        "safe_gapfill_a",
        "safe_gapfill_b",
    }


def test_curriculum_adjacent_success_seed_set_prefers_fresh_reliable_completion_mix_per_budget(
    tmp_path: Path,
    monkeypatch,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 2,
        "max_generated_adjacent_cost_units": 4,
    }

    stale_reliable = EpisodeRecord(
        task_id="stale_reliable",
        prompt="stale reliable",
        workspace="workspace/stale_reliable",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    fresh_gapfill_a = EpisodeRecord(
        task_id="fresh_gapfill_a",
        prompt="fresh gapfill a",
        workspace="workspace/fresh_gapfill_a",
        success=True,
        task_metadata={"benchmark_family": "assurance", "long_horizon_coding_surface": "assurance_cleanup_cert_bundle"},
        termination_reason="success",
        steps=[],
    )
    fresh_gapfill_b = EpisodeRecord(
        task_id="fresh_gapfill_b",
        prompt="fresh gapfill b",
        workspace="workspace/fresh_gapfill_b",
        success=True,
        task_metadata={"benchmark_family": "adjudication", "long_horizon_coding_surface": "adjudication_audit_ruling_bundle"},
        termination_reason="success",
        steps=[],
    )

    stage_keys = {
        "stale_reliable": {"late:validation", "late:governance"},
        "fresh_gapfill_a": {"late:validation"},
        "fresh_gapfill_b": {"late:governance"},
    }
    expected_values = {
        "stale_reliable": 8.0,
        "fresh_gapfill_a": 5.4,
        "fresh_gapfill_b": 5.2,
    }
    projected_completion_mix = {
        "stale_reliable": 1.35,
        "fresh_gapfill_a": 1.15,
        "fresh_gapfill_b": 1.1,
    }
    projected_completion_freshness = {
        "stale_reliable": 0.45,
        "fresh_gapfill_a": 1.25,
        "fresh_gapfill_b": 1.2,
    }
    costs = {
        "stale_reliable": 4,
        "fresh_gapfill_a": 2,
        "fresh_gapfill_b": 2,
    }
    branch_kinds = {
        "stale_reliable": "cleanup",
        "fresh_gapfill_a": "cleanup",
        "fresh_gapfill_b": "audit",
    }
    target_families = {
        "stale_reliable": "validation",
        "fresh_gapfill_a": "validation",
        "fresh_gapfill_b": "governance",
    }

    monkeypatch.setattr(engine, "_late_wave_seed_stage_family_keys", lambda episode: stage_keys[episode.task_id])
    monkeypatch.setattr(engine, "_adjacent_success_seed_expected_value", lambda episode: expected_values[episode.task_id])
    monkeypatch.setattr(
        engine,
        "_adjacent_success_seed_projected_completion_mix",
        lambda episode: projected_completion_mix[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_seed_projected_completion_freshness",
        lambda episode, reference_now=None: projected_completion_freshness[episode.task_id],
    )
    monkeypatch.setattr(engine, "_adjacent_success_seed_cost_units", lambda episode: costs[episode.task_id])
    monkeypatch.setattr(
        engine,
        "_adjacent_success_target_benchmark_family",
        lambda episode: target_families[episode.task_id],
    )
    monkeypatch.setattr(
        engine,
        "_adjacent_success_lineage_branch_kind",
        lambda episode, benchmark_family, variant: branch_kinds[episode.task_id],
    )
    monkeypatch.setattr(engine, "_episode_uses_long_horizon_adjacent_curriculum", lambda episode: True)
    monkeypatch.setattr(
        engine,
        "_episode_benchmark_family",
        lambda episode: str(episode.task_metadata.get("benchmark_family", "")),
    )
    monkeypatch.setattr(
        engine,
        "_long_horizon_adjacent_variant",
        lambda episode, benchmark_family: str(episode.task_metadata.get("long_horizon_coding_surface", "")),
    )

    scheduled = engine._select_adjacent_success_seed_set(
        [stale_reliable, fresh_gapfill_a, fresh_gapfill_b],
        limit=2,
    )

    assert {episode.task_id for episode in scheduled} == {
        "fresh_gapfill_a",
        "fresh_gapfill_b",
    }


def test_curriculum_schedule_generated_seed_episodes_is_budget_aware(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_tasks": 2,
        "max_generated_adjacent_cost_units": 7,
    }

    expensive = EpisodeRecord(
        task_id="expensive_validation_crosscheck",
        prompt="expensive",
        workspace="workspace/expensive_validation_crosscheck",
        success=True,
        task_metadata={
            "benchmark_family": "validation",
            "origin_benchmark_family": "repo_chore",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "validation_audit_gate_bundle",
            "long_horizon_step_count": 12,
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
                "adjudication",
                "validation",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    cheap_cleanup = EpisodeRecord(
        task_id="cheap_assurance_gapfill_cleanup",
        prompt="cheap cleanup",
        workspace="workspace/cheap_assurance_gapfill_cleanup",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "long_horizon_step_count": 4,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    cheap_audit = EpisodeRecord(
        task_id="cheap_adjudication_gapfill_audit",
        prompt="cheap audit",
        workspace="workspace/cheap_adjudication_gapfill_audit",
        success=True,
        task_metadata={
            "benchmark_family": "adjudication",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "adjudication_audit_ruling_bundle",
            "long_horizon_step_count": 4,
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
                "adjudication",
            ],
            "lineage_branch_kinds": [
                "project_release",
                "project_release",
                "workflow_release",
                "tooling_release",
                "integration_release",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
                "audit",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    scheduled = engine.schedule_generated_seed_episodes(
        [expensive, cheap_cleanup, cheap_audit],
        curriculum_kind="adjacent_success",
    )

    assert [episode.task_id for episode in scheduled] == [
        "cheap_assurance_gapfill_cleanup",
        "cheap_adjudication_gapfill_audit",
    ]
    assert engine._adjacent_success_seed_cost_units(expensive) > engine._adjacent_success_seed_cost_units(cheap_cleanup)


def test_curriculum_adjacent_success_seed_cost_prefers_observed_runtime_over_static_shape(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    measured_fast = EpisodeRecord(
        task_id="measured_fast_tail",
        prompt="fast",
        workspace="workspace/measured_fast_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "long_horizon_step_count": 12,
            "observed_runtime_seconds": 1.6,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    heuristic_only = EpisodeRecord(
        task_id="heuristic_only_tail",
        prompt="slow",
        workspace="workspace/heuristic_only_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "long_horizon_step_count": 4,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_cost_units(measured_fast) == 1
    assert engine._adjacent_success_seed_cost_units(heuristic_only) > engine._adjacent_success_seed_cost_units(
        measured_fast
    )


def test_curriculum_adjacent_success_seed_cost_smooths_observed_runtime_with_prior(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    smoothed = EpisodeRecord(
        task_id="smoothed_runtime_tail",
        prompt="smoothed",
        workspace="workspace/smoothed_runtime_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_runtime_seconds": 1.0,
            "observed_runtime_prior_seconds": 5.0,
            "observed_runtime_prior_count": 4,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    noisy_only = EpisodeRecord(
        task_id="noisy_runtime_tail",
        prompt="noisy",
        workspace="workspace/noisy_runtime_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_runtime_seconds": 1.0,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "oversight",
                "assurance",
                "adjudication",
                "validation",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_cost_units(noisy_only) == 1
    assert engine._adjacent_success_seed_cost_units(smoothed) > engine._adjacent_success_seed_cost_units(noisy_only)


def test_curriculum_adjacent_success_seed_set_prefers_reliable_value_over_cheap_risky_tail(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")
    engine._curriculum_controls_cache = {
        "max_generated_adjacent_success_tasks": 1,
        "max_generated_adjacent_cost_units": 4,
    }

    cheap_risky = EpisodeRecord(
        task_id="cheap_risky_assurance_tail",
        prompt="cheap risky",
        workspace="workspace/cheap_risky_assurance_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_runtime_seconds": 1.0,
            "observed_outcome_prior_count": 5,
            "observed_success_prior_rate": 0.2,
            "observed_timeout_prior_rate": 0.4,
            "observed_budget_exceeded_prior_rate": 0.2,
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
        termination_reason="success",
        steps=[],
    )
    reliable_gapfill = EpisodeRecord(
        task_id="reliable_gapfill_assurance_tail",
        prompt="reliable gapfill",
        workspace="workspace/reliable_gapfill_assurance_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_runtime_seconds": 3.8,
            "observed_outcome_prior_count": 5,
            "observed_success_prior_rate": 0.9,
            "observed_timeout_prior_rate": 0.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "lineage_families": [
                "project",
                "repository",
                "workflow",
                "tooling",
                "integration",
                "repo_chore",
                "validation",
                "governance",
                "assurance",
                "assurance",
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
        termination_reason="success",
        steps=[],
    )

    scheduled = engine._select_adjacent_success_seed_set([cheap_risky, reliable_gapfill], limit=1)

    assert [episode.task_id for episode in scheduled] == ["reliable_gapfill_assurance_tail"]
    assert engine._adjacent_success_seed_cost_units(cheap_risky) < engine._adjacent_success_seed_cost_units(
        reliable_gapfill
    )
    assert engine._adjacent_success_seed_expected_value(reliable_gapfill) > engine._adjacent_success_seed_expected_value(
        cheap_risky
    )


def test_curriculum_adjacent_success_seed_expected_value_uses_broader_policy_when_exact_history_is_thin(
    tmp_path: Path,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    thin_exact_but_strong_family_policy = EpisodeRecord(
        task_id="thin_exact_strong_family_policy",
        prompt="thin exact strong family policy",
        workspace="workspace/thin_exact_strong_family_policy",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_family_branch_prior_count": 6,
            "observed_success_family_branch_prior_rate": 0.9,
            "observed_timeout_family_branch_prior_rate": 0.0,
            "observed_budget_exceeded_family_branch_prior_rate": 0.0,
            "observed_outcome_family_prior_count": 8,
            "observed_success_family_prior_rate": 0.85,
            "observed_timeout_family_prior_rate": 0.0,
            "observed_budget_exceeded_family_prior_rate": 0.0,
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
        termination_reason="success",
        steps=[],
    )
    thin_exact_only = EpisodeRecord(
        task_id="thin_exact_only",
        prompt="thin exact only",
        workspace="workspace/thin_exact_only",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
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
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(thin_exact_but_strong_family_policy) > 0.0
    assert engine._adjacent_success_seed_expected_value(thin_exact_but_strong_family_policy) > engine._adjacent_success_seed_expected_value(
        thin_exact_only
    )


def test_curriculum_adjacent_success_seed_expected_value_uses_late_wave_branch_policy_across_families(
    tmp_path: Path,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    assurance_with_cross_family_cleanup_policy = EpisodeRecord(
        task_id="assurance_with_cross_family_cleanup_policy",
        prompt="assurance with cross family cleanup policy",
        workspace="workspace/assurance_with_cross_family_cleanup_policy",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_late_wave_branch_prior_count": 6,
            "observed_success_late_wave_branch_prior_rate": 0.85,
            "observed_timeout_late_wave_branch_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_branch_prior_rate": 0.0,
            "observed_runtime_late_wave_branch_prior_seconds": 3.0,
            "observed_runtime_late_wave_branch_prior_count": 6,
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
        termination_reason="success",
        steps=[],
    )
    assurance_thin_exact_only = EpisodeRecord(
        task_id="assurance_thin_exact_only",
        prompt="assurance thin exact only",
        workspace="workspace/assurance_thin_exact_only",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
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
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(assurance_with_cross_family_cleanup_policy) > 0.0
    assert engine._adjacent_success_seed_expected_value(
        assurance_with_cross_family_cleanup_policy
    ) > engine._adjacent_success_seed_expected_value(assurance_thin_exact_only)
    assert engine._adjacent_success_seed_cost_units(assurance_with_cross_family_cleanup_policy) >= 2


def test_curriculum_adjacent_success_seed_uses_late_wave_phase_policy_before_coarse_branch_policy(
    tmp_path: Path,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    late_cleanup_tail = EpisodeRecord(
        task_id="late_cleanup_tail",
        prompt="late cleanup tail",
        workspace="workspace/late_cleanup_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_late_wave_phase_prior_count": 5,
            "observed_success_late_wave_phase_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_prior_count": 5,
            "observed_outcome_late_wave_branch_prior_count": 8,
            "observed_success_late_wave_branch_prior_rate": 0.2,
            "observed_timeout_late_wave_branch_prior_rate": 0.6,
            "observed_budget_exceeded_late_wave_branch_prior_rate": 0.0,
            "observed_runtime_late_wave_branch_prior_seconds": 1.0,
            "observed_runtime_late_wave_branch_prior_count": 8,
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
                "adjudication",
                "validation",
                "governance",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    coarse_branch_only = EpisodeRecord(
        task_id="coarse_branch_only",
        prompt="coarse branch only",
        workspace="workspace/coarse_branch_only",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_late_wave_branch_prior_count": 8,
            "observed_success_late_wave_branch_prior_rate": 0.2,
            "observed_timeout_late_wave_branch_prior_rate": 0.6,
            "observed_budget_exceeded_late_wave_branch_prior_rate": 0.0,
            "observed_runtime_late_wave_branch_prior_seconds": 1.0,
            "observed_runtime_late_wave_branch_prior_count": 8,
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
                "adjudication",
                "validation",
                "governance",
                "oversight",
                "assurance",
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
                "cleanup",
                "cleanup",
                "cleanup",
                "cleanup",
                "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(late_cleanup_tail) > engine._adjacent_success_seed_expected_value(
        coarse_branch_only
    )
    assert engine._adjacent_success_seed_cost_units(late_cleanup_tail) > engine._adjacent_success_seed_cost_units(
        coarse_branch_only
    )


def test_curriculum_adjacent_success_seed_uses_late_wave_phase_state_policy_before_phase_only(
    tmp_path: Path,
):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    productive_late_cleanup_tail = EpisodeRecord(
        task_id="productive_late_cleanup_tail",
        prompt="productive late cleanup tail",
        workspace="workspace/productive_late_cleanup_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "observed_scheduler_state": "productive",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_late_wave_phase_state_prior_count": 5,
            "observed_success_late_wave_phase_state_prior_rate": 0.95,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 5,
            "observed_outcome_late_wave_phase_prior_count": 8,
            "observed_success_late_wave_phase_prior_rate": 0.2,
            "observed_timeout_late_wave_phase_prior_rate": 0.6,
            "observed_budget_exceeded_late_wave_phase_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_prior_seconds": 1.0,
            "observed_runtime_late_wave_phase_prior_count": 8,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )
    phase_only_cleanup_tail = EpisodeRecord(
        task_id="phase_only_cleanup_tail",
        prompt="phase only cleanup tail",
        workspace="workspace/phase_only_cleanup_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "observed_scheduler_state": "stalled",
            "observed_outcome_prior_count": 1,
            "observed_success_prior_rate": 0.0,
            "observed_timeout_prior_rate": 1.0,
            "observed_budget_exceeded_prior_rate": 0.0,
            "observed_outcome_late_wave_phase_prior_count": 8,
            "observed_success_late_wave_phase_prior_rate": 0.2,
            "observed_timeout_late_wave_phase_prior_rate": 0.6,
            "observed_budget_exceeded_late_wave_phase_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_prior_seconds": 1.0,
            "observed_runtime_late_wave_phase_prior_count": 8,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(productive_late_cleanup_tail) > engine._adjacent_success_seed_expected_value(
        phase_only_cleanup_tail
    )
    assert engine._adjacent_success_seed_cost_units(productive_late_cleanup_tail) > engine._adjacent_success_seed_cost_units(
        phase_only_cleanup_tail
    )


def test_curriculum_adjacent_success_seed_prefers_recency_weighted_phase_state_signal(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    fresh_productive_tail = EpisodeRecord(
        task_id="fresh_productive_tail",
        prompt="fresh productive tail",
        workspace="workspace/fresh_productive_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.95,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    stale_phase_only_tail = EpisodeRecord(
        task_id="stale_phase_only_tail",
        prompt="stale phase only tail",
        workspace="workspace/stale_phase_only_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_prior_count": 3,
            "observed_success_late_wave_phase_prior_rate": 0.45,
            "observed_timeout_late_wave_phase_prior_rate": 0.3,
            "observed_budget_exceeded_late_wave_phase_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_prior_seconds": 2.0,
            "observed_runtime_late_wave_phase_prior_count": 3,
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(fresh_productive_tail) > engine._adjacent_success_seed_expected_value(
        stale_phase_only_tail
    )
    assert engine._adjacent_success_seed_cost_units(fresh_productive_tail) > engine._adjacent_success_seed_cost_units(
        stale_phase_only_tail
    )


def test_curriculum_adjacent_success_seed_prefers_fresh_supported_cluster_over_single_fresh_tail(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    fresh_supported_cluster_tail = EpisodeRecord(
        task_id="fresh_supported_cluster_tail",
        prompt="fresh supported cluster tail",
        workspace="workspace/fresh_supported_cluster_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    single_fresh_tail = EpisodeRecord(
        task_id="single_fresh_tail",
        prompt="single fresh tail",
        workspace="workspace/single_fresh_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 1.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 1.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(fresh_supported_cluster_tail) > engine._adjacent_success_seed_expected_value(
        single_fresh_tail
    )


def test_curriculum_adjacent_success_seed_prefers_dispersed_recency_support_over_narrow_cluster(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    dispersed_tail = EpisodeRecord(
        task_id="dispersed_tail",
        prompt="dispersed tail",
        workspace="workspace/dispersed_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    narrow_tail = EpisodeRecord(
        task_id="narrow_tail",
        prompt="narrow tail",
        workspace="workspace/narrow_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
                "adjudication", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 1.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 1.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(dispersed_tail) > engine._adjacent_success_seed_expected_value(
        narrow_tail
    )


def test_curriculum_adjacent_success_seed_prefers_directional_dispersion_over_lateral_dispersion(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    directional_tail = EpisodeRecord(
        task_id="directional_tail",
        prompt="directional tail",
        workspace="workspace/directional_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    lateral_tail = EpisodeRecord(
        task_id="lateral_tail",
        prompt="lateral tail",
        workspace="workspace/lateral_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 0.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 0.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(directional_tail) > engine._adjacent_success_seed_expected_value(
        lateral_tail
    )


def test_curriculum_adjacent_success_seed_prefers_phase_transition_over_same_stage_downstream(tmp_path: Path):
    engine = CurriculumEngine(memory_root=tmp_path / "episodes")

    phase_transition_tail = EpisodeRecord(
        task_id="phase_transition_tail",
        prompt="phase transition tail",
        workspace="workspace/phase_transition_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 14,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_phase_transition_count": 1.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_phase_transition_count": 1.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )
    same_stage_downstream_tail = EpisodeRecord(
        task_id="same_stage_downstream_tail",
        prompt="same stage downstream tail",
        workspace="workspace/same_stage_downstream_tail",
        success=True,
        task_metadata={
            "benchmark_family": "assurance",
            "origin_benchmark_family": "oversight",
            "difficulty": "long_horizon",
            "long_horizon_coding_surface": "assurance_cleanup_cert_bundle",
            "lineage_depth": 15,
            "lineage_families": [
                "project", "repository", "workflow", "tooling", "integration",
                "repo_chore", "validation", "governance", "oversight", "assurance",
            ],
            "lineage_branch_kinds": [
                "project_release", "project_release", "workflow_release", "tooling_release", "integration_release",
                "cleanup", "cleanup", "cleanup", "cleanup", "cleanup",
            ],
            "observed_outcome_late_wave_phase_state_prior_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_support_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_outcome_late_wave_phase_state_prior_phase_transition_count": 0.0,
            "observed_outcome_late_wave_phase_state_prior_is_recency_weighted": True,
            "observed_success_late_wave_phase_state_prior_rate": 0.9,
            "observed_timeout_late_wave_phase_state_prior_rate": 0.0,
            "observed_budget_exceeded_late_wave_phase_state_prior_rate": 0.0,
            "observed_runtime_late_wave_phase_state_prior_seconds": 4.0,
            "observed_runtime_late_wave_phase_state_prior_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_support_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_dispersion_count": 3.0,
            "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count": 2.0,
            "observed_runtime_late_wave_phase_state_prior_phase_transition_count": 0.0,
            "observed_runtime_late_wave_phase_state_prior_is_recency_weighted": True,
        },
        termination_reason="success",
        steps=[],
    )

    assert engine._adjacent_success_seed_expected_value(phase_transition_tail) > engine._adjacent_success_seed_expected_value(
        same_stage_downstream_tail
    )
