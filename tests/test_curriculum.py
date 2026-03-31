from pathlib import Path
import json

from agent_kernel.curriculum import CurriculumEngine
from agent_kernel.loop import AgentKernel
from agent_kernel.memory import EpisodeMemory
from agent_kernel.policy import Policy
from agent_kernel.schemas import EpisodeRecord, StepRecord
from agent_kernel.config import KernelConfig
from agent_kernel.schemas import ActionDecision
from agent_kernel.task_bank import TaskBank


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
            task_metadata={"benchmark_family": "project"},
            termination_reason="success",
            steps=[],
        )
    )

    episode = EpisodeRecord(
        task_id="deployment_manifest_task",
        prompt="Prepare deployment workspace.",
        workspace="workspace/deployment_manifest_task",
        success=True,
        task_metadata={"benchmark_family": "project"},
        steps=[],
        termination_reason="success",
    )

    task = CurriculumEngine(memory_root=tmp_path / "episodes").generate_adjacent_task(episode)

    assert task.task_id == "deployment_manifest_task_project_adjacent"
    assert task.metadata["curriculum_kind"] == "adjacent_success"
    assert task.metadata["benchmark_family"] == "project"
    assert "project/check.txt" in task.expected_files


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
