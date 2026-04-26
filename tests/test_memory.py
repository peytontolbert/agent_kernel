import json

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.context_budget import ContextBudgeter
from agent_kernel.extensions.extractors import (
    _normalize_command_for_workspace,
    extract_operator_classes,
    dedupe_skills,
    extract_successful_command_skills,
    extract_tool_candidates,
    score_skill_quality,
)
from agent_kernel.learning_compiler import (
    compile_episode_learning_candidates,
    load_learning_candidates,
    matching_learning_candidates,
)
from agent_kernel.memory import EpisodeMemory, GraphMemory
from agent_kernel.schemas import EpisodeRecord, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.tasking import task_bank as task_bank_module
from agent_kernel.tasking.task_bank import (
    TaskBank,
    load_benchmark_candidate_tasks,
    load_discovered_tasks,
    load_episode_replay_tasks,
    load_operator_replay_tasks,
    load_skill_replay_tasks,
    load_skill_transfer_tasks,
    load_tool_replay_tasks,
    load_transition_pressure_tasks,
    load_verifier_candidate_tasks,
    load_verifier_replay_tasks,
)


def test_memory_task_loaders_use_bundled_rule_templates(monkeypatch, tmp_path):
    rules_path = tmp_path / "synthesis_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "memory_task_rules": {
                    "episode_replay": {
                        "task_id_suffix": "_episode_clone",
                        "workspace_suffix": "_episode_clone",
                        "prompt_template": "EPISODE {prompt}",
                        "metadata": {
                            "benchmark_family": "episode_alt",
                            "memory_source": "episode_alt",
                            "requires_retrieval": False,
                        },
                    },
                    "verifier_replay": {
                        "task_id_suffix": "_verify_clone",
                        "workspace_suffix": "_verify_clone",
                        "metadata": {
                            "benchmark_family": "verifier_alt",
                            "memory_source": "verifier_alt",
                        },
                    },
                    "skill_transfer": {
                        "task_id_suffix": "_transfer_clone",
                        "workspace_suffix": "_transfer_clone",
                        "prompt_template": "TRANSFER {prompt}",
                        "metadata": {
                            "benchmark_family": "transfer_alt",
                            "memory_source": "transfer_alt",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(task_bank_module, "_TASK_BANK_SYNTHESIS_RULES_PATH", rules_path)
    task_bank_module._task_bank_synthesis_rules.cache_clear()

    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing hello agent kernel.",
                "workspace": str(tmp_path / "workspace" / "hello_task"),
                "success": True,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                },
                "summary": {"executed_commands": ["printf 'hello agent kernel\n' > hello.txt"]},
            }
        ),
        encoding="utf-8",
    )

    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello agent kernel\n' > hello.txt"]},
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        episode_tasks = load_episode_replay_tasks(episodes_root)
        verifier_tasks = load_verifier_replay_tasks(episodes_root, tmp_path / "missing_skills.json")
        transfer_tasks = load_skill_transfer_tasks(skills_path)
    finally:
        task_bank_module._task_bank_synthesis_rules.cache_clear()

    assert episode_tasks[0].task_id == "hello_task_episode_clone"
    assert episode_tasks[0].workspace_subdir == "hello_task_episode_clone"
    assert episode_tasks[0].prompt.startswith("EPISODE ")
    assert episode_tasks[0].metadata["benchmark_family"] == "episode_alt"
    assert episode_tasks[0].metadata["memory_source"] == "episode_alt"
    assert episode_tasks[0].metadata["requires_retrieval"] is False

    assert verifier_tasks[0].task_id == "hello_task_episode_clone_verify_clone"
    assert verifier_tasks[0].workspace_subdir == "hello_task_episode_clone_verify_clone"
    assert verifier_tasks[0].metadata["benchmark_family"] == "verifier_alt"
    assert verifier_tasks[0].metadata["memory_source"] == "verifier_alt"

    assert transfer_tasks[0].task_id == "hello_task_to_math_task_transfer_clone"
    assert transfer_tasks[0].workspace_subdir == "math_task_transfer_clone"
    assert transfer_tasks[0].prompt.startswith("TRANSFER ")
    assert transfer_tasks[0].metadata["benchmark_family"] == "transfer_alt"
    assert transfer_tasks[0].metadata["memory_source"] == "transfer_alt"



def test_episode_memory_persists_summary_and_fragments(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="episode_task",
        prompt="Create out.txt containing done.",
        workspace=str(tmp_path / "workspace" / "episode_task"),
        success=False,
        task_metadata={"benchmark_family": "workflow", "capability": "workflow_environment"},
        task_contract={
            "prompt": "Create out.txt containing done.",
            "workspace_subdir": "episode_task",
            "setup_commands": [],
            "success_command": "test -f out.txt && grep -q '^done$' out.txt",
            "suggested_commands": ["printf 'done\\n' > out.txt"],
            "expected_files": ["out.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"out.txt": "done\n"},
            "max_steps": 5,
            "metadata": {"benchmark_family": "workflow", "capability": "workflow_environment"},
        },
        termination_reason="repeated_failed_action",
        steps=[
            StepRecord(
                index=1,
                thought="try a broken command",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result={
                    "command": "false",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={
                    "passed": False,
                    "reasons": ["exit code was 1", "missing expected file: out.txt"],
                },
                state_progress_delta=0.0,
                state_regression_count=1,
                state_transition={
                    "progress_delta": 0.0,
                    "regressions": ["out.txt"],
                    "cleared_forbidden_artifacts": [],
                    "newly_materialized_expected_artifacts": [],
                },
                command_governance={
                    "score": -5,
                    "risk_flags": ["network_access_conflict"],
                    "action_categories": [],
                    "environment_alignment": {"network_access_aligned": False},
                    "network_host": "example.com",
                },
            )
        ],
        universe_summary={
            "environment_alignment": {"network_access_aligned": False},
            "environment_snapshot": {"network_access_mode": "allowlist_only"},
        },
    )

    memory.save(episode)
    payload = memory.load("episode_task")

    assert payload["summary"]["termination_reason"] == "repeated_failed_action"
    assert payload["task_metadata"]["benchmark_family"] == "workflow"
    assert payload["task_contract"]["expected_file_contents"]["out.txt"] == "done\n"
    assert payload["summary"]["failure_types"] == ["command_failure", "missing_expected_file"]
    assert payload["summary"]["state_regression_steps"] == 1
    assert payload["summary"]["environment_violation_counts"]["network_access_conflict"] == 1
    assert payload["summary"]["environment_alignment_failures"] == ["network_access_aligned"]
    assert payload["fragments"][0]["kind"] == "command"
    assert payload["fragments"][1]["kind"] == "failure"
    assert any(fragment["kind"] == "state_transition" for fragment in payload["fragments"])
    assert any(fragment["kind"] == "governance" for fragment in payload["fragments"])


def test_successful_verification_passed_reason_does_not_create_failure_data(tmp_path):
    memory = EpisodeMemory(tmp_path / "episodes")
    episode = EpisodeRecord(
        task_id="successful_task",
        prompt="Create done.txt containing done.",
        workspace=str(tmp_path / "workspace" / "successful_task"),
        success=True,
        task_metadata={"benchmark_family": "workflow"},
        task_contract={
            "prompt": "Create done.txt containing done.",
            "workspace_subdir": "successful_task",
            "setup_commands": [],
            "success_command": "test -f done.txt && grep -q '^done$' done.txt",
            "suggested_commands": ["printf 'done\\n' > done.txt"],
            "expected_files": ["done.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"done.txt": "done\n"},
            "max_steps": 3,
            "metadata": {"benchmark_family": "workflow"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="write the file",
                action="code_execute",
                content="printf 'done\\n' > done.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'done\\n' > done.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
            )
        ],
    )

    memory.save(episode)
    payload = memory.load("successful_task")
    assert payload["summary"]["failure_types"] == []

    candidates = compile_episode_learning_candidates(episode)
    assert [candidate["artifact_kind"] for candidate in candidates] == ["success_skill_candidate"]


def test_episode_memory_repairs_historical_success_failure_summary(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("historical_success.json").write_text(
        json.dumps(
            {
                "task_id": "historical_success",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "summary": {"failure_types": ["other"], "executed_commands": []},
                "steps": [
                    {
                        "index": 1,
                        "action": "code_execute",
                        "content": "printf 'done\\n' > done.txt",
                        "verification": {"passed": True, "reasons": ["verification passed"]},
                        "failure_signals": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = EpisodeMemory(episodes_root).load("historical_success")

    assert payload["summary"]["failure_types"] == []
    assert payload["summary"]["executed_commands"] == ["printf 'done\\n' > done.txt"]


def test_load_learning_candidates_repairs_success_only_failure_markers(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success-gap",
                        "artifact_kind": "benchmark_gap",
                        "source_task_id": "historical_success",
                        "termination_reason": "success",
                        "failure_types": ["other"],
                        "transition_failures": [],
                    },
                    {
                        "candidate_id": "learning:success-skill",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "historical_success",
                        "termination_reason": "success",
                        "known_failure_types": ["other"],
                        "procedure": {"commands": ["printf 'done\\n' > done.txt"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_learning_candidates(learning_path)

    assert [candidate["candidate_id"] for candidate in candidates] == ["learning:success-skill"]
    assert candidates[0]["known_failure_types"] == []


def test_load_learning_candidates_respects_explicit_path_under_sqlite_config(tmp_path):
    learning_path = tmp_path / "alt-learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:isolated-file",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "isolated_task",
                        "benchmark_family": "workflow",
                        "command": "false",
                        "verification_reasons": ["exit code was 1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        storage_backend="sqlite",
        runtime_database_path=tmp_path / "runtime" / "agentkernel.sqlite3",
        learning_artifacts_path=tmp_path / "learning" / "run_learning_artifacts.json",
    )
    config.sqlite_store().upsert_learning_candidates(
        [
            {
                "candidate_id": "learning:sqlite-global",
                "artifact_kind": "negative_command_pattern",
                "source_task_id": "global_task",
                "benchmark_family": "workflow",
                "command": "printf 'global\\n'",
                "verification_reasons": ["exit code was 1"],
            }
        ]
    )

    candidates = load_learning_candidates(learning_path, config=config)

    assert [candidate["candidate_id"] for candidate in candidates] == ["learning:isolated-file"]


def test_episode_memory_falls_back_to_json_exports_when_sqlite_is_empty(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("file_only_task.json").write_text(
        json.dumps(
            {
                "task_id": "file_only_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "summary": {"failure_types": [], "executed_commands": ["printf 'done\\n' > done.txt"]},
                "steps": [
                    {
                        "index": 1,
                        "action": "code_execute",
                        "content": "printf 'done\\n' > done.txt",
                        "verification": {"passed": True, "reasons": ["verification passed"]},
                        "failure_signals": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        storage_backend="sqlite",
        runtime_database_path=tmp_path / "runtime" / "agentkernel.sqlite3",
        trajectories_root=episodes_root,
    )

    memory = EpisodeMemory(episodes_root, config=config)

    assert config.sqlite_store().iter_episode_documents() == []
    assert [document["task_id"] for document in memory.list_documents()] == ["file_only_task"]
    assert memory.load("file_only_task")["summary"]["executed_commands"] == ["printf 'done\\n' > done.txt"]


def test_load_learning_candidates_merges_json_exports_missing_from_sqlite(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:shared",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "shared_task",
                        "termination_reason": "success",
                        "procedure": {"commands": ["printf 'db\\n' > done.txt"]},
                    },
                    {
                        "candidate_id": "learning:file-only",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "file_only_task",
                        "termination_reason": "success",
                        "procedure": {"commands": ["printf 'file\\n' > done.txt"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        storage_backend="sqlite",
        runtime_database_path=tmp_path / "runtime" / "agentkernel.sqlite3",
        learning_artifacts_path=learning_path,
    )
    config.sqlite_store().upsert_learning_candidates(
        [
            {
                "candidate_id": "learning:shared",
                "artifact_kind": "success_skill_candidate",
                "source_task_id": "shared_task",
                "benchmark_family": "workflow",
                "memory_source": "",
                "support_count": 1,
                "termination_reason": "success",
                "procedure": {"commands": ["printf 'db\\n' > done.txt"]},
            },
            {
                "candidate_id": "learning:db-only",
                "artifact_kind": "success_skill_candidate",
                "source_task_id": "db_only_task",
                "benchmark_family": "workflow",
                "memory_source": "",
                "support_count": 1,
                "termination_reason": "success",
                "procedure": {"commands": ["printf 'db-only\\n' > done.txt"]},
            },
        ]
    )

    candidates = load_learning_candidates(learning_path, config=config)

    assert [candidate["candidate_id"] for candidate in candidates] == [
        "learning:db-only",
        "learning:shared",
        "learning:file-only",
    ]


def test_failure_reason_classification_keeps_forbidden_and_content_failures(tmp_path):
    memory = EpisodeMemory(tmp_path / "episodes")
    episode = EpisodeRecord(
        task_id="classified_failure_task",
        prompt="Materialize the deployment artifacts without leaving drafts behind.",
        workspace=str(tmp_path / "workspace" / "classified_failure_task"),
        success=False,
        task_metadata={"benchmark_family": "workflow"},
        task_contract={
            "prompt": "Materialize the deployment artifacts without leaving drafts behind.",
            "workspace_subdir": "classified_failure_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": [],
            "expected_files": ["deploy/manifest.txt"],
            "expected_output_substrings": [],
            "forbidden_files": ["staging/draft.txt"],
            "forbidden_output_substrings": ["warning"],
            "expected_file_contents": {"deploy/manifest.txt": "version=1\n"},
            "max_steps": 3,
            "metadata": {"benchmark_family": "workflow"},
        },
        termination_reason="policy_terminated",
        steps=[
            StepRecord(
                index=1,
                thought="write the wrong files",
                action="code_execute",
                content="printf 'draft\\n' > staging/draft.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'draft\\n' > staging/draft.txt",
                    "exit_code": 0,
                    "stdout": "warning\n",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={
                    "passed": False,
                    "reasons": [
                        "forbidden file present: staging/draft.txt",
                        "unexpected file content: deploy/manifest.txt",
                        "forbidden output present: warning",
                    ],
                },
            )
        ],
    )

    memory.save(episode)
    payload = memory.load("classified_failure_task")

    assert payload["summary"]["failure_types"] == [
        "forbidden_file_present",
        "forbidden_output_present",
        "unexpected_file_content",
    ]

    candidates = compile_episode_learning_candidates(episode)
    failure_case = next(candidate for candidate in candidates if candidate["artifact_kind"] == "failure_case")
    assert failure_case["failure_types"] == [
        "forbidden_file_present",
        "forbidden_output_present",
        "unexpected_file_content",
    ]


def test_episode_memory_recurses_into_generated_phase_directories(tmp_path):
    episodes_root = tmp_path / "episodes"
    generated_root = episodes_root / "generated_failure"
    generated_root.mkdir(parents=True)
    generated_root.joinpath("nested_task.json").write_text(
        json.dumps(
            {
                "task_id": "nested_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "summary": {"executed_commands": ["printf 'ok\\n' > status.txt"]},
                "task_contract": {
                    "prompt": "Create status.txt.",
                    "workspace_subdir": "nested_task",
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": ["printf 'ok\\n' > status.txt"],
                    "expected_files": ["status.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"status.txt": "ok\n"},
                    "max_steps": 3,
                    "metadata": {"benchmark_family": "workflow"},
                },
            }
        ),
        encoding="utf-8",
    )

    documents = EpisodeMemory(episodes_root).list_documents()

    assert len(documents) == 1
    assert documents[0]["episode_storage"]["phase"] == "generated_failure"
    assert documents[0]["task_metadata"]["episode_phase"] == "generated_failure"


def test_episode_memory_includes_failure_recovery_learning_documents(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:recovery_case:repo_sync_matrix_task_path_recovery",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "repo_sync_matrix_task_path_recovery",
                        "benchmark_family": "repository",
                        "parent_task": "repo_sync_matrix_task",
                        "success": True,
                        "failure_types": ["missing_expected_file"],
                        "recovery_commands": [
                            "mkdir -p repo && printf 'repository recovered\\n' > repo/status.txt"
                        ],
                        "task_metadata": {
                            "benchmark_family": "repository",
                            "source_task": "repo_sync_matrix_task",
                        },
                    },
                    {
                        "candidate_id": "learning:negative_command:repo_sync_matrix_task:abc123",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "repo_sync_matrix_task",
                        "benchmark_family": "repository",
                        "command": "printf 'wrong\\n' > repo/output.txt",
                        "failure_types": ["unexpected_file_content"],
                        "verification_reasons": ["unexpected file content: repo/status.txt"],
                        "task_metadata": {
                            "benchmark_family": "repository",
                            "source_task": "repo_sync_matrix_task",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    documents = EpisodeMemory(episodes_root).list_documents()

    task_ids = {document["task_id"] for document in documents}
    assert "repo_sync_matrix_task_path_recovery" in task_ids
    assert "repo_sync_matrix_task__negative_command_pattern" in task_ids

    recovery = next(document for document in documents if document["task_id"] == "repo_sync_matrix_task_path_recovery")
    assert recovery["success"] is True
    assert recovery["summary"]["executed_commands"] == [
        "mkdir -p repo && printf 'repository recovered\\n' > repo/status.txt"
    ]
    assert recovery["task_metadata"]["curriculum_kind"] == "failure_recovery"
    assert recovery["episode_storage"]["phase"] == "learning_artifacts"

    negative = next(
        document for document in documents if document["task_id"] == "repo_sync_matrix_task__negative_command_pattern"
    )
    assert negative["success"] is False
    assert negative["summary"]["failure_types"] == ["unexpected_file_content"]
    assert negative["fragments"][0]["reason"] == "unexpected file content: repo/status.txt"


def test_episode_memory_graph_summary_tracks_memory_sources(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="episode_memory_task",
        prompt="Replay the earlier task.",
        workspace=str(tmp_path / "workspace" / "episode_memory_task"),
        success=False,
        task_metadata={"benchmark_family": "episode_memory", "memory_source": "episode"},
        task_contract={
            "prompt": "Replay the earlier task.",
            "workspace_subdir": "episode_memory_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": ["false"],
            "expected_files": [],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 2,
            "metadata": {"benchmark_family": "episode_memory", "memory_source": "episode"},
        },
        termination_reason="no_state_progress",
        steps=[
            StepRecord(
                index=1,
                thought="repeat the bad replay",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result={"command": "false", "exit_code": 1, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["exit code was 1"]},
                failure_signals=["no_state_progress"],
            )
        ],
    )

    memory.save(episode)
    summary = memory.graph_summary()

    assert summary["memory_sources"] == {"episode": 1}
    assert summary["memory_source_failure_signals"] == {"episode": {"no_state_progress": 1}}


def test_episode_memory_graph_summary_tracks_trusted_retrieval_carryover(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="retrieval_memory_task",
        prompt="Use retrieved release guidance to write release.txt.",
        workspace=str(tmp_path / "workspace" / "retrieval_memory_task"),
        success=True,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Use retrieved release guidance to write release.txt.",
            "workspace_subdir": "retrieval_memory_task",
            "setup_commands": [],
            "success_command": "test -f app/release.txt",
            "suggested_commands": ["mkdir -p app && printf 'release ready\\n' > app/release.txt"],
            "expected_files": ["app/release.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"app/release.txt": "release ready\n"},
            "max_steps": 3,
            "metadata": {"benchmark_family": "repository", "memory_source": "episode"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="follow trusted release guidance",
                action="code_execute",
                content="mkdir -p app && printf 'release ready\\n' > app/release.txt",
                selected_skill_id=None,
                command_result={
                    "command": "mkdir -p app && printf 'release ready\\n' > app/release.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                selected_retrieval_span_id="learning:seed:release",
                retrieval_influenced=True,
                trust_retrieval=True,
            )
        ],
    )

    memory.save(episode)
    summary = memory.graph_summary()

    assert summary["retrieval_backed_successes"] == 1
    assert summary["retrieval_influenced_successes"] == 1
    assert summary["trusted_retrieval_successes"] == 1
    assert summary["retrieval_backed_command_counts"] == {
        "mkdir -p app && printf 'release ready\\n' > app/release.txt": 1
    }
    assert summary["trusted_retrieval_command_counts"] == {
        "mkdir -p app && printf 'release ready\\n' > app/release.txt": 1
    }


def test_episode_memory_graph_summary_tracks_trusted_retrieval_procedures(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="retrieval_sequence_task",
        prompt="Use trusted retrieval to restore the report and verify it.",
        workspace=str(tmp_path / "workspace" / "retrieval_sequence_task"),
        success=True,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Use trusted retrieval to restore the report and verify it.",
            "workspace_subdir": "retrieval_sequence_task",
            "setup_commands": [],
            "success_command": "pytest -q tests/test_status.py",
            "suggested_commands": [
                "printf 'status ready\\n' > reports/status.txt",
                "pytest -q tests/test_status.py",
            ],
            "expected_files": ["reports/status.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"reports/status.txt": "status ready\n"},
            "max_steps": 4,
            "metadata": {"benchmark_family": "repository", "memory_source": "episode"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="follow trusted retrieval repair",
                action="code_execute",
                content="printf 'status ready\\n' > reports/status.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'status ready\\n' > reports/status.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                selected_retrieval_span_id="learning:seed:status",
                retrieval_influenced=True,
                trust_retrieval=True,
            ),
            StepRecord(
                index=2,
                thought="verify the repair",
                action="code_execute",
                content="pytest -q tests/test_status.py",
                selected_skill_id=None,
                command_result={
                    "command": "pytest -q tests/test_status.py",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
            ),
        ],
    )

    memory.save(episode)
    summary = memory.graph_summary()

    assert summary["trusted_retrieval_procedures"] == [
        {
            "commands": [
                "printf 'status ready\\n' > reports/status.txt",
                "pytest -q tests/test_status.py",
            ],
            "count": 1,
        }
    ]


def test_episode_memory_graph_summary_surfaces_semantic_obligations_and_recovery_traces(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="semantic_memory_task",
        prompt="Repair the release report and keep docs stable.",
        workspace=str(tmp_path / "workspace" / "semantic_memory_task"),
        success=True,
        task_metadata={
            "benchmark_family": "repository",
            "memory_source": "episode",
            "semantic_verifier": {
                "expected_changed_paths": ["src/release_state.txt"],
                "generated_paths": ["generated/release.patch"],
                "preserved_paths": ["docs/context.md"],
                "report_rules": [{"path": "reports/release_review.txt", "must_mention": ["ready"]}],
                "test_commands": [{"label": "release check", "argv": ["pytest", "-q", "tests/test_release.py"]}],
            },
        },
        task_contract={
            "prompt": "Repair the release report and keep docs stable.",
            "workspace_subdir": "semantic_memory_task",
            "setup_commands": [],
            "success_command": "pytest -q tests/test_release.py",
            "suggested_commands": [
                "printf 'broken\\n' > reports/release_review.txt",
                "printf 'ready\\n' > reports/release_review.txt",
            ],
            "expected_files": ["reports/release_review.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"reports/release_review.txt": "ready\n"},
            "max_steps": 4,
            "metadata": {
                "semantic_verifier": {
                    "expected_changed_paths": ["src/release_state.txt"],
                    "generated_paths": ["generated/release.patch"],
                    "preserved_paths": ["docs/context.md"],
                    "report_rules": [{"path": "reports/release_review.txt", "must_mention": ["ready"]}],
                    "test_commands": [{"label": "release check", "argv": ["pytest", "-q", "tests/test_release.py"]}],
                }
            },
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="write the wrong report",
                action="code_execute",
                content="printf 'broken\\n' > reports/release_review.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'broken\\n' > reports/release_review.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": False, "reasons": ["unexpected file content: reports/release_review.txt"]},
                failure_signals=["state_regression"],
                state_progress_delta=0.0,
                state_transition={"progress_delta": 0.0, "regressions": ["reports/release_review.txt"]},
            ),
            StepRecord(
                index=2,
                thought="repair the report",
                action="code_execute",
                content="printf 'ready\\n' > reports/release_review.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'ready\\n' > reports/release_review.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                state_progress_delta=0.5,
                state_transition={
                    "progress_delta": 0.5,
                    "newly_materialized_expected_artifacts": ["reports/release_review.txt"],
                    "newly_satisfied_expected_contents": ["reports/release_review.txt"],
                    "newly_updated_report_paths": ["reports/release_review.txt"],
                    "edit_patches": [
                        {
                            "path": "reports/release_review.txt",
                            "status": "modified",
                            "patch": "--- a/reports/release_review.txt\n+++ b/reports/release_review.txt\n@@ -1 +1 @@\n-broken\n+ready\n",
                            "patch_summary": "modified reports/release_review.txt (+1 -1)",
                        }
                    ],
                    "state_change_score": 3,
                },
            ),
        ],
    )

    memory.save(episode)
    payload = memory.load("semantic_memory_task")
    summary = memory.graph_summary()

    assert payload["summary"]["verifier_obligation_count"] >= 4
    assert "reports/release_review.txt" in payload["summary"]["changed_paths"]
    assert payload["summary"]["recovery_trace_count"] == 1
    assert any(fragment["kind"] == "verifier_obligation" for fragment in payload["fragments"])
    assert any(fragment["kind"] == "command_outcome" for fragment in payload["fragments"])
    assert any(fragment["kind"] == "edit_patch" for fragment in payload["fragments"])
    assert any(fragment["kind"] == "recovery_trace" for fragment in payload["fragments"])
    assert summary["verifier_obligation_counts"]["write workflow report reports/release_review.txt and mention ready"] == 1
    assert summary["changed_path_counts"]["reports/release_review.txt"] >= 1
    assert summary["edit_patch_path_counts"]["reports/release_review.txt"] >= 1
    assert summary["recovery_command_counts"]["printf 'ready\\n' > reports/release_review.txt"] == 1
    assert summary["semantic_episodes"][0]["task_id"] == "semantic_memory_task"
    assert summary["semantic_episodes"][0]["edit_patches"][0]["path"] == "reports/release_review.txt"
    assert summary["semantic_episodes"][0]["failure_signals"] == ["state_regression"]
    assert summary["semantic_prototypes"][0]["changed_paths"] == ["reports/release_review.txt"]
    assert summary["semantic_prototypes"][0]["success_count"] == 1
    assert summary["semantic_prototypes"][0]["recovery_commands"] == {
        "printf 'ready\\n' > reports/release_review.txt": 1
    }


def test_episode_memory_semantic_recall_ranks_changed_paths_and_obligations(tmp_path):
    memory = EpisodeMemory(tmp_path)
    repair_episode = EpisodeRecord(
        task_id="release_repair_task",
        prompt="Repair the release report and mention ready.",
        workspace=str(tmp_path / "workspace" / "release_repair_task"),
        success=True,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Repair the release report and mention ready.",
            "workspace_subdir": "release_repair_task",
            "setup_commands": [],
            "success_command": "pytest -q tests/test_release.py",
            "suggested_commands": [],
            "expected_files": ["reports/release_review.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"reports/release_review.txt": "ready\n"},
            "max_steps": 3,
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="repair report",
                action="code_execute",
                content="printf 'ready\\n' > reports/release_review.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'ready\\n' > reports/release_review.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                state_progress_delta=1.0,
                state_transition={
                    "progress_delta": 1.0,
                    "newly_updated_report_paths": ["reports/release_review.txt"],
                    "edit_patches": [
                        {
                            "path": "reports/release_review.txt",
                            "status": "modified",
                            "patch": "-broken\n+ready\n",
                            "patch_summary": "modified reports/release_review.txt (+1 -1)",
                        }
                    ],
                },
            )
        ],
    )
    docs_episode = EpisodeRecord(
        task_id="docs_cleanup_task",
        prompt="Refresh docs context.",
        workspace=str(tmp_path / "workspace" / "docs_cleanup_task"),
        success=True,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Refresh docs context.",
            "workspace_subdir": "docs_cleanup_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": [],
            "expected_files": ["docs/context.md"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"docs/context.md": "updated\n"},
            "max_steps": 2,
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="refresh docs",
                action="code_execute",
                content="printf 'updated\\n' > docs/context.md",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'updated\\n' > docs/context.md",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                state_progress_delta=1.0,
                state_transition={
                    "progress_delta": 1.0,
                    "edit_patches": [
                        {
                            "path": "docs/context.md",
                            "status": "modified",
                            "patch": "-old\n+updated\n",
                            "patch_summary": "modified docs/context.md (+1 -1)",
                        }
                    ],
                },
            )
        ],
    )
    memory.save(repair_episode)
    memory.save(docs_episode)

    recalled = memory.semantic_recall(
        benchmark_family="repository",
        changed_paths=["reports/release_review.txt"],
        verifier_obligations=["write workflow report reports/release_review.txt and mention ready"],
        require_success=True,
        limit=2,
    )

    assert [item["task_id"] for item in recalled] == ["release_repair_task", "docs_cleanup_task"]
    assert recalled[0]["changed_paths"] == ["reports/release_review.txt"]


def test_episode_memory_semantic_recall_matches_failure_signals(tmp_path):
    memory = EpisodeMemory(tmp_path)
    regression_episode = EpisodeRecord(
        task_id="recovery_failure_task",
        prompt="Repair the release state after regression.",
        workspace=str(tmp_path / "workspace" / "recovery_failure_task"),
        success=False,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Repair the release state after regression.",
            "workspace_subdir": "recovery_failure_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": [],
            "expected_files": ["reports/release_review.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 2,
        },
        termination_reason="verification_failed",
        steps=[
            StepRecord(
                index=1,
                thought="make a bad edit",
                action="code_execute",
                content="printf 'broken\\n' > reports/release_review.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'broken\\n' > reports/release_review.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": False, "reasons": ["unexpected file content: reports/release_review.txt"]},
                failure_signals=["state_regression"],
                state_progress_delta=0.0,
                state_transition={"progress_delta": 0.0, "regressions": ["reports/release_review.txt"]},
            )
        ],
    )
    other_failure = EpisodeRecord(
        task_id="other_failure_task",
        prompt="Run a broken command.",
        workspace=str(tmp_path / "workspace" / "other_failure_task"),
        success=False,
        task_metadata={"benchmark_family": "tooling", "memory_source": "episode"},
        task_contract={
            "prompt": "Run a broken command.",
            "workspace_subdir": "other_failure_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": [],
            "expected_files": [],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 1,
        },
        termination_reason="verification_failed",
        steps=[
            StepRecord(
                index=1,
                thought="fail for another reason",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result={
                    "command": "false",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": False, "reasons": ["command failed"]},
                failure_signals=["command_failed"],
                state_progress_delta=0.0,
                state_transition={"progress_delta": 0.0},
            )
        ],
    )
    memory.save(regression_episode)
    memory.save(other_failure)

    recalled = memory.semantic_recall(
        failure_signals=["state_regression"],
        require_success=False,
        limit=2,
    )

    assert [item["task_id"] for item in recalled] == ["recovery_failure_task"]
    assert recalled[0]["failure_signals"] == ["state_regression"]


def test_graph_memory_recall_delegates_to_episode_memory(tmp_path):
    memory = EpisodeMemory(tmp_path)
    episode = EpisodeRecord(
        task_id="semantic_recall_task",
        prompt="Repair the release report and keep docs stable.",
        workspace=str(tmp_path / "workspace" / "semantic_recall_task"),
        success=True,
        task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
        task_contract={
            "prompt": "Repair the release report and keep docs stable.",
            "workspace_subdir": "semantic_recall_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": [],
            "expected_files": ["reports/release_review.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 2,
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="repair report",
                action="code_execute",
                content="printf 'ready\\n' > reports/release_review.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'ready\\n' > reports/release_review.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                state_progress_delta=1.0,
                state_transition={
                    "progress_delta": 1.0,
                    "edit_patches": [
                        {
                            "path": "reports/release_review.txt",
                            "status": "modified",
                            "patch": "-broken\n+ready\n",
                            "patch_summary": "modified reports/release_review.txt (+1 -1)",
                        }
                    ],
                },
            )
        ],
    )
    memory.save(episode)

    recalled = GraphMemory(memory).recall(changed_paths=["reports/release_review.txt"], limit=1)

    assert [item["task_id"] for item in recalled] == ["semantic_recall_task"]


def test_episode_memory_graph_summary_builds_semantic_prototypes_across_matching_repairs(tmp_path):
    memory = EpisodeMemory(tmp_path)
    for task_id in ("repair_a", "repair_b"):
        memory.save(
            EpisodeRecord(
                task_id=task_id,
                prompt="Repair the release report.",
                workspace=str(tmp_path / "workspace" / task_id),
                success=True,
                task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
                task_contract={
                    "prompt": "Repair the release report.",
                    "workspace_subdir": task_id,
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": [],
                    "expected_files": ["reports/release_review.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {},
                    "max_steps": 2,
                },
                termination_reason="success",
                steps=[
                    StepRecord(
                        index=1,
                        thought="repair report",
                        action="code_execute",
                        content="printf 'ready\\n' > reports/release_review.txt",
                        selected_skill_id=None,
                        command_result={
                            "command": "printf 'ready\\n' > reports/release_review.txt",
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "timed_out": False,
                        },
                        verification={"passed": True, "reasons": ["verification passed"]},
                        state_progress_delta=1.0,
                        state_transition={
                            "progress_delta": 1.0,
                            "newly_updated_report_paths": ["reports/release_review.txt"],
                            "edit_patches": [
                                {
                                    "path": "reports/release_review.txt",
                                    "status": "modified",
                                    "patch": "-broken\n+ready\n",
                                    "patch_summary": "modified reports/release_review.txt (+1 -1)",
                                }
                            ],
                        },
                    )
                ],
            )
        )

    summary = memory.graph_summary()
    prototype = summary["semantic_prototypes"][0]

    assert prototype["episode_count"] == 2
    assert prototype["success_count"] == 2
    assert prototype["task_ids"] == ["repair_a", "repair_b"]
    assert prototype["application_commands"] == ["printf 'ready\\n' > reports/release_review.txt"]
    assert prototype["command_sequences"] == {
        "printf 'ready\\n' > reports/release_review.txt": 2
    }


def test_episode_memory_semantic_prototype_recall_returns_applicable_repair_patterns(tmp_path):
    memory = EpisodeMemory(tmp_path)
    for task_id in ("repair_a", "repair_b"):
        memory.save(
            EpisodeRecord(
                task_id=task_id,
                prompt="Repair the release report.",
                workspace=str(tmp_path / "workspace" / task_id),
                success=True,
                task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
                task_contract={
                    "prompt": "Repair the release report.",
                    "workspace_subdir": task_id,
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": [],
                    "expected_files": ["reports/release_review.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {},
                    "max_steps": 3,
                },
                termination_reason="success",
                steps=[
                    StepRecord(
                        index=1,
                        thought="repair report",
                        action="code_execute",
                        content="mkdir -p reports",
                        selected_skill_id=None,
                        command_result={"command": "mkdir -p reports", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                        verification={"passed": True, "reasons": ["verification passed"]},
                        state_progress_delta=0.5,
                        state_transition={"progress_delta": 0.5},
                    ),
                    StepRecord(
                        index=2,
                        thought="write report",
                        action="code_execute",
                        content="printf 'ready\\n' > reports/release_review.txt",
                        selected_skill_id=None,
                        command_result={
                            "command": "printf 'ready\\n' > reports/release_review.txt",
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "timed_out": False,
                        },
                        verification={"passed": True, "reasons": ["verification passed"]},
                        state_progress_delta=1.0,
                        state_transition={
                            "progress_delta": 1.0,
                            "newly_updated_report_paths": ["reports/release_review.txt"],
                        },
                    ),
                ],
            )
        )

    recalled = memory.semantic_prototype_recall(
        benchmark_family="repository",
        changed_paths=["reports/release_review.txt"],
        require_success=True,
        limit=1,
    )

    assert recalled[0]["success_count"] == 2
    assert recalled[0]["application_commands"][0] == "printf 'ready\\n' > reports/release_review.txt"
    assert "mkdir -p reports || printf 'ready\\n' > reports/release_review.txt" in recalled[0]["command_sequences"]


def test_context_budgeter_surfaces_trusted_retrieval_carryover_chunks():
    task = TaskSpec(
        task_id="release_task",
        prompt="Prepare the release artifact.",
        workspace_subdir="release_task",
        success_command="test -f app/release.txt",
        expected_files=["app/release.txt"],
        expected_file_contents={"app/release.txt": "release ready\n"},
        metadata={"benchmark_family": "repository"},
    )
    state = AgentState(task=task, current_role="planner")
    state.recent_workspace_summary = "release artifact still missing"
    graph_summary = {
        "document_count": 3,
        "benchmark_families": {"repository": 2},
        "trusted_retrieval_successes": 2,
        "trusted_retrieval_command_counts": {
            "mkdir -p app && printf 'release ready\\n' > app/release.txt": 2
        },
    }
    payload = ContextBudgeter(
        KernelConfig(provider="mock", tolbert_context_char_budget=512, tolbert_context_max_chunks=6)
    ).build_payload(
        state=state,
        task_payload={"task_id": "release_task"},
        history_payload=[],
        history_archive={},
        llm_context_packet=None,
        retrieval_plan={},
        transition_preview=None,
        available_skills=[],
        prompt_adjustments=[],
        allowed_actions=["code_execute"],
        graph_summary=graph_summary,
        universe_summary={},
        world_model_summary={"expected_artifacts": ["app/release.txt"]},
        plan=["materialize expected artifact app/release.txt"],
        active_subgoal="materialize expected artifact app/release.txt",
    )

    assert payload["graph_summary"]["trusted_retrieval_successes"] == 2
    assert payload["graph_summary"]["trusted_retrieval_command_counts"] == {
        "mkdir -p app && printf 'release ready\\n' > app/release.txt": 2
    }
    assert any(
        chunk["source"] == "graph_trusted_retrieval_command"
        and "app/release.txt" in chunk["text"]
        for chunk in payload["state_context_chunks"]
    )


def test_context_budgeter_surfaces_trusted_retrieval_procedure_chunks():
    task = TaskSpec(
        task_id="release_sequence_task",
        prompt="Write the release report and then verify it.",
        workspace_subdir="release_sequence_task",
        success_command="pytest -q tests/test_release_status.py",
        expected_files=["reports/status.txt"],
        expected_file_contents={"reports/status.txt": "status ready\n"},
        metadata={"benchmark_family": "repository"},
    )
    state = AgentState(task=task, current_role="planner")
    payload = ContextBudgeter(
        KernelConfig(provider="mock", tolbert_context_char_budget=512, tolbert_context_max_chunks=6)
    ).build_payload(
        state=state,
        task_payload={"task_id": "release_sequence_task"},
        history_payload=[],
        history_archive={},
        llm_context_packet=None,
        retrieval_plan={},
        transition_preview=None,
        available_skills=[],
        prompt_adjustments=[],
        allowed_actions=["code_execute"],
        graph_summary={
            "trusted_retrieval_procedures": [
                {
                    "commands": [
                        "printf 'status ready\\n' > reports/status.txt",
                        "pytest -q tests/test_release_status.py",
                    ],
                    "count": 2,
                }
            ]
        },
        universe_summary={},
        world_model_summary={"expected_artifacts": ["reports/status.txt"]},
        plan=["materialize expected artifact reports/status.txt"],
        active_subgoal="materialize expected artifact reports/status.txt",
    )

    assert payload["graph_summary"]["trusted_retrieval_procedures"] == [
        {
            "commands": [
                "printf 'status ready\\n' > reports/status.txt",
                "pytest -q tests/test_release_status.py",
            ],
            "count": 2,
        }
    ]
    assert any(
        chunk["source"] == "graph_trusted_retrieval_procedure"
        and "pytest -q tests/test_release_status.py" in chunk["text"]
        for chunk in payload["state_context_chunks"]
    )


def test_context_budgeter_surfaces_semantic_patch_chunks():
    task = TaskSpec(
        task_id="release_patch_task",
        prompt="Repair the release report.",
        workspace_subdir="release_patch_task",
        success_command="pytest -q tests/test_release_status.py",
        expected_files=["reports/release_review.txt"],
        expected_file_contents={"reports/release_review.txt": "READY\n"},
        metadata={"benchmark_family": "integration"},
    )
    state = AgentState(task=task, current_role="planner")
    payload = ContextBudgeter(
        KernelConfig(provider="mock", tolbert_context_char_budget=768, tolbert_context_max_chunks=8)
    ).build_payload(
        state=state,
        task_payload={"task_id": "release_patch_task"},
        history_payload=[],
        history_archive={},
        llm_context_packet=None,
        retrieval_plan={},
        transition_preview=None,
        available_skills=[],
        prompt_adjustments=[],
        allowed_actions=["code_execute"],
        graph_summary={
            "semantic_episodes": [
                {
                    "task_id": "release_patch_memory",
                    "benchmark_family": "integration",
                    "memory_source": "episode_replay",
                    "success": True,
                    "verifier_obligations": ["write workflow report reports/release_review.txt and mention ready"],
                    "changed_paths": ["reports/release_review.txt"],
                    "edit_patches": [
                        {
                            "path": "reports/release_review.txt",
                            "status": "modified",
                            "patch_summary": "modified reports/release_review.txt (+1 -1)",
                            "patch_excerpt": "-BROKEN\n+READY",
                        }
                    ],
                    "recovery_trace": {},
                }
            ]
        },
        universe_summary={},
        world_model_summary={"expected_artifacts": ["reports/release_review.txt"]},
        plan=["materialize expected artifact reports/release_review.txt"],
        active_subgoal="materialize expected artifact reports/release_review.txt",
    )

    assert any(
        chunk["source"] == "graph_semantic_episode"
        and "modified reports/release_review.txt (+1 -1)" in chunk["text"]
        and "+READY" in chunk["text"]
        for chunk in payload["state_context_chunks"]
    )


def test_learning_candidates_record_and_match_memory_source(tmp_path):
    episode = EpisodeRecord(
        task_id="episode_memory_task",
        prompt="Replay the earlier task.",
        workspace=str(tmp_path / "workspace" / "episode_memory_task"),
        success=True,
        task_metadata={"benchmark_family": "episode_memory", "memory_source": "episode"},
        task_contract={
            "prompt": "Replay the earlier task.",
            "workspace_subdir": "episode_memory_task",
            "setup_commands": [],
            "success_command": "true",
            "suggested_commands": ["printf 'done\\n' > done.txt"],
            "expected_files": ["done.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"done.txt": "done\n"},
            "max_steps": 2,
            "metadata": {"benchmark_family": "episode_memory", "memory_source": "episode"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="replay the successful command",
                action="code_execute",
                content="printf 'done\\n' > done.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'done\\n' > done.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": []},
            )
        ],
    )

    candidates = compile_episode_learning_candidates(episode)

    assert candidates
    assert all(candidate["memory_source"] == "episode" for candidate in candidates)
    assert all(candidate["memory_sources"] == ["episode"] for candidate in candidates)

    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:episode-match",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_episode",
                        "benchmark_family": "episode_memory",
                        "memory_source": "episode",
                        "memory_sources": ["episode"],
                        "procedure": {"commands": ["printf 'episode\\n' > replay.txt"]},
                    },
                    {
                        "candidate_id": "learning:verifier-match",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_verifier",
                        "benchmark_family": "episode_memory",
                        "memory_source": "verifier",
                        "memory_sources": ["verifier"],
                        "procedure": {"commands": ["printf 'verifier\\n' > replay.txt"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    matches = matching_learning_candidates(
        learning_path,
        task_id="fresh_replay_task",
        benchmark_family="episode_memory",
        memory_source="episode",
    )

    assert [candidate["candidate_id"] for candidate in matches[:2]] == [
        "learning:episode-match",
        "learning:verifier-match",
    ]


def test_learning_candidates_capture_trusted_retrieval_backed_success(tmp_path):
    episode = EpisodeRecord(
        task_id="retrieval_backed_episode_task",
        prompt="Reuse retrieved guidance to create done.txt.",
        workspace=str(tmp_path / "workspace" / "retrieval_backed_episode_task"),
        success=True,
        task_metadata={"benchmark_family": "workflow"},
        task_contract={
            "prompt": "Reuse retrieved guidance to create done.txt.",
            "workspace_subdir": "retrieval_backed_episode_task",
            "setup_commands": [],
            "success_command": "test -f done.txt",
            "suggested_commands": ["printf 'done\\n' > done.txt"],
            "expected_files": ["done.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"done.txt": "done\n"},
            "max_steps": 2,
            "metadata": {"benchmark_family": "workflow"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="follow the retrieved command",
                action="code_execute",
                content="printf 'done\\n' > done.txt",
                selected_skill_id=None,
                command_result={
                    "command": "printf 'done\\n' > done.txt",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                selected_retrieval_span_id="learning:success_skill:seed_task",
                retrieval_influenced=True,
                trust_retrieval=True,
            )
        ],
    )

    candidates = compile_episode_learning_candidates(episode)

    success_candidate = next(candidate for candidate in candidates if candidate["artifact_kind"] == "success_skill_candidate")
    assert success_candidate["retrieval_backed"] is True
    assert success_candidate["retrieval_selected_steps"] == 1
    assert success_candidate["retrieval_influenced_steps"] == 1
    assert success_candidate["trusted_retrieval_steps"] == 1
    assert success_candidate["selected_retrieval_span_ids"] == ["learning:success_skill:seed_task"]
    assert success_candidate["retrieval_backed_commands"] == ["printf 'done\\n' > done.txt"]
    assert success_candidate["quality"] == 0.93


def test_compile_episode_learning_candidates_strengthens_symbol_aligned_syntax_progress_success():
    episode = EpisodeRecord(
        task_id="syntax_learning_task",
        prompt="Apply a localized Python fix.",
        workspace=".",
        success=True,
        task_metadata={"benchmark_family": "workflow"},
        task_contract={
            "task_id": "syntax_learning_task",
            "prompt": "Apply a localized Python fix.",
            "workspace_subdir": "syntax_learning_task",
            "setup_commands": [],
            "success_command": "python -m py_compile service.py",
            "suggested_commands": [],
            "expected_files": ["service.py"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {"service.py": "ok\n"},
            "max_steps": 2,
            "metadata": {"benchmark_family": "workflow"},
        },
        termination_reason="success",
        steps=[
            StepRecord(
                index=1,
                thought="apply localized python edit",
                action="code_execute",
                content="python scripts/structured_edit.py --path service.py",
                selected_skill_id=None,
                command_result={
                    "command": "python scripts/structured_edit.py --path service.py",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                },
                verification={"passed": True, "reasons": ["verification passed"]},
                proposal_source="structured_edit:line_replace",
                proposal_metadata={
                    "path": "service.py",
                    "syntax_motor_progress": {
                        "symbol_aligned": True,
                        "syntax_safe": True,
                        "strong_progress": True,
                        "edited_symbol_fqn": "service.apply_status",
                    }
                },
            )
        ],
    )

    candidates = compile_episode_learning_candidates(episode)

    success_candidate = next(candidate for candidate in candidates if candidate["artifact_kind"] == "success_skill_candidate")
    assert success_candidate["syntax_motor_symbol_aligned_steps"] == 1
    assert success_candidate["syntax_motor_strong_progress_steps"] == 1
    assert success_candidate["syntax_motor_syntax_safe_steps"] == 1
    assert success_candidate["syntax_motor_edited_symbols"] == ["service.apply_status"]
    assert success_candidate["quality"] == 0.88


def test_matching_learning_candidates_prefers_trusted_retrieval_backed_success(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:generic-family",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "other_workflow_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'generic\\n' > result.txt"]},
                        "support_count": 8,
                    },
                    {
                        "candidate_id": "learning:retrieval-backed",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_workflow_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'retrieval\\n' > result.txt"]},
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "selected_retrieval_span_ids": ["learning:success_skill:seed_workflow_task"],
                        "retrieval_backed_commands": ["printf 'retrieval\\n' > result.txt"],
                        "support_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    matches = matching_learning_candidates(
        learning_path,
        task_id="fresh_workflow_task",
        benchmark_family="workflow",
    )

    assert matches
    assert matches[0]["candidate_id"] == "learning:retrieval-backed"


def test_learning_candidates_match_applicable_transfer_tasks(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:transfer-applicable",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_workflow_task",
                        "benchmark_family": "workflow",
                        "applicable_tasks": ["target_workflow_task"],
                        "procedure": {"commands": ["printf 'transfer\\n' > result.txt"]},
                        "support_count": 1,
                    },
                    {
                        "candidate_id": "learning:family-only",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "other_workflow_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'family\\n' > result.txt"]},
                        "support_count": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    matches = matching_learning_candidates(
        learning_path,
        task_id="target_workflow_task",
        benchmark_family="workflow",
    )

    assert matches
    assert matches[0]["candidate_id"] == "learning:transfer-applicable"


def test_learning_candidates_match_replay_task_lineage(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:lineage-match",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "hello_task",
                        "benchmark_family": "verifier_memory",
                        "command": "false",
                        "verification_reasons": ["exit code was 1"],
                    },
                    {
                        "candidate_id": "learning:family-only",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "other_task",
                        "benchmark_family": "verifier_memory",
                        "command": "printf 'oops\\n'",
                        "verification_reasons": ["exit code was 1"],
                        "support_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    matches = matching_learning_candidates(
        learning_path,
        task_id="hello_task_episode_replay_verifier_replay",
        source_task_id="hello_task_episode_replay",
        benchmark_family="verifier_memory",
        memory_source="verifier",
    )

    assert matches
    assert matches[0]["candidate_id"] == "learning:lineage-match"


def test_learning_candidates_match_source_task_aliases_from_metadata(tmp_path):
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:alias-match",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "config_sync_retrieval_task",
                        "benchmark_family": "workflow",
                        "command": "cp template.env config/app.env",
                        "verification_reasons": ["unexpected file content: config/app.env"],
                        "task_metadata": {
                            "source_task": "config_sync_task",
                        },
                        "support_count": 1,
                    },
                    {
                        "candidate_id": "learning:family-only",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "other_workflow_task",
                        "benchmark_family": "workflow",
                        "command": "printf 'oops\\n' > result.txt",
                        "verification_reasons": ["exit code was 1"],
                        "support_count": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    matches = matching_learning_candidates(
        learning_path,
        task_id="fresh_config_repair_task",
        source_task_id="config_sync_task",
        benchmark_family="workflow",
    )

    assert matches
    assert matches[0]["candidate_id"] == "learning:alias-match"


def test_skill_extractor_emits_structured_skill_records(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                },
                "termination_reason": "success",
                "summary": {"failure_types": ["command_failure"]},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "printf 'hello agent kernel\\n' > hello.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "skill_set"
    assert payload["skills"][0]["skill_id"] == "skill:hello_task:primary"
    assert payload["skills"][0]["procedure"]["commands"] == ["printf 'hello agent kernel\\n' > hello.txt"]
    assert payload["skills"][0]["verifier"]["termination_reason"] == "success"
    assert payload["skills"][0]["quality"] >= 0.75
    assert payload["skills"][0]["benchmark_family"] == "workflow"
    assert payload["skills"][0]["reuse_scope"] == "workflow_specific"
    assert payload["skills"][0]["task_contract"]["expected_files"] == ["hello.txt"]


def test_skill_extractor_reads_retrieval_provenance_from_unattended_command_reports(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing hello agent kernel.",
                "workspace": str(tmp_path / "workspace" / "hello_task"),
                "success": True,
                "termination_reason": "success",
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                },
                "commands": [
                    {
                        "index": 1,
                        "command": "printf 'hello agent kernel\\n' > hello.txt",
                        "verification_passed": True,
                        "verification_reasons": ["verification passed"],
                        "decision_source": "trusted_retrieval_carryover_direct",
                        "selected_retrieval_span_id": "learning:seed:hello",
                        "retrieval_influenced": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    skill = payload["skills"][0]
    assert skill["procedure"]["commands"] == ["printf 'hello agent kernel\\n' > hello.txt"]
    assert skill["retrieval_backed"] is True
    assert skill["retrieval_selected_steps"] == 1
    assert skill["retrieval_influenced_steps"] == 1
    assert skill["trusted_retrieval_steps"] == 1
    assert skill["selected_retrieval_span_ids"] == ["learning:seed:hello"]
    assert skill["retrieval_backed_commands"] == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_skill_extractor_includes_postrun_learning_candidates(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:hello_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "hello_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "applicable_tasks": ["hello_task"],
                        "quality": 0.9,
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "workflow"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["skills"]
    assert payload["skills"][0]["source_task_id"] == "hello_task"
    assert payload["skills"][0]["procedure"]["commands"] == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_skill_extractor_preserves_retrieval_backed_learning_provenance(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:hello_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "hello_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "applicable_tasks": ["hello_task"],
                        "quality": 0.9,
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "selected_retrieval_span_ids": ["learning:seed:hello"],
                        "retrieval_backed_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "workflow"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    skill = payload["skills"][0]
    assert skill["retrieval_backed"] is True
    assert skill["retrieval_selected_steps"] == 1
    assert skill["retrieval_influenced_steps"] == 1
    assert skill["trusted_retrieval_steps"] == 1
    assert skill["selected_retrieval_span_ids"] == ["learning:seed:hello"]
    assert skill["retrieval_backed_commands"] == ["printf 'hello agent kernel\\n' > hello.txt"]
    assert skill["quality"] == 0.98


def test_skill_extractor_uses_successful_recovery_case_via_source_task_alias(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:recovery_case:service_release_task_repository_recovery",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "service_release_task_repository_recovery",
                        "parent_task": "service_release_task",
                        "benchmark_family": "repository",
                        "success": True,
                        "quality": 0.82,
                        "recovery_commands": [
                            "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
                        ],
                        "task_contract": {
                            "prompt": "Prepare release outputs.",
                            "workspace_subdir": "service_release_task_repository_recovery",
                            "setup_commands": [],
                            "success_command": "test -f app/release.txt",
                            "suggested_commands": [],
                            "expected_files": ["app/release.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"app/release.txt": "service release ready\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "repository", "source_task": "service_release_task"},
                        },
                        "task_metadata": {
                            "benchmark_family": "repository",
                            "source_task": "service_release_task",
                            "curriculum_kind": "failure_recovery",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    source_task = TaskBank().get("service_release_task")

    assert [skill["skill_id"] for skill in payload["skills"]] == ["skill:service_release_task:postrun"]
    assert payload["skills"][0]["source_task_id"] == "service_release_task"
    assert payload["skills"][0]["procedure"]["commands"] == [
        "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
    ]
    assert payload["skills"][0]["task_contract"]["expected_files"] == source_task.expected_files
    assert payload["skills"][0]["task_contract"]["metadata"]["source_task"] == "service_release_task"
    assert payload["skills"][0]["known_failure_types"] == []


def test_skill_extractor_can_favor_transfer_candidates(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("workflow_task.json").write_text(
        json.dumps(
            {
                "task_id": "workflow_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 2},
                "fragments": [
                    {"kind": "command", "command": "mkdir -p logs", "passed": True},
                    {"kind": "command", "command": "printf 'done\\n' > logs/out.txt", "passed": True},
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output, min_quality=0.75, transfer_only=True)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["generation_strategy"] == "cross_task_transfer"
    assert payload["skills"]
    assert all(skill["reuse_scope"] == "transfer_candidate" for skill in payload["skills"])


def test_skill_extractor_normalizes_workspace_prefixed_commands():
    command = "mkdir -p rewrite_task && printf 'done\\n' > rewrite_task/note.txt"

    normalized = _normalize_command_for_workspace(command, "rewrite_task")

    assert normalized == "printf 'done\\n' > note.txt"


def test_skill_quality_scoring_penalizes_weaker_commands():
    strong = score_skill_quality(
        {
            "success": True,
            "termination_reason": "success",
            "summary": {"failure_types": ["other"], "step_count": 1},
            "steps": [],
        },
        ["printf 'hello\\n' > hello.txt"],
    )
    weak = score_skill_quality(
        {
            "success": True,
            "termination_reason": "",
            "summary": {"failure_types": ["command_failure"], "step_count": 3},
            "steps": [],
        },
        ["echo 'hello' > hello.txt", "cat hello.txt"],
    )

    assert strong > weak


def test_skill_extractor_dedupes_duplicate_procedures_by_quality(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    shared_command = "printf 'hello agent kernel\\n' > hello.txt"
    for name, step_count in (("hello_task_a", 1), ("hello_task_b", 3)):
        (episodes_root / f"{name}.json").write_text(
            json.dumps(
                {
                    "task_id": "hello_task",
                    "success": True,
                    "termination_reason": "success",
                    "summary": {"failure_types": ["other"], "step_count": step_count},
                    "fragments": [{"kind": "command", "command": shared_command, "passed": True}],
                    "steps": [],
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "skills.json"
    extract_successful_command_skills(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert len(payload["skills"]) == 1
    assert payload["skills"][0]["procedure"]["commands"] == [shared_command]
    assert payload["skills"][0]["quality"] >= 0.75
    assert payload["skills"][0]["skill_signature"] == shared_command


def test_operator_extractor_induces_compact_classes(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": [],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                },
                "fragments": [{"kind": "command", "command": "printf 'hello agent kernel\\n' > hello.txt", "passed": True}],
            }
        ),
        encoding="utf-8",
    )
    episodes_root.joinpath("math_task.json").write_text(
        json.dumps(
            {
                "task_id": "math_task",
                "success": True,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create result.txt containing the number 42.",
                    "workspace_subdir": "math_task",
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": [],
                    "expected_files": ["result.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"result.txt": "42\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                },
                "fragments": [{"kind": "command", "command": "printf '42\\n' > result.txt", "passed": True}],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "operators.json"
    extract_operator_classes(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "operator_class_set"
    assert len(payload["operators"]) == 1
    assert payload["operators"][0]["support"] == 2
    assert payload["operators"][0]["support_count"] == 2
    assert payload["operators"][0]["benchmark_families"] == ["micro"]
    assert payload["operators"][0]["steps"] == ["printf 'hello agent kernel\\n' > hello.txt"]
    assert payload["operators"][0]["task_contract"]["expected_files"] == ["hello.txt"]
    assert payload["operators"][0]["operator_kind"] == "single_emit"


def test_operator_extractor_can_require_cross_family_support(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    for name, family, filename, content in (
        ("hello_task.json", "workflow", "hello.txt", "hello agent kernel\n"),
        ("math_task.json", "project", "result.txt", "42\n"),
    ):
        episodes_root.joinpath(name).write_text(
            json.dumps(
                {
                    "task_id": name.removesuffix(".json"),
                    "success": True,
                    "task_metadata": {"benchmark_family": family, "capability": "file_write"},
                    "task_contract": {
                        "prompt": "Create a file.",
                        "workspace_subdir": name.removesuffix(".json"),
                        "setup_commands": [],
                        "success_command": "true",
                        "suggested_commands": [],
                        "expected_files": [filename],
                        "expected_output_substrings": [],
                        "forbidden_files": [],
                        "forbidden_output_substrings": [],
                        "expected_file_contents": {filename: content},
                        "max_steps": 5,
                        "metadata": {"benchmark_family": family, "capability": "file_write"},
                    },
                    "fragments": [{"kind": "command", "command": f"printf '{content.rstrip()}\\n' > {filename}", "passed": True}],
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "operators.json"
    extract_operator_classes(episodes_root, output, cross_family_only=True)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["generation_strategy"] == "cross_family_operator"
    assert payload["operators"]
    assert len(payload["operators"][0]["applicable_benchmark_families"]) >= 2


def test_dedupe_skills_prefers_higher_quality_and_shorter_sequence():
    deduped = dedupe_skills(
        [
            {
                "skill_id": "skill:hello_task:slow",
                "source_task_id": "hello_task",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt", "cat hello.txt"]},
                "quality": 0.8,
            },
            {
                "skill_id": "skill:hello_task:strong",
                "source_task_id": "hello_task",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                "quality": 0.9,
            },
            {
                "skill_id": "skill:hello_task:tie-break",
                "source_task_id": "hello_task",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                "quality": 0.9,
            },
        ]
    )

    assert len(deduped) == 2
    assert deduped[0]["skill_id"] == "skill:hello_task:strong"


def test_dedupe_skills_prefers_retrieval_backed_when_quality_ties():
    deduped = dedupe_skills(
        [
            {
                "skill_id": "skill:hello_task:plain",
                "source_task_id": "hello_task",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                "quality": 0.9,
            },
            {
                "skill_id": "skill:hello_task:retrieval",
                "source_task_id": "hello_task",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                "quality": 0.9,
                "retrieval_backed": True,
                "retrieval_influenced_steps": 1,
                "trusted_retrieval_steps": 1,
                "retrieval_backed_commands": ["printf 'hello\\n' > hello.txt"],
            },
        ]
    )

    assert len(deduped) == 1
    assert deduped[0]["skill_id"] == "skill:hello_task:retrieval"


def test_episode_replay_tasks_load_from_saved_contract(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing hello agent kernel.",
                "workspace": str(tmp_path / "workspace" / "hello_task"),
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                },
                "summary": {
                    "executed_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                },
            }
        ),
        encoding="utf-8",
    )

    tasks = load_episode_replay_tasks(episodes_root)

    assert len(tasks) == 1
    assert tasks[0].task_id == "hello_task_episode_replay"
    assert tasks[0].metadata["benchmark_family"] == "episode_memory"
    assert tasks[0].metadata["source_task"] == "hello_task"
    assert tasks[0].suggested_commands == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_episode_replay_tasks_uplift_frontier_contract_budget(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "git_parallel_merge_acceptance_task.json").write_text(
        json.dumps(
            {
                "task_id": "git_parallel_merge_acceptance_task",
                "prompt": "accept worker branches into main",
                "workspace": str(tmp_path / "workspace" / "git_parallel_merge_acceptance_task"),
                "success": True,
                "task_metadata": {"benchmark_family": "repo_sandbox", "capability": "repo_environment"},
                "task_contract": {
                    "prompt": "accept worker branches into main",
                    "workspace_subdir": "git_parallel_merge_acceptance_task",
                    "setup_commands": [],
                    "success_command": "test -f reports/test_report.txt",
                    "suggested_commands": [
                        "git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                        "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                        "tests/test_api.sh",
                        "tests/test_docs.sh",
                    ],
                    "expected_files": ["reports/test_report.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"reports/test_report.txt": "api suite passed; docs suite passed\n"},
                    "max_steps": 5,
                    "metadata": {
                        "benchmark_family": "repo_sandbox",
                        "capability": "repo_environment",
                        "difficulty": "git_parallel_merge",
                    },
                },
                "summary": {
                    "executed_commands": [
                        "git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                        "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    tasks = load_episode_replay_tasks(episodes_root)

    assert len(tasks) == 1
    assert tasks[0].max_steps >= 20
    assert tasks[0].metadata["origin_benchmark_family"] == "repo_sandbox"


def test_episode_replay_tasks_include_nested_generated_success_documents(tmp_path):
    episodes_root = tmp_path / "episodes"
    generated_root = episodes_root / "generated_success"
    generated_root.mkdir(parents=True)
    generated_root.joinpath("hello_nested_success.json").write_text(
        json.dumps(
            {
                "task_id": "hello_nested_success",
                "prompt": "Create summary.txt stating hello_task succeeded.",
                "workspace": str(tmp_path / "workspace" / "hello_nested_success"),
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                "task_contract": {
                    "prompt": "Create summary.txt stating hello_task succeeded.",
                    "workspace_subdir": "hello_nested_success",
                    "setup_commands": [],
                    "success_command": "test -f summary.txt",
                    "suggested_commands": ["printf 'hello_task succeeded\\n' > summary.txt"],
                    "expected_files": ["summary.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"summary.txt": "hello_task succeeded\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                },
                "summary": {
                    "executed_commands": ["printf 'hello_task succeeded\\n' > summary.txt"],
                },
            }
        ),
        encoding="utf-8",
    )

    tasks = load_episode_replay_tasks(episodes_root)

    assert len(tasks) == 1
    assert tasks[0].metadata["episode_phase"] == "generated_success"
    assert tasks[0].metadata["episode_relative_path"] == "generated_success/hello_nested_success.json"


def test_episode_replay_tasks_skip_synthetic_lineage_sources(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task_episode_replay.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task_episode_replay",
                "prompt": "Replay a replay.",
                "workspace": str(tmp_path / "workspace" / "hello_task_episode_replay"),
                "success": True,
                "task_metadata": {
                    "benchmark_family": "episode_memory",
                    "memory_source": "episode",
                    "source_task": "hello_task",
                },
                "task_contract": {
                    "prompt": "Replay a replay.",
                    "workspace_subdir": "hello_task_episode_replay",
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": ["true"],
                    "expected_files": [],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {},
                    "max_steps": 2,
                    "metadata": {
                        "benchmark_family": "episode_memory",
                        "memory_source": "episode",
                    },
                },
                "summary": {"executed_commands": ["true"]},
            }
        ),
        encoding="utf-8",
    )

    assert load_episode_replay_tasks(episodes_root) == []
    assert load_verifier_replay_tasks(episodes_root, tmp_path / "skills.json") == []


def test_discovered_and_transition_pressure_skip_synthetic_lineage_sources(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "hello_task_episode_replay_verifier_replay.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task_episode_replay_verifier_replay",
                "prompt": "Replay verifier task.",
                "workspace": str(tmp_path / "workspace" / "hello_task_episode_replay_verifier_replay"),
                "success": False,
                "task_metadata": {
                    "benchmark_family": "verifier_memory",
                    "memory_source": "verifier",
                    "source_task": "hello_task",
                },
                "task_contract": {
                    "prompt": "Replay verifier task.",
                    "workspace_subdir": "hello_task_episode_replay_verifier_replay",
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": ["true"],
                    "expected_files": [],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {},
                    "max_steps": 2,
                    "metadata": {
                        "benchmark_family": "verifier_memory",
                        "memory_source": "verifier",
                    },
                },
                "summary": {
                    "failure_types": ["command_failure"],
                    "transition_failures": ["state_regression"],
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_discovered_tasks(episodes_root) == []
    assert load_transition_pressure_tasks(episodes_root) == []


def test_transition_pressure_tasks_inherit_underlying_source_task_suggestions(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    bank = TaskBank()
    retrieval_task = bank.get("integration_failover_drill_retrieval_task")
    assert retrieval_task.suggested_commands == []
    source_task = bank.get("integration_failover_drill_task")
    assert source_task.suggested_commands

    (episodes_root / "integration_failover_drill_retrieval_task.json").write_text(
        json.dumps(
            {
                "task_id": "integration_failover_drill_retrieval_task",
                "prompt": retrieval_task.prompt,
                "workspace": str(tmp_path / "workspace" / "integration_failover_drill_retrieval_task"),
                "success": False,
                "task_metadata": dict(retrieval_task.metadata),
                "task_contract": {
                    "prompt": retrieval_task.prompt,
                    "workspace_subdir": retrieval_task.workspace_subdir,
                    "setup_commands": list(retrieval_task.setup_commands),
                    "success_command": retrieval_task.success_command,
                    "suggested_commands": list(retrieval_task.suggested_commands),
                    "expected_files": list(retrieval_task.expected_files),
                    "expected_output_substrings": list(retrieval_task.expected_output_substrings),
                    "forbidden_files": list(retrieval_task.forbidden_files),
                    "forbidden_output_substrings": list(retrieval_task.forbidden_output_substrings),
                    "expected_file_contents": dict(retrieval_task.expected_file_contents),
                    "max_steps": retrieval_task.max_steps,
                    "metadata": dict(retrieval_task.metadata),
                },
                "summary": {"transition_failures": ["no_state_progress"]},
            }
        ),
        encoding="utf-8",
    )

    tasks = load_transition_pressure_tasks(episodes_root)

    assert len(tasks) == 1
    assert tasks[0].task_id == "integration_failover_drill_retrieval_task_transition_pressure"
    assert tasks[0].suggested_commands == source_task.suggested_commands


def test_skill_replay_tasks_load_from_skill_contract(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "workflow", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_skill_replay_tasks(skills_path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "hello_task_skill_replay"
    assert tasks[0].metadata["benchmark_family"] == "skill_memory"
    assert tasks[0].metadata["source_task"] == "hello_task"
    assert tasks[0].suggested_commands == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_skill_transfer_tasks_use_raw_procedure_on_different_task(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_skill_transfer_tasks(skills_path)

    assert len(tasks) == 1
    assert tasks[0].metadata["memory_source"] == "skill_transfer"
    assert tasks[0].metadata["transfer_target_task"] == "math_task"
    assert tasks[0].suggested_commands == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_operator_replay_tasks_instantiate_target_specific_commands(tmp_path):
    operators_path = tmp_path / "operators.json"
    operators_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_class_set",
                "lifecycle_state": "retained",
                "operators": [
                    {
                        "operator_id": "operator:file_write:micro",
                        "operator_kind": "single_emit",
                        "source_task_ids": ["hello_task"],
                        "applicable_capabilities": ["file_write"],
                        "applicable_benchmark_families": ["bounded"],
                        "template_procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "template_contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_operator_replay_tasks(operators_path)

    assert len(tasks) == 1
    assert tasks[0].metadata["memory_source"] == "operator"
    assert tasks[0].metadata["transfer_target_task"] == "math_task"
    assert tasks[0].suggested_commands == ["printf '42\\n' > result.txt"]


def test_memory_replay_backfills_empty_capability_and_difficulty(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing hello agent kernel.",
                "workspace": str(tmp_path / "workspace" / "hello_task"),
                "success": True,
                "task_metadata": {
                    "benchmark_family": "micro",
                    "capability": "file_write",
                    "difficulty": "seed",
                },
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {
                        "benchmark_family": "",
                        "capability": "",
                        "difficulty": "",
                    },
                },
                "summary": {
                    "executed_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                },
            }
        ),
        encoding="utf-8",
    )

    tasks = load_episode_replay_tasks(episodes_root)

    assert len(tasks) == 1
    assert tasks[0].metadata["capability"] == "file_write"
    assert tasks[0].metadata["difficulty"] == "seed"
    assert tasks[0].metadata["origin_benchmark_family"] == "micro"


def test_verifier_replay_tasks_synthesize_stricter_contracts(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    episodes_root.joinpath("hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "prompt": "Create hello.txt containing hello agent kernel.",
                "workspace": str(tmp_path / "workspace" / "hello_task"),
                "success": True,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write", "difficulty": "seed"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write", "difficulty": "seed"},
                },
                "summary": {"executed_commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
            }
        ),
        encoding="utf-8",
    )
    skills_path = tmp_path / "skills.json"
    skills_path.write_text("[]", encoding="utf-8")

    tasks = load_verifier_replay_tasks(episodes_root, skills_path)

    assert len(tasks) == 1
    assert tasks[0].metadata["benchmark_family"] == "verifier_memory"
    assert tasks[0].metadata["memory_source"] == "verifier"
    assert tasks[0].metadata["verifier_source"] == "episode"
    assert "hello_task_episode_replay/hello.txt" in tasks[0].forbidden_files


def test_tool_candidate_extractor_emits_local_shell_procedures(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "service_mesh_task.json").write_text(
        json.dumps(
            {
                "task_id": "service_mesh_task",
                "success": True,
                "workspace": str(tmp_path / "workspace" / "service_mesh_task"),
                "task_metadata": {"benchmark_family": "integration"},
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 3},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "mkdir -p gateway services reports",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "printf 'routes synced\\n' > gateway/routes.txt",
                        "passed": True,
                    },
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "tool_candidate_set"
    assert payload["candidates"][0]["tool_id"] == "tool:service_mesh_task:primary"
    assert payload["candidates"][0]["kind"] == "local_shell_procedure"
    assert "set -euo pipefail" in payload["candidates"][0]["script_body"]


def test_tool_candidate_extractor_accepts_single_command_repository_episode(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "repo_sync_matrix_task.json").write_text(
        json.dumps(
            {
                "task_id": "repo_sync_matrix_task",
                "success": True,
                "workspace": str(tmp_path / "workspace" / "repo_sync_matrix_task"),
                "task_metadata": {"benchmark_family": "repository"},
                "task_contract": {
                    "prompt": "Sync repository outputs.",
                    "workspace_subdir": "repo_sync_matrix_task",
                    "setup_commands": [],
                    "success_command": "test -f reports/matrix.txt",
                    "suggested_commands": [],
                    "expected_files": ["reports/matrix.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"reports/matrix.txt": "repository sync recorded\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "repository"},
                },
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 1},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "mkdir -p reports && printf 'repository sync recorded\\n' > reports/matrix.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [candidate["tool_id"] for candidate in payload["candidates"]] == ["tool:repo_sync_matrix_task:primary"]
    assert payload["candidates"][0]["procedure"]["commands"] == [
        "mkdir -p reports && printf 'repository sync recorded\\n' > reports/matrix.txt"
    ]


def test_tool_candidate_extractor_skips_unknown_episode_tasks(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "service_release_task_repository_adjacent.json").write_text(
        json.dumps(
            {
                "task_id": "service_release_task_repository_adjacent",
                "success": True,
                "workspace": str(tmp_path / "workspace" / "service_release_task_repository_adjacent"),
                "task_metadata": {"benchmark_family": "repository"},
                "task_contract": {},
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 1},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "printf 'adjacent\\n' > notes.txt",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["candidates"] == []


def test_tool_candidate_extractor_uses_learning_success_candidates(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:service_release_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "service_release_task",
                        "benchmark_family": "repository",
                        "task_contract": {
                            "prompt": "Prepare release outputs.",
                            "workspace_subdir": "service_release_task",
                            "setup_commands": [],
                            "success_command": "test -f app/release.txt",
                            "suggested_commands": [],
                            "expected_files": ["app/release.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"app/release.txt": "service release ready\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "repository"},
                        },
                        "termination_reason": "success",
                        "quality": 0.85,
                        "procedure": {
                            "commands": [
                                "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
                            ]
                        },
                    },
                    {
                        "candidate_id": "learning:success_skill:service_release_task_repository_adjacent",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "service_release_task_repository_adjacent",
                        "benchmark_family": "repository",
                        "task_contract": {},
                        "termination_reason": "success",
                        "quality": 0.9,
                        "procedure": {"commands": ["printf 'adjacent\\n' > notes.txt"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [candidate["tool_id"] for candidate in payload["candidates"]] == ["tool:service_release_task:primary"]
    assert payload["candidates"][0]["benchmark_family"] == "repository"
    assert payload["candidates"][0]["quality"] == 0.85


def test_tool_candidate_extractor_preserves_retrieval_backed_learning_provenance(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:service_release_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "service_release_task",
                        "benchmark_family": "repository",
                        "task_contract": {
                            "prompt": "Prepare release outputs.",
                            "workspace_subdir": "service_release_task",
                            "setup_commands": [],
                            "success_command": "test -f app/release.txt",
                            "suggested_commands": [],
                            "expected_files": ["app/release.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"app/release.txt": "service release ready\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "repository"},
                        },
                        "termination_reason": "success",
                        "quality": 0.85,
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "selected_retrieval_span_ids": ["learning:seed:release"],
                        "retrieval_backed_commands": [
                            "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
                        ],
                        "procedure": {
                            "commands": [
                                "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    candidate = payload["candidates"][0]
    assert candidate["retrieval_backed"] is True
    assert candidate["retrieval_selected_steps"] == 1
    assert candidate["retrieval_influenced_steps"] == 1
    assert candidate["trusted_retrieval_steps"] == 1
    assert candidate["selected_retrieval_span_ids"] == ["learning:seed:release"]
    assert candidate["retrieval_backed_commands"] == [
        "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
    ]
    assert candidate["quality"] == 0.93


def test_tool_candidate_extractor_uses_successful_recovery_case_via_source_task_alias(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:recovery_case:service_release_task_repository_recovery",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "service_release_task_repository_recovery",
                        "parent_task": "service_release_task",
                        "benchmark_family": "repository",
                        "success": True,
                        "quality": 0.82,
                        "recovery_commands": [
                            "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
                        ],
                        "task_contract": {
                            "prompt": "Prepare release outputs.",
                            "workspace_subdir": "service_release_task_repository_recovery",
                            "setup_commands": [],
                            "success_command": "test -f app/release.txt",
                            "suggested_commands": [],
                            "expected_files": ["app/release.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"app/release.txt": "service release ready\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "repository", "source_task": "service_release_task"},
                        },
                        "task_metadata": {
                            "benchmark_family": "repository",
                            "source_task": "service_release_task",
                            "curriculum_kind": "failure_recovery",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    source_task = TaskBank().get("service_release_task")

    assert [candidate["tool_id"] for candidate in payload["candidates"]] == ["tool:service_release_task:primary"]
    assert payload["candidates"][0]["procedure"]["commands"] == [
        "mkdir -p app config tests && printf 'service release ready\\n' > app/release.txt"
    ]
    assert payload["candidates"][0]["task_contract"]["expected_files"] == source_task.expected_files
    assert payload["candidates"][0]["summary"] == source_task.prompt


def test_tool_candidate_extractor_skips_recovery_case_not_aligned_with_aliased_source_task(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:recovery_case:api_contract_task_tool_recovery",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "api_contract_task_tool_recovery",
                        "parent_task": "api_contract_task",
                        "benchmark_family": "tooling",
                        "success": True,
                        "quality": 0.82,
                        "recovery_commands": [
                            "mkdir -p tool && printf 'tool recovery complete\\n' > tool/recovery.txt && printf 'tool recovery verified\\n' > tool/check.txt && printf 'tool recovered\\n' > tool/status.txt"
                        ],
                        "task_contract": {
                            "prompt": "Recover the tool workspace.",
                            "workspace_subdir": "api_contract_task_tool_recovery",
                            "setup_commands": [],
                            "success_command": "test -f tool/recovery.txt",
                            "suggested_commands": [],
                            "expected_files": [
                                "tool/recovery.txt",
                                "tool/check.txt",
                                "tool/status.txt",
                            ],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {
                                "tool/recovery.txt": "tool recovery complete\n",
                                "tool/check.txt": "tool recovery verified\n",
                                "tool/status.txt": "tool recovered\n",
                            },
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "tooling", "source_task": "api_contract_task"},
                        },
                        "task_metadata": {
                            "benchmark_family": "tooling",
                            "source_task": "api_contract_task",
                            "curriculum_kind": "failure_recovery",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["candidates"] == []


def test_tool_candidate_extractor_dedupes_equivalent_command_variants_and_populates_metadata(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    escaped_command = "printf '{\"route\": \"/health\", \"method\": \"GET\"}\n' > api/request.json"
    literal_newline_command = (
        "printf '{\"route\": \"/health\", \"method\": \"GET\"}" + chr(10) + "' > api/request.json"
    )
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:api_contract_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "api_contract_task",
                        "benchmark_family": "tooling",
                        "task_contract": {
                            "prompt": "Prepare the API contract bundle.",
                            "workspace_subdir": "api_contract_task",
                            "setup_commands": [],
                            "success_command": "test -f api/request.json",
                            "suggested_commands": [],
                            "expected_files": ["api/request.json"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {
                                "api/request.json": '{"route": "/health", "method": "GET"}\n'
                            },
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "tooling"},
                        },
                        "termination_reason": "success",
                        "quality": 0.85,
                        "procedure": {"commands": [escaped_command, literal_newline_command]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["procedure"]["commands"] == [escaped_command]
    assert candidate["command"] == escaped_command
    assert candidate["name"] == "api_contract_task_procedure"
    assert candidate["title"] == "api contract task"
    assert candidate["summary"] == "Prepare the API contract bundle."
    assert candidate["script_body"].count("printf '") == 1


def test_tool_candidate_extractor_skips_incomplete_shared_repo_integrator_trace(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "git_parallel_merge_acceptance_task.json").write_text(
        json.dumps(
            {
                "task_id": "git_parallel_merge_acceptance_task",
                "success": True,
                "workspace": str(tmp_path / "workspace" / "git_parallel_merge_acceptance_task"),
                "task_metadata": {"benchmark_family": "repo_sandbox"},
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 1},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "tests/test_docs.sh",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "mkdir -p reports && printf 'docs only\\n' > reports/merge_report.txt",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "printf 'docs suite passed\\n' > reports/test_report.txt",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "git add reports/merge_report.txt reports/test_report.txt",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "git commit -m 'record merge acceptance reports'",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["candidates"] == []


def test_tool_candidate_extractor_marks_complete_shared_repo_integrator_bundle(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    (episodes_root / "git_parallel_merge_acceptance_task.json").write_text(
        json.dumps(
            {
                "task_id": "git_parallel_merge_acceptance_task",
                "success": True,
                "workspace": str(tmp_path / "workspace" / "git_parallel_merge_acceptance_task"),
                "task_metadata": {"benchmark_family": "repo_sandbox"},
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 1},
                "fragments": [
                    {
                        "kind": "command",
                        "command": "git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "tests/test_api.sh",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "tests/test_docs.sh",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": (
                            "mkdir -p reports && printf 'accepted worker/api-status for src/api_status.txt and "
                            "worker/docs-status for docs/status.md into main without collisions\\n' > "
                            "reports/merge_report.txt"
                        ),
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "printf 'api suite passed; docs suite passed\\n' > reports/test_report.txt",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "git add reports/merge_report.txt reports/test_report.txt",
                        "passed": True,
                    },
                    {
                        "kind": "command",
                        "command": "git commit -m 'record merge acceptance reports'",
                        "passed": True,
                    }
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [candidate["tool_id"] for candidate in payload["candidates"]] == [
        "tool:git_parallel_merge_acceptance_task:primary"
    ]
    bundle = payload["candidates"][0]["shared_repo_bundle"]
    assert bundle["role"] == "integrator"
    assert bundle["bundle_complete"] is True
    assert bundle["observed_merged_branches"] == ["worker/api-status", "worker/docs-status"]


def test_tool_candidate_extractor_skips_incomplete_shared_repo_integrator_learning_candidate(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "run_learning_artifacts.json").write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:git_parallel_merge_acceptance_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "git_parallel_merge_acceptance_task",
                        "benchmark_family": "repo_sandbox",
                        "task_contract": {
                            "prompt": "Accept worker branches into main.",
                            "workspace_subdir": "git_parallel_merge_acceptance_task",
                            "expected_files": ["reports/merge_report.txt"],
                            "metadata": {"benchmark_family": "repo_sandbox"},
                        },
                        "termination_reason": "success",
                        "quality": 0.95,
                        "procedure": {
                            "commands": [
                                "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                                "tests/test_docs.sh",
                                "mkdir -p reports && printf 'docs only\\n' > reports/merge_report.txt",
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tools.json"
    extract_tool_candidates(episodes_root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["candidates"] == []


def test_tool_replay_tasks_load_from_tool_contract(tmp_path):
    tools_path = tmp_path / "tool_candidates.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "mkdir -p gateway services reports",
                                "printf 'routes synced\\n' > gateway/routes.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "Prepare integration workspace.",
                            "workspace_subdir": "service_mesh_task",
                            "setup_commands": [],
                            "success_command": "test -f gateway/routes.txt && grep -q '^routes synced$' gateway/routes.txt",
                            "suggested_commands": [],
                            "expected_files": ["gateway/routes.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"gateway/routes.txt": "routes synced\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "integration", "capability": "integration_environment"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "service_mesh_task_tool_replay"
    assert tasks[0].metadata["benchmark_family"] == "tool_memory"
    assert tasks[0].metadata["memory_source"] == "tool"


def test_tool_replay_tasks_uplift_frontier_contract_budget(tmp_path):
    tools_path = tmp_path / "tool_candidates.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:service_release_task:primary",
                        "source_task_id": "service_release_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "mkdir -p app config tests",
                                "printf 'service release ready\\n' > app/release.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "prepare service release repo slice",
                            "workspace_subdir": "service_release_task",
                            "setup_commands": [],
                            "success_command": "test -f app/release.txt",
                            "suggested_commands": [],
                            "expected_files": ["app/release.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"app/release.txt": "service release ready\n"},
                            "max_steps": 5,
                            "metadata": {
                                "benchmark_family": "repository",
                                "difficulty": "cross_component",
                                "capability": "repo_environment",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert len(tasks) == 1
    assert tasks[0].max_steps >= 14
    assert tasks[0].metadata["origin_benchmark_family"] == "repository"


def test_tool_replay_tasks_skip_unpromoted_candidates(tmp_path):
    tools_path = tmp_path / "tool_candidates.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "promotion_stage": "candidate_procedure",
                        "procedure": {
                            "commands": [
                                "mkdir -p gateway services reports",
                                "printf 'routes synced\\n' > gateway/routes.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "Prepare integration workspace.",
                            "workspace_subdir": "service_mesh_task",
                            "setup_commands": [],
                            "success_command": "test -f gateway/routes.txt && grep -q '^routes synced$' gateway/routes.txt",
                            "suggested_commands": [],
                            "expected_files": ["gateway/routes.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"gateway/routes.txt": "routes synced\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "integration", "capability": "integration_environment"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert tasks == []


def test_tool_replay_tasks_skip_incomplete_shared_repo_integrator_candidates(tmp_path):
    tools_path = tmp_path / "tool_candidates.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:git_parallel_merge_acceptance_task:primary",
                        "source_task_id": "git_parallel_merge_acceptance_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                                "tests/test_docs.sh",
                                "mkdir -p reports && printf 'docs only\\n' > reports/merge_report.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "Accept worker branches into main.",
                            "workspace_subdir": "git_parallel_merge_acceptance_task",
                            "expected_files": ["reports/merge_report.txt"],
                            "metadata": {"benchmark_family": "repo_sandbox"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert tasks == []


def test_tool_replay_tasks_skip_rejected_retention_decision(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "retention_decision": {"state": "reject"},
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {"commands": ["mkdir -p manifests env && printf 'service: payments\\nversion: 3\\n' > manifests/service.yaml && printf 'SERVICE_VERSION=3\\n' > env/runtime.env"]},
                        "task_contract": {
                            "prompt": "Update the deployment manifest and env file.",
                            "workspace_subdir": "service_mesh_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["manifests/service.yaml", "env/runtime.env"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {
                                "manifests/service.yaml": "service: payments\nversion: 3\n",
                                "env/runtime.env": "SERVICE_VERSION=3\n",
                            },
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "integration", "capability": "integration_environment"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert tasks == []


def test_tool_replay_tasks_skip_contract_mismatch_against_source_task(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:api_contract_task:primary",
                        "source_task_id": "api_contract_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "mkdir -p tool && printf 'tool recovery complete\\n' > tool/recovery.txt"
                            ]
                        },
                        "task_contract": {
                            "prompt": "Recover the tool workspace.",
                            "workspace_subdir": "api_contract_task_tool_recovery",
                            "setup_commands": [],
                            "success_command": "test -f tool/recovery.txt",
                            "suggested_commands": [],
                            "expected_files": ["tool/recovery.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"tool/recovery.txt": "tool recovery complete\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "tooling", "capability": "tool_environment"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert tasks == []


def test_tool_replay_tasks_skip_candidate_top_level_artifact(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {"commands": ["printf 'routes synced\\n' > gateway/routes.txt"]},
                        "task_contract": {
                            "prompt": "Prepare integration workspace.",
                            "workspace_subdir": "service_mesh_task",
                            "setup_commands": [],
                            "success_command": "true",
                            "suggested_commands": [],
                            "expected_files": ["gateway/routes.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"gateway/routes.txt": "routes synced\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "integration", "capability": "integration_environment"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tool_replay_tasks(tools_path)

    assert tasks == []


def test_skill_replay_tasks_skip_rejected_artifact(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "rejected",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_skill_replay_tasks(skills_path)

    assert tasks == []


def test_skill_replay_tasks_skip_contract_mismatch_against_source_task(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'wrong\\n' > wrong.txt"]},
                        "task_contract": {
                            "prompt": "Create wrong.txt.",
                            "workspace_subdir": "hello_task_bad",
                            "setup_commands": [],
                            "success_command": "test -f wrong.txt",
                            "suggested_commands": [],
                            "expected_files": ["wrong.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"wrong.txt": "wrong\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_skill_replay_tasks(skills_path)

    assert tasks == []


def test_skill_replay_tasks_skip_proposed_artifact(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "task_contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                            "suggested_commands": [],
                            "expected_files": ["hello.txt"],
                            "expected_output_substrings": [],
                            "forbidden_files": [],
                            "forbidden_output_substrings": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "max_steps": 5,
                            "metadata": {"benchmark_family": "micro", "capability": "file_write"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_skill_replay_tasks(skills_path)

    assert tasks == []


def test_benchmark_candidate_tasks_skip_rejected_artifact(tmp_path):
    candidates_path = tmp_path / "benchmarks.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "benchmark_candidate_set",
                "lifecycle_state": "proposed",
                "retention_decision": {"state": "reject"},
                "proposals": [
                    {
                        "proposal_id": "benchmark:hello_task:failure_cluster",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "kind": "failure_cluster",
                        "prompt": "Create hello.txt containing the string hello agent kernel.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_benchmark_candidate_tasks(candidates_path)

    assert tasks == []


def test_benchmark_candidate_tasks_skip_invalid_top_level_lifecycle_state(tmp_path):
    candidates_path = tmp_path / "benchmarks.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "benchmark_candidate_set",
                "lifecycle_state": "candidate",
                "proposals": [
                    {
                        "proposal_id": "benchmark:hello_task:failure_cluster",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "kind": "failure_cluster",
                        "prompt": "Create hello.txt containing the string hello agent kernel.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_benchmark_candidate_tasks(candidates_path)

    assert tasks == []


def test_verifier_candidate_tasks_skip_rejected_artifact(tmp_path):
    candidates_path = tmp_path / "verifiers.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "verifier_candidate_set",
                "lifecycle_state": "proposed",
                "retention_decision": {"state": "reject"},
                "proposals": [
                    {
                        "proposal_id": "verifier:hello_task:strict",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "forbidden_output_substrings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_verifier_candidate_tasks(candidates_path)

    assert tasks == []


def test_verifier_candidate_tasks_skip_invalid_top_level_lifecycle_state(tmp_path):
    candidates_path = tmp_path / "verifiers.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "verifier_candidate_set",
                "lifecycle_state": "candidate",
                "proposals": [
                    {
                        "proposal_id": "verifier:hello_task:strict",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "forbidden_output_substrings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_verifier_candidate_tasks(candidates_path)

    assert tasks == []


def test_verifier_candidate_tasks_preserve_semantic_verifier_contracts(tmp_path):
    candidates_path = tmp_path / "verifiers.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "verifier_candidate_set",
                "lifecycle_state": "proposed",
                "proposals": [
                    {
                        "proposal_id": "verifier:hello_task:strict",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "contract": {
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                            "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                            "forbidden_output_substrings": [],
                            "semantic_verifier": {
                                "kind": "behavioral_semantic",
                                "behavior_checks": [
                                    {
                                        "label": "hello smoke",
                                        "argv": ["/bin/sh", "-lc", "printf 'hello agent kernel\\n'"],
                                        "expect_exit_code": 0,
                                        "stdout_must_contain": ["hello agent kernel"],
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_verifier_candidate_tasks(candidates_path)

    assert len(tasks) == 1
    assert tasks[0].metadata["semantic_verifier"]["kind"] == "behavioral_semantic"
    assert tasks[0].metadata["semantic_verifier"]["behavior_checks"][0]["label"] == "hello smoke"
