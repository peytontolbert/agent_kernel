import json

from agent_kernel.config import KernelConfig
from agent_kernel.extractors import (
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
from agent_kernel.memory import EpisodeMemory
from agent_kernel.schemas import EpisodeRecord, StepRecord
from agent_kernel.task_bank import (
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
