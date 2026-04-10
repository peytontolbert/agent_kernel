from datetime import UTC, datetime
from pathlib import Path
import importlib.util
import json
from io import StringIO
import sys

from agent_kernel.config import KernelConfig
from evals.metrics import EvalMetrics


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_success_episode(path: Path, *, task_id: str, family: str = "workflow") -> None:
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "success": True,
                "task_metadata": {"benchmark_family": family, "capability": "file_write", "difficulty": "seed"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": task_id,
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": family, "capability": "file_write", "difficulty": "seed"},
                },
                "termination_reason": "success",
                "summary": {"failure_types": [], "step_count": 1, "executed_commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "fragments": [{"kind": "command", "command": "printf 'hello agent kernel\\n' > hello.txt", "passed": True}],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )


def _write_failed_episode(path: Path, *, task_id: str, family: str = "workflow") -> None:
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "success": False,
                "task_metadata": {"benchmark_family": family, "capability": "file_write", "difficulty": "seed"},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": task_id,
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": family, "capability": "file_write", "difficulty": "seed"},
                },
                "summary": {
                    "failure_types": ["missing_expected_file", "command_failure"],
                    "executed_commands": ["false", "printf 'hello agent kernel\\n' > hello.txt"],
                },
                "fragments": [
                    {"kind": "command", "command": "false", "passed": False},
                    {"kind": "failure", "reason": "missing expected file: hello.txt", "failure_types": ["missing_expected_file"]},
                ],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )


def _write_tool_candidate_artifact(path: Path, *, empty: bool = False) -> None:
    candidates = []
    if not empty:
        candidates = [
            {
                "spec_version": "asi_v1",
                "tool_id": "tool:hello_task:primary",
                "kind": "local_shell_procedure",
                "lifecycle_state": "candidate",
                "promotion_stage": "candidate_procedure",
                "source_task_id": "hello_task",
                "benchmark_family": "bounded",
                "quality": 0.91,
                "script_name": "hello_task_tool.sh",
                "script_body": "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'hello\\n' > hello.txt\n",
                "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                "task_contract": {
                    "prompt": "Create hello.txt containing hello.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt && grep -q '^hello$' hello.txt",
                    "suggested_commands": [],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "bounded", "capability": "file_write"},
                },
                "verifier": {"termination_reason": "success"},
            }
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "retention_gate": {
                    "min_quality": 0.75,
                    "require_replay_verification": True,
                    "require_future_task_gain": True,
                },
                "generation_strategy": "procedure_promotion",
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )


def test_extract_skill_tool_and_operator_scripts_use_config_paths(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_success_episode(episodes / "hello_task.json", task_id="hello_task", family="workflow")
    _write_success_episode(episodes / "math_task.json", task_id="math_task", family="project")

    config = KernelConfig(
        trajectories_root=episodes,
        skills_path=tmp_path / "skills" / "command_skills.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
    )

    for script_name, output_path, kind in (
        ("extract_skills.py", config.skills_path, "skill_set"),
        ("extract_tools.py", config.tool_candidates_path, "tool_candidate_set"),
        ("extract_operators.py", config.operator_classes_path, "operator_class_set"),
    ):
        module = _load_script_module(script_name)
        monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
        monkeypatch.setattr(sys, "argv", [script_name])
        stream = StringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        module.main()

        assert output_path.exists()
        assert stream.getvalue().strip() == str(output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["artifact_kind"] == kind


def test_generation_scripts_expose_non_default_artifact_strategies(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_success_episode(episodes / "hello_task.json", task_id="hello_task", family="workflow")
    _write_success_episode(episodes / "math_task.json", task_id="math_task", family="project")
    _write_failed_episode(episodes / "hello_task_failed.json", task_id="hello_task", family="workflow")

    config = KernelConfig(
        trajectories_root=episodes,
        skills_path=tmp_path / "skills" / "command_skills.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "tolbert_model_artifact.json",
        qwen_adapter_artifact_path=tmp_path / "qwen_adapter" / "qwen_adapter_artifact.json",
        verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        world_model_proposals_path=tmp_path / "world_model" / "world_model_proposals.json",
        trust_proposals_path=tmp_path / "trust" / "trust_proposals.json",
        recovery_proposals_path=tmp_path / "recovery" / "recovery_proposals.json",
        delegation_proposals_path=tmp_path / "delegation" / "delegation_proposals.json",
        operator_policy_proposals_path=tmp_path / "operator_policy" / "operator_policy_proposals.json",
        transition_model_proposals_path=tmp_path / "transition_model" / "transition_model_proposals.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        capability_modules_path=tmp_path / "config" / "capabilities.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )

    cases = [
        ("extract_skills.py", ["extract_skills.py", "--transfer-only", "--min-quality", "0.75"], config.skills_path, "generation_strategy", "cross_task_transfer"),
        ("extract_tools.py", ["extract_tools.py", "--replay-hardening"], config.tool_candidates_path, "generation_strategy", "script_hardening"),
        ("extract_operators.py", ["extract_operators.py", "--cross-family-only"], config.operator_classes_path, "generation_strategy", "cross_family_operator"),
        ("synthesize_benchmarks.py", ["synthesize_benchmarks.py", "--focus", "confidence"], config.benchmark_candidates_path, "generation_focus", "confidence"),
        ("propose_retrieval_update.py", ["propose_retrieval_update.py", "--focus", "breadth"], config.retrieval_proposals_path, "generation_focus", "breadth"),
        ("propose_tolbert_model_update.py", ["propose_tolbert_model_update.py", "--focus", "discovered_task_adaptation"], config.tolbert_model_artifact_path, "generation_focus", "discovered_task_adaptation"),
        ("propose_qwen_adapter_update.py", ["propose_qwen_adapter_update.py", "--focus", "teacher_shadow", "--base-model", "Qwen/Qwen3.5-9B"], config.qwen_adapter_artifact_path, "generation_focus", "teacher_shadow"),
        ("propose_world_model_update.py", ["propose_world_model_update.py", "--focus", "workflow_alignment"], config.world_model_proposals_path, "generation_focus", "workflow_alignment"),
        ("propose_trust_update.py", ["propose_trust_update.py", "--focus", "safety"], config.trust_proposals_path, "generation_focus", "safety"),
        ("propose_recovery_update.py", ["propose_recovery_update.py", "--focus", "rollback_safety"], config.recovery_proposals_path, "generation_focus", "rollback_safety"),
        ("propose_delegation_update.py", ["propose_delegation_update.py", "--focus", "worker_depth"], config.delegation_proposals_path, "generation_focus", "worker_depth"),
        ("propose_operator_policy_update.py", ["propose_operator_policy_update.py", "--focus", "git_http_scope"], config.operator_policy_proposals_path, "generation_focus", "git_http_scope"),
        ("propose_transition_model_update.py", ["propose_transition_model_update.py", "--focus", "regression_guard"], config.transition_model_proposals_path, "generation_focus", "regression_guard"),
        ("propose_capability_update.py", ["propose_capability_update.py", "--focus", "tooling_surface"], config.capability_modules_path, "generation_focus", "tooling_surface"),
        ("synthesize_verifiers.py", ["synthesize_verifiers.py", "--strategy", "false_failure_guard"], config.verifier_contracts_path, "generation_strategy", "false_failure_guard"),
    ]
    for script_name, argv, output_path, field, expected in cases:
        module = _load_script_module(script_name)
        monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
        if hasattr(module, "run_eval"):
            monkeypatch.setattr(
                module,
                "run_eval",
                lambda **kwargs: EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2),
            )
        if hasattr(module, "build_tolbert_model_candidate_artifact"):
            monkeypatch.setattr(
                module,
                "build_tolbert_model_candidate_artifact",
                lambda **kwargs: {
                    "spec_version": "asi_v1",
                    "artifact_kind": "tolbert_model_bundle",
                    "lifecycle_state": "candidate",
                    "generation_focus": "discovered_task_adaptation",
                    "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
                    "dataset_manifest": {"total_examples": 4},
                    "decoder_policy": {"allow_retrieval_guidance": True, "allow_skill_commands": True, "allow_task_suggestions": True, "allow_stop_decision": True, "min_stop_completion_ratio": 0.95, "max_task_suggestions": 3},
                    "rollout_policy": {"predicted_progress_gain_weight": 3.0, "predicted_conflict_penalty_weight": 4.0, "predicted_preserved_bonus_weight": 1.0, "predicted_workflow_bonus_weight": 1.5, "latent_progress_bonus_weight": 1.0, "latent_risk_penalty_weight": 2.0, "recover_from_stall_bonus_weight": 1.5, "stop_completion_weight": 8.0, "stop_missing_expected_penalty_weight": 6.0, "stop_forbidden_penalty_weight": 6.0, "stop_preserved_penalty_weight": 4.0, "stable_stop_bonus_weight": 1.5},
                    "runtime_paths": {"config_path": "config.json", "checkpoint_path": "checkpoint.pt", "nodes_path": "nodes.jsonl", "label_map_path": "label_map.json", "source_spans_paths": ["spans.jsonl"], "cache_paths": ["cache.pt"]},
                    "proposals": [{"area": "discovered_task_adaptation", "priority": 5, "reason": "test"}],
                },
            )
        if hasattr(module, "build_qwen_adapter_candidate_artifact"):
            monkeypatch.setattr(
                module,
                "build_qwen_adapter_candidate_artifact",
                lambda **kwargs: {
                    "spec_version": "asi_v1",
                    "artifact_kind": "qwen_adapter_bundle",
                    "lifecycle_state": "candidate",
                    "generation_focus": "teacher_shadow",
                    "runtime_role": "support_runtime",
                    "training_objective": "qlora_sft",
                    "base_model_name": "Qwen/Qwen3.5-9B",
                    "training_dataset_manifest": {"total_examples": 4},
                    "runtime_policy": {"allow_primary_routing": False, "allow_teacher_generation": True},
                    "retention_gate": {"disallow_liftoff_authority": True},
                    "runtime_paths": {"adapter_output_dir": "adapter", "merged_output_dir": "merged"},
                },
            )
        monkeypatch.setattr(sys, "argv", argv)
        stream = StringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        module.main()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload[field] == expected


def test_report_long_horizon_persistence_summarizes_eval_trajectories(monkeypatch):
    module = _load_script_module("report_long_horizon_persistence.py")
    metrics = EvalMetrics(
        total=2,
        passed=1,
        task_trajectories={
            "lh_task": {
                "task_id": "lh_task",
                "benchmark_family": "integration",
                "difficulty": "long_horizon",
                "success": True,
                "termination_reason": "success",
                "steps": [
                    {
                        "index": 1,
                        "world_model_horizon": "long_horizon",
                        "state_progress_delta": 0.3,
                        "state_no_progress": False,
                        "state_regressed": False,
                        "state_regression_count": 0,
                        "active_subgoal": "update workflow path src/release_state.txt",
                        "acting_role": "executor",
                    },
                    {
                        "index": 2,
                        "world_model_horizon": "long_horizon",
                        "state_progress_delta": 0.0,
                        "state_no_progress": True,
                        "state_regressed": False,
                        "state_regression_count": 0,
                        "active_subgoal": "update workflow path src/release_state.txt",
                        "acting_role": "executor",
                    },
                    {
                        "index": 3,
                        "world_model_horizon": "long_horizon",
                        "state_progress_delta": 0.2,
                        "state_no_progress": False,
                        "state_regressed": False,
                        "state_regression_count": 0,
                        "active_subgoal": "materialize expected artifact reports/release.md",
                        "acting_role": "planner",
                    },
                ],
            },
            "bounded_task": {
                "task_id": "bounded_task",
                "benchmark_family": "workflow",
                "difficulty": "seed",
                "success": False,
                "termination_reason": "step_limit",
                "steps": [
                    {
                        "index": 1,
                        "world_model_horizon": "bounded",
                        "state_progress_delta": 0.0,
                        "state_no_progress": True,
                        "state_regressed": False,
                        "state_regression_count": 0,
                        "active_subgoal": "",
                        "acting_role": "executor",
                    }
                ],
            },
        },
    )

    summary = module.summarize_long_horizon_persistence(metrics)

    assert summary["long_horizon_task_count"] == 1
    assert summary["long_horizon_steps"] == 3
    assert summary["productive_long_horizon_steps"] == 2
    assert summary["pressure_events"] == 1
    assert summary["recovery_response_events"] == 1
    assert summary["subgoal_refresh_count"] == 1
    assert summary["max_long_horizon_streak"] == 3
    assert summary["rates"]["productive_long_horizon_step_rate"] == 0.6667
    assert summary["rates"]["recovery_response_rate"] == 1.0


def test_report_long_horizon_persistence_cli_supports_json(monkeypatch):
    module = _load_script_module("report_long_horizon_persistence.py")
    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(
            total=1,
            passed=1,
            task_trajectories={
                "lh_task": {
                    "task_id": "lh_task",
                    "benchmark_family": "integration",
                    "difficulty": "long_horizon",
                    "success": True,
                    "termination_reason": "success",
                    "steps": [
                        {
                            "index": 1,
                            "world_model_horizon": "long_horizon",
                            "state_progress_delta": 0.1,
                            "state_no_progress": False,
                            "state_regressed": False,
                            "state_regression_count": 0,
                            "active_subgoal": "update workflow path src/release_state.txt",
                            "acting_role": "executor",
                        }
                    ],
                }
            },
        ),
    )
    monkeypatch.setattr(sys, "argv", ["report_long_horizon_persistence.py", "--json"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["long_horizon_task_count"] == 1
    assert payload["long_horizon_steps"] == 1


def test_synthesize_benchmark_and_verifier_scripts_use_config_paths(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_success_episode(episodes / "hello_task.json", task_id="hello_task", family="workflow")
    _write_failed_episode(episodes / "hello_task_failed.json", task_id="hello_task", family="workflow")

    config = KernelConfig(
        trajectories_root=episodes,
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "tolbert_model_artifact.json",
        verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
        world_model_proposals_path=tmp_path / "world_model" / "world_model_proposals.json",
        trust_proposals_path=tmp_path / "trust" / "trust_proposals.json",
        recovery_proposals_path=tmp_path / "recovery" / "recovery_proposals.json",
        delegation_proposals_path=tmp_path / "delegation" / "delegation_proposals.json",
        operator_policy_proposals_path=tmp_path / "operator_policy" / "operator_policy_proposals.json",
        transition_model_proposals_path=tmp_path / "transition_model" / "transition_model_proposals.json",
        capability_modules_path=tmp_path / "config" / "capabilities.json",
    )


def test_report_supervised_frontier_ingests_scoped_cycles_and_dedupes_candidates(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_a = improvement_root / "cycles_scope_a.jsonl"
    cycles_b = improvement_root / "cycles_scope_b.jsonl"
    cycles_timeout = improvement_root / "cycles_scope_timeout.jsonl"
    candidate_path = candidates_root / "transition_model" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "transition_model_policy_set", "rules": ["a"]}), encoding="utf-8")

    class FakeConfig:
        improvement_reports_dir = reports_dir
        improvement_cycles_path = improvement_root / "cycles.jsonl"
        candidate_artifacts_root = candidates_root

        def ensure_directories(self):
            self.improvement_reports_dir.mkdir(parents=True, exist_ok=True)

        def uses_sqlite_storage(self):
            return False

    def _write_cycles(path: Path, *, scope_id: str, timed_out: bool = False, generated: bool = True):
        records = [
            {
                "cycle_id": f"cycle:{scope_id}",
                "state": "observe",
                "subsystem": "transition_model" if generated else "observation",
                "metrics_summary": {
                    "protocol": "human_guided",
                    "scope_id": scope_id,
                    "scoped_run": True,
                    "total": 5,
                    "passed": 2 if not timed_out else 0,
                    "pass_rate": 0.4 if not timed_out else 0.0,
                    "generated_total": 1 if generated and not timed_out else 0,
                    "generated_passed": 1 if generated and not timed_out else 0,
                    "generated_pass_rate": 1.0 if generated and not timed_out else 0.0,
                    "observation_curriculum_followups": (
                        [
                            {"kind": "generated_failure", "generated_total": 1, "generated_passed": 1}
                        ]
                        if generated and not timed_out
                        else []
                    ),
                    "observation_timed_out": timed_out,
                    "observation_budget_exceeded": timed_out,
                    "observation_warning": "timed out" if timed_out else "",
                    "observation_elapsed_seconds": 4.2 if timed_out else 1.1,
                    "observation_current_task_timeout_budget_source": (
                        "prestep_subphase:tolbert_query" if timed_out else ""
                    ),
                    "observation_current_task_timeout_budget_seconds": 3.0 if timed_out else 0.0,
                    "observation_returncode": 124 if timed_out else 0,
                },
            }
        ]
        if generated:
            records.append(
                {
                    "cycle_id": f"cycle:{scope_id}",
                    "state": "select",
                    "subsystem": "transition_model",
                    "metrics_summary": {
                        "protocol": "human_guided",
                        "scope_id": scope_id,
                        "scoped_run": True,
                        "selected_variant_id": "regression_guard",
                    },
                }
            )
            records.append(
                {
                    "cycle_id": f"cycle:{scope_id}",
                    "state": "generate",
                    "subsystem": "transition_model",
                    "artifact_kind": "transition_model_policy_set",
                    "artifact_path": str(candidate_path),
                    "candidate_artifact_path": str(candidate_path),
                    "metrics_summary": {"protocol": "human_guided", "scope_id": scope_id, "scoped_run": True},
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    _write_cycles(cycles_a, scope_id="scope_a")
    _write_cycles(cycles_b, scope_id="scope_b")
    _write_cycles(cycles_timeout, scope_id="scope_timeout", timed_out=True, generated=False)

    monkeypatch.setattr(module, "KernelConfig", lambda: FakeConfig())
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "supervised_parallel_frontier_report"
    assert payload["summary"]["scoped_run_count"] == 3
    assert payload["summary"]["completed_runs"] == 3
    assert payload["summary"]["successful_runs"] == 2
    assert payload["summary"]["healthy_runs"] == 2
    assert payload["summary"]["warning_runs"] == 0
    assert payload["summary"]["primary_passed_runs"] == 2
    assert payload["summary"]["generated_success_runs"] == 2
    assert payload["summary"]["generated_failure_runs"] == 2
    assert payload["summary"]["primary_total"] == 15
    assert payload["summary"]["primary_passed"] == 4
    assert payload["summary"]["generated_success_total"] == 2
    assert payload["summary"]["generated_success_passed"] == 2
    assert payload["summary"]["generated_failure_total"] == 2
    assert payload["summary"]["generated_failure_passed"] == 2
    assert payload["summary"]["generated_candidate_runs"] == 2
    assert payload["summary"]["timed_out_runs"] == 1
    assert payload["summary"]["budget_exceeded_runs"] == 1
    assert payload["summary"]["deduped_runs"] == 1
    assert payload["summary"]["timeout_budget_sources"] == {"prestep_subphase:tolbert_query": 1}
    assert len(payload["frontier_candidates"]) == 2
    assert payload["frontier_candidates"][0]["selected_variant_id"] == "regression_guard"
    assert payload["frontier_candidates"][0]["duplicate_count"] == 1
    assert sorted(payload["frontier_candidates"][0]["duplicate_scope_ids"]) == ["scope_b"]
    assert payload["frontier_candidates"][0]["status"] == "healthy"
    assert payload["frontier_candidates"][0]["run_succeeded"] is True
    assert payload["frontier_candidates"][0]["healthy_run"] is True
    assert payload["frontier_candidates"][0]["primary_passed"] == 2
    assert payload["frontier_candidates"][0]["generated_success_passed"] == 1
    assert payload["frontier_candidates"][0]["generated_failure_passed"] == 1
    assert payload["frontier_candidates"][1]["scope_id"] == "scope_timeout"
    assert payload["frontier_candidates"][1]["observation_timed_out"] is True
    assert payload["frontier_candidates"][1]["observation_budget_exceeded"] is True
    assert payload["frontier_candidates"][1]["status"] == "failed"


def test_report_supervised_frontier_marks_over_budget_success_as_warning_not_failure(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_warning.jsonl"
    candidate_path = candidates_root / "retrieval" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:scope_warning",
            "state": "observe",
            "subsystem": "retrieval",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_warning",
                "scoped_run": True,
                "total": 5,
                "passed": 5,
                "pass_rate": 1.0,
                "generated_total": 5,
                "generated_passed": 5,
                "generated_pass_rate": 1.0,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": True,
                "observation_warning": "observation exceeded budget 70.0s with elapsed 70.7s",
                "observation_elapsed_seconds": 105.268,
            },
        },
        {
            "cycle_id": "cycle:scope_warning",
            "state": "select",
            "subsystem": "retrieval",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_warning",
                "scoped_run": True,
                "selected_variant_id": "confidence_gating",
            },
        },
        {
            "cycle_id": "cycle:scope_warning",
            "state": "generate",
            "subsystem": "retrieval",
            "artifact_kind": "retrieval_policy_set",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_warning", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["successful_runs"] == 1
    assert payload["summary"]["warning_runs"] == 1
    assert payload["summary"]["timed_out_runs"] == 0
    assert payload["summary"]["budget_exceeded_runs"] == 1
    assert payload["frontier_candidates"][0]["status"] == "completed_with_warnings"
    assert payload["frontier_candidates"][0]["run_succeeded"] is True
    assert payload["frontier_candidates"][0]["healthy_run"] is False
    assert payload["frontier_candidates"][0]["observation_timed_out"] is False
    assert payload["frontier_candidates"][0]["observation_budget_exceeded"] is True


def test_report_supervised_frontier_prefers_records_matching_scoped_filename(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_live.jsonl"
    old_candidate = candidates_root / "transition_model" / "old.json"
    live_candidate = candidates_root / "transition_model" / "live.json"
    old_candidate.parent.mkdir(parents=True, exist_ok=True)
    old_candidate.write_text(json.dumps({"artifact_kind": "transition_model_policy_set", "rules": ["old"]}), encoding="utf-8")
    live_candidate.write_text(json.dumps({"artifact_kind": "transition_model_policy_set", "rules": ["live"]}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:old",
            "state": "observe",
            "subsystem": "transition_model",
            "metrics_summary": {"protocol": "human_guided", "scope_id": "", "scoped_run": False},
        },
        {
            "cycle_id": "cycle:old",
            "state": "generate",
            "subsystem": "transition_model",
            "artifact_kind": "transition_model_policy_set",
            "artifact_path": str(old_candidate),
            "candidate_artifact_path": str(old_candidate),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "", "scoped_run": False},
        },
        {
            "cycle_id": "cycle:scope_live",
            "state": "observe",
            "subsystem": "transition_model",
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_live", "scoped_run": True},
        },
        {
            "cycle_id": "cycle:scope_live",
            "state": "select",
            "subsystem": "transition_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_live",
                "scoped_run": True,
                "selected_variant_id": "repeat_avoidance",
            },
        },
        {
            "cycle_id": "cycle:scope_live",
            "state": "generate",
            "subsystem": "transition_model",
            "artifact_kind": "transition_model_policy_set",
            "artifact_path": str(live_candidate),
            "candidate_artifact_path": str(live_candidate),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_live", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["scoped_run_count"] == 1
    assert payload["frontier_candidates"][0]["scope_id"] == "scope_live"
    assert payload["frontier_candidates"][0]["selected_variant_id"] == "repeat_avoidance"
    assert payload["frontier_candidates"][0]["candidate_artifact_path"] == str(live_candidate)


def test_report_supervised_frontier_reads_generated_success_counts_from_followups(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_followup.jsonl"
    candidate_path = candidates_root / "tooling" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "tool_candidate_set", "rules": ["a"]}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:scope_followup",
            "state": "observe",
            "subsystem": "tooling",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_followup",
                "scoped_run": True,
                "total": 5,
                "passed": 2,
                "pass_rate": 0.4,
                "generated_pass_rate": 1.0,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": False,
                "observation_warning": "",
                "observation_curriculum_followups": [
                    {
                        "kind": "generated_success",
                        "generated_total": 2,
                        "generated_passed": 2,
                        "timed_out": False,
                        "warning": "",
                    },
                    {
                        "kind": "generated_failure",
                        "generated_total": 1,
                        "generated_passed": 0,
                        "timed_out": False,
                        "warning": "",
                    }
                ],
            },
        },
        {
            "cycle_id": "cycle:scope_followup",
            "state": "select",
            "subsystem": "tooling",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_followup",
                "scoped_run": True,
                "selected_variant_id": "procedure_promotion",
            },
        },
        {
            "cycle_id": "cycle:scope_followup",
            "state": "generate",
            "subsystem": "tooling",
            "artifact_kind": "tool_candidate_set",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_followup", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["generated_success_runs"] == 1
    assert payload["summary"]["generated_success_total"] == 2
    assert payload["summary"]["generated_success_passed"] == 2
    assert payload["summary"]["generated_failure_runs"] == 0
    assert payload["summary"]["generated_failure_total"] == 1
    assert payload["summary"]["generated_failure_passed"] == 0
    assert payload["scoped_runs"][0]["generated_success_total"] == 2
    assert payload["scoped_runs"][0]["generated_success_passed"] == 2
    assert payload["scoped_runs"][0]["generated_failure_total"] == 1
    assert payload["scoped_runs"][0]["generated_failure_passed"] == 0


def test_report_supervised_frontier_includes_tooling_shared_repo_bundle_summary(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_shared_repo.jsonl"
    candidate_path = candidates_root / "tooling" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "candidates": [
                    {
                        "tool_id": "tool:worker",
                        "quality": 0.8,
                        "procedure": {"commands": ["git checkout worker/api-status"]},
                        "shared_repo_bundle": {
                            "shared_repo_id": "repo_sandbox_parallel_merge",
                            "worker_branch": "worker/api-status",
                            "role": "worker",
                            "bundle_complete": True,
                        },
                    },
                    {
                        "tool_id": "tool:integrator",
                        "quality": 0.9,
                        "procedure": {
                            "commands": [
                                "git merge --no-ff worker/api-status",
                                "git merge --no-ff worker/docs-status",
                            ]
                        },
                        "shared_repo_bundle": {
                            "shared_repo_id": "repo_sandbox_parallel_merge",
                            "role": "integrator",
                            "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = [
        {
            "cycle_id": "cycle:scope_shared_repo",
            "state": "observe",
            "subsystem": "tooling",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_shared_repo",
                "scoped_run": True,
                "total": 3,
                "passed": 2,
                "pass_rate": 0.67,
                "generated_total": 1,
                "generated_passed": 1,
                "generated_pass_rate": 1.0,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": False,
                "observation_warning": "",
            },
        },
        {
            "cycle_id": "cycle:scope_shared_repo",
            "state": "select",
            "subsystem": "tooling",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_shared_repo",
                "scoped_run": True,
                "selected_variant_id": "procedure_promotion",
            },
        },
        {
            "cycle_id": "cycle:scope_shared_repo",
            "state": "generate",
            "subsystem": "tooling",
            "artifact_kind": "tool_candidate_set",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_shared_repo", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["frontier_candidates"][0]["shared_repo_bundle_summary"]
    assert summary["shared_repo_candidate_count"] == 2
    assert summary["shared_repo_worker_candidate_count"] == 1
    assert summary["shared_repo_complete_integrator_candidate_count"] == 1
    assert summary["shared_repo_incomplete_integrator_candidate_count"] == 0
    assert summary["shared_repo_complete_candidate_count"] == 2


def test_report_supervised_frontier_includes_retrieval_reuse_summary(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_retrieval_memory.jsonl"
    candidate_path = candidates_root / "skills" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "skills": [
                    {
                        "skill_id": "skill:retrieval",
                        "source_task_id": "repo_chore_task",
                        "quality": 0.91,
                        "procedure": {"commands": ["printf 'release ready\\n' > status.txt"]},
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "selected_retrieval_span_ids": ["learning:seed:release_status"],
                        "retrieval_backed_commands": ["printf 'release ready\\n' > status.txt"],
                    },
                    {
                        "skill_id": "skill:plain",
                        "source_task_id": "repo_chore_task",
                        "quality": 0.8,
                        "procedure": {"commands": ["git status --short"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = [
        {
            "cycle_id": "cycle:scope_retrieval_memory",
            "state": "observe",
            "subsystem": "skills",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_retrieval_memory",
                "scoped_run": True,
                "total": 2,
                "passed": 2,
                "pass_rate": 1.0,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": False,
                "observation_warning": "",
            },
        },
        {
            "cycle_id": "cycle:scope_retrieval_memory",
            "state": "select",
            "subsystem": "skills",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_retrieval_memory",
                "scoped_run": True,
                "selected_variant_id": "memory_promotion",
            },
        },
        {
            "cycle_id": "cycle:scope_retrieval_memory",
            "state": "generate",
            "subsystem": "skills",
            "artifact_kind": "skill_set",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_retrieval_memory", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["frontier_candidates"][0]["retrieval_reuse_summary"]
    assert summary["procedure_count"] == 2
    assert summary["retrieval_backed_procedure_count"] == 1
    assert summary["trusted_retrieval_procedure_count"] == 1
    assert summary["verified_retrieval_command_count"] == 1
    assert payload["summary"]["retrieval_reuse_runs"] == 1
    assert payload["summary"]["trusted_retrieval_procedure_total"] == 1


def test_report_supervised_frontier_includes_long_horizon_summary(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_long_horizon.jsonl"
    candidate_path = candidates_root / "tolbert_model" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:scope_long_horizon",
            "state": "observe",
            "subsystem": "tolbert_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_long_horizon",
                "scoped_run": True,
                "total": 4,
                "passed": 3,
                "pass_rate": 0.75,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": False,
                "observation_warning": "",
                "total_by_difficulty": {"long_horizon": 4},
                "passed_by_difficulty": {"long_horizon": 3},
                "proposal_metrics_by_difficulty": {
                    "long_horizon": {
                        "task_count": 4,
                        "proposal_selected_steps": 3,
                        "novel_valid_command_steps": 2,
                        "novel_valid_command_rate": 0.6667,
                    }
                },
                "world_feedback_by_difficulty": {
                    "long_horizon": {
                        "step_count": 4,
                        "progress_calibration_mae": 0.18,
                        "risk_calibration_mae": 0.14,
                    }
                },
            },
        },
        {
            "cycle_id": "cycle:scope_long_horizon",
            "state": "select",
            "subsystem": "tolbert_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_long_horizon",
                "scoped_run": True,
                "selected_variant_id": "long_horizon_success",
            },
        },
        {
            "cycle_id": "cycle:scope_long_horizon",
            "state": "generate",
            "subsystem": "tolbert_model",
            "artifact_kind": "tolbert_model_bundle",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_long_horizon", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["frontier_candidates"][0]["long_horizon_summary"]
    assert summary["task_count"] == 4
    assert summary["passed"] == 3
    assert summary["pass_rate"] == 0.75
    assert summary["proposal_selected_steps"] == 3
    assert summary["novel_valid_command_rate"] == 0.6667
    assert summary["world_feedback_step_count"] == 4
    assert summary["world_feedback"]["progress_calibration_mae"] == 0.18


def test_report_supervised_frontier_includes_validation_family_summary(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_validation.jsonl"
    candidate_path = candidates_root / "transition_model" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "transition_model"}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:scope_validation",
            "state": "observe",
            "subsystem": "transition_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_validation",
                "scoped_run": True,
                "total": 3,
                "passed": 2,
                "pass_rate": 0.6667,
                "generated_total": 2,
                "generated_passed": 2,
                "generated_pass_rate": 1.0,
                "observation_returncode": 0,
                "observation_timed_out": False,
                "observation_budget_exceeded": False,
                "observation_warning": "",
                "total_by_benchmark_family": {"project": 3},
                "passed_by_benchmark_family": {"project": 2},
                "generated_by_benchmark_family": {"validation": 2},
                "generated_passed_by_benchmark_family": {"validation": 2},
                "proposal_metrics_by_benchmark_family": {
                    "validation": {
                        "task_count": 2,
                        "proposal_selected_steps": 2,
                        "novel_valid_command_steps": 2,
                        "novel_valid_command_rate": 1.0,
                    }
                },
                "world_feedback_by_benchmark_family": {
                    "validation": {
                        "step_count": 2,
                        "progress_calibration_mae": 0.12,
                        "risk_calibration_mae": 0.09,
                    }
                },
            },
        },
        {
            "cycle_id": "cycle:scope_validation",
            "state": "select",
            "subsystem": "transition_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_validation",
                "scoped_run": True,
                "selected_variant_id": "regression_guard",
            },
        },
        {
            "cycle_id": "cycle:scope_validation",
            "state": "generate",
            "subsystem": "transition_model",
            "artifact_kind": "transition_model",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_validation", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["frontier_candidates"][0]["validation_family_summary"]
    assert summary["benchmark_family"] == "validation"
    assert summary["primary_task_count"] == 0
    assert summary["generated_task_count"] == 2
    assert summary["generated_passed"] == 2
    assert summary["generated_pass_rate"] == 1.0
    assert summary["proposal_selected_steps"] == 2
    assert summary["novel_valid_command_rate"] == 1.0
    assert summary["world_feedback_step_count"] == 2
    assert summary["world_feedback"]["progress_calibration_mae"] == 0.12
    assert payload["frontier_candidates"][0]["validation_family_compare_guard_reasons"] == [
        "validation_family_generated_pass_rate_regressed",
        "validation_family_novel_command_rate_regressed",
        "validation_family_world_feedback_regressed",
    ]
    assert payload["summary"]["validation_family_runs"] == 1
    assert payload["summary"]["validation_generated_total"] == 2
    assert payload["summary"]["validation_generated_passed"] == 2
    assert payload["summary"]["validation_family_compare_guard_reason_counts"] == {
        "validation_family_generated_pass_rate_regressed": 1,
        "validation_family_novel_command_rate_regressed": 1,
        "validation_family_world_feedback_regressed": 1,
    }


def test_report_supervised_frontier_includes_observed_benchmark_families(tmp_path, monkeypatch):
    module = _load_script_module("report_supervised_frontier.py")
    improvement_root = tmp_path / "improvement"
    reports_dir = improvement_root / "reports"
    candidates_root = improvement_root / "candidates"
    cycles_path = improvement_root / "cycles_scope_family_mix.jsonl"
    candidate_path = candidates_root / "transition_model" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"artifact_kind": "transition_model"}), encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle:scope_family_mix",
            "state": "observe",
            "subsystem": "transition_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_family_mix",
                "scoped_run": True,
                "total_by_benchmark_family": {"project": 2},
                "passed_by_benchmark_family": {"project": 1},
                "generated_by_benchmark_family": {"validation": 2},
                "generated_passed_by_benchmark_family": {"validation": 2},
                "total_by_origin_benchmark_family": {"repository": 3},
                "passed_by_origin_benchmark_family": {"repository": 2},
                "proposal_metrics_by_benchmark_family": {
                    "workflow": {
                        "task_count": 1,
                        "proposal_selected_steps": 1,
                    }
                },
            },
        },
        {
            "cycle_id": "cycle:scope_family_mix",
            "state": "select",
            "subsystem": "transition_model",
            "metrics_summary": {
                "protocol": "human_guided",
                "scope_id": "scope_family_mix",
                "scoped_run": True,
                "selected_variant_id": "family_mix",
            },
        },
        {
            "cycle_id": "cycle:scope_family_mix",
            "state": "generate",
            "subsystem": "transition_model",
            "artifact_kind": "transition_model",
            "artifact_path": str(candidate_path),
            "candidate_artifact_path": str(candidate_path),
            "metrics_summary": {"protocol": "human_guided", "scope_id": "scope_family_mix", "scoped_run": True},
        },
    ]
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    config = KernelConfig(
        storage_backend="json",
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        candidate_artifacts_root=candidates_root,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["report_supervised_frontier.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["frontier_candidates"][0]["observed_benchmark_families"] == [
        "project",
        "repository",
        "validation",
        "workflow",
    ]


def test_finalize_latest_candidate_from_cycles_accepts_frontier_report(tmp_path, monkeypatch):
    module = _load_script_module("finalize_latest_candidate_from_cycles.py")
    frontier_path = tmp_path / "reports" / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_a",
                        "cycle_id": "cycle:transition_model:abc",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/transition_model.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_latest_candidate_from_cycles.py",
            "--frontier-report",
            str(frontier_path),
            "--subsystem",
            "transition_model",
            "--variant-id",
            "regression_guard",
            "--dry-run",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    command = stream.getvalue().strip()
    assert "--cycle-id cycle:transition_model:abc" in command
    assert "--subsystem transition_model" in command
    assert "--artifact-path candidates/transition_model.json" in command


def test_finalize_latest_candidate_from_cycles_surfaces_trust_breadth_gate_in_dry_run(tmp_path, monkeypatch):
    module = _load_script_module("finalize_latest_candidate_from_cycles.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    trust_path = reports_dir / "unattended_trust_ledger.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_tooling",
                        "cycle_id": "cycle:tooling:abc",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "candidate_artifact_path": "candidates/tooling.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trust_path.write_text(
        json.dumps(
            {
                "coverage_summary": {
                    "family_breadth_min_distinct_task_roots": 2,
                    "required_family_clean_task_root_counts": {"repository": 1},
                    "required_families_missing_clean_task_root_breadth": ["repository"],
                }
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(unattended_trust_ledger_path=trust_path)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_latest_candidate_from_cycles.py",
            "--frontier-report",
            str(frontier_path),
            "--subsystem",
            "tooling",
            "--variant-id",
            "procedure_promotion",
            "--dry-run",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
    assert lines[0].startswith("python scripts/finalize_improvement_cycle.py")
    assert "--cycle-id cycle:tooling:abc" in lines[0]
    assert lines[1] == (
        "finalize_gate_reason="
        "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
    )


def test_report_frontier_promotion_plan_emits_ranked_compare_and_finalize_commands(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    promotion_pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path = reports_dir / "unattended_trust_ledger.json"
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_fast",
                        "cycle_id": "cycle:fast",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/fast.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 0.8,
                        "duplicate_count": 1,
                    },
                    {
                        "scope_id": "scope_slow",
                        "cycle_id": "cycle:slow",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/slow.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 8.0,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_timeout",
                        "cycle_id": "cycle:timeout",
                        "selected_subsystem": "retrieval",
                        "selected_variant_id": "routing_depth",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/timeout.json",
                        "observation_timed_out": True,
                        "observation_elapsed_seconds": 4.0,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion_pass_path.write_text(
        json.dumps({"report_kind": "supervised_frontier_promotion_pass", "results": []}),
        encoding="utf-8",
    )
    trust_path.write_text(
        json.dumps(
            {
                "coverage_summary": {
                    "family_breadth_min_distinct_task_roots": 2,
                    "required_family_clean_task_root_counts": {"repository": 1},
                    "required_families_missing_clean_task_root_breadth": ["repository"],
                }
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir, unattended_trust_ledger_path=trust_path)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
            "--max-per-subsystem",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "supervised_frontier_promotion_plan"
    assert payload["summary"]["eligible_candidates"] == 2
    assert payload["summary"]["selected_candidates"] == 2
    assert payload["summary"]["history_penalized_candidates"] == 0
    assert payload["summary"]["required_families_missing_clean_task_root_breadth"] == ["repository"]
    assert payload["summary"]["required_family_clean_task_root_counts"] == {"repository": 1}
    assert payload["summary"]["family_breadth_min_distinct_task_roots"] == 2
    assert payload["summary"]["bootstrap_finalize_trust_breadth_gate_reason"] == (
        "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
    )
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_fast"
    assert payload["promotion_candidates"][0]["promotion_base_score"] > 0.0
    assert payload["promotion_candidates"][0]["promotion_history_penalty"] == 0.0
    assert payload["promotion_candidates"][0]["bootstrap_finalize_trust_breadth_gated"] is True
    assert payload["promotion_candidates"][0]["required_families_missing_clean_task_root_breadth"] == ["repository"]
    assert payload["promotion_candidates"][0]["required_family_clean_task_root_counts"] == {"repository": 1}
    assert "compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/fast.json" in payload["promotion_candidates"][0]["compare_command"]
    assert "--frontier-report" in payload["promotion_candidates"][0]["finalize_command"]
    assert "--candidate-index 0" in payload["promotion_candidates"][0]["finalize_command"]


def test_report_frontier_promotion_plan_penalizes_recent_failed_promotion_history(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    promotion_pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_reject",
                        "cycle_id": "cycle:reject",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/reject.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 0.8,
                        "duplicate_count": 1,
                    },
                    {
                        "scope_id": "scope_clean",
                        "cycle_id": "cycle:clean",
                        "selected_subsystem": "retrieval",
                        "selected_variant_id": "routing_depth",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/clean.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 0.9,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion_pass_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_pass",
                "results": [
                    {
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "compare_status": "compared",
                        "finalize_state": "reject",
                    },
                    {
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "compare_status": "bootstrap_first_retain",
                        "finalize_skip_reason": "bootstrap_requires_review",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--promotion-pass-report",
            str(promotion_pass_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["history_penalized_candidates"] == 1
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_clean"
    penalized = next(item for item in payload["promotion_candidates"] if item["scope_id"] == "scope_reject")
    assert penalized["promotion_history_penalty"] > 0.0
    assert "recent_finalize_reject" in penalized["promotion_history_reasons"]
    assert "recent_bootstrap_requires_review" in penalized["promotion_history_reasons"]
    assert penalized["promotion_score"] < penalized["promotion_base_score"]


def test_report_frontier_promotion_plan_filters_incompatible_tooling_candidates(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    promotion_pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
    valid_candidate = tmp_path / "candidates" / "tooling_valid.json"
    invalid_candidate = tmp_path / "candidates" / "tooling_invalid.json"
    _write_tool_candidate_artifact(valid_candidate, empty=False)
    _write_tool_candidate_artifact(invalid_candidate, empty=True)
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_valid",
                        "cycle_id": "cycle:valid",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": str(valid_candidate),
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 1.2,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_invalid",
                        "cycle_id": "cycle:invalid",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": str(invalid_candidate),
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 0.7,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion_pass_path.write_text(
        json.dumps({"report_kind": "supervised_frontier_promotion_pass", "results": []}),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "5",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["eligible_candidates"] == 1
    assert payload["summary"]["selected_candidates"] == 1
    assert payload["summary"]["incompatible_candidates"] == 1
    assert [item["scope_id"] for item in payload["promotion_candidates"]] == ["scope_valid"]
    assert payload["promotion_candidates"][0]["candidate_compatible"] is True


def test_report_frontier_promotion_plan_carries_tooling_shared_repo_bundle_summary(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    candidate_path = tmp_path / "candidates" / "tooling_shared_repo.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "spec_version": "asi_v1",
                        "tool_id": "tool:worker",
                        "kind": "local_shell_procedure",
                        "lifecycle_state": "candidate",
                        "promotion_stage": "candidate_procedure",
                        "source_task_id": "worker_task",
                        "benchmark_family": "project",
                        "quality": 0.82,
                        "script_name": "worker_tool.sh",
                        "script_body": "#!/usr/bin/env bash\nset -euo pipefail\ngit checkout worker/api-status\n",
                        "procedure": {"commands": ["git checkout worker/api-status"]},
                        "task_contract": {"prompt": "Switch to worker/api-status", "metadata": {"benchmark_family": "project"}},
                        "verifier": {"termination_reason": "success"},
                        "shared_repo_bundle": {
                            "shared_repo_id": "repo_sandbox_parallel_merge",
                            "worker_branch": "worker/api-status",
                            "role": "worker",
                            "bundle_complete": True,
                        },
                    },
                    {
                        "spec_version": "asi_v1",
                        "tool_id": "tool:integrator",
                        "kind": "local_shell_procedure",
                        "lifecycle_state": "candidate",
                        "promotion_stage": "candidate_procedure",
                        "source_task_id": "integrator_task",
                        "benchmark_family": "project",
                        "quality": 0.88,
                        "script_name": "integrator_tool.sh",
                        "script_body": (
                            "#!/usr/bin/env bash\nset -euo pipefail\n"
                            "git merge --no-ff worker/api-status\n"
                            "git merge --no-ff worker/docs-status\n"
                        ),
                        "procedure": {
                            "commands": [
                                "git merge --no-ff worker/api-status",
                                "git merge --no-ff worker/docs-status",
                            ]
                        },
                        "task_contract": {"prompt": "Merge worker branches", "metadata": {"benchmark_family": "project"}},
                        "verifier": {"termination_reason": "success"},
                        "shared_repo_bundle": {
                            "shared_repo_id": "repo_sandbox_parallel_merge",
                            "role": "integrator",
                            "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_shared_repo_tooling",
                        "cycle_id": "cycle:tooling:shared_repo",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": str(candidate_path),
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 1.0,
                        "duplicate_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        module,
        "assess_artifact_compatibility",
        lambda **kwargs: {"compatible": True, "checked_rules": ["candidates"], "violations": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["promotion_candidates"][0]["shared_repo_bundle_summary"]
    assert summary["shared_repo_candidate_count"] == 2
    assert summary["shared_repo_worker_candidate_count"] == 1
    assert summary["shared_repo_complete_integrator_candidate_count"] == 1
    assert summary["shared_repo_incomplete_integrator_candidate_count"] == 0


def test_report_frontier_promotion_plan_rewards_long_horizon_evidence(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_long",
                        "cycle_id": "cycle:tolbert_model:long",
                        "selected_subsystem": "tolbert_model",
                        "selected_variant_id": "long_horizon_success",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/long.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "long_horizon_summary": {
                            "task_count": 4,
                            "passed": 4,
                            "pass_rate": 1.0,
                            "proposal_selected_steps": 3,
                            "novel_valid_command_steps": 2,
                            "novel_valid_command_rate": 1.0,
                            "world_feedback_step_count": 4,
                            "world_feedback": {"progress_calibration_mae": 0.18},
                        },
                    },
                    {
                        "scope_id": "scope_flat",
                        "cycle_id": "cycle:tolbert_model:flat",
                        "selected_subsystem": "tolbert_model",
                        "selected_variant_id": "long_horizon_success",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/flat.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_long"
    assert payload["promotion_candidates"][0]["promotion_long_horizon_bonus"] > 0.0
    assert payload["promotion_candidates"][0]["promotion_base_score"] > payload["promotion_candidates"][1]["promotion_base_score"]


def test_report_frontier_promotion_plan_rewards_retrieval_reuse_evidence(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    retrieval_candidate_path = tmp_path / "candidates" / "skills_retrieval.json"
    retrieval_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_candidate_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "skills": [
                    {
                        "skill_id": "skill:retrieval",
                        "source_task_id": "repo_chore_task",
                        "quality": 0.91,
                        "procedure": {"commands": ["printf 'release ready\\n' > status.txt"]},
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "selected_retrieval_span_ids": ["learning:seed:release_status"],
                        "retrieval_backed_commands": ["printf 'release ready\\n' > status.txt"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    flat_candidate_path = tmp_path / "candidates" / "skills_flat.json"
    flat_candidate_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "skills": [
                    {
                        "skill_id": "skill:plain",
                        "source_task_id": "repo_chore_task",
                        "quality": 0.91,
                        "procedure": {"commands": ["git status --short"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_retrieval",
                        "cycle_id": "cycle:skills:retrieval",
                        "selected_subsystem": "skills",
                        "selected_variant_id": "memory_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": str(retrieval_candidate_path),
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_plain",
                        "cycle_id": "cycle:skills:plain",
                        "selected_subsystem": "skills",
                        "selected_variant_id": "memory_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": str(flat_candidate_path),
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        module,
        "assess_artifact_compatibility",
        lambda **kwargs: {"compatible": True, "checked_rules": ["skills"], "violations": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_retrieval"
    assert payload["promotion_candidates"][0]["promotion_retrieval_reuse_bonus"] > 0.0
    assert payload["promotion_candidates"][0]["promotion_base_score"] > payload["promotion_candidates"][1]["promotion_base_score"]
    assert payload["promotion_candidates"][0]["retrieval_reuse_summary"]["trusted_retrieval_procedure_count"] == 1
    assert payload["summary"]["retrieval_reuse_ranked_candidates"] == 1


def test_report_frontier_promotion_plan_rewards_validation_family_evidence(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_validation",
                        "cycle_id": "cycle:transition_model:validation",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/validation.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "validation_family_summary": {
                            "benchmark_family": "validation",
                            "primary_task_count": 0,
                            "primary_passed": 0,
                            "primary_pass_rate": 0.0,
                            "generated_task_count": 3,
                            "generated_passed": 3,
                            "generated_pass_rate": 1.0,
                            "proposal_selected_steps": 3,
                            "novel_valid_command_steps": 2,
                            "novel_valid_command_rate": 0.6667,
                            "world_feedback_step_count": 3,
                            "world_feedback": {"progress_calibration_mae": 0.1},
                        },
                    },
                    {
                        "scope_id": "scope_flat",
                        "cycle_id": "cycle:transition_model:flat",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/flat.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_validation"
    assert payload["promotion_candidates"][0]["promotion_validation_family_bonus"] > 0.0
    assert payload["promotion_candidates"][0]["validation_family_compare_guard_reasons"] == [
        "validation_family_generated_pass_rate_regressed",
        "validation_family_novel_command_rate_regressed",
        "validation_family_world_feedback_regressed",
    ]
    assert payload["promotion_candidates"][0]["promotion_base_score"] > payload["promotion_candidates"][1]["promotion_base_score"]
    assert payload["summary"]["validation_ranked_candidates"] == 1
    assert payload["summary"]["validation_compare_guard_reason_counts"] == {
        "validation_family_generated_pass_rate_regressed": 1,
        "validation_family_novel_command_rate_regressed": 1,
        "validation_family_world_feedback_regressed": 1,
    }


def test_report_frontier_promotion_plan_flags_policy_bootstrap_without_generated_evidence(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_policy_zero_generated",
                        "cycle_id": "cycle:policy:bootstrap",
                        "selected_subsystem": "policy",
                        "selected_variant_id": "retrieval_caution",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/policy.json",
                        "candidate_artifact_kind": "prompt_proposal_set",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "primary_pass_rate": 1.0,
                        "healthy_run": True,
                        "generated_success_total": 0,
                        "generated_failure_total": 0,
                    },
                    {
                        "scope_id": "scope_policy_generated",
                        "cycle_id": "cycle:policy:generated",
                        "selected_subsystem": "policy",
                        "selected_variant_id": "retrieval_caution",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/policy_generated.json",
                        "candidate_artifact_kind": "prompt_proposal_set",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "primary_pass_rate": 1.0,
                        "healthy_run": True,
                        "generated_success_total": 2,
                        "generated_failure_total": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        module,
        "_artifact_compatibility",
        lambda candidate, repo_root: {"compatible": True, "checked_rules": [], "violations": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = {candidate["scope_id"]: candidate for candidate in payload["promotion_candidates"]}
    assert candidates["scope_policy_zero_generated"]["bootstrap_review_guard_reasons"] == [
        "policy_bootstrap_generated_evidence_missing"
    ]
    assert candidates["scope_policy_generated"]["bootstrap_review_guard_reasons"] == []
    assert payload["summary"]["bootstrap_review_guard_reason_counts"] == {
        "policy_bootstrap_generated_evidence_missing": 1
    }


def test_report_frontier_promotion_plan_penalizes_incomplete_shared_repo_bundle_histories(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_complete_bundle",
                        "cycle_id": "cycle:tooling:complete",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/complete.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "shared_repo_bundle_summary": {
                            "shared_repo_candidate_count": 2,
                            "shared_repo_worker_candidate_count": 1,
                            "shared_repo_complete_integrator_candidate_count": 1,
                            "shared_repo_incomplete_integrator_candidate_count": 0,
                            "shared_repo_complete_candidate_count": 2,
                        },
                    },
                    {
                        "scope_id": "scope_incomplete_bundle",
                        "cycle_id": "cycle:tooling:incomplete",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/incomplete.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 2.0,
                        "duplicate_count": 0,
                        "shared_repo_bundle_summary": {
                            "shared_repo_candidate_count": 1,
                            "shared_repo_worker_candidate_count": 0,
                            "shared_repo_complete_integrator_candidate_count": 0,
                            "shared_repo_incomplete_integrator_candidate_count": 1,
                            "shared_repo_complete_candidate_count": 0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_complete_bundle"
    assert payload["promotion_candidates"][0]["promotion_shared_repo_bundle_bonus"] > 0.0
    penalized = next(item for item in payload["promotion_candidates"] if item["scope_id"] == "scope_incomplete_bundle")
    assert penalized["promotion_shared_repo_bundle_penalty"] > 0.0
    assert payload["promotion_candidates"][0]["promotion_base_score"] > penalized["promotion_base_score"]


def test_report_frontier_promotion_plan_penalizes_superseded_same_variant_candidates(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_old",
                        "cycle_id": "cycle:transition_model:20260401T000000000000Z:old11111",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/old.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 8.0,
                        "primary_pass_rate": 0.0,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_new",
                        "cycle_id": "cycle:transition_model:20260402T000000000000Z:new22222",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/new.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 20.0,
                        "primary_pass_rate": 1.0,
                        "healthy_run": True,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert [item["scope_id"] for item in payload["promotion_candidates"]] == ["scope_new", "scope_old"]
    superseded = next(item for item in payload["promotion_candidates"] if item["scope_id"] == "scope_old")
    assert superseded["promotion_superseded_penalty"] > 0.0
    assert "superseded_by_newer_same_variant_candidate" in superseded["promotion_history_reasons"]


def test_report_frontier_promotion_plan_keeps_stronger_older_candidate_ahead_of_newer_warning_run(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_strong_old",
                        "cycle_id": "cycle:tooling:20260401T000000000000Z:old11111",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/strong_old.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 55.0,
                        "primary_pass_rate": 1.0,
                        "healthy_run": True,
                        "generated_success_passed": 4,
                        "generated_success_total": 4,
                        "generated_success_pass_rate": 1.0,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_warning_new",
                        "cycle_id": "cycle:tooling:20260402T000000000000Z:new22222",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/warning_new.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 50.0,
                        "primary_pass_rate": 1.0,
                        "healthy_run": False,
                        "generated_success_passed": 0,
                        "generated_success_total": 0,
                        "generated_success_pass_rate": 0.0,
                        "observation_budget_exceeded": True,
                        "observation_warning": "supplemental curriculum follow-up warning: observation child exceeded max runtime of 20.0 seconds",
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert [item["scope_id"] for item in payload["promotion_candidates"]] == ["scope_strong_old", "scope_warning_new"]
    strong = payload["promotion_candidates"][0]
    warning = payload["promotion_candidates"][1]
    assert strong["promotion_generated_success_bonus"] > warning["promotion_generated_success_bonus"]
    assert strong["promotion_health_bonus"] > warning["promotion_health_bonus"]
    assert strong["promotion_warning_penalty"] == 0.0
    assert warning["promotion_warning_category"] == "supplemental_followup_budget"
    assert warning["promotion_warning_penalty"] > 0.0
    assert strong.get("promotion_superseded_penalty", 0.0) == 0.0


def test_report_frontier_promotion_plan_rewards_observation_pass_rate(tmp_path, monkeypatch):
    module = _load_script_module("report_frontier_promotion_plan.py")
    reports_dir = tmp_path / "reports"
    frontier_path = reports_dir / "supervised_parallel_frontier.json"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_parallel_frontier_report",
                "frontier_candidates": [
                    {
                        "scope_id": "scope_fast_fail",
                        "cycle_id": "cycle:transition_model:20260401T000000000000Z:fast1111",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "repeat_avoidance",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/fast_fail.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 1.0,
                        "primary_pass_rate": 0.0,
                        "duplicate_count": 0,
                    },
                    {
                        "scope_id": "scope_slow_success",
                        "cycle_id": "cycle:transition_model:20260402T000000000000Z:slow2222",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "repeat_avoidance",
                        "generated_candidate": True,
                        "candidate_exists": True,
                        "candidate_artifact_path": "candidates/slow_success.json",
                        "observation_timed_out": False,
                        "observation_elapsed_seconds": 20.0,
                        "primary_pass_rate": 1.0,
                        "duplicate_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_frontier_promotion_plan.py",
            "--frontier-report",
            str(frontier_path),
            "--top-k",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert [item["scope_id"] for item in payload["promotion_candidates"]] == ["scope_slow_success", "scope_fast_fail"]
    assert payload["promotion_candidates"][0]["promotion_base_score"] > payload["promotion_candidates"][1]["promotion_base_score"]


def test_run_frontier_promotion_pass_emits_multi_candidate_execution_report(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_a",
                        "cycle_id": "cycle:a",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/a.json",
                        "promotion_score": 7.4,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/a.json --before-cycle-id cycle:a",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_a --candidate-index 0 --dry-run",
                    },
                    {
                        "scope_id": "scope_b",
                        "cycle_id": "cycle:b",
                        "selected_subsystem": "retrieval",
                        "selected_variant_id": "routing_depth",
                        "candidate_artifact_path": "candidates/b.json",
                        "promotion_score": 6.2,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem retrieval --artifact-path candidates/b.json --before-cycle-id cycle:b",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem retrieval --scope-id scope_b --candidate-index 1 --dry-run",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command and "candidates/a.json" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "baseline_pass_rate=0.70 current_pass_rate=0.75", "stderr": ""})()
        if "compare_retained_baseline.py" in command and "candidates/b.json" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "baseline_pass_rate=0.70 current_pass_rate=0.68", "stderr": ""})()
        if "finalize_latest_candidate_from_cycles.py" in command and "scope_a" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "cycle_id=cycle:a subsystem=transition_model state=retain reason=improved", "stderr": ""})()
        if "finalize_latest_candidate_from_cycles.py" in command and "scope_b" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "cycle_id=cycle:b subsystem=retrieval state=reject reason=regressed", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
            "--limit",
            "2",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "supervised_frontier_promotion_pass"
    assert payload["summary"]["candidate_count"] == 2
    assert payload["summary"]["executed_candidates"] == 2
    assert payload["summary"]["apply_finalize"] is False
    assert payload["summary"]["compare_failures"] == 0
    assert payload["summary"]["bootstrap_candidates"] == 0
    assert payload["summary"]["retained"] == 1
    assert payload["summary"]["rejected"] == 1
    assert payload["results"][0]["compare_status"] == "compared"
    assert payload["results"][1]["compare_status"] == "compared"
    assert payload["results"][0]["finalize_state"] == "retain"
    assert payload["results"][1]["finalize_state"] == "reject"
    assert "--dry-run" in payload["results"][0]["finalize"]["command"]


def test_run_frontier_promotion_pass_routes_missing_baseline_into_bootstrap_path(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_bootstrap",
                        "cycle_id": "cycle:bootstrap",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/bootstrap.json",
                        "promotion_score": 7.8,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/bootstrap.json --before-cycle-id cycle:bootstrap",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_bootstrap --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "no prior retained baseline exists for subsystem=transition_model",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "cycle_id=cycle:bootstrap subsystem=transition_model state=retain reason=bootstrap_first_retain",
                    "stderr": "",
                },
            )()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["executed_candidates"] == 1
    assert payload["summary"]["compare_failures"] == 0
    assert payload["summary"]["bootstrap_candidates"] == 1
    assert payload["summary"]["retained"] == 1
    assert payload["results"][0]["promotion_route"] == "bootstrap_first_retain"
    assert payload["results"][0]["compare_status"] == "bootstrap_first_retain"
    assert payload["results"][0]["finalize_state"] == "retain"


def test_run_frontier_promotion_pass_skips_bootstrap_finalize_without_allow(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_bootstrap",
                        "cycle_id": "cycle:bootstrap",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/bootstrap.json",
                        "promotion_score": 7.8,
                        "bootstrap_finalize_trust_breadth_gate_reason": (
                            "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
                        ),
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/bootstrap.json --before-cycle-id cycle:bootstrap",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_bootstrap --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "no prior retained baseline exists for subsystem=transition_model",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            raise AssertionError("bootstrap finalize should be skipped without --allow-bootstrap-finalize")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
            "--apply-finalize",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["bootstrap_candidates"] == 1
    assert payload["summary"]["skipped_bootstrap_finalize"] == 1
    assert payload["summary"]["retained"] == 0
    assert payload["policy"]["allow_bootstrap_finalize"] is False
    assert payload["results"][0]["promotion_route"] == "bootstrap_first_retain"
    assert payload["results"][0]["finalize_skipped"] is True
    assert payload["results"][0]["finalize_skip_reason"] == "bootstrap_requires_review"
    assert payload["results"][0]["finalize_gate_reason"] == (
        "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
    )
    assert payload["results"][0]["finalize_state"] == ""


def test_run_frontier_promotion_pass_records_finalize_gate_reason_from_finalize_output(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_bootstrap",
                        "cycle_id": "cycle:bootstrap",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/bootstrap.json",
                        "promotion_score": 7.8,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/bootstrap.json --before-cycle-id cycle:bootstrap",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_bootstrap --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "no prior retained baseline exists for subsystem=transition_model",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        "cycle_id=cycle:bootstrap subsystem=transition_model state=retain "
                        "reason=bootstrap_first_retain\n"
                        "finalize_gate_reason=bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
                    ),
                    "stderr": "",
                },
            )()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["results"][0]["finalize_state"] == "retain"
    assert payload["results"][0]["finalize_gate_reason"] == (
        "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
    )


def test_run_frontier_promotion_pass_skips_bootstrap_finalize_for_unlisted_subsystem(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_bootstrap",
                        "cycle_id": "cycle:bootstrap",
                        "selected_subsystem": "trust",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/bootstrap.json",
                        "promotion_score": 7.8,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem trust --artifact-path candidates/bootstrap.json --before-cycle-id cycle:bootstrap",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem trust --scope-id scope_bootstrap --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "no prior retained baseline exists for subsystem=trust",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            raise AssertionError("bootstrap finalize should be skipped for unlisted subsystem")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
            "--apply-finalize",
            "--allow-bootstrap-finalize",
            "--allow-bootstrap-subsystem",
            "transition_model",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["summary"]["bootstrap_candidates"] == 1
    assert payload["summary"]["skipped_bootstrap_subsystem_not_allowed"] == 1
    assert payload["policy"]["allowed_bootstrap_subsystems"] == ["transition_model"]
    assert payload["results"][0]["finalize_skipped"] is True
    assert payload["results"][0]["finalize_skip_reason"] == "bootstrap_subsystem_not_allowed"


def test_run_frontier_promotion_pass_skips_finalize_when_compare_reports_long_horizon_regression(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_long_horizon_regressed",
                        "cycle_id": "cycle:tolbert_model:regressed",
                        "selected_subsystem": "tolbert_model",
                        "selected_variant_id": "long_horizon_success",
                        "candidate_artifact_path": "candidates/tolbert.json",
                        "promotion_score": 8.1,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem tolbert_model --artifact-path candidates/tolbert.json --before-cycle-id cycle:tolbert_model:regressed",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tolbert_model --scope-id scope_long_horizon_regressed --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "long_horizon_delta pass_rate_delta=-0.10 novel_valid_command_rate_delta=0.00 progress_calibration_mae_gain=0.02",
                    "stderr": "",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            raise AssertionError("finalize should be skipped when compare reports long-horizon regression")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["summary"]["compare_failures"] == 1
    assert payload["summary"]["retained"] == 0
    assert payload["results"][0]["compare_status"] == "compare_failed"
    assert payload["results"][0]["compare_guard_reason"] == "long_horizon_pass_rate_regressed"
    assert payload["results"][0]["finalize_skipped"] is True
    assert payload["results"][0]["finalize_skip_reason"] == "compare_failed"


def test_run_frontier_promotion_pass_skips_finalize_when_compare_reports_shared_repo_bundle_regression(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_bundle_regressed",
                        "cycle_id": "cycle:tooling:bundle_regressed",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "candidate_artifact_path": "candidates/tooling.json",
                        "promotion_score": 7.9,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path candidates/tooling.json --before-cycle-id cycle:tooling:bundle_regressed",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tooling --scope-id scope_bundle_regressed --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "shared_repo_bundle_delta baseline_complete_candidate_count=2 current_complete_candidate_count=1 complete_candidate_delta=-1 baseline_worker_candidate_count=1 current_worker_candidate_count=1 worker_candidate_delta=0 baseline_complete_integrator_candidate_count=1 current_complete_integrator_candidate_count=0 complete_integrator_candidate_delta=-1 baseline_incomplete_integrator_candidate_count=0 current_incomplete_integrator_candidate_count=1 incomplete_integrator_candidate_delta=1 bundle_coherence_delta=-2",
                    "stderr": "",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            raise AssertionError("finalize should be skipped when compare reports shared-repo bundle regression")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["summary"]["compare_failures"] == 1
    assert payload["results"][0]["compare_status"] == "compare_failed"
    assert payload["results"][0]["compare_guard_reason"] == "shared_repo_bundle_coherence_regressed"
    assert payload["results"][0]["finalize_skipped"] is True
    assert payload["results"][0]["finalize_skip_reason"] == "compare_failed"


def test_run_frontier_promotion_pass_skips_finalize_when_compare_reports_validation_family_regression(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_validation_regressed",
                        "cycle_id": "cycle:transition_model:validation_regressed",
                        "selected_subsystem": "transition_model",
                        "selected_variant_id": "regression_guard",
                        "candidate_artifact_path": "candidates/transition_model.json",
                        "promotion_score": 7.4,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/transition_model.json --before-cycle-id cycle:transition_model:validation_regressed",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_validation_regressed --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert shell is True
        assert text is True
        assert capture_output is True
        if "compare_retained_baseline.py" in command:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "validation_family_delta primary_pass_rate_delta=0.00 generated_pass_rate_delta=-0.50 novel_valid_command_rate_delta=-1.00 progress_calibration_mae_gain=-0.08",
                    "stderr": "",
                },
            )()
        if "finalize_latest_candidate_from_cycles.py" in command:
            raise AssertionError("finalize should be skipped when compare reports validation-family regression")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["summary"]["compare_failures"] == 1
    assert payload["results"][0]["compare_status"] == "compare_failed"
    assert payload["results"][0]["compare_guard_reason"] == "validation_family_generated_pass_rate_regressed"
    assert payload["results"][0]["finalize_skipped"] is True
    assert payload["results"][0]["finalize_skip_reason"] == "compare_failed"


def test_run_frontier_promotion_pass_skips_incompatible_artifact_before_compare(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    invalid_candidate = tmp_path / "candidates" / "tooling_invalid.json"
    _write_tool_candidate_artifact(invalid_candidate, empty=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_invalid",
                        "cycle_id": "cycle:invalid",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "procedure_promotion",
                        "candidate_artifact_path": str(invalid_candidate),
                        "promotion_score": 7.0,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path candidates/invalid.json --before-cycle-id cycle:invalid",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tooling --scope-id scope_invalid --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        raise AssertionError(f"subprocess should not run for incompatible candidate: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(Path(stream.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["skipped_candidates"] == 1
    assert payload["summary"]["executed_candidates"] == 0
    assert payload["results"][0]["promotion_route"] == "skipped_incompatible_artifact"
    assert payload["results"][0]["skip_reason"] == "incompatible_artifact"
    assert payload["results"][0]["compatibility_violations"] == ["artifact must contain a non-empty candidates list"]


def test_supervisor_loop_builds_actions_with_paused_lane_and_operator_gate(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_trust_ledger_path=tmp_path / "reports" / "unattended_trust_ledger.json",
    )
    policy = module.SupervisorPolicy(
        autonomy_mode="shadow",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )

    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["retrieval", "transition_model", "world_model"],
    )

    queue_state = {"active_leases": [], "active_job_count": 0}
    frontier_state = {
        "frontier_candidates": [
            {
                "selected_subsystem": "transition_model",
                "generated_candidate": True,
                "candidate_exists": True,
            }
        ]
    }
    promotion_pass_state = {"results": [{"selected_subsystem": "retrieval", "finalize_state": "reject"}]}
    trust_ledger = {"overall_assessment": {"passed": True, "status": "trusted"}}
    recent_outcomes = [
        {"selected_subsystem": "retrieval", "observation_timed_out": True, "generated_candidate": False},
        {"selected_subsystem": "retrieval", "observation_timed_out": True, "generated_candidate": False},
    ]

    decisions = module._build_round_actions(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state=queue_state,
        frontier_state=frontier_state,
        promotion_pass_state=promotion_pass_state,
        trust_ledger=trust_ledger,
        recent_outcomes=recent_outcomes,
    )

    assert "retrieval" in decisions["paused_subsystems"]
    assert decisions["paused_subsystems"]["retrieval"]["reason"] == "timeout_cooldown"
    assert decisions["operator_gated_reasons"] == ["autonomy_mode=shadow"]
    assert decisions["rollout_gate"]["blocked_subsystems"] == []
    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["transition_model", "world_model"]
    promotion = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion["enabled"] is False
    assert promotion["allow_bootstrap_finalize"] is False


def test_supervisor_loop_blocks_protected_meta_promotion_until_canary_gate_passes(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(
        json.dumps(
            {
                "protected_subsystems": ["trust", "delegation"],
                "protected_paths": ["scripts/run_supervisor_loop.py"],
            }
        ),
        encoding="utf-8",
    )
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "trust",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 5},
        },
        recent_outcomes=[],
    )

    assert decisions["rollout_gate"]["protected_frontier_subsystems"] == ["trust"]
    assert decisions["rollout_gate"]["blocked_subsystems"] == ["trust"]
    assert decisions["rollout_gate"]["candidate_classification"][0]["protected_reasons"] == ["protected_subsystem:trust"]
    promotion = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion["blocked_subsystems"] == ["trust"]
    assert promotion["apply_finalize"] is True


def test_supervisor_loop_recommends_autonomy_widening_for_healthy_non_protected_lane(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 15,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 2,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "required_families_missing_clean_task_root_breadth": [],
                "required_family_clean_task_root_counts": {},
                "family_breadth_min_distinct_task_roots": 2,
            },
        },
        recent_outcomes=[],
    )

    summary = decisions["autonomy_widening_summary"]
    assert summary["escalation_available"] is True
    assert summary["recommended_autonomy_mode"] == "promote"
    assert summary["recommended_rollout_stage"] == "broad"
    assert summary["recommended_bootstrap_finalize_policy"] == "evidence"
    assert summary["promotion_scope"] == "non_protected_broad"
    assert summary["eligible_non_protected_subsystems"] == ["tooling"]
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion_action["allow_subsystems"] == ["tooling"]
    widening_action = next(
        action for action in decisions["actions"] if action["kind"] == "prepare_autonomy_widening_package"
    )
    assert widening_action["enabled"] is True


def test_supervisor_loop_widens_only_for_non_protected_candidates_with_satisfied_family_breadth(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository"],
                },
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "policy": {"required_benchmark_families": ["repository"]},
            "coverage_summary": {
                "required_families_missing_clean_task_root_breadth": ["repository"],
                "required_family_clean_task_root_counts": {"repository": 1},
                "family_breadth_min_distinct_task_roots": 2,
            },
            "family_assessments": {
                "repository": {"passed": True, "status": "trusted"},
            },
        },
        recent_outcomes=[],
    )

    summary = decisions["autonomy_widening_summary"]
    assert summary["escalation_available"] is True
    assert summary["recommended_autonomy_mode"] == "promote"
    assert summary["recommended_rollout_stage"] == "broad"
    assert summary["eligible_non_protected_candidate_count"] == 1
    assert summary["total_non_protected_candidate_count"] == 2
    assert summary["eligible_non_protected_subsystems"] == ["tooling"]
    assert summary["blocked_non_protected_subsystems"] == ["transition_model"]
    assert summary["blocked_non_protected_reasons_by_subsystem"]["transition_model"] == [
        "trust_family_clean_task_root_breadth:repository:1<2"
    ]
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion_action["allow_subsystems"] == ["tooling"]


def test_supervisor_loop_allows_canary_bridge_from_light_supervision_evidence(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository"],
                }
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": False, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 0},
            "policy": {"required_benchmark_families": ["repository"]},
            "coverage_summary": {
                "missing_required_counted_gated_families": ["repository"],
                "required_families_missing_clean_task_root_breadth": [],
                "required_family_clean_task_root_counts": {"repository": 2},
                "required_family_light_supervision_clean_success_counts": {"repository": 2},
                "required_family_contract_clean_failure_recovery_clean_success_counts": {"repository": 0},
                "family_breadth_min_distinct_task_roots": 2,
            },
            "family_assessments": {
                "repository": {"passed": False, "status": "bootstrap"},
            },
        },
        recent_outcomes=[],
    )

    summary = decisions["autonomy_widening_summary"]
    assert summary["escalation_available"] is True
    assert summary["recommended_autonomy_mode"] == "promote"
    assert summary["recommended_rollout_stage"] == "canary"
    assert summary["promotion_scope"] == "non_protected_canary_bridge"
    assert summary["eligible_non_protected_subsystems"] == ["tooling"]
    assert summary["bridge_non_protected_subsystems"] == ["tooling"]
    assert summary["readiness_signals"]["trust_status"] == "bootstrap"
    assert summary["readiness_signals"]["light_supervision_bridge_candidate_count"] == 1
    assert "light_supervision_bridge_subsystems=tooling" in summary["recommendation_reasons"]
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion_action["allow_subsystems"] == ["tooling"]


def test_supervisor_loop_holds_compare_only_when_light_supervision_bridge_is_insufficient(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository"],
                }
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": False, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 0},
            "policy": {"required_benchmark_families": ["repository"]},
            "coverage_summary": {
                "missing_required_counted_gated_families": ["repository"],
                "required_families_missing_clean_task_root_breadth": [],
                "required_family_clean_task_root_counts": {"repository": 2},
                "required_family_light_supervision_clean_success_counts": {"repository": 1},
                "required_family_contract_clean_failure_recovery_clean_success_counts": {"repository": 0},
                "family_breadth_min_distinct_task_roots": 2,
            },
            "family_assessments": {
                "repository": {"passed": False, "status": "bootstrap"},
            },
        },
        recent_outcomes=[],
    )

    summary = decisions["autonomy_widening_summary"]
    assert summary["escalation_available"] is False
    assert summary["recommended_rollout_stage"] == "compare_only"
    assert summary["bridge_non_protected_candidate_count"] == 0
    assert "trust_status=bootstrap" in summary["blockers"]
    assert "no_non_protected_candidates_with_satisfied_trust_or_bridge_evidence" in summary["blockers"]


def test_supervisor_loop_prioritizes_widenable_lane_in_discovery_planning(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "tooling"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 15,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 2,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["workflow"],
                }
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "required_families_missing_clean_task_root_breadth": [],
                "required_family_clean_task_root_counts": {},
                "family_breadth_min_distinct_task_roots": 2,
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["tooling"]
    assert decisions["discovery_priority_summary"]["prioritized_widening_subsystems"] == ["tooling"]


def test_supervisor_loop_combines_repository_breadth_reservation_with_widenable_lane_priority(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "tooling"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository"],
                },
            ],
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "policy": {"required_benchmark_families": ["repository"]},
            "coverage_summary": {
                "required_families_missing_clean_task_root_breadth": ["repository"],
                "required_family_clean_task_root_counts": {"repository": 1},
                "family_breadth_min_distinct_task_roots": 2,
            },
            "family_assessments": {
                "repository": {"passed": True, "status": "trusted"},
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["trust", "tooling"]
    assert decisions["discovery_priority_summary"]["prioritized_widening_subsystems"] == ["tooling"]
    assert decisions["discovery_priority_summary"]["blocked_widening_subsystems"] == ["transition_model"]


def test_execute_prepare_autonomy_widening_package_writes_report(tmp_path):
    module = _load_script_module("run_supervisor_loop.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    config.ensure_directories()
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    report_path = config.improvement_reports_dir / "supervisor_autonomy_widening_plan.json"
    result = module._execute_action(
        action={
            "kind": "prepare_autonomy_widening_package",
            "enabled": True,
            "report_path": str(report_path),
            "summary": {
                "recommended_autonomy_mode": "promote",
                "recommended_rollout_stage": "canary",
                "widening_command": "python scripts/run_supervisor_loop.py --autonomy-mode promote --rollout-stage canary",
            },
        },
        config=config,
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round:test",
    )

    assert result["returncode"] == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "prepare_autonomy_widening_package"
    assert payload["autonomy_widening_summary"]["recommended_autonomy_mode"] == "promote"
    assert payload["autonomy_widening_summary"]["recommended_rollout_stage"] == "canary"
    assert payload["widening_command"].endswith("--rollout-stage canary")


def test_supervisor_loop_allocates_lanes_and_plans_rollback_for_failed_protected_trust(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    work_manifest = {
        "lanes": [
            {
                "lane_id": "lane_world",
                "title": "World",
                "owned_paths": ["agent_kernel/world_model.py"],
            },
            {
                "lane_id": "lane_supervisor",
                "title": "Supervisor",
                "owned_paths": ["scripts/run_supervisor_loop.py"],
            },
        ]
    }
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(
        json.dumps(
            {
                "protected_subsystems": [],
                "protected_paths": ["scripts/run_supervisor_loop.py"],
            }
        ),
        encoding="utf-8",
    )
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="canary",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model"])
    monkeypatch.setattr(module, "_load_work_manifest", lambda repo_root: work_manifest)

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={
            "active_leases": [
                {
                    "job_id": "job-1",
                    "claimed_paths": ["scripts/run_supervisor_loop.py"],
                }
            ]
        },
        frontier_state={
            "frontier_candidates": [
                {
                    "scope_id": "scope_meta",
                    "cycle_id": "cycle:meta",
                    "selected_subsystem": "supervisor",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "candidate_artifact_path": "candidates/supervisor.json",
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "supervisor",
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/supervisor.json",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": False, "status": "restricted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    assert decisions["lane_allocator"]["assignments"][0]["lane_id"] == "lane_world"
    assert decisions["rollback_plan"]["required"] is True
    rollback_action = next(action for action in decisions["actions"] if action["kind"] == "rollback_artifact")
    assert rollback_action["artifact_path"] == "candidates/supervisor.json"
    validate_action = next(action for action in decisions["actions"] if action["kind"] == "validate_rollback_artifact")
    assert validate_action["artifact_path"] == "candidates/supervisor.json"
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion_action["enabled"] is False
    assert "rollback_pending" in promotion_action["promotion_blocked_reasons"]
    assert "rollback_pending" in decisions["blocked_conditions"]
    assert decisions["canary_lifecycle"]["state"] == "rollback_pending"


def test_supervisor_loop_plans_rollback_for_failed_non_protected_retain(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "scope_id": "scope_retrieval",
                    "cycle_id": "cycle:retrieval",
                    "selected_subsystem": "retrieval",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "candidate_artifact_path": "candidates/retrieval.json",
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "retrieval",
                    "apply_finalize": True,
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/retrieval.json",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": False, "status": "restricted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    assert decisions["rollback_plan"]["required"] is True
    assert decisions["rollback_plan"]["trigger_reason"] == "retain_with_failed_trust"
    rollback_action = next(action for action in decisions["actions"] if action["kind"] == "rollback_artifact")
    assert rollback_action["artifact_path"] == "candidates/retrieval.json"
    assert rollback_action["reason"] == "trust_regressed_after_governed_retain"
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert promotion_action["enabled"] is False
    assert "rollback_pending" in promotion_action["promotion_blocked_reasons"]
    assert decisions["canary_lifecycle"]["state"] == "rollback_pending"


def test_supervisor_loop_requires_one_stable_canary_round_before_resuming_promotion(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="canary",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    first = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "trust",
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/trust.json",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )
    first_promotion = next(action for action in first["actions"] if action["kind"] == "run_promotion_pass")
    assert first["canary_lifecycle"]["state"] == "canary_monitoring"
    assert first_promotion["enabled"] is False
    assert "canary_observation_pending" in first_promotion["promotion_blocked_reasons"]

    second = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "trust",
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/trust.json",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_canary_lifecycle={"state": "canary_monitoring"},
    )
    second_promotion = next(action for action in second["actions"] if action["kind"] == "run_promotion_pass")
    assert second["canary_lifecycle"]["state"] == "resume_ready"
    assert second_promotion["enabled"] is True


def test_supervisor_loop_requires_one_stable_canary_round_for_non_protected_retain(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="canary",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    first = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "retrieval", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "retrieval",
                    "apply_finalize": True,
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/retrieval.json",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )
    first_promotion = next(action for action in first["actions"] if action["kind"] == "run_promotion_pass")
    assert first["canary_lifecycle"]["state"] == "canary_monitoring"
    assert first["canary_lifecycle"]["tracked_candidates"] == [
        {
            "selected_subsystem": "retrieval",
            "candidate_artifact_path": "candidates/retrieval.json",
            "protected": False,
            "required_trust_families": [],
            "trust_evidence_blocked_reasons": [],
        }
    ]
    assert first_promotion["enabled"] is False
    assert "canary_observation_pending" in first_promotion["promotion_blocked_reasons"]

    second = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "retrieval", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "retrieval",
                    "apply_finalize": True,
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/retrieval.json",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_canary_lifecycle={"state": "canary_monitoring"},
    )
    second_promotion = next(action for action in second["actions"] if action["kind"] == "run_promotion_pass")
    assert second["canary_lifecycle"]["state"] == "resume_ready"
    assert second_promotion["enabled"] is True


def test_supervisor_loop_rolls_back_retain_when_candidate_trust_family_evidence_is_insufficient(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="canary",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "scope_id": "scope_project",
                    "cycle_id": "cycle:project",
                    "selected_subsystem": "retrieval",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "candidate_artifact_path": "candidates/retrieval.json",
                    "observed_benchmark_families": ["project"],
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "scope_id": "scope_project",
                    "cycle_id": "cycle:project",
                    "selected_subsystem": "retrieval",
                    "apply_finalize": True,
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/retrieval.json",
                }
            ]
        },
        trust_ledger={
            "policy": {"required_benchmark_families": ["project"]},
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "family_assessments": {
                "project": {
                    "passed": True,
                    "status": "bootstrap",
                }
            },
            "coverage_summary": {
                "required_families": ["project"],
                "missing_required_counted_gated_families": ["project"],
                "required_family_clean_task_root_counts": {"project": 1},
                "required_families_missing_clean_task_root_breadth": ["project"],
                "family_breadth_min_distinct_task_roots": 2,
            },
        },
        recent_outcomes=[],
    )

    assert decisions["rollout_gate"]["blocked_subsystems"] == ["retrieval"]
    assert decisions["rollback_plan"]["required"] is True
    assert decisions["rollback_plan"]["trigger_reason"] == "retain_with_insufficient_trust_evidence"
    rollback_action = next(action for action in decisions["actions"] if action["kind"] == "rollback_artifact")
    assert rollback_action["artifact_path"] == "candidates/retrieval.json"
    assert rollback_action["reason"] == "trust_evidence_insufficient_after_governed_retain"
    assert decisions["canary_lifecycle"]["state"] == "rollback_pending"
    tracked = decisions["canary_lifecycle"]["tracked_candidates"][0]
    assert tracked["required_trust_families"] == ["project"]
    assert tracked["trust_evidence_blocked_reasons"] == [
        "trust_family_clean_task_root_breadth:project:1<2",
        "trust_family_counted_gated_evidence_missing:project",
        "trust_family_status=bootstrap:project",
    ]


def test_supervisor_loop_prioritizes_candidate_trust_evidence_family_for_discovery(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "scope_id": "scope_project",
                    "cycle_id": "cycle:project",
                    "selected_subsystem": "retrieval",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "candidate_artifact_path": "candidates/retrieval.json",
                    "observed_benchmark_families": ["project"],
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "policy": {"required_benchmark_families": ["project"]},
            "overall_assessment": {"passed": True, "status": "trusted"},
            "family_assessments": {
                "project": {
                    "passed": True,
                    "status": "bootstrap",
                }
            },
            "coverage_summary": {
                "required_families": ["project"],
                "missing_required_counted_gated_families": ["project"],
                "required_family_clean_task_root_counts": {"project": 2},
                "required_families_missing_clean_task_root_breadth": [],
                "family_breadth_min_distinct_task_roots": 2,
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["trust"]
    assert discovery["priority_benchmark_families"] == ["project"]
    assert decisions["discovery_priority_summary"]["prioritized_trust_evidence_subsystems"] == ["trust", "recovery"]
    assert decisions["effective_trust_evidence_focus"]["priority_benchmark_families"] == ["project"]
    assert decisions["effective_trust_evidence_focus"]["family_details"] == [
        {
            "family": "project",
            "blocked_candidate_count": 1,
            "blocked_subsystems": ["retrieval"],
            "blocked_reason_codes": ["counted_gated_evidence_missing", "status_bootstrap"],
            "severity": 2,
        }
    ]


def test_supervisor_loop_retains_candidate_trust_evidence_priority_from_memory(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_trust_evidence_priority_memory={
            "blocked_candidate_count": 1,
            "blocked_subsystems": ["retrieval"],
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["project"],
            "family_details": [
                {
                    "family": "project",
                    "blocked_candidate_count": 1,
                    "blocked_subsystems": ["retrieval"],
                    "blocked_reason_codes": ["counted_gated_evidence_missing", "status_bootstrap"],
                    "severity": 2,
                }
            ],
            "max_blocked_family_severity": 2,
            "sticky_rounds_remaining": 2,
        },
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["trust"]
    assert discovery["priority_benchmark_families"] == ["project"]
    assert decisions["effective_trust_evidence_focus"]["retained_from_memory"] is True
    assert decisions["trust_evidence_priority_memory"]["sticky_rounds_remaining"] == 1


def test_supervisor_loop_scales_trust_priority_discovery_budget_from_family_pressure(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "retrieval",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository", "project"],
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "coverage_summary": {
                "required_families": ["repository", "project"],
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1, "project": 2},
                "missing_required_counted_gated_families": ["repository", "project"],
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
            "family_assessments": {
                "repository": {"status": "bootstrap", "passed": True},
                "project": {"status": "bootstrap", "passed": True},
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"][0] == "trust"
    assert discovery["worker_count"] == 2
    assert discovery["priority_benchmark_families"] == ["repository", "project"]
    assert discovery["priority_benchmark_family_weights"] == {"project": 2.0, "repository": 3.5}
    assert discovery["task_limit"] == 8
    assert discovery["max_observation_seconds"] == 105.0
    assert decisions["trust_priority_discovery_budget"] == {
        "task_limit": 8,
        "max_observation_seconds": 105.0,
        "task_limit_bonus": 3,
        "observation_bonus_seconds": 45.0,
        "priority_benchmark_family_weights": {"project": 2.0, "repository": 3.5},
        "priority_family_count": 2,
    }


def test_supervisor_loop_prioritizes_zero_coverage_breadth_families_in_discovery(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "retrieval",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "observed_benchmark_families": ["repository", "project"],
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "coverage_summary": {
                "required_families": ["project", "repository", "integration", "repo_chore"],
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {
                    "project": 1,
                    "repository": 1,
                    "integration": 0,
                    "repo_chore": 0,
                },
                "missing_required_counted_gated_families": [
                    "project",
                    "repository",
                    "integration",
                    "repo_chore",
                ],
                "required_families_missing_clean_task_root_breadth": ["integration", "repo_chore"],
            },
            "family_assessments": {
                "project": {"status": "bootstrap", "passed": True},
                "repository": {"status": "bootstrap", "passed": True},
                "integration": {"status": "bootstrap", "passed": True},
                "repo_chore": {"status": "bootstrap", "passed": True},
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["priority_benchmark_families"][:4] == [
        "integration",
        "repo_chore",
        "project",
        "repository",
    ]
    weights = discovery["priority_benchmark_family_weights"]
    assert weights["integration"] == 3.0
    assert weights["repo_chore"] == 3.0
    assert weights["integration"] > weights["repository"]
    assert weights["repo_chore"] > weights["project"]


def test_supervisor_loop_retries_rollback_validation_until_resume_is_safe(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({"protected_subsystems": ["trust"]}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="canary",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=1,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "trust",
                    "finalize_state": "retain",
                    "candidate_artifact_path": "candidates/trust.json",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_canary_lifecycle={
            "state": "rollback_validation_failed",
            "tracked_candidates": [
                {
                    "selected_subsystem": "trust",
                    "candidate_artifact_path": "candidates/trust.json",
                }
            ],
            "validation": {"attempted": True, "passed": False, "failed": True, "results": []},
        },
    )
    validate_action = next(action for action in decisions["actions"] if action["kind"] == "validate_rollback_artifact")
    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert decisions["canary_lifecycle"]["state"] == "rollback_validation_failed"
    assert validate_action["artifact_path"] == "candidates/trust.json"
    assert promotion_action["enabled"] is False
    assert "rollback_validation_failed" in promotion_action["promotion_blocked_reasons"]


def test_apply_execution_results_to_canary_lifecycle_allows_resume_after_validation_passes():
    module = _load_script_module("run_supervisor_loop.py")
    updated = module._apply_execution_results_to_canary_lifecycle(
        canary_lifecycle={
            "state": "rollback_pending",
            "tracked_candidates": [
                {
                    "selected_subsystem": "trust",
                    "candidate_artifact_path": "candidates/trust.json",
                }
            ],
            "validation_required": True,
            "promotion_resume_allowed": False,
            "blocked_reasons": ["rollback_pending"],
            "resume_rule": "",
            "trust_status": "trusted",
            "validation": {},
        },
        executions=[
            {
                "kind": "rollback_artifact",
                "artifact_path": "candidates/trust.json",
                "selected_subsystem": "trust",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
            {
                "kind": "validate_rollback_artifact",
                "artifact_path": "candidates/trust.json",
                "selected_subsystem": "trust",
                "returncode": 0,
                "stdout": "validation_state=passed artifact_path=candidates/trust.json",
                "stderr": "",
            },
        ],
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
    )

    assert updated["state"] == "resume_ready"
    assert updated["promotion_resume_allowed"] is True
    assert updated["validation"]["passed"] is True


def test_supervisor_loop_allows_bootstrap_finalize_only_when_trusted(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    trusted = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )
    trusted_action = next(action for action in trusted["actions"] if action["kind"] == "run_promotion_pass")
    assert trusted_action["apply_finalize"] is True
    assert trusted_action["allow_bootstrap_finalize"] is True
    assert trusted_action["allowed_bootstrap_subsystems"] == ["transition_model"]

    bootstrap = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": False, "status": "bootstrap"}},
        recent_outcomes=[],
    )
    bootstrap_action = next(action for action in bootstrap["actions"] if action["kind"] == "run_promotion_pass")
    assert bootstrap_action["apply_finalize"] is False
    assert bootstrap_action["allow_bootstrap_finalize"] is False
    assert bootstrap_action["allowed_bootstrap_subsystems"] == []


def test_supervisor_loop_allows_bootstrap_finalize_with_clean_bootstrap_evidence(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="evidence",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 2,
                "false_pass_risk_count": 0,
                "unexpected_change_report_count": 0,
            },
        },
        recent_outcomes=[],
    )

    action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert action["apply_finalize"] is True
    assert action["allow_bootstrap_finalize"] is True
    assert action["allowed_bootstrap_subsystems"] == ["transition_model"]
    assert action["bootstrap_policy_reasons"] == []


def test_supervisor_loop_blocks_evidence_bootstrap_finalize_on_risk_signal(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="evidence",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "false_pass_risk_count": 1,
                "unexpected_change_report_count": 1,
            },
        },
        recent_outcomes=[],
    )

    action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert action["apply_finalize"] is True
    assert action["allow_bootstrap_finalize"] is False
    assert action["allowed_bootstrap_subsystems"] == []
    assert "bootstrap_clean_success_streak:1<2" in action["bootstrap_policy_reasons"]
    assert "bootstrap_false_pass_risk_count:1>0" in action["bootstrap_policy_reasons"]
    assert "bootstrap_unexpected_change_report_count:1>0" in action["bootstrap_policy_reasons"]


def test_supervisor_loop_requires_stronger_clean_streak_for_protected_bootstrap_finalize(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(
        json.dumps(
            {
                "protected_subsystems": ["trust"],
                "protected_paths": [],
                "protected_bootstrap_min_clean_success_streak": 5,
            }
        ),
        encoding="utf-8",
    )
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    insufficient = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True},
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 4},
        },
        recent_outcomes=[],
    )
    insufficient_action = next(action for action in insufficient["actions"] if action["kind"] == "run_promotion_pass")
    assert insufficient_action["allow_bootstrap_finalize"] is True
    assert insufficient_action["allowed_bootstrap_subsystems"] == ["transition_model"]
    assert any(
        reason.startswith("protected_bootstrap_clean_success_streak:trust:4<5")
        for reason in insufficient_action["bootstrap_policy_reasons"]
    )

    sufficient = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True},
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 5},
        },
        recent_outcomes=[],
    )
    sufficient_action = next(action for action in sufficient["actions"] if action["kind"] == "run_promotion_pass")
    assert sufficient_action["allowed_bootstrap_subsystems"] == ["transition_model", "trust"]


def test_supervisor_loop_blocks_bootstrap_finalize_when_required_task_root_breadth_is_missing(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="evidence",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 2,
                "false_pass_risk_count": 0,
                "unexpected_change_report_count": 0,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    assert action["apply_finalize"] is True
    assert action["allow_bootstrap_finalize"] is False
    assert action["allowed_bootstrap_subsystems"] == []
    assert "bootstrap_required_family_clean_task_root_breadth:repository:1<2" in action["bootstrap_policy_reasons"]


def test_supervisor_loop_prefers_runtime_managed_campaign_signal_over_clean_gap(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {
                    "integration": 0,
                    "project": 0,
                    "repo_chore": 0,
                    "repo_sandbox": 0,
                    "repository": 0,
                },
                "required_families_missing_clean_task_root_breadth": [
                    "integration",
                    "project",
                    "repo_chore",
                    "repo_sandbox",
                    "repository",
                ],
                "required_family_runtime_managed_signal_counts": {
                    "integration": 1,
                    "project": 1,
                    "repo_chore": 0,
                    "repo_sandbox": 1,
                    "repository": 1,
                },
                "required_family_runtime_managed_decision_yield_counts": {
                    "integration": 0,
                    "project": 1,
                    "repo_chore": 0,
                    "repo_sandbox": 1,
                    "repository": 0,
                },
                "required_families_missing_runtime_managed_decision_yield": [
                    "integration",
                    "repo_chore",
                    "repository",
                ],
                "required_families_missing_runtime_managed_signal": ["repo_chore"],
            },
        },
        recent_outcomes=[],
    )

    assert decisions["current_trust_breadth_focus"]["missing_required_family_runtime_managed_decision_yield"] == [
        "integration",
        "repo_chore",
        "repository",
    ]
    assert decisions["current_trust_breadth_focus"]["missing_required_family_clean_task_root_breadth"] == [
        "integration",
        "repo_chore",
        "repository",
    ]
    assert decisions["current_trust_breadth_focus"]["missing_required_family_runtime_managed_signal_breadth"] == [
        "repo_chore"
    ]
    assert decisions["current_trust_breadth_focus"]["detail_mode"] == "decision_yield"
    assert decisions["trust_breadth_reserved_subsystem_slots"] == ["trust", "trust"]


def test_supervisor_loop_prioritizes_sampled_but_uncredited_family_yield(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {
                    "integration": 0,
                    "project": 0,
                    "repo_chore": 0,
                    "repository": 0,
                },
                "required_families_missing_clean_task_root_breadth": [
                    "integration",
                    "project",
                    "repo_chore",
                    "repository",
                ],
                "required_family_sampled_progress_counts": {
                    "integration": 3,
                    "project": 2,
                    "repo_chore": 1,
                    "repository": 4,
                },
                "required_family_runtime_managed_signal_counts": {
                    "integration": 1,
                    "project": 1,
                    "repo_chore": 0,
                    "repository": 1,
                },
                "required_family_runtime_managed_decision_yield_counts": {
                    "integration": 0,
                    "project": 0,
                    "repo_chore": 0,
                    "repository": 0,
                },
                "required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield": [
                    "integration",
                    "project",
                    "repo_chore",
                    "repository",
                ],
                "required_families_missing_runtime_managed_decision_yield": [
                    "integration",
                    "project",
                    "repo_chore",
                    "repository",
                ],
                "required_families_missing_runtime_managed_signal": ["repo_chore"],
            },
        },
        recent_outcomes=[],
    )

    assert decisions["current_trust_breadth_focus"]["detail_mode"] == "credited_family_yield"
    assert decisions["current_trust_breadth_focus"]["missing_required_family_credited_yield"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert decisions["current_trust_breadth_focus"]["details"][:2] == [
        {"family": "repository", "observed": 0, "sampled_progress": 4, "threshold": 1, "remaining": 1},
        {"family": "integration", "observed": 0, "sampled_progress": 3, "threshold": 1, "remaining": 1},
    ]


def test_supervisor_loop_pauses_discovery_for_bootstrap_review_pending_subsystem(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["transition_model", "retrieval", "trust"],
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )

    assert decisions["paused_subsystems"]["transition_model"]["reason"] == "bootstrap_review_pending"
    assert decisions["paused_subsystems"]["transition_model"]["remediation_queue"] == "baseline_bootstrap"
    assert (
        decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]["selected_subsystem"]
        == "transition_model"
    )
    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["retrieval", "trust"]


def test_supervisor_loop_prioritizes_repository_breadth_trust_discovery_and_routes_to_repository_lane(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["trust"]
    assert discovery["priority_benchmark_families"] == ["repository"]
    assert decisions["discovery_priority_summary"]["prioritized_trust_breadth_subsystems"] == ["trust"]
    assert decisions["lane_allocator"]["assignments"][0]["subsystem"] == "trust"
    assert decisions["lane_allocator"]["assignments"][0]["breadth_focus_families"] == ["repository"]
    assert decisions["lane_allocator"]["assignments"][0]["lane_id"] == "supervised_parallel_lane__task_ecology"


def test_execute_launch_discovery_forwards_repository_breadth_priority_family(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    captured: dict[str, object] = {}

    def fake_command_result(*, command, cwd, timeout_seconds):
        captured["command"] = list(command)
        return {
            "command": list(command),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "started_at": "2026-04-03T00:00:00+00:00",
            "completed_at": "2026-04-03T00:00:01+00:00",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_command_result", fake_command_result)

    module._execute_action(
        action={
            "kind": "launch_discovery",
            "worker_count": 1,
            "subsystems": ["trust"],
            "priority_benchmark_families": ["repository"],
        },
        config=KernelConfig(),
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round_test",
    )

    command = captured["command"]
    assert "--priority-benchmark-family" in command
    idx = command.index("--priority-benchmark-family")
    assert command[idx + 1] == "repository"


def test_execute_launch_discovery_forwards_budget_overrides_and_family_weights(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    captured: dict[str, object] = {}

    def fake_command_result(*, command, cwd, timeout_seconds):
        captured["command"] = list(command)
        captured["timeout_seconds"] = timeout_seconds
        return {
            "command": list(command),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "started_at": "2026-04-03T00:00:00+00:00",
            "completed_at": "2026-04-03T00:00:01+00:00",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_command_result", fake_command_result)

    module._execute_action(
        action={
            "kind": "launch_discovery",
            "worker_count": 1,
            "subsystems": ["trust"],
            "priority_benchmark_families": ["repository"],
            "priority_benchmark_family_weights": {"repository": 3.0},
            "task_limit": 7,
            "max_observation_seconds": 90.0,
        },
        config=KernelConfig(),
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round_test",
    )

    command = captured["command"]
    task_index = command.index("--task-limit")
    assert command[task_index + 1] == "7"
    observation_index = command.index("--max-observation-seconds")
    assert command[observation_index + 1] == "90.0"
    weight_index = command.index("--priority-benchmark-family-weight")
    assert command[weight_index + 1] == "repository=3.00"
    assert captured["timeout_seconds"] == 270.0


def test_execute_launch_discovery_forwards_variant_ids_per_slot(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    captured: dict[str, object] = {}

    def fake_command_result(*, command, cwd, timeout_seconds):
        captured["command"] = list(command)
        return {
            "command": list(command),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "started_at": "2026-04-03T00:00:00+00:00",
            "completed_at": "2026-04-03T00:00:01+00:00",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_command_result", fake_command_result)

    module._execute_action(
        action={
            "kind": "launch_discovery",
            "worker_count": 2,
            "subsystems": ["trust", "recovery"],
            "variant_ids": ["trust_repo_breadth", "rollback_repo_breadth"],
            "variant_roles": ["trust_breadth", "rollback"],
            "variant_strategy_families": ["repository_breadth", "rollback_validation"],
            "priority_benchmark_families": ["repository"],
        },
        config=KernelConfig(),
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round_test",
    )

    command = captured["command"]
    variant_indexes = [index for index, token in enumerate(command) if token == "--variant-id"]
    assert len(variant_indexes) == 2
    assert command[variant_indexes[0] + 1] == "trust_repo_breadth"
    assert command[variant_indexes[1] + 1] == "rollback_repo_breadth"
    strategy_indexes = [index for index, token in enumerate(command) if token == "--variant-strategy-family"]
    assert len(strategy_indexes) == 2
    assert command[strategy_indexes[0] + 1] == "repository_breadth"
    assert command[strategy_indexes[1] + 1] == "rollback_validation"


def test_parallel_supervised_cycles_child_command_forwards_variant_strategy_family():
    module = _load_script_module("run_parallel_supervised_cycles.py")
    args = type(
        "Args",
        (),
        {
            "provider": "mock",
            "model": "test-model",
            "task_limit": 5,
            "max_observation_seconds": 60.0,
            "notes": "supervisor loop discovery batch",
            "priority_benchmark_family": ["repository"],
            "priority_benchmark_family_weight": [],
            "include_episode_memory": False,
            "include_skill_memory": False,
            "include_skill_transfer": False,
            "include_operator_memory": False,
            "include_tool_memory": False,
            "include_verifier_memory": False,
            "include_curriculum": False,
            "include_failure_curriculum": False,
            "generated_curriculum_budget_seconds": 0.0,
            "failure_curriculum_budget_seconds": 0.0,
        },
    )()
    command = module._child_command(
        repo_root=Path(__file__).resolve().parents[1],
        args=args,
        worker_index=0,
        scope_id="scope_1",
        progress_label="scope_1",
        requested_subsystem="recovery",
        requested_variant_id="rollback_repo_breadth",
        requested_variant_strategy_family="rollback_validation",
    )

    strategy_index = command.index("--variant-strategy-family")
    assert command[strategy_index + 1] == "rollback_validation"


def test_guided_cycle_maps_recovery_variant_strategy_family_to_preferred_variant():
    module = _load_script_module("run_human_guided_improvement_cycle.py")
    assert module._variant_strategy_family_preferred_variant_id("rollback_validation") == "rollback_safety"
    assert module._variant_strategy_family_preferred_variant_id("verifier_crosscheck") == "snapshot_coverage"
    assert module._variant_strategy_family_preferred_variant_id("unknown_strategy") == ""


def test_guided_cycle_forwards_recovery_variant_strategy_family_into_eval_kwargs():
    module = _load_script_module("run_human_guided_improvement_cycle.py")
    config = module.KernelConfig()
    args = type(
        "Args",
        (),
        {
            "subsystem": "recovery",
            "variant_strategy_family": "rollback_validation",
            "task_limit": 1,
            "priority_benchmark_family": [],
            "priority_benchmark_family_weight": [],
            "include_episode_memory": False,
            "include_skill_memory": False,
            "include_skill_transfer": False,
            "include_operator_memory": False,
            "include_tool_memory": False,
            "include_verifier_memory": False,
            "include_curriculum": False,
            "include_failure_curriculum": True,
            "generated_curriculum_budget_seconds": 0.0,
            "failure_curriculum_budget_seconds": 0.0,
            "observation_profile": "default",
        },
    )()

    eval_kwargs = module._apply_curriculum_flags(module.autonomous_cycle._observation_eval_kwargs(config, args), args)
    if str(args.subsystem or "").strip() == "recovery":
        requested_variant_strategy_family = str(args.variant_strategy_family or "").strip()
        if requested_variant_strategy_family:
            eval_kwargs["recovery_variant_strategy_family"] = requested_variant_strategy_family

    assert eval_kwargs["recovery_variant_strategy_family"] == "rollback_validation"


def test_supervisor_loop_assigns_family_role_aware_repository_recovery_slots(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])
    monkeypatch.setattr(
        module,
        "_reserved_variant_ids_for_subsystems",
        lambda config, reserved_subsystem_slots, trust_breadth_focus=None: [
            "trust_repo_breadth",
            "rollback_repo_primary",
            "false_pass_repo_primary",
        ],
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
                "hidden_side_effect_risk_rate": 0.1,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_breadth_reserved_subsystem_slots"] == ["trust", "recovery", "recovery"]
    assert decisions["effective_trust_breadth_focus"]["recovery_role_summary"]["prioritized_roles"] == [
        "rollback",
        "false_pass",
        "hidden_side_effect",
    ]
    assert decisions["trust_breadth_reserved_recovery_roles"] == ["rollback", "false_pass"]
    assert decisions["trust_breadth_reserved_recovery_strategy_families"] == [
        "rollback_validation",
        "snapshot_coverage",
    ]
    assert decisions["trust_breadth_reserved_variant_ids"] == [
        "trust_repo_breadth",
        "rollback_repo_primary",
        "false_pass_repo_primary",
    ]
    assert discovery["variant_roles"][:3] == ["trust_breadth", "rollback", "false_pass"]
    assert discovery["variant_strategy_families"][:3] == [
        "repository_breadth",
        "rollback_validation",
        "snapshot_coverage",
    ]
    assert discovery["variant_ids"][:3] == [
        "trust_repo_breadth",
        "rollback_repo_primary",
        "false_pass_repo_primary",
    ]


def test_supervisor_loop_diversifies_repeated_repository_recovery_roles_by_strategy_family(tmp_path):
    module = _load_script_module("run_supervisor_loop.py")
    strategy_families = module._trust_breadth_reserved_recovery_strategy_families(
        {"recovery_role_summary": {"prioritized_roles": ["rollback"]}},
        reserved_subsystem_slots=["trust", "recovery", "recovery"],
        reserved_recovery_roles=["rollback", "rollback"],
    )

    assert strategy_families == [
        "rollback_validation",
        "restore_verification",
    ]


def test_supervisor_loop_reserves_multiple_discovery_slots_for_repository_breadth_recovery(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=3,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 1,
            "targets": [{"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy"}],
            "targeted_subsystems": ["policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    generated_evidence = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert discovery["worker_count"] == 2
    assert discovery["subsystems"] == ["trust", "recovery"]
    assert decisions["trust_breadth_gap_severity"] == 2
    assert decisions["trust_breadth_slot_reservation"] == 2
    assert generated_evidence["worker_count"] == 1


def test_supervisor_loop_scales_repository_breadth_slot_reservation_by_gap_size(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=3,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 2,
            "targets": [
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy"},
                {"selected_subsystem": "world_model", "scope_id": "scope_world_model", "cycle_id": "cycle:world_model"},
            ],
            "targeted_subsystems": ["policy", "world_model"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    generated_evidence = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["trust_breadth_slot_reservation"] == 1
    assert discovery["worker_count"] == 1
    assert discovery["subsystems"] == ["trust"]
    assert generated_evidence["worker_count"] == 2


def test_supervisor_loop_keeps_mild_repository_breadth_gap_trust_heavy_despite_recovery_pressure(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["effective_trust_breadth_focus"]["recovery_pressure"] is True
    assert decisions["discovery_priority_summary"]["prioritized_trust_breadth_subsystems"] == ["trust"]
    assert discovery["subsystems"][0] == "trust"
    assert "recovery" not in discovery["subsystems"]


def test_supervisor_loop_scales_repository_breadth_priority_memory_by_gap_size(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    severe_gap = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )
    narrow_gap = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    assert severe_gap["trust_breadth_gap_severity"] == 2
    assert severe_gap["trust_breadth_slot_reservation"] == 2
    assert severe_gap["trust_breadth_priority_memory"]["sticky_rounds_remaining"] == 4
    assert severe_gap["trust_breadth_priority_memory"]["max_remaining_clean_task_root_breadth_gap"] == 2
    assert narrow_gap["trust_breadth_gap_severity"] == 1
    assert narrow_gap["trust_breadth_slot_reservation"] == 1
    assert narrow_gap["trust_breadth_priority_memory"]["sticky_rounds_remaining"] == 3
    assert narrow_gap["trust_breadth_priority_memory"]["max_remaining_clean_task_root_breadth_gap"] == 1


def test_supervisor_loop_retains_repository_breadth_priority_from_memory(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 2},
                "required_families_missing_clean_task_root_breadth": [],
            },
        },
        recent_outcomes=[],
        previous_trust_breadth_priority_memory={
            "missing_required_family_clean_task_root_breadth": ["repository"],
            "prioritized_subsystems": ["trust", "recovery"],
            "details": [{"family": "repository", "observed": 1, "threshold": 2, "remaining": 1}],
            "family_breadth_min_distinct_task_roots": 2,
            "max_remaining_clean_task_root_breadth_gap": 1,
            "sticky_rounds_remaining": 2,
        },
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"][0] == "trust"
    assert "recovery" not in discovery["subsystems"]
    assert decisions["effective_trust_breadth_focus"]["retained_from_memory"] is True
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["trust_breadth_slot_reservation"] == 1
    assert decisions["trust_breadth_priority_memory"]["sticky_rounds_remaining"] == 1


def test_supervisor_loop_retains_severe_repository_breadth_recovery_mix_from_memory(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=3,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
        previous_trust_breadth_priority_memory={
            "missing_required_family_clean_task_root_breadth": ["repository"],
            "prioritized_subsystems": ["trust", "recovery"],
            "details": [{"family": "repository", "observed": 0, "threshold": 2, "remaining": 2}],
            "family_breadth_min_distinct_task_roots": 2,
            "max_remaining_clean_task_root_breadth_gap": 2,
            "recovery_pressure": True,
            "recovery_pressure_evidence_score": 2,
            "recovery_mix_sticky_rounds_remaining": 2,
            "sticky_rounds_remaining": 4,
        },
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["effective_trust_breadth_focus"]["recovery_mix_retained_from_memory"] is True
    assert decisions["effective_trust_breadth_focus"]["recovery_mix_sticky_rounds_remaining"] == 1
    assert decisions["discovery_priority_summary"]["prioritized_trust_breadth_subsystems"] == ["trust", "recovery"]
    assert decisions["trust_breadth_gap_severity"] == 2
    assert decisions["trust_breadth_slot_reservation"] == 2
    assert discovery["subsystems"][:2] == ["trust", "recovery"]
    assert decisions["trust_breadth_priority_memory"]["recovery_mix_sticky_rounds_remaining"] == 1
    assert decisions["trust_breadth_priority_memory"]["recovery_pressure_evidence_score"] == 1


def test_supervisor_loop_extends_severe_repository_breadth_recovery_mix_for_fresh_repeated_pressure(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])
    monkeypatch.setattr(
        module,
        "_reserved_variant_ids_for_subsystems",
        lambda config, reserved_subsystem_slots, trust_breadth_focus=None: [
            "trust_repo_breadth",
            "rollback_repo_primary",
            "rollback_repo_secondary",
        ],
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
                "hidden_side_effect_risk_rate": 0.1,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
        previous_trust_breadth_priority_memory={
            "missing_required_family_clean_task_root_breadth": ["repository"],
            "prioritized_subsystems": ["trust", "recovery"],
            "details": [{"family": "repository", "observed": 0, "threshold": 2, "remaining": 2}],
            "family_breadth_min_distinct_task_roots": 2,
            "max_remaining_clean_task_root_breadth_gap": 2,
            "recovery_pressure": True,
            "recovery_pressure_evidence_score": 2,
            "recovery_mix_sticky_rounds_remaining": 2,
            "sticky_rounds_remaining": 4,
        },
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["effective_trust_breadth_focus"]["recovery_pressure"] is True
    assert decisions["effective_trust_breadth_focus"]["recovery_pressure_evidence_score"] == 3
    assert decisions["discovery_priority_summary"]["prioritized_trust_breadth_subsystems"] == ["trust", "recovery"]
    assert decisions["trust_breadth_slot_reservation"] == 3
    assert decisions["trust_breadth_reserved_subsystem_slots"] == ["trust", "recovery", "recovery"]
    assert decisions["trust_breadth_reserved_recovery_roles"] == ["rollback", "false_pass"]
    assert decisions["trust_breadth_reserved_recovery_strategy_families"] == [
        "rollback_validation",
        "snapshot_coverage",
    ]
    assert decisions["trust_breadth_reserved_variant_ids"] == [
        "trust_repo_breadth",
        "rollback_repo_primary",
        "rollback_repo_secondary",
    ]
    assert discovery["subsystems"][:3] == ["trust", "recovery", "recovery"]
    assert discovery["variant_ids"][:3] == [
        "trust_repo_breadth",
        "rollback_repo_primary",
        "rollback_repo_secondary",
    ]
    assert discovery["variant_roles"][:3] == ["trust_breadth", "rollback", "false_pass"]
    assert discovery["variant_strategy_families"][:3] == [
        "repository_breadth",
        "rollback_validation",
        "snapshot_coverage",
    ]
    assert decisions["trust_breadth_priority_memory"]["recovery_mix_sticky_rounds_remaining"] == 5
    assert decisions["trust_breadth_priority_memory"]["recovery_pressure_evidence_score"] == 4


def test_supervisor_loop_severe_repository_breadth_gap_with_recovery_pressure_adds_recovery(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_breadth_gap_severity"] == 2
    assert decisions["effective_trust_breadth_focus"]["recovery_pressure"] is True
    assert decisions["discovery_priority_summary"]["prioritized_trust_breadth_subsystems"] == ["trust", "recovery"]
    assert discovery["subsystems"] == ["trust", "recovery"]


def test_supervisor_loop_scales_reserved_recovery_slots_by_fresh_evidence_quality(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["world_model", "policy"])

    stale = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 1},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
        previous_trust_breadth_priority_memory={
            "missing_required_family_clean_task_root_breadth": ["repository"],
            "prioritized_subsystems": ["trust", "recovery"],
            "details": [{"family": "repository", "observed": 0, "threshold": 2, "remaining": 2}],
            "family_breadth_min_distinct_task_roots": 2,
            "max_remaining_clean_task_root_breadth_gap": 2,
            "recovery_pressure": True,
            "recovery_pressure_evidence_score": 2,
            "recovery_mix_sticky_rounds_remaining": 2,
            "sticky_rounds_remaining": 4,
        },
    )
    fresh = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {
                "clean_success_streak": 1,
                "rollback_performed_rate": 0.25,
                "false_pass_risk_rate": 0.2,
                "hidden_side_effect_risk_rate": 0.1,
            },
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
        previous_trust_breadth_priority_memory={
            "missing_required_family_clean_task_root_breadth": ["repository"],
            "prioritized_subsystems": ["trust", "recovery"],
            "details": [{"family": "repository", "observed": 0, "threshold": 2, "remaining": 2}],
            "family_breadth_min_distinct_task_roots": 2,
            "max_remaining_clean_task_root_breadth_gap": 2,
            "recovery_pressure": True,
            "recovery_pressure_evidence_score": 2,
            "recovery_mix_sticky_rounds_remaining": 2,
            "sticky_rounds_remaining": 4,
        },
    )

    assert stale["trust_breadth_slot_reservation"] == 2
    assert stale["trust_breadth_reserved_subsystem_slots"] == ["trust", "recovery"]
    assert fresh["trust_breadth_slot_reservation"] == 3
    assert fresh["trust_breadth_reserved_subsystem_slots"] == ["trust", "recovery", "recovery"]


def test_supervisor_loop_deprioritizes_validation_guarded_subsystems_in_discovery_selection(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["tooling", "retrieval", "world_model"],
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "validation_family_compare_guard_reasons": [
                        "validation_family_generated_pass_rate_regressed",
                        "validation_family_novel_command_rate_regressed",
                    ],
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["retrieval", "world_model"]
    assert discovery["selected_guarded_subsystems"] == []
    assert discovery["deprioritized_guarded_subsystems"] == ["tooling"]
    assert decisions["discovery_priority_summary"]["selected_clean_subsystems"] == ["retrieval", "world_model"]
    assert decisions["discovery_priority_summary"]["deprioritized_guarded_subsystems"] == ["tooling"]
    assert decisions["validation_guard_pressure_summary"]["guarded_subsystems"] == [
        {
            "selected_subsystem": "tooling",
            "guarded_candidate_count": 1,
            "validation_guard_reason_count": 2,
            "validation_guard_severity": 5,
            "validation_family_compare_guard_reasons": [
                "validation_family_generated_pass_rate_regressed",
                "validation_family_novel_command_rate_regressed",
            ],
        }
    ]


def test_supervisor_loop_keeps_guarded_subsystem_eligible_when_clean_discovery_slots_run_out(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["tooling", "retrieval"],
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "validation_family_compare_guard_reasons": [
                        "validation_family_generated_pass_rate_regressed"
                    ],
                }
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["retrieval", "tooling"]
    assert discovery["selected_guarded_subsystems"] == ["tooling"]
    assert discovery["deprioritized_guarded_subsystems"] == []
    assert decisions["discovery_priority_summary"]["selected_guarded_subsystems"] == ["tooling"]


def test_supervisor_loop_carries_validation_guard_pressure_across_quiet_round(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["tooling", "retrieval"],
    )

    previous_validation_guard_memory = {
        "guarded_subsystem_count": 1,
        "guarded_subsystems": [
            {
                "selected_subsystem": "tooling",
                "guarded_candidate_count": 1,
                "validation_guard_reason_count": 1,
                "validation_guard_severity": 3,
                "validation_family_compare_guard_reasons": [
                    "validation_family_generated_pass_rate_regressed"
                ],
                "sticky_rounds_remaining": 2,
            }
        ],
    }

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_validation_guard_memory=previous_validation_guard_memory,
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["retrieval"]
    assert discovery["deprioritized_guarded_subsystems"] == ["tooling"]
    assert decisions["validation_guard_memory"]["guarded_subsystems"] == [
        {
            "selected_subsystem": "tooling",
            "guarded_candidate_count": 1,
            "validation_guard_reason_count": 1,
            "validation_guard_severity": 3,
            "validation_family_compare_guard_reasons": [
                "validation_family_generated_pass_rate_regressed"
            ],
            "sticky_rounds_remaining": 1,
        }
    ]
    assert decisions["effective_validation_guard_pressure_summary"]["retained_only_subsystems"] == ["tooling"]


def test_supervisor_loop_clears_validation_guard_memory_after_clean_compare(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["tooling", "retrieval"],
    )

    previous_validation_guard_memory = {
        "guarded_subsystem_count": 1,
        "guarded_subsystems": [
            {
                "selected_subsystem": "tooling",
                "guarded_candidate_count": 1,
                "validation_guard_reason_count": 1,
                "validation_guard_severity": 3,
                "validation_family_compare_guard_reasons": [
                    "validation_family_generated_pass_rate_regressed"
                ],
                "sticky_rounds_remaining": 2,
            }
        ],
    }

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "tooling",
                    "compare_status": "compared",
                    "compare_guard_reason": "",
                    "finalize_state": "retain",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": True, "status": "trusted"}},
        recent_outcomes=[],
        previous_validation_guard_memory=previous_validation_guard_memory,
    )

    discovery = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery["subsystems"] == ["tooling"]
    assert decisions["validation_guard_memory"]["guarded_subsystems"] == []
    assert decisions["validation_guard_memory"]["clean_compare_cleared_subsystems"] == ["tooling"]
    assert decisions["effective_validation_guard_pressure_summary"]["guarded_subsystems"] == []


def test_supervisor_loop_routes_bootstrap_pause_into_trust_streak_accumulation_queue(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["transition_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": [{"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True}]},
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={"overall_assessment": {"passed": False, "status": "bootstrap"}},
        recent_outcomes=[],
    )

    assert decisions["paused_subsystems"]["transition_model"]["remediation_queue"] == "trust_streak_accumulation"
    assert (
        decisions["bootstrap_remediation_queues"]["trust_streak_accumulation"][0]["selected_subsystem"]
        == "transition_model"
    )
    assert "bootstrap_remediation_pending" in decisions["blocked_conditions"]
    trust_action = next(
        action for action in decisions["actions"] if action["kind"] == "prepare_trust_streak_recovery_package"
    )
    assert trust_action["queue_name"] == "trust_streak_accumulation"
    assert trust_action["entries"][0]["selected_subsystem"] == "transition_model"
    assert trust_action["entries"][0]["required_trust_status"] == "trusted"
    assert trust_action["entries"][0]["current_trust_status"] == "bootstrap"


def test_supervisor_loop_routes_dry_run_bootstrap_first_retain_into_trust_queue(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=0,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")

    decisions = module._build_round_actions(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling:bootstrap",
                    "candidate_artifact_path": "candidates/tooling.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": False,
                    "finalize_skip_reason": "",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 0},
        },
        recent_outcomes=[],
    )

    assert decisions["paused_subsystems"]["tooling"]["reason"] == "bootstrap_review_pending"
    assert decisions["paused_subsystems"]["tooling"]["remediation_queue"] == "trust_streak_accumulation"
    assert decisions["paused_subsystems"]["tooling"]["lane_signal_queue"] == "trust_streak_accumulation"
    assert decisions["paused_subsystems"]["tooling"]["remediation_queues"] == [
        "trust_streak_accumulation",
        "baseline_bootstrap",
    ]
    assert decisions["bootstrap_remediation_queues"]["trust_streak_accumulation"][0]["selected_subsystem"] == "tooling"
    assert decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]["selected_subsystem"] == "tooling"
    assert decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]["lane_signal_queue"] == (
        "trust_streak_accumulation"
    )
    trust_action = next(
        action for action in decisions["actions"] if action["kind"] == "prepare_trust_streak_recovery_package"
    )
    assert trust_action["entries"][0]["selected_subsystem"] == "tooling"
    assert trust_action["entries"][0]["current_trust_status"] == "bootstrap"
    baseline_action = next(action for action in decisions["actions"] if action["kind"] == "prepare_bootstrap_review_package")
    assert baseline_action["entries"][0]["selected_subsystem"] == "tooling"
    assert baseline_action["entries"][0]["required_trust_status"] == "trusted"
    assert "trust-gated" in baseline_action["entries"][0]["review_focus"]


def test_supervisor_loop_routes_bootstrap_pause_into_protected_review_only_queue(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(
        json.dumps(
            {
                "protected_subsystems": ["trust"],
                "protected_paths": [],
                "protected_bootstrap_min_clean_success_streak": 5,
            }
        ),
        encoding="utf-8",
    )
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["trust"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": [{"selected_subsystem": "trust", "generated_candidate": True, "candidate_exists": True}]},
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "trust",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_subsystem_not_allowed",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 4},
        },
        recent_outcomes=[],
    )

    assert decisions["paused_subsystems"]["trust"]["remediation_queue"] == "protected_review_only"
    assert decisions["bootstrap_remediation_queues"]["protected_review_only"][0]["selected_subsystem"] == "trust"
    protected_action = next(action for action in decisions["actions"] if action["kind"] == "prepare_protected_review_package")
    assert protected_action["queue_name"] == "protected_review_only"
    assert protected_action["entries"][0]["selected_subsystem"] == "trust"
    assert protected_action["entries"][0]["required_clean_success_streak"] == 5


def test_supervisor_loop_emits_baseline_bootstrap_review_action(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["transition_model"])
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")

    decisions = module._build_round_actions(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_a",
                    "cycle_id": "cycle:a",
                    "candidate_artifact_path": "candidates/a.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    baseline_action = next(action for action in decisions["actions"] if action["kind"] == "prepare_bootstrap_review_package")
    assert baseline_action["queue_name"] == "baseline_bootstrap"
    assert baseline_action["entries"][0]["selected_subsystem"] == "transition_model"
    assert "--scope-id scope_a" in baseline_action["entries"][0]["review_finalize_command"]


def test_supervisor_loop_reroutes_policy_bootstrap_guarded_candidate_into_protected_review_priority(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["policy"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "policy",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "policy",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "bootstrap_review_guard_reasons": [
                        "policy_bootstrap_generated_evidence_missing"
                    ],
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "policy",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "candidate_artifact_path": "candidates/policy.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    paused = decisions["paused_subsystems"]["policy"]
    assert paused["remediation_queue"] == "protected_review_only"
    assert paused["lane_signal_queue"] == "protected_review_only"
    assert paused["deferred_lane_signal_queue"] == "baseline_bootstrap"
    assert paused["bootstrap_review_required"] is True
    protected_entry = decisions["bootstrap_remediation_queues"]["protected_review_only"][0]
    assert protected_entry["bootstrap_review_guard_reasons"] == [
        "policy_bootstrap_generated_evidence_missing"
    ]
    assert protected_entry["review_focus"].startswith("bootstrap review guard present")
    baseline_entry = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]
    assert baseline_entry["finalize_gate_reason"] == "bootstrap review guard pending before bootstrap lane finalize"


def test_supervisor_loop_launches_bootstrap_generated_evidence_discovery_for_policy_guard(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["policy", "retrieval"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "policy",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "policy",
                    "selected_variant_id": "retrieval_caution",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "bootstrap_review_guard_reasons": [
                        "policy_bootstrap_generated_evidence_missing"
                    ],
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "policy",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "candidate_artifact_path": "candidates/policy.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    evidence_action = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert evidence_action["worker_count"] == 1
    assert evidence_action["generated_curriculum_budget_seconds"] == 30.0
    assert evidence_action["targets"] == [
        {
            "selected_subsystem": "policy",
            "selected_variant_id": "retrieval_caution",
            "scope_id": "scope_policy",
            "cycle_id": "cycle:policy",
            "queue_name": "protected_review_only",
            "bootstrap_generated_evidence_guard_reasons": [
                "policy_bootstrap_generated_evidence_missing"
            ],
        }
    ]
    discovery_action = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert discovery_action["worker_count"] == 1
    assert discovery_action["subsystems"] == ["retrieval"]
    assert decisions["bootstrap_generated_evidence_summary"] == {
        "target_count": 1,
        "targets": [
            {
                "selected_subsystem": "policy",
                "selected_variant_id": "retrieval_caution",
                "scope_id": "scope_policy",
                "cycle_id": "cycle:policy",
                "queue_name": "protected_review_only",
                "bootstrap_generated_evidence_guard_reasons": [
                    "policy_bootstrap_generated_evidence_missing"
                ],
            }
        ],
        "targeted_subsystems": ["policy"],
    }
    status_payload = module._status_payload(
        started_at=datetime.now(UTC),
        now=datetime.now(UTC),
        policy=policy,
        rounds_completed=1,
        latest_round={"policy": policy.to_dict(), "decisions": decisions},
        machine_state={},
        blocked_conditions=[],
        next_retry_at="",
    )
    assert status_payload["bootstrap_generated_evidence_summary"]["targeted_subsystems"] == ["policy"]


def test_supervisor_loop_prioritizes_widenable_bootstrap_generated_evidence_targets(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["retrieval"])
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 2,
            "targets": [
                {
                    "selected_subsystem": "policy",
                    "selected_variant_id": "policy_variant",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "queue_name": "protected_review_only",
                    "required_families": ["repository"],
                },
                {
                    "selected_subsystem": "tooling",
                    "selected_variant_id": "tooling_variant",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "queue_name": "protected_review_only",
                    "required_families": ["workflow"],
                },
            ],
            "targeted_subsystems": ["policy", "tooling"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                }
            ]
        },
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    evidence_action = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert [target["selected_subsystem"] for target in evidence_action["targets"]] == ["tooling", "policy"]


def test_supervisor_loop_scales_bootstrap_generated_evidence_worker_count_for_widenable_lane(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=3,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["trust", "world_model"])
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 2,
            "targets": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "required_families": ["workflow"],
                },
                {
                    "selected_subsystem": "policy",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "required_families": ["repository"],
                },
            ],
            "targeted_subsystems": ["tooling", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                }
            ]
        },
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    evidence_action = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    discovery_action = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["trust_breadth_slot_reservation"] == 1
    assert evidence_action["worker_count"] == 2
    assert [target["selected_subsystem"] for target in evidence_action["targets"]] == ["tooling", "policy"]
    assert discovery_action["worker_count"] == 1
    assert discovery_action["subsystems"] == ["trust"]


def test_supervisor_loop_allows_widenable_bootstrap_lane_to_claim_majority_round_budget(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["trust", "recovery", "world_model"])
    monkeypatch.setattr(
        module,
        "_effective_trust_evidence_focus",
        lambda current_focus, trust_evidence_priority_memory: {
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["repository"],
            "family_details": [{"family": "repository", "severity": 2}],
            "max_blocked_family_severity": 2,
            "retained_from_memory": False,
            "sticky_rounds_remaining": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 3,
            "targets": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "required_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "required_families": ["workflow"],
                },
                {
                    "selected_subsystem": "policy",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "required_families": ["repository"],
                },
            ],
            "targeted_subsystems": ["tooling", "transition_model", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
            ]
        },
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    evidence_action = next(
        action
        for action in decisions["actions"]
        if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    discovery_action = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_evidence_gap_severity"] == 2
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["trust_priority_slot_reservation"] == 2
    assert evidence_action["worker_count"] == 3
    assert [target["selected_subsystem"] for target in evidence_action["targets"]] == [
        "tooling",
        "transition_model",
        "policy",
    ]
    assert discovery_action["worker_count"] == 1
    assert discovery_action["subsystems"] == ["trust"]


def test_supervisor_loop_scales_promotion_pass_limit_for_widenable_lane_majority_budget(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=4,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["trust", "recovery", "world_model", "tooling"],
    )
    monkeypatch.setattr(
        module,
        "_effective_trust_evidence_focus",
        lambda current_focus, trust_evidence_priority_memory: {
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["repository"],
            "family_details": [{"family": "repository", "severity": 2}],
            "max_blocked_family_severity": 2,
            "retained_from_memory": False,
            "sticky_rounds_remaining": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 3,
            "targets": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling", "required_families": ["workflow"]},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model", "required_families": ["workflow"]},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy", "required_families": ["repository"]},
            ],
            "targeted_subsystems": ["tooling", "transition_model", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "world_model",
                    "scope_id": "scope_world_model",
                    "cycle_id": "cycle:world_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
            ],
        },
        promotion_plan_state={
            "promotion_candidates": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling"},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model"},
                {"selected_subsystem": "world_model", "scope_id": "scope_world_model", "cycle_id": "cycle:world_model"},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy"},
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    promotion_action = next(action for action in decisions["actions"] if action["kind"] == "run_promotion_pass")
    evidence_action = next(
        action for action in decisions["actions"] if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert decisions["trust_evidence_gap_severity"] == 2
    assert decisions["trust_breadth_gap_severity"] == 1
    assert decisions["launch_generic_discovery"] is False
    assert promotion_action["allow_subsystems"] == ["tooling", "transition_model", "world_model"]
    assert promotion_action["limit"] == 3
    assert evidence_action["worker_count"] == 3
    assert not any(action["kind"] == "launch_discovery" for action in decisions["actions"])


def test_widening_promotion_pass_limit_reweights_second_pass_by_remaining_scope_and_fresh_feedback():
    module = _load_script_module("run_supervisor_loop.py")

    limit = module._widening_promotion_pass_limit(
        available_worker_slots=4,
        max_promotion_candidates=4,
        promotion_candidates=[
            {"selected_subsystem": "tooling"},
            {"selected_subsystem": "transition_model"},
            {"selected_subsystem": "world_model"},
            {"selected_subsystem": "policy"},
        ],
        trust_evidence_gap_severity=2,
        trust_breadth_gap_severity=1,
        widening_summary={
            "eligible_non_protected_subsystems": [
                "tooling",
                "transition_model",
                "world_model",
            ],
            "eligible_non_protected_candidate_count": 3,
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_count": 1,
            "eligible_non_protected_retained_subsystems": ["tooling"],
        },
        promotion_pass_subsystem_scope={
            "allow_subsystems": ["transition_model", "world_model"],
            "prioritized_subsystems": ["transition_model", "world_model"],
            "blocked_subsystems": ["tooling"],
            "require_allow_subsystem_match": True,
        },
    )

    assert limit == 2


def test_promotion_execution_feedback_summary_scores_strong_retains_above_bootstrap_retains():
    module = _load_script_module("run_supervisor_loop.py")

    summary = module._promotion_execution_feedback_summary(
        promotion_results=[
            {"selected_subsystem": "tooling", "finalize_state": "retain", "compare_status": "compared"},
            {"selected_subsystem": "policy", "finalize_state": "retain", "compare_status": "bootstrap_first_retain"},
            {"selected_subsystem": "planner", "finalize_state": "reject", "compare_status": "compared"},
        ],
        widening_summary={
            "eligible_non_protected_subsystems": ["tooling", "policy", "planner"],
            "eligible_non_protected_cluster_by_subsystem": {
                "tooling": "families:repository+workflow",
                "policy": "families:project",
                "planner": "families:repository+workflow",
            },
        },
    )

    assert summary["eligible_non_protected_retained_count"] == 2
    assert summary["eligible_non_protected_retained_quality_score"] == 5
    assert summary["eligible_non_protected_retained_quality_by_subsystem"] == {
        "tooling": 3,
        "policy": 2,
    }
    assert summary["eligible_non_protected_attempt_quality_by_subsystem"] == {
        "tooling": 3,
        "policy": 2,
        "planner": 2,
    }
    assert summary["eligible_non_protected_retained_quality_by_cluster"] == {
        "families:repository+workflow": 3,
        "families:project": 2,
    }
    assert summary["eligible_non_protected_attempt_quality_by_cluster"] == {
        "families:repository+workflow": 3,
        "families:project": 2,
    }


def test_promotion_pass_subsystem_scope_reweights_second_pass_priority_by_attempt_quality():
    module = _load_script_module("run_supervisor_loop.py")

    scope = module._promotion_pass_subsystem_scope(
        widening_summary={
            "eligible_non_protected_subsystems": [
                "tooling",
                "transition_model",
                "world_model",
                "policy",
            ],
            "eligible_non_protected_priority_by_subsystem": {
                "tooling": 10,
                "transition_model": 7,
                "world_model": 6,
                "policy": 8,
            },
            "eligible_non_protected_cluster_by_subsystem": {
                "tooling": "families:repository+workflow",
                "transition_model": "families:project",
                "world_model": "families:repository+workflow",
                "policy": "families:project",
            },
            "eligible_non_protected_priority_by_cluster": {
                "families:repository+workflow": 10,
                "families:project": 8,
            },
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_subsystems": ["tooling"],
            "eligible_non_protected_attempt_quality_by_subsystem": {
                "tooling": 3,
                "world_model": 2,
            },
            "eligible_non_protected_attempt_quality_by_cluster": {
                "families:repository+workflow": 3,
            },
        },
    )

    assert scope["allow_subsystems"] == ["world_model", "policy", "transition_model"]
    assert scope["prioritized_subsystems"] == ["world_model", "policy", "transition_model"]
    assert scope["blocked_subsystems"] == ["tooling"]
    assert scope["require_allow_subsystem_match"] is True


def test_promotion_pass_subsystem_scope_reweights_second_pass_priority_by_cluster_freshness_and_strength():
    module = _load_script_module("run_supervisor_loop.py")

    scope = module._promotion_pass_subsystem_scope(
        widening_summary={
            "eligible_non_protected_subsystems": [
                "tooling",
                "world_model",
                "policy",
                "transition_model",
            ],
            "eligible_non_protected_priority_by_subsystem": {
                "tooling": 10,
                "world_model": 6,
                "policy": 8,
                "transition_model": 7,
            },
            "eligible_non_protected_cluster_by_subsystem": {
                "tooling": "families:repository+workflow",
                "world_model": "families:repository+workflow",
                "policy": "families:project",
                "transition_model": "families:project",
            },
            "eligible_non_protected_priority_by_cluster": {
                "families:repository+workflow": 10,
                "families:project": 8,
            },
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_subsystems": ["tooling"],
            "eligible_non_protected_attempt_quality_by_subsystem": {
                "world_model": 1,
                "policy": 2,
            },
            "eligible_non_protected_attempt_quality_by_cluster": {
                "families:repository+workflow": 1,
                "families:project": 2,
            },
        },
    )

    assert scope["allow_subsystems"] == ["policy", "transition_model", "world_model"]
    assert scope["prioritized_subsystems"] == ["policy", "transition_model", "world_model"]
    assert scope["blocked_subsystems"] == ["tooling"]


def test_widening_promotion_pass_limit_uses_retain_quality_not_just_count():
    module = _load_script_module("run_supervisor_loop.py")

    weak_limit = module._widening_promotion_pass_limit(
        available_worker_slots=6,
        max_promotion_candidates=6,
        promotion_candidates=[
            {"selected_subsystem": "tooling"},
            {"selected_subsystem": "transition_model"},
            {"selected_subsystem": "world_model"},
            {"selected_subsystem": "policy"},
            {"selected_subsystem": "planner"},
            {"selected_subsystem": "critic"},
        ],
        trust_evidence_gap_severity=2,
        trust_breadth_gap_severity=0,
        widening_summary={
            "eligible_non_protected_subsystems": [
                "tooling",
                "transition_model",
                "world_model",
                "policy",
                "planner",
                "critic",
            ],
            "eligible_non_protected_candidate_count": 6,
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_count": 1,
            "eligible_non_protected_retained_quality_score": 1,
            "eligible_non_protected_retained_subsystems": ["tooling"],
        },
        promotion_pass_subsystem_scope={
            "allow_subsystems": ["transition_model", "world_model", "policy", "planner", "critic"],
            "prioritized_subsystems": ["transition_model", "world_model", "policy", "planner", "critic"],
            "blocked_subsystems": ["tooling"],
            "require_allow_subsystem_match": True,
        },
    )
    strong_limit = module._widening_promotion_pass_limit(
        available_worker_slots=6,
        max_promotion_candidates=6,
        promotion_candidates=[
            {"selected_subsystem": "tooling"},
            {"selected_subsystem": "transition_model"},
            {"selected_subsystem": "world_model"},
            {"selected_subsystem": "policy"},
            {"selected_subsystem": "planner"},
            {"selected_subsystem": "critic"},
        ],
        trust_evidence_gap_severity=2,
        trust_breadth_gap_severity=0,
        widening_summary={
            "eligible_non_protected_subsystems": [
                "tooling",
                "transition_model",
                "world_model",
                "policy",
                "planner",
                "critic",
            ],
            "eligible_non_protected_candidate_count": 6,
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_count": 1,
            "eligible_non_protected_retained_quality_score": 3,
            "eligible_non_protected_retained_subsystems": ["tooling"],
        },
        promotion_pass_subsystem_scope={
            "allow_subsystems": ["transition_model", "world_model", "policy", "planner", "critic"],
            "prioritized_subsystems": ["transition_model", "world_model", "policy", "planner", "critic"],
            "blocked_subsystems": ["tooling"],
            "require_allow_subsystem_match": True,
        },
    )

    assert weak_limit == 4
    assert strong_limit == 5


def test_widening_promotion_pass_limit_returns_zero_when_no_widenable_scope_remains():
    module = _load_script_module("run_supervisor_loop.py")

    limit = module._widening_promotion_pass_limit(
        available_worker_slots=4,
        max_promotion_candidates=4,
        promotion_candidates=[
            {"selected_subsystem": "tooling"},
            {"selected_subsystem": "transition_model"},
        ],
        trust_evidence_gap_severity=2,
        trust_breadth_gap_severity=1,
        widening_summary={
            "eligible_non_protected_subsystems": ["tooling", "transition_model"],
            "eligible_non_protected_candidate_count": 2,
        },
        promotion_feedback_summary={
            "eligible_non_protected_retained_count": 2,
            "eligible_non_protected_retained_subsystems": ["tooling", "transition_model"],
        },
        promotion_pass_subsystem_scope={
            "allow_subsystems": [],
            "prioritized_subsystems": [],
            "blocked_subsystems": ["tooling", "transition_model"],
            "require_allow_subsystem_match": True,
        },
    )

    assert limit == 0


def test_supervisor_loop_reallocates_round_actions_toward_promotion_and_bootstrap_when_widening_dominates(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=4,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["trust", "recovery", "world_model", "tooling"],
    )
    monkeypatch.setattr(
        module,
        "_effective_trust_evidence_focus",
        lambda current_focus, trust_evidence_priority_memory: {
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["repository"],
            "family_details": [{"family": "repository", "severity": 2}],
            "max_blocked_family_severity": 2,
            "retained_from_memory": False,
            "sticky_rounds_remaining": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 3,
            "targets": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling", "required_families": ["workflow"]},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model", "required_families": ["workflow"]},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy", "required_families": ["repository"]},
            ],
            "targeted_subsystems": ["tooling", "transition_model", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "world_model",
                    "scope_id": "scope_world_model",
                    "cycle_id": "cycle:world_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
            ],
        },
        promotion_plan_state={
            "promotion_candidates": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling"},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model"},
                {"selected_subsystem": "world_model", "scope_id": "scope_world_model", "cycle_id": "cycle:world_model"},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy"},
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    kinds = [action["kind"] for action in decisions["actions"]]
    assert decisions["launch_generic_discovery"] is False
    assert "run_promotion_pass" in kinds
    assert "launch_bootstrap_generated_evidence_discovery" in kinds
    assert "launch_discovery" not in kinds


def test_supervisor_loop_keeps_generic_discovery_when_repository_breadth_debt_is_severe(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=4,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["trust", "recovery", "world_model", "tooling"],
    )
    monkeypatch.setattr(
        module,
        "_effective_trust_evidence_focus",
        lambda current_focus, trust_evidence_priority_memory: {
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["repository"],
            "family_details": [{"family": "repository", "severity": 2}],
            "max_blocked_family_severity": 2,
            "retained_from_memory": False,
            "sticky_rounds_remaining": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 3,
            "targets": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling", "required_families": ["workflow"]},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model", "required_families": ["workflow"]},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy", "required_families": ["repository"]},
            ],
            "targeted_subsystems": ["tooling", "transition_model", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "world_model",
                    "scope_id": "scope_world_model",
                    "cycle_id": "cycle:world_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
            ],
        },
        promotion_plan_state={
            "promotion_candidates": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling"},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model"},
                {"selected_subsystem": "world_model", "scope_id": "scope_world_model", "cycle_id": "cycle:world_model"},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy"},
            ]
        },
        promotion_pass_state={"results": []},
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 0},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    kinds = [action["kind"] for action in decisions["actions"]]
    discovery_action = next(action for action in decisions["actions"] if action["kind"] == "launch_discovery")
    assert decisions["trust_breadth_gap_severity"] == 2
    assert decisions["launch_generic_discovery"] is True
    assert "run_promotion_pass" in kinds
    assert "launch_bootstrap_generated_evidence_discovery" in kinds
    assert discovery_action["worker_count"] == 1


def test_supervisor_loop_uses_same_round_promotion_results_to_trim_remaining_widening_budget(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=4,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=4,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["trust", "recovery", "world_model", "tooling"],
    )
    monkeypatch.setattr(
        module,
        "_effective_trust_evidence_focus",
        lambda current_focus, trust_evidence_priority_memory: {
            "prioritized_subsystems": ["trust", "recovery"],
            "priority_benchmark_families": ["repository"],
            "family_details": [{"family": "repository", "severity": 2}],
            "max_blocked_family_severity": 2,
            "retained_from_memory": False,
            "sticky_rounds_remaining": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_generated_evidence_summary",
        lambda queues: {
            "target_count": 3,
            "targets": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling", "required_families": ["workflow"]},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model", "required_families": ["workflow"]},
                {"selected_subsystem": "policy", "scope_id": "scope_policy", "cycle_id": "cycle:policy", "required_families": ["repository"]},
            ],
            "targeted_subsystems": ["tooling", "transition_model", "policy"],
        },
    )

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "summary": {
                "completed_runs": 20,
                "healthy_runs": 16,
                "timed_out_runs": 1,
                "budget_exceeded_runs": 1,
                "generated_candidate_runs": 12,
            },
            "frontier_candidates": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_tooling",
                    "cycle_id": "cycle:tooling",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_transition_model",
                    "cycle_id": "cycle:transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
                {
                    "selected_subsystem": "world_model",
                    "scope_id": "scope_world_model",
                    "cycle_id": "cycle:world_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                    "benchmark_families": ["workflow"],
                },
            ],
        },
        promotion_plan_state={
            "promotion_candidates": [
                {"selected_subsystem": "tooling", "scope_id": "scope_tooling", "cycle_id": "cycle:tooling"},
                {"selected_subsystem": "transition_model", "scope_id": "scope_transition_model", "cycle_id": "cycle:transition_model"},
                {"selected_subsystem": "world_model", "scope_id": "scope_world_model", "cycle_id": "cycle:world_model"},
            ]
        },
        promotion_pass_state={
            "results": [
                {"selected_subsystem": "tooling", "finalize_state": "retain", "compare_status": "compared"},
                {"selected_subsystem": "transition_model", "finalize_state": "retain", "compare_status": "compared"},
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    evidence_action = next(
        action for action in decisions["actions"] if action["kind"] == "launch_bootstrap_generated_evidence_discovery"
    )
    assert decisions["promotion_execution_feedback_summary"]["eligible_non_protected_retained_count"] == 2
    assert decisions["promotion_execution_feedback_summary"]["eligible_non_protected_retained_subsystems"] == [
        "tooling",
        "transition_model",
    ]
    assert evidence_action["worker_count"] == 1
    assert [target["selected_subsystem"] for target in evidence_action["targets"]] == ["policy"]
    assert decisions["launch_generic_discovery"] is False
    assert not any(action["kind"] == "launch_discovery" for action in decisions["actions"])


def test_run_supervisor_loop_executes_second_promotion_pass_after_successful_widening_feedback(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    improvement_root = tmp_path / "improvement"
    jobs_root = tmp_path / "jobs"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        delegated_job_queue_path=jobs_root / "queue.json",
        delegated_job_runtime_state_path=jobs_root / "runtime_state.json",
        unattended_trust_ledger_path=reports_dir / "unattended_trust_ledger.json",
    )
    improvement_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    config.delegated_job_queue_path.write_text("[]\n", encoding="utf-8")
    config.delegated_job_runtime_state_path.write_text("{}\n", encoding="utf-8")
    config.unattended_trust_ledger_path.write_text(
        json.dumps({"overall_assessment": {"status": "trusted", "passed": True}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(module, "_queue_state", lambda _config: {"active_leases": []})
    monkeypatch.setattr(module, "_frontier_state", lambda _config: {"summary": {}, "frontier_candidates": []})
    monkeypatch.setattr(module, "_promotion_plan_state", lambda _config: {"promotion_candidates": []})
    monkeypatch.setattr(module, "_promotion_pass_state", lambda _config: {"summary": {}, "results": []})
    monkeypatch.setattr(module, "_load_trust_ledger", lambda path: {"overall_assessment": {"status": "trusted", "passed": True}})
    monkeypatch.setattr(module, "_recent_outcomes", lambda config, limit=20: [])
    monkeypatch.setattr(module, "_load_previous_machine_state", lambda path: {})
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    decisions_sequence = [
        {
            "actions": [
                {"kind": "refresh_frontier", "enabled": True},
                {"kind": "refresh_promotion_plan", "enabled": True},
            ],
            "blocked_conditions": [],
            "canary_lifecycle": {},
            "meta_policy": {},
            "rollout_gate": {},
            "claim_ledger": {},
            "lane_allocator": {},
            "rollback_plan": {},
            "validation_guard_memory": {},
            "trust_evidence_priority_memory": {},
            "trust_breadth_priority_memory": {},
            "bootstrap_retrieval_priority_memory": {},
            "autonomy_widening_summary": {},
            "promotion_execution_feedback_summary": {},
        },
        {
            "actions": [
                {"kind": "run_promotion_pass", "enabled": True, "limit": 1},
            ],
            "blocked_conditions": [],
            "canary_lifecycle": {},
            "meta_policy": {},
            "rollout_gate": {},
            "claim_ledger": {},
            "lane_allocator": {},
            "rollback_plan": {},
            "validation_guard_memory": {},
            "trust_evidence_priority_memory": {},
            "trust_breadth_priority_memory": {},
            "bootstrap_retrieval_priority_memory": {},
            "autonomy_widening_summary": {},
            "promotion_execution_feedback_summary": {},
        },
        {
            "actions": [
                {"kind": "run_promotion_pass", "enabled": True, "limit": 1},
                {"kind": "launch_bootstrap_generated_evidence_discovery", "enabled": True, "worker_count": 1, "targets": []},
            ],
            "blocked_conditions": [],
            "canary_lifecycle": {},
            "meta_policy": {},
            "rollout_gate": {},
            "claim_ledger": {},
            "lane_allocator": {},
            "rollback_plan": {},
            "validation_guard_memory": {},
            "trust_evidence_priority_memory": {},
            "trust_breadth_priority_memory": {},
            "bootstrap_retrieval_priority_memory": {},
            "autonomy_widening_summary": {},
            "promotion_execution_feedback_summary": {
                "eligible_non_protected_retained_count": 1,
                "eligible_non_protected_retained_subsystems": ["tooling"],
            },
        },
        {
            "actions": [
                {"kind": "launch_bootstrap_generated_evidence_discovery", "enabled": True, "worker_count": 1, "targets": []},
            ],
            "blocked_conditions": [],
            "canary_lifecycle": {},
            "meta_policy": {},
            "rollout_gate": {},
            "claim_ledger": {},
            "lane_allocator": {},
            "rollback_plan": {},
            "validation_guard_memory": {},
            "trust_evidence_priority_memory": {},
            "trust_breadth_priority_memory": {},
            "bootstrap_retrieval_priority_memory": {},
            "autonomy_widening_summary": {},
            "promotion_execution_feedback_summary": {
                "eligible_non_protected_retained_count": 1,
                "eligible_non_protected_retained_subsystems": ["tooling"],
            },
        },
    ]
    build_calls = {"count": 0}

    def fake_build_round_actions(**kwargs):
        index = build_calls["count"]
        build_calls["count"] += 1
        if index >= len(decisions_sequence):
            return decisions_sequence[-1]
        return decisions_sequence[index]

    executions = []

    def fake_execute_or_skip_action(*, action, round_id):
        executions.append(str(action.get("kind", "")).strip())
        return {
            "kind": str(action.get("kind", "")).strip(),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_build_round_actions", fake_build_round_actions)
    monkeypatch.setattr(module, "_execute_or_skip_action", fake_execute_or_skip_action)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_supervisor_loop.py",
            "--autonomy-mode",
            "promote",
            "--max-rounds",
            "1",
            "--provider",
            "mock",
            "--model",
            "test-model",
            "--max-discovery-workers",
            "2",
            "--max-promotion-candidates",
            "2",
            "--rollout-stage",
            "broad",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert executions.count("run_promotion_pass") == 2
    assert executions[-1] == "launch_bootstrap_generated_evidence_discovery"


def test_supervisor_loop_reroutes_validation_guarded_bootstrap_pause_into_protected_review_priority(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["transition_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "validation_family_compare_guard_reasons": [
                        "validation_family_generated_pass_rate_regressed",
                        "validation_family_novel_command_rate_regressed",
                    ],
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "candidate_artifact_path": "candidates/guarded.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    paused = decisions["paused_subsystems"]["transition_model"]
    assert paused["remediation_queue"] == "protected_review_only"
    assert paused["lane_signal_queue"] == "protected_review_only"
    assert paused["deferred_lane_signal_queue"] == "baseline_bootstrap"
    assert paused["validation_family_review_required"] is True
    assert paused["remediation_queues"] == ["protected_review_only", "baseline_bootstrap"]
    protected_entry = decisions["bootstrap_remediation_queues"]["protected_review_only"][0]
    assert protected_entry["selected_subsystem"] == "transition_model"
    assert protected_entry["validation_family_review_required"] is True
    assert protected_entry["validation_guard_reason_count"] == 2
    assert protected_entry["review_focus"].startswith("validation-family compare guard present")
    baseline_entry = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]
    assert baseline_entry["finalize_gate_reason"] == (
        "validation-family review pending before bootstrap lane finalize"
    )
    remediation_kinds = [
        action["kind"]
        for action in decisions["actions"]
        if str(action.get("kind", "")).startswith("prepare_")
    ]
    assert remediation_kinds[0] == "prepare_protected_review_package"
    status_payload = module._status_payload(
        started_at=datetime.now(UTC),
        now=datetime.now(UTC),
        policy=policy,
        rounds_completed=1,
        latest_round={"policy": policy.to_dict(), "decisions": decisions},
        machine_state={},
        blocked_conditions=[],
        next_retry_at="",
    )
    assert status_payload["selected_queue_kind"] == "protected_review_only"


def test_supervisor_loop_surfaces_trust_breadth_finalize_gate_reason_in_bootstrap_queue(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["transition_model"])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {
                    "selected_subsystem": "transition_model",
                    "generated_candidate": True,
                    "candidate_exists": True,
                }
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_breadth",
                    "cycle_id": "cycle:breadth",
                    "candidate_artifact_path": "candidates/breadth.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 3},
            "coverage_summary": {
                "family_breadth_min_distinct_task_roots": 2,
                "required_family_clean_task_root_counts": {"repository": 1},
                "required_families_missing_clean_task_root_breadth": ["repository"],
            },
        },
        recent_outcomes=[],
    )

    baseline_entry = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"][0]
    assert baseline_entry["finalize_gate_reason"] == (
        "bootstrap finalize still gated by required clean task-root breadth (repository:1/2)"
    )


def test_supervisor_loop_deprioritizes_validation_guarded_entries_inside_bootstrap_lane(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "retrieval", "generated_candidate": True, "candidate_exists": True},
                {"selected_subsystem": "transition_model", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "retrieval",
                    "scope_id": "scope_safe",
                    "cycle_id": "cycle:safe",
                },
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "validation_family_compare_guard_reasons": [
                        "validation_family_generated_pass_rate_regressed"
                    ],
                },
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_guarded",
                    "cycle_id": "cycle:guarded",
                    "candidate_artifact_path": "candidates/guarded.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
                {
                    "selected_subsystem": "retrieval",
                    "scope_id": "scope_safe",
                    "cycle_id": "cycle:safe",
                    "candidate_artifact_path": "candidates/safe.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    baseline_entries = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"]
    assert [entry["selected_subsystem"] for entry in baseline_entries] == ["retrieval", "transition_model"]
    protected_entries = decisions["bootstrap_remediation_queues"]["protected_review_only"]
    assert [entry["selected_subsystem"] for entry in protected_entries] == ["transition_model"]
    baseline_action = next(action for action in decisions["actions"] if action["kind"] == "prepare_bootstrap_review_package")
    assert [entry["selected_subsystem"] for entry in baseline_action["entries"]] == ["retrieval", "transition_model"]


def test_supervisor_loop_prioritizes_retrieval_reuse_inside_bootstrap_lane(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="broad",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "retrieval", "generated_candidate": True, "candidate_exists": True},
                {"selected_subsystem": "tooling", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_plan_state={
            "promotion_candidates": [
                {
                    "selected_subsystem": "retrieval",
                    "scope_id": "scope_plain",
                    "cycle_id": "cycle:plain",
                },
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_reuse",
                    "cycle_id": "cycle:reuse",
                    "retrieval_reuse_summary": {
                        "retrieval_backed_procedure_count": 1,
                        "trusted_retrieval_procedure_count": 1,
                        "verified_retrieval_command_count": 2,
                        "selected_retrieval_span_count": 2,
                        "retrieval_selected_step_count": 3,
                    },
                },
            ]
        },
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "retrieval",
                    "scope_id": "scope_plain",
                    "cycle_id": "cycle:plain",
                    "candidate_artifact_path": "candidates/plain.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_reuse",
                    "cycle_id": "cycle:reuse",
                    "candidate_artifact_path": "candidates/reuse.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
    )

    baseline_entries = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"]
    assert [entry["selected_subsystem"] for entry in baseline_entries] == ["tooling", "retrieval"]
    assert baseline_entries[0]["retrieval_reuse_priority_score"] > 0
    assert baseline_entries[1]["retrieval_reuse_priority_score"] == 0
    assert decisions["bootstrap_retrieval_priority_summary"] == {
        "retrieval_ranked_entry_count": 1,
        "retrieval_ranked_subsystems": ["tooling"],
        "trusted_retrieval_entry_count": 1,
        "verified_retrieval_command_total": 2,
        "retrieval_reuse_priority_total": baseline_entries[0]["retrieval_reuse_priority_score"],
        "retained_retrieval_ranked_entry_count": 0,
        "queue_priority_leaders": [
            {
                "queue_name": "baseline_bootstrap",
                "selected_subsystem": "tooling",
                "retrieval_reuse_priority_score": baseline_entries[0]["retrieval_reuse_priority_score"],
            },
        ],
    }
    status_payload = module._status_payload(
        started_at=datetime.now(UTC),
        now=datetime.now(UTC),
        policy=policy,
        rounds_completed=1,
        latest_round={"policy": policy.to_dict(), "decisions": decisions},
        machine_state={},
        blocked_conditions=[],
        next_retry_at="",
    )
    assert status_payload["selected_queue_kind"] == "baseline_bootstrap"
    assert status_payload["selected_subsystem"] == "tooling"
    assert status_payload["bootstrap_retrieval_priority_summary"]["retrieval_ranked_subsystems"] == ["tooling"]


def test_supervisor_loop_carries_bootstrap_retrieval_priority_across_rounds(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    previous_bootstrap_retrieval_priority_memory = {
        "prioritized_subsystem_count": 1,
        "prioritized_subsystems": [
            {
                "selected_subsystem": "tooling",
                "retrieval_reuse_priority_score": 16,
                "retrieval_reuse_summary": {
                    "retrieval_backed_procedure_count": 1,
                    "trusted_retrieval_procedure_count": 1,
                    "verified_retrieval_command_count": 2,
                },
                "sticky_rounds_remaining": 2,
            }
        ],
    }

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={
            "frontier_candidates": [
                {"selected_subsystem": "retrieval", "generated_candidate": True, "candidate_exists": True},
                {"selected_subsystem": "tooling", "generated_candidate": True, "candidate_exists": True},
            ]
        },
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "retrieval",
                    "scope_id": "scope_plain",
                    "cycle_id": "cycle:plain",
                    "candidate_artifact_path": "candidates/plain.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_reuse",
                    "cycle_id": "cycle:reuse",
                    "candidate_artifact_path": "candidates/reuse.json",
                    "compare_status": "bootstrap_first_retain",
                    "finalize_skipped": True,
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "finalize_state": "",
                },
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
        previous_bootstrap_retrieval_priority_memory=previous_bootstrap_retrieval_priority_memory,
    )

    baseline_entries = decisions["bootstrap_remediation_queues"]["baseline_bootstrap"]
    assert [entry["selected_subsystem"] for entry in baseline_entries] == ["tooling", "retrieval"]
    assert baseline_entries[0]["retrieval_reuse_priority_score"] == 0
    assert baseline_entries[0]["retained_retrieval_reuse_priority_score"] == 16
    assert baseline_entries[0]["effective_retrieval_reuse_priority_score"] == 16
    assert baseline_entries[0]["retrieval_reuse_priority_sticky_rounds_remaining"] == 1
    assert decisions["bootstrap_retrieval_priority_memory"] == {
        "prioritized_subsystem_count": 1,
        "prioritized_subsystems": [
            {
                "selected_subsystem": "tooling",
                "retrieval_reuse_priority_score": 16,
                "retrieval_reuse_summary": {
                    "retrieval_backed_procedure_count": 1,
                    "trusted_retrieval_procedure_count": 1,
                    "verified_retrieval_command_count": 2,
                },
                "sticky_rounds_remaining": 1,
            }
        ],
        "clean_compare_cleared_subsystems": [],
    }
    assert decisions["effective_bootstrap_retrieval_priority_summary"]["retained_only_subsystems"] == ["tooling"]
    assert decisions["bootstrap_retrieval_priority_summary"]["retained_retrieval_ranked_entry_count"] == 1
    status_payload = module._status_payload(
        started_at=datetime.now(UTC),
        now=datetime.now(UTC),
        policy=policy,
        rounds_completed=1,
        latest_round={"policy": policy.to_dict(), "decisions": decisions},
        machine_state={},
        blocked_conditions=[],
        next_retry_at="",
    )
    assert status_payload["selected_subsystem"] == "tooling"
    assert status_payload["bootstrap_retrieval_priority_memory"]["prioritized_subsystems"][0]["selected_subsystem"] == "tooling"
    assert status_payload["effective_bootstrap_retrieval_priority_summary"]["retained_only_subsystems"] == ["tooling"]


def test_supervisor_loop_clears_bootstrap_retrieval_priority_memory_after_clean_compare(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    meta_policy_path = tmp_path / "config" / "supervisor_meta_policy.json"
    meta_policy_path.parent.mkdir(parents=True, exist_ok=True)
    meta_policy_path.write_text(json.dumps({}), encoding="utf-8")
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(meta_policy_path),
    )
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: [])

    previous_bootstrap_retrieval_priority_memory = {
        "prioritized_subsystem_count": 1,
        "prioritized_subsystems": [
            {
                "selected_subsystem": "tooling",
                "retrieval_reuse_priority_score": 16,
                "retrieval_reuse_summary": {
                    "retrieval_backed_procedure_count": 1,
                    "trusted_retrieval_procedure_count": 1,
                    "verified_retrieval_command_count": 2,
                },
                "sticky_rounds_remaining": 2,
            }
        ],
    }

    decisions = module._build_round_actions(
        config=KernelConfig(),
        repo_root=Path(__file__).resolve().parents[1],
        policy=policy,
        queue_state={"active_leases": []},
        frontier_state={"frontier_candidates": []},
        promotion_plan_state={"promotion_candidates": []},
        promotion_pass_state={
            "results": [
                {
                    "selected_subsystem": "tooling",
                    "compare_status": "compared",
                    "compare_guard_reason": "",
                    "finalize_state": "retain",
                }
            ]
        },
        trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "overall_summary": {"clean_success_streak": 3},
        },
        recent_outcomes=[],
        previous_bootstrap_retrieval_priority_memory=previous_bootstrap_retrieval_priority_memory,
    )

    assert decisions["bootstrap_retrieval_priority_memory"]["prioritized_subsystems"] == []
    assert decisions["bootstrap_retrieval_priority_memory"]["clean_compare_cleared_subsystems"] == ["tooling"]
    assert decisions["effective_bootstrap_retrieval_priority_summary"]["prioritized_subsystems"] == []
    assert decisions["bootstrap_retrieval_priority_summary"]["retrieval_ranked_subsystems"] == []


def test_execute_action_writes_bootstrap_review_package_report(tmp_path):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    config = KernelConfig(improvement_reports_dir=reports_dir)
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    report_path = reports_dir / "supervisor_baseline_bootstrap_queue.json"

    result = module._execute_action(
        action={
            "kind": "prepare_bootstrap_review_package",
            "queue_name": "baseline_bootstrap",
            "queue_kind": "baseline_bootstrap",
            "autonomy_mode": "dry_run",
            "rollout_stage": "compare_only",
            "bootstrap_finalize_policy": "trusted",
            "trust_status": "bootstrap",
            "report_path": str(report_path),
            "entries": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_a",
                    "candidate_artifact_path": "candidates/a.json",
                    "promotion_block_reason_code": "shared_repo_bundle_coherence_regressed",
                    "compare_guard_reason": "shared_repo_bundle_coherence_regressed",
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "validation_family_compare_guard_reasons": [
                        "validation_family_generated_pass_rate_regressed",
                        "validation_family_novel_command_rate_regressed",
                    ],
                    "review_finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --scope-id scope_a --dry-run",
                }
            ],
        },
        config=config,
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round1",
    )

    assert result["returncode"] == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "prepare_bootstrap_review_package"
    assert payload["queue_name"] == "baseline_bootstrap"
    assert payload["queue_kind"] == "baseline_bootstrap"
    assert payload["autonomy_mode"] == "dry_run"
    assert payload["rollout_stage"] == "compare_only"
    assert payload["bootstrap_finalize_policy"] == "trusted"
    assert payload["trust_status"] == "bootstrap"
    assert payload["selected_subsystem"] == "transition_model"
    assert payload["scope_id"] == "scope_a"
    assert payload["candidate_artifact_path"] == "candidates/a.json"
    assert payload["promotion_block_reason_code"] == "shared_repo_bundle_coherence_regressed"
    assert payload["compare_guard_reason"] == "shared_repo_bundle_coherence_regressed"
    assert payload["finalize_skip_reason"] == "bootstrap_requires_review"
    assert payload["validation_family_compare_guard_reasons"] == [
        "validation_family_generated_pass_rate_regressed",
        "validation_family_novel_command_rate_regressed",
    ]
    assert payload["review_finalize_command"] == (
        "python scripts/finalize_latest_candidate_from_cycles.py --scope-id scope_a --dry-run"
    )
    assert payload["primary_entry"]["selected_subsystem"] == "transition_model"
    assert payload["summary"]["entry_count"] == 1
    assert payload["summary"]["promotion_block_reason_counts"] == {"shared_repo_bundle_coherence_regressed": 1}
    assert payload["summary"]["compare_guard_reason_counts"] == {"shared_repo_bundle_coherence_regressed": 1}
    assert payload["summary"]["finalize_skip_reason_counts"] == {"bootstrap_requires_review": 1}
    assert payload["summary"]["validation_compare_guard_reason_counts"] == {
        "validation_family_generated_pass_rate_regressed": 1,
        "validation_family_novel_command_rate_regressed": 1,
    }
    assert payload["entries"][0]["selected_subsystem"] == "transition_model"


def test_execute_action_writes_bootstrap_review_package_report_with_retrieval_reuse_summary(tmp_path):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    config = KernelConfig(improvement_reports_dir=reports_dir)
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    report_path = reports_dir / "supervisor_baseline_bootstrap_queue.json"

    result = module._execute_action(
        action={
            "kind": "prepare_bootstrap_review_package",
            "queue_name": "baseline_bootstrap",
            "queue_kind": "baseline_bootstrap",
            "autonomy_mode": "dry_run",
            "rollout_stage": "compare_only",
            "bootstrap_finalize_policy": "trusted",
            "trust_status": "bootstrap",
            "report_path": str(report_path),
            "entries": [
                {
                    "selected_subsystem": "tooling",
                    "scope_id": "scope_reuse",
                    "candidate_artifact_path": "candidates/reuse.json",
                    "promotion_block_reason_code": "",
                    "compare_guard_reason": "",
                    "finalize_skip_reason": "bootstrap_requires_review",
                    "validation_family_compare_guard_reasons": [],
                    "retrieval_reuse_summary": {
                        "retrieval_backed_procedure_count": 1,
                        "trusted_retrieval_procedure_count": 1,
                        "verified_retrieval_command_count": 2,
                    },
                    "retrieval_reuse_priority_score": 16,
                    "review_finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --scope-id scope_reuse --dry-run",
                }
            ],
        },
        config=config,
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round1",
    )

    assert result["returncode"] == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_subsystem"] == "tooling"
    assert payload["retrieval_reuse_summary"] == {
        "retrieval_backed_procedure_count": 1,
        "trusted_retrieval_procedure_count": 1,
        "verified_retrieval_command_count": 2,
    }
    assert payload["retrieval_reuse_priority_score"] == 16
    assert payload["effective_retrieval_reuse_summary"] == {
        "retrieval_backed_procedure_count": 1,
        "trusted_retrieval_procedure_count": 1,
        "verified_retrieval_command_count": 2,
    }
    assert payload["effective_retrieval_reuse_priority_score"] == 16
    assert payload["retained_retrieval_reuse_summary"] == {}
    assert payload["retained_retrieval_reuse_priority_score"] == 0
    assert payload["retrieval_reuse_priority_sticky_rounds_remaining"] == 0
    assert payload["summary"]["retrieval_ranked_entry_count"] == 1
    assert payload["summary"]["trusted_retrieval_entry_count"] == 1
    assert payload["summary"]["verified_retrieval_command_total"] == 2
    assert payload["summary"]["retrieval_reuse_priority_total"] == 16
    assert payload["summary"]["retained_retrieval_ranked_entry_count"] == 0


def test_execute_action_launches_bootstrap_generated_evidence_discovery_with_targeted_variant(
    tmp_path, monkeypatch
):
    module = _load_script_module("run_supervisor_loop.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    policy = module.SupervisorPolicy(
        autonomy_mode="dry_run",
        max_discovery_workers=2,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="operator_review",
        provider="mock-provider",
        model_name="mock-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )
    captured: dict[str, object] = {}

    def fake_command_result(*, command, cwd, timeout_seconds):
        captured["command"] = list(command)
        captured["cwd"] = str(cwd)
        captured["timeout_seconds"] = timeout_seconds
        return {"command": list(command), "returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_command_result", fake_command_result)

    result = module._execute_action(
        action={
            "kind": "launch_bootstrap_generated_evidence_discovery",
            "worker_count": 1,
            "generated_curriculum_budget_seconds": 30.0,
            "targets": [
                {
                    "selected_subsystem": "policy",
                    "selected_variant_id": "retrieval_caution",
                    "scope_id": "scope_policy",
                    "cycle_id": "cycle:policy",
                    "bootstrap_generated_evidence_guard_reasons": [
                        "policy_bootstrap_generated_evidence_missing"
                    ],
                }
            ],
        },
        config=config,
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round1",
    )

    assert result["returncode"] == 0
    command = captured["command"]
    assert command[0] == sys.executable
    assert "scripts/run_parallel_supervised_cycles.py" in command[1]
    assert "--include-curriculum" in command
    assert "--generated-curriculum-budget-seconds" in command
    assert "30.0" in command
    assert "--no-auto-diversify-subsystems" in command
    assert "--no-auto-diversify-variants" in command
    assert command.count("--subsystem") == 1
    assert "policy" in command
    assert command.count("--variant-id") == 1
    assert "retrieval_caution" in command
    assert "--provider" in command and "mock-provider" in command
    assert "--model" in command and "mock-model" in command


def test_execute_action_passes_bootstrap_finalize_flag_to_promotion_pass(tmp_path):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    config = KernelConfig(improvement_reports_dir=reports_dir)
    policy = module.SupervisorPolicy(
        autonomy_mode="promote",
        max_discovery_workers=1,
        discovery_task_limit=5,
        discovery_observation_budget_seconds=60.0,
        max_promotion_candidates=2,
        command_timeout_seconds=120,
        lane_failure_threshold=2,
        sleep_seconds=30.0,
        include_curriculum=False,
        include_failure_curriculum=False,
        generated_curriculum_budget_seconds=0.0,
        failure_curriculum_budget_seconds=0.0,
        bootstrap_finalize_policy="trusted",
        provider="mock",
        model_name="test-model",
        rollout_stage="compare_only",
        max_meta_promotions_per_round=1,
        meta_trust_clean_success_streak=2,
        meta_policy_path=str(tmp_path / "config" / "supervisor_meta_policy.json"),
    )

    seen = {}

    def fake_command_result(*, command, cwd, timeout_seconds):
        seen["command"] = list(command)
        return {"command": list(command), "returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

    module._command_result = fake_command_result
    result = module._execute_action(
        action={
            "kind": "run_promotion_pass",
            "limit": 2,
            "apply_finalize": True,
            "allow_bootstrap_finalize": True,
            "allow_subsystems": ["tooling"],
            "prioritized_subsystems": ["tooling"],
            "require_allow_subsystem_match": True,
            "allowed_bootstrap_subsystems": ["transition_model"],
            "blocked_subsystems": [],
        },
        config=config,
        policy=policy,
        repo_root=Path(__file__).resolve().parents[1],
        round_id="round1",
    )

    assert result["returncode"] == 0
    assert "--apply-finalize" in seen["command"]
    assert "--allow-bootstrap-finalize" in seen["command"]
    assert "--allow-subsystem" in seen["command"]
    assert "tooling" in seen["command"]
    assert "--prioritize-subsystem" in seen["command"]
    assert "--require-allow-subsystem-match" in seen["command"]
    assert "--allow-bootstrap-subsystem" in seen["command"]


def test_run_supervisor_loop_writes_status_history_and_report(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    improvement_root = tmp_path / "improvement"
    jobs_root = tmp_path / "jobs"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        delegated_job_queue_path=jobs_root / "queue.json",
        delegated_job_runtime_state_path=jobs_root / "runtime_state.json",
        unattended_trust_ledger_path=reports_dir / "unattended_trust_ledger.json",
    )
    improvement_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    config.delegated_job_queue_path.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        module,
        "_planner_ranked_subsystems",
        lambda config, worker_count: ["transition_model", "retrieval"],
    )

    commands: list[list[str]] = []

    def fake_run(command, cwd, text, capture_output, timeout):
        commands.append(list(command))
        script_name = Path(str(command[1])).name
        if script_name == "report_supervised_frontier.py":
            frontier_path = reports_dir / "supervised_parallel_frontier.json"
            frontier_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_parallel_frontier_report",
                        "summary": {"frontier_candidate_count": 1},
                        "frontier_candidates": [
                            {
                                "scope_id": "scope_a",
                                "cycle_id": "cycle:a",
                                "selected_subsystem": "transition_model",
                                "selected_variant_id": "regression_guard",
                                "generated_candidate": True,
                                "candidate_exists": True,
                                "candidate_artifact_path": "candidates/a.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(frontier_path), "stderr": ""})()
        if script_name == "report_frontier_promotion_plan.py":
            plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_plan",
                        "promotion_candidates": [
                            {
                                "scope_id": "scope_a",
                                "cycle_id": "cycle:a",
                                "selected_subsystem": "transition_model",
                                "selected_variant_id": "regression_guard",
                                "candidate_artifact_path": "candidates/a.json",
                                "promotion_score": 7.0,
                                "compare_command": "python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path candidates/a.json --before-cycle-id cycle:a",
                                "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem transition_model --scope-id scope_a --candidate-index 0 --dry-run",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(plan_path), "stderr": ""})()
        if script_name == "run_frontier_promotion_pass.py":
            pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
            pass_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_pass",
                        "summary": {"candidate_count": 1, "executed_candidates": 1, "apply_finalize": False},
                        "results": [
                            {
                                "selected_subsystem": "transition_model",
                                "finalize_state": "retain",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(pass_path), "stderr": ""})()
        if script_name == "run_parallel_supervised_cycles.py":
            batch_path = reports_dir / "parallel_supervised_preview_test.json"
            batch_path.write_text(
                json.dumps(
                    {
                        "report_kind": "parallel_supervised_preview_report",
                        "summary": {"completed_runs": 1},
                        "runs": [{"scope_id": "supervisor_scope_1", "selected_subsystem": "transition_model"}],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(batch_path), "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_supervisor_loop.py",
            "--autonomy-mode",
            "dry_run",
            "--max-rounds",
            "1",
            "--provider",
            "mock",
            "--model",
            "test-model",
            "--max-discovery-workers",
            "1",
            "--max-promotion-candidates",
            "1",
            "--rollout-stage",
            "compare_only",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "supervisor_loop_report"
    assert payload["round_count"] == 1
    latest_round = payload["rounds"][0]
    assert latest_round["machine_state"]["trust_overall_assessment"]["status"] in {"bootstrap", "trusted"}
    assert any(entry["kind"] == "refresh_frontier" for entry in latest_round["executions"])
    assert any(entry["kind"] == "refresh_promotion_plan" for entry in latest_round["executions"])
    assert any(entry["kind"] == "run_promotion_pass" for entry in latest_round["executions"])
    assert any(entry["kind"] == "launch_discovery" for entry in latest_round["executions"])

    status_path = reports_dir / "supervisor_loop_status.json"
    history_path = reports_dir / "supervisor_loop_history.jsonl"
    assert status_path.exists()
    assert history_path.exists()
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["report_kind"] == "supervisor_loop_status"
    assert status_payload["trust_status"] in {"", "bootstrap", "trusted"}
    assert status_payload["bootstrap_finalize_policy"] == "operator_review"
    assert "selected_queue_kind" in status_payload
    assert "selected_queue_entry" in status_payload
    assert status_payload["machine_state"]["meta_policy"]["exists"] is True
    assert "claim_ledger" in status_payload["machine_state"]
    assert "lane_allocator" in status_payload["machine_state"]
    assert "rollback_plan" in status_payload["machine_state"]
    history_entries = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(history_entries) == 1
    assert any("run_frontier_promotion_pass.py" in " ".join(command) for command in commands)


def test_run_supervisor_loop_surfaces_machine_readable_promotion_block_summary(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    improvement_root = tmp_path / "improvement"
    jobs_root = tmp_path / "jobs"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        delegated_job_queue_path=jobs_root / "queue.json",
        delegated_job_runtime_state_path=jobs_root / "runtime_state.json",
        unattended_trust_ledger_path=reports_dir / "unattended_trust_ledger.json",
    )
    improvement_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    config.delegated_job_queue_path.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["retrieval"])

    def fake_run(command, cwd, text, capture_output, timeout):
        script_name = Path(str(command[1])).name
        if script_name == "report_supervised_frontier.py":
            frontier_path = reports_dir / "supervised_parallel_frontier.json"
            frontier_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_parallel_frontier_report",
                        "summary": {"frontier_candidate_count": 1},
                        "frontier_candidates": [
                            {
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:compare",
                                "selected_subsystem": "tooling",
                                "selected_variant_id": "procedure_promotion",
                                "generated_candidate": True,
                                "candidate_exists": True,
                                "candidate_artifact_path": "candidates/tooling.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(frontier_path), "stderr": ""})()
        if script_name == "report_frontier_promotion_plan.py":
            plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_plan",
                        "summary": {
                            "validation_compare_guard_reason_counts": {
                                "validation_family_generated_pass_rate_regressed": 1
                            }
                        },
                        "promotion_candidates": [
                            {
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:compare",
                                "selected_subsystem": "tooling",
                                "selected_variant_id": "procedure_promotion",
                                "candidate_artifact_path": "candidates/tooling.json",
                                "promotion_score": 6.0,
                                "validation_family_compare_guard_reasons": [
                                    "validation_family_generated_pass_rate_regressed"
                                ],
                                "compare_command": "python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path candidates/tooling.json --before-cycle-id cycle:tooling:compare",
                                "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tooling --scope-id scope_tooling --candidate-index 0 --dry-run",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(plan_path), "stderr": ""})()
        if script_name == "run_frontier_promotion_pass.py":
            pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
            pass_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_pass",
                        "summary": {"candidate_count": 1, "executed_candidates": 1, "apply_finalize": False},
                        "results": [
                            {
                                "selected_subsystem": "tooling",
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:compare",
                                "candidate_artifact_path": "candidates/tooling.json",
                                "compare_status": "compare_failed",
                                "compare_guard_reason": "shared_repo_bundle_coherence_regressed",
                                "finalize_state": "",
                                "finalize_skipped": True,
                                "finalize_skip_reason": "compare_failed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(pass_path), "stderr": ""})()
        if script_name == "run_parallel_supervised_cycles.py":
            batch_path = reports_dir / "parallel_supervised_preview_test.json"
            batch_path.write_text(
                json.dumps(
                    {
                        "report_kind": "parallel_supervised_preview_report",
                        "summary": {"completed_runs": 1},
                        "runs": [{"scope_id": "supervisor_scope_1", "selected_subsystem": "retrieval"}],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(batch_path), "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_supervisor_loop.py",
            "--autonomy-mode",
            "dry_run",
            "--max-rounds",
            "1",
            "--provider",
            "mock",
            "--model",
            "test-model",
            "--max-discovery-workers",
            "1",
            "--max-promotion-candidates",
            "1",
            "--rollout-stage",
            "compare_only",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    latest_round = payload["rounds"][0]
    assert latest_round["decisions"]["precompare_guard_summary"]["validation_compare_guard_reason_counts"] == {
        "validation_family_generated_pass_rate_regressed": 1
    }
    assert latest_round["decisions"]["precompare_guard_summary"]["validation_guarded_candidate_count"] == 1
    assert latest_round["decisions"]["promotion_block_summary"]["blocked_promotion_count"] == 1
    assert latest_round["decisions"]["promotion_block_summary"]["promotion_block_reason_counts"] == {
        "shared_repo_bundle_coherence_regressed": 1
    }
    assert latest_round["decisions"]["promotion_block_summary"]["compare_guard_reason_counts"] == {
        "shared_repo_bundle_coherence_regressed": 1
    }
    status_payload = json.loads((reports_dir / "supervisor_loop_status.json").read_text(encoding="utf-8"))
    assert status_payload["precompare_guard_summary"]["validation_compare_guard_reason_counts"] == {
        "validation_family_generated_pass_rate_regressed": 1
    }
    assert status_payload["validation_guard_memory"]["guarded_subsystems"] == [
        {
            "selected_subsystem": "tooling",
            "guarded_candidate_count": 1,
            "validation_guard_reason_count": 1,
            "validation_guard_severity": 3,
            "validation_family_compare_guard_reasons": [
                "validation_family_generated_pass_rate_regressed"
            ],
            "sticky_rounds_remaining": 2,
        }
    ]
    assert status_payload["effective_validation_guard_pressure_summary"]["guarded_subsystem_count"] == 1
    assert status_payload["promotion_block_summary"]["blocked_promotion_count"] == 1
    assert status_payload["promotion_block_summary"]["promotion_block_reason_counts"] == {
        "shared_repo_bundle_coherence_regressed": 1
    }


def test_run_supervisor_loop_rebuilds_lane_signal_from_fresh_promotion_pass(tmp_path, monkeypatch):
    module = _load_script_module("run_supervisor_loop.py")
    reports_dir = tmp_path / "reports"
    improvement_root = tmp_path / "improvement"
    jobs_root = tmp_path / "jobs"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=improvement_root / "cycles.jsonl",
        delegated_job_queue_path=jobs_root / "queue.json",
        delegated_job_runtime_state_path=jobs_root / "runtime_state.json",
        unattended_trust_ledger_path=reports_dir / "unattended_trust_ledger.json",
    )
    improvement_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    config.delegated_job_queue_path.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(module, "_planner_ranked_subsystems", lambda config, worker_count: ["tooling", "retrieval"])

    def fake_write_unattended_trust_ledger(_config):
        payload = {
            "overall_assessment": {"passed": True, "status": "bootstrap"},
            "overall_summary": {"clean_success_streak": 0},
        }
        _config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        _config.unattended_trust_ledger_path.write_text(json.dumps(payload), encoding="utf-8")
        return _config.unattended_trust_ledger_path

    monkeypatch.setattr(module, "write_unattended_trust_ledger", fake_write_unattended_trust_ledger)

    commands: list[list[str]] = []

    def fake_run(command, cwd, text, capture_output, timeout):
        commands.append(list(command))
        script_name = Path(str(command[1])).name
        if script_name == "report_supervised_frontier.py":
            frontier_path = reports_dir / "supervised_parallel_frontier.json"
            frontier_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_parallel_frontier_report",
                        "summary": {"frontier_candidate_count": 1},
                        "frontier_candidates": [
                            {
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:bootstrap",
                                "selected_subsystem": "tooling",
                                "selected_variant_id": "procedure_promotion",
                                "generated_candidate": True,
                                "candidate_exists": True,
                                "candidate_artifact_path": "candidates/tooling.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(frontier_path), "stderr": ""})()
        if script_name == "report_frontier_promotion_plan.py":
            plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_plan",
                        "promotion_candidates": [
                            {
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:bootstrap",
                                "selected_subsystem": "tooling",
                                "selected_variant_id": "procedure_promotion",
                                "candidate_artifact_path": "candidates/tooling.json",
                                "promotion_score": 6.5,
                                "compare_command": "python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path candidates/tooling.json --before-cycle-id cycle:tooling:bootstrap",
                                "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tooling --scope-id scope_tooling --candidate-index 0 --dry-run",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(plan_path), "stderr": ""})()
        if script_name == "run_frontier_promotion_pass.py":
            pass_path = reports_dir / "supervised_frontier_promotion_pass.json"
            pass_path.write_text(
                json.dumps(
                    {
                        "report_kind": "supervised_frontier_promotion_pass",
                        "summary": {
                            "candidate_count": 1,
                            "executed_candidates": 1,
                            "apply_finalize": False,
                            "bootstrap_candidates": 1,
                        },
                        "results": [
                            {
                                "selected_subsystem": "tooling",
                                "scope_id": "scope_tooling",
                                "cycle_id": "cycle:tooling:bootstrap",
                                "candidate_artifact_path": "candidates/tooling.json",
                                "compare_status": "bootstrap_first_retain",
                                "finalize_state": "",
                                "finalize_skipped": False,
                                "finalize_skip_reason": "",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(pass_path), "stderr": ""})()
        if script_name == "run_parallel_supervised_cycles.py":
            batch_path = reports_dir / "parallel_supervised_preview_test.json"
            batch_path.write_text(
                json.dumps(
                    {
                        "report_kind": "parallel_supervised_preview_report",
                        "summary": {"completed_runs": 1},
                        "runs": [{"scope_id": "supervisor_scope_1", "selected_subsystem": "retrieval"}],
                    }
                ),
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": str(batch_path), "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_supervisor_loop.py",
            "--autonomy-mode",
            "dry_run",
            "--max-rounds",
            "1",
            "--provider",
            "mock",
            "--model",
            "test-model",
            "--max-discovery-workers",
            "1",
            "--max-promotion-candidates",
            "1",
            "--rollout-stage",
            "compare_only",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    latest_round = payload["rounds"][0]
    paused = latest_round["decisions"]["paused_subsystems"]
    assert paused["tooling"]["lane_signal_queue"] == "trust_streak_accumulation"
    assert paused["tooling"]["remediation_queues"] == ["trust_streak_accumulation", "baseline_bootstrap"]
    remediation = latest_round["decisions"]["bootstrap_remediation_queues"]
    assert remediation["trust_streak_accumulation"][0]["selected_subsystem"] == "tooling"
    assert remediation["baseline_bootstrap"][0]["selected_subsystem"] == "tooling"
    assert any(entry["kind"] == "prepare_trust_streak_recovery_package" for entry in latest_round["executions"])
    assert any(entry["kind"] == "prepare_bootstrap_review_package" for entry in latest_round["executions"])
    trust_queue_payload = json.loads(
        (reports_dir / "supervisor_trust_streak_recovery_queue.json").read_text(encoding="utf-8")
    )
    assert trust_queue_payload["queue_kind"] == "trust_streak_accumulation"
    assert trust_queue_payload["trust_status"] == "bootstrap"
    assert trust_queue_payload["rollout_stage"] == "compare_only"
    assert trust_queue_payload["autonomy_mode"] == "dry_run"
    assert trust_queue_payload["bootstrap_finalize_policy"] == "operator_review"
    status_payload = json.loads((reports_dir / "supervisor_loop_status.json").read_text(encoding="utf-8"))
    assert status_payload["trust_status"] == "bootstrap"
    assert status_payload["rollout_stage"] == "compare_only"
    assert status_payload["bootstrap_finalize_policy"] == "operator_review"
    assert status_payload["selected_queue_kind"] == "trust_streak_accumulation"
    assert status_payload["selected_subsystem"] == "tooling"
    assert status_payload["scope_id"] == remediation["trust_streak_accumulation"][0]["scope_id"]
    assert status_payload["selected_queue_entry"]["selected_subsystem"] == "tooling"
    discovery_commands = [command for command in commands if Path(str(command[1])).name == "run_parallel_supervised_cycles.py"]
    assert len(discovery_commands) == 1
    assert "retrieval" in discovery_commands[0]
    assert "tooling" not in discovery_commands[0]


def test_run_frontier_promotion_pass_skips_blocked_subsystems(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_meta",
                        "cycle_id": "cycle:meta",
                        "selected_subsystem": "trust",
                        "selected_variant_id": "safety",
                        "candidate_artifact_path": "candidates/meta.json",
                        "promotion_score": 9.0,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem trust --artifact-path candidates/meta.json --before-cycle-id cycle:meta",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem trust --scope-id scope_meta --candidate-index 0 --dry-run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
            "--block-subsystem",
            "trust",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["skipped_candidates"] == 1
    assert payload["summary"]["compare_failures"] == 0
    assert payload["policy"]["blocked_subsystems"] == ["trust"]
    assert payload["results"][0]["promotion_route"] == "skipped_by_policy"


def test_run_frontier_promotion_pass_applies_allowlist_and_priority_before_limit(tmp_path, monkeypatch):
    module = _load_script_module("run_frontier_promotion_pass.py")
    reports_dir = tmp_path / "reports"
    plan_path = reports_dir / "supervised_frontier_promotion_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "report_kind": "supervised_frontier_promotion_plan",
                "promotion_candidates": [
                    {
                        "scope_id": "scope_trust",
                        "cycle_id": "cycle:trust",
                        "selected_subsystem": "trust",
                        "selected_variant_id": "breadth",
                        "candidate_artifact_path": "candidates/trust.json",
                        "promotion_score": 9.0,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem trust --artifact-path candidates/trust.json --before-cycle-id cycle:trust",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem trust --scope-id scope_trust --candidate-index 0 --dry-run",
                    },
                    {
                        "scope_id": "scope_tooling",
                        "cycle_id": "cycle:tooling",
                        "selected_subsystem": "tooling",
                        "selected_variant_id": "bridge",
                        "candidate_artifact_path": "candidates/tooling.json",
                        "promotion_score": 8.0,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path candidates/tooling.json --before-cycle-id cycle:tooling",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem tooling --scope-id scope_tooling --candidate-index 0 --dry-run",
                    },
                    {
                        "scope_id": "scope_policy",
                        "cycle_id": "cycle:policy",
                        "selected_subsystem": "policy",
                        "selected_variant_id": "planner",
                        "candidate_artifact_path": "candidates/policy.json",
                        "promotion_score": 7.5,
                        "compare_command": "python scripts/compare_retained_baseline.py --subsystem policy --artifact-path candidates/policy.json --before-cycle-id cycle:policy",
                        "finalize_command": "python scripts/finalize_latest_candidate_from_cycles.py --frontier-report reports/frontier.json --subsystem policy --scope-id scope_policy --candidate-index 0 --dry-run",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(improvement_reports_dir=reports_dir)
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)

    def fake_run(command, cwd, shell, text, capture_output):
        assert "policy" in command
        if "compare_retained_baseline.py" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "baseline_pass_rate=0.70 current_pass_rate=0.75", "stderr": ""})()
        if "finalize_latest_candidate_from_cycles.py" in command:
            return type("Completed", (), {"returncode": 0, "stdout": "cycle_id=cycle:policy subsystem=policy state=retain reason=improved", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_promotion_pass.py",
            "--promotion-plan",
            str(plan_path),
            "--allow-subsystem",
            "tooling",
            "--allow-subsystem",
            "policy",
            "--prioritize-subsystem",
            "policy",
            "--require-allow-subsystem-match",
            "--limit",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 2
    assert payload["summary"]["executed_candidates"] == 1
    assert payload["policy"]["allowed_subsystems"] == ["policy", "tooling"]
    assert payload["policy"]["prioritized_subsystems"] == ["policy"]
    assert payload["policy"]["require_allow_subsystem_match"] is True
    assert [result["selected_subsystem"] for result in payload["results"]] == ["policy"]


def test_migrate_tolbert_shared_store_script_prunes_unreferenced_legacy_candidates(tmp_path, monkeypatch):
    module = _load_script_module("migrate_tolbert_shared_store.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "tolbert_model_artifact.json",
        qwen_adapter_artifact_path=tmp_path / "qwen_adapter" / "qwen_adapter_artifact.json",
    )
    legacy_root = config.candidate_artifacts_root / "tolbert_model" / "cycle_legacy"
    legacy_bundle = legacy_root / "tolbert_model_artifact"
    legacy_bundle.mkdir(parents=True, exist_ok=True)
    (legacy_bundle / "marker.txt").write_text("legacy", encoding="utf-8")
    config.tolbert_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["migrate_tolbert_shared_store.py", "--apply"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert str(legacy_root) in report["removed_legacy_cycle_roots"]
    assert not legacy_root.exists()


def test_migrate_tolbert_shared_store_script_removes_empty_candidate_dirs(tmp_path, monkeypatch):
    module = _load_script_module("migrate_tolbert_shared_store.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "tolbert_model_artifact.json",
    )
    empty_cycle = config.candidate_artifacts_root / "tolbert_model" / "cycle_empty"
    empty_cycle.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["migrate_tolbert_shared_store.py", "--apply"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert str(empty_cycle) in report["removed_empty_candidate_dirs"]
    assert not empty_cycle.exists()

    for script_name, output_path, kind in (
        ("synthesize_benchmarks.py", config.benchmark_candidates_path, "benchmark_candidate_set"),
        ("propose_retrieval_update.py", config.retrieval_proposals_path, "retrieval_policy_set"),
        ("propose_tolbert_model_update.py", config.tolbert_model_artifact_path, "tolbert_model_bundle"),
        ("propose_qwen_adapter_update.py", config.qwen_adapter_artifact_path, "qwen_adapter_bundle"),
        ("propose_world_model_update.py", config.world_model_proposals_path, "world_model_policy_set"),
        ("propose_trust_update.py", config.trust_proposals_path, "trust_policy_set"),
        ("propose_recovery_update.py", config.recovery_proposals_path, "recovery_policy_set"),
        ("propose_delegation_update.py", config.delegation_proposals_path, "delegated_runtime_policy_set"),
        ("propose_operator_policy_update.py", config.operator_policy_proposals_path, "operator_policy_set"),
        ("propose_transition_model_update.py", config.transition_model_proposals_path, "transition_model_policy_set"),
        ("propose_capability_update.py", config.capability_modules_path, "capability_module_set"),
        ("synthesize_verifiers.py", config.verifier_contracts_path, "verifier_candidate_set"),
    ):
        module = _load_script_module(script_name)
        monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
        if hasattr(module, "run_eval"):
            monkeypatch.setattr(
                module,
                "run_eval",
                lambda **kwargs: EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2),
            )
        if hasattr(module, "build_tolbert_model_candidate_artifact"):
            monkeypatch.setattr(
                module,
                "build_tolbert_model_candidate_artifact",
                lambda **kwargs: {
                    "spec_version": "asi_v1",
                    "artifact_kind": "tolbert_model_bundle",
                    "lifecycle_state": "candidate",
                    "generation_focus": "balanced",
                    "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
                    "dataset_manifest": {"total_examples": 4},
                    "decoder_policy": {"allow_retrieval_guidance": True, "allow_skill_commands": True, "allow_task_suggestions": True, "allow_stop_decision": True, "min_stop_completion_ratio": 0.95, "max_task_suggestions": 3},
                    "rollout_policy": {"predicted_progress_gain_weight": 3.0, "predicted_conflict_penalty_weight": 4.0, "predicted_preserved_bonus_weight": 1.0, "predicted_workflow_bonus_weight": 1.5, "latent_progress_bonus_weight": 1.0, "latent_risk_penalty_weight": 2.0, "recover_from_stall_bonus_weight": 1.5, "stop_completion_weight": 8.0, "stop_missing_expected_penalty_weight": 6.0, "stop_forbidden_penalty_weight": 6.0, "stop_preserved_penalty_weight": 4.0, "stable_stop_bonus_weight": 1.5},
                    "runtime_paths": {"config_path": "config.json", "checkpoint_path": "checkpoint.pt", "nodes_path": "nodes.jsonl", "label_map_path": "label_map.json", "source_spans_paths": ["spans.jsonl"], "cache_paths": ["cache.pt"]},
                    "proposals": [{"area": "balanced", "priority": 5, "reason": "test"}],
                },
            )
        if hasattr(module, "build_qwen_adapter_candidate_artifact"):
            monkeypatch.setattr(
                module,
                "build_qwen_adapter_candidate_artifact",
                lambda **kwargs: {
                    "spec_version": "asi_v1",
                    "artifact_kind": "qwen_adapter_bundle",
                    "lifecycle_state": "candidate",
                    "generation_focus": "coding_lane_sft",
                    "runtime_role": "support_runtime",
                    "training_objective": "qlora_sft",
                    "base_model_name": "Qwen/Qwen3.5-9B",
                    "training_dataset_manifest": {"total_examples": 4},
                    "runtime_policy": {"allow_primary_routing": False, "allow_teacher_generation": True},
                    "retention_gate": {"disallow_liftoff_authority": True},
                    "runtime_paths": {"adapter_output_dir": "adapter", "merged_output_dir": "merged"},
                },
            )
        monkeypatch.setattr(sys, "argv", [script_name])
        stream = StringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        module.main()

        assert output_path.exists()
        assert stream.getvalue().strip() == str(output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["artifact_kind"] == kind


def test_materialize_retrieval_asset_bundle_script_uses_config_path(tmp_path, monkeypatch):
    module = _load_script_module("materialize_retrieval_asset_bundle.py")
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    checkpoint_path = tmp_path / "tolbert" / "checkpoint.pt"
    cache_path = tmp_path / "tolbert" / "cache.pt"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "asset_strategy": "balanced_rebuild",
                "asset_controls": {"include_tool_candidate_spans": False},
                "asset_rebuild_plan": {"rebuild_required": True},
                "overrides": {"tolbert_branch_results": 2},
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        retrieval_proposals_path=retrieval_path,
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_cache_paths=(str(cache_path),),
    )

    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["materialize_retrieval_asset_bundle.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert bundle_manifest_path.exists()
    assert stream.getvalue().strip() == str(bundle_manifest_path)
    payload = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "tolbert_retrieval_asset_bundle"


def test_proposal_scripts_write_prompt_and_curriculum_artifacts(tmp_path, monkeypatch):
    prompt_module = _load_script_module("propose_prompt_update.py")
    curriculum_module = _load_script_module("propose_curriculum_update.py")
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
    )

    monkeypatch.setattr(prompt_module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(curriculum_module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        prompt_module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2),
    )
    monkeypatch.setattr(
        curriculum_module,
        "run_eval",
        lambda **kwargs: EvalMetrics(
            total=10,
            passed=8,
            generated_total=10,
            generated_passed=5,
            generated_by_kind={"failure_recovery": 5},
            generated_passed_by_kind={"failure_recovery": 2},
        ),
    )
    monkeypatch.setattr(prompt_module.ImprovementPlanner, "failure_counts", lambda self: {"command_failure": 2})

    prompt_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", prompt_stream)
    monkeypatch.setattr(sys, "argv", ["propose_prompt_update.py"])
    prompt_module.main()
    prompt_payload = json.loads(config.prompt_proposals_path.read_text(encoding="utf-8"))

    assert prompt_stream.getvalue().strip() == str(config.prompt_proposals_path)
    assert prompt_payload["artifact_kind"] == "prompt_proposal_set"
    assert prompt_payload["control_schema"] == "policy_behavior_controls_v3"
    assert "planner_controls" in prompt_payload
    assert "improvement_planner_controls" in prompt_payload
    assert "role_directives" in prompt_payload
    assert prompt_payload["retention_gate"]["require_failure_recovery_non_regression"] is True
    assert prompt_payload["retention_gate"]["max_regressed_families"] == 0
    assert prompt_payload["proposals"]

    curriculum_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", curriculum_stream)
    monkeypatch.setattr(sys, "argv", ["propose_curriculum_update.py"])
    curriculum_module.main()
    curriculum_payload = json.loads(config.curriculum_proposals_path.read_text(encoding="utf-8"))

    assert curriculum_stream.getvalue().strip() == str(config.curriculum_proposals_path)
    assert curriculum_payload["artifact_kind"] == "curriculum_proposal_set"
    assert curriculum_payload["control_schema"] == "curriculum_behavior_controls_v3"
    assert curriculum_payload["retention_gate"]["max_regressed_families"] == 0
    assert curriculum_payload["retention_gate"]["max_generated_regressed_families"] == 0


def test_promote_runtime_artifacts_normalizes_live_runtime_files(tmp_path, monkeypatch):
    module = _load_script_module("promote_runtime_artifacts.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    benchmark_path = tmp_path / "benchmarks" / "benchmark_candidates.json"
    tools_path = tmp_path / "tools" / "tool_candidates.json"
    verifier_path = tmp_path / "verifiers" / "verifier_contracts.json"
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    world_model_path = tmp_path / "world_model" / "world_model_proposals.json"
    trust_path = tmp_path / "trust" / "trust_proposals.json"
    recovery_path = tmp_path / "recovery" / "recovery_proposals.json"
    delegation_path = tmp_path / "delegation" / "delegation_proposals.json"
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    transition_model_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    curriculum_path = tmp_path / "curriculum" / "curriculum_proposals.json"
    capability_path = tmp_path / "config" / "capabilities.json"
    universe_contract_path = tmp_path / "universe" / "universe_contract.json"
    universe_constitution_path = tmp_path / "universe" / "universe_constitution.json"
    operating_envelope_path = tmp_path / "universe" / "operating_envelope.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    tools_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    world_model_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    delegation_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    transition_model_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    universe_contract_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps({"artifact_kind": "retrieval_policy_set", "proposals": [{"proposal_id": "retrieval:test"}]}),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps({"artifact_kind": "benchmark_candidate_set", "proposals": [{"proposal_id": "benchmark:test"}]}),
        encoding="utf-8",
    )
    tools_path.write_text(
        json.dumps([{"tool_id": "tool:test:primary", "source_task_id": "hello_task"}]),
        encoding="utf-8",
    )
    verifier_path.write_text(
        json.dumps({"artifact_kind": "verifier_candidate_set", "proposals": [{"proposal_id": "verifier:test"}]}),
        encoding="utf-8",
    )
    prompt_path.write_text(
        json.dumps({"artifact_kind": "prompt_proposal_set", "proposals": [{"proposal_id": "prompt:test"}]}),
        encoding="utf-8",
    )
    world_model_path.write_text(
        json.dumps({"artifact_kind": "world_model_policy_set", "proposals": [{"proposal_id": "world_model:test"}]}),
        encoding="utf-8",
    )
    trust_path.write_text(
        json.dumps({"artifact_kind": "trust_policy_set", "proposals": [{"proposal_id": "trust:test"}]}),
        encoding="utf-8",
    )
    recovery_path.write_text(
        json.dumps({"artifact_kind": "recovery_policy_set", "proposals": [{"proposal_id": "recovery:test"}]}),
        encoding="utf-8",
    )
    delegation_path.write_text(
        json.dumps({"artifact_kind": "delegated_runtime_policy_set", "proposals": [{"proposal_id": "delegation:test"}]}),
        encoding="utf-8",
    )
    operator_policy_path.write_text(
        json.dumps({"artifact_kind": "operator_policy_set", "proposals": [{"proposal_id": "operator_policy:test"}]}),
        encoding="utf-8",
    )
    transition_model_path.write_text(
        json.dumps(
            {
                "artifact_kind": "transition_model_policy_set",
                "signatures": [{"signal": "no_state_progress", "command": "false", "support": 1, "regressions": []}],
                "proposals": [{"proposal_id": "transition_model:test"}],
            }
        ),
        encoding="utf-8",
    )
    curriculum_path.write_text(
        json.dumps({"artifact_kind": "curriculum_proposal_set", "proposals": [{"proposal_id": "curriculum:test"}]}),
        encoding="utf-8",
    )
    capability_path.write_text(
        json.dumps({"modules": [{"module_id": "github", "enabled": True, "capabilities": ["github_read"]}]}),
        encoding="utf-8",
    )
    universe_contract_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["legacy universe invariant"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "action_risk_controls": {"verification_bonus": 4},
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "blocked",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.github.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["python", "pytest"],
                "proposals": [{"proposal_id": "universe:test"}],
            }
        ),
        encoding="utf-8",
    )

    config = KernelConfig(
        improvement_cycles_path=cycles_path,
        retrieval_proposals_path=retrieval_path,
        benchmark_candidates_path=benchmark_path,
        tool_candidates_path=tools_path,
        verifier_contracts_path=verifier_path,
        prompt_proposals_path=prompt_path,
        world_model_proposals_path=world_model_path,
        trust_proposals_path=trust_path,
        recovery_proposals_path=recovery_path,
        delegation_proposals_path=delegation_path,
        operator_policy_proposals_path=operator_policy_path,
        transition_model_proposals_path=transition_model_path,
        curriculum_proposals_path=curriculum_path,
        capability_modules_path=capability_path,
        universe_contract_path=universe_contract_path,
        universe_constitution_path=universe_constitution_path,
        operating_envelope_path=operating_envelope_path,
    )
    monkeypatch.setattr(module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(sys, "argv", ["promote_runtime_artifacts.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tool_payload = json.loads(tools_path.read_text(encoding="utf-8"))
    verifier_payload = json.loads(verifier_path.read_text(encoding="utf-8"))
    prompt_payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    world_model_payload = json.loads(world_model_path.read_text(encoding="utf-8"))
    trust_payload = json.loads(trust_path.read_text(encoding="utf-8"))
    recovery_payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    delegation_payload = json.loads(delegation_path.read_text(encoding="utf-8"))
    operator_policy_payload = json.loads(operator_policy_path.read_text(encoding="utf-8"))
    transition_model_payload = json.loads(transition_model_path.read_text(encoding="utf-8"))
    curriculum_payload = json.loads(curriculum_path.read_text(encoding="utf-8"))
    capability_payload = json.loads(capability_path.read_text(encoding="utf-8"))
    universe_contract_payload = json.loads(universe_contract_path.read_text(encoding="utf-8"))
    universe_constitution_payload = json.loads(universe_constitution_path.read_text(encoding="utf-8"))
    operating_envelope_payload = json.loads(operating_envelope_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert retrieval_payload["lifecycle_state"] == "proposed"
    assert retrieval_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert benchmark_payload["lifecycle_state"] == "proposed"
    assert tool_payload["artifact_kind"] == "tool_candidate_set"
    assert tool_payload["candidates"][0]["promotion_stage"] == "candidate_procedure"
    assert verifier_payload["artifact_kind"] == "verifier_candidate_set"
    assert verifier_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert prompt_payload["artifact_kind"] == "prompt_proposal_set"
    assert prompt_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert world_model_payload["artifact_kind"] == "world_model_policy_set"
    assert world_model_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert trust_payload["artifact_kind"] == "trust_policy_set"
    assert trust_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert recovery_payload["artifact_kind"] == "recovery_policy_set"
    assert recovery_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert delegation_payload["artifact_kind"] == "delegated_runtime_policy_set"
    assert delegation_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert operator_policy_payload["artifact_kind"] == "operator_policy_set"
    assert operator_policy_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert transition_model_payload["artifact_kind"] == "transition_model_policy_set"
    assert transition_model_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert curriculum_payload["artifact_kind"] == "curriculum_proposal_set"
    assert curriculum_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert capability_payload["artifact_kind"] == "capability_module_set"
    assert capability_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert universe_constitution_payload["artifact_kind"] == "universe_constitution"
    assert universe_constitution_payload["lifecycle_state"] == "retained"
    assert universe_constitution_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert "legacy universe invariant" in universe_constitution_payload["invariants"]
    assert operating_envelope_payload["artifact_kind"] == "operating_envelope"
    assert operating_envelope_payload["lifecycle_state"] == "retained"
    assert operating_envelope_payload["generation_context"]["migrated_from_legacy_runtime_artifact"] is True
    assert operating_envelope_payload["allowed_http_hosts"] == ["api.github.com"]
    assert universe_contract_payload["artifact_kind"] == "universe_contract"
    assert universe_contract_payload["lifecycle_state"] == "retained"
    assert universe_contract_payload["generation_context"]["synchronized_from_split_universe_bundle"] is True
    assert universe_contract_payload["allowed_http_hosts"] == ["api.github.com"]
    assert len(records) == 17
    assert any(record["subsystem"] == "tolbert_model" for record in records)
    assert any(record["subsystem"] == "universe_constitution" for record in records)
    assert any(record["subsystem"] == "operating_envelope" for record in records)
    assert not any(record["subsystem"] == "universe" for record in records)


def test_proposal_scripts_expose_variant_focus_flags(tmp_path, monkeypatch):
    prompt_module = _load_script_module("propose_prompt_update.py")
    curriculum_module = _load_script_module("propose_curriculum_update.py")
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )

    monkeypatch.setattr(prompt_module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(curriculum_module, "KernelConfig", lambda config=config: config)
    monkeypatch.setattr(
        prompt_module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2),
    )
    monkeypatch.setattr(
        curriculum_module,
        "run_eval",
        lambda **kwargs: EvalMetrics(
            total=10,
            passed=8,
            generated_total=10,
            generated_passed=5,
            generated_by_kind={"failure_recovery": 5},
            generated_passed_by_kind={"failure_recovery": 2},
            generated_by_benchmark_family={"workflow": 5},
            generated_passed_by_benchmark_family={"workflow": 2},
        ),
    )
    monkeypatch.setattr(prompt_module.ImprovementPlanner, "failure_counts", lambda self: {"command_failure": 2})

    monkeypatch.setattr(sys, "argv", ["propose_prompt_update.py", "--focus", "verifier_alignment"])
    prompt_module.main()
    prompt_payload = json.loads(config.prompt_proposals_path.read_text(encoding="utf-8"))
    assert prompt_payload["generation_focus"] == "verifier_alignment"

    monkeypatch.setattr(
        sys,
        "argv",
        ["propose_curriculum_update.py", "--focus", "benchmark_family", "--family", "workflow"],
    )
    curriculum_module.main()
    curriculum_payload = json.loads(config.curriculum_proposals_path.read_text(encoding="utf-8"))
    assert curriculum_payload["generation_focus"] == "benchmark_family"
