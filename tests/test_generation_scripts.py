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
        monkeypatch.setattr(sys, "argv", argv)
        stream = StringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        module.main()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload[field] == expected


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
                    "observation_timed_out": timed_out,
                    "observation_budget_exceeded": timed_out,
                    "observation_warning": "timed out" if timed_out else "",
                    "observation_elapsed_seconds": 4.2 if timed_out else 1.1,
                    "observation_current_task_timeout_budget_source": (
                        "prestep_subphase:tolbert_query" if timed_out else ""
                    ),
                    "observation_current_task_timeout_budget_seconds": 3.0 if timed_out else 0.0,
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
    assert payload["summary"]["generated_candidate_runs"] == 2
    assert payload["summary"]["timed_out_runs"] == 1
    assert payload["summary"]["deduped_runs"] == 1
    assert payload["summary"]["timeout_budget_sources"] == {"prestep_subphase:tolbert_query": 1}
    assert len(payload["frontier_candidates"]) == 2
    assert payload["frontier_candidates"][0]["selected_variant_id"] == "regression_guard"
    assert payload["frontier_candidates"][0]["duplicate_count"] == 1
    assert sorted(payload["frontier_candidates"][0]["duplicate_scope_ids"]) == ["scope_b"]
    assert payload["frontier_candidates"][1]["scope_id"] == "scope_timeout"
    assert payload["frontier_candidates"][1]["observation_timed_out"] is True


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

    class FakeConfig:
        improvement_reports_dir = reports_dir
        improvement_cycles_path = improvement_root / "cycles.jsonl"
        candidate_artifacts_root = candidates_root

        def ensure_directories(self):
            self.improvement_reports_dir.mkdir(parents=True, exist_ok=True)

        def uses_sqlite_storage(self):
            return False

    monkeypatch.setattr(module, "KernelConfig", lambda: FakeConfig())
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


def test_report_frontier_promotion_plan_emits_ranked_compare_and_finalize_commands(tmp_path, monkeypatch):
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
    assert payload["promotion_candidates"][0]["scope_id"] == "scope_fast"
    assert payload["promotion_candidates"][0]["promotion_base_score"] > 0.0
    assert payload["promotion_candidates"][0]["promotion_history_penalty"] == 0.0
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
    assert payload["results"][0]["finalize_state"] == ""


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
            "report_path": str(report_path),
            "entries": [
                {
                    "selected_subsystem": "transition_model",
                    "scope_id": "scope_a",
                    "candidate_artifact_path": "candidates/a.json",
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
    assert payload["summary"]["entry_count"] == 1
    assert payload["entries"][0]["selected_subsystem"] == "transition_model"


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
    assert status_payload["machine_state"]["meta_policy"]["exists"] is True
    assert "claim_ledger" in status_payload["machine_state"]
    assert "lane_allocator" in status_payload["machine_state"]
    assert "rollback_plan" in status_payload["machine_state"]
    history_entries = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(history_entries) == 1
    assert any("run_frontier_promotion_pass.py" in " ".join(command) for command in commands)


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


def test_migrate_tolbert_shared_store_script_prunes_unreferenced_legacy_candidates(tmp_path, monkeypatch):
    module = _load_script_module("migrate_tolbert_shared_store.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "tolbert_model_artifact.json",
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
    assert curriculum_payload["control_schema"] == "curriculum_behavior_controls_v2"
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
