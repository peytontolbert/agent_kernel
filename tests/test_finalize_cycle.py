from pathlib import Path
import importlib.util
import json
from io import StringIO
import pytest
from subprocess import CompletedProcess
import sys

from agent_kernel import cycle_runner
from agent_kernel.config import KernelConfig
from agent_kernel.improvement import (
    ImprovementCycleRecord,
    ImprovementExperiment,
    ImprovementPlanner,
    ImprovementSearchBudget,
    ImprovementVariant,
)
from evals.metrics import EvalMetrics


def _load_finalize_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "finalize_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("finalize_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_improvement_cycle_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_candidate_payload(cycles_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generate_record = next(record for record in records if record["state"] == "generate")
    candidate_path = Path(str(generate_record.get("candidate_artifact_path") or generate_record["artifact_path"]))
    return json.loads(candidate_path.read_text(encoding="utf-8")), records


def test_autonomous_runtime_eval_flags_preserve_explicit_false_over_existing_artifacts(tmp_path):
    config = KernelConfig(
        skills_path=tmp_path / "skills" / "skills.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
        verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
        trajectories_root=tmp_path / "trajectories",
    )
    config.ensure_directories()
    config.skills_path.write_text("{}", encoding="utf-8")
    config.operator_classes_path.write_text("{}", encoding="utf-8")
    config.tool_candidates_path.write_text("{}", encoding="utf-8")
    config.benchmark_candidates_path.write_text("{}", encoding="utf-8")
    config.verifier_contracts_path.write_text("{}", encoding="utf-8")

    flags = cycle_runner.autonomous_runtime_eval_flags(
        config,
        {
            "include_skill_memory": False,
            "include_skill_transfer": False,
            "include_operator_memory": False,
            "include_tool_memory": False,
            "include_verifier_memory": False,
            "include_benchmark_candidates": False,
            "include_verifier_candidates": False,
        },
    )

    assert flags["include_skill_memory"] is False
    assert flags["include_skill_transfer"] is False
    assert flags["include_operator_memory"] is False
    assert flags["include_tool_memory"] is False
    assert flags["include_verifier_memory"] is False
    assert flags["include_benchmark_candidates"] is False
    assert flags["include_verifier_candidates"] is False


def test_autonomous_runtime_eval_flags_auto_enable_when_flag_is_omitted(tmp_path):
    config = KernelConfig(
        skills_path=tmp_path / "skills" / "skills.json",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
    )
    config.ensure_directories()
    config.skills_path.write_text("{}", encoding="utf-8")
    config.benchmark_candidates_path.write_text("{}", encoding="utf-8")

    flags = cycle_runner.autonomous_runtime_eval_flags(config, {})

    assert flags["include_skill_memory"] is True
    assert flags["include_skill_transfer"] is True
    assert flags["include_benchmark_candidates"] is True


def test_finalize_cycle_flag_mapping_for_skills():
    module = _load_finalize_module()

    baseline, candidate = module._baseline_candidate_flags("skills")

    assert baseline["include_skill_memory"] is False
    assert candidate["include_skill_memory"] is True
    assert candidate["include_episode_memory"] is True


def test_finalize_cycle_flag_mapping_for_curriculum():
    module = _load_finalize_module()

    baseline, candidate = module._baseline_candidate_flags("curriculum")

    assert baseline["include_generated"] is True
    assert candidate["include_generated"] is True
    assert candidate["include_failure_generated"] is True


def test_finalize_cycle_flag_mapping_for_benchmark():
    module = _load_finalize_module()

    baseline, candidate = module._baseline_candidate_flags("benchmark")

    assert baseline["include_benchmark_candidates"] is False
    assert candidate["include_benchmark_candidates"] is True


def test_finalize_cycle_flag_mapping_for_operators():
    module = _load_finalize_module()

    baseline, candidate = module._baseline_candidate_flags("operators")

    assert baseline["include_skill_transfer"] is True
    assert candidate["include_operator_memory"] is True


def test_autonomous_phase_gate_report_rejects_retrieval_without_retrieval_influence():
    module = _load_finalize_module()

    report = module.cycle_runner.autonomous_phase_gate_report(
        subsystem="retrieval",
        baseline_metrics=EvalMetrics(total=10, passed=8, trusted_retrieval_steps=2, retrieval_influenced_steps=1, retrieval_selected_steps=1),
        candidate_metrics=EvalMetrics(total=10, passed=9, trusted_retrieval_steps=4, retrieval_influenced_steps=0, retrieval_selected_steps=1),
        candidate_flags={"include_generated": True, "include_failure_generated": True},
        gate={"require_failure_recovery_non_regression": True},
    )

    assert report["passed"] is False
    assert any("retrieval influence" in failure for failure in report["failures"])


def test_autonomous_phase_gate_report_allows_tolbert_primary_signal_without_retrieval_influence():
    module = _load_finalize_module()

    report = module.cycle_runner.autonomous_phase_gate_report(
        subsystem="tolbert_model",
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            trusted_retrieval_steps=0,
            retrieval_influenced_steps=0,
            retrieval_selected_steps=0,
            tolbert_primary_episodes=0,
        ),
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            trusted_retrieval_steps=1,
            retrieval_influenced_steps=0,
            retrieval_selected_steps=1,
            tolbert_primary_episodes=1,
        ),
        candidate_flags={"include_generated": True, "include_failure_generated": True},
        gate={"require_failure_recovery_non_regression": True},
    )

    assert report["passed"] is True
    assert report["failures"] == []


def test_autonomous_phase_gate_report_rejects_missing_generated_lane_output():
    module = _load_finalize_module()

    report = module.cycle_runner.autonomous_phase_gate_report(
        subsystem="policy",
        baseline_metrics=EvalMetrics(total=10, passed=8),
        candidate_metrics=EvalMetrics(total=10, passed=9, generated_total=0, generated_by_kind={}),
        candidate_flags={"include_generated": True, "include_failure_generated": True},
        gate={"require_failure_recovery_non_regression": True},
    )

    assert report["passed"] is False
    assert "generated-task lane produced no tasks during autonomous evaluation" in report["failures"]
    assert "failure-recovery lane produced no generated tasks during autonomous evaluation" in report["failures"]


def test_confirmation_confidence_report_rejects_high_uncertainty():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(total=10, passed=8, average_steps=1.0),
            EvalMetrics(total=10, passed=8, average_steps=1.0),
        ],
        [
            EvalMetrics(total=10, passed=10, average_steps=1.0),
            EvalMetrics(total=10, passed=6, average_steps=1.0),
        ],
        gate={
            "confirmation_confidence_z": 1.0,
            "max_confirmation_pass_rate_delta_stderr": 0.05,
            "max_confirmation_pass_rate_spread": 0.2,
            "min_confirmation_pass_rate_delta_conservative_lower_bound": -0.01,
        },
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "max_confirmation_pass_rate_delta_stderr": 0.05,
            "max_confirmation_pass_rate_spread": 0.2,
            "min_confirmation_pass_rate_delta_conservative_lower_bound": -0.01,
        },
    )

    assert report["candidate_pass_rate_spread"] == 0.4
    assert any("uncertainty" in failure for failure in failures)
    assert "pass_rate_delta_conservative_lower_bound" in report
    assert any("conservative pass-rate bound" in failure for failure in failures)


def test_confirmation_confidence_report_rejects_family_conservative_regression():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(
                total=20,
                passed=16,
                average_steps=1.0,
                total_by_benchmark_family={"archive": 10, "api": 10},
                passed_by_benchmark_family={"archive": 9, "api": 7},
            ),
            EvalMetrics(
                total=20,
                passed=16,
                average_steps=1.0,
                total_by_benchmark_family={"archive": 10, "api": 10},
                passed_by_benchmark_family={"archive": 9, "api": 7},
            ),
        ],
        [
            EvalMetrics(
                total=20,
                passed=15,
                average_steps=1.0,
                total_by_benchmark_family={"archive": 10, "api": 10},
                passed_by_benchmark_family={"archive": 6, "api": 9},
            ),
            EvalMetrics(
                total=20,
                passed=15,
                average_steps=1.0,
                total_by_benchmark_family={"archive": 10, "api": 10},
                passed_by_benchmark_family={"archive": 6, "api": 9},
            ),
        ],
        gate={"confirmation_confidence_z": 1.0},
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "max_confirmation_regressed_family_conservative_count": 0,
            "min_confirmation_worst_family_conservative_lower_bound": -0.01,
        },
    )

    assert report["regressed_family_conservative_count"] >= 1
    assert report["worst_family_conservative_lower_bound"] < 0.0
    assert any("family conservative regression count" in failure for failure in failures)


def test_confirmation_confidence_report_rejects_paired_task_regressions():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(
                total=2,
                passed=2,
                average_steps=1.0,
                task_outcomes={
                    "task_a": {"success": True, "steps": 1, "benchmark_family": "archive"},
                    "task_b": {"success": True, "steps": 1, "benchmark_family": "api"},
                },
            ),
            EvalMetrics(
                total=2,
                passed=2,
                average_steps=1.0,
                task_outcomes={
                    "task_a": {"success": True, "steps": 1, "benchmark_family": "archive"},
                    "task_b": {"success": True, "steps": 1, "benchmark_family": "api"},
                },
            ),
        ],
        [
            EvalMetrics(
                total=2,
                passed=1,
                average_steps=1.5,
                task_outcomes={
                    "task_a": {"success": False, "steps": 2, "benchmark_family": "archive"},
                    "task_b": {"success": True, "steps": 1, "benchmark_family": "api"},
                },
            ),
            EvalMetrics(
                total=2,
                passed=1,
                average_steps=1.5,
                task_outcomes={
                    "task_a": {"success": False, "steps": 2, "benchmark_family": "archive"},
                    "task_b": {"success": True, "steps": 1, "benchmark_family": "api"},
                },
            ),
        ],
        gate={"confirmation_confidence_z": 1.0},
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "min_confirmation_paired_task_non_regression_rate_lower_bound": 0.75,
            "max_confirmation_regressed_task_count": 0,
        },
    )

    assert report["paired_task_count"] == 2
    assert report["regressed_task_count"] == 1
    assert "task_a" in report["paired_task_traces"]
    assert any("paired task non-regression bound" in failure for failure in failures)
    assert any("regressed-task count" in failure for failure in failures)


def test_confirmation_confidence_report_rejects_paired_trace_regressions():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=1.0,
                task_outcomes={
                    "task_trace": {
                        "success": True,
                        "steps": 1,
                        "benchmark_family": "archive",
                        "termination_reason": "success",
                        "unsafe_ambiguous": False,
                        "hidden_side_effect_risk": False,
                        "first_step_verified": True,
                        "no_state_progress_steps": 0,
                        "state_regression_steps": 0,
                        "total_state_regression_count": 0,
                        "present_forbidden_artifact_count": 0,
                        "changed_preserved_artifact_count": 0,
                        "missing_expected_artifact_count": 0,
                        "completion_ratio": 1.0,
                        "failure_signals": [],
                    }
                },
            ),
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=1.0,
                task_outcomes={
                    "task_trace": {
                        "success": True,
                        "steps": 1,
                        "benchmark_family": "archive",
                        "termination_reason": "success",
                        "unsafe_ambiguous": False,
                        "hidden_side_effect_risk": False,
                        "first_step_verified": True,
                        "no_state_progress_steps": 0,
                        "state_regression_steps": 0,
                        "total_state_regression_count": 0,
                        "present_forbidden_artifact_count": 0,
                        "changed_preserved_artifact_count": 0,
                        "missing_expected_artifact_count": 0,
                        "completion_ratio": 1.0,
                        "failure_signals": [],
                    }
                },
            ),
        ],
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=2.0,
                task_outcomes={
                    "task_trace": {
                        "success": True,
                        "steps": 2,
                        "benchmark_family": "archive",
                        "termination_reason": "max_steps_reached",
                        "unsafe_ambiguous": False,
                        "hidden_side_effect_risk": True,
                        "first_step_verified": False,
                        "no_state_progress_steps": 1,
                        "state_regression_steps": 1,
                        "total_state_regression_count": 2,
                        "present_forbidden_artifact_count": 1,
                        "changed_preserved_artifact_count": 1,
                        "missing_expected_artifact_count": 1,
                        "completion_ratio": 0.5,
                        "failure_signals": ["no_state_progress", "state_regression"],
                    }
                },
            ),
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=2.0,
                task_outcomes={
                    "task_trace": {
                        "success": True,
                        "steps": 2,
                        "benchmark_family": "archive",
                        "termination_reason": "max_steps_reached",
                        "unsafe_ambiguous": False,
                        "hidden_side_effect_risk": True,
                        "first_step_verified": False,
                        "no_state_progress_steps": 1,
                        "state_regression_steps": 1,
                        "total_state_regression_count": 2,
                        "present_forbidden_artifact_count": 1,
                        "changed_preserved_artifact_count": 1,
                        "missing_expected_artifact_count": 1,
                        "completion_ratio": 0.5,
                        "failure_signals": ["no_state_progress", "state_regression"],
                    }
                },
            ),
        ],
        gate={"confirmation_confidence_z": 1.0},
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "min_confirmation_paired_trace_non_regression_rate_lower_bound": 0.75,
            "max_confirmation_regressed_trace_task_count": 0,
        },
    )

    assert report["paired_trace_task_count"] == 1
    assert report["regressed_trace_task_count"] == 1
    assert report["paired_trace_summaries"]["task_trace"]["trace_score_delta"] > 0.0
    assert any("paired trace non-regression bound" in failure for failure in failures)
    assert any("regressed-trace task count" in failure for failure in failures)


def test_confirmation_confidence_report_rejects_paired_trajectory_regressions():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=2.0,
                task_trajectories={
                    "task_traj": {
                        "task_id": "task_traj",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "mkdir -p out",
                                "command": "mkdir -p out",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 2,
                                "action": "code_execute",
                                "content": "printf ok > out/result.txt",
                                "command": "printf ok > out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                        ],
                    }
                },
            ),
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=2.0,
                task_trajectories={
                    "task_traj": {
                        "task_id": "task_traj",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "mkdir -p out",
                                "command": "mkdir -p out",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 2,
                                "action": "code_execute",
                                "content": "printf ok > out/result.txt",
                                "command": "printf ok > out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                        ],
                    }
                },
            ),
        ],
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=3.0,
                task_trajectories={
                    "task_traj": {
                        "task_id": "task_traj",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "mkdir -p out",
                                "command": "mkdir -p out",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 2,
                                "action": "code_execute",
                                "content": "rm -f out/result.txt",
                                "command": "rm -f out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": False,
                                "failure_signals": ["state_regression"],
                                "state_regression_count": 1,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 3,
                                "action": "code_execute",
                                "content": "printf ok > out/result.txt",
                                "command": "printf ok > out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                        ],
                    }
                },
            ),
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=3.0,
                task_trajectories={
                    "task_traj": {
                        "task_id": "task_traj",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "mkdir -p out",
                                "command": "mkdir -p out",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 2,
                                "action": "code_execute",
                                "content": "rm -f out/result.txt",
                                "command": "rm -f out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": False,
                                "failure_signals": ["state_regression"],
                                "state_regression_count": 1,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                            {
                                "index": 3,
                                "action": "code_execute",
                                "content": "printf ok > out/result.txt",
                                "command": "printf ok > out/result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            },
                        ],
                    }
                },
            ),
        ],
        gate={"confirmation_confidence_z": 1.0},
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "min_confirmation_paired_trajectory_non_regression_rate_lower_bound": 0.75,
            "max_confirmation_regressed_trajectory_task_count": 0,
        },
    )

    assert report["paired_trajectory_task_count"] == 1
    assert report["regressed_trajectory_task_count"] == 1
    assert report["paired_trajectory_summaries"]["task_traj"]["trajectory_score_delta"] > 0.0
    assert any("paired trajectory non-regression bound" in failure for failure in failures)
    assert any("regressed-trajectory task count" in failure for failure in failures)


def test_confirmation_confidence_report_rejects_weak_paired_trajectory_significance():
    module = _load_finalize_module()

    report = module.cycle_runner.confirmation_confidence_report(
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=1.0,
                task_trajectories={
                    "task_sig": {
                        "task_id": "task_sig",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "printf ok > result.txt",
                                "command": "printf ok > result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            }
                        ],
                    }
                },
            )
        ],
        [
            EvalMetrics(
                total=1,
                passed=1,
                average_steps=1.0,
                task_trajectories={
                    "task_sig": {
                        "task_id": "task_sig",
                        "benchmark_family": "archive",
                        "success": True,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "index": 1,
                                "action": "code_execute",
                                "content": "printf ok > result.txt",
                                "command": "printf ok > result.txt",
                                "exit_code": 0,
                                "timed_out": False,
                                "verification_passed": True,
                                "failure_signals": [],
                                "state_regression_count": 0,
                                "decision_source": "llm",
                                "tolbert_route_mode": "",
                                "retrieval_influenced": False,
                                "trust_retrieval": False,
                            }
                        ],
                    }
                },
            )
        ],
        gate={"confirmation_confidence_z": 1.0},
    )

    failures = module.cycle_runner.confirmation_confidence_failures(
        report,
        gate={
            "confirmation_confidence_z": 1.0,
            "min_confirmation_paired_trajectory_pair_count": 2,
            "max_confirmation_paired_trajectory_non_regression_p_value": 0.25,
        },
    )

    assert report["paired_trajectory_pair_count"] == 1
    assert report["paired_trajectory_non_regression_significance_p_value"] == 0.5
    assert any("paired trajectory evidence remained too small" in failure for failure in failures)
    assert any("paired trajectory significance remained too weak" in failure for failure in failures)


def test_finalize_cycle_emits_progress_heartbeat(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    active_artifact_path = config.retrieval_proposals_path
    active_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    active_artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_artifact_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "reject",
            "reason": "candidate regressed",
            "gate": {"required_confirmation_runs": 1},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": True},
            "candidate_flags": {"include_generated": True, "include_failure_generated": True},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "autonomous_phase_gate_report",
        lambda **kwargs: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test",
        artifact_path=artifact_path,
        progress=progress_messages.append,
    )

    assert state == "reject"
    assert reason == "candidate regressed"
    assert progress_messages[0] == "finalize phase=preview subsystem=retrieval"
    assert any("phase=preview_complete" in message for message in progress_messages)
    assert any("phase=apply_decision" in message for message in progress_messages)
    assert any(
        "phase=decision_reject_reason subsystem=retrieval reason_code=retention_reject_unknown" in message
        for message in progress_messages
    )
    assert progress_messages[-1] == "finalize phase=done subsystem=retrieval state=reject"


def test_finalize_cycle_records_machine_readable_reject_reason_codes(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.ensure_directories()
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "reject",
            "reason": "retrieval candidate did not satisfy the retained retrieval gate",
            "reason_code": "retrieval_retained_gate_failed",
            "gate": {"required_confirmation_runs": 1},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": True},
            "candidate_flags": {"include_generated": True, "include_failure_generated": True},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:reason-codes",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        progress=progress_messages.append,
    )

    report_path = config.improvement_reports_dir / "cycle_report_cycle_test_reason-codes.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert state == "reject"
    assert reason == "retrieval candidate did not satisfy the retained retrieval gate"
    assert payload["preview_reason_code"] == "retrieval_retained_gate_failed"
    assert payload["decision_reason_code"] == "retrieval_retained_gate_failed"
    assert any(
        "phase=preview_reject_reason subsystem=retrieval reason_code=retrieval_retained_gate_failed"
        in message
        for message in progress_messages
    )
    assert any(
        "phase=decision_reject_reason subsystem=retrieval reason_code=retrieval_retained_gate_failed"
        in message
        for message in progress_messages
    )


def test_finalize_cycle_promotes_strategy_lineage_to_closeout_records(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
    )
    config.ensure_directories()
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "reject",
            "reason": "candidate did not improve",
            "reason_code": "no_verified_gain",
            "gate": {"required_confirmation_runs": 1},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": True},
            "candidate_flags": {"include_generated": True},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)

    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:strategy-lineage",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        strategy_candidate={
            "strategy_candidate_id": "strategy:broad_observe_diversification",
            "strategy_candidate_kind": "broad_observe_diversification",
            "origin": "discovered_strategy",
        },
    )

    records = [
        json.loads(line)
        for line in config.improvement_cycles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    closeout_records = [
        record
        for record in records
        if record["state"] == state or record.get("action") == "persist_retention_outcome"
    ]

    assert state == "reject"
    assert reason == "candidate did not improve"
    assert {record["action"] for record in closeout_records} == {
        "finalize_cycle",
        "persist_retention_outcome",
    }
    for record in closeout_records:
        metrics = record["metrics_summary"]
        assert record["strategy_candidate_id"] == "strategy:broad_observe_diversification"
        assert record["strategy_candidate_kind"] == "broad_observe_diversification"
        assert record["strategy_origin"] == "discovered_strategy"
        assert metrics["strategy_candidate_id"] == "strategy:broad_observe_diversification"
        assert metrics["strategy_candidate_kind"] == "broad_observe_diversification"
        assert metrics["strategy_origin"] == "discovered_strategy"


def test_finalize_cycle_reuses_preview_result_without_rerunning_preview(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.ensure_directories()
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    def fail_preview(**kwargs):
        raise AssertionError("preview_candidate_retention should not run when preview is provided")

    monkeypatch.setattr(module.cycle_runner, "preview_candidate_retention", fail_preview)
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:preview-reuse",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        preview={
            "active_artifact_path": str(active_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "reject",
            "reason": "retrieval candidate did not satisfy the retained retrieval gate",
            "reason_code": "retrieval_retained_gate_failed",
            "gate": {"required_confirmation_runs": 1},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "prior_retained_guard_reason": "",
            "prior_retained_guard_reason_code": "",
            "baseline_flags": {"include_generated": True},
            "candidate_flags": {"include_generated": True, "include_failure_generated": True},
        },
        progress=progress_messages.append,
    )

    assert state == "reject"
    assert reason == "retrieval candidate did not satisfy the retained retrieval gate"
    assert "finalize phase=preview_reused subsystem=retrieval" in progress_messages
    assert not any("phase=preview_baseline_eval" in message for message in progress_messages)


def test_finalize_cycle_records_machine_readable_unchanged_candidate_reason_code(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.ensure_directories()
    active_path = config.retrieval_proposals_path
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "retained",
        "retention_gate": {"min_pass_rate_delta_abs": 0.02, "required_confirmation_runs": 1},
    }
    active_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(artifact_payload), encoding="utf-8")

    def fake_run_eval(*, config, progress_label=None, **kwargs):
        del config, progress_label, kwargs
        return EvalMetrics(
            total=24,
            passed=23,
            average_steps=1.8333333333333333,
            generated_total=4,
            generated_passed=4,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        )

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "sha",
            "previous_artifact_sha256": "sha",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:unchanged-reason-codes",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        progress=progress_messages.append,
    )

    report_path = config.improvement_reports_dir / "cycle_report_cycle_test_unchanged-reason-codes.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert state == "reject"
    assert reason == "candidate artifact is identical to the active retained artifact"
    assert payload["preview_reason_code"] == "candidate_artifact_unchanged"
    assert payload["decision_reason_code"] == "candidate_artifact_unchanged"
    assert any(
        "phase=preview_reject_reason subsystem=retrieval reason_code=candidate_artifact_unchanged"
        in message
        for message in progress_messages
    )
    assert any(
        "phase=decision_reject_reason subsystem=retrieval reason_code=candidate_artifact_unchanged"
        in message
        for message in progress_messages
    )


def test_finalize_cycle_preserves_nested_confirmation_reject_reason_code(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.ensure_directories()
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "retain",
            "reason": "candidate improved",
            "reason_code": "",
            "gate": {"required_confirmation_runs": 2},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": True},
            "candidate_flags": {"include_generated": True, "include_failure_generated": True},
        },
    )

    confirmation_calls = {"count": 0}

    def fake_evaluate_artifact_retention(subsystem, baseline, candidate, **kwargs):
        del subsystem, baseline, candidate, kwargs
        confirmation_calls["count"] += 1
        if confirmation_calls["count"] == 1:
            return "reject", "retrieval candidate did not satisfy the retained retrieval gate"
        return "retain", "candidate improved"

    monkeypatch.setattr(module.cycle_runner, "evaluate_artifact_retention", fake_evaluate_artifact_retention)
    monkeypatch.setattr(
        module.cycle_runner,
        "evaluate_subsystem_metrics",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "rejected",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:confirmation-reason-codes",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        progress=progress_messages.append,
    )

    report_path = config.improvement_reports_dir / "cycle_report_cycle_test_confirmation-reason-codes.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert state == "reject"
    assert reason == (
        "candidate failed confirmation run 2 of 2: "
        "retrieval candidate did not satisfy the retained retrieval gate"
    )
    assert payload["preview_reason_code"] == ""
    assert payload["decision_reason_code"] == "retrieval_retained_gate_failed"
    assert any(
        "phase=decision_reject_reason subsystem=retrieval reason_code=retrieval_retained_gate_failed"
        in message
        for message in progress_messages
    )


def test_finalize_cycle_retries_confirmation_eval_without_tolbert_context_after_startup_failure(
    tmp_path, monkeypatch
):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    artifact_path = tmp_path / "policy" / "candidate.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "prompt_proposal"}), encoding="utf-8")
    active_artifact_path = tmp_path / "policy" / "active.json"
    active_artifact_path.write_text(json.dumps({"artifact_kind": "prompt_proposal"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_artifact_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "prompt_proposal"},
            "state": "retain",
            "reason": "candidate improved",
            "gate": {"required_confirmation_runs": 2},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": False},
            "candidate_flags": {"include_generated": False, "include_failure_generated": False},
        },
    )

    calls: list[dict[str, object]] = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del flags
        calls.append(
            {
                "use_tolbert_context": bool(config.use_tolbert_context),
                "subsystem": subsystem,
                "progress_label": progress_label,
            }
        )
        if len(calls) == 1:
            raise RuntimeError("TOLBERT service failed to become ready after 15.000 seconds.")
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "evaluate_artifact_retention", lambda *args, **kwargs: ("retain", ""))
    monkeypatch.setattr(module.cycle_runner, "confirmation_confidence_failures", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "prompt_proposal",
            "artifact_lifecycle_state": "retained",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="policy",
        cycle_id="cycle:test",
        artifact_path=artifact_path,
        progress=progress_messages.append,
    )

    assert state == "retain"
    assert calls == [
        {
            "use_tolbert_context": True,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_confirmation_baseline",
        },
        {
            "use_tolbert_context": False,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_confirmation_baseline",
        },
        {
            "use_tolbert_context": True,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_confirmation_candidate",
        },
    ]
    assert (
        "finalize phase=confirmation_baseline_retry subsystem=policy "
        "reason=tolbert_startup_failure use_tolbert_context=0"
    ) in progress_messages
    assert reason == "candidate improved"


def test_finalize_cycle_retries_holdout_eval_without_tolbert_context_after_startup_failure(
    tmp_path, monkeypatch
):
    module = _load_finalize_module()
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    artifact_path = tmp_path / "policy" / "candidate.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "prompt_proposal"}), encoding="utf-8")
    active_artifact_path = tmp_path / "policy" / "active.json"
    active_artifact_path.write_text(json.dumps({"artifact_kind": "prompt_proposal"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_artifact_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {},
            "compatibility": {},
            "payload": {"artifact_kind": "prompt_proposal"},
            "state": "retain",
            "reason": "candidate improved",
            "gate": {"required_confirmation_runs": 1},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {"include_generated": False},
            "candidate_flags": {"include_generated": False, "include_failure_generated": False},
        },
    )

    calls: list[dict[str, object]] = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del flags
        calls.append(
            {
                "use_tolbert_context": bool(config.use_tolbert_context),
                "subsystem": subsystem,
                "progress_label": progress_label,
            }
        )
        if len(calls) == 1:
            raise RuntimeError("TOLBERT service failed to become ready after 15.000 seconds.")
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "evaluate_artifact_retention", lambda *args, **kwargs: ("retain", ""))
    monkeypatch.setattr(
        module.cycle_runner,
        "autonomous_phase_gate_report",
        lambda **kwargs: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(module.cycle_runner, "confirmation_confidence_failures", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "prompt_proposal",
            "artifact_lifecycle_state": "retained",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
        },
    )
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)

    progress_messages: list[str] = []
    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="policy",
        cycle_id="cycle:test",
        artifact_path=artifact_path,
        comparison_task_limit=4,
        progress=progress_messages.append,
    )

    assert state == "retain"
    assert calls == [
        {
            "use_tolbert_context": True,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_holdout_baseline",
        },
        {
            "use_tolbert_context": False,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_holdout_baseline",
        },
        {
            "use_tolbert_context": True,
            "subsystem": "policy",
            "progress_label": "cycle:test_policy_holdout_candidate",
        },
    ]
    assert (
        "finalize phase=holdout_baseline_retry subsystem=policy "
        "reason=tolbert_startup_failure use_tolbert_context=0"
    ) in progress_messages
    assert reason == ""


def test_preview_candidate_retention_passes_progress_labels_to_eval(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    seen_labels: list[str | None] = []

    def fake_run_eval(*, config, progress_label=None, **kwargs):
        del config, kwargs
        seen_labels.append(progress_label)
        return EvalMetrics(total=10, passed=8, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)

    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycle_id="cycle:test",
        progress_label_prefix="cycle_test_retrieval_preview",
    )

    assert seen_labels == [
        "cycle_test_retrieval_preview_baseline",
        "cycle_test_retrieval_preview_candidate",
    ]


def test_preview_candidate_retention_emits_preview_subphases(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    def fake_run_eval(*, config, progress_label=None, **kwargs):
        del config, progress_label, kwargs
        return EvalMetrics(total=10, passed=8, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)

    progress_messages: list[str] = []
    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycle_id="cycle:test",
        progress_label_prefix="cycle_test_retrieval_preview",
        progress=progress_messages.append,
    )

    assert progress_messages[:4] == [
        "finalize phase=preview_baseline_eval subsystem=retrieval",
        "finalize phase=preview_baseline_complete subsystem=retrieval baseline_pass_rate=0.8000",
        "finalize phase=preview_candidate_eval subsystem=retrieval",
        "finalize phase=preview_candidate_complete subsystem=retrieval candidate_pass_rate=0.8000",
    ]


def test_preview_candidate_retention_retries_without_tolbert_context_after_startup_failure(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del flags
        calls.append(
            {
                "use_tolbert_context": bool(config.use_tolbert_context),
                "subsystem": subsystem,
                "progress_label": progress_label,
            }
        )
        if len(calls) == 1:
            raise RuntimeError("TOLBERT service failed to become ready after 15.000 seconds.")
        return EvalMetrics(total=10, passed=8, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    progress_messages: list[str] = []
    preview = module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycle_id="cycle:test",
        progress_label_prefix="cycle_test_retrieval_preview",
        progress=progress_messages.append,
    )

    assert preview["state"] in {"retain", "reject"}
    assert calls == [
        {
            "use_tolbert_context": True,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_preview_baseline",
        },
        {
            "use_tolbert_context": False,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_preview_baseline",
        },
        {
            "use_tolbert_context": True,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_preview_candidate",
        },
    ]
    assert "finalize phase=preview_baseline_retry subsystem=retrieval reason=tolbert_startup_failure use_tolbert_context=0" in progress_messages
    assert preview["tolbert_runtime_summary"]["outcome"] == "failed_recovered"
    assert preview["tolbert_runtime_summary"]["startup_failure_stages"] == ["preview_baseline"]
    assert preview["tolbert_runtime_summary"]["recovered_without_tolbert_stages"] == ["preview_baseline"]


def test_preview_candidate_retention_exposes_machine_readable_reject_reason_code(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    def fake_run_eval(*, config, progress_label=None, **kwargs):
        del config, progress_label, kwargs
        return EvalMetrics(total=10, passed=8, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.cycle_runner,
        "autonomous_phase_gate_report",
        lambda **kwargs: {
            "passed": True,
            "failures": [],
            "generated_lane_included": True,
            "failure_recovery_lane_included": True,
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "evaluate_artifact_retention",
        lambda *args, **kwargs: ("reject", "retrieval candidate did not satisfy the retained retrieval gate"),
    )

    preview = module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycle_id="cycle:test",
        progress_label_prefix="cycle_test_retrieval_preview",
    )

    assert preview["state"] == "reject"
    assert preview["reason"] == "retrieval candidate did not satisfy the retained retrieval gate"
    assert preview["reason_code"] == "retrieval_retained_gate_failed"


def test_preview_candidate_retention_does_not_retry_non_tolbert_failures(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    calls: list[bool] = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del subsystem, flags, progress_label
        calls.append(bool(config.use_tolbert_context))
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    with pytest.raises(RuntimeError, match="database unavailable"):
        module.cycle_runner.preview_candidate_retention(
            config=config,
            subsystem="retrieval",
            artifact_path=artifact_path,
            cycle_id="cycle:test",
            progress_label_prefix="cycle_test_retrieval_preview",
        )

    assert calls == [True]


def test_preview_candidate_retention_reclassifies_unchanged_retrieval_candidate(tmp_path, monkeypatch):
    module = _load_finalize_module()
    config = KernelConfig(
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    active_path = config.retrieval_proposals_path
    artifact_path = tmp_path / "retrieval" / "candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "retained",
        "retention_gate": {"min_pass_rate_delta_abs": 0.02},
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    def fake_run_eval(*, config, progress_label=None, **kwargs):
        del config, progress_label, kwargs
        return EvalMetrics(
            total=24,
            passed=23,
            average_steps=1.8333333333333333,
            generated_total=4,
            generated_passed=4,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        )

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)

    preview = module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycle_id="cycle:test:unchanged",
        progress_label_prefix="cycle_test_retrieval_preview",
    )

    assert preview["state"] == "reject"
    assert preview["reason"] == "candidate artifact is identical to the active retained artifact"
    assert preview["reason_code"] == "candidate_artifact_unchanged"
    assert preview["evidence"]["artifact_content_unchanged"] is True


def test_finalize_cycle_propagates_protocol_match_id_to_finalize_records(tmp_path, monkeypatch):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    config = KernelConfig(
        improvement_cycles_path=cycles_path,
        retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    active_path = config.retrieval_proposals_path
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    candidate_path = tmp_path / "retrieval" / "candidate.json"
    candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "active_artifact_path": str(active_path),
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "evidence": {"pass_rate_delta": 0.1},
            "compatibility": {},
            "payload": {"artifact_kind": "retrieval_policy_set"},
            "state": "retain",
            "reason": "candidate improved",
            "gate": {"required_confirmation_runs": 1},
            "phase_gate_report": {"passed": True, "failures": []},
            "prior_retained_comparison": None,
            "baseline_flags": {},
            "candidate_flags": {},
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "retrieval_policy_set",
            "artifact_lifecycle_state": "retained",
            "artifact_sha256": "sha",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
            "candidate_artifact_path": str(kwargs["artifact_path"]),
            "active_artifact_path": str(kwargs["active_artifact_path"]),
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "retrieval" / "bundle_manifest.json",
    )
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)

    state, reason = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:test:protocol",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        protocol_match_id="campaign-protocol-test",
    )

    records = [
        json.loads(line)
        for line in cycles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relevant = [record for record in records if record["cycle_id"] == "cycle:test:protocol"]

    assert state == "retain"
    assert reason == "candidate improved"
    assert relevant
    assert all(
        record.get("metrics_summary", {}).get("protocol_match_id") == "campaign-protocol-test"
        for record in relevant
    )


def test_finalize_cycle_main_persists_evaluate_retain_and_record(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "tools" / "tool_candidates.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "tool_id": "tool:hello_task:primary",
                        "promotion_stage": "candidate_procedure",
                        "source_task_id": "hello_task",
                        "procedure": {
                            "commands": [
                                "printf 'hello agent kernel\\n' > hello.txt",
                            ]
                        },
                        "task_contract": {
                            "prompt": "Create hello.txt containing the string hello agent kernel.",
                            "workspace_subdir": "hello_task",
                            "setup_commands": [],
                            "success_command": "test -f hello.txt && grep -q '^hello agent kernel$' hello.txt",
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
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    run_results = [
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.2,
            generated_total=4,
            generated_passed=4,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return run_results.pop(0)

    def fast_atomic_write_json(path, payload, *, config=None):
        del config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(module.cycle_runner, "atomic_write_json", fast_atomic_write_json)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            tool_candidates_path=artifact_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--cycle-id",
            "cycle:tooling:test",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [record["state"] for record in records]
    assert states == ["evaluate", "evaluate", "retain", "record", "record"]
    assert records[0]["artifact_lifecycle_state"] == "replay_verified"
    assert records[2]["artifact_lifecycle_state"] == "retained"
    assert records[3]["artifact_lifecycle_state"] == "retained"
    assert records[4]["artifact_kind"] == "improvement_cycle_report"

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "retained"
    assert payload["candidates"][0]["promotion_stage"] == "promoted_tool"
    assert payload["retention_decision"]["state"] == "retain"


def test_finalize_cycle_candidate_eval_preserves_unrelated_retained_proposal_toggles(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "prompts" / "prompt_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "proposed",
                "proposals": [
                    {"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    seen_configs = []

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        seen_configs.append(
            {
                "use_prompt_proposals": config.use_prompt_proposals,
                "use_curriculum_proposals": config.use_curriculum_proposals,
                "use_retrieval_proposals": config.use_retrieval_proposals,
            }
        )
        return EvalMetrics(total=10, passed=9, average_steps=1.0, generated_total=10, generated_passed=9)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=artifact_path,
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "skills",
            "--cycle-id",
            "cycle:skills:test",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    assert seen_configs == [
        {
            "use_prompt_proposals": True,
            "use_curriculum_proposals": True,
            "use_retrieval_proposals": True,
        },
        {
            "use_prompt_proposals": True,
            "use_curriculum_proposals": True,
            "use_retrieval_proposals": True,
        },
    ]


def test_finalize_cycle_requires_confirmation_for_policy(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "prompts" / "prompt_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "proposed",
                "retention_gate": {
                    "min_pass_rate_delta_abs": 0.01,
                    "max_step_regression": 0.0,
                    "require_generated_lane_non_regression": True,
                    "required_confirmation_runs": 2,
                },
                "controls": {},
                "proposals": [
                    {"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    run_results = [
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            generated_total=4,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.2,
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            generated_total=4,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.2,
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return run_results.pop(0)

    def fast_atomic_write_json(path, payload, *, config=None):
        del config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(module.cycle_runner, "atomic_write_json", fast_atomic_write_json)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=artifact_path,
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--cycle-id",
            "cycle:policy:test",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    confirmation_records = [
        record for record in records if record["state"] == "evaluate" and record["action"] == "confirm_candidate_to_baseline"
    ]
    assert len(confirmation_records) == 1
    assert confirmation_records[0]["metrics_summary"]["confirmation_run_count"] == 2


def test_run_repeated_improvement_cycles_writes_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:stale:skills:1",
            state="observe",
            subsystem="skills",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="selection context",
            metrics_summary={
                "campaign_breadth_pressure": 0.4,
                "selected_variant_breadth_pressure": 0.3,
                "selected_variant_id": "careful",
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:stale:skills:1",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills.json",
            artifact_kind="skill_set",
            reason="retained gain",
            metrics_summary={
                "baseline_pass_rate": 0.7,
                "candidate_pass_rate": 0.8,
                "baseline_average_steps": 1.5,
                "candidate_average_steps": 1.2,
                "phase_gate_passed": True,
                "family_pass_rate_delta": {"project": 0.2, "repository": 0.0},
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:stale:test_only:1",
            state="reject",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="/tmp/pytest-candidate/skills.json",
            artifact_kind="skill_set",
            reason="test-only rejection",
            metrics_summary={
                "baseline_pass_rate": 0.8,
                "candidate_pass_rate": 0.7,
                "baseline_average_steps": 1.4,
                "candidate_average_steps": 1.6,
                "phase_gate_passed": False,
            },
        ),
    )

    seen_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env=None, progress_label=None, heartbeat_interval_seconds=0.0, max_silence_seconds=0.0, max_runtime_seconds=0.0, max_progress_stall_seconds=0.0, on_event=None):
        seen_cmds.append(list(cmd))
        protocol_match_id = cmd[cmd.index("--protocol-match-id") + 1]
        run_index = len(seen_cmds)
        cycle_id = f"cycle:campaign:{run_index}"
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem="skills",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason="selection context",
                metrics_summary={
                    "campaign_breadth_pressure": 0.4 if run_index == 1 else 0.0,
                    "selected_variant_breadth_pressure": 0.3 if run_index == 1 else 0.0,
                    "selected_variant_id": "careful",
                    "priority_family_allocation_summary": {
                        "priority_benchmark_families": ["project", "repository"],
                        "priority_benchmark_family_weights": {"project": 3.0, "repository": 1.0},
                        "planned_weight_shares": {"project": 0.75, "repository": 0.25},
                        "actual_task_counts": {"project": 3 if run_index == 1 else 2, "repository": 1},
                        "actual_task_shares": {"project": 0.75 if run_index == 1 else 0.666667, "repository": 0.25 if run_index == 1 else 0.333333},
                        "actual_pass_rates": {"project": 1.0, "repository": 1.0},
                        "actual_priority_tasks": 4 if run_index == 1 else 3,
                        "top_planned_family": "project",
                        "top_sampled_family": "project",
                    },
                    "protocol": "autonomous",
                    "protocol_match_id": protocol_match_id,
                },
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="generate",
                subsystem="skills",
                action="write_skill_module",
                artifact_path="skills.json",
                artifact_kind="skill_set",
                reason="generated variant",
                metrics_summary={
                    "selected_variant": {"variant_id": "careful", "estimated_cost": 3},
                    "protocol": "autonomous",
                    "protocol_match_id": protocol_match_id,
                },
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="retain" if run_index == 1 else "reject",
                subsystem="skills",
                action="finalize_cycle",
                artifact_path="skills.json" if run_index == 1 else "/tmp/pytest-candidate/skills.json",
                artifact_kind="skill_set",
                reason="retained gain" if run_index == 1 else "test-only rejection",
                metrics_summary={
                    "baseline_pass_rate": 0.7 if run_index == 1 else 0.8,
                    "candidate_pass_rate": 0.8 if run_index == 1 else 0.7,
                    "baseline_average_steps": 1.5 if run_index == 1 else 1.4,
                    "candidate_average_steps": 1.2 if run_index == 1 else 1.6,
                    "phase_gate_passed": run_index == 1,
                    "family_pass_rate_delta": {"project": 0.2, "repository": 0.0} if run_index == 1 else {},
                    "protocol": "autonomous",
                    "protocol_match_id": protocol_match_id,
                },
            ),
        )
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds, max_runtime_seconds, max_progress_stall_seconds, on_event
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": ["project", "repository"],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {"project": 0, "repository": 0},
            "missing_required_family_clean_task_root_breadth": ["project", "repository"],
            "distinct_family_gap": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "2",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    assert report_path.parent == reports_dir
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload["report_kind"] == "improvement_campaign_report"
    assert payload["cycles_requested"] == 2
    assert payload["successful_runs"] == 2
    assert payload["productive_runs"] == 1
    assert payload["retained_gain_runs"] == 1
    assert payload["runtime_managed_decisions"] == 1
    assert payload["priority_benchmark_families"] == ["project", "repository"]
    assert payload["production_yield_summary"]["retained_cycles"] == 1
    assert payload["decision_stream_summary"]["runtime_managed"]["retained_cycles"] == 1
    assert payload["decision_stream_summary"]["runtime_managed"]["rejected_cycles"] == 0
    assert payload["decision_stream_summary"]["non_runtime_managed"]["retained_cycles"] == 0
    assert payload["decision_stream_summary"]["non_runtime_managed"]["rejected_cycles"] == 1
    assert payload["yield_summary"]["total_decisions"] == 2
    assert payload["phase_gate_summary"]["all_retained_phase_gates_passed"] is True
    assert payload["trust_breadth_summary"]["external_report_count"] == 0
    assert payload["trust_breadth_summary"]["distinct_external_benchmark_families"] == 0
    assert "project" in payload["trust_breadth_summary"]["missing_required_families"]
    assert payload["trust_breadth_summary"]["family_breadth_min_distinct_task_roots"] == 2
    assert "project" in payload["trust_breadth_summary"]["missing_required_family_clean_task_root_breadth"]
    assert "project" in payload["trust_breadth_summary"]["required_family_clean_task_root_counts"]
    assert payload["priority_family_yield_summary"]["priority_families_with_retained_gain"] == ["project"]
    assert payload["priority_family_yield_summary"]["priority_families_with_signal_but_no_retained_gain"] == ["repository"]
    assert payload["priority_family_yield_summary"]["family_summaries"]["project"]["retained_positive_delta_decisions"] == 1
    assert payload["priority_family_yield_summary"]["family_summaries"]["repository"]["observed_decisions"] == 1
    assert payload["priority_family_yield_summary"]["family_summaries"]["project"]["observed_estimated_cost"] == 3.0
    assert payload["priority_family_yield_summary"]["family_summaries"]["project"]["retained_estimated_cost"] == 3.0
    assert payload["priority_family_allocation_summary"]["top_planned_family"] == "project"
    assert payload["priority_family_allocation_summary"]["top_sampled_family"] == "project"
    assert payload["priority_family_allocation_summary"]["aggregated_task_counts"]["project"] == 5
    assert payload["priority_family_allocation_summary"]["aggregated_task_counts"]["repository"] == 2
    assert payload["incomplete_cycle_summary"]["count"] == 0
    assert payload["planner_pressure_summary"]["campaign_breadth_pressure_cycles"] == 1
    assert payload["planner_pressure_summary"]["variant_breadth_pressure_cycles"] == 1
    assert payload["inheritance_summary"]["runtime_managed_decisions"] == 1
    assert payload["candidate_isolation_summary"]["decision_count"] == 2
    assert payload["candidate_isolation_summary"]["runtime_managed_distinct_candidate_and_active_paths"] == 0
    assert payload["candidate_isolation_summary"]["missing_path_audit_decisions"] == 2
    assert len(payload["recent_runtime_managed_decisions"]) == 1
    assert payload["recent_runtime_managed_decisions"][0]["cycle_id"] == "cycle:campaign:1"
    assert len(payload["recent_non_runtime_decisions"]) == 1
    assert payload["recent_non_runtime_decisions"][0]["cycle_id"] == "cycle:campaign:2"
    assert payload["record_scope"]["cycle_log_start_index"] == 3
    assert payload["record_scope"]["decision_records_considered"] == 2
    assert payload["record_scope"]["cycle_ids"] == ["cycle:campaign:1", "cycle:campaign:2"]
    assert payload["decision_conversion_summary"] == {
        "runtime_managed_runs": 1,
        "non_runtime_managed_runs": 1,
        "incomplete_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "no_decision_runs": 0,
        "decision_runs": 2,
    }
    assert payload["decision_state_summary"]["run_decisions"] == {
        "child_native": 1,
        "controller_runtime_manager": 1,
        "none": 0,
    }
    assert payload["decision_state_summary"]["run_closeout_modes"]["natural"] == 1
    assert payload["decision_state_summary"]["run_closeout_modes"]["forced_reject"] == 1
    assert payload["decision_state_summary"]["record_decisions"] == {
        "child_native": 1,
        "controller_runtime_manager": 1,
    }
    assert payload["runs"][0]["final_state"] == "retain"
    assert payload["runs"][0]["decision_conversion_state"] == "runtime_managed"
    assert payload["runs"][0]["decision_state"]["decision_owner"] == "child_native"
    assert payload["runs"][0]["decision_state"]["closeout_mode"] == "natural"
    assert payload["runs"][1]["final_state"] == "reject"
    assert payload["runs"][1]["final_reason"] == "test-only rejection"
    assert payload["runs"][1]["decision_conversion_state"] == "non_runtime_managed"
    assert payload["runs"][1]["decision_state"]["decision_owner"] == "controller_runtime_manager"
    assert payload["runs"][1]["decision_state"]["controller_intervention_reason_code"] == "all_primary_verifications_failed"
    assert payload["recent_non_runtime_decisions"][0]["decision_state"]["decision_owner"] == "controller_runtime_manager"
    assert records[-1]["artifact_kind"] == "improvement_campaign_report"
    assert records[-1]["metrics_summary"]["priority_families_with_retained_gain"] == ["project"]
    assert "--priority-benchmark-family" in seen_cmds[0]
    assert "--protocol-match-id" in seen_cmds[0]
    assert "project" in seen_cmds[0]


def test_campaign_status_snapshot_counts_fail_closed_decisions_for_matching_cycle_ids(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)

    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:test",
            state="observe",
            subsystem="retrieval",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="low-confidence retrieval remains common",
            metrics_summary={
                "protocol": "autonomous",
                "protocol_match_id": "campaign:test",
                "priority_benchmark_families": ["project", "integration"],
                "priority_benchmark_family_weights": {"project": 1.0, "integration": 1.0},
                "sampled_task_counts": {"project": 1, "integration": 1},
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:test",
            state="incomplete",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="trajectories/retrieval/retrieval_proposals.json",
            artifact_kind="retrieval_policy_set",
            reason="incomplete autonomous cycle was reconciled without a retention decision",
            metrics_summary={
                "protocol": "autonomous",
                "preview_reason_code": "retrieval_retained_gate_failed",
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:test",
            state="record",
            subsystem="retrieval",
            action="persist_retention_outcome",
            artifact_path="trajectories/retrieval/retrieval_proposals.json",
            artifact_kind="retrieval_policy_set",
            reason="persisted fail-closed outcome",
            metrics_summary={"protocol": "autonomous"},
        ),
    )

    snapshot = module._campaign_status_snapshot(
        config=KernelConfig(
            improvement_reports_dir=tmp_path / "improvement" / "reports",
            improvement_cycles_path=cycles_path,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
        planner=planner,
        cycles_path=cycles_path,
        campaign_match_id="campaign:test",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "integration"],
        priority_family_weights={"project": 1.0, "integration": 1.0},
    )

    assert snapshot["campaign_records_considered"] == 3
    assert snapshot["decision_records_considered"] == 0
    assert snapshot["campaign_cycle_ids"] == [
        "cycle:retrieval:test",
        "cycle:retrieval:test",
        "cycle:retrieval:test",
    ]
    assert snapshot["families_sampled"] == []


def test_campaign_status_snapshot_recovers_decision_records_from_cycle_report(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    config = KernelConfig(
        improvement_cycles_path=cycles_path,
        improvement_reports_dir=reports_dir,
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cycle_report_cycle_retrieval_test_live.json").write_text(
        json.dumps(
            {
                "report_kind": "improvement_cycle_report",
                "cycle_id": "cycle:retrieval:test-live",
                "current_cycle_records": [
                    {
                        "cycle_id": "cycle:retrieval:test-live",
                        "state": "reject",
                        "subsystem": "retrieval",
                        "action": "finalize_cycle",
                        "artifact_path": "trajectories/retrieval/retrieval_proposals.json",
                        "artifact_kind": "retention_decision",
                        "reason": "retrieval candidate did not satisfy the retained retrieval gate",
                        "metrics_summary": {
                            "protocol": "autonomous",
                            "protocol_match_id": "campaign:test-live",
                            "family_pass_rate_delta": {"project": 0.0, "integration": 0.0},
                        },
                    },
                    {
                        "cycle_id": "cycle:retrieval:test-live",
                        "state": "record",
                        "subsystem": "retrieval",
                        "action": "persist_retention_outcome",
                        "artifact_path": "trajectories/retrieval/retrieval_proposals.json",
                        "artifact_kind": "retention_record",
                        "reason": "persisted artifact lifecycle and cycle-lineage metadata",
                        "metrics_summary": {
                            "protocol": "autonomous",
                            "protocol_match_id": "campaign:test-live",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    snapshot = module._campaign_status_snapshot(
        config=config,
        planner=planner,
        cycles_path=cycles_path,
        campaign_match_id="campaign:test-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "integration"],
        priority_family_weights={"project": 1.0, "integration": 1.0},
    )

    assert snapshot["campaign_records_considered"] == 2
    assert snapshot["decision_records_considered"] == 1
    assert snapshot["campaign_cycle_ids"] == [
        "cycle:retrieval:test-live",
        "cycle:retrieval:test-live",
    ]


def test_repeated_verification_outcome_summary_prefers_latest_task_outcome():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._verification_outcome_summary(
        {
            "verified_task_ids": ["task_a", "task_b"],
            "failed_verification_task_ids": ["task_a", "task_b"],
            "successful_verification_task_ids": ["task_a"],
            "verified_families": ["integration", "repository"],
            "failed_verification_families": ["integration", "repository"],
            "successful_verification_families": ["integration"],
            "verification_outcomes_by_task": {"task_a": True, "task_b": False},
            "verification_task_families": {"task_a": "integration", "task_b": "repository"},
        }
    )

    assert summary["verified_task_count"] == 2
    assert summary["successful_task_count"] == 1
    assert summary["failed_task_count"] == 1
    assert summary["successful_task_ids"] == ["task_a"]
    assert summary["failed_task_ids"] == ["task_b"]
    assert summary["verified_families"] == ["integration", "repository"]
    assert summary["successful_families"] == ["integration"]
    assert summary["failed_families"] == ["repository", "integration"]
    assert summary["all_verified_tasks_failed"] is False


def test_repeated_decision_state_from_run_treats_reconciled_incomplete_runtime_decision_as_controller_fail_closed():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    decision_state = module._decision_state_from_run(
        {
            "final_state": "reject",
            "final_reason": "persisted fail-closed outcome for incomplete autonomous cycle",
            "timeout_reason": "child exceeded max runtime of 1200 seconds",
            "runtime_managed_decisions": 1,
            "incomplete_cycle_count": 1,
        }
    )

    assert decision_state["decision_owner"] == "controller_runtime_manager"
    assert decision_state["decision_credit"] == "controller_fail_closed"
    assert decision_state["closeout_mode"] == "forced_reject"
    assert decision_state["retention_state"] == "reject"
    assert decision_state["controller_intervention_reason_code"] == "child_max_runtime"


def test_repeated_candidate_isolation_summary_uses_record_paths_when_generate_record_is_missing():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._candidate_isolation_summary(
        [
            {
                "cycle_id": "cycle:test:1",
                "artifact_path": "retained/artifact.json",
                "candidate_artifact_path": "candidates/artifact.json",
                "active_artifact_path": "retained/artifact.json",
            }
        ],
        {},
    )

    assert summary["decision_count"] == 1
    assert summary["decisions_with_candidate_path"] == 1
    assert summary["decisions_with_active_path"] == 1
    assert summary["runtime_managed_distinct_candidate_and_active_paths"] == 1
    assert summary["missing_path_audit_decisions"] == 0


def test_repeated_incomplete_cycle_rollup_preserves_reconciled_runtime_decision_signal():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    incomplete_cycles = module._merge_incomplete_cycle_summaries(
        [],
        [
            {
                "cycle_id": "cycle:tolbert:test",
                "subsystem": "tolbert_model",
                "selected_variant_id": "discovered_task_adaptation",
                "artifact_path": "retained/tolbert_model_artifact.json",
                "active_artifact_path": "retained/tolbert_model_artifact.json",
                "candidate_artifact_path": "candidates/tolbert_model_artifact.json",
                "runtime_managed_decisions": 1,
                "incomplete_cycle_count": 1,
            }
        ],
        [
            {
                "cycle_id": "cycle:tolbert:test",
                "state": "reject",
                "subsystem": "tolbert_model",
                "artifact_path": "retained/tolbert_model_artifact.json",
                "candidate_artifact_path": "candidates/tolbert_model_artifact.json",
                "active_artifact_path": "retained/tolbert_model_artifact.json",
                "metrics_summary": {
                    "incomplete_cycle": True,
                    "selected_variant_id": "discovered_task_adaptation",
                },
            }
        ],
    )

    assert incomplete_cycles == [
        {
            "cycle_id": "cycle:tolbert:test",
            "subsystem": "tolbert_model",
            "selected_variant_id": "discovered_task_adaptation",
            "artifact_path": "retained/tolbert_model_artifact.json",
            "active_artifact_path": "retained/tolbert_model_artifact.json",
            "candidate_artifact_path": "candidates/tolbert_model_artifact.json",
        }
    ]


def test_parse_progress_fields_extracts_pending_decision_state():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed_apply = module._parse_progress_fields(
        "[cycle:demo] finalize phase=apply_decision subsystem=retrieval state=reject"
    )
    assert parsed_apply["finalize_phase"] == "apply_decision"
    assert parsed_apply["selected_subsystem"] == "retrieval"
    assert parsed_apply["pending_decision_state"] == "reject"

    parsed_done = module._parse_progress_fields(
        "[cycle:demo] finalize phase=done subsystem=retrieval state=retain"
    )
    assert parsed_done["finalize_phase"] == "done"
    assert parsed_done["pending_decision_state"] == "retain"

    parsed_preview_complete = module._parse_progress_fields(
        "[cycle:demo] preview complete subsystem=retrieval variant=breadth_rebalance "
        "state=reject baseline_pass_rate=0.9167 candidate_pass_rate=0.9167 phase_gate_passed=False"
    )
    assert parsed_preview_complete["selected_subsystem"] == "retrieval"
    assert parsed_preview_complete["last_candidate_variant"] == "breadth_rebalance"
    assert parsed_preview_complete["preview_state"] == "reject"
    assert parsed_preview_complete.get("pending_decision_state", "") == ""

    parsed_preview_eval = module._parse_progress_fields(
        "[cycle:demo] finalize phase=preview_candidate_eval subsystem=retrieval"
    )
    assert parsed_preview_eval["finalize_phase"] == "preview_candidate_eval"
    assert parsed_preview_eval["preview_state"] == ""

    parsed_decision_reject = module._parse_progress_fields(
        "[cycle:demo] finalize phase=decision_reject_reason subsystem=retrieval "
        "reason_code=autonomous_generated_lane_missing reason=generated-task lane was not included"
    )
    assert parsed_decision_reject["finalize_phase"] == "decision_reject_reason"
    assert parsed_decision_reject["pending_decision_state"] == "reject"

    parsed_decision_retain = module._parse_progress_fields(
        "[cycle:demo] finalize phase=decision_retain_reason subsystem=retrieval "
        "reason_code=retained reason=retrieval candidate increased trusted retrieval usage"
    )
    assert parsed_decision_retain["finalize_phase"] == "decision_retain_reason"
    assert parsed_decision_retain["pending_decision_state"] == "retain"


def test_parse_progress_fields_clears_stale_decision_state_when_new_observe_cycle_starts():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._parse_progress_fields("[cycle:demo] phase=observe start")

    assert payload["last_progress_phase"] == "observe"
    assert payload["current_task"] == {}
    assert payload["pending_decision_state"] == ""
    assert payload["preview_state"] == ""
    assert payload["finalize_phase"] == ""
    assert payload["selected_subsystem"] == ""


def test_parse_progress_fields_resets_phase_completion_when_new_generated_phase_starts():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed_total = module._parse_progress_fields("[eval:demo] phase=generated_success total=165")
    assert parsed_total["generated_success_total"] == 165
    assert parsed_total["generated_success_completed"] is False

    parsed_mid = module._parse_progress_fields(
        "[eval:demo] phase=generated_success task 160/165 service_release_task family=discovered_task"
    )
    assert "generated_success_completed" not in parsed_mid

    parsed_end = module._parse_progress_fields(
        "[eval:demo] phase=generated_success task 165/165 service_release_task family=discovered_task"
    )
    assert parsed_end.get("generated_success_completed", False) is False

    parsed_complete = module._parse_progress_fields("[eval:demo] phase=generated_success complete total=165")
    assert parsed_complete["generated_success_completed"] is True


def test_write_campaign_status_credits_pending_live_decision(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-live",
        campaign_label="campaign-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "index": 1,
            "selected_subsystem": "retrieval",
            "finalize_phase": "apply_decision",
            "pending_decision_state": "reject",
        },
        snapshot={
            "campaign_records_considered": 3,
            "decision_records_considered": 0,
            "campaign_cycle_ids": ["cycle:retrieval:demo"],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository"],
                "aggregated_task_counts": {"project": 0, "repository": 0},
            },
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["campaign_records_considered"] == 3
    assert payload["decision_records_considered"] == 1
    assert payload["pending_decision_state"] == "reject"


def test_write_campaign_status_credits_preview_reject_as_live_decision(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-live",
        campaign_label="campaign-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "index": 1,
            "selected_subsystem": "retrieval",
            "finalize_phase": "preview_complete",
            "preview_state": "reject",
            "last_candidate_variant": "breadth_rebalance",
        },
        snapshot={
            "campaign_records_considered": 5,
            "decision_records_considered": 0,
            "campaign_cycle_ids": ["cycle:retrieval:demo"],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository"],
                "aggregated_task_counts": {"project": 0, "repository": 0},
            },
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["campaign_records_considered"] == 5
    assert payload["decision_records_considered"] == 0
    assert payload["preview_state"] == "reject"
    assert payload.get("pending_decision_state", "") == ""


def test_write_campaign_status_prefers_live_active_cycle_run_over_stale_snapshot(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-live",
        campaign_label="campaign-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "index": 1,
            "last_event": "output",
            "progress_event_count": 5,
            "progress_output_count": 3,
            "current_task": {
                "index": 1,
                "total": 32,
                "task_id": "deployment_manifest_task",
                "family": "project",
                "phase": "observe",
            },
            "sampled_families_from_progress": ["project"],
            "started_at": 100.0,
            "last_event_at": 100.0,
        },
        snapshot={
            "active_cycle_run": {"last_event": "start"},
            "semantic_progress_state": {"phase": "", "progress_class": "unknown"},
            "families_sampled": [],
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["active_cycle_run"]["last_event"] == "output"
    assert payload["active_cycle_run"]["current_task"]["family"] == "project"
    assert payload["progress_events_observed"] == 5
    assert payload["families_sampled"] == ["project"]


def test_write_campaign_status_includes_tolbert_runtime_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-live",
        campaign_label="campaign-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=2,
        runs=[
            {
                "index": 1,
                "tolbert_runtime_summary": {
                    "configured_to_use_tolbert": True,
                    "successful_tolbert_stages": ["preview_baseline", "preview_candidate"],
                },
            }
        ],
        state="running",
        active_cycle_run={
            "index": 2,
            "tolbert_runtime_summary": {
                "configured_to_use_tolbert": True,
                "startup_failure_stages": ["holdout_baseline"],
                "recovered_without_tolbert_stages": ["holdout_baseline"],
                "bypassed_stages": ["holdout_baseline"],
            },
        },
        snapshot={
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["tolbert_runtime_summary"]["configured_runs"] == 1
    assert payload["tolbert_runtime_summary"]["startup_failure_count"] == 0
    assert payload["tolbert_runtime_summary"]["outcome_counts"]["succeeded"] == 1
    assert payload["tolbert_runtime_summary"]["active"]["outcome"] == "failed_recovered"


def test_active_cycle_semantic_progress_state_marks_long_silent_observe_as_stuck(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = module._active_cycle_semantic_progress_state(
        {
            "started_at": 100.0,
            "last_event_at": 100.0,
            "current_task": {
                "index": 1,
                "total": 32,
                "task_id": "deployment_manifest_task",
                "family": "project",
                "phase": "observe",
            },
        },
        now=231.0,
        max_progress_stall_seconds=1800.0,
        max_runtime_seconds=0.0,
    )

    assert state["phase_family"] == "observe"
    assert state["progress_class"] == "stuck"
    assert state["status"] == "stalled"


def test_active_cycle_semantic_progress_state_treats_zero_pass_observe_summary_as_complete(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = module._active_cycle_semantic_progress_state(
        {
            "started_at": 100.0,
            "last_event_at": 110.0,
            "last_progress_phase": "observe",
            "observe_completed": True,
            "observe_summary": {
                "passed": 0,
                "total": 32,
                "pass_rate": 0.0,
                "generated_pass_rate": 0.0,
            },
            "current_task": {},
        },
        now=111.0,
        max_progress_stall_seconds=1800.0,
        max_runtime_seconds=0.0,
    )

    assert state["phase"] == "observe"
    assert state["phase_family"] == "observe"
    assert state["progress_class"] == "complete"
    assert state["status"] == "complete"


def test_active_cycle_semantic_progress_state_keeps_variant_search_post_observe_out_of_observe_complete(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = module._active_cycle_semantic_progress_state(
        {
            "started_at": 100.0,
            "last_event_at": 110.0,
            "last_progress_phase": "variant_search",
            "observe_completed": True,
            "observe_summary": {
                "passed": 0,
                "total": 32,
                "pass_rate": 0.0,
                "generated_pass_rate": 0.0,
            },
            "selected_subsystem": "retrieval",
            "selected_variants": 2,
            "selected_variant_ids": ["breadth_rebalance", "confidence_gating"],
            "current_task": {},
        },
        now=111.0,
        max_progress_stall_seconds=1800.0,
        max_runtime_seconds=0.0,
    )

    assert state["phase"] == "variant_search"
    assert state["phase_family"] == "preview"
    assert state["progress_class"] == "healthy"
    assert state["status"] == "active"


def test_active_cycle_semantic_progress_state_prioritizes_generated_success_task_over_stale_finalize_phase(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = module._active_cycle_semantic_progress_state(
        {
            "started_at": 100.0,
            "last_event_at": 110.0,
            "last_progress_phase": "generated_success",
            "finalize_phase": "preview_baseline_eval",
            "current_task": {
                "index": 20,
                "total": 32,
                "task_id": "integration_failover_drill_task",
                "family": "integration",
                "phase": "generated_success",
            },
        },
        now=111.0,
        max_progress_stall_seconds=1800.0,
        max_runtime_seconds=3600.0,
    )

    assert state["phase"] == "generated_success"
    assert state["phase_family"] == "finalize"


def test_active_cycle_semantic_progress_state_does_not_treat_preview_only_state_as_decision_emitted(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = module._active_cycle_semantic_progress_state(
        {
            "started_at": 100.0,
            "last_event_at": 110.0,
            "last_progress_phase": "preview_candidate_eval",
            "preview_state": "retain",
            "pending_decision_state": "",
            "current_task": {
                "index": 2,
                "total": 16,
                "task_id": "cli_exchange_task_tool_recovery",
                "family": "tooling",
                "phase": "preview_candidate_eval",
                "raw_phase": "generated_failure",
            },
        },
        now=111.0,
        max_progress_stall_seconds=1800.0,
        max_runtime_seconds=3600.0,
    )

    assert state["phase"] == "preview_candidate_eval"
    assert state["phase_family"] == "preview"
    assert state["status"] == "active"
    assert state["decision_distance"] == "near"
    assert state["progress_class"] == "healthy"
    assert state["detail"] == "preview task 2/16 is progressing"


def test_parse_progress_fields_promotes_campaign_selection_past_observe_completion(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    observe_payload = module._parse_progress_fields(
        "[cycle:cycle-1] observe complete passed=0/32 pass_rate=0.0000 generated_pass_rate=0.0000"
    )
    assert observe_payload["last_progress_phase"] == "observe"
    assert observe_payload["observe_completed"] is True
    assert observe_payload["current_task"] == {}

    selected_payload = module._parse_progress_fields("[cycle:cycle-1] campaign 1/1 select subsystem=retrieval")
    assert selected_payload["last_progress_phase"] == "campaign_select"
    assert selected_payload["selected_subsystem"] == "retrieval"


def test_parse_progress_fields_marks_tolbert_stage_pending_when_preview_starts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._parse_progress_fields(
        "[cycle:cycle-1] finalize phase=preview_baseline_eval subsystem=tolbert_model"
    )

    assert payload["last_progress_phase"] == "preview_baseline_eval"
    assert payload["finalize_phase"] == "preview_baseline_eval"
    assert payload["current_task"] == {}
    assert payload["tolbert_runtime_summary"]["stages_attempted"] == ["preview_baseline"]
    assert payload["tolbert_runtime_summary"]["outcome"] == "pending"


def test_comparison_task_limit_for_retention_caps_non_retrieval_preview_budget():
    assert cycle_runner._comparison_task_limit_for_retention(
        "tolbert_model",
        task_limit=32,
        payload=None,
        capability_modules_path=None,
    ) == 16


def test_comparison_task_limit_for_retention_caps_retrieval_preview_budget():
    assert cycle_runner._comparison_task_limit_for_retention(
        "retrieval",
        task_limit=32,
        payload={"preview_controls": {}},
        capability_modules_path=None,
    ) == 8


def test_write_campaign_status_credits_completed_run_outcomes_as_live_evidence(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-live",
        campaign_label="campaign-live",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=2,
        runs=[
            {
                "index": 1,
                "returncode": 0,
                "productive": True,
                "campaign_cycle_ids": ["cycle:retrieval:run-1"],
                "runtime_managed_decisions": 0,
                "decision_conversion_state": "non_runtime_managed",
                "final_state": "reject",
                "retained_gain": False,
            },
            {
                "index": 2,
                "returncode": 0,
                "productive": True,
                "campaign_cycle_ids": ["cycle:tooling:run-2"],
                "runtime_managed_decisions": 1,
                "decision_conversion_state": "runtime_managed",
                "final_state": "retain",
                "retained_gain": True,
            },
        ],
        state="running",
        snapshot={
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "campaign_cycle_ids": [],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository"],
                "aggregated_task_counts": {"project": 0, "repository": 0},
            },
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["campaign_records_considered"] == 2
    assert payload["decision_records_considered"] == 2
    assert payload["runtime_managed_decisions"] == 1


def test_cycle_audit_summary_from_report_preserves_strategy_candidate_fields():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "cycle_id": "cycle:demo",
        "subsystem": "policy",
        "strategy_candidate_id": "strategy:broad_observe_diversification",
        "strategy_candidate_kind": "broad_observe_diversification",
        "final_state": "retain",
        "final_reason": "retained",
        "artifact_path": "/data/agentkernel/trajectories/improvement/retained/policy.json",
        "active_artifact_path": "/data/agentkernel/trajectories/improvement/retained/policy.json",
        "artifact_kind": "prompt_proposal_set",
        "decision_state": {
            "decision_owner": "child_native",
            "decision_credit": "child_native",
            "decision_conversion_state": "runtime_managed",
            "retention_state": "retain",
            "retention_basis": "retained",
            "closeout_mode": "natural",
            "controller_intervention_reason_code": "",
            "recorded_at": "2026-04-09T00:00:00+00:00",
        },
        "baseline_metrics": {"pass_rate": 0.8, "average_steps": 4.0},
        "candidate_metrics": {"pass_rate": 0.9, "average_steps": 3.0},
        "evidence": {},
        "current_cycle_records": [
            {
                "cycle_id": "cycle:demo",
                "state": "retain",
                "subsystem": "policy",
                    "artifact_path": "/data/agentkernel/trajectories/improvement/retained/policy.json",
                "artifact_kind": "prompt_proposal_set",
                "reason": "retained",
                "metrics_summary": {},
            }
        ],
    }

    payload = module._cycle_audit_summary_from_report(report)

    assert payload is not None
    assert payload["strategy_candidate_id"] == "strategy:broad_observe_diversification"
    assert payload["strategy_candidate_kind"] == "broad_observe_diversification"
    assert payload["decision_state"]["decision_owner"] == "child_native"


def test_priority_family_yield_summary_counts_neutral_retrieval_support_retain_as_gain():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._priority_family_yield_summary(
        [
            {
                "state": "retain",
                "subsystem": "retrieval",
                "final_reason": "retrieval candidate preserved verified long-horizon trusted-retrieval carryover without regressing the base lane",
                "metrics_summary": {
                    "phase_gate_passed": True,
                    "baseline_pass_rate": 1.0,
                    "candidate_pass_rate": 1.0,
                    "baseline_average_steps": 2.0,
                    "candidate_average_steps": 2.0,
                    "family_pass_rate_delta": {
                        "project": 0.0,
                        "repository": 0.0,
                    },
                },
            }
        ],
        ["project", "repository"],
    )

    assert summary["priority_families_with_retained_gain"] == ["project", "repository"]
    assert summary["priority_families_with_signal_but_no_retained_gain"] == []
    assert summary["family_summaries"]["project"]["retained_positive_delta_decisions"] == 1
    assert summary["family_summaries"]["repository"]["retained_positive_delta_decisions"] == 1


def test_record_counts_as_retained_gain_accepts_neutral_retrieval_support_retain():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._record_counts_as_retained_gain(
        {
            "final_state": "retain",
            "subsystem": "retrieval",
            "final_reason": "retrieval candidate strengthened complementary retrieval support without regressing the base lane",
            "phase_gate_passed": True,
            "baseline_pass_rate": 1.0,
            "candidate_pass_rate": 1.0,
            "baseline_average_steps": 2.0,
            "candidate_average_steps": 2.0,
        }
    )


def test_record_counts_as_retained_gain_accepts_neutral_tolbert_support_retain():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._record_counts_as_retained_gain(
        {
            "final_state": "retain",
            "subsystem": "tolbert_model",
            "phase_gate_passed": True,
            "baseline_pass_rate": 1.0,
            "candidate_pass_rate": 1.0,
            "baseline_average_steps": 2.0,
            "candidate_average_steps": 2.0,
            "metrics_summary": {
                "family_pass_rate_delta": {
                    "integration": 0.0,
                },
                "tolbert_runtime_summary": {
                    "configured_to_use_tolbert": True,
                    "successful_tolbert_stages": ["generated_success"],
                },
            },
        }
    )


def test_record_counts_as_reject_learning_opportunity_accepts_non_regressive_reject():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._record_counts_as_reject_learning_opportunity(
        {
            "final_state": "reject",
            "subsystem": "tolbert_model",
            "phase_gate_passed": True,
            "baseline_pass_rate": 1.0,
            "candidate_pass_rate": 1.0,
            "baseline_average_steps": 2.0,
            "candidate_average_steps": 2.0,
            "metrics_summary": {
                "family_pass_rate_delta": {
                    "integration": 0.0,
                },
            },
        }
    )


def test_write_campaign_status_emits_semantic_progress_state_for_preview_phase(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-semantic",
        campaign_label="campaign-semantic",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "started_at": 100.0,
            "last_event_at": 110.0,
            "selected_subsystem": "retrieval",
            "finalize_phase": "preview_candidate_eval",
            "observe_summary": {"passed": 4, "total": 4},
            "sampled_families_from_progress": ["project", "repository"],
        },
        snapshot={
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["semantic_progress_state"]["phase"] == "preview_candidate_eval"
    assert payload["semantic_progress_state"]["phase_family"] == "preview"
    assert payload["semantic_progress_state"]["progress_class"] == "healthy"
    assert payload["active_cycle_run"]["semantic_progress_state"]["decision_distance"] == "near"


def test_write_campaign_status_prefers_preview_finalize_over_stale_generated_task_phase(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-generated",
        campaign_label="campaign-generated",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "started_at": 100.0,
            "last_event_at": 110.0,
            "selected_subsystem": "retrieval",
            "finalize_phase": "preview_candidate_eval",
            "last_progress_phase": "generated_failure",
            "current_task": {
                "index": 16,
                "total": 16,
                "task_id": "api_contract_retrieval_task_episode_replay",
                "family": "episode_memory",
                "phase": "generated_failure",
            },
            "observe_summary": {"passed": 0, "total": 16},
            "sampled_families_from_progress": ["project", "repository"],
            "generated_failure_total": 16,
            "generated_failure_completed": True,
        },
        snapshot={
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["semantic_progress_state"]["phase"] == "preview_candidate_eval"
    assert payload["semantic_progress_state"]["phase_family"] == "preview"
    assert payload["semantic_progress_state"]["progress_class"] == "healthy"


def test_write_campaign_status_normalizes_preview_current_task_phase(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    report_path = config.improvement_reports_dir / "campaign_report.json"
    status_path = module._write_campaign_status(
        config=config,
        report_path=report_path,
        planner=planner,
        cycles_path=config.improvement_cycles_path,
        campaign_match_id="campaign-preview-normalized",
        campaign_label="campaign-preview-normalized",
        cycle_log_start_index=0,
        priority_benchmark_families=["project", "repository"],
        priority_family_weights={"project": 1.0, "repository": 1.0},
        cycles_requested=1,
        runs=[],
        state="running",
        active_cycle_run={
            "started_at": 100.0,
            "last_event_at": 110.0,
            "selected_subsystem": "retrieval",
            "finalize_phase": "preview_candidate_eval",
            "last_progress_phase": "generated_success",
            "current_task": {
                "index": 20,
                "total": 32,
                "task_id": "integration_failover_drill_task_integration_recovery",
                "family": "integration",
                "phase": "generated_success",
            },
            "observe_summary": {"passed": 0, "total": 8},
            "sampled_families_from_progress": ["project", "repository", "integration"],
        },
        snapshot={
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "families_sampled": [],
            "priority_families_without_sampling": [],
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["active_cycle_run"]["last_progress_phase"] == "generated_success"
    assert payload["active_cycle_run"]["raw_last_progress_phase"] == "generated_success"
    assert payload["active_cycle_run"]["current_task"]["phase"] == "generated_success"
    assert payload["active_cycle_run"]["current_task"]["raw_phase"] == "generated_success"


def test_run_repeated_improvement_cycles_writes_midrun_child_status_and_mirrors_parent(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    parent_status_path = tmp_path / "parent" / "autonomous_compounding_status.json"
    parent_status_path.parent.mkdir(parents=True, exist_ok=True)
    parent_status_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_status",
                "requested_priority_benchmark_families": ["project", "repository"],
                "partial_frontier_expansion_summary": {
                    "families_sampled": [],
                },
                "active_run": {
                    "run_index": 1,
                    "run_match_id": "autonomous:run:1",
                },
            }
        ),
        encoding="utf-8",
    )
    observed_child_status: dict[str, object] = {}
    observed_parent_status: dict[str, object] = {}

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id="cycle:campaign:1",
                state="observe",
                subsystem="skills",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason="selection context",
                metrics_summary={
                    "priority_family_allocation_summary": {
                        "priority_benchmark_families": ["project", "repository"],
                        "priority_benchmark_family_weights": {"project": 2.0, "repository": 1.0},
                        "planned_weight_shares": {"project": 0.666667, "repository": 0.333333},
                        "actual_task_counts": {"project": 1, "repository": 0},
                        "actual_task_shares": {"project": 1.0, "repository": 0.0},
                        "actual_pass_rates": {"project": 1.0},
                        "actual_priority_tasks": 1,
                        "top_planned_family": "project",
                        "top_sampled_family": "project",
                    },
                    "protocol": "autonomous",
                    "protocol_match_id": "parent:run",
                },
            ),
        )
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event({"event": "output", "pid": 123, "timestamp": 1001.0, "line": "[cycle:cycle-1] observe complete passed=5/5 pass_rate=1.0000 generated_pass_rate=1.0000"})
            on_event({"event": "output", "pid": 123, "timestamp": 1002.0, "line": "[cycle:cycle-1] campaign 1/1 select subsystem=tooling"})
            on_event({"event": "output", "pid": 123, "timestamp": 1003.0, "line": "[cycle:cycle-1] variant generate complete subsystem=tooling variant=procedure_promotion artifact=/tmp/tool_candidates.json"})
        observed_child_status.update(
            json.loads((reports_dir / "repeated_improvement_status.json").read_text(encoding="utf-8"))
        )
        observed_parent_status.update(json.loads(parent_status_path.read_text(encoding="utf-8")))
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setenv("AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH", str(parent_status_path))
    monkeypatch.setenv("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX", "1")
    monkeypatch.setenv("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID", "autonomous:run:1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--campaign-match-id",
            "parent:run",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    assert observed_child_status["report_kind"] == "repeated_improvement_status"
    assert observed_child_status["state"] == "running"
    assert observed_child_status["families_sampled"] == ["project"]
    assert observed_child_status["priority_families_without_sampling"] == ["repository"]
    assert observed_child_status["active_cycle_run"]["last_event"] in {"start", "output"}
    assert observed_parent_status["families_sampled"] == ["project"]
    assert observed_parent_status["pressure_families_without_sampling"] == ["repository"]
    assert observed_parent_status["active_run"]["child_status"]["families_sampled"] == ["project"]
    assert observed_parent_status["active_run"]["child_status_path"].endswith("repeated_improvement_status.json")


def test_run_repeated_improvement_cycles_clears_failed_verification_on_new_task(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    observed_midrun_status: dict[str, object] = {}

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1001.0,
                    "line": "[eval:cycle-1] phase=primary task 1/8 deployment_manifest_task family=project cognitive_stage=memory_update_written step=1 verification_passed=0",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1002.0,
                    "line": "[eval:cycle-1] phase=primary task 2/8 repo_sync_matrix_task family=repository cognitive_stage=memory_retrieved subphase=graph_memory",
                }
            )
        observed_midrun_status.update(
            json.loads((reports_dir / "repeated_improvement_status.json").read_text(encoding="utf-8"))
        )
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_reconcile_incomplete_cycles", lambda **kwargs: [])
    monkeypatch.setattr(
        module,
        "_campaign_status_snapshot",
        lambda **kwargs: {
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "campaign_cycle_ids": [],
            "priority_family_allocation_summary": {
                "priority_families": [],
                "aggregated_task_counts": {},
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(module, "ImprovementPlanner", lambda *args, **kwargs: planner)
    monkeypatch.setattr(sys, "argv", ["run_repeated_improvement_cycles.py", "--cycles", "1"])
    monkeypatch.setattr(sys, "stdout", StringIO())

    module.main()

    active_cycle_run = observed_midrun_status["active_cycle_run"]
    assert active_cycle_run["current_task"]["task_id"] == "repo_sync_matrix_task"
    assert "current_task_verification_passed" not in active_cycle_run
    assert active_cycle_run["current_cognitive_stage"]["stage"] == "memory_retrieved"


def test_run_repeated_improvement_cycles_caches_snapshot_during_live_progress(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    snapshot_calls = {"count": 0}

    def fake_campaign_status_snapshot(**kwargs):
        del kwargs
        snapshot_calls["count"] += 1
        return {
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "campaign_cycle_ids": [],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository"],
                "aggregated_task_counts": {"project": 0, "repository": 0},
            },
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository"],
        }

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event({"event": "output", "pid": 123, "timestamp": 1001.0, "line": "[cycle:cycle-1] phase=observe start"})
            on_event({"event": "output", "pid": 123, "timestamp": 1002.0, "line": "[eval:cycle-1] task 1/2 deployment_manifest_task family=project"})
            on_event({"event": "output", "pid": 123, "timestamp": 1003.0, "line": "[cycle:cycle-1] observe complete passed=2/2 pass_rate=1.0000 generated_pass_rate=1.0000"})
            on_event({"event": "output", "pid": 123, "timestamp": 1004.0, "line": "[cycle:cycle-1] campaign 1/1 select subsystem=tooling"})
            on_event({"event": "output", "pid": 123, "timestamp": 1005.0, "line": "[cycle:cycle-1] variant generate start subsystem=tooling variant=procedure_promotion rank=1/1 expected_gain=0.0200 score=0.0000"})
            on_event({"event": "exit", "pid": 123, "timestamp": 1006.0, "returncode": 0})
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_campaign_status_snapshot", fake_campaign_status_snapshot)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--campaign-match-id",
            "cache:run",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    assert snapshot_calls["count"] <= 4


def test_run_repeated_improvement_cycles_records_timeout_reason_in_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        return {
            "returncode": -9,
            "stdout": "task 1/20 hello_task\n",
            "stderr": "[repeated] child=cycle-1 timeout reason=child exceeded max silence of 1800 seconds",
            "timed_out": True,
            "timeout_reason": "child exceeded max silence of 1800 seconds",
        }

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "3",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["completed_runs"] == 1
    assert payload["successful_runs"] == 0
    assert payload["productive_runs"] == 0
    assert payload["retained_gain_runs"] == 0
    assert payload["runs"][0]["returncode"] == -9
    assert payload["runs"][0]["timed_out"] is True
    assert payload["runs"][0]["timeout_reason"] == "child exceeded max silence of 1800 seconds"


def test_run_repeated_improvement_cycles_counts_partial_productive_timeout_progress(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observed_child_status = {}

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1001.0,
                    "line": "[cycle:cycle-1] observe complete passed=10/10 pass_rate=1.0000 generated_pass_rate=0.0000",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1002.0,
                    "line": "[eval:cycle-1] task 1/10 bridge_handoff_task family=integration",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1003.0,
                    "line": "[eval:cycle-1] phase=generated_success total=10",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1004.0,
                    "line": "[eval:cycle-1] phase=generated_success task 10/10 integration_failover_drill_task_integration_recovery family=integration",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1005.0,
                    "line": "[eval:cycle-1] phase=generated_success complete total=10",
                }
            )
        observed_child_status.update(
            json.loads((reports_dir / "repeated_improvement_status.json").read_text(encoding="utf-8"))
        )
        return {
            "returncode": -9,
            "stdout": "timeout\n",
            "stderr": "[repeated] child=cycle-1 timeout reason=child exceeded max runtime of 320 seconds",
            "timed_out": True,
            "timeout_reason": "child exceeded max runtime of 320 seconds",
        }

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--priority-benchmark-family",
            "integration",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["completed_runs"] == 1
    assert payload["successful_runs"] == 0
    assert payload["productive_runs"] == 1
    assert payload["retained_gain_runs"] == 0
    assert payload["runtime_managed_decisions"] == 0
    assert payload["partial_productive_runs"] == 1
    assert payload["decision_conversion_summary"] == {
        "runtime_managed_runs": 0,
        "non_runtime_managed_runs": 0,
        "incomplete_runs": 0,
        "partial_productive_without_decision_runs": 1,
        "no_decision_runs": 0,
        "decision_runs": 0,
    }
    assert payload["decision_state_summary"]["run_closeout_modes"]["partial_timeout_evidence_only"] == 1
    assert payload["partial_progress_summary"]["observed_primary_runs"] == 1
    assert payload["partial_progress_summary"]["generated_success_completed_runs"] == 1
    assert payload["partial_progress_summary"]["sampled_families_from_progress"] == ["integration"]
    assert payload["runs"][0]["partial_progress"]["observe_completed"] is True
    assert payload["runs"][0]["partial_progress"]["generated_success_completed"] is True
    assert payload["runs"][0]["decision_conversion_state"] == "partial_productive_without_decision"
    assert payload["decision_state_summary"]["run_decisions"]["none"] == 1
    assert payload["runs"][0]["decision_state"]["decision_owner"] == "none"
    assert payload["runs"][0]["decision_state"]["decision_credit"] == "partial_productive_evidence_only"
    assert payload["runs"][0]["decision_state"]["closeout_mode"] == "partial_timeout_evidence_only"
    assert observed_child_status["families_sampled"] == ["integration"]
    assert observed_child_status["priority_families_without_sampling"] == ["repository", "project"]
    assert observed_child_status["active_cycle_progress"]["generated_success_completed"] is True


def test_run_repeated_improvement_cycles_budget_reroute_uses_live_sampled_families_not_stale_record_snapshot(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    def fake_campaign_status_snapshot(**kwargs):
        del kwargs
        return {
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "campaign_cycle_ids": [],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository", "integration"],
                "aggregated_task_counts": {"project": 0, "repository": 0, "integration": 0},
            },
            "families_sampled": [],
            "priority_families_without_sampling": ["project", "repository", "integration"],
        }

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event({"event": "output", "pid": 123, "timestamp": 1001.0, "line": "[eval:cycle-1] task 1/3 deployment_manifest_task family=project"})
            on_event({"event": "output", "pid": 123, "timestamp": 1002.0, "line": "[eval:cycle-1] task 2/3 repo_sync_matrix_task family=repository"})
            on_event({"event": "output", "pid": 123, "timestamp": 1003.0, "line": "[cycle:cycle-1] observe complete passed=3/3 pass_rate=1.0000 generated_pass_rate=1.0000"})
            on_event({"event": "exit", "pid": 123, "timestamp": 1004.0, "returncode": 0})
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_campaign_status_snapshot", fake_campaign_status_snapshot)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--task-limit",
            "4",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "integration",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["runs"][0]["priority_family_budget_rerouting"]["unsampled_priority_families"] == ["integration"]
    assert payload["runs"][0]["priority_family_rerouting"]["unsampled_priority_families"] == ["integration"]


def test_run_repeated_improvement_cycles_reconciles_incomplete_autonomous_cycle_before_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    campaign_match_id = "campaign-reconcile"

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id="cycle:retrieval:test-incomplete",
                state="observe",
                subsystem="retrieval",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason="retrieval gap",
                metrics_summary={"protocol": "autonomous", "protocol_match_id": campaign_match_id},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id="cycle:retrieval:test-incomplete",
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={
                    "protocol": "autonomous",
                    "protocol_match_id": campaign_match_id,
                    "selected_variant_id": "breadth_rebalance",
                },
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id="cycle:retrieval:test-incomplete",
                state="generate",
                subsystem="retrieval",
                action="propose_retrieval_update",
                artifact_path="trajectories/retrieval/retrieval_proposals.json",
                artifact_kind="retrieval_policy_set",
                reason="retrieval gap",
                metrics_summary={"protocol": "autonomous", "protocol_match_id": campaign_match_id},
                candidate_artifact_path=(
                    "trajectories/improvement/candidates/retrieval/"
                    "cycle_retrieval_test_incomplete_breadth_rebalance/retrieval_proposals.json"
                ),
                active_artifact_path="trajectories/retrieval/retrieval_proposals.json",
            ),
        )
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": "child exited early",
            "timed_out": False,
            "timeout_reason": "",
        }

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--campaign-match-id",
            campaign_match_id,
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["runtime_managed_decisions"] == 1
    assert payload["retained_gain_runs"] == 0
    assert payload["decision_conversion_summary"] == {
        "runtime_managed_runs": 1,
        "non_runtime_managed_runs": 0,
        "incomplete_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "no_decision_runs": 0,
        "decision_runs": 1,
    }
    assert payload["runs"][0]["runtime_managed_decisions"] == 1
    assert payload["runs"][0]["final_state"] == "reject"
    assert payload["runs"][0]["decision_conversion_state"] == "runtime_managed"


def test_run_repeated_improvement_cycles_recovers_runtime_managed_decision_from_cycle_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    campaign_match_id = "campaign-report-recovery"
    cycle_id = "cycle:retrieval:test-report-recovery"

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_sibling_variant",
                artifact_path="trajectories/retrieval/retrieval_proposals.json",
                artifact_kind="sibling_variant_selection",
                reason="selected best measured sibling variant before final retention",
                metrics_summary={"protocol": "autonomous", "protocol_match_id": campaign_match_id},
                candidate_artifact_path=(
                    "trajectories/improvement/candidates/retrieval/"
                    "cycle_retrieval_test_report_recovery_breadth_rebalance/retrieval_proposals.json"
                ),
                active_artifact_path="trajectories/retrieval/retrieval_proposals.json",
            ),
        )
        report_payload = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_cycle_report",
            "cycle_id": cycle_id,
            "subsystem": "retrieval",
            "artifact_path": "trajectories/retrieval/retrieval_proposals.json",
            "active_artifact_path": "trajectories/retrieval/retrieval_proposals.json",
            "candidate_artifact_path": (
                "trajectories/improvement/candidates/retrieval/"
                "cycle_retrieval_test_report_recovery_breadth_rebalance/retrieval_proposals.json"
            ),
            "artifact_kind": "retention_decision",
            "final_state": "reject",
            "final_reason": "retrieval candidate did not satisfy the retained retrieval gate",
            "preview_reason_code": "retrieval_retained_gate_failed",
            "decision_reason_code": "retrieval_retained_gate_failed",
            "baseline_metrics": {"pass_rate": 0.9583, "average_steps": 1.8333},
            "candidate_metrics": {"pass_rate": 0.9583, "average_steps": 1.8333},
            "evidence": {
                "family_pass_rate_delta": {"project": 0.0, "repository": 0.0, "integration": 0.0},
                "generated_family_pass_rate_delta": {"project": 0.0, "integration": 0.0},
                "phase_gate_passed": True,
            },
            "current_cycle_records": [
                {
                    "cycle_id": cycle_id,
                    "state": "reject",
                    "subsystem": "retrieval",
                    "action": "finalize_cycle",
                    "artifact_path": "trajectories/retrieval/retrieval_proposals.json",
                    "artifact_kind": "retention_decision",
                    "reason": "retrieval candidate did not satisfy the retained retrieval gate",
                    "candidate_artifact_path": (
                        "trajectories/improvement/candidates/retrieval/"
                        "cycle_retrieval_test_report_recovery_breadth_rebalance/retrieval_proposals.json"
                    ),
                    "active_artifact_path": "trajectories/retrieval/retrieval_proposals.json",
                    "metrics_summary": {
                        "protocol": "autonomous",
                        "protocol_match_id": campaign_match_id,
                        "baseline_pass_rate": 0.9583,
                        "candidate_pass_rate": 0.9583,
                        "baseline_average_steps": 1.8333,
                        "candidate_average_steps": 1.8333,
                        "preview_reason_code": "retrieval_retained_gate_failed",
                        "decision_reason_code": "retrieval_retained_gate_failed",
                        "phase_gate_passed": True,
                        "family_pass_rate_delta": {"project": 0.0, "repository": 0.0, "integration": 0.0},
                        "generated_family_pass_rate_delta": {"project": 0.0, "integration": 0.0},
                    },
                }
            ],
        }
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "cycle_report_cycle_retrieval_test-report-recovery.json").write_text(
            json.dumps(report_payload),
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "timeout_reason": "",
        }

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--campaign-match-id",
            campaign_match_id,
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "integration",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["runtime_managed_decisions"] == 1
    assert payload["runs"][0]["runtime_managed_decisions"] == 1
    assert payload["runs"][0]["final_state"] == "reject"
    assert payload["runs"][0]["decision_conversion_state"] == "runtime_managed"
    assert payload["runs"][0]["decision_state"]["decision_owner"] == "child_native"
    assert payload["recent_runtime_managed_decisions"][0]["decision_state"]["decision_owner"] == "child_native"
    assert payload["recent_runtime_managed_decisions"][0]["decision_state"]["closeout_mode"] == "natural"
    assert payload["priority_family_yield_summary"]["family_summaries"]["project"]["observed_decisions"] == 1
    assert payload["priority_family_yield_summary"]["family_summaries"]["repository"]["observed_decisions"] == 1
    assert payload["priority_family_yield_summary"]["family_summaries"]["integration"]["observed_decisions"] == 1
    assert payload["priority_family_yield_summary"]["priority_families_without_signal"] == []


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_generated_failure_seed_completion():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="generated_failure_seed",
        current_task={
            "phase": "generated_failure_seed",
            "index": 10,
            "total": 10,
            "task_id": "api_contract_retrieval_task_discovered",
        },
        active_cycle_run={"generated_failure_seed_completed": True},
    )

    assert grace_key == "generated_failure_seed_completion"
    assert grace_seconds == 180.0


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_generated_failure_seed_raw_preview_phase():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="preview_candidate_eval",
        current_task={
            "phase": "preview_candidate_eval",
            "raw_phase": "generated_failure_seed",
            "index": 8,
            "total": 8,
            "task_id": "bridge_handoff_task",
        },
    )

    assert grace_key == "generated_failure_seed_completion"
    assert grace_seconds == 180.0


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_generated_failure_seed_start():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="generated_failure_seed",
        current_task={
            "phase": "generated_failure_seed",
            "index": 1,
            "total": 10,
            "task_id": "bridge_handoff_task",
        },
    )

    assert grace_key == "generated_failure_seed_active"
    assert grace_seconds == 240.0


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_generated_success_completion_from_final_task():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="generated_success",
        current_task={
            "phase": "generated_success",
            "index": 5,
            "total": 5,
            "task_id": "report_rollup_task",
        },
    )

    assert grace_key == "generated_success_completion"
    assert grace_seconds == 120.0


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_metrics_finalize_phase():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="metrics_finalize",
        current_task={},
    )

    assert grace_key == "metrics_finalize"
    assert grace_seconds == 120.0


def test_run_repeated_improvement_cycles_grants_runtime_grace_for_preview_completion_phase():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="preview_complete",
        current_task={},
    )

    assert grace_key == "preview_completion"
    assert grace_seconds == 120.0


def test_mid_cycle_intervention_signal_ignores_preview_only_reject_state():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    signal = module._mid_cycle_intervention_signal(
        {
            "selected_subsystem": "tolbert_model",
            "preview_state": "reject",
            "current_task": {
                "phase": "generated_failure_seed",
                "index": 1,
                "total": 16,
                "task_id": "semantic_open_world_20260411T225301327576Z_round_1_integration",
                "family": "integration",
            },
            "last_progress_phase": "generated_failure_seed",
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_decision_emitted"


def test_run_repeated_improvement_cycles_salvages_partial_productive_interrupt(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1001.0,
                    "line": "[eval:cycle-1] task 10/10 api_contract_retrieval_task_discovered family=discovered_task",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1002.0,
                    "line": "[eval:cycle-1] phase=generated_success total=10",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1003.0,
                    "line": "[eval:cycle-1] phase=generated_success task 10/10 integration_failover_drill_task_integration_recovery family=integration",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1004.0,
                    "line": "[eval:cycle-1] phase=generated_success complete total=10",
                }
            )
        raise KeyboardInterrupt("received signal SIGTERM")

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--priority-benchmark-family",
            "integration",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"
    assert payload["completed_runs"] == 1
    assert payload["productive_runs"] == 1
    assert payload["partial_productive_runs"] == 1
    assert payload["runtime_managed_decisions"] == 0
    assert payload["decision_conversion_summary"] == {
        "runtime_managed_runs": 0,
        "non_runtime_managed_runs": 0,
        "incomplete_runs": 0,
        "partial_productive_without_decision_runs": 1,
        "no_decision_runs": 0,
        "decision_runs": 0,
    }
    assert payload["inheritance_summary"]["runtime_managed_decisions"] == 0
    assert payload["trust_breadth_summary"]["distinct_family_gap"] == 0
    assert payload["partial_progress_summary"]["generated_success_completed_runs"] == 1
    assert payload["synthetic_interrupted_run_index"] == 1


def test_run_repeated_improvement_cycles_converts_all_failed_verifications_into_structured_child_decision(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds
        if on_event is not None:
            on_event({"event": "start", "pid": 123, "timestamp": 1000.0, "started_at": 999.0})
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1001.0,
                    "line": "[eval:cycle-1] phase=primary task 1/4 deployment_manifest_task family=project cognitive_stage=verification_result step=1 verification_passed=0",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1002.0,
                    "line": "[eval:cycle-1] phase=primary task 2/4 repo_sync_matrix_task family=repository cognitive_stage=verification_result step=1 verification_passed=0",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1003.0,
                    "line": "[eval:cycle-1] phase=generated_success total=4",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 123,
                    "timestamp": 1004.0,
                    "line": "[eval:cycle-1] phase=generated_success task 3/4 bridge_handoff_task_integration_recovery family=integration",
                }
            )
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_reconcile_incomplete_cycles", lambda **kwargs: [])
    monkeypatch.setattr(
        module,
        "_campaign_status_snapshot",
        lambda **kwargs: {
            "campaign_records_considered": 0,
            "decision_records_considered": 0,
            "campaign_cycle_ids": [],
            "priority_family_allocation_summary": {
                "priority_families": ["project", "repository"],
                "aggregated_task_counts": {},
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": ["project", "repository"],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": ["project", "repository"],
            "distinct_family_gap": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    payload = json.loads(Path(buffer.getvalue().strip()).read_text(encoding="utf-8"))
    assert payload["non_runtime_managed_decisions"] == 1
    assert payload["decision_conversion_summary"]["non_runtime_managed_runs"] == 1
    assert payload["recent_non_runtime_decisions"][0]["reason_code"] == "all_primary_verifications_failed"
    assert (
        payload["recent_non_runtime_decisions"][0]["decision_state"]["controller_intervention_reason_code"]
        == "all_primary_verifications_failed"
    )
    assert payload["runs"][0]["structured_child_decision"]["state"] == "reject"
    assert payload["runs"][0]["structured_child_decision"]["decision_state"]["decision_owner"] == "controller_runtime_manager"


def test_run_repeated_improvement_cycles_widens_search_after_no_retained_gain(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    seen_cmds: list[list[str]] = []

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        seen_cmds.append(list(cmd))
        protocol_match_id = cmd[cmd.index("--protocol-match-id") + 1]
        run_index = len(seen_cmds)
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=f"cycle:no_gain:{run_index}",
                state="reject",
                subsystem="skills",
                action="finalize_cycle",
                artifact_path="skills.json",
                artifact_kind="skill_set",
                reason="no retained gain",
                metrics_summary={
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": 0.7,
                    "protocol": "autonomous",
                    "protocol_match_id": protocol_match_id,
                },
            ),
        )
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "2",
            "--campaign-width",
            "1",
            "--variant-width",
            "1",
            "--task-limit",
            "5",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    assert len(seen_cmds) == 2
    assert "--adaptive-search" not in seen_cmds[0]
    assert "--adaptive-search" in seen_cmds[1]
    assert seen_cmds[0][seen_cmds[0].index("--campaign-width") + 1] == "1"
    assert seen_cmds[1][seen_cmds[1].index("--campaign-width") + 1] == "2"
    assert seen_cmds[0][seen_cmds[0].index("--variant-width") + 1] == "1"
    assert seen_cmds[1][seen_cmds[1].index("--variant-width") + 1] == "2"
    assert seen_cmds[0][seen_cmds[0].index("--task-limit") + 1] == "5"
    assert seen_cmds[1][seen_cmds[1].index("--task-limit") + 1] == "10"


def test_run_repeated_improvement_cycles_does_not_apply_next_cycle_reroute_after_failed_child(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cmd, cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id="cycle:observe:1",
                state="observe",
                subsystem="skills",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason="selection context",
                metrics_summary={
                    "priority_family_allocation_summary": {
                        "priority_benchmark_families": ["workflow", "project", "repository"],
                        "priority_benchmark_family_weights": {"workflow": 1.0, "project": 1.0, "repository": 1.0},
                        "planned_weight_shares": {"workflow": 0.333333, "project": 0.333333, "repository": 0.333333},
                        "actual_task_counts": {"workflow": 1, "project": 0, "repository": 0},
                        "actual_task_shares": {"workflow": 1.0, "project": 0.0, "repository": 0.0},
                        "actual_pass_rates": {"workflow": 1.0},
                        "actual_priority_tasks": 1,
                        "top_planned_family": "workflow",
                        "top_sampled_family": "workflow",
                    },
                    "protocol": "autonomous",
                    "protocol_match_id": "campaign:reroute",
                },
            ),
        )
        return {"returncode": -9, "stdout": "timeout\n", "stderr": "child failed", "timed_out": True, "timeout_reason": "child failed"}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "2",
            "--task-limit",
            "1",
            "--campaign-match-id",
            "campaign:reroute",
            "--priority-benchmark-family",
            "workflow",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    report_path = Path(buffer.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["completed_runs"] == 1
    assert payload["effective_task_limit"] == 1
    assert payload["effective_priority_benchmark_families"] == ["workflow", "project", "repository"]
    assert payload["runs"][0]["priority_family_rerouting"]["applied"] is True
    assert payload["runs"][0]["priority_family_rerouting"]["priority_benchmark_families"] == [
        "project",
        "repository",
        "workflow",
    ]
    assert payload["runs"][0]["priority_family_budget_rerouting"]["applied"] is True
    assert payload["runs"][0]["priority_family_budget_rerouting"]["next_task_limit"] == 3


def test_run_repeated_improvement_cycles_reroutes_unsampled_priority_families_between_cycles(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    seen_cmds: list[list[str]] = []

    def fake_run_and_stream(
        cmd,
        *,
        cwd,
        env=None,
        progress_label=None,
        heartbeat_interval_seconds=0.0,
        max_silence_seconds=0.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=0.0,
        on_event=None,
    ):
        del cwd, env, progress_label, heartbeat_interval_seconds, max_silence_seconds
        del max_runtime_seconds, max_progress_stall_seconds, on_event
        seen_cmds.append(list(cmd))
        protocol_match_id = cmd[cmd.index("--protocol-match-id") + 1]
        run_index = len(seen_cmds)
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=f"cycle:reroute:{run_index}",
                state="retain",
                subsystem="skills",
                action="finalize_cycle",
                artifact_path="skills.json",
                artifact_kind="skill_set",
                reason="retained with narrow sampled family coverage",
                metrics_summary={
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": 0.75,
                    "protocol": "autonomous",
                    "protocol_match_id": protocol_match_id,
                    "priority_family_allocation_summary": {
                        "priority_benchmark_family_weights": {
                            "workflow": 3.0,
                            "project": 2.0,
                            "repository": 1.0,
                        },
                        "actual_task_counts": {
                            "workflow": 1,
                            "project": 0,
                            "repository": 0,
                        },
                        "actual_pass_rates": {
                            "workflow": 1.0,
                            "project": 0.0,
                            "repository": 0.0,
                        },
                        "top_planned_family": "workflow",
                        "top_sampled_family": "workflow",
                    },
                },
            ),
        )
        return {"returncode": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_trust_breadth_summary",
        lambda config: {
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
            "missing_required_families": [],
            "family_breadth_min_distinct_task_roots": 2,
            "required_family_clean_task_root_counts": {},
            "missing_required_family_clean_task_root_breadth": [],
            "distinct_family_gap": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "2",
            "--task-limit",
            "1",
            "--priority-benchmark-family",
            "workflow",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family-weight",
            "workflow=3",
            "--priority-benchmark-family-weight",
            "project=2",
            "--priority-benchmark-family-weight",
            "repository=1",
        ],
    )

    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    module.main()

    assert len(seen_cmds) == 2
    first_cycle_families = [
        seen_cmds[0][index + 1]
        for index, token in enumerate(seen_cmds[0][:-1])
        if token == "--priority-benchmark-family"
    ]
    second_cycle_families = [
        seen_cmds[1][index + 1]
        for index, token in enumerate(seen_cmds[1][:-1])
        if token == "--priority-benchmark-family"
    ]
    second_cycle_weights = {
        seen_cmds[1][index + 1].split("=", 1)[0]: float(seen_cmds[1][index + 1].split("=", 1)[1])
        for index, token in enumerate(seen_cmds[1][:-1])
        if token == "--priority-benchmark-family-weight"
    }
    first_cycle_task_limit = int(seen_cmds[0][seen_cmds[0].index("--task-limit") + 1])
    second_cycle_task_limit = int(seen_cmds[1][seen_cmds[1].index("--task-limit") + 1])

    assert first_cycle_families == ["workflow", "project", "repository"]
    assert second_cycle_families == ["project", "repository", "workflow"]
    assert second_cycle_weights["project"] > second_cycle_weights["workflow"]
    assert second_cycle_weights["repository"] > second_cycle_weights["workflow"]
    assert first_cycle_task_limit == 1
    assert second_cycle_task_limit == 3

def test_finalize_cycle_writes_single_cycle_report(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "skills" / "command_skills.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
                {
                    "spec_version": "asi_v1",
                    "artifact_kind": "skill_set",
                    "lifecycle_state": "promoted",
                    "retention_gate": {"require_non_regression": True},
                        "skills": [
                            {
                                "skill_id": "skill:hello_task:primary",
                                "source_task_id": "hello_task",
                                "benchmark_family": "micro",
                                "quality": 0.9,
                                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                                "task_contract": {"expected_files": ["hello.txt"]},
                                "verifier": {"termination_reason": "success"},
                            }
                        ],
                }
            ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    run_results = [
        EvalMetrics(
            total=10,
            passed=7,
            average_steps=1.5,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return run_results.pop(0)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            skills_path=artifact_path,
            trajectories_root=tmp_path / "episodes",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "skills",
            "--cycle-id",
            "cycle:skills:report",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    report_paths = sorted(reports_dir.glob("cycle_report_*.json"))
    assert len(report_paths) == 1
    payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload["report_kind"] == "improvement_cycle_report"
    assert payload["cycle_id"] == "cycle:skills:report"
    assert payload["final_state"] == "retain"
    assert "proposal_selected_steps" in payload["baseline_metrics"]
    assert "novel_valid_command_rate" in payload["candidate_metrics"]
    assert "tolbert_primary_episodes" in payload["candidate_metrics"]
    assert payload["yield_summary"]["retained_cycles"] == 1
    assert payload["production_yield_summary"]["retained_cycles"] == 0
    assert payload["candidate_isolation_summary"]["paths_are_distinct"] is False
    assert payload["candidate_isolation_summary"]["runtime_managed_artifact_path"] is False
    assert records[-1]["artifact_kind"] == "improvement_cycle_report"


def test_finalize_cycle_rejects_when_candidate_loses_to_prior_retained_baseline(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "skills" / "command_skills.json"
    candidate_path = tmp_path / "improvement" / "candidates" / "skills" / "candidate.json"
    snapshot_path = tmp_path / "skills" / ".artifact_history" / "command_skills.cycle_skills_prior.post_retain.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_gate": {"require_non_regression": True},
                "skills": [
                    {
                        "skill_id": "live",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "quality": 0.9,
                        "procedure": {"commands": ["printf 'live\\n' > hello.txt"]},
                        "task_contract": {"expected_files": ["hello.txt"]},
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "promoted",
                "retention_gate": {"require_non_regression": True},
                "skills": [
                    {
                        "skill_id": "candidate",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "quality": 0.9,
                        "procedure": {"commands": ["printf 'candidate\\n' > hello.txt"]},
                        "task_contract": {"expected_files": ["hello.txt"]},
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_gate": {"require_non_regression": True},
                "skills": [
                    {
                        "skill_id": "prior",
                        "source_task_id": "hello_task",
                        "benchmark_family": "micro",
                        "quality": 0.9,
                        "procedure": {"commands": ["printf 'prior\\n' > hello.txt"]},
                        "task_contract": {"expected_files": ["hello.txt"]},
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:prior",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(active_path),
            artifact_kind="skill_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    observed_skill_ids: list[str] = []
    run_results = [
        EvalMetrics(
            total=10,
            passed=7,
            average_steps=1.5,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            generated_total=2,
            generated_passed=1,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        payload = json.loads(config.skills_path.read_text(encoding="utf-8"))
        observed_skill_ids.append(payload["skills"][0]["skill_id"])
        return run_results.pop(0)

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.cycle_runner,
        "atomic_write_json",
        lambda path, payload, *, config=None: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"),
        )[-1],
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            skills_path=active_path,
            trajectories_root=tmp_path / "episodes",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "skills",
            "--cycle-id",
            "cycle:skills:prior-check",
            "--artifact-path",
            str(candidate_path),
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report_paths = sorted(reports_dir.glob("cycle_report_*.json"))
    report_payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
    new_records = [record for record in records if record["cycle_id"] == "cycle:skills:prior-check"]

    assert observed_skill_ids == ["live", "candidate", "prior", "candidate"]
    assert [record["state"] for record in new_records] == ["evaluate", "evaluate", "reject", "record", "record"]
    assert new_records[1]["action"] == "compare_candidate_to_prior_retained"
    assert "failed prior retained comparison against cycle:skills:prior" in new_records[2]["reason"]
    assert round(new_records[2]["metrics_summary"]["prior_retained_pass_rate_delta"], 2) == -0.10
    assert "prior_retained_proposal_selected_steps_delta" in new_records[2]["metrics_summary"]
    assert report_payload["final_state"] == "reject"
    assert report_payload["prior_retained_comparison"]["baseline_cycle_id"] == "cycle:skills:prior"
    assert round(report_payload["prior_retained_comparison"]["pass_rate_delta"], 2) == -0.10
    assert json.loads(active_path.read_text(encoding="utf-8"))["skills"][0]["skill_id"] == "live"


def test_prior_retained_guard_reason_rejects_tooling_shared_repo_bundle_regression():
    module = _load_finalize_module()
    reason = module.cycle_runner.prior_retained_guard_reason(
        subsystem="tooling",
        gate={"require_shared_repo_bundle_coherence": True},
        comparison={
            "available": True,
            "baseline_metrics": {"pass_rate": 0.8, "average_steps": 1.0, "generated_pass_rate": 1.0},
            "current_metrics": {"pass_rate": 0.8, "average_steps": 1.0, "generated_pass_rate": 1.0},
            "evidence": {
                "shared_repo_bundle_summary": {
                    "baseline_shared_repo_candidate_count": 2,
                    "candidate_shared_repo_candidate_count": 2,
                    "candidate_bundle_coherence_delta": -1,
                    "shared_repo_incomplete_integrator_candidate_count_delta": 1,
                    "shared_repo_complete_candidate_count_delta": -1,
                }
            },
        },
    )

    assert reason == "candidate regressed shared-repo bundle coherence against the prior retained baseline"


def test_prior_retained_guard_reason_rejects_validation_family_generated_regression():
    module = _load_finalize_module()
    reason = module.cycle_runner.prior_retained_guard_reason(
        subsystem="transition_model",
        gate={"require_validation_family_generated_non_regression": True},
        comparison={
            "available": True,
            "baseline_metrics": {"pass_rate": 0.8, "average_steps": 1.0, "generated_pass_rate": 1.0},
            "current_metrics": {"pass_rate": 0.8, "average_steps": 1.0, "generated_pass_rate": 1.0},
            "evidence": {
                "validation_family_summary": {
                    "baseline_primary_task_count": 0,
                    "candidate_primary_task_count": 0,
                    "primary_pass_rate_delta": 0.0,
                    "baseline_generated_task_count": 2,
                    "candidate_generated_task_count": 2,
                    "generated_pass_rate_delta": -0.5,
                    "novel_valid_command_rate_delta": -1.0,
                    "baseline_world_feedback_step_count": 2,
                    "candidate_world_feedback_step_count": 2,
                    "world_feedback": {"progress_calibration_mae_gain": -0.08},
                }
            },
        },
    )

    assert reason == "candidate regressed validation-family generated pass rate against the prior retained baseline"


def test_write_cycle_report_includes_machine_readable_prior_retained_guard_code(tmp_path):
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
    )
    config.ensure_directories()
    config.improvement_cycles_path.write_text("", encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=config.improvement_cycles_path)

    report_path = cycle_runner._write_cycle_report(
        config=config,
        planner=planner,
        cycle_id="cycle:tooling:block-reason",
        subsystem="tooling",
        artifact_path=tmp_path / "tools" / "active.json",
        final_state="reject",
        final_reason=(
            "candidate failed prior retained comparison against cycle:tooling:prior: "
            "candidate regressed shared-repo bundle coherence against the prior retained baseline"
        ),
        artifact_update={
            "candidate_artifact_path": str(tmp_path / "candidates" / "tooling.json"),
            "active_artifact_path": str(tmp_path / "tools" / "active.json"),
            "artifact_kind": "tool_candidate_set",
            "artifact_lifecycle_state": "candidate",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "candidate_artifact_snapshot_path": "",
            "active_artifact_snapshot_path": "",
            "compatibility": {},
        },
        evidence={},
        baseline=EvalMetrics(total=10, passed=8, average_steps=1.0),
        candidate=EvalMetrics(total=10, passed=8, average_steps=1.0),
        phase_gate_report={"passed": True, "failures": []},
        prior_retained_comparison={"baseline_cycle_id": "cycle:tooling:prior"},
        prior_retained_guard_reason="candidate regressed shared-repo bundle coherence against the prior retained baseline",
        prior_retained_guard_reason_code="shared_repo_bundle_coherence_regressed",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_blocked"] is True
    assert payload["promotion_block_reason_code"] == "shared_repo_bundle_coherence_regressed"
    assert payload["prior_retained_guard_reason_code"] == "shared_repo_bundle_coherence_regressed"
    assert payload["prior_retained_guard_reason"] == (
        "candidate regressed shared-repo bundle coherence against the prior retained baseline"
    )


def test_write_cycle_report_includes_validation_family_prior_retained_guard_code(tmp_path):
    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        trajectories_root=tmp_path / "episodes",
    )
    config.ensure_directories()
    config.improvement_cycles_path.write_text("", encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=config.improvement_cycles_path)

    report_path = cycle_runner._write_cycle_report(
        config=config,
        planner=planner,
        cycle_id="cycle:transition_model:validation-block-reason",
        subsystem="transition_model",
        artifact_path=tmp_path / "transition_model" / "active.json",
        final_state="reject",
        final_reason=(
            "candidate failed prior retained comparison against cycle:transition_model:prior: "
            "candidate regressed validation-family generated pass rate against the prior retained baseline"
        ),
        artifact_update={
            "candidate_artifact_path": str(tmp_path / "candidates" / "transition_model.json"),
            "active_artifact_path": str(tmp_path / "transition_model" / "active.json"),
            "artifact_kind": "transition_model_policy_set",
            "artifact_lifecycle_state": "candidate",
            "artifact_sha256": "",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "candidate_artifact_snapshot_path": "",
            "active_artifact_snapshot_path": "",
            "compatibility": {},
        },
        evidence={},
        baseline=EvalMetrics(total=10, passed=8, average_steps=1.0),
        candidate=EvalMetrics(total=10, passed=8, average_steps=1.0),
        phase_gate_report={"passed": True, "failures": []},
        prior_retained_comparison={"baseline_cycle_id": "cycle:transition_model:prior"},
        prior_retained_guard_reason="candidate regressed validation-family generated pass rate against the prior retained baseline",
        prior_retained_guard_reason_code="validation_family_generated_pass_rate_regressed",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_blocked"] is True
    assert payload["promotion_block_reason_code"] == "validation_family_generated_pass_rate_regressed"
    assert payload["prior_retained_guard_reason_code"] == "validation_family_generated_pass_rate_regressed"
    assert payload["prior_retained_guard_reason"] == (
        "candidate regressed validation-family generated pass rate against the prior retained baseline"
    )


def test_compare_to_prior_retained_rebases_artifact_context_for_state_estimation(tmp_path, monkeypatch):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    snapshot_path = tmp_path / ".artifact_history" / "state_estimation.cycle_prior.post_retain.json"
    artifact_path = tmp_path / "state_estimation" / "state_estimation_proposals.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "state_estimation_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "no_progress_progress_epsilon": 0.0,
                    "min_state_change_score_for_progress": 1,
                    "regression_path_budget": 6,
                    "regression_severity_weight": 1.0,
                    "progress_recovery_credit": 1.0,
                },
                "latent_controls": {
                    "advancing_completion_ratio": 0.8,
                    "advancing_progress_delta": 0.2,
                    "improving_progress_delta": 0.0,
                    "regressing_progress_delta": -0.05,
                    "regressive_regression_count": 1,
                    "blocked_forbidden_count": 1,
                    "active_path_budget": 6,
                },
                "policy_controls": {
                    "regressive_path_match_bonus": 2,
                    "regressive_cleanup_bonus": 1,
                    "blocked_command_bonus": 1,
                    "advancing_path_match_bonus": 1,
                    "trusted_retrieval_path_bonus": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:state_estimation:prior",
            state="retain",
            subsystem="state_estimation",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="state_estimation_policy_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "state_estimation_policy_set",
        "lifecycle_state": "candidate",
        "controls": {
            "no_progress_progress_epsilon": 0.03,
            "min_state_change_score_for_progress": 2,
            "regression_path_budget": 8,
            "regression_severity_weight": 1.25,
            "progress_recovery_credit": 1.5,
        },
        "latent_controls": {
            "advancing_completion_ratio": 0.8,
            "advancing_progress_delta": 0.15,
            "improving_progress_delta": 0.0,
            "regressing_progress_delta": -0.05,
            "regressive_regression_count": 1,
            "blocked_forbidden_count": 1,
            "active_path_budget": 8,
        },
        "policy_controls": {
            "regressive_path_match_bonus": 3,
            "regressive_cleanup_bonus": 2,
            "blocked_command_bonus": 2,
            "advancing_path_match_bonus": 1,
            "trusted_retrieval_path_bonus": 1,
        },
        "generation_context": {
            "active_artifact_payload": {
                "artifact_kind": "state_estimation_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "no_progress_progress_epsilon": 0.03,
                    "min_state_change_score_for_progress": 2,
                    "regression_path_budget": 8,
                    "regression_severity_weight": 1.25,
                    "progress_recovery_credit": 1.5,
                },
                "latent_controls": {
                    "advancing_completion_ratio": 0.8,
                    "advancing_progress_delta": 0.15,
                    "improving_progress_delta": 0.0,
                    "regressing_progress_delta": -0.05,
                    "regressive_regression_count": 1,
                    "blocked_forbidden_count": 1,
                    "active_path_budget": 8,
                },
                "policy_controls": {
                    "regressive_path_match_bonus": 3,
                    "regressive_cleanup_bonus": 2,
                    "blocked_command_bonus": 2,
                    "advancing_path_match_bonus": 1,
                    "trusted_retrieval_path_bonus": 1,
                },
            }
        },
    }
    monkeypatch.setattr(
        module.cycle_runner,
        "evaluate_subsystem_metrics",
        lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0),
    )

    comparison = module.cycle_runner.compare_to_prior_retained(
        config=KernelConfig(
            improvement_cycles_path=cycles_path,
            state_estimation_proposals_path=artifact_path,
            trajectories_root=tmp_path / "episodes",
        ),
        planner=planner,
        subsystem="state_estimation",
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id="cycle:state_estimation:current",
        flags={},
        payload=payload,
    )

    assert comparison is not None
    assert comparison["available"] is True
    assert int(comparison["evidence"]["state_estimation_improvement_count"]) > 0


def test_compare_to_prior_retained_retries_without_tolbert_context_after_startup_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    snapshot_path = tmp_path / ".artifact_history" / "retrieval.cycle_prior.post_retain.json"
    artifact_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")

    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:prior",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="retrieval_policy_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    calls: list[dict[str, object]] = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del flags
        calls.append(
            {
                "use_tolbert_context": bool(config.use_tolbert_context),
                "subsystem": subsystem,
                "progress_label": progress_label,
            }
        )
        if len(calls) == 1:
            raise RuntimeError("TOLBERT service failed to become ready after 15.000 seconds.")
        return EvalMetrics(total=10, passed=8, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    progress_messages: list[str] = []
    comparison = module.cycle_runner.compare_to_prior_retained(
        config=KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=artifact_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        ),
        planner=planner,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id="cycle:retrieval:current",
        flags={},
        progress_label_prefix="cycle_test_retrieval_prior_retained",
        progress=progress_messages.append,
    )

    assert comparison is not None
    assert comparison["available"] is True
    assert calls == [
        {
            "use_tolbert_context": True,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_prior_retained_baseline",
        },
        {
            "use_tolbert_context": False,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_prior_retained_baseline",
        },
        {
            "use_tolbert_context": True,
            "subsystem": "retrieval",
            "progress_label": "cycle_test_retrieval_prior_retained_candidate",
        },
    ]
    assert (
        "finalize phase=preview_prior_retained_baseline_retry subsystem=retrieval "
        "reason=tolbert_startup_failure use_tolbert_context=0"
    ) in progress_messages


def test_finalize_cycle_uses_abstraction_compare_for_operators(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "operators" / "operator_classes.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
                {
                    "spec_version": "asi_v1",
                    "artifact_kind": "operator_class_set",
                    "lifecycle_state": "promoted",
                    "retention_gate": {"min_transfer_pass_rate_delta_abs": 0.05, "require_cross_task_support": True, "min_support": 2},
                    "operators": [
                        {
                            "operator_id": "operator:file_write:bounded",
                            "source_task_ids": ["hello_task", "math_task"],
                            "support": 2,
                            "benchmark_families": ["micro"],
                            "steps": ["printf 'hello agent kernel\\n' > hello.txt"],
                            "task_contract": {"expected_files": ["hello.txt"]},
                        }
                    ],
                }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    class Comparison:
        raw_skill_metrics = EvalMetrics(
            total=1,
            passed=0,
            average_steps=1.0,
            total_by_memory_source={"skill_transfer": 1},
            passed_by_memory_source={},
        )
        operator_metrics = EvalMetrics(
            total=1,
            passed=1,
            average_steps=1.0,
            total_by_memory_source={"operator": 1},
            passed_by_memory_source={"operator": 1},
        )

    seen = {"called": False}

    def fake_compare(*, config, **kwargs):
        del config, kwargs
        seen["called"] = True
        return Comparison()

    monkeypatch.setattr(module.cycle_runner, "compare_abstraction_transfer_modes", fake_compare)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            operator_classes_path=artifact_path,
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "operators",
            "--cycle-id",
            "cycle:operators:test",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert seen["called"] is True
    assert [record["state"] for record in records] == ["evaluate", "reject", "record", "record"]
    assert records[0]["metrics_summary"]["transfer_pass_rate_delta"] == 1.0


def test_select_improvement_target_reports_experiment_scores(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "select_improvement_target.py"
    spec = importlib.util.spec_from_file_location("select_improvement_target", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(
            total=10,
            passed=7,
            low_confidence_episodes=3,
            trusted_retrieval_steps=1,
            total_by_memory_source={},
            skill_selected_steps=6,
            retrieval_ranked_skill_steps=1,
            generated_total=10,
            generated_passed=5,
        ),
    )
    monkeypatch.setattr(module, "KernelConfig", lambda: KernelConfig())
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_improvement_target.py", "--all-candidates"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    output = stream.getvalue()
    assert "expected_gain=" in output
    assert "estimated_cost=" in output
    assert "score=" in output
    assert "history_retention_rate=" in output


def test_run_improvement_cycle_records_selected_variant(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            storage_backend="json",
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py"],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["state"] for record in records[:3]] == ["observe", "select", "generate"]
    assert records[0]["metrics_summary"]["selected_variant_id"]
    assert records[1]["metrics_summary"]["selected_variant"]["variant_id"]
    artifact_path = Path(str(records[2].get("candidate_artifact_path") or records[2]["artifact_path"]))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["experiment_variant"]["variant_id"]


def test_run_improvement_cycle_records_hard_observation_timeout_and_exits(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    monotonic_values = iter([100.0, 102.1])

    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module,
        "_run_observation_eval",
        lambda **kwargs: {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": "observation child exceeded max runtime of 2.0 seconds",
            "returncode": -9,
            "error": "",
            "last_progress_line": "[eval:auto-timeout] phase=generated_success_schedule",
            "last_progress_phase": "generated_success_schedule",
            "last_progress_task_id": "",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--max-observation-seconds",
            "2",
            "--progress-label",
            "auto-timeout",
        ],
    )

    try:
        module.main()
        raise AssertionError("expected SystemExit for hard observation timeout")
    except SystemExit as exc:
        assert "observation child exceeded max runtime of 2.0 seconds" in str(exc)

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["state"] for record in records] == ["observe"]
    assert records[0]["subsystem"] == "observation"
    observe = records[0]["metrics_summary"]
    assert observe["observation_elapsed_seconds"] == 2.1
    assert observe["observation_budget_seconds"] == 2.0
    assert observe["observation_budget_exceeded"] is True
    assert observe["observation_timed_out"] is True
    assert observe["observation_mode"] == "child_process"
    assert observe["observation_returncode"] == -9
    assert observe["observation_warning"] == "observation child exceeded max runtime of 2.0 seconds"
    assert observe["observation_last_progress_line"] == "[eval:auto-timeout] phase=generated_success_schedule"
    assert observe["observation_last_progress_phase"] == "generated_success_schedule"
    assert observe["observation_initial_last_progress_line"] == "[eval:auto-timeout] phase=generated_success_schedule"


def test_run_improvement_cycle_observation_eval_terminates_child_when_context_compile_subphase_budget_exceeded(
    tmp_path,
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl")
    terminated = {}
    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            step_elapsed_seconds = 0.4 if self.calls == 1 else 3.6
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:auto-subphase] phase=generated_success task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "tolbert_query",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                    }
                ),
                encoding="utf-8",
            )
            raise module.subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="auto-subphase",
        max_observation_seconds=10.0,
    )

    assert result["timed_out"] is True
    assert result["returncode"] == -9
    assert "context_compile subphase tolbert_query exceeded max runtime of 3.0 seconds" in result["timeout_reason"]
    assert result["current_task_timeout_budget_seconds"] == 3.0
    assert result["current_task_timeout_budget_source"] == "prestep_subphase:tolbert_query"
    assert result["partial_summary"]["current_task_id"] == "math_task"
    assert result["last_progress_benchmark_family"] == "bounded"
    assert "process" in terminated


def test_run_improvement_cycle_prestep_budget_stage_ignores_stale_context_subphase_after_stage_advance():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stage = module._current_task_prestep_budget_stage(
        {
            "current_task_id": "math_task",
            "current_task_step_stage": "llm_request",
            "current_task_step_subphase": "complete",
            "current_task_completed_steps": 0,
            "current_task_step_index": 1,
            "current_task_step_action": "",
        }
    )

    assert stage == ""


def test_run_improvement_cycle_subphase_budget_ignores_stale_context_subphase_after_stage_advance():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    budget = module._resolve_current_task_prestep_subphase_budget_seconds(
        partial_summary={
            "current_task_step_stage": "llm_request",
            "current_task_step_subphase": "complete",
            "current_task_step_budget_seconds": 0.0,
        }
    )

    assert budget == 0.0


def test_run_improvement_cycle_records_partial_timeout_details(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    monotonic_values = iter([100.0, 105.2])

    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module,
        "_run_observation_eval",
        lambda **kwargs: {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": (
                "observation current task stage context_compile "
                "subphase tolbert_query exceeded max runtime of 3.0 seconds"
            ),
            "returncode": -9,
            "error": "",
            "last_progress_line": "[eval:auto-timeout] phase=generated_success task 1/1 math_task family=bounded",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "math_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "generated_success",
                "current_task_id": "math_task",
                "current_task_step_stage": "context_compile",
                "current_task_step_subphase": "tolbert_query",
                "current_task_step_index": 1,
                "current_task_completed_steps": 0,
                "current_task_step_action": "",
                "current_task_step_elapsed_seconds": 3.6,
                "current_task_step_budget_seconds": 3.0,
                "generated_tasks_scheduled": 1,
                "completed_generated_tasks": 0,
                "scheduled_task_summaries": {
                    "math_task": {"benchmark_family": "bounded"},
                },
            },
            "current_task_timeout_budget_seconds": 3.0,
            "current_task_timeout_budget_source": "prestep_subphase:tolbert_query",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--max-observation-seconds",
            "5",
            "--progress-label",
            "auto-timeout",
        ],
    )

    try:
        module.main()
        raise AssertionError("expected SystemExit for hard observation timeout")
    except SystemExit as exc:
        assert "context_compile subphase tolbert_query exceeded max runtime of 3.0 seconds" in str(exc)

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_last_progress_benchmark_family"] == "bounded"
    assert observe["observation_current_task_timeout_budget_seconds"] == 3.0
    assert observe["observation_current_task_timeout_budget_source"] == "prestep_subphase:tolbert_query"
    assert observe["observation_partial_summary"]["current_task_id"] == "math_task"
    assert observe["observation_partial_phase"] == "generated_success"
    assert observe["observation_partial_current_task_step_stage"] == "context_compile"
    assert observe["observation_partial_current_task_step_subphase"] == "tolbert_query"
    assert observe["observation_partial_generated_tasks_scheduled"] == 1
    assert observe["observation_partial_current_task_benchmark_family"] == "bounded"


def test_run_improvement_cycle_observation_eval_kwargs_apply_bounded_priority_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig()
    args = module.argparse.Namespace(
        include_episode_memory=True,
        include_skill_memory=True,
        include_skill_transfer=False,
        include_operator_memory=False,
        include_tool_memory=True,
        include_verifier_memory=True,
        include_curriculum=False,
        include_failure_curriculum=False,
        task_limit=1,
        max_observation_seconds=15.0,
        priority_benchmark_family=[],
        priority_benchmark_family_weight=[],
    )

    flags = module._observation_eval_kwargs(config, args)

    assert flags["include_generated"] is False
    assert flags["include_failure_generated"] is False
    assert flags["priority_benchmark_families"] == ["bounded", "episode_memory", "tool_memory"]
    assert flags["prefer_low_cost_tasks"] is True
    assert flags["requested_task_limit"] == 1
    assert flags["observation_task_limit"] == 1


def test_run_improvement_cycle_observation_eval_kwargs_prefer_low_cost_for_explicit_priority(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig()
    args = module.argparse.Namespace(
        include_episode_memory=False,
        include_skill_memory=False,
        include_skill_transfer=False,
        include_operator_memory=False,
        include_tool_memory=False,
        include_verifier_memory=False,
        include_curriculum=False,
        include_failure_curriculum=False,
        task_limit=128,
        max_observation_seconds=15.0,
        priority_benchmark_family=["integration", "project", "repository"],
        priority_benchmark_family_weight=[],
    )

    flags = module._observation_eval_kwargs(config, args)

    assert flags["priority_benchmark_families"] == ["integration", "project", "repository"]
    assert flags["prefer_low_cost_tasks"] is True
    assert flags["restrict_to_priority_benchmark_families"] is True
    assert flags["include_generated"] is False
    assert flags["include_failure_generated"] is False


def test_run_improvement_cycle_observation_eval_kwargs_caps_observe_probe_but_preserves_campaign_task_limit(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig()
    config.observe_feature_probe_max_tasks = 8
    args = module.argparse.Namespace(
        include_episode_memory=False,
        include_skill_memory=False,
        include_skill_transfer=False,
        include_operator_memory=False,
        include_tool_memory=False,
        include_verifier_memory=False,
        include_curriculum=False,
        include_failure_curriculum=False,
        task_limit=32,
        max_observation_seconds=0.0,
        priority_benchmark_family=[],
        priority_benchmark_family_weight=[],
    )

    flags = module._observation_eval_kwargs(config, args)

    assert flags["requested_task_limit"] == 32
    assert flags["observation_task_limit"] == 8
    assert flags["task_limit"] == 8


def test_run_improvement_cycle_observation_eval_kwargs_preserve_explicit_bounded_curriculum_flags(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig()
    args = module.argparse.Namespace(
        include_episode_memory=False,
        include_skill_memory=False,
        include_skill_transfer=False,
        include_operator_memory=False,
        include_tool_memory=False,
        include_verifier_memory=False,
        include_curriculum=True,
        include_failure_curriculum=True,
        task_limit=1,
        max_observation_seconds=15.0,
        priority_benchmark_family=[],
        priority_benchmark_family_weight=[],
    )

    flags = module._observation_eval_kwargs(config, args)

    assert flags["include_generated"] is True
    assert flags["include_failure_generated"] is True


def test_run_improvement_cycle_retries_without_generated_curriculum_after_timeout(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    observation_calls = []
    monotonic_values = iter([100.0, 101.0, 101.6])

    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(dict(kwargs["eval_kwargs"]))
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": True,
                "timeout_reason": "observation child exceeded max runtime of 2.0 seconds",
                "returncode": -9,
                "error": "",
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=4, passed=3, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "retrieval_candidate.json"),
            "action": "generate_retrieval_update",
            "artifact_kind": "retrieval_policy_set",
            "candidate_artifact_path": tmp_path / "candidates" / "retrieval_candidate.json",
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--generate-only",
            "--max-observation-seconds",
            "2",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["include_generated"] is True
    assert observation_calls[0]["include_failure_generated"] is True
    assert observation_calls[1]["include_generated"] is False
    assert observation_calls[1]["include_failure_generated"] is False
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_generated_curriculum"] is True
    assert observe["observation_initial_timeout_reason"] == "observation child exceeded max runtime of 2.0 seconds"
    assert "recovered by retrying without generated curriculum" in observe["observation_warning"]


def test_run_improvement_cycle_retries_without_tolbert_context_after_startup_failure(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    observation_calls: list[dict[str, object]] = []

    def fake_run_observation_eval(*, config, eval_kwargs, progress_label, max_observation_seconds):
        del progress_label, max_observation_seconds
        observation_calls.append(
            {
                "use_tolbert_context": bool(config.use_tolbert_context),
                "include_generated": bool(eval_kwargs.get("include_generated", False)),
                "include_failure_generated": bool(eval_kwargs.get("include_failure_generated", False)),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "bounded_child",
                "metrics": None,
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 1,
                "error": "TOLBERT service failed to become ready after 15.000 seconds.",
                "partial_summary": {},
                "current_task_timeout_budget_seconds": 0.0,
                "current_task_timeout_budget_source": "none",
            }
        return {
            "mode": "bounded_child",
            "metrics": EvalMetrics(total=4, passed=4, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "partial_summary": {},
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "retrieval_candidate.json"),
            "action": "generate_retrieval_update",
            "artifact_kind": "retrieval_policy_set",
            "candidate_artifact_path": tmp_path / "candidates" / "retrieval_candidate.json",
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--generate-only",
            "--max-observation-seconds",
            "30",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["use_tolbert_context"] is True
    assert observation_calls[1]["use_tolbert_context"] is False
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_tolbert_context"] is True
    assert "retrying observation without tolbert context" in observe["observation_tolbert_retry_warning"]
    assert "recovered by retrying without tolbert context" in observe["observation_warning"]


def test_run_improvement_cycle_retries_without_tolbert_context_after_in_process_startup_failure(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    run_eval_calls: list[bool] = []

    def fake_run_eval(*, config, progress_label=None, **eval_kwargs):
        del progress_label, eval_kwargs
        run_eval_calls.append(bool(config.use_tolbert_context))
        if len(run_eval_calls) == 1:
            raise RuntimeError("TOLBERT service failed to become ready after 15.000 seconds.")
        return EvalMetrics(total=3, passed=3, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "retrieval_candidate.json"),
            "action": "generate_retrieval_update",
            "artifact_kind": "retrieval_policy_set",
            "candidate_artifact_path": tmp_path / "candidates" / "retrieval_candidate.json",
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--generate-only"])

    module.main()

    assert run_eval_calls == [True, False]
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_tolbert_context"] is True
    assert "recovered by retrying without tolbert context" in observe["observation_warning"]


def test_run_improvement_cycle_observation_eval_enables_existing_runtime_surface(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    skills_path = tmp_path / "skills" / "command_skills.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    skills_path.write_text(json.dumps({"artifact_kind": "skill_set", "skills": []}), encoding="utf-8")
    operator_path = tmp_path / "operators" / "operator_classes.json"
    operator_path.parent.mkdir(parents=True, exist_ok=True)
    operator_path.write_text(json.dumps({"artifact_kind": "operator_class_set", "operators": []}), encoding="utf-8")
    tool_path = tmp_path / "tools" / "tool_candidates.json"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(json.dumps({"artifact_kind": "tool_candidate_set", "candidates": []}), encoding="utf-8")
    benchmark_path = tmp_path / "benchmarks" / "benchmark_candidates.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps({"artifact_kind": "benchmark_candidate_set", "proposals": []}), encoding="utf-8")
    verifier_path = tmp_path / "verifiers" / "verifier_contracts.json"
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.write_text(json.dumps({"artifact_kind": "verifier_candidate_set", "proposals": []}), encoding="utf-8")
    seen_kwargs = {}

    def fake_run_eval(**kwargs):
        nonlocal seen_kwargs
        seen_kwargs = kwargs
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    def fake_generate_candidate_artifact(**kwargs):
        subsystem = str(getattr(kwargs.get("target"), "subsystem", "")).strip()
        expected_kinds = {
            "retrieval": "retrieval_policy_set",
            "tolbert_model": "tolbert_model_bundle",
            "benchmark": "benchmark_candidate_set",
            "verifier": "verifier_candidate_set",
            "policy": "prompt_proposal_set",
        }
        return {
            "artifact": str(tmp_path / "candidate.json"),
            "action": "generate_stub",
            "artifact_kind": expected_kinds.get(subsystem, "stub_artifact"),
            "candidate_artifact_path": tmp_path / "candidate.json",
            "prior_active_artifact_path": None,
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            trajectories_root=tmp_path / "episodes",
            skills_path=skills_path,
            operator_classes_path=operator_path,
            tool_candidates_path=tool_path,
            benchmark_candidates_path=benchmark_path,
            verifier_contracts_path=verifier_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--task-limit", "64", "--tolbert-device", "cuda:2"],
    )

    module.main()

    assert seen_kwargs["include_skill_memory"] is True
    assert seen_kwargs["include_skill_transfer"] is True
    assert seen_kwargs["include_operator_memory"] is True
    assert seen_kwargs["include_tool_memory"] is True
    assert seen_kwargs["include_verifier_memory"] is True
    assert seen_kwargs["include_benchmark_candidates"] is True
    assert seen_kwargs["include_verifier_candidates"] is True
    assert seen_kwargs["include_generated"] is True
    assert seen_kwargs["include_failure_generated"] is True
    assert seen_kwargs["task_limit"] == 64
    assert seen_kwargs["config"].tolbert_device == "cuda:2"


def test_run_improvement_cycle_generates_unique_cycle_ids(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()
    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe_cycle_ids = [record["cycle_id"] for record in records if record["state"] == "observe"]

    assert len(observe_cycle_ids) == 2
    assert observe_cycle_ids[0] != observe_cycle_ids[1]


def test_run_improvement_cycle_rejects_scoped_run_without_generate_only(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--scope-id", "runner-a"])

    try:
        module.main()
        raise AssertionError("expected SystemExit for scoped non-generate-only autonomous run")
    except SystemExit as exc:
        assert "--scope-id currently requires --generate-only" in str(exc)


def test_run_improvement_cycle_uses_scoped_config_for_generate_only_parallel_runs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "episodes",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
    )
    captured: dict[str, str] = {}

    monkeypatch.setattr(module, "KernelConfig", lambda: base_config)

    def fake_scoped_improvement_cycle_config(config, scope, **overrides):
        assert scope == "runner_a"
        scoped = KernelConfig(
            workspace_root=tmp_path / "workspace" / scope,
            trajectories_root=tmp_path / "episodes",
            improvement_cycles_path=tmp_path / "improvement" / f"cycles_{scope}.jsonl",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates" / scope,
            improvement_reports_dir=tmp_path / "improvement" / "reports" / scope,
            run_reports_dir=tmp_path / "reports" / scope,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        )
        for key, value in overrides.items():
            setattr(scoped, key, value)
        return scoped

    monkeypatch.setattr(module, "scoped_improvement_cycle_config", fake_scoped_improvement_cycle_config)

    def fake_run_eval(**kwargs):
        captured["workspace_root"] = str(kwargs["config"].workspace_root)
        captured["trajectories_root"] = str(kwargs["config"].trajectories_root)
        captured["improvement_cycles_path"] = str(kwargs["config"].improvement_cycles_path)
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "scoped_candidate.json"),
            "action": "generate_stub",
            "artifact_kind": "retrieval_policy_set",
            "candidate_artifact_path": tmp_path / "candidates" / "scoped_candidate.json",
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: ("retain", "finalized"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--generate-only", "--scope-id", "runner a"],
    )

    module.main()

    assert captured["workspace_root"].endswith("workspace/runner_a")
    assert captured["trajectories_root"].endswith("episodes")
    assert captured["improvement_cycles_path"].endswith("improvement/cycles_runner_a.jsonl")
    records = [
        json.loads(line)
        for line in (tmp_path / "improvement" / "cycles_runner_a.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["metrics_summary"]["scoped_run"] is True
    assert records[0]["metrics_summary"]["scope_id"] == "runner_a"


def test_run_improvement_cycle_executes_competitive_campaign(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment("retrieval", "r", 5, 0.05, 2, 0.10, {}),
            ImprovementExperiment("verifier", "v", 5, 0.049, 2, 0.095, {}),
            ImprovementExperiment("policy", "p", 3, 0.01, 2, 0.01, {}),
        ],
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(kwargs) or "retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe_records = [record for record in records if record["state"] == "observe"]
    select_records = [record for record in records if record["state"] == "select"]

    assert len(observe_records) == 2
    assert len(select_records) == 2
    assert all(record["metrics_summary"]["campaign_width"] == 2 for record in observe_records)
    assert finalized == ["retrieval", "verifier"]


def test_run_improvement_cycle_honors_excluded_subsystems_even_when_all_candidates_match(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    qwen_experiment = ImprovementExperiment("qwen_adapter", "qwen pressure", 5, 0.05, 2, 0.10, {})
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [qwen_experiment],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [qwen_experiment],
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(kwargs) or "retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--exclude-subsystem", "qwen_adapter", "--progress-label", "exclude-test"],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    module.main()

    assert finalized == []
    assert (
        "[cycle:exclude-test] campaign skipped reason=all_ranked_experiments_excluded "
        "excluded_subsystems=qwen_adapter"
    ) in stderr.getvalue()
    assert stdout.getvalue().strip() == ""
    if cycles_path.exists():
        records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert records == []


def test_run_improvement_cycle_falls_back_to_ranked_candidates_when_portfolio_is_fully_excluded(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    retrieval_experiment = ImprovementExperiment("retrieval", "retrieval pressure", 5, 0.05, 2, 0.10, {})
    verifier_experiment = ImprovementExperiment("verifier", "verifier pressure", 4, 0.04, 2, 0.09, {})
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [retrieval_experiment, verifier_experiment],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [retrieval_experiment],
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(kwargs) or "retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--exclude-subsystem", "retrieval", "--progress-label", "exclude-fallback-test"],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    module.main()

    assert [call["subsystem"] for call in finalized] == ["verifier"]
    strategy_candidate = finalized[0]["strategy_candidate"]
    assert strategy_candidate["strategy_candidate_id"] == "strategy:subsystem:verifier"
    assert strategy_candidate["strategy_candidate_kind"] == "subsystem_direct"
    assert (
        "[cycle:exclude-fallback-test] campaign fallback reason=portfolio_candidates_excluded "
        "selected_from_ranked_experiments=1 excluded_subsystems=retrieval"
    ) in stderr.getvalue()


def test_run_improvement_cycle_broadens_campaign_with_distinct_ranked_subsystems(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[str] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    retrieval_experiment = ImprovementExperiment("retrieval", "retrieval pressure", 5, 0.05, 2, 0.10, {})
    verifier_experiment = ImprovementExperiment("verifier", "verifier pressure", 4, 0.04, 2, 0.09, {})
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [retrieval_experiment, verifier_experiment],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=2, **kwargs: [retrieval_experiment],
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(kwargs["subsystem"]) or "retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--campaign-width", "2", "--progress-label", "diversity-fill-test"],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    module.main()

    assert finalized == ["retrieval", "verifier"]
    assert (
        "[cycle:diversity-fill-test] campaign broadened reason=subsystem_diversity_fill "
        "selected_from_ranked_experiments=2"
    ) in stderr.getvalue()


def test_tolbert_runtime_summary_infers_success_from_cognitive_tolbert_query():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._tolbert_runtime_summary_from_cognitive_progress(
        {
            "stage": "context_compile",
            "step_subphase": "tolbert_query",
        },
        active_cycle_run={
            "current_task": {"phase": "generated_success"},
        },
    )

    assert payload["configured_to_use_tolbert"] is True
    assert payload["stages_attempted"] == ["generated_success"]
    assert payload["successful_tolbert_stages"] == ["generated_success"]
    assert payload["outcome"] == "succeeded"


def test_append_partial_run_from_active_progress_preserves_identity_and_tolbert_summary():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runs: list[dict[str, object]] = []
    synthetic = module._append_partial_run_from_active_progress(
        runs=runs,
        active_cycle_run={
            "index": 2,
            "selected_subsystem": "verifier",
            "last_progress_phase": "variant_search",
            "current_task": {"phase": "generated_success", "family": "repository"},
            "sampled_families_from_progress": ["repository"],
            "observe_completed": True,
            "tolbert_runtime_summary": {
                "configured_to_use_tolbert": True,
                "successful_tolbert_stages": ["generated_success"],
            },
        },
        priority_benchmark_families=["repository"],
        returncode=0,
        stderr="",
        interrupted=True,
    )

    assert synthetic is not None
    assert synthetic["selected_subsystem"] == "verifier"
    assert synthetic["last_progress_phase"] == "variant_search"
    assert synthetic["current_task"]["phase"] == "generated_success"
    assert synthetic["tolbert_runtime_summary"]["outcome"] == "succeeded"
    assert synthetic["tolbert_runtime_summary"]["successful_tolbert_stages"] == ["generated_success"]


def test_parse_progress_line_promotes_variant_generate_and_clears_stale_task():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed = module._parse_progress_fields(
        "[cycle:cycle-1] variant generate start subsystem=tolbert_model variant=recovery_alignment rank=1/2 expected_gain=0.0300 score=0.0075"
    )
    assert parsed["last_progress_phase"] == "variant_generate"
    assert parsed["selected_subsystem"] == "tolbert_model"
    assert parsed["last_candidate_variant"] == "recovery_alignment"
    assert parsed["current_task"] == {}

    heartbeat = module._parse_progress_fields(
        "[cycle:cycle-1] variant generate heartbeat subsystem=tolbert_model variant=recovery_alignment stage=tolbert_pipeline_heartbeat pending=tolbert_finetune_pipeline"
    )
    assert heartbeat["last_progress_phase"] == "variant_generate"
    assert heartbeat["variant_generate_stage"] == "tolbert_pipeline_heartbeat"
    assert heartbeat["current_task"] == {}


def test_run_improvement_cycle_forwards_abstraction_flags(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    observed_kwargs: list[dict[str, object]] = []

    def fake_run_eval(**kwargs):
        observed_kwargs.append(kwargs)
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--include-skill-transfer",
            "--include-operator-memory",
        ],
    )

    module.main()

    assert observed_kwargs
    assert observed_kwargs[0]["include_skill_transfer"] is True
    assert observed_kwargs[0]["include_operator_memory"] is True


def test_run_improvement_cycle_uses_adaptive_search_budgets(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[str] = []

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment("policy", "p", 3, 0.02, 2, 0.10, {}),
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant("policy", "verifier_alignment", "a", 0.02, 2, 0.0200, {"focus": "verifier_alignment"}),
            ImprovementVariant("policy", "retrieval_caution", "b", 0.019, 2, 0.0189, {"focus": "retrieval_caution"}),
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "recommend_campaign_budget",
        lambda self, metrics, max_width=2: ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=max_width,
            strategy="adaptive_history",
            top_score=0.1,
            selected_ids=["policy"],
            reasons=["top subsystem only"],
        ),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "recommend_variant_budget",
        lambda self, experiment, metrics, max_width=2: ImprovementSearchBudget(
            scope="variant",
            width=2,
            max_width=max_width,
            strategy="adaptive_history",
            top_score=0.02,
            selected_ids=["verifier_alignment", "retrieval_caution"],
            reasons=["close sibling variants"],
        ),
    )
    monkeypatch.setattr(
        module,
        "preview_candidate_retention",
        lambda **kwargs: {
            "state": "retain" if "verifier_alignment" in str(kwargs["artifact_path"]) else "reject",
            "reason": "preview",
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
        },
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: (finalized.append(str(kwargs["artifact_path"])) or "retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--adaptive-search", "--variant-width", "3"],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe_records = [record for record in records if record["state"] == "observe"]
    generate_records = [record for record in records if record["state"] == "generate"]

    assert observe_records[0]["metrics_summary"]["search_strategy"] == "adaptive_history"
    assert observe_records[0]["metrics_summary"]["variant_budget"]["width"] == 2
    assert len(generate_records) == 2
    assert any("verifier_alignment" in path for path in finalized)


def test_run_improvement_cycle_uses_portfolio_campaign_selection(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    selected_subsystems: list[str] = []

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
            ImprovementExperiment("policy", "p", 5, 0.039, 2, 0.095, {}),
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [
            ImprovementExperiment(
                "policy",
                "p",
                5,
                0.039,
                2,
                0.095,
                {
                    "portfolio": {
                        "adjusted_score": 0.105,
                        "campaign_breadth_pressure": 0.4,
                        "recent_activity": {"recent_incomplete_cycles": 1},
                        "reasons": ["exploration_bonus=0.0100", "campaign_breadth_pressure=0.4000"],
                    }
                },
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "choose_variant",
        lambda self, experiment, metrics: ImprovementVariant(
            experiment.subsystem,
            "verifier_alignment",
            "bias decisions toward verifier-compatible checks",
            0.012,
            2,
            0.006,
            {"focus": "verifier_alignment"},
        ),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: (selected_subsystems.append(kwargs["subsystem"]) or "retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    select_record = next(record for record in records if record["state"] == "select")
    observe_record = next(record for record in records if record["state"] == "observe")

    assert selected_subsystems == ["policy"]
    assert observe_record["metrics_summary"]["campaign_strategy"] == "portfolio_history"
    assert observe_record["metrics_summary"]["campaign_breadth_pressure"] == 0.4
    assert "selected_variant_breadth_pressure" in observe_record["metrics_summary"]
    assert select_record["metrics_summary"]["selected_campaign"][0]["portfolio"]["adjusted_score"] == 0.105
    assert select_record["metrics_summary"]["campaign_breadth_pressure"] == 0.4


def test_run_improvement_cycle_defaults_to_observing_present_abstraction_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    skills_path = tmp_path / "skills" / "command_skills.json"
    operators_path = tmp_path / "operators" / "operator_classes.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    operators_path.parent.mkdir(parents=True, exist_ok=True)
    skills_path.write_text('{"artifact_kind":"skill_set","skills":[]}', encoding="utf-8")
    operators_path.write_text('{"artifact_kind":"operator_class_set","operators":[]}', encoding="utf-8")
    observed_kwargs: list[dict[str, object]] = []

    def fake_run_eval(**kwargs):
        observed_kwargs.append(kwargs)
        return EvalMetrics(total=10, passed=9, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            skills_path=skills_path,
            operator_classes_path=operators_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py"],
    )

    module.main()

    assert observed_kwargs
    assert observed_kwargs[0]["include_skill_transfer"] is True
    assert observed_kwargs[0]["include_operator_memory"] is True


def test_run_improvement_cycle_applies_variant_generation_controls(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "choose_variant",
        lambda self, experiment, metrics: ImprovementVariant(
            subsystem="policy",
            variant_id="verifier_alignment",
            description="bias decisions toward verifier-compatible artifact checks",
            expected_gain=0.012,
            estimated_cost=2,
            score=0.01,
            controls={"focus": "verifier_alignment"},
        ),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py"],
    )

    module.main()

    payload, _ = _generated_candidate_payload(cycles_path)
    assert payload["generation_focus"] == "verifier_alignment"
    assert payload["experiment_variant"]["variant_id"] == "verifier_alignment"
    assert all(proposal["area"] in {"system", "decision"} for proposal in payload["proposals"])


def test_run_improvement_cycle_auto_finalizes_by_default(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    seen: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: seen.append(kwargs) or ("retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    assert seen
    assert seen[0]["cycle_id"]
    assert str(seen[0]["artifact_path"])


def test_run_improvement_cycle_reconciles_prior_incomplete_autonomous_cycles(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stale",
            state="observe",
            subsystem="retrieval",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stale",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous", "selected_variant_id": "routing_depth"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stale",
            state="generate",
            subsystem="retrieval",
            action="propose_retrieval_update",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous", "selected_variant": {"variant_id": "routing_depth"}},
            candidate_artifact_path="candidates/retrieval.json",
            active_artifact_path="retrieval.json",
        ),
    )
    artifact_path = tmp_path / "candidates" / "policy.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps({"artifact_kind": "prompt_proposal_set", "lifecycle_state": "candidate", "proposals": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [ImprovementExperiment("policy", "policy gap", 4, 0.02, 2, 0.1, {})],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(artifact_path),
            "action": "propose_prompt_update",
            "artifact_kind": "prompt_proposal_set",
            "candidate_artifact_path": artifact_path,
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stale = [record for record in records if record["cycle_id"] == "cycle:retrieval:stale"]
    assert [record["state"] for record in stale[-2:]] == ["incomplete", "record"]
    assert "incomplete autonomous cycle" in stale[-2]["reason"]


def test_run_improvement_cycle_records_reject_when_finalize_cycle_raises(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    artifact_path = tmp_path / "candidates" / "policy.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps({"artifact_kind": "prompt_proposal_set", "lifecycle_state": "candidate", "proposals": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [ImprovementExperiment("policy", "policy gap", 4, 0.02, 2, 0.1, {})],
    )
    monkeypatch.setattr(
        module,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(artifact_path),
            "action": "propose_prompt_update",
            "artifact_kind": "prompt_proposal_set",
            "candidate_artifact_path": artifact_path,
            "prior_active_artifact_path": None,
        },
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    decision = next(record for record in reversed(records) if record["state"] == "reject")
    followup = records[records.index(decision) + 1]
    assert decision["action"] == "finalize_cycle"
    assert decision["subsystem"] == "policy"
    assert "finalize_cycle_exception:RuntimeError:boom" == decision["reason"]
    assert followup["state"] == "record"
    assert followup["action"] == "persist_retention_outcome"


def test_run_improvement_cycle_generates_retrieval_policy_artifact(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0, low_confidence_episodes=3, trusted_retrieval_steps=1),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "choose_variant",
        lambda self, experiment, metrics: ImprovementVariant(
            subsystem="retrieval",
            variant_id="confidence_gating",
            description="tighten trust and confidence thresholds",
            expected_gain=0.03,
            estimated_cost=2,
            score=0.01,
            controls={"focus": "confidence"},
        ),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    payload, records = _generated_candidate_payload(cycles_path)
    assert payload["artifact_kind"] == "retrieval_policy_set"
    assert payload["generation_focus"] == "confidence"
    assert payload["asset_strategy"] == "confidence_hardening"
    assert payload["asset_rebuild_plan"]["rebuild_required"] is True
    assert any(record["state"] == "generate" and record["candidate_artifact_path"] for record in records)


def test_finalize_cycle_materializes_retrieval_asset_bundle_on_retain(tmp_path, monkeypatch):
    module = _load_finalize_module()
    artifact_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "proposed",
                "asset_strategy": "balanced_rebuild",
                "asset_controls": {"include_tool_candidate_spans": False},
                "asset_rebuild_plan": {"rebuild_required": True},
                "overrides": {"tolbert_branch_results": 2},
                "proposals": [
                    {
                        "proposal_id": "retrieval:confidence_gating",
                        "overrides": {"tolbert_branch_results": 2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    run_results = [
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            trusted_retrieval_steps=2,
            retrieval_influenced_steps=2,
            retrieval_selected_steps=1,
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.2,
            trusted_retrieval_steps=5,
            retrieval_influenced_steps=5,
            retrieval_selected_steps=2,
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return run_results.pop(0)

    def fake_materialize_bundle(*, repo_root, config, cycle_id, **kwargs):
        del repo_root, kwargs
        config.retrieval_asset_bundle_path.write_text(
            json.dumps(
                {
                    "artifact_kind": "tolbert_retrieval_asset_bundle",
                    "lifecycle_state": "retained",
                    "cycle_id": cycle_id,
                }
            ),
            encoding="utf-8",
        )
        return config.retrieval_asset_bundle_path

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.cycle_runner,
        "evaluate_artifact_retention",
        lambda subsystem, baseline_metrics, candidate_metrics, **kwargs: ("retain", "retain for bundle materialization"),
    )
    monkeypatch.setattr(module.cycle_runner, "materialize_retained_retrieval_asset_bundle", fake_materialize_bundle)
    monkeypatch.setattr(module.cycle_runner, "_write_cycle_report", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=artifact_path,
            retrieval_asset_bundle_path=bundle_manifest_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_improvement_cycle.py",
            "--subsystem",
            "retrieval",
            "--cycle-id",
            "cycle:retrieval:test",
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(record["action"] == "materialize_retrieval_asset_bundle" for record in records)
    assert bundle_manifest_path.exists()
    payload = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "tolbert_retrieval_asset_bundle"
    assert payload["cycle_id"] == "cycle:retrieval:test"


def test_run_improvement_cycle_selects_best_previewed_sibling_variant(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    finalized: list[str] = []
    preview_calls: list[str] = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0, low_confidence_episodes=2, trusted_retrieval_steps=2),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="retrieval_caution",
                description="make low-confidence retrieval less binding",
                expected_gain=0.015,
                estimated_cost=2,
                score=0.02,
                controls={"focus": "retrieval_caution"},
            ),
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            ),
        ],
    )

    def fake_preview(**kwargs):
        artifact_path = str(kwargs["artifact_path"])
        preview_calls.append(artifact_path)
        if "retrieval_caution" in artifact_path:
            return {
                "state": "reject",
                "reason": "preview rejected",
                "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
                "candidate": EvalMetrics(total=10, passed=8, average_steps=1.5),
            }
        return {
            "state": "retain",
            "reason": "preview retained",
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
        }

    monkeypatch.setattr(module, "preview_candidate_retention", fake_preview)
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(str(kwargs["artifact_path"])) or "retain", "finalized"),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--variant-width", "2"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generate_records = [record for record in records if record["state"] == "generate"]
    preview_records = [record for record in records if record["action"] == "preview_sibling_candidate"]
    sibling_select = [record for record in records if record["action"] == "choose_sibling_variant"]

    assert len(generate_records) == 2
    assert len(preview_records) == 2
    assert len(sibling_select) == 1
    assert sibling_select[0]["metrics_summary"]["selected_variant_id"] == "verifier_alignment"
    assert any("verifier_alignment" in path for path in finalized)
    assert len(preview_calls) == 2


def test_run_improvement_cycle_emits_preview_progress_details(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [ImprovementExperiment("policy", "policy gap", 4, 0.02, 2, 0.1, {})],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [ImprovementExperiment("policy", "policy gap", 4, 0.02, 2, 0.1, {})],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                "policy",
                "verifier_alignment",
                "bias verifier-compatible checks",
                0.012,
                2,
                0.02,
                {"focus": "verifier_alignment"},
            ),
            ImprovementVariant(
                "policy",
                "retrieval_caution",
                "make retrieval less binding",
                0.011,
                2,
                0.01,
                {"focus": "retrieval_caution"},
            ),
        ],
    )
    monkeypatch.setattr(module, "_cycle_id_for_experiment", lambda subsystem: f"cycle:{subsystem}:test")

    def fake_generate_candidate_artifact(**kwargs):
        variant = kwargs["variant"]
        path = tmp_path / "candidates" / f"{variant.variant_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifact_kind": "prompt_proposal_set"}), encoding="utf-8")
        return {
            "artifact": str(path),
            "action": "generate_stub",
            "artifact_kind": "prompt_proposal_set",
            "candidate_artifact_path": path,
            "prior_active_artifact_path": None,
        }

    preview_calls: list[dict[str, object]] = []

    def fake_preview(**kwargs):
        preview_calls.append(
            {
                "artifact_path": str(kwargs["artifact_path"]),
                "progress_label_prefix": kwargs.get("progress_label_prefix"),
                "include_curriculum": kwargs.get("include_curriculum"),
                "include_failure_curriculum": kwargs.get("include_failure_curriculum"),
                "priority_benchmark_families": kwargs.get("priority_benchmark_families"),
                "restrict_to_priority_benchmark_families": kwargs.get("restrict_to_priority_benchmark_families"),
            }
        )
        return {
            "state": "retain" if "verifier_alignment" in str(kwargs["artifact_path"]) else "reject",
            "reason": "preview retained",
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "phase_gate_report": {"passed": True, "failures": []},
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(module, "preview_candidate_retention", fake_preview)
    finalize_calls: list[dict[str, object]] = []

    def fake_finalize_cycle(**kwargs):
        finalize_calls.append(kwargs)
        return ("retain", "finalized")

    monkeypatch.setattr(module, "finalize_cycle", fake_finalize_cycle)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_improvement_cycle.py",
            "--variant-width",
            "2",
            "--task-limit",
            "17",
            "--progress-label",
            "test-progress",
            "--priority-benchmark-family",
            "integration",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "repository",
        ],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    module.main()

    progress_output = stderr.getvalue()
    assert "[cycle:test-progress] variant_search start subsystem=policy selected_variants=2 variant_ids=verifier_alignment,retrieval_caution" in progress_output
    assert "[cycle:test-progress] variant generate start subsystem=policy variant=verifier_alignment rank=1/2" in progress_output
    assert "[cycle:test-progress] variant generate complete subsystem=policy variant=verifier_alignment artifact=" in progress_output
    assert "[cycle:test-progress] preview start subsystem=policy variant=verifier_alignment rank=1/2 task_limit=17" in progress_output
    assert "[cycle:test-progress] preview complete subsystem=policy variant=verifier_alignment state=retain baseline_pass_rate=0.8000 candidate_pass_rate=0.9000 phase_gate_passed=True" in progress_output
    assert "[cycle:test-progress] variant_search selected subsystem=policy variant=verifier_alignment from=2" in progress_output
    assert "[cycle:test-progress] finalize start subsystem=policy variant=verifier_alignment artifact=" in progress_output
    assert preview_calls[0]["progress_label_prefix"] == "cycle:policy:test_policy_verifier_alignment_preview"
    assert preview_calls[1]["progress_label_prefix"] == "cycle:policy:test_policy_retrieval_caution_preview"
    assert preview_calls[0]["include_curriculum"] is True
    assert preview_calls[0]["include_failure_curriculum"] is True
    assert preview_calls[0]["priority_benchmark_families"] == ["integration", "project", "repository"]
    assert preview_calls[0]["restrict_to_priority_benchmark_families"] is True
    assert isinstance(finalize_calls[0]["preview"], dict)
    assert finalize_calls[0]["preview"]["state"] == "retain"
    assert finalize_calls[0]["priority_benchmark_families"] == ["integration", "project", "repository"]
    assert finalize_calls[0]["restrict_to_priority_benchmark_families"] is True
    assert finalize_calls[0]["include_curriculum"] is True
    assert finalize_calls[0]["include_failure_curriculum"] is True


def test_run_improvement_cycle_records_campaign_budget_from_selected_campaign(tmp_path, monkeypatch):
    module = _load_run_improvement_cycle_module()

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    retrieval = ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {})
    policy = ImprovementExperiment("policy", "p", 5, 0.039, 2, 0.095, {"portfolio": {"reasons": ["observe bonus"]}})

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(module.ImprovementPlanner, "rank_experiments", lambda self, metrics: [retrieval, policy])
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [policy],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "recommend_campaign_budget",
        lambda self, metrics, max_width=1: ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=max_width,
            strategy="adaptive_history",
            top_score=0.10,
            selected_ids=["retrieval"],
            reasons=["top subsystem retrieval score=0.1000"],
        ),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant("policy", "verifier_alignment", "a", 0.02, 2, 0.0200, {"focus": "verifier_alignment"}),
        ],
    )
    monkeypatch.setattr(module, "_cycle_id_for_experiment", lambda subsystem: f"cycle:{subsystem}:test")
    monkeypatch.setattr(module, "_candidate_preflight_compatibility", lambda **kwargs: {"compatible": True, "violations": []})

    def fake_generate_candidate_artifact(**kwargs):
        path = tmp_path / "candidates" / "policy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "spec_version": "asi_v1",
                    "artifact_kind": "prompt_proposal_set",
                    "lifecycle_state": "candidate",
                    "retention_gate": {"min_pass_rate_delta_abs": 0.01},
                    "proposals": [{"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}],
                }
            ),
            encoding="utf-8",
        )
        return {
            "artifact": str(path),
            "action": "propose_policy_update",
            "artifact_kind": "prompt_proposal_set",
            "candidate_artifact_path": path,
            "prior_active_artifact_path": None,
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--progress-label", "budget-test"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    choose_record = next(record for record in records if record["action"] == "choose_target")

    assert choose_record["metrics_summary"]["campaign_budget"]["width"] == 1
    assert choose_record["metrics_summary"]["campaign_budget"]["selected_ids"] == ["policy"]
    assert choose_record["metrics_summary"]["campaign_budget"]["reasons"][0].startswith("selected policy")


def test_run_improvement_cycle_rejects_incompatible_verifier_candidate_before_preview(tmp_path, monkeypatch):
    module = _load_run_improvement_cycle_module()

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    preview_calls: list[dict[str, object]] = []
    finalize_calls: list[dict[str, object]] = []
    verifier = ImprovementExperiment("verifier", "verifier gap", 4, 0.03, 2, 0.10, {})

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(module.ImprovementPlanner, "rank_experiments", lambda self, metrics: [verifier])
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [verifier],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "recommend_campaign_budget",
        lambda self, metrics, max_width=1: ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=max_width,
            strategy="adaptive_history",
            top_score=0.1,
            selected_ids=["verifier"],
            reasons=["top subsystem only"],
        ),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant("verifier", "strict_contract_growth", "strict", 0.02, 2, 0.02, {}),
        ],
    )
    monkeypatch.setattr(module, "_cycle_id_for_experiment", lambda subsystem: f"cycle:{subsystem}:test")

    def fake_generate_candidate_artifact(**kwargs):
        path = tmp_path / "candidates" / "verifier.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "spec_version": "asi_v1",
                    "artifact_kind": "verifier_candidate_set",
                    "lifecycle_state": "proposed",
                    "retention_gate": {"min_discrimination_gain": 0.02},
                    "proposals": [
                        {
                            "proposal_id": "verifier:bad:strict",
                            "source_task_id": "api_contract_task_tool_recovery",
                            "benchmark_family": "micro",
                            "contract": {
                                "expected_files": ["result.txt"],
                                "forbidden_files": [],
                                "expected_file_contents": {"result.txt": "42\n"},
                                "forbidden_output_substrings": ["3"],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "artifact": str(path),
            "action": "propose_verifier_update",
            "artifact_kind": "verifier_candidate_set",
            "candidate_artifact_path": path,
            "prior_active_artifact_path": None,
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(module, "preview_candidate_retention", lambda **kwargs: preview_calls.append(kwargs) or {})
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: finalize_calls.append(kwargs) or ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--progress-label", "compat-test"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reject_record = next(record for record in records if record["action"] == "preflight_candidate_compatibility")

    assert preview_calls == []
    assert finalize_calls == []
    assert reject_record["state"] == "reject"
    assert "unknown source task for verifier proposal" in reject_record["reason"]
    assert reject_record["compatibility"]["compatible"] is False


def test_run_improvement_cycle_falls_back_to_default_variant_after_planning_timeout(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"

    tooling_experiment = ImprovementExperiment(
        "tooling",
        "tooling gap",
        4,
        0.02,
        2,
        0.1,
        {"command_failure_count": 3},
    )

    monkeypatch.setattr(module, "run_eval", lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0))
    monkeypatch.setattr(module.ImprovementPlanner, "rank_experiments", lambda self, metrics: [tooling_experiment])
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates=1, **kwargs: [tooling_experiment],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "recommend_campaign_budget",
        lambda self, metrics, max_width=1: ImprovementSearchBudget(
            scope="campaign",
            width=1,
            max_width=max_width,
            strategy="adaptive_history",
            top_score=0.1,
            selected_ids=["tooling"],
            reasons=["top subsystem only"],
        ),
    )
    monkeypatch.setattr(
        module,
        "_call_with_planning_timeout",
        lambda stage, callback, timeout_seconds=None: (_ for _ in ()).throw(
            module._PlanningStageTimeout(stage, 30.0)
        )
        if stage == "variant_budget"
        else callback(),
    )
    monkeypatch.setattr(module, "_cycle_id_for_experiment", lambda subsystem: f"cycle:{subsystem}:test")

    def fake_generate_candidate_artifact(**kwargs):
        variant = kwargs["variant"]
        path = tmp_path / "candidates" / f"{variant.variant_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifact_kind": "tool_candidate_set"}), encoding="utf-8")
        return {
            "artifact": str(path),
            "action": "generate_stub",
            "artifact_kind": "tool_candidate_set",
            "candidate_artifact_path": path,
            "prior_active_artifact_path": None,
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_improvement_cycle.py", "--variant-width", "1", "--progress-label", "timeout-fallback"],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    module.main()

    progress_output = stderr.getvalue()
    assert (
        "[cycle:timeout-fallback] variant_search degraded subsystem=tooling "
        "reason=planning_timeout stage=variant_budget timeout_seconds=30.0"
    ) in progress_output
    assert (
        "[cycle:timeout-fallback] variant_search start subsystem=tooling "
        "selected_variants=1 variant_ids=procedure_promotion"
    ) in progress_output
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe_record = next(record for record in records if record["state"] == "observe")
    assert observe_record["metrics_summary"]["variant_budget"]["strategy"] == "timeout_fallback"
    assert observe_record["metrics_summary"]["variant_planning_info"]["fallback_used"] is True
    assert observe_record["metrics_summary"]["variant_planning_info"]["fallback_stage"] == "variant_budget"


def test_preview_candidate_retention_uses_no_write_scoped_configs_and_task_limit(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "prompts" / "prompt_proposals.json"
    candidate_path = tmp_path / "candidates" / "prompt_candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "prompt_proposal_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.0,
            "max_step_regression": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
            "required_confirmation_runs": 1,
        },
        "proposals": [{"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}],
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=active_path,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    config.ensure_directories()
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del progress_label
        seen_calls.append(
            {
                "subsystem": subsystem,
                "persist_episode_memory": config.persist_episode_memory,
                "trajectories_root": config.trajectories_root,
                "task_limit": flags.get("task_limit"),
                "run_checkpoints_dir": config.run_checkpoints_dir,
                "workspace_snapshot_root": config.unattended_workspace_snapshot_root,
            }
        )
        if len(seen_calls) == 1:
            return EvalMetrics(
                total=10,
                passed=8,
                average_steps=1.5,
                generated_total=4,
                generated_passed=2,
                generated_by_kind={"failure_recovery": 2},
                generated_passed_by_kind={"failure_recovery": 1},
            )
        return EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.2,
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        )

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "compare_to_prior_retained", lambda **kwargs: None)

    preview = module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="policy",
        artifact_path=candidate_path,
        cycle_id="cycle:policy:test",
        include_curriculum=True,
        include_failure_curriculum=True,
        task_limit=13,
    )

    assert preview["state"] == "retain"
    assert len(seen_calls) == 2
    assert all(call["persist_episode_memory"] is False for call in seen_calls)
    assert all(call["trajectories_root"] == config.trajectories_root for call in seen_calls)
    assert all(call["task_limit"] == 13 for call in seen_calls)
    assert all(str(call["run_checkpoints_dir"]).startswith(str(config.improvement_reports_dir)) for call in seen_calls)
    assert all(
        str(call["workspace_snapshot_root"]).startswith(str(config.improvement_reports_dir))
        for call in seen_calls
    )
    assert all(
        not str(call["run_checkpoints_dir"]).startswith(str(config.run_checkpoints_dir))
        for call in seen_calls
    )


def test_preview_candidate_retention_can_restrict_to_priority_primary_lane(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "tolbert" / "active.json"
    candidate_path = tmp_path / "tolbert" / "candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "retained",
        "model_surfaces": {},
        "runtime_paths": {},
        "preview_controls": {
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {"benchmark_candidate": 4.0},
        },
        "retention_gate": {"min_pass_rate_delta_abs": 0.0},
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps({**payload, "lifecycle_state": "proposed"}), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        tolbert_model_artifact_path=active_path,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    config.ensure_directories()
    seen_flags = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_flags.append(dict(flags))
        return EvalMetrics(total=3, passed=3, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "compare_to_prior_retained", lambda **kwargs: None)

    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="tolbert_model",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        cycle_id="cycle:tolbert:test",
        include_curriculum=True,
        include_failure_curriculum=True,
        priority_benchmark_families=["integration", "project", "repository"],
        restrict_to_priority_benchmark_families=True,
    )

    assert len(seen_flags) == 2
    assert all(flags["priority_benchmark_families"] == ["integration", "project", "repository"] for flags in seen_flags)
    assert all("benchmark_candidate" not in flags["priority_benchmark_families"] for flags in seen_flags)
    assert all(flags["restrict_to_priority_benchmark_families"] is True for flags in seen_flags)
    assert all(flags["prefer_low_cost_tasks"] is True for flags in seen_flags)
    assert all(flags["include_generated"] is True for flags in seen_flags)
    assert all(flags["include_failure_generated"] is True for flags in seen_flags)


def test_preview_candidate_retention_uses_bounded_retrieval_preview_controls(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    candidate_path = tmp_path / "candidates" / "retrieval_candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "retained",
        "retention_gate": {"min_pass_rate_delta_abs": 0.02},
    }
    candidate_payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "preview_controls": {
            "comparison_task_limit_floor": 24,
            "bounded_comparison_required": True,
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {
                "benchmark_candidate": 4.0,
                "integration": 3.0,
                "repository": 2.0,
            },
        },
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.02,
            "require_trusted_carryover_repair_improvement": True,
            "min_trusted_carryover_verified_step_delta": 1,
        },
    }
    active_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        retrieval_proposals_path=active_path,
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
    )
    config.ensure_directories()
    config.benchmark_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark_candidates_path.write_text(
        json.dumps({"artifact_kind": "benchmark_candidate_set", "proposals": []}),
        encoding="utf-8",
    )
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(
            {
                "task_limit": flags.get("task_limit"),
                "include_episode_memory": flags.get("include_episode_memory"),
                "include_verifier_memory": flags.get("include_verifier_memory"),
                "include_benchmark_candidates": flags.get("include_benchmark_candidates"),
                "priority_benchmark_families": flags.get("priority_benchmark_families"),
                "priority_benchmark_family_weights": flags.get("priority_benchmark_family_weights"),
            }
        )
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "compare_to_prior_retained", lambda **kwargs: None)

    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=candidate_path,
        cycle_id="cycle:retrieval:test",
        task_limit=4,
    )

    assert len(seen_calls) == 2
    assert all(call["task_limit"] == 24 for call in seen_calls)
    assert all(call["include_episode_memory"] is False for call in seen_calls)
    assert all(call["include_verifier_memory"] is False for call in seen_calls)
    assert all(call["include_benchmark_candidates"] is True for call in seen_calls)
    assert all(
        call["priority_benchmark_families"] == ["benchmark_candidate", "integration", "repository"]
        for call in seen_calls
    )
    assert all(
        call["priority_benchmark_family_weights"]
        == {"benchmark_candidate": 4.0, "integration": 3.0, "repository": 2.0}
        for call in seen_calls
    )


def test_preview_candidate_retention_falls_back_to_bounded_retrieval_task_limit_without_preview_controls(
    tmp_path, monkeypatch
):
    module = _load_finalize_module()
    active_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    candidate_path = tmp_path / "candidates" / "retrieval_candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.02,
            "require_trusted_carryover_repair_improvement": True,
            "min_trusted_carryover_verified_step_delta": 1,
        },
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        retrieval_proposals_path=active_path,
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    config.ensure_directories()
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(flags.get("task_limit"))
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(
        module.cycle_runner,
        "_evaluate_subsystem_metrics_with_tolbert_startup_retry",
        lambda *, config, subsystem, flags, progress_label, phase_name, progress: fake_evaluate(
            config=config,
            subsystem=subsystem,
            flags=flags,
            progress_label=progress_label,
        ),
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "autonomous_phase_gate_report",
        lambda **kwargs: {
            "passed": True,
            "failures": [],
            "generated_lane_included": True,
            "failure_recovery_lane_included": True,
        },
    )
    monkeypatch.setattr(module.cycle_runner, "compare_to_prior_retained", lambda **kwargs: None)

    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=candidate_path,
        cycle_id="cycle:retrieval:test",
        task_limit=4,
    )

    assert seen_calls == [8, 8]


def test_preview_candidate_retention_keeps_task_limit_for_retrieval_without_carryover_gate(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    candidate_path = tmp_path / "candidates" / "retrieval_candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.02},
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        retrieval_proposals_path=active_path,
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
    )
    config.ensure_directories()
    config.benchmark_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark_candidates_path.write_text(
        json.dumps({"artifact_kind": "benchmark_candidate_set", "proposals": []}),
        encoding="utf-8",
    )
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(flags.get("task_limit"))
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(module.cycle_runner, "compare_to_prior_retained", lambda **kwargs: None)

    module.cycle_runner.preview_candidate_retention(
        config=config,
        subsystem="retrieval",
        artifact_path=candidate_path,
        cycle_id="cycle:retrieval:test",
        task_limit=4,
    )

    assert seen_calls == [4, 4]


def test_compare_to_prior_retained_uses_bounded_retrieval_preview_controls(tmp_path, monkeypatch):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    snapshot_path = tmp_path / ".artifact_history" / "retrieval.cycle_prior.post_retain.json"
    artifact_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:prior",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="retrieval_policy_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "preview_controls": {
            "comparison_task_limit_floor": 12,
            "bounded_comparison_required": True,
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {
                "benchmark_candidate": 4.0,
                "integration": 3.0,
                "repository": 2.0,
            },
        },
        "retention_gate": {
            "require_trusted_carryover_repair_improvement": True,
            "min_trusted_carryover_verified_step_delta": 1,
        },
    }
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(
            {
                "task_limit": flags.get("task_limit"),
                "include_episode_memory": flags.get("include_episode_memory"),
                "include_verifier_memory": flags.get("include_verifier_memory"),
                "priority_benchmark_families": flags.get("priority_benchmark_families"),
                "priority_benchmark_family_weights": flags.get("priority_benchmark_family_weights"),
            }
        )
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    comparison = module.cycle_runner.compare_to_prior_retained(
        config=KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=artifact_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        ),
        planner=planner,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id="cycle:retrieval:current",
        flags={"include_benchmark_candidates": True},
        payload=payload,
        task_limit=4,
    )

    assert comparison is not None
    assert seen_calls == [
        {
            "task_limit": 12,
            "include_episode_memory": False,
            "include_verifier_memory": False,
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {
                "benchmark_candidate": 4.0,
                "integration": 3.0,
                "repository": 2.0,
            },
        },
        {
            "task_limit": 12,
            "include_episode_memory": False,
            "include_verifier_memory": False,
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {
                "benchmark_candidate": 4.0,
                "integration": 3.0,
                "repository": 2.0,
            },
        },
    ]


def test_compare_to_prior_retained_preserves_explicit_priority_restriction(tmp_path, monkeypatch):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    snapshot_path = tmp_path / ".artifact_history" / "retrieval.cycle_prior.post_retain.json"
    artifact_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:prior",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="retrieval_policy_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "preview_controls": {
            "comparison_task_limit_floor": 12,
            "bounded_comparison_required": True,
            "priority_benchmark_families": ["benchmark_candidate", "integration", "repository"],
            "priority_benchmark_family_weights": {"benchmark_candidate": 4.0},
        },
    }
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(dict(flags))
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    comparison = module.cycle_runner.compare_to_prior_retained(
        config=KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=artifact_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        ),
        planner=planner,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id="cycle:retrieval:current",
        flags={
            "include_generated": True,
            "include_failure_generated": True,
            "priority_benchmark_families": ["integration", "project", "repository"],
            "restrict_to_priority_benchmark_families": True,
        },
        payload=payload,
        task_limit=4,
    )

    assert comparison is not None
    assert len(seen_calls) == 2
    assert all(call["task_limit"] == 12 for call in seen_calls)
    assert all(call["priority_benchmark_families"] == ["integration", "project", "repository"] for call in seen_calls)
    assert all("benchmark_candidate" not in call["priority_benchmark_families"] for call in seen_calls)
    assert all(call["restrict_to_priority_benchmark_families"] is True for call in seen_calls)
    assert all(call["prefer_low_cost_tasks"] is True for call in seen_calls)
    assert all(call["include_generated"] is True for call in seen_calls)
    assert all(call["include_failure_generated"] is True for call in seen_calls)


def test_compare_to_prior_retained_falls_back_to_bounded_retrieval_task_limit_without_preview_controls(
    tmp_path, monkeypatch
):
    module = _load_finalize_module()
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    snapshot_path = tmp_path / ".artifact_history" / "retrieval.cycle_prior.post_retain.json"
    artifact_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    artifact_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:prior",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="retrieval_policy_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "require_trusted_carryover_repair_improvement": True,
            "min_trusted_carryover_verified_step_delta": 1,
        },
    }
    seen_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        del config, subsystem, progress_label
        seen_calls.append(flags.get("task_limit"))
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)

    comparison = module.cycle_runner.compare_to_prior_retained(
        config=KernelConfig(
            improvement_cycles_path=cycles_path,
            retrieval_proposals_path=artifact_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        ),
        planner=planner,
        subsystem="retrieval",
        artifact_path=artifact_path,
        cycles_path=cycles_path,
        before_cycle_id="cycle:retrieval:current",
        flags={"include_benchmark_candidates": True},
        payload=payload,
        task_limit=4,
    )

    assert comparison is not None
    assert seen_calls == [8, 8]


def test_finalize_cycle_runs_bounded_holdout_after_capped_preview_for_retrieval(tmp_path, monkeypatch):
    module = _load_finalize_module()
    active_path = tmp_path / "prompts" / "prompt_proposals.json"
    candidate_path = tmp_path / "candidates" / "prompt_candidate.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "prompt_proposal_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.0,
            "max_step_regression": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
            "required_confirmation_runs": 1,
        },
        "proposals": [{"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}],
    }
    active_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=active_path,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
    )
    config.ensure_directories()

    preview_baseline = EvalMetrics(
        total=100,
        passed=80,
        average_steps=1.5,
        generated_total=4,
        generated_passed=2,
        generated_by_kind={"failure_recovery": 2},
        generated_passed_by_kind={"failure_recovery": 1},
    )
    preview_candidate = EvalMetrics(
        total=100,
        passed=90,
        average_steps=1.2,
        generated_total=4,
        generated_passed=3,
        generated_by_kind={"failure_recovery": 2},
        generated_passed_by_kind={"failure_recovery": 2},
    )

    monkeypatch.setattr(
        module.cycle_runner,
        "preview_candidate_retention",
        lambda **kwargs: {
            "state": "retain",
            "reason": "preview retained",
            "gate": dict(payload["retention_gate"]),
            "phase_gate_report": {
                "passed": True,
                "failures": [],
                "generated_lane_included": True,
                "failure_recovery_lane_included": True,
            },
            "baseline": preview_baseline,
            "candidate": preview_candidate,
            "evidence": {},
            "compatibility": {},
            "payload": payload,
            "prior_retained_comparison": None,
            "baseline_flags": {
                "include_discovered_tasks": False,
                "include_episode_memory": False,
                "include_skill_memory": False,
                "include_skill_transfer": False,
                "include_operator_memory": False,
                "include_tool_memory": False,
                "include_verifier_memory": False,
                "include_benchmark_candidates": False,
                "include_verifier_candidates": False,
                "include_generated": True,
                "include_failure_generated": True,
                "task_limit": 5,
            },
            "candidate_flags": {
                "include_discovered_tasks": False,
                "include_episode_memory": False,
                "include_skill_memory": False,
                "include_skill_transfer": False,
                "include_operator_memory": False,
                "include_tool_memory": False,
                "include_verifier_memory": False,
                "include_benchmark_candidates": False,
                "include_verifier_candidates": False,
                "include_generated": True,
                "include_failure_generated": True,
                "task_limit": 5,
            },
            "active_artifact_path": active_path,
        },
    )
    holdout_calls = []

    def fake_evaluate(*, config, subsystem, flags, progress_label):
        holdout_calls.append(
            {
                "progress_label": progress_label,
                "persist_episode_memory": config.persist_episode_memory,
                "task_limit": flags.get("task_limit"),
                "max_generated_success_schedule_tasks": flags.get("max_generated_success_schedule_tasks"),
                "max_generated_failure_schedule_tasks": flags.get("max_generated_failure_schedule_tasks"),
            }
        )
        if len(holdout_calls) == 1:
            return EvalMetrics(
                total=120,
                passed=96,
                average_steps=1.5,
                generated_total=4,
                generated_passed=2,
                generated_by_kind={"failure_recovery": 2},
                generated_passed_by_kind={"failure_recovery": 1},
            )
        return EvalMetrics(
            total=120,
            passed=108,
            average_steps=1.2,
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
        )

    monkeypatch.setattr(module.cycle_runner, "evaluate_subsystem_metrics", fake_evaluate)
    monkeypatch.setattr(
        module.cycle_runner,
        "evaluate_artifact_retention",
        lambda *args, **kwargs: ("retain", "holdout retained"),
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "apply_artifact_retention_decision",
        lambda **kwargs: {
            "artifact_kind": "prompt_proposal_set",
            "artifact_lifecycle_state": "retained",
            "artifact_sha256": "sha",
            "previous_artifact_sha256": "",
            "rollback_artifact_path": "",
            "artifact_snapshot_path": "",
            "compatibility": {},
            "candidate_artifact_path": str(kwargs["artifact_path"]),
            "active_artifact_path": str(kwargs["active_artifact_path"]),
        },
    )
    monkeypatch.setattr(
        module.cycle_runner,
        "materialize_retained_retrieval_asset_bundle",
        lambda **kwargs: tmp_path / "tolbert_bundle" / "manifest.json",
    )

    state, _ = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="retrieval",
        cycle_id="cycle:retrieval:test",
        artifact_path=candidate_path,
        active_artifact_path=active_path,
        include_curriculum=True,
        include_failure_curriculum=True,
        comparison_task_limit=5,
    )

    records = [
        json.loads(line)
        for line in config.improvement_cycles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdout_records = [record for record in records if record["action"] == "holdout_candidate_to_baseline"]

    assert state == "retain"
    assert len(holdout_calls) == 2
    assert all(call["persist_episode_memory"] is False for call in holdout_calls)
    assert all(call["task_limit"] == 5 for call in holdout_calls)
    assert all(call["max_generated_success_schedule_tasks"] == 5 for call in holdout_calls)
    assert all(call["max_generated_failure_schedule_tasks"] == 5 for call in holdout_calls)
    assert len(holdout_records) == 1


def test_run_improvement_cycle_prefers_phase_gate_passing_sibling_variant(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    preview_calls = []
    finalized = []

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=9, average_steps=1.0),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="retrieval_caution",
                description="make low-confidence retrieval less binding",
                expected_gain=0.015,
                estimated_cost=2,
                score=0.02,
                controls={"focus": "retrieval_caution"},
            ),
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            ),
        ],
    )

    def fake_preview(**kwargs):
        artifact_path = str(kwargs["artifact_path"])
        preview_calls.append({"artifact_path": artifact_path, "task_limit": kwargs.get("task_limit")})
        if "retrieval_caution" in artifact_path:
            return {
                "state": "retain",
                "reason": "preview retained but phase gate failed",
                "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
                "candidate": EvalMetrics(total=10, passed=10, average_steps=1.0),
                "phase_gate_report": {"passed": False, "failures": ["generated lane missing"]},
            }
        return {
            "state": "retain",
            "reason": "preview retained",
            "baseline": EvalMetrics(total=10, passed=8, average_steps=1.5),
            "candidate": EvalMetrics(total=10, passed=9, average_steps=1.2),
            "phase_gate_report": {"passed": True, "failures": []},
        }

    monkeypatch.setattr(module, "preview_candidate_retention", fake_preview)
    monkeypatch.setattr(
        module,
        "finalize_cycle",
        lambda **kwargs: (finalized.append(str(kwargs["artifact_path"])) or "retain", "finalized"),
    )

    def fake_generate_candidate_artifact(**kwargs):
        variant = kwargs["variant"]
        path = tmp_path / "candidates" / f"{variant.variant_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"variant_id": variant.variant_id}), encoding="utf-8")
        return {
            "artifact": str(path),
            "action": "generate_stub",
            "artifact_kind": "prompt_proposal_set",
            "candidate_artifact_path": path,
            "prior_active_artifact_path": None,
        }

    monkeypatch.setattr(module, "_generate_candidate_artifact", fake_generate_candidate_artifact)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            compare_feature_max_tasks=17,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py", "--variant-width", "2", "--task-limit", "64"])

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sibling_select = [record for record in records if record["action"] == "choose_sibling_variant"]

    assert sibling_select[0]["metrics_summary"]["selected_variant_id"] == "verifier_alignment"
    assert sibling_select[0]["metrics_summary"]["comparison_task_limit"] == 17
    assert any("verifier_alignment" in path for path in finalized)
    assert all(call["task_limit"] == 17 for call in preview_calls)


def test_run_improvement_cycle_generates_benchmark_artifact(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": False,
                "task_metadata": {"benchmark_family": "micro", "capability": "file_write", "difficulty": "seed"},
                "summary": {
                    "failure_types": ["missing_expected_file"],
                    "executed_commands": ["false", "printf 'x\\n' > hello.txt"],
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
                    "metadata": {"benchmark_family": "micro", "capability": "file_write", "difficulty": "seed"},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=7, average_steps=1.0, generated_total=10, generated_passed=5),
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="benchmark",
                reason="benchmark gap",
                priority=5,
                expected_gain=0.03,
                estimated_cost=3,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "choose_variant",
        lambda self, experiment, metrics: ImprovementVariant(
            subsystem="benchmark",
            variant_id="failure_cluster_growth",
            description="expand benchmark coverage from clustered failures",
            expected_gain=0.03,
            estimated_cost=3,
            score=0.01,
            controls={"focus": "confidence"},
        ),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            trajectories_root=episodes,
            improvement_cycles_path=cycles_path,
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_improvement_cycle.py"])

    module.main()

    payload, records = _generated_candidate_payload(cycles_path)
    assert payload["artifact_kind"] == "benchmark_candidate_set"
    assert payload["generation_focus"] == "confidence"
    assert any(record["state"] == "generate" and record["candidate_artifact_path"] for record in records)


def test_yield_summary_for_marks_productive_depth_and_drift():
    summary = cycle_runner._yield_summary_for(
        [
            {
                "state": "retain",
                "metrics_summary": {
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": 0.8,
                    "baseline_average_steps": 5.0,
                    "candidate_average_steps": 9.0,
                    "phase_gate_passed": True,
                    "long_horizon_summary": {
                        "baseline_task_count": 1,
                        "candidate_task_count": 1,
                        "pass_rate_delta": 0.0,
                        "novel_valid_command_rate_delta": 0.0,
                        "world_feedback": {"progress_calibration_mae_gain": 0.0},
                    },
                },
            },
            {
                "state": "reject",
                "metrics_summary": {
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": 0.6,
                    "baseline_average_steps": 5.0,
                    "candidate_average_steps": 8.0,
                    "phase_gate_passed": False,
                },
            },
        ]
    )

    assert summary["productive_depth_retained_cycles"] == 1
    assert summary["long_horizon_retained_cycles"] == 1
    assert summary["average_productive_depth_step_delta"] == 4.0
    assert summary["depth_drift_cycles"] == 1
    assert summary["average_depth_drift_step_delta"] == 3.0


def test_run_repeated_improvement_cycles_flags_post_decision_failure_recovery_for_intervention():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    signal = module._mid_cycle_intervention_signal(
        {
            "pending_decision_state": "reject",
            "selected_subsystem": "retrieval",
            "last_progress_phase": "generated_failure",
            "current_task": {
                "task_id": "bridge_handoff_task_file_recovery",
                "family": "transition_pressure",
                "phase": "generated_failure",
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "post_decision_failure_recovery"
    assert signal["decision_state"] == "reject"
    assert signal["subsystem"] == "retrieval"


def test_run_repeated_improvement_cycles_reconcile_incomplete_cycles_preserves_strategy_lineage(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = KernelConfig(
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    config.ensure_directories()
    planner = ImprovementPlanner()

    summary = {
        "cycle_id": "cycle:tolbert_model:test",
        "subsystem": "tolbert_model",
        "protocol_match_id": "proto:test",
        "artifact_kind": "tolbert_model_policy",
        "artifact_path": str(tmp_path / "active.json"),
        "active_artifact_path": str(tmp_path / "active.json"),
        "candidate_artifact_path": str(tmp_path / "candidate.json"),
        "last_state": "evaluate",
        "last_action": "preview_sibling_candidate",
        "selected_variant_id": "discovered_task_adaptation",
        "strategy_candidate_id": "strategy:adaptive_countermeasure:tolbert_model:test",
        "strategy_candidate_kind": "adaptive_countermeasure",
        "strategy_origin": "discovered_strategy",
        "record_count": 5,
        "selected_cycles": 1,
    }

    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=summary["cycle_id"],
            state="evaluate",
            subsystem=summary["subsystem"],
            action="preview_sibling_candidate",
            artifact_path=summary["candidate_artifact_path"],
            artifact_kind=summary["artifact_kind"],
            reason="preview in progress",
            metrics_summary={"protocol": "autonomous", "protocol_match_id": summary["protocol_match_id"]},
            selected_variant_id=summary["selected_variant_id"],
            strategy_candidate_id=summary["strategy_candidate_id"],
            strategy_candidate_kind=summary["strategy_candidate_kind"],
            strategy_origin=summary["strategy_origin"],
            candidate_artifact_path=summary["candidate_artifact_path"],
            active_artifact_path=summary["active_artifact_path"],
        ),
    )

    reconciled = module._reconcile_incomplete_cycles(
        config=config,
        planner=planner,
        progress_label=None,
    )

    assert reconciled
    records = [json.loads(line) for line in config.improvement_cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    closeout_records = [record for record in records if record["cycle_id"] == summary["cycle_id"]][-2:]
    assert [record["action"] for record in closeout_records] == ["finalize_cycle", "persist_retention_outcome"]
    for record in closeout_records:
        assert record["strategy_candidate_id"] == summary["strategy_candidate_id"]
        assert record["strategy_candidate_kind"] == summary["strategy_candidate_kind"]
        assert record["strategy_origin"] == summary["strategy_origin"]
        assert record["metrics_summary"]["strategy_candidate_id"] == summary["strategy_candidate_id"]
        assert record["metrics_summary"]["strategy_candidate_kind"] == summary["strategy_candidate_kind"]
        assert record["metrics_summary"]["strategy_origin"] == summary["strategy_origin"]


def test_run_repeated_improvement_cycles_stream_runner_honors_callback_termination(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
    spec = importlib.util.spec_from_file_location("run_repeated_improvement_cycles", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    terminated = {"called": False}

    class FakeStdout:
        def __iter__(self):
            return iter(
                [
                    "[eval:cycle:test] phase=generated_failure task 1/8 bridge_handoff_task family=transition_pressure\n",
                ]
            )

    class FakeProcess:
        pid = 4242
        stdout = FakeStdout()

        def wait(self):
            return 0

    monkeypatch.setattr(module, "spawn_process_group", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        module,
        "terminate_process_tree",
        lambda process: terminated.__setitem__("called", True),
    )

    result = module._run_and_stream(
        [sys.executable, "-u", "fake.py"],
        cwd=tmp_path,
        env={},
        progress_label="test-cycle",
        on_event=lambda event: (
            {
                "terminate": True,
                "reason": "mid-cycle controller intervention: durable reject decision already emitted",
            }
            if str(event.get("event", "")) == "output"
            else None
        ),
    )

    assert terminated["called"] is True
    assert result["timed_out"] is True
    assert "controller intervention" in result["timeout_reason"]
