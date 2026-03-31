from pathlib import Path
import importlib.util
import json
from io import StringIO
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
    assert progress_messages[-1] == "finalize phase=done subsystem=retrieval state=reject"


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

    monkeypatch.setattr(module.cycle_runner, "run_eval", fake_run_eval)
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
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
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
    assert records[-1]["artifact_kind"] == "improvement_campaign_report"
    assert records[-1]["metrics_summary"]["priority_families_with_retained_gain"] == ["project"]
    assert "--priority-benchmark-family" in seen_cmds[0]
    assert "--protocol-match-id" in seen_cmds[0]
    assert "project" in seen_cmds[0]


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
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
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
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
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
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "quality": 0.9,
                        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
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
                "skills": [{"skill_id": "live"}],
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "skills": [{"skill_id": "candidate"}],
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
                "skills": [{"skill_id": "prior"}],
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
                "operators": [
                    {
                        "operator_id": "operator:file_write:bounded",
                        "source_task_ids": ["hello_task", "math_task"],
                        "template_procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                        "template_contract": {"expected_files": ["hello.txt"]},
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
    assert [record["state"] for record in records] == ["evaluate", "retain", "record", "record"]
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
        lambda self, metrics, max_candidates=1: [
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
        lambda self, metrics, max_candidates=1: [
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
    finalized: list[str] = []

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
        lambda self, metrics, max_candidates=1: [
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
    assert [record["state"] for record in stale[-2:]] == ["reject", "record"]
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
        lambda self, metrics, max_candidates=1: [ImprovementExperiment("policy", "policy gap", 4, 0.02, 2, 0.1, {})],
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
        ["run_improvement_cycle.py", "--variant-width", "2", "--task-limit", "17", "--progress-label", "test-progress"],
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


def test_finalize_cycle_runs_uncapped_holdout_after_capped_preview(tmp_path, monkeypatch):
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

    state, _ = module.cycle_runner.finalize_cycle(
        config=config,
        subsystem="policy",
        cycle_id="cycle:policy:test",
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
    assert all(call["task_limit"] is None for call in holdout_calls)
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
