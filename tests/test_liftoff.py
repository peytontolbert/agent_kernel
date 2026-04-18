from pathlib import Path
import importlib.util
import json
from io import StringIO
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.evaluation.liftoff import build_liftoff_gate_report
from evals.metrics import EvalMetrics


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_liftoff_gate_report_promotes_non_regressive_families():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 5, "project": 5},
            passed_by_benchmark_family={"workflow": 5, "project": 4},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3, "project": 2},
            proposal_metrics_by_benchmark_family={
                "workflow": {
                    "task_count": 5,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 2,
                    "novel_valid_command_rate": 1.0,
                },
                "project": {
                    "task_count": 5,
                    "proposal_selected_steps": 1,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            total_by_benchmark_family={"workflow": 5, "project": 5},
            passed_by_benchmark_family={"workflow": 4, "project": 4},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            proposal_metrics_by_benchmark_family={
                "workflow": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
                "project": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
            },
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "retain"
    assert "workflow" in report.primary_takeover_families
    assert report.family_takeover_evidence["workflow"]["decision"] == "promoted"
    assert report.family_takeover_evidence["workflow"]["proposal_selected_steps_delta"] == 2
    assert report.family_takeover_evidence["workflow"]["novel_valid_command_rate_delta"] == 1.0


def test_build_liftoff_gate_report_requires_shadow_signal_for_takeover():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 5},
            passed_by_benchmark_family={"workflow": 5},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            total_by_benchmark_family={"workflow": 5},
            passed_by_benchmark_family={"workflow": 4},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
    )

    assert report.state == "shadow_only"
    assert "workflow" in report.insufficient_shadow_families


def test_build_liftoff_gate_report_rejects_universal_decoder_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 2},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        artifact_payload={
            "artifact_kind": "tolbert_model_bundle",
            "liftoff_gate": {
                "require_universal_decoder_eval": True,
                "min_universal_decoder_exact_match_delta": 0.0,
                "min_universal_decoder_token_f1_delta": 0.0,
                "min_universal_decoder_win_rate_delta": 0.0,
                "require_takeover_drift_eval": False,
            },
        },
        universal_decoder_eval={
            "available": True,
            "hybrid_exact_match_rate": 0.25,
            "baseline_exact_match_rate": 0.5,
            "hybrid_token_f1": 0.6,
            "baseline_token_f1": 0.8,
            "hybrid_win_rate": 0.0,
            "baseline_win_rate": 0.5,
        },
    )

    assert report.state == "reject"
    assert "universal decoder exact-match quality" in report.reason
    assert report.universal_decoder_exact_match_delta == -0.25


def test_build_liftoff_gate_report_demotes_family_without_novel_command_evidence():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"repository": 5},
            passed_by_benchmark_family={"repository": 5},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"repository": 3},
            proposal_metrics_by_benchmark_family={
                "repository": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                }
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            total_by_benchmark_family={"repository": 5},
            passed_by_benchmark_family={"repository": 4},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            proposal_metrics_by_benchmark_family={
                "repository": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                }
            },
        ),
        artifact_payload={
            "artifact_kind": "tolbert_model_bundle",
            "liftoff_gate": {
                "require_family_novel_command_evidence": True,
                "proposal_gate_by_benchmark_family": {
                    "repository": {
                        "require_novel_command_signal": True,
                        "min_proposal_selected_steps_delta": 1,
                        "min_novel_valid_command_steps": 1,
                        "min_novel_valid_command_rate_delta": 0.1,
                    }
                },
            },
        },
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "shadow_only"
    assert "novel-command evidence" in report.reason
    assert "repository" in report.insufficient_proposal_families
    assert report.family_takeover_evidence["repository"]["decision"] == "insufficient_proposal"
    assert (
        report.proposal_gate_failure_reasons_by_benchmark_family["repository"]
        == "missing proposal-selected commands"
    )
    assert (
        report.family_takeover_evidence["repository"]["failure_reason"]
        == "missing proposal-selected commands"
    )


def test_build_liftoff_gate_report_accepts_primary_routing_signal_without_novel_commands():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.1,
            total_by_benchmark_family={"repository": 10},
            passed_by_benchmark_family={"repository": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"repository": 3},
            tolbert_primary_episodes_by_benchmark_family={"repository": 2},
            proposal_metrics_by_benchmark_family={
                "repository": {
                    "task_count": 10,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                }
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.2,
            total_by_benchmark_family={"repository": 10},
            passed_by_benchmark_family={"repository": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            proposal_metrics_by_benchmark_family={
                "repository": {
                    "task_count": 10,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                }
            },
        ),
        artifact_payload={
            "artifact_kind": "tolbert_model_bundle",
            "runtime_policy": {
                "shadow_benchmark_families": ["repository"],
                "primary_benchmark_families": ["repository"],
            },
            "liftoff_gate": {
                "require_family_novel_command_evidence": True,
                "require_shadow_signal": True,
                "min_shadow_episodes_per_promoted_family": 1,
                "proposal_gate_by_benchmark_family": {
                    "repository": {
                        "require_novel_command_signal": True,
                        "min_proposal_selected_steps_delta": 0,
                        "min_novel_valid_command_steps": 0,
                        "min_novel_valid_command_rate_delta": 0.0,
                        "allow_primary_routing_signal": True,
                        "min_primary_episodes": 1,
                    }
                },
            },
        },
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "retain"
    assert "repository" in report.primary_takeover_families
    assert report.family_takeover_evidence["repository"]["decision"] == "promoted"
    assert report.family_takeover_evidence["repository"]["failure_reason"] == ""


def test_build_liftoff_gate_report_rejects_safety_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            unsafe_ambiguous_episodes=2,
            hidden_side_effect_risk_episodes=1,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            unsafe_ambiguous_episodes=0,
            hidden_side_effect_risk_episodes=0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
    )

    assert report.state == "reject"
    assert "unsafe-ambiguous" in report.reason


def test_build_liftoff_gate_report_rejects_long_horizon_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_difficulty={"long_horizon": 4},
            passed_by_difficulty={"long_horizon": 2},
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
            proposal_metrics_by_difficulty={
                "long_horizon": {
                    "task_count": 4,
                    "proposal_selected_steps": 1,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                }
            },
            world_feedback_by_difficulty={
                "long_horizon": {
                    "step_count": 4,
                    "progress_calibration_mae": 0.4,
                    "risk_calibration_mae": 0.2,
                }
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            total_by_difficulty={"long_horizon": 4},
            passed_by_difficulty={"long_horizon": 3},
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            proposal_metrics_by_difficulty={
                "long_horizon": {
                    "task_count": 4,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 1,
                    "novel_valid_command_rate": 0.5,
                }
            },
            world_feedback_by_difficulty={
                "long_horizon": {
                    "step_count": 4,
                    "progress_calibration_mae": 0.2,
                    "risk_calibration_mae": 0.1,
                }
            },
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
    )

    assert report.state == "reject"
    assert "long-horizon" in report.reason


def test_build_liftoff_gate_report_rejects_long_horizon_persistence_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_difficulty={"long_horizon": 4},
            passed_by_difficulty={"long_horizon": 4},
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
            proposal_metrics_by_difficulty={
                "long_horizon": {
                    "task_count": 4,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 1,
                    "novel_valid_command_rate": 0.5,
                }
            },
            world_feedback_by_difficulty={
                "long_horizon": {
                    "step_count": 4,
                    "progress_calibration_mae": 0.2,
                    "risk_calibration_mae": 0.1,
                }
            },
            long_horizon_persistence_summary={
                "long_horizon_steps": 8,
                "productive_long_horizon_step_rate": 0.25,
                "recovery_response_rate": 0.5,
                "horizon_drop_rate": 0.25,
                "long_horizon_success_rate": 1.0,
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            total_by_difficulty={"long_horizon": 4},
            passed_by_difficulty={"long_horizon": 4},
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            proposal_metrics_by_difficulty={
                "long_horizon": {
                    "task_count": 4,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 1,
                    "novel_valid_command_rate": 0.5,
                }
            },
            world_feedback_by_difficulty={
                "long_horizon": {
                    "step_count": 4,
                    "progress_calibration_mae": 0.2,
                    "risk_calibration_mae": 0.1,
                }
            },
            long_horizon_persistence_summary={
                "long_horizon_steps": 8,
                "productive_long_horizon_step_rate": 0.75,
                "recovery_response_rate": 0.5,
                "horizon_drop_rate": 0.0,
                "long_horizon_success_rate": 1.0,
            },
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
    )

    assert report.state == "reject"
    assert "productive persistence" in report.reason


def test_build_liftoff_gate_report_rejects_transfer_alignment_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
            transfer_alignment_summary={
                "transfer_step_count": 4,
                "verified_transfer_step_rate": 0.5,
                "trusted_retrieval_alignment_mean": 0.4,
                "graph_environment_alignment_mean": -0.1,
                "safe_transfer_step_rate": 0.25,
                "unsafe_transfer_step_rate": 0.75,
            },
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            transfer_alignment_summary={
                "transfer_step_count": 4,
                "verified_transfer_step_rate": 0.75,
                "trusted_retrieval_alignment_mean": 0.7,
                "graph_environment_alignment_mean": 0.4,
                "safe_transfer_step_rate": 1.0,
                "unsafe_transfer_step_rate": 0.0,
            },
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
    )

    assert report.state == "reject"
    assert "transfer-safe" in report.reason


def test_evaluate_tolbert_liftoff_shadow_enrolls_priority_families_for_candidate_probe(tmp_path, monkeypatch):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": [],
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "tolbert_model" / "report.json"
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=tmp_path / "reports",
    )

    candidate_artifact_paths: list[Path] = []

    def fake_run_eval(*, config, **kwargs):
        assert kwargs["write_unattended_reports"] is True
        assert kwargs["prefer_low_cost_tasks"] is True
        assert kwargs["task_limit"] == 12
        assert kwargs["priority_benchmark_families"] == ["repository", "project", "workflow"]
        if config.use_tolbert_model_artifacts:
            candidate_artifact_paths.append(config.tolbert_model_artifact_path)
            payload = json.loads(config.tolbert_model_artifact_path.read_text(encoding="utf-8"))
            assert {"repository", "project", "workflow"}.issubset(
                set(payload["runtime_policy"]["shadow_benchmark_families"])
            )
            return EvalMetrics(
                total=12,
                passed=10,
                average_steps=1.0,
                total_by_benchmark_family={"repository": 4, "project": 4, "workflow": 4},
                passed_by_benchmark_family={"repository": 3, "project": 3, "workflow": 4},
                tolbert_shadow_episodes_by_benchmark_family={"repository": 4, "project": 4, "workflow": 4},
            )
        return EvalMetrics(
            total=12,
            passed=9,
            average_steps=1.1,
            total_by_benchmark_family={"repository": 4, "project": 4, "workflow": 4},
            passed_by_benchmark_family={"repository": 3, "project": 2, "workflow": 4},
        )

    def fake_build_trust_ledger(run_config):
        del run_config
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 0.9,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {
                "repository": {"passed": True, "status": "trusted"},
                "project": {"passed": True, "status": "trusted"},
                "workflow": {"passed": True, "status": "trusted"},
            },
        }

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(
        module,
        "write_unattended_trust_ledger",
        lambda run_config, **kwargs: Path(str(kwargs.get("ledger_path", run_config.unattended_trust_ledger_path))),
    )
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: (_ for _ in ()).throw(AssertionError("drift should be skipped")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "12",
            "--takeover-drift-step-budget",
            "0",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--priority-benchmark-family",
            "workflow",
        ],
    )

    module.main()

    assert candidate_artifact_paths
    assert candidate_artifact_paths[0] != artifact_path
    original_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert original_payload["runtime_policy"]["shadow_benchmark_families"] == []


def test_evaluate_tolbert_liftoff_can_preserve_report_history(tmp_path, monkeypatch):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")
    report_path = tmp_path / "tolbert_model" / "report.json"
    reports_dir = tmp_path / "reports"
    baseline_dir = reports_dir / "tolbert_liftoff_baseline"
    candidate_dir = reports_dir / "tolbert_liftoff_candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline_marker.json").write_text("{}", encoding="utf-8")
    (candidate_dir / "candidate_marker.json").write_text("{}", encoding="utf-8")
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=reports_dir,
    )

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    seen_trust_limits: list[int] = []

    def fake_build_trust_ledger(run_config):
        seen_trust_limits.append(int(run_config.unattended_trust_recent_report_limit))
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {},
        }

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(
        module,
        "write_unattended_trust_ledger",
        lambda run_config, **kwargs: Path(str(kwargs.get("ledger_path", run_config.unattended_trust_ledger_path))),
    )
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: (_ for _ in ()).throw(AssertionError("drift should be skipped")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "4",
            "--takeover-drift-step-budget",
            "0",
            "--preserve-report-history",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert (baseline_dir / "baseline_marker.json").exists()
    assert (candidate_dir / "candidate_marker.json").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["preserve_report_history"] is True
    assert seen_trust_limits == [0, 0]


def test_evaluate_tolbert_liftoff_preserved_history_uses_full_required_family_window(tmp_path, monkeypatch):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")
    report_path = tmp_path / "tolbert_model" / "report.json"
    reports_dir = tmp_path / "reports"
    candidate_dir = reports_dir / "tolbert_liftoff_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = reports_dir / "tolbert_liftoff_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    def _write_report(root: Path, *, name: str, family: str) -> None:
        payload = {
            "report_kind": "unattended_task_report",
            "generated_at": f"2026-04-03T00:00:{name.zfill(2)}+00:00",
            "benchmark_family": family,
            "trust_scope": "gated",
            "outcome": "success",
            "success": True,
            "summary": {"command_steps": 1, "unexpected_change_files": 0},
            "commands": [{"cmd": "true"}],
            "side_effects": {"hidden_side_effect_risk": False},
            "supervision": {"mode": "unattended", "independent_execution": True},
        }
        (root / f"task_report_{family}_{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    _write_report(candidate_dir, name="01", family="project")
    _write_report(candidate_dir, name="02", family="repository")
    _write_report(candidate_dir, name="99", family="repo_sandbox")
    _write_report(baseline_dir, name="01", family="project")
    _write_report(baseline_dir, name="02", family="repository")
    _write_report(baseline_dir, name="99", family="repo_sandbox")
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=reports_dir,
        unattended_trust_recent_report_limit=1,
        unattended_trust_required_benchmark_families=("project", "repository", "repo_sandbox"),
    )

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return EvalMetrics(total=3, passed=3, average_steps=1.0)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: (_ for _ in ()).throw(AssertionError("drift should be skipped")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "3",
            "--takeover-drift-step-budget",
            "0",
            "--preserve-report-history",
            "--priority-benchmark-family",
            "repo_sandbox",
        ],
    )

    module.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["candidate_trust"]["policy"]["recent_report_limit"] == 0
    assert payload["candidate_trust"]["coverage_summary"]["required_families_with_counted_gated_reports"] == [
        "project",
        "repo_sandbox",
        "repository",
    ]


def test_evaluate_tolbert_liftoff_enables_repo_sandbox_operator_policy_for_required_family(
    tmp_path,
    monkeypatch,
):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")
    report_path = tmp_path / "tolbert_model" / "report.json"
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=tmp_path / "reports",
        unattended_trust_required_benchmark_families=("repo_sandbox", "project"),
        unattended_allow_git_commands=False,
        unattended_allow_generated_path_mutations=False,
    )

    seen_policy: list[tuple[bool, bool]] = []

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        seen_policy.append(
            (
                bool(config.unattended_allow_git_commands),
                bool(config.unattended_allow_generated_path_mutations),
            )
        )
        return EvalMetrics(total=4, passed=4, average_steps=1.0)

    def fake_build_trust_ledger(run_config):
        del run_config
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {},
        }

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(
        module,
        "write_unattended_trust_ledger",
        lambda run_config, **kwargs: Path(str(kwargs.get("ledger_path", run_config.unattended_trust_ledger_path))),
    )
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: (_ for _ in ()).throw(AssertionError("drift should be skipped")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "4",
            "--takeover-drift-step-budget",
            "0",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert seen_policy == [(True, True), (True, True)]


def test_evaluate_tolbert_liftoff_scopes_takeover_drift_to_shadow_candidate_and_priority_family(
    tmp_path,
    monkeypatch,
):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")
    report_path = tmp_path / "tolbert_model" / "report.json"
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=tmp_path / "reports",
        unattended_trust_required_benchmark_families=("repo_sandbox",),
        unattended_allow_git_commands=False,
        unattended_allow_generated_path_mutations=False,
    )

    captured: dict[str, object] = {}

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return EvalMetrics(
            total=4,
            passed=4,
            average_steps=1.0,
            total_by_benchmark_family={"repo_sandbox": 4},
            passed_by_benchmark_family={"repo_sandbox": 4},
            tolbert_shadow_episodes_by_benchmark_family={"repo_sandbox": 1},
        )

    def fake_build_trust_ledger(run_config):
        del run_config
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"repo_sandbox": {"passed": True, "status": "trusted"}},
        }

    class FakeDriftReport:
        def to_dict(self):
            return {
                "budget_reached": True,
                "worst_pass_rate_delta": 0.0,
                "worst_unsafe_ambiguous_rate_delta": 0.0,
                "worst_hidden_side_effect_rate_delta": 0.0,
                "worst_trust_success_rate_delta": 0.0,
                "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
            }

    def fake_run_takeover_drift_eval(**kwargs):
        captured.update(kwargs)
        return FakeDriftReport()

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(
        module,
        "write_unattended_trust_ledger",
        lambda run_config, **kwargs: Path(str(kwargs.get("ledger_path", run_config.unattended_trust_ledger_path))),
    )
    monkeypatch.setattr(module, "run_takeover_drift_eval", fake_run_takeover_drift_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "4",
            "--takeover-drift-step-budget",
            "5",
            "--priority-benchmark-family",
            "repo_sandbox",
        ],
    )

    module.main()

    assert Path(str(captured["artifact_path"])).name.endswith(".liftoff_shadow.json")
    assert bool(captured["config"].unattended_allow_git_commands) is True
    assert bool(captured["config"].unattended_allow_generated_path_mutations) is True
    assert captured["eval_kwargs"]["priority_benchmark_families"] == ["repo_sandbox"]
    assert captured["eval_kwargs"]["prefer_low_cost_tasks"] is True


def test_evaluate_tolbert_liftoff_preserved_history_widens_eval_breadth_and_weights_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["workflow"],
                    "primary_benchmark_families": ["repository"],
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "tolbert_model" / "report.json"
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
        run_reports_dir=tmp_path / "reports",
        unattended_trust_required_benchmark_families=("integration", "project", "repo_chore", "repo_sandbox", "repository"),
        unattended_allow_git_commands=False,
        unattended_allow_generated_path_mutations=False,
    )

    captured_eval_calls: list[dict[str, object]] = []
    captured_drift: dict[str, object] = {}

    def fake_run_eval(*, config, **kwargs):
        del config
        captured_eval_calls.append(dict(kwargs))
        return EvalMetrics(
            total=10,
            passed=10,
            average_steps=1.0,
            total_by_benchmark_family={"repo_sandbox": 4, "repository": 2, "project": 2, "repo_chore": 1, "integration": 1},
            passed_by_benchmark_family={"repo_sandbox": 4, "repository": 2, "project": 2, "repo_chore": 1, "integration": 1},
            tolbert_shadow_episodes_by_benchmark_family={
                "repo_sandbox": 2,
                "repository": 1,
                "project": 1,
                "repo_chore": 1,
                "integration": 1,
            },
        )

    def fake_build_trust_ledger(run_config):
        del run_config
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {
                "integration": {"passed": True, "status": "bootstrap"},
                "project": {"passed": True, "status": "trusted"},
                "repo_chore": {"passed": True, "status": "trusted"},
                "repo_sandbox": {"passed": True, "status": "trusted"},
                "repository": {"passed": True, "status": "trusted"},
            },
        }

    class FakeDriftReport:
        def to_dict(self):
            return {
                "budget_reached": True,
                "worst_pass_rate_delta": 0.0,
                "worst_unsafe_ambiguous_rate_delta": 0.0,
                "worst_hidden_side_effect_rate_delta": 0.0,
                "worst_trust_success_rate_delta": 0.0,
                "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
            }

    def fake_run_takeover_drift_eval(**kwargs):
        captured_drift.update(kwargs)
        return FakeDriftReport()

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(
        module,
        "write_unattended_trust_ledger",
        lambda run_config, **kwargs: Path(str(kwargs.get("ledger_path", run_config.unattended_trust_ledger_path))),
    )
    monkeypatch.setattr(module, "run_takeover_drift_eval", fake_run_takeover_drift_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--task-limit",
            "10",
            "--takeover-drift-step-budget",
            "5",
            "--preserve-report-history",
            "--priority-benchmark-family",
            "repo_sandbox",
        ],
    )

    module.main()

    assert len(captured_eval_calls) == 2
    expected_families = ["repo_sandbox", "integration", "project", "repo_chore", "repository", "workflow"]
    expected_weights = {
        "repo_sandbox": 2.0,
        "integration": 1.0,
        "project": 1.0,
        "repo_chore": 1.0,
        "repository": 1.0,
        "workflow": 1.0,
    }
    assert captured_eval_calls[0]["priority_benchmark_families"] == expected_families
    assert captured_eval_calls[0]["priority_benchmark_family_weights"] == expected_weights
    assert captured_eval_calls[0]["surface_shared_repo_bundles"] is False
    assert captured_drift["eval_kwargs"]["priority_benchmark_families"] == expected_families
    assert captured_drift["eval_kwargs"]["priority_benchmark_family_weights"] == expected_weights
    assert captured_drift["eval_kwargs"]["surface_shared_repo_bundles"] is False


def test_scoped_liftoff_config_scopes_workspace_and_reports(tmp_path):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    base = KernelConfig(
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        run_reports_dir=tmp_path / "reports",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust_ledger.json",
    )

    scoped = module._scoped_liftoff_config(base, "liftoff_scope")

    assert scoped.workspace_root != base.workspace_root
    assert scoped.workspace_root.name == "liftoff_scope"
    assert scoped.run_reports_dir != base.run_reports_dir
    assert scoped.run_reports_dir.name == "liftoff_scope"
    assert scoped.unattended_workspace_snapshot_root != base.unattended_workspace_snapshot_root


def test_liftoff_scope_name_uses_report_stem():
    module = _load_script_module("evaluate_tolbert_liftoff.py")

    scope = module._liftoff_scope_name(
        "tolbert_liftoff_candidate",
        report_path=Path("trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverbreadth_mixed_task10.json"),
    )

    assert scope == "tolbert_liftoff_candidate_liftoff_gate_report_20260403_reposandbox_takeoverbreadth_mixed_task10"


def test_build_liftoff_gate_report_rejects_trust_ledger_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        candidate_trust_ledger={
            "overall_assessment": {"passed": False, "status": "restricted"},
            "gated_summary": {
                "success_rate": 0.75,
                "unsafe_ambiguous_rate": 0.10,
                "hidden_side_effect_risk_rate": 0.10,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"workflow": {"passed": False, "status": "restricted"}},
        },
        baseline_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 0.90,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"workflow": {"passed": True, "status": "trusted"}},
        },
    )

    assert report.state == "reject"
    assert "trust-ledger" in report.reason


def test_build_liftoff_gate_report_prefers_counted_trust_summary_over_gated_attempts():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        candidate_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "counted_gated_summary": {
                "success_rate": 0.90,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "gated_summary": {
                "success_rate": 0.70,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.20,
                "success_hidden_side_effect_risk_rate": 0.10,
            },
            "family_assessments": {"workflow": {"passed": True, "status": "trusted"}},
        },
        baseline_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "counted_gated_summary": {
                "success_rate": 0.90,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "gated_summary": {
                "success_rate": 0.90,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"workflow": {"passed": True, "status": "trusted"}},
        },
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "retain"
    assert report.trust_success_rate_delta == 0.0


def test_build_liftoff_gate_report_surfaces_autonomy_bridge_evidence():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"repository": 10},
            passed_by_benchmark_family={"repository": 9},
            tolbert_shadow_episodes_by_benchmark_family={"repository": 3},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"repository": 10},
            passed_by_benchmark_family={"repository": 9},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        candidate_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "counted_gated_summary": {
                "light_supervision_clean_success_count": 4,
                "contract_clean_failure_recovery_clean_success_count": 2,
            },
            "coverage_summary": {
                "required_family_light_supervision_clean_success_counts": {"repository": 4},
                "required_family_contract_clean_failure_recovery_clean_success_counts": {"repository": 2},
            },
        },
        baseline_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "counted_gated_summary": {
                "light_supervision_clean_success_count": 1,
                "contract_clean_failure_recovery_clean_success_count": 0,
            },
            "coverage_summary": {
                "required_family_light_supervision_clean_success_counts": {"repository": 1},
                "required_family_contract_clean_failure_recovery_clean_success_counts": {"repository": 0},
            },
        },
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.autonomy_bridge_evidence["light_supervision_clean_success_delta"] == 3
    assert report.autonomy_bridge_evidence["contract_clean_failure_recovery_clean_success_delta"] == 2
    assert report.autonomy_bridge_evidence["candidate"]["required_family_light_supervision_clean_success_counts"] == {
        "repository": 4
    }


def test_build_liftoff_gate_report_rejects_takeover_drift_regression():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.0,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 9},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3},
        ),
        baseline_metrics=EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.1,
            total_by_benchmark_family={"workflow": 10},
            passed_by_benchmark_family={"workflow": 8},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        takeover_drift_report={
            "budget_reached": True,
            "worst_pass_rate_delta": -0.2,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "reject"
    assert "takeover-drift" in report.reason


def test_build_liftoff_gate_report_requires_takeover_drift_budget_reached():
    report = build_liftoff_gate_report(
        candidate_metrics=EvalMetrics(
            total=4,
            passed=4,
            average_steps=1.0,
            total_by_benchmark_family={"repo_sandbox": 4},
            passed_by_benchmark_family={"repo_sandbox": 4},
            tolbert_shadow_episodes_by_benchmark_family={"repo_sandbox": 1},
        ),
        baseline_metrics=EvalMetrics(
            total=4,
            passed=4,
            average_steps=1.0,
            total_by_benchmark_family={"repo_sandbox": 4},
            passed_by_benchmark_family={"repo_sandbox": 4},
        ),
        artifact_payload={"artifact_kind": "tolbert_model_bundle"},
        candidate_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"repo_sandbox": {"passed": True, "status": "trusted"}},
        },
        baseline_trust_ledger={
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 1.0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {"repo_sandbox": {"passed": True, "status": "trusted"}},
        },
        takeover_drift_report={
            "budget_reached": False,
            "worst_pass_rate_delta": 0.0,
            "worst_unsafe_ambiguous_rate_delta": 0.0,
            "worst_hidden_side_effect_rate_delta": 0.0,
            "worst_trust_success_rate_delta": 0.0,
            "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
        },
    )

    assert report.state == "shadow_only"
    assert "required step budget" in report.reason


def test_evaluate_tolbert_liftoff_script_writes_report_and_applies_routing(tmp_path, monkeypatch):
    module = _load_script_module("evaluate_tolbert_liftoff.py")
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["workflow"],
                    "primary_benchmark_families": [],
                    "min_path_confidence": 0.75,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 2,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                    "require_family_novel_command_evidence": True,
                    "proposal_gate_by_benchmark_family": {
                        "project": {
                            "require_novel_command_signal": True,
                            "min_proposal_selected_steps_delta": 1,
                            "min_novel_valid_command_steps": 1,
                            "min_novel_valid_command_rate_delta": 0.1,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "tolbert_model" / "report.json"
    config = KernelConfig(
        tolbert_model_artifact_path=artifact_path,
        tolbert_liftoff_report_path=report_path,
    )
    calls = {"count": 0}

    def fake_run_eval(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return EvalMetrics(
                total=10,
                passed=8,
                average_steps=1.3,
                total_by_benchmark_family={"workflow": 5, "project": 5},
                passed_by_benchmark_family={"workflow": 4, "project": 4},
                generated_total=2,
                generated_passed=2,
                generated_by_kind={"failure_recovery": 1},
                generated_passed_by_kind={"failure_recovery": 1},
                proposal_metrics_by_benchmark_family={
                    "workflow": {
                        "task_count": 5,
                        "proposal_selected_steps": 0,
                        "novel_valid_command_steps": 0,
                        "novel_valid_command_rate": 0.0,
                    },
                    "project": {
                        "task_count": 5,
                        "proposal_selected_steps": 0,
                        "novel_valid_command_steps": 0,
                        "novel_valid_command_rate": 0.0,
                    },
                },
            )
        return EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.1,
            total_by_benchmark_family={"workflow": 5, "project": 5},
            passed_by_benchmark_family={"workflow": 5, "project": 4},
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3, "project": 1},
            proposal_metrics_by_benchmark_family={
                "workflow": {
                    "task_count": 5,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 2,
                    "novel_valid_command_rate": 1.0,
                },
                "project": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
            },
        )

    def fake_build_trust_ledger(run_config):
        if run_config.use_tolbert_model_artifacts:
            return {
                "overall_assessment": {"passed": True, "status": "trusted"},
                "gated_summary": {
                    "success_rate": 0.9,
                    "unsafe_ambiguous_rate": 0.0,
                    "hidden_side_effect_risk_rate": 0.0,
                    "success_hidden_side_effect_risk_rate": 0.0,
                },
                "family_assessments": {
                    "workflow": {"passed": True, "status": "trusted"},
                    "project": {"passed": True, "status": "trusted"},
                },
            }
        return {
            "overall_assessment": {"passed": True, "status": "trusted"},
            "gated_summary": {
                "success_rate": 0.8,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
            },
            "family_assessments": {
                "workflow": {"passed": True, "status": "trusted"},
                "project": {"passed": True, "status": "trusted"},
            },
        }

    class FakeDriftReport:
        def to_dict(self):
            return {
                "budget_reached": True,
                "waves_completed": 2,
                "worst_pass_rate_delta": 0.0,
                "worst_unsafe_ambiguous_rate_delta": 0.0,
                "worst_hidden_side_effect_rate_delta": 0.0,
                "worst_trust_success_rate_delta": 0.0,
                "worst_trust_unsafe_ambiguous_rate_delta": 0.0,
            }

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_unattended_trust_ledger", fake_build_trust_ledger)
    monkeypatch.setattr(module, "write_unattended_trust_ledger", lambda run_config: run_config.unattended_trust_ledger_path)
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: FakeDriftReport())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tolbert_liftoff.py",
            "--apply-routing",
        ],
    )
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    module.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["report"]["state"] == "retain"
    assert payload["report"]["family_takeover_evidence"]["workflow"]["decision"] == "promoted"
    assert payload["report"]["family_takeover_evidence"]["project"]["decision"] == "insufficient_proposal"
    assert payload["report"]["family_takeover_evidence"]["project"]["failure_reason"] == "missing proposal-selected commands"
    assert payload["report"]["proposal_metrics_by_benchmark_family"]["workflow"]["proposal_selected_steps"] >= 0
    assert payload["summary"]["family_takeover"]["project"]["failure_reason"] == "missing proposal-selected commands"
    assert payload["summary"]["proposal_gate_failure_reasons_by_benchmark_family"]["project"] == "missing proposal-selected commands"
    assert payload["takeover_drift"]["budget_reached"] is True
    assert "workflow" in artifact["runtime_policy"]["primary_benchmark_families"]
    assert artifact["hybrid_runtime"]["primary_enabled"] is True
    assert "demotion family=project reason=missing proposal-selected commands" in err_stream.getvalue()
    assert stream.getvalue().strip() == str(report_path)
