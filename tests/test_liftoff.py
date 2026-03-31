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
