from pathlib import Path
import importlib.util
import json
from io import StringIO
import sys

from agent_kernel.config import KernelConfig
from evals.metrics import EvalMetrics


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_report_skips_storage_governance(tmp_path, monkeypatch):
    module = _load_script("run_tolbert_liftoff_loop.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()
    captured: dict[str, object] = {}

    def fake_atomic_write_json(path, payload, *, config=None, govern_storage=True):
        captured["path"] = path
        captured["payload"] = payload
        captured["config"] = config
        captured["govern_storage"] = govern_storage
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "atomic_write_json", fake_atomic_write_json)

    report_path = config.improvement_reports_dir / "liftoff_report.json"
    payload = {"report_kind": "tolbert_liftoff_loop_report", "status": "running"}
    module._write_report(report_path, payload, config=config)

    assert captured["path"] == report_path
    assert captured["payload"] == payload
    assert captured["config"] == config
    assert captured["govern_storage"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


def test_apply_routing_to_artifact_skips_storage_governance(tmp_path, monkeypatch):
    module = _load_script("run_tolbert_liftoff_loop.py")
    config = KernelConfig(
        candidate_artifacts_root=tmp_path / "candidates",
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )
    config.ensure_directories()
    captured: dict[str, object] = {}

    def fake_atomic_write_json(path, payload, *, config=None, govern_storage=True):
        captured["path"] = path
        captured["payload"] = payload
        captured["config"] = config
        captured["govern_storage"] = govern_storage
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = type(
        "Report",
        (),
        {
            "state": "retain",
            "primary_takeover_families": ["workflow"],
            "shadow_only_families": ["project"],
        },
    )()
    artifact_path = config.candidate_artifacts_root / "candidate.json"
    artifact = {
        "runtime_policy": {
            "shadow_benchmark_families": [],
            "primary_benchmark_families": [],
        },
        "hybrid_runtime": {},
    }

    monkeypatch.setattr(module, "atomic_write_json", fake_atomic_write_json)

    module._apply_routing_to_artifact(artifact_path, artifact, report, config=config)

    assert captured["path"] == artifact_path
    assert captured["config"] == config
    assert captured["govern_storage"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["runtime_policy"]["primary_benchmark_families"] == [
        "workflow"
    ]


def test_run_tolbert_liftoff_loop_writes_readiness_report(tmp_path, monkeypatch):
    module = _load_script("run_tolbert_liftoff_loop.py")
    candidate_root = tmp_path / "improvement" / "candidates"
    reports_dir = tmp_path / "improvement" / "reports"
    retained_artifact_path = tmp_path / "tolbert_model" / "retained_artifact.json"
    config = KernelConfig(
        candidate_artifacts_root=candidate_root,
        improvement_reports_dir=reports_dir,
        tolbert_model_artifact_path=retained_artifact_path,
        trajectories_root=tmp_path / "episodes",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )

    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps({"report_kind": "improvement_campaign_report"}),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, kwargs
        assert "run_repeated_improvement_cycles.py" in " ".join(cmd)
        assert "--task-limit" in cmd
        assert "64" in cmd
        assert "--tolbert-device" in cmd
        assert "cuda:2" in cmd
        assert env["AGENT_KERNEL_USE_TOLBERT_MODEL_ARTIFACTS"] == "1"
        assert env["AGENT_KERNEL_TOLBERT_DEVICE"] == "cuda:2"
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    metrics_calls = {"count": 0}

    seen_task_limits: list[int | None] = []

    def fake_run_eval(*, config, progress_label, task_limit=None, **kwargs):
        del config
        assert kwargs["write_unattended_reports"] is True
        assert kwargs["prefer_low_cost_tasks"] is True
        seen_task_limits.append(task_limit)
        metrics_calls["count"] += 1
        if progress_label == "tolbert_liftoff_baseline":
            return EvalMetrics(
                total=10,
                passed=8,
                average_steps=1.4,
                proposal_selected_steps=0,
                novel_command_steps=0,
                novel_valid_command_steps=0,
                generated_total=2,
                generated_passed=1,
                generated_by_kind={"failure_recovery": 1},
                generated_passed_by_kind={"failure_recovery": 1},
                total_by_difficulty={"long_horizon": 4},
                passed_by_difficulty={"long_horizon": 3},
                total_by_benchmark_family={"workflow": 6, "project": 4},
                passed_by_benchmark_family={"workflow": 5, "project": 3},
                proposal_metrics_by_benchmark_family={
                    "workflow": {
                        "task_count": 6,
                        "proposal_selected_steps": 0,
                        "novel_valid_command_steps": 0,
                        "novel_valid_command_rate": 0.0,
                    },
                    "project": {
                        "task_count": 4,
                        "proposal_selected_steps": 0,
                        "novel_valid_command_steps": 0,
                        "novel_valid_command_rate": 0.0,
                    },
                },
                proposal_metrics_by_difficulty={
                    "long_horizon": {
                        "task_count": 4,
                        "proposal_selected_steps": 1,
                        "novel_valid_command_steps": 1,
                        "novel_valid_command_rate": 0.5,
                    }
                },
                world_feedback_summary={
                    "step_count": 8,
                    "progress_calibration_mae": 0.30,
                    "risk_calibration_mae": 0.28,
                    "decoder_progress_calibration_mae": 0.32,
                    "decoder_risk_calibration_mae": 0.27,
                },
                world_feedback_by_benchmark_family={
                    "workflow": {"step_count": 5, "progress_calibration_mae": 0.25, "risk_calibration_mae": 0.22},
                    "project": {"step_count": 3, "progress_calibration_mae": 0.38, "risk_calibration_mae": 0.36},
                },
                world_feedback_by_difficulty={
                    "long_horizon": {"step_count": 4, "progress_calibration_mae": 0.30, "risk_calibration_mae": 0.24},
                },
                long_horizon_persistence_summary={
                    "long_horizon_steps": 6,
                    "productive_long_horizon_step_rate": 0.5,
                    "recovery_response_rate": 0.25,
                    "horizon_drop_rate": 0.1667,
                    "long_horizon_success_rate": 0.75,
                },
                transfer_alignment_summary={
                    "transfer_step_count": 4,
                    "verified_transfer_step_rate": 0.5,
                    "trusted_retrieval_alignment_mean": 0.3,
                    "graph_environment_alignment_mean": 0.1,
                    "safe_transfer_step_rate": 0.5,
                    "unsafe_transfer_step_rate": 0.5,
                },
            )
        return EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.1,
            proposal_selected_steps=3,
            novel_command_steps=3,
            novel_valid_command_steps=2,
            generated_total=2,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 1},
            generated_passed_by_kind={"failure_recovery": 1},
            total_by_difficulty={"long_horizon": 4},
            passed_by_difficulty={"long_horizon": 4},
            total_by_benchmark_family={"workflow": 6, "project": 4},
            passed_by_benchmark_family={"workflow": 6, "project": 3},
            tolbert_shadow_episodes=4,
            tolbert_shadow_episodes_by_benchmark_family={"workflow": 3, "project": 1},
            proposal_metrics_by_benchmark_family={
                "workflow": {
                    "task_count": 6,
                    "proposal_selected_steps": 3,
                    "novel_valid_command_steps": 2,
                    "novel_valid_command_rate": 0.6667,
                },
                "project": {
                    "task_count": 4,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
            },
            proposal_metrics_by_difficulty={
                "long_horizon": {
                    "task_count": 4,
                    "proposal_selected_steps": 3,
                    "novel_valid_command_steps": 2,
                    "novel_valid_command_rate": 1.0,
                }
            },
            world_feedback_summary={
                "step_count": 10,
                "progress_calibration_mae": 0.18,
                "risk_calibration_mae": 0.16,
                "decoder_progress_calibration_mae": 0.14,
                "decoder_risk_calibration_mae": 0.12,
            },
            world_feedback_by_benchmark_family={
                "workflow": {"step_count": 6, "progress_calibration_mae": 0.14, "risk_calibration_mae": 0.12},
                "project": {"step_count": 4, "progress_calibration_mae": 0.24, "risk_calibration_mae": 0.22},
            },
            world_feedback_by_difficulty={
                "long_horizon": {"step_count": 4, "progress_calibration_mae": 0.18, "risk_calibration_mae": 0.14},
            },
            long_horizon_persistence_summary={
                "long_horizon_steps": 8,
                "productive_long_horizon_step_rate": 0.75,
                "recovery_response_rate": 0.5,
                "horizon_drop_rate": 0.0,
                "long_horizon_success_rate": 1.0,
            },
            transfer_alignment_summary={
                "transfer_step_count": 4,
                "verified_transfer_step_rate": 0.75,
                "trusted_retrieval_alignment_mean": 0.6,
                "graph_environment_alignment_mean": 0.4,
                "safe_transfer_step_rate": 1.0,
                "unsafe_transfer_step_rate": 0.0,
            },
        )

    def fake_build_candidate_artifact(*, config, repo_root, output_dir, metrics, focus, current_payload):
        del config, repo_root, metrics, current_payload
        checkpoint_path = output_dir / "training" / "checkpoints" / "tolbert_epoch2.pt"
        cache_path = output_dir / "retrieval_cache" / "tolbert_epoch2.pt"
        universal_manifest_path = output_dir / "universal_dataset" / "universal_decoder_dataset_manifest.json"
        universal_bundle_path = output_dir / "universal_runtime" / "hybrid_bundle_manifest.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        universal_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        universal_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("pt", encoding="utf-8")
        cache_path.write_text("cache", encoding="utf-8")
        universal_manifest_path.write_text(json.dumps({"artifact_kind": "tolbert_universal_decoder_dataset"}), encoding="utf-8")
        universal_bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
        return {
            "artifact_kind": "tolbert_model_bundle",
            "generation_focus": focus or "balanced",
            "runtime_policy": {
                "shadow_benchmark_families": ["workflow", "project"],
                "primary_benchmark_families": [],
            },
            "liftoff_gate": {
                "min_pass_rate_delta": 0.0,
                "max_step_regression": 0.0,
                "max_regressed_families": 0,
                "require_generated_lane_non_regression": True,
                "require_failure_recovery_non_regression": True,
                "require_shadow_signal": True,
                "min_shadow_episodes_per_promoted_family": 1,
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
            "build_policy": {
                "allow_kernel_autobuild": True,
                "allow_kernel_rebuild": False,
                "ready_total_examples": 640,
                "ready_synthetic_examples": 96,
                "min_total_examples": 512,
                "min_synthetic_examples": 64,
            },
            "dataset_manifest": {
                "total_examples": 640,
                "trajectory_examples": 500,
                "synthetic_trajectory_examples": 96,
                "transition_failure_examples": 120,
                "discovered_task_examples": 32,
                "verifier_label_examples": 12,
                "benchmark_families": ["workflow", "project"],
            },
            "universal_dataset_manifest": {
                "artifact_kind": "tolbert_universal_decoder_dataset",
                "manifest_path": str(universal_manifest_path),
                "train_examples": 18,
                "eval_examples": 2,
            },
            "job_records": [{"job_id": "job:train", "state": "completed", "outcome": "success"}],
            "external_training_backends": [{"backend_id": "tolbert"}],
            "runtime_paths": {
                "checkpoint_path": str(checkpoint_path),
                "cache_paths": [str(cache_path)],
                "universal_bundle_manifest_path": str(universal_bundle_path),
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
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "build_tolbert_model_candidate_artifact", fake_build_candidate_artifact)
    monkeypatch.setattr(module, "run_takeover_drift_eval", lambda **kwargs: FakeDriftReport())
    monkeypatch.setattr(
        module,
        "evaluate_universal_decoder_against_seed",
        lambda **kwargs: type(
            "Report",
            (),
            {
                "to_dict": lambda self: {
                    "artifact_kind": "tolbert_universal_decoder_eval",
                    "hybrid_exact_match_rate": 0.5,
                    "baseline_exact_match_rate": 0.75,
                    "hybrid_token_f1": 0.6,
                    "baseline_token_f1": 0.8,
                    "example_count": 2,
                    "slices": {
                        "docs_markdown": {
                            "source_type": "docs_markdown",
                            "example_count": 1,
                            "hybrid_exact_match_rate": 1.0,
                            "baseline_exact_match_rate": 1.0,
                            "hybrid_token_f1": 1.0,
                            "baseline_token_f1": 1.0,
                        }
                    },
                }
            },
        )(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tolbert_liftoff_loop.py",
            "--cycles",
            "2",
            "--task-limit",
            "64",
            "--tolbert-device",
            "cuda:2",
            "--apply-routing",
            "--promote-on-retain",
        ],
    )
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert metrics_calls["count"] == 2
    assert seen_task_limits == [64, 64]
    assert payload["report_kind"] == "tolbert_liftoff_loop_report"
    assert payload["campaign"]["completed"] is True
    assert payload["campaign"]["campaign_report_path"] == str(campaign_report_path)
    assert payload["dataset_readiness"]["allow_kernel_autobuild"] is True
    assert payload["training"]["checkpoint_exists"] is True
    assert payload["training"]["cache_exists"] is True
    assert payload["comparison"]["pass_rate_delta"] > 0.0
    assert payload["comparison"]["proposal_selected_steps_delta"] == 3
    assert payload["comparison"]["novel_valid_command_rate_delta"] > 0.0
    assert payload["comparison"]["proposal_metrics_by_benchmark_family"]["workflow"]["proposal_selected_steps_delta"] == 3
    assert payload["comparison"]["proposal_metrics_by_difficulty"]["long_horizon"]["proposal_selected_steps_delta"] == 2
    assert payload["comparison"]["world_feedback_calibration"]["progress_calibration_mae_gain"] == 0.12
    assert payload["comparison"]["world_feedback_by_benchmark_family"]["workflow"]["progress_calibration_mae_gain"] == 0.11
    assert payload["comparison"]["long_horizon"]["pass_rate_delta"] == 0.25
    assert payload["comparison"]["long_horizon"]["world_feedback"]["progress_calibration_mae_gain"] == 0.12
    assert payload["comparison"]["long_horizon"]["persistence"]["productive_long_horizon_step_rate_delta"] == 0.25
    assert payload["comparison"]["transfer_alignment"]["graph_environment_alignment_mean_delta"] == 0.3
    assert payload["comparison"]["transfer_alignment"]["safe_transfer_step_rate_delta"] == 0.5
    assert payload["comparison"]["family_takeover_summary"]["project"]["failure_reason"] == "missing proposal-selected commands"
    assert (
        payload["comparison"]["proposal_gate_failure_reasons_by_benchmark_family"]["project"]
        == "missing proposal-selected commands"
    )
    assert payload["liftoff_report"]["family_takeover_evidence"]["workflow"]["decision"] in {"promoted", "shadow_only"}
    assert payload["liftoff_report"]["family_takeover_evidence"]["project"]["decision"] == "insufficient_proposal"
    assert payload["takeover_drift"]["budget_reached"] is True
    assert payload["universal_decoder_eval"]["available"] is True
    assert payload["universal_decoder_eval"]["slices"]["docs_markdown"]["hybrid_exact_match_rate"] == 1.0
    assert payload["liftoff_report"]["state"] in {"retain", "shadow_only"}
    assert "demotion family=project reason=missing proposal-selected commands" in err_stream.getvalue()


def test_run_tolbert_liftoff_loop_resume_preserves_metrics_and_trust_payloads(tmp_path, monkeypatch):
    module = _load_script("run_tolbert_liftoff_loop.py")
    candidate_root = tmp_path / "improvement" / "candidates"
    reports_dir = tmp_path / "improvement" / "reports"
    retained_artifact_path = tmp_path / "tolbert_model" / "retained_artifact.json"
    config = KernelConfig(
        candidate_artifacts_root=candidate_root,
        improvement_reports_dir=reports_dir,
        tolbert_model_artifact_path=retained_artifact_path,
        trajectories_root=tmp_path / "episodes",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )
    config.ensure_directories()
    report_path = reports_dir / "liftoff_resume.json"
    candidate_artifact_path = candidate_root / "resume_candidate.json"
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "liftoff_gate": {
                    "require_takeover_drift_eval": False,
                    "takeover_drift_step_budget": 0,
                },
                "dataset_manifest": {"benchmark_families": ["repository"]},
                "build_policy": {"allow_kernel_autobuild": True},
                "runtime_paths": {},
            }
        ),
        encoding="utf-8",
    )

    baseline_metrics = EvalMetrics(
        total=4,
        passed=2,
        average_steps=2.0,
        total_by_difficulty={"long_horizon": 2},
        passed_by_difficulty={"long_horizon": 1},
        total_by_benchmark_family={"repository": 4},
        passed_by_benchmark_family={"repository": 2},
        long_horizon_persistence_summary={
            "long_horizon_steps": 4,
            "productive_long_horizon_step_rate": 0.25,
        },
        contract_clean_failure_recovery_summary={
            "task_count": 1,
            "long_horizon_pass_rate": 0.0,
        },
        transfer_alignment_summary={
            "verified_transfer_step_rate": 0.25,
            "graph_environment_alignment_mean": 0.1,
        },
    )
    candidate_metrics = EvalMetrics(
        total=4,
        passed=3,
        average_steps=1.5,
        total_by_difficulty={"long_horizon": 2},
        passed_by_difficulty={"long_horizon": 2},
        total_by_benchmark_family={"repository": 4},
        passed_by_benchmark_family={"repository": 3},
        long_horizon_persistence_summary={
            "long_horizon_steps": 5,
            "productive_long_horizon_step_rate": 0.8,
        },
        contract_clean_failure_recovery_summary={
            "task_count": 2,
            "long_horizon_pass_rate": 1.0,
        },
        transfer_alignment_summary={
            "verified_transfer_step_rate": 0.75,
            "graph_environment_alignment_mean": 0.6,
        },
    )
    report_path.write_text(
        json.dumps(
            {
                "report_kind": "tolbert_liftoff_loop_report",
                "campaign": {"completed": True},
                "candidate_artifact_path": str(candidate_artifact_path),
                "baseline_metrics": module._metric_summary(baseline_metrics),
                "candidate_metrics": module._metric_summary(candidate_metrics),
                "baseline_trust": {
                    "overall_summary": {"light_supervision_clean_success_count": 1},
                },
                "candidate_trust": {
                    "overall_summary": {"light_supervision_clean_success_count": 3},
                },
                "takeover_drift": {"budget_reached": True},
                "universal_decoder_eval": {"available": False},
            }
        ),
        encoding="utf-8",
    )

    def fail_run_eval(**kwargs):
        raise AssertionError(f"run_eval should not run on resume: {kwargs}")

    def fail_run_and_stream(*args, **kwargs):
        raise AssertionError(f"campaign should not run on resume: {args} {kwargs}")

    captured: dict[str, object] = {}

    def fake_build_liftoff_gate_report(
        *,
        candidate_metrics,
        baseline_metrics,
        artifact_payload,
        candidate_trust_ledger,
        baseline_trust_ledger,
        takeover_drift_report,
    ):
        assert artifact_payload["artifact_kind"] == "tolbert_model_bundle"
        assert baseline_metrics.long_horizon_persistence_summary["productive_long_horizon_step_rate"] == 0.25
        assert candidate_metrics.long_horizon_persistence_summary["productive_long_horizon_step_rate"] == 0.8
        assert baseline_metrics.contract_clean_failure_recovery_summary["task_count"] == 1
        assert candidate_metrics.transfer_alignment_summary["graph_environment_alignment_mean"] == 0.6
        assert baseline_trust_ledger["overall_summary"]["light_supervision_clean_success_count"] == 1
        assert candidate_trust_ledger["overall_summary"]["light_supervision_clean_success_count"] == 3
        assert takeover_drift_report == {"budget_reached": True}
        captured["called"] = True
        return type(
            "Report",
            (),
            {
                "state": "shadow_only",
                "reason": "resume_check",
                "primary_takeover_families": [],
                "shadow_only_families": ["repository"],
                "insufficient_proposal_families": [],
                "proposal_gate_failure_reasons_by_benchmark_family": {},
                "family_takeover_evidence": {},
                "to_dict": lambda self: {
                    "state": "shadow_only",
                    "reason": "resume_check",
                    "shadow_only_families": ["repository"],
                },
            },
        )()

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "run_eval", fail_run_eval)
    monkeypatch.setattr(module, "_run_and_stream", fail_run_and_stream)
    monkeypatch.setattr(module, "build_liftoff_gate_report", fake_build_liftoff_gate_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tolbert_liftoff_loop.py",
            "--report-path",
            str(report_path),
            "--resume",
            "--skip-improvement",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert captured["called"] is True
    assert payload["candidate_trust"]["overall_summary"]["light_supervision_clean_success_count"] == 3
    assert payload["takeover_drift"]["budget_reached"] is True
    assert payload["universal_decoder_eval"]["available"] is False
