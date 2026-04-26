from pathlib import Path
import importlib.util
import json
import sys


def _load_export_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "export_autonomy_evidence.py"
    spec = importlib.util.spec_from_file_location("export_autonomy_evidence", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retained_report() -> dict[str, object]:
    return {
        "cycle_id": "cycle:transition_model:20260423T223800Z:604f5429r6",
        "subsystem": "transition_model",
        "created_at": "2026-04-23T23:13:04Z",
        "final_state": "retain",
        "final_reason": "transition-model candidate improved retained bad-transition guidance without broader regression",
        "decision_state": {
            "decision_owner": "child_native",
            "decision_conversion_state": "runtime_managed",
            "retention_state": "retain",
            "retention_basis": "transition-model candidate improved retained bad-transition guidance without broader regression",
            "closeout_mode": "natural",
        },
        "phase_gate_report": {"passed": True, "failures": []},
        "compatibility": {"compatible": True, "violations": []},
        "candidate_isolation_summary": {"runtime_managed_artifact_path": True},
        "baseline_metrics": {"pass_rate": 1.0},
        "candidate_metrics": {"pass_rate": 1.0},
        "evidence": {
            "pass_rate_delta": 0.0,
            "average_step_delta": 0.0,
            "family_pass_rate_delta": {
                "integration": 0.0,
                "project": 0.0,
                "repository": 0.0,
                "repo_chore": 0.0,
                "repo_sandbox": 0.0,
            },
            "transition_model_improvement_count": 5,
            "transition_signature_count": 13,
            "confirmation_run_count": 2,
            "confirmation_paired_task_non_regression_rate_lower_bound": 0.9411764705882353,
            "confirmation_paired_trace_non_regression_rate_lower_bound": 0.9411764705882353,
            "confirmation_paired_trajectory_non_regression_rate_lower_bound": 0.9411764705882353,
            "confirmation_regressed_trace_task_count": 0,
            "confirmation_regressed_trajectory_task_count": 0,
        },
        "active_artifact_path": "trajectories/transition_model/transition_model_proposals.json",
        "candidate_artifact_path": "trajectories/improvement/candidates/transition_model/candidate.json",
        "artifact_sha256": "abc",
        "artifact_lifecycle_state": "retained",
    }


def test_build_a4_evidence_packet_marks_retained_r6_shape_supported():
    module = _load_export_module()

    packet = module.build_a4_evidence_packet(
        _retained_report(),
        source_report_path="report.json",
        source_report_sha256="report-sha",
    )

    assert packet["claim"]["level"] == "A4"
    assert packet["claim"]["scope"] == "narrow_direct_transition_model_route"
    assert packet["claim"]["status"] == "supported"
    assert all(packet["gates"].values())
    assert packet["source"]["cycle_id"] == "cycle:transition_model:20260423T223800Z:604f5429r6"
    assert packet["source"]["source_report_sha256"] == "report-sha"
    assert packet["decision_state"]["decision_owner"] == "child_native"
    assert packet["decision_state"]["decision_conversion_state"] == "runtime_managed"
    assert packet["decision_state"]["closeout_mode"] == "natural"
    assert packet["required_families"]["missing"] == []
    assert packet["metrics"]["transition_model_improvement_count"] == 5
    assert packet["metrics"]["regressed_trace_task_count"] == 0
    assert packet["metrics"]["regressed_trajectory_task_count"] == 0


def test_export_autonomy_evidence_cli_writes_packet(tmp_path, monkeypatch, capsys):
    module = _load_export_module()
    report_path = tmp_path / "cycle_report.json"
    output_path = tmp_path / "a4_packet.json"
    report_path.write_text(json.dumps(_retained_report()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_autonomy_evidence.py",
            "--cycle-report",
            str(report_path),
            "--output-json",
            str(output_path),
        ],
    )

    module.main()

    packet = json.loads(output_path.read_text(encoding="utf-8"))
    assert packet["claim"]["status"] == "supported"
    assert f"output_json={output_path}" in capsys.readouterr().out


def test_verify_a4_evidence_packet_accepts_matching_packet():
    module = _load_export_module()
    packet = module.build_a4_evidence_packet(
        _retained_report(),
        source_report_path="report.json",
        source_report_sha256="report-sha",
    )

    assert module.verify_a4_evidence_packet(packet, packet) == []


def test_verify_a4_evidence_packet_rejects_stale_source_hash():
    module = _load_export_module()
    expected = module.build_a4_evidence_packet(
        _retained_report(),
        source_report_path="report.json",
        source_report_sha256="new-sha",
    )
    stale = {
        **expected,
        "source": {
            **expected["source"],
            "source_report_sha256": "old-sha",
        },
    }

    failures = module.verify_a4_evidence_packet(stale, expected)

    assert any("source.source_report_sha256 mismatch" in failure for failure in failures)


def test_build_a4_evidence_packet_exposes_missing_gate():
    module = _load_export_module()
    report = _retained_report()
    report["decision_state"] = {
        **report["decision_state"],
        "closeout_mode": "forced_reject",
    }

    packet = module.build_a4_evidence_packet(report, source_report_path="report.json")

    assert packet["claim"]["status"] == "not_supported"
    assert packet["gates"]["natural_closeout"] is False


def test_verify_static_autonomy_packet_accepts_a5_substrate_packet():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A5_substrate",
            "scope": "isolated_mock_delegated_queue",
            "status": "supported",
            "summary": "Queue substrate check.",
        },
        "evidence": {
            "focused_tests": {
                "command": "pytest -q tests/test_job_queue.py -k queue",
                "result": "4 passed",
            },
            "isolated_cli_run": {"state_counts": {"completed": 1}},
        },
        "open_limits": ["Not an A5 claim."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_accepts_a4_packet_shape():
    module = _load_export_module()
    packet = module.build_a4_evidence_packet(
        _retained_report(),
        source_report_path="report.json",
        source_report_sha256="report-sha",
    )

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_a5_without_cli_evidence():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A5_substrate",
            "scope": "isolated_mock_delegated_queue",
            "status": "supported",
            "summary": "Queue substrate check.",
        },
        "evidence": {
            "focused_tests": {
                "command": "pytest -q tests/test_job_queue.py -k queue",
                "result": "4 passed",
            },
        },
        "open_limits": ["Not an A5 claim."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A5_substrate requires at least one isolated_cli_* evidence block" in failures


def test_verify_static_autonomy_packet_accepts_full_a5_confirmation_packet():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A5",
            "scope": "production_role_confirmation",
            "status": "supported",
            "summary": "Production role confirmation.",
        },
        "evidence": {
            "a5_confirmation": {
                "role_duration_seconds": 7200,
                "product_native_intake": True,
                "product_user_workstream_intake": True,
                "interruption_resume_proven": True,
                "trust_status": "trusted",
                "active_lease_count": 0,
                "required_family_counted_gated_report_counts": {
                    "integration": 2,
                    "project": 2,
                    "repository": 2,
                    "repo_chore": 2,
                    "repo_sandbox": 2,
                },
                "native_role_closeout": {
                    "closeout_ready": True,
                    "closeout_mode": "queue_empty_trusted",
                    "operator_steering_required": False,
                    "active_leases": 0,
                },
            }
        },
        "open_limits": ["Does not imply A6."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_full_a5_short_role():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A5",
            "scope": "production_role_confirmation",
            "status": "supported",
            "summary": "Production role confirmation.",
        },
        "evidence": {
            "a5_confirmation": {
                "role_duration_seconds": 769,
                "product_native_intake": True,
                "product_user_workstream_intake": True,
                "interruption_resume_proven": True,
                "trust_status": "trusted",
                "active_lease_count": 0,
                "required_family_counted_gated_report_counts": {
                    "integration": 2,
                    "project": 2,
                    "repository": 2,
                    "repo_chore": 2,
                    "repo_sandbox": 2,
                },
                "native_role_closeout": {
                    "closeout_ready": True,
                    "closeout_mode": "queue_empty_trusted",
                    "operator_steering_required": False,
                    "active_leases": 0,
                },
            }
        },
        "open_limits": ["Too short."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A5 role_duration_seconds must be at least 7200" in failures


def test_verify_static_autonomy_packet_accepts_a6_confirmation_packet():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A6",
            "scope": "retained_self_improvement_confirmation",
            "status": "supported",
            "summary": "Retained self-improvement confirmation.",
        },
        "evidence": {
            "a6_confirmation": {
                "retained_gain_runs": 2,
                "candidate_baseline_report_count": 2,
                "regression_gates_survived": True,
                "retained_changes_affect_runtime": True,
                "autonomous_compounding_claim_ready": True,
                "non_collapsing_repeated_runs": True,
            }
        },
        "open_limits": ["Does not imply A7."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_a6_single_retained_gain():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A6",
            "scope": "retained_self_improvement_confirmation",
            "status": "supported",
            "summary": "Retained self-improvement confirmation.",
        },
        "evidence": {
            "a6_confirmation": {
                "retained_gain_runs": 1,
                "candidate_baseline_report_count": 2,
                "regression_gates_survived": True,
                "retained_changes_affect_runtime": True,
                "autonomous_compounding_claim_ready": True,
                "non_collapsing_repeated_runs": True,
            }
        },
        "open_limits": ["One retained gain is not repeated A6 evidence."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A6 retained_gain_runs must be at least 2" in failures
