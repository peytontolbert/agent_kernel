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


def test_verify_static_autonomy_packet_accepts_a7_readiness_packet():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7_readiness",
            "scope": "repeated_live_vllm_heldout_unfamiliar_transfer",
            "status": "supported",
            "summary": "Repeated held-out transfer readiness.",
        },
        "evidence": {
            "a7_readiness": {
                "rotation_count": 5,
                "non_identical_manifests": True,
                "total_held_out_frontier_reports": 30,
                "total_held_out_frontier_clean_successes": 30,
                "aggregate_clean_success_rate": 1.0,
                "families_repeated_in_both_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "families_repeated_across_all_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "distinct_frontier_slice_count": 30,
                "hard_hintless_rotation_count": 2,
                "hard_hintless_clean_success_count": 15,
                "hard_hintless_clean_success_rate": 1.0,
                "stateful_repair_rotation_count": 1,
                "stateful_repair_clean_success_count": 5,
                "stateful_repair_clean_success_rate": 1.0,
                "diagnostic_synthesis_rotation_count": 1,
                "diagnostic_synthesis_clean_success_count": 5,
                "diagnostic_synthesis_clean_success_rate": 1.0,
                "existing_path_only": True,
                "per_task_architecture_changes": False,
                "provider": {"provider": "vllm"},
            }
        },
        "open_limits": ["A7 readiness is not A7."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_a7_readiness_with_mock_provider():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7_readiness",
            "scope": "repeated_mock_heldout_unfamiliar_transfer",
            "status": "supported",
            "summary": "Repeated held-out transfer readiness.",
        },
        "evidence": {
            "a7_readiness": {
                "rotation_count": 2,
                "non_identical_manifests": True,
                "total_held_out_frontier_reports": 10,
                "aggregate_clean_success_rate": 1.0,
                "families_repeated_in_both_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "distinct_frontier_slice_count": 10,
                "existing_path_only": True,
                "per_task_architecture_changes": False,
                "provider": {"provider": "mock"},
            }
        },
        "open_limits": ["Mock provider cannot support A7 readiness."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A7_readiness provider.provider must be a live non-mock provider" in failures


def test_verify_static_autonomy_packet_rejects_a7_readiness_with_weak_hard_hintless_evidence():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7_readiness",
            "scope": "repeated_live_vllm_heldout_unfamiliar_transfer_with_hard_hintless_rotation",
            "status": "supported",
            "summary": "Repeated held-out transfer readiness with weak hard hintless evidence.",
        },
        "evidence": {
            "a7_readiness": {
                "rotation_count": 3,
                "non_identical_manifests": True,
                "total_held_out_frontier_reports": 15,
                "total_held_out_frontier_clean_successes": 15,
                "aggregate_clean_success_rate": 1.0,
                "families_repeated_in_both_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "distinct_frontier_slice_count": 15,
                "hard_hintless_rotation_count": 0,
                "hard_hintless_clean_success_count": 4,
                "hard_hintless_clean_success_rate": 0.6,
                "existing_path_only": True,
                "per_task_architecture_changes": False,
                "provider": {"provider": "vllm"},
            }
        },
        "open_limits": ["A7 readiness is not A7."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A7_readiness hard_hintless_rotation_count must be at least 2" in failures
    assert "A7_readiness hard_hintless_clean_success_count must be at least 15" in failures
    assert "A7_readiness hard_hintless_clean_success_rate must be at least 0.9" in failures


def test_verify_static_autonomy_packet_rejects_a7_readiness_with_weak_stateful_repair_evidence():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7_readiness",
            "scope": "repeated_live_vllm_heldout_unfamiliar_transfer_with_stateful_repair",
            "status": "supported",
            "summary": "Repeated held-out transfer readiness with weak stateful repair evidence.",
        },
        "evidence": {
            "a7_readiness": {
                "rotation_count": 5,
                "non_identical_manifests": True,
                "total_held_out_frontier_reports": 30,
                "total_held_out_frontier_clean_successes": 30,
                "aggregate_clean_success_rate": 1.0,
                "families_repeated_in_both_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "distinct_frontier_slice_count": 30,
                "stateful_repair_rotation_count": 0,
                "stateful_repair_clean_success_count": 4,
                "stateful_repair_clean_success_rate": 0.6,
                "existing_path_only": True,
                "per_task_architecture_changes": False,
                "provider": {"provider": "vllm"},
            }
        },
        "open_limits": ["A7 readiness is not A7."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A7_readiness stateful_repair_rotation_count must be at least 1" in failures
    assert "A7_readiness stateful_repair_clean_success_count must be at least 5" in failures
    assert "A7_readiness stateful_repair_clean_success_rate must be at least 0.9" in failures


def test_verify_static_autonomy_packet_rejects_a7_readiness_with_weak_diagnostic_synthesis_evidence():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7_readiness",
            "scope": "repeated_live_vllm_heldout_unfamiliar_transfer_with_diagnostic_synthesis",
            "status": "supported",
            "summary": "Repeated held-out transfer readiness with weak diagnostic synthesis evidence.",
        },
        "evidence": {
            "a7_readiness": {
                "rotation_count": 6,
                "non_identical_manifests": True,
                "total_held_out_frontier_reports": 35,
                "total_held_out_frontier_clean_successes": 35,
                "aggregate_clean_success_rate": 1.0,
                "families_repeated_in_both_rotations": [
                    "integration",
                    "project",
                    "repository",
                    "repo_chore",
                    "repo_sandbox",
                ],
                "distinct_frontier_slice_count": 35,
                "diagnostic_synthesis_rotation_count": 0,
                "diagnostic_synthesis_clean_success_count": 4,
                "diagnostic_synthesis_clean_success_rate": 0.6,
                "existing_path_only": True,
                "per_task_architecture_changes": False,
                "provider": {"provider": "vllm"},
            }
        },
        "open_limits": ["A7 readiness is not A7."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A7_readiness diagnostic_synthesis_rotation_count must be at least 1" in failures
    assert "A7_readiness diagnostic_synthesis_clean_success_count must be at least 5" in failures
    assert "A7_readiness diagnostic_synthesis_clean_success_rate must be at least 0.9" in failures


def test_verify_static_autonomy_packet_accepts_full_a7_gate_shape():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7",
            "scope": "declared_coding_task_universe_unfamiliar_transfer",
            "status": "supported",
            "summary": "Broad unfamiliar transfer with limited redesign and baseline comparison.",
        },
        "evidence": {
            "a7": {
                "unfamiliar_domain_slice_count": 5,
                "broad_transfer_clean_success_rate": 0.85,
                "strong_baseline_win_rate": 0.55,
                "long_horizon_transfer_slice_count": 1,
                "limited_redesign": True,
                "conservative_comparison_report": True,
                "full_a7_claim_ready": True,
            }
        },
        "open_limits": ["A7 is coding-task-universe scoped, not literature-wide AGI."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_full_a7_without_frontier_gates():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A7",
            "scope": "local_readiness_only",
            "status": "supported",
            "summary": "Readiness is being incorrectly promoted to A7.",
        },
        "evidence": {
            "a7": {
                "unfamiliar_domain_slice_count": 2,
                "broad_transfer_clean_success_rate": 0.6,
                "strong_baseline_win_rate": 0.0,
                "long_horizon_transfer_slice_count": 0,
                "limited_redesign": False,
                "conservative_comparison_report": False,
                "full_a7_claim_ready": False,
            }
        },
        "open_limits": ["Should reject weak A7 evidence."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A7 unfamiliar_domain_slice_count must be at least 5" in failures
    assert "A7 broad_transfer_clean_success_rate must be at least 0.8" in failures
    assert "A7 strong_baseline_win_rate must be at least 0.5" in failures
    assert "A7 long_horizon_transfer_slice_count must be at least 1" in failures
    assert "A7 limited_redesign must be true" in failures
    assert "A7 conservative_comparison_report must be true" in failures
    assert "A7 full_a7_claim_ready must be true" in failures


def test_verify_static_autonomy_packet_accepts_a8_gate_shape():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A8",
            "scope": "declared_task_universe_superhuman_verified_compounding",
            "status": "supported",
            "summary": "Decisive strong-baseline outperformance with verified recursive compounding.",
        },
        "evidence": {
            "a8": {
                "strong_baseline_comparison_slice_count": 5,
                "superhuman_coding_window_count": 3,
                "superhuman_coding_task_count": 100,
                "strong_human_baseline_win_rate": 0.85,
                "strong_human_baseline_win_rate_lower_bound": 0.65,
                "unfamiliar_domain_slice_count": 5,
                "long_horizon_transfer_slice_count": 3,
                "recursive_compounding_retained_gain_runs": 5,
                "recursive_compounding_window_count": 3,
                "regression_rate": 0.01,
                "codeforces_rating_equivalent": 3000,
                "mle_bench_gold_medal_rate": 0.2,
                "swe_bench_verified_resolve_rate": 0.8,
                "swe_rebench_resolve_rate": 0.6,
                "re_bench_human_expert_win_rate": 0.5,
                "decisive_outperformance": True,
                "conservative_comparison_report": True,
                "verified_recursive_compounding": True,
                "sustained_superhuman_coding": True,
                "a8_claim_ready": True,
            }
        },
        "open_limits": ["A8 claim is scoped to the declared task universe."],
    }

    assert module.verify_static_autonomy_packet(packet) == []


def test_verify_static_autonomy_packet_rejects_a8_without_superhuman_and_compounding_gates():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "claim": {
            "level": "A8",
            "scope": "readiness_promoted_without_baseline",
            "status": "supported",
            "summary": "A8 must not be inferred from local held-out readiness.",
        },
        "evidence": {
            "a8": {
                "strong_baseline_comparison_slice_count": 0,
                "superhuman_coding_window_count": 1,
                "superhuman_coding_task_count": 30,
                "strong_human_baseline_win_rate": 0.4,
                "strong_human_baseline_win_rate_lower_bound": 0.2,
                "unfamiliar_domain_slice_count": 2,
                "long_horizon_transfer_slice_count": 0,
                "recursive_compounding_retained_gain_runs": 1,
                "recursive_compounding_window_count": 1,
                "regression_rate": 0.1,
                "codeforces_rating_equivalent": 2400,
                "mle_bench_gold_medal_rate": 0.0,
                "swe_bench_verified_resolve_rate": 0.4,
                "swe_rebench_resolve_rate": 0.3,
                "re_bench_human_expert_win_rate": 0.0,
                "decisive_outperformance": False,
                "conservative_comparison_report": False,
                "verified_recursive_compounding": False,
                "sustained_superhuman_coding": False,
                "a8_claim_ready": False,
            }
        },
        "open_limits": ["Should reject weak A8 evidence."],
    }

    failures = module.verify_static_autonomy_packet(packet)

    assert "A8 strong_baseline_comparison_slice_count must be at least 5" in failures
    assert "A8 superhuman_coding_window_count must be at least 3" in failures
    assert "A8 superhuman_coding_task_count must be at least 100" in failures
    assert "A8 strong_human_baseline_win_rate must be at least 0.8" in failures
    assert "A8 strong_human_baseline_win_rate_lower_bound must be at least 0.6" in failures
    assert "A8 unfamiliar_domain_slice_count must be at least 5" in failures
    assert "A8 long_horizon_transfer_slice_count must be at least 3" in failures
    assert "A8 recursive_compounding_retained_gain_runs must be at least 5" in failures
    assert "A8 recursive_compounding_window_count must be at least 3" in failures
    assert "A8 regression_rate must be at most 0.02" in failures
    assert "A8 codeforces_rating_equivalent must be at least 3000" in failures
    assert "A8 mle_bench_gold_medal_rate must be at least 0.2" in failures
    assert "A8 swe_bench_verified_resolve_rate must be at least 0.8" in failures
    assert "A8 swe_rebench_resolve_rate must be at least 0.6" in failures
    assert "A8 re_bench_human_expert_win_rate must be at least 0.5" in failures
    assert "A8 decisive_outperformance must be true" in failures
    assert "A8 conservative_comparison_report must be true" in failures
    assert "A8 verified_recursive_compounding must be true" in failures
    assert "A8 sustained_superhuman_coding must be true" in failures
    assert "A8 a8_claim_ready must be true" in failures


def test_verify_a8_benchmark_target_packet_accepts_current_target_shape():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "a8_coding_superhuman_target_packet",
        "target": {
            "level": "A8",
            "domain": "coding",
            "thresholds": {
                "codeforces_rating_equivalent": 3000,
                "mle_bench_gold_medal_rate": 0.2,
                "swe_bench_verified_resolve_rate": 0.8,
                "swe_rebench_resolve_rate": 0.6,
                "re_bench_human_expert_win_rate": 0.5,
                "strong_baseline_comparison_slice_count": 5,
                "superhuman_coding_window_count": 3,
                "superhuman_coding_task_count": 100,
                "strong_human_baseline_win_rate": 0.8,
                "strong_human_baseline_win_rate_lower_bound": 0.6,
                "unfamiliar_domain_slice_count": 5,
                "long_horizon_transfer_slice_count": 3,
                "recursive_compounding_retained_gain_runs": 5,
                "recursive_compounding_window_count": 3,
                "max_regression_rate": 0.02,
            },
            "benchmark_sources": {
                "codeforces": {"evidence_metric": "rating_equivalent"},
                "mle_bench": {"evidence_metric": "gold_medal_rate"},
                "swe_bench_verified": {"evidence_metric": "resolve_rate"},
                "swe_rebench": {"evidence_metric": "resolve_rate"},
                "re_bench": {"evidence_metric": "human_expert_win_rate"},
            },
            "acceptance_policy": {
                "requires_sustained_windows": True,
                "requires_conservative_comparison": True,
                "requires_recursive_compounding": True,
                "forbids_readiness_only_promotion": True,
            },
        },
    }

    assert module.verify_a8_benchmark_target_packet(packet) == []


def test_verify_a8_benchmark_target_packet_rejects_weakened_targets():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "a8_coding_superhuman_target_packet",
        "target": {
            "level": "A8",
            "domain": "coding",
            "thresholds": {
                "codeforces_rating_equivalent": 2400,
                "mle_bench_gold_medal_rate": 0.0,
                "swe_bench_verified_resolve_rate": 0.8,
                "swe_rebench_resolve_rate": 0.6,
                "re_bench_human_expert_win_rate": 0.5,
                "strong_baseline_comparison_slice_count": 1,
                "superhuman_coding_window_count": 1,
                "superhuman_coding_task_count": 30,
                "strong_human_baseline_win_rate": 0.8,
                "strong_human_baseline_win_rate_lower_bound": 0.6,
                "unfamiliar_domain_slice_count": 5,
                "long_horizon_transfer_slice_count": 3,
                "recursive_compounding_retained_gain_runs": 5,
                "recursive_compounding_window_count": 3,
                "max_regression_rate": 0.02,
            },
            "benchmark_sources": {
                "codeforces": {"evidence_metric": "rating_equivalent"},
                "mle_bench": {"evidence_metric": "gold_medal_rate"},
            },
            "acceptance_policy": {
                "requires_sustained_windows": False,
                "requires_conservative_comparison": True,
                "requires_recursive_compounding": True,
                "forbids_readiness_only_promotion": False,
            },
        },
    }

    failures = module.verify_a8_benchmark_target_packet(packet)

    assert "target.thresholds.codeforces_rating_equivalent mismatch: observed=2400 expected=3000" in failures
    assert "target.thresholds.mle_bench_gold_medal_rate mismatch: observed=0.0 expected=0.2" in failures
    assert "target.thresholds.strong_baseline_comparison_slice_count mismatch: observed=1 expected=5" in failures
    assert "target.benchmark_sources.swe_bench_verified object is required" in failures
    assert "target.benchmark_sources.swe_rebench object is required" in failures
    assert "target.benchmark_sources.re_bench object is required" in failures
    assert "target.acceptance_policy.requires_sustained_windows must be true" in failures
    assert "target.acceptance_policy.forbids_readiness_only_promotion must be true" in failures


def _a8_benchmark_results() -> list[dict[str, object]]:
    return [
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "codeforces",
            "metrics": {
                "rating_equivalent": 3000,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "mle_bench",
            "metrics": {
                "gold_medal_rate": 0.2,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "swe_bench_verified",
            "metrics": {
                "resolve_rate": 0.8,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "swe_rebench",
            "metrics": {
                "resolve_rate": 0.6,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "re_bench",
            "metrics": {
                "human_expert_win_rate": 0.5,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "sustained_coding_window",
            "metrics": {
                "window_count": 3,
                "task_count": 100,
                "strong_human_baseline_win_rate": 0.85,
                "strong_human_baseline_win_rate_lower_bound": 0.65,
                "unfamiliar_domain_slice_count": 5,
                "long_horizon_transfer_slice_count": 3,
                "strong_baseline_comparison_slice_count": 5,
                "regression_rate": 0.01,
                "conservative_comparison_report": True,
            },
        },
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "recursive_compounding",
            "metrics": {
                "retained_gain_runs": 5,
                "window_count": 3,
                "verified_recursive_compounding": True,
                "conservative_comparison_report": True,
            },
        },
    ]


def test_verify_a8_benchmark_result_packet_accepts_required_metrics():
    module = _load_export_module()

    for packet in _a8_benchmark_results():
        assert module.verify_a8_benchmark_result_packet(packet) == []


def test_verify_a8_benchmark_result_packet_accepts_swe_bench_live():
    module = _load_export_module()

    failures = module.verify_a8_benchmark_result_packet(
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_result",
            "benchmark": "swe_bench_live",
            "metrics": {
                "resolve_rate": 0.82,
                "conservative_comparison_report": True,
            },
        }
    )

    assert failures == []


def test_verify_a8_benchmark_result_packet_rejects_missing_metric():
    module = _load_export_module()
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_result",
        "benchmark": "swe_bench_verified",
        "metrics": {},
    }

    failures = module.verify_a8_benchmark_result_packet(packet)

    assert "metrics object is required" in failures
    assert "metrics.resolve_rate is required for benchmark swe_bench_verified" in failures


def test_build_a8_autonomy_packet_from_benchmark_results_marks_supported_when_targets_met():
    module = _load_export_module()

    packet = module.build_a8_autonomy_packet_from_benchmark_results(
        _a8_benchmark_results(),
        source_paths=["codeforces.json", "mle.json"],
    )

    assert packet["claim"]["status"] == "supported"
    assert packet["source"]["benchmark_result_count"] == 7
    assert packet["evidence"]["a8"]["a8_claim_ready"] is True
    assert module.verify_static_autonomy_packet(packet) == []


def test_build_a8_autonomy_packet_from_benchmark_results_keeps_missing_adapter_not_supported():
    module = _load_export_module()
    weak_results = [
        result
        for result in _a8_benchmark_results()
        if result["benchmark"] != "swe_rebench"
    ]

    packet = module.build_a8_autonomy_packet_from_benchmark_results(weak_results)

    assert packet["claim"]["status"] == "not_supported"
    assert packet["evidence"]["a8"]["a8_claim_ready"] is False
    failures = module.verify_static_autonomy_packet(packet)
    assert "claim.status must be supported" in failures
    assert "A8 swe_rebench_resolve_rate must be at least 0.6" in failures
    assert "A8 conservative_comparison_report must be true" in failures


def test_export_autonomy_evidence_cli_writes_a8_aggregate_packet(tmp_path, monkeypatch, capsys):
    module = _load_export_module()
    result_paths = []
    for index, result in enumerate(_a8_benchmark_results()):
        result_path = tmp_path / f"result_{index}.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        result_paths.extend(["--a8-benchmark-result", str(result_path)])
    output_path = tmp_path / "a8_packet.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_autonomy_evidence.py",
            *result_paths,
            "--output-json",
            str(output_path),
        ],
    )

    module.main()

    packet = json.loads(output_path.read_text(encoding="utf-8"))
    assert packet["claim"]["level"] == "A8"
    assert packet["claim"]["status"] == "supported"
    assert "benchmark_result_count=7" in capsys.readouterr().out
