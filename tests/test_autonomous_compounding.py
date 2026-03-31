from pathlib import Path
import importlib.util
import json
from io import StringIO
from subprocess import CompletedProcess
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_autonomous_compounding_check_writes_report(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    bundle_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_retrieval_asset_bundle"}), encoding="utf-8")
    world_model_path = tmp_path / "world_model" / "world_model_proposals.json"
    world_model_path.parent.mkdir(parents=True, exist_ok=True)
    world_model_path.write_text(json.dumps({"artifact_kind": "world_model_policy_set"}), encoding="utf-8")
    trust_path = tmp_path / "trust" / "trust_proposals.json"
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.write_text(json.dumps({"artifact_kind": "trust_policy_set"}), encoding="utf-8")
    recovery_path = tmp_path / "recovery" / "recovery_proposals.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(json.dumps({"artifact_kind": "recovery_policy_set"}), encoding="utf-8")
    delegation_path = tmp_path / "delegation" / "delegation_proposals.json"
    delegation_path.parent.mkdir(parents=True, exist_ok=True)
    delegation_path.write_text(json.dumps({"artifact_kind": "delegated_runtime_policy_set"}), encoding="utf-8")
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(json.dumps({"artifact_kind": "operator_policy_set"}), encoding="utf-8")
    transition_model_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    transition_model_path.parent.mkdir(parents=True, exist_ok=True)
    transition_model_path.write_text(json.dumps({"artifact_kind": "transition_model_policy_set"}), encoding="utf-8")
    queue_path = tmp_path / "jobs" / "queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    runtime_state_path = tmp_path / "jobs" / "runtime_state.json"
    runtime_state_path.write_text(json.dumps({"active": []}), encoding="utf-8")
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "resume.json").write_text(json.dumps({"checkpoint": True}), encoding="utf-8")
    snapshot_root = tmp_path / "recovery" / "workspaces"
    (snapshot_root / "hello_task").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "hello_task" / "state.txt").write_text("snapshot", encoding="utf-8")
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(json.dumps({"modules": [{"module_id": "github", "enabled": True}]}), encoding="utf-8")
    trust_ledger_path = tmp_path / "reports" / "trust.json"
    trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    trust_ledger_path.write_text(json.dumps({"ledger_kind": "unattended_trust_ledger"}), encoding="utf-8")

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check
        assert "AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH" in env
        assert "--task-limit" in cmd
        assert cmd[cmd.index("--task-limit") + 1] == "40"
        forwarded_families = [
            cmd[index + 1]
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family"
        ]
        assert forwarded_families == ["workflow", "project", "repository", "tooling", "integration"]
        assert Path(env["AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH"]).exists()
        assert env["AGENT_KERNEL_USE_PROMPT_PROPOSALS"] == "0"
        assert env["AGENT_KERNEL_USE_CURRICULUM_PROPOSALS"] == "0"
        assert env["AGENT_KERNEL_USE_RETRIEVAL_PROPOSALS"] == "0"
        assert "AGENT_KERNEL_RUN_CHECKPOINTS_DIR" in env
        assert Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]).exists()
        assert json.loads((Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]) / "resume.json").read_text(encoding="utf-8"))["checkpoint"] is True
        assert json.loads(Path(env["AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "world_model_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_TRUST_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "trust_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_RECOVERY_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "recovery_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_DELEGATION_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "delegated_runtime_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "operator_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "transition_model_policy_set"
        assert "AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH" in env
        assert Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]).parent.exists()
        assert "AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH" in env
        assert Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]).parent.exists()
        assert "AGENT_KERNEL_CAPABILITY_MODULES_PATH" in env
        assert json.loads(Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]).read_text(encoding="utf-8"))["modules"][0]["module_id"] == "github"
        assert "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH" in env
        assert json.loads(Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]).read_text(encoding="utf-8"))["ledger_kind"] == "unattended_trust_ledger"
        assert (Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]) / "hello_task" / "state.txt").read_text(encoding="utf-8") == "snapshot"
        assert "AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD" in env
        isolated_report_dir = Path(env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"])
        isolated_report_dir.mkdir(parents=True, exist_ok=True)
        run_index = 1 if "autonomous-run-1" in cmd else 2
        report_path = isolated_report_dir / f"campaign_report_{run_index}.json"
        payload = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_campaign_report",
            "campaign_label": f"autonomous-run-{run_index}",
            "campaign_match_id": f"autonomous:match:{run_index}",
            "priority_benchmark_families": ["workflow", "project", "repository", "tooling", "integration"],
            "record_scope": {
                "protocol": "autonomous",
                "campaign_match_id": f"autonomous:match:{run_index}",
                "records_considered": run_index + 2,
                "decision_records_considered": run_index,
                "cycle_ids": [f"cycle:policy:{run_index}"],
            },
            "production_yield_summary": {
                "retained_cycles": run_index,
                "rejected_cycles": 0,
                "total_decisions": run_index,
                "retained_by_subsystem": {"policy": run_index},
                "rejected_by_subsystem": {},
                "average_retained_pass_rate_delta": 0.05 * run_index,
                "average_retained_step_delta": -0.1 * run_index,
                "average_rejected_pass_rate_delta": 0.0,
                "average_rejected_step_delta": 0.0,
            },
            "phase_gate_summary": {
                "all_retained_phase_gates_passed": True,
            },
            "decision_stream_summary": {
                "runtime_managed": {
                    "retained_cycles": run_index,
                    "rejected_cycles": 0,
                    "total_decisions": run_index,
                },
                "non_runtime_managed": {
                    "retained_cycles": 0,
                    "rejected_cycles": 0,
                    "total_decisions": 0,
                },
            },
            "trust_breadth_summary": {
                "required_families": ["workflow", "project", "repository", "tooling", "integration"],
                "required_families_with_reports": (
                    ["workflow", "project"] if run_index == 1 else ["workflow", "repository", "tooling"]
                ),
                "missing_required_families": ["integration"] if run_index == 1 else ["project", "integration"],
                "distinct_family_gap": 1 if run_index == 1 else 2,
                "external_benchmark_families": (
                    ["workflow", "project"] if run_index == 1 else ["workflow", "repository", "tooling"]
                ),
                "distinct_external_benchmark_families": 2 if run_index == 1 else 3,
            },
            "priority_family_yield_summary": {
                "priority_families": ["workflow", "project", "repository", "tooling", "integration"],
                "family_summaries": (
                    {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.06,
                            "retained_estimated_cost": 4.0,
                        },
                        "project": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.04,
                            "retained_estimated_cost": 5.0,
                        },
                        "repository": {
                            "observed_decisions": 0,
                            "retained_positive_delta_decisions": 0,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.0,
                        },
                        "tooling": {
                            "observed_decisions": 0,
                            "retained_positive_delta_decisions": 0,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.0,
                        },
                        "integration": {
                            "observed_decisions": 0,
                            "retained_positive_delta_decisions": 0,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.0,
                        },
                    }
                    if run_index == 1
                    else {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.07,
                            "retained_estimated_cost": 4.0,
                        },
                        "project": {
                            "observed_decisions": 0,
                            "retained_positive_delta_decisions": 0,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.0,
                        },
                        "repository": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.03,
                            "retained_estimated_cost": 6.0,
                        },
                        "tooling": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.02,
                            "retained_estimated_cost": 7.0,
                        },
                        "integration": {
                            "observed_decisions": 0,
                            "retained_positive_delta_decisions": 0,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.0,
                        },
                    }
                ),
            },
            "priority_family_allocation_summary": {
                "priority_families": ["workflow", "project", "repository", "tooling", "integration"],
                "planned_weight_shares": {
                    "workflow": 0.2,
                    "project": 0.2,
                    "repository": 0.2,
                    "tooling": 0.2,
                    "integration": 0.2,
                },
                "aggregated_task_counts": {
                    "workflow": 3 if run_index == 1 else 2,
                    "project": 2 if run_index == 1 else 3,
                    "repository": 1,
                    "tooling": 0,
                    "integration": 0,
                },
                "aggregated_task_shares": {
                    "workflow": 0.5 if run_index == 1 else 0.333333,
                    "project": 0.333333 if run_index == 1 else 0.5,
                    "repository": 0.166667,
                    "tooling": 0.0,
                    "integration": 0.0,
                },
                "top_planned_family": "workflow",
                "top_sampled_family": "workflow" if run_index == 1 else "project",
            },
            "recent_runtime_managed_decisions": [
                {
                    "cycle_id": f"cycle:policy:{run_index}",
                    "state": "retain",
                    "subsystem": "policy",
                    "artifact_kind": "prompt_policy_set",
                    "artifact_path": f"/runtime/policy_{run_index}.json",
                    "metrics_summary": {"candidate_pass_rate": 0.8},
                }
            ],
            "inheritance_summary": {
                "decision_count": run_index,
                "inherited_decisions": max(0, run_index - 1),
                "runtime_managed_decisions": run_index,
                "non_runtime_managed_decisions": 0,
            },
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=bundle_path,
            verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            world_model_proposals_path=world_model_path,
            trust_proposals_path=trust_path,
            recovery_proposals_path=recovery_path,
            delegation_proposals_path=delegation_path,
            operator_policy_proposals_path=operator_policy_path,
            transition_model_proposals_path=transition_model_path,
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            capability_modules_path=modules_path,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            run_checkpoints_dir=checkpoints_dir,
            unattended_workspace_snapshot_root=snapshot_root,
            unattended_trust_ledger_path=trust_ledger_path,
            use_prompt_proposals=False,
            use_curriculum_proposals=False,
            use_retrieval_proposals=False,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "2", "--cycles", "2"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    planner = ImprovementPlanner(memory_root=episodes_root, cycles_path=cycles_path)
    records = planner.load_cycle_records(cycles_path)

    assert payload["report_kind"] == "autonomous_compounding_report"
    assert payload["priority_benchmark_family_source"] == "default_non_replay_transfer_families"
    assert payload["priority_benchmark_families"] == [
        "workflow",
        "project",
        "repository",
        "tooling",
        "integration",
    ]
    assert payload["summary"]["runs"] == 2
    assert payload["summary"]["successful_runs"] == 2
    assert payload["summary"]["runs_with_retention"] == 2
    assert payload["summary"]["autonomous_compounding_viable"] is True
    assert payload["task_limit"] == 40
    assert payload["task_limit_source"] == "config_compare_feature_max_tasks"
    assert payload["priority_benchmark_family_weight_source"] == "default_equal_weight"
    assert payload["priority_benchmark_family_weights"] == {
        "workflow": 1.0,
        "project": 1.0,
        "repository": 1.0,
        "tooling": 1.0,
        "integration": 1.0,
    }
    assert payload["summary"]["runs_with_retained_phase_gate_failures"] == 0
    assert payload["summary"]["retained_cycle_spread"] == 1.0
    assert payload["summary"]["claim_gate_summary"]["starting_state_consistent"] is True
    assert payload["summary"]["claim_gate_summary"]["retention_criteria_stable"] is True
    assert payload["runs"][0]["seed_fingerprint"] == payload["runs"][1]["seed_fingerprint"]
    assert payload["runs"][0]["retention_criteria_fingerprint"] == payload["runs"][1]["retention_criteria_fingerprint"]
    assert payload["summary"]["claim_gate_summary"]["autonomous_compounding_claim_ready"] is True
    assert payload["summary"]["claim_gate_summary"]["blockers"] == []
    assert payload["summary"]["claim_gate_summary"]["min_runtime_managed_decisions"] == 1
    family_transfer_summary = payload["summary"]["claim_gate_summary"]["family_transfer_summary"]
    assert family_transfer_summary["target_non_replay_families"] == [
        "workflow",
        "project",
        "repository",
        "tooling",
        "integration",
    ]
    assert family_transfer_summary["distinct_target_families_observed"] == 4
    assert family_transfer_summary["distinct_target_families_with_retained_gain"] == 4
    assert family_transfer_summary["families_with_repeated_retained_gain"] == ["workflow"]
    family_transfer_timeline = payload["summary"]["claim_gate_summary"]["family_transfer_timeline"]
    assert family_transfer_timeline["families_with_non_declining_repeated_retained_gain"] == ["workflow"]
    assert family_transfer_timeline["families_with_declining_repeated_retained_gain"] == []
    assert family_transfer_timeline["families_with_cost_acceptable_non_declining_repeated_retained_gain"] == ["workflow"]
    assert family_transfer_timeline["families_with_costly_non_declining_repeated_retained_gain"] == []
    assert family_transfer_timeline["family_timelines"]["workflow"][0]["retained_positive_pass_rate_delta_sum"] == 0.06
    assert family_transfer_timeline["family_timelines"]["workflow"][1]["retained_positive_pass_rate_delta_sum"] == 0.07
    assert family_transfer_timeline["family_timelines"]["workflow"][0]["retained_return_on_cost"] == 0.015
    assert family_transfer_timeline["family_timelines"]["workflow"][1]["retained_return_on_cost"] == 0.0175
    family_transfer_investment_ranking = payload["summary"]["claim_gate_summary"]["family_transfer_investment_ranking"]
    assert family_transfer_investment_ranking["top_transfer_investment_family"] == "workflow"
    assert family_transfer_investment_ranking["ranked_families_by_transfer_investment"][0] == "workflow"
    assert family_transfer_investment_ranking["ranked_families_by_transfer_investment"][-1] == "integration"
    assert family_transfer_investment_ranking["family_rankings"][0]["category"] == "cost_acceptable_persistent"
    assert family_transfer_investment_ranking["family_rankings"][0]["investment_score"] > family_transfer_investment_ranking["family_rankings"][1]["investment_score"]
    priority_family_allocation_audit = payload["summary"]["claim_gate_summary"]["priority_family_allocation_audit"]
    assert priority_family_allocation_audit["runs_with_allocation_summary"] == 2
    assert priority_family_allocation_audit["runs_with_top_planned_family_as_top_sampled"] == 1
    assert priority_family_allocation_audit["top_planned_family"] == "workflow"
    assert priority_family_allocation_audit["actual_task_totals"]["project"] == 5
    result_stream_audit = payload["summary"]["claim_gate_summary"]["result_stream_audit"]
    assert result_stream_audit["runs_with_scoped_campaign_record_stream"] == 2
    assert result_stream_audit["runs_with_runtime_managed_result_stream"] == 2
    assert result_stream_audit["warnings"] == []
    assert result_stream_audit["result_streams"][0]["record_scope"]["protocol"] == "autonomous"
    assert result_stream_audit["result_streams"][0]["decision_stream_summary"]["runtime_managed"]["total_decisions"] == 1
    assert result_stream_audit["result_streams"][0]["recent_runtime_managed_decisions"][0]["cycle_id"] == "cycle:policy:1"
    assert records[-1]["artifact_kind"] == "autonomous_compounding_report"


def test_run_autonomous_compounding_check_reuses_prior_family_investment_ranking(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T000000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_families = [
            cmd[index + 1]
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family"
        ]
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert "--task-limit" in cmd
        assert cmd[cmd.index("--task-limit") + 1] == "40"
        assert forwarded_families == ["repository", "workflow", "project", "tooling", "integration"]
        assert forwarded_weights["repository"] > forwarded_weights["workflow"] > forwarded_weights["project"]
        assert forwarded_weights["project"] > forwarded_weights["tooling"] > forwarded_weights["integration"]
        report_path = reports_dir / "campaign_report_seeded.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": forwarded_families,
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 2,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "trust_breadth_summary": {
                        "required_families_with_reports": ["repository", "workflow"],
                        "external_benchmark_families": ["repository", "workflow"],
                    },
                    "priority_family_yield_summary": {
                        "family_summaries": {
                            "repository": {
                                "observed_decisions": 1,
                                "retained_positive_delta_decisions": 1,
                                "retained_negative_delta_decisions": 0,
                                "retained_positive_pass_rate_delta_sum": 0.05,
                                "retained_estimated_cost": 2.0,
                            },
                            "workflow": {
                                "observed_decisions": 1,
                                "retained_positive_delta_decisions": 1,
                                "retained_negative_delta_decisions": 0,
                                "retained_positive_pass_rate_delta_sum": 0.04,
                                "retained_estimated_cost": 2.0,
                            },
                        }
                    },
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_source"] == "prior_compounding_ranking"
    assert payload["task_limit"] == 40
    assert payload["task_limit_source"] == "config_compare_feature_max_tasks"
    assert payload["priority_benchmark_families"] == ["repository", "workflow", "project", "tooling", "integration"]
    assert payload["priority_benchmark_family_weight_source"] == "prior_compounding_investment_score_plus_rank_weight"
    assert payload["priority_benchmark_family_weights"]["repository"] > payload["priority_benchmark_family_weights"]["workflow"]
    assert payload["runs"][0]["retention_criteria_manifest"]["run_parameters"]["priority_benchmark_family_source"] == "prior_compounding_ranking"
    assert payload["runs"][0]["retention_criteria_manifest"]["run_parameters"]["task_limit"] == 40
    assert payload["runs"][0]["retention_criteria_manifest"]["run_parameters"]["priority_benchmark_family_weights"]["repository"] > payload["runs"][0]["retention_criteria_manifest"]["run_parameters"]["priority_benchmark_family_weights"]["workflow"]


def test_run_autonomous_compounding_check_compensates_under_sampled_prior_family(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T010000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.45,
                                "workflow": 0.1,
                                "project": 0.2,
                                "tooling": 0.15,
                                "integration": 0.1,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.3,
                                "workflow": 0.1,
                                "project": 0.25,
                                "tooling": 0.25,
                                "integration": 0.1,
                            },
                            "latest_positive_gap_streak_by_family": {
                                "workflow": 1,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 9,
                                "workflow": 2,
                                "project": 4,
                                "tooling": 3,
                                "integration": 2,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["workflow"] > forwarded_weights["repository"]
        report_path = reports_dir / "campaign_report_compensated.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_weight_source"] == (
        "prior_compounding_investment_score_plus_rank_weight_and_allocation_compensation_and_normalization"
    )
    assert payload["priority_benchmark_family_allocation_compensation"]["gap_source"] == "latest_allocation_summary"
    assert payload["priority_benchmark_family_allocation_compensation"]["positive_gap_families"] == ["workflow"]
    assert "repository" in payload["priority_benchmark_family_allocation_compensation"]["negative_gap_families"]
    assert payload["priority_benchmark_family_allocation_compensation"]["share_gap_by_family"]["workflow"] == 0.4
    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["latest_task_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["recent_average_task_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["bonus_allocation_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_allocation_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_bonus_allocation_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_streak_by_family"]["workflow"] == 1
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_multiplier_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_bonus_by_family"]["workflow"] == 4.0
    assert payload["priority_benchmark_family_allocation_compensation"]["latest_summary_run_index"] == 3
    assert payload["priority_benchmark_family_weights"]["workflow"] > payload["priority_benchmark_family_weights"]["repository"]
    assert (
        payload["runs"][0]["retention_criteria_manifest"]["run_parameters"]["priority_benchmark_family_allocation_compensation"]["positive_gap_families"]
        == ["workflow"]
    )


def test_run_autonomous_compounding_check_applies_retained_allocation_confidence_policy(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "priority_family_allocation_confidence": {
                        "minimum_runs": 4,
                        "target_priority_tasks": 20,
                        "target_family_tasks": 5,
                        "history_window_runs": 2,
                        "history_weight": 0.25,
                        "bonus_history_weight": 1.0,
                        "normalization_history_weight": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    prior_report_path = reports_dir / "autonomous_compounding_20260329T015500000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.35,
                                "workflow": 0.25,
                                "project": 0.15,
                                "tooling": 0.15,
                                "integration": 0.1,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.3,
                                "workflow": 0.1,
                                "project": 0.15,
                                "tooling": 0.25,
                                "integration": 0.1,
                            },
                            "latest_positive_gap_streak_by_family": {
                                "workflow": 1,
                            },
                            "latest_oversampled_streak_by_family": {
                                "repository": 1,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 0.4,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 0.8,
                                "workflow": 0.2,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 6,
                                "workflow": 2,
                                "project": 3,
                                "tooling": 3,
                                "integration": 2,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cmd, cwd, capture_output, text, check, env
        report_path = reports_dir / "campaign_report_policy_weighted.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "priority_family_allocation_summary": {
                        "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                        "planned_weight_shares": {
                            "repository": 0.3,
                            "workflow": 0.3,
                            "project": 0.15,
                            "tooling": 0.15,
                            "integration": 0.1,
                        },
                        "aggregated_task_shares": {
                            "repository": 0.4,
                            "workflow": 0.2,
                            "project": 0.2,
                            "tooling": 0.1,
                            "integration": 0.1,
                        },
                        "aggregated_task_counts": {
                            "repository": 2,
                            "workflow": 1,
                            "project": 1,
                            "tooling": 1,
                            "integration": 1,
                        },
                        "total_priority_tasks": 6,
                        "top_planned_family": "repository",
                        "top_sampled_family": "repository",
                    },
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess([], 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
            verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
            prompt_proposals_path=prompt_path,
            world_model_proposals_path=tmp_path / "world_model" / "world_model_proposals.json",
            trust_proposals_path=tmp_path / "trust" / "trust_proposals.json",
            recovery_proposals_path=tmp_path / "recovery" / "recovery_proposals.json",
            delegation_proposals_path=tmp_path / "delegation" / "delegation_proposals.json",
            operator_policy_proposals_path=tmp_path / "operator_policy" / "operator_policy_proposals.json",
            transition_model_proposals_path=tmp_path / "transition_model" / "transition_model_proposals.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            capability_modules_path=tmp_path / "config" / "capabilities.json",
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    compensation = payload["priority_benchmark_family_allocation_compensation"]
    assert compensation["allocation_confidence_min_runs"] == 4
    assert compensation["allocation_confidence_target_priority_tasks"] == 20
    assert compensation["allocation_confidence_target_family_tasks"] == 5
    assert compensation["allocation_confidence_history_window_runs"] == 2
    assert compensation["allocation_confidence_history_weight"] == 0.25
    assert compensation["bonus_allocation_confidence_history_weight"] == 1.0
    assert compensation["normalization_allocation_confidence_history_weight"] == 0.0
    assert compensation["applied_bonus_allocation_confidence_by_family"]["workflow"] == 0.2
    assert compensation["weight_bonus_by_family"]["workflow"] == 0.8
    assert compensation["applied_normalization_allocation_confidence_by_family"]["repository"] == 0.4
    assert compensation["weight_normalization_by_family"]["repository"] == 0.2
    audit = payload["summary"]["claim_gate_summary"]["priority_family_allocation_audit"]
    assert audit["allocation_confidence"] == 0.25
    assert audit["allocation_confidence_by_family"]["repository"] == 0.25
    assert audit["allocation_confidence_by_family"]["workflow"] == 0.2
    assert audit["allocation_confidence_components"]["minimum_runs"] == 4
    assert audit["allocation_confidence_components"]["target_priority_tasks"] == 20
    assert audit["allocation_confidence_components"]["target_family_tasks"] == 5
    assert audit["allocation_confidence_components"]["target_family_tasks_override"] == 5
    assert audit["allocation_confidence_components"]["history_window_runs"] == 2
    assert audit["allocation_confidence_components"]["history_weight"] == 0.25
    assert audit["allocation_confidence_components"]["bonus_history_weight"] == 1.0
    assert audit["allocation_confidence_components"]["normalization_history_weight"] == 0.0


def test_run_autonomous_compounding_check_drops_compensation_after_latest_gap_closes(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T020000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 1,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.35,
                                "workflow": 0.2,
                                "project": 0.2,
                                "tooling": 0.15,
                                "integration": 0.1,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.2,
                                "workflow": 0.52,
                                "project": 0.13,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 7,
                                "workflow": 6,
                                "project": 4,
                                "tooling": 3,
                                "integration": 2,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "workflow",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["repository"] > forwarded_weights["workflow"]
        report_path = reports_dir / "campaign_report_recovered.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_weight_source"] == "prior_compounding_investment_score_plus_rank_weight"
    assert payload["priority_benchmark_family_allocation_compensation"]["gap_source"] == "latest_allocation_summary"
    assert payload["priority_benchmark_family_allocation_compensation"]["positive_gap_families"] == []
    assert payload["priority_benchmark_family_allocation_compensation"]["negative_gap_families"] == []
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_bonus_by_family"] == {}
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_streak_by_family"] == {}
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_normalization_by_family"] == {}
    assert payload["priority_benchmark_family_allocation_compensation"]["recovered_gap_families"] == ["workflow"]
    assert payload["priority_benchmark_family_allocation_compensation"]["average_share_gap_by_family"]["workflow"] == 0.3
    assert payload["priority_benchmark_family_allocation_compensation"]["latest_share_gap_by_family"]["project"] == 0.02
    assert payload["priority_benchmark_family_weights"]["repository"] > payload["priority_benchmark_family_weights"]["workflow"]


def test_run_autonomous_compounding_check_applies_gentle_normalization_after_overshoot(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T030000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 1,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.28,
                                "workflow": 0.32,
                                "project": 0.17,
                                "tooling": 0.14,
                                "integration": 0.09,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.33,
                                "workflow": 0.49,
                                "project": 0.12,
                                "tooling": 0.08,
                                "integration": 0.05,
                            },
                            "latest_oversampled_streak_by_family": {
                                "repository": 1,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 8,
                                "workflow": 6,
                                "project": 4,
                                "tooling": 2,
                                "integration": 1,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["repository"] < 5.09
        assert forwarded_weights["repository"] > forwarded_weights["workflow"]
        report_path = reports_dir / "campaign_report_normalized.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_weight_source"] == (
        "prior_compounding_investment_score_plus_rank_weight_and_allocation_normalization"
    )
    assert payload["priority_benchmark_family_allocation_compensation"]["negative_gap_families"] == ["repository"]
    assert payload["priority_benchmark_family_allocation_compensation"]["oversampled_share_gap_by_family"]["repository"] == 0.13
    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_allocation_confidence_by_family"]["repository"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_allocation_confidence_by_family"]["repository"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_normalization_allocation_confidence_by_family"]["repository"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_streak_by_family"] == {}
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_streak_by_family"]["repository"] == 1
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_multiplier_by_family"]["repository"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_normalization_by_family"]["repository"] == 0.5
    assert payload["priority_benchmark_family_allocation_compensation"]["net_weight_adjustment_by_family"]["repository"] == -0.5
    assert payload["priority_benchmark_family_allocation_compensation"]["latest_oversampled_share_gap_by_family"]["repository"] == 0.13
    assert payload["priority_benchmark_family_weights"]["repository"] == 4.59


def test_run_autonomous_compounding_check_strengthens_normalization_for_repeated_overshoot(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T040000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.31,
                                "workflow": 0.36,
                                "project": 0.13,
                                "tooling": 0.12,
                                "integration": 0.08,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.33,
                                "workflow": 0.49,
                                "project": 0.12,
                                "tooling": 0.08,
                                "integration": 0.05,
                            },
                            "latest_oversampled_streak_by_family": {
                                "repository": 3,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 10,
                                "workflow": 5,
                                "project": 4,
                                "tooling": 3,
                                "integration": 2,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["repository"] == 4.09
        report_path = reports_dir / "campaign_report_normalized_streak.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_weight_source"] == (
        "prior_compounding_investment_score_plus_rank_weight_and_allocation_normalization"
    )
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_streak_by_family"]["repository"] == 3
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_multiplier_by_family"]["repository"] == 2.0
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_normalization_by_family"]["repository"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["net_weight_adjustment_by_family"]["repository"] == -1.0
    assert payload["priority_benchmark_family_weights"]["repository"] == 4.09


def test_run_autonomous_compounding_check_strengthens_compensation_for_repeated_under_sampling(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T050000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.34,
                                "workflow": 0.18,
                                "project": 0.22,
                                "tooling": 0.16,
                                "integration": 0.1,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.3,
                                "workflow": 0.1,
                                "project": 0.25,
                                "tooling": 0.25,
                                "integration": 0.1,
                            },
                            "latest_positive_gap_streak_by_family": {
                                "workflow": 3,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 1.0,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 1.0,
                                "integration": 1.0,
                            },
                            "actual_task_totals": {
                                "repository": 9,
                                "workflow": 2,
                                "project": 4,
                                "tooling": 3,
                                "integration": 2,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["workflow"] == 12.04
        assert forwarded_weights["workflow"] > forwarded_weights["repository"]
        report_path = reports_dir / "campaign_report_repeated_bonus.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_weight_source"] == (
        "prior_compounding_investment_score_plus_rank_weight_and_allocation_compensation_and_normalization"
    )
    assert payload["priority_benchmark_family_allocation_compensation"]["positive_gap_families"] == ["workflow"]
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_streak_by_family"]["workflow"] == 3
    assert payload["priority_benchmark_family_allocation_compensation"]["compensation_multiplier_by_family"]["workflow"] == 2.0
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_bonus_by_family"]["workflow"] == 8.0
    assert payload["priority_benchmark_family_allocation_compensation"]["net_weight_adjustment_by_family"]["workflow"] == 8.0
    assert payload["priority_benchmark_family_weights"]["workflow"] == 12.04


def test_run_autonomous_compounding_check_dampens_adjustments_when_allocation_confidence_is_low(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T060000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 1,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.33,
                                "workflow": 0.1,
                                "project": 0.12,
                                "tooling": 0.25,
                                "integration": 0.2,
                            },
                            "latest_summary_run_index": 1,
                            "latest_summary_run_match_id": "autonomous:prior:1",
                            "latest_total_priority_tasks": 4,
                            "allocation_confidence": 0.333333,
                            "allocation_confidence_by_family": {
                                "repository": 0.333333,
                                "workflow": 0.333333,
                                "project": 0.333333,
                                "tooling": 0.0,
                                "integration": 0.0,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 0.333333,
                                "workflow": 0.333333,
                                "project": 0.333333,
                                "tooling": 0.0,
                                "integration": 0.0,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 0.333333,
                                "workflow": 0.333333,
                                "project": 0.333333,
                                "tooling": 0.0,
                                "integration": 0.0,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 0.666667,
                                "workflow": 0.333333,
                                "project": 0.333333,
                                "tooling": 0.0,
                                "integration": 0.0,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 0.333333,
                                "workflow": 0.333333,
                                "project": 0.333333,
                                "tooling": 0.0,
                                "integration": 0.0,
                            },
                            "allocation_confidence_components": {
                                "run_confidence": 0.333333,
                                "task_confidence": 0.333333,
                                "minimum_runs": 3,
                                "target_priority_tasks": 12,
                                "target_family_tasks": 3,
                                "history_window_runs": 3,
                                "history_weight": 0.5,
                                "bonus_history_weight": 0.75,
                                "normalization_history_weight": 0.25,
                            },
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.33,
                                "workflow": 0.1,
                                "project": 0.12,
                                "tooling": 0.25,
                                "integration": 0.2,
                            },
                            "latest_positive_gap_streak_by_family": {
                                "workflow": 3,
                            },
                            "latest_oversampled_streak_by_family": {
                                "repository": 3,
                            },
                            "actual_task_totals": {
                                "repository": 2,
                                "workflow": 1,
                                "project": 1,
                                "tooling": 0,
                                "integration": 0,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "repository",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["repository"] == 4.756667
        assert forwarded_weights["workflow"] == 6.706664
        report_path = reports_dir / "campaign_report_low_confidence.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence"] == 0.333333
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_bonus_by_family"]["workflow"] == 2.666664
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_normalization_by_family"]["repository"] == 0.333333
    assert payload["priority_benchmark_family_allocation_compensation"]["net_weight_adjustment_by_family"]["workflow"] == 2.666664
    assert payload["priority_benchmark_family_allocation_compensation"]["net_weight_adjustment_by_family"]["repository"] == -0.333333
    assert payload["priority_benchmark_family_weights"]["workflow"] == 6.706664
    assert payload["priority_benchmark_family_weights"]["repository"] == 4.756667


def test_run_autonomous_compounding_check_uses_family_specific_allocation_confidence(tmp_path, monkeypatch):
    module = _load_script("run_autonomous_compounding_check.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prior_report_path = reports_dir / "autonomous_compounding_20260329T070000000000Z.json"
    prior_report_path.write_text(
        json.dumps(
            {
                "report_kind": "autonomous_compounding_report",
                "summary": {
                    "claim_gate_summary": {
                        "family_transfer_investment_ranking": {
                            "ranked_families_by_transfer_investment": ["repository", "workflow", "project"],
                            "family_rankings": [
                                {"family": "repository", "investment_score": 0.09},
                                {"family": "workflow", "investment_score": 0.04},
                                {"family": "project", "investment_score": 0.02},
                            ],
                        },
                        "priority_family_allocation_audit": {
                            "runs_with_allocation_summary": 3,
                            "runs_with_top_planned_family_as_top_sampled": 0,
                            "priority_families": ["repository", "workflow", "project", "tooling", "integration"],
                            "average_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "average_actual_shares": {
                                "repository": 0.31,
                                "workflow": 0.22,
                                "project": 0.15,
                                "tooling": 0.17,
                                "integration": 0.15,
                            },
                            "latest_summary_run_index": 3,
                            "latest_summary_run_match_id": "autonomous:prior:3",
                            "latest_total_priority_tasks": 12,
                            "latest_task_counts_by_family": {
                                "repository": 1,
                                "workflow": 5,
                                "project": 3,
                                "tooling": 2,
                                "integration": 1,
                            },
                            "allocation_confidence": 1.0,
                            "allocation_confidence_by_family": {
                                "repository": 0.555556,
                                "workflow": 0.777778,
                                "project": 0.833333,
                                "tooling": 0.5,
                                "integration": 0.277778,
                            },
                            "bonus_allocation_confidence_by_family": {
                                "repository": 0.666667,
                                "workflow": 0.666667,
                                "project": 0.75,
                                "tooling": 0.416667,
                                "integration": 0.25,
                            },
                            "normalization_allocation_confidence_by_family": {
                                "repository": 0.444444,
                                "workflow": 0.888889,
                                "project": 0.916667,
                                "tooling": 0.583333,
                                "integration": 0.305556,
                            },
                            "latest_task_confidence_by_family": {
                                "repository": 0.333333,
                                "workflow": 1.0,
                                "project": 1.0,
                                "tooling": 0.666667,
                                "integration": 0.333333,
                            },
                            "recent_average_task_confidence_by_family": {
                                "repository": 0.777778,
                                "workflow": 0.555556,
                                "project": 0.666667,
                                "tooling": 0.333333,
                                "integration": 0.222222,
                            },
                            "allocation_confidence_components": {
                                "run_confidence": 1.0,
                                "task_confidence": 1.0,
                                "minimum_runs": 3,
                                "target_priority_tasks": 12,
                                "target_family_tasks": 3,
                                "history_window_runs": 3,
                                "history_weight": 0.5,
                                "bonus_history_weight": 0.75,
                                "normalization_history_weight": 0.25,
                            },
                            "latest_planned_shares": {
                                "repository": 0.2,
                                "workflow": 0.5,
                                "project": 0.15,
                                "tooling": 0.1,
                                "integration": 0.05,
                            },
                            "latest_actual_shares": {
                                "repository": 0.33,
                                "workflow": 0.1,
                                "project": 0.15,
                                "tooling": 0.27,
                                "integration": 0.15,
                            },
                            "latest_positive_gap_streak_by_family": {
                                "workflow": 3,
                            },
                            "latest_oversampled_streak_by_family": {
                                "repository": 3,
                            },
                            "actual_task_totals": {
                                "repository": 7,
                                "workflow": 11,
                                "project": 9,
                                "tooling": 5,
                                "integration": 4,
                            },
                            "top_planned_family": "workflow",
                            "top_sampled_family": "workflow",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check, env
        forwarded_weights = {
            cmd[index + 1].split("=", 1)[0]: float(cmd[index + 1].split("=", 1)[1])
            for index, token in enumerate(cmd[:-1])
            if token == "--priority-benchmark-family-weight"
        }
        assert forwarded_weights["workflow"] == 9.373336
        assert forwarded_weights["repository"] == 4.645556
        report_path = reports_dir / "campaign_report_family_confidence.json"
        report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "campaign_label": "autonomous-run-1",
                    "campaign_match_id": "autonomous:match:1",
                    "priority_benchmark_families": ["repository", "workflow", "project", "tooling", "integration"],
                    "record_scope": {
                        "protocol": "autonomous",
                        "campaign_match_id": "autonomous:match:1",
                        "records_considered": 1,
                        "decision_records_considered": 1,
                        "cycle_ids": ["cycle:policy:1"],
                    },
                    "production_yield_summary": {
                        "retained_cycles": 1,
                        "rejected_cycles": 0,
                        "total_decisions": 1,
                        "average_retained_pass_rate_delta": 0.05,
                        "average_retained_step_delta": -0.1,
                        "average_retained_estimated_cost": 2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                    "decision_stream_summary": {
                        "runtime_managed": {"total_decisions": 1},
                        "non_runtime_managed": {"total_decisions": 0},
                    },
                    "priority_family_yield_summary": {"family_summaries": {}},
                    "inheritance_summary": {"runtime_managed_decisions": 1},
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, stdout=f"{report_path}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=tmp_path / "retrieval" / "retrieval_asset_bundle.json",
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
            delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
            delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
            run_checkpoints_dir=tmp_path / "checkpoints",
            unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
            unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_autonomous_compounding_check.py", "--runs", "1", "--cycles", "1"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["latest_task_confidence_by_family"]["workflow"] == 1.0
    assert payload["priority_benchmark_family_allocation_compensation"]["recent_average_task_confidence_by_family"]["workflow"] == 0.555556
    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence_by_family"]["workflow"] == 0.777778
    assert payload["priority_benchmark_family_allocation_compensation"]["allocation_confidence_by_family"]["repository"] == 0.555556
    assert payload["priority_benchmark_family_allocation_compensation"]["bonus_allocation_confidence_by_family"]["workflow"] == 0.666667
    assert payload["priority_benchmark_family_allocation_compensation"]["normalization_allocation_confidence_by_family"]["repository"] == 0.444444
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_allocation_confidence_by_family"]["workflow"] == 0.666667
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_allocation_confidence_by_family"]["repository"] == 0.444444
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_bonus_allocation_confidence_by_family"]["workflow"] == 0.666667
    assert payload["priority_benchmark_family_allocation_compensation"]["applied_normalization_allocation_confidence_by_family"]["repository"] == 0.444444
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_bonus_by_family"]["workflow"] == 5.333336
    assert payload["priority_benchmark_family_allocation_compensation"]["weight_normalization_by_family"]["repository"] == 0.444444
    assert payload["priority_benchmark_family_weights"]["workflow"] == 9.373336
    assert payload["priority_benchmark_family_weights"]["repository"] == 4.645556


def test_claim_gate_blocks_missing_result_stream_audit_fields():
    module = _load_script("run_autonomous_compounding_check.py")

    results = [
        {
            "run_match_id": "autonomous:one",
            "run_index": 1,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.1,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 2.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "campaign_label": "autonomous-run-1",
                "campaign_match_id": "autonomous:one",
                "inheritance_summary": {
                    "runtime_managed_decisions": 1,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
        },
        {
            "run_match_id": "autonomous:two",
            "run_index": 2,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.04,
                "average_retained_step_delta": -0.05,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 2.5,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "campaign_label": "autonomous-run-2",
                "campaign_match_id": "autonomous:two",
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:two",
                    "records_considered": 3,
                    "decision_records_considered": 1,
                    "cycle_ids": ["cycle:policy:2"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 1},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "recent_runtime_managed_decisions": [
                    {
                        "cycle_id": "cycle:policy:2",
                        "state": "retain",
                        "subsystem": "policy",
                        "artifact_kind": "prompt_policy_set",
                        "artifact_path": "/runtime/policy_2.json",
                    }
                ],
                "inheritance_summary": {
                    "runtime_managed_decisions": 1,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
        },
    ]

    claim_gate = module._claim_gate_summary(results)

    assert claim_gate["autonomous_compounding_claim_ready"] is False
    assert "one_or_more_runs_missing_scoped_campaign_record_stream" in claim_gate["blockers"]
    assert "one_or_more_runs_missing_runtime_managed_result_stream" in claim_gate["blockers"]
    assert claim_gate["result_stream_audit"]["missing_scoped_campaign_record_stream_runs"] == [1]
    assert claim_gate["result_stream_audit"]["missing_runtime_managed_result_stream_runs"] == [1]


def test_claim_gate_blocks_when_non_replay_transfer_gain_is_too_narrow():
    module = _load_script("run_autonomous_compounding_check.py")

    results = [
        {
            "run_match_id": "autonomous:one",
            "run_index": 1,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.1,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 2.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:one",
                    "records_considered": 3,
                    "decision_records_considered": 1,
                    "cycle_ids": ["cycle:policy:1"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 1},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow"],
                    "external_benchmark_families": ["workflow"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.05,
                            "retained_estimated_cost": 4.0,
                        }
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 1,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project", "repository", "tooling", "integration"]
                }
            },
        },
        {
            "run_match_id": "autonomous:two",
            "run_index": 2,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.04,
                "average_retained_step_delta": -0.05,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 2.5,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:two",
                    "records_considered": 3,
                    "decision_records_considered": 1,
                    "cycle_ids": ["cycle:policy:2"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 1},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow"],
                    "external_benchmark_families": ["workflow"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.04,
                            "retained_estimated_cost": 4.0,
                        }
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 1,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project", "repository", "tooling", "integration"]
                }
            },
        },
    ]

    claim_gate = module._claim_gate_summary(results)

    assert claim_gate["autonomous_compounding_claim_ready"] is False
    assert "non_replay_transfer_family_observation_too_narrow" in claim_gate["blockers"]
    assert "non_replay_transfer_retained_gain_too_narrow" in claim_gate["blockers"]
    assert "non_replay_transfer_retained_gain_not_persistent_over_time" in claim_gate["blockers"]
    assert claim_gate["family_transfer_summary"]["distinct_target_families_observed"] == 1
    assert claim_gate["family_transfer_summary"]["distinct_target_families_with_retained_gain"] == 1


def test_claim_gate_blocks_when_transfer_gain_declines_over_time_despite_breadth():
    module = _load_script("run_autonomous_compounding_check.py")

    results = [
        {
            "run_match_id": "autonomous:one",
            "run_index": 1,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.08,
                "average_retained_step_delta": -0.1,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 3.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:one",
                    "records_considered": 4,
                    "decision_records_considered": 2,
                    "cycle_ids": ["cycle:policy:1", "cycle:tooling:1"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 2},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow", "project", "repository"],
                    "external_benchmark_families": ["workflow", "project", "repository"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.08,
                            "retained_estimated_cost": 4.0,
                        },
                        "project": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.05,
                            "retained_estimated_cost": 4.0,
                        },
                        "repository": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.04,
                            "retained_estimated_cost": 4.0,
                        },
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 2,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project", "repository"]
                }
            },
        },
        {
            "run_match_id": "autonomous:two",
            "run_index": 2,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.04,
                "average_retained_step_delta": -0.05,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 3.5,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:two",
                    "records_considered": 4,
                    "decision_records_considered": 2,
                    "cycle_ids": ["cycle:policy:2", "cycle:tooling:2"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 2},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow", "project", "repository"],
                    "external_benchmark_families": ["workflow", "project", "repository"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.03,
                            "retained_estimated_cost": 4.0,
                        },
                        "project": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.02,
                            "retained_estimated_cost": 4.0,
                        },
                        "repository": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.01,
                            "retained_estimated_cost": 4.0,
                        },
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 2,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project", "repository"]
                }
            },
        },
    ]

    claim_gate = module._claim_gate_summary(results)

    assert claim_gate["autonomous_compounding_claim_ready"] is False
    assert "non_replay_transfer_retained_gain_not_persistent_over_time" in claim_gate["blockers"]
    assert claim_gate["family_transfer_timeline"]["families_with_non_declining_repeated_retained_gain"] == []
    assert claim_gate["family_transfer_timeline"]["families_with_declining_repeated_retained_gain"] == [
        "workflow",
        "project",
        "repository",
    ]


def test_claim_gate_blocks_when_transfer_is_persistent_but_too_expensive():
    module = _load_script("run_autonomous_compounding_check.py")

    results = [
        {
            "run_match_id": "autonomous:one",
            "run_index": 1,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.1,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 12.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:one",
                    "records_considered": 4,
                    "decision_records_considered": 2,
                    "cycle_ids": ["cycle:policy:1", "cycle:tooling:1"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 2},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow", "project"],
                    "external_benchmark_families": ["workflow", "project"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.05,
                            "retained_estimated_cost": 8.0,
                        },
                        "project": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.04,
                            "retained_estimated_cost": 6.0,
                        },
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 2,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project"]
                }
            },
        },
        {
            "run_match_id": "autonomous:two",
            "run_index": 2,
            "returncode": 0,
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.08,
                "worst_family_delta": 0.0,
                "worst_generated_family_delta": 0.0,
                "worst_failure_recovery_delta": 0.0,
                "average_retained_estimated_cost": 12.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            "report_payload": {
                "record_scope": {
                    "protocol": "autonomous",
                    "campaign_match_id": "autonomous:two",
                    "records_considered": 4,
                    "decision_records_considered": 2,
                    "cycle_ids": ["cycle:policy:2", "cycle:tooling:2"],
                },
                "decision_stream_summary": {
                    "runtime_managed": {"total_decisions": 2},
                    "non_runtime_managed": {"total_decisions": 0},
                },
                "trust_breadth_summary": {
                    "required_families_with_reports": ["workflow", "project"],
                    "external_benchmark_families": ["workflow", "project"],
                },
                "priority_family_yield_summary": {
                    "family_summaries": {
                        "workflow": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.06,
                            "retained_estimated_cost": 6.5,
                        },
                        "project": {
                            "observed_decisions": 1,
                            "retained_positive_delta_decisions": 1,
                            "retained_negative_delta_decisions": 0,
                            "retained_positive_pass_rate_delta_sum": 0.05,
                            "retained_estimated_cost": 5.5,
                        },
                    }
                },
                "inheritance_summary": {
                    "runtime_managed_decisions": 2,
                },
            },
            "seed_fingerprint": "seed-shared",
            "retention_criteria_fingerprint": "criteria-shared",
            "retention_criteria_manifest": {
                "run_parameters": {
                    "priority_benchmark_families": ["workflow", "project"]
                }
            },
        },
    ]

    claim_gate = module._claim_gate_summary(results)

    assert claim_gate["autonomous_compounding_claim_ready"] is False
    assert "non_replay_transfer_return_on_cost_too_low" in claim_gate["blockers"]
    assert claim_gate["family_transfer_timeline"]["families_with_non_declining_repeated_retained_gain"] == [
        "workflow",
        "project",
    ]
    assert claim_gate["family_transfer_timeline"]["families_with_cost_acceptable_non_declining_repeated_retained_gain"] == []
    assert claim_gate["family_transfer_timeline"]["families_with_costly_non_declining_repeated_retained_gain"] == [
        "workflow",
        "project",
    ]
    investment_ranking = claim_gate["family_transfer_investment_ranking"]
    assert investment_ranking["ranked_families_by_transfer_investment"][:2] == ["workflow", "project"]
    assert investment_ranking["family_rankings"][0]["category"] == "costly_persistent"
