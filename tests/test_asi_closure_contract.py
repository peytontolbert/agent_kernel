import importlib.util
from pathlib import Path

from agent_kernel.llm import MockLLMClient
from agent_kernel.memory import EpisodeMemory
from agent_kernel.policy import LLMDecisionPolicy
from agent_kernel.schemas import CommandResult, EpisodeRecord, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.verifier import Verifier
from agent_kernel.world_model import WorldModel


def _load_script(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_name.replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_asi_closure_contract_runtime_surfaces(tmp_path):
    memory = EpisodeMemory(tmp_path / "episodes")
    for task_id in ("repair_a", "repair_b"):
        memory.save(
            EpisodeRecord(
                task_id=task_id,
                prompt="Repair the release report.",
                workspace=str(tmp_path / "workspace" / task_id),
                success=True,
                task_metadata={"benchmark_family": "repository", "memory_source": "episode"},
                task_contract={
                    "prompt": "Repair the release report.",
                    "workspace_subdir": task_id,
                    "setup_commands": [],
                    "success_command": "true",
                    "suggested_commands": [],
                    "expected_files": ["reports/release_review.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {},
                    "max_steps": 3,
                },
                termination_reason="success",
                steps=[
                    StepRecord(
                        index=1,
                        thought="prepare reports dir",
                        action="code_execute",
                        content="mkdir -p reports",
                        selected_skill_id=None,
                        command_result={
                            "command": "mkdir -p reports",
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "timed_out": False,
                        },
                        verification={"passed": True, "reasons": ["verification passed"]},
                        state_progress_delta=0.5,
                        state_transition={"progress_delta": 0.5},
                    ),
                    StepRecord(
                        index=2,
                        thought="write release report",
                        action="code_execute",
                        content="printf 'ready\\n' > reports/release_review.txt",
                        selected_skill_id=None,
                        command_result={
                            "command": "printf 'ready\\n' > reports/release_review.txt",
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "timed_out": False,
                        },
                        verification={"passed": True, "reasons": ["verification passed"]},
                        state_progress_delta=1.0,
                        state_transition={
                            "progress_delta": 1.0,
                            "newly_updated_report_paths": ["reports/release_review.txt"],
                        },
                    ),
                ],
            )
        )

    prototypes = memory.semantic_prototype_recall(
        benchmark_family="repository",
        changed_paths=["reports/final_review.txt"],
        require_success=True,
        limit=1,
    )
    assert prototypes
    assert prototypes[0]["application_commands"][0] == "printf 'ready\\n' > reports/release_review.txt"
    assert "mkdir -p reports || printf 'ready\\n' > reports/release_review.txt" in prototypes[0]["command_sequences"]
    assert prototypes[0]["transform_semantics"]["target_kinds"]["report"] >= 1
    assert "{target_path}" in next(iter(prototypes[0]["application_command_templates"]))
    assert "printf 'ready\\n' > reports/final_review.txt" in prototypes[0]["instantiated_application_commands"]

    model = WorldModel()
    summary = {
        "benchmark_family": "repository",
        "horizon": "long_horizon",
        "expected_artifacts": ["reports/final_review.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["reports/final_review.txt"],
        "present_forbidden_artifacts": [],
        "workflow_expected_changed_paths": ["src/release_state.txt"],
        "workflow_generated_paths": [],
        "workflow_report_paths": ["reports/final_review.txt"],
        "semantic_prototypes": prototypes,
    }
    rollout = model.simulate_command_sequence_effect(
        summary,
        ["mkdir -p reports", "printf 'ready\\n' > reports/final_review.txt"],
    )
    assert rollout["sequence_alignment_bonus"] > 0.0
    assert rollout["predicted_progress_gain"] >= 1.0
    assert rollout["latent_transition_score"] > 0.0
    assert rollout["latent_transition"]["future_state_signature"]

    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="closure_contract_task",
            prompt="Repair the release report.",
            workspace_subdir="closure_contract_task",
            suggested_commands=[
                "printf 'draft\\n' > reports/final_review.txt",
                "mkdir -p reports",
                "printf 'ready\\n' > reports/final_review.txt",
                "python -m pytest tests/test_release.py",
            ],
            expected_files=["reports/final_review.txt"],
            metadata={
                "benchmark_family": "repository",
                "difficulty": "long_horizon",
                "semantic_verifier": {
                    "expected_changed_paths": ["src/release_state.txt"],
                    "report_rules": [{"path": "reports/final_review.txt", "must_mention": ["ready"]}],
                },
            },
        )
    )
    state.world_model_summary = dict(summary)
    state.graph_summary = {"semantic_prototypes": prototypes}
    state.latest_state_transition = {"no_progress": True, "regressions": []}
    state.latent_state_summary = {
        "active_paths": ["src/release_state.txt", "reports/final_review.txt"],
        "learned_world_state": {"progress_signal": 0.1, "risk_signal": 0.6},
    }

    preview = policy._transition_preview(state)

    assert preview["candidates"]
    assert preview["planning_mode"] == "hierarchical"
    assert preview["subgoal_plan"]
    assert any(
        candidate["command"] == "printf 'ready\\n' > reports/final_review.txt"
        for candidate in preview["candidates"]
    )
    assert any(str(branch.get("subgoal", "")).strip() for branch in preview["search_branches"])

    verifier = Verifier()
    verify_workspace = tmp_path / "verify_workspace"
    verify_workspace.mkdir(parents=True, exist_ok=True)
    (verify_workspace / "reports").mkdir(parents=True, exist_ok=True)
    (verify_workspace / "reports" / "final_review.txt").write_text("ready\n", encoding="utf-8")
    (verify_workspace / "reports" / "status.json").write_text(
        '{"status":"ready","score":9}\n',
        encoding="utf-8",
    )
    verify_task = TaskSpec(
        task_id="closure_contract_verify",
        prompt="Run semantic behavior checks with workspace side effects.",
        workspace_subdir="closure_contract_verify",
        metadata={
            "semantic_verifier": {
                "behavior_checks": [
                    {
                        "label": "workspace repair",
                        "argv": [
                            "/bin/sh",
                            "-lc",
                            "mkdir -p reports && printf 'ready\\n' > reports/release_review.txt",
                        ],
                        "file_expectations": [
                            {
                                "path": "reports/release_review.txt",
                                "must_exist": True,
                                "must_contain": ["ready"],
                            }
                        ],
                        "repo_invariants": [
                            {
                                "kind": "file_contains",
                                "path": "reports/release_review.txt",
                                "must_contain": ["ready"],
                            }
                        ],
                    }
                ],
                "semantic_assertions": [
                    {
                        "label": "stdout score",
                        "source": "stdout_text",
                        "capture_regex": r"score=(\d+)",
                        "min": 8,
                    },
                    {
                        "label": "workspace report text",
                        "source": "workspace_file_text",
                        "path": "reports/final_review.txt",
                        "contains": ["ready"],
                        "not_regex": r"broken|failed",
                    },
                    {
                        "label": "workspace status json",
                        "source": "workspace_file_json",
                        "path": "reports/status.json",
                        "json_fields": [
                            {"path": "status", "equals": "ready"},
                            {"path": "score", "min": 8},
                        ],
                    },
                ],
            }
        },
    )
    verification = verifier.verify(
        verify_task,
        verify_workspace,
        CommandResult(command="true", exit_code=0, stdout="score=9\n", stderr=""),
    )
    assert verification.passed is True


def test_locked_asi_closure_contract_unattended_status_surface():
    module = _load_script("run_unattended_campaign.py")

    projected = module._live_status_projection(
        payload={
            "status": "interrupted",
            "reason": "runtime ceiling",
            "event_log_path": "/tmp/unattended.events.jsonl",
            "lock_path": "/tmp/unattended.lock",
            "unattended_evidence": {
                "retained_open_world_gain": True,
                "external_summary": {
                    "total": 1,
                    "distinct_benchmark_families": 1,
                    "benchmark_families": ["repository"],
                },
                "semantic_hub_summary": {
                    "reports": 2,
                    "distinct_benchmark_families": 2,
                    "benchmark_families": ["integration", "project"],
                },
                "replay_derived_summary": {
                    "reports": 1,
                    "distinct_benchmark_families": 1,
                    "benchmark_families": ["repository"],
                },
                },
                "trust_breadth_summary": {
                    "required_families": ["integration", "project", "repository", "repo_chore"],
                    "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                    "missing_required_families": [],
                    "external_report_count": 1,
                    "distinct_external_benchmark_families": 1,
                    "distinct_family_gap": 0,
                },
            "campaign_report": {
                "recent_production_decisions": [
                    {
                        "cycle_id": "cycle:retrieval:1",
                        "subsystem": "retrieval",
                        "state": "retain",
                        "metrics_summary": {
                            "baseline_trusted_carryover_repair_rate": 0.25,
                            "trusted_carryover_repair_rate": 0.5,
                            "baseline_trusted_carryover_verified_steps": 1,
                            "trusted_carryover_verified_steps": 2,
                            "trusted_carryover_verified_step_delta": 1,
                        },
                    },
                    {
                        "cycle_id": "cycle:tolbert:1",
                        "subsystem": "tolbert_model",
                        "state": "retain",
                    }
                ],
                "runtime_managed_decisions": 2,
                "retained_gain_runs": 1,
                "runs": [
                    {
                        "runtime_managed_decisions": 2,
                        "decision_records_considered": 2,
                        "retained_gain": True,
                        "final_state": "retain",
                    }
                ],
            },
        },
        active_run={
            "provider": "hybrid",
            "policy": {"priority_benchmark_families": ["integration", "project", "repository"]},
            "child_status": {"mirrored": True},
        },
        active_child={
            "last_progress_phase": "metrics_finalize",
            "last_report_path": "/tmp/campaign_report.json",
            "sampled_families_from_progress": ["integration", "project", "repository"],
            "active_cycle_progress": {
                "productive_partial": True,
                "candidate_generated": True,
                "generated_success_started": True,
                "generated_success_completed": True,
                "generated_failure_completed": True,
                "raw_phase": "metrics_finalize",
            },
            "pending_decision_state": "retain",
            "runtime_managed_decisions": 2,
        },
    )

    for field in (
        "open_world_breadth_summary",
        "provider_independence_summary",
        "retrieval_carryover_summary",
        "tolbert_retained_routing_summary",
        "long_horizon_finalize_summary",
        "handoff_summary",
        "closure_gap_summary",
        "benchmark_dominance_summary",
    ):
        assert field in projected
    assert projected["open_world_breadth_summary"]["retained_open_world_gain"] is True
    assert set(projected["open_world_breadth_summary"]["benchmark_families"]) == {
        "integration",
        "project",
        "repository",
    }
    assert projected["closure_gap_summary"] == {
        "retained_conversion": "closed",
        "trust_breadth": "closed",
        "open_world_breadth": "closed",
        "decoder_runtime_independence": "closed",
        "tolbert_retained_routing": "closed",
        "retrieval_carryover": "closed",
        "long_horizon_finalize_execution": "closed",
        "evidence_hygiene_handoff": "closed",
    }
    assert projected["benchmark_dominance_summary"]["required_family_statuses"] == {
        "integration": "closed",
        "project": "closed",
        "repository": "closed",
        "repo_chore": "closed",
    }
    assert projected["benchmark_dominance_summary"]["contract_state"] == "closed"
