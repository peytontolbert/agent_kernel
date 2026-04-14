from __future__ import annotations

import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.strategy.semantic_hub import query_semantic_items, record_semantic_note
from agent_kernel.strategy_memory import (
    StrategyNode,
    finalize_strategy_node,
    load_strategy_nodes,
    load_strategy_snapshots,
    record_pending_strategy_node,
)
from agent_kernel.strategy_memory.sampler import summarize_strategy_priors


def _config(tmp_path):
    return KernelConfig(
        workspace_root=tmp_path,
        trajectories_root=tmp_path / "trajectories",
        improvement_cycles_path=tmp_path / "trajectories/improvement/cycles.jsonl",
        candidate_artifacts_root=tmp_path / "trajectories/improvement/candidates",
        improvement_reports_dir=tmp_path / "trajectories/improvement/reports",
        strategy_memory_nodes_path=tmp_path / "trajectories/improvement/strategy_memory/nodes.jsonl",
        strategy_memory_snapshots_path=tmp_path / "trajectories/improvement/strategy_memory/snapshots.json",
        semantic_hub_root=tmp_path / "trajectories/semantic_hub",
        run_reports_dir=tmp_path / "trajectories/reports",
        run_checkpoints_dir=tmp_path / "trajectories/checkpoints",
        runtime_database_path=tmp_path / "var/runtime/agentkernel.sqlite3",
        unattended_trust_ledger_path=tmp_path / "trajectories/reports/unattended_trust_ledger.json",
    )


def test_strategy_memory_records_pending_and_finalized_nodes(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()

    pending = record_pending_strategy_node(
        config,
        cycle_id="cycle:retrieval:test",
        subsystem="retrieval",
        selected_variant_id="confidence_gating",
        strategy_candidate={
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "origin": "discovered_strategy",
            "parent_strategy_node_ids": ["strategy_node:older"],
        },
        motivation="reduce retrieval churn",
        controls={"tolbert_confidence_threshold": 0.2},
        score=0.5,
        family_coverage={"repository": 3},
        artifact_paths={"active_artifact_path": "trajectories/retrieval/retrieval_proposals.json"},
    )
    assert pending.strategy_node_id

    report = {
        "cycle_id": "cycle:retrieval:test",
        "subsystem": "retrieval",
        "strategy_candidate": {
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "origin": "discovered_strategy",
            "selected_variant_id": "confidence_gating",
        },
        "strategy_candidate_id": "strategy:retrieval_support_rebalance",
        "strategy_candidate_kind": "retrieval_support_rebalance",
        "final_state": "retain",
        "final_reason": "retrieval candidate improved broad coding-family support without regressing the base lane",
        "artifact_path": "trajectories/retrieval/retrieval_proposals.json",
        "candidate_artifact_path": "trajectories/improvement/candidates/retrieval/test/retrieval_proposals.json",
        "artifact_snapshot_path": "trajectories/improvement/candidates/retrieval/test/snapshot.json",
        "report_path": "trajectories/improvement/reports/cycle_report_cycle_retrieval_test.json",
        "evidence": {
            "pass_rate_delta": 0.1,
            "average_step_delta": -1.0,
            "trusted_carryover_repair_rate_delta": 0.2,
            "false_failure_rate": 0.0,
            "family_pass_rate_delta": {"repository": 0.2},
        },
    }
    finalized = finalize_strategy_node(config, report)

    nodes = load_strategy_nodes(config)
    assert len(nodes) == 1
    assert nodes[0].strategy_node_id == pending.strategy_node_id
    assert nodes[0].strategy_id == "strategy:retrieval_support_rebalance"
    assert nodes[0].strategy_origin == "discovered_strategy"
    assert finalized.retention_state == "retain"
    assert finalized.retained_gain > 0.0
    assert finalized.continuation_parent_node_id == "strategy_node:older"
    assert finalized.parent_strategy_node_ids == ["strategy_node:older"]
    assert finalized.artifact_paths["semantic_attempt_path"].endswith(".json")
    semantic_attempt = json.loads(
        Path(finalized.artifact_paths["semantic_attempt_path"]).read_text(encoding="utf-8")
    )
    assert semantic_attempt["continuation_parent_node_id"] == "strategy_node:older"
    assert semantic_attempt["status"] == "retain"
    assert semantic_attempt["strategy_id"] == "strategy:retrieval_support_rebalance"
    assert semantic_attempt["strategy_origin"] == "discovered_strategy"
    assert semantic_attempt["execution_evidence"]["family_breadth_gain"] == 1
    note_paths = list((tmp_path / "trajectories/semantic_hub/notes").glob("*.json"))
    assert note_paths
    semantic_note = json.loads(
        note_paths[0].read_text(encoding="utf-8")
    )
    assert semantic_note["retention_state"] == "retain"
    assert semantic_note["execution_evidence"]["phase_gate_passed"] is False
    snapshots = load_strategy_snapshots(config.strategy_memory_snapshots_path)
    assert snapshots["summary"]["retained_nodes"] == 1
    assert (
        snapshots["best_retained_by_subsystem"]["retrieval"]["strategy_node_id"]
        == pending.strategy_node_id
    )
    assert (
        snapshots["best_retained_by_subsystem"]["retrieval"]["strategy_id"]
        == "strategy:retrieval_support_rebalance"
    )
    assert snapshots["best_retained_by_subsystem"]["retrieval"]["strategy_origin"] == "discovered_strategy"
    assert snapshots["best_retained_by_strategy_kind"]["retrieval_support_rebalance"]["analysis_lesson"]
    assert snapshots["best_retained_by_family_cluster"]["repository"]["results_summary"]["status"] == "retain"


def test_strategy_sampler_penalizes_repeated_reject_only_history(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()
    for cycle_id in ("cycle:retrieval:a", "cycle:retrieval:b"):
        record_pending_strategy_node(
            config,
            cycle_id=cycle_id,
            subsystem="retrieval",
            selected_variant_id="confidence_gating",
            strategy_candidate={
                "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                "strategy_candidate_kind": "retrieval_support_rebalance",
            },
            motivation="reduce retrieval churn",
            controls={},
            score=0.0,
            family_coverage={},
            artifact_paths={},
        )
        finalize_strategy_node(
            config,
            {
                "cycle_id": cycle_id,
                "subsystem": "retrieval",
                "strategy_candidate": {
                    "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                    "strategy_candidate_kind": "retrieval_support_rebalance",
                    "selected_variant_id": "confidence_gating",
                },
                "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                "strategy_candidate_kind": "retrieval_support_rebalance",
                "final_state": "reject",
                "final_reason": "retrieval candidate produced no material change from the retained artifact",
                "evidence": {
                    "pass_rate_delta": 0.0,
                    "average_step_delta": 0.0,
                    "trusted_carryover_repair_rate_delta": 0.0,
                    "false_failure_rate": 0.0,
                    "family_pass_rate_delta": {},
                },
            },
        )

    summary = summarize_strategy_priors(
        load_strategy_nodes(config),
        subsystem="retrieval",
        strategy_candidate_id="strategy:retrieval_support_rebalance",
    )
    assert summary["avoid_reselection"] is True
    assert summary["score_delta"] < 0.0
    assert summary["recent_rejects"] >= 2


def test_strategy_sampler_surfaces_continuation_parent_metadata(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()
    record_pending_strategy_node(
        config,
        cycle_id="cycle:retrieval:seed",
        subsystem="retrieval",
        selected_variant_id="confidence_gating",
        strategy_candidate={
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "continuation_workspace_ref": "workspace/retrieval_parent",
            "continuation_artifact_path": "trajectories/improvement/candidates/retrieval/seed/retrieval.json",
            "continuation_branch": "retrieval-parent",
        },
        motivation="seed retained parent",
        controls={},
        score=0.0,
        family_coverage={},
        artifact_paths={},
    )
    finalize_strategy_node(
        config,
        {
            "cycle_id": "cycle:retrieval:seed",
            "subsystem": "retrieval",
            "strategy_candidate": {
                "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                "strategy_candidate_kind": "retrieval_support_rebalance",
                "selected_variant_id": "confidence_gating",
            },
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "final_state": "retain",
            "final_reason": "retained",
            "candidate_artifact_path": "trajectories/improvement/candidates/retrieval/seed/retrieval.json",
            "evidence": {
                "pass_rate_delta": 0.05,
                "average_step_delta": 0.0,
                "trusted_carryover_repair_rate_delta": 0.0,
                "false_failure_rate": 0.0,
                "family_pass_rate_delta": {},
            },
        },
    )

    summary = summarize_strategy_priors(
        load_strategy_nodes(config),
        subsystem="retrieval",
        strategy_candidate_id="strategy:retrieval_support_rebalance",
    )
    assert summary["continuation_parent_node_id"].startswith("strategy_node:")
    assert summary["continuation_artifact_path"].endswith("retrieval.json")
    assert summary["continuation_workspace_ref"] == "workspace/retrieval_parent"
    assert summary["continuation_branch"] == "retrieval-parent"
    assert summary["best_retained_snapshot"]["strategy_id"] == "strategy:retrieval_support_rebalance"
    assert summary["best_retained_snapshot"]["execution_evidence"]["family_breadth_gain"] == 0
    assert summary["selected_parent_nodes"][0]["strategy_node_id"] == summary["continuation_parent_node_id"]


def test_strategy_sampler_exposes_parent_control_surface(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()
    record_pending_strategy_node(
        config,
        cycle_id="cycle:policy:seed",
        subsystem="policy",
        selected_variant_id="state_focus",
        strategy_candidate={
            "strategy_candidate_id": "strategy:adaptive_countermeasure:policy:seed",
            "strategy_candidate_kind": "adaptive_countermeasure",
            "semantic_hypotheses": ["state regression confidence gap"],
        },
        motivation="seed retained parent",
        controls={"planner_temperature": 0.2, "prefer_preview_yield": True},
        score=0.0,
        family_coverage={},
        artifact_paths={},
    )
    finalize_strategy_node(
        config,
        {
            "cycle_id": "cycle:policy:seed",
            "subsystem": "policy",
            "strategy_candidate": {
                "strategy_candidate_id": "strategy:adaptive_countermeasure:policy:seed",
                "strategy_candidate_kind": "adaptive_countermeasure",
                "selected_variant_id": "state_focus",
            },
            "strategy_candidate_id": "strategy:adaptive_countermeasure:policy:seed",
            "strategy_candidate_kind": "adaptive_countermeasure",
            "final_state": "retain",
            "final_reason": "retained with broad closeout-ready support",
            "closeout_mode": "natural",
            "phase_gate_report": {"passed": True, "failures": []},
            "evidence": {
                "pass_rate_delta": 0.08,
                "average_step_delta": -0.5,
                "trusted_carryover_repair_rate_delta": 0.0,
                "false_failure_rate": 0.0,
                "family_pass_rate_delta": {"integration": 0.1, "repository": 0.08},
            },
        },
    )

    summary = summarize_strategy_priors(
        load_strategy_nodes(config),
        subsystem="policy",
        strategy_candidate_id="strategy:adaptive_countermeasure:policy:seed",
    )

    assert summary["parent_control_surface"]["preferred_controls"]["planner_temperature"] == 0.2
    assert summary["parent_control_surface"]["prefer_family_breadth"] is True
    assert summary["parent_control_surface"]["prefer_unattended_closeout"] is True
    assert "integration" in summary["parent_control_surface"]["required_family_gains"]


def test_strategy_memory_updates_descendants_for_all_parent_links(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()
    parent_a = record_pending_strategy_node(
        config,
        cycle_id="cycle:retrieval:parent_a",
        subsystem="retrieval",
        selected_variant_id="seed_a",
        strategy_candidate={
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
        },
        motivation="seed parent a",
        controls={},
        score=0.0,
        family_coverage={},
        artifact_paths={},
    )
    finalize_strategy_node(
        config,
        {
            "cycle_id": "cycle:retrieval:parent_a",
            "subsystem": "retrieval",
            "strategy_candidate": {
                "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                "strategy_candidate_kind": "retrieval_support_rebalance",
                "selected_variant_id": "seed_a",
            },
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "final_state": "retain",
            "final_reason": "retained parent a",
            "evidence": {
                "pass_rate_delta": 0.03,
                "average_step_delta": 0.0,
                "trusted_carryover_repair_rate_delta": 0.0,
                "false_failure_rate": 0.0,
                "family_pass_rate_delta": {},
            },
        },
    )
    parent_b = record_pending_strategy_node(
        config,
        cycle_id="cycle:retrieval:parent_b",
        subsystem="retrieval",
        selected_variant_id="seed_b",
        strategy_candidate={
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
        },
        motivation="seed parent b",
        controls={},
        score=0.0,
        family_coverage={},
        artifact_paths={},
    )
    finalize_strategy_node(
        config,
        {
            "cycle_id": "cycle:retrieval:parent_b",
            "subsystem": "retrieval",
            "strategy_candidate": {
                "strategy_candidate_id": "strategy:retrieval_support_rebalance",
                "strategy_candidate_kind": "retrieval_support_rebalance",
                "selected_variant_id": "seed_b",
            },
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "final_state": "retain",
            "final_reason": "retained parent b",
            "evidence": {
                "pass_rate_delta": 0.02,
                "average_step_delta": 0.0,
                "trusted_carryover_repair_rate_delta": 0.0,
                "false_failure_rate": 0.0,
                "family_pass_rate_delta": {},
            },
        },
    )

    child = record_pending_strategy_node(
        config,
        cycle_id="cycle:retrieval:child",
        subsystem="retrieval",
        selected_variant_id="branch",
        strategy_candidate={
            "strategy_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_id": "strategy:retrieval_support_rebalance",
            "strategy_candidate_kind": "retrieval_support_rebalance",
            "continuation_parent_node_id": parent_a.strategy_node_id,
            "parent_strategy_node_ids": [parent_a.strategy_node_id, parent_b.strategy_node_id],
        },
        motivation="branch from two retained parents",
        controls={},
        score=0.0,
        family_coverage={},
        artifact_paths={},
    )

    nodes = {node.strategy_node_id: node for node in load_strategy_nodes(config)}
    assert child.parent_strategy_node_ids == [parent_a.strategy_node_id, parent_b.strategy_node_id]
    assert child.strategy_id == "strategy:retrieval_support_rebalance"
    assert child.strategy_node_id in nodes[parent_a.strategy_node_id].descendant_node_ids
    assert child.strategy_node_id in nodes[parent_b.strategy_node_id].descendant_node_ids


def test_semantic_hub_queries_artifacts_by_terms_and_filters(tmp_path):
    config = _config(tmp_path)
    config.ensure_directories()
    record_semantic_note(
        config,
        note_id="confidence-gap",
        payload={
            "strategy_id": "strategy:retrieval_support_rebalance",
            "subsystem": "retrieval",
            "retention_state": "retain",
            "analysis_lesson": "confidence gap recovers state regression with trusted carryover",
            "semantic_hypotheses": ["state regression confidence gap"],
        },
    )
    record_semantic_note(
        config,
        note_id="unrelated",
        payload={
            "strategy_id": "strategy:policy_budget",
            "subsystem": "policy",
            "retention_state": "reject",
            "analysis_lesson": "budget-only edits did not change verifier yield",
        },
    )

    matches = query_semantic_items(
        config,
        query="state regression confidence",
        categories=["notes"],
        filters={"subsystem": "retrieval"},
        limit=2,
    )

    assert [match["item_id"] for match in matches] == ["confidence-gap"]
    assert set(matches[0]["matched_terms"]) >= {"state", "regression", "confidence"}


def test_strategy_sampler_uses_semantic_hypotheses_for_cross_subsystem_parent_selection():
    semantic_parent = StrategyNode(
        strategy_node_id="strategy_node:semantic",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
        subsystem="retrieval",
        strategy_id="strategy:retrieval_support_rebalance",
        strategy_candidate_id="strategy:retrieval_support_rebalance",
        strategy_candidate_kind="retrieval_support_rebalance",
        retention_state="retain",
        retained_gain=0.2,
        analysis_lesson="trusted carryover repaired confidence gaps",
        reuse_conditions=["state regression under low retrieval confidence"],
        semantic_hypotheses=["state regression confidence gap"],
        visit_count=2,
        family_coverage={"repository": 0.1},
    )
    unrelated_parent = StrategyNode(
        strategy_node_id="strategy_node:unrelated",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-03T00:00:00+00:00",
        subsystem="policy",
        strategy_id="strategy:policy_budget",
        strategy_candidate_id="strategy:policy_budget",
        strategy_candidate_kind="budget_tuning",
        retention_state="retain",
        retained_gain=0.3,
        analysis_lesson="budget tuning helped direct execution",
        semantic_hypotheses=["step budget expansion"],
        visit_count=3,
    )

    summary = summarize_strategy_priors(
        [unrelated_parent, semantic_parent],
        subsystem="planner",
        strategy_candidate_id="strategy:new_countermeasure",
        semantic_hypotheses=["state regression confidence gap"],
    )

    assert summary["parent_selection_mode"] == "semantic"
    assert summary["selected_parent_strategy_node_ids"][0] == "strategy_node:semantic"
    assert summary["selected_parent_nodes"][0]["semantic_hypotheses"] == ["state regression confidence gap"]
    assert set(summary["semantic_parent_matches"][0]["semantic_matched_terms"]) >= {
        "state",
        "regression",
        "confidence",
    }
