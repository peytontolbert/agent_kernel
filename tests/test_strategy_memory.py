from __future__ import annotations

import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.strategy_memory import (
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
    note_paths = list((tmp_path / "trajectories/semantic_hub/notes").glob("*.json"))
    assert note_paths
    semantic_note = json.loads(
        note_paths[0].read_text(encoding="utf-8")
    )
    assert semantic_note["retention_state"] == "retain"
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
    assert summary["selected_parent_nodes"][0]["strategy_node_id"] == summary["continuation_parent_node_id"]


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
