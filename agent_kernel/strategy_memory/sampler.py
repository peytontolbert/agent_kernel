from __future__ import annotations

from math import log, sqrt

from .node import StrategyNode


def node_value(node: StrategyNode, *, total_visits: int) -> float:
    base = float(node.retained_gain)
    if node.retention_state == "retain" and base <= 0.0:
        base = 0.15
    elif node.retention_state == "reject" and base >= 0.0:
        base = -0.2
    family_bonus = min(0.08, 0.02 * len([key for key, value in node.family_coverage.items() if value]))
    exploration = sqrt(log(max(2, total_visits + 1)) / max(1, node.visit_count + 1))
    return round(base + family_bonus + 0.05 * exploration, 4)


def summarize_strategy_priors(
    nodes: list[StrategyNode],
    *,
    subsystem: str,
    strategy_candidate_id: str,
    limit: int = 3,
) -> dict[str, object]:
    if not nodes:
        return {
            "selected_parent_strategy_node_ids": [],
            "selected_parent_nodes": [],
            "best_retained_strategy_node_id": "",
            "continuation_parent_node_id": "",
            "continuation_artifact_path": "",
            "continuation_workspace_ref": "",
            "continuation_branch": "",
            "best_retained_snapshot": {},
            "score_delta": 0.0,
            "avoid_reselection": False,
            "retained_count": 0,
            "rejected_count": 0,
            "recent_rejects": 0,
            "recent_retains": 0,
        }
    subsystem = str(subsystem).strip()
    strategy_candidate_id = str(strategy_candidate_id).strip()
    exact_matches = [
        node
        for node in nodes
        if (
            strategy_candidate_id
            and (node.strategy_id == strategy_candidate_id or node.strategy_candidate_id == strategy_candidate_id)
        )
    ]
    subsystem_matches = [node for node in nodes if subsystem and node.subsystem == subsystem]
    matching = exact_matches or subsystem_matches or list(nodes)
    total_visits = sum(max(1, node.visit_count) for node in nodes)
    ranked = sorted(
        matching,
        key=lambda node: (
            node.retention_state != "retain",
            -node_value(node, total_visits=total_visits),
            -(node.visit_count),
            node.strategy_node_id,
        ),
    )
    selected = ranked[: max(0, limit)]
    recent = sorted(matching, key=lambda node: (node.updated_at, node.created_at, node.strategy_node_id))[-3:]
    recent_rejects = sum(1 for node in recent if node.retention_state == "reject")
    recent_retains = sum(1 for node in recent if node.retention_state == "retain")
    best_retained = next((node for node in ranked if node.retention_state == "retain"), None)
    retained_count = sum(1 for node in matching if node.retention_state == "retain")
    rejected_count = sum(1 for node in matching if node.retention_state == "reject")
    score_delta = 0.0
    if best_retained is not None:
        score_delta += min(0.06, max(0.0, best_retained.retained_gain) * 0.1 + 0.01)
    if recent_rejects > 0 and recent_retains == 0:
        score_delta -= min(0.08, 0.02 * recent_rejects)
    selected_parent_nodes = [
        {
            "strategy_node_id": node.strategy_node_id,
            "strategy_id": node.strategy_id or node.strategy_candidate_id,
            "strategy_candidate_kind": node.strategy_candidate_kind,
            "retention_state": node.retention_state,
            "retained_gain": float(node.retained_gain),
            "analysis_lesson": node.analysis_lesson,
            "reuse_conditions": list(node.reuse_conditions),
            "avoid_conditions": list(node.avoid_conditions),
            "family_coverage": dict(node.family_coverage),
            "continuation_artifact_path": str(node.continuation_artifact_path),
        }
        for node in selected
    ]
    best_retained_snapshot = {}
    if best_retained is not None:
        best_retained_snapshot = {
            "strategy_node_id": best_retained.strategy_node_id,
            "strategy_id": best_retained.strategy_id or best_retained.strategy_candidate_id,
            "strategy_candidate_kind": best_retained.strategy_candidate_kind,
            "retained_gain": float(best_retained.retained_gain),
            "analysis_lesson": best_retained.analysis_lesson,
            "reuse_conditions": list(best_retained.reuse_conditions),
            "avoid_conditions": list(best_retained.avoid_conditions),
            "parent_strategy_node_ids": list(best_retained.parent_strategy_node_ids),
            "continuation_parent_node_id": str(best_retained.continuation_parent_node_id),
            "continuation_artifact_path": str(best_retained.continuation_artifact_path),
            "continuation_workspace_ref": str(best_retained.continuation_workspace_ref),
            "continuation_branch": str(best_retained.continuation_branch),
            "results_summary": dict(best_retained.results_summary),
            "actor_summary": dict(best_retained.actor_summary),
            "family_coverage": dict(best_retained.family_coverage),
        }
    return {
        "selected_parent_strategy_node_ids": [node.strategy_node_id for node in selected],
        "selected_parent_nodes": selected_parent_nodes,
        "best_retained_strategy_node_id": "" if best_retained is None else best_retained.strategy_node_id,
        "continuation_parent_node_id": "" if best_retained is None else best_retained.strategy_node_id,
        "continuation_artifact_path": "" if best_retained is None else str(best_retained.continuation_artifact_path),
        "continuation_workspace_ref": "" if best_retained is None else str(best_retained.continuation_workspace_ref),
        "continuation_branch": "" if best_retained is None else str(best_retained.continuation_branch),
        "best_retained_snapshot": best_retained_snapshot,
        "best_retained_gain": 0.0 if best_retained is None else float(best_retained.retained_gain),
        "score_delta": round(score_delta, 4),
        "avoid_reselection": recent_rejects >= 2 and recent_retains == 0,
        "retained_count": retained_count,
        "rejected_count": rejected_count,
        "recent_rejects": recent_rejects,
        "recent_retains": recent_retains,
    }
