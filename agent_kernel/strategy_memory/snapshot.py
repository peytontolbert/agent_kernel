from __future__ import annotations

import json
from pathlib import Path

from .node import StrategyNode


def load_strategy_snapshots(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_strategy_snapshots(nodes: list[StrategyNode]) -> dict[str, object]:
    retained = [node for node in nodes if node.retention_state == "retain"]
    by_subsystem: dict[str, dict[str, object]] = {}
    by_family: dict[str, dict[str, object]] = {}
    by_kind: dict[str, dict[str, object]] = {}

    def _node_summary(node: StrategyNode) -> dict[str, object]:
        return {
            "strategy_node_id": node.strategy_node_id,
            "cycle_id": node.cycle_id,
            "subsystem": node.subsystem,
            "selected_variant_id": node.selected_variant_id,
            "strategy_id": node.strategy_id or node.strategy_candidate_id,
            "strategy_candidate_id": node.strategy_candidate_id,
            "strategy_candidate_kind": node.strategy_candidate_kind,
            "parent_strategy_node_ids": list(node.parent_strategy_node_ids),
            "retained_gain": float(node.retained_gain),
            "score": float(node.score),
            "analysis_lesson": node.analysis_lesson,
            "reuse_conditions": list(node.reuse_conditions),
            "avoid_conditions": list(node.avoid_conditions),
            "continuation_parent_node_id": node.continuation_parent_node_id,
            "continuation_artifact_path": node.continuation_artifact_path,
            "actor_summary": dict(node.actor_summary),
            "results_summary": dict(node.results_summary),
            "family_coverage": dict(node.family_coverage),
            "artifact_paths": dict(node.artifact_paths),
            "updated_at": node.updated_at,
        }

    for node in retained:
        current = by_subsystem.get(node.subsystem)
        if current is None or float(current.get("retained_gain", 0.0) or 0.0) < node.retained_gain:
            by_subsystem[node.subsystem] = _node_summary(node)
        current_kind = by_kind.get(node.strategy_candidate_kind)
        if current_kind is None or float(current_kind.get("retained_gain", 0.0) or 0.0) < node.retained_gain:
            by_kind[node.strategy_candidate_kind] = _node_summary(node)
        for family, value in node.family_coverage.items():
            if not value:
                continue
            current_family = by_family.get(family)
            if current_family is None or float(current_family.get("retained_gain", 0.0) or 0.0) < node.retained_gain:
                by_family[str(family)] = _node_summary(node)
    return {
        "summary": {
            "total_nodes": len(nodes),
            "retained_nodes": len(retained),
            "rejected_nodes": sum(1 for node in nodes if node.retention_state == "reject"),
            "pending_nodes": sum(1 for node in nodes if node.retention_state == "pending"),
            "latest_updated_at": max((node.updated_at for node in nodes), default=""),
        },
        "best_retained_by_subsystem": by_subsystem,
        "best_retained_by_family_cluster": by_family,
        "best_retained_by_strategy_kind": by_kind,
    }
