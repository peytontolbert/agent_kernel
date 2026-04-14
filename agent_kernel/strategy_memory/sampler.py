from __future__ import annotations

from math import log, sqrt

from ..extensions.strategy.semantic_hub import semantic_query_terms
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


def _scalar_control_mapping(controls: dict[str, object]) -> dict[str, object]:
    filtered: dict[str, object] = {}
    for raw_key, value in dict(controls).items():
        key = str(raw_key).strip()
        if not key or key in {"history", "base_subsystem"} or key.startswith("strategy_memory"):
            continue
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
    return filtered


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _parent_control_surface(selected: list[StrategyNode], matching: list[StrategyNode], *, best_retained: StrategyNode | None) -> dict[str, object]:
    preferred_controls = _scalar_control_mapping(best_retained.controls) if best_retained is not None else {}
    rejected_controls = [
        _scalar_control_mapping(node.controls)
        for node in matching
        if node.retention_state == "reject"
    ]
    rejected_controls = [controls for controls in rejected_controls if controls][:3]
    reuse_conditions = _ordered_unique(
        [
            *(
                condition
                for node in selected
                if node.retention_state == "retain"
                for condition in list(node.reuse_conditions)
            ),
        ]
    )
    avoid_conditions = _ordered_unique(
        [
            *(
                condition
                for node in selected
                for condition in list(node.avoid_conditions)
            ),
        ]
    )
    semantic_hypotheses = _ordered_unique(
        [
            *(
                hypothesis
                for node in selected
                for hypothesis in list(node.semantic_hypotheses)
            ),
        ]
    )[:12]
    required_family_gains = sorted(
        {
            str(family).strip()
            for node in selected
            if node.retention_state == "retain"
            for family, value in dict(node.family_coverage).items()
            if str(family).strip() and float(value or 0.0) > 0.0
        }
    )
    reject_family_pressure: dict[str, int] = {}
    for node in matching:
        if node.retention_state != "reject":
            continue
        for family, value in dict(node.family_coverage).items():
            family_token = str(family).strip()
            if not family_token or float(value or 0.0) >= 0.0:
                continue
            reject_family_pressure[family_token] = int(reject_family_pressure.get(family_token, 0) or 0) + 1
    closeout_modes = _ordered_unique(
        [
            str(dict(node.execution_evidence).get("closeout_mode", "")).strip()
            for node in selected
            if isinstance(node.execution_evidence, dict)
        ]
    )
    prefer_family_breadth = any(
        int(dict(node.execution_evidence).get("family_breadth_gain", 0) or 0) >= 2
        for node in selected
        if isinstance(node.execution_evidence, dict)
    ) or len(required_family_gains) >= 2
    prefer_unattended_closeout = any(
        bool(dict(node.execution_evidence).get("closeout_ready", False))
        for node in selected
        if isinstance(node.execution_evidence, dict)
    )
    return {
        "best_parent_strategy_node_id": "" if best_retained is None else best_retained.strategy_node_id,
        "preferred_controls": preferred_controls,
        "rejected_controls": rejected_controls,
        "reuse_conditions": reuse_conditions,
        "avoid_conditions": avoid_conditions,
        "semantic_hypotheses": semantic_hypotheses,
        "required_family_gains": required_family_gains,
        "reject_family_pressure": dict(sorted(reject_family_pressure.items())),
        "prefer_family_breadth": prefer_family_breadth,
        "prefer_unattended_closeout": prefer_unattended_closeout,
        "closeout_modes": closeout_modes,
    }


def summarize_strategy_priors(
    nodes: list[StrategyNode],
    *,
    subsystem: str,
    strategy_candidate_id: str,
    semantic_hypotheses: list[str] | None = None,
    context_terms: list[object] | None = None,
    limit: int = 3,
) -> dict[str, object]:
    query_terms = semantic_query_terms(
        subsystem,
        strategy_candidate_id,
        list(semantic_hypotheses or []),
        list(context_terms or []),
    )
    if not nodes:
        return {
            "selected_parent_strategy_node_ids": [],
            "selected_parent_nodes": [],
            "semantic_parent_matches": [],
            "semantic_query_terms": query_terms,
            "parent_selection_mode": "empty",
            "best_retained_strategy_node_id": "",
            "continuation_parent_node_id": "",
            "continuation_artifact_path": "",
            "continuation_workspace_ref": "",
            "continuation_branch": "",
            "best_retained_snapshot": {},
            "parent_control_surface": {},
            "score_delta": 0.0,
            "avoid_reselection": False,
            "retained_count": 0,
            "rejected_count": 0,
            "recent_rejects": 0,
            "recent_retains": 0,
        }
    subsystem = str(subsystem).strip()
    strategy_candidate_id = str(strategy_candidate_id).strip()
    relevance_by_id = {
        node.strategy_node_id: _parent_relevance(
            node,
            subsystem=subsystem,
            strategy_candidate_id=strategy_candidate_id,
            query_terms=query_terms,
        )
        for node in nodes
    }
    exact_matches = [
        node
        for node in nodes
        if (
            strategy_candidate_id
            and (node.strategy_id == strategy_candidate_id or node.strategy_candidate_id == strategy_candidate_id)
        )
    ]
    subsystem_matches = [node for node in nodes if subsystem and node.subsystem == subsystem]
    semantic_matches = [
        node
        for node in nodes
        if relevance_by_id.get(node.strategy_node_id, {}).get("semantic_score", 0.0)
    ]
    matching = [
        node
        for node in nodes
        if relevance_by_id.get(node.strategy_node_id, {}).get("score", 0.0) > 0.0
    ] or list(nodes)
    if exact_matches:
        parent_selection_mode = "exact_or_semantic"
    elif semantic_matches:
        parent_selection_mode = "semantic"
    elif subsystem_matches:
        parent_selection_mode = "subsystem"
    else:
        parent_selection_mode = "global"
    total_visits = sum(max(1, node.visit_count) for node in nodes)
    ranked = sorted(
        matching,
        key=lambda node: (
            node.retention_state != "retain",
            -float(relevance_by_id.get(node.strategy_node_id, {}).get("score", 0.0) or 0.0),
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
            "strategy_origin": node.strategy_origin,
            "retention_state": node.retention_state,
            "retained_gain": float(node.retained_gain),
            "analysis_lesson": node.analysis_lesson,
            "reuse_conditions": list(node.reuse_conditions),
            "avoid_conditions": list(node.avoid_conditions),
            "semantic_hypotheses": list(node.semantic_hypotheses),
            "semantic_match_score": float(
                relevance_by_id.get(node.strategy_node_id, {}).get("semantic_score", 0.0) or 0.0
            ),
            "semantic_matched_terms": list(
                relevance_by_id.get(node.strategy_node_id, {}).get("matched_terms", [])
            ),
            "controls": _scalar_control_mapping(node.controls),
            "family_coverage": dict(node.family_coverage),
            "execution_evidence": dict(node.execution_evidence),
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
            "strategy_origin": best_retained.strategy_origin,
            "retained_gain": float(best_retained.retained_gain),
            "analysis_lesson": best_retained.analysis_lesson,
            "reuse_conditions": list(best_retained.reuse_conditions),
            "avoid_conditions": list(best_retained.avoid_conditions),
            "semantic_hypotheses": list(best_retained.semantic_hypotheses),
            "semantic_match_score": float(
                relevance_by_id.get(best_retained.strategy_node_id, {}).get("semantic_score", 0.0) or 0.0
            ),
            "semantic_matched_terms": list(
                relevance_by_id.get(best_retained.strategy_node_id, {}).get("matched_terms", [])
            ),
            "controls": _scalar_control_mapping(best_retained.controls),
            "parent_strategy_node_ids": list(best_retained.parent_strategy_node_ids),
            "continuation_parent_node_id": str(best_retained.continuation_parent_node_id),
            "continuation_artifact_path": str(best_retained.continuation_artifact_path),
            "continuation_workspace_ref": str(best_retained.continuation_workspace_ref),
            "continuation_branch": str(best_retained.continuation_branch),
            "results_summary": dict(best_retained.results_summary),
            "execution_evidence": dict(best_retained.execution_evidence),
            "actor_summary": dict(best_retained.actor_summary),
            "family_coverage": dict(best_retained.family_coverage),
        }
    return {
        "selected_parent_strategy_node_ids": [node.strategy_node_id for node in selected],
        "selected_parent_nodes": selected_parent_nodes,
        "parent_control_surface": _parent_control_surface(selected, matching, best_retained=best_retained),
        "semantic_parent_matches": _semantic_parent_matches(
            ranked,
            relevance_by_id=relevance_by_id,
            limit=limit,
        ),
        "semantic_query_terms": query_terms,
        "parent_selection_mode": parent_selection_mode,
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


def _node_semantic_terms(node: StrategyNode) -> list[str]:
    return semantic_query_terms(
        node.subsystem,
        node.strategy_id,
        node.strategy_candidate_id,
        node.strategy_candidate_kind,
        node.motivation,
        node.analysis_lesson,
        node.reuse_conditions,
        node.avoid_conditions,
        node.semantic_hypotheses,
        node.family_coverage,
        node.controls,
        node.actor_summary,
        node.results_summary,
    )


def _parent_relevance(
    node: StrategyNode,
    *,
    subsystem: str,
    strategy_candidate_id: str,
    query_terms: list[str],
) -> dict[str, object]:
    exact = bool(
        strategy_candidate_id
        and (node.strategy_id == strategy_candidate_id or node.strategy_candidate_id == strategy_candidate_id)
    )
    subsystem_match = bool(subsystem and node.subsystem == subsystem)
    node_terms = set(_node_semantic_terms(node))
    matched_terms = [term for term in query_terms if term in node_terms]
    hypothesis_terms = set(semantic_query_terms(node.semantic_hypotheses))
    hypothesis_matches = [term for term in query_terms if term in hypothesis_terms]
    semantic_score = 0.0
    if query_terms:
        semantic_score = round(
            (len(matched_terms) / max(1, len(query_terms)))
            + (0.5 * len(hypothesis_matches) / max(1, len(query_terms))),
            4,
        )
    score = 0.0
    if exact:
        score += 8.0
    if subsystem_match:
        score += 3.0
    score += semantic_score
    return {
        "score": round(score, 4),
        "semantic_score": semantic_score,
        "matched_terms": matched_terms,
        "exact": exact,
        "subsystem_match": subsystem_match,
    }


def _semantic_parent_matches(
    ranked: list[StrategyNode],
    *,
    relevance_by_id: dict[str, dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for node in ranked:
        relevance = relevance_by_id.get(node.strategy_node_id, {})
        semantic_score = float(relevance.get("semantic_score", 0.0) or 0.0)
        if semantic_score <= 0.0:
            continue
        matches.append(
            {
                "strategy_node_id": node.strategy_node_id,
                "strategy_id": node.strategy_id or node.strategy_candidate_id,
                "retention_state": node.retention_state,
                "semantic_match_score": semantic_score,
                "semantic_matched_terms": list(relevance.get("matched_terms", [])),
                "semantic_hypotheses": list(node.semantic_hypotheses),
            }
        )
    return matches[: max(0, int(limit))]
