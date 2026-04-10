from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _string_list(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return normalized


def _string_map(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    return {str(key): str(value) for key, value in values.items() if str(key).strip() and str(value).strip()}


def _object_map(values: object) -> dict[str, object]:
    if not isinstance(values, dict):
        return {}
    return {str(key): value for key, value in values.items() if str(key).strip()}


@dataclass(slots=True)
class StrategyNode:
    strategy_node_id: str
    created_at: str
    updated_at: str
    parent_strategy_node_ids: list[str] = field(default_factory=list)
    cycle_id: str = ""
    subsystem: str = ""
    selected_variant_id: str = ""
    strategy_id: str = ""
    strategy_candidate_id: str = ""
    strategy_candidate_kind: str = ""
    motivation: str = ""
    controls: dict[str, object] = field(default_factory=dict)
    actor_summary: dict[str, object] = field(default_factory=dict)
    results_summary: dict[str, object] = field(default_factory=dict)
    retention_state: str = "pending"
    retained_gain: float = 0.0
    analysis_lesson: str = ""
    reuse_conditions: list[str] = field(default_factory=list)
    avoid_conditions: list[str] = field(default_factory=list)
    continuation_parent_node_id: str = ""
    continuation_workspace_ref: str = ""
    continuation_artifact_path: str = ""
    continuation_branch: str = ""
    semantic_hypotheses: list[str] = field(default_factory=list)
    stagnation_count: int = 0
    descendant_node_ids: list[str] = field(default_factory=list)
    transfer_artifact_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    visit_count: int = 0
    family_coverage: dict[str, object] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_node_id": self.strategy_node_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_strategy_node_ids": list(self.parent_strategy_node_ids),
            "cycle_id": self.cycle_id,
            "subsystem": self.subsystem,
            "selected_variant_id": self.selected_variant_id,
            "strategy_id": self.strategy_id or self.strategy_candidate_id,
            "strategy_candidate_id": self.strategy_candidate_id,
            "strategy_candidate_kind": self.strategy_candidate_kind,
            "motivation": self.motivation,
            "controls": dict(self.controls),
            "actor_summary": dict(self.actor_summary),
            "results_summary": dict(self.results_summary),
            "retention_state": self.retention_state,
            "retained_gain": float(self.retained_gain),
            "analysis_lesson": self.analysis_lesson,
            "reuse_conditions": list(self.reuse_conditions),
            "avoid_conditions": list(self.avoid_conditions),
            "continuation_parent_node_id": self.continuation_parent_node_id,
            "continuation_workspace_ref": self.continuation_workspace_ref,
            "continuation_artifact_path": self.continuation_artifact_path,
            "continuation_branch": self.continuation_branch,
            "semantic_hypotheses": list(self.semantic_hypotheses),
            "stagnation_count": int(self.stagnation_count),
            "descendant_node_ids": list(self.descendant_node_ids),
            "transfer_artifact_ids": list(self.transfer_artifact_ids),
            "score": float(self.score),
            "visit_count": int(self.visit_count),
            "family_coverage": dict(self.family_coverage),
            "artifact_paths": dict(self.artifact_paths),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "StrategyNode | None":
        if not isinstance(payload, dict):
            return None
        strategy_node_id = str(payload.get("strategy_node_id", "")).strip()
        if not strategy_node_id:
            return None
        return cls(
            strategy_node_id=strategy_node_id,
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip() or str(payload.get("created_at", "")).strip(),
            parent_strategy_node_ids=_string_list(payload.get("parent_strategy_node_ids", [])),
            cycle_id=str(payload.get("cycle_id", "")).strip(),
            subsystem=str(payload.get("subsystem", "")).strip(),
            selected_variant_id=str(payload.get("selected_variant_id", "")).strip(),
            strategy_id=str(payload.get("strategy_id", payload.get("strategy_candidate_id", ""))).strip(),
            strategy_candidate_id=str(payload.get("strategy_candidate_id", payload.get("strategy_id", ""))).strip(),
            strategy_candidate_kind=str(payload.get("strategy_candidate_kind", "")).strip(),
            motivation=str(payload.get("motivation", "")).strip(),
            controls=_object_map(payload.get("controls", {})),
            actor_summary=_object_map(payload.get("actor_summary", {})),
            results_summary=_object_map(payload.get("results_summary", {})),
            retention_state=str(payload.get("retention_state", "pending")).strip() or "pending",
            retained_gain=float(payload.get("retained_gain", 0.0) or 0.0),
            analysis_lesson=str(payload.get("analysis_lesson", "")).strip(),
            reuse_conditions=_string_list(payload.get("reuse_conditions", [])),
            avoid_conditions=_string_list(payload.get("avoid_conditions", [])),
            continuation_parent_node_id=str(payload.get("continuation_parent_node_id", "")).strip(),
            continuation_workspace_ref=str(payload.get("continuation_workspace_ref", "")).strip(),
            continuation_artifact_path=str(payload.get("continuation_artifact_path", "")).strip(),
            continuation_branch=str(payload.get("continuation_branch", "")).strip(),
            semantic_hypotheses=_string_list(payload.get("semantic_hypotheses", [])),
            stagnation_count=max(0, int(payload.get("stagnation_count", 0) or 0)),
            descendant_node_ids=_string_list(payload.get("descendant_node_ids", [])),
            transfer_artifact_ids=_string_list(payload.get("transfer_artifact_ids", [])),
            score=float(payload.get("score", 0.0) or 0.0),
            visit_count=max(0, int(payload.get("visit_count", 0) or 0)),
            family_coverage=_object_map(payload.get("family_coverage", {})),
            artifact_paths=_string_map(payload.get("artifact_paths", {})),
        )
