from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from .resource_registry import ResourceRegistry
from .resource_types import RESOURCE_KIND_ARTIFACT, subsystem_resource_id


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_string_list(values: list[object] | tuple[object, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ResourceEditPlan:
    resource_id: str
    resource_kind: str
    target_subsystem: str
    mutation_kind: str
    selected_variant_id: str
    selected_variant_description: str
    active_version_id: str = ""
    active_resource_path: str = ""
    expected_artifact_kind: str = ""
    controls: dict[str, object] = field(default_factory=dict)
    generation_kwargs: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "target_subsystem": self.target_subsystem,
            "mutation_kind": self.mutation_kind,
            "selected_variant_id": self.selected_variant_id,
            "selected_variant_description": self.selected_variant_description,
            "active_version_id": self.active_version_id,
            "active_resource_path": self.active_resource_path,
            "expected_artifact_kind": self.expected_artifact_kind,
            "controls": dict(self.controls),
            "generation_kwargs": dict(self.generation_kwargs),
        }


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    selection_id: str
    cycle_id: str
    subsystem: str
    resource_id: str
    resource_kind: str
    rationale: str
    selected_variant_id: str
    selected_variant_description: str
    selected_variant_expected_gain: float
    selected_variant_estimated_cost: int
    selected_variant_score: float
    strategy_candidate_id: str = ""
    strategy_candidate_kind: str = ""
    strategy_origin: str = ""
    strategy_label: str = ""
    strategy_node_id: str = ""
    portfolio_reasons: tuple[str, ...] = ()
    campaign_index: int = 0
    campaign_width: int = 0
    variant_rank: int = 1
    variant_width: int = 1
    search_strategy: str = ""
    campaign_budget: dict[str, object] = field(default_factory=dict)
    variant_budget: dict[str, object] = field(default_factory=dict)
    edit_plan: ResourceEditPlan | None = None
    generated_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "cycle_id": self.cycle_id,
            "subsystem": self.subsystem,
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "rationale": self.rationale,
            "selected_variant": {
                "variant_id": self.selected_variant_id,
                "description": self.selected_variant_description,
                "expected_gain": round(float(self.selected_variant_expected_gain), 4),
                "estimated_cost": int(self.selected_variant_estimated_cost),
                "score": round(float(self.selected_variant_score), 4),
            },
            "strategy": {
                "strategy_candidate_id": self.strategy_candidate_id,
                "strategy_candidate_kind": self.strategy_candidate_kind,
                "origin": self.strategy_origin,
                "strategy_label": self.strategy_label,
                "strategy_node_id": self.strategy_node_id,
            },
            "portfolio_reasons": list(self.portfolio_reasons),
            "search": {
                "campaign_index": int(self.campaign_index),
                "campaign_width": int(self.campaign_width),
                "variant_rank": int(self.variant_rank),
                "variant_width": int(self.variant_width),
                "search_strategy": self.search_strategy,
                "campaign_budget": dict(self.campaign_budget),
                "variant_budget": dict(self.variant_budget),
            },
            "edit_plan": None if self.edit_plan is None else self.edit_plan.to_dict(),
            "generated_at": self.generated_at,
        }


def build_selection_record(
    *,
    cycle_id: str,
    target_subsystem: str,
    reason: str,
    variant_id: str,
    variant_description: str,
    variant_expected_gain: float,
    variant_estimated_cost: int,
    variant_score: float,
    variant_controls: dict[str, object] | None = None,
    generation_kwargs: dict[str, object] | None = None,
    expected_artifact_kind: str = "",
    strategy_candidate: dict[str, object] | None = None,
    campaign_index: int = 0,
    campaign_width: int = 0,
    variant_rank: int = 1,
    variant_width: int = 1,
    search_strategy: str = "",
    campaign_budget: dict[str, object] | None = None,
    variant_budget: dict[str, object] | None = None,
    resource_registry: ResourceRegistry | None = None,
    generated_at: str | None = None,
) -> SelectionRecord:
    subsystem = str(target_subsystem).strip()
    resource_id = subsystem_resource_id(subsystem)
    strategy = dict(strategy_candidate or {})
    active_version = None
    if resource_registry is not None and resource_registry.has(resource_id):
        active_version = resource_registry.resolve_active_version(resource_id)
    rationale = str(strategy.get("rationale", "")).strip() or str(reason).strip()
    portfolio_reasons = _normalized_string_list(list(strategy.get("portfolio_reasons", []) or []))
    edit_plan = ResourceEditPlan(
        resource_id=resource_id,
        resource_kind=RESOURCE_KIND_ARTIFACT,
        target_subsystem=subsystem,
        mutation_kind="generate_candidate_artifact",
        selected_variant_id=str(variant_id).strip(),
        selected_variant_description=str(variant_description).strip(),
        active_version_id="" if active_version is None else str(active_version.version_id).strip(),
        active_resource_path="" if active_version is None else str(active_version.path),
        expected_artifact_kind=str(expected_artifact_kind).strip(),
        controls=dict(variant_controls or {}),
        generation_kwargs=dict(generation_kwargs or {}),
    )
    return SelectionRecord(
        selection_id=f"select:{cycle_id}",
        cycle_id=cycle_id,
        subsystem=subsystem,
        resource_id=resource_id,
        resource_kind=RESOURCE_KIND_ARTIFACT,
        rationale=rationale,
        selected_variant_id=str(variant_id).strip(),
        selected_variant_description=str(variant_description).strip(),
        selected_variant_expected_gain=float(variant_expected_gain),
        selected_variant_estimated_cost=max(0, int(variant_estimated_cost)),
        selected_variant_score=float(variant_score),
        strategy_candidate_id=str(strategy.get("strategy_candidate_id", strategy.get("strategy_id", ""))).strip(),
        strategy_candidate_kind=str(
            strategy.get("strategy_candidate_kind", strategy.get("strategy_kind", ""))
        ).strip(),
        strategy_origin=str(strategy.get("origin", strategy.get("strategy_origin", ""))).strip(),
        strategy_label=str(strategy.get("strategy_label", "")).strip(),
        strategy_node_id=str(strategy.get("strategy_node_id", "")).strip(),
        portfolio_reasons=portfolio_reasons,
        campaign_index=max(0, int(campaign_index)),
        campaign_width=max(0, int(campaign_width)),
        variant_rank=max(0, int(variant_rank)),
        variant_width=max(0, int(variant_width)),
        search_strategy=str(search_strategy).strip(),
        campaign_budget=dict(campaign_budget or {}),
        variant_budget=dict(variant_budget or {}),
        edit_plan=edit_plan,
        generated_at=str(generated_at).strip() if generated_at is not None else _utcnow_iso(),
    )


__all__ = [
    "ResourceEditPlan",
    "SelectionRecord",
    "build_selection_record",
]
