from __future__ import annotations

from typing import Any

from ...improvement_engine import ImprovementExperiment, ImprovementVariant


def recent_strategy_activity_summary(
    planner: Any,
    *,
    strategy_candidate_id: str,
    output_path=None,
    recent_cycle_window: int = 6,
) -> dict[str, object]:
    from ... import improvement as core

    strategy_id = str(strategy_candidate_id).strip()
    if not strategy_id:
        return {"selected_cycles": 0, "retained_cycles": 0, "rejected_cycles": 0}
    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return {"selected_cycles": 0, "retained_cycles": 0, "rejected_cycles": 0}
    records = [
        record for record in planner.load_cycle_records(resolved) if core._record_strategy_candidate_id(record) == strategy_id
    ]
    recent = records[-max(1, recent_cycle_window) :]
    return {
        "selected_cycles": len(
            {
                str(record.get("cycle_id", "")).strip()
                for record in recent
                if str(record.get("state", "")).strip() in {"observe", "select", "generate", "evaluate"}
            }
        ),
        "retained_cycles": len(
            {str(record.get("cycle_id", "")).strip() for record in recent if str(record.get("state", "")).strip() == "retain"}
        ),
        "rejected_cycles": len(
            {str(record.get("cycle_id", "")).strip() for record in recent if str(record.get("state", "")).strip() == "reject"}
        ),
    }


def strategy_memory_variant_lineage_adjustment(
    strategy_candidate: dict[str, object] | None,
    variant: ImprovementVariant,
) -> tuple[float, dict[str, object]]:
    candidate = strategy_candidate if isinstance(strategy_candidate, dict) else {}
    control_surface = (
        dict(candidate.get("parent_control_surface", {}))
        if isinstance(candidate.get("parent_control_surface", {}), dict)
        else {}
    )
    preferred_controls = (
        dict(control_surface.get("preferred_controls", {}))
        if isinstance(control_surface.get("preferred_controls", {}), dict)
        else {}
    )
    rejected_controls = [
        dict(item)
        for item in list(control_surface.get("rejected_controls", []) or [])
        if isinstance(item, dict)
    ]
    variant_controls = dict(variant.controls)
    preferred_exact_matches = sum(
        1 for key, value in preferred_controls.items() if key in variant_controls and variant_controls.get(key) == value
    )
    preferred_key_matches = sum(1 for key in preferred_controls if key in variant_controls)
    rejected_exact_matches = 0
    for rejected in rejected_controls:
        matches = sum(1 for key, value in rejected.items() if key in variant_controls and variant_controls.get(key) == value)
        rejected_exact_matches = max(rejected_exact_matches, matches)
    score_adjustment = min(0.05, preferred_exact_matches * 0.015 + max(0, preferred_key_matches - preferred_exact_matches) * 0.005)
    score_adjustment -= min(0.05, rejected_exact_matches * 0.015)
    evidence = {
        "preferred_control_matches": preferred_exact_matches,
        "preferred_control_key_overlap": preferred_key_matches,
        "rejected_control_matches": rejected_exact_matches,
        "score_adjustment": round(score_adjustment, 4),
        "prefer_family_breadth": bool(control_surface.get("prefer_family_breadth", False)),
        "prefer_unattended_closeout": bool(control_surface.get("prefer_unattended_closeout", False)),
    }
    return round(score_adjustment, 4), evidence


def subsystem_history_summary(
    planner: Any,
    *,
    subsystem: str,
    output_path=None,
) -> dict[str, object]:
    decisions = [
        record
        for record in planner._decision_records(output_path)
        if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
    ]
    return planner._decision_summary(decisions)


def variant_history_summary(
    planner: Any,
    *,
    subsystem: str,
    variant_id: str,
    output_path=None,
) -> dict[str, object]:
    cycle_variants = planner._cycle_variant_index(output_path)
    decisions = [
        record
        for record in planner._decision_records(output_path)
        if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
        and cycle_variants.get(str(record.get("cycle_id", ""))) == variant_id
    ]
    return planner._decision_summary(decisions)


def apply_improvement_planner_mutation(
    planner: Any,
    candidate: ImprovementExperiment,
    *,
    planner_controls: dict[str, object],
) -> tuple[ImprovementExperiment, dict[str, object]]:
    expected_gain_multiplier = planner._planner_control_subsystem_float(
        planner_controls,
        "subsystem_expected_gain_multiplier",
        candidate.subsystem,
        fallback_subsystem=planner._base_subsystem(candidate.subsystem),
        default=1.0,
        min_value=0.25,
        max_value=4.0,
    )
    cost_multiplier = planner._planner_control_subsystem_float(
        planner_controls,
        "subsystem_cost_multiplier",
        candidate.subsystem,
        fallback_subsystem=planner._base_subsystem(candidate.subsystem),
        default=1.0,
        min_value=0.5,
        max_value=4.0,
    )
    if expected_gain_multiplier == 1.0 and cost_multiplier == 1.0:
        return candidate, {}
    expected_gain = round(max(0.0, candidate.expected_gain * expected_gain_multiplier), 4)
    estimated_cost = max(1, int(round(candidate.estimated_cost * cost_multiplier)))
    return (
        ImprovementExperiment(
            subsystem=candidate.subsystem,
            reason=candidate.reason,
            priority=candidate.priority,
            expected_gain=expected_gain,
            estimated_cost=estimated_cost,
            score=candidate.score,
            evidence=dict(candidate.evidence),
        ),
        {
            "expected_gain_multiplier": expected_gain_multiplier,
            "cost_multiplier": cost_multiplier,
        },
    )


def apply_variant_planner_mutation(
    planner: Any,
    variant: ImprovementVariant,
    *,
    planner_controls: dict[str, object],
) -> tuple[ImprovementVariant, dict[str, object]]:
    expected_gain_multiplier = planner._planner_control_variant_float(
        planner_controls,
        "variant_expected_gain_multiplier",
        variant.subsystem,
        variant.variant_id,
        fallback_subsystem=planner._base_subsystem(variant.subsystem),
        default=1.0,
        min_value=0.25,
        max_value=4.0,
    )
    cost_multiplier = planner._planner_control_variant_float(
        planner_controls,
        "variant_cost_multiplier",
        variant.subsystem,
        variant.variant_id,
        fallback_subsystem=planner._base_subsystem(variant.subsystem),
        default=1.0,
        min_value=0.5,
        max_value=4.0,
    )
    if expected_gain_multiplier == 1.0 and cost_multiplier == 1.0:
        return variant, {}
    expected_gain = round(max(0.0, variant.expected_gain * expected_gain_multiplier), 4)
    estimated_cost = max(1, int(round(variant.estimated_cost * cost_multiplier)))
    return (
        ImprovementVariant(
            subsystem=variant.subsystem,
            variant_id=variant.variant_id,
            description=variant.description,
            expected_gain=expected_gain,
            estimated_cost=estimated_cost,
            score=variant.score,
            controls=dict(variant.controls),
        ),
        {
            "expected_gain_multiplier": expected_gain_multiplier,
            "cost_multiplier": cost_multiplier,
        },
    )


def with_variant_expansions(
    planner: Any,
    variants: list[ImprovementVariant],
    *,
    planner_controls: dict[str, object],
) -> list[ImprovementVariant]:
    combined = list(variants)
    seen_variant_ids = {variant.variant_id for variant in variants}
    expansions = planner_controls.get("variant_expansions", {})
    if not isinstance(expansions, dict) or not variants:
        return combined
    subsystem = variants[0].subsystem
    effective_subsystem = planner._base_subsystem(subsystem)
    for key in dict.fromkeys([subsystem, effective_subsystem]):
        entries = expansions.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            variant_id = str(entry.get("variant_id", "")).strip()
            description = str(entry.get("description", "")).strip()
            if not variant_id or not description or variant_id in seen_variant_ids:
                continue
            try:
                expected_gain = max(0.0, float(entry.get("expected_gain", 0.0)))
                estimated_cost = max(1, int(entry.get("estimated_cost", 1)))
            except (TypeError, ValueError):
                continue
            controls = entry.get("controls", {})
            combined.append(
                planner._variant(
                    subsystem,
                    variant_id,
                    description,
                    expected_gain,
                    estimated_cost,
                    dict(controls) if isinstance(controls, dict) else {},
                )
            )
            seen_variant_ids.add(variant_id)
    return combined


def planner_control_float(
    planner_controls: dict[str, object],
    field: str,
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    value = planner_controls.get(field, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def planner_control_variant_float(
    planner_controls: dict[str, object],
    field: str,
    subsystem: str,
    variant_id: str,
    *,
    fallback_subsystem: str | None = None,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    mapping = planner_controls.get(field, {})
    if not isinstance(mapping, dict):
        return default
    subsystem_mapping = mapping.get(subsystem, {})
    if not isinstance(subsystem_mapping, dict) and fallback_subsystem is not None:
        subsystem_mapping = mapping.get(fallback_subsystem, {})
    elif (
        isinstance(subsystem_mapping, dict)
        and variant_id not in subsystem_mapping
        and fallback_subsystem is not None
        and fallback_subsystem != subsystem
    ):
        fallback_mapping = mapping.get(fallback_subsystem, {})
        if isinstance(fallback_mapping, dict):
            subsystem_mapping = fallback_mapping
    if not isinstance(subsystem_mapping, dict):
        return default
    value = subsystem_mapping.get(variant_id, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def planner_guardrail_float(
    planner_controls: dict[str, object],
    *,
    scope: str,
    field: str,
    legacy_field: str,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    search_guardrails = planner_controls.get("search_guardrails", {})
    if isinstance(search_guardrails, dict):
        scope_mapping = search_guardrails.get(scope, {})
        if isinstance(scope_mapping, dict) and field in scope_mapping:
            try:
                parsed = float(scope_mapping.get(field, default))
            except (TypeError, ValueError):
                return default
            return max(min_value, min(max_value, parsed))
    return planner_control_float(
        planner_controls,
        legacy_field,
        default,
        min_value=min_value,
        max_value=max_value,
    )


def planner_control_subsystem_float(
    planner_controls: dict[str, object],
    field: str,
    subsystem: str,
    *,
    fallback_subsystem: str | None = None,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    mapping = planner_controls.get(field, {})
    if not isinstance(mapping, dict):
        return default
    value = mapping.get(subsystem, default)
    if value == default and fallback_subsystem is not None:
        value = mapping.get(fallback_subsystem, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))
