from __future__ import annotations

from copy import deepcopy
from typing import Any

from .improvement_catalog import catalog_object


def normalized_generation_focus(focus: str | None, *, default: str = "balanced") -> str:
    normalized = str(focus or default).strip()
    return normalized or default


def artifact_payload_in_lifecycle_states(
    payload: object,
    *,
    artifact_kind: str | None = None,
    allowed_states: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if artifact_kind and str(payload.get("artifact_kind", "")).strip() != artifact_kind:
        return None
    retention_decision = payload.get("retention_decision", {})
    if isinstance(retention_decision, dict) and str(retention_decision.get("state", "")).strip() == "reject":
        return None
    allowed = {str(state).strip() for state in allowed_states if str(state).strip()}
    lifecycle_state = str(payload.get("lifecycle_state", "")).strip()
    if not lifecycle_state or lifecycle_state not in allowed:
        return None
    return payload


def retained_artifact_payload(payload: object, *, artifact_kind: str) -> dict[str, Any] | None:
    return artifact_payload_in_lifecycle_states(
        payload,
        artifact_kind=artifact_kind,
        allowed_states={"retained"},
    )


def retained_mapping_section(
    payload: object,
    *,
    artifact_kind: str,
    section: str,
) -> dict[str, Any]:
    retained = retained_artifact_payload(payload, artifact_kind=artifact_kind)
    if retained is None:
        return {}
    value = retained.get(section, {})
    return deepcopy(value) if isinstance(value, dict) else {}


def retained_sequence_section(
    payload: object,
    *,
    artifact_kind: str,
    section: str,
) -> list[Any]:
    retained = retained_artifact_payload(payload, artifact_kind=artifact_kind)
    if retained is None:
        return []
    value = retained.get(section, [])
    return deepcopy(value) if isinstance(value, list) else []


def retention_gate_preset(preset: str, **overrides: Any) -> dict[str, Any]:
    presets = catalog_object("improvement", "retention_gate_presets")
    if not isinstance(presets, dict):
        presets = {}
    gate = deepcopy(presets.get(preset, {}))
    if not isinstance(gate, dict):
        gate = {}
    for key, value in overrides.items():
        gate[str(key)] = deepcopy(value)
    return gate


def filter_proposals_by_area(
    proposals: list[dict[str, Any]],
    *,
    allowed_areas: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if not allowed_areas:
        return deepcopy(proposals)
    allowed = {str(area).strip() for area in allowed_areas if str(area).strip()}
    return [
        deepcopy(proposal)
        for proposal in proposals
        if str(proposal.get("area", "")).strip() in allowed
    ]


def ensure_proposals(
    proposals: list[dict[str, Any]],
    *,
    fallback: dict[str, Any],
) -> list[dict[str, Any]]:
    if proposals:
        return deepcopy(proposals)
    return [deepcopy(fallback)]


def normalized_string_list(values: object, *, lowercase: bool = False) -> list[str]:
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, (list, tuple, set)):
        candidates = [str(value) for value in values]
    else:
        candidates = []
    normalized: set[str] = set()
    for value in candidates:
        item = str(value).strip()
        if not item:
            continue
        normalized.add(item.lower() if lowercase else item)
    return sorted(normalized)


def merged_string_lists(left: object, right: object, *, lowercase: bool = False) -> list[str]:
    return sorted(set(normalized_string_list(left, lowercase=lowercase)) | set(normalized_string_list(right, lowercase=lowercase)))


def normalized_control_mapping(
    controls: dict[str, Any] | None,
    *,
    bool_fields: tuple[str, ...] = (),
    int_fields: tuple[str, ...] = (),
    nonnegative_int_fields: tuple[str, ...] = (),
    positive_int_fields: tuple[str, ...] = (),
    list_fields: tuple[str, ...] = (),
    lowercase_list_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(controls, dict):
        return {}
    normalized = deepcopy(controls)
    for key in bool_fields:
        if key in normalized:
            normalized[key] = bool(normalized[key])
    for key in int_fields:
        if key not in normalized:
            continue
        try:
            normalized[key] = int(normalized[key])
        except (TypeError, ValueError):
            normalized.pop(key, None)
    for key in nonnegative_int_fields:
        if key not in normalized:
            continue
        try:
            normalized[key] = max(0, int(normalized[key]))
        except (TypeError, ValueError):
            normalized.pop(key, None)
    for key in positive_int_fields:
        if key not in normalized:
            continue
        try:
            normalized[key] = max(1, int(normalized[key]))
        except (TypeError, ValueError):
            normalized.pop(key, None)
    for key in list_fields:
        if key in normalized:
            normalized[key] = normalized_string_list(normalized[key])
    for key in lowercase_list_fields:
        if key in normalized:
            normalized[key] = normalized_string_list(normalized[key], lowercase=True)
    return normalized


def overlay_control_mapping(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
    *,
    bool_fields: tuple[str, ...] = (),
    int_fields: tuple[str, ...] = (),
    nonnegative_int_fields: tuple[str, ...] = (),
    positive_int_fields: tuple[str, ...] = (),
    list_fields: tuple[str, ...] = (),
    lowercase_list_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    merged = deepcopy(base)
    normalized_overrides = normalized_control_mapping(
        overrides,
        bool_fields=bool_fields,
        int_fields=int_fields,
        nonnegative_int_fields=nonnegative_int_fields,
        positive_int_fields=positive_int_fields,
        list_fields=list_fields,
        lowercase_list_fields=lowercase_list_fields,
    )
    for key, value in normalized_overrides.items():
        if key in merged:
            merged[key] = deepcopy(value)
    return merged


def build_standard_proposal_artifact(
    *,
    artifact_kind: str,
    generation_focus: str,
    retention_gate: dict[str, Any],
    control_schema: str | None = None,
    controls: dict[str, Any] | None = None,
    proposals: list[dict[str, Any]] | None = None,
    extra_sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "spec_version": "asi_v1",
        "artifact_kind": artifact_kind,
        "lifecycle_state": "proposed",
        "generation_focus": generation_focus,
        "retention_gate": deepcopy(retention_gate),
    }
    if control_schema:
        artifact["control_schema"] = control_schema
    if controls is not None:
        artifact["controls"] = deepcopy(controls)
    artifact["proposals"] = deepcopy(proposals) if isinstance(proposals, list) else []
    if isinstance(extra_sections, dict):
        for key, value in extra_sections.items():
            artifact[key] = deepcopy(value)
    return artifact
