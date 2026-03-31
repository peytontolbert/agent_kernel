from __future__ import annotations

MODELING_COUNTERFACTUAL_GROUPS: tuple[str, ...] = (
    "retrieval",
    "state",
    "policy",
    "transition",
    "risk",
)


def parse_modeling_counterfactual_groups(raw: str) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for item in (piece.strip().lower() for piece in raw.split(",")):
        if not item or item in seen:
            continue
        if item not in MODELING_COUNTERFACTUAL_GROUPS:
            continue
        groups.append(item)
        seen.add(item)
    return groups
