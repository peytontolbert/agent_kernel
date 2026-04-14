from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..extensions.improvement.improvement_catalog import catalog_mapping, catalog_string_set

_FRONTIER_FAMILY_STEP_FLOORS: dict[str, int] = {
    str(key): int(value)
    for key, value in catalog_mapping("task_budget", "frontier_family_step_floors").items()
}

_DIFFICULTY_STEP_FLOORS: dict[str, int] = {
    str(key): int(value)
    for key, value in catalog_mapping("task_budget", "difficulty_step_floors").items()
}

_DERIVED_FAMILIES = catalog_string_set("task_budget", "derived_families")


def uplifted_task_max_steps(
    current_max_steps: int,
    *,
    metadata: Mapping[str, object] | None = None,
    suggested_commands: Sequence[str] | None = None,
) -> int:
    normalized_current = max(1, int(current_max_steps or 1))
    if normalized_current > 5:
        return normalized_current

    metadata = dict(metadata or {})
    benchmark_family = _effective_benchmark_family(metadata)
    difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip().lower()
    horizon = str(metadata.get("horizon", "")).strip().lower()

    budget = normalized_current
    budget = max(budget, _coerce_non_negative_int(metadata.get("budget_step_floor")))
    budget = max(budget, _FRONTIER_FAMILY_STEP_FLOORS.get(benchmark_family, normalized_current))
    budget = max(budget, _DIFFICULTY_STEP_FLOORS.get(difficulty, normalized_current))
    if horizon == "long_horizon":
        budget = max(budget, _DIFFICULTY_STEP_FLOORS["long_horizon"])

    long_horizon_step_count = _coerce_non_negative_int(metadata.get("long_horizon_step_count"))
    if long_horizon_step_count > 0:
        budget = max(budget, long_horizon_step_count + 3)

    synthetic_edit_plan = metadata.get("synthetic_edit_plan", [])
    if isinstance(synthetic_edit_plan, list):
        budget = max(budget, len(synthetic_edit_plan) + 3)

    command_count = len([command for command in (suggested_commands or []) if str(command).strip()])
    if command_count > 0:
        budget = max(budget, command_count + 2)

    return budget


def _effective_benchmark_family(metadata: Mapping[str, object]) -> str:
    benchmark_family = str(metadata.get("benchmark_family", "bounded")).strip().lower() or "bounded"
    if benchmark_family in _DERIVED_FAMILIES:
        origin = str(metadata.get("origin_benchmark_family", "")).strip().lower()
        if origin:
            return origin
    return benchmark_family


def _coerce_non_negative_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
