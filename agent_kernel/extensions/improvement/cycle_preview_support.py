from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from evals.harness import scoped_eval_config

from ...config import KernelConfig
from ...extensions.strategy.subsystems import (
    base_subsystem_for,
    comparison_config_for_subsystem_artifact,
)
from .artifacts import retention_gate_for_payload

_A4_REQUIRED_FAMILY_SET = {
    "integration",
    "project",
    "repository",
    "repo_chore",
    "repo_sandbox",
}


def _priority_family_set(flags: dict[str, object] | None) -> set[str]:
    if not isinstance(flags, dict):
        return set()
    return {
        str(value).strip()
        for value in list(flags.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    }


def _is_direct_a4_transition_model_route(
    subsystem: str,
    *,
    comparison_task_limit: int | None,
    baseline_flags: dict[str, object] | None = None,
    candidate_flags: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> bool:
    if base_subsystem_for(subsystem, capability_modules_path) != "transition_model":
        return False
    if not isinstance(comparison_task_limit, int) or comparison_task_limit <= 0 or comparison_task_limit > 24:
        return False
    active_flags = candidate_flags if isinstance(candidate_flags, dict) else baseline_flags
    if not isinstance(active_flags, dict):
        return False
    if not bool(active_flags.get("restrict_to_priority_benchmark_families", False)):
        return False
    return _priority_family_set(active_flags) == _A4_REQUIRED_FAMILY_SET


def retention_eval_config(
    *,
    base_config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
    scope: str,
) -> KernelConfig:
    scoped = scoped_eval_config(
        base_config,
        scope,
        trajectories_root=base_config.trajectories_root,
        persist_episode_memory=False,
        persist_learning_candidates=False,
        storage_write_learning_exports=False,
    )
    preview_runtime_root = scoped.improvement_reports_dir / "preview_runtime"
    scoped = replace(
        scoped,
        run_checkpoints_dir=preview_runtime_root / "checkpoints",
        unattended_workspace_snapshot_root=preview_runtime_root / "workspace_snapshots",
    )
    scoped.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scoped.unattended_workspace_snapshot_root.mkdir(parents=True, exist_ok=True)
    return comparison_config_for_subsystem_artifact(scoped, subsystem, artifact_path)


def comparison_task_limit_for_retention(
    subsystem: str,
    *,
    task_limit: int | None,
    payload: dict[str, object] | None = None,
    flags: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> int | None:
    preview_comparison_task_limit_cap = 16
    retrieval_preview_comparison_task_limit_cap = 8
    retrieval_bounded_compare_fallback = 8
    if not isinstance(task_limit, int) or task_limit <= 0:
        task_limit = None
    if _is_direct_a4_transition_model_route(
        subsystem,
        comparison_task_limit=task_limit,
        baseline_flags=flags,
        candidate_flags=flags,
        capability_modules_path=capability_modules_path,
    ):
        if task_limit is None:
            return 8
        return min(task_limit, 8)
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval" or not isinstance(payload, dict):
        if task_limit is None:
            return None
        return min(task_limit, preview_comparison_task_limit_cap)
    preview_controls = payload.get("preview_controls", {})
    if isinstance(preview_controls, dict):
        comparison_task_limit_floor = preview_controls.get("comparison_task_limit_floor")
        try:
            comparison_task_limit_floor = int(comparison_task_limit_floor)
        except (TypeError, ValueError):
            comparison_task_limit_floor = 0
        if comparison_task_limit_floor > 0:
            bounded_floor = max(retrieval_preview_comparison_task_limit_cap, comparison_task_limit_floor)
            if task_limit is None:
                return bounded_floor
            return max(task_limit, bounded_floor)
    gate = retention_gate_for_payload(
        subsystem,
        payload,
        capability_modules_path=capability_modules_path,
    )
    if bool(gate.get("require_trusted_carryover_repair_improvement", False)):
        bounded_limit = max(retrieval_preview_comparison_task_limit_cap, retrieval_bounded_compare_fallback)
        if task_limit is None:
            return bounded_limit
        return max(task_limit, bounded_limit)
    if task_limit is None:
        return retrieval_preview_comparison_task_limit_cap
    return min(task_limit, retrieval_preview_comparison_task_limit_cap)


def holdout_task_limit_for_retention(
    subsystem: str,
    *,
    comparison_task_limit: int | None,
    baseline_flags: dict[str, object] | None = None,
    candidate_flags: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> int | None:
    if not isinstance(comparison_task_limit, int) or comparison_task_limit <= 0:
        return None
    if _is_direct_a4_transition_model_route(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        baseline_flags=baseline_flags,
        candidate_flags=candidate_flags,
        capability_modules_path=capability_modules_path,
    ):
        return min(comparison_task_limit, 8)
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval":
        return min(comparison_task_limit, 16)
    return max(4, comparison_task_limit)


def holdout_generated_schedule_limit_for_retention(
    subsystem: str,
    *,
    comparison_task_limit: int | None,
    baseline_flags: dict[str, object] | None = None,
    candidate_flags: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> int:
    holdout_task_limit = holdout_task_limit_for_retention(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        baseline_flags=baseline_flags,
        candidate_flags=candidate_flags,
        capability_modules_path=capability_modules_path,
    )
    if not isinstance(holdout_task_limit, int) or holdout_task_limit <= 0:
        return 0
    return holdout_task_limit


def retrieval_preview_priority_overrides(
    payload: dict[str, object] | None,
) -> tuple[list[str], dict[str, float]]:
    if not isinstance(payload, dict):
        return [], {}
    preview_controls = payload.get("preview_controls", {})
    if not isinstance(preview_controls, dict):
        return [], {}
    families = [
        str(family).strip()
        for family in list(preview_controls.get("priority_benchmark_families", []))
        if str(family).strip()
    ]
    weights_value = preview_controls.get("priority_benchmark_family_weights", {})
    weights: dict[str, float] = {}
    if isinstance(weights_value, dict):
        for family, weight in weights_value.items():
            normalized_family = str(family).strip()
            if not normalized_family:
                continue
            try:
                weights[normalized_family] = float(weight)
            except (TypeError, ValueError):
                continue
    return families, weights


def retrieval_bounded_preview_required(
    subsystem: str,
    *,
    payload: dict[str, object] | None,
    capability_modules_path: Path | None = None,
) -> bool:
    if base_subsystem_for(subsystem, capability_modules_path) != "retrieval" or not isinstance(payload, dict):
        return False
    preview_controls = payload.get("preview_controls", {})
    if not isinstance(preview_controls, dict):
        return False
    return bool(preview_controls.get("bounded_comparison_required", False))


def apply_retrieval_bounded_preview_filters(
    subsystem: str,
    *,
    flags: dict[str, object],
    payload: dict[str, object] | None,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    scoped_flags = dict(flags)
    if not retrieval_bounded_preview_required(
        subsystem,
        payload=payload,
        capability_modules_path=capability_modules_path,
    ):
        return scoped_flags
    scoped_flags["include_episode_memory"] = False
    scoped_flags["include_verifier_memory"] = False
    preview_controls = payload.get("preview_controls", {}) if isinstance(payload, dict) else {}
    if isinstance(preview_controls, dict) and bool(preview_controls.get("prefer_long_horizon_tasks", False)):
        scoped_flags["prefer_long_horizon_tasks"] = True
    return scoped_flags


def merge_priority_families(
    explicit_families: list[str] | None,
    explicit_weights: dict[str, float] | None,
    payload: dict[str, object] | None,
    *,
    allow_payload_overrides: bool = True,
) -> tuple[list[str], dict[str, float]]:
    merged_families: list[str] = []
    merged_weights: dict[str, float] = {}
    for family in list(explicit_families or []):
        normalized = str(family).strip()
        if not normalized:
            continue
        if normalized not in merged_families:
            merged_families.append(normalized)
    for family, weight in dict(explicit_weights or {}).items():
        normalized = str(family).strip()
        if not normalized:
            continue
        try:
            merged_weights[normalized] = float(weight)
        except (TypeError, ValueError):
            continue
        if normalized not in merged_families:
            merged_families.append(normalized)
    if allow_payload_overrides:
        retrieval_families, retrieval_weights = retrieval_preview_priority_overrides(payload)
        for family in retrieval_families:
            if family not in merged_families:
                merged_families.append(family)
        for family, weight in retrieval_weights.items():
            merged_weights[family] = max(weight, float(merged_weights.get(family, 0.0) or 0.0))
            if family not in merged_families:
                merged_families.append(family)
    return merged_families, merged_weights


def apply_priority_family_restriction(
    flags: dict[str, object],
    *,
    restrict_to_priority_families: bool,
    merged_priority_families: list[str],
    preserve_generated_lanes: bool,
) -> None:
    if not (restrict_to_priority_families and merged_priority_families):
        return
    flags["restrict_to_priority_benchmark_families"] = True
    flags["prefer_low_cost_tasks"] = True
    if not preserve_generated_lanes:
        flags["include_generated"] = False
        flags["include_failure_generated"] = False


def apply_direct_a4_transition_model_compare_preferences(
    subsystem: str,
    *,
    flags: dict[str, object],
    comparison_task_limit: int | None,
    capability_modules_path: Path | None = None,
) -> None:
    if not _is_direct_a4_transition_model_route(
        subsystem,
        comparison_task_limit=comparison_task_limit,
        baseline_flags=flags,
        candidate_flags=flags,
        capability_modules_path=capability_modules_path,
    ):
        return
    # The narrow A4 transition_model lane needs discriminative replay/recovery pressure,
    # not the cheapest light-supervision tasks that saturate confirmation at 1.0/1.0.
    flags["prefer_low_cost_tasks"] = False
    flags["prefer_retrieval_tasks"] = True
    flags["prefer_long_horizon_tasks"] = True
    flags["include_discovered_tasks"] = True
    priority_families = [
        str(value).strip()
        for value in list(flags.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if "transition_pressure" not in priority_families:
        priority_families.append("transition_pressure")
    flags["priority_benchmark_families"] = priority_families
