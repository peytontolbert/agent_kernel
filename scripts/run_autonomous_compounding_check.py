from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.improvement.curriculum_improvement import retained_curriculum_controls
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner
from agent_kernel.extensions.improvement.improvement_common import build_standard_proposal_artifact, retention_gate_preset
from agent_kernel.extensions.improvement.prompt_improvement import resolve_improvement_planner_controls, retained_improvement_planner_controls

DEFAULT_NON_REPLAY_TRANSFER_FAMILIES = ("workflow", "project", "repository", "tooling", "integration")
MIN_ACCEPTABLE_TRANSFER_RETURN_ON_COST = 0.01
MATERIAL_ALLOCATION_SHARE_GAP = 0.05
ALLOCATION_NORMALIZATION_DAMPING = 0.5
ALLOCATION_COMPENSATION_STREAK_STEP = 0.5
ALLOCATION_COMPENSATION_MAX_MULTIPLIER = 2.0
ALLOCATION_NORMALIZATION_STREAK_STEP = 0.5
ALLOCATION_NORMALIZATION_MAX_MULTIPLIER = 2.0
ALLOCATION_CONFIDENCE_MIN_RUNS = 3
ALLOCATION_CONFIDENCE_TARGET_PRIORITY_TASKS = 12
ALLOCATION_CONFIDENCE_HISTORY_WINDOW = 3
ALLOCATION_CONFIDENCE_HISTORY_WEIGHT = 0.5
AUTONOMOUS_FRONTIER_TASK_LIMIT_FLOOR = 2
AUTONOMOUS_FRONTIER_TASK_LIMIT_MAX = 4
AUTONOMOUS_FRONTIER_MISSING_FAMILY_WEIGHT_BONUS = 3.0
AUTONOMOUS_FRONTIER_UNSAMPLED_PRESSURE_WEIGHT_BONUS = 2.5
AUTONOMOUS_FRONTIER_UNDER_SAMPLED_WEIGHT_BONUS = 1.5
AUTONOMOUS_FRONTIER_UNSAMPLED_TARGET_WEIGHT_BONUS = 1.0
AUTONOMOUS_FRONTIER_GENERALIZATION_WEIGHT_BONUS = 1.0
AUTONOMOUS_SEED_FINGERPRINT_MAX_FILES = 256
AUTONOMOUS_SEED_FINGERPRINT_MAX_DIRECTORIES = 256
AUTONOMOUS_SEED_FINGERPRINT_MAX_BYTES = 64 * 1024 * 1024
AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_FILES = 256
AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_DIRECTORIES = 256
AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_BYTES = 64 * 1024 * 1024


def _status_path(config: KernelConfig) -> Path:
    return config.improvement_reports_dir / "autonomous_compounding_status.json"


def _ordered_unique_strings(*groups: object) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for values in groups:
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, (list, tuple, set)):
            candidates = [str(value) for value in values]
        else:
            candidates = []
        for value in candidates:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _load_json_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_list(payload: dict[str, object], key: str) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _effective_active_run_snapshot(
    *,
    config: KernelConfig,
    active_run: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    prior_payload = _load_json_payload(_status_path(config))
    prior_active_run = prior_payload.get("active_run", {})
    if not isinstance(prior_active_run, dict):
        prior_active_run = {}
    effective_active_run = dict(prior_active_run)
    if isinstance(active_run, dict):
        effective_active_run.update(active_run)

    child_status_payload = effective_active_run.get("child_status", {})
    if not isinstance(child_status_payload, dict):
        child_status_payload = {}
    if not child_status_payload:
        inherited_child_status = prior_active_run.get("child_status", {})
        if isinstance(inherited_child_status, dict):
            child_status_payload = dict(inherited_child_status)

    child_status_path_value = str(effective_active_run.get("child_status_path", "")).strip()
    if not child_status_path_value:
        child_status_path_value = str(prior_active_run.get("child_status_path", "")).strip()
    if child_status_path_value:
        effective_active_run["child_status_path"] = child_status_path_value
    if not child_status_payload and child_status_path_value:
        child_status_payload = _load_json_payload(Path(child_status_path_value))
    if child_status_payload:
        effective_active_run["child_status"] = child_status_payload

    return effective_active_run, prior_payload


def _retained_child_sampling_snapshot(
    *,
    prior_payload: dict[str, object],
    active_run: dict[str, object],
    requested_priority_benchmark_families: list[str],
) -> dict[str, list[str]]:
    child_status = active_run.get("child_status", {})
    if not isinstance(child_status, dict):
        child_status = {}
    if not child_status:
        return {
            "families_sampled": _string_list(prior_payload, "families_sampled"),
            "families_never_sampled": _string_list(prior_payload, "families_never_sampled"),
            "pressure_families_without_sampling": _string_list(prior_payload, "pressure_families_without_sampling"),
        }

    sampled = _ordered_unique_strings(
        _string_list(prior_payload, "families_sampled"),
        _string_list(child_status, "families_sampled"),
    )
    priority_families = _ordered_unique_strings(
        requested_priority_benchmark_families,
        _string_list(active_run, "priority_benchmark_families"),
        _string_list(child_status, "priority_benchmark_families"),
    )
    if priority_families:
        sampled_set = set(sampled)
        unsampled = [family for family in priority_families if family not in sampled_set]
    else:
        unsampled = _ordered_unique_strings(
            _string_list(prior_payload, "pressure_families_without_sampling"),
            _string_list(prior_payload, "families_never_sampled"),
            _string_list(child_status, "priority_families_without_sampling"),
            _string_list(child_status, "pressure_families_without_sampling"),
        )
    return {
        "families_sampled": sampled,
        "families_never_sampled": unsampled,
        "pressure_families_without_sampling": unsampled,
    }


def _match_id(index: int) -> str:
    return f"autonomous:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}:{index}"


def _run_root(config: KernelConfig, *, run_match_id: str) -> Path:
    safe_match_id = run_match_id.replace(":", "_")
    return config.improvement_reports_dir / ".autonomous_compounding" / safe_match_id


def _child_run_status_path(run_root: Path) -> Path:
    return run_root / "trajectories" / "improvement" / "reports" / "repeated_improvement_status.json"


def _runtime_feature_env(config: KernelConfig) -> dict[str, str]:
    return config.to_env()


def _resolved_improvement_planner_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(getattr(config, "use_prompt_proposals", True)):
        return resolve_improvement_planner_controls()
    prompt_path = getattr(config, "prompt_proposals_path", None)
    if prompt_path is None:
        return resolve_improvement_planner_controls()
    path = Path(prompt_path)
    if not path.exists():
        return resolve_improvement_planner_controls()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return resolve_improvement_planner_controls()
    return retained_improvement_planner_controls(payload)


def _priority_family_allocation_confidence_controls(config: KernelConfig) -> dict[str, float]:
    planner_controls = _resolved_improvement_planner_controls(config)
    controls = planner_controls.get("priority_family_allocation_confidence", {})
    if not isinstance(controls, dict):
        controls = {}
    return {
        "minimum_runs": max(1, int(controls.get("minimum_runs", ALLOCATION_CONFIDENCE_MIN_RUNS) or 0)),
        "target_priority_tasks": max(
            1,
            int(controls.get("target_priority_tasks", ALLOCATION_CONFIDENCE_TARGET_PRIORITY_TASKS) or 0),
        ),
        "target_family_tasks": max(0, int(controls.get("target_family_tasks", 0) or 0)),
        "history_window_runs": max(1, int(controls.get("history_window_runs", ALLOCATION_CONFIDENCE_HISTORY_WINDOW) or 0)),
        "history_weight": max(
            0.0,
            min(1.0, float(controls.get("history_weight", ALLOCATION_CONFIDENCE_HISTORY_WEIGHT) or 0.0)),
        ),
        "bonus_history_weight": max(0.0, min(1.0, float(controls.get("bonus_history_weight", 0.75) or 0.0))),
        "normalization_history_weight": max(
            0.0,
            min(1.0, float(controls.get("normalization_history_weight", 0.25) or 0.0)),
        ),
    }


def _latest_prior_compounding_claim_gate_summary(config: KernelConfig) -> dict[str, object]:
    reports_dir = Path(config.improvement_reports_dir)
    if not reports_dir.exists():
        return {}
    for path in sorted(reports_dir.glob("autonomous_compounding_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            continue
        claim_gate = summary.get("claim_gate_summary", {})
        if not isinstance(claim_gate, dict):
            continue
        ranking = claim_gate.get("family_transfer_investment_ranking", {})
        if not isinstance(ranking, dict):
            continue
        families = [str(value).strip() for value in ranking.get("ranked_families_by_transfer_investment", []) if str(value).strip()]
        if families:
            return claim_gate
    return {}


def _resolved_task_limit(config: KernelConfig, args: argparse.Namespace) -> tuple[int, str]:
    requested = max(0, int(getattr(args, "task_limit", 0) or 0))
    if requested > 0:
        return requested, "explicit_cli"
    default_limit = max(0, int(getattr(config, "compare_feature_max_tasks", 0) or 0))
    if default_limit > 0:
        return default_limit, "config_compare_feature_max_tasks"
    return 0, "unbounded"


def _default_priority_family_weights(families: list[str]) -> dict[str, float]:
    return {family: 1.0 for family in families}


def _rank_weighted_priority_family_weights(families: list[str]) -> dict[str, float]:
    count = len(families)
    return {
        family: float(max(1, count - index))
        for index, family in enumerate(families)
    }


def _priority_family_allocation_compensation(
    claim_gate_summary: dict[str, object],
    families: list[str],
    *,
    confidence_controls: dict[str, float | int],
) -> tuple[dict[str, float], dict[str, object]]:
    allocation_audit = claim_gate_summary.get("priority_family_allocation_audit", {})
    if not isinstance(allocation_audit, dict):
        allocation_audit = {}
    average_planned_shares = allocation_audit.get("average_planned_shares", {})
    average_actual_shares = allocation_audit.get("average_actual_shares", {})
    latest_planned_shares = allocation_audit.get("latest_planned_shares", {})
    latest_actual_shares = allocation_audit.get("latest_actual_shares", {})
    if not isinstance(average_planned_shares, dict):
        average_planned_shares = {}
    if not isinstance(average_actual_shares, dict):
        average_actual_shares = {}
    if not isinstance(latest_planned_shares, dict):
        latest_planned_shares = {}
    if not isinstance(latest_actual_shares, dict):
        latest_actual_shares = {}
    latest_positive_gap_streak_by_family = allocation_audit.get("latest_positive_gap_streak_by_family", {})
    if not isinstance(latest_positive_gap_streak_by_family, dict):
        latest_positive_gap_streak_by_family = {}
    latest_oversampled_streak_by_family = allocation_audit.get("latest_oversampled_streak_by_family", {})
    if not isinstance(latest_oversampled_streak_by_family, dict):
        latest_oversampled_streak_by_family = {}
    allocation_confidence_by_family = allocation_audit.get("allocation_confidence_by_family", {})
    if not isinstance(allocation_confidence_by_family, dict):
        allocation_confidence_by_family = {}
    bonus_allocation_confidence_by_family = allocation_audit.get("bonus_allocation_confidence_by_family", {})
    if not isinstance(bonus_allocation_confidence_by_family, dict):
        bonus_allocation_confidence_by_family = {}
    normalization_allocation_confidence_by_family = allocation_audit.get("normalization_allocation_confidence_by_family", {})
    if not isinstance(normalization_allocation_confidence_by_family, dict):
        normalization_allocation_confidence_by_family = {}
    latest_task_confidence_by_family = allocation_audit.get("latest_task_confidence_by_family", {})
    if not isinstance(latest_task_confidence_by_family, dict):
        latest_task_confidence_by_family = {}
    recent_average_task_confidence_by_family = allocation_audit.get("recent_average_task_confidence_by_family", {})
    if not isinstance(recent_average_task_confidence_by_family, dict):
        recent_average_task_confidence_by_family = {}
    allocation_confidence = float(allocation_audit.get("allocation_confidence", 1.0) or 0.0)
    allocation_confidence = max(0.0, min(1.0, allocation_confidence))
    gap_source = "latest_allocation_summary"
    planned_shares = latest_planned_shares
    actual_shares = latest_actual_shares
    if not planned_shares and not actual_shares:
        gap_source = "average_allocation_summary"
        planned_shares = average_planned_shares
        actual_shares = average_actual_shares
    positive_gap_families: list[str] = []
    negative_gap_families: list[str] = []
    share_gap_by_family: dict[str, float] = {}
    oversampled_share_gap_by_family: dict[str, float] = {}
    weight_bonus_by_family: dict[str, float] = {}
    compensation_streak_by_family: dict[str, int] = {}
    compensation_multiplier_by_family: dict[str, float] = {}
    weight_normalization_by_family: dict[str, float] = {}
    normalization_streak_by_family: dict[str, int] = {}
    normalization_multiplier_by_family: dict[str, float] = {}
    applied_allocation_confidence_by_family: dict[str, float] = {}
    applied_bonus_allocation_confidence_by_family: dict[str, float] = {}
    applied_normalization_allocation_confidence_by_family: dict[str, float] = {}
    net_weight_adjustment_by_family: dict[str, float] = {}
    minimum_runs = max(1, int(confidence_controls.get("minimum_runs", ALLOCATION_CONFIDENCE_MIN_RUNS) or 0))
    target_priority_tasks = max(
        1,
        int(confidence_controls.get("target_priority_tasks", ALLOCATION_CONFIDENCE_TARGET_PRIORITY_TASKS) or 0),
    )
    target_family_tasks_override = max(0, int(confidence_controls.get("target_family_tasks", 0) or 0))
    history_window_runs = max(
        1,
        int(confidence_controls.get("history_window_runs", ALLOCATION_CONFIDENCE_HISTORY_WINDOW) or 0),
    )
    history_weight = max(
        0.0,
        min(1.0, float(confidence_controls.get("history_weight", ALLOCATION_CONFIDENCE_HISTORY_WEIGHT) or 0.0)),
    )
    bonus_history_weight = max(0.0, min(1.0, float(confidence_controls.get("bonus_history_weight", 0.75) or 0.0)))
    normalization_history_weight = max(
        0.0,
        min(1.0, float(confidence_controls.get("normalization_history_weight", 0.25) or 0.0)),
    )
    bonus_scale = max(1.0, float(len(families) * 2))
    normalization_scale = bonus_scale * ALLOCATION_NORMALIZATION_DAMPING
    for family in families:
        planned_share = float(planned_shares.get(family, 0.0) or 0.0)
        actual_share = float(actual_shares.get(family, 0.0) or 0.0)
        positive_gap = max(0.0, planned_share - actual_share)
        negative_gap = max(0.0, actual_share - planned_share)
        family_bonus_confidence = float(allocation_confidence_by_family.get(family, allocation_confidence) or 0.0)
        if family in latest_task_confidence_by_family or family in recent_average_task_confidence_by_family:
            family_bonus_confidence = min(
                allocation_confidence,
                float(latest_task_confidence_by_family.get(family, family_bonus_confidence) or 0.0)
                * (1.0 - bonus_history_weight)
                + float(recent_average_task_confidence_by_family.get(family, family_bonus_confidence) or 0.0)
                * bonus_history_weight,
            )
        elif family in bonus_allocation_confidence_by_family:
            family_bonus_confidence = float(bonus_allocation_confidence_by_family.get(family, family_bonus_confidence) or 0.0)
        family_bonus_confidence = max(0.0, min(1.0, family_bonus_confidence))
        family_normalization_confidence = float(allocation_confidence_by_family.get(family, allocation_confidence) or 0.0)
        if family in latest_task_confidence_by_family or family in recent_average_task_confidence_by_family:
            family_normalization_confidence = min(
                allocation_confidence,
                float(latest_task_confidence_by_family.get(family, family_normalization_confidence) or 0.0)
                * (1.0 - normalization_history_weight)
                + float(
                    recent_average_task_confidence_by_family.get(family, family_normalization_confidence) or 0.0
                )
                * normalization_history_weight,
            )
        elif family in normalization_allocation_confidence_by_family:
            family_normalization_confidence = float(
                normalization_allocation_confidence_by_family.get(family, family_normalization_confidence) or 0.0
            )
        family_normalization_confidence = max(0.0, min(1.0, family_normalization_confidence))
        if positive_gap >= MATERIAL_ALLOCATION_SHARE_GAP:
            positive_gap_families.append(family)
            share_gap_by_family[family] = round(positive_gap, 6)
            applied_allocation_confidence_by_family[family] = round(family_bonus_confidence, 6)
            applied_bonus_allocation_confidence_by_family[family] = round(family_bonus_confidence, 6)
            positive_gap_streak = max(1, int(latest_positive_gap_streak_by_family.get(family, 0) or 0))
            compensation_streak_by_family[family] = positive_gap_streak
            compensation_multiplier_by_family[family] = round(
                min(
                    ALLOCATION_COMPENSATION_MAX_MULTIPLIER,
                    1.0 + ALLOCATION_COMPENSATION_STREAK_STEP * max(0, positive_gap_streak - 1),
                ),
                6,
            )
            weight_bonus_by_family[family] = round(
                positive_gap * bonus_scale * compensation_multiplier_by_family[family] * family_bonus_confidence,
                6,
            )
        elif negative_gap >= MATERIAL_ALLOCATION_SHARE_GAP:
            negative_gap_families.append(family)
            oversampled_share_gap_by_family[family] = round(negative_gap, 6)
            applied_allocation_confidence_by_family[family] = round(family_normalization_confidence, 6)
            applied_normalization_allocation_confidence_by_family[family] = round(family_normalization_confidence, 6)
            oversampled_streak = max(1, int(latest_oversampled_streak_by_family.get(family, 0) or 0))
            normalization_streak_by_family[family] = oversampled_streak
            normalization_multiplier_by_family[family] = round(
                min(
                    ALLOCATION_NORMALIZATION_MAX_MULTIPLIER,
                    1.0 + ALLOCATION_NORMALIZATION_STREAK_STEP * max(0, oversampled_streak - 1),
                ),
                6,
            )
            weight_normalization_by_family[family] = round(
                min(negative_gap, MATERIAL_ALLOCATION_SHARE_GAP * 2.0)
                * normalization_scale
                * normalization_multiplier_by_family[family],
                6,
            )
            weight_normalization_by_family[family] = round(
                weight_normalization_by_family[family] * family_normalization_confidence,
                6,
            )
        net_adjustment = round(
            weight_bonus_by_family.get(family, 0.0) - weight_normalization_by_family.get(family, 0.0),
            6,
        )
        if abs(net_adjustment) > 1e-12:
            net_weight_adjustment_by_family[family] = net_adjustment
    return (
        net_weight_adjustment_by_family,
        {
            "gap_source": gap_source,
            "positive_gap_families": positive_gap_families,
            "negative_gap_families": negative_gap_families,
            "share_gap_by_family": share_gap_by_family,
            "oversampled_share_gap_by_family": oversampled_share_gap_by_family,
            "weight_bonus_by_family": weight_bonus_by_family,
            "compensation_streak_by_family": compensation_streak_by_family,
            "compensation_multiplier_by_family": compensation_multiplier_by_family,
            "weight_normalization_by_family": weight_normalization_by_family,
            "normalization_streak_by_family": normalization_streak_by_family,
            "normalization_multiplier_by_family": normalization_multiplier_by_family,
            "net_weight_adjustment_by_family": net_weight_adjustment_by_family,
            "material_gap_threshold": MATERIAL_ALLOCATION_SHARE_GAP,
            "allocation_confidence": round(allocation_confidence, 6),
            "allocation_confidence_by_family": {
                family: round(max(0.0, min(1.0, float(value or 0.0))), 6)
                for family, value in allocation_confidence_by_family.items()
                if str(family).strip()
            },
            "bonus_allocation_confidence_by_family": {
                family: round(max(0.0, min(1.0, float(value or 0.0))), 6)
                for family, value in bonus_allocation_confidence_by_family.items()
                if str(family).strip()
            },
            "normalization_allocation_confidence_by_family": {
                family: round(max(0.0, min(1.0, float(value or 0.0))), 6)
                for family, value in normalization_allocation_confidence_by_family.items()
                if str(family).strip()
            },
            "latest_task_confidence_by_family": {
                family: round(max(0.0, min(1.0, float(value or 0.0))), 6)
                for family, value in latest_task_confidence_by_family.items()
                if str(family).strip()
            },
            "recent_average_task_confidence_by_family": {
                family: round(max(0.0, min(1.0, float(value or 0.0))), 6)
                for family, value in recent_average_task_confidence_by_family.items()
                if str(family).strip()
            },
            "applied_allocation_confidence_by_family": applied_allocation_confidence_by_family,
            "applied_bonus_allocation_confidence_by_family": applied_bonus_allocation_confidence_by_family,
            "applied_normalization_allocation_confidence_by_family": applied_normalization_allocation_confidence_by_family,
            "allocation_confidence_min_runs": minimum_runs,
            "allocation_confidence_target_priority_tasks": target_priority_tasks,
            "allocation_confidence_target_family_tasks": target_family_tasks_override,
            "allocation_confidence_history_window_runs": history_window_runs,
            "allocation_confidence_history_weight": round(history_weight, 6),
            "bonus_allocation_confidence_history_weight": round(bonus_history_weight, 6),
            "normalization_allocation_confidence_history_weight": round(normalization_history_weight, 6),
            "compensation_streak_step": ALLOCATION_COMPENSATION_STREAK_STEP,
            "compensation_max_multiplier": ALLOCATION_COMPENSATION_MAX_MULTIPLIER,
            "normalization_damping": ALLOCATION_NORMALIZATION_DAMPING,
            "normalization_streak_step": ALLOCATION_NORMALIZATION_STREAK_STEP,
            "normalization_max_multiplier": ALLOCATION_NORMALIZATION_MAX_MULTIPLIER,
            "average_share_gap_by_family": {
                family: round(
                    max(
                        0.0,
                        float(average_planned_shares.get(family, 0.0) or 0.0)
                        - float(average_actual_shares.get(family, 0.0) or 0.0),
                    ),
                    6,
                )
                for family in families
                if max(
                    0.0,
                    float(average_planned_shares.get(family, 0.0) or 0.0)
                    - float(average_actual_shares.get(family, 0.0) or 0.0),
                )
                > 0.0
            },
            "latest_share_gap_by_family": {
                family: round(
                    max(
                        0.0,
                        float(latest_planned_shares.get(family, 0.0) or 0.0)
                        - float(latest_actual_shares.get(family, 0.0) or 0.0),
                    ),
                    6,
                )
                for family in families
                if max(
                    0.0,
                    float(latest_planned_shares.get(family, 0.0) or 0.0)
                    - float(latest_actual_shares.get(family, 0.0) or 0.0),
                )
                > 0.0
            },
            "latest_oversampled_share_gap_by_family": {
                family: round(
                    max(
                        0.0,
                        float(latest_actual_shares.get(family, 0.0) or 0.0)
                        - float(latest_planned_shares.get(family, 0.0) or 0.0),
                    ),
                    6,
                )
                for family in families
                if max(
                    0.0,
                    float(latest_actual_shares.get(family, 0.0) or 0.0)
                    - float(latest_planned_shares.get(family, 0.0) or 0.0),
                )
                > 0.0
            },
            "recovered_gap_families": [
                family
                for family in families
                if max(
                    0.0,
                    float(average_planned_shares.get(family, 0.0) or 0.0)
                    - float(average_actual_shares.get(family, 0.0) or 0.0),
                )
                >= 0.05
                and max(
                    0.0,
                    float(latest_planned_shares.get(family, 0.0) or 0.0)
                    - float(latest_actual_shares.get(family, 0.0) or 0.0),
                )
                < 0.05
            ],
            "latest_summary_run_index": int(allocation_audit.get("latest_summary_run_index", 0) or 0),
            "latest_summary_run_match_id": str(allocation_audit.get("latest_summary_run_match_id", "")).strip(),
            "top_planned_family": str(allocation_audit.get("top_planned_family", "")).strip(),
            "top_sampled_family": str(allocation_audit.get("top_sampled_family", "")).strip(),
            "runs_with_allocation_summary": int(allocation_audit.get("runs_with_allocation_summary", 0) or 0),
            "runs_with_top_planned_family_as_top_sampled": int(
                allocation_audit.get("runs_with_top_planned_family_as_top_sampled", 0) or 0
            ),
        },
    )


def _priority_benchmark_family_plan(
    config: KernelConfig,
    args: argparse.Namespace,
) -> tuple[list[str], str, dict[str, float], str, dict[str, object]]:
    confidence_controls = _priority_family_allocation_confidence_controls(config)
    selected = [str(value).strip() for value in getattr(args, "priority_benchmark_family", []) if str(value).strip()]
    if selected:
        return (
            selected,
            "explicit_cli",
            _default_priority_family_weights(selected),
            "explicit_cli_equal_weights",
            {
                "positive_gap_families": [],
                "share_gap_by_family": {},
                "weight_bonus_by_family": {},
                "allocation_confidence_min_runs": int(confidence_controls["minimum_runs"]),
                "allocation_confidence_target_priority_tasks": int(confidence_controls["target_priority_tasks"]),
                "allocation_confidence_target_family_tasks": int(confidence_controls["target_family_tasks"]),
                "allocation_confidence_history_window_runs": int(confidence_controls["history_window_runs"]),
                "allocation_confidence_history_weight": round(float(confidence_controls["history_weight"]), 6),
                "bonus_allocation_confidence_history_weight": round(confidence_controls["bonus_history_weight"], 6),
                "normalization_allocation_confidence_history_weight": round(
                    confidence_controls["normalization_history_weight"],
                    6,
                ),
            },
        )
    prior_claim_gate_summary = _latest_prior_compounding_claim_gate_summary(config)
    prior_ranking = prior_claim_gate_summary.get("family_transfer_investment_ranking", {})
    if not isinstance(prior_ranking, dict):
        prior_ranking = {}
    prior_ranked_families = [
        str(value).strip()
        for value in prior_ranking.get("ranked_families_by_transfer_investment", [])
        if str(value).strip()
    ]
    if prior_ranked_families:
        ordered = [family for family in prior_ranked_families if family in DEFAULT_NON_REPLAY_TRANSFER_FAMILIES]
        ordered.extend(family for family in DEFAULT_NON_REPLAY_TRANSFER_FAMILIES if family not in ordered)
        allocation_bonus_by_family, allocation_compensation = _priority_family_allocation_compensation(
            prior_claim_gate_summary,
            ordered,
            confidence_controls=confidence_controls,
        )
        family_rankings = prior_ranking.get("family_rankings", [])
        if isinstance(family_rankings, list):
            score_map: dict[str, float] = {}
            for entry in family_rankings:
                if not isinstance(entry, dict):
                    continue
                family = str(entry.get("family", "")).strip()
                if not family:
                    continue
                score = float(entry.get("investment_score", 0.0) or 0.0)
                if score > 0.0:
                    score_map[family] = score
            if score_map:
                rank_weights = _rank_weighted_priority_family_weights(ordered)
                weight_source = "prior_compounding_investment_score_plus_rank_weight"
                if allocation_compensation.get("positive_gap_families") and allocation_compensation.get("negative_gap_families"):
                    weight_source = (
                        "prior_compounding_investment_score_plus_rank_weight_and_allocation_compensation_and_normalization"
                    )
                elif allocation_compensation.get("positive_gap_families"):
                    weight_source = "prior_compounding_investment_score_plus_rank_weight_and_allocation_compensation"
                elif allocation_compensation.get("negative_gap_families"):
                    weight_source = "prior_compounding_investment_score_plus_rank_weight_and_allocation_normalization"
                return (
                    ordered,
                    "prior_compounding_ranking",
                    {
                        family: round(
                            score_map.get(family, 0.0)
                            + rank_weights[family]
                            + allocation_bonus_by_family.get(family, 0.0),
                            6,
                        )
                        for family in ordered
                    },
                    weight_source,
                    allocation_compensation,
                )
        rank_weights = _rank_weighted_priority_family_weights(ordered)
        weight_source = "prior_compounding_rank_weight"
        if allocation_compensation.get("positive_gap_families") and allocation_compensation.get("negative_gap_families"):
            weight_source = "prior_compounding_rank_weight_and_allocation_compensation_and_normalization"
        elif allocation_compensation.get("positive_gap_families"):
            weight_source = "prior_compounding_rank_weight_and_allocation_compensation"
        elif allocation_compensation.get("negative_gap_families"):
            weight_source = "prior_compounding_rank_weight_and_allocation_normalization"
        return (
            ordered,
            "prior_compounding_ranking",
            {
                family: round(
                    rank_weights[family]
                    + allocation_bonus_by_family.get(family, 0.0),
                    6,
                )
                for family in ordered
            },
            weight_source,
            allocation_compensation,
        )
    default_families = list(DEFAULT_NON_REPLAY_TRANSFER_FAMILIES)
    return (
        default_families,
        "default_non_replay_transfer_families",
        _default_priority_family_weights(default_families),
        "default_equal_weight",
        {
            "positive_gap_families": [],
            "share_gap_by_family": {},
            "weight_bonus_by_family": {},
            "allocation_confidence_min_runs": int(confidence_controls["minimum_runs"]),
            "allocation_confidence_target_priority_tasks": int(confidence_controls["target_priority_tasks"]),
            "allocation_confidence_target_family_tasks": int(confidence_controls["target_family_tasks"]),
            "allocation_confidence_history_window_runs": int(confidence_controls["history_window_runs"]),
            "allocation_confidence_history_weight": round(float(confidence_controls["history_weight"]), 6),
            "bonus_allocation_confidence_history_weight": round(confidence_controls["bonus_history_weight"], 6),
            "normalization_allocation_confidence_history_weight": round(
                confidence_controls["normalization_history_weight"],
                6,
            ),
        },
    )


def _run_env(config: KernelConfig, run_root: Path) -> dict[str, str]:
    trajectories_root = run_root / "trajectories"
    env = config.to_env()
    env.update(
        {
        "AGENT_KERNEL_WORKSPACE_ROOT": str(run_root / "workspace"),
        "AGENT_KERNEL_TRAJECTORIES_ROOT": str(trajectories_root / "episodes"),
        "AGENT_KERNEL_SKILLS_PATH": str(trajectories_root / "skills" / "command_skills.json"),
        "AGENT_KERNEL_OPERATOR_CLASSES_PATH": str(trajectories_root / "operators" / "operator_classes.json"),
        "AGENT_KERNEL_TOOL_CANDIDATES_PATH": str(trajectories_root / "tools" / "tool_candidates.json"),
        "AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH": str(trajectories_root / "benchmarks" / "benchmark_candidates.json"),
        "AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH": str(trajectories_root / "retrieval" / "retrieval_proposals.json"),
        "AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH": str(trajectories_root / "retrieval" / "retrieval_asset_bundle.json"),
        "AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH": str(trajectories_root / "tolbert_model" / "tolbert_model_artifact.json"),
        "AGENT_KERNEL_TOLBERT_SUPERVISED_DATASETS_DIR": str(trajectories_root / "tolbert_model" / "datasets"),
        "AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH": str(trajectories_root / "tolbert_model" / "liftoff_gate_report.json"),
        "AGENT_KERNEL_VERIFIER_CONTRACTS_PATH": str(trajectories_root / "verifiers" / "verifier_contracts.json"),
        "AGENT_KERNEL_PROMPT_PROPOSALS_PATH": str(trajectories_root / "prompts" / "prompt_proposals.json"),
        "AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH": str(trajectories_root / "world_model" / "world_model_proposals.json"),
        "AGENT_KERNEL_STATE_ESTIMATION_PROPOSALS_PATH": str(trajectories_root / "state_estimation" / "state_estimation_proposals.json"),
        "AGENT_KERNEL_TRUST_PROPOSALS_PATH": str(trajectories_root / "trust" / "trust_proposals.json"),
        "AGENT_KERNEL_RECOVERY_PROPOSALS_PATH": str(trajectories_root / "recovery" / "recovery_proposals.json"),
        "AGENT_KERNEL_DELEGATION_PROPOSALS_PATH": str(trajectories_root / "delegation" / "delegation_proposals.json"),
        "AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH": str(trajectories_root / "operator_policy" / "operator_policy_proposals.json"),
        "AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH": str(trajectories_root / "transition_model" / "transition_model_proposals.json"),
        "AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH": str(trajectories_root / "curriculum" / "curriculum_proposals.json"),
        "AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH": str(trajectories_root / "improvement" / "cycles.jsonl"),
        "AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT": str(trajectories_root / "improvement" / "candidates"),
        "AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR": str(trajectories_root / "improvement" / "reports"),
        "AGENT_KERNEL_RUN_REPORTS_DIR": str(trajectories_root / "reports"),
        "AGENT_KERNEL_CAPABILITY_MODULES_PATH": str(run_root / "config" / "capabilities.json"),
        "AGENT_KERNEL_RUN_CHECKPOINTS_DIR": str(trajectories_root / "checkpoints"),
        "AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH": str(trajectories_root / "jobs" / "queue.json"),
        "AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH": str(trajectories_root / "jobs" / "runtime_state.json"),
        "AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT": str(trajectories_root / "recovery" / "workspaces"),
        "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH": str(trajectories_root / "reports" / "unattended_trust_ledger.json"),
        }
    )
    return env


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(
    src: Path,
    dst: Path,
    *,
    max_files: int = 0,
    max_directories: int = 0,
    max_bytes: int = 0,
) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if max_files <= 0 and max_directories <= 0 and max_bytes <= 0:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    copied_files = 0
    copied_directories = 0
    copied_bytes = 0
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        copied_directories += 1
        dirs.sort()
        files.sort()
        if max_directories > 0 and copied_directories > max_directories:
            dirs[:] = []
            break
        root_path = Path(root)
        relative_root = root_path.relative_to(src)
        target_root = dst / relative_root if str(relative_root) != "." else dst
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            if max_files > 0 and copied_files >= max_files:
                dirs[:] = []
                break
            if max_bytes > 0 and copied_bytes >= max_bytes:
                dirs[:] = []
                break
            source_file = root_path / name
            target_file = target_root / name
            shutil.copy2(source_file, target_file)
            copied_files += 1
            copied_bytes += int(source_file.stat().st_size)
        else:
            continue
        break


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_path(path: Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.exists():
        return {"exists": False, "type": "missing", "digest": ""}
    if resolved.is_file():
        return {
            "exists": True,
            "type": "file",
            "digest": _sha256_for_file(resolved),
            "size_bytes": int(resolved.stat().st_size),
        }
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    total_bytes = 0
    truncated = False
    for root, dirs, files in os.walk(resolved):
        directory_count += 1
        dirs.sort()
        files.sort()
        if directory_count > AUTONOMOUS_SEED_FINGERPRINT_MAX_DIRECTORIES:
            truncated = True
            dirs[:] = []
            break
        root_path = Path(root)
        for name in files:
            if (
                file_count >= AUTONOMOUS_SEED_FINGERPRINT_MAX_FILES
                or total_bytes >= AUTONOMOUS_SEED_FINGERPRINT_MAX_BYTES
            ):
                truncated = True
                dirs[:] = []
                break
            child = root_path / name
            relative = child.relative_to(resolved).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_for_file(child).encode("ascii"))
            digest.update(b"\0")
            file_count += 1
            total_bytes += int(child.stat().st_size)
        if truncated:
            break
    return {
        "exists": True,
        "type": "directory",
        "digest": digest.hexdigest(),
        "file_count": file_count,
        "directory_count": directory_count,
        "size_bytes": total_bytes,
        "truncated": truncated,
        "max_files": AUTONOMOUS_SEED_FINGERPRINT_MAX_FILES,
        "max_directories": AUTONOMOUS_SEED_FINGERPRINT_MAX_DIRECTORIES,
        "max_size_bytes": AUTONOMOUS_SEED_FINGERPRINT_MAX_BYTES,
    }


def _seed_manifest(config: KernelConfig) -> dict[str, object]:
    seed_paths = {
        "trajectories_root": config.trajectories_root,
        "run_checkpoints_dir": config.run_checkpoints_dir,
        "unattended_workspace_snapshot_root": config.unattended_workspace_snapshot_root,
        "skills_path": config.skills_path,
        "operator_classes_path": config.operator_classes_path,
        "tool_candidates_path": config.tool_candidates_path,
        "benchmark_candidates_path": config.benchmark_candidates_path,
        "retrieval_proposals_path": config.retrieval_proposals_path,
        "retrieval_asset_bundle_path": config.retrieval_asset_bundle_path,
        "tolbert_model_artifact_path": config.tolbert_model_artifact_path,
        "verifier_contracts_path": config.verifier_contracts_path,
        "prompt_proposals_path": config.prompt_proposals_path,
        "world_model_proposals_path": config.world_model_proposals_path,
        "state_estimation_proposals_path": config.state_estimation_proposals_path,
        "trust_proposals_path": config.trust_proposals_path,
        "recovery_proposals_path": config.recovery_proposals_path,
        "delegation_proposals_path": config.delegation_proposals_path,
        "operator_policy_proposals_path": config.operator_policy_proposals_path,
        "transition_model_proposals_path": config.transition_model_proposals_path,
        "curriculum_proposals_path": config.curriculum_proposals_path,
        "improvement_cycles_path": config.improvement_cycles_path,
        "capability_modules_path": config.capability_modules_path,
        "delegated_job_queue_path": config.delegated_job_queue_path,
        "delegated_job_runtime_state_path": config.delegated_job_runtime_state_path,
        "unattended_trust_ledger_path": config.unattended_trust_ledger_path,
    }
    return {
        "paths": {
            name: {
                "source_path": str(path),
                **_fingerprint_path(path),
            }
            for name, path in seed_paths.items()
        }
    }


def _retention_criteria_manifest(
    config: KernelConfig,
    args: argparse.Namespace,
    *,
    task_limit: int,
    task_limit_source: str,
    priority_benchmark_families: list[str],
    priority_benchmark_family_source: str,
    priority_benchmark_family_weights: dict[str, float],
    priority_benchmark_family_weight_source: str,
    priority_benchmark_family_allocation_compensation: dict[str, object],
    autonomous_frontier_curriculum_pressure: dict[str, object],
    priority_benchmark_family_live_routing: dict[str, object],
) -> dict[str, object]:
    subsystems = (
        "benchmark",
        "curriculum",
        "verifier",
        "policy",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "capabilities",
        "retrieval",
        "tolbert_model",
    )
    return {
        "gate_presets": {subsystem: retention_gate_preset(subsystem) for subsystem in subsystems},
        "run_parameters": {
            "cycles": max(1, args.cycles),
            "campaign_width": max(1, args.campaign_width),
            "variant_width": max(1, args.variant_width),
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": task_limit,
            "task_limit_source": task_limit_source,
            "priority_benchmark_families": priority_benchmark_families,
            "priority_benchmark_family_source": priority_benchmark_family_source,
            "priority_benchmark_family_weights": priority_benchmark_family_weights,
            "priority_benchmark_family_weight_source": priority_benchmark_family_weight_source,
            "priority_benchmark_family_allocation_compensation": priority_benchmark_family_allocation_compensation,
            "priority_benchmark_family_live_routing": priority_benchmark_family_live_routing,
            "autonomous_frontier_curriculum_pressure": autonomous_frontier_curriculum_pressure,
            "include_episode_memory": bool(args.include_episode_memory),
            "include_skill_memory": bool(args.include_skill_memory),
            "include_skill_transfer": bool(args.include_skill_transfer),
            "include_operator_memory": bool(args.include_operator_memory),
            "include_tool_memory": bool(args.include_tool_memory),
            "include_verifier_memory": bool(args.include_verifier_memory),
            "include_curriculum": bool(args.include_curriculum),
            "include_failure_curriculum": bool(args.include_failure_curriculum),
        },
        "runtime_config": {
            "provider": str(config.provider),
            "model_name": str(config.model_name),
            "compare_feature_max_tasks": int(config.compare_feature_max_tasks),
            "persist_episode_memory": bool(config.persist_episode_memory),
        },
    }


def _manifest_fingerprint(manifest: dict[str, object]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _autonomous_frontier_curriculum_pressure(
    *,
    priority_benchmark_families: list[str],
    priority_benchmark_family_allocation_compensation: dict[str, object],
    prior_claim_gate_summary: dict[str, object] | None,
) -> dict[str, object]:
    summary = prior_claim_gate_summary.get("family_transfer_summary", {}) if isinstance(prior_claim_gate_summary, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    timeline = prior_claim_gate_summary.get("family_transfer_timeline", {}) if isinstance(prior_claim_gate_summary, dict) else {}
    if not isinstance(timeline, dict):
        timeline = {}
    ranking = prior_claim_gate_summary.get("family_transfer_investment_ranking", {}) if isinstance(prior_claim_gate_summary, dict) else {}
    if not isinstance(ranking, dict):
        ranking = {}
    frontier_expansion = (
        prior_claim_gate_summary.get("frontier_expansion_summary", {}) if isinstance(prior_claim_gate_summary, dict) else {}
    )
    if not isinstance(frontier_expansion, dict):
        frontier_expansion = {}
    target_families = _ordered_unique_strings(
        priority_benchmark_families,
        ranking.get("ranked_families_by_transfer_investment", []),
        DEFAULT_NON_REPLAY_TRANSFER_FAMILIES,
    )
    target_family_set = set(target_families)
    missing_observation_families = [
        family for family in _ordered_unique_strings(summary.get("families_missing_observation", [])) if family in target_family_set
    ]
    without_retained_gain_families = [
        family for family in _ordered_unique_strings(summary.get("families_without_retained_gain", [])) if family in target_family_set
    ]
    declining_transfer_families = [
        family
        for family in _ordered_unique_strings(
            summary.get("families_with_negative_retained_delta", []),
            timeline.get("families_with_declining_repeated_retained_gain", []),
            timeline.get("families_with_declining_repeated_return_on_cost", []),
            timeline.get("families_with_costly_non_declining_repeated_retained_gain", []),
        )
        if family in target_family_set
    ]
    persistent_gain_families = [
        family
        for family in _ordered_unique_strings(timeline.get("families_with_cost_acceptable_non_declining_repeated_retained_gain", []))
        if family in target_family_set
    ]
    under_sampled_families = [
        family
        for family in _ordered_unique_strings(priority_benchmark_family_allocation_compensation.get("positive_gap_families", []))
        if family in target_family_set
    ]
    unsampled_target_families = [
        family
        for family in _ordered_unique_strings(frontier_expansion.get("families_never_sampled", []))
        if family in target_family_set
    ]
    unsampled_pressure_families = [
        family
        for family in _ordered_unique_strings(frontier_expansion.get("pressure_families_without_sampling", []))
        if family in target_family_set
    ]
    generalization_priority_families = _ordered_unique_strings(
        without_retained_gain_families,
        declining_transfer_families,
    )
    priority_families = _ordered_unique_strings(
        missing_observation_families,
        unsampled_pressure_families,
        under_sampled_families,
        unsampled_target_families,
        generalization_priority_families,
        persistent_gain_families,
        target_families,
    )
    missing_families = _ordered_unique_strings(
        missing_observation_families,
        unsampled_pressure_families,
        under_sampled_families,
        unsampled_target_families,
    )
    retention_priority_families = _ordered_unique_strings(without_retained_gain_families, declining_transfer_families)
    higher_pressure = bool(missing_families or retention_priority_families or generalization_priority_families)
    controls = {
        "frontier_priority_families": priority_families,
        "frontier_missing_families": missing_families,
        "frontier_retention_priority_families": retention_priority_families,
        "frontier_generalization_priority_families": generalization_priority_families,
        "frontier_priority_family_bonus": 3 if priority_families else 2,
        "frontier_missing_family_bonus": 5 if missing_families else 4,
        "frontier_retention_priority_bonus": 4 if retention_priority_families else 2,
        "frontier_generalization_bonus": 4 if generalization_priority_families else 3,
        "frontier_outward_branch_bonus": 3 if higher_pressure else 2,
        "frontier_lineage_breadth_bonus": 2 if generalization_priority_families else 1,
        "frontier_harder_task_bonus": 4 if higher_pressure else 2,
        "frontier_min_lineage_depth": 2 if retention_priority_families else (1 if missing_families else 0),
        "max_generated_adjacent_tasks": max(4, min(6, len(priority_families) + 1)),
    }
    return {
        "target_non_replay_families": target_families,
        "priority_families": priority_families,
        "missing_observation_families": missing_observation_families,
        "under_sampled_families": under_sampled_families,
        "unsampled_target_families": unsampled_target_families,
        "unsampled_pressure_families": unsampled_pressure_families,
        "without_retained_gain_families": without_retained_gain_families,
        "declining_transfer_families": declining_transfer_families,
        "persistent_gain_families": persistent_gain_families,
        "generalization_priority_families": generalization_priority_families,
        "controls": controls,
    }


def _autonomous_frontier_live_priority_routing(
    *,
    task_limit: int,
    task_limit_source: str,
    priority_benchmark_families: list[str],
    priority_benchmark_family_weights: dict[str, float],
    priority_benchmark_family_weight_source: str,
    autonomous_frontier_curriculum_pressure: dict[str, object],
) -> tuple[int, str, list[str], dict[str, float], str, dict[str, object]]:
    pressure = autonomous_frontier_curriculum_pressure if isinstance(autonomous_frontier_curriculum_pressure, dict) else {}
    missing_families = _ordered_unique_strings(pressure.get("missing_observation_families", []))
    unsampled_pressure_families = _ordered_unique_strings(pressure.get("unsampled_pressure_families", []))
    under_sampled_families = _ordered_unique_strings(pressure.get("under_sampled_families", []))
    unsampled_target_families = _ordered_unique_strings(pressure.get("unsampled_target_families", []))
    generalization_priority_families = _ordered_unique_strings(pressure.get("generalization_priority_families", []))
    routed_families = _ordered_unique_strings(
        missing_families,
        unsampled_pressure_families,
        under_sampled_families,
        unsampled_target_families,
        generalization_priority_families,
        pressure.get("priority_families", []),
        priority_benchmark_families,
        DEFAULT_NON_REPLAY_TRANSFER_FAMILIES,
    )
    base_weights = {
        family: float(priority_benchmark_family_weights.get(family, 0.0) or 0.0)
        for family in routed_families
    }
    default_rank_weights = _rank_weighted_priority_family_weights(routed_families)
    for family in routed_families:
        if base_weights[family] <= 0.0:
            base_weights[family] = float(default_rank_weights.get(family, 1.0) or 1.0)
    weight_bonus_by_family: dict[str, float] = {}
    routed_weights: dict[str, float] = {}
    for family in routed_families:
        bonus = 0.0
        if family in missing_families:
            bonus += AUTONOMOUS_FRONTIER_MISSING_FAMILY_WEIGHT_BONUS
        if family in unsampled_pressure_families:
            bonus += AUTONOMOUS_FRONTIER_UNSAMPLED_PRESSURE_WEIGHT_BONUS
        if family in under_sampled_families:
            bonus += AUTONOMOUS_FRONTIER_UNDER_SAMPLED_WEIGHT_BONUS
        if family in unsampled_target_families:
            bonus += AUTONOMOUS_FRONTIER_UNSAMPLED_TARGET_WEIGHT_BONUS
        if family in generalization_priority_families:
            bonus += AUTONOMOUS_FRONTIER_GENERALIZATION_WEIGHT_BONUS
        if bonus > 0.0:
            weight_bonus_by_family[family] = round(bonus, 6)
        routed_weights[family] = round(base_weights[family] + bonus, 6)
    routing_pressure_families = _ordered_unique_strings(
        missing_families,
        unsampled_pressure_families,
        under_sampled_families,
        unsampled_target_families,
        generalization_priority_families,
    )
    pressure_weight_floor_by_family: dict[str, float] = {}
    if routing_pressure_families:
        highest_non_pressure_weight = max(
            [routed_weights[family] for family in routed_families if family not in routing_pressure_families] or [0.0]
        )
        pressure_family_count = len(routing_pressure_families)
        for index, family in enumerate(routing_pressure_families):
            minimum_weight = round(highest_non_pressure_weight + float(max(1, pressure_family_count - index)), 6)
            if routed_weights[family] >= minimum_weight:
                continue
            additional_bonus = round(minimum_weight - routed_weights[family], 6)
            routed_weights[family] = minimum_weight
            weight_bonus_by_family[family] = round(weight_bonus_by_family.get(family, 0.0) + additional_bonus, 6)
            pressure_weight_floor_by_family[family] = minimum_weight
    effective_task_limit = max(0, int(task_limit))
    effective_task_limit_source = str(task_limit_source)
    task_limit_floor = 0
    if effective_task_limit > 0 and routing_pressure_families:
        task_limit_floor = min(
            AUTONOMOUS_FRONTIER_TASK_LIMIT_MAX,
            max(AUTONOMOUS_FRONTIER_TASK_LIMIT_FLOOR, len(routing_pressure_families)),
        )
        if effective_task_limit < task_limit_floor:
            effective_task_limit = task_limit_floor
            effective_task_limit_source = f"{effective_task_limit_source}_and_autonomous_frontier_pressure_floor"
    effective_weight_source = str(priority_benchmark_family_weight_source)
    if weight_bonus_by_family:
        effective_weight_source = f"{effective_weight_source}_and_autonomous_frontier_pressure"
    return (
        effective_task_limit,
        effective_task_limit_source,
        routed_families,
        routed_weights,
        effective_weight_source,
        {
            "routing_pressure_families": routing_pressure_families,
            "missing_observation_families": missing_families,
            "unsampled_pressure_families": unsampled_pressure_families,
            "under_sampled_families": under_sampled_families,
            "unsampled_target_families": unsampled_target_families,
            "generalization_priority_families": generalization_priority_families,
            "weight_bonus_by_family": weight_bonus_by_family,
            "pressure_weight_floor_by_family": pressure_weight_floor_by_family,
            "base_weight_by_family": {family: round(base_weights[family], 6) for family in routed_families},
            "routed_weight_by_family": routed_weights,
            "task_limit_floor": task_limit_floor,
            "task_limit_floor_applied": bool(task_limit_floor and effective_task_limit == task_limit_floor),
        },
    )


def _write_scoped_curriculum_proposals(
    *,
    config: KernelConfig,
    scoped_path: Path,
    autonomous_frontier_curriculum_pressure: dict[str, object],
) -> dict[str, object]:
    current_payload = _load_json_payload(config.curriculum_proposals_path)
    baseline_controls = retained_curriculum_controls(current_payload)
    controls = dict(baseline_controls)
    derived_controls = autonomous_frontier_curriculum_pressure.get("controls", {})
    if isinstance(derived_controls, dict):
        controls.update(derived_controls)
    priority_families = _ordered_unique_strings(autonomous_frontier_curriculum_pressure.get("priority_families", []))
    generalization_priority_families = _ordered_unique_strings(
        autonomous_frontier_curriculum_pressure.get("generalization_priority_families", [])
    )
    missing_families = _ordered_unique_strings(autonomous_frontier_curriculum_pressure.get("missing_observation_families", []))
    proposals: list[dict[str, object]] = []
    if missing_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 6,
                "reason": f"autonomous compounding still lacks live transfer observation on {', '.join(missing_families[:3])}",
                "suggestion": "Generate outward long-horizon followups that open those missing coding families before repeating already-observed lanes.",
            }
        )
    if generalization_priority_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 6,
                "reason": f"retained gains remain narrow or unstable on {', '.join(generalization_priority_families[:3])}",
                "suggestion": "Prefer harder outward-branch followups in those families so retention pressure rewards generalization, not replay of the same narrow seed shape.",
            }
        )
    elif priority_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": f"next autonomous batch should keep pressure on {priority_families[0]} before re-saturating easier lanes",
                "suggestion": "Spend generated-success budget on broader repo and workflow settings first, then deepen the strongest lineages by one harder step.",
            }
        )
    retained = current_payload if str(current_payload.get("artifact_kind", "")).strip() == "curriculum_proposal_set" else {}
    retained_proposals = list(retained.get("proposals", [])) if isinstance(retained.get("proposals", []), list) else []
    artifact = build_standard_proposal_artifact(
        artifact_kind="curriculum_proposal_set",
        generation_focus=str(retained.get("generation_focus", "balanced") or "balanced"),
        retention_gate=dict(retained.get("retention_gate", retention_gate_preset("curriculum"))),
        control_schema=str(retained.get("control_schema", "curriculum_behavior_controls_v3") or "curriculum_behavior_controls_v3"),
        controls=controls,
        proposals=[*proposals, *retained_proposals],
        extra_sections={
            "autonomous_frontier_curriculum_pressure": autonomous_frontier_curriculum_pressure,
        },
    )
    artifact["lifecycle_state"] = "retained"
    artifact["retention_decision"] = {"state": "retain"}
    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    scoped_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _seed_runtime(config: KernelConfig, env: dict[str, str]) -> None:
    bounded_seed_kwargs = {
        "max_files": AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_FILES,
        "max_directories": AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_DIRECTORIES,
        "max_bytes": AUTONOMOUS_TRAJECTORY_SEED_COPY_MAX_BYTES,
    }
    _copy_tree(
        config.trajectories_root,
        Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]),
        **bounded_seed_kwargs,
    )
    _copy_tree(config.run_checkpoints_dir, Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]), **bounded_seed_kwargs)
    _copy_tree(
        config.unattended_workspace_snapshot_root,
        Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]),
        **bounded_seed_kwargs,
    )
    _copy_file(config.skills_path, Path(env["AGENT_KERNEL_SKILLS_PATH"]))
    _copy_file(config.operator_classes_path, Path(env["AGENT_KERNEL_OPERATOR_CLASSES_PATH"]))
    _copy_file(config.tool_candidates_path, Path(env["AGENT_KERNEL_TOOL_CANDIDATES_PATH"]))
    _copy_file(config.benchmark_candidates_path, Path(env["AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH"]))
    _copy_file(config.retrieval_proposals_path, Path(env["AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH"]))
    _copy_file(config.retrieval_asset_bundle_path, Path(env["AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH"]))
    _copy_file(config.tolbert_model_artifact_path, Path(env["AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH"]))
    _copy_file(config.verifier_contracts_path, Path(env["AGENT_KERNEL_VERIFIER_CONTRACTS_PATH"]))
    _copy_file(config.prompt_proposals_path, Path(env["AGENT_KERNEL_PROMPT_PROPOSALS_PATH"]))
    _copy_file(config.world_model_proposals_path, Path(env["AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH"]))
    _copy_file(config.state_estimation_proposals_path, Path(env["AGENT_KERNEL_STATE_ESTIMATION_PROPOSALS_PATH"]))
    _copy_file(config.trust_proposals_path, Path(env["AGENT_KERNEL_TRUST_PROPOSALS_PATH"]))
    _copy_file(config.recovery_proposals_path, Path(env["AGENT_KERNEL_RECOVERY_PROPOSALS_PATH"]))
    _copy_file(config.delegation_proposals_path, Path(env["AGENT_KERNEL_DELEGATION_PROPOSALS_PATH"]))
    _copy_file(config.operator_policy_proposals_path, Path(env["AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH"]))
    _copy_file(config.transition_model_proposals_path, Path(env["AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH"]))
    _copy_file(config.curriculum_proposals_path, Path(env["AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH"]))
    _copy_file(config.improvement_cycles_path, Path(env["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"]))
    _copy_file(config.capability_modules_path, Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]))
    _copy_file(config.delegated_job_queue_path, Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]))
    _copy_file(config.delegated_job_runtime_state_path, Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]))
    _copy_file(config.unattended_trust_ledger_path, Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]))
    Path(env["AGENT_KERNEL_WORKSPACE_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_RUN_REPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_TOLBERT_SUPERVISED_DATASETS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]).parent.mkdir(parents=True, exist_ok=True)


def _run_command(
    repo_root: Path,
    args: argparse.Namespace,
    *,
    run_match_id: str,
    run_index: int,
    task_limit: int,
    priority_benchmark_families: list[str],
    priority_benchmark_family_weights: dict[str, float],
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_repeated_improvement_cycles.py"),
        "--cycles",
        str(max(1, args.cycles)),
        "--campaign-width",
        str(max(1, args.campaign_width)),
        "--variant-width",
        str(max(1, args.variant_width)),
        "--campaign-label",
        f"autonomous-run-{run_index}",
        "--campaign-match-id",
        run_match_id,
    ]
    if task_limit > 0:
        cmd.extend(["--task-limit", str(task_limit)])
    if args.adaptive_search:
        cmd.append("--adaptive-search")
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.model:
        cmd.extend(["--model", args.model])
    for family in priority_benchmark_families:
        cmd.extend(["--priority-benchmark-family", family])
    for family in priority_benchmark_families:
        weight = float(priority_benchmark_family_weights.get(family, 0.0) or 0.0)
        if weight > 0.0:
            cmd.extend(["--priority-benchmark-family-weight", f"{family}={weight:.6f}"])
    for flag, enabled in (
        ("--include-episode-memory", args.include_episode_memory),
        ("--include-skill-memory", args.include_skill_memory),
        ("--include-skill-transfer", args.include_skill_transfer),
        ("--include-operator-memory", args.include_operator_memory),
        ("--include-tool-memory", args.include_tool_memory),
        ("--include-verifier-memory", args.include_verifier_memory),
        ("--include-curriculum", args.include_curriculum),
        ("--include-failure-curriculum", args.include_failure_curriculum),
    ):
        if enabled:
            cmd.append(flag)
    return cmd


def _run_once(
    *,
    repo_root: Path,
    config: KernelConfig,
    args: argparse.Namespace,
    run_match_id: str,
    run_index: int,
    priority_benchmark_family_source: str,
    priority_benchmark_family_weight_source: str,
    priority_benchmark_family_allocation_compensation: dict[str, object],
    autonomous_frontier_curriculum_pressure: dict[str, object],
    routed_task_limit: int,
    routed_task_limit_source: str,
    routed_priority_benchmark_families: list[str],
    routed_priority_benchmark_family_weights: dict[str, float],
    routed_priority_benchmark_family_weight_source: str,
    priority_benchmark_family_live_routing: dict[str, object],
) -> dict[str, object]:
    seed_manifest = _seed_manifest(config)
    retention_criteria_manifest = _retention_criteria_manifest(
        config,
        args,
        task_limit=routed_task_limit,
        task_limit_source=routed_task_limit_source,
        priority_benchmark_families=routed_priority_benchmark_families,
        priority_benchmark_family_source=priority_benchmark_family_source,
        priority_benchmark_family_weights=routed_priority_benchmark_family_weights,
        priority_benchmark_family_weight_source=routed_priority_benchmark_family_weight_source,
        priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
        autonomous_frontier_curriculum_pressure=autonomous_frontier_curriculum_pressure,
        priority_benchmark_family_live_routing=priority_benchmark_family_live_routing,
    )
    root = _run_root(config, run_match_id=run_match_id)
    env_overrides = _run_env(config, root)
    _seed_runtime(config, env_overrides)
    _write_scoped_curriculum_proposals(
        config=config,
        scoped_path=Path(env_overrides["AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH"]),
        autonomous_frontier_curriculum_pressure=autonomous_frontier_curriculum_pressure,
    )
    env = dict(os.environ)
    env.update(env_overrides)
    env["AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH"] = str(_status_path(config))
    env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX"] = str(run_index)
    env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID"] = run_match_id
    env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUNTIME_ROOT"] = str(root)
    completed = subprocess.run(
        _run_command(
            repo_root,
            args,
            run_match_id=run_match_id,
            run_index=run_index,
            task_limit=routed_task_limit,
            priority_benchmark_families=routed_priority_benchmark_families,
            priority_benchmark_family_weights=routed_priority_benchmark_family_weights,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    stdout = str(completed.stdout).strip()
    report_path = Path(stdout.splitlines()[-1]) if stdout else Path()
    report_payload: dict[str, object] = {}
    if report_path.exists():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report_payload = loaded
    production_summary = report_payload.get("production_yield_summary", {})
    if not isinstance(production_summary, dict):
        production_summary = {}
    phase_gate_summary = report_payload.get("phase_gate_summary", {})
    if not isinstance(phase_gate_summary, dict):
        phase_gate_summary = {}
    return {
        "run_match_id": run_match_id,
        "run_index": run_index,
        "returncode": int(completed.returncode),
        "stdout": stdout,
        "stderr": str(completed.stderr).strip(),
        "runtime_root": str(root),
        "report_path": str(report_path),
        "report_payload": report_payload,
        "production_yield_summary": production_summary,
        "phase_gate_summary": phase_gate_summary,
        "seed_manifest": seed_manifest,
        "seed_fingerprint": _manifest_fingerprint(seed_manifest),
        "retention_criteria_manifest": retention_criteria_manifest,
        "retention_criteria_fingerprint": _manifest_fingerprint(retention_criteria_manifest),
        "task_limit": routed_task_limit,
        "task_limit_source": routed_task_limit_source,
        "priority_benchmark_families": routed_priority_benchmark_families,
        "priority_benchmark_family_weights": routed_priority_benchmark_family_weights,
        "priority_benchmark_family_weight_source": routed_priority_benchmark_family_weight_source,
        "priority_benchmark_family_live_routing": priority_benchmark_family_live_routing,
        "autonomous_frontier_curriculum_pressure": autonomous_frontier_curriculum_pressure,
    }


def _average(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _coerced_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerced_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _retrieval_carryover_summary(results: list[dict[str, object]]) -> dict[str, object]:
    runs_with_metrics: list[int] = []
    runs_with_verified_improvement: list[int] = []
    runs_with_verified_non_regression: list[int] = []
    runs_missing_verified_non_regression: list[int] = []
    observed_retrieval_decision_count = 0
    best_trusted_carryover_repair_rate = 0.0
    best_trusted_carryover_verified_steps = 0

    carryover_metric_keys = {
        "trusted_carryover_repair_rate",
        "baseline_trusted_carryover_repair_rate",
        "trusted_carryover_verified_steps",
        "baseline_trusted_carryover_verified_steps",
        "trusted_carryover_verified_step_delta",
    }

    for result in results:
        run_index = int(result.get("run_index", 0) or 0)
        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        decisions = report_payload.get("recent_runtime_managed_decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        run_observed_metrics = False
        run_verified_improvement = False
        run_verified_non_regression = False
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            subsystem = str(decision.get("subsystem", "")).strip()
            if subsystem and subsystem != "retrieval":
                continue
            metrics = decision.get("metrics_summary", {})
            if not isinstance(metrics, dict) or not any(key in metrics for key in carryover_metric_keys):
                continue
            observed_retrieval_decision_count += 1
            run_observed_metrics = True
            trusted_carryover_repair_rate = max(
                0.0,
                _coerced_float(metrics.get("trusted_carryover_repair_rate", 0.0)),
            )
            baseline_trusted_carryover_repair_rate = max(
                0.0,
                _coerced_float(metrics.get("baseline_trusted_carryover_repair_rate", 0.0)),
            )
            trusted_carryover_verified_steps = max(
                0,
                _coerced_int(metrics.get("trusted_carryover_verified_steps", 0)),
            )
            baseline_trusted_carryover_verified_steps = max(
                0,
                _coerced_int(metrics.get("baseline_trusted_carryover_verified_steps", 0)),
            )
            trusted_carryover_verified_step_delta = _coerced_int(
                metrics.get("trusted_carryover_verified_step_delta", 0)
            )
            best_trusted_carryover_repair_rate = max(
                best_trusted_carryover_repair_rate,
                trusted_carryover_repair_rate,
            )
            best_trusted_carryover_verified_steps = max(
                best_trusted_carryover_verified_steps,
                trusted_carryover_verified_steps,
            )
            if (
                trusted_carryover_verified_step_delta > 0
                or trusted_carryover_repair_rate > baseline_trusted_carryover_repair_rate
            ):
                run_verified_improvement = True
            if (
                trusted_carryover_verified_steps > 0
                and trusted_carryover_repair_rate >= baseline_trusted_carryover_repair_rate
                and trusted_carryover_verified_steps >= baseline_trusted_carryover_verified_steps
            ):
                run_verified_non_regression = True
        if run_observed_metrics:
            runs_with_metrics.append(run_index)
            if run_verified_improvement:
                runs_with_verified_improvement.append(run_index)
            if run_verified_non_regression:
                runs_with_verified_non_regression.append(run_index)
            else:
                runs_missing_verified_non_regression.append(run_index)

    return {
        "runs_checked": len(results),
        "observed_retrieval_decision_count": observed_retrieval_decision_count,
        "runs_with_retrieval_carryover_metrics": runs_with_metrics,
        "runs_with_verified_carryover_improvement": runs_with_verified_improvement,
        "runs_with_verified_carryover_non_regression": runs_with_verified_non_regression,
        "runs_missing_verified_carryover_non_regression": runs_missing_verified_non_regression,
        "best_trusted_carryover_repair_rate": round(best_trusted_carryover_repair_rate, 4),
        "best_trusted_carryover_verified_steps": best_trusted_carryover_verified_steps,
        "carryover_pressure": bool(runs_missing_verified_non_regression),
    }


def _result_stream_for_run(result: dict[str, object]) -> dict[str, object]:
    report_payload = result.get("report_payload", {})
    if not isinstance(report_payload, dict):
        report_payload = {}
    record_scope = report_payload.get("record_scope", {})
    if not isinstance(record_scope, dict):
        record_scope = {}
    decision_stream_summary = report_payload.get("decision_stream_summary", {})
    if not isinstance(decision_stream_summary, dict):
        decision_stream_summary = {}
    runtime_stream = decision_stream_summary.get("runtime_managed", {})
    if not isinstance(runtime_stream, dict):
        runtime_stream = {}
    non_runtime_stream = decision_stream_summary.get("non_runtime_managed", {})
    if not isinstance(non_runtime_stream, dict):
        non_runtime_stream = {}
    phase_gate_summary = report_payload.get("phase_gate_summary", {})
    if not isinstance(phase_gate_summary, dict):
        phase_gate_summary = {}
    trust_breadth_summary = report_payload.get("trust_breadth_summary", {})
    if not isinstance(trust_breadth_summary, dict):
        trust_breadth_summary = {}
    priority_family_yield_summary = report_payload.get("priority_family_yield_summary", {})
    if not isinstance(priority_family_yield_summary, dict):
        priority_family_yield_summary = {}
    priority_family_allocation_summary = report_payload.get("priority_family_allocation_summary", {})
    if not isinstance(priority_family_allocation_summary, dict):
        priority_family_allocation_summary = {}
    recent_runtime_managed_decisions = report_payload.get("recent_runtime_managed_decisions", [])
    if not isinstance(recent_runtime_managed_decisions, list):
        recent_runtime_managed_decisions = []
    comparable_recent_runtime_decisions: list[dict[str, object]] = []
    for decision in recent_runtime_managed_decisions[-10:]:
        if not isinstance(decision, dict):
            continue
        comparable_recent_runtime_decisions.append(
            {
                "cycle_id": str(decision.get("cycle_id", "")).strip(),
                "state": str(decision.get("state", "")).strip(),
                "subsystem": str(decision.get("subsystem", "")).strip(),
                "artifact_kind": str(decision.get("artifact_kind", "")).strip(),
                "artifact_path": str(decision.get("artifact_path", "")).strip(),
            }
        )
    scoped_protocol = str(record_scope.get("protocol", "")).strip()
    scoped_campaign_match_id = str(record_scope.get("campaign_match_id", "")).strip()
    return {
        "run_match_id": str(result.get("run_match_id", "")).strip(),
        "run_index": int(result.get("run_index", 0) or 0),
        "campaign_label": str(report_payload.get("campaign_label", "")).strip(),
        "campaign_match_id": str(report_payload.get("campaign_match_id", "")).strip(),
        "record_scope": {
            "protocol": scoped_protocol,
            "campaign_match_id": scoped_campaign_match_id,
            "records_considered": int(record_scope.get("records_considered", 0) or 0),
            "decision_records_considered": int(record_scope.get("decision_records_considered", 0) or 0),
            "cycle_ids": list(record_scope.get("cycle_ids", []))
            if isinstance(record_scope.get("cycle_ids", []), list)
            else [],
        },
        "decision_stream_summary": {
            "runtime_managed": runtime_stream,
            "non_runtime_managed": non_runtime_stream,
        },
        "phase_gate_summary": phase_gate_summary,
        "trust_breadth_summary": trust_breadth_summary,
        "priority_benchmark_families": list(report_payload.get("priority_benchmark_families", []))
        if isinstance(report_payload.get("priority_benchmark_families", []), list)
        else [],
        "priority_family_yield_summary": priority_family_yield_summary,
        "priority_family_allocation_summary": priority_family_allocation_summary,
        "recent_runtime_managed_decisions": comparable_recent_runtime_decisions,
        "result_stream_ready": bool(
            scoped_protocol == "autonomous"
            and scoped_campaign_match_id
            and int(record_scope.get("records_considered", 0) or 0) > 0
            and "total_decisions" in runtime_stream
        ),
    }


def _result_stream_audit(results: list[dict[str, object]]) -> dict[str, object]:
    streams = [_result_stream_for_run(result) for result in results]
    missing_scoped_runs = [
        int(stream.get("run_index", 0) or 0)
        for stream in streams
        if not (
            str(dict(stream.get("record_scope", {})).get("protocol", "")).strip() == "autonomous"
            and str(dict(stream.get("record_scope", {})).get("campaign_match_id", "")).strip()
        )
    ]
    missing_runtime_stream_runs = [
        int(stream.get("run_index", 0) or 0)
        for stream in streams
        if "total_decisions" not in dict(dict(stream.get("decision_stream_summary", {})).get("runtime_managed", {}))
    ]
    warnings: list[str] = []
    if missing_scoped_runs:
        warnings.append("one_or_more_runs_missing_scoped_campaign_record_stream")
    if missing_runtime_stream_runs:
        warnings.append("one_or_more_runs_missing_runtime_managed_result_stream")
    return {
        "runs_checked": len(streams),
        "runs_with_scoped_campaign_record_stream": len(streams) - len(missing_scoped_runs),
        "runs_with_runtime_managed_result_stream": len(streams) - len(missing_runtime_stream_runs),
        "missing_scoped_campaign_record_stream_runs": missing_scoped_runs,
        "missing_runtime_managed_result_stream_runs": missing_runtime_stream_runs,
        "warnings": warnings,
        "result_streams": streams,
    }


def _family_transfer_summary(results: list[dict[str, object]]) -> dict[str, object]:
    target_families: list[str] = []
    for result in results:
        retention_manifest = result.get("retention_criteria_manifest", {})
        if not isinstance(retention_manifest, dict):
            retention_manifest = {}
        run_parameters = retention_manifest.get("run_parameters", {})
        if not isinstance(run_parameters, dict):
            run_parameters = {}
        for value in run_parameters.get("priority_benchmark_families", []):
            family = str(value).strip()
            if family and family not in target_families:
                target_families.append(family)
    if not target_families:
        target_families = list(DEFAULT_NON_REPLAY_TRANSFER_FAMILIES)
    family_summaries: dict[str, dict[str, object]] = {
        family: {
            "runs_with_observation": 0,
            "runs_with_required_report": 0,
            "runs_with_external_coverage": 0,
            "runs_with_retained_gain": 0,
            "runs_with_negative_retained_delta": 0,
            "observed_decisions": 0,
            "observed_estimated_cost": 0.0,
            "retained_positive_delta_decisions": 0,
            "retained_positive_pass_rate_delta_sum": 0.0,
            "retained_estimated_cost": 0.0,
        }
        for family in target_families
    }
    for result in results:
        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        trust_breadth_summary = report_payload.get("trust_breadth_summary", {})
        if not isinstance(trust_breadth_summary, dict):
            trust_breadth_summary = {}
        priority_family_yield_summary = report_payload.get("priority_family_yield_summary", {})
        if not isinstance(priority_family_yield_summary, dict):
            priority_family_yield_summary = {}
        required_families_with_reports = {
            str(value).strip() for value in trust_breadth_summary.get("required_families_with_reports", []) if str(value).strip()
        }
        external_benchmark_families = {
            str(value).strip() for value in trust_breadth_summary.get("external_benchmark_families", []) if str(value).strip()
        }
        family_yield_summaries = priority_family_yield_summary.get("family_summaries", {})
        if not isinstance(family_yield_summaries, dict):
            family_yield_summaries = {}
        for family in target_families:
            family_summary = family_summaries[family]
            yield_summary = family_yield_summaries.get(family, {})
            if not isinstance(yield_summary, dict):
                yield_summary = {}
            observed_decisions = int(yield_summary.get("observed_decisions", 0) or 0)
            retained_positive_delta_decisions = int(yield_summary.get("retained_positive_delta_decisions", 0) or 0)
            retained_negative_delta_decisions = int(yield_summary.get("retained_negative_delta_decisions", 0) or 0)
            retained_positive_pass_rate_delta_sum = float(
                yield_summary.get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0
            )
            observed_estimated_cost = float(yield_summary.get("observed_estimated_cost", 0.0) or 0.0)
            retained_estimated_cost = float(yield_summary.get("retained_estimated_cost", 0.0) or 0.0)
            observed = family in required_families_with_reports or family in external_benchmark_families or observed_decisions > 0
            if observed:
                family_summary["runs_with_observation"] = int(family_summary["runs_with_observation"]) + 1
            if family in required_families_with_reports:
                family_summary["runs_with_required_report"] = int(family_summary["runs_with_required_report"]) + 1
            if family in external_benchmark_families:
                family_summary["runs_with_external_coverage"] = int(family_summary["runs_with_external_coverage"]) + 1
            if retained_positive_delta_decisions > 0:
                family_summary["runs_with_retained_gain"] = int(family_summary["runs_with_retained_gain"]) + 1
            if retained_negative_delta_decisions > 0:
                family_summary["runs_with_negative_retained_delta"] = int(
                    family_summary["runs_with_negative_retained_delta"]
                ) + 1
            family_summary["observed_decisions"] = int(family_summary["observed_decisions"]) + observed_decisions
            family_summary["observed_estimated_cost"] = float(family_summary["observed_estimated_cost"]) + observed_estimated_cost
            family_summary["retained_positive_delta_decisions"] = int(
                family_summary["retained_positive_delta_decisions"]
            ) + retained_positive_delta_decisions
            family_summary["retained_positive_pass_rate_delta_sum"] = float(
                family_summary["retained_positive_pass_rate_delta_sum"]
            ) + retained_positive_pass_rate_delta_sum
            family_summary["retained_estimated_cost"] = float(family_summary["retained_estimated_cost"]) + retained_estimated_cost
    families_observed = [family for family in target_families if int(family_summaries[family]["runs_with_observation"]) > 0]
    families_with_retained_gain = [
        family for family in target_families if int(family_summaries[family]["runs_with_retained_gain"]) > 0
    ]
    families_with_repeated_retained_gain = [
        family for family in target_families if int(family_summaries[family]["runs_with_retained_gain"]) > 1
    ]
    families_with_negative_retained_delta = [
        family for family in target_families if int(family_summaries[family]["runs_with_negative_retained_delta"]) > 0
    ]
    return {
        "runs_checked": len(results),
        "target_non_replay_families": target_families,
        "distinct_target_families_observed": len(families_observed),
        "distinct_target_families_with_retained_gain": len(families_with_retained_gain),
        "families_observed": families_observed,
        "families_missing_observation": [family for family in target_families if family not in families_observed],
        "families_with_retained_gain": families_with_retained_gain,
        "families_without_retained_gain": [family for family in target_families if family not in families_with_retained_gain],
        "families_with_repeated_retained_gain": families_with_repeated_retained_gain,
        "families_with_negative_retained_delta": families_with_negative_retained_delta,
        "family_summaries": family_summaries,
    }


def _family_transfer_timeline(results: list[dict[str, object]]) -> dict[str, object]:
    summary = _family_transfer_summary(results)
    target_families = list(summary.get("target_non_replay_families", []))
    timelines: dict[str, list[dict[str, object]]] = {family: [] for family in target_families}
    for result in results:
        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        trust_breadth_summary = report_payload.get("trust_breadth_summary", {})
        if not isinstance(trust_breadth_summary, dict):
            trust_breadth_summary = {}
        priority_family_yield_summary = report_payload.get("priority_family_yield_summary", {})
        if not isinstance(priority_family_yield_summary, dict):
            priority_family_yield_summary = {}
        required_families_with_reports = {
            str(value).strip() for value in trust_breadth_summary.get("required_families_with_reports", []) if str(value).strip()
        }
        external_benchmark_families = {
            str(value).strip() for value in trust_breadth_summary.get("external_benchmark_families", []) if str(value).strip()
        }
        family_yield_summaries = priority_family_yield_summary.get("family_summaries", {})
        if not isinstance(family_yield_summaries, dict):
            family_yield_summaries = {}
        for family in target_families:
            yield_summary = family_yield_summaries.get(family, {})
            if not isinstance(yield_summary, dict):
                yield_summary = {}
            observed_decisions = int(yield_summary.get("observed_decisions", 0) or 0)
            retained_positive_delta_decisions = int(yield_summary.get("retained_positive_delta_decisions", 0) or 0)
            retained_positive_pass_rate_delta_sum = float(
                yield_summary.get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0
            )
            retained_estimated_cost = float(yield_summary.get("retained_estimated_cost", 0.0) or 0.0)
            retained_return_on_cost = 0.0
            if retained_estimated_cost > 0.0:
                retained_return_on_cost = retained_positive_pass_rate_delta_sum / retained_estimated_cost
            timelines[family].append(
                {
                    "run_index": int(result.get("run_index", 0) or 0),
                    "run_match_id": str(result.get("run_match_id", "")).strip(),
                    "observed": bool(
                        family in required_families_with_reports
                        or family in external_benchmark_families
                        or observed_decisions > 0
                    ),
                    "required_report": family in required_families_with_reports,
                    "external_coverage": family in external_benchmark_families,
                    "observed_decisions": observed_decisions,
                    "retained_gain": retained_positive_delta_decisions > 0,
                    "retained_positive_delta_decisions": retained_positive_delta_decisions,
                    "retained_positive_pass_rate_delta_sum": retained_positive_pass_rate_delta_sum,
                    "retained_estimated_cost": retained_estimated_cost,
                    "retained_return_on_cost": retained_return_on_cost,
                }
            )
    families_with_repeated_observation: list[str] = []
    families_with_repeated_retained_gain: list[str] = []
    families_with_non_declining_repeated_retained_gain: list[str] = []
    families_with_declining_repeated_retained_gain: list[str] = []
    families_without_repeated_retained_gain: list[str] = []
    families_with_cost_acceptable_non_declining_repeated_retained_gain: list[str] = []
    families_with_costly_non_declining_repeated_retained_gain: list[str] = []
    families_with_declining_repeated_return_on_cost: list[str] = []
    for family in target_families:
        timeline = timelines[family]
        observed_runs = [item for item in timeline if bool(item.get("observed", False))]
        retained_gain_runs = [item for item in timeline if bool(item.get("retained_gain", False))]
        if len(observed_runs) > 1:
            families_with_repeated_observation.append(family)
        if len(retained_gain_runs) > 1:
            families_with_repeated_retained_gain.append(family)
            first_delta = float(retained_gain_runs[0].get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0)
            latest_delta = float(retained_gain_runs[-1].get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0)
            if latest_delta + 1e-12 >= first_delta:
                families_with_non_declining_repeated_retained_gain.append(family)
                first_return_on_cost = float(retained_gain_runs[0].get("retained_return_on_cost", 0.0) or 0.0)
                latest_return_on_cost = float(retained_gain_runs[-1].get("retained_return_on_cost", 0.0) or 0.0)
                if latest_return_on_cost + 1e-12 >= first_return_on_cost:
                    if latest_return_on_cost + 1e-12 >= MIN_ACCEPTABLE_TRANSFER_RETURN_ON_COST:
                        families_with_cost_acceptable_non_declining_repeated_retained_gain.append(family)
                    else:
                        families_with_costly_non_declining_repeated_retained_gain.append(family)
                else:
                    families_with_declining_repeated_return_on_cost.append(family)
            else:
                families_with_declining_repeated_retained_gain.append(family)
        else:
            families_without_repeated_retained_gain.append(family)
    return {
        "runs_checked": len(results),
        "target_non_replay_families": target_families,
        "families_with_repeated_observation": families_with_repeated_observation,
        "families_with_repeated_retained_gain": families_with_repeated_retained_gain,
        "families_with_non_declining_repeated_retained_gain": families_with_non_declining_repeated_retained_gain,
        "families_with_declining_repeated_retained_gain": families_with_declining_repeated_retained_gain,
        "families_with_cost_acceptable_non_declining_repeated_retained_gain": (
            families_with_cost_acceptable_non_declining_repeated_retained_gain
        ),
        "families_with_costly_non_declining_repeated_retained_gain": families_with_costly_non_declining_repeated_retained_gain,
        "families_with_declining_repeated_return_on_cost": families_with_declining_repeated_return_on_cost,
        "families_without_repeated_retained_gain": families_without_repeated_retained_gain,
        "minimum_acceptable_return_on_cost": MIN_ACCEPTABLE_TRANSFER_RETURN_ON_COST,
        "family_timelines": timelines,
    }


def _family_transfer_investment_ranking(results: list[dict[str, object]]) -> dict[str, object]:
    timeline = _family_transfer_timeline(results)
    target_families = list(timeline.get("target_non_replay_families", []))
    ranked_entries: list[dict[str, object]] = []
    for family in target_families:
        family_timeline = list(dict(timeline.get("family_timelines", {})).get(family, []))
        observed_runs = [item for item in family_timeline if bool(item.get("observed", False))]
        retained_gain_runs = [item for item in family_timeline if bool(item.get("retained_gain", False))]
        latest_entry = family_timeline[-1] if family_timeline else {}
        latest_return_on_cost = float(latest_entry.get("retained_return_on_cost", 0.0) or 0.0)
        average_return_on_cost = (
            _average([float(item.get("retained_return_on_cost", 0.0) or 0.0) for item in retained_gain_runs])
            if retained_gain_runs
            else 0.0
        )
        latest_pass_rate_delta = float(latest_entry.get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0)
        category = "unobserved"
        if family in timeline["families_with_cost_acceptable_non_declining_repeated_retained_gain"]:
            category = "cost_acceptable_persistent"
        elif family in timeline["families_with_costly_non_declining_repeated_retained_gain"]:
            category = "costly_persistent"
        elif family in timeline["families_with_declining_repeated_return_on_cost"] or family in timeline["families_with_declining_repeated_retained_gain"]:
            category = "declining"
        elif retained_gain_runs:
            category = "single_run_gain"
        elif observed_runs:
            category = "observed_no_gain"
        score = round(
            latest_return_on_cost
            + average_return_on_cost * 0.5
            + (0.02 if category == "cost_acceptable_persistent" else 0.0)
            + (0.01 if category == "costly_persistent" else 0.0)
            + min(0.01, 0.0025 * len(retained_gain_runs))
            + min(0.005, 0.001 * len(observed_runs)),
            6,
        )
        ranked_entries.append(
            {
                "family": family,
                "category": category,
                "investment_score": score,
                "latest_return_on_cost": latest_return_on_cost,
                "average_return_on_cost": average_return_on_cost,
                "latest_retained_pass_rate_delta": latest_pass_rate_delta,
                "observed_runs": len(observed_runs),
                "retained_gain_runs": len(retained_gain_runs),
                "cost_acceptable_persistent": family in timeline["families_with_cost_acceptable_non_declining_repeated_retained_gain"],
                "non_declining_persistent": family in timeline["families_with_non_declining_repeated_retained_gain"],
            }
        )
    ranked_entries.sort(
        key=lambda item: (
            1 if bool(item.get("cost_acceptable_persistent", False)) else 0,
            1 if bool(item.get("non_declining_persistent", False)) else 0,
            float(item.get("investment_score", 0.0) or 0.0),
            float(item.get("latest_return_on_cost", 0.0) or 0.0),
            float(item.get("average_return_on_cost", 0.0) or 0.0),
            int(item.get("retained_gain_runs", 0) or 0),
            int(item.get("observed_runs", 0) or 0),
            str(item.get("family", "")),
        ),
        reverse=True,
    )
    return {
        "runs_checked": len(results),
        "target_non_replay_families": target_families,
        "ranked_families_by_transfer_investment": [str(item.get("family", "")).strip() for item in ranked_entries],
        "family_rankings": ranked_entries,
        "top_transfer_investment_family": str(ranked_entries[0].get("family", "")).strip() if ranked_entries else "",
    }


def _frontier_expansion_summary(results: list[dict[str, object]]) -> dict[str, object]:
    target_families: list[str] = []
    pressure_families: list[str] = []
    missing_observation_priority_families: list[str] = []
    generalization_priority_families: list[str] = []
    required_sampled_family_count = 0
    sampled_run_counts_by_family: dict[str, int] = {}
    sampled_task_totals_by_family: dict[str, int] = {}
    signal_run_counts_by_family: dict[str, int] = {}
    retained_gain_run_counts_by_family: dict[str, int] = {}
    runs_with_broad_sampling: list[int] = []
    runs_missing_broad_sampling: list[int] = []
    runs_missing_pressure_sampling: list[int] = []
    runs_missing_missing_family_sampling: list[int] = []
    runs_missing_generalization_sampling: list[int] = []

    for result in results:
        run_index = int(result.get("run_index", 0) or 0)
        retention_manifest = result.get("retention_criteria_manifest", {})
        if not isinstance(retention_manifest, dict):
            retention_manifest = {}
        run_parameters = retention_manifest.get("run_parameters", {})
        if not isinstance(run_parameters, dict):
            run_parameters = {}
        target_families = _ordered_unique_strings(target_families, run_parameters.get("priority_benchmark_families", []))
        pressure = run_parameters.get("autonomous_frontier_curriculum_pressure", {})
        if not isinstance(pressure, dict):
            pressure = {}
        target_families = _ordered_unique_strings(target_families, pressure.get("target_non_replay_families", []))
        pressure_families = _ordered_unique_strings(
            pressure_families,
            pressure.get("priority_families", []),
            pressure.get("unsampled_pressure_families", []),
        )
        missing_observation_priority_families = _ordered_unique_strings(
            missing_observation_priority_families,
            pressure.get("missing_observation_families", []),
            pressure.get("unsampled_target_families", []),
        )
        generalization_priority_families = _ordered_unique_strings(
            generalization_priority_families,
            pressure.get("generalization_priority_families", []),
        )

        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        allocation_summary = report_payload.get("priority_family_allocation_summary", {})
        if not isinstance(allocation_summary, dict):
            allocation_summary = {}
        actual_counts = allocation_summary.get("aggregated_task_counts", {})
        if not isinstance(actual_counts, dict):
            actual_counts = {}
        priority_yield_summary = report_payload.get("priority_family_yield_summary", {})
        if not isinstance(priority_yield_summary, dict):
            priority_yield_summary = {}
        family_yield_summaries = priority_yield_summary.get("family_summaries", {})
        if not isinstance(family_yield_summaries, dict):
            family_yield_summaries = {}
        sampled_families: list[str] = []
        signaled_families: list[str] = []
        retained_gain_families: list[str] = []
        for family in _ordered_unique_strings(target_families):
            raw_count = actual_counts.get(family, 0)
            try:
                task_count = max(0, int(raw_count or 0))
            except (TypeError, ValueError):
                task_count = 0
            if task_count > 0:
                sampled_families.append(family)
                sampled_run_counts_by_family[family] = int(sampled_run_counts_by_family.get(family, 0)) + 1
                sampled_task_totals_by_family[family] = int(sampled_task_totals_by_family.get(family, 0)) + task_count
            family_yield = family_yield_summaries.get(family, {})
            if not isinstance(family_yield, dict):
                family_yield = {}
            observed_decisions = max(0, int(family_yield.get("observed_decisions", 0) or 0))
            retained_positive_delta_decisions = max(
                0,
                int(family_yield.get("retained_positive_delta_decisions", 0) or 0),
            )
            if observed_decisions > 0:
                signaled_families.append(family)
                signal_run_counts_by_family[family] = int(signal_run_counts_by_family.get(family, 0)) + 1
            if retained_positive_delta_decisions > 0:
                retained_gain_families.append(family)
                retained_gain_run_counts_by_family[family] = int(retained_gain_run_counts_by_family.get(family, 0)) + 1
        required_sampled_family_count = max(required_sampled_family_count, min(2, len(_ordered_unique_strings(target_families))))
        if len(sampled_families) >= required_sampled_family_count:
            runs_with_broad_sampling.append(run_index)
        else:
            runs_missing_broad_sampling.append(run_index)
        if pressure_families and not any(family in sampled_families for family in pressure_families):
            runs_missing_pressure_sampling.append(run_index)
        if missing_observation_priority_families and not any(
            family in sampled_families for family in missing_observation_priority_families
        ):
            runs_missing_missing_family_sampling.append(run_index)
        if generalization_priority_families and not any(
            family in sampled_families for family in generalization_priority_families
        ):
            runs_missing_generalization_sampling.append(run_index)

    target_families = _ordered_unique_strings(target_families, DEFAULT_NON_REPLAY_TRANSFER_FAMILIES)
    sampled_families = [
        family
        for family in target_families
        if int(sampled_run_counts_by_family.get(family, 0) or 0) > 0
    ]
    families_never_sampled = [family for family in target_families if family not in sampled_families]
    pressure_families_with_sampling = [
        family
        for family in pressure_families
        if int(sampled_run_counts_by_family.get(family, 0) or 0) > 0
    ]
    pressure_families_without_sampling = [
        family for family in pressure_families if family not in pressure_families_with_sampling
    ]
    return {
        "runs_checked": len(results),
        "target_non_replay_families": target_families,
        "required_sampled_family_count": required_sampled_family_count,
        "distinct_target_families_sampled": len(sampled_families),
        "families_sampled": sampled_families,
        "families_never_sampled": families_never_sampled,
        "sampled_run_counts_by_family": {
            family: int(sampled_run_counts_by_family.get(family, 0) or 0) for family in target_families
        },
        "sampled_task_totals_by_family": {
            family: int(sampled_task_totals_by_family.get(family, 0) or 0) for family in target_families
        },
        "signal_run_counts_by_family": {
            family: int(signal_run_counts_by_family.get(family, 0) or 0) for family in target_families
        },
        "retained_gain_run_counts_by_family": {
            family: int(retained_gain_run_counts_by_family.get(family, 0) or 0) for family in target_families
        },
        "pressure_families": pressure_families,
        "pressure_families_with_sampling": pressure_families_with_sampling,
        "pressure_families_without_sampling": pressure_families_without_sampling,
        "missing_observation_priority_families": missing_observation_priority_families,
        "generalization_priority_families": generalization_priority_families,
        "runs_with_broad_priority_sampling": runs_with_broad_sampling,
        "runs_missing_broad_priority_sampling": runs_missing_broad_sampling,
        "runs_missing_priority_pressure_sampling": runs_missing_pressure_sampling,
        "runs_missing_missing_family_sampling": runs_missing_missing_family_sampling,
        "runs_missing_generalization_sampling": runs_missing_generalization_sampling,
    }


def _required_family_clean_task_root_breadth_summary(results: list[dict[str, object]]) -> dict[str, object]:
    family_run_missing: dict[str, list[int]] = {}
    family_run_counts: dict[str, list[int]] = {}
    family_run_thresholds: dict[str, list[int]] = {}
    for result in results:
        run_index = int(result.get("run_index", 0) or 0)
        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        trust_breadth_summary = report_payload.get("trust_breadth_summary", {})
        if not isinstance(trust_breadth_summary, dict):
            trust_breadth_summary = {}
        required_counts = trust_breadth_summary.get("required_family_clean_task_root_counts", {})
        if not isinstance(required_counts, dict):
            required_counts = {}
        missing_families = [
            str(value).strip()
            for value in trust_breadth_summary.get("missing_required_family_clean_task_root_breadth", [])
            if str(value).strip()
        ]
        threshold = max(0, int(trust_breadth_summary.get("family_breadth_min_distinct_task_roots", 0) or 0))
        for family in missing_families:
            family_run_missing.setdefault(family, []).append(run_index)
            family_run_counts.setdefault(family, []).append(max(0, int(required_counts.get(family, 0) or 0)))
            family_run_thresholds.setdefault(family, []).append(threshold)
    families = sorted(family_run_missing)
    return {
        "families_missing_clean_task_root_breadth": families,
        "missing_family_run_indices": {family: family_run_missing[family] for family in families},
        "required_family_clean_task_root_counts": {
            family: min(family_run_counts.get(family, [0])) if family_run_counts.get(family) else 0
            for family in families
        },
        "family_breadth_min_distinct_task_roots": max(
            [max(values) for values in family_run_thresholds.values() if values] or [0]
        ),
    }


def _priority_family_allocation_audit(
    results: list[dict[str, object]],
    *,
    confidence_controls: dict[str, float | int] | None = None,
) -> dict[str, object]:
    resolved_confidence_controls = confidence_controls or {
        "minimum_runs": ALLOCATION_CONFIDENCE_MIN_RUNS,
        "target_priority_tasks": ALLOCATION_CONFIDENCE_TARGET_PRIORITY_TASKS,
        "target_family_tasks": 0,
        "history_window_runs": ALLOCATION_CONFIDENCE_HISTORY_WINDOW,
        "history_weight": ALLOCATION_CONFIDENCE_HISTORY_WEIGHT,
        "bonus_history_weight": 0.75,
        "normalization_history_weight": 0.25,
    }
    families: list[str] = []
    planned_share_totals: dict[str, float] = {}
    actual_share_totals: dict[str, float] = {}
    actual_task_totals: dict[str, int] = {}
    runs_with_allocation_summary = 0
    runs_with_top_planned_family_as_top_sampled = 0
    run_share_history: list[tuple[int, dict[str, float], dict[str, float]]] = []
    run_task_count_history: list[tuple[int, dict[str, int]]] = []
    latest_total_priority_tasks = 0
    latest_summary: dict[str, object] = {}
    latest_summary_run_index = 0
    latest_summary_run_match_id = ""
    for result in results:
        report_payload = result.get("report_payload", {})
        if not isinstance(report_payload, dict):
            report_payload = {}
        summary = report_payload.get("priority_family_allocation_summary", {})
        if not isinstance(summary, dict) or not summary:
            continue
        runs_with_allocation_summary += 1
        run_index = int(result.get("run_index", 0) or 0)
        if run_index >= latest_summary_run_index:
            latest_summary_run_index = run_index
            latest_summary_run_match_id = str(result.get("run_match_id", "")).strip()
            latest_summary = dict(summary)
        if str(summary.get("top_planned_family", "")).strip() == str(summary.get("top_sampled_family", "")).strip():
            runs_with_top_planned_family_as_top_sampled += 1
        for family in summary.get("priority_families", []):
            token = str(family).strip()
            if token and token not in families:
                families.append(token)
        planned_weight_shares = summary.get("planned_weight_shares", {})
        if not isinstance(planned_weight_shares, dict):
            planned_weight_shares = {}
        aggregated_task_shares = summary.get("aggregated_task_shares", {})
        if not isinstance(aggregated_task_shares, dict):
            aggregated_task_shares = {}
        aggregated_task_counts = summary.get("aggregated_task_counts", {})
        if not isinstance(aggregated_task_counts, dict):
            aggregated_task_counts = {}
        run_share_history.append(
            (
                run_index,
                {str(family).strip(): float(planned_weight_shares.get(family, 0.0) or 0.0) for family in families},
                {str(family).strip(): float(aggregated_task_shares.get(family, 0.0) or 0.0) for family in families},
            )
        )
        run_task_count_history.append(
            (
                run_index,
                {str(family).strip(): int(aggregated_task_counts.get(family, 0) or 0) for family in families},
            )
        )
        for family in families:
            planned_share_totals[family] = planned_share_totals.get(family, 0.0) + float(
                planned_weight_shares.get(family, 0.0) or 0.0
            )
            actual_share_totals[family] = actual_share_totals.get(family, 0.0) + float(
                aggregated_task_shares.get(family, 0.0) or 0.0
            )
            actual_task_totals[family] = actual_task_totals.get(family, 0) + int(
                aggregated_task_counts.get(family, 0) or 0
            )
    latest_planned_shares_raw = latest_summary.get("planned_weight_shares", {})
    latest_actual_shares_raw = latest_summary.get("aggregated_task_shares", {})
    latest_total_priority_tasks = int(latest_summary.get("total_priority_tasks", 0) or 0)
    latest_aggregated_task_counts_raw = latest_summary.get("aggregated_task_counts", {})
    if not isinstance(latest_planned_shares_raw, dict):
        latest_planned_shares_raw = {}
    if not isinstance(latest_actual_shares_raw, dict):
        latest_actual_shares_raw = {}
    if not isinstance(latest_aggregated_task_counts_raw, dict):
        latest_aggregated_task_counts_raw = {}
    if latest_total_priority_tasks <= 0:
        latest_total_priority_tasks = sum(int(latest_aggregated_task_counts_raw.get(family, 0) or 0) for family in families)
    minimum_runs = max(1, int(resolved_confidence_controls.get("minimum_runs", ALLOCATION_CONFIDENCE_MIN_RUNS) or 0))
    target_priority_tasks = max(
        1,
        int(resolved_confidence_controls.get("target_priority_tasks", ALLOCATION_CONFIDENCE_TARGET_PRIORITY_TASKS) or 0),
    )
    target_family_tasks_override = max(0, int(resolved_confidence_controls.get("target_family_tasks", 0) or 0))
    history_window_runs = max(
        1,
        int(resolved_confidence_controls.get("history_window_runs", ALLOCATION_CONFIDENCE_HISTORY_WINDOW) or 0),
    )
    history_weight = max(
        0.0,
        min(1.0, float(resolved_confidence_controls.get("history_weight", ALLOCATION_CONFIDENCE_HISTORY_WEIGHT) or 0.0)),
    )
    target_family_tasks = max(
        1,
        target_family_tasks_override
        or (target_priority_tasks + max(1, len(families)) - 1) // max(1, len(families)),
    )
    average_planned_shares = {
        family: 0.0 if runs_with_allocation_summary <= 0 else round(planned_share_totals.get(family, 0.0) / runs_with_allocation_summary, 6)
        for family in families
    }
    average_actual_shares = {
        family: 0.0 if runs_with_allocation_summary <= 0 else round(actual_share_totals.get(family, 0.0) / runs_with_allocation_summary, 6)
        for family in families
    }
    latest_planned_shares = {
        family: round(float(latest_planned_shares_raw.get(family, 0.0) or 0.0), 6)
        for family in families
    }
    latest_actual_shares = {
        family: round(float(latest_actual_shares_raw.get(family, 0.0) or 0.0), 6)
        for family in families
    }
    average_share_gap_by_family = {
        family: round(max(0.0, average_planned_shares[family] - average_actual_shares[family]), 6)
        for family in families
        if max(0.0, average_planned_shares[family] - average_actual_shares[family]) > 0.0
    }
    latest_share_gap_by_family = {
        family: round(max(0.0, latest_planned_shares[family] - latest_actual_shares[family]), 6)
        for family in families
        if max(0.0, latest_planned_shares[family] - latest_actual_shares[family]) > 0.0
    }
    latest_oversampled_share_gap_by_family = {
        family: round(max(0.0, latest_actual_shares[family] - latest_planned_shares[family]), 6)
        for family in families
        if max(0.0, latest_actual_shares[family] - latest_planned_shares[family]) > 0.0
    }
    latest_positive_gap_streak_by_family: dict[str, int] = {}
    latest_oversampled_streak_by_family: dict[str, int] = {}
    for family in families:
        positive_streak = 0
        oversampled_streak = 0
        for _, planned_map, actual_map in sorted(run_share_history, key=lambda item: item[0], reverse=True):
            positive_gap = max(0.0, float(planned_map.get(family, 0.0) or 0.0) - float(actual_map.get(family, 0.0) or 0.0))
            oversampled_gap = max(0.0, float(actual_map.get(family, 0.0) or 0.0) - float(planned_map.get(family, 0.0) or 0.0))
            if positive_gap >= MATERIAL_ALLOCATION_SHARE_GAP and oversampled_streak == 0:
                positive_streak += 1
            elif positive_streak > 0:
                break
            if oversampled_gap >= MATERIAL_ALLOCATION_SHARE_GAP and positive_streak == 0:
                oversampled_streak += 1
            elif oversampled_streak > 0:
                break
        if positive_streak > 0:
            latest_positive_gap_streak_by_family[family] = positive_streak
        if oversampled_streak > 0:
            latest_oversampled_streak_by_family[family] = oversampled_streak
    run_confidence = min(1.0, runs_with_allocation_summary / float(minimum_runs))
    task_confidence = min(1.0, latest_total_priority_tasks / float(target_priority_tasks))
    allocation_confidence = round(min(run_confidence, task_confidence), 6)
    latest_task_counts_by_family = {
        family: int(latest_aggregated_task_counts_raw.get(family, 0) or 0)
        for family in families
    }
    latest_task_confidence_by_family = {
        family: round(min(1.0, latest_task_counts_by_family[family] / float(target_family_tasks)), 6)
        for family in families
    }
    recent_count_history = [
        counts
        for _, counts in sorted(run_task_count_history, key=lambda item: item[0], reverse=True)[:history_window_runs]
    ]
    recent_average_task_confidence_by_family = {
        family: round(
            _average(
                [
                    min(1.0, int(counts.get(family, 0) or 0) / float(target_family_tasks))
                    for counts in recent_count_history
                ]
            ),
            6,
        )
        for family in families
    }
    allocation_confidence_by_family = {
        family: round(
            min(
                allocation_confidence,
                latest_task_confidence_by_family[family] * (1.0 - history_weight)
                + recent_average_task_confidence_by_family[family] * history_weight,
            ),
            6,
        )
        for family in families
    }
    bonus_history_weight = max(
        0.0,
        min(1.0, float(resolved_confidence_controls.get("bonus_history_weight", 0.75) or 0.0)),
    )
    bonus_allocation_confidence_by_family = {
        family: round(
            min(
                allocation_confidence,
                latest_task_confidence_by_family[family] * (1.0 - bonus_history_weight)
                + recent_average_task_confidence_by_family[family] * bonus_history_weight,
            ),
            6,
        )
        for family in families
    }
    normalization_history_weight = max(
        0.0,
        min(1.0, float(resolved_confidence_controls.get("normalization_history_weight", 0.25) or 0.0)),
    )
    normalization_allocation_confidence_by_family = {
        family: round(
            min(
                allocation_confidence,
                latest_task_confidence_by_family[family] * (1.0 - normalization_history_weight)
                + recent_average_task_confidence_by_family[family] * normalization_history_weight,
            ),
            6,
        )
        for family in families
    }
    return {
        "runs_checked": len(results),
        "runs_with_allocation_summary": runs_with_allocation_summary,
        "runs_with_top_planned_family_as_top_sampled": runs_with_top_planned_family_as_top_sampled,
        "priority_families": families,
        "average_planned_shares": average_planned_shares,
        "average_actual_shares": average_actual_shares,
        "average_share_gap_by_family": average_share_gap_by_family,
        "latest_summary_run_index": latest_summary_run_index,
        "latest_summary_run_match_id": latest_summary_run_match_id,
        "latest_total_priority_tasks": latest_total_priority_tasks,
        "latest_task_counts_by_family": latest_task_counts_by_family,
        "allocation_confidence": allocation_confidence,
        "allocation_confidence_by_family": allocation_confidence_by_family,
        "bonus_allocation_confidence_by_family": bonus_allocation_confidence_by_family,
        "normalization_allocation_confidence_by_family": normalization_allocation_confidence_by_family,
        "latest_task_confidence_by_family": latest_task_confidence_by_family,
        "recent_average_task_confidence_by_family": recent_average_task_confidence_by_family,
        "allocation_confidence_components": {
            "run_confidence": round(run_confidence, 6),
            "task_confidence": round(task_confidence, 6),
            "minimum_runs": minimum_runs,
            "target_priority_tasks": target_priority_tasks,
            "target_family_tasks": target_family_tasks,
            "target_family_tasks_override": target_family_tasks_override,
            "history_window_runs": history_window_runs,
            "history_weight": round(history_weight, 6),
            "bonus_history_weight": round(bonus_history_weight, 6),
            "normalization_history_weight": round(normalization_history_weight, 6),
        },
        "latest_planned_shares": latest_planned_shares,
        "latest_actual_shares": latest_actual_shares,
        "latest_share_gap_by_family": latest_share_gap_by_family,
        "latest_positive_gap_streak_by_family": latest_positive_gap_streak_by_family,
        "latest_oversampled_share_gap_by_family": latest_oversampled_share_gap_by_family,
        "latest_oversampled_streak_by_family": latest_oversampled_streak_by_family,
        "latest_positive_gap_families": [
            family for family in families if latest_share_gap_by_family.get(family, 0.0) >= MATERIAL_ALLOCATION_SHARE_GAP
        ],
        "latest_oversampled_families": [
            family
            for family in families
            if latest_oversampled_share_gap_by_family.get(family, 0.0) >= MATERIAL_ALLOCATION_SHARE_GAP
        ],
        "recovered_gap_families": [
            family
            for family in families
            if average_share_gap_by_family.get(family, 0.0) >= MATERIAL_ALLOCATION_SHARE_GAP
            and latest_share_gap_by_family.get(family, 0.0) < MATERIAL_ALLOCATION_SHARE_GAP
        ],
        "actual_task_totals": actual_task_totals,
        "top_planned_family": (
            max(families, key=lambda family: (average_planned_shares[family], family))
            if families
            else ""
        ),
        "top_sampled_family": (
            max(families, key=lambda family: (actual_task_totals[family], average_actual_shares[family], family))
            if families
            else ""
        ),
    }


def _claim_gate_summary(
    results: list[dict[str, object]],
    *,
    confidence_controls: dict[str, float] | None = None,
) -> dict[str, object]:
    runs = len(results)
    successful_runs = sum(1 for result in results if int(result.get("returncode", 1)) == 0)
    retained_cycles = [
        float(dict(result.get("production_yield_summary", {})).get("retained_cycles", 0.0))
        for result in results
    ]
    rejected_cycles = [
        float(dict(result.get("production_yield_summary", {})).get("rejected_cycles", 0.0))
        for result in results
    ]
    retained_pass_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("average_retained_pass_rate_delta", 0.0))
        for result in results
    ]
    retained_step_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("average_retained_step_delta", 0.0))
        for result in results
    ]
    retained_phase_gates = [
        bool(dict(result.get("phase_gate_summary", {})).get("all_retained_phase_gates_passed", True))
        for result in results
    ]
    worst_family_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("worst_family_delta", 0.0))
        for result in results
    ]
    worst_generated_family_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("worst_generated_family_delta", 0.0))
        for result in results
    ]
    worst_failure_recovery_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("worst_failure_recovery_delta", 0.0))
        for result in results
    ]
    retained_estimated_costs = [
        float(dict(result.get("production_yield_summary", {})).get("average_retained_estimated_cost", 0.0))
        for result in results
    ]
    runtime_managed_decisions = [
        int(dict(dict(result.get("report_payload", {})).get("inheritance_summary", {})).get("runtime_managed_decisions", 0))
        for result in results
    ]
    seed_fingerprints = [str(result.get("seed_fingerprint", "")).strip() for result in results]
    criteria_fingerprints = [str(result.get("retention_criteria_fingerprint", "")).strip() for result in results]
    result_stream_audit = _result_stream_audit(results)
    family_transfer_summary = _family_transfer_summary(results)
    family_transfer_timeline = _family_transfer_timeline(results)
    family_transfer_investment_ranking = _family_transfer_investment_ranking(results)
    frontier_expansion_summary = _frontier_expansion_summary(results)
    required_family_clean_task_root_breadth = _required_family_clean_task_root_breadth_summary(results)
    retrieval_carryover_summary = _retrieval_carryover_summary(results)
    priority_family_allocation_audit = _priority_family_allocation_audit(
        results,
        confidence_controls=confidence_controls,
    )
    blockers: list[str] = []
    if runs < 2:
        blockers.append("requires_at_least_two_isolated_runs")
    if successful_runs != runs:
        blockers.append("one_or_more_runs_failed")
    if any(value <= 0.0 for value in retained_cycles):
        blockers.append("one_or_more_runs_retained_no_runtime_managed_gain")
    if any(value > 0.0 for value in rejected_cycles):
        blockers.append("one_or_more_runs_rejected_runtime_managed_candidates")
    if any(value < 0.0 for value in retained_pass_deltas):
        blockers.append("one_or_more_runs_regressed_retained_pass_rate")
    if any(value > 0.0 for value in retained_step_deltas):
        blockers.append("one_or_more_runs_increased_retained_average_steps")
    if not all(retained_phase_gates):
        blockers.append("one_or_more_runs_failed_retained_phase_gates")
    if any(value < 0.0 for value in worst_family_deltas):
        blockers.append("one_or_more_runs_showed_cross_family_regression")
    if any(value < 0.0 for value in worst_generated_family_deltas):
        blockers.append("one_or_more_runs_showed_generated_family_regression")
    if any(value < 0.0 for value in worst_failure_recovery_deltas):
        blockers.append("one_or_more_runs_regressed_failure_recovery")
    if any(value <= 0 for value in runtime_managed_decisions):
        blockers.append("one_or_more_runs_lacked_runtime_managed_decisions")
    unique_seed_fingerprints = {value for value in seed_fingerprints if value}
    unique_criteria_fingerprints = {value for value in criteria_fingerprints if value}
    if len(unique_seed_fingerprints) != max(1, min(runs, 1)):
        blockers.append("isolated_runs_did_not_share_the_same_seeded_starting_state")
    if len(unique_criteria_fingerprints) != max(1, min(runs, 1)):
        blockers.append("isolated_runs_did_not_share_stable_retention_criteria")
    if result_stream_audit["missing_scoped_campaign_record_stream_runs"]:
        blockers.append("one_or_more_runs_missing_scoped_campaign_record_stream")
    if result_stream_audit["missing_runtime_managed_result_stream_runs"]:
        blockers.append("one_or_more_runs_missing_runtime_managed_result_stream")
    required_transfer_family_count = min(2, len(family_transfer_summary["target_non_replay_families"]))
    if family_transfer_summary["distinct_target_families_observed"] < required_transfer_family_count:
        blockers.append("non_replay_transfer_family_observation_too_narrow")
    if family_transfer_summary["distinct_target_families_with_retained_gain"] < required_transfer_family_count:
        blockers.append("non_replay_transfer_retained_gain_too_narrow")
    if frontier_expansion_summary["distinct_target_families_sampled"] < required_transfer_family_count:
        blockers.append("autonomous_frontier_sampling_too_narrow")
    if frontier_expansion_summary["pressure_families_without_sampling"]:
        blockers.append("autonomous_frontier_priority_pressure_not_exercised")
    if not family_transfer_timeline["families_with_non_declining_repeated_retained_gain"]:
        blockers.append("non_replay_transfer_retained_gain_not_persistent_over_time")
    elif not family_transfer_timeline["families_with_cost_acceptable_non_declining_repeated_retained_gain"]:
        blockers.append("non_replay_transfer_return_on_cost_too_low")
    if required_family_clean_task_root_breadth["families_missing_clean_task_root_breadth"]:
        blockers.append("required_family_clean_task_root_breadth_too_narrow")
    if bool(retrieval_carryover_summary.get("carryover_pressure", False)):
        blockers.append("trusted_retrieval_carryover_not_proven")
    retained_cycle_spread = 0.0 if not retained_cycles else max(retained_cycles) - min(retained_cycles)
    retained_pass_rate_delta_spread = (
        0.0 if not retained_pass_deltas else max(retained_pass_deltas) - min(retained_pass_deltas)
    )
    return {
        "runs_checked": runs,
        "successful_runs": successful_runs,
        "min_required_runs": 2,
        "retained_cycle_spread": retained_cycle_spread,
        "retained_pass_rate_delta_spread": retained_pass_rate_delta_spread,
        "min_runtime_managed_decisions": 0 if not runtime_managed_decisions else min(runtime_managed_decisions),
        "worst_family_delta": 0.0 if not worst_family_deltas else min(worst_family_deltas),
        "worst_generated_family_delta": 0.0 if not worst_generated_family_deltas else min(worst_generated_family_deltas),
        "worst_failure_recovery_delta": 0.0 if not worst_failure_recovery_deltas else min(worst_failure_recovery_deltas),
        "average_retained_estimated_cost": _average(retained_estimated_costs),
        "seed_fingerprint": next(iter(unique_seed_fingerprints), ""),
        "retention_criteria_fingerprint": next(iter(unique_criteria_fingerprints), ""),
        "starting_state_consistent": len(unique_seed_fingerprints) == 1 if runs else True,
        "retention_criteria_stable": len(unique_criteria_fingerprints) == 1 if runs else True,
        "result_stream_audit": result_stream_audit,
        "family_transfer_summary": family_transfer_summary,
        "family_transfer_timeline": family_transfer_timeline,
        "family_transfer_investment_ranking": family_transfer_investment_ranking,
        "frontier_expansion_summary": frontier_expansion_summary,
        "required_family_clean_task_root_breadth": required_family_clean_task_root_breadth,
        "retrieval_carryover_summary": retrieval_carryover_summary,
        "priority_family_allocation_audit": priority_family_allocation_audit,
        "autonomous_compounding_claim_ready": not blockers,
        "blockers": blockers,
    }


def _summary(
    results: list[dict[str, object]],
    *,
    confidence_controls: dict[str, float] | None = None,
) -> dict[str, object]:
    retained_cycles = [
        float(dict(result.get("production_yield_summary", {})).get("retained_cycles", 0.0))
        for result in results
    ]
    rejected_cycles = [
        float(dict(result.get("production_yield_summary", {})).get("rejected_cycles", 0.0))
        for result in results
    ]
    retained_pass_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("average_retained_pass_rate_delta", 0.0))
        for result in results
    ]
    retained_step_deltas = [
        float(dict(result.get("production_yield_summary", {})).get("average_retained_step_delta", 0.0))
        for result in results
    ]
    retained_phase_gates = [
        bool(dict(result.get("phase_gate_summary", {})).get("all_retained_phase_gates_passed", True))
        for result in results
    ]
    successful_runs = sum(1 for result in results if int(result.get("returncode", 1)) == 0)
    claim_gate = _claim_gate_summary(results, confidence_controls=confidence_controls)
    return {
        "runs": len(results),
        "successful_runs": successful_runs,
        "runs_with_retention": sum(1 for value in retained_cycles if value > 0.0),
        "runs_without_rejection": sum(1 for value in rejected_cycles if value <= 0.0),
        "average_retained_cycles": _average(retained_cycles),
        "min_retained_cycles": 0.0 if not retained_cycles else min(retained_cycles),
        "max_retained_cycles": 0.0 if not retained_cycles else max(retained_cycles),
        "average_rejected_cycles": _average(rejected_cycles),
        "average_retained_pass_rate_delta": _average(retained_pass_deltas),
        "min_retained_pass_rate_delta": 0.0 if not retained_pass_deltas else min(retained_pass_deltas),
        "max_retained_pass_rate_delta": 0.0 if not retained_pass_deltas else max(retained_pass_deltas),
        "average_retained_step_delta": _average(retained_step_deltas),
        "min_retained_step_delta": 0.0 if not retained_step_deltas else min(retained_step_deltas),
        "max_retained_step_delta": 0.0 if not retained_step_deltas else max(retained_step_deltas),
        "retained_cycle_spread": 0.0 if not retained_cycles else max(retained_cycles) - min(retained_cycles),
        "retained_pass_rate_delta_spread": 0.0 if not retained_pass_deltas else max(retained_pass_deltas) - min(retained_pass_deltas),
        "runs_with_retained_phase_gate_failures": sum(1 for passed in retained_phase_gates if not passed),
        "autonomous_compounding_viable": bool(claim_gate["autonomous_compounding_claim_ready"]),
        "claim_gate_summary": claim_gate,
    }


def _write_status(
    *,
    config: KernelConfig,
    args: argparse.Namespace,
    started_at: str,
    state: str,
    task_limit: int,
    task_limit_source: str,
    priority_benchmark_families: list[str],
    priority_benchmark_family_source: str,
    priority_benchmark_family_weights: dict[str, float],
    priority_benchmark_family_weight_source: str,
    priority_benchmark_family_allocation_compensation: dict[str, object],
    results: list[dict[str, object]],
    confidence_controls: dict[str, float] | None = None,
    active_run: dict[str, object] | None = None,
    final_report_path: Path | None = None,
) -> Path:
    effective_active_run, prior_payload = _effective_active_run_snapshot(
        config=config,
        active_run=active_run,
    )
    summary = _summary(results, confidence_controls=confidence_controls) if results else {}
    claim_gate_summary = summary.get("claim_gate_summary", {}) if isinstance(summary, dict) else {}
    if not isinstance(claim_gate_summary, dict):
        claim_gate_summary = {}
    frontier_expansion_summary = claim_gate_summary.get("frontier_expansion_summary", {})
    if not isinstance(frontier_expansion_summary, dict):
        frontier_expansion_summary = {}
    retained_child_snapshot = _retained_child_sampling_snapshot(
        prior_payload=prior_payload,
        active_run=effective_active_run,
        requested_priority_benchmark_families=priority_benchmark_families,
    )
    families_sampled = _string_list(frontier_expansion_summary, "families_sampled")
    if "families_sampled" not in frontier_expansion_summary:
        families_sampled = retained_child_snapshot["families_sampled"]
    families_never_sampled = _string_list(frontier_expansion_summary, "families_never_sampled")
    if "families_never_sampled" not in frontier_expansion_summary:
        families_never_sampled = retained_child_snapshot["families_never_sampled"]
    pressure_families_without_sampling = _string_list(frontier_expansion_summary, "pressure_families_without_sampling")
    if "pressure_families_without_sampling" not in frontier_expansion_summary:
        pressure_families_without_sampling = retained_child_snapshot["pressure_families_without_sampling"]
    family_transfer_summary = claim_gate_summary.get("family_transfer_summary", {})
    if not isinstance(family_transfer_summary, dict):
        family_transfer_summary = {}
    retrieval_carryover_summary = claim_gate_summary.get("retrieval_carryover_summary", {})
    if not isinstance(retrieval_carryover_summary, dict):
        retrieval_carryover_summary = {}
    latest_completed_run = results[-1] if results else {}
    completed_run_summaries = [
        {
            "run_index": int(result.get("run_index", 0) or 0),
            "run_match_id": str(result.get("run_match_id", "")).strip(),
            "returncode": int(result.get("returncode", 0) or 0),
            "report_path": str(result.get("report_path", "")).strip(),
            "task_limit": int(result.get("task_limit", 0) or 0),
            "task_limit_source": str(result.get("task_limit_source", "")).strip(),
            "priority_benchmark_families": list(result.get("priority_benchmark_families", []))
            if isinstance(result.get("priority_benchmark_families", []), list)
            else [],
        }
        for result in results
    ]
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_compounding_status",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "state": state,
        "runs_requested": max(1, args.runs),
        "runs_completed": len(results),
        "runs_remaining": max(0, max(1, args.runs) - len(results)),
        "cycles_per_run": max(1, args.cycles),
        "requested_task_limit": task_limit,
        "requested_task_limit_source": task_limit_source,
        "requested_priority_benchmark_families": priority_benchmark_families,
        "priority_benchmark_family_source": priority_benchmark_family_source,
        "requested_priority_benchmark_family_weights": priority_benchmark_family_weights,
        "requested_priority_benchmark_family_weight_source": priority_benchmark_family_weight_source,
        "priority_benchmark_family_allocation_compensation": priority_benchmark_family_allocation_compensation,
        "active_run": effective_active_run,
        "latest_completed_run": (
            {
                "run_index": int(latest_completed_run.get("run_index", 0) or 0),
                "run_match_id": str(latest_completed_run.get("run_match_id", "")).strip(),
                "report_path": str(latest_completed_run.get("report_path", "")).strip(),
                "returncode": int(latest_completed_run.get("returncode", 0) or 0),
            }
            if latest_completed_run
            else {}
        ),
        "completed_runs": completed_run_summaries,
        "partial_summary": summary,
        "partial_frontier_expansion_summary": frontier_expansion_summary,
        "partial_family_transfer_summary": family_transfer_summary,
        "partial_retrieval_carryover_summary": retrieval_carryover_summary,
        "pressure_families_without_sampling": pressure_families_without_sampling,
        "families_sampled": families_sampled,
        "families_never_sampled": families_never_sampled,
        "partial_blockers": list(claim_gate_summary.get("blockers", []))
        if isinstance(claim_gate_summary.get("blockers", []), list)
        else [],
        "final_report_path": str(final_report_path) if final_report_path is not None else "",
    }
    status_path = _status_path(config)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return status_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--campaign-width", type=int, default=2)
    parser.add_argument("--variant-width", type=int, default=2)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()
    allocation_confidence_controls = _priority_family_allocation_confidence_controls(config)
    task_limit, task_limit_source = _resolved_task_limit(config, args)
    (
        priority_benchmark_families,
        priority_benchmark_family_source,
        priority_benchmark_family_weights,
        priority_benchmark_family_weight_source,
        priority_benchmark_family_allocation_compensation,
    ) = _priority_benchmark_family_plan(config, args)

    repo_root = Path(__file__).resolve().parents[1]
    results: list[dict[str, object]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    active_run_status: dict[str, object] = {}
    interrupted_signal_name = ""

    def _handle_interrupt(signum: int, _frame: object) -> None:
        nonlocal interrupted_signal_name
        try:
            interrupted_signal_name = signal.Signals(signum).name
        except ValueError:
            interrupted_signal_name = str(signum)
        raise KeyboardInterrupt

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)
    _write_status(
        config=config,
        args=args,
        started_at=started_at,
        state="starting",
        task_limit=task_limit,
        task_limit_source=task_limit_source,
        priority_benchmark_families=priority_benchmark_families,
        priority_benchmark_family_source=priority_benchmark_family_source,
        priority_benchmark_family_weights=priority_benchmark_family_weights,
        priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
        priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
        results=results,
        confidence_controls=allocation_confidence_controls,
    )
    try:
        for index in range(1, max(1, args.runs) + 1):
            run_match_id = _match_id(index)
            prior_claim_gate_summary = _latest_prior_compounding_claim_gate_summary(config)
            autonomous_frontier_curriculum_pressure = _autonomous_frontier_curriculum_pressure(
                priority_benchmark_families=priority_benchmark_families,
                priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
                prior_claim_gate_summary=prior_claim_gate_summary,
            )
            (
                routed_task_limit,
                routed_task_limit_source,
                routed_priority_benchmark_families,
                routed_priority_benchmark_family_weights,
                routed_priority_benchmark_family_weight_source,
                priority_benchmark_family_live_routing,
            ) = _autonomous_frontier_live_priority_routing(
                task_limit=task_limit,
                task_limit_source=task_limit_source,
                priority_benchmark_families=priority_benchmark_families,
                priority_benchmark_family_weights=priority_benchmark_family_weights,
                priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
                autonomous_frontier_curriculum_pressure=autonomous_frontier_curriculum_pressure,
            )
            active_run_status = {
                "run_index": index,
                "run_match_id": run_match_id,
                "runtime_root": str(_run_root(config, run_match_id=run_match_id)),
                "child_status_path": str(_child_run_status_path(_run_root(config, run_match_id=run_match_id))),
                "task_limit": routed_task_limit,
                "task_limit_source": routed_task_limit_source,
                "priority_benchmark_families": routed_priority_benchmark_families,
                "priority_benchmark_family_weights": routed_priority_benchmark_family_weights,
                "priority_benchmark_family_weight_source": routed_priority_benchmark_family_weight_source,
                "priority_benchmark_family_live_routing": priority_benchmark_family_live_routing,
                "autonomous_frontier_curriculum_pressure": autonomous_frontier_curriculum_pressure,
            }
            _write_status(
                config=config,
                args=args,
                started_at=started_at,
                state="running",
                task_limit=task_limit,
                task_limit_source=task_limit_source,
                priority_benchmark_families=priority_benchmark_families,
                priority_benchmark_family_source=priority_benchmark_family_source,
                priority_benchmark_family_weights=priority_benchmark_family_weights,
                priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
                priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
                results=results,
                confidence_controls=allocation_confidence_controls,
                active_run=active_run_status,
            )
            results.append(
                    _run_once(
                        repo_root=repo_root,
                        config=config,
                        args=args,
                        run_match_id=run_match_id,
                        run_index=index,
                        priority_benchmark_family_source=priority_benchmark_family_source,
                        priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
                        priority_benchmark_family_weight_source=routed_priority_benchmark_family_weight_source,
                        autonomous_frontier_curriculum_pressure=autonomous_frontier_curriculum_pressure,
                        routed_task_limit=routed_task_limit,
                        routed_task_limit_source=routed_task_limit_source,
                        routed_priority_benchmark_families=routed_priority_benchmark_families,
                        routed_priority_benchmark_family_weights=routed_priority_benchmark_family_weights,
                        routed_priority_benchmark_family_weight_source=routed_priority_benchmark_family_weight_source,
                        priority_benchmark_family_live_routing=priority_benchmark_family_live_routing,
                    )
                )
            active_run_status = {}
            _write_status(
                config=config,
                args=args,
                started_at=started_at,
                state="running" if index < max(1, args.runs) else "finalizing",
                task_limit=task_limit,
                task_limit_source=task_limit_source,
                priority_benchmark_families=priority_benchmark_families,
                priority_benchmark_family_source=priority_benchmark_family_source,
                priority_benchmark_family_weights=priority_benchmark_family_weights,
                priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
                priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
                results=results,
                confidence_controls=allocation_confidence_controls,
            )
    except KeyboardInterrupt:
        interrupted_active_run = dict(active_run_status)
        if interrupted_signal_name:
            interrupted_active_run["interrupted_signal"] = interrupted_signal_name
        _write_status(
            config=config,
            args=args,
            started_at=started_at,
            state="aborted",
            task_limit=task_limit,
            task_limit_source=task_limit_source,
            priority_benchmark_families=priority_benchmark_families,
            priority_benchmark_family_source=priority_benchmark_family_source,
            priority_benchmark_family_weights=priority_benchmark_family_weights,
            priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
            priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
            results=results,
            confidence_controls=allocation_confidence_controls,
            active_run=interrupted_active_run,
        )
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    summary = _summary(results, confidence_controls=allocation_confidence_controls)
    report = {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_compounding_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_requested": max(1, args.runs),
        "cycles_per_run": max(1, args.cycles),
        "task_limit": results[0].get("task_limit", task_limit) if results else task_limit,
        "task_limit_source": results[0].get("task_limit_source", task_limit_source) if results else task_limit_source,
        "requested_task_limit": task_limit,
        "requested_task_limit_source": task_limit_source,
        "priority_benchmark_families": (
            results[0].get("priority_benchmark_families", priority_benchmark_families) if results else priority_benchmark_families
        ),
        "priority_benchmark_family_source": priority_benchmark_family_source,
        "priority_benchmark_family_weights": (
            results[0].get("priority_benchmark_family_weights", priority_benchmark_family_weights)
            if results
            else priority_benchmark_family_weights
        ),
        "priority_benchmark_family_weight_source": (
            results[0].get("priority_benchmark_family_weight_source", priority_benchmark_family_weight_source)
            if results
            else priority_benchmark_family_weight_source
        ),
        "priority_benchmark_family_allocation_compensation": priority_benchmark_family_allocation_compensation,
        "priority_benchmark_family_live_routing": (
            results[0].get("priority_benchmark_family_live_routing", {}) if results else {}
        ),
        "autonomous_frontier_curriculum_pressure": (
            results[0].get("autonomous_frontier_curriculum_pressure", {}) if results else {}
        ),
        "summary": summary,
        "runs": results,
    }
    report_path = config.improvement_reports_dir / (
        f"autonomous_compounding_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_status(
        config=config,
        args=args,
        started_at=started_at,
        state="finished",
        task_limit=task_limit,
        task_limit_source=task_limit_source,
        priority_benchmark_families=priority_benchmark_families,
        priority_benchmark_family_source=priority_benchmark_family_source,
        priority_benchmark_family_weights=priority_benchmark_family_weights,
        priority_benchmark_family_weight_source=priority_benchmark_family_weight_source,
        priority_benchmark_family_allocation_compensation=priority_benchmark_family_allocation_compensation,
        results=results,
        confidence_controls=allocation_confidence_controls,
        final_report_path=report_path,
    )

    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=f"autonomous_compounding:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            state="record",
            subsystem="autonomous_compounding",
            action="summarize_isolated_autonomous_campaigns",
            artifact_path=str(report_path),
            artifact_kind="autonomous_compounding_report",
            reason="summarize repeated autonomous improvement campaigns from isolated starting states",
            metrics_summary={
                "runs_requested": max(1, args.runs),
                "cycles_per_run": max(1, args.cycles),
                "task_limit": results[0].get("task_limit", task_limit) if results else task_limit,
                "task_limit_source": results[0].get("task_limit_source", task_limit_source) if results else task_limit_source,
                "requested_task_limit": task_limit,
                "requested_task_limit_source": task_limit_source,
                "priority_benchmark_families": (
                    results[0].get("priority_benchmark_families", priority_benchmark_families)
                    if results
                    else priority_benchmark_families
                ),
                "priority_benchmark_family_source": priority_benchmark_family_source,
                "priority_benchmark_family_weights": (
                    results[0].get("priority_benchmark_family_weights", priority_benchmark_family_weights)
                    if results
                    else priority_benchmark_family_weights
                ),
                "priority_benchmark_family_weight_source": (
                    results[0].get("priority_benchmark_family_weight_source", priority_benchmark_family_weight_source)
                    if results
                    else priority_benchmark_family_weight_source
                ),
                "priority_benchmark_family_allocation_compensation": priority_benchmark_family_allocation_compensation,
                "priority_benchmark_family_live_routing": (
                    results[0].get("priority_benchmark_family_live_routing", {}) if results else {}
                ),
                "autonomous_frontier_curriculum_pressure": (
                    results[0].get("autonomous_frontier_curriculum_pressure", {}) if results else {}
                ),
                "successful_runs": summary["successful_runs"],
                "runs_with_retention": summary["runs_with_retention"],
                "autonomous_compounding_viable": summary["autonomous_compounding_viable"],
                "retained_cycle_spread": summary["retained_cycle_spread"],
                "retained_pass_rate_delta_spread": summary["retained_pass_rate_delta_spread"],
            },
        ),
    )
    print(report_path)


if __name__ == "__main__":
    main()
