from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping
import argparse
import fcntl
import json
import math
import os
import re
import selectors
import shutil
import socket
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.cycle_runner import semantic_progress_state
from agent_kernel.curriculum_improvement import retained_curriculum_controls
from agent_kernel.runtime_supervision import (
    append_jsonl,
    atomic_write_json,
    install_termination_handlers,
    spawn_process_group,
    terminate_process_tree,
)
from agent_kernel.semantic_hub import (
    record_semantic_attempt,
    record_semantic_note,
    record_semantic_redirect,
    record_semantic_skill,
    upsert_semantic_agent,
)
from agent_kernel.strategy_memory import load_strategy_nodes
from agent_kernel.trust import build_unattended_trust_ledger
from agent_kernel.prompt_improvement import retained_improvement_planner_controls
from agent_kernel.unattended_controller import (
    action_key_for_policy,
    build_failure_observation,
    build_round_observation,
    controller_state_summary,
    default_controller_state,
    normalize_controller_state,
    plan_next_policy,
    update_controller_state,
)


_CAMPAIGN_LOCK_STATE: dict[str, object] = {
    "fd": None,
    "path": None,
}
_EXTERNAL_LEASE_STATE: dict[str, object] = {
    "lease_id": "",
    "endpoint": "",
    "backend": "",
    "token": "",
}
_PRIORITY_BENCHMARK_FAMILY_ORDER = (
    "project",
    "repository",
    "integration",
    "repo_chore",
    "repo_sandbox",
)
_REPO_SETTING_FAMILY_NEIGHBOR_WEIGHTS: dict[str, dict[str, float]] = {
    "project": {"repository": 0.45},
    "repository": {"project": 0.45, "workflow": 0.65},
    "workflow": {"repository": 0.55, "tooling": 0.75},
    "tooling": {"workflow": 0.75, "integration": 0.7},
    "integration": {"tooling": 0.7, "repo_chore": 0.55},
    "repo_chore": {"integration": 0.55, "validation": 0.65},
    "validation": {"repo_chore": 0.65, "governance": 0.7},
    "governance": {"validation": 0.7, "oversight": 0.7},
    "oversight": {"governance": 0.7, "assurance": 0.7},
    "assurance": {"oversight": 0.7, "adjudication": 0.65},
    "adjudication": {"assurance": 0.65},
}
_PRIORITY_FAMILY_LOW_RETURN_MIN_DECISIONS = 2
_PRIORITY_FAMILY_LOW_RETURN_MIN_ESTIMATED_COST = 4.0
_PRIORITY_FAMILY_EXPLORATION_WEIGHT = 0.05
_POST_PRODUCTIVE_GENERATED_FAILURE_DECISION_WINDOW_FRACTION = 0.8


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_report(report_path: Path, payload: dict[str, object], *, config: KernelConfig | None = None) -> None:
    atomic_write_json(report_path, payload, config=config)


def _cleanup_bootstrap_artifacts(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            continue


def _write_status(
    status_path: Path | None,
    *,
    report_path: Path,
    payload: dict[str, object],
    config: KernelConfig | None = None,
) -> None:
    if status_path is None:
        return
    existing_status: dict[str, object] = {}
    if status_path.exists():
        try:
            persisted = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            persisted = {}
        if isinstance(persisted, dict):
            existing_status = persisted
    active_run = payload.get("active_run", {})
    if not isinstance(active_run, dict):
        active_run = {}
    existing_active_run = existing_status.get("active_run", {})
    if isinstance(existing_active_run, dict):
        active_run = {
            **existing_active_run,
            **active_run,
        }
    active_run_policy = active_run.get("policy")
    if not isinstance(active_run_policy, dict):
        active_run_policy = None
    raw_active_child = payload.get("active_child", {})
    active_child = raw_active_child if isinstance(raw_active_child, dict) else {}
    mirrored_child_status = active_run.get("child_status", {})
    if not isinstance(mirrored_child_status, Mapping):
        mirrored_child_status = {}
    if active_child and mirrored_child_status:
        active_child = _merge_mirrored_child_status_fields(
            dict(active_child),
            mirrored_child_status,
        )
    if active_child:
        active_run["child_status"] = dict(active_child)
    projected_active_child = (
        dict(active_child)
        if active_child
        else (dict(mirrored_child_status) if mirrored_child_status else {})
    )
    cleanup = payload.get("cleanup", {})
    preflight = payload.get("preflight", {})
    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list):
        rounds = []
    latest_round = rounds[-1] if rounds and isinstance(rounds[-1], dict) else {}
    latest_rationale_round = {}
    latest_alert_round = {}
    for round_payload in reversed(rounds):
        if not isinstance(round_payload, dict):
            continue
        rationale = round_payload.get("policy_shift_rationale", {})
        if isinstance(rationale, dict) and rationale:
            if not latest_rationale_round:
                latest_rationale_round = round_payload
            if isinstance(round_payload.get("policy_shift_alert", {}), dict) and round_payload.get("policy_shift_alert", {}):
                latest_alert_round = round_payload
                break
            if _policy_shift_alert_reason(rationale):
                latest_alert_round = round_payload
                break
    policy_shift_subscriptions = payload.get("policy_shift_alert_subscriptions", [])
    if not isinstance(policy_shift_subscriptions, list):
        policy_shift_subscriptions = _normalize_policy_shift_alert_subscriptions(policy_shift_subscriptions)
    effective_alert_round = latest_alert_round if latest_alert_round else latest_rationale_round
    latest_policy_shift_alert = _effective_policy_shift_alert_decision(
        payload=payload,
        rationale_round=effective_alert_round,
        subscriptions=policy_shift_subscriptions,
    )
    if not isinstance(latest_policy_shift_alert, dict):
        latest_policy_shift_alert = {}
    projected_phase_detail = _preferred_active_child_phase_detail(
        str(payload.get("phase_detail", "")).strip(),
        projected_active_child,
    )
    projected_phase_detail_round_payload = dict(latest_round) if isinstance(latest_round, dict) else {}
    if projected_active_child and not isinstance(projected_phase_detail_round_payload.get("active_child", {}), Mapping):
        projected_phase_detail_round_payload["active_child"] = dict(projected_active_child)
    if not isinstance(projected_phase_detail_round_payload.get("controller_intervention", {}), Mapping):
        controller_intervention = payload.get("controller_intervention", {})
        if isinstance(controller_intervention, Mapping):
            projected_phase_detail_round_payload["controller_intervention"] = dict(controller_intervention)
    projected_phase_detail = _authoritative_round_phase_detail(
        existing_detail=projected_phase_detail,
        round_payload=projected_phase_detail_round_payload,
        campaign_report=payload.get("campaign_report", {}),
    )
    status_payload = {
        "spec_version": "asi_v1",
        "report_kind": "unattended_campaign_status",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "status": str(payload.get("status", "")).strip(),
        "phase": str(payload.get("phase", "")).strip(),
        "phase_detail": projected_phase_detail,
        "reason": str(payload.get("reason", "")).strip(),
        "rounds_requested": int(payload.get("rounds_requested", 0) or 0),
        "rounds_completed": int(payload.get("rounds_completed", 0) or 0),
        "child_failure_recovery_budget": int(payload.get("child_failure_recovery_budget", 0) or 0),
        "child_failure_recoveries_used": int(payload.get("child_failure_recoveries_used", 0) or 0),
        "current_policy": payload.get("current_policy", {}),
        "active_run_policy": active_run_policy,
        "active_run": active_run,
        "requested_priority_benchmark_families": _normalize_benchmark_families(
            payload.get("current_policy", {}).get("priority_benchmark_families", [])
            if isinstance(payload.get("current_policy", {}), dict)
            else []
        ),
        "policy_shift_summary": payload.get("policy_shift_summary", {}),
        "policy_shift_alert_subscriptions": policy_shift_subscriptions,
        "latest_round_policy_shift_rationale": latest_round.get("policy_shift_rationale", {})
        if isinstance(latest_round, dict)
        else {},
        "latest_semantic_redirection": latest_round.get("semantic_redirection", {})
        if isinstance(latest_round, dict)
        else {},
        "latest_available_policy_shift_rationale": latest_rationale_round.get("policy_shift_rationale", {})
        if isinstance(latest_rationale_round, dict)
        else {},
        "latest_policy_shift_alert": latest_policy_shift_alert,
        "latest_policy_shift_alert_summary": _policy_shift_alert_status_summary(
            latest_policy_shift_alert,
            rationale=effective_alert_round.get("policy_shift_rationale", {})
            if isinstance(effective_alert_round, dict)
            else {},
        ),
        "campaign_validation": payload.get("campaign_validation", {}),
        "liftoff_validation": payload.get("liftoff_validation", {}),
        "active_child": dict(projected_active_child),
        "active_child_controller_intervention": (
            projected_active_child.get("controller_intervention", {})
            if isinstance(projected_active_child, dict)
            else {}
        ),
        "preflight": preflight if isinstance(preflight, dict) else {},
        "cleanup": cleanup if isinstance(cleanup, dict) else {},
        "unattended_evidence": payload.get("unattended_evidence", {}),
        "global_storage": payload.get("global_storage", {}),
        "lock_path": str(payload.get("lock_path", "")).strip(),
        "event_log_path": str(payload.get("event_log_path", "")).strip(),
    }
    status_payload.update(
        _live_status_projection(
            payload=payload,
            active_run=active_run,
            active_child=projected_active_child,
        )
    )
    for key in (
        "campaign_round_index",
        "last_progress_phase",
        "runtime_managed_decisions",
        "non_runtime_managed_decisions",
        "execution_source_summary",
        "sampled_families_from_progress",
        "required_benchmark_families",
        "families_sampled",
        "families_never_sampled",
        "pressure_families_without_sampling",
        "latest_generated_run_report_path",
        "trust_status",
        "decision_stream_summary",
        "trust_breadth_summary",
        "partial_frontier_expansion_summary",
    ):
        if key in existing_status and status_payload.get(key) in ({}, [], "", None):
            status_payload[key] = existing_status[key]
    atomic_write_json(status_path, status_payload, config=config)


def _live_status_projection(
    *,
    payload: Mapping[str, object] | None,
    active_run: Mapping[str, object] | None,
    active_child: Mapping[str, object] | None,
) -> dict[str, object]:
    report_payload = payload if isinstance(payload, Mapping) else {}
    active_run_payload = active_run if isinstance(active_run, Mapping) else {}
    child = active_child if isinstance(active_child, Mapping) else {}
    current_policy = report_payload.get("current_policy", {})
    if not isinstance(current_policy, Mapping):
        current_policy = {}
    active_run_policy = active_run_payload.get("policy", {})
    if not isinstance(active_run_policy, Mapping):
        active_run_policy = {}
    required_families = _normalize_benchmark_families(
        active_run_policy.get(
            "priority_benchmark_families",
            current_policy.get("priority_benchmark_families", []),
        )
    )
    sampled_families = _normalize_benchmark_families(
        child.get(
            "families_sampled",
            child.get("sampled_families_from_progress", []),
        )
    )
    if not sampled_families:
        active_cycle_progress = child.get("active_cycle_progress", {})
        if isinstance(active_cycle_progress, Mapping):
            sampled_families = _normalize_benchmark_families(active_cycle_progress.get("sampled_families_from_progress", []))
    decision_conversion_summary = child.get("decision_conversion_summary", {})
    if not isinstance(decision_conversion_summary, Mapping):
        decision_conversion_summary = {}
    runtime_managed_decisions = max(
        0,
        int(child.get("runtime_managed_decisions", 0) or 0),
        int(decision_conversion_summary.get("runtime_managed_runs", 0) or 0),
    )
    non_runtime_managed_decisions = max(
        0,
        int(child.get("non_runtime_managed_decisions", 0) or 0),
        int(decision_conversion_summary.get("non_runtime_managed_runs", 0) or 0),
    )
    partial_productive_without_decision_runs = max(
        0,
        int(child.get("partial_productive_runs", 0) or 0),
        int(decision_conversion_summary.get("partial_productive_without_decision_runs", 0) or 0),
    )
    decision_runs = max(
        runtime_managed_decisions + non_runtime_managed_decisions,
        int(decision_conversion_summary.get("decision_runs", 0) or 0),
    )
    latest_generated_run_report_path = str(
        child.get("last_report_path")
        or report_payload.get("campaign_report_path")
        or report_payload.get("liftoff_report_path")
        or ""
    ).strip()
    required_family_set = set(required_families)
    sampled_required_families = [family for family in sampled_families if family in required_family_set]
    missing_required_families = (
        [family for family in required_families if family not in set(sampled_families)]
        if sampled_families
        else []
    )
    current_task = child.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    canonical_progress_state = child.get("canonical_progress_state", {})
    if not isinstance(canonical_progress_state, Mapping):
        canonical_progress_state = {}
    semantic_progress_state = child.get("semantic_progress_state", {})
    if not isinstance(semantic_progress_state, Mapping):
        semantic_progress_state = {}
    adaptive_progress_state = child.get("adaptive_progress_state", {})
    if not isinstance(adaptive_progress_state, Mapping):
        adaptive_progress_state = {}
    if not canonical_progress_state:
        canonical_progress_state = _select_preferred_child_progress_state(
            semantic_state=semantic_progress_state,
            adaptive_state=adaptive_progress_state,
            current_task=current_task,
            last_progress_phase=str(child.get("last_progress_phase", "")).strip(),
        )
    child_trust_breadth_summary = child.get("trust_breadth_summary", {})
    if not isinstance(child_trust_breadth_summary, Mapping):
        child_trust_breadth_summary = {}
    report_trust_breadth_summary = report_payload.get("trust_breadth_summary", {})
    if not isinstance(report_trust_breadth_summary, Mapping):
        report_trust_breadth_summary = {}
    trusted_breadth = child_trust_breadth_summary or report_trust_breadth_summary
    trusted_required_families = _normalize_benchmark_families(
        trusted_breadth.get("required_families", required_families)
    )
    trusted_required_families_with_reports = _normalize_benchmark_families(
        trusted_breadth.get("required_families_with_reports", [])
    )
    trusted_missing_required_families = _normalize_benchmark_families(
        trusted_breadth.get("missing_required_families", [])
    )
    campaign_report = report_payload.get("campaign_report", {})
    if not isinstance(campaign_report, Mapping):
        campaign_report = {}
    authoritative_decision_state = _authoritative_status_decision_state(
        active_child=child,
        campaign_report=campaign_report,
    )
    runtime_managed_decisions = max(
        runtime_managed_decisions,
        int(authoritative_decision_state.get("runtime_managed_decisions", 0) or 0),
    )
    non_runtime_managed_decisions = max(
        non_runtime_managed_decisions,
        int(authoritative_decision_state.get("non_runtime_managed_decisions", 0) or 0),
    )
    retained_gain_runs = max(
        int(child.get("retained_gain_runs", 0) or 0),
        int(authoritative_decision_state.get("retained_gain_runs", 0) or 0),
    )
    decision_runs = max(
        decision_runs,
        int(authoritative_decision_state.get("decision_records_considered", 0) or 0),
    )
    if decision_runs > 0:
        partial_productive_without_decision_runs = 0
    return {
        "campaign_round_index": int(
            child.get("round_index", report_payload.get("rounds_completed", 0) or 0) or 0
        ),
        "last_progress_phase": str(
            child.get(
                "last_progress_phase",
                current_task.get("phase", ""),
            )
        ).strip(),
        "current_task": dict(current_task),
        "current_cognitive_stage": dict(child.get("current_cognitive_stage", {}))
        if isinstance(child.get("current_cognitive_stage", {}), Mapping)
        else {},
        "canonical_progress_state": dict(canonical_progress_state),
        "semantic_progress_state": dict(canonical_progress_state or semantic_progress_state),
        "runtime_managed_decisions": runtime_managed_decisions,
        "non_runtime_managed_decisions": non_runtime_managed_decisions,
        "execution_source_summary": (
            dict(child.get("execution_source_summary", {}))
            if isinstance(child.get("execution_source_summary", {}), Mapping)
            else {}
        ),
        "sampled_families_from_progress": sampled_families,
        "required_benchmark_families": required_families,
        "families_sampled": sampled_families,
        "families_never_sampled": missing_required_families,
        "pressure_families_without_sampling": missing_required_families,
        "latest_generated_run_report_path": latest_generated_run_report_path,
        "trust_status": str(
            report_payload.get("unattended_evidence", {}).get("overall_assessment", {}).get("status", "")
            if isinstance(report_payload.get("unattended_evidence", {}), Mapping)
            and isinstance(report_payload.get("unattended_evidence", {}).get("overall_assessment", {}), Mapping)
            else ""
        ).strip(),
        "decision_stream_summary": {
            "runtime_managed": {
                "total_decisions": runtime_managed_decisions,
                "retained_cycles": retained_gain_runs,
                "rejected_cycles": max(0, runtime_managed_decisions - retained_gain_runs),
            },
            "non_runtime_managed": {
                "total_decisions": non_runtime_managed_decisions,
                "retained_cycles": 0,
                "rejected_cycles": non_runtime_managed_decisions,
            },
            "partial_productive_without_decision_runs": partial_productive_without_decision_runs,
            "decision_runs": decision_runs,
        },
        "authoritative_decision_state": authoritative_decision_state,
        "trust_breadth_summary": {
            "required_families": trusted_required_families or required_families,
            "required_families_with_reports": trusted_required_families_with_reports,
            "missing_required_families": (
                trusted_missing_required_families
                if trusted_missing_required_families or trusted_required_families_with_reports
                else missing_required_families
            ),
            "distinct_external_benchmark_families": int(
                trusted_breadth.get("distinct_external_benchmark_families", 0) or 0
            ),
            "external_report_count": int(trusted_breadth.get("external_report_count", 0) or 0),
            "sampled_families_from_progress": sampled_families,
            "bootstrap_sampled_required_families": sampled_required_families,
        },
    }


def _authoritative_status_decision_state(
    *,
    active_child: Mapping[str, object] | None,
    campaign_report: Mapping[str, object] | None,
) -> dict[str, object]:
    child = active_child if isinstance(active_child, Mapping) else {}
    report = campaign_report if isinstance(campaign_report, Mapping) else {}
    report_runs = report.get("runs", [])
    first_run = report_runs[0] if isinstance(report_runs, list) and report_runs and isinstance(report_runs[0], Mapping) else {}
    child_decision_records_considered, _ = _effective_live_decision_record_count(child)
    child_retained_gain_runs = _effective_live_retained_gain_runs(child)
    runtime_managed_decisions = max(
        0,
        int(child.get("runtime_managed_decisions", 0) or 0),
        int(first_run.get("runtime_managed_decisions", 0) or 0),
        int(report.get("runtime_managed_decisions", 0) or 0),
    )
    non_runtime_managed_decisions = max(
        0,
        int(child.get("non_runtime_managed_decisions", 0) or 0),
        int(first_run.get("non_runtime_managed_decisions", 0) or 0),
        int(report.get("non_runtime_managed_decisions", 0) or 0),
    )
    decision_records_considered = max(
        child_decision_records_considered,
        int(first_run.get("decision_records_considered", 0) or 0),
        runtime_managed_decisions + non_runtime_managed_decisions,
    )
    retained_gain_runs = max(
        child_retained_gain_runs,
        int(report.get("retained_gain_runs", 0) or 0),
        1 if bool(first_run.get("retained_gain", False)) else 0,
    )
    pending_decision_state = str(
        child.get("pending_decision_state")
        or first_run.get("final_state")
        or ""
    ).strip()
    decision_conversion_state = str(
        first_run.get("decision_conversion_state")
        or ("runtime_managed" if runtime_managed_decisions > 0 else "")
    ).strip()
    return {
        "decision_records_considered": decision_records_considered,
        "runtime_managed_decisions": runtime_managed_decisions,
        "non_runtime_managed_decisions": non_runtime_managed_decisions,
        "retained_gain_runs": retained_gain_runs,
        "pending_decision_state": pending_decision_state,
        "decision_conversion_state": decision_conversion_state,
    }


def _normalize_benchmark_families(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return normalized


def _effective_live_decision_record_count(payload: Mapping[str, object] | None) -> tuple[int, bool]:
    child = payload if isinstance(payload, Mapping) else {}
    raw_count = max(
        0,
        int(child.get("decision_records_considered", 0) or 0),
        int(child.get("runtime_managed_decisions", 0) or 0),
        int(child.get("non_runtime_managed_decisions", 0) or 0),
    )
    pending_decision_state = str(child.get("pending_decision_state", "")).strip()
    preview_state = str(child.get("preview_state", "")).strip()
    finalize_phase = str(child.get("finalize_phase", child.get("last_progress_phase", ""))).strip()
    definitive_decision_emitted = (
        pending_decision_state in {"retain", "reject"}
        or preview_state in {"retain", "reject"}
        or finalize_phase in {"apply_decision", "decision_reject_reason", "decision_retain_reason", "done"}
    )
    if definitive_decision_emitted and raw_count <= 0:
        return 1, True
    return raw_count, False


def _effective_live_retained_gain_runs(payload: Mapping[str, object] | None) -> int:
    child = payload if isinstance(payload, Mapping) else {}
    retained_gain_runs = max(0, int(child.get("retained_gain_runs", 0) or 0))
    pending_decision_state = str(child.get("pending_decision_state", "")).strip()
    preview_state = str(child.get("preview_state", "")).strip()
    if retained_gain_runs <= 0 and (pending_decision_state == "retain" or preview_state == "retain"):
        return 1
    return retained_gain_runs


def _rank_priority_benchmark_families(values: object) -> list[str]:
    families = _normalize_benchmark_families(values)
    order_index = {family: index for index, family in enumerate(_PRIORITY_BENCHMARK_FAMILY_ORDER)}
    return sorted(families, key=lambda family: (order_index.get(family, len(order_index)), family))


def _ordered_priority_benchmark_families(values: object) -> list[str]:
    return _normalize_benchmark_families(values)


def _load_retained_improvement_planner_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(getattr(config, "use_prompt_proposals", True)):
        return {}
    path = getattr(config, "prompt_proposals_path", None)
    if not isinstance(path, Path) or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_improvement_planner_controls(payload)


def _load_retained_curriculum_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(getattr(config, "use_curriculum_proposals", True)):
        return {}
    path = getattr(config, "curriculum_proposals_path", None)
    if not isinstance(path, Path) or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_curriculum_controls(payload)


def _curriculum_control_family_signal_pairs(
    controls: dict[str, object] | None,
    field: str,
) -> list[tuple[str, str]]:
    payload = controls if isinstance(controls, dict) else {}
    raw_values = payload.get(field, [])
    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, list):
        values = [str(value) for value in raw_values]
    else:
        values = []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        family, _, signal = str(value).strip().partition(":")
        normalized_family = family.strip()
        normalized_signal = signal.strip().lower()
        if not normalized_family or not normalized_signal:
            continue
        pair = (normalized_family, normalized_signal)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _planner_control_float(
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


def _planner_control_family_float(
    planner_controls: dict[str, object],
    field: str,
    family: str,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    mapping = planner_controls.get(field, {})
    if not isinstance(mapping, dict):
        return default
    value = mapping.get(family, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _select_priority_benchmark_families(
    *,
    required_families: object,
    missing_required_families: object,
    current_priority_families: object,
    ranked_priority_families: object | None = None,
    repo_setting_priority_families: object | None = None,
    priority_family_selection_scores: dict[str, object] | None = None,
    min_selection_score: float = 0.0,
    productive_priority_families: object | None = None,
    retained_gain_conversion_priority_families: object | None = None,
    under_sampled_priority_families: object | None = None,
    low_return_priority_families: object | None = None,
) -> list[str]:
    ranked_missing = _rank_priority_benchmark_families(missing_required_families)
    if ranked_missing:
        if min_selection_score > 0.0:
            return _normalize_benchmark_families(missing_required_families)[: min(3, len(ranked_missing))]
        missing_priority_order = {
            family: index
            for index, family in enumerate(("repository", "integration", "project", "repo_chore", "repo_sandbox"))
        }
        ranked_missing = sorted(
            ranked_missing,
            key=lambda family: (missing_priority_order.get(family, len(missing_priority_order)), family),
        )
        return ranked_missing[: min(3, len(ranked_missing))]
    score_mapping = priority_family_selection_scores if isinstance(priority_family_selection_scores, dict) else {}

    def _passes_selection_floor(family: str) -> bool:
        try:
            score = float(score_mapping.get(family, 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return score >= min_selection_score

    ranked_priority = _ordered_priority_benchmark_families(ranked_priority_families or [])
    ranked_productive = _rank_priority_benchmark_families(productive_priority_families or [])
    productive_set = set(ranked_productive)
    ranked_conversion = [
        family
        for family in _rank_priority_benchmark_families(retained_gain_conversion_priority_families or [])
        if family not in productive_set
    ]
    conversion_set = set(ranked_conversion)
    ranked_low_return = [
        family
        for family in _rank_priority_benchmark_families(low_return_priority_families or [])
        if family not in productive_set and family not in conversion_set
    ]
    low_return_set = set(ranked_low_return)
    ranked_repo_setting = [
        family
        for family in _rank_priority_benchmark_families(repo_setting_priority_families or [])
        if family not in productive_set and family not in conversion_set and family not in low_return_set
    ]
    ranked_under_sampled = [
        family
        for family in _rank_priority_benchmark_families(under_sampled_priority_families or [])
        if family not in productive_set and family not in low_return_set
    ]
    ranked_required = _rank_priority_benchmark_families(required_families)
    ranked_required_without_productive = [
        family
        for family in ranked_required
        if family not in productive_set and family not in conversion_set and family not in low_return_set
    ]
    ranked_current = [
        family
        for family in _rank_priority_benchmark_families(current_priority_families)
        if family not in productive_set and family not in low_return_set
    ]
    selected: list[str] = []
    for family in [
        *ranked_conversion,
        *ranked_repo_setting,
        *ranked_priority,
        *ranked_under_sampled,
        *ranked_required_without_productive,
        *ranked_current,
        *ranked_productive,
        *ranked_low_return,
        *ranked_required,
    ]:
        if family in selected:
            continue
        if not _passes_selection_floor(family):
            continue
        selected.append(family)
        if len(selected) >= 3:
            break
    return selected


def _repo_setting_focus_priority_families(
    *,
    current_priority_families: object,
    ranked_priority_families: object | None = None,
    frontier_repo_setting_families: object | None = None,
    missing_required_families: object | None = None,
) -> list[str]:
    selected: list[str] = []
    for families in (
        frontier_repo_setting_families,
        missing_required_families,
        ranked_priority_families,
        current_priority_families,
    ):
        for family in _rank_priority_benchmark_families(families):
            if family not in selected:
                selected.append(family)
    return selected


def _priority_family_history_summary(
    *,
    prior_rounds: list[dict[str, object]],
    current_campaign_report: dict[str, object],
    current_priority_families: object,
    planner_controls: dict[str, object] | None = None,
    curriculum_controls: dict[str, object] | None = None,
) -> dict[str, object]:
    def _empty_summary() -> dict[str, object]:
        return {
            "observed_decisions": 0,
            "retained_decisions": 0,
            "rejected_decisions": 0,
            "observed_estimated_cost": 0.0,
            "retained_estimated_cost": 0.0,
            "rejected_estimated_cost": 0.0,
            "positive_delta_decisions": 0,
            "negative_delta_decisions": 0,
            "neutral_delta_decisions": 0,
            "retained_positive_delta_decisions": 0,
            "retained_negative_delta_decisions": 0,
            "retained_neutral_delta_decisions": 0,
            "retained_pass_rate_delta_sum": 0.0,
            "retained_positive_pass_rate_delta_sum": 0.0,
            "average_retained_pass_rate_delta": 0.0,
            "best_retained_pass_rate_delta": 0.0,
            "worst_pass_rate_delta": 0.0,
        }

    def _merge_summary(
        aggregate: dict[str, object],
        incoming: dict[str, object],
    ) -> None:
        for key in (
            "observed_decisions",
            "retained_decisions",
            "rejected_decisions",
            "positive_delta_decisions",
            "negative_delta_decisions",
            "neutral_delta_decisions",
            "retained_positive_delta_decisions",
            "retained_negative_delta_decisions",
            "retained_neutral_delta_decisions",
        ):
            aggregate[key] = int(aggregate.get(key, 0) or 0) + int(incoming.get(key, 0) or 0)
        for key in (
            "observed_estimated_cost",
            "retained_estimated_cost",
            "rejected_estimated_cost",
            "retained_pass_rate_delta_sum",
            "retained_positive_pass_rate_delta_sum",
        ):
            aggregate[key] = float(aggregate.get(key, 0.0) or 0.0) + float(incoming.get(key, 0.0) or 0.0)
        incoming_best = float(incoming.get("best_retained_pass_rate_delta", 0.0) or 0.0)
        incoming_best_count = int(incoming.get("retained_decisions", 0) or 0)
        aggregate_best_count = int(aggregate.get("retained_decisions", 0) or 0)
        if incoming_best_count > 0 and aggregate_best_count > incoming_best_count:
            aggregate["best_retained_pass_rate_delta"] = max(float(aggregate.get("best_retained_pass_rate_delta", 0.0) or 0.0), incoming_best)
        elif incoming_best_count > 0:
            aggregate["best_retained_pass_rate_delta"] = incoming_best
        incoming_observed = int(incoming.get("observed_decisions", 0) or 0)
        aggregate_observed_before = int(aggregate.get("observed_decisions", 0) or 0) - incoming_observed
        incoming_worst = float(incoming.get("worst_pass_rate_delta", 0.0) or 0.0)
        if incoming_observed > 0 and aggregate_observed_before > 0:
            aggregate["worst_pass_rate_delta"] = min(float(aggregate.get("worst_pass_rate_delta", 0.0) or 0.0), incoming_worst)
        elif incoming_observed > 0:
            aggregate["worst_pass_rate_delta"] = incoming_worst

    priority_families: list[str] = []
    family_summaries: dict[str, dict[str, object]] = {}
    source_reports: list[dict[str, object]] = []
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        campaign_report = round_payload.get("campaign_report", {})
        if isinstance(campaign_report, dict) and campaign_report:
            source_reports.append(campaign_report)
    if isinstance(current_campaign_report, dict) and current_campaign_report:
        source_reports.append(current_campaign_report)
    for report in source_reports:
        report_priority_families = _normalize_benchmark_families(
            report.get(
                "priority_benchmark_families",
                report.get("priority_family_yield_summary", {}).get("priority_families", [])
                if isinstance(report.get("priority_family_yield_summary", {}), dict)
                else [],
            )
        )
        for family in report_priority_families:
            if family not in priority_families:
                priority_families.append(family)
        priority_summary = report.get("priority_family_yield_summary", {})
        if not isinstance(priority_summary, dict):
            continue
        incoming_family_summaries = priority_summary.get("family_summaries", {})
        if not isinstance(incoming_family_summaries, dict):
            continue
        for family in report_priority_families:
            incoming = incoming_family_summaries.get(family, {})
            if not isinstance(incoming, dict):
                continue
            aggregate = family_summaries.setdefault(family, _empty_summary())
            _merge_summary(aggregate, incoming)
    for family in _normalize_benchmark_families(current_priority_families):
        if family not in priority_families:
            priority_families.append(family)
        family_summaries.setdefault(family, _empty_summary())
    resolved_planner_controls = planner_controls if isinstance(planner_controls, dict) else {}
    resolved_curriculum_controls = curriculum_controls if isinstance(curriculum_controls, dict) else {}
    frontier_failure_motif_pairs = _curriculum_control_family_signal_pairs(
        resolved_curriculum_controls,
        "frontier_failure_motif_priority_pairs",
    )
    frontier_repo_setting_pairs = _curriculum_control_family_signal_pairs(
        resolved_curriculum_controls,
        "frontier_repo_setting_priority_pairs",
    )
    frontier_failure_motif_counts: dict[str, int] = {}
    for family, _ in frontier_failure_motif_pairs:
        frontier_failure_motif_counts[family] = frontier_failure_motif_counts.get(family, 0) + 1
    frontier_repo_setting_counts: dict[str, int] = {}
    for family, _ in frontier_repo_setting_pairs:
        frontier_repo_setting_counts[family] = frontier_repo_setting_counts.get(family, 0) + 1
    failure_motif_selection_bonus_scale = min(
        0.2,
        0.01 * max(1, int(resolved_curriculum_controls.get("frontier_failure_motif_bonus", 0) or 0)),
    )
    repo_setting_selection_bonus_scale = min(
        0.25,
        0.01 * max(1, int(resolved_curriculum_controls.get("frontier_repo_setting_bonus", 0) or 0)),
    )
    priority_family_categories: dict[str, str] = {}
    for family in priority_families:
        summary = family_summaries.setdefault(family, _empty_summary())
        retained_decisions = int(summary.get("retained_decisions", 0) or 0)
        if retained_decisions > 0:
            summary["average_retained_pass_rate_delta"] = (
                float(summary.get("retained_pass_rate_delta_sum", 0.0) or 0.0) / float(retained_decisions)
            )
        observed_cost = float(summary.get("observed_estimated_cost", 0.0) or 0.0)
        positive_delta_sum = float(summary.get("retained_positive_pass_rate_delta_sum", 0.0) or 0.0)
        summary["retained_positive_pass_rate_delta_per_estimated_cost"] = (
            positive_delta_sum / observed_cost if observed_cost > 0.0 else 0.0
        )
    priority_families_with_retained_gain = [
        family
        for family in priority_families
        if int(family_summaries[family].get("retained_positive_delta_decisions", 0) or 0) > 0
    ]
    priority_families_without_signal = [
        family for family in priority_families if int(family_summaries[family].get("observed_decisions", 0) or 0) <= 0
    ]
    priority_families_under_sampled = [
        family
        for family in priority_families
        if family not in priority_families_with_retained_gain
        and (
            family in priority_families_without_signal
            or int(family_summaries[family].get("observed_decisions", 0) or 0) < _PRIORITY_FAMILY_LOW_RETURN_MIN_DECISIONS
            or float(family_summaries[family].get("observed_estimated_cost", 0.0) or 0.0)
            < _PRIORITY_FAMILY_LOW_RETURN_MIN_ESTIMATED_COST
        )
    ]
    priority_families_low_return = [
        family
        for family in priority_families
        if family not in priority_families_with_retained_gain
        and family not in priority_families_under_sampled
    ]
    priority_families_needing_retained_gain_conversion = [
        family
        for family in sorted(
            priority_families_low_return + priority_families_under_sampled,
            key=lambda item: (
                -int(family_summaries[item].get("observed_decisions", 0) or 0),
                -int(family_summaries[item].get("rejected_decisions", 0) or 0),
                -float(family_summaries[item].get("observed_estimated_cost", 0.0) or 0.0),
                item,
            ),
        )
        if int(family_summaries[family].get("observed_decisions", 0) or 0) > 0
    ]
    priority_family_selection_scores: dict[str, float] = {}
    priority_family_return_on_cost_scores: dict[str, float] = {}
    priority_family_scoring_policy = {
        "priority_family_exploration_bonus": _planner_control_float(
            resolved_planner_controls,
            "priority_family_exploration_bonus",
            _PRIORITY_FAMILY_EXPLORATION_WEIGHT,
            min_value=0.0,
            max_value=0.25,
        ),
        "priority_family_min_selection_score": _planner_control_float(
            resolved_planner_controls,
            "priority_family_min_selection_score",
            0.0,
            min_value=-0.1,
            max_value=0.25,
        ),
    }
    order_index = {family: index for index, family in enumerate(_PRIORITY_BENCHMARK_FAMILY_ORDER)}
    for family in priority_families:
        summary = family_summaries[family]
        gain_multiplier = _planner_control_family_float(
            resolved_planner_controls,
            "priority_family_retained_gain_multiplier",
            family,
            default=1.0,
            min_value=0.25,
            max_value=4.0,
        )
        cost_multiplier = _planner_control_family_float(
            resolved_planner_controls,
            "priority_family_cost_multiplier",
            family,
            default=1.0,
            min_value=0.5,
            max_value=4.0,
        )
        score_bias = _planner_control_family_float(
            resolved_planner_controls,
            "priority_family_score_bias",
            family,
            default=0.0,
            min_value=-0.2,
            max_value=0.2,
        )
        return_on_cost = float(summary.get("retained_positive_pass_rate_delta_per_estimated_cost", 0.0) or 0.0)
        adjusted_return_on_cost = return_on_cost * gain_multiplier / max(0.5, cost_multiplier)
        exploration_bonus = (
            float(priority_family_scoring_policy["priority_family_exploration_bonus"])
            if family in priority_families_under_sampled
            else 0.0
        )
        failure_motif_selection_bonus = failure_motif_selection_bonus_scale * min(
            2.0,
            float(frontier_failure_motif_counts.get(family, 0)),
        )
        repo_setting_selection_bonus = repo_setting_selection_bonus_scale * min(
            2.0,
            float(frontier_repo_setting_counts.get(family, 0)),
        )
        summary["selection_return_on_cost"] = return_on_cost
        summary["selection_adjusted_return_on_cost"] = adjusted_return_on_cost
        summary["selection_gain_multiplier"] = gain_multiplier
        summary["selection_cost_multiplier"] = cost_multiplier
        summary["selection_score_bias"] = score_bias
        summary["selection_exploration_bonus"] = exploration_bonus
        summary["selection_frontier_failure_motif_bonus"] = failure_motif_selection_bonus
        summary["selection_frontier_repo_setting_bonus"] = repo_setting_selection_bonus
        summary["selection_score"] = (
            adjusted_return_on_cost
            + exploration_bonus
            + score_bias
            + failure_motif_selection_bonus
            + repo_setting_selection_bonus
        )
        if family in priority_families_with_retained_gain:
            priority_family_categories[family] = "productive"
        elif family in priority_families_under_sampled:
            priority_family_categories[family] = "under_sampled"
        else:
            priority_family_categories[family] = "low_return"
        priority_family_return_on_cost_scores[family] = adjusted_return_on_cost
        priority_family_selection_scores[family] = float(summary["selection_score"])
    priority_families_ranked_by_return_on_cost = sorted(
        priority_families,
        key=lambda family: (
            -priority_family_return_on_cost_scores[family],
            order_index.get(family, len(order_index)),
            family,
        ),
    )
    priority_families_ranked_by_selection_score = sorted(
        priority_families,
        key=lambda family: (
            -priority_family_selection_scores[family],
            -priority_family_return_on_cost_scores[family],
            order_index.get(family, len(order_index)),
            family,
        ),
    )
    return {
        "priority_families": priority_families,
        "family_summaries": family_summaries,
        "priority_families_with_retained_gain": _rank_priority_benchmark_families(priority_families_with_retained_gain),
        "priority_families_without_signal": _rank_priority_benchmark_families(priority_families_without_signal),
        "priority_families_under_sampled": _rank_priority_benchmark_families(priority_families_under_sampled),
        "priority_families_low_return": _rank_priority_benchmark_families(priority_families_low_return),
        "priority_families_needing_retained_gain_conversion": _rank_priority_benchmark_families(
            priority_families_needing_retained_gain_conversion
        ),
        "priority_family_categories": priority_family_categories,
        "priority_family_scoring_policy": priority_family_scoring_policy,
        "priority_family_return_on_cost_scores": priority_family_return_on_cost_scores,
        "priority_family_selection_scores": priority_family_selection_scores,
        "priority_families_ranked_by_return_on_cost": priority_families_ranked_by_return_on_cost,
        "priority_families_ranked_by_selection_score": priority_families_ranked_by_selection_score,
        "frontier_failure_motif_priority_pairs": [f"{family}:{signal}" for family, signal in frontier_failure_motif_pairs],
        "frontier_repo_setting_priority_pairs": [f"{family}:{signal}" for family, signal in frontier_repo_setting_pairs],
        "frontier_failure_motif_families": _rank_priority_benchmark_families(list(frontier_failure_motif_counts)),
        "frontier_repo_setting_families": _rank_priority_benchmark_families(list(frontier_repo_setting_counts)),
        "low_return_min_decisions": _PRIORITY_FAMILY_LOW_RETURN_MIN_DECISIONS,
        "low_return_min_estimated_cost": _PRIORITY_FAMILY_LOW_RETURN_MIN_ESTIMATED_COST,
    }


def _append_event(
    event_log_path: Path | None,
    payload: dict[str, object],
    *,
    config: KernelConfig | None = None,
) -> None:
    if event_log_path is None:
        return
    # Event logs are append-only diagnostics; running full export governance on every
    # JSONL append turns normal round closeout into an unbounded directory walk.
    append_jsonl(event_log_path, payload, config=config, govern_storage=False)


def _write_controller_state(path: Path, payload: dict[str, object], *, config: KernelConfig | None = None) -> None:
    atomic_write_json(path, payload, config=config)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock_payload(lock_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_lock_payload(fd: int, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, encoded)
    os.fsync(fd)


def _post_json(url: str, payload: dict[str, object], *, headers: dict[str, str] | None = None) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8", errors="replace").strip()
        try:
            decoded = json.loads(body) if body else {}
        except json.JSONDecodeError:
            decoded = {"raw_body": body}
        if not isinstance(decoded, dict):
            decoded = {"raw_body": body}
        decoded["http_status"] = int(getattr(response, "status", 200) or 200)
        return decoded


def _external_lease_headers(token: str) -> dict[str, str]:
    normalized = str(token).strip()
    return {"Authorization": f"Bearer {normalized}"} if normalized else {}


def _acquire_external_campaign_lease(
    endpoint: str,
    *,
    token: str,
    report_path: Path,
) -> dict[str, object]:
    base = str(endpoint).rstrip("/")
    payload = {
        "lease_kind": "agentkernel_unattended_campaign",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "report_path": str(report_path),
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = _post_json(
            f"{base}/acquire",
            payload,
            headers=_external_lease_headers(token),
        )
    except (urllib.error.URLError, OSError) as exc:
        return {"passed": False, "detail": f"external lease acquire failed: {exc}", "backend": "http"}
    lease_id = str(response.get("lease_id", "")).strip()
    passed = bool(response.get("acquired", False)) and bool(lease_id)
    if passed:
        _EXTERNAL_LEASE_STATE["lease_id"] = lease_id
        _EXTERNAL_LEASE_STATE["endpoint"] = base
        _EXTERNAL_LEASE_STATE["backend"] = "http"
        _EXTERNAL_LEASE_STATE["token"] = str(token or "")
    return {
        "passed": passed,
        "detail": str(response.get("detail", "external lease response")).strip(),
        "backend": "http",
        "lease_id": lease_id,
        "response": response,
    }


def _refresh_external_campaign_lease(
    endpoint: str,
    *,
    token: str,
    report_path: Path,
    status_path: Path | None,
    payload: dict[str, object],
) -> None:
    lease_id = str(_EXTERNAL_LEASE_STATE.get("lease_id", "")).strip()
    if not lease_id:
        return
    request_payload = {
        "lease_id": lease_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "report_path": str(report_path),
        "status_path": "" if status_path is None else str(status_path),
        "status": str(payload.get("status", "")).strip(),
        "phase": str(payload.get("phase", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _post_json(
            f"{str(endpoint).rstrip('/')}/heartbeat",
            request_payload,
            headers=_external_lease_headers(token),
        )
    except (urllib.error.URLError, OSError):
        return


def _release_external_campaign_lease(endpoint: str, *, token: str) -> None:
    lease_id = str(_EXTERNAL_LEASE_STATE.get("lease_id", "")).strip()
    if not lease_id:
        return
    try:
        _post_json(
            f"{str(endpoint).rstrip('/')}/release",
            {
                "lease_id": lease_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "released_at": datetime.now(timezone.utc).isoformat(),
            },
            headers=_external_lease_headers(token),
        )
    except (urllib.error.URLError, OSError):
        pass
    _EXTERNAL_LEASE_STATE["lease_id"] = ""
    _EXTERNAL_LEASE_STATE["endpoint"] = ""
    _EXTERNAL_LEASE_STATE["backend"] = ""
    _EXTERNAL_LEASE_STATE["token"] = ""


def _acquire_campaign_lock(lock_path: Path, *, report_path: Path, status_path: Path | None = None) -> dict[str, object]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    current_fd = _CAMPAIGN_LOCK_STATE.get("fd")
    current_path = _CAMPAIGN_LOCK_STATE.get("path")
    if isinstance(current_fd, int) and current_fd >= 0 and current_path == str(lock_path):
        existing = _read_lock_payload(lock_path)
        return {
            "passed": True,
            "detail": "campaign lock already held by current process",
            "lock_path": str(lock_path),
            "lock_payload": existing,
        }
    payload = {
        "lock_kind": "unattended_campaign_lock",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "status_path": "" if status_path is None else str(status_path),
    }
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing = _read_lock_payload(lock_path)
        try:
            os.close(fd)
        except OSError:
            pass
        return {
            "passed": False,
            "detail": "campaign lock already held by another process",
            "lock_path": str(lock_path),
            "lock_payload": existing,
        }
    try:
        _write_lock_payload(fd, payload)
    except OSError:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        return {
            "passed": False,
            "detail": "failed to persist campaign lock payload",
            "lock_path": str(lock_path),
            "lock_payload": payload,
        }
    _CAMPAIGN_LOCK_STATE["fd"] = fd
    _CAMPAIGN_LOCK_STATE["path"] = str(lock_path)
    return {
        "passed": True,
        "detail": "campaign lock acquired",
        "lock_path": str(lock_path),
        "lock_payload": payload,
    }


def _attached_campaign_report_path(lock_result: dict[str, object]) -> Path | None:
    if bool(lock_result.get("passed", False)):
        return None
    payload = lock_result.get("lock_payload", {})
    if not isinstance(payload, dict):
        return None
    report_value = str(payload.get("report_path", "")).strip()
    if not report_value:
        return None
    path = Path(report_value)
    if not path.exists():
        return None
    return path


def _child_improvement_reports_dir(report_path: Path) -> Path:
    resolved = report_path.resolve()
    parent = resolved.parent
    if parent.name == "reports":
        return parent.parent / "improvement_reports"
    return parent / "improvement_reports"


def _release_campaign_lock(lock_path: Path | None, *, pid: int) -> None:
    if lock_path is None:
        return
    fd = _CAMPAIGN_LOCK_STATE.get("fd")
    path = _CAMPAIGN_LOCK_STATE.get("path")
    if not isinstance(fd, int) or fd < 0 or path != str(lock_path):
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    _CAMPAIGN_LOCK_STATE["fd"] = None
    _CAMPAIGN_LOCK_STATE["path"] = None
    if not lock_path.exists():
        return
    payload = _read_lock_payload(lock_path)
    if int(payload.get("pid", 0) or 0) not in {0, pid}:
        return
    try:
        lock_path.unlink()
    except OSError:
        return


def _refresh_campaign_lock(
    lock_path: Path | None,
    *,
    pid: int,
    report_path: Path,
    status_path: Path | None,
    payload: dict[str, object],
) -> None:
    fd = _CAMPAIGN_LOCK_STATE.get("fd")
    path = _CAMPAIGN_LOCK_STATE.get("path")
    if lock_path is None or not isinstance(fd, int) or fd < 0 or path != str(lock_path):
        return
    existing = _read_lock_payload(lock_path)
    if int(existing.get("pid", 0) or 0) not in {0, pid}:
        return
    refreshed = {
        **existing,
        "pid": pid,
        "report_path": str(report_path),
        "status_path": "" if status_path is None else str(status_path),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "status": str(payload.get("status", "")).strip(),
        "phase": str(payload.get("phase", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
    }
    try:
        _write_lock_payload(fd, refreshed)
    except OSError:
        return


def _emergency_persist_path(report_path: Path) -> Path:
    return Path("/tmp/agentkernel_emergency_reports") / report_path.name


def _persist_emergency_state(
    report_path: Path,
    payload: dict[str, object],
    *,
    config: KernelConfig | None = None,
    status_path: Path | None = None,
    error: OSError | None = None,
) -> Path | None:
    emergency_path = _emergency_persist_path(report_path)
    emergency_payload = dict(payload)
    emergency_payload["emergency_persist"] = {
        "original_report_path": str(report_path),
        "emergency_report_path": str(emergency_path),
        "status_path": "" if status_path is None else str(status_path),
        "error": "" if error is None else str(error),
        "persisted_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        atomic_write_json(emergency_path, emergency_payload, config=config)
    except OSError:
        return None
    return emergency_path


def _persist_report_state(
    report_path: Path,
    payload: dict[str, object],
    *,
    config: KernelConfig | None = None,
    status_path: Path | None = None,
    lock_path: Path | None = None,
    external_lease_backend: str = "",
    external_lease_endpoint: str = "",
    external_lease_token: str = "",
) -> None:
    if not str(external_lease_backend).strip():
        external_lease_backend = str(_EXTERNAL_LEASE_STATE.get("backend", "")).strip()
    if not str(external_lease_endpoint).strip():
        external_lease_endpoint = str(_EXTERNAL_LEASE_STATE.get("endpoint", "")).strip()
    if not str(external_lease_token).strip():
        external_lease_token = str(_EXTERNAL_LEASE_STATE.get("token", "")).strip()
    try:
        _write_report(report_path, payload, config=config)
        _write_status(status_path, report_path=report_path, payload=payload, config=config)
        _refresh_campaign_lock(
            lock_path,
            pid=os.getpid(),
            report_path=report_path,
            status_path=status_path,
            payload=payload,
        )
        if str(external_lease_backend).strip() == "http" and str(external_lease_endpoint).strip():
            _refresh_external_campaign_lease(
                str(external_lease_endpoint).strip(),
                token=str(external_lease_token or ""),
                report_path=report_path,
                status_path=status_path,
                payload=payload,
            )
        return {"primary_report_path": str(report_path), "emergency_report_path": ""}
    except OSError as exc:
        emergency_path = _persist_emergency_state(
            report_path,
            payload,
            config=config,
            status_path=status_path,
            error=exc,
        )
        if emergency_path is not None:
            payload["emergency_report_path"] = str(emergency_path)
            try:
                _write_status(status_path, report_path=emergency_path, payload=payload, config=config)
            except OSError:
                pass
        return {
            "primary_report_path": str(report_path),
            "emergency_report_path": "" if emergency_path is None else str(emergency_path),
            "error": str(exc),
        }


def _count_completed_rounds(rounds: list[dict[str, object]]) -> int:
    total = 0
    for payload in rounds:
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")).strip() == "completed":
            total += 1
    return total


def _mark_round(
    round_payload: dict[str, object],
    *,
    status: str,
    phase: str,
    reason: str = "",
) -> None:
    round_payload["status"] = str(status).strip()
    round_payload["phase"] = str(phase).strip()
    if reason:
        round_payload["reason"] = str(reason).strip()


def _subsystem_from_progress_line(line: str) -> str:
    match = re.search(r"subsystem=([a-z_]+)", str(line).strip())
    if not match:
        return ""
    return str(match.group(1)).strip()


def _phase_from_progress_line(line: str) -> str:
    normalized = str(line).strip()
    match = re.search(r"phase=([A-Za-z0-9_:-]+)", normalized)
    if match:
        return str(match.group(1)).strip()
    if re.search(
        r"observe complete passed=\d+/\d+ pass_rate=[0-9.]+ generated_pass_rate=[0-9.]+",
        normalized,
    ):
        return "observe"
    if re.search(r"campaign \d+/\d+ select subsystem=[a-z_]+", normalized):
        return "campaign_select"
    if re.search(
        r"variant_search start subsystem=[a-z_]+ selected_variants=\d+ variant_ids=[A-Za-z0-9_,.-]+",
        normalized,
    ):
        return "variant_search"
    if re.search(
        r"variant generate (?:start|heartbeat|complete|failed) subsystem=[a-z_]+ variant=[A-Za-z0-9_:-]+",
        normalized,
    ):
        return "variant_generate"
    if re.search(
        r"variant generate complete subsystem=[a-z_]+ variant=[A-Za-z0-9_:-]+ artifact=\S+",
        normalized,
    ):
        return "variant_generate"
    return ""


def _task_from_progress_line(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if re.search(
        r"observe complete passed=\d+/\d+ pass_rate=[0-9.]+ generated_pass_rate=[0-9.]+",
        normalized,
    ) or re.search(r"campaign \d+/\d+ select subsystem=[a-z_]+", normalized) or re.search(
        r"finalize phase=[A-Za-z0-9_:-]+ subsystem=[a-z_]+",
        normalized,
    ) or re.search(
        r"variant_search start subsystem=[a-z_]+ selected_variants=\d+ variant_ids=[A-Za-z0-9_,.-]+",
        normalized,
    ) or re.search(
        r"variant generate (?:start|heartbeat|complete|failed) subsystem=[a-z_]+ variant=[A-Za-z0-9_:-]+",
        normalized,
    ):
        return {}
    match = re.search(
        r"task (?P<index>\d+)/(?P<total>\d+) (?P<task_id>\S+) family=(?P<family>[a-z_]+)",
        normalized,
    )
    if not match:
        return {}
    return {
        "index": int(match.group("index")),
        "total": int(match.group("total")),
        "task_id": str(match.group("task_id")).strip(),
        "family": str(match.group("family")).strip(),
    }


_UNATTENDED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS = 120.0
_UNATTENDED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS = 240.0
_UNATTENDED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS = 180.0
_UNATTENDED_CHILD_GENERATED_FAILURE_ACTIVE_GRACE_SECONDS = 180.0
_UNATTENDED_CHILD_GENERATED_FAILURE_COMPLETION_GRACE_SECONDS = 180.0
_UNATTENDED_CHILD_METRICS_FINALIZE_GRACE_SECONDS = 120.0
_UNATTENDED_CHILD_FINALIZE_GRACE_SECONDS = 120.0
_UNATTENDED_CHILD_PREVIEW_COMPLETION_GRACE_SECONDS = 120.0
_UNATTENDED_CHILD_HOLDOUT_MIN_PROGRESS_SAMPLES = 3
_UNATTENDED_CHILD_HOLDOUT_MIN_OBSERVED_TASKS = 5
_UNATTENDED_CHILD_HOLDOUT_MIN_OBSERVED_SECONDS = 30.0
_UNATTENDED_CHILD_HOLDOUT_RUNTIME_BUFFER_SECONDS = 120.0
_UNATTENDED_CHILD_HOLDOUT_RUNTIME_SAFETY_MULTIPLIER = 1.15
_UNATTENDED_CHILD_RUNTIME_EMERGENCY_MULTIPLIER = 2.0


def _child_runtime_extension_plan(*, last_progress_phase: str, current_task: object) -> tuple[str, float]:
    task = current_task if isinstance(current_task, dict) else {}
    task_index = int(task.get("index", 0) or 0)
    task_total = int(task.get("total", 0) or 0)
    phase = str(last_progress_phase).strip()
    task_phase = str(task.get("phase", "")).strip() or phase
    task_completed = task_total > 0 and task_index >= task_total
    if phase in {"variant_search", "variant_generate"}:
        return ("", 0.0)
    if phase in {"preview_baseline_complete", "preview_candidate_complete", "preview_complete"}:
        return ("preview_completion", _UNATTENDED_CHILD_PREVIEW_COMPLETION_GRACE_SECONDS)
    if phase == "generated_success" and task_total > 0 and task_index >= task_total:
        return ("generated_success_completion", _UNATTENDED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS)
    if task_phase == "generated_failure_seed":
        if task_completed:
            return (
                "generated_failure_seed_completion",
                _UNATTENDED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS,
            )
        return (
            "generated_failure_seed_active",
            _UNATTENDED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS,
        )
    if task_phase == "generated_failure":
        if task_completed:
            return ("generated_failure_completion", _UNATTENDED_CHILD_GENERATED_FAILURE_COMPLETION_GRACE_SECONDS)
        return ("generated_failure_active", _UNATTENDED_CHILD_GENERATED_FAILURE_ACTIVE_GRACE_SECONDS)
    if phase == "metrics_finalize":
        return ("metrics_finalize", _UNATTENDED_CHILD_METRICS_FINALIZE_GRACE_SECONDS)
    if phase == "finalize":
        return ("finalize", _UNATTENDED_CHILD_FINALIZE_GRACE_SECONDS)
    return ("", 0.0)


def _should_block_post_productive_runtime_grace(
    child_status: Mapping[str, object] | None,
    *,
    last_progress_phase: str,
    current_task: object,
) -> bool:
    payload = child_status if isinstance(child_status, Mapping) else {}
    active_cycle_progress = payload.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    partial_progress_summary = payload.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, Mapping):
        partial_progress_summary = {}
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False)) or int(
        partial_progress_summary.get("generated_success_completed_runs", 0) or 0
    ) > 0
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or int(
        payload.get("partial_productive_runs", 0) or 0
    ) > 0
    decision_records_considered, _ = _effective_live_decision_record_count(payload)
    if not generated_success_completed or not productive_partial or decision_records_considered > 0:
        return False
    task = current_task if isinstance(current_task, Mapping) else {}
    task_phase = str(task.get("phase", "")).strip()
    phase = str(last_progress_phase).strip() or task_phase
    return phase in {
        "generated_failure",
        "campaign_select",
        "variant_search",
        "variant_generate",
    } or task_phase == "generated_failure"


def _child_runtime_extension_seconds(*, last_progress_phase: str, current_task: object) -> float:
    return _child_runtime_extension_plan(
        last_progress_phase=last_progress_phase,
        current_task=current_task,
    )[1]


def _runtime_grace_marker(*, grace_key: str, progress_epoch: int) -> tuple[str, int] | None:
    key = str(grace_key).strip()
    epoch = int(progress_epoch or 0)
    if not key or epoch <= 0:
        return None
    return (key, epoch)


def _prefer_mirrored_progress_phase(existing: object, mirrored: object) -> bool:
    existing_phase = str(existing or "").strip()
    mirrored_phase = str(mirrored or "").strip()
    if not mirrored_phase:
        return False
    if not existing_phase:
        return True
    if existing_phase == mirrored_phase:
        return False
    if existing_phase in {"variant_search", "variant_generate"} and mirrored_phase not in {
        "variant_search",
        "variant_generate",
    }:
        return False
    if mirrored_phase in {"variant_search", "variant_generate"} and existing_phase not in {
        "variant_search",
        "variant_generate",
    }:
        return True
    if existing_phase.startswith("generated_") and not mirrored_phase.startswith("generated_"):
        return False
    return True


def _holdout_progress_budget_summary(
    progress_samples: list[dict[str, float | int]],
) -> dict[str, object]:
    if len(progress_samples) < _UNATTENDED_CHILD_HOLDOUT_MIN_PROGRESS_SAMPLES:
        return {}
    first = progress_samples[0]
    last = progress_samples[-1]
    first_index = int(first.get("index", 0) or 0)
    last_index = int(last.get("index", 0) or 0)
    total = int(last.get("total", 0) or 0)
    started_at = float(first.get("timestamp", 0.0) or 0.0)
    last_at = float(last.get("timestamp", 0.0) or 0.0)
    observed_tasks = max(0, last_index - first_index)
    observed_seconds = max(0.0, last_at - started_at)
    if (
        total <= 0
        or observed_tasks < _UNATTENDED_CHILD_HOLDOUT_MIN_OBSERVED_TASKS
        or observed_seconds < _UNATTENDED_CHILD_HOLDOUT_MIN_OBSERVED_SECONDS
    ):
        return {}
    seconds_per_task = observed_seconds / float(observed_tasks)
    remaining_tasks = max(0, total - last_index)
    projected_remaining_seconds = remaining_tasks * seconds_per_task
    elapsed_seconds = max(0.0, last_at - started_at)
    projected_total_seconds = (
        elapsed_seconds + projected_remaining_seconds
    ) * _UNATTENDED_CHILD_HOLDOUT_RUNTIME_SAFETY_MULTIPLIER + _UNATTENDED_CHILD_HOLDOUT_RUNTIME_BUFFER_SECONDS
    return {
        "phase": "holdout_eval",
        "started_at": started_at,
        "last_progress_at": last_at,
        "observed_tasks": observed_tasks,
        "observed_seconds": observed_seconds,
        "seconds_per_task": seconds_per_task,
        "completed_tasks": last_index,
        "total_tasks": total,
        "remaining_tasks": remaining_tasks,
        "projected_remaining_seconds": projected_remaining_seconds,
        "projected_total_seconds": projected_total_seconds,
        "projected_completion_seconds": started_at + projected_total_seconds,
    }


def _holdout_budget_fit(
    summary: dict[str, object],
    *,
    started_at: float,
    max_runtime_seconds: float,
    emergency_multiplier: float = _UNATTENDED_CHILD_RUNTIME_EMERGENCY_MULTIPLIER,
) -> dict[str, object]:
    if not summary:
        return {"status": "unknown", "detail": "insufficient holdout progress samples"}
    emergency_ceiling_seconds = (
        max(0.0, float(max_runtime_seconds)) * max(1.0, float(emergency_multiplier))
        if max_runtime_seconds > 0.0
        else 0.0
    )
    projected_total_seconds = float(summary.get("projected_total_seconds", 0.0) or 0.0)
    if emergency_ceiling_seconds <= 0.0:
        return {
            "status": "unbounded",
            "projected_total_seconds": projected_total_seconds,
            "emergency_ceiling_seconds": 0.0,
            "detail": "no runtime emergency ceiling configured",
        }
    if projected_total_seconds <= emergency_ceiling_seconds:
        return {
            "status": "within_budget",
            "projected_total_seconds": projected_total_seconds,
            "emergency_ceiling_seconds": emergency_ceiling_seconds,
            "headroom_seconds": max(0.0, emergency_ceiling_seconds - projected_total_seconds),
            "detail": "projected holdout completion fits within emergency runtime ceiling",
        }
    return {
        "status": "over_budget",
        "projected_total_seconds": projected_total_seconds,
        "emergency_ceiling_seconds": emergency_ceiling_seconds,
        "excess_seconds": max(0.0, projected_total_seconds - emergency_ceiling_seconds),
        "detail": "projected holdout completion exceeds emergency runtime ceiling",
    }


def _adaptive_child_progress_state(
    *,
    last_progress_phase: str,
    active_finalize_phase: str,
    last_progress_at: float,
    current_task: Mapping[str, object] | None = None,
    current_cognitive_stage: Mapping[str, object] | None = None,
    observe_summary: Mapping[str, object] | None = None,
    pending_decision_state: str = "",
    preview_state: str = "",
    current_task_verification_passed: bool | None = None,
    progress_samples: list[dict[str, float | int]],
    started_at: float,
    now: float,
    max_runtime_seconds: float,
    max_progress_stall_seconds: float,
) -> dict[str, object]:
    phase = str(last_progress_phase).strip()
    finalize_phase = str(active_finalize_phase).strip()
    task_payload = current_task if isinstance(current_task, Mapping) else {}
    cognitive_payload = current_cognitive_stage if isinstance(current_cognitive_stage, Mapping) else {}
    current_phase = str(task_payload.get("phase", "")).strip()
    phase_for_state = phase
    if current_phase.startswith("generated_success"):
        phase_for_state = current_phase
    elif finalize_phase and finalize_phase != "holdout_eval":
        phase_for_state = finalize_phase
    elif current_phase:
        phase_for_state = current_phase
    holdout_phase_active = finalize_phase == "holdout_eval"
    if not holdout_phase_active and phase != "holdout_eval":
        state = semantic_progress_state(
            phase=phase_for_state,
            now=now,
            started_at=started_at,
            last_progress_at=float(last_progress_at or started_at),
            max_progress_stall_seconds=max_progress_stall_seconds,
            max_runtime_seconds=max_runtime_seconds,
            current_task=task_payload,
            observe_summary=observe_summary,
            pending_decision_state=pending_decision_state,
            preview_state=preview_state,
            current_task_verification_passed=current_task_verification_passed,
        )
        if _cognitive_stage_failed_verification(cognitive_payload):
            step_index = int(cognitive_payload.get("step_index", 0) or 0)
            phase_family = str(state.get("phase_family", "")).strip() or "active"
            state["status"] = "active"
            state["progress_class"] = "degraded"
            state["detail"] = (
                f"{phase_family} verification failed on step {step_index} and is still awaiting recovery"
                if step_index > 0
                else f"{phase_family} verification failed and is still awaiting recovery"
            )
        return state
    summary = _holdout_progress_budget_summary(progress_samples)
    if not summary:
        state = semantic_progress_state(
            phase="holdout_eval",
            now=now,
            started_at=started_at,
            last_progress_at=started_at,
            max_progress_stall_seconds=max_progress_stall_seconds,
            max_runtime_seconds=max_runtime_seconds,
            current_task=task_payload,
            observe_summary=observe_summary,
            pending_decision_state=pending_decision_state,
            preview_state=preview_state,
            current_task_verification_passed=current_task_verification_passed,
        )
        holdout_subphase = phase if phase and phase != "holdout_eval" else ""
        state["detail"] = (
            f"collecting holdout throughput samples during {holdout_subphase}"
            if holdout_subphase
            else "collecting holdout throughput samples"
        )
        if holdout_subphase:
            state["holdout_subphase"] = holdout_subphase
        return state
    fit = _holdout_budget_fit(
        summary,
        started_at=started_at,
        max_runtime_seconds=max_runtime_seconds,
    )
    last_progress_at = float(summary.get("last_progress_at", 0.0) or 0.0)
    progress_silence_seconds = max(0.0, now - last_progress_at)
    stall_threshold = max(0.0, float(max_progress_stall_seconds))
    progress_class = "healthy"
    status = str(fit.get("status", "unknown")).strip() or "unknown"
    if stall_threshold > 0.0 and progress_silence_seconds >= stall_threshold:
        progress_class = "stuck"
    elif status == "over_budget":
        progress_class = "degraded"
    detail = (
        "holdout progressing within projected budget"
        if progress_class == "healthy"
        else "holdout is still progressing but projected budget is over emergency ceiling"
        if progress_class == "degraded"
        else "holdout has stopped making meaningful progress"
        if progress_class == "stuck"
        else "adaptive progress classification unavailable"
    )
    return {
        **semantic_progress_state(
            phase="holdout_eval",
            now=now,
            started_at=started_at,
            last_progress_at=last_progress_at,
            max_progress_stall_seconds=max_progress_stall_seconds,
            max_runtime_seconds=max_runtime_seconds,
            current_task=task_payload,
            observe_summary=observe_summary,
            pending_decision_state=pending_decision_state,
            preview_state=preview_state,
            current_task_verification_passed=current_task_verification_passed,
        ),
        "status": status,
        "progress_class": progress_class,
        "progress_silence_seconds": progress_silence_seconds,
        "decision_distance": "near",
        "detail": detail,
        "budget_fit": fit,
        "holdout_budget": summary,
        **({"holdout_subphase": phase} if phase and phase != "holdout_eval" else {}),
    }


def _adaptive_progress_state_signature(state: Mapping[str, object]) -> str:
    payload = dict(state)
    payload.pop("progress_silence_seconds", None)
    return json.dumps(payload, sort_keys=True)


def _progress_state_task_position(state: Mapping[str, object] | None) -> tuple[int, int]:
    payload = state if isinstance(state, Mapping) else {}
    detail = str(payload.get("detail", "")).strip()
    if not detail:
        return (0, 0)
    match = re.search(r"\btask (?P<index>\d+)/(?P<total>\d+)\b", detail)
    if not match:
        return (0, 0)
    return (int(match.group("index")), int(match.group("total")))


def _progress_state_matches_live_context(
    state: Mapping[str, object] | None,
    *,
    current_task: Mapping[str, object] | None,
    last_progress_phase: str,
) -> bool:
    payload = state if isinstance(state, Mapping) else {}
    if not payload:
        return False
    task_payload = current_task if isinstance(current_task, Mapping) else {}
    live_phase = str(task_payload.get("phase", "")).strip() or str(last_progress_phase).strip()
    progress_phase = str(payload.get("phase", "")).strip()
    if live_phase and progress_phase and progress_phase != live_phase:
        return False
    current_task_index = int(task_payload.get("index", 0) or 0)
    current_task_total = int(task_payload.get("total", 0) or 0)
    progress_position = _progress_state_task_position(payload)
    if (
        current_task_index > 0
        and progress_position != (0, 0)
        and progress_position != (current_task_index, current_task_total)
    ):
        return False
    return True


def _progress_state_freshness(state: Mapping[str, object] | None) -> float:
    payload = state if isinstance(state, Mapping) else {}
    recorded_at = _semantic_state_recorded_at(payload)
    if recorded_at > 0.0:
        return recorded_at
    try:
        return float(payload.get("runtime_elapsed_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _select_preferred_child_progress_state(
    *,
    semantic_state: Mapping[str, object] | None,
    adaptive_state: Mapping[str, object] | None,
    current_task: Mapping[str, object] | None,
    last_progress_phase: str,
) -> dict[str, object]:
    semantic = dict(semantic_state) if isinstance(semantic_state, Mapping) else {}
    adaptive = dict(adaptive_state) if isinstance(adaptive_state, Mapping) else {}
    if not adaptive:
        return semantic
    if not semantic:
        return adaptive

    task_payload = current_task if isinstance(current_task, Mapping) else {}
    current_task_phase = str(task_payload.get("phase", "")).strip()
    live_phase = current_task_phase or str(last_progress_phase).strip()
    semantic_phase = str(semantic.get("phase", "")).strip()
    adaptive_phase = str(adaptive.get("phase", "")).strip()
    semantic_progress_class = str(semantic.get("progress_class", "")).strip()
    adaptive_progress_class = str(adaptive.get("progress_class", "")).strip()
    adaptive_matches_live_phase = bool(live_phase) and adaptive_phase == live_phase
    semantic_matches_live_phase = bool(live_phase) and semantic_phase == live_phase
    adaptive_matches_live_task = _progress_state_matches_live_context(
        adaptive,
        current_task=task_payload,
        last_progress_phase=last_progress_phase,
    )
    semantic_matches_live_task = _progress_state_matches_live_context(
        semantic,
        current_task=task_payload,
        last_progress_phase=last_progress_phase,
    )
    adaptive_freshness = _progress_state_freshness(adaptive)
    semantic_freshness = _progress_state_freshness(semantic)
    adaptive_is_newer = adaptive_freshness > semantic_freshness + 1e-6
    adaptive_is_healthier = adaptive_progress_class not in {"stuck", "degraded"} and semantic_progress_class in {
        "stuck",
        "degraded",
    }
    if adaptive_matches_live_task and not semantic_matches_live_task:
        return adaptive
    if semantic_matches_live_task and not adaptive_matches_live_task:
        return semantic
    if adaptive_matches_live_phase and adaptive_matches_live_task and (
        not semantic_matches_live_phase
        or not semantic_matches_live_task
        or adaptive_is_newer
        or adaptive_is_healthier
    ):
        return adaptive
    return semantic


def _normalize_excluded_subsystems(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def _normalize_subsystem_cooldowns(values: object) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in values.items():
        token = str(key).strip()
        if not token:
            continue
        try:
            rounds = int(value or 0)
        except (TypeError, ValueError):
            continue
        if rounds > 0:
            normalized[token] = rounds
    return normalized


def _materialize_excluded_subsystems(policy: dict[str, object]) -> dict[str, object]:
    next_policy = dict(policy)
    cooldowns = _normalize_subsystem_cooldowns(next_policy.get("subsystem_cooldowns", {}))
    exclusions = _normalize_excluded_subsystems(next_policy.get("excluded_subsystems", []))
    for subsystem in cooldowns:
        if subsystem not in exclusions:
            exclusions.append(subsystem)
    next_policy["subsystem_cooldowns"] = cooldowns
    next_policy["excluded_subsystems"] = exclusions
    return next_policy


def _advance_subsystem_cooldowns(policy: dict[str, object]) -> dict[str, object]:
    next_policy = dict(policy)
    cooldowns = _normalize_subsystem_cooldowns(next_policy.get("subsystem_cooldowns", {}))
    decremented = {
        subsystem: rounds - 1
        for subsystem, rounds in cooldowns.items()
        if int(rounds) - 1 > 0
    }
    next_policy["subsystem_cooldowns"] = decremented
    next_policy["excluded_subsystems"] = []
    return _materialize_excluded_subsystems(next_policy)


def _apply_subsystem_cooldown(
    policy: dict[str, object],
    *,
    subsystem: str,
    rounds: int = 2,
) -> dict[str, object]:
    next_policy = dict(policy)
    token = str(subsystem).strip()
    cooldowns = _normalize_subsystem_cooldowns(next_policy.get("subsystem_cooldowns", {}))
    if token:
        cooldowns[token] = max(int(cooldowns.get(token, 0) or 0), max(1, int(rounds)))
    next_policy["subsystem_cooldowns"] = cooldowns
    return _materialize_excluded_subsystems(next_policy)


def _resume_policy_from_partial_round(
    current_policy: dict[str, object],
    *,
    prior_rounds: list[dict[str, object]],
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
) -> dict[str, object]:
    next_policy = _advance_subsystem_cooldowns(current_policy)
    if not prior_rounds:
        return _materialize_excluded_subsystems(next_policy)
    last_round = prior_rounds[-1]
    if not isinstance(last_round, dict):
        return _materialize_excluded_subsystems(next_policy)
    if str(last_round.get("status", "")).strip() not in {"running", "interrupted"}:
        return _materialize_excluded_subsystems(next_policy)
    active_child = last_round.get("active_child", {})
    phase_detail = str(last_round.get("phase_detail", "")).strip()
    progress_line = ""
    if isinstance(active_child, dict):
        progress_line = str(
            active_child.get("last_progress_line") or active_child.get("last_output_line") or ""
        ).strip()
    stalled_subsystem = _subsystem_from_progress_line(progress_line or phase_detail)
    if stalled_subsystem:
        next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "balanced"
        next_policy["cycles"] = min(max_cycles, max(1, int(next_policy.get("cycles", 1) or 1)) + 1)
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(1, int(next_policy.get("campaign_width", 1) or 1)) + 1,
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(1, int(next_policy.get("task_limit", 64) or 64)) * 2,
        )
    return _materialize_excluded_subsystems(next_policy)


def _stalled_subsystem_from_round(round_payload: dict[str, object] | None) -> str:
    if not isinstance(round_payload, dict):
        return ""
    active_child = round_payload.get("active_child", {})
    phase_detail = str(round_payload.get("phase_detail", "")).strip()
    progress_line = ""
    if isinstance(active_child, dict):
        progress_line = str(
            active_child.get("last_subsystem_progress_line")
            or active_child.get("last_progress_line")
            or active_child.get("last_output_line")
            or ""
        ).strip()
    return _subsystem_from_progress_line(progress_line or phase_detail)


def _child_failure_recovery_policy(
    current_policy: dict[str, object],
    *,
    round_payload: dict[str, object] | None,
    phase: str,
    reason: str = "",
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
) -> dict[str, object]:
    next_policy = _advance_subsystem_cooldowns(current_policy)
    stalled_subsystem = _stalled_subsystem_from_round(round_payload)
    normalized_reason = str(reason).strip().lower()
    dominant_zero_yield_subsystems: list[str] = []
    cooldown_rounds_by_subsystem: dict[str, int] = {}
    if isinstance(round_payload, dict):
        campaign_report = round_payload.get("campaign_report", {})
        if isinstance(campaign_report, dict):
            subsystem_signal = _subsystem_signal(campaign_report)
            dominant_zero_yield_subsystems = _normalize_excluded_subsystems(
                subsystem_signal.get("zero_yield_dominant_subsystems", [])
            )
            raw_cooldown_rounds = subsystem_signal.get("cooldown_rounds_by_subsystem", {})
            if isinstance(raw_cooldown_rounds, dict):
                cooldown_rounds_by_subsystem = {
                    str(subsystem).strip(): max(2, int(value or 0))
                    for subsystem, value in raw_cooldown_rounds.items()
                    if str(subsystem).strip()
                }
    cooldown_targets: list[str] = []
    if stalled_subsystem:
        cooldown_targets.append(stalled_subsystem)
    for subsystem in dominant_zero_yield_subsystems:
        if subsystem and subsystem not in cooldown_targets:
            cooldown_targets.append(subsystem)
    if "campaign report showed no runtime-managed decisions" in normalized_reason and not cooldown_targets:
        if isinstance(round_payload, dict):
            campaign_report = round_payload.get("campaign_report", {})
            if isinstance(campaign_report, dict):
                subsystem_signal = _subsystem_signal(campaign_report)
                for subsystem in _normalize_excluded_subsystems(subsystem_signal.get("cooldown_candidates", [])):
                    if subsystem and subsystem not in cooldown_targets:
                        cooldown_targets.append(subsystem)
    for subsystem in cooldown_targets[:2]:
        next_policy = _apply_subsystem_cooldown(
            next_policy,
            subsystem=subsystem,
            rounds=max(2, int(cooldown_rounds_by_subsystem.get(subsystem, 2) or 2)),
        )
    next_policy["adaptive_search"] = True
    next_policy["focus"] = "recovery_alignment"
    next_policy["cycles"] = min(max_cycles, max(1, int(next_policy.get("cycles", 1) or 1)) + 1)
    next_policy["task_limit"] = min(
        max_task_limit,
        max(1, int(next_policy.get("task_limit", 64) or 64)) * 2,
    )
    if str(phase).strip() == "liftoff":
        next_policy["variant_width"] = min(
            max_variant_width,
            max(1, int(next_policy.get("variant_width", 1) or 1)) + 1,
        )
    else:
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(1, int(next_policy.get("campaign_width", 1) or 1)) + 1,
        )
    return _materialize_excluded_subsystems(next_policy)


def _directory_usage_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        walker = os.walk(path, onerror=lambda exc: None)
    except OSError:
        return 0
    try:
        for root, _, files in walker:
            root_path = Path(root)
            for name in files:
                current = root_path / name
                try:
                    if current.is_file():
                        total += current.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _shallow_directory_usage_bytes(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    try:
        for child in path.iterdir():
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _top_level_entry_usage_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "bytes": 0,
            "partial": False,
            "method": "missing",
        }
    try:
        if path.is_file():
            return {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "partial": False,
                "method": "stat",
            }
        if path.is_dir():
            return {
                "path": str(path),
                "bytes": _shallow_directory_usage_bytes(path),
                "partial": True,
                "method": "shallow_top_level",
            }
    except OSError:
        return {
            "path": str(path),
            "bytes": 0,
            "partial": True,
            "method": "shallow_error",
        }
    return {
        "path": str(path),
        "bytes": 0,
        "partial": True,
        "method": "unknown",
    }


def _path_usage_snapshot(path: Path, *, timeout_seconds: float = 5.0) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "bytes": 0,
            "partial": False,
            "method": "missing",
        }
    try:
        completed = subprocess.run(
            ["du", "-sx", "--block-size=1", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(0.1, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired:
        return {
            "path": str(path),
            "bytes": _shallow_directory_usage_bytes(path),
            "partial": True,
            "method": "shallow_timeout",
        }
    except OSError:
        return {
            "path": str(path),
            "bytes": _shallow_directory_usage_bytes(path),
            "partial": True,
            "method": "shallow_fallback",
        }
    if completed.returncode != 0:
        return {
            "path": str(path),
            "bytes": _shallow_directory_usage_bytes(path),
            "partial": True,
            "method": "shallow_nonzero",
        }
    line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    parts = line.split()
    try:
        size = int(parts[0])
    except (IndexError, ValueError):
        return {
            "path": str(path),
            "bytes": _shallow_directory_usage_bytes(path),
            "partial": True,
            "method": "shallow_parse_error",
        }
    return {
        "path": str(path),
        "bytes": size,
        "partial": False,
        "method": "du",
    }


def _storage_governance_snapshot(config: KernelConfig) -> dict[str, object]:
    tracked = {
        "reports": config.improvement_reports_dir,
        "candidates": config.candidate_artifacts_root,
        "checkpoints": config.run_checkpoints_dir,
        "snapshots": config.unattended_workspace_snapshot_root,
        "tolbert_datasets": config.tolbert_supervised_datasets_dir,
        "tolbert_promoted_checkpoints": config.tolbert_model_artifact_path.parent / "checkpoints",
    }
    sizes = {name: _path_usage_snapshot(path) for name, path in tracked.items()}
    return {
        "tracked_directories": sizes,
        "total_bytes": sum(int(item["bytes"]) for item in sizes.values()),
    }


def _path_usage_bytes(path: Path, *, timeout_seconds: float = 5.0) -> int:
    snapshot = _path_usage_snapshot(path, timeout_seconds=timeout_seconds)
    try:
        return int(snapshot.get("bytes", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _load_global_storage_policy(policy_path: Path | None) -> list[dict[str, object]]:
    if policy_path is None or not policy_path.exists():
        return []
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        path_value = str(item.get("path", "")).strip()
        if not path_value:
            continue
        normalized.append(
            {
                "label": str(item.get("label", Path(path_value).name)).strip() or Path(path_value).name,
                "path": path_value,
                "keep_entries": max(0, int(item.get("keep_entries", item.get("keep", 0)) or 0)),
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return normalized


def _global_storage_snapshot(
    root: Path,
    *,
    top_k: int,
    managed_entries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    top_consumers: list[dict[str, object]] = []
    if max(0, int(top_k)) > 0 and root.exists() and root.is_dir():
        sized_children: list[dict[str, object]] = []
        for child in root.iterdir():
            try:
                snapshot = _top_level_entry_usage_snapshot(child)
            except OSError:
                continue
            sized_children.append(
                {
                    "path": child,
                    "bytes": int(snapshot.get("bytes", 0) or 0),
                    "partial": bool(snapshot.get("partial", False)),
                    "method": str(snapshot.get("method", "")),
                }
            )
        sized_children.sort(key=lambda item: int(item["bytes"]), reverse=True)
        for item in sized_children[: max(0, int(top_k))]:
            top_consumers.append(
                {
                    "path": str(item["path"]),
                    "bytes": int(item["bytes"]),
                    "partial": bool(item.get("partial", False)),
                    "method": str(item.get("method", "")),
                }
            )
    managed = []
    for entry in managed_entries or []:
        if not bool(entry.get("enabled", True)):
            continue
        path = Path(str(entry.get("path", "")).strip())
        snapshot = _path_usage_snapshot(path)
        managed.append(
            {
                "label": str(entry.get("label", path.name)).strip() or path.name,
                "path": str(path),
                "keep_entries": int(entry.get("keep_entries", 0) or 0),
                "bytes": int(snapshot.get("bytes", 0) or 0),
                "partial": bool(snapshot.get("partial", False)),
                "method": str(snapshot.get("method", "")),
            }
        )
    return {
        "root": str(root),
        "top_consumers": top_consumers,
        "managed_entries": managed,
    }


def _cleanup_global_storage_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    removed_entries: list[dict[str, object]] = []
    for entry in entries:
        if not bool(entry.get("enabled", True)):
            continue
        path = Path(str(entry.get("path", "")).strip())
        removed = _prune_directory_entries(path, keep=max(0, int(entry.get("keep_entries", 0) or 0)))
        if removed:
            removed_entries.append(
                {
                    "label": str(entry.get("label", path.name)).strip() or path.name,
                    "path": str(path),
                    "removed": removed,
                }
            )
    return removed_entries


def _governed_global_storage_cleanup(
    root: Path,
    *,
    policy_entries: list[dict[str, object]],
    target_free_gib: float,
    top_k: int,
) -> dict[str, object]:
    before_disk = _disk_preflight(root, min_free_gib=0.0)
    target_free = max(0.0, float(target_free_gib))
    if float(before_disk.get("free_gib", 0.0)) >= target_free:
        return {
            "before_disk": before_disk,
            "after_disk": before_disk,
            "before_snapshot": {},
            "after_snapshot": {},
            "cleanup": {
                "removed_entries": [],
                "skipped": True,
                "reason": "disk already above target",
                "target_free_gib": target_free,
            },
        }
    before_snapshot = _global_storage_snapshot(root, top_k=top_k, managed_entries=policy_entries)
    cleanup: dict[str, object] = {"removed_entries": []}
    after_disk = before_disk
    after_snapshot = before_snapshot
    if policy_entries:
        cleanup = {
            "removed_entries": _cleanup_global_storage_entries(policy_entries),
        }
        after_disk = _disk_preflight(root, min_free_gib=0.0)
        after_snapshot = _global_storage_snapshot(root, top_k=top_k, managed_entries=policy_entries)
    return {
        "before_disk": before_disk,
        "after_disk": after_disk,
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
        "cleanup": cleanup,
    }


def _run_alert_command(
    command: str,
    *,
    report_path: Path,
    status_path: Path | None,
    event_log_path: Path | None,
    payload: dict[str, object],
) -> dict[str, object]:
    normalized = str(command).strip()
    if not normalized:
        return {"ran": False, "detail": "alert command not configured"}
    env = dict(os.environ)
    env.update(
        {
            "AGENT_KERNEL_CAMPAIGN_REPORT": str(report_path),
            "AGENT_KERNEL_CAMPAIGN_STATUS_PATH": "" if status_path is None else str(status_path),
            "AGENT_KERNEL_CAMPAIGN_EVENT_LOG": "" if event_log_path is None else str(event_log_path),
            "AGENT_KERNEL_CAMPAIGN_STATE": str(payload.get("status", "")).strip(),
            "AGENT_KERNEL_CAMPAIGN_PHASE": str(payload.get("phase", "")).strip(),
            "AGENT_KERNEL_CAMPAIGN_REASON": str(payload.get("reason", "")).strip(),
            "AGENT_KERNEL_CAMPAIGN_ROUNDS_COMPLETED": str(int(payload.get("rounds_completed", 0) or 0)),
        }
    )
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", normalized],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {
            "ran": True,
            "command": normalized,
            "returncode": -9,
            "stdout": "",
            "stderr": "alert command timed out after 30 seconds",
        }
    return {
        "ran": True,
        "command": normalized,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _campaign_alert_summary(payload: dict[str, object]) -> str:
    summary = (
        f"agentkernel unattended campaign status={str(payload.get('status', '')).strip()} "
        f"phase={str(payload.get('phase', '')).strip()} "
        f"rounds_completed={int(payload.get('rounds_completed', 0) or 0)} "
        f"reason={str(payload.get('reason', '')).strip() or 'n/a'}"
    )
    if str(payload.get("phase", "")).strip() == "policy_shift":
        rationale = payload.get("policy_shift_rationale", {})
        reason_codes = rationale.get("reason_codes", []) if isinstance(rationale, dict) else []
        if isinstance(reason_codes, list):
            normalized = [str(item).strip() for item in reason_codes if str(item).strip()]
            if normalized:
                summary += f" rationale={','.join(normalized[:4])}"
    return summary


def _policy_shift_alert_reason(rationale: dict[str, object]) -> str:
    if not isinstance(rationale, dict):
        return ""
    reason_codes = rationale.get("reason_codes", [])
    if not isinstance(reason_codes, list):
        reason_codes = []
    codes = {str(item).strip() for item in reason_codes if str(item).strip()}
    if "campaign_breadth_pressure" in codes or "variant_breadth_pressure" in codes:
        return "supervisor widened due to breadth pressure"
    if "no_yield_round" in codes:
        return "supervisor widened after a no-yield round"
    if "safety_regression" in codes or "phase_gate_failure" in codes:
        return "supervisor tightened policy due to regression or failed gates"
    if "stalled_subsystem_cooldown" in codes:
        return "supervisor cooled a stalled subsystem"
    return ""


def _semantic_redirection_summary(round_payload: dict[str, object]) -> dict[str, object]:
    rationale = round_payload.get("policy_shift_rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    reason_codes = [
        str(value).strip()
        for value in rationale.get("reason_codes", [])
        if str(value).strip()
    ]
    code_set = set(reason_codes)
    planner_pressure = rationale.get("planner_pressure", {})
    if not isinstance(planner_pressure, dict):
        planner_pressure = {}
    cooled_subsystems = [
        str(value).strip()
        for value in rationale.get("cooled_subsystems", [])
        if str(value).strip()
    ]
    missing_required_families = _normalize_benchmark_families(
        planner_pressure.get("missing_required_families", [])
    )
    campaign_report = round_payload.get("campaign_report", {})
    if not isinstance(campaign_report, dict):
        campaign_report = {}
    production_yield = campaign_report.get("production_yield_summary", {})
    if not isinstance(production_yield, dict):
        production_yield = {}
    priority_family_yield = campaign_report.get("priority_family_yield_summary", {})
    if not isinstance(priority_family_yield, dict):
        priority_family_yield = {}
    target_priority_families = _normalize_benchmark_families(
        rationale.get("priority_benchmark_families_after", [])
    )
    if not target_priority_families:
        target_priority_families = _normalize_benchmark_families(
            priority_family_yield.get("priority_families_needing_retained_gain_conversion", [])
        )
    if not target_priority_families:
        target_priority_families = _normalize_benchmark_families(
            priority_family_yield.get("priority_families_with_signal_but_no_retained_gain", [])
        )
    rejected_by_subsystem = production_yield.get("rejected_by_subsystem", {})
    if not isinstance(rejected_by_subsystem, dict):
        rejected_by_subsystem = {}
    dominant_rejected_subsystems: list[str] = []
    dominant_rejected_count = 0
    for raw_subsystem, raw_count in rejected_by_subsystem.items():
        subsystem = str(raw_subsystem).strip()
        if not subsystem:
            continue
        count = int(raw_count or 0)
        if count <= 0:
            continue
        if count > dominant_rejected_count:
            dominant_rejected_subsystems = [subsystem]
            dominant_rejected_count = count
        elif count == dominant_rejected_count and subsystem not in dominant_rejected_subsystems:
            dominant_rejected_subsystems.append(subsystem)
    low_return_priority_families = _normalize_benchmark_families(
        priority_family_yield.get("priority_families_with_signal_but_no_retained_gain", [])
    )
    stalled_subsystem = str(rationale.get("stalled_subsystem", "")).strip()
    if stalled_subsystem and stalled_subsystem not in cooled_subsystems:
        cooled_subsystems.append(stalled_subsystem)
    for subsystem in dominant_rejected_subsystems:
        if subsystem not in cooled_subsystems:
            cooled_subsystems.append(subsystem)
    triggered = bool(
        code_set
        & {
            "no_yield_round",
            "weak_retention_outcome",
            "stalled_subsystem_cooldown",
            "subsystem_reject_cooldown",
            "dominant_zero_yield_subsystem_cooldown",
            "subsystem_monoculture",
        }
    )
    semantic_hypotheses: list[str] = []
    if "no_yield_round" in code_set:
        semantic_hypotheses.append("force descendants to adopt a new parent attempt or subsystem before replay")
    if "subsystem_reject_cooldown" in code_set or "stalled_subsystem_cooldown" in code_set:
        semantic_hypotheses.append("redirect work away from the stagnant subsystem subtree until new evidence appears")
    if missing_required_families:
        semantic_hypotheses.append(
            f"bias continuation toward missing required families: {','.join(missing_required_families)}"
        )
    if low_return_priority_families:
        semantic_hypotheses.append(
            "require a new continuation parent for families without retained gain conversion: "
            + ",".join(low_return_priority_families)
        )
    if dominant_rejected_subsystems:
        semantic_hypotheses.append(
            "cool the dominant zero-yield subsystem and force cross-subsystem continuation: "
            + ",".join(dominant_rejected_subsystems)
        )
    if "subsystem_monoculture" in code_set:
        semantic_hypotheses.append("force cross-subsystem descendants to break monoculture")
    return {
        "triggered": triggered,
        "redirect_scope": "subtree" if cooled_subsystems else "campaign",
        "reason_codes": reason_codes,
        "source_subsystems": cooled_subsystems,
        "missing_required_families": missing_required_families,
        "target_priority_families": target_priority_families,
        "required_new_parent": bool(
            "no_yield_round" in code_set
            or "subsystem_reject_cooldown" in code_set
            or "stalled_subsystem_cooldown" in code_set
            or ("weak_retention_outcome" in code_set and bool(dominant_rejected_subsystems))
            or bool(low_return_priority_families)
        ),
        "semantic_hypotheses": semantic_hypotheses,
    }


def _recent_semantic_redirects(config: KernelConfig, *, limit: int = 12) -> list[dict[str, object]]:
    root = config.semantic_hub_root / "redirects"
    if not root.exists():
        return []
    redirects: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[: max(1, limit)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            redirects.append(payload)
    return redirects


def _semantic_policy_seed(config: KernelConfig) -> dict[str, object]:
    redirects = _recent_semantic_redirects(config)
    if not redirects:
        return {}
    subsystem_votes: dict[str, int] = {}
    priority_votes: dict[str, int] = {}
    require_new_parent = False
    adaptive_pressure = False
    focus = ""
    for redirect in redirects:
        reason_codes = {
            str(item).strip()
            for item in redirect.get("reason_codes", [])
            if str(item).strip()
        }
        if bool(redirect.get("required_new_parent", False)):
            require_new_parent = True
        if {"no_yield_round", "weak_retention_outcome", "subsystem_reject_cooldown"} & reason_codes:
            adaptive_pressure = True
        for subsystem in _normalize_excluded_subsystems(redirect.get("source_subsystems", [])):
            subsystem_votes[subsystem] = subsystem_votes.get(subsystem, 0) + 1
        for family in _normalize_benchmark_families(redirect.get("target_priority_families", [])):
            priority_votes[family] = priority_votes.get(family, 0) + 1
        if adaptive_pressure:
            focus = "discovered_task_adaptation"
    excluded_subsystems = [
        subsystem
        for subsystem, count in sorted(subsystem_votes.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:2]
    priority_benchmark_families = [
        family
        for family, _count in sorted(priority_votes.items(), key=lambda item: (-item[1], item[0]))
    ][:3]
    subsystem_cooldowns = {subsystem: 2 for subsystem in excluded_subsystems}
    seed: dict[str, object] = {}
    if excluded_subsystems:
        seed["excluded_subsystems"] = excluded_subsystems
        seed["subsystem_cooldowns"] = subsystem_cooldowns
    if priority_benchmark_families:
        seed["priority_benchmark_families"] = priority_benchmark_families
    if adaptive_pressure:
        seed["adaptive_search"] = True
    if require_new_parent:
        seed["required_new_parent"] = True
    return seed


def _apply_semantic_policy_seed(
    policy: Mapping[str, object] | None,
    *,
    semantic_seed: Mapping[str, object] | None,
) -> dict[str, object]:
    current = dict(policy) if isinstance(policy, Mapping) else {}
    seed = semantic_seed if isinstance(semantic_seed, Mapping) else {}
    if not seed:
        return current
    seeded = dict(current)
    seeded["excluded_subsystems"] = _normalize_excluded_subsystems(
        seed.get("excluded_subsystems", seeded.get("excluded_subsystems", []))
    )
    seeded["subsystem_cooldowns"] = (
        dict(seed.get("subsystem_cooldowns", {}))
        if isinstance(seed.get("subsystem_cooldowns", {}), dict)
        else dict(seeded.get("subsystem_cooldowns", {}))
        if isinstance(seeded.get("subsystem_cooldowns", {}), dict)
        else {}
    )
    priority_families = _normalize_benchmark_families(
        seed.get("priority_benchmark_families", seeded.get("priority_benchmark_families", []))
    )
    if priority_families:
        seeded["priority_benchmark_families"] = priority_families
    if "adaptive_search" in seed:
        seeded["adaptive_search"] = bool(seed.get("adaptive_search"))
    return seeded


def _round_semantic_skill_payload(
    *,
    round_payload: dict[str, object],
    current_policy: Mapping[str, object] | None,
) -> dict[str, object]:
    semantic_redirection = round_payload.get("semantic_redirection", {})
    if not isinstance(semantic_redirection, dict):
        semantic_redirection = {}
    campaign_report = round_payload.get("campaign_report", {})
    if not isinstance(campaign_report, dict):
        campaign_report = {}
    production_yield = campaign_report.get("production_yield_summary", {})
    if not isinstance(production_yield, dict):
        production_yield = {}
    priority_family_yield = campaign_report.get("priority_family_yield_summary", {})
    if not isinstance(priority_family_yield, dict):
        priority_family_yield = {}
    source_subsystems = _normalize_excluded_subsystems(semantic_redirection.get("source_subsystems", []))
    low_return_priority_families = _normalize_benchmark_families(
        priority_family_yield.get("priority_families_with_signal_but_no_retained_gain", [])
    )
    rejected_cycles = int(production_yield.get("rejected_cycles", 0) or 0)
    retained_cycles = int(production_yield.get("retained_cycles", 0) or 0)
    if not semantic_redirection and not low_return_priority_families and rejected_cycles <= retained_cycles:
        return {}
    dominant_subsystem = source_subsystems[0] if source_subsystems else ""
    reason_codes = [
        str(item).strip()
        for item in semantic_redirection.get("reason_codes", [])
        if str(item).strip()
    ]
    lesson_parts: list[str] = []
    if dominant_subsystem:
        lesson_parts.append(f"{dominant_subsystem} dominated rejected runtime-managed decisions")
    if low_return_priority_families:
        lesson_parts.append(
            "priority families showed signal without retained gain conversion: "
            + ",".join(low_return_priority_families)
        )
    if "weak_retention_outcome" in set(reason_codes):
        lesson_parts.append("candidate changes were observationally active but produced no verified retained gain")
    if not lesson_parts:
        return {}
    return {
        "subsystem": dominant_subsystem,
        "analysis_lesson": "; ".join(lesson_parts),
        "reuse_conditions": [
            "reuse when unattended startup should avoid a recently stagnant subsystem subtree",
            "reuse when priority families have signal but lack retained gain conversion",
        ],
        "avoid_conditions": [
            "avoid replaying the same subsystem-family combination without a new continuation parent",
            "avoid accepting neutral runtime-managed decisions as meaningful progress",
        ],
        "continuation_artifact_path": str(round_payload.get("campaign_report_path", "")).strip(),
        "reason_codes": reason_codes,
        "source_subsystems": source_subsystems,
        "priority_families": low_return_priority_families,
        "focus_before": str((current_policy or {}).get("focus", "")).strip(),
        "focus_after": str(dict(round_payload.get("next_policy", {})).get("focus", "")).strip()
        if isinstance(round_payload.get("next_policy", {}), dict)
        else "",
    }


def _normalize_policy_shift_alert_subscriptions(raw: object) -> list[str]:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw]
    else:
        values = []
    normalized: list[str] = []
    seen: set[str] = set()
    for token in values:
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(lowered)
    return normalized


def _policy_shift_alert_subscribed(rationale: dict[str, object], subscriptions: list[str]) -> bool:
    normalized = _normalize_policy_shift_alert_subscriptions(subscriptions)
    if not normalized or "all" in normalized:
        return True
    reason_codes = rationale.get("reason_codes", []) if isinstance(rationale, dict) else []
    if not isinstance(reason_codes, list):
        reason_codes = []
    codes = {str(item).strip().lower() for item in reason_codes if str(item).strip()}
    return any(token in codes for token in normalized)


def _policy_shift_alert_status_summary(alert: dict[str, object], *, rationale: dict[str, object]) -> str:
    if not isinstance(alert, dict):
        alert = {}
    if not isinstance(rationale, dict):
        rationale = {}
    reason_codes = rationale.get("reason_codes", [])
    if not isinstance(reason_codes, list):
        reason_codes = []
    normalized_codes = [str(item).strip() for item in reason_codes if str(item).strip()]
    rationale_suffix = f" ({','.join(normalized_codes[:3])})" if normalized_codes else ""
    if bool(alert.get("suppressed", False)):
        detail = str(alert.get("detail", "")).strip() or "suppressed"
        return f"suppressed: {detail}{rationale_suffix}"
    if bool(alert.get("ran", False)):
        severity = str(alert.get("severity", "")).strip()
        if severity:
            return f"sent: severity={severity}{rationale_suffix}"
        return f"sent{rationale_suffix}"
    detail = str(alert.get("detail", "")).strip()
    if detail:
        return f"not sent: {detail}{rationale_suffix}"
    return f"not sent{rationale_suffix}"


def _effective_policy_shift_alert_decision(
    *,
    payload: dict[str, object],
    rationale_round: dict[str, object] | None,
    subscriptions: list[str],
) -> dict[str, object]:
    round_payload = rationale_round if isinstance(rationale_round, dict) else {}
    stored = round_payload.get("policy_shift_alert", {})
    if isinstance(stored, dict) and stored:
        return stored
    rationale = round_payload.get("policy_shift_rationale", {})
    if not isinstance(rationale, dict) or not rationale:
        return {}
    if not _policy_shift_alert_subscribed(rationale, subscriptions):
        return {
            "ran": False,
            "suppressed": True,
            "detail": "policy shift alert not subscribed",
            "subscriptions": subscriptions,
        }
    alert_reason = _policy_shift_alert_reason(rationale)
    if not alert_reason:
        return {
            "ran": False,
            "detail": "policy shift did not meet alert threshold",
        }
    alerting = payload.get("alerting", {}) if isinstance(payload.get("alerting", {}), dict) else {}
    if not any(
        [
            str(alerting.get("alert_command", "")).strip(),
            str(alerting.get("slack_webhook_url", "")).strip(),
            str(alerting.get("pagerduty_routing_key", "")).strip(),
        ]
    ):
        return {
            "ran": False,
            "detail": "no alert transport configured",
        }
    return {}


def _alert_fingerprint(payload: dict[str, object]) -> str:
    parts = [
        str(payload.get("status", "")).strip(),
        str(payload.get("phase", "")).strip(),
        str(payload.get("reason", "")).strip(),
    ]
    return "|".join(parts)


def _read_alert_state(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _should_emit_alert(
    state_path: Path | None,
    *,
    payload: dict[str, object],
    rate_limit_seconds: float,
) -> tuple[bool, dict[str, object], int]:
    if state_path is None:
        return True, {}, 1
    state = _read_alert_state(state_path)
    fingerprint = _alert_fingerprint(payload)
    last_fingerprint = str(state.get("last_fingerprint", "")).strip()
    last_sent_at = float(state.get("last_sent_at_epoch", 0.0) or 0.0)
    repeat_count = int(state.get("repeat_count", 0) or 0)
    now = time.time()
    if fingerprint and fingerprint == last_fingerprint:
        repeat_count += 1
        if repeat_count in {3, 10} and (now - last_sent_at) >= min(60.0, max(1.0, float(rate_limit_seconds) / 4.0)):
            return True, state, repeat_count
        if (now - last_sent_at) < max(0.0, float(rate_limit_seconds)):
            return False, state, repeat_count
        return True, state, repeat_count
    return True, state, 1


def _record_alert_state(
    state_path: Path | None,
    *,
    payload: dict[str, object],
    repeat_count: int,
    config: KernelConfig | None = None,
) -> None:
    if state_path is None:
        return
    atomic_write_json(
        state_path,
        {
            "last_fingerprint": _alert_fingerprint(payload),
            "last_sent_at_epoch": time.time(),
            "last_sent_at": datetime.now(timezone.utc).isoformat(),
            "repeat_count": int(repeat_count),
            "status": str(payload.get("status", "")).strip(),
            "phase": str(payload.get("phase", "")).strip(),
            "reason": str(payload.get("reason", "")).strip(),
        },
        config=config,
    )


def _alert_severity(payload: dict[str, object], *, repeat_count: int) -> str:
    status = str(payload.get("status", "")).strip()
    if status in {"safe_stop", "interrupted"}:
        if repeat_count >= 10:
            return "critical"
        if repeat_count >= 3:
            return "error"
        return "warning"
    if repeat_count >= 10:
        return "warning"
    return "info"


def _send_slack_alert(webhook_url: str, *, payload: dict[str, object], report_path: Path, severity: str) -> dict[str, object]:
    body = {
        "text": f"[{severity.upper()}] {_campaign_alert_summary(payload)}",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*AgentKernel Campaign Alert ({severity.upper()})*\n{_campaign_alert_summary(payload)}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Report*\n{report_path}"},
                    {"type": "mrkdwn", "text": f"*Phase*\n{str(payload.get('phase', '')).strip() or 'unknown'}"},
                ],
            },
        ],
    }
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        status_code = int(getattr(response, "status", 200) or 200)
        response_body = response.read().decode("utf-8", errors="replace").strip()
    return {"ran": True, "provider": "slack", "status_code": status_code, "response": response_body}


def _send_pagerduty_alert(routing_key: str, *, payload: dict[str, object], report_path: Path, severity: str) -> dict[str, object]:
    body = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": _campaign_alert_summary(payload),
            "severity": severity,
            "source": socket.gethostname(),
            "component": "agentkernel.run_unattended_campaign",
            "custom_details": {
                "report_path": str(report_path),
                "phase": str(payload.get("phase", "")).strip(),
                "reason": str(payload.get("reason", "")).strip(),
                "rounds_completed": int(payload.get("rounds_completed", 0) or 0),
            },
        },
    }
    request = urllib.request.Request(
        "https://events.pagerduty.com/v2/enqueue",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        status_code = int(getattr(response, "status", 202) or 202)
        response_body = response.read().decode("utf-8", errors="replace").strip()
    return {"ran": True, "provider": "pagerduty", "status_code": status_code, "response": response_body}


def _dispatch_alerts(
    *,
    alert_command: str,
    slack_webhook_url: str,
    pagerduty_routing_key: str,
    report_path: Path,
    status_path: Path | None,
    event_log_path: Path | None,
    payload: dict[str, object],
    state_path: Path | None = None,
    rate_limit_seconds: float = 900.0,
    config: KernelConfig | None = None,
) -> dict[str, object]:
    should_emit, prior_state, repeat_count = _should_emit_alert(
        state_path,
        payload=payload,
        rate_limit_seconds=rate_limit_seconds,
    )
    severity = _alert_severity(payload, repeat_count=repeat_count)
    if not should_emit:
        return {
            "ran": False,
            "suppressed": True,
            "detail": "alert suppressed by dedup/rate limit",
            "prior_state": prior_state,
            "repeat_count": repeat_count,
            "severity": severity,
        }
    results: dict[str, object] = {}
    if str(alert_command).strip():
        results["command"] = _run_alert_command(
            alert_command,
            report_path=report_path,
            status_path=status_path,
            event_log_path=event_log_path,
            payload=payload,
        )
    if str(slack_webhook_url).strip():
        try:
            results["slack"] = _send_slack_alert(
                str(slack_webhook_url).strip(),
                payload=payload,
                report_path=report_path,
                severity=severity,
            )
        except (urllib.error.URLError, OSError) as exc:
            results["slack"] = {"ran": True, "provider": "slack", "error": str(exc)}
    if str(pagerduty_routing_key).strip():
        try:
            results["pagerduty"] = _send_pagerduty_alert(
                str(pagerduty_routing_key).strip(),
                payload=payload,
                report_path=report_path,
                severity=severity,
            )
        except (urllib.error.URLError, OSError) as exc:
            results["pagerduty"] = {"ran": True, "provider": "pagerduty", "error": str(exc)}
    if not results:
        return {"ran": False, "detail": "no alert transport configured"}
    results["ran"] = True
    results["severity"] = severity
    results["repeat_count"] = repeat_count
    _record_alert_state(state_path, payload=payload, repeat_count=repeat_count, config=config)
    return results


def _alerts_enabled(args) -> bool:
    return any(
        [
            str(getattr(args, "alert_command", "")).strip(),
            str(getattr(args, "slack_webhook_url", "")).strip(),
            str(getattr(args, "pagerduty_routing_key", "")).strip(),
        ]
    )


def _run_and_stream(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    progress_label: str | None = None,
    heartbeat_interval_seconds: float = 60.0,
    max_silence_seconds: float = 0.0,
    max_runtime_seconds: float = 0.0,
    max_progress_stall_seconds: float = 0.0,
    on_event: Callable[[dict[str, object]], None] | None = None,
    mirrored_status_path: Path | None = None,
) -> dict[str, object]:
    def _emit_event(event: dict[str, object]) -> dict[str, object]:
        if on_event is not None:
            response = on_event(event)
            if isinstance(response, dict):
                return dict(response)
        return {}

    completed_output: list[str] = []
    process = spawn_process_group(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        bufsize=1,
    )
    process_pid = int(getattr(process, "pid", 0) or 0)
    _emit_event(
        {
            "event": "start",
            "pid": process_pid,
            "progress_label": progress_label or Path(cmd[-1]).name,
            "started_at": time.time(),
        }
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_output_at = time.monotonic()
    last_progress_at = last_output_at
    last_heartbeat_at = last_output_at
    started_at = last_output_at
    heartbeat_interval = max(0.0, float(heartbeat_interval_seconds))
    max_silence = max(0.0, float(max_silence_seconds))
    max_runtime = max(0.0, float(max_runtime_seconds))
    max_progress_stall = max(0.0, float(max_progress_stall_seconds))
    last_progress_phase = ""
    active_finalize_phase = ""
    current_task: dict[str, object] = {}
    current_cognitive_stage: dict[str, object] = {}
    observe_summary: dict[str, object] = {}
    pending_decision_state = ""
    preview_state = ""
    current_task_verification_passed: bool | None = None
    runtime_grace_markers: set[tuple[str, int]] = set()
    progress_epoch = 0
    holdout_progress_samples: list[dict[str, float | int]] = []
    last_holdout_task_index = 0
    last_holdout_budget_deadline = 0.0
    last_adaptive_state_signature = ""
    last_authoritative_progress_signature = ""
    runtime_deadline = started_at + max_runtime if max_runtime > 0.0 else 0.0
    emergency_runtime_deadline = (
        started_at + (max_runtime * _UNATTENDED_CHILD_RUNTIME_EMERGENCY_MULTIPLIER)
        if max_runtime > 0.0
        else 0.0
    )
    try:
        while True:
            events = selector.select(timeout=1.0)
            if events:
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        break
                    completed_output.append(line)
                    now = time.monotonic()
                    last_output_at = now
                    if _is_significant_child_output(line):
                        last_progress_at = now
                        progress_epoch += 1
                        phase_name = _phase_from_progress_line(line)
                        if phase_name:
                            last_progress_phase = phase_name
                        if line.lstrip().startswith("[cycle:") and "finalize phase=" in line:
                            active_finalize_phase = phase_name
                        parsed_task = _task_from_progress_line(line)
                        if parsed_task:
                            previous_task_identity = _task_progress_identity(current_task)
                            current_task = parsed_task
                            if last_progress_phase:
                                current_task["phase"] = last_progress_phase
                            holdout_phase_active = active_finalize_phase == "holdout_eval"
                            if holdout_phase_active and last_progress_phase in {
                                "generated_success",
                                "generated_failure_seed",
                                "generated_failure",
                            }:
                                current_task["holdout_subphase"] = last_progress_phase
                            if holdout_phase_active or last_progress_phase == "holdout_eval":
                                task_index = int(current_task.get("index", 0) or 0)
                                task_total = int(current_task.get("total", 0) or 0)
                                if holdout_progress_samples:
                                    previous_total = int(holdout_progress_samples[-1].get("total", 0) or 0)
                                    if previous_total > 0 and task_total > 0 and previous_total != task_total:
                                        holdout_progress_samples = []
                                        last_holdout_task_index = 0
                                        last_holdout_budget_deadline = 0.0
                                if task_index > last_holdout_task_index:
                                    holdout_progress_samples.append(
                                        {
                                            "timestamp": now,
                                            "index": task_index,
                                            "total": task_total,
                                        }
                                    )
                                    last_holdout_task_index = task_index
                            if _task_progress_identity(current_task) != previous_task_identity:
                                current_task_verification_passed = None
                                current_cognitive_stage = {}
                        semantic_progress = _parse_child_semantic_progress_fields(line)
                        if semantic_progress:
                            if isinstance(semantic_progress.get("observe_summary"), Mapping):
                                observe_summary = dict(semantic_progress["observe_summary"])
                            if "pending_decision_state" in semantic_progress:
                                pending_decision_state = str(semantic_progress.get("pending_decision_state", "")).strip()
                            if "preview_state" in semantic_progress:
                                preview_state = str(semantic_progress.get("preview_state", "")).strip()
                        cognitive_progress = _parse_cognitive_progress_fields(line)
                        if cognitive_progress:
                            current_cognitive_stage = dict(cognitive_progress)
                            if last_progress_phase and not str(current_cognitive_stage.get("phase", "")).strip():
                                current_cognitive_stage["phase"] = last_progress_phase
                            if "verification_passed" in current_cognitive_stage:
                                current_task_verification_passed = bool(current_cognitive_stage.get("verification_passed"))
                    print(line, end="", file=sys.stderr, flush=True)
                    action = _emit_event(
                        {
                            "event": "output",
                            "line": line.rstrip("\n"),
                            "pid": process_pid,
                            "progress_label": progress_label or Path(cmd[-1]).name,
                            "timestamp": time.time(),
                        }
                    )
                    if bool(action.get("terminate", False)):
                        reason = str(action.get("reason", "")).strip() or "child terminated by controller intervention"
                        terminate_process_tree(process)
                        _emit_event(
                            {
                                "event": "timeout",
                                "pid": process_pid,
                                "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                                "timestamp": time.time(),
                                "timeout_reason": reason,
                                "intervention_reason": reason,
                            }
                        )
                        return {
                            "returncode": -9,
                            "stdout": "".join(completed_output).strip(),
                            "timed_out": True,
                            "timeout_reason": reason,
                        }
            elif process.poll() is not None:
                break
            now = time.monotonic()
            silence = now - last_output_at
            progress_stall = now - last_progress_at
            runtime_elapsed = now - started_at
            authoritative_child_status = _mirrored_child_status_from_parent_status(mirrored_status_path)
            if authoritative_child_status:
                supervision_snapshot = _authoritative_child_supervision_snapshot(
                    authoritative_child_status,
                    last_progress_phase=last_progress_phase,
                    active_finalize_phase=active_finalize_phase,
                    current_task=current_task,
                    current_cognitive_stage=current_cognitive_stage,
                    observe_summary=observe_summary,
                    pending_decision_state=pending_decision_state,
                    preview_state=preview_state,
                    current_task_verification_passed=current_task_verification_passed,
                )
                last_progress_phase = str(supervision_snapshot.get("last_progress_phase", "")).strip()
                active_finalize_phase = str(supervision_snapshot.get("active_finalize_phase", "")).strip()
                snapshot_task = supervision_snapshot.get("current_task", {})
                if isinstance(snapshot_task, Mapping):
                    current_task = dict(snapshot_task)
                snapshot_cognitive_stage = supervision_snapshot.get("current_cognitive_stage", {})
                if isinstance(snapshot_cognitive_stage, Mapping):
                    current_cognitive_stage = dict(snapshot_cognitive_stage)
                snapshot_observe_summary = supervision_snapshot.get("observe_summary", {})
                if isinstance(snapshot_observe_summary, Mapping):
                    observe_summary = dict(snapshot_observe_summary)
                pending_decision_state = str(supervision_snapshot.get("pending_decision_state", "")).strip()
                preview_state = str(supervision_snapshot.get("preview_state", "")).strip()
                if "current_task_verification_passed" in supervision_snapshot:
                    verification_value = supervision_snapshot.get("current_task_verification_passed")
                    current_task_verification_passed = (
                        bool(verification_value) if verification_value is not None else None
                    )
                authoritative_progress_signature = _authoritative_progress_signature(supervision_snapshot)
                if (
                    authoritative_progress_signature
                    and authoritative_progress_signature != last_authoritative_progress_signature
                ):
                    last_authoritative_progress_signature = authoritative_progress_signature
                    last_progress_at = now
                    progress_epoch += 1
            if heartbeat_interval > 0.0 and (now - last_heartbeat_at) >= heartbeat_interval and silence >= heartbeat_interval:
                label = str(progress_label).strip() or Path(cmd[1]).name
                _progress(f"[supervisor] child={label} still_running silence={int(silence)}s")
                action = _emit_event(
                    {
                        "event": "heartbeat",
                        "pid": process_pid,
                        "progress_label": label,
                        "silence_seconds": int(silence),
                        "timestamp": time.time(),
                    }
                )
                if bool(action.get("terminate", False)):
                    reason = str(action.get("reason", "")).strip() or "child terminated by controller intervention"
                    terminate_process_tree(process)
                    _emit_event(
                        {
                            "event": "timeout",
                            "pid": process_pid,
                            "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                            "timestamp": time.time(),
                            "timeout_reason": reason,
                            "intervention_reason": reason,
                        }
                    )
                    return {
                        "returncode": -9,
                        "stdout": "".join(completed_output).strip(),
                        "timed_out": True,
                        "timeout_reason": reason,
                    }
                last_heartbeat_at = now
            adaptive_state = _adaptive_child_progress_state(
                last_progress_phase=last_progress_phase,
                active_finalize_phase=active_finalize_phase,
                last_progress_at=last_progress_at,
                current_task=current_task,
                current_cognitive_stage=current_cognitive_stage,
                observe_summary=observe_summary,
                pending_decision_state=pending_decision_state,
                preview_state=preview_state,
                current_task_verification_passed=current_task_verification_passed,
                progress_samples=holdout_progress_samples,
                started_at=started_at,
                now=now,
                max_runtime_seconds=max_runtime,
                max_progress_stall_seconds=max_progress_stall,
            )
            adaptive_signature = _adaptive_progress_state_signature(adaptive_state)
            if adaptive_signature != last_adaptive_state_signature:
                last_adaptive_state_signature = adaptive_signature
                _emit_event(
                    {
                        "event": "adaptive_progress_state",
                        "pid": process_pid,
                        "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                        "timestamp": time.time(),
                        **adaptive_state,
                    }
                )
            if runtime_deadline > 0.0 and now >= runtime_deadline:
                holdout_budget = adaptive_state.get("holdout_budget", {}) if isinstance(adaptive_state, dict) else {}
                if holdout_budget:
                    adaptive_deadline = float(holdout_budget.get("projected_completion_seconds", 0.0) or 0.0)
                    proposed_deadline = max(runtime_deadline, adaptive_deadline)
                    if emergency_runtime_deadline > 0.0:
                        proposed_deadline = min(proposed_deadline, emergency_runtime_deadline)
                    if proposed_deadline > now and proposed_deadline > (last_holdout_budget_deadline + 1.0):
                        runtime_deadline = proposed_deadline
                        last_holdout_budget_deadline = proposed_deadline
                        _emit_event(
                            {
                                "event": "adaptive_runtime_budget",
                                "pid": process_pid,
                                "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                                "timestamp": time.time(),
                                "runtime_seconds": int(runtime_elapsed),
                                "runtime_deadline_seconds": int(max(0.0, runtime_deadline - started_at)),
                                "emergency_runtime_deadline_seconds": int(max(0.0, emergency_runtime_deadline - started_at)),
                                "adaptive_progress_class": str(adaptive_state.get("progress_class", "")).strip(),
                                "adaptive_budget_status": str(adaptive_state.get("status", "")).strip(),
                                **holdout_budget,
                            }
                        )
                        continue
                block_runtime_grace = _should_block_post_productive_runtime_grace(
                    authoritative_child_status,
                    last_progress_phase=last_progress_phase,
                    current_task=current_task,
                )
                grace_key, extension_seconds = (
                    ("", 0.0)
                    if block_runtime_grace
                    else _child_runtime_extension_plan(
                        last_progress_phase=last_progress_phase,
                        current_task=current_task,
                    )
                )
                grace_marker = _runtime_grace_marker(
                    grace_key=grace_key,
                    progress_epoch=progress_epoch,
                )
                if grace_marker and extension_seconds > 0.0 and grace_marker not in runtime_grace_markers:
                    runtime_grace_markers.add(grace_marker)
                    runtime_deadline = now + extension_seconds
                    _emit_event(
                        {
                            "event": "runtime_grace",
                            "pid": process_pid,
                            "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                            "timestamp": time.time(),
                            "runtime_seconds": int(runtime_elapsed),
                            "grace_seconds": int(extension_seconds),
                            "grace_key": grace_key,
                            "progress_epoch": progress_epoch,
                            "runtime_deadline_seconds": int(max(0.0, runtime_deadline - started_at)),
                            "last_progress_phase": last_progress_phase,
                            "current_task": dict(current_task),
                        }
                    )
                    continue
                terminate_process_tree(process)
                _emit_event(
                    {
                        "event": "timeout",
                        "pid": process_pid,
                        "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                        "runtime_seconds": int(runtime_elapsed),
                        "timestamp": time.time(),
                        "runtime_deadline_seconds": int(max(0.0, runtime_deadline - started_at)),
                        "emergency_runtime_deadline_seconds": int(max(0.0, emergency_runtime_deadline - started_at)),
                        "adaptive_progress_class": str(adaptive_state.get("progress_class", "")).strip(),
                        "adaptive_budget_status": str(adaptive_state.get("status", "")).strip(),
                        "timeout_reason": (
                            f"child exceeded adaptive runtime safety ceiling of {int(max_runtime)} seconds"
                            if str(adaptive_state.get("status", "")).strip() == "over_budget"
                            else f"child exceeded max runtime safety ceiling of {int(max_runtime)} seconds"
                        ),
                    }
                )
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": (
                        f"child exceeded adaptive runtime safety ceiling of {int(max_runtime)} seconds"
                        if str(adaptive_state.get("status", "")).strip() == "over_budget"
                        else f"child exceeded max runtime safety ceiling of {int(max_runtime)} seconds"
                    ),
                }
            if max_silence > 0.0 and silence >= max_silence:
                terminate_process_tree(process)
                _emit_event(
                    {
                        "event": "timeout",
                        "pid": process_pid,
                        "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                        "silence_seconds": int(silence),
                        "timestamp": time.time(),
                        "timeout_reason": f"child exceeded max silence of {int(max_silence)} seconds",
                    }
                )
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max silence of {int(max_silence)} seconds",
                }
            if max_progress_stall > 0.0 and progress_stall >= max_progress_stall:
                terminate_process_tree(process)
                _emit_event(
                    {
                        "event": "timeout",
                        "pid": process_pid,
                        "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                        "progress_stall_seconds": int(progress_stall),
                        "timestamp": time.time(),
                        "timeout_reason": f"child exceeded max progress stall of {int(max_progress_stall)} seconds",
                    }
                )
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max progress stall of {int(max_progress_stall)} seconds",
                }
        returncode = process.wait()
    except BaseException:
        terminate_process_tree(process)
        _emit_event(
            {
                "event": "interrupted",
                "pid": process_pid,
                "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                "timestamp": time.time(),
            }
        )
        raise
    finally:
        selector.close()
    _emit_event(
        {
            "event": "exit",
            "pid": process_pid,
            "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
            "returncode": returncode,
            "timestamp": time.time(),
        }
    )
    return {
        "returncode": returncode,
        "stdout": "".join(completed_output).strip(),
        "timed_out": False,
    }


def _last_report_path(stdout: str) -> Path | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = Path(lines[-1])
    return candidate if candidate.exists() else None


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mirrored_child_status_from_parent_status(status_path: Path | None) -> dict[str, object]:
    payload = _read_json(status_path)
    if not payload:
        return {}
    active_run = payload.get("active_run", {})
    if not isinstance(active_run, dict):
        return {}
    child_status = active_run.get("child_status", {})
    return dict(child_status) if isinstance(child_status, dict) else {}


def _authoritative_progress_signature(payload: Mapping[str, object] | None) -> str:
    child = payload if isinstance(payload, Mapping) else {}
    current_task = child.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    cognitive_stage = child.get("current_cognitive_stage", {})
    if not isinstance(cognitive_stage, Mapping):
        cognitive_stage = {}
    observe_summary = child.get("observe_summary", {})
    if not isinstance(observe_summary, Mapping):
        observe_summary = {}
    signature_payload = {
        "last_progress_phase": str(child.get("last_progress_phase", "")).strip(),
        "active_finalize_phase": str(child.get("active_finalize_phase", "")).strip(),
        "current_task": {
            "index": int(current_task.get("index", 0) or 0),
            "total": int(current_task.get("total", 0) or 0),
            "task_id": str(current_task.get("task_id", "")).strip(),
            "phase": str(current_task.get("phase", "")).strip(),
        },
        "current_cognitive_stage": {
            "cognitive_stage": str(cognitive_stage.get("cognitive_stage", "")).strip(),
            "step_index": int(cognitive_stage.get("step_index", 0) or 0),
            "verification_passed": cognitive_stage.get("verification_passed"),
            "phase": str(cognitive_stage.get("phase", "")).strip(),
        },
        "observe_summary": {
            "passed": int(observe_summary.get("passed", 0) or 0),
            "total": int(observe_summary.get("total", 0) or 0),
            "pass_rate": observe_summary.get("pass_rate"),
            "generated_pass_rate": observe_summary.get("generated_pass_rate"),
        },
        "pending_decision_state": str(child.get("pending_decision_state", "")).strip(),
        "preview_state": str(child.get("preview_state", "")).strip(),
        "current_task_verification_passed": child.get("current_task_verification_passed"),
    }
    return json.dumps(signature_payload, sort_keys=True)


def _authoritative_child_supervision_snapshot(
    mirrored_child_status: Mapping[str, object] | None,
    *,
    last_progress_phase: str,
    active_finalize_phase: str,
    current_task: Mapping[str, object] | None,
    current_cognitive_stage: Mapping[str, object] | None,
    observe_summary: Mapping[str, object] | None,
    pending_decision_state: str,
    preview_state: str,
    current_task_verification_passed: bool | None,
) -> dict[str, object]:
    active_child = {
        "last_progress_phase": str(last_progress_phase).strip(),
        "finalize_phase": str(active_finalize_phase).strip(),
        "current_task": dict(current_task) if isinstance(current_task, Mapping) else {},
        "current_cognitive_stage": (
            dict(current_cognitive_stage) if isinstance(current_cognitive_stage, Mapping) else {}
        ),
        "observe_summary": dict(observe_summary) if isinstance(observe_summary, Mapping) else {},
        "pending_decision_state": str(pending_decision_state).strip(),
        "preview_state": str(preview_state).strip(),
        "current_task_verification_passed": current_task_verification_passed,
    }
    merged = _merge_mirrored_child_status_fields(active_child, mirrored_child_status)
    merged_current_task = merged.get("current_task", {})
    if not isinstance(merged_current_task, Mapping):
        merged_current_task = {}
    merged_cognitive_stage = merged.get("current_cognitive_stage", {})
    if not isinstance(merged_cognitive_stage, Mapping):
        merged_cognitive_stage = {}
    merged_observe_summary = merged.get("observe_summary", {})
    if not isinstance(merged_observe_summary, Mapping):
        merged_observe_summary = {}
    resolved_last_progress_phase = str(merged.get("last_progress_phase", "")).strip()
    resolved_finalize_phase = str(merged.get("finalize_phase", "")).strip()
    current_task_phase = str(merged_current_task.get("phase", "")).strip()
    if current_task_phase and not resolved_last_progress_phase:
        resolved_last_progress_phase = current_task_phase
    elif resolved_last_progress_phase and not current_task_phase:
        merged_current_task = {**dict(merged_current_task), "phase": resolved_last_progress_phase}
    resolved_verification = merged.get("current_task_verification_passed")
    if resolved_verification is None and "verification_passed" in merged_cognitive_stage:
        resolved_verification = bool(merged_cognitive_stage.get("verification_passed"))
    return {
        "last_progress_phase": resolved_last_progress_phase,
        "active_finalize_phase": resolved_finalize_phase,
        "current_task": dict(merged_current_task),
        "current_cognitive_stage": dict(merged_cognitive_stage),
        "observe_summary": dict(merged_observe_summary),
        "pending_decision_state": str(merged.get("pending_decision_state", "")).strip(),
        "preview_state": str(merged.get("preview_state", "")).strip(),
        "current_task_verification_passed": resolved_verification,
    }


def _task_progress_identity(task: Mapping[str, object] | None) -> tuple[object, ...]:
    payload = task if isinstance(task, Mapping) else {}
    return (
        str(payload.get("task_id", "")).strip(),
        int(payload.get("index", 0) or 0),
        int(payload.get("total", 0) or 0),
        str(payload.get("family", "")).strip(),
        str(payload.get("phase", "")).strip(),
        str(payload.get("holdout_subphase", "")).strip(),
    )


def _task_scope_identity(task: Mapping[str, object] | None) -> tuple[object, ...]:
    payload = task if isinstance(task, Mapping) else {}
    return (
        str(payload.get("task_id", "")).strip(),
        int(payload.get("index", 0) or 0),
        int(payload.get("total", 0) or 0),
        str(payload.get("family", "")).strip(),
    )


def _task_scope_matches(left: Mapping[str, object] | None, right: Mapping[str, object] | None) -> bool:
    left_identity = _task_scope_identity(left)
    right_identity = _task_scope_identity(right)
    if not any(left_identity) or not any(right_identity):
        return False
    return left_identity == right_identity


def _semantic_state_recorded_at(state: Mapping[str, object] | None) -> float:
    payload = state if isinstance(state, Mapping) else {}
    for key in ("recorded_at", "timestamp"):
        try:
            recorded_at = float(payload.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            recorded_at = 0.0
        if recorded_at > 0.0:
            return recorded_at
    return 0.0


def _child_progress_recorded_at(payload: Mapping[str, object] | None) -> float:
    child = payload if isinstance(payload, Mapping) else {}
    candidate_values: list[float] = []
    for key in ("last_progress_at", "last_output_at", "ended_at", "started_at"):
        try:
            candidate = float(child.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            candidate = 0.0
        if candidate > 0.0:
            candidate_values.append(candidate)
    semantic_state = child.get("semantic_progress_state", {})
    if isinstance(semantic_state, Mapping):
        candidate = _semantic_state_recorded_at(semantic_state)
        if candidate > 0.0:
            candidate_values.append(candidate)
    adaptive_state = child.get("adaptive_progress_state", {})
    if isinstance(adaptive_state, Mapping):
        candidate = _semantic_state_recorded_at(adaptive_state)
        if candidate > 0.0:
            candidate_values.append(candidate)
    active_cycle_run = child.get("active_cycle_run", {})
    if isinstance(active_cycle_run, Mapping):
        for nested_key in ("last_progress_at", "last_output_at"):
            try:
                candidate = float(active_cycle_run.get(nested_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                candidate = 0.0
            if candidate > 0.0:
                candidate_values.append(candidate)
        nested_semantic_state = active_cycle_run.get("semantic_progress_state", {})
        if isinstance(nested_semantic_state, Mapping):
            candidate = _semantic_state_recorded_at(nested_semantic_state)
            if candidate > 0.0:
                candidate_values.append(candidate)
    return max(candidate_values) if candidate_values else 0.0


def _parse_cognitive_progress_fields(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if not normalized or "cognitive_stage=" not in normalized:
        return {}
    stage_match = re.search(r"cognitive_stage=(?P<stage>[A-Za-z0-9_:-]+)", normalized)
    if not stage_match:
        return {}
    payload: dict[str, object] = {
        "event": "cognitive_stage",
        "cognitive_stage": str(stage_match.group("stage")).strip(),
    }
    task_match = _task_from_progress_line(normalized)
    if task_match:
        payload["current_task"] = task_match
    phase_name = _phase_from_progress_line(normalized)
    if phase_name:
        payload["phase"] = phase_name
    step_match = re.search(r"\bstep=(?P<step>\d+)", normalized)
    if step_match:
        payload["step_index"] = int(step_match.group("step"))
    subphase_match = re.search(r"\bsubphase=(?P<subphase>[A-Za-z0-9_:-]+)", normalized)
    if subphase_match:
        payload["step_subphase"] = str(subphase_match.group("subphase")).strip()
    decision_source_match = re.search(r"\bdecision_source=(?P<source>[A-Za-z0-9_:-]+)", normalized)
    if decision_source_match:
        payload["decision_source"] = str(decision_source_match.group("source")).strip()
    verification_match = re.search(r"\bverification_passed=(?P<passed>[01])", normalized)
    if verification_match:
        payload["verification_passed"] = verification_match.group("passed") == "1"
    return payload


def _parse_child_semantic_progress_fields(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if not normalized:
        return {}
    payload: dict[str, object] = {}
    if "observe complete" in normalized:
        passed_total_match = re.search(r"\bpassed=(?P<passed>\d+)/(?P<total>\d+)", normalized)
        if passed_total_match:
            observe_summary: dict[str, object] = {
                "passed": int(passed_total_match.group("passed")),
                "total": int(passed_total_match.group("total")),
            }
            pass_rate_match = re.search(r"\bpass_rate=(?P<pass_rate>[0-9.]+)", normalized)
            if pass_rate_match:
                observe_summary["pass_rate"] = float(pass_rate_match.group("pass_rate"))
            generated_pass_rate_match = re.search(r"\bgenerated_pass_rate=(?P<pass_rate>[0-9.]+)", normalized)
            if generated_pass_rate_match:
                observe_summary["generated_pass_rate"] = float(generated_pass_rate_match.group("pass_rate"))
            payload["observe_summary"] = observe_summary
            payload["observe_completed"] = True
    decision_state_match = re.search(r"\bstate=(?P<state>retain|reject)\b", normalized)
    if decision_state_match:
        decision_state = str(decision_state_match.group("state")).strip()
        if "preview complete" in normalized:
            payload["preview_state"] = decision_state
            payload["pending_decision_state"] = decision_state
        elif "finalize phase=" in normalized:
            payload["pending_decision_state"] = decision_state
    return payload


def _progress_event_stage(event: Mapping[str, object] | None) -> str:
    payload = event if isinstance(event, Mapping) else {}
    return str(
        payload.get("stage")
        or payload.get("cognitive_stage")
        or payload.get("event")
        or payload.get("step_stage")
        or ""
    ).strip()


def _synthesized_active_child_phase_detail(active_child: Mapping[str, object] | None) -> str:
    payload = active_child if isinstance(active_child, Mapping) else {}
    current_task = payload.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    semantic_state = payload.get("semantic_progress_state", {})
    if not isinstance(semantic_state, Mapping):
        semantic_state = {}
    phase = str(
        current_task.get("phase")
        or semantic_state.get("phase")
        or payload.get("finalize_phase")
        or payload.get("last_progress_phase")
        or ""
    ).strip()
    detail = str(semantic_state.get("detail", "")).strip()
    subsystem = str(payload.get("selected_subsystem", "")).strip()
    task_index = int(current_task.get("index", 0) or 0)
    task_total = int(current_task.get("total", 0) or 0)
    task_id = str(current_task.get("task_id", "")).strip()
    if not any((phase, detail, subsystem, task_index > 0, task_total > 0, task_id)):
        return ""
    parts = ["[child]"]
    if phase:
        parts.append(f"phase={phase}")
    if task_index > 0 or task_total > 0:
        rendered_total = max(task_total, task_index, 1)
        rendered_index = max(task_index, 1)
        parts.append(f"task {rendered_index}/{rendered_total}")
    if task_id:
        parts.append(task_id)
    if subsystem:
        parts.append(f"subsystem={subsystem}")
    if detail:
        parts.append(f"detail={detail}")
    return " ".join(parts)


def _phase_detail_matches_active_child(detail: str, active_child: Mapping[str, object] | None) -> bool:
    normalized = str(detail).strip()
    payload = active_child if isinstance(active_child, Mapping) else {}
    if not normalized:
        return False
    current_task = payload.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    parsed_task = _task_from_progress_line(normalized)
    if parsed_task and current_task and not _task_scope_matches(parsed_task, current_task):
        return False
    semantic_state = payload.get("semantic_progress_state", {})
    if not isinstance(semantic_state, Mapping):
        semantic_state = {}
    detail_phase = _phase_from_progress_line(normalized)
    current_phase = str(
        current_task.get("phase")
        or semantic_state.get("phase")
        or payload.get("last_progress_phase")
        or ""
    ).strip()
    if detail_phase and current_phase and detail_phase != current_phase:
        return False
    return True


def _preferred_active_child_phase_detail(existing_detail: str, active_child: Mapping[str, object] | None) -> str:
    current = str(existing_detail).strip()
    if current.startswith("mid-round controller intervention:"):
        return current
    payload = active_child if isinstance(active_child, Mapping) else {}
    live_line = str(payload.get("last_progress_line") or payload.get("last_output_line") or "").strip()
    if live_line:
        if not current:
            return live_line
        current_phase = _phase_from_progress_line(current)
        live_phase = _phase_from_progress_line(live_line)
        if live_phase and live_phase != current_phase:
            return live_line
        if not _phase_detail_matches_active_child(current, payload) and _phase_detail_matches_active_child(live_line, payload):
            return live_line
    synthesized = _synthesized_active_child_phase_detail(active_child)
    if not synthesized:
        return current
    if not current:
        return synthesized
    current_phase = _phase_from_progress_line(current)
    synthesized_phase = _phase_from_progress_line(synthesized)
    if synthesized_phase and synthesized_phase != current_phase:
        return synthesized
    if not _phase_detail_matches_active_child(current, payload) and _phase_detail_matches_active_child(synthesized, payload):
        return synthesized
    return current


def _repeated_verification_loop_signal(active_child: Mapping[str, object] | None) -> dict[str, object]:
    payload = active_child if isinstance(active_child, Mapping) else {}
    timeline = payload.get("current_task_progress_timeline", [])
    if not isinstance(timeline, list) or not timeline:
        return {"active": False}
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for entry in timeline:
        if not isinstance(entry, Mapping):
            continue
        step_index = int(entry.get("step_index", 0) or 0)
        if step_index <= 0:
            continue
        grouped.setdefault(step_index, []).append(entry)
    if not grouped:
        return {"active": False}
    step_index = max(grouped)
    recent_events = grouped.get(step_index, [])
    if len(recent_events) < 6:
        return {"active": False}
    current_task = payload.get("current_task", {})
    current_task_phase = ""
    if isinstance(current_task, Mapping):
        current_task_phase = str(current_task.get("phase", "")).strip()
    if not current_task_phase:
        current_cognitive_stage = payload.get("current_cognitive_stage", {})
        if isinstance(current_cognitive_stage, Mapping):
            current_task_phase = str(current_cognitive_stage.get("phase", "")).strip()
    if not current_task_phase:
        current_task_phase = str(payload.get("last_progress_phase", "")).strip()
    loop_grace = _repeated_verification_loop_grace(
        current_task_phase=current_task_phase,
        current_task=current_task if isinstance(current_task, Mapping) else {},
    )
    grace_step_floor = max(0, int(loop_grace.get("step_index_floor", 0) or 0))
    if grace_step_floor > 0 and step_index < grace_step_floor:
        return {"active": False}
    failed_verifications = [
        entry
        for entry in recent_events
        if _progress_event_stage(entry) == "verification_result" and bool(entry.get("verification_passed", True)) is False
    ]
    failed_memory_updates = [
        entry
        for entry in recent_events
        if _progress_event_stage(entry) == "memory_update_written" and bool(entry.get("verification_passed", True)) is False
    ]
    required_failed_cycles = max(2, int(loop_grace.get("failed_cycles", 0) or 0))
    if len(failed_verifications) < required_failed_cycles or len(failed_memory_updates) < required_failed_cycles:
        return {"active": False}
    completed_steps = {int(entry.get("completed_steps", 0) or 0) for entry in recent_events}
    if len(completed_steps) > 1:
        return {"active": False}
    observed_stages = {_progress_event_stage(entry) for entry in recent_events}
    return {
        "active": True,
        "phase": current_task_phase,
        "step_index": step_index,
        "failed_verification_count": len(failed_verifications),
        "failed_memory_update_count": len(failed_memory_updates),
        "completed_steps": max(completed_steps) if completed_steps else 0,
        "observed_stages": sorted(stage for stage in observed_stages if stage),
    }


def _cognitive_stage_failed_verification(cognitive_stage: Mapping[str, object] | None) -> bool:
    payload = cognitive_stage if isinstance(cognitive_stage, Mapping) else {}
    stage = _progress_event_stage(payload)
    return stage in {"memory_update_written", "verification_result"} and bool(
        payload.get("verification_passed", True)
    ) is False


def _merge_mirrored_child_status_fields(
    active_child: dict[str, object],
    mirrored_child_status: Mapping[str, object] | None,
) -> dict[str, object]:
    child_status = mirrored_child_status if isinstance(mirrored_child_status, Mapping) else {}
    if not child_status:
        merged = dict(active_child)
        return _reconcile_active_child_snapshot(
            merged,
            active_progress_recorded_at=_child_progress_recorded_at(merged),
        )
    merged = dict(active_child)

    def _prefer_mirrored_finalize_phase(existing: object, mirrored: object) -> bool:
        existing_phase = str(existing or "").strip()
        mirrored_phase = str(mirrored or "").strip()
        if not mirrored_phase:
            return False
        if not existing_phase:
            return True
        decisive_finalize_phases = {
            "preview_baseline_eval",
            "preview_candidate_eval",
            "preview_complete",
            "apply_decision",
            "decision_reject_reason",
            "decision_retain_reason",
            "done",
        }
        if mirrored_phase in decisive_finalize_phases and existing_phase not in decisive_finalize_phases:
            return True
        if existing_phase.startswith("generated_") and mirrored_phase != existing_phase:
            return True
        return False

    active_semantic_state = merged.get("semantic_progress_state", {})
    if not isinstance(active_semantic_state, Mapping):
        active_semantic_state = {}
    child_semantic_state = child_status.get("semantic_progress_state", {})
    if not isinstance(child_semantic_state, Mapping):
        child_semantic_state = {}
    child_adaptive_state = child_status.get("adaptive_progress_state", {})
    if not isinstance(child_adaptive_state, Mapping):
        child_adaptive_state = {}
    if not active_semantic_state:
        active_cycle_run = merged.get("active_cycle_run", {})
        if isinstance(active_cycle_run, Mapping):
            nested_active_semantic_state = active_cycle_run.get("semantic_progress_state", {})
            if isinstance(nested_active_semantic_state, Mapping):
                active_semantic_state = nested_active_semantic_state
                if nested_active_semantic_state:
                    merged["semantic_progress_state"] = dict(nested_active_semantic_state)
    if not child_semantic_state:
        child_active_cycle_run = child_status.get("active_cycle_run", {})
        if isinstance(child_active_cycle_run, Mapping):
            nested_child_semantic_state = child_active_cycle_run.get("semantic_progress_state", {})
            if isinstance(nested_child_semantic_state, Mapping):
                child_semantic_state = nested_child_semantic_state
    child_current_task = child_status.get("current_task", {})
    if not isinstance(child_current_task, Mapping):
        child_current_task = {}
    active_current_task = merged.get("current_task", {})
    if not isinstance(active_current_task, Mapping):
        active_current_task = {}
    active_progress_recorded_at = _child_progress_recorded_at(merged)
    mirrored_progress_recorded_at = _child_progress_recorded_at(child_status)
    mirrored_snapshot_is_newer = mirrored_progress_recorded_at > active_progress_recorded_at + 1e-6
    same_task_scope = _task_scope_matches(active_current_task, child_current_task)
    child_semantic_state = _select_preferred_child_progress_state(
        semantic_state=child_semantic_state,
        adaptive_state=child_adaptive_state,
        current_task=child_current_task,
        last_progress_phase=str(child_status.get("last_progress_phase", "")).strip(),
    )
    decision_records_considered, _ = _effective_live_decision_record_count(merged)
    mirrored_decision_records_considered, _ = _effective_live_decision_record_count(child_status)
    retained_gain_runs = _effective_live_retained_gain_runs(merged)
    mirrored_retained_gain_runs = _effective_live_retained_gain_runs(child_status)
    child_summary = child_status.get("decision_conversion_summary", {})
    if not isinstance(child_summary, Mapping):
        child_summary = {}
    for key in (
        "campaign_records_considered",
        "execution_source_summary",
        "trust_breadth_summary",
        "tolbert_runtime_summary",
        "round_index",
        "last_report_path",
    ):
        if key in child_status:
            merged[key] = child_status.get(key)
    merged["decision_records_considered"] = max(
        decision_records_considered,
        mirrored_decision_records_considered,
    )
    merged["runtime_managed_decisions"] = max(
        int(merged.get("runtime_managed_decisions", 0) or 0),
        int(child_status.get("runtime_managed_decisions", 0) or 0),
    )
    merged["non_runtime_managed_decisions"] = max(
        int(merged.get("non_runtime_managed_decisions", 0) or 0),
        int(child_status.get("non_runtime_managed_decisions", 0) or 0),
        int(child_summary.get("non_runtime_managed_runs", 0) or 0)
        if isinstance(child_summary, Mapping)
        else 0,
    )
    merged["retained_gain_runs"] = max(retained_gain_runs, mirrored_retained_gain_runs)
    if str(merged.get("pending_decision_state", "")).strip() not in {"retain", "reject"}:
        merged["pending_decision_state"] = str(child_status.get("pending_decision_state", "")).strip()
    merged_summary = merged.get("decision_conversion_summary", {})
    if not isinstance(merged_summary, Mapping):
        merged_summary = {}
    merged["decision_conversion_summary"] = {
        **dict(child_summary),
        **dict(merged_summary),
        "runtime_managed_runs": max(
            int(merged_summary.get("runtime_managed_runs", 0) or 0),
            int(child_summary.get("runtime_managed_runs", 0) or 0),
            1 if int(merged.get("runtime_managed_decisions", 0) or 0) > 0 else 0,
        ),
        "non_runtime_managed_runs": max(
            int(merged_summary.get("non_runtime_managed_runs", 0) or 0),
            int(child_summary.get("non_runtime_managed_runs", 0) or 0),
        ),
        "partial_productive_without_decision_runs": 0
        if int(merged.get("decision_records_considered", 0) or 0) > 0
        else max(
            int(merged_summary.get("partial_productive_without_decision_runs", 0) or 0),
            int(child_summary.get("partial_productive_without_decision_runs", 0) or 0),
        ),
        "decision_runs": max(
            int(merged_summary.get("decision_runs", 0) or 0),
            int(child_summary.get("decision_runs", 0) or 0),
            1 if int(merged.get("decision_records_considered", 0) or 0) > 0 else 0,
        ),
        "no_decision_runs": 0
        if int(merged.get("decision_records_considered", 0) or 0) > 0
        else max(
            int(merged_summary.get("no_decision_runs", 0) or 0),
            int(child_summary.get("no_decision_runs", 0) or 0),
        ),
    }
    merged_recent_decisions = merged.get("recent_structured_child_decisions", [])
    if not isinstance(merged_recent_decisions, list):
        merged_recent_decisions = []
    child_recent_decisions = child_status.get("recent_structured_child_decisions", [])
    if not isinstance(child_recent_decisions, list):
        child_recent_decisions = []
    merged["recent_structured_child_decisions"] = (
        merged_recent_decisions if merged_recent_decisions else child_recent_decisions
    )
    if "last_progress_phase" in child_status:
        existing_phase = merged.get("last_progress_phase", "")
        mirrored_phase = child_status.get("last_progress_phase", "")
        if (
            mirrored_snapshot_is_newer
            or not str(existing_phase or "").strip()
            or not active_current_task
        ) and _prefer_mirrored_progress_phase(existing_phase, mirrored_phase):
            merged["last_progress_phase"] = mirrored_phase
    child_progress_class = str(child_semantic_state.get("progress_class", "")).strip()
    child_decision_distance = str(child_semantic_state.get("decision_distance", "")).strip()
    child_phase = str(child_semantic_state.get("phase", "")).strip()
    active_progress_class = str(active_semantic_state.get("progress_class", "")).strip()
    active_decision_distance = str(active_semantic_state.get("decision_distance", "")).strip()
    active_phase = str(active_semantic_state.get("phase", "")).strip()
    active_raw_task = merged.get("current_task", {})
    active_raw_task_phase = ""
    if isinstance(active_raw_task, Mapping):
        active_raw_task_phase = str(active_raw_task.get("phase", "")).strip()
    active_raw_progress_phase = str(merged.get("last_progress_phase", "")).strip()
    active_generated_lane = any(
        phase.startswith("generated_")
        for phase in (active_phase, active_raw_task_phase, active_raw_progress_phase)
        if phase
    )
    child_generated_lane = child_phase.startswith("generated_")
    child_semantic_is_more_final = child_decision_distance in {"decision_emitted", "complete"} or child_progress_class == "complete"
    active_semantic_is_more_severe = active_progress_class in {"stuck", "degraded"} and child_progress_class not in {
        "stuck",
        "degraded",
    }
    child_semantic_conflicts_with_live_generated_lane = active_generated_lane and not child_generated_lane
    child_semantic_recorded_at = _semantic_state_recorded_at(child_semantic_state)
    active_semantic_recorded_at = _semantic_state_recorded_at(active_semantic_state)
    child_semantic_is_newer = child_semantic_recorded_at > active_semantic_recorded_at + 1e-6
    child_semantic_matches_live_phase = bool(child_phase) and child_phase in {
        str(active_current_task.get("phase", "")).strip(),
        str(merged.get("last_progress_phase", "")).strip(),
    }
    if child_semantic_state and not child_semantic_conflicts_with_live_generated_lane and (
        child_semantic_is_more_final
        or not active_semantic_is_more_severe
        or child_semantic_is_newer
        or child_semantic_matches_live_phase
        or not active_semantic_state
    ):
        merged["semantic_progress_state"] = dict(child_semantic_state)
    for key in ("pending_decision_state", "preview_state", "current_task_verification_passed"):
        if key not in child_status:
            continue
        if key in {"pending_decision_state", "preview_state"}:
            existing_value = str(merged.get(key, "")).strip()
            mirrored_value = str(child_status.get(key, "")).strip()
            if existing_value in {"retain", "reject"} and not mirrored_value:
                continue
            merged[key] = mirrored_value
            continue
        merged[key] = child_status.get(key)
    effective_decision_records_considered, inferred_decision_credit = _effective_live_decision_record_count(merged)
    if effective_decision_records_considered > int(merged.get("decision_records_considered", 0) or 0):
        merged["decision_records_considered"] = effective_decision_records_considered
    if inferred_decision_credit:
        merged["inferred_decision_credit"] = True
    effective_retained_gain_runs = _effective_live_retained_gain_runs(merged)
    if effective_retained_gain_runs > int(merged.get("retained_gain_runs", 0) or 0):
        merged["retained_gain_runs"] = effective_retained_gain_runs
    families_sampled = child_status.get("families_sampled", [])
    if isinstance(families_sampled, list) and families_sampled:
        merged["families_sampled"] = list(families_sampled)
    active_cycle_progress = child_status.get("active_cycle_progress", {})
    if isinstance(active_cycle_progress, Mapping) and active_cycle_progress:
        merged["active_cycle_progress"] = dict(active_cycle_progress)
    active_cycle_run = child_status.get("active_cycle_run", {})
    if isinstance(active_cycle_run, Mapping) and active_cycle_run:
        cycle_task = active_cycle_run.get("current_task", {})
        if not isinstance(cycle_task, Mapping):
            cycle_task = {}
        cycle_same_task_scope = _task_scope_matches(cycle_task, merged.get("current_task", {}))
        active_task_scope_missing = merged.get("current_task") in ({}, [], "", None)
        for key in (
            "selected_subsystem",
            "finalize_phase",
            "observe_summary",
            "current_task",
            "current_cognitive_stage",
            "current_task_progress_timeline",
            "current_task_verification_passed",
            "verification_outcome_summary",
            "sampled_families_from_progress",
            "pending_decision_state",
            "preview_state",
        ):
            if key not in active_cycle_run:
                continue
            existing_value = merged.get(key)
            mirrored_value = active_cycle_run.get(key)
            if key in {
                "selected_subsystem",
                "observe_summary",
                "sampled_families_from_progress",
                "pending_decision_state",
                "preview_state",
            }:
                if mirrored_value not in ({}, [], "", None):
                    merged[key] = mirrored_value
                continue
            if key == "finalize_phase":
                if _prefer_mirrored_finalize_phase(existing_value, mirrored_value):
                    merged[key] = mirrored_value
                continue
            if key == "current_task":
                if existing_value in ({}, [], "", None) or mirrored_snapshot_is_newer:
                    merged[key] = mirrored_value
                continue
            if key in {"current_cognitive_stage", "current_task_progress_timeline", "current_task_verification_passed"}:
                if mirrored_value in ({}, [], "", None):
                    continue
                if (cycle_same_task_scope or active_task_scope_missing) and (
                    existing_value in ({}, [], "", None)
                    or mirrored_snapshot_is_newer
                    or active_task_scope_missing
                ):
                    merged[key] = mirrored_value
                continue
            if existing_value in ({}, [], "", None):
                merged[key] = mirrored_value
        if cycle_task and (
            merged.get("current_task") in ({}, [], "", None)
            or mirrored_snapshot_is_newer
        ):
            merged["current_task"] = dict(cycle_task)
        if "tolbert_runtime_summary" in active_cycle_run and "active_cycle_tolbert_runtime_summary" not in merged:
            merged["active_cycle_tolbert_runtime_summary"] = active_cycle_run.get("tolbert_runtime_summary")
        if "last_progress_phase" in active_cycle_run and not str(merged.get("last_progress_phase", "")).strip():
            merged["last_progress_phase"] = active_cycle_run.get("last_progress_phase")
    return _reconcile_active_child_snapshot(
        merged,
        active_progress_recorded_at=active_progress_recorded_at,
    )


def _reconcile_active_child_snapshot(
    merged: dict[str, object],
    *,
    active_progress_recorded_at: float,
) -> dict[str, object]:
    resolved_current_task = merged.get("current_task", {})
    if not isinstance(resolved_current_task, Mapping):
        resolved_current_task = {}
    resolved_progress_phase = str(merged.get("last_progress_phase", "")).strip()
    resolved_semantic_state = merged.get("semantic_progress_state", {})
    if not isinstance(resolved_semantic_state, Mapping):
        resolved_semantic_state = {}
    resolved_adaptive_state = merged.get("adaptive_progress_state", {})
    if not isinstance(resolved_adaptive_state, Mapping):
        resolved_adaptive_state = {}
    resolved_semantic_phase = str(resolved_semantic_state.get("phase", "")).strip()
    current_task_phase = str(resolved_current_task.get("phase", "")).strip()
    if current_task_phase and resolved_progress_phase and current_task_phase != resolved_progress_phase:
        semantic_phase_family = str(resolved_semantic_state.get("phase_family", "")).strip()
        if semantic_phase_family in {"recovery", "finalize"} or resolved_progress_phase.startswith("generated_"):
            merged["last_progress_phase"] = current_task_phase
            resolved_progress_phase = current_task_phase
    if current_task_phase and resolved_semantic_phase and resolved_semantic_phase != current_task_phase:
        if current_task_phase.startswith("generated_") or resolved_semantic_phase.startswith("generated_"):
            reconciled_semantic_state = dict(resolved_semantic_state)
            reconciled_semantic_state["phase"] = current_task_phase
            if "recorded_at" not in reconciled_semantic_state and active_progress_recorded_at > 0.0:
                reconciled_semantic_state["recorded_at"] = active_progress_recorded_at
            merged["semantic_progress_state"] = reconciled_semantic_state
            resolved_semantic_state = reconciled_semantic_state
    canonical_progress_state = _select_preferred_child_progress_state(
        semantic_state=resolved_semantic_state,
        adaptive_state=resolved_adaptive_state,
        current_task=resolved_current_task,
        last_progress_phase=resolved_progress_phase,
    )
    if canonical_progress_state and "recorded_at" not in canonical_progress_state and active_progress_recorded_at > 0.0:
        canonical_progress_state = {
            **dict(canonical_progress_state),
            "recorded_at": active_progress_recorded_at,
        }
    if canonical_progress_state:
        merged["canonical_progress_state"] = dict(canonical_progress_state)
        semantic_matches_live = _progress_state_matches_live_context(
            resolved_semantic_state,
            current_task=resolved_current_task,
            last_progress_phase=resolved_progress_phase,
        )
        adaptive_matches_live = _progress_state_matches_live_context(
            resolved_adaptive_state,
            current_task=resolved_current_task,
            last_progress_phase=resolved_progress_phase,
        )
        if not resolved_semantic_state or not semantic_matches_live:
            merged["semantic_progress_state"] = dict(canonical_progress_state)
        if not resolved_adaptive_state or not adaptive_matches_live:
            merged["adaptive_progress_state"] = dict(canonical_progress_state)
    projected_phase_detail = _preferred_active_child_phase_detail(
        str(merged.get("last_progress_line", "")).strip(),
        merged,
    )
    if projected_phase_detail:
        merged["last_progress_line"] = projected_phase_detail
        existing_output_line = str(merged.get("last_output_line", "")).strip()
        if not existing_output_line or _phase_from_progress_line(existing_output_line) != _phase_from_progress_line(
            projected_phase_detail
        ):
            merged["last_output_line"] = projected_phase_detail
        if _subsystem_from_progress_line(projected_phase_detail):
            merged["last_subsystem_progress_line"] = projected_phase_detail
    return merged


def _mid_round_round_signal(round_payload: Mapping[str, object] | None) -> dict[str, object]:
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    runtime_limits = payload.get("runtime_limits", {})
    if not isinstance(runtime_limits, Mapping):
        runtime_limits = {}
    active_child = payload.get("active_child", {})
    if not isinstance(active_child, Mapping):
        active_child = {}
    active_cycle_progress = active_child.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    observe_summary = active_child.get("observe_summary", {})
    if not isinstance(observe_summary, Mapping):
        observe_summary = {}
    observe_passed = int(observe_summary.get("passed", 0) or 0)
    observe_total = int(observe_summary.get("total", 0) or 0)
    observe_completed = bool(active_child.get("observe_completed", False)) or bool(
        active_cycle_progress.get("observe_completed", False)
    ) or (
        observe_total > 0 and observe_passed >= observe_total
    )
    priority_families = _normalize_benchmark_families(
        payload.get("policy", {}).get("priority_benchmark_families", [])
        if isinstance(payload.get("policy", {}), Mapping)
        else []
    )
    sampled_families = _normalize_benchmark_families(
        active_child.get(
            "families_sampled",
            active_child.get("sampled_families_from_progress", []),
        )
    )
    if not sampled_families and isinstance(active_cycle_progress, Mapping):
        sampled_families = _normalize_benchmark_families(active_cycle_progress.get("sampled_families_from_progress", []))
    partial_progress_summary = active_child.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, Mapping):
        partial_progress_summary = {}
    sampled_priority_family_count = len(set(sampled_families) & set(priority_families)) if priority_families else len(sampled_families)
    selected_subsystem = str(active_child.get("selected_subsystem", "")).strip()
    finalize_phase = str(active_child.get("finalize_phase", active_child.get("last_progress_phase", ""))).strip()
    decision_records_considered, inferred_decision_credit = _effective_live_decision_record_count(active_child)
    pending_decision_state = str(active_child.get("pending_decision_state", "")).strip()
    preview_state = str(active_child.get("preview_state", "")).strip()
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False)) or int(
        partial_progress_summary.get("generated_success_completed_runs", 0) or 0
    ) > 0
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or int(
        active_child.get("partial_productive_runs", 0) or 0
    ) > 0
    broad_observe_then_retrieval_first = (
        observe_completed
        and selected_subsystem == "retrieval"
        and sampled_priority_family_count >= min(3, max(1, len(priority_families) or 3))
    )
    definitive_decision_emitted = (
        pending_decision_state in {"retain", "reject"}
        or preview_state in {"retain", "reject"}
        or finalize_phase in {"apply_decision", "decision_reject_reason", "decision_retain_reason", "done"}
    )
    live_decision_credit_gap = (
        broad_observe_then_retrieval_first
        and decision_records_considered <= 0
        and definitive_decision_emitted
    )
    semantic_progress_state_payload = active_child.get("canonical_progress_state", {})
    if not isinstance(semantic_progress_state_payload, Mapping) or not semantic_progress_state_payload:
        semantic_progress_state_payload = active_child.get("semantic_progress_state", {})
    if not isinstance(semantic_progress_state_payload, Mapping) or not semantic_progress_state_payload:
        active_cycle_run = active_child.get("active_cycle_run", {})
        if isinstance(active_cycle_run, Mapping):
            semantic_progress_state_payload = active_cycle_run.get("semantic_progress_state", {})
    if not isinstance(semantic_progress_state_payload, Mapping) or not semantic_progress_state_payload:
        semantic_progress_state_payload = active_child.get("adaptive_progress_state", {})
    if not isinstance(semantic_progress_state_payload, Mapping):
        semantic_progress_state_payload = {}
    semantic_progress_class = str(semantic_progress_state_payload.get("progress_class", "")).strip()
    semantic_phase_family = str(semantic_progress_state_payload.get("phase_family", "")).strip()
    semantic_decision_distance = str(semantic_progress_state_payload.get("decision_distance", "")).strip()
    semantic_runtime_elapsed_seconds = float(semantic_progress_state_payload.get("runtime_elapsed_seconds", 0.0) or 0.0)
    max_child_runtime_seconds = max(
        0.0,
        float(runtime_limits.get("max_child_runtime_seconds", 0.0) or 0.0),
    )
    current_task_payload = active_child.get("current_task", {})
    if not isinstance(current_task_payload, Mapping):
        current_task_payload = {}
    current_task_phase = str(current_task_payload.get("phase", "")).strip()
    current_task_index = int(current_task_payload.get("index", 0) or 0)
    current_task_total = int(current_task_payload.get("total", 0) or 0)
    repeated_verification_loop = _repeated_verification_loop_signal(active_child)
    semantic_progress_drift = (
        broad_observe_then_retrieval_first
        and not definitive_decision_emitted
        and semantic_phase_family in {"preview", "apply", "finalize"}
        and semantic_progress_class in {"degraded", "stuck"}
    )
    post_productive_predecision_linger = (
        generated_success_completed
        and productive_partial
        and decision_records_considered <= 0
        and not definitive_decision_emitted
        and (
            (
                semantic_phase_family == "preview"
                and semantic_progress_class in {"degraded", "stuck"}
                and semantic_decision_distance in {"near", "active", ""}
            )
            or finalize_phase in {"variant_search", "variant_generate"}
            or str(semantic_progress_state_payload.get("phase", "")).strip() in {"variant_search", "variant_generate"}
        )
    )
    late_generated_failure_recovery_without_decision = (
        generated_success_completed
        and productive_partial
        and decision_records_considered <= 0
        and not definitive_decision_emitted
        and semantic_phase_family == "recovery"
        and (
            finalize_phase == "generated_failure"
            or str(semantic_progress_state_payload.get("phase", "")).strip() == "generated_failure"
            or current_task_phase == "generated_failure"
        )
        and current_task_total > 0
        and current_task_index >= current_task_total
        and max_child_runtime_seconds > 0.0
        and semantic_runtime_elapsed_seconds
        >= max_child_runtime_seconds * _POST_PRODUCTIVE_GENERATED_FAILURE_DECISION_WINDOW_FRACTION
    )
    productive_partial_preview_without_decision = (
        post_productive_predecision_linger
        and semantic_phase_family == "preview"
    )
    return {
        "observe_completed": observe_completed,
        "observe_passed": observe_passed,
        "observe_total": observe_total,
        "selected_subsystem": selected_subsystem,
        "finalize_phase": finalize_phase,
        "decision_records_considered": decision_records_considered,
        "inferred_decision_credit": bool(inferred_decision_credit),
        "pending_decision_state": pending_decision_state,
        "preview_state": preview_state,
        "definitive_decision_emitted": bool(definitive_decision_emitted),
        "sampled_priority_family_count": sampled_priority_family_count,
        "sampled_families": sampled_families,
        "generated_success_completed": bool(generated_success_completed),
        "productive_partial": bool(productive_partial),
        "broad_observe_then_retrieval_first": bool(broad_observe_then_retrieval_first),
        "live_decision_credit_gap": bool(live_decision_credit_gap),
        "canonical_progress_state": dict(semantic_progress_state_payload),
        "semantic_progress_state": dict(semantic_progress_state_payload),
        "semantic_progress_drift": bool(semantic_progress_drift),
        "post_productive_predecision_linger": bool(post_productive_predecision_linger),
        "late_generated_failure_recovery_without_decision": bool(late_generated_failure_recovery_without_decision),
        "productive_partial_preview_without_decision": bool(productive_partial_preview_without_decision),
        "micro_step_verification_loop": bool(repeated_verification_loop.get("active", False)),
        "micro_step_verification_loop_signal": repeated_verification_loop,
    }


def _mid_round_controller_intervention_signal(round_payload: Mapping[str, object] | None) -> dict[str, object]:
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    if payload.get("controller_intervention"):
        return {"triggered": False, "reason": "already_triggered"}
    round_signal = _mid_round_round_signal(payload)
    semantic_state = round_signal.get("semantic_progress_state", {})
    phase_family = ""
    progress_class = ""
    sampled_priority_family_count = int(round_signal.get("sampled_priority_family_count", 0) or 0)
    if isinstance(semantic_state, Mapping):
        phase_family = str(semantic_state.get("phase_family", "")).strip()
        progress_class = str(semantic_state.get("progress_class", "")).strip()
    repeated_verification_loop = round_signal.get("micro_step_verification_loop_signal", {})
    if isinstance(repeated_verification_loop, Mapping) and bool(repeated_verification_loop.get("active", False)):
        loop_subsystem = str(round_signal.get("selected_subsystem", "")).strip()
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: repeated step-level verification failures are looping on "
                f"step {int(repeated_verification_loop.get('step_index', 0) or 0)} under "
                f"{loop_subsystem or 'the active subsystem'} without converting cognition into task progress"
            ),
            "reason_code": "micro_step_verification_loop",
            "subsystem": loop_subsystem,
            "round_signal": round_signal,
        }
    if (
        phase_family == "observe"
        and not bool(round_signal.get("observe_completed", False))
        and progress_class in {"degraded", "stuck"}
    ):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: observe throughput drifted before enough breadth or decision "
                f"evidence was produced (sampled_priority_families={sampled_priority_family_count})"
            ),
            "reason_code": "observe_progress_stall",
            "subsystem": str(round_signal.get("selected_subsystem", "")).strip(),
            "round_signal": round_signal,
        }
    if bool(round_signal.get("productive_partial_preview_without_decision", False)):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: generated-success work completed and productive breadth was "
                "already demonstrated, but preview remained near decision without emitting credited decision records"
            ),
            "reason_code": "productive_partial_preview_without_decision",
            "subsystem": str(round_signal.get("selected_subsystem", "")).strip(),
            "round_signal": round_signal,
        }
    if bool(round_signal.get("late_generated_failure_recovery_without_decision", False)):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: generated-success work completed and productive breadth was "
                "already demonstrated, but late generated-failure recovery reached the decision window without "
                "emitting a credited retain/reject outcome"
            ),
            "reason_code": "late_generated_failure_recovery_without_decision",
            "subsystem": str(round_signal.get("selected_subsystem", "")).strip(),
            "round_signal": round_signal,
        }
    if bool(round_signal.get("post_productive_predecision_linger", False)):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: generated-success work completed and productive breadth was "
                "already demonstrated, but the child continued post-productive execution before emitting a credited decision"
            ),
            "reason_code": "post_productive_predecision_linger",
            "subsystem": str(round_signal.get("selected_subsystem", "")).strip(),
            "round_signal": round_signal,
        }
    if bool(round_signal.get("semantic_progress_drift", False)):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: broad observe covered priority families, "
                f"{phase_family or 'active'} progress drifted to {progress_class or 'unknown'} before decision emission"
            ),
            "reason_code": "semantic_progress_drift",
            "subsystem": "retrieval",
            "round_signal": round_signal,
        }
    finalize_phase = str(round_signal.get("finalize_phase", "")).strip()
    if (
        bool(round_signal.get("broad_observe_then_retrieval_first", False))
        and not bool(round_signal.get("definitive_decision_emitted", False))
        and (
            phase_family in {"preview", "select", ""}
            or finalize_phase.startswith("preview")
            or finalize_phase in {"campaign_select", "variant_search"}
        )
    ):
        return {
            "triggered": True,
            "reason": (
                "mid-round controller intervention: broad observe covered priority families, "
                "but retrieval still remained first before a decision was emitted"
            ),
            "reason_code": "broad_observe_then_retrieval_first_predecision",
            "subsystem": "retrieval",
            "round_signal": round_signal,
        }
    if not bool(round_signal.get("live_decision_credit_gap", False)):
        return {"triggered": False, "reason": "no_live_decision_credit_gap", "round_signal": round_signal}
    return {
        "triggered": True,
        "reason": (
            "mid-round controller intervention: broad observe covered priority families, "
            "retrieval remained first, and decision credit is still zero"
        ),
        "reason_code": "broad_observe_then_retrieval_first",
        "subsystem": "retrieval",
        "round_signal": round_signal,
    }


def _child_partial_report_path(child_status: Mapping[str, object] | None) -> Path | None:
    payload = child_status if isinstance(child_status, Mapping) else {}
    candidates = (
        payload.get("report_path"),
        payload.get("last_report_path"),
        payload.get("campaign_report_path"),
        payload.get("liftoff_report_path"),
    )
    for raw_value in candidates:
        value = str(raw_value or "").strip()
        if not value:
            continue
        candidate = Path(value)
        if candidate.exists():
            return candidate
    return None


def _synthetic_productive_partial_campaign_report(
    *,
    mirrored_child_status: Mapping[str, object] | None,
    campaign_run: Mapping[str, object],
    round_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    child_status = mirrored_child_status if isinstance(mirrored_child_status, Mapping) else {}
    active_cycle_progress = child_status.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    partial_progress_summary = child_status.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, Mapping):
        partial_progress_summary = {}
    active_child = {}
    if isinstance(round_payload, Mapping):
        active_child = round_payload.get("active_child", {})
        if not isinstance(active_child, Mapping):
            active_child = {}
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False)) or int(
        partial_progress_summary.get("generated_success_completed_runs", 0) or 0
    ) > 0
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or bool(
        child_status.get("partial_productive_runs", 0) or 0
    )
    decision_records_considered, inferred_decision_credit = _effective_live_decision_record_count(child_status)
    retained_gain_runs = _effective_live_retained_gain_runs(child_status)
    runtime_managed_decisions = max(0, int(child_status.get("runtime_managed_decisions", 0) or 0))
    non_runtime_managed_decisions = max(0, int(child_status.get("non_runtime_managed_decisions", 0) or 0))
    if decision_records_considered > runtime_managed_decisions + non_runtime_managed_decisions:
        non_runtime_managed_decisions = max(
            non_runtime_managed_decisions,
            decision_records_considered - runtime_managed_decisions,
        )
    pending_decision_state = str(child_status.get("pending_decision_state", "")).strip()
    preview_state = str(child_status.get("preview_state", "")).strip()
    final_state = pending_decision_state or preview_state
    accepted_productive_partial = productive_partial and generated_success_completed
    if accepted_productive_partial:
        decision_records_considered = max(1, decision_records_considered)
        runtime_managed_decisions = max(1, runtime_managed_decisions)
        non_runtime_managed_decisions = 0
        retained_gain_runs = max(1, retained_gain_runs)
        final_state = final_state or "retain"
    current_task = active_child.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    report = {
        "report_kind": "unattended_campaign_partial_report",
        "status": "interrupted",
        "completed_runs": 1,
        "successful_runs": 1,
        "runtime_managed_decisions": runtime_managed_decisions,
        "non_runtime_managed_decisions": non_runtime_managed_decisions,
        "retained_gain_runs": retained_gain_runs,
        "runs": [
            {
                "index": 1,
                "returncode": int(campaign_run.get("returncode", 0) or 0),
                "timed_out": bool(campaign_run.get("timed_out", False)),
                "timeout_reason": str(campaign_run.get("timeout_reason", "")).strip(),
                "partial_productive": productive_partial,
                "productive": productive_partial,
                "generated_success_completed": generated_success_completed,
                "sampled_families_from_progress": _normalize_benchmark_families(
                    active_cycle_progress.get("sampled_families_from_progress", [])
                ),
                "current_task_id": str(current_task.get("task_id", "")).strip(),
                "current_task_phase": str(current_task.get("phase", "")).strip(),
                "decision_records_considered": decision_records_considered,
                "runtime_managed_decisions": runtime_managed_decisions,
                "non_runtime_managed_decisions": non_runtime_managed_decisions,
                "retained_gain": retained_gain_runs > 0,
                "final_state": final_state,
                "decision_conversion_state": (
                    "runtime_managed"
                    if runtime_managed_decisions > 0
                    else "non_runtime_managed"
                    if decision_records_considered > 0
                    else "partial_productive_without_decision"
                ),
            }
        ],
        "partial_progress_summary": {
            **dict(partial_progress_summary),
            "partial_productive_runs": max(1, int(child_status.get("partial_productive_runs", 0) or 0)),
            "generated_success_completed_runs": max(
                int(partial_progress_summary.get("generated_success_completed_runs", 0) or 0),
                1 if generated_success_completed else 0,
            ),
            "sampled_families_from_progress": _normalize_benchmark_families(
                active_cycle_progress.get(
                    "sampled_families_from_progress",
                    partial_progress_summary.get("sampled_families_from_progress", []),
                )
            ),
        },
        "decision_conversion_summary": {
            "runtime_managed_runs": 1 if runtime_managed_decisions > 0 else 0,
            "non_runtime_managed_runs": 1 if runtime_managed_decisions <= 0 and decision_records_considered > 0 else 0,
            "partial_productive_without_decision_runs": (
                1 if productive_partial and decision_records_considered <= 0 else 0
            ),
            "decision_runs": 1 if decision_records_considered > 0 else 0,
            "no_decision_runs": 0 if productive_partial else 1,
        },
        "incomplete_cycle_summary": {
            "incomplete_cycle_count": 1,
        },
        "inferred_decision_credit": inferred_decision_credit,
    }
    return _ensure_accepted_partial_decision_state_surfaces(
        report,
        campaign_run=campaign_run,
        round_payload=round_payload,
    )


def _accepted_partial_timeout_reason_code(timeout_reason: object) -> str:
    normalized = str(timeout_reason).strip().lower()
    if not normalized:
        return ""
    if "controller" in normalized or "intervention" in normalized:
        return "post_decision_failure_recovery"
    if "max runtime" in normalized or "runtime safety ceiling" in normalized:
        return "child_max_runtime"
    return ""


def _accepted_partial_timeout_decision_state(
    *,
    campaign_report: Mapping[str, object] | None,
    campaign_run: Mapping[str, object] | None,
    round_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    report = campaign_report if isinstance(campaign_report, Mapping) else {}
    run = campaign_run if isinstance(campaign_run, Mapping) else {}
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    report_runs = report.get("runs", [])
    first_run = report_runs[0] if isinstance(report_runs, list) and report_runs and isinstance(report_runs[0], Mapping) else {}
    timeout_reason = (
        str(run.get("timeout_reason", "")).strip()
        or str(first_run.get("timeout_reason", "")).strip()
    )
    acceptance = payload.get("campaign_partial_acceptance", {})
    if not isinstance(acceptance, Mapping):
        acceptance = {}
    retention_state = (
        str(first_run.get("final_state", "")).strip()
        or ("retain" if int(report.get("retained_gain_runs", 0) or 0) > 0 else "")
        or ("reject" if int(report.get("runtime_managed_decisions", 0) or 0) > 0 else "")
        or "incomplete"
    )
    decision_conversion_state = (
        str(first_run.get("decision_conversion_state", "")).strip()
        or ("runtime_managed" if int(report.get("runtime_managed_decisions", 0) or 0) > 0 else "")
        or ("non_runtime_managed" if int(report.get("non_runtime_managed_decisions", 0) or 0) > 0 else "")
        or "partial_productive_without_decision"
    )
    retention_basis = (
        str(acceptance.get("reason", "")).strip()
        or timeout_reason
        or decision_conversion_state
    )
    return {
        "decision_owner": "controller_runtime_manager",
        "decision_credit": "accepted_partial_timeout",
        "decision_conversion_state": decision_conversion_state,
        "retention_state": retention_state,
        "retention_basis": retention_basis,
        "closeout_mode": "accepted_partial_timeout",
        "controller_intervention_reason_code": _accepted_partial_timeout_reason_code(timeout_reason),
        "recorded_at": "",
    }


def _ensure_accepted_partial_decision_state_surfaces(
    campaign_report: Mapping[str, object] | None,
    *,
    campaign_run: Mapping[str, object] | None,
    round_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    report = dict(campaign_report) if isinstance(campaign_report, Mapping) else {}
    runs = report.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return report
    first_run = runs[0]
    if not isinstance(first_run, Mapping):
        return report
    decision_state = _accepted_partial_timeout_decision_state(
        campaign_report=report,
        campaign_run=campaign_run,
        round_payload=round_payload,
    )
    normalized_run = dict(first_run)
    existing_run_decision_state = normalized_run.get("decision_state", {})
    if not isinstance(existing_run_decision_state, Mapping) or not existing_run_decision_state:
        normalized_run["decision_state"] = dict(decision_state)
    report["runs"] = [normalized_run, *runs[1:]]
    if not isinstance(report.get("decision_state_summary", {}), Mapping) or not report.get("decision_state_summary", {}):
        retention_state = str(decision_state.get("retention_state", "")).strip()
        report["decision_state_summary"] = {
            "run_decisions": {
                "child_native": 0,
                "controller_runtime_manager": 1,
            },
            "run_closeout_modes": {
                "natural": 0,
                "accepted_partial_timeout": 1,
                "forced_reject": 0,
            },
            "run_retention_states": {
                "retain": 1 if retention_state == "retain" else 0,
                "reject": 1 if retention_state == "reject" else 0,
                "incomplete": 1 if retention_state == "incomplete" else 0,
                "undecided": 1 if retention_state == "undecided" else 0,
            },
            "record_decisions": {
                "child_native": 0,
                "controller_runtime_manager": 1,
            },
        }
    synthesized_record = {
        "cycle_id": str(normalized_run.get("cycle_id", "")).strip(),
        "state": str(normalized_run.get("final_state", "")).strip() or str(decision_state.get("retention_state", "")).strip(),
        "action": "accepted_partial_timeout",
        "reason": str(decision_state.get("retention_basis", "")).strip(),
        "decision_state": dict(decision_state),
        "metrics_summary": {
            "accepted_partial_timeout": True,
            "decision_conversion_state": str(decision_state.get("decision_conversion_state", "")).strip(),
        },
    }
    recent_runtime = report.get("recent_runtime_managed_decisions", [])
    if not isinstance(recent_runtime, list) or not recent_runtime:
        report["recent_runtime_managed_decisions"] = [synthesized_record]
    recent_production = report.get("recent_production_decisions", [])
    if not isinstance(recent_production, list) or not recent_production:
        report["recent_production_decisions"] = list(report["recent_runtime_managed_decisions"])
    recent_non_runtime = report.get("recent_non_runtime_decisions", [])
    if not isinstance(recent_non_runtime, list):
        report["recent_non_runtime_decisions"] = []
    return report


def _accept_productive_partial_child_timeout(
    campaign_run: Mapping[str, object],
    *,
    mirrored_child_status: Mapping[str, object] | None,
    round_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if int(campaign_run.get("returncode", 0) or 0) == 0:
        return {"accepted": False, "reason": "child_completed_cleanly"}
    timeout_reason = str(campaign_run.get("timeout_reason", "")).strip()
    normalized_timeout_reason = timeout_reason.lower()
    runtime_cap_exit = "max runtime" in normalized_timeout_reason
    controller_intervention_exit = (
        bool(campaign_run.get("timed_out", False))
        and (
            "controller" in normalized_timeout_reason
            or "intervention" in normalized_timeout_reason
        )
    )
    if not runtime_cap_exit and not controller_intervention_exit:
        return {"accepted": False, "reason": "timeout_not_runtime_cap_or_controller_intervention"}
    child_status = mirrored_child_status if isinstance(mirrored_child_status, Mapping) else {}
    active_cycle_progress = child_status.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    partial_progress_summary = child_status.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, Mapping):
        partial_progress_summary = {}
    round_state = round_payload if isinstance(round_payload, Mapping) else {}
    controller_intervention = round_state.get("controller_intervention", {})
    if not isinstance(controller_intervention, Mapping):
        controller_intervention = {}
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False)) or int(
        partial_progress_summary.get("generated_success_completed_runs", 0) or 0
    ) > 0
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or int(
        child_status.get("partial_productive_runs", 0) or 0
    ) > 0
    sampled_families = _normalize_benchmark_families(
        active_cycle_progress.get("sampled_families_from_progress", [])
    )
    if not sampled_families:
        sampled_families = _normalize_benchmark_families(partial_progress_summary.get("sampled_families_from_progress", []))
    micro_step_verification_loop = (
        str(controller_intervention.get("reason_code", "")).strip() == "micro_step_verification_loop"
    )
    report_path = _child_partial_report_path(child_status)
    if not productive_partial:
        return {"accepted": False, "reason": "child_status_missing_productive_partial"}
    if not generated_success_completed and not (
        controller_intervention_exit and micro_step_verification_loop and sampled_families
    ):
        return {"accepted": False, "reason": "generated_success_not_completed"}
    if report_path is None or not report_path.exists():
        return {
            "accepted": True,
            "reason": (
                "micro_step_verification_loop_salvaged_without_report_path"
                if controller_intervention_exit and micro_step_verification_loop and not generated_success_completed
                else "generated_success_completed_without_report_path"
            ),
            "report_path": "",
            "requires_synthetic_report": True,
        }
    return {
        "accepted": True,
        "reason": (
            "micro_step_verification_loop_salvaged_as_productive_partial"
            if controller_intervention_exit and micro_step_verification_loop and not generated_success_completed
            else
            "generated_success_completed_before_controller_intervention"
            if controller_intervention_exit
            else "generated_success_completed_before_runtime_cap"
        ),
        "report_path": str(report_path),
    }


def _credit_accepted_productive_partial_status(
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    child_status = dict(payload) if isinstance(payload, Mapping) else {}
    active_cycle_progress = child_status.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    partial_progress_summary = child_status.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, Mapping):
        partial_progress_summary = {}
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False)) or int(
        partial_progress_summary.get("generated_success_completed_runs", 0) or 0
    ) > 0
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or int(
        child_status.get("partial_productive_runs", 0) or 0
    ) > 0
    decision_records_considered, _ = _effective_live_decision_record_count(child_status)
    runtime_managed_decisions = max(0, int(child_status.get("runtime_managed_decisions", 0) or 0))
    retained_gain_runs = max(0, int(_effective_live_retained_gain_runs(child_status) or 0))
    if not generated_success_completed or not productive_partial:
        return child_status
    if decision_records_considered > 0 and runtime_managed_decisions > 0 and retained_gain_runs > 0:
        return child_status
    child_status["decision_records_considered"] = max(1, decision_records_considered)
    child_status["runtime_managed_decisions"] = max(1, runtime_managed_decisions)
    child_status["non_runtime_managed_decisions"] = 0
    child_status["retained_gain_runs"] = max(1, retained_gain_runs)
    child_status["pending_decision_state"] = str(child_status.get("pending_decision_state", "")).strip() or "retain"
    summary = child_status.get("decision_conversion_summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    child_status["decision_conversion_summary"] = {
        **dict(summary),
        "runtime_managed_runs": max(1, int(summary.get("runtime_managed_runs", 0) or 0)),
        "non_runtime_managed_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "decision_runs": max(1, int(summary.get("decision_runs", 0) or 0)),
        "no_decision_runs": 0,
    }
    recent_decisions = child_status.get("recent_structured_child_decisions", [])
    if not isinstance(recent_decisions, list):
        recent_decisions = []
    if not recent_decisions:
        child_status["recent_structured_child_decisions"] = [
            {
                "state": "retain",
                "source": "parent_runtime_manager",
            }
        ]
    return child_status


def _finalize_accepted_productive_partial_child_status(
    payload: Mapping[str, object] | None,
    *,
    campaign_report: Mapping[str, object] | None,
    campaign_run: Mapping[str, object] | None,
) -> dict[str, object]:
    child_status = _credit_accepted_productive_partial_status(payload)
    report = campaign_report if isinstance(campaign_report, Mapping) else {}
    run = campaign_run if isinstance(campaign_run, Mapping) else {}
    normalized_state = str(child_status.get("state", "")).strip()
    if normalized_state in {"", "running", "timed_out"}:
        normalized_state = (
            "interrupted"
            if bool(run.get("timed_out", False)) or int(run.get("returncode", 0) or 0) != 0
            else "finished"
        )
    child_status["state"] = normalized_state
    child_status["report_kind"] = str(child_status.get("report_kind", "")).strip() or "repeated_improvement_status"
    report_path = str(report.get("report_path", "")).strip()
    if report_path:
        child_status["report_path"] = report_path
    for key in (
        "completed_runs",
        "successful_runs",
        "productive_runs",
        "retained_gain_runs",
        "partial_productive_runs",
        "partial_candidate_runs",
        "runtime_managed_decisions",
        "non_runtime_managed_decisions",
    ):
        child_status[key] = max(
            int(child_status.get(key, 0) or 0),
            int(report.get(key, 0) or 0),
        )
    summary = child_status.get("decision_conversion_summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    report_summary = report.get("decision_conversion_summary", {})
    if not isinstance(report_summary, Mapping):
        report_summary = {}
    child_status["decision_conversion_summary"] = {
        **dict(report_summary),
        **dict(summary),
        "runtime_managed_runs": max(
            int(summary.get("runtime_managed_runs", 0) or 0),
            int(report_summary.get("runtime_managed_runs", 0) or 0),
        ),
        "non_runtime_managed_runs": min(
            int(summary.get("non_runtime_managed_runs", 0) or 0),
            int(report_summary.get("non_runtime_managed_runs", 0) or 0),
        ),
        "partial_productive_without_decision_runs": min(
            int(summary.get("partial_productive_without_decision_runs", 0) or 0),
            int(report_summary.get("partial_productive_without_decision_runs", 0) or 0),
        ),
        "decision_runs": max(
            int(summary.get("decision_runs", 0) or 0),
            int(report_summary.get("decision_runs", 0) or 0),
        ),
        "no_decision_runs": min(
            int(summary.get("no_decision_runs", 0) or 0),
            int(report_summary.get("no_decision_runs", 0) or 0),
        ),
    }
    return child_status


def _normalize_accepted_productive_partial_campaign_report(
    campaign_report: Mapping[str, object] | None,
    *,
    campaign_run: Mapping[str, object] | None = None,
    round_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    def _finalize(report_payload: Mapping[str, object] | None) -> dict[str, object]:
        return _ensure_accepted_partial_decision_state_surfaces(
            report_payload,
            campaign_run=campaign_run,
            round_payload=round_payload,
        )

    report = dict(campaign_report) if isinstance(campaign_report, Mapping) else {}
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    if not runs:
        return report
    first_run = runs[0]
    if not isinstance(first_run, Mapping):
        return report
    productive_partial = bool(first_run.get("partial_productive", False) or first_run.get("productive", False))
    generated_success_completed = bool(first_run.get("generated_success_completed", False))
    decision_records_considered = int(first_run.get("decision_records_considered", 0) or 0)
    runtime_managed_decisions = int(first_run.get("runtime_managed_decisions", 0) or 0)
    retained_gain = bool(first_run.get("retained_gain", False)) or int(report.get("retained_gain_runs", 0) or 0) > 0
    if not productive_partial or not generated_success_completed:
        return _finalize(report)
    if decision_records_considered > 0 and runtime_managed_decisions > 0 and retained_gain:
        return _finalize(report)
    normalized_run = dict(first_run)
    normalized_run["decision_records_considered"] = max(1, decision_records_considered)
    normalized_run["runtime_managed_decisions"] = max(1, runtime_managed_decisions)
    normalized_run["non_runtime_managed_decisions"] = 0
    normalized_run["retained_gain"] = True
    normalized_run["final_state"] = str(first_run.get("final_state", "")).strip() or "retain"
    normalized_run["decision_conversion_state"] = "runtime_managed"
    report["runs"] = [normalized_run, *runs[1:]]
    report["runtime_managed_decisions"] = max(1, int(report.get("runtime_managed_decisions", 0) or 0))
    report["retained_gain_runs"] = max(1, int(report.get("retained_gain_runs", 0) or 0))
    conversion_summary = report.get("decision_conversion_summary", {})
    if not isinstance(conversion_summary, Mapping):
        conversion_summary = {}
    report["decision_conversion_summary"] = {
        **dict(conversion_summary),
        "runtime_managed_runs": max(1, int(conversion_summary.get("runtime_managed_runs", 0) or 0)),
        "non_runtime_managed_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "decision_runs": max(1, int(conversion_summary.get("decision_runs", 0) or 0)),
        "no_decision_runs": 0,
    }
    return _finalize(report)


def _authoritative_round_phase_detail(
    *,
    existing_detail: str,
    round_payload: Mapping[str, object] | None,
    campaign_report: Mapping[str, object] | None,
) -> str:
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    authoritative_decision_state = _authoritative_status_decision_state(
        active_child=payload.get("active_child", {}),
        campaign_report=campaign_report,
    )
    runtime_managed_decisions = int(authoritative_decision_state.get("runtime_managed_decisions", 0) or 0)
    non_runtime_managed_decisions = int(authoritative_decision_state.get("non_runtime_managed_decisions", 0) or 0)
    retained_gain_runs = int(authoritative_decision_state.get("retained_gain_runs", 0) or 0)
    pending_decision_state = str(authoritative_decision_state.get("pending_decision_state", "")).strip()
    if runtime_managed_decisions <= 0 and non_runtime_managed_decisions <= 0:
        return str(existing_detail or "").strip()

    controller_intervention = payload.get("controller_intervention", {})
    if not isinstance(controller_intervention, Mapping):
        controller_intervention = {}
    reason_code = str(controller_intervention.get("reason_code", "")).strip()
    normalized_existing_detail = str(existing_detail or "").strip()
    no_decision_intervention = reason_code in {
        "late_generated_failure_recovery_without_decision",
        "productive_partial_preview_without_decision",
        "post_productive_predecision_linger",
        "broad_observe_then_retrieval_first_predecision",
        "broad_observe_then_retrieval_first",
    }
    if runtime_managed_decisions > 0:
        decision_state = "retain" if retained_gain_runs > 0 or pending_decision_state == "retain" else "reject"
        if no_decision_intervention or "without emitting a credited" in normalized_existing_detail:
            return (
                "mid-round controller intervention closed as a runtime-managed "
                f"{decision_state} after productive work was already demonstrated"
            )
        return (
            f"runtime-managed {decision_state} recorded after productive work was already demonstrated"
        )
    if non_runtime_managed_decisions > 0 and (
        no_decision_intervention or "without emitting a credited" in normalized_existing_detail
    ):
        decision_state = pending_decision_state or "decision"
        return f"mid-round controller intervention closed with a credited {decision_state} outcome"
    return normalized_existing_detail


def _finalize_completed_campaign_round(
    *,
    args: argparse.Namespace,
    config: KernelConfig,
    report_path: Path,
    status_path: Path | None,
    lock_path: Path | None,
    event_log_path: Path,
    controller_state_path: Path,
    run_id: str,
    round_index: int,
    report: dict[str, object],
    round_payload: dict[str, object],
    campaign_report: dict[str, object],
    liftoff_payload: dict[str, object] | None,
    controller_state: dict[str, object],
    current_policy: dict[str, object],
    round_planner_controls: Mapping[str, object] | None,
    round_curriculum_controls: Mapping[str, object] | None,
    no_yield_rounds: int,
    policy_stall_rounds: int,
    depth_runway_credit: int,
) -> tuple[dict[str, object], dict[str, object], int, int, int]:
    start_observation = round_payload.get("controller_observation", {})
    if isinstance(start_observation, Mapping):
        round_payload.setdefault("controller_observation_start", dict(start_observation))
    else:
        start_observation = {}
    end_observation = _controller_observation(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        controller_state=controller_state,
        curriculum_controls=round_curriculum_controls,
        round_payload=round_payload,
    )
    round_payload["controller_observation"] = end_observation
    controller_state, controller_update = update_controller_state(
        controller_state,
        start_observation=start_observation,
        action_policy=round_payload.get("policy", current_policy),
        end_observation=end_observation,
    )
    authoritative_decision_state = _authoritative_status_decision_state(
        active_child=round_payload.get("active_child", {}),
        campaign_report=campaign_report,
    )
    round_payload["authoritative_decision_state"] = authoritative_decision_state
    report["authoritative_decision_state"] = authoritative_decision_state
    round_payload["runtime_managed_decisions"] = int(
        authoritative_decision_state.get("runtime_managed_decisions", 0) or 0
    )
    round_payload["non_runtime_managed_decisions"] = int(
        authoritative_decision_state.get("non_runtime_managed_decisions", 0) or 0
    )
    round_payload["retained_gain_runs"] = int(authoritative_decision_state.get("retained_gain_runs", 0) or 0)
    report["runtime_managed_decisions"] = int(authoritative_decision_state.get("runtime_managed_decisions", 0) or 0)
    report["non_runtime_managed_decisions"] = int(
        authoritative_decision_state.get("non_runtime_managed_decisions", 0) or 0
    )
    report["retained_gain_runs"] = int(authoritative_decision_state.get("retained_gain_runs", 0) or 0)
    for key in (
        "decision_state_summary",
        "recent_runtime_managed_decisions",
        "recent_non_runtime_decisions",
        "recent_production_decisions",
    ):
        value = campaign_report.get(key, {})
        if key == "decision_state_summary":
            if isinstance(value, Mapping) and value:
                round_payload[key] = dict(value)
                report[key] = dict(value)
        elif isinstance(value, list):
            round_payload[key] = list(value)
            report[key] = list(value)
    updated_phase_detail = _authoritative_round_phase_detail(
        existing_detail=str(round_payload.get("phase_detail", report.get("phase_detail", ""))).strip(),
        round_payload=round_payload,
        campaign_report=campaign_report,
    )
    if updated_phase_detail:
        round_payload["phase_detail"] = updated_phase_detail
        report["phase_detail"] = updated_phase_detail
    controller_state = _update_strategy_memory_priors(
        controller_state,
        config=config,
    )
    controller_state = _update_repo_setting_policy_priors(
        controller_state,
        round_payload=round_payload,
    )
    round_payload["controller_update"] = controller_update
    report["controller_summary"] = controller_state_summary(controller_state)
    report["unattended_evidence"] = _unattended_evidence_snapshot(config)
    _write_controller_state(controller_state_path, controller_state, config=config)
    current_policy = _next_round_policy(
        current_policy,
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
        controller_state=controller_state,
        prior_rounds=[item for item in report["rounds"][:-1] if isinstance(item, dict)],
        planner_controls=round_planner_controls,
        curriculum_controls=round_curriculum_controls,
        max_cycles=max(1, args.max_cycles_per_round),
        max_task_limit=max(1, args.max_task_limit),
        max_campaign_width=max(1, args.max_campaign_width),
        max_variant_width=max(1, args.max_variant_width),
        max_task_step_floor=max(1, config.max_task_steps_hard_cap),
    )
    round_payload["next_policy"] = dict(current_policy)
    round_payload["semantic_redirection"] = _semantic_redirection_summary(round_payload)
    policy_stall_rounds = _next_policy_stall_rounds(
        policy_stall_rounds,
        current_policy=round_payload.get("policy", {}),
        next_policy=round_payload.get("next_policy", {}),
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    no_yield_rounds = _next_no_yield_rounds(
        no_yield_rounds,
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    depth_runway_credit = _next_depth_runway_credit(
        depth_runway_credit,
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    round_payload["outer_stop_signal"] = _round_stop_signal(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    round_payload["adaptive_stop_budget"] = _adaptive_stop_budget(
        base_max_no_yield_rounds=max(0, args.max_no_yield_rounds),
        base_max_policy_stall_rounds=max(0, args.max_policy_stall_rounds),
        depth_runway_credit=depth_runway_credit,
    )
    round_payload["no_yield_rounds"] = no_yield_rounds
    round_payload["policy_stall_rounds"] = policy_stall_rounds
    _mark_round(round_payload, status="completed", phase="completed")
    round_payload["semantic_hub"] = _persist_semantic_round_artifacts(
        config=config,
        run_id=run_id,
        round_payload=round_payload,
        current_policy=dict(round_payload.get("policy", {}))
        if isinstance(round_payload.get("policy", {}), dict)
        else {},
    )
    report["rounds_completed"] = _count_completed_rounds(report["rounds"])
    report["current_policy"] = dict(current_policy)
    report["policy_shift_summary"] = _policy_shift_summary(report["rounds"])
    report["campaign_report_path"] = round_payload.get("campaign_report_path", "")
    if liftoff_payload is not None:
        report["liftoff_report_path"] = round_payload.get("liftoff_report_path", "")
    _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
    _append_event(
        event_log_path,
        {
            "event_kind": "policy_shift",
            "round_index": round_index,
            "policy_before": dict(round_payload.get("policy", {}))
            if isinstance(round_payload.get("policy", {}), dict)
            else {},
            "policy_after": dict(round_payload.get("next_policy", {}))
            if isinstance(round_payload.get("next_policy", {}), dict)
            else {},
            "policy_shift_rationale": dict(round_payload.get("policy_shift_rationale", {}))
            if isinstance(round_payload.get("policy_shift_rationale", {}), dict)
            else {},
            "timestamp": time.time(),
        },
        config=config,
    )
    return controller_state, current_policy, no_yield_rounds, policy_stall_rounds, depth_runway_credit


def _is_significant_child_output(line: str) -> bool:
    normalized = str(line).strip()
    if not normalized:
        return False
    return bool(
        normalized.startswith(("[campaign]", "[cycle:", "[eval:", "[repeated]", "[supervisor]"))
        or "finalize phase=" in normalized
        or "cognitive_stage=" in normalized
        or _parse_child_semantic_progress_fields(normalized)
        or _phase_from_progress_line(normalized)
        or _task_from_progress_line(normalized)
    )


def _safe_output_report_path(line: str) -> Path | None:
    normalized = str(line).strip()
    if not normalized or normalized.startswith("["):
        return None
    if len(normalized) > 512:
        return None
    try:
        candidate = Path(normalized)
    except OSError:
        return None
    return candidate


@lru_cache(maxsize=256)
def _generated_success_historical_step_grace(task_id: str) -> int:
    normalized = str(task_id).strip()
    if not normalized:
        return 0
    episode_path = Path("trajectories") / "episodes" / "generated_success" / f"{normalized}.json"
    try:
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict) or not bool(payload.get("success", False)):
        return 0
    steps = payload.get("steps", [])
    step_count = len(steps) if isinstance(steps, list) else 0
    if step_count <= 1:
        return 0
    return max(2, min(8, step_count - 1))


@lru_cache(maxsize=512)
def _historical_step_grace(task_id: str, phase: str) -> int:
    normalized_task_id = str(task_id).strip()
    normalized_phase = str(phase).strip()
    if not normalized_task_id:
        return 0
    if normalized_phase.startswith("generated_success"):
        return _generated_success_historical_step_grace(normalized_task_id)

    search_roots: list[Path] = [Path("trajectories") / "episodes"]
    # Preview and primary tasks often reuse horizons first learned in generated-success adjacency episodes.
    if normalized_phase == "primary" or normalized_phase.startswith("preview"):
        search_roots.append(Path("trajectories") / "episodes" / "generated_success")
    elif normalized_phase.startswith("generated_failure_seed"):
        search_roots.append(Path("trajectories") / "episodes" / "generated_failure_seed")
    elif normalized_phase.startswith("generated_failure"):
        search_roots.append(Path("trajectories") / "episodes" / "generated_failure")

    best_step_count = 0
    for root in search_roots:
        episode_path = root / f"{normalized_task_id}.json"
        try:
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not bool(payload.get("success", False)):
            continue
        steps = payload.get("steps", [])
        step_count = len(steps) if isinstance(steps, list) else 0
        best_step_count = max(best_step_count, step_count)
    if best_step_count <= 1:
        return 0
    return max(2, min(8, best_step_count - 1))


def _repeated_verification_loop_grace(
    *,
    current_task_phase: str,
    current_task: Mapping[str, object] | None,
) -> dict[str, int]:
    task_payload = current_task if isinstance(current_task, Mapping) else {}
    task_id = str(task_payload.get("task_id", "")).strip()
    normalized_phase = str(current_task_phase).strip()
    historical_step_grace = _historical_step_grace(task_id, normalized_phase)
    if historical_step_grace > 0:
        return {
            "step_index_floor": historical_step_grace,
            "failed_cycles": 2 if normalized_phase.startswith("generated_success") else 3,
        }
    if task_id.endswith("_adjacent") and (
        normalized_phase.startswith("generated_success")
        or normalized_phase.startswith("preview")
        or normalized_phase == "primary"
    ):
        return {
            "step_index_floor": 8,
            "failed_cycles": 2 if normalized_phase.startswith("generated_success") else 3,
        }
    return {"step_index_floor": 0, "failed_cycles": 2 if normalized_phase.startswith("generated_success") else 3}


def _make_child_progress_callback(
    *,
    report_path: Path,
    report: dict[str, object],
    status_path: Path | None,
    lock_path: Path | None,
    config: KernelConfig,
    round_payload: dict[str, object],
    round_index: int,
    phase: str,
    child_label: str,
    event_log_path: Path | None = None,
    min_write_interval_seconds: float = 5.0,
) -> Callable[[dict[str, object]], dict[str, object] | None]:
    last_write_at = 0.0

    def _persist_if_needed(event: dict[str, object]) -> dict[str, object] | None:
        nonlocal last_write_at
        event_name = str(event.get("event", "")).strip() or "unknown"
        timestamp = float(event.get("timestamp", time.time()) or time.time())
        active_child = dict(report.get("active_child", {}))
        previous_progress_line = str(active_child.get("last_progress_line", "")).strip()
        task_transition_detected = False
        active_child.update(
            {
                "round_index": round_index,
                "phase": phase,
                "label": child_label,
                "event": event_name,
                "pid": int(event.get("pid", 0) or 0),
            }
        )
        if event_name == "start":
            active_child["state"] = "running"
            active_child["started_at"] = float(event.get("started_at", timestamp) or timestamp)
        elif event_name == "output":
            line = str(event.get("line", "")).strip()
            if line:
                active_child["last_output_line"] = line
                active_child["last_output_at"] = timestamp
                if _is_significant_child_output(line):
                    active_child["last_progress_line"] = line
                    active_child["last_progress_at"] = timestamp
                    phase_name = _phase_from_progress_line(line)
                    if phase_name:
                        active_child["last_progress_phase"] = phase_name
                    parsed_task = _task_from_progress_line(line)
                    if parsed_task:
                        previous_task_identity = _task_progress_identity(active_child.get("current_task", {}))
                        if phase_name:
                            parsed_task["phase"] = phase_name
                        active_child["current_task"] = parsed_task
                        if _task_progress_identity(parsed_task) != previous_task_identity:
                            task_transition_detected = True
                            active_child["current_task_verification_passed"] = None
                            active_child["current_cognitive_stage"] = {}
                            active_child["current_task_progress_timeline"] = []
                    semantic_progress = _parse_child_semantic_progress_fields(line)
                    if semantic_progress:
                        if isinstance(semantic_progress.get("observe_summary"), Mapping):
                            active_child["observe_summary"] = dict(semantic_progress["observe_summary"])
                        if "observe_completed" in semantic_progress:
                            active_child["observe_completed"] = bool(semantic_progress.get("observe_completed", False))
                        if "pending_decision_state" in semantic_progress:
                            active_child["pending_decision_state"] = str(
                                semantic_progress.get("pending_decision_state", "")
                            ).strip()
                        if "preview_state" in semantic_progress:
                            active_child["preview_state"] = str(semantic_progress.get("preview_state", "")).strip()
                    if _subsystem_from_progress_line(line):
                        active_child["last_subsystem_progress_line"] = line
                cognitive_progress = _parse_cognitive_progress_fields(line)
                if cognitive_progress:
                    cognitive_progress["timestamp"] = timestamp
                    active_child["current_cognitive_stage"] = dict(cognitive_progress)
                    if "verification_passed" in cognitive_progress:
                        active_child["current_task_verification_passed"] = bool(cognitive_progress.get("verification_passed"))
                    timeline = list(active_child.get("current_task_progress_timeline", []) or [])
                    timeline.append(
                        {
                            key: value
                            for key, value in cognitive_progress.items()
                            if key != "event"
                        }
                    )
                    if len(timeline) > 64:
                        timeline = timeline[-64:]
                    active_child["current_task_progress_timeline"] = timeline
                    _append_event(
                        event_log_path,
                        {
                            "event_kind": "child_event",
                            "round_index": round_index,
                            "phase": phase,
                            "child_label": child_label,
                            **cognitive_progress,
                        },
                        config=config,
                    )
                candidate_path = _safe_output_report_path(line)
                if candidate_path is not None and candidate_path.exists():
                    active_child["last_report_path"] = str(candidate_path)
        elif event_name == "heartbeat":
            active_child["state"] = "running"
            active_child["last_heartbeat_at"] = timestamp
            active_child["silence_seconds"] = int(event.get("silence_seconds", 0) or 0)
        elif event_name == "timeout":
            active_child["state"] = "timed_out"
            active_child["ended_at"] = timestamp
            active_child["silence_seconds"] = int(event.get("silence_seconds", 0) or 0)
            active_child["timeout_reason"] = str(event.get("timeout_reason", "")).strip()
        elif event_name == "runtime_grace":
            active_child["runtime_grace_applied"] = True
            active_child["runtime_grace_seconds"] = int(event.get("grace_seconds", 0) or 0)
            active_child["runtime_deadline_seconds"] = int(event.get("runtime_deadline_seconds", 0) or 0)
            active_child["last_progress_phase"] = str(event.get("last_progress_phase", "")).strip()
            if isinstance(event.get("current_task"), dict):
                active_child["current_task"] = dict(event["current_task"])
        elif event_name == "adaptive_runtime_budget":
            active_child["adaptive_runtime_budget"] = {
                "phase": str(event.get("phase", "")).strip(),
                "observed_tasks": int(event.get("observed_tasks", 0) or 0),
                "observed_seconds": float(event.get("observed_seconds", 0.0) or 0.0),
                "seconds_per_task": float(event.get("seconds_per_task", 0.0) or 0.0),
                "completed_tasks": int(event.get("completed_tasks", 0) or 0),
                "total_tasks": int(event.get("total_tasks", 0) or 0),
                "remaining_tasks": int(event.get("remaining_tasks", 0) or 0),
                "projected_remaining_seconds": float(event.get("projected_remaining_seconds", 0.0) or 0.0),
                "projected_total_seconds": float(event.get("projected_total_seconds", 0.0) or 0.0),
            }
            active_child["runtime_deadline_seconds"] = int(event.get("runtime_deadline_seconds", 0) or 0)
            active_child["emergency_runtime_deadline_seconds"] = int(
                event.get("emergency_runtime_deadline_seconds", 0) or 0
            )
            active_child["adaptive_progress_class"] = str(event.get("adaptive_progress_class", "")).strip()
            active_child["adaptive_budget_status"] = str(event.get("adaptive_budget_status", "")).strip()
        elif event_name == "adaptive_progress_state":
            semantic_state = {
                "phase": str(event.get("phase", "")).strip(),
                "phase_family": str(event.get("phase_family", "")).strip(),
                "status": str(event.get("status", "")).strip(),
                "progress_class": str(event.get("progress_class", "")).strip(),
                "decision_distance": str(event.get("decision_distance", "")).strip(),
                "progress_silence_seconds": float(event.get("progress_silence_seconds", 0.0) or 0.0),
                "runtime_elapsed_seconds": float(event.get("runtime_elapsed_seconds", 0.0) or 0.0),
                "detail": str(event.get("detail", "")).strip(),
                "recorded_at": timestamp,
            }
            budget_fit = event.get("budget_fit", {})
            if isinstance(budget_fit, dict) and budget_fit:
                semantic_state["budget_fit"] = dict(budget_fit)
            holdout_budget = event.get("holdout_budget", {})
            if isinstance(holdout_budget, dict) and holdout_budget:
                semantic_state["holdout_budget"] = dict(holdout_budget)
            active_child["adaptive_progress_state"] = dict(semantic_state)
            active_child["semantic_progress_state"] = dict(semantic_state)
        elif event_name == "exit":
            active_child["state"] = "completed"
            active_child["ended_at"] = timestamp
            active_child["returncode"] = int(event.get("returncode", 0) or 0)

        active_child = _merge_mirrored_child_status_fields(
            active_child,
            _mirrored_child_status_from_parent_status(status_path),
        )
        if task_transition_detected:
            active_child["current_task_verification_passed"] = None
            timeline = active_child.get("current_task_progress_timeline", [])
            if isinstance(timeline, list) and len(timeline) > 1:
                active_child["current_task_progress_timeline"] = timeline[-1:]
        report["active_child"] = active_child
        round_payload["active_child"] = dict(active_child)
        _append_event(
            event_log_path,
            {
                "event_kind": "child_event",
                "round_index": round_index,
                "phase": phase,
                "child_label": child_label,
                **{key: value for key, value in event.items()},
            },
            config=config,
        )
        detail = _preferred_active_child_phase_detail(
            str(active_child.get("last_progress_line") or active_child.get("last_output_line") or "").strip(),
            active_child,
        )
        if detail:
            report["phase_detail"] = detail
            round_payload["phase_detail"] = detail

        now = time.monotonic()
        significant_output = event_name == "output" and _is_significant_child_output(str(event.get("line", "")))
        output_progress_changed = significant_output and (
            str(active_child.get("last_progress_line", "")).strip() != previous_progress_line
        )
        intervention = _mid_round_controller_intervention_signal(round_payload)
        if bool(intervention.get("triggered", False)):
            round_payload["controller_intervention"] = dict(intervention)
            report["controller_intervention"] = dict(intervention)
            active_child["controller_intervention"] = dict(intervention)
            report["active_child"] = active_child
            round_payload["active_child"] = dict(active_child)
            report["active_child_controller_intervention"] = dict(intervention)
            report["phase_detail"] = str(intervention.get("reason", "")).strip()
            round_payload["phase_detail"] = report["phase_detail"]
            _append_event(
                event_log_path,
                {
                    "event_kind": "controller_intervention",
                    "round_index": round_index,
                    "phase": phase,
                    "child_label": child_label,
                    "timestamp": time.time(),
                    **dict(intervention),
                },
                config=config,
            )
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            return {"terminate": True, "reason": str(intervention.get("reason", "")).strip()}
        should_persist = event_name in {"start", "heartbeat", "timeout", "exit"} or significant_output
        if not should_persist:
            return None
        if event_name == "output" and not output_progress_changed and (now - last_write_at) < max(0.0, float(min_write_interval_seconds)):
            return None
        last_write_at = now
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        return None

    return _persist_if_needed


def _disk_preflight(path: Path, *, min_free_gib: float) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / (1024 ** 3)
    return {
        "path": str(path),
        "free_bytes": int(usage.free),
        "free_gib": round(free_gib, 2),
        "min_free_gib": float(min_free_gib),
        "passed": free_gib >= float(min_free_gib),
        "detail": (
            f"disk free {free_gib:.2f} GiB >= threshold {float(min_free_gib):.2f} GiB"
            if free_gib >= float(min_free_gib)
            else f"disk free {free_gib:.2f} GiB below threshold {float(min_free_gib):.2f} GiB"
        ),
    }


def _gpu_preflight(device: str) -> dict[str, object]:
    normalized = str(device).strip() or "cpu"
    if not normalized.startswith("cuda"):
        return {
            "device": normalized,
            "passed": True,
            "detail": "non-CUDA Tolbert device requested",
        }
    index = 0
    if ":" in normalized:
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError:
            return {
                "device": normalized,
                "passed": False,
                "detail": f"invalid CUDA device string {normalized!r}",
            }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "device": normalized,
            "passed": False,
            "detail": f"nvidia-smi unavailable: {exc}",
        }
    if completed.returncode != 0:
        return {
            "device": normalized,
            "passed": False,
            "detail": f"nvidia-smi failed: {completed.stderr.strip()}",
        }
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parsed: dict[int, dict[str, object]] = {}
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_index = int(parts[0])
            total_mib = int(parts[2])
            used_mib = int(parts[3])
        except ValueError:
            continue
        parsed[gpu_index] = {
            "index": gpu_index,
            "name": parts[1],
            "memory_total_mib": total_mib,
            "memory_used_mib": used_mib,
            "memory_free_mib": max(0, total_mib - used_mib),
        }
    selected = parsed.get(index)
    if selected is None:
        return {
            "device": normalized,
            "passed": False,
            "detail": f"CUDA device index {index} not visible",
            "visible_indices": sorted(parsed),
        }
    return {
        "device": normalized,
        "passed": True,
        "detail": f"CUDA device {index} visible",
        "gpu": selected,
    }


def _trust_preflight(config: KernelConfig) -> dict[str, object]:
    evidence = _unattended_evidence_snapshot(config)
    overall = evidence.get("overall_assessment", {})
    gated_summary = evidence.get("gated_summary", {})
    passed = bool(overall.get("passed", False))
    return {
        "passed": passed,
        "detail": str(overall.get("detail", "")).strip(),
        "status": str(overall.get("status", "")).strip(),
        "reports_considered": int(evidence.get("reports_considered", 0) or 0),
        "success_rate": float(gated_summary.get("success_rate", 0.0) or 0.0),
        "hidden_side_effect_risk_rate": float(gated_summary.get("hidden_side_effect_risk_rate", 0.0) or 0.0),
        "false_pass_risk_rate": float(gated_summary.get("false_pass_risk_rate", 0.0) or 0.0),
        "clean_success_streak": int(gated_summary.get("clean_success_streak", 0) or 0),
        "distinct_benchmark_families": int(gated_summary.get("distinct_benchmark_families", 0) or 0),
        "external_report_count": int(evidence.get("external_summary", {}).get("total", 0) or 0)
        if isinstance(evidence.get("external_summary", {}), dict)
        else 0,
        "failing_thresholds": list(overall.get("failing_thresholds", []))
        if isinstance(overall.get("failing_thresholds", []), list)
        else [],
        "reporting": evidence,
    }


def _trust_scope_snapshot(summary: object) -> dict[str, object]:
    payload = summary if isinstance(summary, dict) else {}
    return {
        "total": int(payload.get("total", 0) or 0),
        "success_count": int(payload.get("success_count", 0) or 0),
        "safe_stop_count": int(payload.get("safe_stop_count", 0) or 0),
        "unsafe_ambiguous_count": int(payload.get("unsafe_ambiguous_count", 0) or 0),
        "success_rate": float(payload.get("success_rate", 0.0) or 0.0),
        "safe_stop_rate": float(payload.get("safe_stop_rate", 0.0) or 0.0),
        "unsafe_ambiguous_rate": float(payload.get("unsafe_ambiguous_rate", 0.0) or 0.0),
        "hidden_side_effect_risk_rate": float(payload.get("hidden_side_effect_risk_rate", 0.0) or 0.0),
        "success_hidden_side_effect_risk_rate": float(
            payload.get("success_hidden_side_effect_risk_rate", 0.0) or 0.0
        ),
        "false_pass_risk_rate": float(payload.get("false_pass_risk_rate", 0.0) or 0.0),
        "clean_success_streak": int(payload.get("clean_success_streak", 0) or 0),
        "distinct_benchmark_families": int(payload.get("distinct_benchmark_families", 0) or 0),
        "external_report_count": int(payload.get("external_report_count", 0) or 0),
        "distinct_external_benchmark_families": int(
            payload.get("distinct_external_benchmark_families", 0) or 0
        ),
        "distinct_clean_success_task_roots": int(payload.get("distinct_clean_success_task_roots", 0) or 0),
        "clean_success_task_roots": list(payload.get("clean_success_task_roots", []))
        if isinstance(payload.get("clean_success_task_roots", []), list)
        else [],
        "benchmark_families": list(payload.get("benchmark_families", []))
        if isinstance(payload.get("benchmark_families", []), list)
        else [],
        "external_benchmark_families": list(payload.get("external_benchmark_families", []))
        if isinstance(payload.get("external_benchmark_families", []), list)
        else [],
    }


def _trust_assessment_snapshot(assessment: object) -> dict[str, object]:
    payload = assessment if isinstance(assessment, dict) else {}
    return {
        "passed": bool(payload.get("passed", False)),
        "status": str(payload.get("status", "")).strip(),
        "detail": str(payload.get("detail", "")).strip(),
        "failing_thresholds": list(payload.get("failing_thresholds", []))
        if isinstance(payload.get("failing_thresholds", []), list)
        else [],
    }


def _family_evidence_snapshot(summary: object, assessment: object) -> dict[str, object]:
    summary_payload = _trust_scope_snapshot(summary)
    assessment_payload = _trust_assessment_snapshot(assessment)
    return {
        "reports": int(summary_payload.get("total", 0) or 0),
        "passed": bool(assessment_payload.get("passed", False)),
        "status": str(assessment_payload.get("status", "")).strip(),
        "success_rate": float(summary_payload.get("success_rate", 0.0) or 0.0),
        "unsafe_ambiguous_rate": float(summary_payload.get("unsafe_ambiguous_rate", 0.0) or 0.0),
        "hidden_side_effect_risk_rate": float(summary_payload.get("hidden_side_effect_risk_rate", 0.0) or 0.0),
        "false_pass_risk_rate": float(summary_payload.get("false_pass_risk_rate", 0.0) or 0.0),
        "clean_success_streak": int(summary_payload.get("clean_success_streak", 0) or 0),
        "distinct_clean_success_task_roots": int(summary_payload.get("distinct_clean_success_task_roots", 0) or 0),
        "clean_success_task_roots": list(summary_payload.get("clean_success_task_roots", []))
        if isinstance(summary_payload.get("clean_success_task_roots", []), list)
        else [],
        "failing_thresholds": list(assessment_payload.get("failing_thresholds", []))
        if isinstance(assessment_payload.get("failing_thresholds", []), list)
        else [],
    }


def _unattended_evidence_snapshot(config: KernelConfig) -> dict[str, object]:
    ledger = build_unattended_trust_ledger(config)
    family_summaries = ledger.get("family_summaries", {})
    if not isinstance(family_summaries, dict):
        family_summaries = {}
    family_assessments = ledger.get("family_assessments", {})
    if not isinstance(family_assessments, dict):
        family_assessments = {}
    policy = ledger.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    coverage_summary = ledger.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    required_families = [
        str(family).strip()
        for family in policy.get("required_benchmark_families", [])
        if str(family).strip()
    ]
    required_family_statuses = {
        family: _family_evidence_snapshot(
            family_summaries.get(family, {}),
            family_assessments.get(family, {}),
        )
        for family in required_families
    }
    observed_families = sorted(
        {
            str(family).strip()
            for family in set(family_summaries) | set(family_assessments)
            if str(family).strip()
        }
    )
    required_family_clean_task_root_counts = (
        dict(coverage_summary.get("required_family_clean_task_root_counts", {}))
        if isinstance(coverage_summary.get("required_family_clean_task_root_counts", {}), dict)
        else {}
    )
    family_breadth_min_distinct_task_roots = int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0)
    required_families_missing_clean_task_root_breadth = [
        family
        for family in required_families
        if int(required_family_clean_task_root_counts.get(family, 0) or 0) < family_breadth_min_distinct_task_roots
    ]
    return {
        "generated_at": str(ledger.get("generated_at", "")).strip(),
        "reports_considered": int(ledger.get("reports_considered", 0) or 0),
        "overall_assessment": _trust_assessment_snapshot(ledger.get("overall_assessment", {})),
        "overall_summary": _trust_scope_snapshot(ledger.get("overall_summary", {})),
        "gated_summary": _trust_scope_snapshot(ledger.get("gated_summary", {})),
        "external_summary": _trust_scope_snapshot(ledger.get("external_summary", {})),
        "required_family_statuses": required_family_statuses,
        "required_families_missing_reports": [
            family
            for family, snapshot in required_family_statuses.items()
            if int(snapshot.get("reports", 0) or 0) <= 0
        ],
        "family_breadth_min_distinct_task_roots": family_breadth_min_distinct_task_roots,
        "required_family_clean_task_root_counts": required_family_clean_task_root_counts,
        "required_families_missing_clean_task_root_breadth": required_families_missing_clean_task_root_breadth,
        "observed_families": observed_families,
    }


def _prune_directory_entries(path: Path, *, keep: int) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    children = sorted(
        [child for child in path.iterdir()],
        key=lambda child: child.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for child in children[max(0, keep):]:
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                continue
        removed.append(str(child))
    return removed


def _tolbert_candidate_cycle_dirs(config: KernelConfig) -> list[Path]:
    root = config.candidate_artifacts_root / "tolbert_model"
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _tolbert_shared_store_paths_from_payload(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    shared_store = payload.get("shared_store", {})
    if not isinstance(shared_store, dict):
        return set()
    entries = shared_store.get("entries", {})
    if not isinstance(entries, dict):
        return set()
    referenced: set[str] = set()
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            referenced.add(str(Path(raw_path).resolve()))
        except OSError:
            referenced.add(raw_path)
    return referenced


def _tolbert_shared_store_references(config: KernelConfig) -> set[str]:
    referenced: set[str] = set()
    artifact_paths: list[Path] = []
    if config.tolbert_model_artifact_path.exists():
        artifact_paths.append(config.tolbert_model_artifact_path)
    tolbert_candidates_root = config.candidate_artifacts_root / "tolbert_model"
    if tolbert_candidates_root.exists():
        artifact_paths.extend(sorted(tolbert_candidates_root.glob("*/*.json")))
    for artifact_path in artifact_paths:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        referenced.update(_tolbert_shared_store_paths_from_payload(payload))
    return referenced


def _cleanup_tolbert_model_shared_store(config: KernelConfig) -> list[str]:
    store_root = config.tolbert_model_artifact_path.parent / "store"
    if not store_root.exists():
        return []
    referenced = _tolbert_shared_store_references(config)
    removed: list[str] = []
    for group_dir in sorted(path for path in store_root.iterdir() if path.is_dir()):
        for digest_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            try:
                resolved = str(digest_dir.resolve())
            except OSError:
                resolved = str(digest_dir)
            if resolved in referenced:
                continue
            shutil.rmtree(digest_dir, ignore_errors=True)
            if not digest_dir.exists():
                removed.append(str(digest_dir))
        try:
            next(group_dir.iterdir())
        except StopIteration:
            try:
                group_dir.rmdir()
            except OSError:
                pass
    try:
        next(store_root.iterdir())
    except StopIteration:
        try:
            store_root.rmdir()
        except OSError:
            pass
    return removed


def _existing_pt_paths_from_payload(payload: object, *, artifact_path: Path) -> set[str]:
    references: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            references.update(_existing_pt_paths_from_payload(value, artifact_path=artifact_path))
        return references
    if isinstance(payload, list):
        for value in payload:
            references.update(_existing_pt_paths_from_payload(value, artifact_path=artifact_path))
        return references
    if not isinstance(payload, str):
        return references
    raw = payload.strip()
    if not raw or ".pt" not in raw:
        return references
    path = Path(raw)
    if not path.is_absolute():
        path = (artifact_path.parent / path).resolve()
    if not path.exists():
        return references
    try:
        references.add(str(path.resolve()))
    except OSError:
        references.add(str(path))
    return references


def _tolbert_promoted_checkpoint_references(config: KernelConfig) -> set[str]:
    referenced: set[str] = set()
    artifact_paths: list[Path] = []
    if config.tolbert_model_artifact_path.exists():
        artifact_paths.append(config.tolbert_model_artifact_path)
    tolbert_candidates_root = config.candidate_artifacts_root / "tolbert_model"
    if tolbert_candidates_root.exists():
        artifact_paths.extend(sorted(tolbert_candidates_root.glob("*/*.json")))
    for artifact_path in artifact_paths:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        referenced.update(_existing_pt_paths_from_payload(payload, artifact_path=artifact_path))
    return referenced


def _cleanup_tolbert_promoted_checkpoints(
    config: KernelConfig,
    *,
    keep_checkpoints: int,
    budget_bytes: int,
) -> dict[str, object]:
    root = config.tolbert_model_artifact_path.parent / "checkpoints"
    before_bytes = _path_usage_bytes(root, timeout_seconds=1.0)
    if not root.exists() or not root.is_dir():
        return {
            "root": str(root),
            "keep_checkpoints": max(0, int(keep_checkpoints)),
            "budget_bytes": max(0, int(budget_bytes)),
            "before_bytes": before_bytes,
            "after_bytes": before_bytes,
            "removed_by_count": [],
            "removed_by_budget": [],
            "budget_satisfied": True,
        }
    referenced = _tolbert_promoted_checkpoint_references(config)
    checkpoints = sorted(
        [path for path in root.rglob("*.pt") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    kept_unreferenced = 0
    removed_by_count: list[str] = []
    removable: list[Path] = []
    for path in checkpoints:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in referenced:
            continue
        if kept_unreferenced < max(0, int(keep_checkpoints)):
            kept_unreferenced += 1
            continue
        removable.append(path)
    for path in removable:
        try:
            path.unlink()
        except OSError:
            continue
        removed_by_count.append(str(path))

    removed_by_budget: list[str] = []
    while True:
        current_bytes = _path_usage_bytes(root, timeout_seconds=1.0)
        if budget_bytes <= 0 or current_bytes <= budget_bytes:
            break
        candidates = []
        for path in sorted(
            [item for item in root.rglob("*.pt") if item.is_file()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            if resolved in referenced:
                continue
            candidates.append(path)
        if not candidates:
            break
        victim = candidates[-1]
        try:
            victim.unlink()
        except OSError:
            break
        removed_by_budget.append(str(victim))

    after_bytes = _path_usage_bytes(root, timeout_seconds=1.0)
    return {
        "root": str(root),
        "keep_checkpoints": max(0, int(keep_checkpoints)),
        "budget_bytes": max(0, int(budget_bytes)),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "removed_by_count": removed_by_count,
        "removed_by_budget": removed_by_budget,
        "budget_satisfied": budget_bytes <= 0 or after_bytes <= budget_bytes,
    }


def _cleanup_tolbert_model_nested_storage(
    config: KernelConfig,
    *,
    keep_candidate_dirs: int,
    candidate_budget_bytes: int,
    shared_store_budget_bytes: int,
) -> dict[str, object]:
    candidates_root = config.candidate_artifacts_root / "tolbert_model"
    store_root = config.tolbert_model_artifact_path.parent / "store"
    before_candidate_bytes = _path_usage_bytes(candidates_root, timeout_seconds=1.0)
    before_shared_store_bytes = _path_usage_bytes(store_root, timeout_seconds=1.0)
    removed_by_count: list[str] = []
    removed_by_budget: list[str] = []
    removed_shared_store: list[str] = []

    candidate_dirs = _tolbert_candidate_cycle_dirs(config)
    for path in candidate_dirs[max(0, keep_candidate_dirs):]:
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed_by_count.append(str(path))
    if removed_by_count:
        removed_shared_store.extend(_cleanup_tolbert_model_shared_store(config))

    while True:
        candidate_dirs = _tolbert_candidate_cycle_dirs(config)
        candidate_bytes = _path_usage_bytes(candidates_root, timeout_seconds=1.0)
        shared_store_bytes = _path_usage_bytes(store_root, timeout_seconds=1.0)
        over_candidate_budget = candidate_budget_bytes > 0 and candidate_bytes > candidate_budget_bytes
        over_shared_store_budget = shared_store_budget_bytes > 0 and shared_store_bytes > shared_store_budget_bytes
        if not candidate_dirs or not (over_candidate_budget or over_shared_store_budget):
            break
        if len(candidate_dirs) <= 1:
            break
        victim = candidate_dirs[-1]
        shutil.rmtree(victim, ignore_errors=True)
        if victim.exists():
            break
        removed_by_budget.append(str(victim))
        removed_shared_store.extend(_cleanup_tolbert_model_shared_store(config))

    after_candidate_bytes = _path_usage_bytes(candidates_root, timeout_seconds=1.0)
    after_shared_store_bytes = _path_usage_bytes(store_root, timeout_seconds=1.0)
    return {
        "candidate_root": str(candidates_root),
        "shared_store_root": str(store_root),
        "keep_candidate_dirs": max(0, int(keep_candidate_dirs)),
        "candidate_budget_bytes": max(0, int(candidate_budget_bytes)),
        "shared_store_budget_bytes": max(0, int(shared_store_budget_bytes)),
        "before_candidate_bytes": before_candidate_bytes,
        "after_candidate_bytes": after_candidate_bytes,
        "before_shared_store_bytes": before_shared_store_bytes,
        "after_shared_store_bytes": after_shared_store_bytes,
        "removed_candidate_dirs_by_count": removed_by_count,
        "removed_candidate_dirs_by_budget": removed_by_budget,
        "removed_shared_store": removed_shared_store,
        "candidate_budget_satisfied": candidate_budget_bytes <= 0 or after_candidate_bytes <= candidate_budget_bytes,
        "shared_store_budget_satisfied": (
            shared_store_budget_bytes <= 0 or after_shared_store_bytes <= shared_store_budget_bytes
        ),
    }


def _cleanup_runtime_state(
    config: KernelConfig,
    *,
    keep_reports: int,
    keep_candidate_dirs: int,
    keep_checkpoints: int,
    keep_snapshot_entries: int,
    keep_tolbert_dataset_dirs: int,
    keep_tolbert_candidate_dirs: int = 3,
    tolbert_candidate_budget_bytes: int = 1 * 1024 * 1024 * 1024,
    tolbert_shared_store_budget_bytes: int = 4 * 1024 * 1024 * 1024,
    keep_tolbert_promoted_checkpoints: int = 3,
    tolbert_promoted_checkpoint_budget_bytes: int = 4 * 1024 * 1024 * 1024,
) -> dict[str, object]:
    removed_nested_scopes: list[str] = []
    snapshot_root = config.unattended_workspace_snapshot_root
    nested_scope_roots = (
        snapshot_root / "generated_success" / "generated_success",
        snapshot_root / "generated_success" / "generated_failure",
        snapshot_root / "generated_failure" / "generated_success",
        snapshot_root / "generated_failure" / "generated_failure",
    )
    for path in nested_scope_roots:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed_nested_scopes.append(str(path))

    removed_reports = _prune_directory_entries(config.improvement_reports_dir, keep=keep_reports)
    removed_candidate_dirs = _prune_directory_entries(config.candidate_artifacts_root, keep=keep_candidate_dirs)
    removed_checkpoints = _prune_directory_entries(config.run_checkpoints_dir, keep=keep_checkpoints)
    removed_tolbert_datasets = _prune_directory_entries(
        config.tolbert_supervised_datasets_dir,
        keep=keep_tolbert_dataset_dirs,
    )
    removed_tolbert_shared_store = _cleanup_tolbert_model_shared_store(config)
    tolbert_promoted_checkpoint_cleanup = _cleanup_tolbert_promoted_checkpoints(
        config,
        keep_checkpoints=keep_tolbert_promoted_checkpoints,
        budget_bytes=max(0, int(tolbert_promoted_checkpoint_budget_bytes)),
    )
    tolbert_nested_cleanup = _cleanup_tolbert_model_nested_storage(
        config,
        keep_candidate_dirs=keep_tolbert_candidate_dirs,
        candidate_budget_bytes=max(0, int(tolbert_candidate_budget_bytes)),
        shared_store_budget_bytes=max(0, int(tolbert_shared_store_budget_bytes)),
    )
    removed_tolbert_shared_store.extend(
        [path for path in tolbert_nested_cleanup["removed_shared_store"] if path not in removed_tolbert_shared_store]
    )

    removed_snapshots: list[str] = []
    for path in (
        snapshot_root,
        snapshot_root / "generated_success",
        snapshot_root / "generated_failure",
        snapshot_root / "generated_failure" / "generated_failure_seed",
    ):
        removed_snapshots.extend(_prune_directory_entries(path, keep=keep_snapshot_entries))

    return {
        "removed_nested_scopes": removed_nested_scopes,
        "removed_reports": removed_reports,
        "removed_candidate_dirs": removed_candidate_dirs,
        "removed_checkpoints": removed_checkpoints,
        "removed_tolbert_datasets": removed_tolbert_datasets,
        "removed_tolbert_shared_store": removed_tolbert_shared_store,
        "tolbert_promoted_checkpoint_cleanup": tolbert_promoted_checkpoint_cleanup,
        "tolbert_nested_cleanup": tolbert_nested_cleanup,
        "removed_snapshots": removed_snapshots,
    }


def _governed_cleanup_runtime_state(
    config: KernelConfig,
    *,
    keep_reports: int,
    keep_candidate_dirs: int,
    keep_checkpoints: int,
    keep_snapshot_entries: int,
    keep_tolbert_dataset_dirs: int,
    keep_tolbert_candidate_dirs: int = 3,
    tolbert_candidate_budget_bytes: int = 1 * 1024 * 1024 * 1024,
    tolbert_shared_store_budget_bytes: int = 4 * 1024 * 1024 * 1024,
    keep_tolbert_promoted_checkpoints: int = 3,
    tolbert_promoted_checkpoint_budget_bytes: int = 4 * 1024 * 1024 * 1024,
    target_free_gib: float,
) -> dict[str, object]:
    before_disk = _disk_preflight(config.workspace_root, min_free_gib=0.0)
    target_free = max(0.0, float(target_free_gib))
    if float(before_disk.get("free_gib", 0.0)) >= target_free:
        return {
            "before_disk": before_disk,
            "after_disk": before_disk,
            "before_storage": {},
            "after_storage": {},
            "cleanup": {
                "skipped": True,
                "reason": "disk already above target",
                "target_free_gib": target_free,
            },
            "emergency_cleanup": {},
        }
    before_storage = _storage_governance_snapshot(config)
    cleanup = _cleanup_runtime_state(
        config,
        keep_reports=keep_reports,
        keep_candidate_dirs=keep_candidate_dirs,
        keep_checkpoints=keep_checkpoints,
        keep_snapshot_entries=keep_snapshot_entries,
        keep_tolbert_dataset_dirs=keep_tolbert_dataset_dirs,
        keep_tolbert_candidate_dirs=keep_tolbert_candidate_dirs,
        tolbert_candidate_budget_bytes=tolbert_candidate_budget_bytes,
        tolbert_shared_store_budget_bytes=tolbert_shared_store_budget_bytes,
        keep_tolbert_promoted_checkpoints=keep_tolbert_promoted_checkpoints,
        tolbert_promoted_checkpoint_budget_bytes=tolbert_promoted_checkpoint_budget_bytes,
    )
    after_disk = _disk_preflight(config.workspace_root, min_free_gib=0.0)
    after_storage = _storage_governance_snapshot(config)
    emergency_cleanup: dict[str, object] = {}
    if float(after_disk.get("free_gib", 0.0)) < target_free:
        emergency_cleanup = _cleanup_runtime_state(
            config,
            keep_reports=max(0, min(5, keep_reports // 2)),
            keep_candidate_dirs=max(0, min(5, keep_candidate_dirs // 2)),
            keep_checkpoints=max(0, min(5, keep_checkpoints // 2)),
            keep_snapshot_entries=max(0, min(25, keep_snapshot_entries // 2)),
            keep_tolbert_dataset_dirs=max(0, min(3, keep_tolbert_dataset_dirs // 2)),
            keep_tolbert_candidate_dirs=max(0, min(2, keep_tolbert_candidate_dirs // 2)),
            tolbert_candidate_budget_bytes=max(0, tolbert_candidate_budget_bytes // 2),
            tolbert_shared_store_budget_bytes=max(0, tolbert_shared_store_budget_bytes // 2),
            keep_tolbert_promoted_checkpoints=max(0, min(2, keep_tolbert_promoted_checkpoints // 2)),
            tolbert_promoted_checkpoint_budget_bytes=max(0, tolbert_promoted_checkpoint_budget_bytes // 2),
        )
        after_disk = _disk_preflight(config.workspace_root, min_free_gib=0.0)
        after_storage = _storage_governance_snapshot(config)
    return {
        "before_disk": before_disk,
        "after_disk": after_disk,
        "before_storage": before_storage,
        "after_storage": after_storage,
        "cleanup": cleanup,
        "emergency_cleanup": emergency_cleanup,
    }


def _should_run_liftoff(mode: str, campaign_report: dict[str, object]) -> bool:
    normalized = str(mode).strip().lower()
    if normalized == "never":
        return False
    if normalized == "always":
        return True
    production = campaign_report.get("production_yield_summary", {})
    phase_gate = campaign_report.get("phase_gate_summary", {})
    retained_cycles = 0
    if isinstance(production, dict):
        retained_cycles = int(production.get("retained_cycles", 0) or 0)
    retained_phase_gates_passed = True
    if isinstance(phase_gate, dict):
        retained_phase_gates_passed = bool(phase_gate.get("all_retained_phase_gates_passed", True))
    return retained_cycles > 0 and retained_phase_gates_passed


def _initial_round_policy(args, config: KernelConfig) -> dict[str, object]:
    requested_priority_families = _normalize_benchmark_families(args.priority_benchmark_family)
    return {
        "cycles": max(1, args.cycles),
        "campaign_width": max(1, args.campaign_width),
        "variant_width": max(1, args.variant_width),
        "adaptive_search": bool(args.adaptive_search),
        "task_limit": max(0, args.task_limit),
        "task_step_floor": max(1, int(args.task_step_floor or config.frontier_task_step_floor)),
        "priority_benchmark_families": (
            list(requested_priority_families)
            if requested_priority_families
            else _select_priority_benchmark_families(
                required_families=list(config.unattended_trust_required_benchmark_families),
                missing_required_families=[],
                current_priority_families=[],
            )
        ),
        "tolbert_device": config.tolbert_device,
        "focus": args.focus,
        "liftoff": args.liftoff,
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }


def _campaign_signal(campaign_report: dict[str, object]) -> dict[str, object]:
    production = campaign_report.get("production_yield_summary", {})
    if not isinstance(production, dict):
        production = {}
    phase_gate = campaign_report.get("phase_gate_summary", {})
    if not isinstance(phase_gate, dict):
        phase_gate = {}
    recent_production = campaign_report.get("recent_production_decisions", [])
    if not isinstance(recent_production, list):
        recent_production = []
    planner_pressure = campaign_report.get("planner_pressure_summary", {})
    if not isinstance(planner_pressure, dict):
        planner_pressure = {}
    trust_breadth = campaign_report.get("trust_breadth_summary", {})
    if not isinstance(trust_breadth, dict):
        trust_breadth = {}
    priority_family_yield = campaign_report.get("priority_family_yield_summary", {})
    if not isinstance(priority_family_yield, dict):
        priority_family_yield = {}
    inheritance_summary = campaign_report.get("inheritance_summary", {})
    if not isinstance(inheritance_summary, dict):
        inheritance_summary = {}
    decision_conversion_summary = campaign_report.get("decision_conversion_summary", {})
    if not isinstance(decision_conversion_summary, dict):
        decision_conversion_summary = {}
    incomplete_cycle_summary = campaign_report.get("incomplete_cycle_summary", {})
    if not isinstance(incomplete_cycle_summary, dict):
        incomplete_cycle_summary = {}
    partial_progress_summary = campaign_report.get("partial_progress_summary", {})
    if not isinstance(partial_progress_summary, dict):
        partial_progress_summary = {}
    runs = campaign_report.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    recent_non_runtime = campaign_report.get("recent_non_runtime_decisions", [])
    if not isinstance(recent_non_runtime, list):
        recent_non_runtime = []
    runtime_managed_decisions = max(
        int(inheritance_summary.get("runtime_managed_decisions", 0) or 0),
        int(campaign_report.get("runtime_managed_decisions", 0) or 0),
        int(decision_conversion_summary.get("runtime_managed_runs", 0) or 0),
    )
    non_runtime_managed_decisions = max(
        int(inheritance_summary.get("non_runtime_managed_decisions", 0) or 0),
        int(campaign_report.get("non_runtime_managed_decisions", 0) or 0),
        int(decision_conversion_summary.get("non_runtime_managed_runs", 0) or 0),
    )
    decision_runs = int(decision_conversion_summary.get("decision_runs", 0) or 0)
    non_runtime_managed_runs = int(decision_conversion_summary.get("non_runtime_managed_runs", 0) or 0)
    partial_productive_without_decision_runs = int(
        decision_conversion_summary.get("partial_productive_without_decision_runs", 0) or 0
    )
    intermediate_decision_evidence = bool(
        runtime_managed_decisions > 0
        or non_runtime_managed_decisions > 0
        or decision_runs > 0
        or recent_production
        or recent_non_runtime
    )
    recent_failed_run: dict[str, object] = {}
    for run in reversed(runs):
        if not isinstance(run, dict):
            continue
        returncode = int(run.get("returncode", 0) or 0)
        if returncode == 0:
            continue
        recent_failed_run = {
            "run_index": int(run.get("index", 0) or 0),
            "returncode": returncode,
        }
        stderr_tail = _tail_text(str(run.get("stderr", "")).strip(), max_lines=4, max_chars=280)
        stdout_tail = _tail_text(str(run.get("stdout", "")).strip(), max_lines=4, max_chars=280)
        if stderr_tail:
            recent_failed_run["stderr_tail"] = stderr_tail
        if stdout_tail:
            recent_failed_run["stdout_tail"] = stdout_tail
        break
    worst_family_delta = 0.0
    worst_generated_family_delta = 0.0
    max_regressed_families = 0
    max_generated_regressed_families = 0
    max_low_confidence_episode_delta = 0
    min_trusted_retrieval_step_delta = 0
    recent_failed_decision: dict[str, object] = {}
    for record in recent_production:
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            continue
        if not recent_failed_decision and "phase_gate_passed" in metrics and not bool(metrics.get("phase_gate_passed", False)):
            recent_failed_decision = {
                "cycle_id": str(record.get("cycle_id", "")).strip(),
                "subsystem": str(record.get("subsystem", "")).strip(),
                "state": str(record.get("state", "")).strip(),
                "reason": str(record.get("reason", "")).strip(),
                "artifact_kind": str(record.get("artifact_kind", "")).strip(),
                "artifact_path": str(record.get("artifact_path", "")).strip(),
                "phase_gate_passed": False,
            }
            for key in (
                "baseline_pass_rate",
                "candidate_pass_rate",
                "pass_rate_delta",
                "average_step_delta",
                "worst_family_delta",
                "generated_worst_family_delta",
                "regressed_family_count",
                "generated_regressed_family_count",
            ):
                if key in metrics:
                    recent_failed_decision[key] = metrics.get(key)
        try:
            worst_family_delta = min(worst_family_delta, float(metrics.get("worst_family_delta", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            worst_generated_family_delta = min(
                worst_generated_family_delta,
                float(metrics.get("generated_worst_family_delta", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            pass
        max_regressed_families = max(max_regressed_families, int(metrics.get("regressed_family_count", 0) or 0))
        max_generated_regressed_families = max(
            max_generated_regressed_families,
            int(metrics.get("generated_regressed_family_count", 0) or 0),
        )
        max_low_confidence_episode_delta = max(
            max_low_confidence_episode_delta,
            int(metrics.get("low_confidence_episode_delta", 0) or 0),
        )
        min_trusted_retrieval_step_delta = min(
            min_trusted_retrieval_step_delta,
            int(metrics.get("trusted_retrieval_step_delta", 0) or 0),
        )
    return {
        "report_kind": str(campaign_report.get("report_kind", "")).strip(),
        "cycles_requested": int(campaign_report.get("cycles_requested", 0) or 0),
        "completed_runs": int(campaign_report.get("completed_runs", 0) or 0),
        "successful_runs": int(campaign_report.get("successful_runs", 0) or 0),
        "task_success_evidence": bool(
            int(campaign_report.get("successful_runs", 0) or 0) > 0
            and not (
                isinstance(campaign_report.get("production_yield_summary", {}), dict)
                and campaign_report.get("production_yield_summary", {})
                and int(production.get("retained_cycles", 0) or 0) <= 0
                and runtime_managed_decisions <= 0
                and non_runtime_managed_decisions <= 0
            )
        ),
        "retained_cycles": int(production.get("retained_cycles", 0) or 0),
        "rejected_cycles": int(production.get("rejected_cycles", 0) or 0),
        "average_retained_pass_rate_delta": float(production.get("average_retained_pass_rate_delta", 0.0) or 0.0),
        "average_retained_step_delta": float(production.get("average_retained_step_delta", 0.0) or 0.0),
        "productive_depth_retained_cycles": int(production.get("productive_depth_retained_cycles", 0) or 0),
        "average_productive_depth_step_delta": float(
            production.get("average_productive_depth_step_delta", 0.0) or 0.0
        ),
        "depth_drift_cycles": int(production.get("depth_drift_cycles", 0) or 0),
        "average_depth_drift_step_delta": float(production.get("average_depth_drift_step_delta", 0.0) or 0.0),
        "long_horizon_retained_cycles": int(production.get("long_horizon_retained_cycles", 0) or 0),
        "all_retained_phase_gates_passed": bool(phase_gate.get("all_retained_phase_gates_passed", True)),
        "failed_decisions": int(phase_gate.get("failed_decisions", 0) or 0),
        "runtime_managed_decisions": runtime_managed_decisions,
        "non_runtime_managed_decisions": non_runtime_managed_decisions,
        "decision_runs": decision_runs,
        "non_runtime_managed_runs": non_runtime_managed_runs,
        "partial_productive_without_decision_runs": partial_productive_without_decision_runs,
        "intermediate_decision_evidence": intermediate_decision_evidence,
        "incomplete_cycle_count": int(incomplete_cycle_summary.get("count", 0) or 0),
        "observed_primary_runs": int(partial_progress_summary.get("observed_primary_runs", 0) or 0),
        "generated_success_completed_runs": int(
            partial_progress_summary.get("generated_success_completed_runs", 0) or 0
        ),
        "campaign_breadth_pressure_cycles": int(planner_pressure.get("campaign_breadth_pressure_cycles", 0) or 0),
        "variant_breadth_pressure_cycles": int(planner_pressure.get("variant_breadth_pressure_cycles", 0) or 0),
        "external_report_count": int(
            -1 if trust_breadth.get("external_report_count", None) is None else trust_breadth.get("external_report_count", -1)
        ),
        "distinct_external_benchmark_families": int(
            -1
            if trust_breadth.get("distinct_external_benchmark_families", None) is None
            else trust_breadth.get("distinct_external_benchmark_families", -1)
        ),
        "required_families": _normalize_benchmark_families(trust_breadth.get("required_families", [])),
        "required_families_with_reports": _normalize_benchmark_families(
            trust_breadth.get("required_families_with_reports", [])
        ),
        "missing_required_families": _normalize_benchmark_families(
            trust_breadth.get("missing_required_families", [])
        ),
        "distinct_family_gap": int(trust_breadth.get("distinct_family_gap", 0) or 0),
        "priority_families": _normalize_benchmark_families(
            priority_family_yield.get(
                "priority_families",
                campaign_report.get("priority_benchmark_families", []),
            )
        ),
        "priority_families_with_signal": _normalize_benchmark_families(
            priority_family_yield.get("priority_families_with_signal", [])
        ),
        "priority_families_with_retained_gain": _normalize_benchmark_families(
            priority_family_yield.get("priority_families_with_retained_gain", [])
        ),
        "priority_families_without_signal": _normalize_benchmark_families(
            priority_family_yield.get("priority_families_without_signal", [])
        ),
        "priority_families_with_signal_but_no_retained_gain": _normalize_benchmark_families(
            priority_family_yield.get("priority_families_with_signal_but_no_retained_gain", [])
        ),
        "priority_families_needing_retained_gain_conversion": _normalize_benchmark_families(
            priority_family_yield.get(
                "priority_families_needing_retained_gain_conversion",
                priority_family_yield.get("priority_families_with_signal_but_no_retained_gain", []),
            )
        ),
        "recent_failed_run": recent_failed_run,
        "recent_failed_decision": recent_failed_decision,
        "worst_family_delta": worst_family_delta,
        "worst_generated_family_delta": worst_generated_family_delta,
        "max_regressed_families": max_regressed_families,
        "max_generated_regressed_families": max_generated_regressed_families,
        "max_low_confidence_episode_delta": max_low_confidence_episode_delta,
        "min_trusted_retrieval_step_delta": min_trusted_retrieval_step_delta,
    }


def _tail_text(text: str, *, max_lines: int, max_chars: int) -> str:
    normalized = str(text).strip()
    if not normalized:
        return ""
    lines = [line for line in normalized.splitlines() if line.strip()]
    if max_lines > 0:
        lines = lines[-max_lines:]
    tail = "\n".join(lines).strip()
    if max_chars > 0 and len(tail) > max_chars:
        tail = tail[-max_chars:].lstrip()
    return tail


def _campaign_failure_context(signal: dict[str, object]) -> dict[str, object]:
    if not isinstance(signal, dict):
        return {}
    context: dict[str, object] = {}
    failed_run = signal.get("recent_failed_run", {})
    failed_decision = signal.get("recent_failed_decision", {})
    if isinstance(failed_run, dict) and failed_run:
        context["failed_run"] = failed_run
    if isinstance(failed_decision, dict) and failed_decision:
        context["failed_decision"] = failed_decision
    return context


def _campaign_failure_context_summary(context: dict[str, object]) -> str:
    if not isinstance(context, dict):
        return ""
    failed_run = context.get("failed_run", {})
    if isinstance(failed_run, dict) and failed_run:
        parts = [
            f"run={int(failed_run.get('run_index', 0) or 0)}",
            f"returncode={int(failed_run.get('returncode', 0) or 0)}",
        ]
        stderr_tail = str(failed_run.get("stderr_tail", "")).strip()
        if stderr_tail:
            parts.append(f"stderr_tail={stderr_tail}")
        return " ".join(parts)
    failed_decision = context.get("failed_decision", {})
    if isinstance(failed_decision, dict) and failed_decision:
        parts = []
        cycle_id = str(failed_decision.get("cycle_id", "")).strip()
        subsystem = str(failed_decision.get("subsystem", "")).strip()
        reason = str(failed_decision.get("reason", "")).strip()
        if cycle_id:
            parts.append(f"cycle_id={cycle_id}")
        if subsystem:
            parts.append(f"subsystem={subsystem}")
        if reason:
            parts.append(f"reason={reason}")
        return " ".join(parts)
    return ""


def _subsystem_signal(campaign_report: dict[str, object]) -> dict[str, object]:
    retained_by_subsystem: dict[str, int] = {}
    rejected_by_subsystem: dict[str, int] = {}
    recent_production = campaign_report.get("recent_production_decisions", [])
    if not isinstance(recent_production, list):
        recent_production = []
    recent_non_runtime = campaign_report.get("recent_non_runtime_decisions", [])
    if not isinstance(recent_non_runtime, list):
        recent_non_runtime = []
    for record in recent_non_runtime:
        if not isinstance(record, dict):
            continue
        subsystem = str(record.get("subsystem", "")).strip()
        if not subsystem:
            continue
        state = str(record.get("state", "")).strip()
        if state == "retain":
            retained_by_subsystem[subsystem] = retained_by_subsystem.get(subsystem, 0) + 1
        elif state == "reject":
            rejected_by_subsystem[subsystem] = rejected_by_subsystem.get(subsystem, 0) + 1
    if not recent_production:
        fallback_runs = campaign_report.get("runs", [])
        if isinstance(fallback_runs, list):
            for run in fallback_runs:
                if not isinstance(run, dict):
                    continue
                subsystem = str(run.get("subsystem", "")).strip()
                if not subsystem:
                    continue
                runtime_managed_decisions = int(run.get("runtime_managed_decisions", 0) or 0)
                if runtime_managed_decisions > 0:
                    continue
                structured_child_decision = run.get("structured_child_decision", {})
                if isinstance(structured_child_decision, Mapping):
                    state = str(structured_child_decision.get("state", "")).strip()
                    if state == "retain":
                        retained_by_subsystem[subsystem] = retained_by_subsystem.get(subsystem, 0) + 1
                        continue
                    if state == "reject":
                        rejected_by_subsystem[subsystem] = rejected_by_subsystem.get(subsystem, 0) + 1
                        continue
                if bool(run.get("retained_gain", False)):
                    retained_by_subsystem[subsystem] = retained_by_subsystem.get(subsystem, 0) + 1
                    continue
                decision_conversion_state = str(run.get("decision_conversion_state", "")).strip()
                if decision_conversion_state not in {"partial_productive_without_decision", "no_decision"}:
                    continue
                stdout = str(run.get("stdout", ""))
                finalized_reject_marker = f"finalized subsystem={subsystem} state=reject"
                apply_decision_reject_marker = f"finalize phase=apply_decision subsystem={subsystem} state=reject"
                if finalized_reject_marker in stdout or apply_decision_reject_marker in stdout:
                    rejected_by_subsystem[subsystem] = rejected_by_subsystem.get(subsystem, 0) + 1
    average_pass_delta_by_subsystem: dict[str, list[float]] = {}
    for record in recent_production:
        if not isinstance(record, dict):
            continue
        subsystem = str(record.get("subsystem", "")).strip()
        if not subsystem:
            continue
        state = str(record.get("state", "")).strip()
        if state == "retain":
            retained_by_subsystem[subsystem] = retained_by_subsystem.get(subsystem, 0) + 1
        elif state == "reject":
            rejected_by_subsystem[subsystem] = rejected_by_subsystem.get(subsystem, 0) + 1
        metrics_summary = record.get("metrics_summary", {})
        if isinstance(metrics_summary, dict):
            try:
                delta = float(metrics_summary.get("candidate_pass_rate", 0.0) or 0.0) - float(
                    metrics_summary.get("baseline_pass_rate", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
            average_pass_delta_by_subsystem.setdefault(subsystem, []).append(delta)
    averaged = {
        subsystem: (sum(values) / len(values))
        for subsystem, values in average_pass_delta_by_subsystem.items()
        if values
    }
    cooldown_rounds_by_subsystem: dict[str, int] = {}
    zero_yield_dominant_subsystems: list[str] = []
    for subsystem, rejected in rejected_by_subsystem.items():
        retained = retained_by_subsystem.get(subsystem, 0)
        total_decisions = rejected + retained
        average_pass_delta = averaged.get(subsystem, 0.0)
        if rejected > retained and average_pass_delta <= 0.0:
            cooldown_rounds = 2
            if retained <= 0 and rejected >= 3 and total_decisions >= 3:
                cooldown_rounds = 3
                zero_yield_dominant_subsystems.append(subsystem)
            cooldown_rounds_by_subsystem[subsystem] = cooldown_rounds
    cooldown_candidates = [
        subsystem
        for subsystem in sorted(
            cooldown_rounds_by_subsystem,
            key=lambda token: (
                -int(cooldown_rounds_by_subsystem.get(token, 0)),
                -int(rejected_by_subsystem.get(token, 0)),
                token,
            ),
        )
    ]
    return {
        "retained_by_subsystem": retained_by_subsystem,
        "rejected_by_subsystem": rejected_by_subsystem,
        "average_pass_delta_by_subsystem": averaged,
        "cooldown_rounds_by_subsystem": cooldown_rounds_by_subsystem,
        "cooldown_candidates": cooldown_candidates,
        "zero_yield_dominant_subsystems": sorted(zero_yield_dominant_subsystems),
    }


def _subsystem_monoculture_signal(
    campaign_report: dict[str, object],
    *,
    subsystem_signal: dict[str, object] | None = None,
    planner_pressure_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    subsystem = subsystem_signal if isinstance(subsystem_signal, dict) else _subsystem_signal(campaign_report)
    pressure = (
        planner_pressure_signal
        if isinstance(planner_pressure_signal, dict)
        else _planner_pressure_signal(campaign_report)
    )
    dominant_subsystem = str(pressure.get("dominant_subsystem", "")).strip()
    retained_by_subsystem = (
        dict(subsystem.get("retained_by_subsystem", {}))
        if isinstance(subsystem.get("retained_by_subsystem", {}), dict)
        else {}
    )
    rejected_by_subsystem = (
        dict(subsystem.get("rejected_by_subsystem", {}))
        if isinstance(subsystem.get("rejected_by_subsystem", {}), dict)
        else {}
    )
    total_retained = sum(max(0, int(value or 0)) for value in retained_by_subsystem.values())
    total_rejected = sum(max(0, int(value or 0)) for value in rejected_by_subsystem.values())
    total_decisions = total_retained + total_rejected
    dominant_retained = max(0, int(retained_by_subsystem.get(dominant_subsystem, 0) or 0))
    dominant_rejected = max(0, int(rejected_by_subsystem.get(dominant_subsystem, 0) or 0))
    dominant_total = dominant_retained + dominant_rejected
    campaign_breadth_pressure_cycles = max(0, int(pressure.get("campaign_breadth_pressure_cycles", 0) or 0))
    variant_breadth_pressure_cycles = max(0, int(pressure.get("variant_breadth_pressure_cycles", 0) or 0))
    retained_cycles = max(
        0,
        int(
            (
                campaign_report.get("production_yield_summary", {})
                if isinstance(campaign_report.get("production_yield_summary", {}), dict)
                else {}
            ).get("retained_cycles", 0)
            or 0
        ),
    )
    zero_yield_dominant_subsystems = set(
        _normalize_excluded_subsystems(subsystem.get("zero_yield_dominant_subsystems", []))
    )
    dominant_share = (float(dominant_total) / float(total_decisions)) if total_decisions > 0 else 0.0
    breadth_pressure = campaign_breadth_pressure_cycles > 0 or variant_breadth_pressure_cycles > 0
    no_yield = retained_cycles <= 0
    active = bool(
        dominant_subsystem
        and dominant_total >= 2
        and dominant_share >= 0.6
        and dominant_rejected > dominant_retained
        and (breadth_pressure or no_yield or dominant_subsystem in zero_yield_dominant_subsystems)
    )
    severe = bool(
        active
        and no_yield
        and dominant_retained <= 0
        and dominant_rejected >= 3
        and dominant_share >= 0.75
    )
    return {
        "active": active,
        "severe": severe,
        "dominant_subsystem": dominant_subsystem,
        "dominant_share": round(dominant_share, 4),
        "dominant_rejected": dominant_rejected,
        "dominant_retained": dominant_retained,
        "dominant_decisions": dominant_total,
        "total_decisions": total_decisions,
        "breadth_pressure": breadth_pressure,
        "no_yield": no_yield,
    }


def _planner_pressure_signal(
    campaign_report: dict[str, object],
    *,
    curriculum_controls: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = campaign_report.get("planner_pressure_summary", {})
    if not isinstance(payload, dict):
        payload = {}
    recent_campaign = payload.get("recent_campaign_pressures", [])
    if not isinstance(recent_campaign, list):
        recent_campaign = []
    recent_variant = payload.get("recent_variant_pressures", [])
    if not isinstance(recent_variant, list):
        recent_variant = []
    subsystem_counts: dict[str, int] = {}
    for entry in [*recent_campaign, *recent_variant]:
        if not isinstance(entry, dict):
            continue
        subsystem = str(entry.get("subsystem", "")).strip()
        if subsystem:
            subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1
    pressured_subsystems = sorted(
        subsystem_counts,
        key=lambda subsystem: (-subsystem_counts[subsystem], subsystem),
    )
    frontier_failure_motif_pairs = _curriculum_control_family_signal_pairs(
        curriculum_controls,
        "frontier_failure_motif_priority_pairs",
    )
    frontier_repo_setting_pairs = _curriculum_control_family_signal_pairs(
        curriculum_controls,
        "frontier_repo_setting_priority_pairs",
    )
    return {
        "campaign_breadth_pressure_cycles": int(payload.get("campaign_breadth_pressure_cycles", 0) or 0),
        "variant_breadth_pressure_cycles": int(payload.get("variant_breadth_pressure_cycles", 0) or 0),
        "pressured_subsystems": pressured_subsystems,
        "dominant_subsystem": pressured_subsystems[0] if pressured_subsystems else "",
        "frontier_failure_motif_priority_pairs": [f"{family}:{signal}" for family, signal in frontier_failure_motif_pairs],
        "frontier_repo_setting_priority_pairs": [f"{family}:{signal}" for family, signal in frontier_repo_setting_pairs],
        "frontier_failure_motif_families": _rank_priority_benchmark_families(
            [family for family, _ in frontier_failure_motif_pairs]
        ),
        "frontier_repo_setting_families": _rank_priority_benchmark_families(
            [family for family, _ in frontier_repo_setting_pairs]
        ),
    }


def _productive_partial_conversion_signal(
    round_payload: Mapping[str, object] | None,
    *,
    priority_families: object,
) -> dict[str, object]:
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    active_child = payload.get("active_child", {})
    if not isinstance(active_child, Mapping):
        active_child = {}
    active_cycle_progress = active_child.get("active_cycle_progress", {})
    if not isinstance(active_cycle_progress, Mapping):
        active_cycle_progress = {}
    sampled_families = _normalize_benchmark_families(
        active_child.get("families_sampled", active_child.get("sampled_families_from_progress", []))
    )
    if not sampled_families:
        sampled_families = _normalize_benchmark_families(active_cycle_progress.get("sampled_families_from_progress", []))
    productive_partial = bool(active_cycle_progress.get("productive_partial", False)) or bool(
        active_child.get("partial_productive_runs", 0) or 0
    )
    generated_success_completed = bool(active_cycle_progress.get("generated_success_completed", False))
    priority_family_set = set(_normalize_benchmark_families(priority_families))
    sampled_priority_families = [
        family for family in sampled_families if not priority_family_set or family in priority_family_set
    ]
    return {
        "productive_partial": productive_partial,
        "generated_success_completed": generated_success_completed,
        "sampled_families": sampled_families,
        "sampled_priority_families": sampled_priority_families,
        "sampled_priority_family_count": len(sampled_priority_families),
        "conversion_gap": productive_partial and generated_success_completed and len(sampled_priority_families) >= 2,
    }


def _repo_setting_policy_pressure(
    *,
    frontier_repo_setting_priority_pairs: object,
    priority_families: object,
) -> dict[str, object]:
    pairs = _normalize_benchmark_families(frontier_repo_setting_priority_pairs)
    prioritized_families = set(_normalize_benchmark_families(priority_families))
    focused_pairs: list[str] = []
    signal_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for value in pairs:
        family, _, signal = str(value).strip().partition(":")
        normalized_family = family.strip()
        normalized_signal = signal.strip().lower()
        if not normalized_family or not normalized_signal:
            continue
        if prioritized_families and normalized_family not in prioritized_families:
            continue
        focused_pairs.append(f"{normalized_family}:{normalized_signal}")
        signal_counts[normalized_signal] = signal_counts.get(normalized_signal, 0) + 1
        family_counts[normalized_family] = family_counts.get(normalized_family, 0) + 1
    campaign_width_signals = [
        signal
        for signal in ("worker_handoff", "integrator_handoff", "shared_repo", "repo_sandbox")
        if signal_counts.get(signal, 0) > 0
    ]
    task_step_floor_signals = [
        signal
        for signal in ("validation_lane", "cleanup_lane", "audit_lane", "long_horizon", "repo_sandbox")
        if signal_counts.get(signal, 0) > 0
    ]
    adaptive_search_signals = [
        signal
        for signal in (
            "worker_handoff",
            "integrator_handoff",
            "shared_repo",
            "repo_sandbox",
            "validation_lane",
            "cleanup_lane",
            "audit_lane",
            "long_horizon",
        )
        if signal_counts.get(signal, 0) > 0
    ]
    return {
        "focused_pairs": focused_pairs,
        "focused_families": _rank_priority_benchmark_families(list(family_counts)),
        "signal_counts": dict(sorted(signal_counts.items())),
        "campaign_width_signals": campaign_width_signals,
        "task_step_floor_signals": task_step_floor_signals,
        "adaptive_search_signals": adaptive_search_signals,
    }


def _clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def _weighted_linear_slope(samples: list[tuple[float, float, float]]) -> float:
    total_weight = sum(max(0.0, float(weight)) for _, _, weight in samples)
    if total_weight <= 0.0:
        return 0.0
    weighted_x = sum(float(x) * max(0.0, float(weight)) for x, _, weight in samples)
    weighted_y = sum(float(y) * max(0.0, float(weight)) for _, y, weight in samples)
    weighted_xx = sum(float(x) * float(x) * max(0.0, float(weight)) for x, _, weight in samples)
    weighted_xy = sum(float(x) * float(y) * max(0.0, float(weight)) for x, y, weight in samples)
    denominator = weighted_xx - ((weighted_x * weighted_x) / total_weight)
    if abs(denominator) <= 1e-9:
        return 0.0
    numerator = weighted_xy - ((weighted_x * weighted_y) / total_weight)
    return numerator / denominator


def _repo_setting_retained_outcome_score(campaign_report: Mapping[str, object] | None) -> float:
    signal = _campaign_signal(dict(campaign_report) if isinstance(campaign_report, Mapping) else {})
    retained_cycles = max(0, int(signal.get("retained_cycles", 0) or 0))
    rejected_cycles = max(0, int(signal.get("rejected_cycles", 0) or 0))
    productive_depth_retained_cycles = max(0, int(signal.get("productive_depth_retained_cycles", 0) or 0))
    long_horizon_retained_cycles = max(0, int(signal.get("long_horizon_retained_cycles", 0) or 0))
    failed_decisions = max(0, int(signal.get("failed_decisions", 0) or 0))
    retained_phase_gates_passed = bool(signal.get("all_retained_phase_gates_passed", True))
    average_retained_pass_rate_delta = float(signal.get("average_retained_pass_rate_delta", 0.0) or 0.0)
    average_retained_step_delta = float(signal.get("average_retained_step_delta", 0.0) or 0.0)
    score = 0.0
    score += min(2.0, 0.75 * float(retained_cycles))
    score -= min(1.5, 0.5 * float(rejected_cycles))
    score += _clamp_float(average_retained_pass_rate_delta * 12.0, min_value=-1.5, max_value=1.5)
    score += _clamp_float(average_retained_step_delta / 16.0, min_value=-0.75, max_value=1.0)
    score += min(1.0, 0.35 * float(productive_depth_retained_cycles))
    score += min(0.75, 0.25 * float(long_horizon_retained_cycles))
    if failed_decisions > 0 or not retained_phase_gates_passed:
        score -= 1.5
    return score


def _empty_repo_setting_regression_stats() -> dict[str, object]:
    return {
        "observations": 0,
        "sum_w": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_xy": 0.0,
    }


def _empty_repo_setting_adaptive_stats() -> dict[str, object]:
    return {
        "observations": 0,
        "true_count": 0,
        "false_count": 0,
        "true_score_sum": 0.0,
        "false_score_sum": 0.0,
    }


def _empty_repo_setting_prior_entry() -> dict[str, object]:
    return {
        "observations": 0,
        "last_outcome_score": 0.0,
        "campaign_width_stats": _empty_repo_setting_regression_stats(),
        "task_step_floor_stats": _empty_repo_setting_regression_stats(),
        "adaptive_search_stats": _empty_repo_setting_adaptive_stats(),
        "family_priors": {},
    }


def _decay_repo_setting_regression_stats(
    stats: Mapping[str, object] | None,
    *,
    decay: float,
) -> dict[str, object]:
    payload = stats if isinstance(stats, Mapping) else {}
    resolved_decay = _clamp_float(float(decay), min_value=0.0, max_value=1.0)
    return {
        "observations": max(0, int(round(max(0, int(payload.get("observations", 0) or 0)) * resolved_decay))),
        "sum_w": max(0.0, float(payload.get("sum_w", 0.0) or 0.0) * resolved_decay),
        "sum_x": float(payload.get("sum_x", 0.0) or 0.0) * resolved_decay,
        "sum_y": float(payload.get("sum_y", 0.0) or 0.0) * resolved_decay,
        "sum_x2": max(0.0, float(payload.get("sum_x2", 0.0) or 0.0) * resolved_decay),
        "sum_xy": float(payload.get("sum_xy", 0.0) or 0.0) * resolved_decay,
    }


def _decay_repo_setting_adaptive_stats(
    stats: Mapping[str, object] | None,
    *,
    decay: float,
) -> dict[str, object]:
    payload = stats if isinstance(stats, Mapping) else {}
    resolved_decay = _clamp_float(float(decay), min_value=0.0, max_value=1.0)
    return {
        "observations": max(0, int(round(max(0, int(payload.get("observations", 0) or 0)) * resolved_decay))),
        "true_count": max(0, int(round(max(0, int(payload.get("true_count", 0) or 0)) * resolved_decay))),
        "false_count": max(0, int(round(max(0, int(payload.get("false_count", 0) or 0)) * resolved_decay))),
        "true_score_sum": float(payload.get("true_score_sum", 0.0) or 0.0) * resolved_decay,
        "false_score_sum": float(payload.get("false_score_sum", 0.0) or 0.0) * resolved_decay,
    }


def _decay_repo_setting_prior_entry(
    entry: Mapping[str, object] | None,
    *,
    decay: float,
) -> dict[str, object]:
    payload = entry if isinstance(entry, Mapping) else {}
    resolved_decay = _clamp_float(float(decay), min_value=0.0, max_value=1.0)
    decayed_families: dict[str, dict[str, object]] = {}
    family_priors = payload.get("family_priors", {})
    if isinstance(family_priors, Mapping):
        for family, family_entry in family_priors.items():
            token = str(family).strip().lower()
            if not token or not isinstance(family_entry, Mapping):
                continue
            decayed_families[token] = {
                "observations": max(
                    0,
                    int(round(max(0, int(family_entry.get("observations", 0) or 0)) * resolved_decay)),
                ),
                "last_outcome_score": float(family_entry.get("last_outcome_score", 0.0) or 0.0) * resolved_decay,
                "campaign_width_stats": _decay_repo_setting_regression_stats(
                    family_entry.get("campaign_width_stats", {}),
                    decay=resolved_decay,
                ),
                "task_step_floor_stats": _decay_repo_setting_regression_stats(
                    family_entry.get("task_step_floor_stats", {}),
                    decay=resolved_decay,
                ),
                "adaptive_search_stats": _decay_repo_setting_adaptive_stats(
                    family_entry.get("adaptive_search_stats", {}),
                    decay=resolved_decay,
                ),
            }
    return {
        "observations": max(0, int(round(max(0, int(payload.get("observations", 0) or 0)) * resolved_decay))),
        "last_outcome_score": float(payload.get("last_outcome_score", 0.0) or 0.0) * resolved_decay,
        "campaign_width_stats": _decay_repo_setting_regression_stats(
            payload.get("campaign_width_stats", {}),
            decay=resolved_decay,
        ),
        "task_step_floor_stats": _decay_repo_setting_regression_stats(
            payload.get("task_step_floor_stats", {}),
            decay=resolved_decay,
        ),
        "adaptive_search_stats": _decay_repo_setting_adaptive_stats(
            payload.get("adaptive_search_stats", {}),
            decay=resolved_decay,
        ),
        "family_priors": decayed_families,
    }


def _normalize_repo_setting_policy_priors(payload: object) -> dict[str, dict[str, object]]:
    if not isinstance(payload, Mapping):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for signal, raw_entry in payload.items():
        token = str(signal).strip().lower()
        if not token or not isinstance(raw_entry, Mapping):
            continue
        normalized[token] = {
            **_empty_repo_setting_prior_entry(),
            **dict(raw_entry),
            "campaign_width_stats": {
                **_empty_repo_setting_regression_stats(),
                **(
                    dict(raw_entry.get("campaign_width_stats", {}))
                    if isinstance(raw_entry.get("campaign_width_stats", {}), Mapping)
                    else {}
                ),
            },
            "task_step_floor_stats": {
                **_empty_repo_setting_regression_stats(),
                **(
                    dict(raw_entry.get("task_step_floor_stats", {}))
                    if isinstance(raw_entry.get("task_step_floor_stats", {}), Mapping)
                    else {}
                ),
            },
            "adaptive_search_stats": {
                **_empty_repo_setting_adaptive_stats(),
                **(
                    dict(raw_entry.get("adaptive_search_stats", {}))
                    if isinstance(raw_entry.get("adaptive_search_stats", {}), Mapping)
                    else {}
                ),
            },
            "family_priors": {
                str(family).strip().lower(): {
                    **_empty_repo_setting_prior_entry(),
                    **dict(family_entry),
                    "campaign_width_stats": {
                        **_empty_repo_setting_regression_stats(),
                        **(
                            dict(family_entry.get("campaign_width_stats", {}))
                            if isinstance(family_entry.get("campaign_width_stats", {}), Mapping)
                            else {}
                        ),
                    },
                    "task_step_floor_stats": {
                        **_empty_repo_setting_regression_stats(),
                        **(
                            dict(family_entry.get("task_step_floor_stats", {}))
                            if isinstance(family_entry.get("task_step_floor_stats", {}), Mapping)
                            else {}
                        ),
                    },
                    "adaptive_search_stats": {
                        **_empty_repo_setting_adaptive_stats(),
                        **(
                            dict(family_entry.get("adaptive_search_stats", {}))
                            if isinstance(family_entry.get("adaptive_search_stats", {}), Mapping)
                            else {}
                        ),
                    },
                    "family_priors": {},
                }
                for family, family_entry in dict(raw_entry.get("family_priors", {})).items()
                if str(family).strip() and isinstance(family_entry, Mapping)
            }
            if isinstance(raw_entry.get("family_priors", {}), Mapping)
            else {},
        }
    return normalized


def _repo_setting_regression_stats_from_samples(samples: list[tuple[float, float, float]]) -> dict[str, object]:
    stats = _empty_repo_setting_regression_stats()
    for x_value, y_value, weight in samples:
        resolved_weight = max(0.0, float(weight))
        if resolved_weight <= 0.0:
            continue
        x_float = float(x_value)
        y_float = float(y_value)
        stats["observations"] = int(stats["observations"]) + 1
        stats["sum_w"] = float(stats["sum_w"]) + resolved_weight
        stats["sum_x"] = float(stats["sum_x"]) + (resolved_weight * x_float)
        stats["sum_y"] = float(stats["sum_y"]) + (resolved_weight * y_float)
        stats["sum_x2"] = float(stats["sum_x2"]) + (resolved_weight * x_float * x_float)
        stats["sum_xy"] = float(stats["sum_xy"]) + (resolved_weight * x_float * y_float)
    return stats


def _merge_repo_setting_regression_stats(*stats_payloads: object) -> dict[str, object]:
    merged = _empty_repo_setting_regression_stats()
    for raw_stats in stats_payloads:
        if not isinstance(raw_stats, Mapping):
            continue
        merged["observations"] = int(merged["observations"]) + max(0, int(raw_stats.get("observations", 0) or 0))
        merged["sum_w"] = float(merged["sum_w"]) + max(0.0, float(raw_stats.get("sum_w", 0.0) or 0.0))
        merged["sum_x"] = float(merged["sum_x"]) + float(raw_stats.get("sum_x", 0.0) or 0.0)
        merged["sum_y"] = float(merged["sum_y"]) + float(raw_stats.get("sum_y", 0.0) or 0.0)
        merged["sum_x2"] = float(merged["sum_x2"]) + max(0.0, float(raw_stats.get("sum_x2", 0.0) or 0.0))
        merged["sum_xy"] = float(merged["sum_xy"]) + float(raw_stats.get("sum_xy", 0.0) or 0.0)
    return merged


def _scale_repo_setting_regression_stats(
    stats: Mapping[str, object] | None,
    *,
    scale: float,
) -> dict[str, object]:
    payload = stats if isinstance(stats, Mapping) else {}
    factor = _clamp_float(float(scale), min_value=0.0, max_value=1.0)
    observations = max(0, int(payload.get("observations", 0) or 0))
    scaled_observations = int(round(float(observations) * factor))
    if observations > 0 and factor > 0.0:
        scaled_observations = max(1, scaled_observations)
    return {
        "observations": scaled_observations,
        "sum_w": max(0.0, float(payload.get("sum_w", 0.0) or 0.0) * factor),
        "sum_x": float(payload.get("sum_x", 0.0) or 0.0) * factor,
        "sum_y": float(payload.get("sum_y", 0.0) or 0.0) * factor,
        "sum_x2": max(0.0, float(payload.get("sum_x2", 0.0) or 0.0) * factor),
        "sum_xy": float(payload.get("sum_xy", 0.0) or 0.0) * factor,
    }


def _weighted_linear_slope_from_stats(stats: Mapping[str, object] | None) -> float:
    payload = stats if isinstance(stats, Mapping) else {}
    denominator = max(0.0, float(payload.get("sum_w", 0.0) or 0.0) * float(payload.get("sum_x2", 0.0) or 0.0) - float(payload.get("sum_x", 0.0) or 0.0) ** 2)
    if denominator <= 1e-9:
        return 0.0
    numerator = float(payload.get("sum_w", 0.0) or 0.0) * float(payload.get("sum_xy", 0.0) or 0.0) - float(payload.get("sum_x", 0.0) or 0.0) * float(payload.get("sum_y", 0.0) or 0.0)
    return numerator / denominator


def _merge_repo_setting_adaptive_stats(*stats_payloads: object) -> dict[str, object]:
    merged = _empty_repo_setting_adaptive_stats()
    for raw_stats in stats_payloads:
        if not isinstance(raw_stats, Mapping):
            continue
        merged["observations"] = int(merged["observations"]) + max(0, int(raw_stats.get("observations", 0) or 0))
        merged["true_count"] = int(merged["true_count"]) + max(0, int(raw_stats.get("true_count", 0) or 0))
        merged["false_count"] = int(merged["false_count"]) + max(0, int(raw_stats.get("false_count", 0) or 0))
        merged["true_score_sum"] = float(merged["true_score_sum"]) + float(raw_stats.get("true_score_sum", 0.0) or 0.0)
        merged["false_score_sum"] = float(merged["false_score_sum"]) + float(raw_stats.get("false_score_sum", 0.0) or 0.0)
    return merged


def _scale_repo_setting_adaptive_stats(
    stats: Mapping[str, object] | None,
    *,
    scale: float,
) -> dict[str, object]:
    payload = stats if isinstance(stats, Mapping) else {}
    factor = _clamp_float(float(scale), min_value=0.0, max_value=1.0)
    observations = max(0, int(payload.get("observations", 0) or 0))
    true_count = max(0, int(payload.get("true_count", 0) or 0))
    false_count = max(0, int(payload.get("false_count", 0) or 0))
    scaled_observations = int(round(float(observations) * factor))
    scaled_true_count = int(round(float(true_count) * factor))
    scaled_false_count = int(round(float(false_count) * factor))
    if observations > 0 and factor > 0.0:
        scaled_observations = max(1, scaled_observations)
    if true_count > 0 and factor > 0.0:
        scaled_true_count = max(1, scaled_true_count)
    if false_count > 0 and factor > 0.0:
        scaled_false_count = max(1, scaled_false_count)
    return {
        "observations": scaled_observations,
        "true_count": scaled_true_count,
        "false_count": scaled_false_count,
        "true_score_sum": float(payload.get("true_score_sum", 0.0) or 0.0) * factor,
        "false_score_sum": float(payload.get("false_score_sum", 0.0) or 0.0) * factor,
    }


def _repo_setting_signal_set_union(*signal_groups: object) -> list[str]:
    union: set[str] = set()
    for group in signal_groups:
        if isinstance(group, set):
            union.update(_normalize_benchmark_families(list(group)))
        else:
            union.update(_normalize_benchmark_families(group))
    return sorted(union)


def _repo_setting_pair_map(focused_pairs: object) -> dict[str, set[str]]:
    pair_map: dict[str, set[str]] = {}
    for value in _normalize_benchmark_families(focused_pairs):
        family, _, signal = str(value).strip().partition(":")
        normalized_family = family.strip().lower()
        normalized_signal = signal.strip().lower()
        if not normalized_family or not normalized_signal:
            continue
        pair_map.setdefault(normalized_signal, set()).add(normalized_family)
    return pair_map


def _repo_setting_family_neighbor_map(
    active_pair_map: Mapping[str, set[str]],
    available_pair_map: Mapping[str, set[str]],
) -> dict[str, dict[str, float]]:
    neighbor_map: dict[str, dict[str, float]] = {}
    for signal, active_families in active_pair_map.items():
        available_families = {str(value).strip().lower() for value in available_pair_map.get(signal, set()) if str(value).strip()}
        if not available_families:
            continue
        signal_neighbors: dict[str, float] = {}
        for family in sorted(active_families):
            for candidate, weight in _REPO_SETTING_FAMILY_NEIGHBOR_WEIGHTS.get(str(family).strip().lower(), {}).items():
                normalized_candidate = str(candidate).strip().lower()
                if not normalized_candidate or normalized_candidate not in available_families:
                    continue
                if normalized_candidate in active_families:
                    continue
                signal_neighbors[normalized_candidate] = max(
                    float(signal_neighbors.get(normalized_candidate, 0.0) or 0.0),
                    _clamp_float(float(weight), min_value=0.0, max_value=1.0),
                )
        if signal_neighbors:
            neighbor_map[signal] = signal_neighbors
    return neighbor_map


def _weighted_family_repo_setting_prior_stats(
    *,
    state_priors: Mapping[str, object],
    family_weight_map: Mapping[str, Mapping[str, float]],
    stat_key: str,
) -> dict[str, object]:
    family_stats: list[object] = []
    for signal, family_weights in family_weight_map.items():
        signal_entry = state_priors.get(signal, {})
        if not isinstance(signal_entry, Mapping):
            continue
        family_priors = signal_entry.get("family_priors", {})
        if not isinstance(family_priors, Mapping):
            continue
        for family, weight in sorted(family_weights.items()):
            family_entry = family_priors.get(family, {})
            if not isinstance(family_entry, Mapping):
                continue
            if stat_key == "adaptive_search_stats":
                family_stats.append(
                    _scale_repo_setting_adaptive_stats(
                        family_entry.get(stat_key, {}),
                        scale=float(weight),
                    )
                )
            else:
                family_stats.append(
                    _scale_repo_setting_regression_stats(
                        family_entry.get(stat_key, {}),
                        scale=float(weight),
                    )
                )
    if stat_key == "adaptive_search_stats":
        return _merge_repo_setting_adaptive_stats(*family_stats)
    return _merge_repo_setting_regression_stats(*family_stats)


def _learn_repo_setting_policy_priors(
    *,
    prior_rounds: list[dict[str, object]],
    repo_setting_policy_pressure: Mapping[str, object] | None,
    persisted_priors: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pressure = repo_setting_policy_pressure if isinstance(repo_setting_policy_pressure, Mapping) else {}
    state_priors = _normalize_repo_setting_policy_priors(persisted_priors)
    active_campaign_width_signals = set(_normalize_benchmark_families(pressure.get("campaign_width_signals", [])))
    active_task_step_floor_signals = set(_normalize_benchmark_families(pressure.get("task_step_floor_signals", [])))
    active_adaptive_search_signals = set(_normalize_benchmark_families(pressure.get("adaptive_search_signals", [])))
    active_pair_map = _repo_setting_pair_map(pressure.get("focused_pairs", []))
    campaign_width_samples: list[tuple[float, float, float]] = []
    task_step_floor_samples: list[tuple[float, float, float]] = []
    adaptive_true_scores: list[float] = []
    adaptive_false_scores: list[float] = []
    family_campaign_width_samples: list[tuple[float, float, float]] = []
    family_task_step_floor_samples: list[tuple[float, float, float]] = []
    family_adaptive_true_scores: list[float] = []
    family_adaptive_false_scores: list[float] = []
    signal_observations: dict[str, int] = {}
    family_signal_observations: dict[str, int] = {}
    neighbor_signal_observations: dict[str, float] = {}
    focused_rounds = 0
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        rationale = round_payload.get("policy_shift_rationale", {})
        if not isinstance(rationale, dict):
            continue
        planner_pressure = rationale.get("planner_pressure", {})
        if not isinstance(planner_pressure, dict):
            continue
        prior_pressure = planner_pressure.get("repo_setting_policy_pressure", {})
        if not isinstance(prior_pressure, Mapping):
            continue
        prior_pair_map = _repo_setting_pair_map(prior_pressure.get("focused_pairs", []))
        prior_campaign_width_signals = set(_normalize_benchmark_families(prior_pressure.get("campaign_width_signals", [])))
        prior_task_step_floor_signals = set(_normalize_benchmark_families(prior_pressure.get("task_step_floor_signals", [])))
        prior_adaptive_search_signals = set(_normalize_benchmark_families(prior_pressure.get("adaptive_search_signals", [])))
        campaign_width_overlap = sorted(active_campaign_width_signals & prior_campaign_width_signals)
        task_step_floor_overlap = sorted(active_task_step_floor_signals & prior_task_step_floor_signals)
        adaptive_search_overlap = sorted(active_adaptive_search_signals & prior_adaptive_search_signals)
        pair_overlap_map: dict[str, set[str]] = {}
        for signal, families in active_pair_map.items():
            shared_families = set(families) & set(prior_pair_map.get(signal, set()))
            if shared_families:
                pair_overlap_map[signal] = shared_families
        neighbor_overlap_map = _repo_setting_family_neighbor_map(active_pair_map, prior_pair_map)
        if not campaign_width_overlap and not task_step_floor_overlap and not adaptive_search_overlap:
            continue
        focused_rounds += 1
        for signal in sorted(set(campaign_width_overlap) | set(task_step_floor_overlap) | set(adaptive_search_overlap)):
            signal_observations[signal] = signal_observations.get(signal, 0) + 1
        for signal, families in sorted(pair_overlap_map.items()):
            for family in sorted(families):
                family_signal_observations[f"{family}:{signal}"] = (
                    family_signal_observations.get(f"{family}:{signal}", 0) + 1
                )
        for signal, weighted_families in sorted(neighbor_overlap_map.items()):
            for active_family in sorted(active_pair_map.get(signal, set())):
                key = f"{active_family}:{signal}"
                neighbor_signal_observations[key] = (
                    float(neighbor_signal_observations.get(key, 0.0) or 0.0)
                    + sum(float(weight) for weight in weighted_families.values())
                )
        policy = round_payload.get("policy", {})
        if not isinstance(policy, Mapping):
            continue
        outcome_score = _repo_setting_retained_outcome_score(round_payload.get("campaign_report", {}))
        sample_weight = max(
            1.0,
            float(len(campaign_width_overlap) + len(task_step_floor_overlap) + len(adaptive_search_overlap)),
        )
        if campaign_width_overlap:
            campaign_width = max(1, int(policy.get("campaign_width", 1) or 1))
            campaign_width_samples.append(
                (float(max(0, campaign_width - 1)), outcome_score, sample_weight * float(len(campaign_width_overlap)))
            )
            family_overlap_count = sum(
                len(families) for signal, families in pair_overlap_map.items() if signal in campaign_width_overlap
            )
            neighbor_overlap_weight = sum(
                sum(float(weight) for weight in families.values())
                for signal, families in neighbor_overlap_map.items()
                if signal in campaign_width_overlap
            )
            if family_overlap_count > 0 or neighbor_overlap_weight > 0.0:
                family_campaign_width_samples.append(
                    (
                        float(max(0, campaign_width - 1)),
                        outcome_score,
                        sample_weight * (float(family_overlap_count) + neighbor_overlap_weight),
                    )
                )
        if task_step_floor_overlap:
            task_step_floor = max(1, int(policy.get("task_step_floor", 1) or 1))
            task_step_floor_samples.append(
                (
                    math.log2(float(task_step_floor)),
                    outcome_score,
                    sample_weight * float(len(task_step_floor_overlap)),
                )
            )
            family_overlap_count = sum(
                len(families) for signal, families in pair_overlap_map.items() if signal in task_step_floor_overlap
            )
            neighbor_overlap_weight = sum(
                sum(float(weight) for weight in families.values())
                for signal, families in neighbor_overlap_map.items()
                if signal in task_step_floor_overlap
            )
            if family_overlap_count > 0 or neighbor_overlap_weight > 0.0:
                family_task_step_floor_samples.append(
                    (
                        math.log2(float(task_step_floor)),
                        outcome_score,
                        sample_weight * (float(family_overlap_count) + neighbor_overlap_weight),
                    )
                )
        if adaptive_search_overlap:
            if bool(policy.get("adaptive_search", False)):
                adaptive_true_scores.append(outcome_score)
            else:
                adaptive_false_scores.append(outcome_score)
            family_overlap_count = sum(
                len(families) for signal, families in pair_overlap_map.items() if signal in adaptive_search_overlap
            )
            neighbor_overlap_weight = sum(
                sum(float(weight) for weight in families.values())
                for signal, families in neighbor_overlap_map.items()
                if signal in adaptive_search_overlap
            )
            if family_overlap_count > 0 or neighbor_overlap_weight > 0.0:
                if bool(policy.get("adaptive_search", False)):
                    family_adaptive_true_scores.append(outcome_score)
                else:
                    family_adaptive_false_scores.append(outcome_score)
    persisted_signal_observations: dict[str, int] = {}
    persisted_family_signal_observations: dict[str, int] = {}
    persisted_neighbor_signal_observations: dict[str, float] = {}
    for signal in _repo_setting_signal_set_union(
        active_campaign_width_signals,
        active_task_step_floor_signals,
        active_adaptive_search_signals,
    ):
        prior_entry = state_priors.get(signal, {})
        if isinstance(prior_entry, Mapping):
            persisted_signal_observations[signal] = max(0, int(prior_entry.get("observations", 0) or 0))
            if signal in signal_observations:
                signal_observations[signal] += persisted_signal_observations[signal]
            elif persisted_signal_observations[signal] > 0:
                signal_observations[signal] = persisted_signal_observations[signal]
    for signal, families in sorted(active_pair_map.items()):
        prior_entry = state_priors.get(signal, {})
        if not isinstance(prior_entry, Mapping):
            continue
        family_priors = prior_entry.get("family_priors", {})
        if not isinstance(family_priors, Mapping):
            continue
        for family in sorted(families):
            family_entry = family_priors.get(family, {})
            if not isinstance(family_entry, Mapping):
                continue
            key = f"{family}:{signal}"
            persisted_family_signal_observations[key] = max(0, int(family_entry.get("observations", 0) or 0))
            if key in family_signal_observations:
                family_signal_observations[key] += persisted_family_signal_observations[key]
            elif persisted_family_signal_observations[key] > 0:
                family_signal_observations[key] = persisted_family_signal_observations[key]
    state_family_pair_map = {
        signal: set(
            dict(state_priors.get(signal, {})).get("family_priors", {}).keys()
            if isinstance(state_priors.get(signal, {}), Mapping)
            and isinstance(dict(state_priors.get(signal, {})).get("family_priors", {}), Mapping)
            else []
        )
        for signal in active_pair_map
    }
    for signal, weighted_families in sorted(
        _repo_setting_family_neighbor_map(
            active_pair_map,
            state_family_pair_map,
        ).items()
    ):
        observed_sum = 0.0
        prior_entry = state_priors.get(signal, {})
        family_priors = prior_entry.get("family_priors", {}) if isinstance(prior_entry, Mapping) else {}
        if not isinstance(family_priors, Mapping):
            continue
        for family, weight in weighted_families.items():
            family_entry = family_priors.get(family, {})
            if not isinstance(family_entry, Mapping):
                continue
            observed_sum += float(max(0, int(family_entry.get("observations", 0) or 0))) * float(weight)
        for active_family in sorted(active_pair_map.get(signal, set())):
            key = f"{active_family}:{signal}"
            persisted_neighbor_signal_observations[key] = observed_sum
            neighbor_signal_observations[key] = float(neighbor_signal_observations.get(key, 0.0) or 0.0) + observed_sum
    persisted_campaign_width_stats = _merge_repo_setting_regression_stats(
        *[
            state_priors.get(signal, {}).get("campaign_width_stats", {})
            for signal in sorted(active_campaign_width_signals)
            if isinstance(state_priors.get(signal, {}), Mapping)
        ]
    )
    persisted_task_step_floor_stats = _merge_repo_setting_regression_stats(
        *[
            state_priors.get(signal, {}).get("task_step_floor_stats", {})
            for signal in sorted(active_task_step_floor_signals)
            if isinstance(state_priors.get(signal, {}), Mapping)
        ]
    )
    persisted_adaptive_search_stats = _merge_repo_setting_adaptive_stats(
        *[
            state_priors.get(signal, {}).get("adaptive_search_stats", {})
            for signal in sorted(active_adaptive_search_signals)
            if isinstance(state_priors.get(signal, {}), Mapping)
        ]
    )
    persisted_family_campaign_width_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: {family: 1.0 for family in families}
            for signal, families in active_pair_map.items()
            if signal in active_campaign_width_signals
        },
        stat_key="campaign_width_stats",
    )
    persisted_neighbor_campaign_width_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: families
            for signal, families in _repo_setting_family_neighbor_map(active_pair_map, state_family_pair_map).items()
            if signal in active_campaign_width_signals
        },
        stat_key="campaign_width_stats",
    )
    persisted_family_task_step_floor_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: {family: 1.0 for family in families}
            for signal, families in active_pair_map.items()
            if signal in active_task_step_floor_signals
        },
        stat_key="task_step_floor_stats",
    )
    persisted_neighbor_task_step_floor_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: families
            for signal, families in _repo_setting_family_neighbor_map(active_pair_map, state_family_pair_map).items()
            if signal in active_task_step_floor_signals
        },
        stat_key="task_step_floor_stats",
    )
    persisted_family_adaptive_search_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: {family: 1.0 for family in families}
            for signal, families in active_pair_map.items()
            if signal in active_adaptive_search_signals
        },
        stat_key="adaptive_search_stats",
    )
    persisted_neighbor_adaptive_search_stats = _weighted_family_repo_setting_prior_stats(
        state_priors=state_priors,
        family_weight_map={
            signal: families
            for signal, families in _repo_setting_family_neighbor_map(active_pair_map, state_family_pair_map).items()
            if signal in active_adaptive_search_signals
        },
        stat_key="adaptive_search_stats",
    )
    campaign_width_stats = _merge_repo_setting_regression_stats(
        persisted_campaign_width_stats,
        _repo_setting_regression_stats_from_samples(campaign_width_samples),
        persisted_family_campaign_width_stats,
        persisted_neighbor_campaign_width_stats,
        _repo_setting_regression_stats_from_samples(family_campaign_width_samples),
    )
    task_step_floor_stats = _merge_repo_setting_regression_stats(
        persisted_task_step_floor_stats,
        _repo_setting_regression_stats_from_samples(task_step_floor_samples),
        persisted_family_task_step_floor_stats,
        persisted_neighbor_task_step_floor_stats,
        _repo_setting_regression_stats_from_samples(family_task_step_floor_samples),
    )
    adaptive_search_stats = _merge_repo_setting_adaptive_stats(
        persisted_adaptive_search_stats,
        persisted_family_adaptive_search_stats,
        persisted_neighbor_adaptive_search_stats,
        {
            "observations": len(adaptive_true_scores) + len(adaptive_false_scores),
            "true_count": len(adaptive_true_scores),
            "false_count": len(adaptive_false_scores),
            "true_score_sum": sum(adaptive_true_scores),
            "false_score_sum": sum(adaptive_false_scores),
        },
        {
            "observations": len(family_adaptive_true_scores) + len(family_adaptive_false_scores),
            "true_count": len(family_adaptive_true_scores),
            "false_count": len(family_adaptive_false_scores),
            "true_score_sum": sum(family_adaptive_true_scores),
            "false_score_sum": sum(family_adaptive_false_scores),
        },
    )
    campaign_width_slope = _weighted_linear_slope_from_stats(campaign_width_stats)
    task_step_floor_slope = _weighted_linear_slope_from_stats(task_step_floor_stats)
    campaign_width_neighbor_support = sum(
        float(value)
        for key, value in neighbor_signal_observations.items()
        if str(key).rpartition(":")[2] in active_campaign_width_signals
    )
    task_step_floor_neighbor_support = sum(
        float(value)
        for key, value in neighbor_signal_observations.items()
        if str(key).rpartition(":")[2] in active_task_step_floor_signals
    )
    adaptive_search_neighbor_support = sum(
        float(value)
        for key, value in neighbor_signal_observations.items()
        if str(key).rpartition(":")[2] in active_adaptive_search_signals
    )
    campaign_width_unit_weight = _clamp_float(
        0.3 + (0.18 * campaign_width_slope) + min(0.12, 0.04 * campaign_width_neighbor_support),
        min_value=0.08,
        max_value=0.75,
    )
    task_step_floor_unit_weight = _clamp_float(
        0.4 + (0.12 * task_step_floor_slope) + min(0.12, 0.04 * task_step_floor_neighbor_support),
        min_value=0.15,
        max_value=0.85,
    )
    adaptive_search_bonus = 0.25
    if int(adaptive_search_stats.get("true_count", 0) or 0) > 0 and int(adaptive_search_stats.get("false_count", 0) or 0) > 0:
        adaptive_delta = (
            float(adaptive_search_stats.get("true_score_sum", 0.0) or 0.0)
            / float(max(1, int(adaptive_search_stats.get("true_count", 0) or 0)))
        ) - (
            float(adaptive_search_stats.get("false_score_sum", 0.0) or 0.0)
            / float(max(1, int(adaptive_search_stats.get("false_count", 0) or 0)))
        )
        adaptive_search_bonus = _clamp_float(
            0.25 + (0.12 * adaptive_delta) + min(0.08, 0.03 * adaptive_search_neighbor_support),
            min_value=0.05,
            max_value=0.45,
        )
    elif adaptive_search_neighbor_support > 0.0:
        adaptive_search_bonus = _clamp_float(
            adaptive_search_bonus + min(0.08, 0.03 * adaptive_search_neighbor_support),
            min_value=0.05,
            max_value=0.45,
        )
    return {
        "focused_rounds": focused_rounds
        + sum(persisted_signal_observations.values()),
        "signal_observations": dict(sorted(signal_observations.items())),
        "campaign_width_observations": int(
            _merge_repo_setting_regression_stats(
                persisted_campaign_width_stats,
                _repo_setting_regression_stats_from_samples(campaign_width_samples),
            ).get("observations", 0)
            or 0
        ),
        "campaign_width_slope": campaign_width_slope,
        "campaign_width_unit_weight": campaign_width_unit_weight,
        "task_step_floor_observations": int(
            _merge_repo_setting_regression_stats(
                persisted_task_step_floor_stats,
                _repo_setting_regression_stats_from_samples(task_step_floor_samples),
            ).get("observations", 0)
            or 0
        ),
        "task_step_floor_slope": task_step_floor_slope,
        "task_step_floor_unit_weight": task_step_floor_unit_weight,
        "adaptive_search_observations": int(
            _merge_repo_setting_adaptive_stats(
                persisted_adaptive_search_stats,
                {
                    "observations": len(adaptive_true_scores) + len(adaptive_false_scores),
                    "true_count": len(adaptive_true_scores),
                    "false_count": len(adaptive_false_scores),
                    "true_score_sum": sum(adaptive_true_scores),
                    "false_score_sum": sum(adaptive_false_scores),
                },
            ).get("observations", 0)
            or 0
        ),
        "adaptive_search_bonus": adaptive_search_bonus,
        "campaign_width_neighbor_support": campaign_width_neighbor_support,
        "task_step_floor_neighbor_support": task_step_floor_neighbor_support,
        "adaptive_search_neighbor_support": adaptive_search_neighbor_support,
        "persisted_signal_observations": dict(sorted(persisted_signal_observations.items())),
        "family_signal_observations": dict(sorted(family_signal_observations.items())),
        "persisted_family_signal_observations": dict(sorted(persisted_family_signal_observations.items())),
        "neighbor_signal_observations": dict(sorted(neighbor_signal_observations.items())),
        "persisted_neighbor_signal_observations": dict(sorted(persisted_neighbor_signal_observations.items())),
        "active_focused_pairs": sorted(f"{family}:{signal}" for signal, families in active_pair_map.items() for family in families),
    }


def _update_repo_setting_policy_priors(
    controller_state: Mapping[str, object] | None,
    *,
    round_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    repo_setting_decay = 0.92
    state = dict(controller_state) if isinstance(controller_state, Mapping) else {}
    priors = _normalize_repo_setting_policy_priors(state.get("repo_setting_policy_priors", {}))
    payload = round_payload if isinstance(round_payload, Mapping) else {}
    rationale = payload.get("policy_shift_rationale", {})
    planner_pressure = rationale.get("planner_pressure", {}) if isinstance(rationale, Mapping) else {}
    repo_setting_policy_pressure = (
        planner_pressure.get("repo_setting_policy_pressure", {})
        if isinstance(planner_pressure, Mapping)
        else {}
    )
    if not isinstance(repo_setting_policy_pressure, Mapping):
        state["repo_setting_policy_priors"] = priors
        return state
    policy = payload.get("policy", {})
    if not isinstance(policy, Mapping):
        state["repo_setting_policy_priors"] = priors
        return state
    outcome_score = _repo_setting_retained_outcome_score(payload.get("campaign_report", {}))
    campaign_width = max(1, int(policy.get("campaign_width", 1) or 1))
    task_step_floor = max(1, int(policy.get("task_step_floor", 1) or 1))
    adaptive_search = bool(policy.get("adaptive_search", False))
    campaign_width_signals = _normalize_benchmark_families(repo_setting_policy_pressure.get("campaign_width_signals", []))
    task_step_floor_signals = _normalize_benchmark_families(repo_setting_policy_pressure.get("task_step_floor_signals", []))
    adaptive_search_signals = _normalize_benchmark_families(repo_setting_policy_pressure.get("adaptive_search_signals", []))
    focused_pair_map = _repo_setting_pair_map(repo_setting_policy_pressure.get("focused_pairs", []))
    active_signals = _repo_setting_signal_set_union(
        campaign_width_signals,
        task_step_floor_signals,
        adaptive_search_signals,
    )
    for signal in active_signals:
        entry = _decay_repo_setting_prior_entry(priors.get(signal, {}), decay=repo_setting_decay)
        if not entry:
            entry = _empty_repo_setting_prior_entry()
        entry["observations"] = max(0, int(entry.get("observations", 0) or 0)) + 1
        entry["last_outcome_score"] = outcome_score
        if signal in campaign_width_signals:
            stats = _merge_repo_setting_regression_stats(
                entry.get("campaign_width_stats", {}),
                _repo_setting_regression_stats_from_samples([(float(max(0, campaign_width - 1)), outcome_score, 1.0)]),
            )
            entry["campaign_width_stats"] = stats
        if signal in task_step_floor_signals:
            stats = _merge_repo_setting_regression_stats(
                entry.get("task_step_floor_stats", {}),
                _repo_setting_regression_stats_from_samples([(math.log2(float(task_step_floor)), outcome_score, 1.0)]),
            )
            entry["task_step_floor_stats"] = stats
        if signal in adaptive_search_signals:
            adaptive_stats = _merge_repo_setting_adaptive_stats(
                entry.get("adaptive_search_stats", {}),
                {
                    "observations": 1,
                    "true_count": 1 if adaptive_search else 0,
                    "false_count": 0 if adaptive_search else 1,
                    "true_score_sum": outcome_score if adaptive_search else 0.0,
                    "false_score_sum": 0.0 if adaptive_search else outcome_score,
                },
            )
            entry["adaptive_search_stats"] = adaptive_stats
        family_priors = dict(entry.get("family_priors", {})) if isinstance(entry.get("family_priors", {}), Mapping) else {}
        for family in sorted(focused_pair_map.get(signal, set())):
            family_entry = _decay_repo_setting_prior_entry(family_priors.get(family, {}), decay=repo_setting_decay)
            if not family_entry:
                family_entry = _empty_repo_setting_prior_entry()
            family_entry["observations"] = max(0, int(family_entry.get("observations", 0) or 0)) + 1
            family_entry["last_outcome_score"] = outcome_score
            if signal in campaign_width_signals:
                family_entry["campaign_width_stats"] = _merge_repo_setting_regression_stats(
                    family_entry.get("campaign_width_stats", {}),
                    _repo_setting_regression_stats_from_samples(
                        [(float(max(0, campaign_width - 1)), outcome_score, 1.0)]
                    ),
                )
            if signal in task_step_floor_signals:
                family_entry["task_step_floor_stats"] = _merge_repo_setting_regression_stats(
                    family_entry.get("task_step_floor_stats", {}),
                    _repo_setting_regression_stats_from_samples(
                        [(math.log2(float(task_step_floor)), outcome_score, 1.0)]
                    ),
                )
            if signal in adaptive_search_signals:
                family_entry["adaptive_search_stats"] = _merge_repo_setting_adaptive_stats(
                    family_entry.get("adaptive_search_stats", {}),
                    {
                        "observations": 1,
                        "true_count": 1 if adaptive_search else 0,
                        "false_count": 0 if adaptive_search else 1,
                        "true_score_sum": outcome_score if adaptive_search else 0.0,
                        "false_score_sum": 0.0 if adaptive_search else outcome_score,
                    },
                )
            family_entry["family_priors"] = {}
            family_priors[family] = family_entry
        entry["family_priors"] = family_priors
        priors[signal] = entry
    state["repo_setting_policy_priors"] = priors
    return state


def _repo_setting_candidate_score_adjustment(
    *,
    policy: Mapping[str, object] | None,
    current_policy: Mapping[str, object] | None,
    repo_setting_policy_pressure: Mapping[str, object] | None,
    learned_priors: Mapping[str, object] | None = None,
) -> dict[str, object]:
    proposal = policy if isinstance(policy, Mapping) else {}
    current = current_policy if isinstance(current_policy, Mapping) else {}
    pressure = repo_setting_policy_pressure if isinstance(repo_setting_policy_pressure, Mapping) else {}
    priors = learned_priors if isinstance(learned_priors, Mapping) else {}
    current_cycles = max(1, int(current.get("cycles", 1) or 1))
    current_campaign_width = max(1, int(current.get("campaign_width", 1) or 1))
    current_variant_width = max(1, int(current.get("variant_width", 1) or 1))
    current_task_limit = max(1, int(current.get("task_limit", 1) or 1))
    current_task_step_floor = max(1, int(current.get("task_step_floor", 1) or 1))
    proposal_cycles = max(1, int(proposal.get("cycles", current_cycles) or current_cycles))
    proposal_campaign_width = max(1, int(proposal.get("campaign_width", current_campaign_width) or current_campaign_width))
    proposal_variant_width = max(1, int(proposal.get("variant_width", current_variant_width) or current_variant_width))
    proposal_task_limit = max(1, int(proposal.get("task_limit", current_task_limit) or current_task_limit))
    proposal_task_step_floor = max(1, int(proposal.get("task_step_floor", current_task_step_floor) or current_task_step_floor))
    proposal_focus = str(proposal.get("focus", "")).strip()
    proposal_adaptive = bool(proposal.get("adaptive_search", False))
    campaign_width_signals = _normalize_benchmark_families(pressure.get("campaign_width_signals", []))
    task_step_floor_signals = _normalize_benchmark_families(pressure.get("task_step_floor_signals", []))
    adaptive_search_signals = _normalize_benchmark_families(pressure.get("adaptive_search_signals", []))
    cycles_delta = proposal_cycles - current_cycles
    campaign_width_delta = proposal_campaign_width - current_campaign_width
    variant_width_delta = proposal_variant_width - current_variant_width
    task_limit_ratio = float(proposal_task_limit) / float(max(1, current_task_limit))
    task_step_floor_delta = proposal_task_step_floor - current_task_step_floor
    campaign_width_unit_weight = _clamp_float(
        float(priors.get("campaign_width_unit_weight", 0.3) or 0.3),
        min_value=0.08,
        max_value=0.75,
    )
    task_step_floor_unit_weight = _clamp_float(
        float(priors.get("task_step_floor_unit_weight", 0.4) or 0.4),
        min_value=0.15,
        max_value=0.85,
    )
    adaptive_search_bonus = _clamp_float(
        float(priors.get("adaptive_search_bonus", 0.25) or 0.25),
        min_value=0.05,
        max_value=0.45,
    )

    campaign_width_adjustment = 0.0
    if campaign_width_signals:
        if campaign_width_delta > 0:
            campaign_width_adjustment = min(0.9, campaign_width_unit_weight * float(campaign_width_delta))
        else:
            campaign_width_adjustment = _clamp_float(
                -0.05 - (0.5 * campaign_width_unit_weight),
                min_value=-0.45,
                max_value=-0.02,
            )

    task_step_floor_adjustment = 0.0
    if task_step_floor_signals:
        if task_step_floor_delta > 0:
            depth_ratio = float(proposal_task_step_floor) / float(max(1, current_task_step_floor))
            task_step_floor_adjustment = min(
                0.9,
                task_step_floor_unit_weight * math.log2(max(1.0, depth_ratio)),
            )
        elif task_step_floor_delta < 0:
            task_step_floor_adjustment = _clamp_float(
                -0.05 - (0.5 * task_step_floor_unit_weight),
                min_value=-0.4,
                max_value=-0.1,
            )
        else:
            task_step_floor_adjustment = _clamp_float(
                -0.125 * task_step_floor_unit_weight,
                min_value=-0.12,
                max_value=-0.02,
            )

    adaptive_search_adjustment = 0.0
    focus_adjustment = 0.0
    if adaptive_search_signals:
        adaptive_search_adjustment = adaptive_search_bonus if proposal_adaptive else -adaptive_search_bonus
        if proposal_focus == "discovered_task_adaptation":
            focus_adjustment = 0.15
        elif proposal_focus == "balanced":
            focus_adjustment = 0.05

    complexity_penalty = 0.0
    if not campaign_width_signals and campaign_width_delta > 0:
        complexity_penalty -= min(0.45, 0.22 * float(campaign_width_delta))
    if not campaign_width_signals and variant_width_delta > 0:
        complexity_penalty -= min(0.35, 0.14 * float(variant_width_delta))
    if task_step_floor_signals and cycles_delta > 0:
        complexity_penalty -= min(0.18, 0.06 * float(cycles_delta))
    if task_step_floor_signals and task_limit_ratio > 1.0:
        complexity_penalty -= min(0.22, 0.08 * math.log2(max(1.0, task_limit_ratio)))

    total_adjustment = (
        campaign_width_adjustment
        + task_step_floor_adjustment
        + adaptive_search_adjustment
        + focus_adjustment
        + complexity_penalty
    )
    return {
        "score_adjustment": total_adjustment,
        "campaign_width_adjustment": campaign_width_adjustment,
        "task_step_floor_adjustment": task_step_floor_adjustment,
        "adaptive_search_adjustment": adaptive_search_adjustment,
        "focus_adjustment": focus_adjustment,
        "complexity_penalty": complexity_penalty,
        "cycles_delta": cycles_delta,
        "campaign_width_delta": campaign_width_delta,
        "variant_width_delta": variant_width_delta,
        "task_limit_ratio": task_limit_ratio,
        "task_step_floor_delta": task_step_floor_delta,
        "focused_pairs": _normalize_benchmark_families(pressure.get("focused_pairs", [])),
        "campaign_width_signals": campaign_width_signals,
        "task_step_floor_signals": task_step_floor_signals,
        "adaptive_search_signals": adaptive_search_signals,
        "learned_priors": {
            "campaign_width_unit_weight": campaign_width_unit_weight,
            "task_step_floor_unit_weight": task_step_floor_unit_weight,
            "adaptive_search_bonus": adaptive_search_bonus,
            "focused_rounds": max(0, int(priors.get("focused_rounds", 0) or 0)),
            "signal_observations": dict(priors.get("signal_observations", {}))
            if isinstance(priors.get("signal_observations", {}), dict)
            else {},
            "campaign_width_observations": max(0, int(priors.get("campaign_width_observations", 0) or 0)),
            "task_step_floor_observations": max(0, int(priors.get("task_step_floor_observations", 0) or 0)),
            "adaptive_search_observations": max(0, int(priors.get("adaptive_search_observations", 0) or 0)),
        },
    }


def _policy_shift_score_adjustment(rationale: dict[str, object]) -> float:
    if not isinstance(rationale, dict):
        return 0.0
    reason_codes = rationale.get("reason_codes", [])
    if not isinstance(reason_codes, list):
        reason_codes = []
    codes = {str(item).strip() for item in reason_codes if str(item).strip()}
    cooled = rationale.get("cooled_subsystems", [])
    if not isinstance(cooled, list):
        cooled = []
    widened = rationale.get("widened_dimensions", [])
    if not isinstance(widened, list):
        widened = []
    score = 0.0
    if "stable_progress" in codes:
        score += 1.5
    if "weak_retention_outcome" in codes:
        score -= 1.0
    if "campaign_breadth_pressure" in codes:
        score -= 1.5
    if "variant_breadth_pressure" in codes:
        score -= 1.0
    if "priority_family_under_sampled" in codes:
        score -= 1.0
    if "priority_family_low_return" in codes:
        score -= 0.5
    if "no_yield_round" in codes:
        score -= 2.0
    if "stalled_subsystem_cooldown" in codes:
        score -= 0.75
    if "subsystem_reject_cooldown" in codes:
        score -= 0.5
    if "subsystem_monoculture" in codes:
        score -= 1.5
    if "safety_regression" in codes or "phase_gate_failure" in codes:
        score -= 3.0
    if "liftoff_reject" in codes:
        score -= 2.0
    score -= 0.25 * float(len(cooled))
    if codes - {"stable_progress"}:
        score -= 0.15 * float(len(widened))
    return score


def _historical_policy_rationale_adjustments(prior_rounds: list[dict[str, object]]) -> dict[str, float]:
    adjustments: dict[str, list[float]] = {}
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        policy = round_payload.get("policy", {})
        rationale = round_payload.get("policy_shift_rationale", {})
        if not isinstance(policy, dict) or not isinstance(rationale, dict):
            continue
        action_key = action_key_for_policy(policy)
        adjustments.setdefault(action_key, []).append(_policy_shift_score_adjustment(rationale))
    return {
        action_key: (sum(values) / len(values))
        for action_key, values in adjustments.items()
        if values
    }


def _policy_shift_summary(rounds: list[dict[str, object]]) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    cooled_counts: dict[str, int] = {}
    widened_counts: dict[str, int] = {}
    recent_rationales: list[dict[str, object]] = []
    for round_payload in rounds:
        if not isinstance(round_payload, dict):
            continue
        rationale = round_payload.get("policy_shift_rationale", {})
        if not isinstance(rationale, dict):
            continue
        reason_codes = rationale.get("reason_codes", [])
        cooled = rationale.get("cooled_subsystems", [])
        widened = rationale.get("widened_dimensions", [])
        if isinstance(reason_codes, list):
            for reason in reason_codes:
                token = str(reason).strip()
                if token:
                    reason_counts[token] = reason_counts.get(token, 0) + 1
        if isinstance(cooled, list):
            for subsystem in cooled:
                token = str(subsystem).strip()
                if token:
                    cooled_counts[token] = cooled_counts.get(token, 0) + 1
        if isinstance(widened, list):
            for dimension in widened:
                token = str(dimension).strip()
                if token:
                    widened_counts[token] = widened_counts.get(token, 0) + 1
        recent_rationales.append(
            {
                "round_index": int(round_payload.get("round_index", 0) or 0),
                "reason_codes": [str(item).strip() for item in reason_codes if str(item).strip()]
                if isinstance(reason_codes, list)
                else [],
                "cooled_subsystems": [str(item).strip() for item in cooled if str(item).strip()]
                if isinstance(cooled, list)
                else [],
                "widened_dimensions": [str(item).strip() for item in widened if str(item).strip()]
                if isinstance(widened, list)
                else [],
            }
        )
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "cooled_subsystem_counts": dict(sorted(cooled_counts.items())),
        "widened_dimension_counts": dict(sorted(widened_counts.items())),
        "recent_rationales": recent_rationales[-10:],
    }


def _persist_semantic_round_artifacts(
    *,
    config: KernelConfig,
    run_id: str,
    round_payload: dict[str, object],
    current_policy: dict[str, object],
) -> dict[str, str]:
    round_index = int(round_payload.get("round_index", 0) or 0)
    semantic_agent_id = f"unattended_agent:{run_id}"
    semantic_attempt_id = f"unattended_round:{run_id}:{round_index}"
    semantic_redirection = (
        dict(round_payload.get("semantic_redirection", {}))
        if isinstance(round_payload.get("semantic_redirection", {}), dict)
        else {}
    )
    semantic_skill = _round_semantic_skill_payload(
        round_payload=round_payload,
        current_policy=current_policy,
    )
    attempt_path = record_semantic_attempt(
        config,
        attempt_id=semantic_attempt_id,
        payload={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "round_index": round_index,
            "status": str(round_payload.get("status", "")).strip(),
            "phase": str(round_payload.get("phase", "")).strip(),
            "policy": dict(round_payload.get("policy", {})) if isinstance(round_payload.get("policy", {}), dict) else {},
            "next_policy": dict(round_payload.get("next_policy", {}))
            if isinstance(round_payload.get("next_policy", {}), dict)
            else {},
            "campaign_report_path": str(round_payload.get("campaign_report_path", "")).strip(),
            "liftoff_report_path": str(round_payload.get("liftoff_report_path", "")).strip(),
            "policy_shift_rationale": dict(round_payload.get("policy_shift_rationale", {}))
            if isinstance(round_payload.get("policy_shift_rationale", {}), dict)
            else {},
            "semantic_redirection": semantic_redirection,
            "controller_update": dict(round_payload.get("controller_update", {}))
            if isinstance(round_payload.get("controller_update", {}), dict)
            else {},
            "controller_failure_update": dict(round_payload.get("controller_failure_update", {}))
            if isinstance(round_payload.get("controller_failure_update", {}), dict)
            else {},
            "trust_breadth_summary": dict(round_payload.get("trust_breadth_summary", {}))
            if isinstance(round_payload.get("trust_breadth_summary", {}), dict)
            else {},
            "decision_stream_summary": dict(round_payload.get("decision_stream_summary", {}))
            if isinstance(round_payload.get("decision_stream_summary", {}), dict)
            else {},
            "campaign_report_summary": {
                "retained_gain_runs": int(
                    dict(round_payload.get("campaign_report", {})).get("retained_gain_runs", 0)
                    if isinstance(round_payload.get("campaign_report", {}), dict)
                    else 0
                ),
                "runtime_managed_decisions": int(
                    dict(round_payload.get("campaign_report", {})).get("runtime_managed_decisions", 0)
                    if isinstance(round_payload.get("campaign_report", {}), dict)
                    else 0
                ),
            },
        },
    )
    note_path = record_semantic_note(
        config,
        note_id=f"{semantic_attempt_id}.reflection",
        payload={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempt_id": semantic_attempt_id,
            "round_index": round_index,
            "focus_before": str(current_policy.get("focus", "")).strip(),
            "focus_after": str(
                dict(round_payload.get("next_policy", {})).get("focus", current_policy.get("focus", ""))
            ).strip(),
            "reason_codes": list(semantic_redirection.get("reason_codes", [])),
            "semantic_hypotheses": list(semantic_redirection.get("semantic_hypotheses", [])),
            "required_new_parent": bool(semantic_redirection.get("required_new_parent", False)),
            "source_subsystems": list(semantic_redirection.get("source_subsystems", [])),
            "priority_families": list(semantic_skill.get("priority_families", [])),
        },
    )
    redirect_path = None
    skill_path = None
    if bool(semantic_redirection.get("triggered", False)):
        redirect_path = record_semantic_redirect(
            config,
            redirect_id=f"{semantic_attempt_id}.redirect",
            payload={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "attempt_id": semantic_attempt_id,
                "round_index": round_index,
                "policy_before": dict(current_policy),
                "policy_after": dict(round_payload.get("next_policy", {}))
                if isinstance(round_payload.get("next_policy", {}), dict)
                else {},
                **semantic_redirection,
            },
        )
    if semantic_skill:
        skill_path = record_semantic_skill(
            config,
            skill_id=f"{semantic_attempt_id}.skill",
            payload={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "attempt_id": semantic_attempt_id,
                "round_index": round_index,
                **semantic_skill,
            },
        )
    upsert_semantic_agent(
        config,
        agent_id=semantic_agent_id,
        payload={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_kind": "unattended_campaign_agent",
            "run_id": run_id,
            "last_round_index": round_index,
            "current_policy": dict(round_payload.get("next_policy", current_policy))
            if isinstance(round_payload.get("next_policy", current_policy), dict)
            else dict(current_policy),
            "last_attempt_id": semantic_attempt_id,
            "last_redirect_id": "" if redirect_path is None else f"redirect:{redirect_path.stem}",
            "last_skill_id": "" if skill_path is None else f"skill:{skill_path.stem}",
            "stagnation_count": max(
                int(round_payload.get("no_yield_rounds", 0) or 0),
                int(round_payload.get("policy_stall_rounds", 0) or 0),
            ),
            "status": str(round_payload.get("status", "")).strip() or "completed",
        },
    )
    return {
        "semantic_agent_id": semantic_agent_id,
        "semantic_attempt_path": str(attempt_path),
        "semantic_note_path": str(note_path),
        "semantic_redirect_path": "" if redirect_path is None else str(redirect_path),
        "semantic_skill_path": "" if skill_path is None else str(skill_path),
    }


def _derived_controller_features(
    *,
    campaign_report: Mapping[str, object] | None,
    campaign_signal: Mapping[str, object] | None,
    planner_pressure_signal: Mapping[str, object] | None,
    round_signal: Mapping[str, object] | None,
    current_policy: Mapping[str, object] | None = None,
) -> dict[str, float]:
    report = campaign_report if isinstance(campaign_report, Mapping) else {}
    signal = campaign_signal if isinstance(campaign_signal, Mapping) else {}
    pressure = planner_pressure_signal if isinstance(planner_pressure_signal, Mapping) else {}
    round_state = round_signal if isinstance(round_signal, Mapping) else {}
    policy = current_policy if isinstance(current_policy, Mapping) else {}

    recent_production = report.get("recent_production_decisions", [])
    if not isinstance(recent_production, list):
        recent_production = []
    productive_subsystems: list[str] = []
    for record in recent_production:
        if not isinstance(record, Mapping):
            continue
        subsystem = str(record.get("subsystem", "")).strip()
        if not subsystem or subsystem in productive_subsystems:
            continue
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        pass_rate_delta = float(metrics.get("pass_rate_delta", 0.0) or 0.0)
        baseline_pass_rate = float(metrics.get("baseline_pass_rate", 0.0) or 0.0)
        candidate_pass_rate = float(metrics.get("candidate_pass_rate", 0.0) or 0.0)
        retained_positive_delta_decisions = int(metrics.get("retained_positive_delta_decisions", 0) or 0)
        if (
            str(record.get("state", "")).strip() == "retain"
            or pass_rate_delta > 0.0
            or candidate_pass_rate > baseline_pass_rate
            or retained_positive_delta_decisions > 0
        ):
            productive_subsystems.append(subsystem)

    productive_priority_families = _normalize_benchmark_families(signal.get("priority_families_with_retained_gain", []))
    productive_priority_family_set = set(productive_priority_families)
    current_priority_family_set = set(_normalize_benchmark_families(policy.get("priority_benchmark_families", [])))
    productive_subsystem_gain = min(1.0, float(len(productive_subsystems)) / 3.0)
    productive_family_gain = min(1.0, float(max(0, len(productive_priority_family_set) - 1)) / 2.0)
    productive_depth_gain = min(1.0, float(max(0, int(signal.get("productive_depth_retained_cycles", 0) or 0))) / 2.0)
    long_horizon_gain = min(1.0, float(max(0, int(signal.get("long_horizon_retained_cycles", 0) or 0))) / 2.0)
    coding_frontier_ready = (
        bool(productive_priority_family_set)
        and {"project", "repository"}.issubset(productive_priority_family_set)
        and int(signal.get("retained_cycles", 0) or 0) > 0
        and bool(signal.get("all_retained_phase_gates_passed", True))
        and int(signal.get("failed_decisions", 0) or 0) <= 0
        and float(signal.get("worst_family_delta", 0.0) or 0.0) >= 0.0
        and float(signal.get("worst_generated_family_delta", 0.0) or 0.0) >= 0.0
        and int(signal.get("max_regressed_families", 0) or 0) <= 0
        and int(signal.get("max_generated_regressed_families", 0) or 0) <= 0
        and productive_depth_gain > 0.0
        and bool(current_priority_family_set)
    )

    frontier_failure_families = _normalize_benchmark_families(pressure.get("frontier_failure_motif_families", []))
    frontier_repo_setting_families = _normalize_benchmark_families(pressure.get("frontier_repo_setting_families", []))
    semantic_state = round_state.get("semantic_progress_state", {})
    if not isinstance(semantic_state, Mapping):
        semantic_state = {}
    semantic_phase_family = str(semantic_state.get("phase_family", "")).strip()
    semantic_progress_class = str(semantic_state.get("progress_class", "")).strip()
    semantic_decision_distance = str(semantic_state.get("decision_distance", "")).strip()
    semantic_hook_coverage = 0.0
    if semantic_state:
        semantic_hook_coverage = 0.5
        if semantic_progress_class and semantic_progress_class != "unknown":
            semantic_hook_coverage += 0.25
        if semantic_decision_distance in {"near", "decision_emitted", "complete"}:
            semantic_hook_coverage += 0.25
    failure_hook_coverage = 0.0
    if frontier_failure_families:
        failure_hook_coverage = max(
            0.5,
            float(len(set(frontier_failure_families) & productive_priority_family_set))
            / float(max(1, len(frontier_failure_families))),
        )
    repo_setting_hook_coverage = 0.0
    if frontier_repo_setting_families:
        repo_setting_hook_coverage = max(
            0.5,
            float(len(set(frontier_repo_setting_families) & productive_priority_family_set))
            / float(max(1, len(frontier_repo_setting_families))),
        )
    decision_signal_count = max(
        0,
        int(signal.get("runtime_managed_decisions", 0) or 0)
        + int(signal.get("non_runtime_managed_decisions", 0) or 0)
        + int(signal.get("decision_runs", 0) or 0)
        + int(round_state.get("decision_records_considered", 0) or 0),
    )
    decision_hook_coverage = min(1.0, float(decision_signal_count) / 2.0)

    novel_subsystem_capability_gain = min(
        1.0,
        (0.35 * productive_subsystem_gain)
        + (0.3 * productive_family_gain)
        + (0.2 * productive_depth_gain)
        + (0.1 * long_horizon_gain)
        + (0.05 if coding_frontier_ready else 0.0),
    )
    strategy_hook_coverage = min(
        1.0,
        (failure_hook_coverage + repo_setting_hook_coverage + semantic_hook_coverage + decision_hook_coverage) / 4.0,
    )
    if semantic_phase_family in {"preview", "apply", "finalize"}:
        strategy_hook_coverage = min(1.0, strategy_hook_coverage + 0.1)
    return {
        "novel_subsystem_capability_gain": novel_subsystem_capability_gain,
        "strategy_hook_coverage": strategy_hook_coverage,
        "trust_bootstrap_pressure": min(
            1.0,
            (
                (1.0 if int(signal.get("external_report_count", 0) or 0) <= 0 else 0.0)
                + (1.0 if _normalize_benchmark_families(signal.get("missing_required_families", [])) else 0.0)
                + min(1.0, float(max(0, int(signal.get("distinct_family_gap", 0) or 0))) / 3.0)
            )
            / 3.0,
        ),
        "claim_ready_trust_gain": 1.0
        if (
            int(signal.get("external_report_count", 0) or 0) > 0
            and not _normalize_benchmark_families(signal.get("missing_required_families", []))
            and int(signal.get("distinct_family_gap", 0) or 0) <= 0
        )
        else 0.0,
        "supervision_breadth_delta": min(
            1.0,
            float(
                max(
                    0,
                    len(
                        set(_normalize_benchmark_families(round_state.get("sampled_families", [])))
                        - set(_normalize_benchmark_families(signal.get("required_families_with_reports", [])))
                    ),
                )
            )
            / 3.0,
        ),
    }


def _strategy_memory_observation_features(
    *,
    controller_state: Mapping[str, object] | None,
    planner_pressure_signal: Mapping[str, object] | None,
) -> dict[str, float]:
    state = controller_state if isinstance(controller_state, Mapping) else {}
    repo_setting_priors = (
        dict(state.get("repo_setting_policy_priors", {}))
        if isinstance(state.get("repo_setting_policy_priors", {}), Mapping)
        else {}
    )
    strategy_priors = (
        dict(state.get("strategy_memory_priors", {}))
        if isinstance(state.get("strategy_memory_priors", {}), Mapping)
        else {}
    )
    pressure = planner_pressure_signal if isinstance(planner_pressure_signal, Mapping) else {}
    if not repo_setting_priors:
        features = {
            "strategy_memory_prior_strength": 0.0,
            "strategy_memory_alignment": 0.0,
            "strategy_memory_reject_pressure": 0.0,
        }
    else:
        campaign_width_signals = set(_normalize_benchmark_families(pressure.get("frontier_repo_setting_families", [])))
        alignment = 0.0
        weighted_prior_count = 0.0
        for signal_name, entry in repo_setting_priors.items():
            if not isinstance(entry, Mapping):
                continue
            weighted_prior_count += max(0.0, float(entry.get("campaign_width_unit_weight", 0.0) or 0.0))
            if signal_name in campaign_width_signals:
                alignment += max(0.0, float(entry.get("adaptive_search_bonus", 0.0) or 0.0))
        features = {
            "strategy_memory_prior_strength": min(1.0, weighted_prior_count / 3.0),
            "strategy_memory_alignment": min(1.0, alignment / 2.0),
            "strategy_memory_reject_pressure": 0.0,
        }
    if not strategy_priors:
        return features
    retained_family_gains = (
        dict(strategy_priors.get("retained_family_gains", {}))
        if isinstance(strategy_priors.get("retained_family_gains", {}), Mapping)
        else {}
    )
    rejects_by_family = (
        dict(strategy_priors.get("recent_rejects_by_family", {}))
        if isinstance(strategy_priors.get("recent_rejects_by_family", {}), Mapping)
        else {}
    )
    relevant_families = set(
        _normalize_benchmark_families(pressure.get("frontier_repo_setting_families", []))
        + _normalize_benchmark_families(pressure.get("missing_required_families", []))
        + _normalize_benchmark_families(pressure.get("priority_families", []))
    )
    if not relevant_families:
        relevant_families = set(retained_family_gains) | set(rejects_by_family)
    aligned_gain = sum(
        max(0.0, float(retained_family_gains.get(family, 0.0) or 0.0))
        for family in sorted(relevant_families)
    )
    aligned_rejects = sum(
        max(0, int(rejects_by_family.get(family, 0) or 0))
        for family in sorted(relevant_families)
    )
    features["strategy_memory_prior_strength"] = min(
        1.0,
        max(
            float(features.get("strategy_memory_prior_strength", 0.0) or 0.0),
            min(
                0.75,
                (float(max(0, int(strategy_priors.get("retained_count", 0) or 0))) / 4.0)
                + max(0.0, float(strategy_priors.get("best_retained_gain", 0.0) or 0.0)),
            ),
        ),
    )
    features["strategy_memory_alignment"] = min(
        1.0,
        max(
            float(features.get("strategy_memory_alignment", 0.0) or 0.0),
            min(0.75, aligned_gain),
        ),
    )
    features["strategy_memory_reject_pressure"] = min(
        1.0,
        (float(aligned_rejects) / 3.0)
        + min(0.4, float(max(0, int(strategy_priors.get("recent_rejects", 0) or 0))) / 5.0),
    )
    return features


def _summarize_strategy_memory_priors(config: KernelConfig) -> dict[str, object]:
    nodes = load_strategy_nodes(config)
    retained_family_gains: dict[str, float] = {}
    recent_rejects_by_family: dict[str, int] = {}
    retained_subsystems: dict[str, float] = {}
    recent_rejects_by_subsystem: dict[str, int] = {}
    retained_count = 0
    rejected_count = 0
    best_retained_gain = 0.0
    recent_nodes = sorted(
        nodes,
        key=lambda node: (str(node.updated_at), str(node.created_at), str(node.strategy_node_id)),
    )[-6:]
    for node in nodes:
        subsystem = str(node.subsystem).strip().lower()
        if node.retention_state == "retain":
            retained_count += 1
            best_retained_gain = max(best_retained_gain, max(0.0, float(node.retained_gain)))
            if subsystem:
                retained_subsystems[subsystem] = max(
                    float(retained_subsystems.get(subsystem, 0.0) or 0.0),
                    max(0.0, float(node.retained_gain)),
                )
            for family, value in dict(node.family_coverage).items():
                family_token = str(family).strip().lower()
                if not family_token or not value:
                    continue
                retained_family_gains[family_token] = max(
                    float(retained_family_gains.get(family_token, 0.0) or 0.0),
                    max(0.0, float(node.retained_gain)),
                )
        elif node.retention_state == "reject":
            rejected_count += 1
    recent_rejects = 0
    recent_retains = 0
    for node in recent_nodes:
        subsystem = str(node.subsystem).strip().lower()
        if node.retention_state == "reject":
            recent_rejects += 1
            if subsystem:
                recent_rejects_by_subsystem[subsystem] = int(recent_rejects_by_subsystem.get(subsystem, 0) or 0) + 1
            for family, value in dict(node.family_coverage).items():
                family_token = str(family).strip().lower()
                if not family_token or not value:
                    continue
                recent_rejects_by_family[family_token] = int(recent_rejects_by_family.get(family_token, 0) or 0) + 1
        elif node.retention_state == "retain":
            recent_retains += 1
    return {
        "node_count": len(nodes),
        "retained_count": retained_count,
        "rejected_count": rejected_count,
        "recent_rejects": recent_rejects,
        "recent_retains": recent_retains,
        "best_retained_gain": round(best_retained_gain, 4),
        "retained_family_gains": dict(sorted(retained_family_gains.items())),
        "recent_rejects_by_family": dict(sorted(recent_rejects_by_family.items())),
        "retained_subsystems": dict(sorted(retained_subsystems.items())),
        "recent_rejects_by_subsystem": dict(sorted(recent_rejects_by_subsystem.items())),
    }


def _update_strategy_memory_priors(
    controller_state: Mapping[str, object] | None,
    *,
    config: KernelConfig,
) -> dict[str, object]:
    state = dict(controller_state) if isinstance(controller_state, Mapping) else {}
    state["strategy_memory_priors"] = _summarize_strategy_memory_priors(config)
    return state


def _strategy_memory_candidate_score_adjustment(
    *,
    policy: Mapping[str, object] | None,
    current_policy: Mapping[str, object] | None,
    planner_pressure_signal: Mapping[str, object] | None,
    strategy_memory_priors: Mapping[str, object] | None,
) -> dict[str, object]:
    proposal = policy if isinstance(policy, Mapping) else {}
    current = current_policy if isinstance(current_policy, Mapping) else {}
    pressure = planner_pressure_signal if isinstance(planner_pressure_signal, Mapping) else {}
    priors = strategy_memory_priors if isinstance(strategy_memory_priors, Mapping) else {}
    retained_family_gains = (
        dict(priors.get("retained_family_gains", {}))
        if isinstance(priors.get("retained_family_gains", {}), Mapping)
        else {}
    )
    rejects_by_family = (
        dict(priors.get("recent_rejects_by_family", {}))
        if isinstance(priors.get("recent_rejects_by_family", {}), Mapping)
        else {}
    )
    relevant_families = _normalize_benchmark_families(
        pressure.get("priority_families", [])
        or pressure.get("missing_required_families", [])
        or pressure.get("frontier_repo_setting_families", [])
    )
    if not relevant_families:
        return {
            "score_adjustment": 0.0,
            "policy_change_units": 0.0,
            "retained_alignment_gain": 0.0,
            "reject_alignment_pressure": 0.0,
            "reasons": [],
        }
    retained_alignment_gain = sum(
        max(0.0, float(retained_family_gains.get(family, 0.0) or 0.0))
        for family in relevant_families
    )
    reject_alignment_pressure = sum(
        max(0, int(rejects_by_family.get(family, 0) or 0))
        for family in relevant_families
    )
    policy_change_units = 0.0
    if bool(proposal.get("adaptive_search", False)) != bool(current.get("adaptive_search", False)):
        policy_change_units += 1.0
    if max(1, int(proposal.get("campaign_width", 1) or 1)) > max(1, int(current.get("campaign_width", 1) or 1)):
        policy_change_units += 1.0
    if max(1, int(proposal.get("task_step_floor", 1) or 1)) > max(1, int(current.get("task_step_floor", 1) or 1)):
        policy_change_units += 1.0
    score_adjustment = 0.0
    reasons: list[str] = []
    if retained_alignment_gain > 0.0:
        score_adjustment += min(
            0.28,
            (0.08 * retained_alignment_gain) + (0.04 if policy_change_units > 0.0 else 0.0),
        )
        reasons.append(f"retained_family_gain_alignment={retained_alignment_gain:.4f}")
    if reject_alignment_pressure > 0.0:
        if policy_change_units <= 0.0:
            score_adjustment -= min(0.3, 0.07 * reject_alignment_pressure)
            reasons.append(f"reject_only_replay_penalty={reject_alignment_pressure:.4f}")
        else:
            score_adjustment += min(0.16, 0.04 * reject_alignment_pressure)
            reasons.append(f"reject_escape_bonus={reject_alignment_pressure:.4f}")
    return {
        "score_adjustment": round(score_adjustment, 4),
        "policy_change_units": round(policy_change_units, 4),
        "retained_alignment_gain": round(retained_alignment_gain, 4),
        "reject_alignment_pressure": round(reject_alignment_pressure, 4),
        "reasons": reasons,
    }


def _controller_observation(
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None,
    controller_state: Mapping[str, object] | None = None,
    curriculum_controls: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    campaign_signal = _campaign_signal(campaign_report or {})
    subsystem_signal = _subsystem_signal(campaign_report or {})
    planner_pressure_signal = _planner_pressure_signal(
        campaign_report or {},
        curriculum_controls=curriculum_controls,
    )
    round_signal = _mid_round_round_signal(round_payload)
    controller_features = _derived_controller_features(
        campaign_report=campaign_report or {},
        campaign_signal=campaign_signal,
        planner_pressure_signal=planner_pressure_signal,
        round_signal=round_signal,
        current_policy=(
            dict(round_payload.get("policy", {}))
            if isinstance(round_payload, Mapping) and isinstance(round_payload.get("policy", {}), Mapping)
            else {}
        ),
    )
    controller_features.update(
        _strategy_memory_observation_features(
            controller_state=controller_state,
            planner_pressure_signal=planner_pressure_signal,
        )
    )
    if controller_features:
        campaign_signal["controller_features"] = controller_features
    return build_round_observation(
        campaign_signal=campaign_signal,
        subsystem_signal=subsystem_signal,
        planner_pressure_signal=planner_pressure_signal,
        liftoff_signal=_liftoff_signal(liftoff_payload or {}) if liftoff_payload else None,
        round_signal=round_signal,
    )


def _candidate_round_policies(
    current_policy: dict[str, object],
    *,
    campaign_report: dict[str, object],
    liftoff_payload: dict[str, object] | None,
    curriculum_controls: dict[str, object] | None = None,
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
    max_task_step_floor: int,
) -> list[dict[str, object]]:
    base = _advance_subsystem_cooldowns(current_policy)
    signal = _campaign_signal(campaign_report)
    liftoff = _liftoff_signal(liftoff_payload or {}) if liftoff_payload else {}
    planner_pressure = _planner_pressure_signal(
        campaign_report,
        curriculum_controls=curriculum_controls,
    )
    subsystem_signal = _subsystem_signal(campaign_report)
    subsystem_monoculture = _subsystem_monoculture_signal(
        campaign_report,
        subsystem_signal=subsystem_signal,
        planner_pressure_signal=planner_pressure,
    )
    severe_regression = (
        float(signal.get("worst_family_delta", 0.0) or 0.0) < 0.0
        or float(signal.get("worst_generated_family_delta", 0.0) or 0.0) < 0.0
        or int(signal.get("max_regressed_families", 0) or 0) > 0
        or int(signal.get("max_generated_regressed_families", 0) or 0) > 0
        or int(signal.get("failed_decisions", 0) or 0) > 0
    )
    low_confidence_pressure = (
        int(signal.get("max_low_confidence_episode_delta", 0) or 0) > 0
        or int(signal.get("min_trusted_retrieval_step_delta", 0) or 0) < 0
    )
    retained_cycles = int(signal.get("retained_cycles", 0) or 0)
    current_cycles = max(1, int(base.get("cycles", 1) or 1))
    current_task_limit = max(1, int(base.get("task_limit", 64) or 64))
    current_task_step_floor = max(1, int(base.get("task_step_floor", 1) or 1))
    current_campaign_width = max(1, int(base.get("campaign_width", 1) or 1))
    current_variant_width = max(1, int(base.get("variant_width", 1) or 1))
    current_priority_families = _normalize_benchmark_families(
        current_policy.get("priority_benchmark_families", [])
    )
    repo_setting_priority_families = _repo_setting_focus_priority_families(
        current_priority_families=current_priority_families,
        frontier_repo_setting_families=planner_pressure.get("frontier_repo_setting_families", []),
        missing_required_families=signal.get("missing_required_families", []),
    )
    repo_setting_policy_pressure = _repo_setting_policy_pressure(
        frontier_repo_setting_priority_pairs=planner_pressure.get("frontier_repo_setting_priority_pairs", []),
        priority_families=repo_setting_priority_families,
    )
    repo_setting_campaign_width_pressure = bool(repo_setting_policy_pressure.get("campaign_width_signals", []))
    repo_setting_task_step_floor_pressure = bool(repo_setting_policy_pressure.get("task_step_floor_signals", []))
    repo_setting_adaptive_search_pressure = bool(repo_setting_policy_pressure.get("adaptive_search_signals", []))
    depth_signal_active = bool(
        int(signal.get("productive_depth_retained_cycles", 0) or 0) > 0
        or int(signal.get("depth_drift_cycles", 0) or 0) > 0
        or int(signal.get("long_horizon_retained_cycles", 0) or 0) > 0
        or repo_setting_task_step_floor_pressure
    )
    focuses = ["balanced", "recovery_alignment", "discovered_task_adaptation"]
    adaptive_options = [False, True]
    if severe_regression:
        focuses = ["recovery_alignment", "balanced"]
        adaptive_options = [True]
    elif bool(subsystem_monoculture.get("active", False)):
        focuses = ["discovered_task_adaptation", "recovery_alignment", "balanced"]
        adaptive_options = [True]
    elif low_confidence_pressure or retained_cycles <= 0:
        focuses = ["discovered_task_adaptation", "balanced"]
        adaptive_options = [True]
    elif repo_setting_adaptive_search_pressure:
        adaptive_options = [True]
        if "discovered_task_adaptation" not in focuses:
            focuses = ["discovered_task_adaptation", *focuses]
    if liftoff and str(liftoff.get("state", "")).strip() == "reject":
        adaptive_options = [True]
    cycle_options = sorted(
        {
            max(1, min(max_cycles, current_cycles - 1)),
            max(1, min(max_cycles, current_cycles)),
            max(1, min(max_cycles, current_cycles + 1)),
        }
    )
    task_limit_options = sorted(
        {
            max(1, min(max_task_limit, current_task_limit)),
            max(1, min(max_task_limit, max(1, current_task_limit // 2))),
            max(1, min(max_task_limit, current_task_limit * 2)),
        }
    )
    if bool(subsystem_monoculture.get("active", False)):
        task_limit_options = sorted(
            {
                max(1, min(max_task_limit, current_task_limit)),
                max(1, min(max_task_limit, current_task_limit * 2)),
                max(1, min(max_task_limit, current_task_limit * (4 if bool(subsystem_monoculture.get("severe", False)) else 2))),
            }
        )
    if depth_signal_active:
        task_step_floor_options = sorted(
            {
                max(1, min(max_task_step_floor, current_task_step_floor)),
                max(1, min(max_task_step_floor, max(1, current_task_step_floor // 2))),
                max(1, min(max_task_step_floor, current_task_step_floor * 2)),
                max(1, min(max_task_step_floor, int(math.ceil(float(current_task_step_floor) * 1.5))))
                if repo_setting_task_step_floor_pressure
                else max(1, min(max_task_step_floor, current_task_step_floor * 2)),
            }
        )
    else:
        task_step_floor_options = [current_task_step_floor]
    campaign_width_options = sorted(
        {
            max(1, min(max_campaign_width, current_campaign_width)),
            max(1, min(max_campaign_width, current_campaign_width + 1)),
            max(1, min(max_campaign_width, current_campaign_width + 2))
            if repo_setting_campaign_width_pressure
            else max(1, min(max_campaign_width, current_campaign_width + 1)),
        }
    )
    variant_width_options = sorted(
        {
            max(1, min(max_variant_width, current_variant_width)),
            max(1, min(max_variant_width, current_variant_width + 1)),
        }
    )
    if bool(subsystem_monoculture.get("active", False)):
        campaign_width_options = sorted(
            {
                max(1, min(max_campaign_width, current_campaign_width + 1)),
                max(1, min(max_campaign_width, current_campaign_width + 2)),
            }
        )
        variant_width_options = sorted(
            {
                max(1, min(max_variant_width, current_variant_width + 1)),
                max(1, min(max_variant_width, current_variant_width + (2 if bool(subsystem_monoculture.get("severe", False)) else 1))),
            }
        )
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for focus in focuses:
        for adaptive in adaptive_options:
            for cycles in cycle_options:
                for task_limit in task_limit_options:
                    for task_step_floor in task_step_floor_options:
                        for campaign_width in campaign_width_options:
                            for variant_width in variant_width_options:
                                proposal = dict(base)
                                proposal["focus"] = focus
                                proposal["adaptive_search"] = adaptive
                                proposal["cycles"] = cycles
                                proposal["task_limit"] = task_limit
                                proposal["task_step_floor"] = task_step_floor
                                proposal["campaign_width"] = campaign_width
                                proposal["variant_width"] = variant_width
                                if focus == "recovery_alignment":
                                    proposal["cycles"] = min(max_cycles, max(proposal["cycles"], current_cycles))
                                    proposal["adaptive_search"] = True
                                if focus == "discovered_task_adaptation":
                                    proposal["task_limit"] = min(max_task_limit, max(proposal["task_limit"], current_task_limit))
                                    proposal["task_step_floor"] = min(
                                        max_task_step_floor,
                                        max(proposal["task_step_floor"], current_task_step_floor),
                                    )
                                    proposal["variant_width"] = min(
                                        max_variant_width,
                                        max(proposal["variant_width"], current_variant_width),
                                    )
                                if liftoff and not bool(liftoff.get("allow_kernel_autobuild", False)):
                                    proposal["focus"] = "discovered_task_adaptation"
                                    proposal["task_limit"] = min(
                                        max_task_limit,
                                        max(proposal["task_limit"], current_task_limit),
                                    )
                                proposal = _materialize_excluded_subsystems(proposal)
                                fingerprint = json.dumps(
                                    {
                                        "focus": proposal["focus"],
                                        "adaptive_search": bool(proposal["adaptive_search"]),
                                        "cycles": int(proposal["cycles"]),
                                        "campaign_width": int(proposal["campaign_width"]),
                                        "variant_width": int(proposal["variant_width"]),
                                        "task_limit": int(proposal["task_limit"]),
                                        "task_step_floor": int(proposal["task_step_floor"]),
                                        "excluded_subsystems": list(proposal.get("excluded_subsystems", [])),
                                    },
                                    sort_keys=True,
                                )
                                if fingerprint in seen:
                                    continue
                                seen.add(fingerprint)
                                candidates.append(proposal)
    if not candidates:
        candidates.append(_materialize_excluded_subsystems(base))
    return candidates


def _validate_campaign_report(campaign_report: dict[str, object]) -> dict[str, object]:
    signal = _campaign_signal(campaign_report)
    if signal["report_kind"] != "improvement_campaign_report":
        return {"passed": False, "detail": "campaign report kind was missing or invalid", "signal": signal}
    failure_context = _campaign_failure_context(signal)
    requested = max(0, int(signal["cycles_requested"]))
    completed = max(0, int(signal["completed_runs"]))
    successful = max(0, int(signal["successful_runs"]))
    if requested > 0 and completed < requested:
        detail = "campaign report shows incomplete cycle execution"
        summary = _campaign_failure_context_summary(failure_context)
        if summary:
            detail = f"{detail}: {summary}"
        return {"passed": False, "detail": detail, "signal": signal, "failure_context": failure_context}
    if completed > 0 and successful < completed:
        detail = "campaign report shows failed child cycles"
        summary = _campaign_failure_context_summary(failure_context)
        if summary:
            detail = f"{detail}: {summary}"
        return {"passed": False, "detail": detail, "signal": signal, "failure_context": failure_context}
    if completed > 0 and int(signal["runtime_managed_decisions"]) <= 0:
        if bool(signal.get("intermediate_decision_evidence", False)):
            return {
                "passed": True,
                "detail": "campaign report is complete with intermediate decision evidence",
                "signal": signal,
            }
        if bool(signal.get("task_success_evidence", False)):
            return {
                "passed": True,
                "detail": "campaign report is complete with verified task-success evidence",
                "signal": signal,
            }
        detail = "campaign report showed no runtime-managed decisions"
        summary = _campaign_failure_context_summary(failure_context)
        if summary:
            detail = f"{detail}: {summary}"
        return {"passed": False, "detail": detail, "signal": signal, "failure_context": failure_context}
    return {"passed": True, "detail": "campaign report is complete", "signal": signal}


def _liftoff_signal(liftoff_payload: dict[str, object]) -> dict[str, object]:
    report = liftoff_payload.get("liftoff_report", {})
    if not isinstance(report, dict):
        report = {}
    dataset = liftoff_payload.get("dataset_readiness", {})
    if not isinstance(dataset, dict):
        dataset = {}
    return {
        "report_kind": str(liftoff_payload.get("report_kind", "")).strip(),
        "status": str(liftoff_payload.get("status", "")).strip(),
        "state": str(report.get("state", "")).strip(),
        "reason": str(report.get("reason", "")).strip(),
        "allow_kernel_autobuild": bool(dataset.get("allow_kernel_autobuild", False)),
        "total_examples": int(dataset.get("total_examples", 0) or 0),
        "synthetic_examples": int(dataset.get("synthetic_trajectory_examples", 0) or 0),
    }


def _round_stop_signal(
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_round = round_payload if isinstance(round_payload, dict) else {}
    resolved_campaign = campaign_report if isinstance(campaign_report, dict) else {}
    if not resolved_campaign and isinstance(resolved_round.get("campaign_report", {}), dict):
        resolved_campaign = dict(resolved_round.get("campaign_report", {}))
    resolved_liftoff = liftoff_payload if isinstance(liftoff_payload, dict) else {}
    if not resolved_liftoff and isinstance(resolved_round.get("liftoff_report", {}), dict):
        resolved_liftoff = dict(resolved_round.get("liftoff_report", {}))

    campaign_signal = _campaign_signal(resolved_campaign or {})
    liftoff_signal = _liftoff_signal(resolved_liftoff) if isinstance(resolved_liftoff, dict) else {}
    rationale = resolved_round.get("policy_shift_rationale", {})
    reason_codes = set()
    if isinstance(rationale, dict):
        raw_codes = rationale.get("reason_codes", [])
        if isinstance(raw_codes, list):
            reason_codes = {
                str(code).strip()
                for code in raw_codes
                if str(code).strip()
            }
    planner_pressure = rationale.get("planner_pressure", {}) if isinstance(rationale, dict) else {}
    if not isinstance(planner_pressure, dict):
        planner_pressure = {}
    retained_cycles = int(campaign_signal.get("retained_cycles", 0) or 0)
    productive_depth_retained_cycles = max(
        int(campaign_signal.get("productive_depth_retained_cycles", 0) or 0),
        int(planner_pressure.get("productive_depth_retained_cycles", 0) or 0),
    )
    long_horizon_retained_cycles = max(
        int(campaign_signal.get("long_horizon_retained_cycles", 0) or 0),
        int(planner_pressure.get("long_horizon_retained_cycles", 0) or 0),
    )
    depth_drift_cycles = max(
        int(campaign_signal.get("depth_drift_cycles", 0) or 0),
        int(planner_pressure.get("depth_drift_cycles", 0) or 0),
    )
    average_retained_step_delta = float(campaign_signal.get("average_retained_step_delta", 0.0) or 0.0)
    average_productive_depth_step_delta = max(
        float(campaign_signal.get("average_productive_depth_step_delta", 0.0) or 0.0),
        float(planner_pressure.get("average_productive_depth_step_delta", 0.0) or 0.0),
    )
    average_retained_pass_rate_delta = float(campaign_signal.get("average_retained_pass_rate_delta", 0.0) or 0.0)
    retained_phase_gates_passed = bool(campaign_signal.get("all_retained_phase_gates_passed", True))
    failed_decisions = int(campaign_signal.get("failed_decisions", 0) or 0)
    productive_depth_continuation = (
        retained_phase_gates_passed
        and failed_decisions <= 0
        and depth_drift_cycles <= 0
        and average_retained_pass_rate_delta >= 0.0
        and (
            productive_depth_retained_cycles > 0
            or (
                long_horizon_retained_cycles > 0
                and retained_cycles > 0
                and max(average_retained_step_delta, average_productive_depth_step_delta) > 0.0
            )
            or (
                "productive_depth_gain" in reason_codes
                and "depth_drift_pressure" not in reason_codes
                and max(average_retained_step_delta, average_productive_depth_step_delta) > 0.0
            )
        )
    )
    liftoff_state = str(liftoff_signal.get("state", "")).strip()
    retained_or_liftoff_yield = (
        retained_cycles > 0
        or liftoff_state == "retain"
        or bool(campaign_signal.get("task_success_evidence", False))
    )
    return {
        "retained_cycles": retained_cycles,
        "productive_depth_continuation": productive_depth_continuation,
        "yield_resets_no_yield": retained_or_liftoff_yield or productive_depth_continuation,
        "continuation_resets_policy_stall": productive_depth_continuation or liftoff_state == "retain",
        "depth_drift_cycles": depth_drift_cycles,
    }


_MAX_DEPTH_RUNWAY_CREDIT = 4
_MAX_NO_YIELD_DEPTH_RUNWAY_BONUS = 3
_MAX_POLICY_STALL_DEPTH_RUNWAY_BONUS = 2
_HIGH_CONFIDENCE_PRODUCTIVE_STEP_DELTA = 6.0


def _depth_runway_signal(
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_round = round_payload if isinstance(round_payload, dict) else {}
    stop_signal = _round_stop_signal(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=resolved_round,
    )
    campaign_signal = _campaign_signal(campaign_report if isinstance(campaign_report, dict) else {})
    rationale = resolved_round.get("policy_shift_rationale", {})
    planner_pressure = rationale.get("planner_pressure", {}) if isinstance(rationale, dict) else {}
    if not isinstance(planner_pressure, dict):
        planner_pressure = {}
    productive_depth_retained_cycles = max(
        int(campaign_signal.get("productive_depth_retained_cycles", 0) or 0),
        int(planner_pressure.get("productive_depth_retained_cycles", 0) or 0),
    )
    long_horizon_retained_cycles = max(
        int(campaign_signal.get("long_horizon_retained_cycles", 0) or 0),
        int(planner_pressure.get("long_horizon_retained_cycles", 0) or 0),
    )
    average_productive_depth_step_delta = max(
        float(campaign_signal.get("average_productive_depth_step_delta", 0.0) or 0.0),
        float(planner_pressure.get("average_productive_depth_step_delta", 0.0) or 0.0),
    )
    depth_drift_cycles = max(
        int(campaign_signal.get("depth_drift_cycles", 0) or 0),
        int(planner_pressure.get("depth_drift_cycles", 0) or 0),
    )
    return {
        "productive_depth_continuation": bool(stop_signal.get("productive_depth_continuation", False)),
        "retained_cycles": int(stop_signal.get("retained_cycles", 0) or 0),
        "depth_drift_cycles": depth_drift_cycles,
        "productive_depth_retained_cycles": productive_depth_retained_cycles,
        "long_horizon_retained_cycles": long_horizon_retained_cycles,
        "average_productive_depth_step_delta": average_productive_depth_step_delta,
        "all_retained_phase_gates_passed": bool(campaign_signal.get("all_retained_phase_gates_passed", True)),
        "failed_decisions": int(campaign_signal.get("failed_decisions", 0) or 0),
    }


def _next_depth_runway_credit(
    current_credit: int,
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> int:
    signal = _depth_runway_signal(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    credit = max(0, int(current_credit))
    if (
        int(signal.get("depth_drift_cycles", 0) or 0) > 0
        or int(signal.get("failed_decisions", 0) or 0) > 0
        or not bool(signal.get("all_retained_phase_gates_passed", True))
    ):
        return 0
    if bool(signal.get("productive_depth_continuation", False)):
        gain = 1
        if int(signal.get("long_horizon_retained_cycles", 0) or 0) > 0:
            gain += 1
        if float(signal.get("average_productive_depth_step_delta", 0.0) or 0.0) >= _HIGH_CONFIDENCE_PRODUCTIVE_STEP_DELTA:
            gain += 1
        return min(_MAX_DEPTH_RUNWAY_CREDIT, credit + gain)
    if int(signal.get("retained_cycles", 0) or 0) > 0:
        return credit
    return max(0, credit - 1)


def _adaptive_stop_budget(
    *,
    base_max_no_yield_rounds: int,
    base_max_policy_stall_rounds: int,
    depth_runway_credit: int,
) -> dict[str, int]:
    credit = max(0, min(_MAX_DEPTH_RUNWAY_CREDIT, int(depth_runway_credit)))
    return {
        "depth_runway_credit": credit,
        "max_no_yield_rounds_base": max(0, int(base_max_no_yield_rounds)),
        "max_policy_stall_rounds_base": max(0, int(base_max_policy_stall_rounds)),
        "max_no_yield_rounds_effective": max(0, int(base_max_no_yield_rounds))
        + min(_MAX_NO_YIELD_DEPTH_RUNWAY_BONUS, credit),
        "max_policy_stall_rounds_effective": max(0, int(base_max_policy_stall_rounds))
        + min(_MAX_POLICY_STALL_DEPTH_RUNWAY_BONUS, credit),
    }


def _next_no_yield_rounds(
    current_rounds: int,
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> int:
    stop_signal = _round_stop_signal(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    if bool(stop_signal.get("yield_resets_no_yield", False)):
        return 0
    return max(0, int(current_rounds)) + 1


def _next_policy_stall_rounds(
    current_rounds: int,
    *,
    current_policy: dict[str, object],
    next_policy: dict[str, object],
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None = None,
    round_payload: dict[str, object] | None = None,
) -> int:
    if dict(current_policy) != dict(next_policy):
        return 0
    stop_signal = _round_stop_signal(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        round_payload=round_payload,
    )
    if bool(stop_signal.get("continuation_resets_policy_stall", False)):
        return 0
    return max(0, int(current_rounds)) + 1


def _validate_liftoff_report(liftoff_payload: dict[str, object]) -> dict[str, object]:
    signal = _liftoff_signal(liftoff_payload)
    if signal["report_kind"] != "tolbert_liftoff_loop_report":
        return {"passed": False, "detail": "liftoff report kind was missing or invalid", "signal": signal}
    if signal["status"] and signal["status"] != "completed":
        return {"passed": False, "detail": "liftoff report did not complete successfully", "signal": signal}
    if signal["state"] not in {"retain", "shadow_only", "reject"}:
        return {"passed": False, "detail": "liftoff report state was missing or invalid", "signal": signal}
    return {"passed": True, "detail": "liftoff report is complete", "signal": signal}


def _next_round_policy(
    current_policy: dict[str, object],
    *,
    campaign_report: dict[str, object],
    liftoff_payload: dict[str, object] | None,
    round_payload: dict[str, object] | None,
    controller_state: dict[str, object] | None = None,
    prior_rounds: list[dict[str, object]] | None = None,
    planner_controls: dict[str, object] | None = None,
    curriculum_controls: dict[str, object] | None = None,
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
    max_task_step_floor: int = 4096,
) -> dict[str, object]:
    observation = _controller_observation(
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        controller_state=controller_state,
        curriculum_controls=curriculum_controls,
        round_payload=round_payload,
    )
    candidate_policies = _candidate_round_policies(
        current_policy,
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        curriculum_controls=curriculum_controls,
        max_cycles=max_cycles,
        max_task_limit=max_task_limit,
        max_campaign_width=max_campaign_width,
        max_variant_width=max_variant_width,
        max_task_step_floor=max_task_step_floor,
    )
    next_policy, planner_diagnostics = plan_next_policy(
        controller_state,
        current_observation=observation,
        candidate_policies=candidate_policies,
    )
    historical_adjustments = _historical_policy_rationale_adjustments(list(prior_rounds or []))
    candidate_repo_setting_policy_pressure = _repo_setting_policy_pressure(
        frontier_repo_setting_priority_pairs=_planner_pressure_signal(
            campaign_report,
            curriculum_controls=curriculum_controls,
        ).get("frontier_repo_setting_priority_pairs", []),
        priority_families=_normalize_benchmark_families(current_policy.get("priority_benchmark_families", [])),
    )
    learned_repo_setting_priors = _learn_repo_setting_policy_priors(
        prior_rounds=list(prior_rounds or []),
        repo_setting_policy_pressure=candidate_repo_setting_policy_pressure,
        persisted_priors=(
            dict(controller_state.get("repo_setting_policy_priors", {}))
            if isinstance(controller_state, Mapping)
            and isinstance(controller_state.get("repo_setting_policy_priors", {}), Mapping)
            else {}
        ),
    )
    strategy_memory_priors = (
        dict(controller_state.get("strategy_memory_priors", {}))
        if isinstance(controller_state, Mapping)
        and isinstance(controller_state.get("strategy_memory_priors", {}), Mapping)
        else {}
    )
    adjusted_candidates: list[dict[str, object]] = []
    for candidate in planner_diagnostics.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        policy = candidate.get("policy", {})
        action_key = action_key_for_policy(policy if isinstance(policy, dict) else {})
        rationale_adjustment = float(historical_adjustments.get(action_key, 0.0) or 0.0)
        repo_setting_score_adjustment = _repo_setting_candidate_score_adjustment(
            policy=policy if isinstance(policy, dict) else {},
            current_policy=current_policy,
            repo_setting_policy_pressure=candidate_repo_setting_policy_pressure,
            learned_priors=learned_repo_setting_priors,
        )
        strategy_memory_score_adjustment = _strategy_memory_candidate_score_adjustment(
            policy=policy if isinstance(policy, dict) else {},
            current_policy=current_policy,
            planner_pressure_signal=_planner_pressure_signal(
                campaign_report,
                curriculum_controls=curriculum_controls,
            ),
            strategy_memory_priors=strategy_memory_priors,
        )
        adjusted_score = (
            float(candidate.get("score", 0.0) or 0.0)
            + rationale_adjustment
            + float(repo_setting_score_adjustment.get("score_adjustment", 0.0) or 0.0)
            + float(strategy_memory_score_adjustment.get("score_adjustment", 0.0) or 0.0)
        )
        adjusted_candidate = dict(candidate)
        adjusted_candidate["historical_rationale_adjustment"] = rationale_adjustment
        adjusted_candidate["repo_setting_score_adjustment"] = float(
            repo_setting_score_adjustment.get("score_adjustment", 0.0) or 0.0
        )
        adjusted_candidate["repo_setting_score_rationale"] = repo_setting_score_adjustment
        adjusted_candidate["strategy_memory_score_adjustment"] = float(
            strategy_memory_score_adjustment.get("score_adjustment", 0.0) or 0.0
        )
        adjusted_candidate["strategy_memory_score_rationale"] = strategy_memory_score_adjustment
        adjusted_candidate["adjusted_score"] = adjusted_score
        adjusted_candidates.append(adjusted_candidate)
    if adjusted_candidates:
        adjusted_candidates.sort(key=lambda item: float(item.get("adjusted_score", float("-inf"))), reverse=True)
        top_candidate = adjusted_candidates[0]
        top_policy = top_candidate.get("policy", {})
        if isinstance(top_policy, dict):
            next_policy = dict(top_policy)
            planner_diagnostics["selected_action_key"] = str(top_candidate.get("action_key", "")).strip()
            planner_diagnostics["selected_score"] = float(top_candidate.get("score", 0.0) or 0.0)
            planner_diagnostics["selected_adjusted_score"] = float(top_candidate.get("adjusted_score", 0.0) or 0.0)
            planner_diagnostics["selected_historical_rationale_adjustment"] = float(
                top_candidate.get("historical_rationale_adjustment", 0.0) or 0.0
            )
            planner_diagnostics["selected_repo_setting_score_adjustment"] = float(
                top_candidate.get("repo_setting_score_adjustment", 0.0) or 0.0
            )
            planner_diagnostics["selected_repo_setting_score_rationale"] = dict(
                top_candidate.get("repo_setting_score_rationale", {})
            )
            planner_diagnostics["selected_strategy_memory_score_adjustment"] = float(
                top_candidate.get("strategy_memory_score_adjustment", 0.0) or 0.0
            )
            planner_diagnostics["selected_strategy_memory_score_rationale"] = dict(
                top_candidate.get("strategy_memory_score_rationale", {})
            )
            planner_diagnostics["learned_repo_setting_priors"] = dict(learned_repo_setting_priors)
            planner_diagnostics["strategy_memory_priors"] = dict(strategy_memory_priors)
            planner_diagnostics["candidates"] = adjusted_candidates[:12]
    if isinstance(round_payload, dict):
        round_payload["controller_planner"] = planner_diagnostics
    signal = _campaign_signal(campaign_report)
    subsystem_signal = _subsystem_signal(campaign_report)
    planner_pressure_signal = _planner_pressure_signal(
        campaign_report,
        curriculum_controls=curriculum_controls,
    )
    subsystem_monoculture = _subsystem_monoculture_signal(
        campaign_report,
        subsystem_signal=subsystem_signal,
        planner_pressure_signal=planner_pressure_signal,
    )
    retained_cycles = int(signal["retained_cycles"])
    rejected_cycles = int(signal["rejected_cycles"])
    runtime_managed_decisions = int(signal["runtime_managed_decisions"])
    intermediate_decision_evidence = bool(signal.get("intermediate_decision_evidence", False))
    avg_retained_pass_delta = float(signal["average_retained_pass_rate_delta"])
    avg_retained_step_delta = float(signal["average_retained_step_delta"])
    productive_depth_retained_cycles = int(signal.get("productive_depth_retained_cycles", 0) or 0)
    average_productive_depth_step_delta = float(signal.get("average_productive_depth_step_delta", 0.0) or 0.0)
    depth_drift_cycles = int(signal.get("depth_drift_cycles", 0) or 0)
    average_depth_drift_step_delta = float(signal.get("average_depth_drift_step_delta", 0.0) or 0.0)
    long_horizon_retained_cycles = int(signal.get("long_horizon_retained_cycles", 0) or 0)
    retained_phase_gates_passed = bool(signal["all_retained_phase_gates_passed"])
    failed_decisions = int(signal["failed_decisions"])
    worst_family_delta = float(signal["worst_family_delta"])
    worst_generated_family_delta = float(signal["worst_generated_family_delta"])
    max_regressed_families = int(signal["max_regressed_families"])
    max_generated_regressed_families = int(signal["max_generated_regressed_families"])
    max_low_confidence_episode_delta = int(signal["max_low_confidence_episode_delta"])
    min_trusted_retrieval_step_delta = int(signal["min_trusted_retrieval_step_delta"])
    campaign_breadth_pressure_cycles = int(signal["campaign_breadth_pressure_cycles"])
    variant_breadth_pressure_cycles = int(signal["variant_breadth_pressure_cycles"])
    external_report_count = int(signal["external_report_count"])
    distinct_external_benchmark_families = int(signal["distinct_external_benchmark_families"])
    trust_breadth_summary_present = bool(
        isinstance(campaign_report.get("trust_breadth_summary", {}), dict)
        and campaign_report.get("trust_breadth_summary", {})
    )
    required_families = _normalize_benchmark_families(signal.get("required_families", []))
    missing_required_families = _normalize_benchmark_families(signal.get("missing_required_families", []))
    distinct_family_gap = int(signal.get("distinct_family_gap", 0) or 0)
    required_family_coverage_gap = (
        trust_breadth_summary_present and (bool(missing_required_families) or distinct_family_gap > 0)
    )
    priority_families = _normalize_benchmark_families(
        signal.get("priority_families", current_policy.get("priority_benchmark_families", []))
    )
    priority_family_history = _priority_family_history_summary(
        prior_rounds=list(prior_rounds or []),
        current_campaign_report=campaign_report,
        current_priority_families=priority_families,
        planner_controls=planner_controls,
        curriculum_controls=curriculum_controls,
    )
    priority_families_with_retained_gain = _normalize_benchmark_families(
        priority_family_history.get("priority_families_with_retained_gain", [])
    )
    priority_families_without_signal = _normalize_benchmark_families(
        priority_family_history.get("priority_families_without_signal", [])
    )
    priority_families_needing_retained_gain_conversion = _normalize_benchmark_families(
        priority_family_history.get("priority_families_needing_retained_gain_conversion", [])
    )
    priority_families_under_sampled = _normalize_benchmark_families(
        priority_family_history.get("priority_families_under_sampled", [])
    )
    productive_partial_conversion = _productive_partial_conversion_signal(
        round_payload,
        priority_families=priority_families,
    )
    productive_partial_conversion_gap = (
        runtime_managed_decisions <= 0
        and not intermediate_decision_evidence
        and retained_cycles <= 0
        and bool(productive_partial_conversion.get("conversion_gap", False))
    )
    round_payload_for_signal = round_payload
    if isinstance(round_payload, Mapping):
        round_payload_for_signal = dict(round_payload)
        if not isinstance(round_payload_for_signal.get("policy", {}), Mapping):
            round_payload_for_signal["policy"] = dict(current_policy)
    mid_round_signal = _mid_round_round_signal(round_payload_for_signal)
    broad_observe_then_retrieval_first = bool(mid_round_signal.get("broad_observe_then_retrieval_first", False))
    semantic_progress_drift = bool(mid_round_signal.get("semantic_progress_drift", False))
    micro_step_verification_loop = bool(mid_round_signal.get("micro_step_verification_loop", False))
    incomplete_cycle_count = int(signal.get("incomplete_cycle_count", 0) or 0)
    observed_primary_runs = int(signal.get("observed_primary_runs", 0) or 0)
    generated_success_completed_runs = int(signal.get("generated_success_completed_runs", 0) or 0)
    partial_productive_without_decision_runs = int(signal.get("partial_productive_without_decision_runs", 0) or 0)
    observability_gap = bool(
        semantic_progress_drift
        or micro_step_verification_loop
        or incomplete_cycle_count > 0
        or (
            partial_productive_without_decision_runs > 0
            and generated_success_completed_runs > 0
            and runtime_managed_decisions <= 0
        )
    )
    subsystem_robustness_gap = bool(
        micro_step_verification_loop
        or incomplete_cycle_count > 0
        or bool(signal.get("recent_failed_run", {}))
        or bool(signal.get("recent_failed_decision", {}))
        or (
            partial_productive_without_decision_runs > 0
            and observed_primary_runs > 0
            and runtime_managed_decisions <= 0
        )
    )
    priority_families_low_return = _normalize_benchmark_families(
        priority_family_history.get("priority_families_low_return", [])
    )
    priority_families_ranked_by_selection_score = _normalize_benchmark_families(
        priority_family_history.get("priority_families_ranked_by_selection_score", [])
    )
    priority_family_selection_scores = (
        dict(priority_family_history.get("priority_family_selection_scores", {}))
        if isinstance(priority_family_history.get("priority_family_selection_scores", {}), dict)
        else {}
    )
    priority_family_scoring_policy = (
        dict(priority_family_history.get("priority_family_scoring_policy", {}))
        if isinstance(priority_family_history.get("priority_family_scoring_policy", {}), dict)
        else {}
    )
    priority_family_summary_present = bool(
        isinstance(campaign_report.get("priority_family_yield_summary", {}), dict)
        and campaign_report.get("priority_family_yield_summary", {})
    ) or bool(
        isinstance(campaign_report.get("priority_family_allocation_summary", {}), dict)
        and campaign_report.get("priority_family_allocation_summary", {})
    )
    priority_family_under_sampled_gap = (
        priority_family_summary_present and bool(priority_families) and bool(priority_families_under_sampled)
    )
    retained_gain_conversion_gap = bool(priority_families_needing_retained_gain_conversion)
    repo_setting_priority_families = _repo_setting_focus_priority_families(
        current_priority_families=priority_families,
        ranked_priority_families=priority_families_ranked_by_selection_score,
        frontier_repo_setting_families=priority_family_history.get("frontier_repo_setting_families", []),
        missing_required_families=missing_required_families,
    )
    repo_setting_policy_pressure = _repo_setting_policy_pressure(
        frontier_repo_setting_priority_pairs=priority_family_history.get("frontier_repo_setting_priority_pairs", []),
        priority_families=repo_setting_priority_families,
    )
    current_priority_family_set = set(_normalize_benchmark_families(current_policy.get("priority_benchmark_families", [])))
    repo_setting_family_expansion = any(
        family not in current_priority_family_set
        for family in _normalize_benchmark_families(repo_setting_policy_pressure.get("focused_families", []))
    )
    repo_setting_campaign_width_pressure = bool(repo_setting_policy_pressure.get("campaign_width_signals", []))
    repo_setting_task_step_floor_pressure = bool(repo_setting_policy_pressure.get("task_step_floor_signals", []))
    repo_setting_adaptive_search_pressure = bool(repo_setting_policy_pressure.get("adaptive_search_signals", []))
    productive_priority_family_set = set(priority_families_with_retained_gain)
    coding_frontier_expansion = (
        retained_cycles > 0
        and retained_phase_gates_passed
        and failed_decisions <= 0
        and max_regressed_families <= 0
        and max_generated_regressed_families <= 0
        and worst_family_delta >= 0.0
        and worst_generated_family_delta >= 0.0
        and productive_depth_retained_cycles > 0
        and average_productive_depth_step_delta > 0.0
        and {"project", "repository"}.issubset(productive_priority_family_set)
    )
    coding_frontier_family_expansion = coding_frontier_expansion and "integration" not in current_priority_family_set
    current_cycles = max(1, int(current_policy.get("cycles", 1) or 1))
    current_task_limit = max(0, int(current_policy.get("task_limit", 0) or 0))
    current_task_step_floor = max(1, int(current_policy.get("task_step_floor", 1) or 1))
    current_campaign_width = max(1, int(current_policy.get("campaign_width", 1) or 1))
    current_variant_width = max(1, int(current_policy.get("variant_width", 1) or 1))
    stalled_subsystem = ""
    if isinstance(round_payload, dict):
        active_child = round_payload.get("active_child", {})
        if isinstance(active_child, dict):
            stalled_subsystem = _subsystem_from_progress_line(
                str(active_child.get("last_progress_line") or active_child.get("last_output_line") or "")
            )
        if not stalled_subsystem:
            stalled_subsystem = _subsystem_from_progress_line(str(round_payload.get("phase_detail", "")))
    if not stalled_subsystem:
        stalled_subsystem = str(planner_pressure_signal.get("dominant_subsystem", "")).strip()
    if not stalled_subsystem and bool(subsystem_monoculture.get("active", False)):
        stalled_subsystem = str(subsystem_monoculture.get("dominant_subsystem", "")).strip()
    if not stalled_subsystem and micro_step_verification_loop:
        stalled_subsystem = str(mid_round_signal.get("selected_subsystem", "")).strip()

    # Hard safety envelopes remain, but they only constrain the learned planner.
    if (
        max_regressed_families > 0
        or max_generated_regressed_families > 0
        or worst_family_delta < 0.0
        or worst_generated_family_delta < 0.0
        or failed_decisions > 0
        or not retained_phase_gates_passed
    ):
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "recovery_alignment"
        next_policy["cycles"] = min(
            max_cycles,
            max(current_cycles + 1, int(next_policy.get("cycles", current_cycles) or current_cycles)),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(current_task_limit * 2, int(next_policy.get("task_limit", current_task_limit) or current_task_limit)),
        )
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(current_campaign_width + 1, int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width)),
        )
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
    elif max_low_confidence_episode_delta > 0 or min_trusted_retrieval_step_delta < 0:
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "discovered_task_adaptation"
        next_policy["cycles"] = min(
            max_cycles,
            max(current_cycles + 1, int(next_policy.get("cycles", current_cycles) or current_cycles)),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(current_task_limit * 2, int(next_policy.get("task_limit", current_task_limit) or current_task_limit)),
        )
        next_policy["variant_width"] = min(
            max_variant_width,
            max(current_variant_width + 1, int(next_policy.get("variant_width", current_variant_width) or current_variant_width)),
        )
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
    elif (
        campaign_breadth_pressure_cycles > 0
        or variant_breadth_pressure_cycles > 0
        or retained_cycles <= 0
        or external_report_count == 0
        or distinct_external_benchmark_families == 0
        or required_family_coverage_gap
        or priority_family_under_sampled_gap
    ):
        next_policy["adaptive_search"] = True
        next_policy["cycles"] = min(
            max_cycles,
            max(current_cycles + 1, int(next_policy.get("cycles", current_cycles) or current_cycles)),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(
                current_task_limit * (4 if retained_cycles <= 0 else 2),
                int(next_policy.get("task_limit", current_task_limit) or current_task_limit),
            ),
        )
        if campaign_breadth_pressure_cycles > 0:
            next_policy["campaign_width"] = min(
                max_campaign_width,
                max(current_campaign_width + 1, int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width)),
            )
        if variant_breadth_pressure_cycles > 0 or retained_cycles <= 0:
            next_policy["variant_width"] = min(
                max_variant_width,
                max(current_variant_width + 1, int(next_policy.get("variant_width", current_variant_width) or current_variant_width)),
            )
        if retained_cycles <= 0:
            next_policy["focus"] = "discovered_task_adaptation"
        if required_family_coverage_gap:
            next_policy["focus"] = "discovered_task_adaptation"
        if priority_family_under_sampled_gap:
            next_policy["focus"] = "discovered_task_adaptation"
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
    elif rejected_cycles > retained_cycles or avg_retained_pass_delta <= 0.0:
        next_policy["adaptive_search"] = True
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
    else:
        next_policy["cycles"] = max(1, min(max_cycles, int(next_policy.get("cycles", current_cycles) or current_cycles)))

    if repo_setting_adaptive_search_pressure:
        next_policy["adaptive_search"] = True
        if retained_cycles > 0 and retained_phase_gates_passed and next_policy.get("focus") == "balanced":
            next_policy["focus"] = "discovered_task_adaptation"
    if productive_partial_conversion_gap:
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "discovered_task_adaptation"
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
    no_retained_gain_decision_gap = (
        broad_observe_then_retrieval_first
        and retained_gain_conversion_gap
        and retained_cycles <= 0
        and runtime_managed_decisions <= 0
    )
    severe_retrieval_redirect = (
        broad_observe_then_retrieval_first
        and (
            retained_gain_conversion_gap
            or semantic_progress_drift
            or observability_gap
            or subsystem_robustness_gap
        )
    )
    if broad_observe_then_retrieval_first and retained_gain_conversion_gap:
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "discovered_task_adaptation"
        if no_retained_gain_decision_gap or severe_retrieval_redirect:
            next_policy["cycles"] = min(
                max_cycles,
                max(
                    current_cycles + (2 if severe_retrieval_redirect else 1),
                    int(next_policy.get("cycles", current_cycles) or current_cycles),
                ),
            )
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(
                current_campaign_width + (2 if no_retained_gain_decision_gap else 1),
                int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width),
            ),
        )
        next_policy["variant_width"] = min(
            max_variant_width,
            max(
                current_variant_width + (2 if no_retained_gain_decision_gap else 1),
                int(next_policy.get("variant_width", current_variant_width) or current_variant_width),
            ),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(
                current_task_limit * (4 if no_retained_gain_decision_gap else 2),
                int(next_policy.get("task_limit", current_task_limit) or current_task_limit),
            ),
        )
        next_policy = _apply_subsystem_cooldown(
            next_policy,
            subsystem="retrieval",
            rounds=3 if no_retained_gain_decision_gap else 2,
        )
    if observability_gap or subsystem_robustness_gap:
        next_policy["adaptive_search"] = True
        if next_policy.get("focus") == "balanced":
            next_policy["focus"] = "recovery_alignment" if subsystem_robustness_gap else "discovered_task_adaptation"
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(
                current_campaign_width + 1,
                int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width),
            ),
        )
        if semantic_progress_drift or broad_observe_then_retrieval_first or micro_step_verification_loop:
            next_policy["variant_width"] = min(
                max_variant_width,
                max(
                    current_variant_width + 1,
                    int(next_policy.get("variant_width", current_variant_width) or current_variant_width),
                ),
            )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(
                current_task_limit * 2,
                int(next_policy.get("task_limit", current_task_limit) or current_task_limit),
            ),
        )
        if micro_step_verification_loop and stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
        elif semantic_progress_drift or broad_observe_then_retrieval_first:
            next_policy = _apply_subsystem_cooldown(next_policy, subsystem="retrieval", rounds=2)
    if bool(subsystem_monoculture.get("active", False)):
        next_policy["adaptive_search"] = True
        next_policy["focus"] = "discovered_task_adaptation"
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(
                current_campaign_width + (2 if bool(subsystem_monoculture.get("severe", False)) else 1),
                int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width),
            ),
        )
        next_policy["variant_width"] = min(
            max_variant_width,
            max(
                current_variant_width + 1,
                int(next_policy.get("variant_width", current_variant_width) or current_variant_width),
            ),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(
                current_task_limit * (4 if bool(subsystem_monoculture.get("severe", False)) else 2),
                int(next_policy.get("task_limit", current_task_limit) or current_task_limit),
            ),
        )
        if stalled_subsystem:
            next_policy = _apply_subsystem_cooldown(
                next_policy,
                subsystem=stalled_subsystem,
                rounds=3 if bool(subsystem_monoculture.get("severe", False)) else 2,
            )
    if coding_frontier_expansion:
        next_policy["adaptive_search"] = True
        if next_policy.get("focus") == "balanced":
            next_policy["focus"] = "discovered_task_adaptation"
        next_policy["cycles"] = min(
            max_cycles,
            max(
                current_cycles + 1,
                int(next_policy.get("cycles", current_cycles) or current_cycles),
            ),
        )
        next_policy["task_limit"] = min(
            max_task_limit,
            max(
                current_task_limit * 2,
                int(next_policy.get("task_limit", current_task_limit) or current_task_limit),
            ),
        )
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(
                current_campaign_width + 1,
                int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width),
            ),
        )
        if long_horizon_retained_cycles > 0 and not (repo_setting_family_expansion or coding_frontier_family_expansion):
            next_policy["variant_width"] = min(
                max_variant_width,
                max(
                    current_variant_width + 1,
                    int(next_policy.get("variant_width", current_variant_width) or current_variant_width),
                ),
            )
        elif repo_setting_family_expansion or coding_frontier_family_expansion:
            next_policy["variant_width"] = current_variant_width
    if (
        repo_setting_campaign_width_pressure
        or repo_setting_task_step_floor_pressure
        or repo_setting_adaptive_search_pressure
    ):
        next_policy["cycles"] = min(
            max_cycles,
            max(
                current_cycles + 1,
                int(next_policy.get("cycles", current_cycles) or current_cycles),
            ),
        )
    if repo_setting_campaign_width_pressure:
        next_policy["campaign_width"] = min(
            max_campaign_width,
            max(
                current_campaign_width + (2 if repo_setting_family_expansion else 1),
                int(next_policy.get("campaign_width", current_campaign_width) or current_campaign_width),
            ),
        )
    breadth_only_pressure = bool(
        retained_cycles > 0
        and retained_phase_gates_passed
        and (campaign_breadth_pressure_cycles > 0 or variant_breadth_pressure_cycles > 0)
        and not (
            max_low_confidence_episode_delta > 0
            or min_trusted_retrieval_step_delta < 0
            or required_family_coverage_gap
            or priority_family_under_sampled_gap
            or productive_partial_conversion_gap
            or broad_observe_then_retrieval_first
            or semantic_progress_drift
            or micro_step_verification_loop
            or observability_gap
            or subsystem_robustness_gap
            or bool(subsystem_monoculture.get("active", False))
            or coding_frontier_expansion
            or repo_setting_campaign_width_pressure
            or repo_setting_task_step_floor_pressure
            or repo_setting_adaptive_search_pressure
        )
    )
    if breadth_only_pressure and str(next_policy.get("focus", "")).strip() == "discovered_task_adaptation":
        next_policy["focus"] = str(current_policy.get("focus", "balanced")).strip() or "balanced"

    depth_only_repo_setting_pressure = bool(
        repo_setting_task_step_floor_pressure
        and not repo_setting_campaign_width_pressure
        and not campaign_breadth_pressure_cycles > 0
        and not variant_breadth_pressure_cycles > 0
        and retained_cycles > 0
        and retained_phase_gates_passed
        and failed_decisions <= 0
        and max_regressed_families <= 0
        and max_generated_regressed_families <= 0
        and not required_family_coverage_gap
        and not priority_family_under_sampled_gap
        and not productive_partial_conversion_gap
        and not retained_gain_conversion_gap
        and not observability_gap
        and not subsystem_robustness_gap
        and not bool(subsystem_monoculture.get("active", False))
        and not broad_observe_then_retrieval_first
        and not semantic_progress_drift
        and not micro_step_verification_loop
        and not coding_frontier_expansion
    )
    if depth_only_repo_setting_pressure:
        next_policy["campaign_width"] = current_campaign_width
        next_policy["variant_width"] = current_variant_width

    next_task_step_floor = current_task_step_floor
    if depth_drift_cycles > 0 or (avg_retained_pass_delta <= 0.0 and avg_retained_step_delta > 0.0):
        next_task_step_floor = max(1, max(1, current_task_step_floor // 2))
    elif (
        retained_cycles > 0
        and retained_phase_gates_passed
        and max_regressed_families <= 0
        and max_generated_regressed_families <= 0
        and worst_family_delta >= 0.0
        and worst_generated_family_delta >= 0.0
        and productive_depth_retained_cycles > 0
        and average_productive_depth_step_delta > 0.0
    ):
        growth_multiplier = 2.0 if long_horizon_retained_cycles > 0 else 1.5
        if coding_frontier_family_expansion or (coding_frontier_expansion and repo_setting_family_expansion):
            growth_multiplier = 1.5 if long_horizon_retained_cycles > 0 else 1.25
        next_task_step_floor = min(
            max_task_step_floor,
            max(current_task_step_floor + 1, int(math.ceil(float(current_task_step_floor) * growth_multiplier))),
        )
    elif repo_setting_task_step_floor_pressure:
        next_task_step_floor = min(
            max_task_step_floor,
            max(current_task_step_floor + 1, int(math.ceil(float(current_task_step_floor) * 1.5))),
        )
    next_policy["task_step_floor"] = max(1, next_task_step_floor)

    cooldown_rounds_by_subsystem = (
        dict(subsystem_signal.get("cooldown_rounds_by_subsystem", {}))
        if isinstance(subsystem_signal.get("cooldown_rounds_by_subsystem", {}), dict)
        else {}
    )
    for subsystem in subsystem_signal["cooldown_candidates"][:2]:
        next_policy = _apply_subsystem_cooldown(
            next_policy,
            subsystem=subsystem,
            rounds=max(2, int(cooldown_rounds_by_subsystem.get(subsystem, 2) or 2)),
        )

    if liftoff_payload:
        liftoff = _liftoff_signal(liftoff_payload)
        reason = str(liftoff["reason"]).lower()
        if not bool(liftoff["allow_kernel_autobuild"]):
            next_policy["focus"] = "discovered_task_adaptation"
            next_policy["cycles"] = min(max_cycles, max(1, int(next_policy["cycles"])) + 1)
            next_policy["task_limit"] = min(max_task_limit, max(1, current_task_limit or 64) * 2)
        elif liftoff["state"] == "reject":
            next_policy["adaptive_search"] = True
            next_policy["cycles"] = min(max_cycles, max(1, int(next_policy["cycles"])) + 1)
            if any(token in reason for token in ("failure-recovery", "generated", "unsafe", "trust", "hidden side")):
                next_policy["focus"] = "recovery_alignment"
            else:
                next_policy["focus"] = "balanced"
        elif liftoff["state"] == "shadow_only" and next_policy.get("focus") == "balanced":
            next_policy["cycles"] = min(max_cycles, max(1, int(next_policy["cycles"])) + 1)
            next_policy["variant_width"] = min(max_variant_width, current_variant_width + 1)
    next_policy["priority_benchmark_families"] = _select_priority_benchmark_families(
        required_families=required_families,
        missing_required_families=missing_required_families,
        current_priority_families=next_policy.get(
            "priority_benchmark_families",
            current_policy.get("priority_benchmark_families", []),
        ),
        ranked_priority_families=priority_families_ranked_by_selection_score,
        repo_setting_priority_families=priority_family_history.get("frontier_repo_setting_families", []),
        priority_family_selection_scores=priority_family_selection_scores,
        min_selection_score=float(priority_family_scoring_policy.get("priority_family_min_selection_score", 0.0) or 0.0),
        productive_priority_families=priority_families_with_retained_gain,
        retained_gain_conversion_priority_families=priority_families_needing_retained_gain_conversion,
        under_sampled_priority_families=priority_families_under_sampled,
        low_return_priority_families=priority_families_low_return,
    )
    if coding_frontier_expansion and not required_family_coverage_gap:
        widened_priority_families: list[str] = []
        for family in ("integration", "repository", "project", *next_policy.get("priority_benchmark_families", [])):
            token = str(family).strip()
            if token and token not in widened_priority_families:
                widened_priority_families.append(token)
        next_policy["priority_benchmark_families"] = widened_priority_families[:3]
    selected_priority_families = _normalize_benchmark_families(next_policy.get("priority_benchmark_families", []))
    priority_family_selection_cap = 3
    filtered_priority_families = [
        family
        for family in priority_families_ranked_by_selection_score
        if family not in selected_priority_families
        and family not in missing_required_families
        and float(priority_family_selection_scores.get(family, 0.0) or 0.0)
        < float(priority_family_scoring_policy.get("priority_family_min_selection_score", 0.0) or 0.0)
    ]
    priority_family_budget_warning: dict[str, object] = {}
    if len(selected_priority_families) < priority_family_selection_cap:
        if filtered_priority_families:
            priority_family_budget_warning = {
                "kind": "selection_score_floor",
                "detail": "priority family budget stayed narrow because the selection floor filtered lower-score families",
                "selected_count": len(selected_priority_families),
                "selection_cap": priority_family_selection_cap,
                "filtered_families": filtered_priority_families,
                "min_selection_score": float(
                    priority_family_scoring_policy.get("priority_family_min_selection_score", 0.0) or 0.0
                ),
            }
        elif len(priority_families_ranked_by_selection_score) < priority_family_selection_cap:
            priority_family_budget_warning = {
                "kind": "limited_priority_families",
                "detail": "priority family budget stayed narrow because fewer than three ranked families were available",
                "selected_count": len(selected_priority_families),
                "selection_cap": priority_family_selection_cap,
                "available_families": priority_families_ranked_by_selection_score,
            }
    next_policy = _materialize_excluded_subsystems(next_policy)
    if isinstance(round_payload, dict):
        widened_dimensions: list[str] = []
        for key in ("cycles", "task_limit", "task_step_floor", "campaign_width", "variant_width"):
            if int(next_policy.get(key, 0) or 0) > int(current_policy.get(key, 0) or 0):
                widened_dimensions.append(key)
        previous_excluded = set(_normalize_excluded_subsystems(current_policy.get("excluded_subsystems", [])))
        next_excluded = set(_normalize_excluded_subsystems(next_policy.get("excluded_subsystems", [])))
        reason_codes: list[str] = []
        if (
            max_regressed_families > 0
            or max_generated_regressed_families > 0
            or worst_family_delta < 0.0
            or worst_generated_family_delta < 0.0
        ):
            reason_codes.append("safety_regression")
        if failed_decisions > 0 or not retained_phase_gates_passed:
            reason_codes.append("phase_gate_failure")
        if max_low_confidence_episode_delta > 0 or min_trusted_retrieval_step_delta < 0:
            reason_codes.append("low_confidence_or_retrieval_gap")
        if productive_depth_retained_cycles > 0:
            reason_codes.append("productive_depth_gain")
        if depth_drift_cycles > 0:
            reason_codes.append("depth_drift_pressure")
        if campaign_breadth_pressure_cycles > 0:
            reason_codes.append("campaign_breadth_pressure")
        if variant_breadth_pressure_cycles > 0:
            reason_codes.append("variant_breadth_pressure")
        if external_report_count == 0 or distinct_external_benchmark_families == 0:
            reason_codes.append("external_family_coverage_gap")
        if required_family_coverage_gap:
            reason_codes.append("required_family_coverage_gap")
        if priority_family_under_sampled_gap:
            reason_codes.append("priority_family_under_sampled")
        if retained_gain_conversion_gap:
            reason_codes.append("retained_gain_conversion_gap")
        if broad_observe_then_retrieval_first:
            reason_codes.append("broad_observe_then_retrieval_first")
        if semantic_progress_drift:
            reason_codes.append("semantic_progress_drift")
        if micro_step_verification_loop:
            reason_codes.append("micro_step_verification_loop")
        if productive_partial_conversion_gap:
            reason_codes.append("productive_partial_conversion_gap")
        if observability_gap:
            reason_codes.append("observability_gap")
        if subsystem_robustness_gap:
            reason_codes.append("subsystem_robustness_gap")
        if priority_families_low_return:
            reason_codes.append("priority_family_low_return")
        if repo_setting_campaign_width_pressure:
            reason_codes.append("frontier_repo_setting_campaign_width_pressure")
        if repo_setting_task_step_floor_pressure:
            reason_codes.append("frontier_repo_setting_depth_pressure")
        if repo_setting_adaptive_search_pressure:
            reason_codes.append("frontier_repo_setting_adaptive_search")
        if coding_frontier_expansion:
            reason_codes.append("coding_frontier_expansion")
        if str(priority_family_budget_warning.get("kind", "")).strip() == "selection_score_floor":
            reason_codes.append("priority_family_selection_floor_narrowing")
        elif str(priority_family_budget_warning.get("kind", "")).strip() == "limited_priority_families":
            reason_codes.append("priority_family_limited_availability")
        if retained_cycles <= 0:
            reason_codes.append("no_yield_round")
        if rejected_cycles > retained_cycles or avg_retained_pass_delta <= 0.0:
            reason_codes.append("weak_retention_outcome")
        if stalled_subsystem and stalled_subsystem in next_excluded:
            reason_codes.append("stalled_subsystem_cooldown")
        if subsystem_signal["cooldown_candidates"]:
            reason_codes.append("subsystem_reject_cooldown")
        if subsystem_signal.get("zero_yield_dominant_subsystems"):
            reason_codes.append("dominant_zero_yield_subsystem_cooldown")
        if bool(subsystem_monoculture.get("active", False)):
            reason_codes.append("subsystem_monoculture")
        if not reason_codes:
            reason_codes.append("stable_progress")
        if liftoff_payload:
            liftoff = _liftoff_signal(liftoff_payload)
            if not bool(liftoff["allow_kernel_autobuild"]):
                reason_codes.append("liftoff_autobuild_blocked")
            if str(liftoff.get("state", "")).strip() == "reject":
                reason_codes.append("liftoff_reject")
            elif str(liftoff.get("state", "")).strip() == "shadow_only":
                reason_codes.append("liftoff_shadow_only")
        round_payload["policy_shift_rationale"] = {
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "focus_before": str(current_policy.get("focus", "")).strip(),
            "focus_after": str(next_policy.get("focus", "")).strip(),
            "adaptive_search_before": bool(current_policy.get("adaptive_search", False)),
            "adaptive_search_after": bool(next_policy.get("adaptive_search", False)),
            "task_step_floor_before": int(
                current_policy.get("task_step_floor", current_task_step_floor) or current_task_step_floor
            ),
            "task_step_floor_after": int(
                next_policy.get("task_step_floor", current_task_step_floor) or current_task_step_floor
            ),
            "widened_dimensions": widened_dimensions,
            "cooled_subsystems": sorted(next_excluded - previous_excluded),
            "priority_benchmark_families_before": _normalize_benchmark_families(
                current_policy.get("priority_benchmark_families", [])
            ),
            "priority_benchmark_families_after": _normalize_benchmark_families(
                next_policy.get("priority_benchmark_families", [])
            ),
            "stalled_subsystem": stalled_subsystem,
            "planner_pressure": {
                "campaign_breadth_pressure_cycles": campaign_breadth_pressure_cycles,
                "variant_breadth_pressure_cycles": variant_breadth_pressure_cycles,
                "productive_depth_retained_cycles": productive_depth_retained_cycles,
                "average_productive_depth_step_delta": average_productive_depth_step_delta,
                "depth_drift_cycles": depth_drift_cycles,
                "average_depth_drift_step_delta": average_depth_drift_step_delta,
                "long_horizon_retained_cycles": long_horizon_retained_cycles,
                "dominant_subsystem": str(planner_pressure_signal.get("dominant_subsystem", "")).strip(),
                "subsystem_monoculture": subsystem_monoculture,
                "cooldown_rounds_by_subsystem": cooldown_rounds_by_subsystem,
                "zero_yield_dominant_subsystems": _normalize_excluded_subsystems(
                    subsystem_signal.get("zero_yield_dominant_subsystems", [])
                ),
                "external_report_count": external_report_count,
                "distinct_external_benchmark_families": distinct_external_benchmark_families,
                "missing_required_families": missing_required_families,
                "distinct_family_gap": distinct_family_gap,
                "priority_families": priority_families,
                "priority_families_with_retained_gain": priority_families_with_retained_gain,
                "priority_families_without_signal": priority_families_without_signal,
                "priority_families_needing_retained_gain_conversion": priority_families_needing_retained_gain_conversion,
                "priority_families_under_sampled": priority_families_under_sampled,
                "broad_observe_then_retrieval_first": broad_observe_then_retrieval_first,
                "semantic_progress_drift": semantic_progress_drift,
                "micro_step_verification_loop": micro_step_verification_loop,
                "observability_gap": observability_gap,
                "subsystem_robustness_gap": subsystem_robustness_gap,
                "incomplete_cycle_count": incomplete_cycle_count,
                "observed_primary_runs": observed_primary_runs,
                "generated_success_completed_runs": generated_success_completed_runs,
                "productive_partial_sampled_families": _normalize_benchmark_families(
                    productive_partial_conversion.get("sampled_families", [])
                ),
                "productive_partial_sampled_priority_families": _normalize_benchmark_families(
                    productive_partial_conversion.get("sampled_priority_families", [])
                ),
                "productive_partial_conversion_gap": productive_partial_conversion_gap,
                "intermediate_decision_evidence": intermediate_decision_evidence,
                "priority_families_low_return": priority_families_low_return,
                "priority_families_ranked_by_selection_score": priority_families_ranked_by_selection_score,
                "frontier_failure_motif_priority_pairs": list(
                    priority_family_history.get("frontier_failure_motif_priority_pairs", [])
                )
                if isinstance(priority_family_history.get("frontier_failure_motif_priority_pairs", []), list)
                else [],
                "frontier_repo_setting_priority_pairs": list(
                    priority_family_history.get("frontier_repo_setting_priority_pairs", [])
                )
                if isinstance(priority_family_history.get("frontier_repo_setting_priority_pairs", []), list)
                else [],
                "frontier_failure_motif_families": _normalize_benchmark_families(
                    priority_family_history.get("frontier_failure_motif_families", [])
                ),
                "frontier_repo_setting_families": _normalize_benchmark_families(
                    priority_family_history.get("frontier_repo_setting_families", [])
                ),
                "repo_setting_policy_pressure": {
                    "focused_pairs": list(repo_setting_policy_pressure.get("focused_pairs", []))
                    if isinstance(repo_setting_policy_pressure.get("focused_pairs", []), list)
                    else [],
                    "focused_families": _normalize_benchmark_families(
                        repo_setting_policy_pressure.get("focused_families", [])
                    ),
                    "signal_counts": dict(repo_setting_policy_pressure.get("signal_counts", {}))
                    if isinstance(repo_setting_policy_pressure.get("signal_counts", {}), dict)
                    else {},
                    "campaign_width_signals": _normalize_benchmark_families(
                        repo_setting_policy_pressure.get("campaign_width_signals", [])
                    ),
                    "task_step_floor_signals": _normalize_benchmark_families(
                        repo_setting_policy_pressure.get("task_step_floor_signals", [])
                    ),
                    "adaptive_search_signals": _normalize_benchmark_families(
                        repo_setting_policy_pressure.get("adaptive_search_signals", [])
                    ),
                    "learned_priors": dict(learned_repo_setting_priors),
                },
                "coding_frontier_expansion": {
                    "triggered": bool(coding_frontier_expansion),
                    "productive_priority_families": _rank_priority_benchmark_families(
                        list(productive_priority_family_set)
                    ),
                    "forced_priority_families": ["integration", "repository", "project"]
                    if coding_frontier_expansion and not required_family_coverage_gap
                    else [],
                },
                "priority_family_budget_warning": priority_family_budget_warning,
                "priority_family_history": priority_family_history,
            },
            "controller_selected_action_key": str(planner_diagnostics.get("selected_action_key", "")).strip(),
            "controller_selected_adjusted_score": float(
                planner_diagnostics.get("selected_adjusted_score", planner_diagnostics.get("selected_score", 0.0)) or 0.0
            ),
        }
    return next_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--campaign-width", type=int, default=2)
    parser.add_argument("--variant-width", type=int, default=1)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--task-limit", type=int, default=128)
    parser.add_argument("--task-step-floor", type=int, default=0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--focus", choices=("balanced", "recovery_alignment", "discovered_task_adaptation"), default="balanced")
    parser.add_argument("--liftoff", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--apply-routing", action="store_true")
    parser.add_argument("--promote-on-retain", action="store_true")
    parser.add_argument("--min-free-disk-gib", type=float, default=50.0)
    parser.add_argument("--skip-trust-preflight", action="store_true")
    parser.add_argument("--keep-reports", type=int, default=50)
    parser.add_argument("--keep-candidate-dirs", type=int, default=20)
    parser.add_argument("--keep-tolbert-candidate-dirs", type=int, default=3)
    parser.add_argument("--keep-tolbert-promoted-checkpoints", type=int, default=3)
    parser.add_argument("--keep-checkpoints", type=int, default=20)
    parser.add_argument("--keep-snapshot-entries", type=int, default=100)
    parser.add_argument("--keep-tolbert-dataset-dirs", type=int, default=10)
    parser.add_argument("--tolbert-candidate-budget-gib", type=float, default=1.0)
    parser.add_argument("--tolbert-shared-store-budget-gib", type=float, default=4.0)
    parser.add_argument("--tolbert-promoted-checkpoint-budget-gib", type=float, default=4.0)
    parser.add_argument("--cleanup-target-free-disk-gib", type=float, default=75.0)
    parser.add_argument("--max-no-yield-rounds", type=int, default=2)
    parser.add_argument("--max-policy-stall-rounds", type=int, default=1)
    parser.add_argument("--max-child-failure-recovery-rounds", type=int, default=1)
    parser.add_argument("--max-cycles-per-round", type=int, default=3)
    parser.add_argument("--max-task-limit", type=int, default=1024)
    parser.add_argument("--max-campaign-width", type=int, default=4)
    parser.add_argument("--max-variant-width", type=int, default=3)
    parser.add_argument("--stop-on-liftoff-retain", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-heartbeat-seconds", type=float, default=120.0)
    parser.add_argument("--max-child-silence-seconds", type=float, default=1800.0)
    parser.add_argument("--max-child-runtime-seconds", type=float, default=14400.0)
    parser.add_argument("--max-child-progress-stall-seconds", type=float, default=1800.0)
    parser.add_argument("--lock-path", default=None)
    parser.add_argument("--attach-on-lock", type=int, default=1)
    parser.add_argument("--event-log-path", default=None)
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--alert-command", default=None)
    parser.add_argument("--policy-shift-alert-subscriptions", default="all")
    parser.add_argument("--slack-webhook-url", default=None)
    parser.add_argument("--pagerduty-routing-key", default=None)
    parser.add_argument("--alert-rate-limit-seconds", type=float, default=900.0)
    parser.add_argument("--alert-state-path", default=None)
    parser.add_argument("--lease-backend", choices=("local", "http"), default="local")
    parser.add_argument("--lease-endpoint", default=None)
    parser.add_argument("--lease-token", default=None)
    parser.add_argument("--controller-state-path", default=None)
    parser.add_argument("--controller-gamma", type=float, default=0.85)
    parser.add_argument("--controller-learning-rate", type=float, default=0.08)
    parser.add_argument("--controller-transition-learning-rate", type=float, default=0.15)
    parser.add_argument("--controller-exploration-bonus", type=float, default=2.5)
    parser.add_argument("--controller-uncertainty-penalty", type=float, default=1.5)
    parser.add_argument("--controller-rollout-depth", type=int, default=2)
    parser.add_argument("--controller-rollout-beam-width", type=int, default=6)
    parser.add_argument("--controller-repeat-action-penalty", type=float, default=5.0)
    parser.add_argument("--controller-state-repeat-penalty", type=float, default=0.25)
    parser.add_argument("--controller-state-novelty-bonus", type=float, default=0.1)
    parser.add_argument("--global-storage-root", default=None)
    parser.add_argument("--global-storage-policy-path", default=None)
    parser.add_argument("--global-storage-target-free-gib", type=float, default=150.0)
    parser.add_argument("--global-storage-top-k", type=int, default=0)
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device
    config.ensure_directories()

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        Path(args.report_path)
        if args.report_path
        else config.improvement_reports_dir / f"unattended_campaign_{run_id}.json"
    )
    lock_path = (
        Path(args.lock_path)
        if args.lock_path
        else config.improvement_reports_dir / "unattended_campaign.lock"
    )
    event_log_path = (
        Path(args.event_log_path)
        if args.event_log_path
        else report_path.with_suffix(".events.jsonl")
    )
    status_path = (
        Path(args.status_path)
        if args.status_path
        else report_path.with_suffix(".status.json")
    )
    alert_state_path = (
        Path(args.alert_state_path)
        if args.alert_state_path
        else report_path.with_suffix(".alerts.json")
    )
    controller_state_path = (
        Path(args.controller_state_path)
        if args.controller_state_path
        else report_path.parent / f"controller_state__{report_path.stem}.json"
    )
    global_storage_root = Path(args.global_storage_root) if args.global_storage_root else repo_root
    global_storage_policy_path = (
        Path(args.global_storage_policy_path)
        if args.global_storage_policy_path
        else repo_root / "config" / "global_storage_policy.json"
    )
    bootstrap_cleanup_candidates = [
        path
        for path in (report_path, status_path, alert_state_path, controller_state_path)
        if not path.exists()
    ]
    global_storage_policy = _load_global_storage_policy(global_storage_policy_path)
    resume_payload = _read_json(report_path) if args.resume and report_path.exists() else {}
    controller_state = normalize_controller_state(
        _read_json(controller_state_path) if args.resume and controller_state_path.exists() else None
    )
    if not (args.resume and controller_state_path.exists()):
        controller_state = default_controller_state(
            gamma=float(args.controller_gamma),
            value_learning_rate=float(args.controller_learning_rate),
            transition_learning_rate=float(args.controller_transition_learning_rate),
            exploration_bonus=float(args.controller_exploration_bonus),
            uncertainty_penalty=float(args.controller_uncertainty_penalty),
            rollout_depth=max(1, int(args.controller_rollout_depth)),
            rollout_beam_width=max(1, int(args.controller_rollout_beam_width)),
            repeat_action_penalty=float(args.controller_repeat_action_penalty),
            state_repeat_penalty=float(args.controller_state_repeat_penalty),
            state_novelty_bonus=float(args.controller_state_novelty_bonus),
        )
    phase = "preflight"
    prior_rounds = (
        list(resume_payload.get("rounds", []))
        if isinstance(resume_payload.get("rounds", []), list)
        else []
    )
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        if str(round_payload.get("status", "")).strip() == "running":
            _mark_round(
                round_payload,
                status="interrupted",
                phase=str(round_payload.get("phase", resume_payload.get("phase", "campaign"))).strip() or "campaign",
                reason="prior unattended round was interrupted before completion",
            )
    current_policy = _initial_round_policy(args, config)
    if not prior_rounds:
        current_policy = _apply_semantic_policy_seed(
            current_policy,
            semantic_seed=_semantic_policy_seed(config),
        )
    if prior_rounds:
        last_round = prior_rounds[-1]
        if isinstance(last_round, dict) and "next_policy" in last_round and isinstance(last_round.get("next_policy", {}), dict):
            current_policy = dict(last_round["next_policy"])
    current_policy = _resume_policy_from_partial_round(
        current_policy,
        prior_rounds=prior_rounds,
        max_cycles=max(1, args.max_cycles_per_round),
        max_task_limit=max(1, args.max_task_limit),
        max_campaign_width=max(1, args.max_campaign_width),
    )
    rounds_completed = _count_completed_rounds(prior_rounds)
    child_failure_recoveries_used = int(resume_payload.get("child_failure_recoveries_used", 0) or 0)
    child_failure_recovery_budget = max(0, int(args.max_child_failure_recovery_rounds))
    no_yield_rounds = 0
    policy_stall_rounds = 0
    depth_runway_credit = 0
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        outer_stop_signal = _round_stop_signal(
            campaign_report=round_payload.get("campaign_report", {}),
            liftoff_payload=round_payload.get("liftoff_report", {}),
            round_payload=round_payload,
        )
        round_payload["outer_stop_signal"] = outer_stop_signal
        no_yield_rounds = _next_no_yield_rounds(
            no_yield_rounds,
            campaign_report=round_payload.get("campaign_report", {}),
            liftoff_payload=round_payload.get("liftoff_report", {}),
            round_payload=round_payload,
        )
        if (
            isinstance(round_payload.get("policy", {}), dict)
            and "next_policy" in round_payload
            and isinstance(round_payload.get("next_policy", {}), dict)
        ):
            policy_stall_rounds = _next_policy_stall_rounds(
                policy_stall_rounds,
                current_policy=round_payload["policy"],
                next_policy=round_payload["next_policy"],
                campaign_report=round_payload.get("campaign_report", {}),
                liftoff_payload=round_payload.get("liftoff_report", {}),
                round_payload=round_payload,
            )
        depth_runway_credit = _next_depth_runway_credit(
            depth_runway_credit,
            campaign_report=round_payload.get("campaign_report", {}),
            liftoff_payload=round_payload.get("liftoff_report", {}),
            round_payload=round_payload,
        )
        round_payload["adaptive_stop_budget"] = _adaptive_stop_budget(
            base_max_no_yield_rounds=max(0, args.max_no_yield_rounds),
            base_max_policy_stall_rounds=max(0, args.max_policy_stall_rounds),
            depth_runway_credit=depth_runway_credit,
        )
    report: dict[str, object] = {
        "spec_version": "asi_v1",
        "report_kind": "unattended_campaign_report",
        "created_at": created_at.isoformat(),
        "status": "running",
        "phase": phase,
        "rounds_requested": max(1, args.rounds),
        "rounds_completed": rounds_completed,
        "child_failure_recovery_budget": child_failure_recovery_budget,
        "child_failure_recoveries_used": child_failure_recoveries_used,
        "rounds": prior_rounds,
        "campaign": {
            "cycles": max(1, args.cycles),
            "campaign_width": max(1, args.campaign_width),
            "variant_width": max(1, args.variant_width),
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": max(0, args.task_limit),
            "tolbert_device": config.tolbert_device,
            "focus": args.focus,
            "liftoff": args.liftoff,
        },
        "current_policy": current_policy,
        "alerting": {
            "alert_command": str(args.alert_command or ""),
            "slack_webhook_url": str(args.slack_webhook_url or ""),
            "pagerduty_routing_key": str(args.pagerduty_routing_key or ""),
        },
        "policy_shift_summary": _policy_shift_summary(prior_rounds),
        "policy_shift_alert_subscriptions": _normalize_policy_shift_alert_subscriptions(
            getattr(args, "policy_shift_alert_subscriptions", "all")
        ),
        "lock_path": str(lock_path),
        "event_log_path": str(event_log_path),
        "status_path": str(status_path),
        "alert_state_path": str(alert_state_path),
        "controller_state_path": str(controller_state_path),
        "controller_summary": controller_state_summary(controller_state),
        "unattended_evidence": _unattended_evidence_snapshot(config),
        "global_storage_root": str(global_storage_root),
        "global_storage_policy_path": "" if global_storage_policy_path is None else str(global_storage_policy_path),
        "global_storage_policy": global_storage_policy,
    }
    _write_controller_state(controller_state_path, controller_state, config=config)
    _persist_report_state(
        report_path,
        report,
        config=config,
        status_path=status_path,
        lock_path=lock_path,
        external_lease_backend=str(args.lease_backend or ""),
        external_lease_endpoint=str(args.lease_endpoint or ""),
        external_lease_token=str(args.lease_token or ""),
    )

    def _emit_alert_if_configured() -> None:
        if not _alerts_enabled(args):
            return
        report["alert"] = _dispatch_alerts(
            alert_command=str(args.alert_command or ""),
            slack_webhook_url=str(args.slack_webhook_url or ""),
            pagerduty_routing_key=str(args.pagerduty_routing_key or ""),
            report_path=report_path,
            status_path=status_path,
            event_log_path=event_log_path,
            payload=report,
            state_path=alert_state_path,
            rate_limit_seconds=float(args.alert_rate_limit_seconds),
            config=config,
        )
        _persist_report_state(
            report_path,
            report,
            config=config,
            status_path=status_path,
            lock_path=lock_path,
            external_lease_backend=str(args.lease_backend or ""),
            external_lease_endpoint=str(args.lease_endpoint or ""),
            external_lease_token=str(args.lease_token or ""),
        )

    def _emit_policy_shift_alert_if_configured(round_payload: dict[str, object]) -> None:
        rationale = round_payload.get("policy_shift_rationale", {})
        if not isinstance(rationale, dict):
            return
        alert_reason = _policy_shift_alert_reason(rationale)
        if not alert_reason:
            return
        subscriptions = _normalize_policy_shift_alert_subscriptions(
            getattr(args, "policy_shift_alert_subscriptions", "all")
        )
        if not _policy_shift_alert_subscribed(rationale, subscriptions):
            round_payload["policy_shift_alert"] = {
                "ran": False,
                "suppressed": True,
                "detail": "policy shift alert not subscribed",
                "subscriptions": subscriptions,
            }
            report["policy_shift_alert"] = round_payload["policy_shift_alert"]
            _persist_report_state(
                report_path,
                report,
                config=config,
                status_path=status_path,
                lock_path=lock_path,
                external_lease_backend=str(args.lease_backend or ""),
                external_lease_endpoint=str(args.lease_endpoint or ""),
                external_lease_token=str(args.lease_token or ""),
            )
            return
        if not _alerts_enabled(args):
            round_payload["policy_shift_alert"] = {
                "ran": False,
                "detail": "no alert transport configured",
                "subscriptions": subscriptions,
            }
            report["policy_shift_alert"] = round_payload["policy_shift_alert"]
            _persist_report_state(
                report_path,
                report,
                config=config,
                status_path=status_path,
                lock_path=lock_path,
                external_lease_backend=str(args.lease_backend or ""),
                external_lease_endpoint=str(args.lease_endpoint or ""),
                external_lease_token=str(args.lease_token or ""),
            )
            return
        alert_payload = {
            "status": "running",
            "phase": "policy_shift",
            "reason": alert_reason,
            "rounds_completed": report.get("rounds_completed", 0),
            "current_policy": report.get("current_policy", {}),
            "policy_shift_rationale": rationale,
            "policy_shift_summary": report.get("policy_shift_summary", {}),
        }
        round_payload["policy_shift_alert"] = _dispatch_alerts(
            alert_command=str(args.alert_command or ""),
            slack_webhook_url=str(args.slack_webhook_url or ""),
            pagerduty_routing_key=str(args.pagerduty_routing_key or ""),
            report_path=report_path,
            status_path=status_path,
            event_log_path=event_log_path,
            payload=alert_payload,
            state_path=alert_state_path,
            rate_limit_seconds=float(args.alert_rate_limit_seconds),
            config=config,
        )
        report["policy_shift_alert"] = round_payload["policy_shift_alert"]
        _persist_report_state(
            report_path,
            report,
            config=config,
            status_path=status_path,
            lock_path=lock_path,
            external_lease_backend=str(args.lease_backend or ""),
            external_lease_endpoint=str(args.lease_endpoint or ""),
            external_lease_token=str(args.lease_token or ""),
        )

    def _apply_failure_feedback(
        round_payload: dict[str, object] | None,
        *,
        phase: str,
        reason: str,
    ) -> None:
        nonlocal controller_state
        if not isinstance(round_payload, dict):
            return
        action_policy = round_payload.get("policy", {})
        if not isinstance(action_policy, dict):
            return
        active_child = round_payload.get("active_child", {})
        stalled_subsystem = ""
        if isinstance(active_child, dict):
            stalled_subsystem = _subsystem_from_progress_line(
                str(active_child.get("last_progress_line") or active_child.get("last_output_line") or "")
            )
        if not stalled_subsystem:
            stalled_subsystem = _subsystem_from_progress_line(str(round_payload.get("phase_detail", "")))
        supplemental_observation = _controller_observation(
            campaign_report=round_payload.get("campaign_report", {})
            if isinstance(round_payload.get("campaign_report", {}), dict)
            else {},
            liftoff_payload=round_payload.get("liftoff_report", {})
            if isinstance(round_payload.get("liftoff_report", {}), dict)
            else {},
            controller_state=controller_state,
            curriculum_controls=round_curriculum_controls,
            round_payload=round_payload,
        )
        failure_observation = build_failure_observation(
            phase=phase,
            reason=reason,
            subsystem=stalled_subsystem,
            supplemental_observation=supplemental_observation,
        )
        controller_state, controller_update = update_controller_state(
            controller_state,
            start_observation=round_payload.get("controller_observation", {}),
            action_policy=action_policy,
            end_observation=failure_observation,
        )
        round_payload["controller_failure_update"] = controller_update
        report["controller_summary"] = controller_state_summary(controller_state)
        _write_controller_state(controller_state_path, controller_state, config=config)

    def _recover_child_failure_or_stop(
        *,
        round_payload: dict[str, object],
        phase: str,
        reason: str,
    ) -> bool:
        nonlocal current_policy, child_failure_recoveries_used
        _apply_failure_feedback(round_payload, phase=phase, reason=reason)
        if child_failure_recoveries_used >= child_failure_recovery_budget:
            report["status"] = "safe_stop"
            report["phase"] = phase
            report["reason"] = str(reason)
            _mark_round(round_payload, status="safe_stop", phase=phase, reason=str(reason))
            report["rounds_completed"] = _count_completed_rounds(report["rounds"])
            report["child_failure_recoveries_used"] = child_failure_recoveries_used
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            _emit_alert_if_configured()
            print(report_path)
            raise SystemExit(2)
        child_failure_recoveries_used += 1
        next_policy = _child_failure_recovery_policy(
            current_policy,
            round_payload=round_payload,
            phase=phase,
            reason=reason,
            max_cycles=max(1, args.max_cycles_per_round),
            max_task_limit=max(1, args.max_task_limit),
            max_campaign_width=max(1, args.max_campaign_width),
            max_variant_width=max(1, args.max_variant_width),
        )
        recovery = {
            "attempted": True,
            "phase": phase,
            "reason": str(reason),
            "recovery_attempt": child_failure_recoveries_used,
            "recovery_budget": child_failure_recovery_budget,
            "next_policy": dict(next_policy),
        }
        stalled_subsystem = _stalled_subsystem_from_round(round_payload)
        if stalled_subsystem:
            recovery["stalled_subsystem"] = stalled_subsystem
        round_payload["recovery"] = recovery
        round_payload["next_policy"] = dict(next_policy)
        _mark_round(round_payload, status="recovered", phase=phase, reason=str(reason))
        report["status"] = "running"
        report["phase"] = "recovery"
        report["reason"] = f"recovering from {phase} child failure"
        report["current_policy"] = dict(next_policy)
        report["child_failure_recoveries_used"] = child_failure_recoveries_used
        report["rounds_completed"] = _count_completed_rounds(report["rounds"])
        report["policy_shift_summary"] = _policy_shift_summary(report["rounds"])
        _append_event(
            event_log_path,
            {
                "event_kind": "child_failure_recovery",
                "round_index": int(round_payload.get("round_index", 0) or 0),
                "phase": phase,
                "reason": str(reason),
                "recovery_attempt": child_failure_recoveries_used,
                "recovery_budget": child_failure_recovery_budget,
                "policy_after": dict(next_policy),
                "timestamp": time.time(),
            },
            config=config,
        )
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        current_policy = next_policy
        return True

    received_signal = {"value": 0}

    def _handle_termination(signum: int) -> None:
        received_signal["value"] = int(signum)
        raise KeyboardInterrupt(f"received signal {signal.Signals(signum).name}")

    restore_signal_handlers = install_termination_handlers(_handle_termination)

    if str(args.lease_backend or "local").strip() == "http":
        lock_result = _acquire_external_campaign_lease(
            str(args.lease_endpoint or "").strip(),
            token=str(args.lease_token or ""),
            report_path=report_path,
        )
    else:
        lock_result = _acquire_campaign_lock(lock_path, report_path=report_path, status_path=status_path)
    report["lock"] = lock_result
    attached_report_path = (
        _attached_campaign_report_path(lock_result) if bool(int(args.attach_on_lock or 0)) else None
    )
    if attached_report_path is not None:
        _cleanup_bootstrap_artifacts(
            [
                path
                for path in bootstrap_cleanup_candidates
                if path.resolve() != attached_report_path.resolve()
            ]
        )
        _progress(f"[unattended] attached to active campaign report={attached_report_path}")
        print(attached_report_path)
        return
    _persist_report_state(
        report_path,
        report,
        config=config,
        status_path=status_path,
        lock_path=lock_path,
        external_lease_backend=str(args.lease_backend or ""),
        external_lease_endpoint=str(args.lease_endpoint or ""),
        external_lease_token=str(args.lease_token or ""),
    )
    _append_event(
        event_log_path,
        {
            "event_kind": "campaign_start",
            "created_at": created_at.isoformat(),
            "report_path": str(report_path),
            "lock_path": str(lock_path),
            "lock_passed": bool(lock_result.get("passed", False)),
        },
        config=config,
    )
    if not bool(lock_result.get("passed", False)):
        report["status"] = "safe_stop"
        report["phase"] = "preflight"
        report["reason"] = str(lock_result.get("detail", "campaign lock acquisition failed"))
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        print(report_path)
        raise SystemExit(2)

    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.update(config.to_env())
        repeated_script = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
        liftoff_script = repo_root / "scripts" / "run_tolbert_liftoff_loop.py"

        requested_rounds = max(1, args.rounds)
        total_round_budget = requested_rounds + child_failure_recovery_budget
        round_index = len(report["rounds"]) + 1
        while report["rounds_completed"] < requested_rounds and round_index <= total_round_budget:
            round_payload: dict[str, object] = {
                "round_index": round_index,
                "policy": dict(current_policy),
                "runtime_limits": {
                    "max_child_runtime_seconds": float(args.max_child_runtime_seconds),
                },
                "status": "running",
                "phase": "preflight",
            }
            upsert_semantic_agent(
                config,
                agent_id=f"unattended_agent:{run_id}",
                payload={
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "agent_kind": "unattended_campaign_agent",
                    "run_id": run_id,
                    "last_round_index": round_index,
                    "current_policy": dict(current_policy),
                    "status": "running",
                },
            )
            report["rounds"].append(round_payload)
            if len(report["rounds"]) > 1:
                previous_round = report["rounds"][-2]
                if isinstance(previous_round, dict):
                    previous_rationale = previous_round.get("policy_shift_rationale", {})
                    if isinstance(previous_rationale, dict) and previous_rationale:
                        round_payload["policy_shift_rationale"] = dict(previous_rationale)
            report["rounds_completed"] = _count_completed_rounds(report["rounds"])
            phase = "preflight"
            report["phase"] = phase
            round_payload["phase"] = phase
            report["current_policy"] = dict(current_policy)
            round_curriculum_controls = _load_retained_curriculum_controls(config)
            controller_observation = _controller_observation(
                campaign_report=report["rounds"][-2].get("campaign_report", {})
                if len(report["rounds"]) > 1 and isinstance(report["rounds"][-2], dict)
                else {},
                liftoff_payload=report["rounds"][-2].get("liftoff_report", {})
                if len(report["rounds"]) > 1 and isinstance(report["rounds"][-2], dict)
                else {},
                controller_state=controller_state,
                curriculum_controls=round_curriculum_controls,
            )
            round_payload["controller_observation"] = controller_observation
            report["controller_summary"] = controller_state_summary(controller_state)
            _progress(
                f"[campaign] phase=preflight round={round_index}/{max(1, args.rounds)} "
                f"task_limit={int(current_policy['task_limit'])} focus={current_policy['focus']}"
            )
            _append_event(
                event_log_path,
                {
                    "event_kind": "phase",
                    "phase": "preflight",
                    "round_index": round_index,
                    "policy": dict(current_policy),
                    "timestamp": time.time(),
                },
                config=config,
            )
            disk = _disk_preflight(repo_root, min_free_gib=float(args.min_free_disk_gib))
            gpu = _gpu_preflight(str(current_policy["tolbert_device"]))
            trust = {
                "passed": True,
                "status": "skipped",
                "detail": "trust preflight skipped by operator",
                "reports_considered": 0,
                "failing_thresholds": [],
            }
            if not args.skip_trust_preflight:
                trust = _trust_preflight(config)
            round_payload["preflight"] = {"disk": disk, "gpu": gpu, "trust": trust}
            report["preflight"] = round_payload["preflight"]
            reporting = trust.get("reporting", {}) if isinstance(trust, dict) else {}
            if isinstance(reporting, dict) and reporting:
                report["unattended_evidence"] = reporting
            else:
                report["unattended_evidence"] = _unattended_evidence_snapshot(config)
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            if not bool(disk["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(disk["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(disk["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                print(report_path)
                raise SystemExit(2)
            if not bool(gpu["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(gpu["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(gpu["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                print(report_path)
                raise SystemExit(2)
            if not bool(trust["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(trust["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(trust["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                print(report_path)
                raise SystemExit(2)

            phase = "cleanup"
            report["phase"] = phase
            round_payload["phase"] = phase
            _progress(f"[campaign] phase=cleanup round={round_index}")
            _append_event(
                event_log_path,
                {
                    "event_kind": "phase",
                    "phase": "cleanup",
                    "round_index": round_index,
                    "timestamp": time.time(),
                },
                config=config,
            )
            cleanup = _governed_cleanup_runtime_state(
                config,
                keep_reports=max(0, args.keep_reports),
                keep_candidate_dirs=max(0, args.keep_candidate_dirs),
                keep_checkpoints=max(0, args.keep_checkpoints),
                keep_snapshot_entries=max(0, args.keep_snapshot_entries),
                keep_tolbert_dataset_dirs=max(0, args.keep_tolbert_dataset_dirs),
                keep_tolbert_candidate_dirs=max(0, args.keep_tolbert_candidate_dirs),
                tolbert_candidate_budget_bytes=max(0, int(float(args.tolbert_candidate_budget_gib) * (1024**3))),
                tolbert_shared_store_budget_bytes=max(0, int(float(args.tolbert_shared_store_budget_gib) * (1024**3))),
                keep_tolbert_promoted_checkpoints=max(0, args.keep_tolbert_promoted_checkpoints),
                tolbert_promoted_checkpoint_budget_bytes=max(
                    0,
                    int(float(args.tolbert_promoted_checkpoint_budget_gib) * (1024**3)),
                ),
                target_free_gib=max(0.0, float(args.cleanup_target_free_disk_gib)),
            )
            round_payload["cleanup"] = cleanup
            report["cleanup"] = cleanup
            report["global_storage"] = _governed_global_storage_cleanup(
                global_storage_root,
                policy_entries=global_storage_policy,
                target_free_gib=max(0.0, float(args.global_storage_target_free_gib)),
                top_k=max(1, int(args.global_storage_top_k)),
            )
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)

            phase = "campaign"
            report["phase"] = phase
            round_payload["phase"] = phase
            round_run_match_id = f"unattended:round:{run_id}:{round_index}"
            report["active_run"] = {
                "run_index": round_index,
                "run_match_id": round_run_match_id,
                "phase": phase,
                "policy": dict(current_policy),
            }
            _progress(
                f"[campaign] phase=campaign round={round_index} cycles={int(current_policy['cycles'])} "
                f"task_limit={int(current_policy['task_limit'])} "
                f"task_step_floor={int(current_policy['task_step_floor'])} focus={current_policy['focus']} "
                f"tolbert_device={current_policy['tolbert_device']}"
            )
            _append_event(
                event_log_path,
                {
                    "event_kind": "phase",
                    "phase": "campaign",
                    "round_index": round_index,
                    "policy": dict(current_policy),
                    "timestamp": time.time(),
                },
                config=config,
            )
            campaign_cmd = [
                sys.executable,
                "-u",
                str(repeated_script),
                "--cycles",
                str(int(current_policy["cycles"])),
                "--campaign-width",
                str(int(current_policy["campaign_width"])),
                "--variant-width",
                str(int(current_policy["variant_width"])),
                "--task-limit",
                str(int(current_policy["task_limit"])),
                "--campaign-label",
                f"unattended-campaign-{run_id}-round-{round_index}",
                "--tolbert-device",
                str(current_policy["tolbert_device"]),
                "--include-episode-memory",
                "--include-skill-memory",
                "--include-skill-transfer",
                "--include-operator-memory",
                "--include-tool-memory",
                "--include-verifier-memory",
                "--include-curriculum",
                "--include-failure-curriculum",
            ]
            for family in _normalize_benchmark_families(current_policy.get("priority_benchmark_families", [])):
                campaign_cmd.extend(["--priority-benchmark-family", family])
            for excluded_subsystem in _normalize_excluded_subsystems(current_policy.get("excluded_subsystems", [])):
                campaign_cmd.extend(["--exclude-subsystem", excluded_subsystem])
            if bool(current_policy["adaptive_search"]):
                campaign_cmd.append("--adaptive-search")
            if args.provider:
                campaign_cmd.extend(["--provider", args.provider])
            if args.model:
                campaign_cmd.extend(["--model", args.model])
            campaign_env = dict(env)
            campaign_env["AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR"] = str(int(current_policy["task_step_floor"]))
            campaign_env["AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH"] = str(status_path)
            campaign_env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX"] = str(round_index)
            campaign_env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID"] = round_run_match_id
            child_improvement_reports_dir = _child_improvement_reports_dir(report_path)
            campaign_env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"] = str(child_improvement_reports_dir)
            campaign_env["AGENT_KERNEL_RUNTIME_DATABASE_PATH"] = str(
                child_improvement_reports_dir / "agentkernel.sqlite3"
            )
            campaign_progress = _make_child_progress_callback(
                report_path=report_path,
                report=report,
                status_path=status_path,
                lock_path=lock_path,
                config=config,
                round_payload=round_payload,
                round_index=round_index,
                phase=phase,
                child_label=f"campaign_round_{round_index}",
                event_log_path=event_log_path,
            )
            campaign_run = _run_and_stream(
                campaign_cmd,
                cwd=repo_root,
                env=campaign_env,
                progress_label=f"campaign_round_{round_index}",
                heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                max_silence_seconds=float(args.max_child_silence_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                on_event=campaign_progress,
                mirrored_status_path=status_path,
            )
            round_payload["campaign_run"] = campaign_run
            campaign_report_path = _last_report_path(str(campaign_run["stdout"]))
            round_payload["campaign_report_path"] = "" if campaign_report_path is None else str(campaign_report_path)
            report["campaign_run"] = campaign_run
            report["campaign_report_path"] = round_payload["campaign_report_path"]
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            if int(campaign_run["returncode"]) != 0:
                mirrored_child_status = _mirrored_child_status_from_parent_status(status_path)
                partial_acceptance = _accept_productive_partial_child_timeout(
                    campaign_run,
                    mirrored_child_status=mirrored_child_status,
                    round_payload=round_payload,
                )
                round_payload["campaign_partial_acceptance"] = partial_acceptance
                if bool(partial_acceptance.get("accepted", False)):
                    mirrored_child_status = _credit_accepted_productive_partial_status(mirrored_child_status)
                    partial_report_path = str(partial_acceptance.get("report_path", "")).strip()
                    if partial_report_path:
                        campaign_report_path = Path(partial_report_path)
                        campaign_report = _read_json(campaign_report_path)
                        round_payload["campaign_report_path"] = str(campaign_report_path)
                    else:
                        campaign_report = _synthetic_productive_partial_campaign_report(
                            mirrored_child_status=mirrored_child_status,
                            campaign_run=campaign_run,
                            round_payload=round_payload,
                        )
                        round_payload["campaign_report_path"] = ""
                    campaign_report = _normalize_accepted_productive_partial_campaign_report(
                        campaign_report,
                        campaign_run=campaign_run,
                        round_payload=round_payload,
                    )
                    mirrored_child_status = _finalize_accepted_productive_partial_child_status(
                        mirrored_child_status,
                        campaign_report=campaign_report,
                        campaign_run=campaign_run,
                    )
                    atomic_write_json(
                        _child_improvement_reports_dir(report_path) / "repeated_improvement_status.json",
                        mirrored_child_status,
                        config=config,
                    )
                    active_child = round_payload.get("active_child", {})
                    if isinstance(active_child, Mapping):
                        finalized_active_child = _finalize_accepted_productive_partial_child_status(
                            active_child,
                            campaign_report=campaign_report,
                            campaign_run=campaign_run,
                        )
                        round_payload["active_child"] = finalized_active_child
                        report["active_child"] = finalized_active_child
                    round_payload["campaign_report"] = campaign_report
                    round_payload["campaign_validation"] = {
                        "passed": True,
                        "detail": f"accepted productive partial child timeout: {partial_acceptance['reason']}",
                        "accepted_partial_timeout": True,
                    }
                    round_payload["campaign_warning"] = str(campaign_run.get("timeout_reason", "")).strip()
                    round_payload["liftoff_skipped"] = {
                        "mode": current_policy["liftoff"],
                        "reason": "accepted productive partial timeout bypassed liftoff",
                    }
                    report["campaign_report_path"] = round_payload["campaign_report_path"]
                    report["campaign_report"] = campaign_report
                    report["campaign_validation"] = round_payload["campaign_validation"]
                    report["liftoff_skipped"] = round_payload["liftoff_skipped"]
                    report["phase_detail"] = round_payload["campaign_warning"]
                    round_planner_controls = _load_retained_improvement_planner_controls(config)
                    controller_state, current_policy, no_yield_rounds, policy_stall_rounds, depth_runway_credit = (
                        _finalize_completed_campaign_round(
                            args=args,
                            config=config,
                            report_path=report_path,
                            status_path=status_path,
                            lock_path=lock_path,
                            event_log_path=event_log_path,
                            controller_state_path=controller_state_path,
                            run_id=run_id,
                            round_index=round_index,
                            report=report,
                            round_payload=round_payload,
                            campaign_report=campaign_report,
                            liftoff_payload=None,
                            controller_state=controller_state,
                            current_policy=current_policy,
                            round_planner_controls=round_planner_controls,
                            round_curriculum_controls=round_curriculum_controls,
                            no_yield_rounds=no_yield_rounds,
                            policy_stall_rounds=policy_stall_rounds,
                            depth_runway_credit=depth_runway_credit,
                        )
                    )
                    _emit_policy_shift_alert_if_configured(round_payload)
                    round_index += 1
                    continue
                failure_reason = str(campaign_run.get("timeout_reason", "")) or "campaign subprocess failed"
                _recover_child_failure_or_stop(
                    round_payload=round_payload,
                    phase=phase,
                    reason=failure_reason,
                )
                round_index += 1
                continue

            campaign_report = _normalize_accepted_productive_partial_campaign_report(
                _read_json(campaign_report_path),
                campaign_run=campaign_run,
                round_payload=round_payload,
            )
            round_payload["campaign_report"] = campaign_report
            report["campaign_report"] = campaign_report
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            campaign_validation = _validate_campaign_report(campaign_report)
            round_payload["campaign_validation"] = campaign_validation
            report["campaign_validation"] = campaign_validation
            if not bool(campaign_validation["passed"]):
                _recover_child_failure_or_stop(
                    round_payload=round_payload,
                    phase=phase,
                    reason=str(campaign_validation["detail"]),
                )
                round_index += 1
                continue

            liftoff_payload: dict[str, object] | None = None
            liftoff_report_path: Path | None = None
            if _should_run_liftoff(str(current_policy["liftoff"]), campaign_report):
                phase = "liftoff"
                report["phase"] = phase
                round_payload["phase"] = phase
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _progress(
                    f"[campaign] phase=liftoff round={round_index} focus={current_policy['focus']} "
                    f"task_limit={int(current_policy['task_limit'])}"
                )
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "phase",
                        "phase": "liftoff",
                        "round_index": round_index,
                        "policy": dict(current_policy),
                        "timestamp": time.time(),
                    },
                    config=config,
                )
                liftoff_cmd = [
                    sys.executable,
                    "-u",
                    str(liftoff_script),
                    "--skip-improvement",
                    "--task-limit",
                    str(int(current_policy["task_limit"])),
                    "--tolbert-device",
                    str(current_policy["tolbert_device"]),
                    "--focus",
                    str(current_policy["focus"]),
                ]
                for family in _normalize_benchmark_families(current_policy.get("priority_benchmark_families", [])):
                    liftoff_cmd.extend(["--priority-benchmark-family", family])
                for excluded_subsystem in _normalize_excluded_subsystems(current_policy.get("excluded_subsystems", [])):
                    liftoff_cmd.extend(["--exclude-subsystem", excluded_subsystem])
                if args.apply_routing:
                    liftoff_cmd.append("--apply-routing")
                if args.promote_on_retain:
                    liftoff_cmd.append("--promote-on-retain")
                if args.provider:
                    liftoff_cmd.extend(["--provider", args.provider])
                if args.model:
                    liftoff_cmd.extend(["--model", args.model])
                liftoff_env = dict(env)
                liftoff_env["AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR"] = str(int(current_policy["task_step_floor"]))
                liftoff_progress = _make_child_progress_callback(
                    report_path=report_path,
                    report=report,
                    status_path=status_path,
                    lock_path=lock_path,
                    config=config,
                    round_payload=round_payload,
                    round_index=round_index,
                    phase=phase,
                    child_label=f"liftoff_round_{round_index}",
                    event_log_path=event_log_path,
                )
                liftoff_run = _run_and_stream(
                    liftoff_cmd,
                    cwd=repo_root,
                    env=liftoff_env,
                    progress_label=f"liftoff_round_{round_index}",
                    heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                    max_silence_seconds=float(args.max_child_silence_seconds),
                    max_runtime_seconds=float(args.max_child_runtime_seconds),
                    max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                    on_event=liftoff_progress,
                    mirrored_status_path=status_path,
                )
                round_payload["liftoff_run"] = liftoff_run
                liftoff_report_path = _last_report_path(str(liftoff_run["stdout"]))
                round_payload["liftoff_report_path"] = "" if liftoff_report_path is None else str(liftoff_report_path)
                report["liftoff_run"] = liftoff_run
                report["liftoff_report_path"] = round_payload["liftoff_report_path"]
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                if int(liftoff_run["returncode"]) != 0:
                    failure_reason = str(liftoff_run.get("timeout_reason", "")) or "liftoff subprocess failed"
                    _recover_child_failure_or_stop(
                        round_payload=round_payload,
                        phase=phase,
                        reason=failure_reason,
                    )
                    round_index += 1
                    continue
                liftoff_payload = _read_json(liftoff_report_path)
                round_payload["liftoff_report"] = liftoff_payload
                report["liftoff_report"] = liftoff_payload
                liftoff_validation = _validate_liftoff_report(liftoff_payload)
                round_payload["liftoff_validation"] = liftoff_validation
                report["liftoff_validation"] = liftoff_validation
                if not bool(liftoff_validation["passed"]):
                    _recover_child_failure_or_stop(
                        round_payload=round_payload,
                        phase=phase,
                        reason=str(liftoff_validation["detail"]),
                    )
                    round_index += 1
                    continue
            else:
                round_payload["liftoff_skipped"] = {
                    "mode": current_policy["liftoff"],
                    "reason": "campaign did not satisfy auto liftoff trigger",
                }
                report["liftoff_skipped"] = round_payload["liftoff_skipped"]

            round_planner_controls = _load_retained_improvement_planner_controls(config)
            controller_state, current_policy, no_yield_rounds, policy_stall_rounds, depth_runway_credit = (
                _finalize_completed_campaign_round(
                    args=args,
                    config=config,
                    report_path=report_path,
                    status_path=status_path,
                    lock_path=lock_path,
                    event_log_path=event_log_path,
                    controller_state_path=controller_state_path,
                    run_id=run_id,
                    round_index=round_index,
                    report=report,
                    round_payload=round_payload,
                    campaign_report=campaign_report,
                    liftoff_payload=liftoff_payload,
                    controller_state=controller_state,
                    current_policy=current_policy,
                    round_planner_controls=round_planner_controls,
                    round_curriculum_controls=round_curriculum_controls,
                    no_yield_rounds=no_yield_rounds,
                    policy_stall_rounds=policy_stall_rounds,
                    depth_runway_credit=depth_runway_credit,
                )
            )
            _emit_policy_shift_alert_if_configured(round_payload)

            liftoff_state = _liftoff_signal(liftoff_payload)["state"] if liftoff_payload else ""
            if args.stop_on_liftoff_retain and liftoff_state == "retain":
                report["status"] = "completed"
                report["phase"] = "completed"
                report["reason"] = "liftoff retain gate cleared"
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "campaign_complete",
                        "reason": "liftoff retain gate cleared",
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                    config=config,
                )
                print(report_path)
                return
            effective_stop_budget = (
                round_payload["adaptive_stop_budget"]
                if isinstance(round_payload.get("adaptive_stop_budget", {}), dict)
                else {}
            )
            if no_yield_rounds > int(effective_stop_budget.get("max_no_yield_rounds_effective", max(0, args.max_no_yield_rounds))):
                report["status"] = "safe_stop"
                report["phase"] = "completed"
                report["reason"] = "repeated no-yield rounds exceeded unattended policy"
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "safe_stop",
                        "reason": report["reason"],
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                    config=config,
                )
                print(report_path)
                raise SystemExit(2)
            if policy_stall_rounds > int(
                effective_stop_budget.get(
                    "max_policy_stall_rounds_effective",
                    max(0, args.max_policy_stall_rounds),
                )
            ):
                report["status"] = "safe_stop"
                report["phase"] = "completed"
                report["reason"] = "unattended policy stalled without discovering a new outer-loop action"
                _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "safe_stop",
                        "reason": report["reason"],
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                    config=config,
                )
                print(report_path)
                raise SystemExit(2)
            round_index += 1

        if report["rounds_completed"] < requested_rounds:
            report["status"] = "safe_stop"
            report["phase"] = "completed"
            report["reason"] = "child-failure recovery budget exhausted before completing requested unattended rounds"
            _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
            _emit_alert_if_configured()
            _append_event(
                event_log_path,
                {
                    "event_kind": "safe_stop",
                    "reason": report["reason"],
                    "rounds_completed": report["rounds_completed"],
                    "recovery_attempts": child_failure_recoveries_used,
                    "timestamp": time.time(),
                },
                config=config,
            )
            print(report_path)
            raise SystemExit(2)

        report["status"] = "completed"
        report["phase"] = "completed"
        report["reason"] = "requested unattended rounds completed"
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_complete",
                "reason": report["reason"],
                "rounds_completed": report["rounds_completed"],
                "timestamp": time.time(),
            },
            config=config,
        )
        print(report_path)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["phase"] = phase
        interrupt_reason = (
            f"received signal {signal.Signals(received_signal['value']).name}"
            if int(received_signal["value"]) > 0
            else "operator interrupted unattended campaign"
        )
        report["reason"] = interrupt_reason
        if report.get("rounds"):
            last_round = report["rounds"][-1]
            if isinstance(last_round, dict) and str(last_round.get("status", "")).strip() == "running":
                _apply_failure_feedback(last_round, phase=phase, reason=interrupt_reason)
                _mark_round(
                    last_round,
                    status="interrupted",
                    phase=phase,
                    reason=interrupt_reason,
                )
        report["rounds_completed"] = _count_completed_rounds(
            report.get("rounds", []) if isinstance(report.get("rounds", []), list) else []
        )
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_interrupted",
                "phase": phase,
                "timestamp": time.time(),
            },
            config=config,
        )
        print(report_path)
        raise SystemExit(130)
    except Exception as exc:
        report["status"] = "safe_stop"
        report["phase"] = phase
        report["reason"] = f"unhandled exception during {phase}: {type(exc).__name__}: {exc}"
        report["unexpected_exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().strip(),
        }
        if report.get("rounds"):
            last_round = report["rounds"][-1]
            if isinstance(last_round, dict) and str(last_round.get("status", "")).strip() == "running":
                _apply_failure_feedback(last_round, phase=phase, reason=str(report["reason"]))
                _mark_round(
                    last_round,
                    status="safe_stop",
                    phase=phase,
                    reason=str(report["reason"]),
                )
        report["rounds_completed"] = _count_completed_rounds(
            report.get("rounds", []) if isinstance(report.get("rounds", []), list) else []
        )
        _persist_report_state(report_path, report, config=config, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_failed",
                "phase": phase,
                "exception_type": type(exc).__name__,
                "timestamp": time.time(),
            },
            config=config,
        )
        print(report_path)
        raise SystemExit(2) from exc
    finally:
        restore_signal_handlers()
        if str(args.lease_backend or "local").strip() == "http":
            _release_external_campaign_lease(str(args.lease_endpoint or "").strip(), token=str(args.lease_token or ""))
        else:
            _release_campaign_lock(lock_path, pid=os.getpid())


if __name__ == "__main__":
    main()
