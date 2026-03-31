from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import argparse
import fcntl
import json
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
from agent_kernel.runtime_supervision import (
    append_jsonl,
    atomic_write_json,
    install_termination_handlers,
    spawn_process_group,
    terminate_process_tree,
)
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
_PRIORITY_FAMILY_LOW_RETURN_MIN_DECISIONS = 2
_PRIORITY_FAMILY_LOW_RETURN_MIN_ESTIMATED_COST = 4.0
_PRIORITY_FAMILY_EXPLORATION_WEIGHT = 0.05


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    atomic_write_json(report_path, payload)


def _cleanup_bootstrap_artifacts(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            continue


def _write_status(status_path: Path | None, *, report_path: Path, payload: dict[str, object]) -> None:
    if status_path is None:
        return
    active_child = payload.get("active_child", {})
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
    status_payload = {
        "spec_version": "asi_v1",
        "report_kind": "unattended_campaign_status",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "status": str(payload.get("status", "")).strip(),
        "phase": str(payload.get("phase", "")).strip(),
        "phase_detail": str(payload.get("phase_detail", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "rounds_requested": int(payload.get("rounds_requested", 0) or 0),
        "rounds_completed": int(payload.get("rounds_completed", 0) or 0),
        "child_failure_recovery_budget": int(payload.get("child_failure_recovery_budget", 0) or 0),
        "child_failure_recoveries_used": int(payload.get("child_failure_recoveries_used", 0) or 0),
        "current_policy": payload.get("current_policy", {}),
        "policy_shift_summary": payload.get("policy_shift_summary", {}),
        "policy_shift_alert_subscriptions": policy_shift_subscriptions,
        "latest_round_policy_shift_rationale": latest_round.get("policy_shift_rationale", {})
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
        "active_child": active_child if isinstance(active_child, dict) else {},
        "preflight": preflight if isinstance(preflight, dict) else {},
        "cleanup": cleanup if isinstance(cleanup, dict) else {},
        "unattended_evidence": payload.get("unattended_evidence", {}),
        "global_storage": payload.get("global_storage", {}),
        "lock_path": str(payload.get("lock_path", "")).strip(),
        "event_log_path": str(payload.get("event_log_path", "")).strip(),
    }
    atomic_write_json(status_path, status_payload)


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
    priority_family_selection_scores: dict[str, object] | None = None,
    min_selection_score: float = 0.0,
    productive_priority_families: object | None = None,
    under_sampled_priority_families: object | None = None,
    low_return_priority_families: object | None = None,
) -> list[str]:
    ranked_missing = _rank_priority_benchmark_families(missing_required_families)
    if ranked_missing:
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
    ranked_low_return = [
        family for family in _rank_priority_benchmark_families(low_return_priority_families or []) if family not in productive_set
    ]
    low_return_set = set(ranked_low_return)
    ranked_under_sampled = [
        family
        for family in _rank_priority_benchmark_families(under_sampled_priority_families or [])
        if family not in productive_set and family not in low_return_set
    ]
    ranked_required = _rank_priority_benchmark_families(required_families)
    ranked_required_without_productive = [
        family for family in ranked_required if family not in productive_set and family not in low_return_set
    ]
    ranked_current = [
        family
        for family in _rank_priority_benchmark_families(current_priority_families)
        if family not in productive_set and family not in low_return_set
    ]
    selected: list[str] = []
    for family in [
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


def _priority_family_history_summary(
    *,
    prior_rounds: list[dict[str, object]],
    current_campaign_report: dict[str, object],
    current_priority_families: object,
    planner_controls: dict[str, object] | None = None,
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
        summary["selection_return_on_cost"] = return_on_cost
        summary["selection_adjusted_return_on_cost"] = adjusted_return_on_cost
        summary["selection_gain_multiplier"] = gain_multiplier
        summary["selection_cost_multiplier"] = cost_multiplier
        summary["selection_score_bias"] = score_bias
        summary["selection_exploration_bonus"] = exploration_bonus
        summary["selection_score"] = adjusted_return_on_cost + exploration_bonus + score_bias
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
        "priority_family_categories": priority_family_categories,
        "priority_family_scoring_policy": priority_family_scoring_policy,
        "priority_family_return_on_cost_scores": priority_family_return_on_cost_scores,
        "priority_family_selection_scores": priority_family_selection_scores,
        "priority_families_ranked_by_return_on_cost": priority_families_ranked_by_return_on_cost,
        "priority_families_ranked_by_selection_score": priority_families_ranked_by_selection_score,
        "low_return_min_decisions": _PRIORITY_FAMILY_LOW_RETURN_MIN_DECISIONS,
        "low_return_min_estimated_cost": _PRIORITY_FAMILY_LOW_RETURN_MIN_ESTIMATED_COST,
    }


def _append_event(event_log_path: Path | None, payload: dict[str, object]) -> None:
    if event_log_path is None:
        return
    append_jsonl(event_log_path, payload)


def _write_controller_state(path: Path, payload: dict[str, object]) -> None:
    atomic_write_json(path, payload)


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
        atomic_write_json(emergency_path, emergency_payload)
    except OSError:
        return None
    return emergency_path


def _persist_report_state(
    report_path: Path,
    payload: dict[str, object],
    *,
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
        _write_report(report_path, payload)
        _write_status(status_path, report_path=report_path, payload=payload)
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
            status_path=status_path,
            error=exc,
        )
        if emergency_path is not None:
            payload["emergency_report_path"] = str(emergency_path)
            try:
                _write_status(status_path, report_path=emergency_path, payload=payload)
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
            active_child.get("last_progress_line") or active_child.get("last_output_line") or ""
        ).strip()
    return _subsystem_from_progress_line(progress_line or phase_detail)


def _child_failure_recovery_policy(
    current_policy: dict[str, object],
    *,
    round_payload: dict[str, object] | None,
    phase: str,
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
) -> dict[str, object]:
    next_policy = _advance_subsystem_cooldowns(current_policy)
    stalled_subsystem = _stalled_subsystem_from_round(round_payload)
    if stalled_subsystem:
        next_policy = _apply_subsystem_cooldown(next_policy, subsystem=stalled_subsystem, rounds=2)
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
    before_snapshot = _global_storage_snapshot(root, top_k=top_k, managed_entries=policy_entries)
    cleanup: dict[str, object] = {"removed_entries": []}
    after_disk = before_disk
    after_snapshot = before_snapshot
    if policy_entries and float(before_disk.get("free_gib", 0.0)) < max(0.0, float(target_free_gib)):
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


def _record_alert_state(state_path: Path | None, *, payload: dict[str, object], repeat_count: int) -> None:
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
    _record_alert_state(state_path, payload=payload, repeat_count=repeat_count)
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
) -> dict[str, object]:
    def _emit_event(event: dict[str, object]) -> None:
        if on_event is not None:
            on_event(event)

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
                    print(line, end="", file=sys.stderr, flush=True)
                    _emit_event(
                        {
                            "event": "output",
                            "line": line.rstrip("\n"),
                            "pid": process_pid,
                            "progress_label": progress_label or Path(cmd[-1]).name,
                            "timestamp": time.time(),
                        }
                    )
            elif process.poll() is not None:
                break
            now = time.monotonic()
            silence = now - last_output_at
            progress_stall = now - last_progress_at
            runtime_elapsed = now - started_at
            if heartbeat_interval > 0.0 and (now - last_heartbeat_at) >= heartbeat_interval and silence >= heartbeat_interval:
                label = str(progress_label).strip() or Path(cmd[1]).name
                _progress(f"[supervisor] child={label} still_running silence={int(silence)}s")
                _emit_event(
                    {
                        "event": "heartbeat",
                        "pid": process_pid,
                        "progress_label": label,
                        "silence_seconds": int(silence),
                        "timestamp": time.time(),
                    }
                )
                last_heartbeat_at = now
            if max_runtime > 0.0 and runtime_elapsed >= max_runtime:
                terminate_process_tree(process)
                _emit_event(
                    {
                        "event": "timeout",
                        "pid": process_pid,
                        "progress_label": str(progress_label).strip() or Path(cmd[-1]).name,
                        "runtime_seconds": int(runtime_elapsed),
                        "timestamp": time.time(),
                        "timeout_reason": f"child exceeded max runtime of {int(max_runtime)} seconds",
                    }
                )
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max runtime of {int(max_runtime)} seconds",
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


def _is_significant_child_output(line: str) -> bool:
    normalized = str(line).strip()
    if not normalized:
        return False
    return normalized.startswith(("[campaign]", "[cycle:", "[eval:", "[repeated]", "[supervisor]")) or "finalize phase=" in normalized


def _make_child_progress_callback(
    *,
    report_path: Path,
    report: dict[str, object],
    status_path: Path | None,
    lock_path: Path | None,
    round_payload: dict[str, object],
    round_index: int,
    phase: str,
    child_label: str,
    event_log_path: Path | None = None,
    min_write_interval_seconds: float = 5.0,
) -> Callable[[dict[str, object]], None]:
    last_write_at = 0.0

    def _persist_if_needed(event: dict[str, object]) -> None:
        nonlocal last_write_at
        event_name = str(event.get("event", "")).strip() or "unknown"
        timestamp = float(event.get("timestamp", time.time()) or time.time())
        active_child = dict(report.get("active_child", {}))
        previous_progress_line = str(active_child.get("last_progress_line", "")).strip()
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
                candidate_path = Path(line)
                if candidate_path.exists():
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
        elif event_name == "exit":
            active_child["state"] = "completed"
            active_child["ended_at"] = timestamp
            active_child["returncode"] = int(event.get("returncode", 0) or 0)

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
        )
        detail = str(active_child.get("last_progress_line") or active_child.get("last_output_line") or "").strip()
        if detail:
            report["phase_detail"] = detail
            round_payload["phase_detail"] = detail

        now = time.monotonic()
        significant_output = event_name == "output" and _is_significant_child_output(str(event.get("line", "")))
        output_progress_changed = significant_output and (
            str(active_child.get("last_progress_line", "")).strip() != previous_progress_line
        )
        should_persist = event_name in {"start", "heartbeat", "timeout", "exit"} or significant_output
        if not should_persist:
            return
        if event_name == "output" and not output_progress_changed and (now - last_write_at) < max(0.0, float(min_write_interval_seconds)):
            return
        last_write_at = now
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)

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
    if float(after_disk.get("free_gib", 0.0)) < max(0.0, float(target_free_gib)):
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
        "priority_benchmark_families": _select_priority_benchmark_families(
            required_families=requested_priority_families or list(config.unattended_trust_required_benchmark_families),
            missing_required_families=requested_priority_families,
            current_priority_families=requested_priority_families,
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
    runs = campaign_report.get("runs", [])
    if not isinstance(runs, list):
        runs = []
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
        "retained_cycles": int(production.get("retained_cycles", 0) or 0),
        "rejected_cycles": int(production.get("rejected_cycles", 0) or 0),
        "average_retained_pass_rate_delta": float(production.get("average_retained_pass_rate_delta", 0.0) or 0.0),
        "average_retained_step_delta": float(production.get("average_retained_step_delta", 0.0) or 0.0),
        "all_retained_phase_gates_passed": bool(phase_gate.get("all_retained_phase_gates_passed", True)),
        "failed_decisions": int(phase_gate.get("failed_decisions", 0) or 0),
        "runtime_managed_decisions": int(
            campaign_report.get("inheritance_summary", {}).get("runtime_managed_decisions", 0)
            if isinstance(campaign_report.get("inheritance_summary", {}), dict)
            else 0
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
    cooldown_candidates = [
        subsystem
        for subsystem, rejected in rejected_by_subsystem.items()
        if rejected > retained_by_subsystem.get(subsystem, 0) and averaged.get(subsystem, 0.0) <= 0.0
    ]
    return {
        "retained_by_subsystem": retained_by_subsystem,
        "rejected_by_subsystem": rejected_by_subsystem,
        "average_pass_delta_by_subsystem": averaged,
        "cooldown_candidates": sorted(cooldown_candidates),
    }


def _planner_pressure_signal(campaign_report: dict[str, object]) -> dict[str, object]:
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
    return {
        "campaign_breadth_pressure_cycles": int(payload.get("campaign_breadth_pressure_cycles", 0) or 0),
        "variant_breadth_pressure_cycles": int(payload.get("variant_breadth_pressure_cycles", 0) or 0),
        "pressured_subsystems": pressured_subsystems,
        "dominant_subsystem": pressured_subsystems[0] if pressured_subsystems else "",
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


def _controller_observation(
    *,
    campaign_report: dict[str, object] | None,
    liftoff_payload: dict[str, object] | None,
) -> dict[str, object]:
    campaign_signal = _campaign_signal(campaign_report or {})
    return build_round_observation(
        campaign_signal=campaign_signal,
        subsystem_signal=_subsystem_signal(campaign_report or {}),
        planner_pressure_signal=_planner_pressure_signal(campaign_report or {}),
        liftoff_signal=_liftoff_signal(liftoff_payload or {}) if liftoff_payload else None,
    )


def _candidate_round_policies(
    current_policy: dict[str, object],
    *,
    campaign_report: dict[str, object],
    liftoff_payload: dict[str, object] | None,
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
) -> list[dict[str, object]]:
    base = _advance_subsystem_cooldowns(current_policy)
    signal = _campaign_signal(campaign_report)
    liftoff = _liftoff_signal(liftoff_payload or {}) if liftoff_payload else {}
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
    current_campaign_width = max(1, int(base.get("campaign_width", 1) or 1))
    current_variant_width = max(1, int(base.get("variant_width", 1) or 1))
    focuses = ["balanced", "recovery_alignment", "discovered_task_adaptation"]
    adaptive_options = [False, True]
    if severe_regression:
        focuses = ["recovery_alignment", "balanced"]
        adaptive_options = [True]
    elif low_confidence_pressure or retained_cycles <= 0:
        focuses = ["discovered_task_adaptation", "balanced"]
        adaptive_options = [True]
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
    campaign_width_options = sorted(
        {
            max(1, min(max_campaign_width, current_campaign_width)),
            max(1, min(max_campaign_width, current_campaign_width + 1)),
        }
    )
    variant_width_options = sorted(
        {
            max(1, min(max_variant_width, current_variant_width)),
            max(1, min(max_variant_width, current_variant_width + 1)),
        }
    )
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for focus in focuses:
        for adaptive in adaptive_options:
            for cycles in cycle_options:
                for task_limit in task_limit_options:
                    for campaign_width in campaign_width_options:
                        for variant_width in variant_width_options:
                            proposal = dict(base)
                            proposal["focus"] = focus
                            proposal["adaptive_search"] = adaptive
                            proposal["cycles"] = cycles
                            proposal["task_limit"] = task_limit
                            proposal["campaign_width"] = campaign_width
                            proposal["variant_width"] = variant_width
                            if focus == "recovery_alignment":
                                proposal["cycles"] = min(max_cycles, max(proposal["cycles"], current_cycles))
                                proposal["adaptive_search"] = True
                            if focus == "discovered_task_adaptation":
                                proposal["task_limit"] = min(max_task_limit, max(proposal["task_limit"], current_task_limit))
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
    max_cycles: int,
    max_task_limit: int,
    max_campaign_width: int,
    max_variant_width: int,
) -> dict[str, object]:
    observation = _controller_observation(campaign_report=campaign_report, liftoff_payload=liftoff_payload)
    candidate_policies = _candidate_round_policies(
        current_policy,
        campaign_report=campaign_report,
        liftoff_payload=liftoff_payload,
        max_cycles=max_cycles,
        max_task_limit=max_task_limit,
        max_campaign_width=max_campaign_width,
        max_variant_width=max_variant_width,
    )
    next_policy, planner_diagnostics = plan_next_policy(
        controller_state,
        current_observation=observation,
        candidate_policies=candidate_policies,
    )
    historical_adjustments = _historical_policy_rationale_adjustments(list(prior_rounds or []))
    adjusted_candidates: list[dict[str, object]] = []
    for candidate in planner_diagnostics.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        policy = candidate.get("policy", {})
        action_key = action_key_for_policy(policy if isinstance(policy, dict) else {})
        rationale_adjustment = float(historical_adjustments.get(action_key, 0.0) or 0.0)
        adjusted_score = float(candidate.get("score", 0.0) or 0.0) + rationale_adjustment
        adjusted_candidate = dict(candidate)
        adjusted_candidate["historical_rationale_adjustment"] = rationale_adjustment
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
            planner_diagnostics["candidates"] = adjusted_candidates[:12]
    if isinstance(round_payload, dict):
        round_payload["controller_planner"] = planner_diagnostics
    signal = _campaign_signal(campaign_report)
    subsystem_signal = _subsystem_signal(campaign_report)
    planner_pressure_signal = _planner_pressure_signal(campaign_report)
    retained_cycles = int(signal["retained_cycles"])
    rejected_cycles = int(signal["rejected_cycles"])
    avg_retained_pass_delta = float(signal["average_retained_pass_rate_delta"])
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
    required_families = _normalize_benchmark_families(signal.get("required_families", []))
    missing_required_families = _normalize_benchmark_families(signal.get("missing_required_families", []))
    distinct_family_gap = int(signal.get("distinct_family_gap", 0) or 0)
    required_family_coverage_gap = bool(missing_required_families) or distinct_family_gap > 0
    priority_families = _normalize_benchmark_families(
        signal.get("priority_families", current_policy.get("priority_benchmark_families", []))
    )
    priority_family_history = _priority_family_history_summary(
        prior_rounds=list(prior_rounds or []),
        current_campaign_report=campaign_report,
        current_priority_families=priority_families,
        planner_controls=planner_controls,
    )
    priority_families_with_retained_gain = _normalize_benchmark_families(
        priority_family_history.get("priority_families_with_retained_gain", [])
    )
    priority_families_without_signal = _normalize_benchmark_families(
        priority_family_history.get("priority_families_without_signal", [])
    )
    priority_families_under_sampled = _normalize_benchmark_families(
        priority_family_history.get("priority_families_under_sampled", [])
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
    priority_family_under_sampled_gap = bool(priority_families) and bool(priority_families_under_sampled)
    current_cycles = max(1, int(next_policy.get("cycles", 1) or 1))
    current_task_limit = max(0, int(next_policy.get("task_limit", 0) or 0))
    current_campaign_width = max(1, int(next_policy.get("campaign_width", 1) or 1))
    current_variant_width = max(1, int(next_policy.get("variant_width", 1) or 1))
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

    for subsystem in subsystem_signal["cooldown_candidates"][:2]:
        next_policy = _apply_subsystem_cooldown(next_policy, subsystem=subsystem, rounds=2)

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
        priority_family_selection_scores=priority_family_selection_scores,
        min_selection_score=float(priority_family_scoring_policy.get("priority_family_min_selection_score", 0.0) or 0.0),
        productive_priority_families=priority_families_with_retained_gain,
        under_sampled_priority_families=priority_families_under_sampled,
        low_return_priority_families=priority_families_low_return,
    )
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
        for key in ("cycles", "task_limit", "campaign_width", "variant_width"):
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
        if priority_families_low_return:
            reason_codes.append("priority_family_low_return")
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
                "dominant_subsystem": str(planner_pressure_signal.get("dominant_subsystem", "")).strip(),
                "external_report_count": external_report_count,
                "distinct_external_benchmark_families": distinct_external_benchmark_families,
                "missing_required_families": missing_required_families,
                "distinct_family_gap": distinct_family_gap,
                "priority_families": priority_families,
                "priority_families_with_retained_gain": priority_families_with_retained_gain,
                "priority_families_without_signal": priority_families_without_signal,
                "priority_families_under_sampled": priority_families_under_sampled,
                "priority_families_low_return": priority_families_low_return,
                "priority_families_ranked_by_selection_score": priority_families_ranked_by_selection_score,
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
    for round_payload in prior_rounds:
        if not isinstance(round_payload, dict):
            continue
        campaign_report = round_payload.get("campaign_report", {})
        if not isinstance(campaign_report, dict):
            continue
        production = campaign_report.get("production_yield_summary", {})
        if isinstance(production, dict) and int(production.get("retained_cycles", 0) or 0) <= 0:
            no_yield_rounds += 1
        else:
            no_yield_rounds = 0
        if (
            isinstance(round_payload.get("policy", {}), dict)
            and "next_policy" in round_payload
            and isinstance(round_payload.get("next_policy", {}), dict)
        ):
            if dict(round_payload["policy"]) == dict(round_payload["next_policy"]):
                policy_stall_rounds += 1
            else:
                policy_stall_rounds = 0
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
    _write_controller_state(controller_state_path, controller_state)
    _persist_report_state(
        report_path,
        report,
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
        )
        _persist_report_state(
            report_path,
            report,
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
        )
        report["policy_shift_alert"] = round_payload["policy_shift_alert"]
        _persist_report_state(
            report_path,
            report,
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
        failure_observation = build_failure_observation(
            phase=phase,
            reason=reason,
            subsystem=stalled_subsystem,
        )
        controller_state, controller_update = update_controller_state(
            controller_state,
            start_observation=round_payload.get("controller_observation", {}),
            action_policy=action_policy,
            end_observation=failure_observation,
        )
        round_payload["controller_failure_update"] = controller_update
        report["controller_summary"] = controller_state_summary(controller_state)
        _write_controller_state(controller_state_path, controller_state)

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
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
            _emit_alert_if_configured()
            print(report_path)
            raise SystemExit(2)
        child_failure_recoveries_used += 1
        next_policy = _child_failure_recovery_policy(
            current_policy,
            round_payload=round_payload,
            phase=phase,
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
        )
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
    )
    if not bool(lock_result.get("passed", False)):
        report["status"] = "safe_stop"
        report["phase"] = "preflight"
        report["reason"] = str(lock_result.get("detail", "campaign lock acquisition failed"))
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
                "status": "running",
                "phase": "preflight",
            }
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
            controller_observation = _controller_observation(
                campaign_report=report["rounds"][-2].get("campaign_report", {})
                if len(report["rounds"]) > 1 and isinstance(report["rounds"][-2], dict)
                else {},
                liftoff_payload=report["rounds"][-2].get("liftoff_report", {})
                if len(report["rounds"]) > 1 and isinstance(report["rounds"][-2], dict)
                else {},
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
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
            if not bool(disk["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(disk["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(disk["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                print(report_path)
                raise SystemExit(2)
            if not bool(gpu["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(gpu["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(gpu["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                print(report_path)
                raise SystemExit(2)
            if not bool(trust["passed"]):
                report["status"] = "safe_stop"
                report["phase"] = "preflight"
                report["reason"] = str(trust["detail"])
                _mark_round(round_payload, status="safe_stop", phase="preflight", reason=str(trust["detail"]))
                report["rounds_completed"] = _count_completed_rounds(report["rounds"])
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)

            phase = "campaign"
            report["phase"] = phase
            round_payload["phase"] = phase
            _progress(
                f"[campaign] phase=campaign round={round_index} cycles={int(current_policy['cycles'])} "
                f"task_limit={int(current_policy['task_limit'])} focus={current_policy['focus']} "
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
            campaign_progress = _make_child_progress_callback(
                report_path=report_path,
                report=report,
                status_path=status_path,
                lock_path=lock_path,
                round_payload=round_payload,
                round_index=round_index,
                phase=phase,
                child_label=f"campaign_round_{round_index}",
                event_log_path=event_log_path,
            )
            campaign_run = _run_and_stream(
                campaign_cmd,
                cwd=repo_root,
                env=env,
                progress_label=f"campaign_round_{round_index}",
                heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                max_silence_seconds=float(args.max_child_silence_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                on_event=campaign_progress,
            )
            round_payload["campaign_run"] = campaign_run
            campaign_report_path = _last_report_path(str(campaign_run["stdout"]))
            round_payload["campaign_report_path"] = "" if campaign_report_path is None else str(campaign_report_path)
            report["campaign_run"] = campaign_run
            report["campaign_report_path"] = round_payload["campaign_report_path"]
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
            if int(campaign_run["returncode"]) != 0:
                failure_reason = str(campaign_run.get("timeout_reason", "")) or "campaign subprocess failed"
                _recover_child_failure_or_stop(
                    round_payload=round_payload,
                    phase=phase,
                    reason=failure_reason,
                )
                round_index += 1
                continue

            campaign_report = _read_json(campaign_report_path)
            round_payload["campaign_report"] = campaign_report
            report["campaign_report"] = campaign_report
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
                liftoff_progress = _make_child_progress_callback(
                    report_path=report_path,
                    report=report,
                    status_path=status_path,
                    lock_path=lock_path,
                    round_payload=round_payload,
                    round_index=round_index,
                    phase=phase,
                    child_label=f"liftoff_round_{round_index}",
                    event_log_path=event_log_path,
                )
                liftoff_run = _run_and_stream(
                    liftoff_cmd,
                    cwd=repo_root,
                    env=env,
                    progress_label=f"liftoff_round_{round_index}",
                    heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                    max_silence_seconds=float(args.max_child_silence_seconds),
                    max_runtime_seconds=float(args.max_child_runtime_seconds),
                    max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                    on_event=liftoff_progress,
                )
                round_payload["liftoff_run"] = liftoff_run
                liftoff_report_path = _last_report_path(str(liftoff_run["stdout"]))
                round_payload["liftoff_report_path"] = "" if liftoff_report_path is None else str(liftoff_report_path)
                report["liftoff_run"] = liftoff_run
                report["liftoff_report_path"] = round_payload["liftoff_report_path"]
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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

            production = campaign_report.get("production_yield_summary", {})
            retained_cycles = int(production.get("retained_cycles", 0) or 0) if isinstance(production, dict) else 0
            if retained_cycles <= 0:
                no_yield_rounds += 1
            else:
                no_yield_rounds = 0
            round_payload["no_yield_rounds"] = no_yield_rounds
            end_observation = _controller_observation(
                campaign_report=campaign_report,
                liftoff_payload=liftoff_payload,
            )
            controller_state, controller_update = update_controller_state(
                controller_state,
                start_observation=round_payload.get("controller_observation", {}),
                action_policy=round_payload.get("policy", current_policy),
                end_observation=end_observation,
            )
            round_payload["controller_update"] = controller_update
            report["controller_summary"] = controller_state_summary(controller_state)
            report["unattended_evidence"] = _unattended_evidence_snapshot(config)
            _write_controller_state(controller_state_path, controller_state)
            round_planner_controls = _load_retained_improvement_planner_controls(config)
            current_policy = _next_round_policy(
                current_policy,
                campaign_report=campaign_report,
                liftoff_payload=liftoff_payload,
                round_payload=round_payload,
                controller_state=controller_state,
                prior_rounds=[item for item in report["rounds"][:-1] if isinstance(item, dict)],
                planner_controls=round_planner_controls,
                max_cycles=max(1, args.max_cycles_per_round),
                max_task_limit=max(1, args.max_task_limit),
                max_campaign_width=max(1, args.max_campaign_width),
                max_variant_width=max(1, args.max_variant_width),
            )
            round_payload["next_policy"] = dict(current_policy)
            if dict(round_payload["policy"]) == dict(round_payload["next_policy"]):
                policy_stall_rounds += 1
            else:
                policy_stall_rounds = 0
            round_payload["policy_stall_rounds"] = policy_stall_rounds
            _mark_round(round_payload, status="completed", phase="completed")
            report["rounds_completed"] = _count_completed_rounds(report["rounds"])
            report["current_policy"] = dict(current_policy)
            report["policy_shift_summary"] = _policy_shift_summary(report["rounds"])
            report["campaign_report_path"] = round_payload.get("campaign_report_path", "")
            if liftoff_payload is not None:
                report["liftoff_report_path"] = round_payload.get("liftoff_report_path", "")
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
            )
            _emit_policy_shift_alert_if_configured(round_payload)

            liftoff_state = _liftoff_signal(liftoff_payload)["state"] if liftoff_payload else ""
            if args.stop_on_liftoff_retain and liftoff_state == "retain":
                report["status"] = "completed"
                report["phase"] = "completed"
                report["reason"] = "liftoff retain gate cleared"
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "campaign_complete",
                        "reason": "liftoff retain gate cleared",
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                )
                print(report_path)
                return
            if no_yield_rounds > max(0, args.max_no_yield_rounds):
                report["status"] = "safe_stop"
                report["phase"] = "completed"
                report["reason"] = "repeated no-yield rounds exceeded unattended policy"
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "safe_stop",
                        "reason": report["reason"],
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                )
                print(report_path)
                raise SystemExit(2)
            if policy_stall_rounds > max(0, args.max_policy_stall_rounds):
                report["status"] = "safe_stop"
                report["phase"] = "completed"
                report["reason"] = "unattended policy stalled without discovering a new outer-loop action"
                _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
                _emit_alert_if_configured()
                _append_event(
                    event_log_path,
                    {
                        "event_kind": "safe_stop",
                        "reason": report["reason"],
                        "round_index": round_index,
                        "timestamp": time.time(),
                    },
                )
                print(report_path)
                raise SystemExit(2)
            round_index += 1

        if report["rounds_completed"] < requested_rounds:
            report["status"] = "safe_stop"
            report["phase"] = "completed"
            report["reason"] = "child-failure recovery budget exhausted before completing requested unattended rounds"
            _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
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
            )
            print(report_path)
            raise SystemExit(2)

        report["status"] = "completed"
        report["phase"] = "completed"
        report["reason"] = "requested unattended rounds completed"
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_complete",
                "reason": report["reason"],
                "rounds_completed": report["rounds_completed"],
                "timestamp": time.time(),
            },
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
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_interrupted",
                "phase": phase,
                "timestamp": time.time(),
            },
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
        _persist_report_state(report_path, report, status_path=status_path, lock_path=lock_path)
        _emit_alert_if_configured()
        _append_event(
            event_log_path,
            {
                "event_kind": "campaign_failed",
                "phase": phase,
                "exception_type": type(exc).__name__,
                "timestamp": time.time(),
            },
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
