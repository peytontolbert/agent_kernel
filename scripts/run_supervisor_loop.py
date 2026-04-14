from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
import time

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner
from agent_kernel.ops.job_queue import DelegatedJobQueue, DelegatedRuntimeController, TERMINAL_JOB_STATES
from agent_kernel.ops.runtime_supervision import append_jsonl, atomic_write_json
from agent_kernel.extensions.trust import write_unattended_trust_ledger
from evals.metrics import EvalMetrics


_AUTONOMY_MODES = ("shadow", "dry_run", "promote")
_ROLLOUT_STAGES = ("shadow", "compare_only", "canary", "broad")
_SUPERVISOR_HISTORY_FILENAME = "supervisor_loop_history.jsonl"
_SUPERVISOR_STATUS_FILENAME = "supervisor_loop_status.json"
_SUPERVISOR_REPORT_FILENAME = "supervisor_loop_report.json"
_SUPERVISOR_AUTONOMY_WIDENING_FILENAME = "supervisor_autonomy_widening_plan.json"
_DISCOVERY_TIMEOUT_MULTIPLIER = 3.0
_VALIDATION_GUARD_MEMORY_ROUNDS = 2
_BOOTSTRAP_RETRIEVAL_PRIORITY_MEMORY_ROUNDS = 2
_TRUST_EVIDENCE_PRIORITY_MEMORY_ROUNDS = 2
_TRUST_EVIDENCE_PRIORITY_MEMORY_MAX_EXTRA_ROUNDS = 2
_TRUST_BREADTH_PRIORITY_MEMORY_ROUNDS = 2
_TRUST_BREADTH_PRIORITY_MEMORY_MAX_EXTRA_ROUNDS = 2
_TRUST_BREADTH_RECOVERY_MIX_MEMORY_ROUNDS = 2
_TRUST_PRIORITY_DISCOVERY_MAX_TASK_LIMIT_BONUS = 4
_TRUST_PRIORITY_DISCOVERY_OBSERVATION_BONUS_SECONDS = 15.0
_BOOTSTRAP_GENERATED_EVIDENCE_DISCOVERY_MIN_BUDGET_SECONDS = 30.0
_DEFAULT_META_PROTECTED_SUBSYSTEMS = (
    "delegation",
    "operator_policy",
    "recovery",
    "trust",
)
_DEFAULT_META_PROTECTED_PATHS = (
    "agent_kernel/job_queue.py",
    "agent_kernel/shared_repo.py",
    "agent_kernel/trust.py",
    "config/supervised_parallel_work_manifest.json",
    "docs/ai_agent_status.md",
    "docs/supervised_agent_runbook.md",
    "docs/supervised_work_queue.md",
    "scripts/report_frontier_promotion_plan.py",
    "scripts/run_frontier_promotion_pass.py",
    "scripts/run_parallel_supervised_cycles.py",
    "scripts/run_supervisor_loop.py",
)
_AUTONOMY_MODE_ORDER = {name: index for index, name in enumerate(_AUTONOMY_MODES)}
_ROLLOUT_STAGE_ORDER = {name: index for index, name in enumerate(_ROLLOUT_STAGES)}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    if config is not None and config.uses_sqlite_storage():
        records = config.sqlite_store().load_cycle_records(output_path=path)
        if records:
            return records
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    payloads: list[dict[str, object]] = []
    for line in lines:
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _eval_metrics_from_summary(summary: object) -> EvalMetrics:
    payload = summary if isinstance(summary, dict) else {}
    return EvalMetrics(
        total=_safe_int(payload.get("total", 0)),
        passed=_safe_int(payload.get("passed", 0)),
        average_steps=_safe_float(payload.get("average_steps", 0.0)),
        generated_total=_safe_int(payload.get("generated_total", 0)),
        generated_passed=_safe_int(payload.get("generated_passed", 0)),
    )


def _latest_observe_metrics(config: KernelConfig) -> EvalMetrics:
    records = _load_jsonl(config.improvement_cycles_path, config=config)
    for record in reversed(records):
        if str(record.get("state", "")).strip() != "observe":
            continue
        return _eval_metrics_from_summary(record.get("metrics_summary", {}))
    return EvalMetrics(total=0, passed=0)


def _recent_cycle_records_for_path(path: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    records = _load_jsonl(path, config=config)
    latest_cycle_id = ""
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if cycle_id:
            latest_cycle_id = cycle_id
            break
    if not latest_cycle_id:
        return []
    return [record for record in records if str(record.get("cycle_id", "")).strip() == latest_cycle_id]


def _record_for_state(records: list[dict[str, object]], state: str) -> dict[str, object]:
    for record in records:
        if str(record.get("state", "")).strip() == state:
            return record
    return {}


def _recent_supervised_outcomes(config: KernelConfig, *, limit: int) -> list[dict[str, object]]:
    improvement_root = config.improvement_cycles_path.parent
    summaries: list[dict[str, object]] = []
    candidate_paths = (
        sorted(
            config.sqlite_store().list_cycle_paths(parent=improvement_root, pattern="cycles_*.jsonl")
        )
        if config.uses_sqlite_storage()
        else sorted(
            improvement_root.glob("cycles_*.jsonl"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
            reverse=True,
        )
    )
    for path in candidate_paths:
        records = _recent_cycle_records_for_path(path, config=config)
        if not records:
            continue
        observe = _record_for_state(records, "observe")
        generate = _record_for_state(records, "generate")
        select = _record_for_state(records, "select")
        observe_metrics = observe.get("metrics_summary", {}) if isinstance(observe.get("metrics_summary", {}), dict) else {}
        select_metrics = select.get("metrics_summary", {}) if isinstance(select.get("metrics_summary", {}), dict) else {}
        summaries.append(
            {
                "cycles_path": str(path),
                "cycle_id": str(records[-1].get("cycle_id", "")).strip(),
                "scope_id": str(observe_metrics.get("scope_id", "")).strip() or path.stem.removeprefix("cycles_"),
                "selected_subsystem": str(select.get("subsystem", "")).strip() or str(generate.get("subsystem", "")).strip(),
                "selected_variant_id": str(select_metrics.get("selected_variant_id", "")).strip(),
                "last_state": str(records[-1].get("state", "")).strip(),
                "generated_candidate": bool(generate),
                "observation_timed_out": bool(observe_metrics.get("observation_timed_out", False)),
                "observation_elapsed_seconds": _safe_float(observe_metrics.get("observation_elapsed_seconds", 0.0)),
            }
        )
        if limit > 0 and len(summaries) >= limit:
            break
    return summaries


def _queue_state(config: KernelConfig) -> dict[str, object]:
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    runtime = DelegatedRuntimeController(config.delegated_job_runtime_state_path)
    jobs = queue.list_jobs()
    active_jobs = [job for job in jobs if job.state not in TERMINAL_JOB_STATES]
    lease_snapshot = runtime.snapshot(config=config)
    return {
        "queue_path": str(config.delegated_job_queue_path),
        "runtime_state_path": str(config.delegated_job_runtime_state_path),
        "job_count": len(jobs),
        "active_job_count": len(active_jobs),
        "queued_job_count": sum(1 for job in jobs if job.state == "queued"),
        "in_progress_job_count": sum(1 for job in jobs if job.state == "in_progress"),
        "cancel_requested_job_count": sum(1 for job in jobs if job.state == "cancel_requested"),
        "active_leases": list(lease_snapshot.get("active_leases", []))
        if isinstance(lease_snapshot.get("active_leases", []), list)
        else [],
        "budget_groups": dict(lease_snapshot.get("budget_groups", {}))
        if isinstance(lease_snapshot.get("budget_groups", {}), dict)
        else {},
    }


def _frontier_state(config: KernelConfig) -> dict[str, object]:
    path = config.improvement_reports_dir / "supervised_parallel_frontier.json"
    payload = _load_json(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "payload": payload,
        "summary": dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), dict) else {},
        "frontier_candidates": list(payload.get("frontier_candidates", []))
        if isinstance(payload.get("frontier_candidates", []), list)
        else [],
    }


def _promotion_pass_state(config: KernelConfig) -> dict[str, object]:
    path = config.improvement_reports_dir / "supervised_frontier_promotion_pass.json"
    payload = _load_json(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "payload": payload,
        "summary": dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), dict) else {},
        "results": list(payload.get("results", [])) if isinstance(payload.get("results", []), list) else [],
    }


def _promotion_plan_state(config: KernelConfig) -> dict[str, object]:
    path = config.improvement_reports_dir / "supervised_frontier_promotion_plan.json"
    payload = _load_json(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "payload": payload,
        "summary": dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), dict) else {},
        "promotion_candidates": list(payload.get("promotion_candidates", []))
        if isinstance(payload.get("promotion_candidates", []), list)
        else [],
    }


def _increment_reason_count(counts: dict[str, int], reason: str) -> None:
    normalized = str(reason).strip()
    if not normalized:
        return
    counts[normalized] = counts.get(normalized, 0) + 1


def _promotion_block_reason_code(result: dict[str, object]) -> str:
    compare_guard_reason = str(result.get("compare_guard_reason", "")).strip()
    if compare_guard_reason:
        return compare_guard_reason
    finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
    if finalize_skip_reason:
        return finalize_skip_reason
    return ""


def _promotion_block_summary(results: list[dict[str, object]]) -> dict[str, object]:
    blocked_subsystems: set[str] = set()
    promotion_block_reason_counts: dict[str, int] = {}
    compare_guard_reason_counts: dict[str, int] = {}
    finalize_skip_reason_counts: dict[str, int] = {}
    blocked_promotion_count = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        compare_guard_reason = str(result.get("compare_guard_reason", "")).strip()
        finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
        promotion_block_reason = _promotion_block_reason_code(result)
        blocked = bool(promotion_block_reason)
        if not blocked and str(result.get("compare_status", "")).strip() == "compare_failed":
            blocked = True
        if not blocked:
            continue
        blocked_promotion_count += 1
        if subsystem:
            blocked_subsystems.add(subsystem)
        _increment_reason_count(promotion_block_reason_counts, promotion_block_reason)
        _increment_reason_count(compare_guard_reason_counts, compare_guard_reason)
        _increment_reason_count(finalize_skip_reason_counts, finalize_skip_reason)
    return {
        "blocked_promotion_count": blocked_promotion_count,
        "blocked_subsystems": sorted(blocked_subsystems),
        "promotion_block_reason_counts": promotion_block_reason_counts,
        "compare_guard_reason_counts": compare_guard_reason_counts,
        "finalize_skip_reason_counts": finalize_skip_reason_counts,
    }


def _precompare_guard_summary(promotion_plan_state: dict[str, object]) -> dict[str, object]:
    summary = promotion_plan_state.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    candidates = promotion_plan_state.get("promotion_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    validation_compare_guard_reason_counts = dict(summary.get("validation_compare_guard_reason_counts", {}))
    if not isinstance(validation_compare_guard_reason_counts, dict):
        validation_compare_guard_reason_counts = {}
    guarded_candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        reasons = candidate.get("validation_family_compare_guard_reasons", [])
        if not isinstance(reasons, list) or not any(str(reason).strip() for reason in reasons):
            continue
        guarded_candidates.append(
            {
                "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                "scope_id": str(candidate.get("scope_id", "")).strip(),
                "cycle_id": str(candidate.get("cycle_id", "")).strip(),
                "validation_family_compare_guard_reasons": [
                    str(reason).strip() for reason in reasons if str(reason).strip()
                ],
            }
        )
    return {
        "validation_compare_guard_reason_counts": {
            str(reason).strip(): int(count or 0)
            for reason, count in sorted(validation_compare_guard_reason_counts.items())
            if str(reason).strip()
        },
        "validation_guarded_candidate_count": len(guarded_candidates),
        "validation_guarded_candidates": guarded_candidates,
    }


def _validation_guard_pressure_summary(promotion_plan_state: dict[str, object]) -> dict[str, object]:
    candidates = promotion_plan_state.get("promotion_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    subsystem_buckets: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        reasons = _validation_family_compare_guard_reasons(candidate)
        if not subsystem or not reasons:
            continue
        bucket = subsystem_buckets.setdefault(
            subsystem,
            {
                "selected_subsystem": subsystem,
                "guarded_candidate_count": 0,
                "validation_guard_reason_count": 0,
                "validation_guard_severity": 0,
                "validation_family_compare_guard_reasons": set(),
            },
        )
        bucket["guarded_candidate_count"] = _safe_int(bucket.get("guarded_candidate_count", 0), 0) + 1
        bucket["validation_guard_reason_count"] = _safe_int(bucket.get("validation_guard_reason_count", 0), 0) + len(
            reasons
        )
        bucket["validation_guard_severity"] = _safe_int(bucket.get("validation_guard_severity", 0), 0) + (
            _validation_guard_severity_score(reasons)
        )
        reason_set = bucket.get("validation_family_compare_guard_reasons", set())
        if not isinstance(reason_set, set):
            reason_set = set()
        reason_set.update(reasons)
        bucket["validation_family_compare_guard_reasons"] = reason_set
    subsystem_pressure = [
        {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(bucket.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(bucket.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(bucket.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": sorted(
                {
                    str(reason).strip()
                    for reason in bucket.get("validation_family_compare_guard_reasons", set())
                    if str(reason).strip()
                }
            ),
        }
        for subsystem, bucket in sorted(subsystem_buckets.items())
    ]
    return {
        "guarded_subsystem_count": len(subsystem_pressure),
        "guarded_subsystems": subsystem_pressure,
    }


def _validation_guard_memory_summary(
    *,
    previous_memory: dict[str, object] | None,
    current_pressure_summary: dict[str, object] | None,
    promotion_results: list[dict[str, object]] | None,
) -> dict[str, object]:
    previous_entries = (
        previous_memory.get("guarded_subsystems", [])
        if isinstance(previous_memory, dict)
        else []
    )
    current_entries = (
        current_pressure_summary.get("guarded_subsystems", [])
        if isinstance(current_pressure_summary, dict)
        else []
    )
    if not isinstance(previous_entries, list):
        previous_entries = []
    if not isinstance(current_entries, list):
        current_entries = []
    if not isinstance(promotion_results, list):
        promotion_results = []

    previous_lookup: dict[str, dict[str, object]] = {}
    for raw in previous_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        previous_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(raw.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(raw.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(raw.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": _validation_family_compare_guard_reasons(raw),
            "sticky_rounds_remaining": max(
                0,
                _safe_int(raw.get("sticky_rounds_remaining", _VALIDATION_GUARD_MEMORY_ROUNDS), 0),
            ),
        }

    current_lookup: dict[str, dict[str, object]] = {}
    for raw in current_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        current_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(raw.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(raw.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(raw.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": _validation_family_compare_guard_reasons(raw),
        }

    clean_compare_subsystems: set[str] = set()
    regression_reasons_by_subsystem: dict[str, set[str]] = {}
    for result in promotion_results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        compare_guard_reason = str(result.get("compare_guard_reason", "")).strip()
        compare_status = str(result.get("compare_status", "")).strip()
        if compare_guard_reason.startswith("validation_family_"):
            regression_reasons_by_subsystem.setdefault(subsystem, set()).add(compare_guard_reason)
            continue
        if compare_status == "compared":
            clean_compare_subsystems.add(subsystem)

    next_lookup: dict[str, dict[str, object]] = {}
    for subsystem, current in current_lookup.items():
        if subsystem in clean_compare_subsystems:
            continue
        regression_reasons = regression_reasons_by_subsystem.get(subsystem, set())
        reasons = sorted(set(current.get("validation_family_compare_guard_reasons", [])) | regression_reasons)
        reason_count = max(_safe_int(current.get("validation_guard_reason_count", 0), 0), len(reasons))
        severity = max(
            _safe_int(current.get("validation_guard_severity", 0), 0),
            _validation_guard_severity_score(reasons),
        )
        sticky_rounds_remaining = _VALIDATION_GUARD_MEMORY_ROUNDS + (1 if regression_reasons else 0)
        next_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": max(_safe_int(current.get("guarded_candidate_count", 0), 0), 1),
            "validation_guard_reason_count": reason_count,
            "validation_guard_severity": severity,
            "validation_family_compare_guard_reasons": reasons,
            "sticky_rounds_remaining": sticky_rounds_remaining,
        }

    for subsystem, regression_reasons in regression_reasons_by_subsystem.items():
        if subsystem in clean_compare_subsystems or subsystem in next_lookup:
            continue
        previous = previous_lookup.get(subsystem, {})
        reasons = sorted(set(_validation_family_compare_guard_reasons(previous)) | regression_reasons)
        next_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": max(_safe_int(previous.get("guarded_candidate_count", 0), 0), 1),
            "validation_guard_reason_count": max(
                _safe_int(previous.get("validation_guard_reason_count", 0), 0),
                len(reasons),
            ),
            "validation_guard_severity": max(
                _safe_int(previous.get("validation_guard_severity", 0), 0),
                _validation_guard_severity_score(reasons),
            ),
            "validation_family_compare_guard_reasons": reasons,
            "sticky_rounds_remaining": _VALIDATION_GUARD_MEMORY_ROUNDS + 1,
        }

    for subsystem, previous in previous_lookup.items():
        if subsystem in clean_compare_subsystems or subsystem in next_lookup:
            continue
        sticky_rounds_remaining = max(0, _safe_int(previous.get("sticky_rounds_remaining", 0), 0) - 1)
        if sticky_rounds_remaining <= 0:
            continue
        next_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(previous.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(previous.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(previous.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": _validation_family_compare_guard_reasons(previous),
            "sticky_rounds_remaining": sticky_rounds_remaining,
        }

    return {
        "guarded_subsystem_count": len(next_lookup),
        "guarded_subsystems": [
            next_lookup[subsystem]
            for subsystem in sorted(next_lookup)
        ],
        "clean_compare_cleared_subsystems": sorted(clean_compare_subsystems),
    }


def _effective_validation_guard_pressure_summary(
    current_pressure_summary: dict[str, object] | None,
    validation_guard_memory: dict[str, object] | None,
) -> dict[str, object]:
    current_entries = (
        current_pressure_summary.get("guarded_subsystems", [])
        if isinstance(current_pressure_summary, dict)
        else []
    )
    retained_entries = (
        validation_guard_memory.get("guarded_subsystems", [])
        if isinstance(validation_guard_memory, dict)
        else []
    )
    if not isinstance(current_entries, list):
        current_entries = []
    if not isinstance(retained_entries, list):
        retained_entries = []

    merged: dict[str, dict[str, object]] = {}
    current_subsystems: set[str] = set()
    retained_only_subsystems: list[str] = []
    for raw in current_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        current_subsystems.add(subsystem)
        merged[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(raw.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(raw.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(raw.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": _validation_family_compare_guard_reasons(raw),
        }
    for raw in retained_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem or subsystem in merged:
            continue
        retained_only_subsystems.append(subsystem)
        merged[subsystem] = {
            "selected_subsystem": subsystem,
            "guarded_candidate_count": _safe_int(raw.get("guarded_candidate_count", 0), 0),
            "validation_guard_reason_count": _safe_int(raw.get("validation_guard_reason_count", 0), 0),
            "validation_guard_severity": _safe_int(raw.get("validation_guard_severity", 0), 0),
            "validation_family_compare_guard_reasons": _validation_family_compare_guard_reasons(raw),
            "sticky_rounds_remaining": _safe_int(raw.get("sticky_rounds_remaining", 0), 0),
        }
    return {
        "guarded_subsystem_count": len(merged),
        "guarded_subsystems": [merged[subsystem] for subsystem in sorted(merged)],
        "current_guarded_subsystem_count": len(current_subsystems),
        "retained_guarded_subsystem_count": len(retained_only_subsystems),
        "retained_only_subsystems": sorted(retained_only_subsystems),
    }


def _validation_family_compare_guard_reasons(payload: dict[str, object]) -> list[str]:
    reasons = payload.get("validation_family_compare_guard_reasons", [])
    if not isinstance(reasons, list):
        return []
    return [str(reason).strip() for reason in reasons if str(reason).strip()]


def _bootstrap_review_guard_reasons(payload: dict[str, object]) -> list[str]:
    reasons = payload.get("bootstrap_review_guard_reasons", [])
    if not isinstance(reasons, list):
        return []
    return [str(reason).strip() for reason in reasons if str(reason).strip()]


def _bootstrap_generated_evidence_guard_reasons(payload: dict[str, object]) -> list[str]:
    return [
        reason
        for reason in _bootstrap_review_guard_reasons(payload)
        if reason == "policy_bootstrap_generated_evidence_missing"
    ]


def _validation_guard_severity_score(reasons: list[str]) -> int:
    score = 0
    for reason in reasons:
        normalized = str(reason).strip()
        if not normalized:
            continue
        if "generated_pass_rate" in normalized:
            score += 3
        elif "novel_command" in normalized:
            score += 2
        elif "world_feedback" in normalized:
            score += 2
        else:
            score += 1
    return score


def _retrieval_reuse_summary(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("retrieval_reuse_summary", {})
    if not isinstance(summary, dict):
        return {}
    return dict(summary)


def _effective_retrieval_reuse_summary(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("effective_retrieval_reuse_summary", {})
    if isinstance(summary, dict) and summary:
        return dict(summary)
    return _retrieval_reuse_summary(payload)


def _retrieval_reuse_priority_score(payload: dict[str, object]) -> int:
    summary = _retrieval_reuse_summary(payload)
    if not summary:
        return 0
    trusted_retrieval_procedures = max(0, _safe_int(summary.get("trusted_retrieval_procedure_count", 0), 0))
    verified_retrieval_commands = max(0, _safe_int(summary.get("verified_retrieval_command_count", 0), 0))
    retrieval_backed_procedures = max(0, _safe_int(summary.get("retrieval_backed_procedure_count", 0), 0))
    selected_retrieval_spans = max(0, _safe_int(summary.get("selected_retrieval_span_count", 0), 0))
    retrieval_selected_steps = max(0, _safe_int(summary.get("retrieval_selected_step_count", 0), 0))
    return (
        trusted_retrieval_procedures * 5
        + verified_retrieval_commands * 4
        + retrieval_backed_procedures * 3
        + min(selected_retrieval_spans, 4)
        + min(retrieval_selected_steps, 3)
    )


def _effective_retrieval_reuse_priority_score(payload: dict[str, object]) -> int:
    return max(
        0,
        _safe_int(
            payload.get(
                "effective_retrieval_reuse_priority_score",
                payload.get("retrieval_reuse_priority_score", 0),
            ),
            0,
        ),
    )


def _bootstrap_retrieval_priority_pressure_summary(
    queues: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    prioritized_subsystems: dict[str, dict[str, object]] = {}
    for raw_entries in queues.values():
        entries = raw_entries if isinstance(raw_entries, list) else []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            subsystem = str(raw.get("selected_subsystem", "")).strip()
            if not subsystem:
                continue
            score = max(0, _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0))
            if score <= 0:
                continue
            current = prioritized_subsystems.get(subsystem, {})
            if score < max(0, _safe_int(current.get("retrieval_reuse_priority_score", 0), 0)):
                continue
            prioritized_subsystems[subsystem] = {
                "selected_subsystem": subsystem,
                "retrieval_reuse_priority_score": score,
                "retrieval_reuse_summary": _retrieval_reuse_summary(raw),
            }
    return {
        "prioritized_subsystem_count": len(prioritized_subsystems),
        "prioritized_subsystems": [
            prioritized_subsystems[subsystem]
            for subsystem in sorted(prioritized_subsystems)
        ],
    }


def _bootstrap_generated_evidence_summary(
    queues: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    targeted: dict[tuple[str, str, str], dict[str, object]] = {}
    for queue_name in ("protected_review_only", "trust_streak_accumulation", "baseline_bootstrap"):
        raw_entries = queues.get(queue_name, [])
        entries = raw_entries if isinstance(raw_entries, list) else []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            reasons = _bootstrap_generated_evidence_guard_reasons(raw)
            if not reasons:
                continue
            subsystem = str(raw.get("selected_subsystem", "")).strip()
            scope_id = str(raw.get("scope_id", "")).strip()
            variant_id = str(raw.get("selected_variant_id", "")).strip()
            if not subsystem:
                continue
            key = (subsystem, variant_id, scope_id)
            current = targeted.get(key)
            if current and str(current.get("queue_name", "")).strip() == "protected_review_only":
                continue
            targeted[key] = {
                "selected_subsystem": subsystem,
                "selected_variant_id": variant_id,
                "scope_id": scope_id,
                "cycle_id": str(raw.get("cycle_id", "")).strip(),
                "queue_name": queue_name,
                "bootstrap_generated_evidence_guard_reasons": reasons,
            }
    targets = [
        targeted[key]
        for key in sorted(
            targeted,
            key=lambda item: (
                0
                if str(targeted[item].get("queue_name", "")).strip() == "protected_review_only"
                else 1,
                item[0],
                item[1],
                item[2],
            ),
        )
    ]
    return {
        "target_count": len(targets),
        "targets": targets,
        "targeted_subsystems": sorted(
            {
                str(target.get("selected_subsystem", "")).strip()
                for target in targets
                if str(target.get("selected_subsystem", "")).strip()
            }
        ),
    }


def _prioritize_bootstrap_generated_evidence_targets(
    *,
    targets: list[dict[str, object]],
    widening_focus: dict[str, object] | None,
) -> list[dict[str, object]]:
    widening = dict(widening_focus) if isinstance(widening_focus, dict) else {}
    prioritized_subsystems = {
        str(value).strip()
        for value in list(widening.get("prioritized_subsystems", []) or [])
        if str(value).strip()
    }
    priority_families = {
        str(value).strip()
        for value in list(widening.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    }
    decorated: list[tuple[int, int, str, str, str, dict[str, object]]] = []
    for target in targets:
        subsystem = str(target.get("selected_subsystem", "")).strip()
        families = [
            str(value).strip()
            for value in list(target.get("required_families", []) or [])
            if str(value).strip()
        ]
        widenable_subsystem = 1 if subsystem in prioritized_subsystems else 0
        widenable_family = 1 if priority_families and any(family in priority_families for family in families) else 0
        decorated.append(
            (
                -widenable_subsystem,
                -widenable_family,
                str(target.get("queue_name", "")).strip(),
                subsystem,
                str(target.get("selected_variant_id", "")).strip(),
                target,
            )
        )
    return [target for *_ignored, target in sorted(decorated)]


def _bootstrap_generated_evidence_worker_budget(
    *,
    available_worker_slots: int,
    trust_priority_slot_reservation: int,
    trust_evidence_gap_severity: int,
    trust_breadth_gap_severity: int,
    widening_focus: dict[str, object] | None,
    bootstrap_generated_evidence_targets: list[dict[str, object]],
) -> int:
    base_budget = min(
        max(0, int(available_worker_slots) - int(trust_priority_slot_reservation)),
        len(bootstrap_generated_evidence_targets),
    )
    widening = dict(widening_focus) if isinstance(widening_focus, dict) else {}
    widenable_subsystems = {
        str(value).strip()
        for value in list(widening.get("prioritized_subsystems", []) or [])
        if str(value).strip()
    }
    widenable_target_count = sum(
        1
        for target in bootstrap_generated_evidence_targets
        if str(target.get("selected_subsystem", "")).strip() in widenable_subsystems
    )
    if widenable_target_count <= 0:
        return base_budget
    if int(trust_breadth_gap_severity) >= 2:
        return base_budget
    widenable_pressure = max(len(widenable_subsystems), widenable_target_count)
    round_share_cap = max(0, int(available_worker_slots) - 1)
    if round_share_cap <= 0:
        return base_budget
    widening_budget = max(base_budget, min(round_share_cap, widenable_target_count))
    # When breadth debt is mild and widening evidence is strong, let bootstrap evidence
    # claim a larger share of the total round budget instead of only consuming leftovers.
    if (
        int(trust_breadth_gap_severity) <= 1
        and int(trust_evidence_gap_severity) > 0
        and widenable_pressure >= 2
    ):
        target_round_share = max(
            widening_budget,
            min(round_share_cap, max(2, widenable_target_count + min(widenable_pressure - 1, 1))),
        )
        widening_budget = max(widening_budget, target_round_share)
    elif int(trust_breadth_gap_severity) == 0 and widenable_pressure >= 2:
        widening_budget = max(
            widening_budget,
            min(round_share_cap, max(2, widenable_target_count + min(widenable_pressure - 1, 1))),
        )
    return widening_budget


def _promotion_execution_feedback_summary(
    *,
    promotion_results: list[dict[str, object]] | None,
    widening_summary: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(promotion_results, list):
        promotion_results = []
    widening = dict(widening_summary) if isinstance(widening_summary, dict) else {}
    eligible_subsystems = {
        str(value).strip()
        for value in list(widening.get("eligible_non_protected_subsystems", []) or [])
        if str(value).strip()
    }
    cluster_by_subsystem = {
        str(key).strip(): str(value).strip()
        for key, value in dict(widening.get("eligible_non_protected_cluster_by_subsystem", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }
    retained_subsystems: list[str] = []
    retained_quality_by_subsystem: dict[str, int] = {}
    attempted_quality_by_subsystem: dict[str, int] = {}
    retained_quality_by_cluster: dict[str, int] = {}
    attempted_quality_by_cluster: dict[str, int] = {}
    for result in promotion_results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        if not subsystem or subsystem not in eligible_subsystems:
            continue
        finalize_state = str(result.get("finalize_state", "")).strip()
        compare_status = str(result.get("compare_status", "")).strip()
        quality = 0
        if compare_status == "compared":
            quality += 2
        elif compare_status == "bootstrap_first_retain":
            quality += 1
        if finalize_state == "retain":
            quality += 1
        attempted_quality_by_subsystem[subsystem] = max(
            quality,
            attempted_quality_by_subsystem.get(subsystem, 0),
        )
        cluster = cluster_by_subsystem.get(subsystem, subsystem)
        attempted_quality_by_cluster[cluster] = max(
            quality,
            attempted_quality_by_cluster.get(cluster, 0),
        )
        if finalize_state == "retain":
            if subsystem not in retained_subsystems:
                retained_subsystems.append(subsystem)
            retained_quality_by_subsystem[subsystem] = max(
                quality,
                retained_quality_by_subsystem.get(subsystem, 0),
            )
            retained_quality_by_cluster[cluster] = max(
                quality,
                retained_quality_by_cluster.get(cluster, 0),
            )
    retained_quality_score = sum(retained_quality_by_subsystem.values())
    return {
        "eligible_non_protected_retained_subsystems": retained_subsystems,
        "eligible_non_protected_retained_count": len(retained_subsystems),
        "eligible_non_protected_retained_quality_score": retained_quality_score,
        "eligible_non_protected_retained_quality_by_subsystem": retained_quality_by_subsystem,
        "eligible_non_protected_retained_quality_by_cluster": retained_quality_by_cluster,
        "eligible_non_protected_attempt_quality_by_subsystem": attempted_quality_by_subsystem,
        "eligible_non_protected_attempt_quality_by_cluster": attempted_quality_by_cluster,
    }


def _promotion_pass_subsystem_scope(
    *,
    widening_summary: dict[str, object] | None,
    promotion_feedback_summary: dict[str, object] | None,
) -> dict[str, object]:
    widening = dict(widening_summary) if isinstance(widening_summary, dict) else {}
    feedback = dict(promotion_feedback_summary) if isinstance(promotion_feedback_summary, dict) else {}
    eligible_subsystems: list[str] = []
    for value in list(widening.get("eligible_non_protected_subsystems", []) or []):
        subsystem = str(value).strip()
        if subsystem and subsystem not in eligible_subsystems:
            eligible_subsystems.append(subsystem)
    retained_subsystems = {
        str(value).strip()
        for value in list(feedback.get("eligible_non_protected_retained_subsystems", []) or [])
        if str(value).strip()
    }
    attempted_quality_by_subsystem = {
        str(key).strip(): max(0, _safe_int(value, 0))
        for key, value in dict(feedback.get("eligible_non_protected_attempt_quality_by_subsystem", {}) or {}).items()
        if str(key).strip()
    }
    attempted_quality_by_cluster = {
        str(key).strip(): max(0, _safe_int(value, 0))
        for key, value in dict(feedback.get("eligible_non_protected_attempt_quality_by_cluster", {}) or {}).items()
        if str(key).strip()
    }
    widening_priority_by_subsystem = {
        str(key).strip(): max(0, _safe_int(value, 0))
        for key, value in dict(widening.get("eligible_non_protected_priority_by_subsystem", {}) or {}).items()
        if str(key).strip()
    }
    widening_cluster_by_subsystem = {
        str(key).strip(): str(value).strip()
        for key, value in dict(widening.get("eligible_non_protected_cluster_by_subsystem", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }
    widening_priority_by_cluster = {
        str(key).strip(): max(0, _safe_int(value, 0))
        for key, value in dict(widening.get("eligible_non_protected_priority_by_cluster", {}) or {}).items()
        if str(key).strip()
    }
    remaining_subsystems = [
        subsystem
        for subsystem in eligible_subsystems
        if subsystem not in retained_subsystems
    ]
    eligible_index = {subsystem: index for index, subsystem in enumerate(eligible_subsystems)}
    prioritized_subsystems = sorted(
        remaining_subsystems,
        key=lambda subsystem: (
            -attempted_quality_by_cluster.get(
                widening_cluster_by_subsystem.get(subsystem, subsystem),
                0,
            ),
            -widening_priority_by_cluster.get(
                widening_cluster_by_subsystem.get(subsystem, subsystem),
                0,
            ),
            -attempted_quality_by_subsystem.get(subsystem, 0),
            -widening_priority_by_subsystem.get(subsystem, 0),
            eligible_index.get(subsystem, len(eligible_index)),
            subsystem,
        ),
    )
    blocked_subsystems = [
        subsystem
        for subsystem in eligible_subsystems
        if subsystem in retained_subsystems
    ]
    return {
        "allow_subsystems": prioritized_subsystems,
        "prioritized_subsystems": prioritized_subsystems,
        "blocked_subsystems": blocked_subsystems,
        "require_allow_subsystem_match": bool(eligible_subsystems),
    }


def _candidate_widening_cluster_key(candidate: dict[str, object]) -> str:
    required_families = sorted(
        {
            str(value).strip()
            for value in list(candidate.get("required_trust_families", []) or [])
            if str(value).strip()
        }
    )
    if required_families:
        return "families:" + "+".join(required_families)
    bridge = dict(candidate.get("autonomy_bridge", {})) if isinstance(candidate.get("autonomy_bridge", {}), dict) else {}
    supported_families = sorted(
        {
            str(entry.get("benchmark_family", "")).strip()
            for entry in list(bridge.get("supported_families", []) or [])
            if isinstance(entry, dict) and str(entry.get("benchmark_family", "")).strip()
        }
    )
    if supported_families:
        return "families:" + "+".join(supported_families)
    subsystem = str(candidate.get("selected_subsystem", "")).strip()
    return f"subsystem:{subsystem}" if subsystem else "subsystem:unknown"


def _apply_promotion_feedback_to_bootstrap_targets(
    *,
    targets: list[dict[str, object]],
    promotion_feedback_summary: dict[str, object] | None,
) -> list[dict[str, object]]:
    retained_subsystems = {
        str(value).strip()
        for value in list(
            dict(promotion_feedback_summary or {}).get("eligible_non_protected_retained_subsystems", []) or []
        )
        if str(value).strip()
    }
    if not retained_subsystems:
        return list(targets)
    filtered_targets: list[dict[str, object]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        subsystem = str(target.get("selected_subsystem", "")).strip()
        if subsystem and subsystem in retained_subsystems:
            continue
        filtered_targets.append(target)
    return filtered_targets


def _widening_promotion_pass_limit(
    *,
    available_worker_slots: int,
    max_promotion_candidates: int,
    promotion_candidates: list[dict[str, object]],
    trust_evidence_gap_severity: int,
    trust_breadth_gap_severity: int,
    widening_summary: dict[str, object] | None,
    promotion_feedback_summary: dict[str, object] | None,
    promotion_pass_subsystem_scope: dict[str, object] | None,
) -> int:
    base_limit = min(max(0, int(max_promotion_candidates)), len(promotion_candidates))
    if base_limit <= 0:
        return 0
    widening = dict(widening_summary) if isinstance(widening_summary, dict) else {}
    promotion_feedback = (
        dict(promotion_feedback_summary) if isinstance(promotion_feedback_summary, dict) else {}
    )
    subsystem_scope = (
        dict(promotion_pass_subsystem_scope) if isinstance(promotion_pass_subsystem_scope, dict) else {}
    )
    widenable_subsystems = [
        str(value).strip()
        for value in list(widening.get("eligible_non_protected_subsystems", []) or [])
        if str(value).strip()
    ]
    widenable_count = len(widenable_subsystems)
    if widenable_count <= 0:
        return base_limit
    remaining_widenable_subsystems = [
        str(value).strip()
        for value in list(subsystem_scope.get("allow_subsystems", []) or [])
        if str(value).strip()
    ]
    remaining_widenable_count = len(remaining_widenable_subsystems)
    fresh_retained_widenable_quality_score = max(
        0,
        _safe_int(promotion_feedback.get("eligible_non_protected_retained_quality_score", 0), 0),
    )
    if remaining_widenable_count <= 0:
        return 0
    if int(trust_breadth_gap_severity) >= 2:
        return min(base_limit, max(1, min(remaining_widenable_count, 1)))
    round_share_cap = max(1, int(available_worker_slots) - 1)
    widening_limit = max(1, min(base_limit, remaining_widenable_count, round_share_cap))
    widening_pressure = max(
        remaining_widenable_count,
        max(0, _safe_int(widening.get("eligible_non_protected_candidate_count", 0), 0)),
    )
    if (
        int(trust_breadth_gap_severity) <= 1
        and int(trust_evidence_gap_severity) > 0
        and widening_pressure >= 2
    ):
        widening_limit = max(
            widening_limit,
            min(
                base_limit,
                round_share_cap,
                max(
                    2,
                    remaining_widenable_count
                    + min(widening_pressure - 1, 1)
                    + min(fresh_retained_widenable_quality_score // 3, 1),
                ),
            ),
        )
    elif int(trust_breadth_gap_severity) == 0 and widening_pressure >= 2:
        widening_limit = max(
            widening_limit,
            min(
                base_limit,
                round_share_cap,
                max(
                    2,
                    remaining_widenable_count
                    + min(widening_pressure - 1, 1)
                    + min(fresh_retained_widenable_quality_score // 3, 1),
                ),
            ),
        )
    if fresh_retained_widenable_quality_score > 0 and remaining_widenable_count > 2:
        second_pass_feedback_cap = max(
            2,
            remaining_widenable_count
            - 1
            + min(fresh_retained_widenable_quality_score // 3, 1),
        )
        widening_limit = min(widening_limit, second_pass_feedback_cap)
    widening_limit = min(widening_limit, remaining_widenable_count)
    return widening_limit


def _should_launch_generic_discovery(
    *,
    available_worker_slots: int,
    remaining_discovery_worker_slots: int,
    trust_evidence_gap_severity: int,
    trust_breadth_gap_severity: int,
    bootstrap_generated_evidence_worker_count: int,
    promotion_pass_limit: int,
    promotion_feedback_summary: dict[str, object] | None,
    widening_summary: dict[str, object] | None,
) -> bool:
    if int(remaining_discovery_worker_slots) <= 0:
        return False
    widening = dict(widening_summary) if isinstance(widening_summary, dict) else {}
    widenable_count = max(
        0,
        len(
            [
                str(value).strip()
                for value in list(widening.get("eligible_non_protected_subsystems", []) or [])
                if str(value).strip()
            ]
        ),
    )
    if widenable_count <= 0:
        return True
    # Keep generic discovery alive while breadth debt is still severe.
    if int(trust_breadth_gap_severity) >= 2:
        return True
    round_share_cap = max(1, int(available_worker_slots) - 1)
    widening_round_majority = (
        int(bootstrap_generated_evidence_worker_count) >= round_share_cap
        and int(promotion_pass_limit) >= min(round_share_cap, max(2, widenable_count))
    )
    if (
        widening_round_majority
        and int(trust_breadth_gap_severity) <= 1
        and int(trust_evidence_gap_severity) > 0
    ):
        return False
    promotion_feedback = dict(promotion_feedback_summary) if isinstance(promotion_feedback_summary, dict) else {}
    retained_widenable_count = max(
        0,
        _safe_int(promotion_feedback.get("eligible_non_protected_retained_count", 0), 0),
    )
    if (
        retained_widenable_count > 0
        and int(trust_breadth_gap_severity) <= 1
        and int(trust_evidence_gap_severity) > 0
        and int(bootstrap_generated_evidence_worker_count) <= 0
    ):
        return False
    return True


def _bootstrap_retrieval_priority_memory_summary(
    *,
    previous_memory: dict[str, object] | None,
    current_priority_summary: dict[str, object] | None,
    promotion_results: list[dict[str, object]] | None,
) -> dict[str, object]:
    previous_entries = (
        previous_memory.get("prioritized_subsystems", [])
        if isinstance(previous_memory, dict)
        else []
    )
    current_entries = (
        current_priority_summary.get("prioritized_subsystems", [])
        if isinstance(current_priority_summary, dict)
        else []
    )
    if not isinstance(previous_entries, list):
        previous_entries = []
    if not isinstance(current_entries, list):
        current_entries = []
    if not isinstance(promotion_results, list):
        promotion_results = []

    previous_lookup: dict[str, dict[str, object]] = {}
    for raw in previous_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        previous_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(
                0,
                _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_summary": _retrieval_reuse_summary(raw),
            "sticky_rounds_remaining": max(
                0,
                _safe_int(
                    raw.get(
                        "sticky_rounds_remaining",
                        _BOOTSTRAP_RETRIEVAL_PRIORITY_MEMORY_ROUNDS,
                    ),
                    0,
                ),
            ),
        }

    current_lookup: dict[str, dict[str, object]] = {}
    for raw in current_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        current_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(
                0,
                _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_summary": _retrieval_reuse_summary(raw),
        }

    resolved_subsystems: set[str] = set()
    for result in promotion_results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        compare_status = str(result.get("compare_status", "")).strip()
        if not subsystem or not compare_status:
            continue
        if compare_status != "bootstrap_first_retain":
            resolved_subsystems.add(subsystem)

    next_lookup: dict[str, dict[str, object]] = {}
    for subsystem, current in current_lookup.items():
        if subsystem in resolved_subsystems:
            continue
        current_score = max(0, _safe_int(current.get("retrieval_reuse_priority_score", 0), 0))
        if current_score <= 0:
            continue
        previous_score = max(
            0,
            _safe_int(previous_lookup.get(subsystem, {}).get("retrieval_reuse_priority_score", 0), 0),
        )
        next_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(current_score, previous_score),
            "retrieval_reuse_summary": _retrieval_reuse_summary(current)
            or _retrieval_reuse_summary(previous_lookup.get(subsystem, {})),
            "sticky_rounds_remaining": _BOOTSTRAP_RETRIEVAL_PRIORITY_MEMORY_ROUNDS,
        }

    for subsystem, previous in previous_lookup.items():
        if subsystem in resolved_subsystems or subsystem in next_lookup:
            continue
        sticky_rounds_remaining = max(0, _safe_int(previous.get("sticky_rounds_remaining", 0), 0) - 1)
        if sticky_rounds_remaining <= 0:
            continue
        next_lookup[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(
                0,
                _safe_int(previous.get("retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_summary": _retrieval_reuse_summary(previous),
            "sticky_rounds_remaining": sticky_rounds_remaining,
        }

    return {
        "prioritized_subsystem_count": len(next_lookup),
        "prioritized_subsystems": [
            next_lookup[subsystem]
            for subsystem in sorted(next_lookup)
        ],
        "clean_compare_cleared_subsystems": sorted(resolved_subsystems),
    }


def _effective_bootstrap_retrieval_priority_summary(
    current_priority_summary: dict[str, object] | None,
    retrieval_priority_memory: dict[str, object] | None,
) -> dict[str, object]:
    current_entries = (
        current_priority_summary.get("prioritized_subsystems", [])
        if isinstance(current_priority_summary, dict)
        else []
    )
    retained_entries = (
        retrieval_priority_memory.get("prioritized_subsystems", [])
        if isinstance(retrieval_priority_memory, dict)
        else []
    )
    if not isinstance(current_entries, list):
        current_entries = []
    if not isinstance(retained_entries, list):
        retained_entries = []

    merged: dict[str, dict[str, object]] = {}
    current_subsystems: set[str] = set()
    retained_only_subsystems: list[str] = []
    for raw in current_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        current_subsystems.add(subsystem)
        merged[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(
                0,
                _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_summary": _retrieval_reuse_summary(raw),
        }
    for raw in retained_entries:
        if not isinstance(raw, dict):
            continue
        subsystem = str(raw.get("selected_subsystem", "")).strip()
        if not subsystem or subsystem in merged:
            continue
        retained_only_subsystems.append(subsystem)
        merged[subsystem] = {
            "selected_subsystem": subsystem,
            "retrieval_reuse_priority_score": max(
                0,
                _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_summary": _retrieval_reuse_summary(raw),
            "sticky_rounds_remaining": max(
                0,
                _safe_int(raw.get("sticky_rounds_remaining", 0), 0),
            ),
        }
    return {
        "prioritized_subsystem_count": len(merged),
        "prioritized_subsystems": [merged[subsystem] for subsystem in sorted(merged)],
        "current_prioritized_subsystem_count": len(current_subsystems),
        "retained_prioritized_subsystem_count": len(retained_only_subsystems),
        "retained_only_subsystems": sorted(retained_only_subsystems),
    }


def _apply_bootstrap_retrieval_priority_memory(
    queues: dict[str, list[dict[str, object]]],
    effective_priority_summary: dict[str, object] | None,
) -> dict[str, list[dict[str, object]]]:
    prioritized_entries = (
        effective_priority_summary.get("prioritized_subsystems", [])
        if isinstance(effective_priority_summary, dict)
        else []
    )
    if not isinstance(prioritized_entries, list):
        prioritized_entries = []
    priority_lookup = {
        str(raw.get("selected_subsystem", "")).strip(): raw
        for raw in prioritized_entries
        if isinstance(raw, dict) and str(raw.get("selected_subsystem", "")).strip()
    }
    for queue_name, entries in queues.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            subsystem = str(entry.get("selected_subsystem", "")).strip()
            retained = priority_lookup.get(subsystem, {})
            retained_summary = _retrieval_reuse_summary(retained)
            retained_score = max(0, _safe_int(retained.get("retrieval_reuse_priority_score", 0), 0))
            current_summary = _retrieval_reuse_summary(entry)
            current_score = max(0, _safe_int(entry.get("retrieval_reuse_priority_score", 0), 0))
            entry["retained_retrieval_reuse_summary"] = retained_summary
            entry["retained_retrieval_reuse_priority_score"] = retained_score
            entry["retrieval_reuse_priority_sticky_rounds_remaining"] = max(
                0,
                _safe_int(retained.get("sticky_rounds_remaining", 0), 0),
            )
            entry["effective_retrieval_reuse_summary"] = current_summary or retained_summary
            entry["effective_retrieval_reuse_priority_score"] = max(current_score, retained_score)
            entry["retrieval_reuse_priority"] = _effective_retrieval_reuse_priority_score(entry) > 0
        entries.sort(key=lambda entry: _bootstrap_queue_entry_sort_key(queue_name, entry))
    return queues


def _bootstrap_retrieval_priority_summary(queues: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    queue_priority_leaders: list[dict[str, object]] = []
    retrieval_ranked_subsystems: set[str] = set()
    retrieval_ranked_entry_count = 0
    trusted_retrieval_entry_count = 0
    verified_retrieval_command_total = 0
    retrieval_reuse_priority_total = 0
    retained_retrieval_ranked_entry_count = 0
    for queue_name, raw_entries in sorted(queues.items()):
        entries = raw_entries if isinstance(raw_entries, list) else []
        primary_entry = _primary_queue_entry(entries)
        if primary_entry:
            queue_priority_leaders.append(
                {
                    "queue_name": queue_name,
                    "selected_subsystem": str(primary_entry.get("selected_subsystem", "")).strip(),
                    "retrieval_reuse_priority_score": _effective_retrieval_reuse_priority_score(primary_entry),
                }
            )
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            score = _effective_retrieval_reuse_priority_score(raw)
            if score <= 0:
                continue
            retrieval_ranked_entry_count += 1
            retrieval_reuse_priority_total += score
            subsystem = str(raw.get("selected_subsystem", "")).strip()
            if subsystem:
                retrieval_ranked_subsystems.add(subsystem)
            current_score = max(0, _safe_int(raw.get("retrieval_reuse_priority_score", 0), 0))
            retained_score = max(0, _safe_int(raw.get("retained_retrieval_reuse_priority_score", 0), 0))
            if retained_score > 0 and current_score <= 0:
                retained_retrieval_ranked_entry_count += 1
            summary = _effective_retrieval_reuse_summary(raw)
            if max(0, _safe_int(summary.get("trusted_retrieval_procedure_count", 0), 0)) > 0:
                trusted_retrieval_entry_count += 1
            verified_retrieval_command_total += max(
                0,
                _safe_int(summary.get("verified_retrieval_command_count", 0), 0),
            )
    return {
        "retrieval_ranked_entry_count": retrieval_ranked_entry_count,
        "retrieval_ranked_subsystems": sorted(retrieval_ranked_subsystems),
        "trusted_retrieval_entry_count": trusted_retrieval_entry_count,
        "verified_retrieval_command_total": verified_retrieval_command_total,
        "retrieval_reuse_priority_total": retrieval_reuse_priority_total,
        "retained_retrieval_ranked_entry_count": retained_retrieval_ranked_entry_count,
        "queue_priority_leaders": queue_priority_leaders,
    }


def _bootstrap_queue_entry_sort_key(queue_name: str, entry: dict[str, object]) -> tuple[object, ...]:
    reasons = _validation_family_compare_guard_reasons(entry)
    severity = _validation_guard_severity_score(reasons)
    retrieval_priority = _effective_retrieval_reuse_priority_score(entry)
    subsystem = str(entry.get("selected_subsystem", "")).strip()
    scope_id = str(entry.get("scope_id", "")).strip()
    cycle_id = str(entry.get("cycle_id", "")).strip()
    if queue_name == "protected_review_only":
        return (0 if reasons else 1, -severity, -retrieval_priority, subsystem, scope_id, cycle_id)
    return (1 if reasons else 0, severity, -retrieval_priority, subsystem, scope_id, cycle_id)


def _bootstrap_action_sort_key(action: dict[str, object]) -> tuple[object, ...]:
    queue_name = str(action.get("queue_name", "")).strip()
    primary_entry = _primary_queue_entry(action.get("entries", []))
    primary_reasons = _validation_family_compare_guard_reasons(primary_entry)
    severity = _validation_guard_severity_score(primary_reasons)
    retrieval_priority = _effective_retrieval_reuse_priority_score(primary_entry)
    if queue_name == "protected_review_only" and primary_reasons:
        return (0, -severity, -retrieval_priority, queue_name)
    queue_rank = {
        "trust_streak_accumulation": 1,
        "baseline_bootstrap": 2,
        "protected_review_only": 3,
    }.get(queue_name, 4)
    return (queue_rank, -retrieval_priority, queue_name)


def _supervisor_status_state(config: KernelConfig) -> dict[str, object]:
    path = config.improvement_reports_dir / _SUPERVISOR_STATUS_FILENAME
    payload = _load_json(path)
    latest_round = payload.get("latest_round", {})
    machine_state = payload.get("machine_state", {})
    return {
        "path": str(path),
        "exists": path.exists(),
        "payload": payload,
        "latest_round": dict(latest_round) if isinstance(latest_round, dict) else {},
        "machine_state": dict(machine_state) if isinstance(machine_state, dict) else {},
    }


def _planner_ranked_subsystems(config: KernelConfig, *, worker_count: int) -> list[str]:
    metrics = _latest_observe_metrics(config)
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    candidates = planner.select_portfolio_campaign(metrics, max_candidates=max(int(worker_count) * 3, int(worker_count)))
    if not candidates:
        candidates = planner.rank_experiments(metrics)
    return [
        str(candidate.subsystem).strip()
        for candidate in candidates
        if str(getattr(candidate, "subsystem", "")).strip()
    ]


def _trust_breadth_focus(trust_ledger: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(trust_ledger, dict):
        trust_ledger = {}
    coverage_summary = trust_ledger.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    sampled_progress_counts = coverage_summary.get("required_family_sampled_progress_counts", {})
    if not isinstance(sampled_progress_counts, dict):
        sampled_progress_counts = {}
    signal_counts = coverage_summary.get("required_family_runtime_managed_signal_counts", {})
    if not isinstance(signal_counts, dict):
        signal_counts = {}
    decision_yield_counts = coverage_summary.get("required_family_runtime_managed_decision_yield_counts", {})
    if not isinstance(decision_yield_counts, dict):
        decision_yield_counts = {}
    credited_yield_gap_families = [
        str(value).strip()
        for value in list(
            coverage_summary.get(
                "required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield",
                [],
            )
            or []
        )
        if str(value).strip()
    ]
    decision_yield_missing_families = [
        str(value).strip()
        for value in list(coverage_summary.get("required_families_missing_runtime_managed_decision_yield", []) or [])
        if str(value).strip()
    ]
    signal_missing_families = [
        str(value).strip()
        for value in list(coverage_summary.get("required_families_missing_runtime_managed_signal", []) or [])
        if str(value).strip()
    ]
    counts = coverage_summary.get("required_family_clean_task_root_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    detail_mode = (
        "credited_family_yield"
        if credited_yield_gap_families
        else "decision_yield"
        if decision_yield_missing_families
        else "runtime_signal"
        if signal_missing_families
        else "clean_task_root"
    )
    missing_families = credited_yield_gap_families or decision_yield_missing_families or signal_missing_families or [
        str(value).strip()
        for value in list(coverage_summary.get("required_families_missing_clean_task_root_breadth", []) or [])
        if str(value).strip()
    ]
    threshold = (
        1
        if credited_yield_gap_families or decision_yield_missing_families
        else max(0, _safe_int(coverage_summary.get("family_breadth_min_distinct_task_roots", 0), 0))
    )
    details = [
        {
            "family": family,
            "observed": max(
                0,
                _safe_int(decision_yield_counts.get(family, 0), 0)
                if credited_yield_gap_families or decision_yield_missing_families
                else _safe_int(signal_counts.get(family, 0), 0)
                if signal_missing_families
                else _safe_int(counts.get(family, 0), 0),
            ),
            "sampled_progress": max(
                0,
                _safe_int(sampled_progress_counts.get(family, 0), 0)
                if credited_yield_gap_families
                else 0,
            ),
            "threshold": threshold,
            "remaining": max(
                0,
                threshold
                - max(
                    0,
                    _safe_int(decision_yield_counts.get(family, 0), 0)
                    if credited_yield_gap_families or decision_yield_missing_families
                    else _safe_int(signal_counts.get(family, 0), 0)
                    if signal_missing_families
                    else _safe_int(counts.get(family, 0), 0),
                ),
            ),
        }
        for family in missing_families
    ]
    if credited_yield_gap_families:
        details.sort(
            key=lambda detail: (
                -max(0, _safe_int(detail.get("sampled_progress", 0), 0)),
                str(detail.get("family", "")),
            )
        )
    max_remaining = max(
        (
            max(0, _safe_int(detail.get("remaining", 0), 0))
            for detail in details
            if isinstance(detail, dict)
        ),
        default=0,
    )
    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    recovery_pressure_evidence_score = _trust_breadth_recovery_pressure_evidence_score(overall_summary)
    recovery_pressure = recovery_pressure_evidence_score > 0
    recovery_role_summary = _trust_breadth_recovery_role_summary(overall_summary)
    prioritized_subsystems: list[str] = []
    if missing_families:
        prioritized_subsystems.append("trust")
        if recovery_pressure and max_remaining > 1:
            prioritized_subsystems.append("recovery")
    return {
        "missing_required_family_clean_task_root_breadth": missing_families,
        "missing_required_family_credited_yield": credited_yield_gap_families,
        "missing_required_family_runtime_managed_decision_yield": decision_yield_missing_families,
        "missing_required_family_runtime_managed_signal_breadth": signal_missing_families,
        "required_family_clean_task_root_counts": {
            str(key).strip(): max(0, _safe_int(value, 0))
            for key, value in counts.items()
            if str(key).strip()
        },
        "required_family_sampled_progress_counts": {
            str(key).strip(): max(0, _safe_int(value, 0))
            for key, value in sampled_progress_counts.items()
            if str(key).strip()
        },
        "required_family_runtime_managed_decision_yield_counts": {
            str(key).strip(): max(0, _safe_int(value, 0))
            for key, value in decision_yield_counts.items()
            if str(key).strip()
        },
        "required_family_runtime_managed_signal_counts": {
            str(key).strip(): max(0, _safe_int(value, 0))
            for key, value in signal_counts.items()
            if str(key).strip()
        },
        "family_breadth_min_distinct_task_roots": threshold,
        "detail_mode": detail_mode,
        "prioritized_subsystems": prioritized_subsystems,
        "recovery_pressure": recovery_pressure,
        "recovery_pressure_evidence_score": recovery_pressure_evidence_score,
        "recovery_role_summary": recovery_role_summary,
        "details": details,
        "max_remaining_clean_task_root_breadth_gap": max_remaining,
    }


def _trust_breadth_recovery_pressure_evidence_score(overall_summary: dict[str, object] | None) -> int:
    if not isinstance(overall_summary, dict):
        return 0
    score = 0
    for field in (
        "rollback_performed_rate",
        "false_pass_risk_rate",
        "hidden_side_effect_risk_rate",
        "success_hidden_side_effect_risk_rate",
    ):
        if _safe_float(overall_summary.get(field, 0.0), 0.0) > 0.0:
            score += 1
    if _safe_int(overall_summary.get("unexpected_change_report_count", 0), 0) > 0:
        score += 1
    return min(score, 4)


def _trust_breadth_recovery_role_summary(overall_summary: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    rollback_rate = _safe_float(overall_summary.get("rollback_performed_rate", 0.0), 0.0)
    false_pass_rate = _safe_float(overall_summary.get("false_pass_risk_rate", 0.0), 0.0)
    hidden_side_effect_rate = max(
        _safe_float(overall_summary.get("hidden_side_effect_risk_rate", 0.0), 0.0),
        _safe_float(overall_summary.get("success_hidden_side_effect_risk_rate", 0.0), 0.0),
        _safe_float(overall_summary.get("unexpected_change_report_rate", 0.0), 0.0),
    )
    role_signals = {
        "rollback": rollback_rate,
        "false_pass": false_pass_rate,
        "hidden_side_effect": hidden_side_effect_rate,
    }
    prioritized_roles = [
        role
        for role, signal in sorted(
            role_signals.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        if float(signal) > 0.0
    ]
    return {
        "role_signals": {role: round(float(signal), 4) for role, signal in role_signals.items()},
        "prioritized_roles": prioritized_roles,
    }


def _trust_breadth_gap_severity(focus: dict[str, object] | None) -> int:
    if not isinstance(focus, dict):
        return 0
    details = focus.get("details", [])
    if not isinstance(details, list):
        details = []
    detail_remaining = max(
        (
            max(0, _safe_int(detail.get("remaining", 0), 0))
            for detail in details
            if isinstance(detail, dict)
        ),
        default=0,
    )
    explicit_remaining = max(
        0,
        _safe_int(focus.get("max_remaining_clean_task_root_breadth_gap", 0), 0),
    )
    return max(detail_remaining, explicit_remaining)


def _trust_breadth_priority_families(focus: dict[str, object] | None) -> list[str]:
    if not isinstance(focus, dict):
        return []
    details = focus.get("details", [])
    if not isinstance(details, list):
        details = []
    ranked: list[tuple[int, int, str]] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        family = str(detail.get("family", "")).strip()
        if not family:
            continue
        remaining = max(0, _safe_int(detail.get("remaining", 0), 0))
        observed = max(0, _safe_int(detail.get("observed", 0), 0))
        ranked.append((-remaining, observed, family))
    if ranked:
        ranked.sort()
        return [family for _, _, family in ranked]
    return [
        str(value).strip()
        for value in list(focus.get("missing_required_family_clean_task_root_breadth", []) or [])
        if str(value).strip()
    ]


def _trust_breadth_reserved_subsystem_slots(
    focus: dict[str, object] | None,
    *,
    available_worker_slots: int,
) -> list[str]:
    if not isinstance(focus, dict):
        return []
    prioritized = [
        str(value).strip()
        for value in list(focus.get("prioritized_subsystems", []) or [])
        if str(value).strip()
    ]
    if available_worker_slots <= 0 or not prioritized:
        return []
    severity = _trust_breadth_gap_severity(focus)
    if severity <= 0:
        return []
    has_trust = "trust" in prioritized
    has_recovery = "recovery" in prioritized
    recovery_evidence_score = max(0, _safe_int(focus.get("recovery_pressure_evidence_score", 0), 0))
    fresh_recovery_pressure = bool(focus.get("recovery_pressure", False))
    retained_recovery_mix = bool(focus.get("recovery_mix_retained_from_memory", False))
    recovery_bonus_slots = 0
    if has_recovery and severity > 1 and recovery_evidence_score >= 3:
        recovery_bonus_slots += 1
    if has_recovery and severity > 1 and recovery_evidence_score >= 5 and fresh_recovery_pressure:
        recovery_bonus_slots += 1
    if retained_recovery_mix and not fresh_recovery_pressure:
        recovery_bonus_slots = max(0, recovery_bonus_slots - 1)
    reservation = min(max(0, available_worker_slots), max(0, severity) + recovery_bonus_slots)
    if reservation <= 0:
        return []
    trust_slots = 1 if has_trust else 0
    recovery_slots = 0
    if has_recovery:
        recovery_slots = min(
            max(0, reservation - trust_slots),
            1 + recovery_bonus_slots,
        )
    slots: list[str] = []
    if trust_slots > 0:
        slots.append("trust")
    slots.extend(["recovery"] * recovery_slots)
    fill_subsystems = [
        subsystem
        for subsystem in prioritized
        if subsystem not in slots
    ]
    fill_index = 0
    while len(slots) < reservation and fill_subsystems:
        slots.append(fill_subsystems[fill_index % len(fill_subsystems)])
        fill_index += 1
    while len(slots) < reservation and has_trust:
        slots.append("trust")
    return slots[:reservation]


def _trust_breadth_reserved_recovery_roles(
    focus: dict[str, object] | None,
    *,
    reserved_subsystem_slots: list[str] | None,
) -> list[str]:
    normalized_slots = [
        str(value).strip()
        for value in list(reserved_subsystem_slots or [])
        if str(value).strip()
    ]
    recovery_slot_count = sum(1 for value in normalized_slots if value == "recovery")
    if recovery_slot_count <= 0 or not isinstance(focus, dict):
        return []
    role_summary = focus.get("recovery_role_summary", {})
    if not isinstance(role_summary, dict):
        role_summary = {}
    prioritized_roles = [
        str(value).strip()
        for value in list(role_summary.get("prioritized_roles", []) or [])
        if str(value).strip()
    ]
    if not prioritized_roles:
        prioritized_roles = ["rollback"]
    roles: list[str] = []
    for index in range(recovery_slot_count):
        roles.append(prioritized_roles[min(index, len(prioritized_roles) - 1)])
    return roles


def _trust_breadth_reserved_recovery_strategy_families(
    focus: dict[str, object] | None,
    *,
    reserved_subsystem_slots: list[str] | None,
    reserved_recovery_roles: list[str] | None,
) -> list[str]:
    normalized_slots = [
        str(value).strip()
        for value in list(reserved_subsystem_slots or [])
        if str(value).strip()
    ]
    recovery_slot_count = sum(1 for value in normalized_slots if value == "recovery")
    normalized_roles = [
        str(value).strip()
        for value in list(reserved_recovery_roles or [])
        if str(value).strip()
    ]
    if recovery_slot_count <= 0 or not normalized_roles:
        return []
    role_to_strategy_families = {
        "rollback": (
            "rollback_validation",
            "restore_verification",
            "snapshot_integrity",
        ),
        "false_pass": (
            "snapshot_coverage",
            "verifier_crosscheck",
            "post_success_replay",
        ),
        "hidden_side_effect": (
            "mutation_residue_scan",
            "unexpected_change_audit",
            "workspace_restore_verification",
        ),
    }
    per_role_counts: dict[str, int] = {}
    strategy_families: list[str] = []
    for role in normalized_roles[:recovery_slot_count]:
        per_role_counts[role] = per_role_counts.get(role, 0) + 1
        options = role_to_strategy_families.get(role, ())
        if not options:
            strategy_families.append("generic_recovery")
            continue
        strategy_families.append(options[(per_role_counts[role] - 1) % len(options)])
    return strategy_families


def _variant_ids_for_recovery_roles(
    planner: ImprovementPlanner,
    experiment: object,
    metrics: EvalMetrics,
    *,
    recovery_roles: list[str] | None,
    strategy_families: list[str] | None,
    slot_count: int,
) -> list[str]:
    ranked_variants = planner.rank_variants(experiment, metrics)
    if slot_count <= 0 or not ranked_variants:
        return []
    role_strategy_preferences = {
        "rollback": ("rollback_validation", "restore_verification", "snapshot_integrity"),
        "false_pass": ("snapshot_coverage", "verifier_crosscheck", "post_success_replay"),
        "hidden_side_effect": ("mutation_residue_scan", "unexpected_change_audit", "workspace_restore_verification"),
    }
    strategy_variant_preferences = {
        "rollback_validation": ("rollback_safety",),
        "restore_verification": ("rollback_safety", "snapshot_coverage"),
        "snapshot_integrity": ("snapshot_coverage", "rollback_safety"),
        "snapshot_coverage": ("snapshot_coverage", "rollback_safety"),
        "verifier_crosscheck": ("snapshot_coverage", "rollback_safety"),
        "post_success_replay": ("snapshot_coverage", "rollback_safety"),
        "mutation_residue_scan": ("snapshot_coverage", "rollback_safety"),
        "unexpected_change_audit": ("snapshot_coverage", "rollback_safety"),
        "workspace_restore_verification": ("rollback_safety", "snapshot_coverage"),
    }
    selected_ids: list[str] = []
    remaining_variants = [
        str(getattr(variant, "variant_id", "")).strip()
        for variant in ranked_variants
        if str(getattr(variant, "variant_id", "")).strip()
    ]
    normalized_roles = [
        str(role).strip()
        for role in list(recovery_roles or [])[:slot_count]
        if str(role).strip()
    ]
    normalized_strategy_families = [
        str(value).strip()
        for value in list(strategy_families or [])[:slot_count]
        if str(value).strip()
    ]
    for slot_index, role in enumerate(normalized_roles):
        preferred_strategy_family = (
            normalized_strategy_families[slot_index]
            if slot_index < len(normalized_strategy_families)
            else ""
        )
        preferred_ids = strategy_variant_preferences.get(preferred_strategy_family, ())
        if not preferred_ids:
            for family in role_strategy_preferences.get(role, ()):
                preferred_ids = strategy_variant_preferences.get(family, ())
                if preferred_ids:
                    break
        selected = ""
        for preferred in preferred_ids:
            if preferred in remaining_variants:
                selected = preferred
                remaining_variants.remove(preferred)
                break
        if selected:
            selected_ids.append(selected)
            continue
        if remaining_variants:
            selected_ids.append(remaining_variants.pop(0))
    if len(selected_ids) < slot_count:
        variant_budget = planner.recommend_variant_budget(experiment, metrics, max_width=slot_count)
        fallback_ids = [
            str(value).strip()
            for value in list(variant_budget.selected_ids or [])
            if str(value).strip()
        ]
        for variant_id in fallback_ids + remaining_variants:
            if len(selected_ids) >= slot_count:
                break
            if variant_id:
                selected_ids.append(variant_id)
    while selected_ids and len(selected_ids) < slot_count:
        selected_ids.append(selected_ids[-1])
    return selected_ids[:slot_count]


def _reserved_variant_ids_for_subsystems(
    config: KernelConfig,
    *,
    reserved_subsystem_slots: list[str] | None,
    trust_breadth_focus: dict[str, object] | None = None,
) -> list[str]:
    normalized_slots = [
        str(value).strip()
        for value in list(reserved_subsystem_slots or [])
        if str(value).strip()
    ]
    if not normalized_slots:
        return []
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    metrics = _latest_observe_metrics(config)
    ranked = planner.rank_experiments(metrics)
    experiments_by_subsystem: dict[str, object] = {}
    for experiment in ranked:
        subsystem = str(getattr(experiment, "subsystem", "")).strip()
        if subsystem and subsystem not in experiments_by_subsystem:
            experiments_by_subsystem[subsystem] = experiment
    variant_ids = [""] * len(normalized_slots)
    subsystem_indexes: dict[str, list[int]] = {}
    for index, subsystem in enumerate(normalized_slots):
        subsystem_indexes.setdefault(subsystem, []).append(index)
    for subsystem, indexes in subsystem_indexes.items():
        experiment = experiments_by_subsystem.get(subsystem)
        if experiment is None:
            continue
        if subsystem == "recovery":
            recovery_roles = _trust_breadth_reserved_recovery_roles(
                trust_breadth_focus,
                reserved_subsystem_slots=normalized_slots,
            )
            selected_ids = _variant_ids_for_recovery_roles(
                planner,
                experiment,
                metrics,
                recovery_roles=recovery_roles,
                strategy_families=_trust_breadth_reserved_recovery_strategy_families(
                    trust_breadth_focus,
                    reserved_subsystem_slots=normalized_slots,
                    reserved_recovery_roles=recovery_roles,
                ),
                slot_count=len(indexes),
            )
            for offset, slot_index in enumerate(indexes):
                if not selected_ids:
                    break
                variant_ids[slot_index] = selected_ids[min(offset, len(selected_ids) - 1)]
            continue
        if len(indexes) <= 1:
            variant = planner.choose_variant(experiment, metrics)
            variant_id = str(getattr(variant, "variant_id", "")).strip()
            if variant_id:
                variant_ids[indexes[0]] = variant_id
            continue
        variant_budget = planner.recommend_variant_budget(experiment, metrics, max_width=len(indexes))
        selected_ids = [
            str(value).strip()
            for value in list(variant_budget.selected_ids or [])
            if str(value).strip()
        ]
        ranked_variants = planner.rank_variants(experiment, metrics)
        for variant in ranked_variants:
            variant_id = str(getattr(variant, "variant_id", "")).strip()
            if variant_id and variant_id not in selected_ids:
                selected_ids.append(variant_id)
            if len(selected_ids) >= len(indexes):
                break
        for offset, slot_index in enumerate(indexes):
            if not selected_ids:
                break
            variant_ids[slot_index] = selected_ids[min(offset, len(selected_ids) - 1)]
    return variant_ids


def _candidate_observed_benchmark_families(candidate: dict[str, object]) -> list[str]:
    families: set[str] = set()
    for value in list(candidate.get("observed_benchmark_families", []) or []):
        normalized = str(value).strip()
        if normalized:
            families.add(normalized)
    validation_summary = candidate.get("validation_family_summary", {})
    if isinstance(validation_summary, dict):
        family = str(validation_summary.get("benchmark_family", "")).strip()
        if family:
            families.add(family)
    return sorted(families)


def _candidate_trust_evidence(candidate: dict[str, object], trust_ledger: dict[str, object]) -> dict[str, object]:
    coverage_summary = trust_ledger.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    policy = trust_ledger.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    required_families = {
        str(value).strip()
        for value in list(
            coverage_summary.get("required_families", [])
            or policy.get("required_benchmark_families", [])
            or []
        )
        if str(value).strip()
    }
    family_assessments = trust_ledger.get("family_assessments", {})
    if not isinstance(family_assessments, dict):
        family_assessments = {}
    missing_counted_gated_families = {
        str(value).strip()
        for value in list(coverage_summary.get("missing_required_counted_gated_families", []) or [])
        if str(value).strip()
    }
    missing_clean_task_root_breadth_families = {
        str(value).strip()
        for value in list(coverage_summary.get("required_families_missing_clean_task_root_breadth", []) or [])
        if str(value).strip()
    }
    required_family_clean_task_root_counts = coverage_summary.get("required_family_clean_task_root_counts", {})
    if not isinstance(required_family_clean_task_root_counts, dict):
        required_family_clean_task_root_counts = {}
    required_family_light_supervision_clean_success_counts = coverage_summary.get(
        "required_family_light_supervision_clean_success_counts",
        {},
    )
    if not isinstance(required_family_light_supervision_clean_success_counts, dict):
        required_family_light_supervision_clean_success_counts = {}
    required_family_contract_clean_failure_recovery_clean_success_counts = coverage_summary.get(
        "required_family_contract_clean_failure_recovery_clean_success_counts",
        {},
    )
    if not isinstance(required_family_contract_clean_failure_recovery_clean_success_counts, dict):
        required_family_contract_clean_failure_recovery_clean_success_counts = {}
    runtime_managed_signal_families = {
        str(value).strip()
        for value in list(coverage_summary.get("required_families_with_runtime_managed_signals", []) or [])
        if str(value).strip()
    }
    runtime_managed_decision_yield_families = {
        str(value).strip()
        for value in list(coverage_summary.get("required_families_with_runtime_managed_decision_yield", []) or [])
        if str(value).strip()
    }
    missing_runtime_managed_decision_yield_families = {
        str(value).strip()
        for value in list(coverage_summary.get("required_families_missing_runtime_managed_decision_yield", []) or [])
        if str(value).strip()
    }
    required_family_runtime_managed_decision_yield_counts = coverage_summary.get(
        "required_family_runtime_managed_decision_yield_counts",
        {},
    )
    if not isinstance(required_family_runtime_managed_decision_yield_counts, dict):
        required_family_runtime_managed_decision_yield_counts = {}
    family_breadth_min_distinct_task_roots = max(
        0,
        _safe_int(coverage_summary.get("family_breadth_min_distinct_task_roots", 0), 0),
    )
    observed_benchmark_families = _candidate_observed_benchmark_families(candidate)
    required_trust_families = [
        family
        for family in observed_benchmark_families
        if family in required_families
    ]
    blocked_reasons: list[str] = []
    bootstrap_families: list[str] = []
    restricted_families: list[str] = []
    missing_counted_families: list[str] = []
    missing_decision_yield_families: list[str] = []
    missing_breadth_families: list[str] = []
    for family in required_trust_families:
        if family in missing_runtime_managed_decision_yield_families:
            missing_decision_yield_families.append(family)
            observed_yield = max(0, _safe_int(required_family_runtime_managed_decision_yield_counts.get(family, 0), 0))
            blocked_reasons.append(f"trust_family_runtime_managed_decision_yield:{family}:{observed_yield}<1")
        if family in missing_counted_gated_families and family not in runtime_managed_signal_families:
            missing_counted_families.append(family)
            blocked_reasons.append(f"trust_family_counted_gated_evidence_missing:{family}")
        assessment = family_assessments.get(family, {})
        if not isinstance(assessment, dict):
            assessment = {}
        status = str(assessment.get("status", "")).strip()
        if status == "bootstrap":
            bootstrap_families.append(family)
            blocked_reasons.append(f"trust_family_status=bootstrap:{family}")
        elif assessment and not bool(assessment.get("passed", False)):
            restricted_families.append(family)
            blocked_reasons.append(f"trust_family_status=restricted:{family}")
        if family in missing_clean_task_root_breadth_families and family not in runtime_managed_signal_families:
            missing_breadth_families.append(family)
            observed = max(0, _safe_int(required_family_clean_task_root_counts.get(family, 0), 0))
            blocked_reasons.append(
                "trust_family_clean_task_root_breadth:"
                f"{family}:{observed}<{family_breadth_min_distinct_task_roots}"
            )
    return {
        "observed_benchmark_families": observed_benchmark_families,
        "required_trust_families": required_trust_families,
        "missing_counted_gated_families": sorted(set(missing_counted_families)),
        "missing_runtime_managed_decision_yield_families": sorted(set(missing_decision_yield_families)),
        "bootstrap_families": sorted(set(bootstrap_families)),
        "restricted_families": sorted(set(restricted_families)),
        "missing_clean_task_root_breadth_families": sorted(set(missing_breadth_families)),
        "required_family_runtime_managed_decision_yield_counts": {
            family: max(0, _safe_int(required_family_runtime_managed_decision_yield_counts.get(family, 0), 0))
            for family in required_trust_families
        },
        "runtime_managed_decision_yield_families": sorted(runtime_managed_decision_yield_families),
        "required_family_light_supervision_clean_success_counts": {
            family: max(0, _safe_int(required_family_light_supervision_clean_success_counts.get(family, 0), 0))
            for family in required_trust_families
        },
        "required_family_contract_clean_failure_recovery_clean_success_counts": {
            family: max(
                0,
                _safe_int(required_family_contract_clean_failure_recovery_clean_success_counts.get(family, 0), 0),
            )
            for family in required_trust_families
        },
        "runtime_managed_signal_families": sorted(runtime_managed_signal_families),
        "blocked_reasons": sorted(set(reason for reason in blocked_reasons if str(reason).strip())),
        "blocked": bool(blocked_reasons),
    }


def _candidate_autonomy_bridge(
    candidate_evidence: dict[str, object],
    *,
    clean_success_threshold: int,
) -> dict[str, object]:
    required_families = [
        str(value).strip()
        for value in list(candidate_evidence.get("required_trust_families", []) or [])
        if str(value).strip()
    ]
    if not required_families:
        return {
            "eligible": False,
            "required_family_count": 0,
            "supported_families": [],
            "unsupported_families": [],
            "minimum_clean_success_threshold": max(1, clean_success_threshold),
            "reason": "no_required_trust_families",
        }
    if list(candidate_evidence.get("restricted_families", []) or []):
        return {
            "eligible": False,
            "required_family_count": len(required_families),
            "supported_families": [],
            "unsupported_families": required_families,
            "minimum_clean_success_threshold": max(1, clean_success_threshold),
            "reason": "restricted_family_present",
        }
    if list(candidate_evidence.get("missing_runtime_managed_decision_yield_families", []) or []):
        return {
            "eligible": False,
            "required_family_count": len(required_families),
            "supported_families": [],
            "unsupported_families": required_families,
            "minimum_clean_success_threshold": max(1, clean_success_threshold),
            "reason": "family_decision_yield_missing",
        }
    if list(candidate_evidence.get("missing_clean_task_root_breadth_families", []) or []):
        return {
            "eligible": False,
            "required_family_count": len(required_families),
            "supported_families": [],
            "unsupported_families": required_families,
            "minimum_clean_success_threshold": max(1, clean_success_threshold),
            "reason": "family_breadth_missing",
        }

    bootstrap_or_missing = {
        str(value).strip()
        for value in list(candidate_evidence.get("missing_counted_gated_families", []) or [])
        + list(candidate_evidence.get("bootstrap_families", []) or [])
        if str(value).strip()
    }
    if not bootstrap_or_missing:
        return {
            "eligible": False,
            "required_family_count": len(required_families),
            "supported_families": [],
            "unsupported_families": [],
            "minimum_clean_success_threshold": max(1, clean_success_threshold),
            "reason": "counted_trust_already_satisfied",
        }

    light_success_counts = (
        dict(candidate_evidence.get("required_family_light_supervision_clean_success_counts", {}))
        if isinstance(candidate_evidence.get("required_family_light_supervision_clean_success_counts", {}), dict)
        else {}
    )
    recovery_success_counts = (
        dict(candidate_evidence.get("required_family_contract_clean_failure_recovery_clean_success_counts", {}))
        if isinstance(candidate_evidence.get("required_family_contract_clean_failure_recovery_clean_success_counts", {}), dict)
        else {}
    )
    threshold = max(1, clean_success_threshold)
    recovery_assisted_threshold = max(1, threshold - 1)
    supported_families: list[dict[str, object]] = []
    unsupported_families: list[dict[str, object]] = []
    for family in required_families:
        if family not in bootstrap_or_missing:
            continue
        light_successes = max(0, _safe_int(light_success_counts.get(family, 0), 0))
        recovery_successes = max(0, _safe_int(recovery_success_counts.get(family, 0), 0))
        recovery_assisted = light_successes >= recovery_assisted_threshold and recovery_successes > 0
        supported = light_successes >= threshold or recovery_assisted
        payload = {
            "family": family,
            "light_supervision_clean_successes": light_successes,
            "contract_clean_failure_recovery_clean_successes": recovery_successes,
            "recovery_assisted": recovery_assisted,
        }
        if supported:
            supported_families.append(payload)
        else:
            unsupported_families.append(payload)
    return {
        "eligible": bool(bootstrap_or_missing) and not unsupported_families and bool(supported_families),
        "required_family_count": len(required_families),
        "supported_families": supported_families,
        "unsupported_families": unsupported_families,
        "minimum_clean_success_threshold": threshold,
        "reason": "supported" if supported_families and not unsupported_families else "insufficient_clean_successes",
    }


def _frontier_candidate_trust_evidence(
    *,
    frontier_state: dict[str, object],
    trust_ledger: dict[str, object],
) -> dict[str, object]:
    frontier_candidates = frontier_state.get("frontier_candidates", [])
    if not isinstance(frontier_candidates, list):
        frontier_candidates = []
    entries: list[dict[str, object]] = []
    blocked_subsystems: set[str] = set()
    blocked_reason_counts: dict[str, int] = {}
    for candidate in frontier_candidates:
        if not isinstance(candidate, dict):
            continue
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        evidence = _candidate_trust_evidence(candidate, trust_ledger)
        entry = {
            "scope_id": str(candidate.get("scope_id", "")).strip(),
            "cycle_id": str(candidate.get("cycle_id", "")).strip(),
            "selected_subsystem": subsystem,
            "candidate_artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
            **evidence,
        }
        entries.append(entry)
        if not subsystem or not bool(entry.get("blocked", False)):
            continue
        blocked_subsystems.add(subsystem)
        for reason in list(entry.get("blocked_reasons", []) or []):
            _increment_reason_count(blocked_reason_counts, f"{subsystem}:{str(reason).strip()}")
    return {
        "entries": entries,
        "blocked_subsystems": sorted(blocked_subsystems),
        "blocked_reason_counts": blocked_reason_counts,
    }


def _matching_frontier_trust_evidence(
    *,
    result: dict[str, object],
    frontier_trust_evidence: dict[str, object],
) -> dict[str, object]:
    entries = frontier_trust_evidence.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    subsystem = str(result.get("selected_subsystem", "")).strip()
    scope_id = str(result.get("scope_id", "")).strip()
    cycle_id = str(result.get("cycle_id", "")).strip()
    artifact_path = str(result.get("candidate_artifact_path", "")).strip()
    fallback: dict[str, object] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("selected_subsystem", "")).strip() != subsystem:
            continue
        if not fallback:
            fallback = entry
        entry_scope_id = str(entry.get("scope_id", "")).strip()
        entry_cycle_id = str(entry.get("cycle_id", "")).strip()
        entry_artifact_path = str(entry.get("candidate_artifact_path", "")).strip()
        scope_matches = not scope_id or not entry_scope_id or entry_scope_id == scope_id
        cycle_matches = not cycle_id or not entry_cycle_id or entry_cycle_id == cycle_id
        artifact_matches = not artifact_path or not entry_artifact_path or entry_artifact_path == artifact_path
        if scope_matches and cycle_matches and artifact_matches:
            return entry
    return fallback


def _trust_evidence_focus(
    *,
    frontier_trust_evidence: dict[str, object] | None,
    trust_ledger: dict[str, object] | None,
) -> dict[str, object]:
    summary = frontier_trust_evidence if isinstance(frontier_trust_evidence, dict) else {}
    ledger = trust_ledger if isinstance(trust_ledger, dict) else {}
    overall_assessment = ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    overall_summary = ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    entries = summary.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    family_details: dict[str, dict[str, object]] = {}
    blocked_subsystems: set[str] = set()
    blocked_reason_counts: dict[str, int] = {}
    blocked_candidate_count = 0
    for entry in entries:
        if not isinstance(entry, dict) or not bool(entry.get("blocked", False)):
            continue
        blocked_candidate_count += 1
        subsystem = str(entry.get("selected_subsystem", "")).strip()
        if subsystem:
            blocked_subsystems.add(subsystem)
        required_families = {
            str(value).strip()
            for value in list(entry.get("required_trust_families", []) or [])
            if str(value).strip()
        }
        missing_counted_gated = {
            str(value).strip()
            for value in list(entry.get("missing_counted_gated_families", []) or [])
            if str(value).strip()
        }
        bootstrap_families = {
            str(value).strip()
            for value in list(entry.get("bootstrap_families", []) or [])
            if str(value).strip()
        }
        restricted_families = {
            str(value).strip()
            for value in list(entry.get("restricted_families", []) or [])
            if str(value).strip()
        }
        missing_breadth_families = {
            str(value).strip()
            for value in list(entry.get("missing_clean_task_root_breadth_families", []) or [])
            if str(value).strip()
        }
        for family in required_families:
            detail = family_details.setdefault(
                family,
                {
                    "family": family,
                    "blocked_candidate_count": 0,
                    "blocked_subsystems": set(),
                    "blocked_reason_codes": set(),
                },
            )
            detail["blocked_candidate_count"] = _safe_int(detail.get("blocked_candidate_count", 0), 0) + 1
            blocked_subsystem_set = detail.get("blocked_subsystems")
            if subsystem and isinstance(blocked_subsystem_set, set):
                blocked_subsystem_set.add(subsystem)
            blocked_reason_set = detail.get("blocked_reason_codes")
            if family in missing_counted_gated:
                if isinstance(blocked_reason_set, set):
                    blocked_reason_set.add("counted_gated_evidence_missing")
                _increment_reason_count(blocked_reason_counts, f"{family}:counted_gated_evidence_missing")
            if family in bootstrap_families:
                if isinstance(blocked_reason_set, set):
                    blocked_reason_set.add("status_bootstrap")
                _increment_reason_count(blocked_reason_counts, f"{family}:status_bootstrap")
            if family in restricted_families:
                if isinstance(blocked_reason_set, set):
                    blocked_reason_set.add("status_restricted")
                _increment_reason_count(blocked_reason_counts, f"{family}:status_restricted")
            if family in missing_breadth_families:
                if isinstance(blocked_reason_set, set):
                    blocked_reason_set.add("clean_task_root_breadth")
                _increment_reason_count(blocked_reason_counts, f"{family}:clean_task_root_breadth")
    resolved_family_details: list[dict[str, object]] = []
    for family, detail in family_details.items():
        blocked_reason_codes = sorted(
            {
                str(value).strip()
                for value in list(detail.get("blocked_reason_codes", set()) or [])
                if str(value).strip()
            }
        )
        blocked_candidate_total = max(0, _safe_int(detail.get("blocked_candidate_count", 0), 0))
        severity = blocked_candidate_total + len(blocked_reason_codes) - (1 if blocked_candidate_total > 0 else 0)
        resolved_family_details.append(
            {
                "family": family,
                "blocked_candidate_count": blocked_candidate_total,
                "blocked_subsystems": sorted(
                    {
                        str(value).strip()
                        for value in list(detail.get("blocked_subsystems", set()) or [])
                        if str(value).strip()
                    }
                ),
                "blocked_reason_codes": blocked_reason_codes,
                "severity": max(0, severity),
            }
        )
    resolved_family_details.sort(
        key=lambda detail: (
            -_safe_int(detail.get("severity", 0), 0),
            -_safe_int(detail.get("blocked_candidate_count", 0), 0),
            str(detail.get("family", "")).strip(),
        )
    )
    priority_benchmark_families = [
        str(detail.get("family", "")).strip()
        for detail in resolved_family_details
        if str(detail.get("family", "")).strip()
    ]
    prioritized_subsystems: list[str] = []
    if priority_benchmark_families:
        prioritized_subsystems.append("trust")
        if (
            not bool(overall_assessment.get("passed", False))
            or any(
                "status_restricted" in list(detail.get("blocked_reason_codes", []) or [])
                for detail in resolved_family_details
            )
            or any(
                "status_bootstrap" in list(detail.get("blocked_reason_codes", []) or [])
                for detail in resolved_family_details
            )
            or float(overall_summary.get("rollback_performed_rate", 0.0) or 0.0) > 0.0
            or float(overall_summary.get("false_pass_risk_rate", 0.0) or 0.0) > 0.0
        ):
            prioritized_subsystems.append("recovery")
    max_family_severity = max(
        (
            _safe_int(detail.get("severity", 0), 0)
            for detail in resolved_family_details
            if isinstance(detail, dict)
        ),
        default=0,
    )
    return {
        "blocked_candidate_count": blocked_candidate_count,
        "blocked_subsystems": sorted(blocked_subsystems),
        "blocked_reason_counts": blocked_reason_counts,
        "prioritized_subsystems": prioritized_subsystems,
        "priority_benchmark_families": priority_benchmark_families,
        "family_details": resolved_family_details,
        "max_blocked_family_severity": max_family_severity,
    }


def _trust_evidence_focus_severity(focus: dict[str, object] | None) -> int:
    if not isinstance(focus, dict):
        return 0
    details = focus.get("family_details", [])
    if not isinstance(details, list):
        details = []
    detail_severity = max(
        (
            max(0, _safe_int(detail.get("severity", 0), 0))
            for detail in details
            if isinstance(detail, dict)
        ),
        default=0,
    )
    explicit_severity = max(0, _safe_int(focus.get("max_blocked_family_severity", 0), 0))
    return max(detail_severity, explicit_severity)


def _trust_evidence_priority_memory_summary(
    *,
    previous_memory: dict[str, object] | None,
    current_focus: dict[str, object] | None,
) -> dict[str, object]:
    current_families = (
        [
            str(value).strip()
            for value in list(current_focus.get("priority_benchmark_families", []) or [])
            if str(value).strip()
        ]
        if isinstance(current_focus, dict)
        else []
    )
    current_severity = _trust_evidence_focus_severity(current_focus)
    if current_families:
        sticky_rounds_remaining = _TRUST_EVIDENCE_PRIORITY_MEMORY_ROUNDS + min(
            current_severity,
            _TRUST_EVIDENCE_PRIORITY_MEMORY_MAX_EXTRA_ROUNDS,
        )
        return {
            "blocked_candidate_count": max(0, _safe_int(current_focus.get("blocked_candidate_count", 0), 0)),
            "blocked_subsystems": [
                str(value).strip()
                for value in list(current_focus.get("blocked_subsystems", []) or [])
                if str(value).strip()
            ],
            "prioritized_subsystems": [
                str(value).strip()
                for value in list(current_focus.get("prioritized_subsystems", []) or [])
                if str(value).strip()
            ],
            "priority_benchmark_families": current_families,
            "family_details": list(current_focus.get("family_details", []) or []),
            "max_blocked_family_severity": current_severity,
            "sticky_rounds_remaining": sticky_rounds_remaining,
        }
    sticky_rounds_remaining = max(
        0,
        _safe_int(
            previous_memory.get("sticky_rounds_remaining", 0) if isinstance(previous_memory, dict) else 0,
            0,
        )
        - 1,
    )
    previous_families = (
        [
            str(value).strip()
            for value in list(previous_memory.get("priority_benchmark_families", []) or [])
            if str(value).strip()
        ]
        if isinstance(previous_memory, dict)
        else []
    )
    if sticky_rounds_remaining <= 0 or not previous_families:
        return {
            "blocked_candidate_count": 0,
            "blocked_subsystems": [],
            "prioritized_subsystems": [],
            "priority_benchmark_families": [],
            "family_details": [],
            "max_blocked_family_severity": 0,
            "sticky_rounds_remaining": 0,
        }
    return {
        "blocked_candidate_count": max(0, _safe_int(previous_memory.get("blocked_candidate_count", 0), 0)),
        "blocked_subsystems": [
            str(value).strip()
            for value in list(previous_memory.get("blocked_subsystems", []) or [])
            if str(value).strip()
        ],
        "prioritized_subsystems": [
            str(value).strip()
            for value in list(previous_memory.get("prioritized_subsystems", []) or [])
            if str(value).strip()
        ],
        "priority_benchmark_families": previous_families,
        "family_details": list(previous_memory.get("family_details", []) or []),
        "max_blocked_family_severity": max(
            0,
            _safe_int(previous_memory.get("max_blocked_family_severity", 0), 0),
        ),
        "sticky_rounds_remaining": sticky_rounds_remaining,
    }


def _effective_trust_evidence_focus(
    current_focus: dict[str, object] | None,
    trust_evidence_priority_memory: dict[str, object] | None,
) -> dict[str, object]:
    current_families = (
        [
            str(value).strip()
            for value in list(current_focus.get("priority_benchmark_families", []) or [])
            if str(value).strip()
        ]
        if isinstance(current_focus, dict)
        else []
    )
    if current_families:
        effective = dict(current_focus) if isinstance(current_focus, dict) else {}
        effective["retained_from_memory"] = False
        effective["sticky_rounds_remaining"] = _safe_int(
            effective.get("sticky_rounds_remaining", _TRUST_EVIDENCE_PRIORITY_MEMORY_ROUNDS),
            _TRUST_EVIDENCE_PRIORITY_MEMORY_ROUNDS,
        )
        return effective
    retained = dict(trust_evidence_priority_memory) if isinstance(trust_evidence_priority_memory, dict) else {}
    retained_families = [
        str(value).strip()
        for value in list(retained.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if not retained_families:
        return {
            "blocked_candidate_count": 0,
            "blocked_subsystems": [],
            "prioritized_subsystems": [],
            "priority_benchmark_families": [],
            "family_details": [],
            "max_blocked_family_severity": 0,
            "sticky_rounds_remaining": 0,
            "retained_from_memory": False,
        }
    retained["retained_from_memory"] = True
    return retained


def _trust_breadth_priority_memory_summary(
    *,
    previous_memory: dict[str, object] | None,
    current_focus: dict[str, object] | None,
) -> dict[str, object]:
    previous_missing = []
    if isinstance(previous_memory, dict):
        previous_missing = [
            str(value).strip()
            for value in list(previous_memory.get("missing_required_family_clean_task_root_breadth", []) or [])
            if str(value).strip()
        ]
    current_missing = []
    current_details = []
    current_prioritized = []
    current_threshold = 0
    current_recovery_pressure = False
    current_recovery_pressure_evidence_score = 0
    if isinstance(current_focus, dict):
        current_missing = [
            str(value).strip()
            for value in list(current_focus.get("missing_required_family_clean_task_root_breadth", []) or [])
            if str(value).strip()
        ]
        current_details = list(current_focus.get("details", []) or [])
        current_prioritized = [
            str(value).strip()
            for value in list(current_focus.get("prioritized_subsystems", []) or [])
            if str(value).strip()
        ]
        current_threshold = max(
            0,
            _safe_int(current_focus.get("family_breadth_min_distinct_task_roots", 0), 0),
        )
        current_recovery_pressure = bool(current_focus.get("recovery_pressure", False))
        current_recovery_pressure_evidence_score = max(
            0,
            _safe_int(current_focus.get("recovery_pressure_evidence_score", 0), 0),
        )
    current_gap_severity = _trust_breadth_gap_severity(current_focus)
    previous_recovery_mix_sticky_rounds = max(
        0,
        _safe_int(
            previous_memory.get("recovery_mix_sticky_rounds_remaining", 0) if isinstance(previous_memory, dict) else 0,
            0,
        ),
    )
    previous_recovery_pressure_evidence_score = max(
        0,
        _safe_int(
            previous_memory.get("recovery_pressure_evidence_score", 0) if isinstance(previous_memory, dict) else 0,
            0,
        ),
    )
    if current_missing:
        effective_prioritized = list(current_prioritized)
        recovery_mix_sticky_rounds_remaining = 0
        effective_recovery_pressure_evidence_score = 0
        if current_gap_severity > 1:
            if current_recovery_pressure:
                effective_recovery_pressure_evidence_score = min(
                    6,
                    max(0, previous_recovery_pressure_evidence_score - 1) + max(1, current_recovery_pressure_evidence_score),
                )
                recovery_mix_sticky_rounds_remaining = (
                    _TRUST_BREADTH_RECOVERY_MIX_MEMORY_ROUNDS
                    + min(current_gap_severity - 1, 1)
                    + min(max(0, effective_recovery_pressure_evidence_score - 1), 2)
                )
            elif previous_recovery_mix_sticky_rounds > 0:
                recovery_mix_sticky_rounds_remaining = max(0, previous_recovery_mix_sticky_rounds - 1)
                effective_recovery_pressure_evidence_score = max(0, previous_recovery_pressure_evidence_score - 1)
            if recovery_mix_sticky_rounds_remaining > 0 and "recovery" not in effective_prioritized:
                effective_prioritized.append("recovery")
        sticky_rounds_remaining = _TRUST_BREADTH_PRIORITY_MEMORY_ROUNDS + min(
            current_gap_severity,
            _TRUST_BREADTH_PRIORITY_MEMORY_MAX_EXTRA_ROUNDS,
        )
        return {
            "missing_required_family_clean_task_root_breadth": current_missing,
            "prioritized_subsystems": effective_prioritized,
            "details": current_details,
            "family_breadth_min_distinct_task_roots": current_threshold,
            "max_remaining_clean_task_root_breadth_gap": current_gap_severity,
            "recovery_pressure": current_recovery_pressure,
            "recovery_pressure_evidence_score": effective_recovery_pressure_evidence_score,
            "recovery_mix_sticky_rounds_remaining": recovery_mix_sticky_rounds_remaining,
            "sticky_rounds_remaining": sticky_rounds_remaining,
        }
    sticky_rounds_remaining = max(
        0,
        _safe_int(
            previous_memory.get("sticky_rounds_remaining", 0) if isinstance(previous_memory, dict) else 0,
            0,
        )
        - 1,
    )
    if sticky_rounds_remaining <= 0 or not previous_missing:
        return {
            "missing_required_family_clean_task_root_breadth": [],
            "prioritized_subsystems": [],
            "details": [],
            "family_breadth_min_distinct_task_roots": 0,
            "max_remaining_clean_task_root_breadth_gap": 0,
            "recovery_pressure": False,
            "recovery_pressure_evidence_score": 0,
            "recovery_mix_sticky_rounds_remaining": 0,
            "sticky_rounds_remaining": 0,
        }
    recovery_mix_sticky_rounds_remaining = max(0, previous_recovery_mix_sticky_rounds - 1)
    recovery_pressure_evidence_score = max(0, previous_recovery_pressure_evidence_score - 1)
    prioritized_from_memory = [
        str(value).strip()
        for value in list(previous_memory.get("prioritized_subsystems", []) or [])
        if str(value).strip()
    ]
    if recovery_mix_sticky_rounds_remaining <= 0 or recovery_pressure_evidence_score <= 0:
        prioritized_from_memory = [subsystem for subsystem in prioritized_from_memory if subsystem != "recovery"]
    return {
        "missing_required_family_clean_task_root_breadth": previous_missing,
        "prioritized_subsystems": prioritized_from_memory,
        "details": list(previous_memory.get("details", []) or []),
        "family_breadth_min_distinct_task_roots": max(
            0,
            _safe_int(previous_memory.get("family_breadth_min_distinct_task_roots", 0), 0),
        ),
        "max_remaining_clean_task_root_breadth_gap": max(
            0,
            _safe_int(previous_memory.get("max_remaining_clean_task_root_breadth_gap", 0), 0),
        ),
        "recovery_pressure": bool(previous_memory.get("recovery_pressure", False)),
        "recovery_pressure_evidence_score": recovery_pressure_evidence_score,
        "recovery_mix_sticky_rounds_remaining": recovery_mix_sticky_rounds_remaining,
        "sticky_rounds_remaining": sticky_rounds_remaining,
    }


def _effective_trust_breadth_focus(
    current_focus: dict[str, object] | None,
    trust_breadth_priority_memory: dict[str, object] | None,
) -> dict[str, object]:
    current_missing = (
        [
            str(value).strip()
            for value in list(current_focus.get("missing_required_family_clean_task_root_breadth", []) or [])
            if str(value).strip()
        ]
        if isinstance(current_focus, dict)
        else []
    )
    if current_missing:
        effective = dict(current_focus) if isinstance(current_focus, dict) else {}
        retained = dict(trust_breadth_priority_memory) if isinstance(trust_breadth_priority_memory, dict) else {}
        if (
            _trust_breadth_gap_severity(effective) > 1
            and _safe_int(retained.get("recovery_mix_sticky_rounds_remaining", 0), 0) > 0
        ):
            prioritized = [
                subsystem
                for subsystem in list(effective.get("prioritized_subsystems", []) or [])
                if isinstance(subsystem, str) and subsystem.strip()
            ]
            if "trust" not in prioritized:
                prioritized.insert(0, "trust")
            if "recovery" not in prioritized:
                prioritized.append("recovery")
            effective["prioritized_subsystems"] = prioritized
            effective["recovery_mix_retained_from_memory"] = True
            effective["recovery_mix_sticky_rounds_remaining"] = _safe_int(
                retained.get("recovery_mix_sticky_rounds_remaining", 0),
                0,
            )
        else:
            effective["recovery_mix_retained_from_memory"] = False
            effective["recovery_mix_sticky_rounds_remaining"] = _safe_int(
                effective.get("recovery_mix_sticky_rounds_remaining", 0),
                0,
            )
        effective["retained_from_memory"] = False
        effective["sticky_rounds_remaining"] = _safe_int(
            effective.get("sticky_rounds_remaining", _TRUST_BREADTH_PRIORITY_MEMORY_ROUNDS),
            _TRUST_BREADTH_PRIORITY_MEMORY_ROUNDS,
        )
        return effective
    retained = dict(trust_breadth_priority_memory) if isinstance(trust_breadth_priority_memory, dict) else {}
    retained_missing = [
        str(value).strip()
        for value in list(retained.get("missing_required_family_clean_task_root_breadth", []) or [])
        if str(value).strip()
    ]
    if not retained_missing:
        return {
            "missing_required_family_clean_task_root_breadth": [],
            "required_family_clean_task_root_counts": {},
            "family_breadth_min_distinct_task_roots": 0,
            "prioritized_subsystems": [],
            "details": [],
            "max_remaining_clean_task_root_breadth_gap": 0,
            "recovery_pressure": False,
            "recovery_pressure_evidence_score": 0,
            "recovery_mix_sticky_rounds_remaining": 0,
            "recovery_mix_retained_from_memory": False,
            "sticky_rounds_remaining": 0,
            "retained_from_memory": False,
        }
    retained["required_family_clean_task_root_counts"] = {}
    retained["recovery_mix_retained_from_memory"] = _safe_int(
        retained.get("recovery_mix_sticky_rounds_remaining", 0),
        0,
    ) > 0 and "recovery" in list(retained.get("prioritized_subsystems", []) or [])
    retained["retained_from_memory"] = True
    return retained


def _trust_priority_discovery_budget(
    *,
    base_task_limit: int,
    base_observation_budget_seconds: float,
    trust_evidence_focus: dict[str, object] | None,
    trust_breadth_focus: dict[str, object] | None,
    widening_focus: dict[str, object] | None,
    priority_benchmark_families: list[str] | None,
) -> dict[str, object]:
    families = [
        str(value).strip()
        for value in list(priority_benchmark_families or [])
        if str(value).strip()
    ]
    if not families:
        return {
            "task_limit": max(0, int(base_task_limit)),
            "max_observation_seconds": max(0.0, float(base_observation_budget_seconds)),
            "task_limit_bonus": 0,
            "observation_bonus_seconds": 0.0,
            "priority_benchmark_family_weights": {},
            "priority_family_count": 0,
        }
    evidence_focus = dict(trust_evidence_focus) if isinstance(trust_evidence_focus, dict) else {}
    breadth_focus = dict(trust_breadth_focus) if isinstance(trust_breadth_focus, dict) else {}
    widening = dict(widening_focus) if isinstance(widening_focus, dict) else {}
    evidence_details = {
        str(detail.get("family", "")).strip(): dict(detail)
        for detail in list(evidence_focus.get("family_details", []) or [])
        if isinstance(detail, dict) and str(detail.get("family", "")).strip()
    }
    breadth_details = {
        str(detail.get("family", "")).strip(): dict(detail)
        for detail in list(breadth_focus.get("details", []) or [])
        if isinstance(detail, dict) and str(detail.get("family", "")).strip()
    }
    breadth_gaps = {
        str(value).strip()
        for value in list(breadth_focus.get("missing_required_family_clean_task_root_breadth", []) or [])
        if str(value).strip()
    }
    widening_priorities = {
        str(value).strip()
        for value in list(widening.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    }
    evidence_severity = _trust_evidence_focus_severity(evidence_focus)
    breadth_severity = _trust_breadth_gap_severity(breadth_focus)
    task_limit_bonus = min(
        _TRUST_PRIORITY_DISCOVERY_MAX_TASK_LIMIT_BONUS,
        max(0, len(families) - 1) + max(0, evidence_severity - 1) + max(0, breadth_severity - 1),
    )
    observation_bonus_seconds = float(task_limit_bonus) * _TRUST_PRIORITY_DISCOVERY_OBSERVATION_BONUS_SECONDS
    family_weights: dict[str, float] = {}
    for family in families:
        weight = 1.0
        if family in evidence_details:
            weight += min(2.0, 0.5 * max(1, _safe_int(evidence_details[family].get("severity", 0), 0)))
        if family in breadth_gaps:
            weight += 1.0
            breadth_detail = breadth_details.get(family, {})
            remaining = max(0, _safe_int(breadth_detail.get("remaining", 0), 0))
            observed = max(0, _safe_int(breadth_detail.get("observed", 0), 0))
            if remaining > 1:
                weight += min(1.0, 0.5 * float(remaining - 1))
            if observed <= 0 and remaining > 0:
                weight += 0.5
        if family in widening_priorities:
            weight += 0.25
        family_weights[family] = round(weight, 2)
    return {
        "task_limit": max(0, int(base_task_limit) + int(task_limit_bonus)),
        "max_observation_seconds": max(0.0, float(base_observation_budget_seconds) + observation_bonus_seconds),
        "task_limit_bonus": int(task_limit_bonus),
        "observation_bonus_seconds": float(observation_bonus_seconds),
        "priority_benchmark_family_weights": family_weights,
        "priority_family_count": len(families),
    }


def _paused_subsystems(
    *,
    recent_outcomes: list[dict[str, object]],
    promotion_results: list[dict[str, object]],
    failure_threshold: int,
) -> dict[str, dict[str, object]]:
    if failure_threshold <= 0:
        return {}
    stats: dict[str, dict[str, int]] = {}
    for outcome in recent_outcomes:
        subsystem = str(outcome.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        bucket = stats.setdefault(
            subsystem,
            {"timeouts": 0, "rejects": 0, "retains": 0, "generated": 0, "bootstrap_review_pending": 0},
        )
        if bool(outcome.get("observation_timed_out", False)):
            bucket["timeouts"] += 1
        if bool(outcome.get("generated_candidate", False)):
            bucket["generated"] += 1
    for result in promotion_results:
        subsystem = str(result.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        bucket = stats.setdefault(
            subsystem,
            {"timeouts": 0, "rejects": 0, "retains": 0, "generated": 0, "bootstrap_review_pending": 0},
        )
        state = str(result.get("finalize_state", "")).strip()
        compare_status = str(result.get("compare_status", "")).strip()
        finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
        if state == "reject":
            bucket["rejects"] += 1
        elif state == "retain":
            bucket["retains"] += 1
        if state != "retain" and (
            compare_status == "bootstrap_first_retain"
            or finalize_skip_reason in {"bootstrap_requires_review", "bootstrap_subsystem_not_allowed"}
        ) and (
            (
                bool(result.get("finalize_skipped", False))
                and finalize_skip_reason in {"bootstrap_requires_review", "bootstrap_subsystem_not_allowed"}
            )
            or not finalize_skip_reason
        ):
            bucket["bootstrap_review_pending"] += 1
    paused: dict[str, dict[str, object]] = {}
    for subsystem, bucket in stats.items():
        timeout_failures = int(bucket.get("timeouts", 0))
        reject_failures = int(bucket.get("rejects", 0))
        bootstrap_review_pending = int(bucket.get("bootstrap_review_pending", 0))
        retains = int(bucket.get("retains", 0))
        if retains > 0:
            continue
        if bootstrap_review_pending > 0:
            paused[subsystem] = {
                "reason": "bootstrap_review_pending",
                "failure_count": bootstrap_review_pending,
                "stats": dict(bucket),
            }
            continue
        if timeout_failures >= failure_threshold:
            paused[subsystem] = {
                "reason": "timeout_cooldown",
                "failure_count": timeout_failures,
                "stats": dict(bucket),
            }
            continue
        if reject_failures >= failure_threshold:
            paused[subsystem] = {
                "reason": "promotion_reject_cooldown",
                "failure_count": reject_failures,
                "stats": dict(bucket),
            }
    return paused


def _bootstrap_remediation_queues(
    *,
    paused_subsystems: dict[str, dict[str, object]],
    promotion_results: list[dict[str, object]],
    promotion_plan_candidates: list[dict[str, object]],
    trust_ledger: dict[str, object],
    rollout_gate: dict[str, object],
    allowed_bootstrap_subsystems: list[str],
    rollout_stage: str,
) -> dict[str, list[dict[str, object]]]:
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    trust_status = str(assessment.get("status", "")).strip() or "unknown"
    trust_breadth_finalize_gate_reason = _trust_breadth_finalize_gate_reason(trust_ledger)
    blocked_subsystems = {
        str(value).strip()
        for value in list(rollout_gate.get("blocked_subsystems", []) or [])
        if str(value).strip()
    }
    allowed_bootstrap = {
        str(value).strip()
        for value in list(allowed_bootstrap_subsystems or [])
        if str(value).strip()
    }
    normalized_rollout_stage = str(rollout_stage).strip()
    queues: dict[str, list[dict[str, object]]] = {
        "baseline_bootstrap": [],
        "trust_streak_accumulation": [],
        "protected_review_only": [],
    }
    plan_candidate_index: dict[tuple[str, str, str], dict[str, object]] = {}
    for candidate in promotion_plan_candidates:
        if not isinstance(candidate, dict):
            continue
        key = (
            str(candidate.get("selected_subsystem", "")).strip(),
            str(candidate.get("scope_id", "")).strip(),
            str(candidate.get("cycle_id", "")).strip(),
        )
        if any(key):
            plan_candidate_index[key] = candidate
    seen: set[tuple[str, str]] = set()
    for result in promotion_results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        paused = paused_subsystems.setdefault(subsystem, {})
        compare_status = str(result.get("compare_status", "")).strip()
        if str(paused.get("reason", "")).strip() != "bootstrap_review_pending" and compare_status != "bootstrap_first_retain":
            continue
        finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
        if not compare_status:
            compare_status = "bootstrap_first_retain"
        if compare_status != "bootstrap_first_retain":
            continue
        automated_queue_name = "baseline_bootstrap"
        automated_remediation_reason = finalize_skip_reason or "first_retain_review_required"
        if subsystem in blocked_subsystems or finalize_skip_reason == "bootstrap_subsystem_not_allowed":
            automated_queue_name = "protected_review_only"
            automated_remediation_reason = finalize_skip_reason or "protected_subsystem_review_required"
        elif trust_status != "trusted":
            automated_queue_name = "trust_streak_accumulation"
            automated_remediation_reason = f"trust_status={trust_status}"
        elif allowed_bootstrap and subsystem not in allowed_bootstrap:
            automated_queue_name = "protected_review_only"
            automated_remediation_reason = "bootstrap_subsystem_not_allowlisted"

        queue_names = [automated_queue_name]
        if (
            automated_queue_name == "trust_streak_accumulation"
            and normalized_rollout_stage == "compare_only"
            and subsystem not in blocked_subsystems
            and finalize_skip_reason != "bootstrap_subsystem_not_allowed"
        ):
            queue_names.append("baseline_bootstrap")

        plan_candidate = plan_candidate_index.get(
            (
                subsystem,
                str(result.get("scope_id", "")).strip(),
                str(result.get("cycle_id", "")).strip(),
            ),
            {},
        )
        validation_family_compare_guard_reasons = (
            _validation_family_compare_guard_reasons(plan_candidate)
            if isinstance(plan_candidate, dict)
            else []
        )
        bootstrap_review_guard_reasons = (
            _bootstrap_review_guard_reasons(plan_candidate)
            if isinstance(plan_candidate, dict)
            else []
        )
        retrieval_reuse_summary = _retrieval_reuse_summary(plan_candidate) if isinstance(plan_candidate, dict) else {}
        retrieval_reuse_priority_score = _retrieval_reuse_priority_score(plan_candidate) if isinstance(plan_candidate, dict) else 0
        validation_guard_reason_count = len(validation_family_compare_guard_reasons)
        validation_guard_severity = _validation_guard_severity_score(validation_family_compare_guard_reasons)
        validation_review_required = bool(validation_family_compare_guard_reasons)
        bootstrap_review_guard_reason_count = len(bootstrap_review_guard_reasons)
        bootstrap_review_required = bool(bootstrap_review_guard_reasons)
        primary_queue_name = automated_queue_name
        remediation_reason = automated_remediation_reason
        if validation_review_required and automated_queue_name != "protected_review_only":
            primary_queue_name = "protected_review_only"
            remediation_reason = "validation_family_compare_guard_review_required"
            queue_names = [primary_queue_name, *queue_names]
        elif bootstrap_review_required and automated_queue_name != "protected_review_only":
            primary_queue_name = "protected_review_only"
            remediation_reason = "bootstrap_review_guard_required"
            queue_names = [primary_queue_name, *queue_names]
        base_entry = {
            "selected_subsystem": subsystem,
            "selected_variant_id": str(plan_candidate.get("selected_variant_id", "")).strip()
            if isinstance(plan_candidate, dict)
            else "",
            "scope_id": str(result.get("scope_id", "")).strip(),
            "cycle_id": str(result.get("cycle_id", "")).strip(),
            "candidate_artifact_path": str(result.get("candidate_artifact_path", "")).strip(),
            "compare_status": compare_status,
            "compare_guard_reason": str(result.get("compare_guard_reason", "")).strip(),
            "finalize_skip_reason": finalize_skip_reason,
            "promotion_block_reason_code": _promotion_block_reason_code(result),
            "validation_family_compare_guard_reasons": validation_family_compare_guard_reasons,
            "validation_guard_reason_count": validation_guard_reason_count,
            "validation_guard_severity": validation_guard_severity,
            "validation_family_review_required": validation_review_required,
            "bootstrap_review_guard_reasons": bootstrap_review_guard_reasons,
            "bootstrap_review_guard_reason_count": bootstrap_review_guard_reason_count,
            "bootstrap_review_required": bootstrap_review_required,
            "retrieval_reuse_summary": retrieval_reuse_summary,
            "retrieval_reuse_priority_score": retrieval_reuse_priority_score,
            "retrieval_reuse_priority": retrieval_reuse_priority_score > 0,
            "lane_signal_queue": primary_queue_name,
        }
        for queue_name in queue_names:
            dedupe_key = (queue_name, subsystem)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entry = dict(base_entry)
            if queue_name == primary_queue_name:
                entry["remediation_reason"] = remediation_reason
                if validation_review_required and queue_name == "protected_review_only":
                    entry["review_focus"] = (
                        "validation-family compare guard present; route this first-retain candidate to explicit review"
                    )
                elif bootstrap_review_required and queue_name == "protected_review_only":
                    entry["review_focus"] = (
                        "bootstrap review guard present; route this first-retain candidate to explicit review"
                    )
            else:
                entry["remediation_reason"] = "first_retain_review_available_during_compare_only"
                if validation_review_required:
                    entry["finalize_gate_reason"] = "validation-family review pending before bootstrap lane finalize"
                elif bootstrap_review_required:
                    entry["finalize_gate_reason"] = "bootstrap review guard pending before bootstrap lane finalize"
                elif trust_breadth_finalize_gate_reason:
                    entry["finalize_gate_reason"] = trust_breadth_finalize_gate_reason
                else:
                    entry["finalize_gate_reason"] = f"bootstrap finalize still gated by trust_status={trust_status}"
            queues[queue_name].append(entry)
        paused["remediation_queue"] = primary_queue_name
        paused["lane_signal_queue"] = primary_queue_name
        paused["remediation_reason"] = remediation_reason
        paused["remediation_queues"] = list(queue_names)
        if validation_review_required:
            paused["validation_family_review_required"] = True
            paused["deferred_lane_signal_queue"] = automated_queue_name
        if bootstrap_review_required:
            paused["bootstrap_review_required"] = True
            paused["deferred_lane_signal_queue"] = automated_queue_name
    for queue_name, entries in queues.items():
        entries.sort(key=lambda entry: _bootstrap_queue_entry_sort_key(queue_name, entry))
    return queues


def _bootstrap_review_finalize_command(
    *,
    repo_root: Path,
    config: KernelConfig,
    entry: dict[str, object],
) -> str:
    subsystem = str(entry.get("selected_subsystem", "")).strip()
    scope_id = str(entry.get("scope_id", "")).strip()
    if not subsystem or not scope_id:
        return ""
    command = [
        sys.executable,
        str(repo_root / "scripts" / "finalize_latest_candidate_from_cycles.py"),
        "--frontier-report",
        str(config.improvement_reports_dir / "supervised_parallel_frontier.json"),
        "--subsystem",
        subsystem,
        "--scope-id",
        scope_id,
        "--candidate-index",
        "0",
        "--dry-run",
    ]
    return " ".join(command)


def _bootstrap_remediation_actions(
    *,
    repo_root: Path,
    config: KernelConfig,
    policy: SupervisorPolicy,
    queues: dict[str, list[dict[str, object]]],
    trust_ledger: dict[str, object],
    meta_policy: dict[str, object],
) -> list[dict[str, object]]:
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    trust_status = str(assessment.get("status", "")).strip() or "unknown"
    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    clean_success_streak = _safe_int(overall_summary.get("clean_success_streak", 0), 0)
    protected_min_clean_success_streak = _safe_int(
        meta_policy.get("protected_bootstrap_min_clean_success_streak", 0),
        0,
    )
    action_specs = [
        (
            "baseline_bootstrap",
            "prepare_bootstrap_review_package",
            "supervisor_baseline_bootstrap_queue.json",
        ),
        (
            "trust_streak_accumulation",
            "prepare_trust_streak_recovery_package",
            "supervisor_trust_streak_recovery_queue.json",
        ),
        (
            "protected_review_only",
            "prepare_protected_review_package",
            "supervisor_protected_review_queue.json",
        ),
    ]
    actions: list[dict[str, object]] = []
    for queue_name, action_kind, filename in action_specs:
        raw_entries = queues.get(queue_name, [])
        if not isinstance(raw_entries, list) or not raw_entries:
            continue
        entries: list[dict[str, object]] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            entry["review_finalize_command"] = _bootstrap_review_finalize_command(
                repo_root=repo_root,
                config=config,
                entry=entry,
            )
            entry["current_trust_status"] = trust_status
            entry["current_clean_success_streak"] = clean_success_streak
            if queue_name == "trust_streak_accumulation":
                entry["required_trust_status"] = "trusted"
                entry["recovery_focus"] = "accumulate clean unattended supervisor rounds before bootstrap finalize"
            elif queue_name == "protected_review_only":
                entry["required_clean_success_streak"] = protected_min_clean_success_streak
                entry["review_focus"] = "protected subsystem first-retain requires explicit protected-lane review"
            else:
                if str(entry.get("lane_signal_queue", "")).strip() == "trust_streak_accumulation":
                    entry["required_trust_status"] = "trusted"
                    entry["review_focus"] = (
                        "prepare first-retain review package now; bootstrap finalize remains trust-gated until trust recovers"
                    )
                else:
                    entry["review_focus"] = "prepare first-retain review package and operator decision"
            entries.append(entry)
        if not entries:
            continue
        actions.append(
            {
                "kind": action_kind,
                "enabled": True,
                "queue_name": queue_name,
                "queue_kind": queue_name,
                "autonomy_mode": policy.autonomy_mode,
                "rollout_stage": policy.rollout_stage,
                "bootstrap_finalize_policy": policy.bootstrap_finalize_policy,
                "trust_status": trust_status,
                "report_path": str(config.improvement_reports_dir / filename),
                "entries": entries,
            }
        )
    actions.sort(key=_bootstrap_action_sort_key)
    return actions


def _primary_queue_entry(entries: object) -> dict[str, object]:
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if isinstance(entry, dict):
            return dict(entry)
    return {}


def _selected_bootstrap_queue_entry(latest_round: dict[str, object] | None) -> tuple[str, dict[str, object]]:
    if not isinstance(latest_round, dict):
        return "", {}
    decisions = latest_round.get("decisions", {})
    if not isinstance(decisions, dict):
        return "", {}
    queues = decisions.get("bootstrap_remediation_queues", {})
    if not isinstance(queues, dict):
        return "", {}
    protected_review_entry = _primary_queue_entry(queues.get("protected_review_only", []))
    if protected_review_entry and _validation_family_compare_guard_reasons(protected_review_entry):
        return "protected_review_only", protected_review_entry
    for queue_name in ("trust_streak_accumulation", "baseline_bootstrap", "protected_review_only"):
        entry = _primary_queue_entry(queues.get(queue_name, []))
        if entry:
            return queue_name, entry
    return "", {}


class SupervisorPolicy:
    def __init__(
        self,
        *,
        autonomy_mode: str,
        max_discovery_workers: int,
        discovery_task_limit: int,
        discovery_observation_budget_seconds: float,
        max_promotion_candidates: int,
        command_timeout_seconds: int,
        lane_failure_threshold: int,
        sleep_seconds: float,
        include_curriculum: bool,
        include_failure_curriculum: bool,
        generated_curriculum_budget_seconds: float,
        failure_curriculum_budget_seconds: float,
        bootstrap_finalize_policy: str,
        provider: str,
        model_name: str,
        rollout_stage: str,
        max_meta_promotions_per_round: int,
        meta_trust_clean_success_streak: int,
        meta_policy_path: str,
    ) -> None:
        self.autonomy_mode = autonomy_mode
        self.max_discovery_workers = max_discovery_workers
        self.discovery_task_limit = discovery_task_limit
        self.discovery_observation_budget_seconds = discovery_observation_budget_seconds
        self.max_promotion_candidates = max_promotion_candidates
        self.command_timeout_seconds = command_timeout_seconds
        self.lane_failure_threshold = lane_failure_threshold
        self.sleep_seconds = sleep_seconds
        self.include_curriculum = include_curriculum
        self.include_failure_curriculum = include_failure_curriculum
        self.generated_curriculum_budget_seconds = generated_curriculum_budget_seconds
        self.failure_curriculum_budget_seconds = failure_curriculum_budget_seconds
        self.bootstrap_finalize_policy = bootstrap_finalize_policy
        self.provider = provider
        self.model_name = model_name
        self.rollout_stage = rollout_stage
        self.max_meta_promotions_per_round = max_meta_promotions_per_round
        self.meta_trust_clean_success_streak = meta_trust_clean_success_streak
        self.meta_policy_path = meta_policy_path

    def to_dict(self) -> dict[str, object]:
        return {
            "autonomy_mode": self.autonomy_mode,
            "max_discovery_workers": self.max_discovery_workers,
            "discovery_task_limit": self.discovery_task_limit,
            "discovery_observation_budget_seconds": self.discovery_observation_budget_seconds,
            "max_promotion_candidates": self.max_promotion_candidates,
            "command_timeout_seconds": self.command_timeout_seconds,
            "lane_failure_threshold": self.lane_failure_threshold,
            "sleep_seconds": self.sleep_seconds,
            "include_curriculum": self.include_curriculum,
            "include_failure_curriculum": self.include_failure_curriculum,
            "generated_curriculum_budget_seconds": self.generated_curriculum_budget_seconds,
            "failure_curriculum_budget_seconds": self.failure_curriculum_budget_seconds,
            "bootstrap_finalize_policy": self.bootstrap_finalize_policy,
            "provider": self.provider,
            "model_name": self.model_name,
            "rollout_stage": self.rollout_stage,
            "max_meta_promotions_per_round": self.max_meta_promotions_per_round,
            "meta_trust_clean_success_streak": self.meta_trust_clean_success_streak,
            "meta_policy_path": self.meta_policy_path,
        }


def _load_work_manifest(repo_root: Path) -> dict[str, object]:
    return _load_json(repo_root / "config" / "supervised_parallel_work_manifest.json")


def _path_overlap(path: str, pattern: str) -> bool:
    normalized_path = str(path).strip().strip("/")
    normalized_pattern = str(pattern).strip().strip("/")
    if not normalized_path or not normalized_pattern:
        return False
    return (
        normalized_path == normalized_pattern
        or normalized_path.startswith(normalized_pattern + "/")
        or normalized_pattern.startswith(normalized_path + "/")
    )


def _manifest_lanes(work_manifest: dict[str, object]) -> list[dict[str, object]]:
    lanes = work_manifest.get("lanes", [])
    if not isinstance(lanes, list):
        return []
    return [lane for lane in lanes if isinstance(lane, dict)]


def _lane_matches_subsystem(lane: dict[str, object], subsystem: str) -> bool:
    token = str(subsystem).strip().lower()
    if not token:
        return False
    lane_tokens = [
        str(lane.get("lane_id", "")).strip().lower(),
        str(lane.get("title", "")).strip().lower(),
        str(lane.get("primary_question", "")).strip().lower(),
        str(lane.get("objective", "")).strip().lower(),
    ]
    if any(token in value for value in lane_tokens if value):
        return True
    owned_paths = lane.get("owned_paths", [])
    if not isinstance(owned_paths, list):
        return False
    return any(token in str(path).strip().lower() for path in owned_paths if str(path).strip())


def _lane_protected_paths(lane: dict[str, object], meta_policy: dict[str, object]) -> list[str]:
    owned_paths = lane.get("owned_paths", [])
    if not isinstance(owned_paths, list):
        return []
    protected_paths = meta_policy.get("protected_paths", [])
    if not isinstance(protected_paths, list):
        return []
    overlaps: list[str] = []
    for owned in owned_paths:
        owned_text = str(owned).strip()
        if not owned_text:
            continue
        if any(_path_overlap(owned_text, str(protected).strip()) for protected in protected_paths if str(protected).strip()):
            overlaps.append(owned_text)
    return sorted(set(overlaps))


def _classify_frontier_candidates(
    *,
    frontier_state: dict[str, object],
    work_manifest: dict[str, object],
    meta_policy: dict[str, object],
) -> list[dict[str, object]]:
    frontier_candidates = frontier_state.get("frontier_candidates", [])
    if not isinstance(frontier_candidates, list):
        return []
    protected_subsystems = {
        str(value).strip()
        for value in meta_policy.get("protected_subsystems", [])
        if str(value).strip()
    }
    lanes = _manifest_lanes(work_manifest)
    classified: list[dict[str, object]] = []
    for candidate in frontier_candidates:
        if not isinstance(candidate, dict):
            continue
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        matched_lanes = [lane for lane in lanes if _lane_matches_subsystem(lane, subsystem)]
        matched_lane_ids = [
            str(lane.get("lane_id", "")).strip()
            for lane in matched_lanes
            if str(lane.get("lane_id", "")).strip()
        ]
        matched_lane_protected_paths: list[str] = []
        for lane in matched_lanes:
            matched_lane_protected_paths.extend(_lane_protected_paths(lane, meta_policy))
        protected = bool(subsystem in protected_subsystems or matched_lane_protected_paths)
        classified.append(
            {
                "scope_id": str(candidate.get("scope_id", "")).strip(),
                "cycle_id": str(candidate.get("cycle_id", "")).strip(),
                "selected_subsystem": subsystem,
                "candidate_artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                "observed_benchmark_families": list(candidate.get("observed_benchmark_families", []) or []),
                "validation_family_summary": (
                    dict(candidate.get("validation_family_summary", {}))
                    if isinstance(candidate.get("validation_family_summary", {}), dict)
                    else {}
                ),
                "matched_lane_ids": matched_lane_ids,
                "matched_protected_paths": sorted(set(matched_lane_protected_paths)),
                "protected": protected,
                "protected_reasons": [
                    *([f"protected_subsystem:{subsystem}"] if subsystem in protected_subsystems else []),
                    *[f"protected_path:{path}" for path in sorted(set(matched_lane_protected_paths))],
                ],
            }
        )
    return classified


def _load_meta_policy(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    protected_subsystems = [
        str(value).strip()
        for value in payload.get("protected_subsystems", _DEFAULT_META_PROTECTED_SUBSYSTEMS)
        if str(value).strip()
    ]
    protected_paths = [
        str(value).strip()
        for value in payload.get("protected_paths", _DEFAULT_META_PROTECTED_PATHS)
        if str(value).strip()
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "protected_subsystems": sorted(set(protected_subsystems)),
        "protected_paths": sorted(set(protected_paths)),
        "protected_bootstrap_min_clean_success_streak": max(
            0,
            _safe_int(payload.get("protected_bootstrap_min_clean_success_streak", 0), 0),
        ),
    }


def _bootstrap_finalize_allowed_subsystems(
    *,
    policy: SupervisorPolicy,
    trust_ledger: dict[str, object],
    meta_policy: dict[str, object],
    frontier_state: dict[str, object],
    rollout_gate: dict[str, object],
    apply_finalize: bool,
) -> tuple[list[str], list[str]]:
    if not apply_finalize:
        return [], []
    normalized_policy = str(policy.bootstrap_finalize_policy).strip()
    if normalized_policy not in {"allow", "trusted", "evidence"}:
        return [], []
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    trust_passed = bool(overall_assessment.get("passed", False))
    trust_status = str(overall_assessment.get("status", "")).strip()
    if normalized_policy == "trusted" and trust_status != "trusted":
        return [], ["bootstrap_trust_status_not_trusted"]

    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    clean_success_streak = _safe_int(overall_summary.get("clean_success_streak", 0), 0)
    false_pass_risk_count = _safe_int(overall_summary.get("false_pass_risk_count", 0), 0)
    unexpected_change_report_count = _safe_int(overall_summary.get("unexpected_change_report_count", 0), 0)
    coverage_summary = trust_ledger.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    required_family_clean_task_root_counts = coverage_summary.get("required_family_clean_task_root_counts", {})
    if not isinstance(required_family_clean_task_root_counts, dict):
        required_family_clean_task_root_counts = {}
    missing_required_family_runtime_managed_signal_breadth = [
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_runtime_managed_signal", [])
        if str(value).strip()
    ]
    missing_required_family_clean_task_root_breadth = [
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_clean_task_root_breadth", [])
        if str(value).strip()
    ]
    breadth_missing_families = missing_required_family_runtime_managed_signal_breadth or missing_required_family_clean_task_root_breadth
    family_breadth_min_distinct_task_roots = max(
        0,
        _safe_int(coverage_summary.get("family_breadth_min_distinct_task_roots", 0), 0),
    )
    if normalized_policy == "evidence":
        reasons: list[str] = []
        required_clean_success_streak = max(1, int(policy.meta_trust_clean_success_streak))
        if not trust_passed:
            reasons.append("bootstrap_trust_not_passed")
        if clean_success_streak < required_clean_success_streak:
            reasons.append(
                f"bootstrap_clean_success_streak:{clean_success_streak}<{required_clean_success_streak}"
            )
        if false_pass_risk_count > 0:
            reasons.append(f"bootstrap_false_pass_risk_count:{false_pass_risk_count}>0")
        if unexpected_change_report_count > 0:
            reasons.append(f"bootstrap_unexpected_change_report_count:{unexpected_change_report_count}>0")
        for family in breadth_missing_families:
            observed = max(0, _safe_int(required_family_clean_task_root_counts.get(family, 0), 0))
            reasons.append(
                f"bootstrap_required_family_clean_task_root_breadth:{family}:{observed}<{family_breadth_min_distinct_task_roots}"
            )
        if reasons:
            return [], reasons
    protected_min_clean_success_streak = max(
        int(policy.meta_trust_clean_success_streak),
        _safe_int(meta_policy.get("protected_bootstrap_min_clean_success_streak", 0), 0),
    )
    protected_subsystems = {
        str(value).strip()
        for value in rollout_gate.get("allowed_protected_subsystems", []) or meta_policy.get("protected_subsystems", [])
        if str(value).strip()
    }
    blocked_subsystems = {
        str(value).strip()
        for value in rollout_gate.get("blocked_subsystems", [])
        if str(value).strip()
    }
    candidates = frontier_state.get("frontier_candidates", [])
    if not isinstance(candidates, list):
        return [], []

    allowed: list[str] = []
    reasons: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if not bool(candidate.get("generated_candidate", False)) or not bool(candidate.get("candidate_exists", False)):
            continue
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        if not subsystem or subsystem in seen or subsystem in blocked_subsystems:
            continue
        if subsystem in protected_subsystems and clean_success_streak < protected_min_clean_success_streak:
            reasons.append(
                f"protected_bootstrap_clean_success_streak:{subsystem}:{clean_success_streak}<{protected_min_clean_success_streak}"
            )
            continue
        allowed.append(subsystem)
        seen.add(subsystem)
    return sorted(allowed), reasons


def _trust_breadth_finalize_gate_reason(trust_ledger: dict[str, object]) -> str:
    coverage_summary = trust_ledger.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    missing_families = [
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_runtime_managed_signal", [])
        if str(value).strip()
    ] or [
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_clean_task_root_breadth", [])
        if str(value).strip()
    ]
    if not missing_families:
        return ""
    required_family_clean_task_root_counts = coverage_summary.get("required_family_clean_task_root_counts", {})
    if not isinstance(required_family_clean_task_root_counts, dict):
        required_family_clean_task_root_counts = {}
    threshold = max(0, _safe_int(coverage_summary.get("family_breadth_min_distinct_task_roots", 0), 0))
    details = ", ".join(
        f"{family}:{max(0, _safe_int(required_family_clean_task_root_counts.get(family, 0), 0))}/{threshold}"
        for family in missing_families
    )
    return f"bootstrap finalize still gated by required clean task-root breadth ({details})"


def _active_claim_ledger(queue_state: dict[str, object]) -> dict[str, object]:
    active_leases = queue_state.get("active_leases", [])
    if not isinstance(active_leases, list):
        active_leases = []
    path_claims: dict[str, list[str]] = {}
    claims: list[dict[str, object]] = []
    for lease in active_leases:
        if not isinstance(lease, dict):
            continue
        claimed_paths = [
            str(value).strip()
            for value in list(lease.get("claimed_paths", []) or [])
            if str(value).strip()
        ]
        claim = {
            "job_id": str(lease.get("job_id", "")).strip(),
            "task_id": str(lease.get("task_id", "")).strip(),
            "shared_repo_id": str(lease.get("shared_repo_id", "")).strip(),
            "worker_branch": str(lease.get("worker_branch", "")).strip(),
            "target_branch": str(lease.get("target_branch", "")).strip(),
            "claimed_paths": claimed_paths,
        }
        claims.append(claim)
        owner = claim["job_id"] or claim["task_id"] or claim["worker_branch"] or "unknown"
        for path in claimed_paths:
            path_claims.setdefault(path, []).append(owner)
    conflicts = [
        {"path": path, "owners": owners}
        for path, owners in sorted(path_claims.items())
        if len(owners) > 1
    ]
    return {
        "active_claim_count": len(claims),
        "claims": claims,
        "path_conflicts": conflicts,
    }


def _frontier_candidate_subsystems(frontier_state: dict[str, object]) -> list[str]:
    candidates = frontier_state.get("frontier_candidates", [])
    if not isinstance(candidates, list):
        return []
    return [
        str(candidate.get("selected_subsystem", "")).strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("selected_subsystem", "")).strip()
    ]


def _lane_matches_breadth_family(lane: dict[str, object], family: str) -> bool:
    token = str(family).strip().lower()
    if not token:
        return False
    lane_tokens = [
        str(lane.get("lane_id", "")).strip().lower(),
        str(lane.get("title", "")).strip().lower(),
        str(lane.get("primary_question", "")).strip().lower(),
        str(lane.get("objective", "")).strip().lower(),
        str(lane.get("recommended_command", "")).strip().lower(),
    ]
    return any(token in value for value in lane_tokens if value)


def _lane_allocator(
    *,
    work_manifest: dict[str, object],
    claim_ledger: dict[str, object],
    paused_subsystems: dict[str, dict[str, object]],
    recommended_subsystems: list[str],
    meta_policy: dict[str, object],
    trust_ledger: dict[str, object] | None = None,
) -> dict[str, object]:
    claims = claim_ledger.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    lanes = _manifest_lanes(work_manifest)
    active_claimed_paths: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for path in list(claim.get("claimed_paths", []) or []):
            text = str(path).strip()
            if text:
                active_claimed_paths.add(text)
    lane_summaries: list[dict[str, object]] = []
    available_lanes: list[dict[str, object]] = []
    breadth_focus = _trust_breadth_focus(trust_ledger)
    breadth_families = [
        str(value).strip()
        for value in list(breadth_focus.get("missing_required_family_clean_task_root_breadth", []) or [])
        if str(value).strip()
    ]
    for lane in lanes:
        owned_paths = [
            str(path).strip()
            for path in list(lane.get("owned_paths", []) or [])
            if str(path).strip()
        ]
        claimed = any(any(_path_overlap(path, claimed_path) for claimed_path in active_claimed_paths) for path in owned_paths)
        protected_paths = _lane_protected_paths(lane, meta_policy)
        lane_summary = {
            "lane_id": str(lane.get("lane_id", "")).strip(),
            "title": str(lane.get("title", "")).strip(),
            "objective": str(lane.get("objective", "")).strip(),
            "primary_question": str(lane.get("primary_question", "")).strip(),
            "recommended_command": str(lane.get("recommended_command", "")).strip(),
            "owned_paths": owned_paths,
            "claimed": claimed,
            "protected": bool(protected_paths),
            "protected_paths": protected_paths,
        }
        lane_summaries.append(lane_summary)
        if not claimed:
            available_lanes.append(lane_summary)
    assignments: list[dict[str, object]] = []
    used_lane_ids: set[str] = set()
    for subsystem in recommended_subsystems:
        if subsystem in paused_subsystems:
            continue
        breadth_matching = []
        if subsystem in {"trust", "recovery"} and breadth_families:
            breadth_matching = [
                lane
                for lane in available_lanes
                if lane["lane_id"] not in used_lane_ids
                and any(_lane_matches_breadth_family(lane, family) for family in breadth_families)
            ]
        matching = [
            lane
            for lane in available_lanes
            if lane["lane_id"] not in used_lane_ids and _lane_matches_subsystem(lane, subsystem)
        ]
        preferred_matching = breadth_matching or matching
        if preferred_matching:
            chosen = preferred_matching[0]
            used_lane_ids.add(str(chosen["lane_id"]))
            assignments.append(
                {
                    "subsystem": subsystem,
                    "lane_id": str(chosen["lane_id"]),
                    "title": str(chosen["title"]),
                    "owned_paths": list(chosen["owned_paths"]),
                    "protected": bool(chosen["protected"]),
                    "breadth_focus_families": list(breadth_families) if breadth_matching else [],
                }
            )
            continue
        assignments.append(
            {
                "subsystem": subsystem,
                "lane_id": "",
                "title": "",
                "owned_paths": [],
                "protected": False,
                "status": "unmapped",
                "breadth_focus_families": list(breadth_families) if subsystem in {"trust", "recovery"} else [],
            }
        )
    return {
        "lanes": lane_summaries,
        "assignments": assignments,
        "available_lane_count": sum(1 for lane in lane_summaries if not bool(lane.get("claimed", False))),
        "claimed_lane_count": sum(1 for lane in lane_summaries if bool(lane.get("claimed", False))),
    }


def _rollback_plan(
    *,
    trust_ledger: dict[str, object],
    frontier_state: dict[str, object],
    promotion_pass_state: dict[str, object],
    rollout_gate: dict[str, object],
) -> dict[str, object]:
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    trust_passed = bool(overall_assessment.get("passed", False))
    results = promotion_pass_state.get("results", [])
    if not isinstance(results, list):
        results = []
    protected_subsystems = {
        str(value).strip()
        for value in rollout_gate.get("protected_frontier_subsystems", [])
        if str(value).strip()
    }
    frontier_trust_evidence = _frontier_candidate_trust_evidence(
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
    )
    rollback_candidates: list[dict[str, object]] = []
    retained_candidate_trust_evidence_failed = False
    for result in results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        artifact_path = str(result.get("candidate_artifact_path", "")).strip()
        if str(result.get("finalize_state", "")).strip() != "retain":
            continue
        if not bool(result.get("apply_finalize", True)):
            continue
        if not artifact_path:
            continue
        trust_evidence = _matching_frontier_trust_evidence(
            result=result,
            frontier_trust_evidence=frontier_trust_evidence,
        )
        trust_evidence_blocked_reasons = [
            str(value).strip()
            for value in list(trust_evidence.get("blocked_reasons", []) or [])
            if str(value).strip()
        ]
        candidate_trust_evidence_failed = bool(trust_evidence_blocked_reasons)
        if not trust_passed and not candidate_trust_evidence_failed:
            rollback_reason = (
                "trust_regressed_after_protected_retain"
                if subsystem in protected_subsystems
                else "trust_regressed_after_governed_retain"
            )
        elif candidate_trust_evidence_failed:
            retained_candidate_trust_evidence_failed = True
            rollback_reason = (
                "trust_evidence_insufficient_after_protected_retain"
                if subsystem in protected_subsystems
                else "trust_evidence_insufficient_after_governed_retain"
            )
        else:
            continue
        rollback_candidates.append(
            {
                "selected_subsystem": subsystem,
                "candidate_artifact_path": artifact_path,
                "reason": rollback_reason,
                "trust_evidence_blocked_reasons": trust_evidence_blocked_reasons,
            }
        )
    return {
        "required": bool(rollback_candidates) and (not trust_passed or retained_candidate_trust_evidence_failed),
        "rollback_candidates": rollback_candidates,
        "trigger_reason": (
            ""
            if not rollback_candidates
            else "retain_with_failed_trust"
            if not trust_passed
            else "retain_with_insufficient_trust_evidence"
        ),
    }


def _canary_tracked_candidates(
    *,
    frontier_state: dict[str, object],
    trust_ledger: dict[str, object],
    promotion_pass_state: dict[str, object],
    rollout_gate: dict[str, object],
    meta_policy: dict[str, object],
    previous_canary_lifecycle: dict[str, object],
) -> list[dict[str, object]]:
    results = promotion_pass_state.get("results", [])
    if not isinstance(results, list):
        results = []
    protected_subsystems = {
        str(value).strip()
        for value in list(meta_policy.get("protected_subsystems", []) or [])
        if str(value).strip()
    }
    protected_subsystems.update(
        str(value).strip()
        for value in list(rollout_gate.get("protected_frontier_subsystems", []) or [])
        if str(value).strip()
    )
    previous_candidates = previous_canary_lifecycle.get("tracked_candidates", [])
    if isinstance(previous_candidates, list):
        protected_subsystems.update(
            str(candidate.get("selected_subsystem", "")).strip()
            for candidate in previous_candidates
            if isinstance(candidate, dict) and str(candidate.get("selected_subsystem", "")).strip()
        )
    frontier_trust_evidence = _frontier_candidate_trust_evidence(
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
    )
    tracked: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        artifact_path = str(result.get("candidate_artifact_path", "")).strip()
        if not subsystem or not artifact_path:
            continue
        if str(result.get("finalize_state", "")).strip() != "retain":
            continue
        if not bool(result.get("apply_finalize", True)):
            continue
        key = (subsystem, artifact_path)
        if key in seen:
            continue
        seen.add(key)
        trust_evidence = _matching_frontier_trust_evidence(
            result=result,
            frontier_trust_evidence=frontier_trust_evidence,
        )
        tracked.append(
            {
                "selected_subsystem": subsystem,
                "candidate_artifact_path": artifact_path,
                "protected": subsystem in protected_subsystems,
                "required_trust_families": list(trust_evidence.get("required_trust_families", []) or []),
                "trust_evidence_blocked_reasons": list(trust_evidence.get("blocked_reasons", []) or []),
            }
        )
    if tracked:
        return tracked
    if isinstance(previous_candidates, list):
        fallback: list[dict[str, object]] = []
        for candidate in previous_candidates:
            if not isinstance(candidate, dict):
                continue
            subsystem = str(candidate.get("selected_subsystem", "")).strip()
            artifact_path = str(candidate.get("candidate_artifact_path", "")).strip()
            if not subsystem or not artifact_path:
                continue
            fallback.append(
                {
                    "selected_subsystem": subsystem,
                    "candidate_artifact_path": artifact_path,
                    "protected": bool(candidate.get("protected", False)),
                    "required_trust_families": list(candidate.get("required_trust_families", []) or []),
                    "trust_evidence_blocked_reasons": list(
                        candidate.get("trust_evidence_blocked_reasons", []) or []
                    ),
                }
            )
        return fallback
    return []


def _canary_lifecycle(
    *,
    policy: SupervisorPolicy,
    frontier_state: dict[str, object],
    trust_ledger: dict[str, object],
    promotion_pass_state: dict[str, object],
    rollout_gate: dict[str, object],
    rollback_plan: dict[str, object],
    meta_policy: dict[str, object],
    previous_canary_lifecycle: dict[str, object],
) -> dict[str, object]:
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    trust_passed = bool(assessment.get("passed", False))
    trust_status = str(assessment.get("status", "")).strip() or "unknown"
    previous_state = str(previous_canary_lifecycle.get("state", "")).strip()
    previous_validation = previous_canary_lifecycle.get("validation", {})
    if not isinstance(previous_validation, dict):
        previous_validation = {}
    tracked_candidates = _canary_tracked_candidates(
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
        promotion_pass_state=promotion_pass_state,
        rollout_gate=rollout_gate,
        meta_policy=meta_policy,
        previous_canary_lifecycle=previous_canary_lifecycle,
    )
    blocked_reasons: list[str] = []
    promotion_resume_allowed = True
    validation_required = False
    state = "idle"
    resume_rule = "promotion can continue immediately because no governed canary is active"
    if rollback_plan.get("required", False):
        state = "rollback_pending"
        blocked_reasons = ["rollback_pending"]
        promotion_resume_allowed = False
        validation_required = True
        resume_rule = "promotion resumes only after rollback succeeds and post-rollback validation passes"
    elif bool(previous_validation.get("failed", False)):
        state = "rollback_validation_failed"
        blocked_reasons = ["rollback_validation_failed"]
        promotion_resume_allowed = False
        validation_required = True
        resume_rule = "promotion resumes only after rollback validation is rerun and passes"
    elif bool(previous_validation.get("attempted", False)) and bool(previous_validation.get("passed", False)):
        if trust_passed:
            state = "resume_ready"
            resume_rule = "promotion may resume because rollback validation passed and trust recovered"
        else:
            state = "resume_blocked"
            blocked_reasons = ["trust_not_recovered_after_rollback"]
            promotion_resume_allowed = False
            resume_rule = "promotion remains blocked until rollback validation passes and trust recovers"
    elif tracked_candidates and policy.rollout_stage == "canary":
        if trust_passed and previous_state == "canary_monitoring":
            state = "resume_ready"
            resume_rule = "promotion may resume after one stable canary observation round with trusted status"
        elif trust_passed:
            state = "canary_monitoring"
            blocked_reasons = ["canary_observation_pending"]
            promotion_resume_allowed = False
            resume_rule = "promotion resumes after one trusted observation round or an explicit rollback/validation path"
        else:
            state = "rollback_pending"
            blocked_reasons = ["rollback_pending"]
            promotion_resume_allowed = False
            validation_required = True
            resume_rule = "promotion resumes only after rollback succeeds and post-rollback validation passes"
    elif tracked_candidates:
        state = "resume_ready" if trust_passed else "resume_blocked"
        if not trust_passed:
            blocked_reasons = [f"trust_status={trust_status}"]
            promotion_resume_allowed = False
        resume_rule = (
            "promotion may resume because the retained candidate is outside canary staging"
            if trust_passed
            else "promotion remains blocked until trust recovers"
        )
    return {
        "state": state,
        "previous_state": previous_state,
        "tracked_candidates": tracked_candidates,
        "validation_required": validation_required,
        "promotion_resume_allowed": promotion_resume_allowed,
        "blocked_reasons": blocked_reasons,
        "resume_rule": resume_rule,
        "trust_status": trust_status,
        "validation": {
            "attempted": False,
            "passed": False,
            "failed": False,
            "results": [],
        },
    }


def _apply_execution_results_to_canary_lifecycle(
    *,
    canary_lifecycle: dict[str, object],
    executions: list[dict[str, object]],
    trust_ledger: dict[str, object],
) -> dict[str, object]:
    updated = {
        "state": str(canary_lifecycle.get("state", "")).strip() or "idle",
        "previous_state": str(canary_lifecycle.get("previous_state", "")).strip(),
        "tracked_candidates": list(canary_lifecycle.get("tracked_candidates", []) or []),
        "validation_required": bool(canary_lifecycle.get("validation_required", False)),
        "promotion_resume_allowed": bool(canary_lifecycle.get("promotion_resume_allowed", True)),
        "blocked_reasons": list(canary_lifecycle.get("blocked_reasons", []) or []),
        "resume_rule": str(canary_lifecycle.get("resume_rule", "")).strip(),
        "trust_status": str(canary_lifecycle.get("trust_status", "")).strip() or "unknown",
        "validation": {
            "attempted": False,
            "passed": False,
            "failed": False,
            "results": [],
        },
    }
    rollback_executions = [
        execution
        for execution in executions
        if isinstance(execution, dict)
        and str(execution.get("kind", "")).strip() == "rollback_artifact"
        and not bool(execution.get("skipped", False))
    ]
    validation_executions = [
        execution
        for execution in executions
        if isinstance(execution, dict)
        and str(execution.get("kind", "")).strip() == "validate_rollback_artifact"
        and not bool(execution.get("skipped", False))
    ]
    if not rollback_executions and not validation_executions:
        return updated
    validation_results = [
        {
            "selected_subsystem": str(execution.get("selected_subsystem", "")).strip(),
            "artifact_path": str(execution.get("artifact_path", "")).strip(),
            "returncode": _safe_int(execution.get("returncode", 1), 1),
            "stdout": str(execution.get("stdout", "")).strip(),
            "stderr": str(execution.get("stderr", "")).strip(),
        }
        for execution in validation_executions
    ]
    updated["validation"] = {
        "attempted": bool(validation_results),
        "passed": bool(validation_results) and all(result["returncode"] == 0 for result in validation_results),
        "failed": any(result["returncode"] != 0 for result in validation_results),
        "results": validation_results,
    }
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    trust_passed = bool(assessment.get("passed", False))
    trust_status = str(assessment.get("status", "")).strip() or updated["trust_status"]
    updated["trust_status"] = trust_status
    if validation_results:
        if updated["validation"]["passed"]:
            updated["validation_required"] = False
            if trust_passed:
                updated["state"] = "resume_ready"
                updated["promotion_resume_allowed"] = True
                updated["blocked_reasons"] = []
                updated["resume_rule"] = "promotion may resume because rollback validation passed and trust recovered"
            else:
                updated["state"] = "resume_blocked"
                updated["promotion_resume_allowed"] = False
                updated["blocked_reasons"] = ["trust_not_recovered_after_rollback"]
                updated["resume_rule"] = "promotion remains blocked until rollback validation passes and trust recovers"
        else:
            updated["state"] = "rollback_validation_failed"
            updated["validation_required"] = True
            updated["promotion_resume_allowed"] = False
            updated["blocked_reasons"] = ["rollback_validation_failed"]
            updated["resume_rule"] = "promotion resumes only after rollback validation is rerun and passes"
        return updated
    updated["state"] = "rollback_validation_pending"
    updated["validation_required"] = True
    updated["promotion_resume_allowed"] = False
    updated["blocked_reasons"] = ["rollback_validation_pending"]
    updated["resume_rule"] = "promotion resumes only after rollback validation runs and passes"
    return updated


def _rollout_gate(
    *,
    policy: SupervisorPolicy,
    trust_ledger: dict[str, object],
    meta_policy: dict[str, object],
    frontier_state: dict[str, object],
    work_manifest: dict[str, object],
) -> dict[str, object]:
    candidate_classification = _classify_frontier_candidates(
        frontier_state=frontier_state,
        work_manifest=work_manifest,
        meta_policy=meta_policy,
    )
    protected_frontier_subsystems = sorted(
        {
            str(candidate.get("selected_subsystem", "")).strip()
            for candidate in candidate_classification
            if bool(candidate.get("protected", False))
            and str(candidate.get("selected_subsystem", "")).strip()
        }
    )
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    trust_passed = bool(overall_assessment.get("passed", False))
    trust_status = str(overall_assessment.get("status", "")).strip() or "unknown"
    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    clean_success_streak = _safe_int(overall_summary.get("clean_success_streak", 0), 0)
    frontier_trust_evidence = _frontier_candidate_trust_evidence(
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
    )
    allowed_protected_subsystems: list[str] = []
    blocked_subsystems: list[str] = []
    reasons: list[str] = []
    meta_finalize_budget = 0
    if protected_frontier_subsystems:
        if policy.rollout_stage in {"shadow", "compare_only"}:
            blocked_subsystems = list(protected_frontier_subsystems)
            reasons.append(f"rollout_stage={policy.rollout_stage}")
        elif policy.rollout_stage == "canary":
            if not trust_passed:
                blocked_subsystems = list(protected_frontier_subsystems)
                reasons.append(f"meta_trust_status={trust_status}")
            elif clean_success_streak < policy.meta_trust_clean_success_streak:
                blocked_subsystems = list(protected_frontier_subsystems)
                reasons.append(
                    "meta_clean_success_streak="
                    f"{clean_success_streak} below required={policy.meta_trust_clean_success_streak}"
                )
            else:
                meta_finalize_budget = max(0, int(policy.max_meta_promotions_per_round))
                allowed_protected_subsystems = list(protected_frontier_subsystems[:meta_finalize_budget])
                blocked_subsystems = list(protected_frontier_subsystems[meta_finalize_budget:])
                if blocked_subsystems:
                    reasons.append(f"canary_budget={meta_finalize_budget}")
        elif policy.rollout_stage == "broad":
            meta_finalize_budget = max(0, int(policy.max_meta_promotions_per_round))
            if meta_finalize_budget > 0:
                allowed_protected_subsystems = list(protected_frontier_subsystems[:meta_finalize_budget])
                blocked_subsystems = list(protected_frontier_subsystems[meta_finalize_budget:])
                if blocked_subsystems:
                    reasons.append(f"meta_budget_cap={meta_finalize_budget}")
    trust_evidence_blocked_subsystems = [
        str(value).strip()
        for value in list(frontier_trust_evidence.get("blocked_subsystems", []) or [])
        if str(value).strip()
    ]
    for reason, count in sorted(
        (
            (str(key).strip(), _safe_int(value, 0))
            for key, value in dict(frontier_trust_evidence.get("blocked_reason_counts", {})).items()
        ),
        key=lambda item: item[0],
    ):
        if reason and count > 0:
            reasons.append(f"{reason} x{count}")
    blocked_subsystems = sorted(set(blocked_subsystems) | set(trust_evidence_blocked_subsystems))
    return {
        "candidate_classification": candidate_classification,
        "protected_frontier_subsystems": protected_frontier_subsystems,
        "allowed_protected_subsystems": allowed_protected_subsystems,
        "blocked_subsystems": blocked_subsystems,
        "blocked_reasons": reasons,
        "meta_finalize_budget": meta_finalize_budget,
        "trust_status": trust_status,
        "clean_success_streak": clean_success_streak,
        "trust_evidence_summary": frontier_trust_evidence,
    }


def _at_least_mode(current: str, target: str) -> bool:
    return _AUTONOMY_MODE_ORDER.get(str(current).strip(), -1) >= _AUTONOMY_MODE_ORDER.get(str(target).strip(), -1)


def _at_least_stage(current: str, target: str) -> bool:
    return _ROLLOUT_STAGE_ORDER.get(str(current).strip(), -1) >= _ROLLOUT_STAGE_ORDER.get(str(target).strip(), -1)


def _autonomy_widening_summary(
    *,
    policy: SupervisorPolicy,
    frontier_state: dict[str, object],
    trust_ledger: dict[str, object],
    rollout_gate: dict[str, object],
    canary_lifecycle: dict[str, object],
    blocked_conditions: list[str],
) -> dict[str, object]:
    frontier_summary = frontier_state.get("summary", {})
    if not isinstance(frontier_summary, dict):
        frontier_summary = {}
    candidate_classification = rollout_gate.get("candidate_classification", [])
    if not isinstance(candidate_classification, list):
        candidate_classification = []
    protected_candidates = [
        candidate for candidate in candidate_classification if isinstance(candidate, dict) and bool(candidate.get("protected", False))
    ]
    non_protected_candidates = [
        candidate for candidate in candidate_classification if isinstance(candidate, dict) and not bool(candidate.get("protected", False))
    ]
    bridge_clean_success_threshold = max(1, int(policy.meta_trust_clean_success_streak))
    non_protected_candidate_evidence = []
    for candidate in non_protected_candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_evidence = _candidate_trust_evidence(candidate, trust_ledger)
        bridge = _candidate_autonomy_bridge(
            candidate_evidence,
            clean_success_threshold=bridge_clean_success_threshold,
        )
        non_protected_candidate_evidence.append(
            {
                "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                "scope_id": str(candidate.get("scope_id", "")).strip(),
                "cycle_id": str(candidate.get("cycle_id", "")).strip(),
                **candidate_evidence,
                "autonomy_bridge": bridge,
            }
        )
    widening_eligible_non_protected = [
        candidate
        for candidate in non_protected_candidate_evidence
        if not bool(candidate.get("blocked", False))
        or bool(dict(candidate.get("autonomy_bridge", {})).get("eligible", False))
    ]
    bridge_eligible_non_protected = [
        candidate
        for candidate in non_protected_candidate_evidence
        if bool(candidate.get("blocked", False))
        and bool(dict(candidate.get("autonomy_bridge", {})).get("eligible", False))
    ]
    blocked_non_protected = [
        candidate
        for candidate in non_protected_candidate_evidence
        if bool(candidate.get("blocked", False))
        and not bool(dict(candidate.get("autonomy_bridge", {})).get("eligible", False))
    ]
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    trust_passed = bool(overall_assessment.get("passed", False))
    trust_status = str(overall_assessment.get("status", "")).strip() or "unknown"
    clean_success_streak = max(0, _safe_int(overall_summary.get("clean_success_streak", 0), 0))
    false_pass_risk_count = max(0, _safe_int(overall_summary.get("false_pass_risk_count", 0), 0))
    hidden_side_effect_risk_count = max(0, _safe_int(overall_summary.get("hidden_side_effect_risk_count", 0), 0))
    rollback_performed_count = max(0, _safe_int(overall_summary.get("rollback_performed_count", 0), 0))
    completed_runs = max(0, _safe_int(frontier_summary.get("completed_runs", 0), 0))
    healthy_runs = max(0, _safe_int(frontier_summary.get("healthy_runs", 0), 0))
    timed_out_runs = max(0, _safe_int(frontier_summary.get("timed_out_runs", 0), 0))
    budget_exceeded_runs = max(0, _safe_int(frontier_summary.get("budget_exceeded_runs", 0), 0))
    generated_candidate_runs = max(0, _safe_int(frontier_summary.get("generated_candidate_runs", 0), 0))
    healthy_run_rate = 0.0 if completed_runs <= 0 else healthy_runs / float(completed_runs)
    timeout_rate = 0.0 if completed_runs <= 0 else timed_out_runs / float(completed_runs)
    budget_exceeded_rate = 0.0 if completed_runs <= 0 else budget_exceeded_runs / float(completed_runs)
    blocked_reason_tokens = [
        str(reason).strip()
        for reason in list(rollout_gate.get("blocked_reasons", []) or [])
        if str(reason).strip()
    ]
    blocked_subsystems = {
        str(value).strip()
        for value in list(rollout_gate.get("blocked_subsystems", []) or [])
        if str(value).strip()
    }
    canary_state = str(canary_lifecycle.get("state", "")).strip() or "idle"
    canary_blocked_reasons = [
        str(reason).strip()
        for reason in list(canary_lifecycle.get("blocked_reasons", []) or [])
        if str(reason).strip()
    ]
    recommendation_reasons: list[str] = []
    blockers: list[str] = []
    target_autonomy_mode = str(policy.autonomy_mode).strip()
    target_rollout_stage = str(policy.rollout_stage).strip()
    target_bootstrap_finalize_policy = str(policy.bootstrap_finalize_policy).strip()
    promotion_scope = "none"
    if non_protected_candidates:
        if not trust_passed and not widening_eligible_non_protected:
            blockers.append(f"trust_status={trust_status}")
        if false_pass_risk_count > 0:
            blockers.append(f"false_pass_risk_count={false_pass_risk_count}")
        if hidden_side_effect_risk_count > 0:
            blockers.append(f"hidden_side_effect_risk_count={hidden_side_effect_risk_count}")
        if rollback_performed_count > 0:
            blockers.append(f"rollback_performed_count={rollback_performed_count}")
        if healthy_run_rate < 0.55:
            blockers.append(f"healthy_run_rate={healthy_run_rate:.2f}<0.55")
        if timeout_rate > 0.20:
            blockers.append(f"timeout_rate={timeout_rate:.2f}>0.20")
        if budget_exceeded_rate > 0.25:
            blockers.append(f"budget_exceeded_rate={budget_exceeded_rate:.2f}>0.25")
        if canary_state in {"rollback_pending", "rollback_validation_failed", "rollback_validation_pending"}:
            blockers.append(f"canary_state={canary_state}")
        if generated_candidate_runs <= 0:
            blockers.append("generated_candidate_runs=0")
        if not widening_eligible_non_protected:
            blockers.append("no_non_protected_candidates_with_satisfied_trust_or_bridge_evidence")
        if not blockers:
            target_autonomy_mode = "promote"
            target_rollout_stage = "canary"
            promotion_scope = "non_protected_canary"
            recommendation_reasons.extend(
                [
                    f"widening_eligible_non_protected_candidates={len(widening_eligible_non_protected)}",
                    f"healthy_run_rate={healthy_run_rate:.2f}",
                    f"timeout_rate={timeout_rate:.2f}",
                    f"budget_exceeded_rate={budget_exceeded_rate:.2f}",
                ]
            )
            if bridge_eligible_non_protected:
                promotion_scope = "non_protected_canary_bridge"
                bridge_subsystems = sorted(
                    {
                        str(candidate.get("selected_subsystem", "")).strip()
                        for candidate in bridge_eligible_non_protected
                        if str(candidate.get("selected_subsystem", "")).strip()
                    }
                )
                recommendation_reasons.append(
                    "light_supervision_bridge_subsystems="
                    + ",".join(bridge_subsystems)
                )
                recommendation_reasons.append(
                    f"light_supervision_bridge_threshold={bridge_clean_success_threshold}"
                )
            if (
                trust_status == "trusted"
                and clean_success_streak >= max(1, int(policy.meta_trust_clean_success_streak))
                and healthy_run_rate >= 0.65
                and timeout_rate <= 0.12
                and budget_exceeded_rate <= 0.15
                and canary_state in {"idle", "resume_ready"}
            ):
                target_rollout_stage = "broad"
                promotion_scope = "non_protected_broad"
                recommendation_reasons.append(
                    f"clean_success_streak={clean_success_streak}>={max(1, int(policy.meta_trust_clean_success_streak))}"
                )
            if trust_status == "trusted":
                target_bootstrap_finalize_policy = "evidence"
    elif protected_candidates:
        blockers.append("only_protected_candidates_present")
    else:
        blockers.append("no_frontier_candidates")
    escalation_available = (
        not blockers
        and (
            not _at_least_mode(policy.autonomy_mode, target_autonomy_mode)
            or not _at_least_stage(policy.rollout_stage, target_rollout_stage)
            or (
                target_bootstrap_finalize_policy != str(policy.bootstrap_finalize_policy).strip()
                and target_bootstrap_finalize_policy
            )
        )
    )
    widening_command = ""
    if escalation_available:
        command = [
            sys.executable,
            "scripts/run_supervisor_loop.py",
            "--autonomy-mode",
            target_autonomy_mode,
            "--rollout-stage",
            target_rollout_stage,
            "--bootstrap-finalize-policy",
            target_bootstrap_finalize_policy or str(policy.bootstrap_finalize_policy).strip() or "operator_review",
        ]
        widening_command = " ".join(command)
    eligible_non_protected_priority_by_subsystem: dict[str, int] = {}
    eligible_non_protected_cluster_by_subsystem: dict[str, str] = {}
    eligible_non_protected_priority_by_cluster: dict[str, int] = {}
    for candidate in widening_eligible_non_protected:
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        cluster_key = _candidate_widening_cluster_key(candidate)
        bridge = dict(candidate.get("autonomy_bridge", {})) if isinstance(candidate.get("autonomy_bridge", {}), dict) else {}
        if not subsystem:
            continue
        if subsystem in blocked_subsystems and not bool(bridge.get("eligible", False)):
            continue
        priority_score = 0
        if bool(bridge.get("eligible", False)):
            priority_score += 8
        priority_score += min(
            6,
            sum(
                max(0, _safe_int(entry.get("light_supervision_clean_successes", 0), 0))
                for entry in list(bridge.get("supported_families", []) or [])
                if isinstance(entry, dict)
            ),
        )
        priority_score += min(
            4,
            sum(
                max(0, _safe_int(entry.get("contract_clean_failure_recovery_clean_successes", 0), 0))
                for entry in list(bridge.get("supported_families", []) or [])
                if isinstance(entry, dict)
            ),
        )
        priority_score += min(3, len(list(candidate.get("required_trust_families", []) or [])))
        priority_score -= min(3, len(list(candidate.get("blocked_reasons", []) or [])))
        priority_score = max(0, priority_score)
        eligible_non_protected_cluster_by_subsystem[subsystem] = cluster_key
        eligible_non_protected_priority_by_subsystem[subsystem] = max(
            priority_score,
            eligible_non_protected_priority_by_subsystem.get(subsystem, 0),
        )
        eligible_non_protected_priority_by_cluster[cluster_key] = max(
            priority_score,
            eligible_non_protected_priority_by_cluster.get(cluster_key, 0),
        )
    eligible_non_protected_prioritized_subsystems = [
        subsystem
        for subsystem, _score in sorted(
            eligible_non_protected_priority_by_subsystem.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return {
        "current_autonomy_mode": str(policy.autonomy_mode).strip(),
        "current_rollout_stage": str(policy.rollout_stage).strip(),
        "current_bootstrap_finalize_policy": str(policy.bootstrap_finalize_policy).strip(),
        "recommended_autonomy_mode": target_autonomy_mode,
        "recommended_rollout_stage": target_rollout_stage,
        "recommended_bootstrap_finalize_policy": target_bootstrap_finalize_policy,
        "promotion_scope": promotion_scope,
        "escalation_available": escalation_available,
        "eligible_non_protected_candidate_count": len(widening_eligible_non_protected),
        "bridge_non_protected_candidate_count": len(bridge_eligible_non_protected),
        "total_non_protected_candidate_count": len(non_protected_candidates),
        "eligible_protected_candidate_count": len(protected_candidates),
        "eligible_non_protected_subsystems": eligible_non_protected_prioritized_subsystems,
        "eligible_non_protected_priority_by_subsystem": eligible_non_protected_priority_by_subsystem,
        "eligible_non_protected_cluster_by_subsystem": eligible_non_protected_cluster_by_subsystem,
        "eligible_non_protected_priority_by_cluster": eligible_non_protected_priority_by_cluster,
        "bridge_non_protected_subsystems": sorted(
            {
                str(candidate.get("selected_subsystem", "")).strip()
                for candidate in bridge_eligible_non_protected
                if str(candidate.get("selected_subsystem", "")).strip()
            }
        ),
        "blocked_non_protected_subsystems": sorted(
            {
                str(candidate.get("selected_subsystem", "")).strip()
                for candidate in blocked_non_protected
                if str(candidate.get("selected_subsystem", "")).strip()
            }
        ),
        "blocked_non_protected_reasons_by_subsystem": {
            str(candidate.get("selected_subsystem", "")).strip(): list(candidate.get("blocked_reasons", []) or [])
            for candidate in blocked_non_protected
            if str(candidate.get("selected_subsystem", "")).strip()
        },
        "blocked_protected_subsystems": sorted(
            {
                str(candidate.get("selected_subsystem", "")).strip()
                for candidate in protected_candidates
                if str(candidate.get("selected_subsystem", "")).strip()
            }
        ),
        "readiness_signals": {
            "trust_passed": trust_passed,
            "trust_status": trust_status,
            "clean_success_streak": clean_success_streak,
            "healthy_run_rate": round(healthy_run_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
            "budget_exceeded_rate": round(budget_exceeded_rate, 4),
            "generated_candidate_runs": generated_candidate_runs,
            "canary_state": canary_state,
            "light_supervision_bridge_candidate_count": len(bridge_eligible_non_protected),
            "light_supervision_bridge_threshold": bridge_clean_success_threshold,
        },
        "recommendation_reasons": recommendation_reasons,
        "blockers": blockers,
        "rollout_gate_blocked_reasons": blocked_reason_tokens,
        "canary_blocked_reasons": canary_blocked_reasons,
        "widening_command": widening_command,
        "summary": (
            f"recommend {target_autonomy_mode}/{target_rollout_stage}"
            if not blockers
            else f"hold {policy.autonomy_mode}/{policy.rollout_stage}: {'; '.join(blockers)}"
        ),
    }


def _widening_discovery_focus(
    *,
    rollout_gate: dict[str, object],
    trust_ledger: dict[str, object],
) -> dict[str, object]:
    candidate_classification = rollout_gate.get("candidate_classification", [])
    if not isinstance(candidate_classification, list):
        candidate_classification = []
    non_protected_candidates = [
        candidate for candidate in candidate_classification if isinstance(candidate, dict) and not bool(candidate.get("protected", False))
    ]
    non_protected_candidate_evidence = []
    for candidate in non_protected_candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_evidence = _candidate_trust_evidence(candidate, trust_ledger)
        non_protected_candidate_evidence.append(
            {
                "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                **candidate_evidence,
                "autonomy_bridge": _candidate_autonomy_bridge(
                    candidate_evidence,
                    clean_success_threshold=1,
                ),
            }
        )
    eligible = [
        candidate
        for candidate in non_protected_candidate_evidence
        if not bool(candidate.get("blocked", False))
        or bool(dict(candidate.get("autonomy_bridge", {})).get("eligible", False))
    ]
    blocked = [
        candidate
        for candidate in non_protected_candidate_evidence
        if bool(candidate.get("blocked", False))
        and not bool(dict(candidate.get("autonomy_bridge", {})).get("eligible", False))
    ]
    priority_benchmark_families: list[str] = []
    for candidate in eligible:
        for family in list(candidate.get("required_trust_families", []) or []):
            family_name = str(family).strip()
            if family_name and family_name not in priority_benchmark_families:
                priority_benchmark_families.append(family_name)
    return {
        "prioritized_subsystems": [
            str(candidate.get("selected_subsystem", "")).strip()
            for candidate in eligible
            if str(candidate.get("selected_subsystem", "")).strip()
        ],
        "blocked_subsystems": [
            str(candidate.get("selected_subsystem", "")).strip()
            for candidate in blocked
            if str(candidate.get("selected_subsystem", "")).strip()
        ],
        "blocked_reasons_by_subsystem": {
            str(candidate.get("selected_subsystem", "")).strip(): list(candidate.get("blocked_reasons", []) or [])
            for candidate in blocked
            if str(candidate.get("selected_subsystem", "")).strip()
        },
        "priority_benchmark_families": priority_benchmark_families,
    }


def _operator_gated_reasons(
    *,
    policy: SupervisorPolicy,
    trust_ledger: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    status = str(assessment.get("status", "")).strip()
    passed = bool(assessment.get("passed", False))
    if policy.autonomy_mode == "shadow":
        reasons.append("autonomy_mode=shadow")
    if policy.autonomy_mode == "promote" and not passed:
        reasons.append(f"trust_status={status or 'unknown'}")
    return reasons


def _recommended_discovery_subsystems(
    *,
    config: KernelConfig,
    paused_subsystems: dict[str, dict[str, object]],
    validation_guard_pressure_summary: dict[str, object] | None,
    trust_evidence_focus: dict[str, object] | None,
    trust_breadth_focus: dict[str, object] | None,
    widening_focus: dict[str, object] | None,
    reserved_priority_subsystem_slots: list[str] | None,
    worker_count: int,
) -> tuple[list[str], dict[str, object]]:
    ranked = _planner_ranked_subsystems(config, worker_count=worker_count)
    evidence_focus = dict(trust_evidence_focus) if isinstance(trust_evidence_focus, dict) else {}
    breadth_focus = dict(trust_breadth_focus) if isinstance(trust_breadth_focus, dict) else {}
    widening_focus = dict(widening_focus) if isinstance(widening_focus, dict) else {}
    prioritized_evidence_subsystems = [
        subsystem
        for subsystem in evidence_focus.get("prioritized_subsystems", [])
        if isinstance(subsystem, str) and subsystem.strip()
    ]
    prioritized_breadth_subsystems = [
        subsystem
        for subsystem in breadth_focus.get("prioritized_subsystems", [])
        if isinstance(subsystem, str) and subsystem.strip()
    ]
    prioritized_subsystems: list[str] = []
    prioritized_widening_subsystems = [
        subsystem
        for subsystem in widening_focus.get("prioritized_subsystems", [])
        if isinstance(subsystem, str) and subsystem.strip()
    ]
    for subsystem in [
        *prioritized_evidence_subsystems,
        *prioritized_breadth_subsystems,
        *prioritized_widening_subsystems,
    ]:
        if subsystem not in prioritized_subsystems:
            prioritized_subsystems.append(subsystem)
    reserved_priority_slots = [
        str(value).strip()
        for value in list(reserved_priority_subsystem_slots or [])
        if str(value).strip()
    ]
    if reserved_priority_slots:
        ranked = reserved_priority_slots + [
            subsystem for subsystem in ranked if subsystem not in reserved_priority_slots
        ]
        if prioritized_widening_subsystems:
            ranked = reserved_priority_slots + [
                subsystem
                for subsystem in [*prioritized_widening_subsystems, *ranked]
                if subsystem not in reserved_priority_slots
            ]
    elif prioritized_subsystems:
        ranked = prioritized_subsystems + [subsystem for subsystem in ranked if subsystem not in prioritized_subsystems]
    selected: list[str] = []
    clean_candidates: list[str] = []
    guarded_candidates: list[tuple[int, int, int, int, str]] = []
    guard_lookup: dict[str, dict[str, object]] = {}
    if isinstance(validation_guard_pressure_summary, dict):
        guarded_subsystems = validation_guard_pressure_summary.get("guarded_subsystems", [])
        if isinstance(guarded_subsystems, list):
            for raw in guarded_subsystems:
                if not isinstance(raw, dict):
                    continue
                subsystem = str(raw.get("selected_subsystem", "")).strip()
                if subsystem:
                    guard_lookup[subsystem] = raw
    for index, subsystem in enumerate(ranked):
        if subsystem in paused_subsystems:
            continue
        guard_pressure = guard_lookup.get(subsystem, {})
        guard_severity = _safe_int(guard_pressure.get("validation_guard_severity", 0), 0)
        guard_count = _safe_int(guard_pressure.get("guarded_candidate_count", 0), 0)
        reason_count = _safe_int(guard_pressure.get("validation_guard_reason_count", 0), 0)
        if guard_severity > 0 or guard_count > 0 or reason_count > 0:
            guarded_candidates.append((guard_severity, guard_count, reason_count, index, subsystem))
        else:
            clean_candidates.append(subsystem)
    for subsystem in clean_candidates:
        if len(selected) >= worker_count:
            break
        selected.append(subsystem)
    if len(selected) < worker_count:
        for _, _, _, _, subsystem in sorted(guarded_candidates):
            if len(selected) >= worker_count:
                break
            selected.append(subsystem)
    selected_guarded_subsystems = [
        subsystem
        for subsystem in selected
        if subsystem in guard_lookup
    ]
    deprioritized_guarded_subsystems = [
        subsystem
        for _, _, _, _, subsystem in sorted(guarded_candidates)
        if subsystem not in selected
    ]
    return selected, {
        "selected_subsystems": list(selected),
        "selected_clean_subsystems": [subsystem for subsystem in selected if subsystem not in guard_lookup],
        "selected_guarded_subsystems": selected_guarded_subsystems,
        "deprioritized_guarded_subsystems": deprioritized_guarded_subsystems,
        "validation_guarded_subsystem_count": len(guard_lookup),
        "trust_evidence_focus": evidence_focus,
        "trust_breadth_focus": breadth_focus,
        "prioritized_trust_evidence_subsystems": prioritized_evidence_subsystems,
        "prioritized_trust_breadth_subsystems": prioritized_breadth_subsystems,
        "prioritized_widening_subsystems": prioritized_widening_subsystems,
        "reserved_priority_subsystem_slots": reserved_priority_slots,
        "blocked_widening_subsystems": [
            str(value).strip()
            for value in list(widening_focus.get("blocked_subsystems", []) or [])
            if str(value).strip()
        ],
        "blocked_widening_reasons_by_subsystem": {
            str(key).strip(): list(value or [])
            for key, value in dict(widening_focus.get("blocked_reasons_by_subsystem", {}) or {}).items()
            if str(key).strip()
        },
        "priority_benchmark_families": list(
            dict.fromkeys(
                [
                    *[
                        str(value).strip()
                        for value in _trust_breadth_priority_families(breadth_focus)
                        if str(value).strip()
                    ],
                    *[
                        str(value).strip()
                        for value in list(widening_focus.get("priority_benchmark_families", []) or [])
                        if str(value).strip()
                    ],
                    *[
                        str(value).strip()
                        for value in list(evidence_focus.get("priority_benchmark_families", []) or [])
                        if str(value).strip()
                    ],
                ]
            )
        ),
    }


def _next_retry_at(*, now: datetime, sleep_seconds: float) -> str:
    delay = max(0.0, float(sleep_seconds or 0.0))
    return _isoformat(now + timedelta(seconds=delay))


def _build_round_actions(
    *,
    config: KernelConfig,
    repo_root: Path,
    policy: SupervisorPolicy,
    queue_state: dict[str, object],
    frontier_state: dict[str, object],
    promotion_plan_state: dict[str, object] | None = None,
    promotion_pass_state: dict[str, object] | None = None,
    trust_ledger: dict[str, object] | None = None,
    recent_outcomes: list[dict[str, object]] | None = None,
    previous_canary_lifecycle: dict[str, object] | None = None,
    previous_validation_guard_memory: dict[str, object] | None = None,
    previous_bootstrap_retrieval_priority_memory: dict[str, object] | None = None,
    previous_trust_evidence_priority_memory: dict[str, object] | None = None,
    previous_trust_breadth_priority_memory: dict[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(promotion_plan_state, dict):
        promotion_plan_state = {}
    if not isinstance(promotion_pass_state, dict):
        promotion_pass_state = {}
    if not isinstance(trust_ledger, dict):
        trust_ledger = {}
    if not isinstance(recent_outcomes, list):
        recent_outcomes = []
    promotion_results = promotion_pass_state.get("results", [])
    if not isinstance(promotion_results, list):
        promotion_results = []
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    trust_status = str(overall_assessment.get("status", "")).strip()
    paused = _paused_subsystems(
        recent_outcomes=recent_outcomes,
        promotion_results=promotion_results,
        failure_threshold=policy.lane_failure_threshold,
    )
    operator_gated_reasons = _operator_gated_reasons(policy=policy, trust_ledger=trust_ledger)
    meta_policy = _load_meta_policy(Path(policy.meta_policy_path))
    work_manifest = _load_work_manifest(repo_root)
    rollout_gate = _rollout_gate(
        policy=policy,
        trust_ledger=trust_ledger,
        meta_policy=meta_policy,
        frontier_state=frontier_state,
        work_manifest=work_manifest,
    )
    validation_guard_pressure_summary = _validation_guard_pressure_summary(promotion_plan_state)
    validation_guard_memory = _validation_guard_memory_summary(
        previous_memory=previous_validation_guard_memory,
        current_pressure_summary=validation_guard_pressure_summary,
        promotion_results=promotion_results,
    )
    effective_validation_guard_pressure_summary = _effective_validation_guard_pressure_summary(
        validation_guard_pressure_summary,
        validation_guard_memory,
    )
    current_trust_evidence_focus = _trust_evidence_focus(
        frontier_trust_evidence=rollout_gate.get("trust_evidence_summary", {}),
        trust_ledger=trust_ledger,
    )
    trust_evidence_priority_memory = _trust_evidence_priority_memory_summary(
        previous_memory=previous_trust_evidence_priority_memory,
        current_focus=current_trust_evidence_focus,
    )
    effective_trust_evidence_focus = _effective_trust_evidence_focus(
        current_trust_evidence_focus,
        trust_evidence_priority_memory,
    )
    current_trust_breadth_focus = _trust_breadth_focus(trust_ledger)
    trust_breadth_priority_memory = _trust_breadth_priority_memory_summary(
        previous_memory=previous_trust_breadth_priority_memory,
        current_focus=current_trust_breadth_focus,
    )
    effective_trust_breadth_focus = _effective_trust_breadth_focus(
        current_trust_breadth_focus,
        trust_breadth_priority_memory,
    )
    widening_discovery_focus = _widening_discovery_focus(
        rollout_gate=rollout_gate,
        trust_ledger=trust_ledger,
    )
    active_leases = queue_state.get("active_leases", [])
    active_lease_count = len(active_leases) if isinstance(active_leases, list) else 0
    available_worker_slots = max(0, int(policy.max_discovery_workers) - active_lease_count)
    trust_breadth_reserved_subsystem_slots = _trust_breadth_reserved_subsystem_slots(
        effective_trust_breadth_focus,
        available_worker_slots=available_worker_slots,
    )
    trust_breadth_reserved_recovery_roles = _trust_breadth_reserved_recovery_roles(
        effective_trust_breadth_focus,
        reserved_subsystem_slots=trust_breadth_reserved_subsystem_slots,
    )
    trust_breadth_reserved_recovery_strategy_families = _trust_breadth_reserved_recovery_strategy_families(
        effective_trust_breadth_focus,
        reserved_subsystem_slots=trust_breadth_reserved_subsystem_slots,
        reserved_recovery_roles=trust_breadth_reserved_recovery_roles,
    )
    try:
        trust_breadth_reserved_variant_ids = _reserved_variant_ids_for_subsystems(
            config,
            reserved_subsystem_slots=trust_breadth_reserved_subsystem_slots,
            trust_breadth_focus=effective_trust_breadth_focus,
        )
    except TypeError:
        trust_breadth_reserved_variant_ids = _reserved_variant_ids_for_subsystems(
            config,
            reserved_subsystem_slots=trust_breadth_reserved_subsystem_slots,
        )
    recommended_subsystems, discovery_priority_summary = _recommended_discovery_subsystems(
        config=config,
        paused_subsystems=paused,
        validation_guard_pressure_summary=effective_validation_guard_pressure_summary,
        trust_evidence_focus=effective_trust_evidence_focus,
        trust_breadth_focus=effective_trust_breadth_focus,
        widening_focus=widening_discovery_focus,
        reserved_priority_subsystem_slots=trust_breadth_reserved_subsystem_slots,
        worker_count=available_worker_slots,
    ) if available_worker_slots > 0 else ([], {"selected_subsystems": [], "selected_clean_subsystems": [], "selected_guarded_subsystems": [], "deprioritized_guarded_subsystems": [], "validation_guarded_subsystem_count": _safe_int(effective_validation_guard_pressure_summary.get("guarded_subsystem_count", 0), 0), "reserved_priority_subsystem_slots": [], "prioritized_widening_subsystems": [], "blocked_widening_subsystems": [], "blocked_widening_reasons_by_subsystem": {}})
    claim_ledger = _active_claim_ledger(queue_state)
    lane_allocator = _lane_allocator(
        work_manifest=work_manifest,
        claim_ledger=claim_ledger,
        paused_subsystems=paused,
        recommended_subsystems=recommended_subsystems,
        meta_policy=meta_policy,
        trust_ledger=trust_ledger,
    )
    rollback_plan = _rollback_plan(
        trust_ledger=trust_ledger,
        frontier_state=frontier_state,
        promotion_pass_state=promotion_pass_state,
        rollout_gate=rollout_gate,
    )
    canary_lifecycle = _canary_lifecycle(
        policy=policy,
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
        promotion_pass_state=promotion_pass_state,
        rollout_gate=rollout_gate,
        rollback_plan=rollback_plan,
        meta_policy=meta_policy,
        previous_canary_lifecycle=previous_canary_lifecycle or {},
    )

    frontier_candidates = frontier_state.get("frontier_candidates", [])
    if not isinstance(frontier_candidates, list):
        frontier_candidates = []
    promotion_plan_candidates = promotion_plan_state.get("promotion_candidates", [])
    if not isinstance(promotion_plan_candidates, list):
        promotion_plan_candidates = []
    promotion_candidates = [
        candidate
        for candidate in promotion_plan_candidates
        if isinstance(candidate, dict)
    ] or [
        candidate
        for candidate in frontier_candidates
        if isinstance(candidate, dict)
        and bool(candidate.get("generated_candidate", False))
        and bool(candidate.get("candidate_exists", False))
    ]
    precompare_guard_summary = _precompare_guard_summary(promotion_plan_state)
    blocked_conditions: list[str] = []
    if available_worker_slots <= 0:
        blocked_conditions.append("discovery_at_capacity")
    if available_worker_slots > 0 and not recommended_subsystems:
        blocked_conditions.append("no_eligible_discovery_subsystems")
    if not promotion_candidates:
        blocked_conditions.append("no_promotion_candidates")
    if operator_gated_reasons:
        blocked_conditions.append("operator_gated")
    if rollout_gate["blocked_subsystems"]:
        blocked_conditions.append("meta_promotion_blocked")
    if rollback_plan["required"]:
        blocked_conditions.append("rollback_pending")
    if not canary_lifecycle["promotion_resume_allowed"]:
        blocked_conditions.append("canary_lifecycle_blocked")
    if claim_ledger.get("path_conflicts", []):
        blocked_conditions.append("claim_conflict")

    autonomy_widening_summary = _autonomy_widening_summary(
        policy=policy,
        frontier_state=frontier_state,
        trust_ledger=trust_ledger,
        rollout_gate=rollout_gate,
        canary_lifecycle=canary_lifecycle,
        blocked_conditions=blocked_conditions,
    )
    promotion_feedback_summary = _promotion_execution_feedback_summary(
        promotion_results=promotion_results,
        widening_summary=autonomy_widening_summary,
    )
    trust_evidence_priority_subsystems = [
        subsystem
        for subsystem in list(effective_trust_evidence_focus.get("prioritized_subsystems", []) or [])
        if isinstance(subsystem, str) and subsystem.strip()
    ]
    trust_breadth_priority_subsystems = [
        subsystem
        for subsystem in list(effective_trust_breadth_focus.get("prioritized_subsystems", []) or [])
        if isinstance(subsystem, str) and subsystem.strip()
    ]
    trust_evidence_gap_severity = _trust_evidence_focus_severity(effective_trust_evidence_focus)
    trust_evidence_slot_reservation = min(
        available_worker_slots,
        len(trust_evidence_priority_subsystems),
        max(0, trust_evidence_gap_severity),
    )
    trust_breadth_gap_severity = _trust_breadth_gap_severity(effective_trust_breadth_focus)
    trust_breadth_slot_reservation = len(trust_breadth_reserved_subsystem_slots)
    trust_priority_subsystems: list[str] = []
    for subsystem in [*trust_evidence_priority_subsystems, *trust_breadth_priority_subsystems]:
        if subsystem not in trust_priority_subsystems:
            trust_priority_subsystems.append(subsystem)
    trust_priority_slot_reservation = min(
        available_worker_slots,
        max(trust_evidence_slot_reservation, trust_breadth_slot_reservation),
    )

    actions = [
        {"kind": "refresh_frontier", "enabled": True},
        {"kind": "refresh_promotion_plan", "enabled": True},
    ]
    if bool(autonomy_widening_summary.get("escalation_available", False)):
        actions.append(
            {
                "kind": "prepare_autonomy_widening_package",
                "enabled": True,
                "report_path": str(config.improvement_reports_dir / _SUPERVISOR_AUTONOMY_WIDENING_FILENAME),
                "summary": autonomy_widening_summary,
            }
        )
    if rollback_plan["required"]:
        for candidate in rollback_plan["rollback_candidates"]:
            actions.append(
                {
                    "kind": "rollback_artifact",
                    "enabled": True,
                    "artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                    "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                    "reason": str(candidate.get("reason", "")).strip(),
                }
            )
            actions.append(
                {
                    "kind": "validate_rollback_artifact",
                    "enabled": True,
                    "artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                    "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                    "reason": "post_rollback_validation",
                }
            )
    elif canary_lifecycle["validation_required"]:
        for candidate in canary_lifecycle["tracked_candidates"]:
            actions.append(
                {
                    "kind": "validate_rollback_artifact",
                    "enabled": True,
                    "artifact_path": str(candidate.get("candidate_artifact_path", "")).strip(),
                    "selected_subsystem": str(candidate.get("selected_subsystem", "")).strip(),
                    "reason": "resume_gate_validation_retry",
                }
            )
    if promotion_candidates:
        promotion_blocked_reasons = list(operator_gated_reasons)
        promotion_blocked_reasons.extend(str(reason).strip() for reason in canary_lifecycle["blocked_reasons"])
        apply_finalize = (
            policy.autonomy_mode == "promote"
            and not operator_gated_reasons
            and canary_lifecycle["promotion_resume_allowed"]
        )
        allowed_bootstrap_subsystems, bootstrap_policy_reasons = _bootstrap_finalize_allowed_subsystems(
            policy=policy,
            trust_ledger=trust_ledger,
            meta_policy=meta_policy,
            frontier_state=frontier_state,
            rollout_gate=rollout_gate,
            apply_finalize=apply_finalize,
        )
        promotion_pass_subsystem_scope = _promotion_pass_subsystem_scope(
            widening_summary=autonomy_widening_summary,
            promotion_feedback_summary=promotion_feedback_summary,
        )
        promotion_pass_limit = _widening_promotion_pass_limit(
            available_worker_slots=available_worker_slots,
            max_promotion_candidates=int(policy.max_promotion_candidates),
            promotion_candidates=promotion_candidates,
            trust_evidence_gap_severity=trust_evidence_gap_severity,
            trust_breadth_gap_severity=trust_breadth_gap_severity,
            widening_summary=autonomy_widening_summary,
            promotion_feedback_summary=promotion_feedback_summary,
            promotion_pass_subsystem_scope=promotion_pass_subsystem_scope,
        )
        actions.append(
            {
                "kind": "run_promotion_pass",
                "enabled": policy.autonomy_mode != "shadow" and canary_lifecycle["promotion_resume_allowed"],
                "apply_finalize": apply_finalize,
                "allow_bootstrap_finalize": bool(allowed_bootstrap_subsystems),
                "allowed_bootstrap_subsystems": list(allowed_bootstrap_subsystems),
                "bootstrap_policy_reasons": list(bootstrap_policy_reasons),
                "limit": max(1, promotion_pass_limit),
                "operator_gated_reasons": list(operator_gated_reasons),
                "promotion_blocked_reasons": promotion_blocked_reasons,
                "blocked_subsystems": list(rollout_gate["blocked_subsystems"])
                + list(promotion_pass_subsystem_scope.get("blocked_subsystems", [])),
                "allow_subsystems": list(promotion_pass_subsystem_scope.get("allow_subsystems", [])),
                "prioritized_subsystems": list(promotion_pass_subsystem_scope.get("prioritized_subsystems", [])),
                "require_allow_subsystem_match": bool(
                    promotion_pass_subsystem_scope.get("require_allow_subsystem_match", False)
                ),
                "meta_blocked_reasons": list(rollout_gate["blocked_reasons"]),
            }
        )
    else:
        allowed_bootstrap_subsystems = []
        promotion_pass_limit = 0
    bootstrap_remediation_queues = _bootstrap_remediation_queues(
        paused_subsystems=paused,
        promotion_results=promotion_results,
        promotion_plan_candidates=promotion_plan_candidates,
        trust_ledger=trust_ledger,
        rollout_gate=rollout_gate,
        allowed_bootstrap_subsystems=list(allowed_bootstrap_subsystems),
        rollout_stage=policy.rollout_stage,
    )
    bootstrap_generated_evidence_summary = _bootstrap_generated_evidence_summary(
        bootstrap_remediation_queues
    )
    current_bootstrap_retrieval_priority_summary = _bootstrap_retrieval_priority_pressure_summary(
        bootstrap_remediation_queues
    )
    bootstrap_retrieval_priority_memory = _bootstrap_retrieval_priority_memory_summary(
        previous_memory=previous_bootstrap_retrieval_priority_memory,
        current_priority_summary=current_bootstrap_retrieval_priority_summary,
        promotion_results=promotion_results,
    )
    effective_bootstrap_retrieval_priority_summary = _effective_bootstrap_retrieval_priority_summary(
        current_bootstrap_retrieval_priority_summary,
        bootstrap_retrieval_priority_memory,
    )
    bootstrap_remediation_queues = _apply_bootstrap_retrieval_priority_memory(
        bootstrap_remediation_queues,
        effective_bootstrap_retrieval_priority_summary,
    )
    bootstrap_retrieval_priority_summary = _bootstrap_retrieval_priority_summary(bootstrap_remediation_queues)
    promotion_block_summary = _promotion_block_summary(promotion_results)
    if any(bootstrap_remediation_queues.values()):
        blocked_conditions.append("bootstrap_remediation_pending")
    if (
        _safe_int(bootstrap_generated_evidence_summary.get("target_count", 0), 0) > 0
        and available_worker_slots <= 0
    ):
        blocked_conditions.append("bootstrap_generated_evidence_pending")
    actions.extend(
        _bootstrap_remediation_actions(
            repo_root=repo_root,
            config=config,
            policy=policy,
            queues=bootstrap_remediation_queues,
            trust_ledger=trust_ledger,
            meta_policy=meta_policy,
        )
    )
    bootstrap_generated_evidence_targets = [
        target
        for target in list(bootstrap_generated_evidence_summary.get("targets", []) or [])
        if isinstance(target, dict)
    ]
    bootstrap_generated_evidence_targets = _prioritize_bootstrap_generated_evidence_targets(
        targets=bootstrap_generated_evidence_targets,
        widening_focus=widening_discovery_focus,
    )
    bootstrap_generated_evidence_targets = _apply_promotion_feedback_to_bootstrap_targets(
        targets=bootstrap_generated_evidence_targets,
        promotion_feedback_summary=promotion_feedback_summary,
    )
    bootstrap_generated_evidence_worker_count = _bootstrap_generated_evidence_worker_budget(
        available_worker_slots=available_worker_slots,
        trust_priority_slot_reservation=trust_priority_slot_reservation,
        trust_evidence_gap_severity=trust_evidence_gap_severity,
        trust_breadth_gap_severity=trust_breadth_gap_severity,
        widening_focus=widening_discovery_focus,
        bootstrap_generated_evidence_targets=bootstrap_generated_evidence_targets,
    )
    trust_priority_discovery_budget = _trust_priority_discovery_budget(
        base_task_limit=policy.discovery_task_limit,
        base_observation_budget_seconds=policy.discovery_observation_budget_seconds,
        trust_evidence_focus=effective_trust_evidence_focus,
        trust_breadth_focus=effective_trust_breadth_focus,
        widening_focus=widening_discovery_focus,
        priority_benchmark_families=list(discovery_priority_summary.get("priority_benchmark_families", []) or []),
    )
    if bootstrap_generated_evidence_worker_count > 0:
        actions.append(
            {
                "kind": "launch_bootstrap_generated_evidence_discovery",
                "enabled": True,
                "worker_count": bootstrap_generated_evidence_worker_count,
                "targets": bootstrap_generated_evidence_targets[:bootstrap_generated_evidence_worker_count],
                "generated_curriculum_budget_seconds": max(
                    float(policy.generated_curriculum_budget_seconds),
                    _BOOTSTRAP_GENERATED_EVIDENCE_DISCOVERY_MIN_BUDGET_SECONDS,
                ),
            }
        )
    remaining_discovery_worker_slots = max(
        0,
        int(available_worker_slots) - int(bootstrap_generated_evidence_worker_count),
    )
    launch_generic_discovery = _should_launch_generic_discovery(
        available_worker_slots=available_worker_slots,
        remaining_discovery_worker_slots=remaining_discovery_worker_slots,
        trust_evidence_gap_severity=trust_evidence_gap_severity,
        trust_breadth_gap_severity=trust_breadth_gap_severity,
        bootstrap_generated_evidence_worker_count=bootstrap_generated_evidence_worker_count,
        promotion_pass_limit=promotion_pass_limit,
        promotion_feedback_summary=promotion_feedback_summary,
        widening_summary=autonomy_widening_summary,
    )
    if launch_generic_discovery and remaining_discovery_worker_slots > 0 and recommended_subsystems:
        reserved_variant_roles: list[str] = []
        reserved_variant_strategy_families: list[str] = []
        recovery_role_index = 0
        for subsystem in trust_breadth_reserved_subsystem_slots[:remaining_discovery_worker_slots]:
            if subsystem == "trust":
                reserved_variant_roles.append("trust_breadth")
                reserved_variant_strategy_families.append("repository_breadth")
            elif subsystem == "recovery":
                role = (
                    trust_breadth_reserved_recovery_roles[recovery_role_index]
                    if recovery_role_index < len(trust_breadth_reserved_recovery_roles)
                    else ""
                )
                reserved_variant_roles.append(role)
                strategy_family = (
                    trust_breadth_reserved_recovery_strategy_families[recovery_role_index]
                    if recovery_role_index < len(trust_breadth_reserved_recovery_strategy_families)
                    else ""
                )
                reserved_variant_strategy_families.append(strategy_family)
                recovery_role_index += 1
            else:
                reserved_variant_roles.append("")
                reserved_variant_strategy_families.append("")
        actions.append(
            {
                "kind": "launch_discovery",
                "enabled": True,
                "worker_count": min(remaining_discovery_worker_slots, len(recommended_subsystems)),
                "subsystems": recommended_subsystems[:remaining_discovery_worker_slots],
                "variant_ids": trust_breadth_reserved_variant_ids[:remaining_discovery_worker_slots],
                "variant_roles": reserved_variant_roles,
                "variant_strategy_families": reserved_variant_strategy_families,
                "selected_guarded_subsystems": list(discovery_priority_summary.get("selected_guarded_subsystems", [])),
                "deprioritized_guarded_subsystems": list(
                    discovery_priority_summary.get("deprioritized_guarded_subsystems", [])
                ),
                "priority_benchmark_families": list(
                    discovery_priority_summary.get("priority_benchmark_families", [])
                ),
                "priority_benchmark_family_weights": dict(
                    trust_priority_discovery_budget.get("priority_benchmark_family_weights", {})
                ),
                "task_limit": max(0, _safe_int(trust_priority_discovery_budget.get("task_limit", 0), 0)),
                "max_observation_seconds": max(
                    0.0,
                    float(trust_priority_discovery_budget.get("max_observation_seconds", 0.0) or 0.0),
                ),
            }
        )
    return {
        "paused_subsystems": paused,
        "operator_gated_reasons": operator_gated_reasons,
        "meta_policy": meta_policy,
        "rollout_gate": rollout_gate,
        "validation_guard_pressure_summary": validation_guard_pressure_summary,
        "validation_guard_memory": validation_guard_memory,
        "effective_validation_guard_pressure_summary": effective_validation_guard_pressure_summary,
        "current_trust_evidence_focus": current_trust_evidence_focus,
        "trust_evidence_priority_memory": trust_evidence_priority_memory,
        "effective_trust_evidence_focus": effective_trust_evidence_focus,
        "trust_evidence_gap_severity": trust_evidence_gap_severity,
        "current_trust_breadth_focus": current_trust_breadth_focus,
        "trust_breadth_priority_memory": trust_breadth_priority_memory,
        "effective_trust_breadth_focus": effective_trust_breadth_focus,
        "widening_discovery_focus": widening_discovery_focus,
        "trust_evidence_slot_reservation": trust_evidence_slot_reservation,
        "trust_breadth_gap_severity": trust_breadth_gap_severity,
        "trust_breadth_reserved_subsystem_slots": trust_breadth_reserved_subsystem_slots,
        "trust_breadth_reserved_recovery_roles": trust_breadth_reserved_recovery_roles,
        "trust_breadth_reserved_recovery_strategy_families": trust_breadth_reserved_recovery_strategy_families,
        "trust_breadth_reserved_variant_ids": trust_breadth_reserved_variant_ids,
        "trust_priority_slot_reservation": trust_priority_slot_reservation,
        "trust_priority_discovery_budget": trust_priority_discovery_budget,
        "discovery_priority_summary": discovery_priority_summary,
        "claim_ledger": claim_ledger,
        "lane_allocator": lane_allocator,
        "rollback_plan": rollback_plan,
        "canary_lifecycle": canary_lifecycle,
        "bootstrap_remediation_queues": bootstrap_remediation_queues,
        "bootstrap_generated_evidence_summary": bootstrap_generated_evidence_summary,
        "trust_breadth_slot_reservation": trust_breadth_slot_reservation,
        "current_bootstrap_retrieval_priority_summary": current_bootstrap_retrieval_priority_summary,
        "bootstrap_retrieval_priority_memory": bootstrap_retrieval_priority_memory,
        "effective_bootstrap_retrieval_priority_summary": effective_bootstrap_retrieval_priority_summary,
        "bootstrap_retrieval_priority_summary": bootstrap_retrieval_priority_summary,
        "promotion_block_summary": promotion_block_summary,
        "precompare_guard_summary": precompare_guard_summary,
        "autonomy_widening_summary": autonomy_widening_summary,
        "promotion_execution_feedback_summary": promotion_feedback_summary,
        "launch_generic_discovery": launch_generic_discovery,
        "work_manifest": work_manifest,
        "blocked_conditions": blocked_conditions,
        "actions": actions,
    }


def _command_result(*, command: list[str], cwd: Path, timeout_seconds: float) -> dict[str, object]:
    started_at = _utcnow()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=max(1.0, float(timeout_seconds)),
    )
    completed_at = _utcnow()
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout).strip(),
        "stderr": str(completed.stderr).strip(),
        "started_at": _isoformat(started_at),
        "completed_at": _isoformat(completed_at),
        "timed_out": False,
    }


def _execute_action(
    *,
    action: dict[str, object],
    config: KernelConfig,
    policy: SupervisorPolicy,
    repo_root: Path,
    round_id: str,
) -> dict[str, object]:
    kind = str(action.get("kind", "")).strip()
    timeout_seconds = max(1.0, float(policy.command_timeout_seconds))
    if kind == "refresh_frontier":
        return _command_result(
            command=[sys.executable, str(repo_root / "scripts" / "report_supervised_frontier.py")],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
    if kind == "refresh_promotion_plan":
        return _command_result(
            command=[
                sys.executable,
                str(repo_root / "scripts" / "report_frontier_promotion_plan.py"),
                "--frontier-report",
                str(config.improvement_reports_dir / "supervised_parallel_frontier.json"),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
    if kind == "run_promotion_pass":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_frontier_promotion_pass.py"),
            "--promotion-plan",
            str(config.improvement_reports_dir / "supervised_frontier_promotion_plan.json"),
            "--limit",
            str(max(1, _safe_int(action.get("limit", 0), 1))),
        ]
        for subsystem in list(action.get("blocked_subsystems", []) or []):
            if str(subsystem).strip():
                command.extend(["--block-subsystem", str(subsystem).strip()])
        for subsystem in list(action.get("allow_subsystems", []) or []):
            if str(subsystem).strip():
                command.extend(["--allow-subsystem", str(subsystem).strip()])
        for subsystem in list(action.get("prioritized_subsystems", []) or []):
            if str(subsystem).strip():
                command.extend(["--prioritize-subsystem", str(subsystem).strip()])
        for subsystem in list(action.get("allowed_bootstrap_subsystems", []) or []):
            if str(subsystem).strip():
                command.extend(["--allow-bootstrap-subsystem", str(subsystem).strip()])
        if bool(action.get("require_allow_subsystem_match", False)):
            command.append("--require-allow-subsystem-match")
        if bool(action.get("apply_finalize", False)):
            command.append("--apply-finalize")
        if bool(action.get("allow_bootstrap_finalize", False)):
            command.append("--allow-bootstrap-finalize")
        return _command_result(command=command, cwd=repo_root, timeout_seconds=timeout_seconds)
    if kind in {
        "prepare_bootstrap_review_package",
        "prepare_trust_streak_recovery_package",
        "prepare_protected_review_package",
    }:
        started_at = _utcnow()
        report_path = Path(str(action.get("report_path", "")).strip())
        entries = [entry for entry in list(action.get("entries", []) or []) if isinstance(entry, dict)]
        if not report_path:
            now = _utcnow()
            return {
                "command": [],
                "returncode": 1,
                "stdout": "",
                "stderr": "missing report_path for remediation package",
                "started_at": _isoformat(started_at),
                "completed_at": _isoformat(now),
                "timed_out": False,
            }
        queue_name = str(action.get("queue_name", "")).strip()
        primary_entry = _primary_queue_entry(entries)
        summary_promotion_block_reason_counts: dict[str, int] = {}
        summary_compare_guard_reason_counts: dict[str, int] = {}
        summary_finalize_skip_reason_counts: dict[str, int] = {}
        summary_validation_compare_guard_reason_counts: dict[str, int] = {}
        retrieval_ranked_entry_count = 0
        trusted_retrieval_entry_count = 0
        verified_retrieval_command_total = 0
        retrieval_reuse_priority_total = 0
        retained_retrieval_ranked_entry_count = 0
        for entry in entries:
            _increment_reason_count(
                summary_promotion_block_reason_counts,
                str(entry.get("promotion_block_reason_code", "")).strip(),
            )
            _increment_reason_count(
                summary_compare_guard_reason_counts,
                str(entry.get("compare_guard_reason", "")).strip(),
            )
            _increment_reason_count(
                summary_finalize_skip_reason_counts,
                str(entry.get("finalize_skip_reason", "")).strip(),
            )
            reasons = entry.get("validation_family_compare_guard_reasons", [])
            if isinstance(reasons, list):
                for reason in reasons:
                    _increment_reason_count(summary_validation_compare_guard_reason_counts, str(reason).strip())
            retrieval_reuse_priority = _effective_retrieval_reuse_priority_score(entry)
            if retrieval_reuse_priority > 0:
                retrieval_ranked_entry_count += 1
                retrieval_reuse_priority_total += retrieval_reuse_priority
                if (
                    max(0, _safe_int(entry.get("retained_retrieval_reuse_priority_score", 0), 0)) > 0
                    and max(0, _safe_int(entry.get("retrieval_reuse_priority_score", 0), 0)) <= 0
                ):
                    retained_retrieval_ranked_entry_count += 1
                retrieval_summary = _effective_retrieval_reuse_summary(entry)
                if max(0, _safe_int(retrieval_summary.get("trusted_retrieval_procedure_count", 0), 0)) > 0:
                    trusted_retrieval_entry_count += 1
                verified_retrieval_command_total += max(
                    0,
                    _safe_int(retrieval_summary.get("verified_retrieval_command_count", 0), 0),
                )
        payload = {
            "report_kind": kind,
            "generated_at": _isoformat(_utcnow()),
            "round_id": round_id,
            "queue_name": queue_name,
            "queue_kind": str(action.get("queue_kind", "")).strip() or queue_name,
            "autonomy_mode": str(action.get("autonomy_mode", "")).strip(),
            "rollout_stage": str(action.get("rollout_stage", "")).strip(),
            "bootstrap_finalize_policy": str(action.get("bootstrap_finalize_policy", "")).strip(),
            "trust_status": str(action.get("trust_status", "")).strip(),
            "primary_entry": primary_entry,
            "selected_subsystem": str(primary_entry.get("selected_subsystem", "")).strip(),
            "scope_id": str(primary_entry.get("scope_id", "")).strip(),
            "cycle_id": str(primary_entry.get("cycle_id", "")).strip(),
            "candidate_artifact_path": str(primary_entry.get("candidate_artifact_path", "")).strip(),
            "promotion_block_reason_code": str(primary_entry.get("promotion_block_reason_code", "")).strip(),
            "compare_guard_reason": str(primary_entry.get("compare_guard_reason", "")).strip(),
            "finalize_skip_reason": str(primary_entry.get("finalize_skip_reason", "")).strip(),
            "validation_family_compare_guard_reasons": list(primary_entry.get("validation_family_compare_guard_reasons", []))
            if isinstance(primary_entry.get("validation_family_compare_guard_reasons", []), list)
            else [],
            "retrieval_reuse_summary": _retrieval_reuse_summary(primary_entry),
            "retrieval_reuse_priority_score": max(0, _safe_int(primary_entry.get("retrieval_reuse_priority_score", 0), 0)),
            "effective_retrieval_reuse_summary": _effective_retrieval_reuse_summary(primary_entry),
            "effective_retrieval_reuse_priority_score": _effective_retrieval_reuse_priority_score(primary_entry),
            "retained_retrieval_reuse_summary": _retrieval_reuse_summary(
                {"retrieval_reuse_summary": primary_entry.get("retained_retrieval_reuse_summary", {})}
            ),
            "retained_retrieval_reuse_priority_score": max(
                0,
                _safe_int(primary_entry.get("retained_retrieval_reuse_priority_score", 0), 0),
            ),
            "retrieval_reuse_priority_sticky_rounds_remaining": max(
                0,
                _safe_int(primary_entry.get("retrieval_reuse_priority_sticky_rounds_remaining", 0), 0),
            ),
            "review_finalize_command": str(primary_entry.get("review_finalize_command", "")).strip(),
            "summary": {
                "entry_count": len(entries),
                "subsystems": sorted(
                    {
                        str(entry.get("selected_subsystem", "")).strip()
                        for entry in entries
                        if str(entry.get("selected_subsystem", "")).strip()
                    }
                ),
                "promotion_block_reason_counts": summary_promotion_block_reason_counts,
                "compare_guard_reason_counts": summary_compare_guard_reason_counts,
                "finalize_skip_reason_counts": summary_finalize_skip_reason_counts,
                "validation_compare_guard_reason_counts": summary_validation_compare_guard_reason_counts,
                "retrieval_ranked_entry_count": retrieval_ranked_entry_count,
                "trusted_retrieval_entry_count": trusted_retrieval_entry_count,
                "verified_retrieval_command_total": verified_retrieval_command_total,
                "retrieval_reuse_priority_total": retrieval_reuse_priority_total,
                "retained_retrieval_ranked_entry_count": retained_retrieval_ranked_entry_count,
            },
            "entries": entries,
        }
        atomic_write_json(report_path, payload, config=config)
        completed_at = _utcnow()
        return {
            "command": [],
            "returncode": 0,
            "stdout": str(report_path),
            "stderr": "",
            "started_at": _isoformat(started_at),
            "completed_at": _isoformat(completed_at),
            "timed_out": False,
            "report_path": str(report_path),
            "queue_name": queue_name,
            "entry_count": len(entries),
        }
    if kind == "prepare_autonomy_widening_package":
        started_at = _utcnow()
        report_path = Path(str(action.get("report_path", "")).strip())
        summary = action.get("summary", {})
        if not report_path:
            now = _utcnow()
            return {
                "command": [],
                "returncode": 1,
                "stdout": "",
                "stderr": "missing report_path for autonomy widening package",
                "started_at": _isoformat(started_at),
                "completed_at": _isoformat(now),
                "timed_out": False,
            }
        payload = {
            "report_kind": "prepare_autonomy_widening_package",
            "generated_at": _isoformat(_utcnow()),
            "round_id": round_id,
            "current_policy": {
                "autonomy_mode": policy.autonomy_mode,
                "rollout_stage": policy.rollout_stage,
                "bootstrap_finalize_policy": policy.bootstrap_finalize_policy,
            },
            "autonomy_widening_summary": dict(summary) if isinstance(summary, dict) else {},
            "widening_command": (
                str(summary.get("widening_command", "")).strip() if isinstance(summary, dict) else ""
            ),
        }
        atomic_write_json(report_path, payload, config=config)
        completed_at = _utcnow()
        return {
            "command": [],
            "returncode": 0,
            "stdout": str(report_path),
            "stderr": "",
            "started_at": _isoformat(started_at),
            "completed_at": _isoformat(completed_at),
            "timed_out": False,
            "report_path": str(report_path),
        }
    if kind == "rollback_artifact":
        artifact_path = str(action.get("artifact_path", "")).strip()
        subsystem = str(action.get("selected_subsystem", "")).strip()
        if not artifact_path:
            return {
                "command": [],
                "returncode": 1,
                "stdout": "",
                "stderr": "missing artifact_path for rollback",
                "started_at": _isoformat(_utcnow()),
                "completed_at": _isoformat(_utcnow()),
                "timed_out": False,
                "artifact_path": "",
                "selected_subsystem": subsystem,
            }
        result = _command_result(
            command=[
                sys.executable,
                str(repo_root / "scripts" / "rollback_artifact.py"),
                "--artifact-path",
                artifact_path,
                "--cycles-path",
                str(config.improvement_cycles_path),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
        result["artifact_path"] = artifact_path
        result["selected_subsystem"] = subsystem
        return result
    if kind == "validate_rollback_artifact":
        artifact_path = str(action.get("artifact_path", "")).strip()
        subsystem = str(action.get("selected_subsystem", "")).strip()
        if not artifact_path:
            return {
                "command": [],
                "returncode": 1,
                "stdout": "",
                "stderr": "missing artifact_path for rollback validation",
                "started_at": _isoformat(_utcnow()),
                "completed_at": _isoformat(_utcnow()),
                "timed_out": False,
                "artifact_path": "",
                "selected_subsystem": subsystem,
            }
        result = _command_result(
            command=[
                sys.executable,
                str(repo_root / "scripts" / "validate_rollback_artifact.py"),
                "--artifact-path",
                artifact_path,
                "--cycles-path",
                str(config.improvement_cycles_path),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
        result["artifact_path"] = artifact_path
        result["selected_subsystem"] = subsystem
        return result
    if kind == "launch_discovery":
        subsystems = [
            str(value).strip()
            for value in list(action.get("subsystems", []) or [])
            if str(value).strip()
        ]
        variant_ids = [
            str(value).strip()
            for value in list(action.get("variant_ids", []) or [])
            if str(value).strip()
        ]
        variant_strategy_families = [
            str(value).strip()
            for value in list(action.get("variant_strategy_families", []) or [])
        ]
        worker_count = max(1, min(_safe_int(action.get("worker_count", 1), 1), len(subsystems) or 1))
        task_limit = max(0, _safe_int(action.get("task_limit", policy.discovery_task_limit), policy.discovery_task_limit))
        max_observation_seconds = max(
            0.0,
            float(action.get("max_observation_seconds", policy.discovery_observation_budget_seconds) or 0.0),
        )
        priority_benchmark_family_weights = {
            str(family).strip(): float(weight)
            for family, weight in dict(action.get("priority_benchmark_family_weights", {}) or {}).items()
            if str(family).strip() and float(weight or 0.0) > 0.0
        }
        scope_prefix = f"supervisor_{round_id}"
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_parallel_supervised_cycles.py"),
            "--workers",
            str(worker_count),
            "--scope-prefix",
            scope_prefix,
            "--progress-label-prefix",
            scope_prefix,
            "--provider",
            policy.provider,
            "--model",
            policy.model_name,
            "--task-limit",
            str(task_limit),
            "--max-observation-seconds",
            str(max_observation_seconds),
            "--notes",
            "supervisor loop discovery batch",
            "--auto-diversify-variants",
        ]
        if policy.include_curriculum:
            command.append("--include-curriculum")
        if policy.include_failure_curriculum:
            command.append("--include-failure-curriculum")
        if policy.generated_curriculum_budget_seconds > 0.0:
            command.extend(["--generated-curriculum-budget-seconds", str(policy.generated_curriculum_budget_seconds)])
        if policy.failure_curriculum_budget_seconds > 0.0:
            command.extend(["--failure-curriculum-budget-seconds", str(policy.failure_curriculum_budget_seconds)])
        for family in list(action.get("priority_benchmark_families", []) or []):
            token = str(family).strip()
            if token:
                command.extend(["--priority-benchmark-family", token])
                weight = float(priority_benchmark_family_weights.get(token, 0.0) or 0.0)
                if weight > 0.0:
                    command.extend(["--priority-benchmark-family-weight", f"{token}={weight:.2f}"])
        if variant_ids and len(variant_ids) < worker_count:
            variant_ids.extend([""] * (worker_count - len(variant_ids)))
        if variant_strategy_families and len(variant_strategy_families) < worker_count:
            variant_strategy_families.extend([""] * (worker_count - len(variant_strategy_families)))
        for index, subsystem in enumerate(subsystems[:worker_count]):
            command.extend(["--subsystem", subsystem])
            if index < len(variant_ids) and variant_ids[index]:
                command.extend(["--variant-id", variant_ids[index]])
            if index < len(variant_strategy_families) and variant_strategy_families[index]:
                command.extend(["--variant-strategy-family", variant_strategy_families[index]])
        discovery_timeout = max(
            timeout_seconds,
            float(worker_count) * max(1.0, float(max_observation_seconds)) * _DISCOVERY_TIMEOUT_MULTIPLIER,
        )
        return _command_result(command=command, cwd=repo_root, timeout_seconds=discovery_timeout)
    if kind == "launch_bootstrap_generated_evidence_discovery":
        raw_targets = list(action.get("targets", []) or [])
        targets = [target for target in raw_targets if isinstance(target, dict)]
        worker_count = max(1, min(_safe_int(action.get("worker_count", 1), 1), len(targets) or 1))
        scope_prefix = f"supervisor_{round_id}_bootstrap_evidence"
        generated_budget_seconds = max(
            _BOOTSTRAP_GENERATED_EVIDENCE_DISCOVERY_MIN_BUDGET_SECONDS,
            _safe_float(action.get("generated_curriculum_budget_seconds", 0.0), 0.0),
        )
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_parallel_supervised_cycles.py"),
            "--workers",
            str(worker_count),
            "--scope-prefix",
            scope_prefix,
            "--progress-label-prefix",
            scope_prefix,
            "--provider",
            policy.provider,
            "--model",
            policy.model_name,
            "--task-limit",
            str(max(1, policy.discovery_task_limit)),
            "--max-observation-seconds",
            str(max(1.0, float(policy.discovery_observation_budget_seconds))),
            "--notes",
            "supervisor bootstrap generated-evidence remediation",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            str(generated_budget_seconds),
            "--no-auto-diversify-subsystems",
            "--no-auto-diversify-variants",
        ]
        selected_targets = targets[:worker_count]
        selected_variants = [
            str(target.get("selected_variant_id", "")).strip()
            for target in selected_targets
        ]
        for target in selected_targets:
            subsystem = str(target.get("selected_subsystem", "")).strip()
            if subsystem:
                command.extend(["--subsystem", subsystem])
        if selected_variants and all(selected_variants):
            for variant_id in selected_variants:
                command.extend(["--variant-id", variant_id])
        discovery_timeout = max(
            timeout_seconds,
            float(worker_count) * max(1.0, float(policy.discovery_observation_budget_seconds)) * _DISCOVERY_TIMEOUT_MULTIPLIER,
        )
        return _command_result(command=command, cwd=repo_root, timeout_seconds=discovery_timeout)
    return {
        "command": [],
        "returncode": 1,
        "stdout": "",
        "stderr": f"unknown action kind: {kind}",
        "started_at": _isoformat(_utcnow()),
        "completed_at": _isoformat(_utcnow()),
        "timed_out": False,
    }


def _status_payload(
    *,
    started_at: datetime,
    now: datetime,
    policy: SupervisorPolicy,
    rounds_completed: int,
    latest_round: dict[str, object] | None,
    machine_state: dict[str, object],
    blocked_conditions: list[str],
    next_retry_at: str,
) -> dict[str, object]:
    trust_overall_assessment = (
        dict(machine_state.get("trust_overall_assessment", {}))
        if isinstance(machine_state.get("trust_overall_assessment", {}), dict)
        else {}
    )
    trust_status = str(trust_overall_assessment.get("status", "")).strip()
    selected_queue_kind, selected_queue_entry = _selected_bootstrap_queue_entry(latest_round)
    rollout_stage = ""
    promotion_block_summary: dict[str, object] = {}
    precompare_guard_summary: dict[str, object] = {}
    validation_guard_memory: dict[str, object] = {}
    trust_evidence_priority_memory: dict[str, object] = {}
    effective_trust_evidence_focus: dict[str, object] = {}
    effective_validation_guard_pressure_summary: dict[str, object] = {}
    bootstrap_retrieval_priority_memory: dict[str, object] = {}
    trust_breadth_priority_memory: dict[str, object] = {}
    effective_trust_breadth_focus: dict[str, object] = {}
    effective_bootstrap_retrieval_priority_summary: dict[str, object] = {}
    bootstrap_retrieval_priority_summary: dict[str, object] = {}
    bootstrap_generated_evidence_summary: dict[str, object] = {}
    autonomy_widening_summary: dict[str, object] = {}
    if isinstance(latest_round, dict):
        latest_policy = latest_round.get("policy", {})
        if isinstance(latest_policy, dict):
            rollout_stage = str(latest_policy.get("rollout_stage", "")).strip()
        decisions = latest_round.get("decisions", {})
        if isinstance(decisions, dict) and isinstance(decisions.get("promotion_block_summary", {}), dict):
            promotion_block_summary = dict(decisions.get("promotion_block_summary", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("precompare_guard_summary", {}), dict):
            precompare_guard_summary = dict(decisions.get("precompare_guard_summary", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("validation_guard_memory", {}), dict):
            validation_guard_memory = dict(decisions.get("validation_guard_memory", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("trust_evidence_priority_memory", {}), dict):
            trust_evidence_priority_memory = dict(decisions.get("trust_evidence_priority_memory", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("effective_trust_evidence_focus", {}), dict):
            effective_trust_evidence_focus = dict(decisions.get("effective_trust_evidence_focus", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("effective_validation_guard_pressure_summary", {}), dict):
            effective_validation_guard_pressure_summary = dict(
                decisions.get("effective_validation_guard_pressure_summary", {})
            )
        if isinstance(decisions, dict) and isinstance(decisions.get("bootstrap_retrieval_priority_memory", {}), dict):
            bootstrap_retrieval_priority_memory = dict(decisions.get("bootstrap_retrieval_priority_memory", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("trust_breadth_priority_memory", {}), dict):
            trust_breadth_priority_memory = dict(decisions.get("trust_breadth_priority_memory", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("effective_trust_breadth_focus", {}), dict):
            effective_trust_breadth_focus = dict(decisions.get("effective_trust_breadth_focus", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("effective_bootstrap_retrieval_priority_summary", {}), dict):
            effective_bootstrap_retrieval_priority_summary = dict(
                decisions.get("effective_bootstrap_retrieval_priority_summary", {})
            )
        if isinstance(decisions, dict) and isinstance(decisions.get("bootstrap_retrieval_priority_summary", {}), dict):
            bootstrap_retrieval_priority_summary = dict(decisions.get("bootstrap_retrieval_priority_summary", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("bootstrap_generated_evidence_summary", {}), dict):
            bootstrap_generated_evidence_summary = dict(decisions.get("bootstrap_generated_evidence_summary", {}))
        if isinstance(decisions, dict) and isinstance(decisions.get("autonomy_widening_summary", {}), dict):
            autonomy_widening_summary = dict(decisions.get("autonomy_widening_summary", {}))
    return {
        "report_kind": "supervisor_loop_status",
        "started_at": _isoformat(started_at),
        "updated_at": _isoformat(now),
        "autonomy_mode": policy.autonomy_mode,
        "bootstrap_finalize_policy": policy.bootstrap_finalize_policy,
        "rollout_stage": rollout_stage,
        "trust_status": trust_status,
        "selected_queue_kind": selected_queue_kind,
        "selected_queue_entry": selected_queue_entry,
        "selected_subsystem": str(selected_queue_entry.get("selected_subsystem", "")).strip(),
        "scope_id": str(selected_queue_entry.get("scope_id", "")).strip(),
        "cycle_id": str(selected_queue_entry.get("cycle_id", "")).strip(),
        "candidate_artifact_path": str(selected_queue_entry.get("candidate_artifact_path", "")).strip(),
        "selected_promotion_block_reason_code": str(selected_queue_entry.get("promotion_block_reason_code", "")).strip(),
        "selected_compare_guard_reason": str(selected_queue_entry.get("compare_guard_reason", "")).strip(),
        "rounds_completed": rounds_completed,
        "latest_round": latest_round or {},
        "promotion_block_summary": promotion_block_summary,
        "precompare_guard_summary": precompare_guard_summary,
        "validation_guard_memory": validation_guard_memory,
        "trust_evidence_priority_memory": trust_evidence_priority_memory,
        "effective_trust_evidence_focus": effective_trust_evidence_focus,
        "trust_breadth_priority_memory": trust_breadth_priority_memory,
        "effective_trust_breadth_focus": effective_trust_breadth_focus,
        "effective_validation_guard_pressure_summary": effective_validation_guard_pressure_summary,
        "bootstrap_retrieval_priority_memory": bootstrap_retrieval_priority_memory,
        "effective_bootstrap_retrieval_priority_summary": effective_bootstrap_retrieval_priority_summary,
        "bootstrap_retrieval_priority_summary": bootstrap_retrieval_priority_summary,
        "bootstrap_generated_evidence_summary": bootstrap_generated_evidence_summary,
        "autonomy_widening_summary": autonomy_widening_summary,
        "machine_state": machine_state,
        "blocked_conditions": list(blocked_conditions),
        "next_retry_at": next_retry_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autonomy-mode", choices=_AUTONOMY_MODES, default="shadow")
    parser.add_argument("--max-rounds", type=int, default=1, help="0 means run until interrupted.")
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    parser.add_argument("--max-discovery-workers", type=int, default=2)
    parser.add_argument("--discovery-task-limit", type=int, default=5)
    parser.add_argument("--discovery-max-observation-seconds", type=float, default=60.0)
    parser.add_argument("--max-promotion-candidates", type=int, default=2)
    parser.add_argument("--command-timeout-seconds", type=int, default=900)
    parser.add_argument("--lane-failure-threshold", type=int, default=2)
    parser.add_argument("--recent-outcomes-limit", type=int, default=12)
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--generated-curriculum-budget-seconds", type=float, default=0.0)
    parser.add_argument("--failure-curriculum-budget-seconds", type=float, default=0.0)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--rollout-stage", choices=_ROLLOUT_STAGES, default="compare_only")
    parser.add_argument(
        "--bootstrap-finalize-policy",
        choices=("operator_review", "trusted", "evidence", "allow"),
        default="operator_review",
    )
    parser.add_argument("--max-meta-promotions-per-round", type=int, default=1)
    parser.add_argument("--meta-trust-clean-success-streak", type=int, default=2)
    parser.add_argument("--meta-policy-path", default="")
    parser.add_argument("--status-path", default="")
    parser.add_argument("--history-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    config = KernelConfig()
    if str(args.provider).strip():
        config.provider = str(args.provider).strip()
    if str(args.model).strip():
        config.model_name = str(args.model).strip()
    config.ensure_directories()
    repo_root = Path(__file__).resolve().parents[1]
    meta_policy_path = (
        Path(str(args.meta_policy_path).strip())
        if str(args.meta_policy_path).strip()
        else repo_root / "config" / "supervisor_meta_policy.json"
    )

    policy = SupervisorPolicy(
        autonomy_mode=str(args.autonomy_mode).strip(),
        max_discovery_workers=max(0, int(args.max_discovery_workers)),
        discovery_task_limit=max(0, int(args.discovery_task_limit)),
        discovery_observation_budget_seconds=max(0.0, float(args.discovery_max_observation_seconds)),
        max_promotion_candidates=max(0, int(args.max_promotion_candidates)),
        command_timeout_seconds=max(1, int(args.command_timeout_seconds)),
        lane_failure_threshold=max(0, int(args.lane_failure_threshold)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        include_curriculum=bool(args.include_curriculum),
        include_failure_curriculum=bool(args.include_failure_curriculum),
        generated_curriculum_budget_seconds=max(0.0, float(args.generated_curriculum_budget_seconds)),
        failure_curriculum_budget_seconds=max(0.0, float(args.failure_curriculum_budget_seconds)),
        bootstrap_finalize_policy=str(args.bootstrap_finalize_policy).strip() or "operator_review",
        provider=config.provider,
        model_name=config.model_name,
        rollout_stage=str(args.rollout_stage).strip(),
        max_meta_promotions_per_round=max(0, int(args.max_meta_promotions_per_round)),
        meta_trust_clean_success_streak=max(0, int(args.meta_trust_clean_success_streak)),
        meta_policy_path=str(meta_policy_path),
    )

    status_path = (
        Path(str(args.status_path).strip())
        if str(args.status_path).strip()
        else config.improvement_reports_dir / _SUPERVISOR_STATUS_FILENAME
    )
    history_path = (
        Path(str(args.history_path).strip())
        if str(args.history_path).strip()
        else config.improvement_reports_dir / _SUPERVISOR_HISTORY_FILENAME
    )
    report_path = (
        Path(str(args.report_path).strip())
        if str(args.report_path).strip()
        else config.improvement_reports_dir / _SUPERVISOR_REPORT_FILENAME
    )

    started_at = _utcnow()
    rounds: list[dict[str, object]] = []
    round_index = 0

    def _execute_or_skip_action(*, action: dict[str, object], round_id: str) -> dict[str, object]:
        kind = str(action.get("kind", "")).strip()
        if not bool(action.get("enabled", False)):
            return {
                "kind": kind,
                "skipped": True,
                "reason": ";".join(
                    str(value)
                    for value in (
                        list(action.get("operator_gated_reasons", []) or [])
                        + list(action.get("promotion_blocked_reasons", []) or [])
                    )
                ),
            }
        result = _execute_action(
            action=action,
            config=config,
            policy=policy,
            repo_root=repo_root,
            round_id=round_id,
        )
        return {"kind": kind, **result}

    while int(args.max_rounds) <= 0 or round_index < int(args.max_rounds):
        round_index += 1
        round_started_at = _utcnow()
        trust_ledger_path = write_unattended_trust_ledger(config)
        trust_ledger = _load_json(trust_ledger_path)
        frontier = _frontier_state(config)
        promotion_plan = _promotion_plan_state(config)
        promotion_pass = _promotion_pass_state(config)
        supervisor_status = _supervisor_status_state(config)
        queue_state = _queue_state(config)
        recent_outcomes = _recent_supervised_outcomes(config, limit=max(0, int(args.recent_outcomes_limit)))
        previous_canary_lifecycle = supervisor_status["machine_state"].get("canary_lifecycle", {})
        previous_validation_guard_memory = supervisor_status["machine_state"].get("validation_guard_memory", {})
        previous_bootstrap_retrieval_priority_memory = supervisor_status["machine_state"].get(
            "bootstrap_retrieval_priority_memory",
            {},
        )
        previous_trust_evidence_priority_memory = supervisor_status["machine_state"].get(
            "trust_evidence_priority_memory",
            {},
        )
        previous_trust_breadth_priority_memory = supervisor_status["machine_state"].get(
            "trust_breadth_priority_memory",
            {},
        )

        def _build_current_decisions(
            current_frontier: dict[str, object],
            current_promotion_plan: dict[str, object],
            current_promotion_pass: dict[str, object],
        ) -> dict[str, object]:
            return _build_round_actions(
                config=config,
                repo_root=repo_root,
                policy=policy,
                queue_state=queue_state,
                frontier_state=current_frontier,
                promotion_plan_state=current_promotion_plan,
                promotion_pass_state=current_promotion_pass,
                trust_ledger=trust_ledger,
                recent_outcomes=recent_outcomes,
                previous_canary_lifecycle=previous_canary_lifecycle,
                previous_validation_guard_memory=previous_validation_guard_memory,
                previous_bootstrap_retrieval_priority_memory=previous_bootstrap_retrieval_priority_memory,
                previous_trust_evidence_priority_memory=previous_trust_evidence_priority_memory,
                previous_trust_breadth_priority_memory=previous_trust_breadth_priority_memory,
            )

        decisions = _build_current_decisions(frontier, promotion_plan, promotion_pass)
        executions: list[dict[str, object]] = []
        round_id = round_started_at.strftime("%Y%m%dT%H%M%S%fZ")

        for action in decisions["actions"]:
            kind = str(action.get("kind", "")).strip()
            if kind not in {"refresh_frontier", "refresh_promotion_plan"}:
                continue
            executions.append(_execute_or_skip_action(action=action, round_id=round_id))

        frontier = _frontier_state(config)
        promotion_plan = _promotion_plan_state(config)
        promotion_pass = _promotion_pass_state(config)
        decisions = _build_current_decisions(frontier, promotion_plan, promotion_pass)

        promotion_action = next(
            (
                action
                for action in decisions.get("actions", [])
                if str(action.get("kind", "")).strip() == "run_promotion_pass"
            ),
            None,
        )
        if isinstance(promotion_action, dict):
            executions.append(_execute_or_skip_action(action=promotion_action, round_id=round_id))
            frontier = _frontier_state(config)
            promotion_plan = _promotion_plan_state(config)
            promotion_pass = _promotion_pass_state(config)

        decisions = _build_current_decisions(frontier, promotion_plan, promotion_pass)
        second_promotion_feedback = decisions.get("promotion_execution_feedback_summary", {})
        second_promotion_retained_count = (
            _safe_int(second_promotion_feedback.get("eligible_non_protected_retained_count", 0), 0)
            if isinstance(second_promotion_feedback, dict)
            else 0
        )
        second_promotion_action = next(
            (
                action
                for action in decisions.get("actions", [])
                if str(action.get("kind", "")).strip() == "run_promotion_pass"
                and bool(action.get("enabled", False))
            ),
            None,
        )
        if second_promotion_retained_count > 0 and isinstance(second_promotion_action, dict):
            executions.append(_execute_or_skip_action(action=second_promotion_action, round_id=round_id))
            frontier = _frontier_state(config)
            promotion_plan = _promotion_plan_state(config)
            promotion_pass = _promotion_pass_state(config)
            decisions = _build_current_decisions(frontier, promotion_plan, promotion_pass)

        for action in decisions["actions"]:
            kind = str(action.get("kind", "")).strip()
            if kind in {"refresh_frontier", "refresh_promotion_plan", "run_promotion_pass"}:
                continue
            executions.append(_execute_or_skip_action(action=action, round_id=round_id))

        round_completed_at = _utcnow()
        machine_canary_lifecycle = _apply_execution_results_to_canary_lifecycle(
            canary_lifecycle=decisions.get("canary_lifecycle", {}),
            executions=executions,
            trust_ledger=trust_ledger,
        )
        round_payload = {
            "round_index": round_index,
            "started_at": _isoformat(round_started_at),
            "completed_at": _isoformat(round_completed_at),
            "policy": policy.to_dict(),
            "machine_state": {
                "frontier_summary": frontier["summary"],
                "promotion_pass_summary": promotion_pass["summary"],
                "trust_overall_assessment": dict(trust_ledger.get("overall_assessment", {}))
                if isinstance(trust_ledger.get("overall_assessment", {}), dict)
                else {},
                "meta_policy": decisions.get("meta_policy", {}),
                "rollout_gate": decisions.get("rollout_gate", {}),
                "claim_ledger": decisions.get("claim_ledger", {}),
                "lane_allocator": decisions.get("lane_allocator", {}),
                "rollback_plan": decisions.get("rollback_plan", {}),
                "canary_lifecycle": machine_canary_lifecycle,
                "validation_guard_memory": decisions.get("validation_guard_memory", {}),
                "trust_evidence_priority_memory": decisions.get("trust_evidence_priority_memory", {}),
                "trust_breadth_priority_memory": decisions.get("trust_breadth_priority_memory", {}),
                "bootstrap_retrieval_priority_memory": decisions.get("bootstrap_retrieval_priority_memory", {}),
                "autonomy_widening_summary": decisions.get("autonomy_widening_summary", {}),
                "queue_state": queue_state,
                "recent_outcomes": recent_outcomes,
            },
            "decisions": decisions,
            "executions": executions,
            "blocked_conditions": list(decisions.get("blocked_conditions", [])),
            "next_retry_at": _next_retry_at(now=round_completed_at, sleep_seconds=policy.sleep_seconds),
        }
        rounds.append(round_payload)
        append_jsonl(history_path, round_payload, config=config)

        status_payload = _status_payload(
            started_at=started_at,
            now=round_completed_at,
            policy=policy,
            rounds_completed=len(rounds),
            latest_round=round_payload,
            machine_state=round_payload["machine_state"],
            blocked_conditions=list(decisions.get("blocked_conditions", [])),
            next_retry_at=str(round_payload.get("next_retry_at", "")).strip(),
        )
        atomic_write_json(status_path, status_payload, config=config)

        if int(args.max_rounds) > 0 and round_index >= int(args.max_rounds):
            break
        time.sleep(max(0.0, policy.sleep_seconds))

    final_payload = {
        "report_kind": "supervisor_loop_report",
        "started_at": _isoformat(started_at),
        "completed_at": _isoformat(_utcnow()),
        "autonomy_mode": policy.autonomy_mode,
        "round_count": len(rounds),
        "status_path": str(status_path),
        "history_path": str(history_path),
        "policy": policy.to_dict(),
        "rounds": rounds,
    }
    atomic_write_json(report_path, final_payload, config=config)
    print(str(report_path))


if __name__ == "__main__":
    main()
