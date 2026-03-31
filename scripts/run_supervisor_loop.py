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
from agent_kernel.job_queue import DelegatedJobQueue, DelegatedRuntimeController, TERMINAL_JOB_STATES
from agent_kernel.runtime_supervision import append_jsonl, atomic_write_json
from agent_kernel.trust import write_unattended_trust_ledger
from evals.metrics import EvalMetrics


_AUTONOMY_MODES = ("shadow", "dry_run", "promote")
_ROLLOUT_STAGES = ("shadow", "compare_only", "canary", "broad")
_SUPERVISOR_HISTORY_FILENAME = "supervisor_loop_history.jsonl"
_SUPERVISOR_STATUS_FILENAME = "supervisor_loop_status.json"
_SUPERVISOR_REPORT_FILENAME = "supervisor_loop_report.json"
_DISCOVERY_TIMEOUT_MULTIPLIER = 3.0
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
        if state == "reject":
            bucket["rejects"] += 1
        elif state == "retain":
            bucket["retains"] += 1
        if bool(result.get("finalize_skipped", False)) and str(result.get("finalize_skip_reason", "")).strip() in {
            "bootstrap_requires_review",
            "bootstrap_subsystem_not_allowed",
        }:
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
    trust_ledger: dict[str, object],
    rollout_gate: dict[str, object],
    allowed_bootstrap_subsystems: list[str],
) -> dict[str, list[dict[str, object]]]:
    assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    trust_status = str(assessment.get("status", "")).strip() or "unknown"
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
    queues: dict[str, list[dict[str, object]]] = {
        "baseline_bootstrap": [],
        "trust_streak_accumulation": [],
        "protected_review_only": [],
    }
    seen: set[tuple[str, str]] = set()
    for result in promotion_results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        if not subsystem:
            continue
        paused = paused_subsystems.get(subsystem, {})
        if str(paused.get("reason", "")).strip() != "bootstrap_review_pending":
            continue
        compare_status = str(result.get("compare_status", "")).strip()
        finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
        if finalize_skip_reason not in {"bootstrap_requires_review", "bootstrap_subsystem_not_allowed"}:
            continue
        if not compare_status:
            compare_status = "bootstrap_first_retain"
        if compare_status != "bootstrap_first_retain":
            continue
        if subsystem in blocked_subsystems or finalize_skip_reason == "bootstrap_subsystem_not_allowed":
            queue_name = "protected_review_only"
            remediation_reason = finalize_skip_reason or "protected_subsystem_review_required"
        elif trust_status != "trusted":
            queue_name = "trust_streak_accumulation"
            remediation_reason = f"trust_status={trust_status}"
        elif allowed_bootstrap and subsystem not in allowed_bootstrap:
            queue_name = "protected_review_only"
            remediation_reason = "bootstrap_subsystem_not_allowlisted"
        else:
            queue_name = "baseline_bootstrap"
            remediation_reason = finalize_skip_reason or "first_retain_review_required"
        dedupe_key = (queue_name, subsystem)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entry = {
            "selected_subsystem": subsystem,
            "scope_id": str(result.get("scope_id", "")).strip(),
            "cycle_id": str(result.get("cycle_id", "")).strip(),
            "candidate_artifact_path": str(result.get("candidate_artifact_path", "")).strip(),
            "compare_status": compare_status,
            "finalize_skip_reason": finalize_skip_reason,
            "remediation_reason": remediation_reason,
        }
        queues[queue_name].append(entry)
        paused["remediation_queue"] = queue_name
        paused["remediation_reason"] = remediation_reason
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
                entry["review_focus"] = "prepare first-retain review package and operator decision"
            entries.append(entry)
        if not entries:
            continue
        actions.append(
            {
                "kind": action_kind,
                "enabled": True,
                "queue_name": queue_name,
                "report_path": str(config.improvement_reports_dir / filename),
                "entries": entries,
            }
        )
    return actions


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
    if normalized_policy not in {"allow", "trusted"}:
        return [], []
    overall_assessment = trust_ledger.get("overall_assessment", {})
    if not isinstance(overall_assessment, dict):
        overall_assessment = {}
    trust_status = str(overall_assessment.get("status", "")).strip()
    if normalized_policy == "trusted" and trust_status != "trusted":
        return [], ["bootstrap_trust_status_not_trusted"]

    overall_summary = trust_ledger.get("overall_summary", {})
    if not isinstance(overall_summary, dict):
        overall_summary = {}
    clean_success_streak = _safe_int(overall_summary.get("clean_success_streak", 0), 0)
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


def _lane_allocator(
    *,
    work_manifest: dict[str, object],
    claim_ledger: dict[str, object],
    paused_subsystems: dict[str, dict[str, object]],
    recommended_subsystems: list[str],
    meta_policy: dict[str, object],
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
        matching = [
            lane
            for lane in available_lanes
            if lane["lane_id"] not in used_lane_ids and _lane_matches_subsystem(lane, subsystem)
        ]
        if matching:
            chosen = matching[0]
            used_lane_ids.add(str(chosen["lane_id"]))
            assignments.append(
                {
                    "subsystem": subsystem,
                    "lane_id": str(chosen["lane_id"]),
                    "title": str(chosen["title"]),
                    "owned_paths": list(chosen["owned_paths"]),
                    "protected": bool(chosen["protected"]),
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
    rollback_candidates: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        subsystem = str(result.get("selected_subsystem", "")).strip()
        artifact_path = str(result.get("candidate_artifact_path", "")).strip()
        if subsystem not in protected_subsystems:
            continue
        if str(result.get("finalize_state", "")).strip() != "retain":
            continue
        if not artifact_path:
            continue
        rollback_candidates.append(
            {
                "selected_subsystem": subsystem,
                "candidate_artifact_path": artifact_path,
                "reason": "trust_regressed_after_protected_retain",
            }
        )
    return {
        "required": bool(rollback_candidates) and not trust_passed,
        "rollback_candidates": rollback_candidates,
        "trigger_reason": "" if trust_passed or not rollback_candidates else "protected_retain_with_failed_trust",
    }


def _canary_tracked_candidates(
    *,
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
        if subsystem not in protected_subsystems:
            continue
        key = (subsystem, artifact_path)
        if key in seen:
            continue
        seen.add(key)
        tracked.append(
            {
                "selected_subsystem": subsystem,
                "candidate_artifact_path": artifact_path,
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
                }
            )
        return fallback
    return []


def _canary_lifecycle(
    *,
    policy: SupervisorPolicy,
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
        promotion_pass_state=promotion_pass_state,
        rollout_gate=rollout_gate,
        meta_policy=meta_policy,
        previous_canary_lifecycle=previous_canary_lifecycle,
    )
    blocked_reasons: list[str] = []
    promotion_resume_allowed = True
    validation_required = False
    state = "idle"
    resume_rule = "promotion can continue immediately because no protected canary is active"
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
            "promotion may resume because the protected retain is outside canary staging"
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
    return {
        "candidate_classification": candidate_classification,
        "protected_frontier_subsystems": protected_frontier_subsystems,
        "allowed_protected_subsystems": allowed_protected_subsystems,
        "blocked_subsystems": blocked_subsystems,
        "blocked_reasons": reasons,
        "meta_finalize_budget": meta_finalize_budget,
        "trust_status": trust_status,
        "clean_success_streak": clean_success_streak,
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
    worker_count: int,
) -> list[str]:
    ranked = _planner_ranked_subsystems(config, worker_count=worker_count)
    selected: list[str] = []
    for subsystem in ranked:
        if subsystem in paused_subsystems:
            continue
        selected.append(subsystem)
        if len(selected) >= worker_count:
            break
    return selected


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
    promotion_pass_state: dict[str, object],
    trust_ledger: dict[str, object],
    recent_outcomes: list[dict[str, object]],
    previous_canary_lifecycle: dict[str, object] | None = None,
) -> dict[str, object]:
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
    active_leases = queue_state.get("active_leases", [])
    active_lease_count = len(active_leases) if isinstance(active_leases, list) else 0
    available_worker_slots = max(0, int(policy.max_discovery_workers) - active_lease_count)
    recommended_subsystems = _recommended_discovery_subsystems(
        config=config,
        paused_subsystems=paused,
        worker_count=available_worker_slots,
    ) if available_worker_slots > 0 else []
    claim_ledger = _active_claim_ledger(queue_state)
    lane_allocator = _lane_allocator(
        work_manifest=work_manifest,
        claim_ledger=claim_ledger,
        paused_subsystems=paused,
        recommended_subsystems=recommended_subsystems,
        meta_policy=meta_policy,
    )
    rollback_plan = _rollback_plan(
        trust_ledger=trust_ledger,
        promotion_pass_state=promotion_pass_state,
        rollout_gate=rollout_gate,
    )
    canary_lifecycle = _canary_lifecycle(
        policy=policy,
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
    promotion_candidates = [
        candidate
        for candidate in frontier_candidates
        if isinstance(candidate, dict)
        and bool(candidate.get("generated_candidate", False))
        and bool(candidate.get("candidate_exists", False))
    ]
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

    actions = [
        {"kind": "refresh_frontier", "enabled": True},
        {"kind": "refresh_promotion_plan", "enabled": True},
    ]
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
        actions.append(
            {
                "kind": "run_promotion_pass",
                "enabled": policy.autonomy_mode != "shadow" and canary_lifecycle["promotion_resume_allowed"],
                "apply_finalize": apply_finalize,
                "allow_bootstrap_finalize": bool(allowed_bootstrap_subsystems),
                "allowed_bootstrap_subsystems": list(allowed_bootstrap_subsystems),
                "bootstrap_policy_reasons": list(bootstrap_policy_reasons),
                "limit": min(int(policy.max_promotion_candidates), len(promotion_candidates)),
                "operator_gated_reasons": list(operator_gated_reasons),
                "promotion_blocked_reasons": promotion_blocked_reasons,
                "blocked_subsystems": list(rollout_gate["blocked_subsystems"]),
                "meta_blocked_reasons": list(rollout_gate["blocked_reasons"]),
            }
        )
    else:
        allowed_bootstrap_subsystems = []
    bootstrap_remediation_queues = _bootstrap_remediation_queues(
        paused_subsystems=paused,
        promotion_results=promotion_results,
        trust_ledger=trust_ledger,
        rollout_gate=rollout_gate,
        allowed_bootstrap_subsystems=list(allowed_bootstrap_subsystems),
    )
    if any(bootstrap_remediation_queues.values()):
        blocked_conditions.append("bootstrap_remediation_pending")
    actions.extend(
        _bootstrap_remediation_actions(
            repo_root=repo_root,
            config=config,
            queues=bootstrap_remediation_queues,
            trust_ledger=trust_ledger,
            meta_policy=meta_policy,
        )
    )
    if available_worker_slots > 0 and recommended_subsystems:
        actions.append(
            {
                "kind": "launch_discovery",
                "enabled": True,
                "worker_count": min(available_worker_slots, len(recommended_subsystems)),
                "subsystems": recommended_subsystems[:available_worker_slots],
            }
        )
    return {
        "paused_subsystems": paused,
        "operator_gated_reasons": operator_gated_reasons,
        "meta_policy": meta_policy,
        "rollout_gate": rollout_gate,
        "claim_ledger": claim_ledger,
        "lane_allocator": lane_allocator,
        "rollback_plan": rollback_plan,
        "canary_lifecycle": canary_lifecycle,
        "bootstrap_remediation_queues": bootstrap_remediation_queues,
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
        for subsystem in list(action.get("allowed_bootstrap_subsystems", []) or []):
            if str(subsystem).strip():
                command.extend(["--allow-bootstrap-subsystem", str(subsystem).strip()])
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
        payload = {
            "report_kind": kind,
            "generated_at": _isoformat(_utcnow()),
            "round_id": round_id,
            "queue_name": queue_name,
            "summary": {
                "entry_count": len(entries),
                "subsystems": sorted(
                    {
                        str(entry.get("selected_subsystem", "")).strip()
                        for entry in entries
                        if str(entry.get("selected_subsystem", "")).strip()
                    }
                ),
            },
            "entries": entries,
        }
        atomic_write_json(report_path, payload)
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
        worker_count = max(1, min(_safe_int(action.get("worker_count", 1), 1), len(subsystems) or 1))
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
            str(max(0, policy.discovery_task_limit)),
            "--max-observation-seconds",
            str(max(0.0, policy.discovery_observation_budget_seconds)),
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
        for subsystem in subsystems[:worker_count]:
            command.extend(["--subsystem", subsystem])
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
    return {
        "report_kind": "supervisor_loop_status",
        "started_at": _isoformat(started_at),
        "updated_at": _isoformat(now),
        "autonomy_mode": policy.autonomy_mode,
        "rounds_completed": rounds_completed,
        "latest_round": latest_round or {},
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
        choices=("operator_review", "trusted", "allow"),
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
    while int(args.max_rounds) <= 0 or round_index < int(args.max_rounds):
        round_index += 1
        round_started_at = _utcnow()
        trust_ledger_path = write_unattended_trust_ledger(config)
        trust_ledger = _load_json(trust_ledger_path)
        frontier = _frontier_state(config)
        promotion_pass = _promotion_pass_state(config)
        supervisor_status = _supervisor_status_state(config)
        queue_state = _queue_state(config)
        recent_outcomes = _recent_supervised_outcomes(config, limit=max(0, int(args.recent_outcomes_limit)))
        decisions = _build_round_actions(
            config=config,
            repo_root=repo_root,
            policy=policy,
            queue_state=queue_state,
            frontier_state=frontier,
            promotion_pass_state=promotion_pass,
            trust_ledger=trust_ledger,
            recent_outcomes=recent_outcomes,
            previous_canary_lifecycle=supervisor_status["machine_state"].get("canary_lifecycle", {}),
        )
        executions: list[dict[str, object]] = []
        round_id = round_started_at.strftime("%Y%m%dT%H%M%S%fZ")
        for action in decisions["actions"]:
            if not bool(action.get("enabled", False)):
                executions.append(
                    {
                        "kind": str(action.get("kind", "")).strip(),
                        "skipped": True,
                        "reason": ";".join(
                            str(value)
                            for value in (
                                list(action.get("operator_gated_reasons", []) or [])
                                + list(action.get("promotion_blocked_reasons", []) or [])
                            )
                        ),
                    }
                )
                continue
            result = _execute_action(
                action=action,
                config=config,
                policy=policy,
                repo_root=repo_root,
                round_id=round_id,
            )
            executions.append({"kind": str(action.get("kind", "")).strip(), **result})

        executed_action_kinds = {
            str(entry.get("kind", "")).strip()
            for entry in executions
            if str(entry.get("kind", "")).strip()
        }
        if "run_promotion_pass" not in executed_action_kinds and "refresh_promotion_plan" in executed_action_kinds:
            refreshed_frontier = _frontier_state(config)
            refreshed_decisions = _build_round_actions(
                config=config,
                repo_root=repo_root,
                policy=policy,
                queue_state=queue_state,
                frontier_state=refreshed_frontier,
                promotion_pass_state=promotion_pass,
                trust_ledger=trust_ledger,
                recent_outcomes=recent_outcomes,
                previous_canary_lifecycle=supervisor_status["machine_state"].get("canary_lifecycle", {}),
            )
            promotion_action = next(
                (
                    action
                    for action in refreshed_decisions.get("actions", [])
                    if str(action.get("kind", "")).strip() == "run_promotion_pass"
                ),
                None,
            )
            if isinstance(promotion_action, dict):
                decisions["actions"].append(dict(promotion_action))
                if bool(promotion_action.get("enabled", False)):
                    result = _execute_action(
                        action=promotion_action,
                        config=config,
                        policy=policy,
                        repo_root=repo_root,
                        round_id=round_id,
                    )
                    executions.append({"kind": "run_promotion_pass", **result})
                else:
                    executions.append(
                        {
                            "kind": "run_promotion_pass",
                            "skipped": True,
                            "reason": ";".join(
                                str(value)
                                for value in (
                                    list(promotion_action.get("operator_gated_reasons", []) or [])
                                    + list(promotion_action.get("promotion_blocked_reasons", []) or [])
                                )
                            ),
                        }
                    )

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
                "queue_state": queue_state,
                "recent_outcomes": recent_outcomes,
            },
            "decisions": decisions,
            "executions": executions,
            "blocked_conditions": list(decisions.get("blocked_conditions", [])),
            "next_retry_at": _next_retry_at(now=round_completed_at, sleep_seconds=policy.sleep_seconds),
        }
        rounds.append(round_payload)
        append_jsonl(history_path, round_payload)

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
        atomic_write_json(status_path, status_payload)

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
    atomic_write_json(report_path, final_payload)
    print(str(report_path))


if __name__ == "__main__":
    main()
