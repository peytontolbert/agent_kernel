from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
from typing import Any
from uuid import uuid4

from ..config import KernelConfig
from ..extensions.delegation_policy import delegation_policy_snapshot
from ..learning_compiler import persist_episode_learning_candidates
from ..loop import AgentKernel
from .preflight import (
    capture_workspace_snapshot,
    classify_run_outcome,
    run_unattended_preflight,
    write_unattended_task_report,
)
from .runtime_supervision import atomic_write_json
from .shared_repo import (
    prepare_runtime_task,
    shared_repo_claim,
    shared_repo_workspace_path,
    uses_shared_repo,
)
from ..schemas import (
    CommandResult,
    EpisodeRecord,
    StepRecord,
    TaskSpec,
    VerificationResult,
    classify_verification_reason,
)
from ..tasking.task_bank import TaskBank
from ..extensions.trust import write_unattended_trust_ledger
from ..verifier import Verifier, structured_artifact_verifier_covers_success_command
from .workspace_recovery import (
    annotate_task_report_recovery,
    recovery_annotation,
    restore_workspace_tree,
    should_restore_on_outcome,
    should_restore_on_runner_exception,
    should_snapshot_workspace,
    snapshot_workspace_tree,
)


TERMINAL_JOB_STATES = {"cancelled", "completed", "expired", "failed", "safe_stop"}
_CHECKPOINT_OUTPUT_TAIL_MAX_CHARS = 4000
_CHECKPOINT_OUTPUT_TAIL_MAX_LINES = 40
_ARTIFACT_GUARD_BACKOFF_EVENT = "artifact_guard_backoff_requeued"
_ARTIFACT_GUARD_BACKOFF_LIMIT = 3
_ARTIFACT_GUARD_BACKOFF_PRIORITY_DELTA = 100


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime | None:
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _job_id(task_id: str) -> str:
    safe_task_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task_id.strip())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"job:{safe_task_id}:{timestamp}:{uuid4().hex[:8]}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _storage_config_for_path(path: Path) -> KernelConfig | None:
    config = KernelConfig(
        storage_backend=os.getenv("AGENT_KERNEL_STORAGE_BACKEND", "sqlite").strip().lower(),
        runtime_database_path=Path(
            os.getenv("AGENT_KERNEL_RUNTIME_DATABASE_PATH", "var/runtime/agentkernel.sqlite3")
        ),
        delegated_job_queue_path=Path(
            os.getenv("AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH", "trajectories/jobs/queue.json")
        ),
        delegated_job_runtime_state_path=Path(
            os.getenv("AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH", "trajectories/jobs/runtime_state.json")
        ),
        run_reports_dir=Path(
            os.getenv("AGENT_KERNEL_RUN_REPORTS_DIR", "trajectories/reports")
        ),
        run_checkpoints_dir=Path(
            os.getenv("AGENT_KERNEL_RUN_CHECKPOINTS_DIR", "trajectories/checkpoints")
        ),
        storage_write_job_state_exports=os.getenv(
            "AGENT_KERNEL_STORAGE_WRITE_JOB_STATE_EXPORTS",
            "1",
        )
        == "1",
        storage_keep_terminal_job_records=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_TERMINAL_JOB_RECORDS", "256")
        ),
        storage_prune_terminal_job_artifacts=os.getenv(
            "AGENT_KERNEL_STORAGE_PRUNE_TERMINAL_JOB_ARTIFACTS",
            "1",
        )
        == "1",
    )
    try:
        resolved = path.resolve()
        if resolved in {
            config.delegated_job_queue_path.resolve(),
            config.delegated_job_runtime_state_path.resolve(),
        }:
            return config
    except OSError:
        return None
    return None


def _resolved_path(path: Path) -> Path | None:
    try:
        return path.resolve()
    except OSError:
        return None


def _path_within_root(path: Path, root: Path) -> bool:
    resolved_path = _resolved_path(path)
    resolved_root = _resolved_path(root)
    if resolved_path is None or resolved_root is None:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _cleanup_pruned_job_artifacts(job: "DelegatedJob", config: KernelConfig) -> None:
    if not config.storage_prune_terminal_job_artifacts:
        return
    for raw_path, managed_root in (
        (job.checkpoint_path, config.run_checkpoints_dir),
        (job.report_path, config.run_reports_dir),
    ):
        normalized = str(raw_path).strip()
        if not normalized:
            continue
        artifact_path = Path(normalized)
        if not artifact_path.exists() or artifact_path.is_dir():
            continue
        if not _path_within_root(artifact_path, managed_root):
            continue
        try:
            artifact_path.unlink()
        except OSError:
            continue
        parent = artifact_path.parent
        while parent != managed_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _terminal_job_sort_key(job: "DelegatedJob") -> tuple[float, float, str]:
    finished = _parse_timestamp(job.finished_at)
    queued = _parse_timestamp(job.queued_at)
    finished_rank = finished.timestamp() if finished is not None else float("-inf")
    queued_rank = queued.timestamp() if queued is not None else float("-inf")
    return (finished_rank, queued_rank, job.job_id)


def _prune_terminal_jobs(jobs: list["DelegatedJob"], config: KernelConfig | None) -> None:
    if config is None:
        return
    keep = max(0, int(config.storage_keep_terminal_job_records))
    terminals = [
        job
        for job in jobs
        if job.state in TERMINAL_JOB_STATES and not str(job.promoted_at).strip()
    ]
    if len(terminals) <= keep:
        return
    ordered = sorted(terminals, key=_terminal_job_sort_key, reverse=True)
    keep_ids = {job.job_id for job in ordered[:keep]}
    retained: list[DelegatedJob] = []
    for job in jobs:
        if job.state in TERMINAL_JOB_STATES and not str(job.promoted_at).strip() and job.job_id not in keep_ids:
            _cleanup_pruned_job_artifacts(job, config)
            continue
        retained.append(job)
    jobs[:] = retained


def _active_budget_group_counts(leases: list["DelegatedJobLease"]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lease in leases:
        group = f"{lease.provider}:{lease.model_name}".strip(":")
        counts[group or "unknown"] = counts.get(group or "unknown", 0) + 1
    return counts


def _budget_group(value: str) -> str:
    normalized = _safe_name(value.strip())
    return normalized or "default"


def _task_payload_benchmark_family(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("benchmark_family", "")).strip()


def _inferred_budget_group(
    *,
    task_id: str,
    budget_group: str,
    runtime_overrides: dict[str, object] | None,
    task_bank: TaskBank | None,
) -> str:
    normalized_budget_group = _budget_group(budget_group)
    if normalized_budget_group != "default":
        return normalized_budget_group
    overrides = dict(runtime_overrides or {})
    benchmark_family = _task_payload_benchmark_family(overrides.get("task_payload"))
    if not benchmark_family and task_bank is not None:
        try:
            task = task_bank.get(task_id)
        except KeyError:
            task = None
        if task is not None:
            benchmark_family = str(getattr(task, "metadata", {}).get("benchmark_family", "")).strip()
    if not benchmark_family:
        return normalized_budget_group
    return _budget_group(f"family_{benchmark_family}")


def _normalize_claim_paths(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = [str(value) for value in values]
    else:
        raw_values = []
    normalized = {
        str(value).strip().strip("/")
        for value in raw_values
        if str(value).strip()
    }
    return tuple(sorted(normalized))


def _claimed_path_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> str | None:
    if not left or not right:
        return None
    for left_path in left:
        for right_path in right:
            if (
                left_path == right_path
                or left_path.startswith(f"{right_path}/")
                or right_path.startswith(f"{left_path}/")
            ):
                return left_path if len(left_path) <= len(right_path) else right_path
    return None


def _scheduler_fairness_boost(job: "DelegatedJob") -> int:
    blocked_count = max(0, int(getattr(job, "scheduler_blocked_count", 0)))
    selected_count = max(0, int(getattr(job, "scheduler_selected_count", 0)))
    outstanding = blocked_count - selected_count
    if bool(getattr(job, "scheduler_blocked_open", False)):
        outstanding = max(outstanding, 1)
    if outstanding <= 0:
        return 0
    return min(3, outstanding)


def _scheduler_candidate_sort_key(
    job: "DelegatedJob",
    *,
    worker_job: bool,
) -> tuple[int, float, int, float, str]:
    state_rank = 0 if job.state == "in_progress" else 1
    deadline = _parse_timestamp(job.deadline_at)
    queued = _parse_timestamp(job.queued_at)
    deadline_rank = deadline.timestamp() if deadline is not None else float("inf")
    queued_rank = queued.timestamp() if queued is not None else float("inf")
    first_blocked = _parse_timestamp(getattr(job, "scheduler_first_blocked_at", ""))
    first_blocked_rank = first_blocked.timestamp() if first_blocked is not None else float("inf")
    fairness_boost = _scheduler_fairness_boost(job)
    worker_rank = 0 if worker_job else 1
    return (
        worker_rank,
        state_rank,
        -(int(job.priority) + fairness_boost),
        first_blocked_rank if fairness_boost > 0 else queued_rank,
        deadline_rank,
        job.job_id,
    )


@dataclass(slots=True)
class DelegatedCoordinationClaim:
    shared_repo_id: str = ""
    target_branch: str = ""
    worker_branch: str = ""
    claimed_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "shared_repo_id": self.shared_repo_id,
            "target_branch": self.target_branch,
            "worker_branch": self.worker_branch,
            "claimed_paths": list(self.claimed_paths),
        }

    @classmethod
    def from_job_and_task(cls, job: "DelegatedJob", task: Any) -> "DelegatedCoordinationClaim":
        claim = shared_repo_claim(
            task,
            runtime_overrides=dict(job.runtime_overrides) if isinstance(job.runtime_overrides, dict) else {},
            job_id=job.job_id,
        )
        return cls(
            shared_repo_id=str(claim.get("shared_repo_id", "")).strip(),
            target_branch=str(claim.get("target_branch", "")).strip(),
            worker_branch=str(claim.get("worker_branch", "")).strip(),
            claimed_paths=_normalize_claim_paths(claim.get("claimed_paths", ())),
        )


@dataclass(slots=True)
class DelegatedJob:
    job_id: str
    task_id: str
    state: str = "queued"
    priority: int = 0
    budget_group: str = "default"
    deadline_at: str = ""
    queued_at: str = field(default_factory=_utcnow)
    started_at: str = ""
    finished_at: str = ""
    cancel_requested_at: str = ""
    attempt_count: int = 0
    checkpoint_path: str = ""
    report_path: str = ""
    notes: str = ""
    runtime_overrides: dict[str, object] = field(default_factory=dict)
    outcome: str = ""
    outcome_reasons: list[str] = field(default_factory=list)
    last_error: str = ""
    scheduler_selected_count: int = 0
    scheduler_blocked_count: int = 0
    scheduler_unblock_count: int = 0
    scheduler_first_blocked_at: str = ""
    scheduler_last_blocked_at: str = ""
    scheduler_last_unblocked_at: str = ""
    scheduler_blocked_open: bool = False
    promoted_at: str = ""
    promotion_detail: str = ""
    history: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "state": self.state,
            "priority": self.priority,
            "budget_group": self.budget_group,
            "deadline_at": self.deadline_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested_at": self.cancel_requested_at,
            "attempt_count": self.attempt_count,
            "checkpoint_path": self.checkpoint_path,
            "report_path": self.report_path,
            "notes": self.notes,
            "runtime_overrides": dict(self.runtime_overrides),
            "outcome": self.outcome,
            "outcome_reasons": list(self.outcome_reasons),
            "last_error": self.last_error,
            "scheduler_selected_count": self.scheduler_selected_count,
            "scheduler_blocked_count": self.scheduler_blocked_count,
            "scheduler_unblock_count": self.scheduler_unblock_count,
            "scheduler_first_blocked_at": self.scheduler_first_blocked_at,
            "scheduler_last_blocked_at": self.scheduler_last_blocked_at,
            "scheduler_last_unblocked_at": self.scheduler_last_unblocked_at,
            "scheduler_blocked_open": self.scheduler_blocked_open,
            "promoted_at": self.promoted_at,
            "promotion_detail": self.promotion_detail,
            "history": [dict(entry) for entry in self.history],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DelegatedJob":
        return cls(
            job_id=str(payload.get("job_id", "")).strip(),
            task_id=str(payload.get("task_id", "")).strip(),
            state=str(payload.get("state", "queued")).strip() or "queued",
            priority=int(payload.get("priority", 0)),
            budget_group=_budget_group(str(payload.get("budget_group", "default"))),
            deadline_at=str(payload.get("deadline_at", "")).strip(),
            queued_at=str(payload.get("queued_at", "")).strip() or _utcnow(),
            started_at=str(payload.get("started_at", "")).strip(),
            finished_at=str(payload.get("finished_at", "")).strip(),
            cancel_requested_at=str(payload.get("cancel_requested_at", "")).strip(),
            attempt_count=int(payload.get("attempt_count", 0)),
            checkpoint_path=str(payload.get("checkpoint_path", "")).strip(),
            report_path=str(payload.get("report_path", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            runtime_overrides=dict(payload.get("runtime_overrides", {}))
            if isinstance(payload.get("runtime_overrides", {}), dict)
            else {},
            outcome=str(payload.get("outcome", "")).strip(),
            outcome_reasons=[
                str(value) for value in payload.get("outcome_reasons", []) if str(value).strip()
            ]
            if isinstance(payload.get("outcome_reasons", []), list)
            else [],
            last_error=str(payload.get("last_error", "")).strip(),
            scheduler_selected_count=int(payload.get("scheduler_selected_count", 0)),
            scheduler_blocked_count=int(payload.get("scheduler_blocked_count", 0)),
            scheduler_unblock_count=int(payload.get("scheduler_unblock_count", 0)),
            scheduler_first_blocked_at=str(payload.get("scheduler_first_blocked_at", "")).strip(),
            scheduler_last_blocked_at=str(payload.get("scheduler_last_blocked_at", "")).strip(),
            scheduler_last_unblocked_at=str(payload.get("scheduler_last_unblocked_at", "")).strip(),
            scheduler_blocked_open=bool(payload.get("scheduler_blocked_open", False)),
            promoted_at=str(payload.get("promoted_at", "")).strip(),
            promotion_detail=str(payload.get("promotion_detail", "")).strip(),
            history=[dict(entry) for entry in payload.get("history", []) if isinstance(entry, dict)]
            if isinstance(payload.get("history", []), list)
            else [],
        )


@dataclass(slots=True)
class DelegatedResourcePolicy:
    max_concurrent_jobs: int
    max_active_jobs_per_budget_group: int
    max_queued_jobs_per_budget_group: int
    max_artifact_bytes_per_job: int
    max_subprocesses_per_job: int
    max_consecutive_selections_per_budget_group: int
    provider: str
    model_name: str
    command_timeout_seconds: int
    llm_timeout_seconds: int
    max_steps: int
    frontier_task_step_floor: int

    def to_dict(self) -> dict[str, object]:
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_active_jobs_per_budget_group": self.max_active_jobs_per_budget_group,
            "max_queued_jobs_per_budget_group": self.max_queued_jobs_per_budget_group,
            "max_artifact_bytes_per_job": self.max_artifact_bytes_per_job,
            "max_subprocesses_per_job": self.max_subprocesses_per_job,
            "max_consecutive_selections_per_budget_group": self.max_consecutive_selections_per_budget_group,
            "provider": self.provider,
            "model_name": self.model_name,
            "command_timeout_seconds": self.command_timeout_seconds,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "max_steps": self.max_steps,
            "frontier_task_step_floor": self.frontier_task_step_floor,
        }

    @classmethod
    def from_config(cls, config: KernelConfig) -> "DelegatedResourcePolicy":
        snapshot = delegation_policy_snapshot(config)
        return cls(
            max_concurrent_jobs=max(1, int(snapshot["delegated_job_max_concurrency"])),
            max_active_jobs_per_budget_group=max(0, int(snapshot["delegated_job_max_active_per_budget_group"])),
            max_queued_jobs_per_budget_group=max(0, int(snapshot["delegated_job_max_queued_per_budget_group"])),
            max_artifact_bytes_per_job=max(0, int(snapshot["delegated_job_max_artifact_bytes"])),
            max_subprocesses_per_job=max(1, int(snapshot["delegated_job_max_subprocesses_per_job"])),
            max_consecutive_selections_per_budget_group=max(
                0, int(snapshot["delegated_job_max_consecutive_selections_per_budget_group"])
            ),
            provider=config.provider,
            model_name=config.model_name,
            command_timeout_seconds=int(snapshot["command_timeout_seconds"]),
            llm_timeout_seconds=int(snapshot["llm_timeout_seconds"]),
            max_steps=int(snapshot["max_steps"]),
            frontier_task_step_floor=int(config.frontier_task_step_floor),
        )


@dataclass(slots=True)
class DelegatedJobLease:
    job_id: str
    task_id: str
    budget_group: str
    workspace_path: str
    checkpoint_path: str
    report_path: str
    runner_id: str
    runner_host: str
    runner_pid: int
    claimed_at: str
    provider: str
    model_name: str
    shared_repo_id: str = ""
    target_branch: str = ""
    worker_branch: str = ""
    claimed_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "budget_group": self.budget_group,
            "workspace_path": self.workspace_path,
            "checkpoint_path": self.checkpoint_path,
            "report_path": self.report_path,
            "runner_id": self.runner_id,
            "runner_host": self.runner_host,
            "runner_pid": self.runner_pid,
            "claimed_at": self.claimed_at,
            "provider": self.provider,
            "model_name": self.model_name,
            "shared_repo_id": self.shared_repo_id,
            "target_branch": self.target_branch,
            "worker_branch": self.worker_branch,
            "claimed_paths": list(self.claimed_paths),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DelegatedJobLease":
        return cls(
            job_id=str(payload.get("job_id", "")).strip(),
            task_id=str(payload.get("task_id", "")).strip(),
            budget_group=_budget_group(str(payload.get("budget_group", "default"))),
            workspace_path=str(payload.get("workspace_path", "")).strip(),
            checkpoint_path=str(payload.get("checkpoint_path", "")).strip(),
            report_path=str(payload.get("report_path", "")).strip(),
            runner_id=str(payload.get("runner_id", "")).strip(),
            runner_host=str(payload.get("runner_host", "")).strip(),
            runner_pid=int(payload.get("runner_pid", 0)),
            claimed_at=str(payload.get("claimed_at", "")).strip(),
            provider=str(payload.get("provider", "")).strip(),
            model_name=str(payload.get("model_name", "")).strip(),
            shared_repo_id=str(payload.get("shared_repo_id", "")).strip(),
            target_branch=str(payload.get("target_branch", "")).strip(),
            worker_branch=str(payload.get("worker_branch", "")).strip(),
            claimed_paths=_normalize_claim_paths(payload.get("claimed_paths", [])),
        )


class DelegatedRuntimeController:
    def __init__(self, runtime_state_path: Path, *, runner_id: str | None = None) -> None:
        self.runtime_state_path = runtime_state_path
        self.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner_id = runner_id or f"{socket.gethostname()}:{os.getpid()}"

    def snapshot(self, *, config: KernelConfig | None = None) -> dict[str, object]:
        policy = DelegatedResourcePolicy.from_config(config) if config is not None else None
        with self._locked_state() as payload:
            self._normalize_state_payload(payload, policy)
            lease_objects = self._drop_stale_leases(payload)
            leases = [lease.to_dict() for lease in lease_objects]
            return {
                "spec_version": "asi_v1",
                "runtime_kind": "delegated_job_runtime_state",
                "policy": policy.to_dict() if policy is not None else dict(payload.get("policy", {})),
                "active_leases": leases,
                "budget_groups": _active_budget_group_counts(lease_objects),
                "scheduler": dict(payload.get("scheduler", {})) if isinstance(payload.get("scheduler", {}), dict) else {},
                "history": [dict(entry) for entry in payload.get("history", []) if isinstance(entry, dict)],
            }

    def scheduler_state(self) -> dict[str, object]:
        payload = self._read_payload()
        scheduler = payload.get("scheduler", {})
        if not isinstance(scheduler, dict):
            return {
                "last_selected_budget_group": "",
                "consecutive_budget_group_selections": 0,
            }
        return {
            "last_selected_budget_group": str(scheduler.get("last_selected_budget_group", "")).strip(),
            "consecutive_budget_group_selections": int(scheduler.get("consecutive_budget_group_selections", 0)),
        }

    def record_budget_group_selection(self, budget_group: str) -> None:
        normalized = _budget_group(budget_group)
        with self._locked_state() as payload:
            self._normalize_state_payload(payload)
            scheduler = payload.setdefault("scheduler", {})
            if not isinstance(scheduler, dict):
                scheduler = {}
                payload["scheduler"] = scheduler
            last_group = str(scheduler.get("last_selected_budget_group", "")).strip()
            current = int(scheduler.get("consecutive_budget_group_selections", 0))
            scheduler["last_selected_budget_group"] = normalized
            scheduler["consecutive_budget_group_selections"] = current + 1 if last_group == normalized else 1
            scheduler["last_selected_at"] = _utcnow()
            self._append_runtime_history(
                payload,
                event="scheduler_budget_group_selected",
                job_id="",
                detail=(
                    f"budget_group={normalized} consecutive="
                    f"{int(scheduler.get('consecutive_budget_group_selections', 0))}"
                ),
            )

    def acquire(
        self,
        *,
        job: DelegatedJob,
        task: Any,
        config: KernelConfig,
        checkpoint_path: Path,
        report_path: Path,
    ) -> tuple[DelegatedJobLease | None, str | None]:
        policy = DelegatedResourcePolicy.from_config(config)
        workspace_path = (
            shared_repo_workspace_path(
                config.workspace_root,
                task,
                runtime_overrides=job.runtime_overrides,
                job_id=job.job_id,
            )
            if uses_shared_repo(task, runtime_overrides=job.runtime_overrides)
            else (config.workspace_root / task.workspace_subdir).resolve()
        )
        checkpoint_resolved = checkpoint_path.resolve()
        report_resolved = report_path.resolve()
        coordination = DelegatedCoordinationClaim.from_job_and_task(job, task)
        with self._locked_state() as payload:
            self._normalize_state_payload(payload, policy)
            leases = self._drop_stale_leases(payload)
            if len(leases) >= policy.max_concurrent_jobs:
                return None, "resource_limit:max_concurrent_jobs"
            for lease in leases:
                if lease.workspace_path == str(workspace_path):
                    return None, f"lease_collision:workspace:{workspace_path}"
                if lease.checkpoint_path == str(checkpoint_resolved):
                    return None, f"lease_collision:checkpoint:{checkpoint_resolved}"
                if lease.report_path == str(report_resolved):
                    return None, f"lease_collision:report:{report_resolved}"
                if coordination.shared_repo_id and lease.shared_repo_id == coordination.shared_repo_id:
                    if coordination.worker_branch and lease.worker_branch == coordination.worker_branch:
                        return None, (
                            f"lease_collision:worker_branch:{coordination.shared_repo_id}:{coordination.worker_branch}"
                        )
                    overlap = _claimed_path_overlap(coordination.claimed_paths, lease.claimed_paths)
                    if overlap is not None:
                        return None, f"lease_collision:claimed_path:{coordination.shared_repo_id}:{overlap}"
            if policy.max_active_jobs_per_budget_group > 0:
                active_in_budget_group = sum(1 for lease in leases if lease.budget_group == job.budget_group)
                if active_in_budget_group >= policy.max_active_jobs_per_budget_group:
                    return None, f"resource_limit:budget_group:{job.budget_group}"
            lease = DelegatedJobLease(
                job_id=job.job_id,
                task_id=job.task_id,
                budget_group=job.budget_group,
                workspace_path=str(workspace_path),
                checkpoint_path=str(checkpoint_resolved),
                report_path=str(report_resolved),
                runner_id=self.runner_id,
                runner_host=socket.gethostname(),
                runner_pid=os.getpid(),
                claimed_at=_utcnow(),
                provider=config.provider,
                model_name=config.model_name,
                shared_repo_id=coordination.shared_repo_id,
                target_branch=coordination.target_branch,
                worker_branch=coordination.worker_branch,
                claimed_paths=coordination.claimed_paths,
            )
            payload["policy"] = policy.to_dict()
            payload["active_leases"] = [existing.to_dict() for existing in leases] + [lease.to_dict()]
            self._append_runtime_history(
                payload,
                event="lease_acquired",
                job_id=job.job_id,
                detail=(
                    f"runner={lease.runner_id} workspace={lease.workspace_path} "
                    f"shared_repo_id={lease.shared_repo_id or '-'} "
                    f"worker_branch={lease.worker_branch or '-'} "
                    f"claimed_paths={','.join(lease.claimed_paths) or '-'}"
                ),
            )
            return lease, None

    def release(
        self,
        job_id: str,
        *,
        final_state: str,
        detail: str,
        artifact_bytes: int,
    ) -> None:
        with self._locked_state() as payload:
            leases = self._leases_from_payload(payload)
            remaining = [lease for lease in leases if lease.job_id != job_id]
            payload["active_leases"] = [lease.to_dict() for lease in remaining]
            self._append_runtime_history(
                payload,
                event="lease_released",
                job_id=job_id,
                detail=f"state={final_state} artifact_bytes={artifact_bytes} {detail}".strip(),
            )

    def _drop_stale_leases(self, payload: dict[str, object]) -> list[DelegatedJobLease]:
        leases = self._leases_from_payload(payload)
        kept: list[DelegatedJobLease] = []
        for lease in leases:
            if self._lease_is_stale(lease):
                self._append_runtime_history(
                    payload,
                    event="lease_reaped",
                    job_id=lease.job_id,
                    detail=f"stale runner {lease.runner_host}:{lease.runner_pid}",
                )
                continue
            kept.append(lease)
        if len(kept) != len(leases):
            payload["active_leases"] = [lease.to_dict() for lease in kept]
        return kept

    @staticmethod
    def _lease_is_stale(lease: DelegatedJobLease) -> bool:
        if not lease.runner_host or lease.runner_host != socket.gethostname():
            return False
        if lease.runner_pid <= 0:
            return False
        try:
            os.kill(lease.runner_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    @staticmethod
    def _append_runtime_history(
        payload: dict[str, object],
        *,
        event: str,
        job_id: str,
        detail: str,
    ) -> None:
        history = payload.setdefault("history", [])
        if not isinstance(history, list):
            history = []
            payload["history"] = history
        history.append(
            {
                "recorded_at": _utcnow(),
                "event": event,
                "job_id": job_id,
                "detail": detail,
            }
        )
        if len(history) > 200:
            del history[:-200]

    @staticmethod
    def _leases_from_payload(payload: dict[str, object]) -> list[DelegatedJobLease]:
        raw_leases = payload.get("active_leases", [])
        if not isinstance(raw_leases, list):
            return []
        return [DelegatedJobLease.from_dict(lease) for lease in raw_leases if isinstance(lease, dict)]

    @staticmethod
    def _normalize_state_payload(
        payload: dict[str, object],
        policy: DelegatedResourcePolicy | None = None,
    ) -> None:
        payload["spec_version"] = "asi_v1"
        payload["runtime_kind"] = "delegated_job_runtime_state"
        payload["policy"] = policy.to_dict() if policy is not None else dict(payload.get("policy", {}))
        if not isinstance(payload.get("active_leases"), list):
            payload["active_leases"] = []
        if not isinstance(payload.get("history"), list):
            payload["history"] = []
        scheduler = payload.get("scheduler", {})
        if not isinstance(scheduler, dict):
            scheduler = {}
            payload["scheduler"] = scheduler
        scheduler.setdefault("last_selected_budget_group", "")
        scheduler.setdefault("consecutive_budget_group_selections", 0)

    def _read_payload(self) -> dict[str, object]:
        storage_config = _storage_config_for_path(self.runtime_state_path)
        if storage_config is not None and storage_config.uses_sqlite_storage():
            payload = storage_config.sqlite_store().load_runtime_state(runtime_path=self.runtime_state_path)
            if payload:
                return payload
        if not self.runtime_state_path.exists():
            return {
                "spec_version": "asi_v1",
                "runtime_kind": "delegated_job_runtime_state",
                "policy": {},
                "active_leases": [],
                "scheduler": {},
                "history": [],
            }
        with self.runtime_state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                handle.seek(0)
                raw = handle.read().strip()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if not raw:
            return {
                "spec_version": "asi_v1",
                "runtime_kind": "delegated_job_runtime_state",
                "policy": {},
                "active_leases": [],
                "scheduler": {},
                "history": [],
            }
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            return {
                "spec_version": "asi_v1",
                "runtime_kind": "delegated_job_runtime_state",
                "policy": {},
                "active_leases": [],
                "scheduler": {},
                "history": [],
            }
        return payload

    @contextmanager
    def _locked_state(self) -> Any:
        self.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        storage_config = _storage_config_for_path(self.runtime_state_path)
        with self.runtime_state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if storage_config is not None and storage_config.uses_sqlite_storage():
                payload = storage_config.sqlite_store().load_runtime_state(runtime_path=self.runtime_state_path)
            else:
                handle.seek(0)
                raw = handle.read().strip()
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {}
            if not isinstance(payload, dict):
                payload = {}
            try:
                yield payload
            finally:
                if storage_config is not None and storage_config.uses_sqlite_storage():
                    storage_config.sqlite_store().replace_runtime_state(
                        runtime_path=self.runtime_state_path,
                        payload=payload,
                    )
                    if storage_config.storage_write_job_state_exports:
                        handle.seek(0)
                        handle.truncate()
                        handle.write(json.dumps(payload, indent=2))
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                else:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps(payload, indent=2))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DelegatedJobQueue:
    def __init__(self, queue_path: Path) -> None:
        self.queue_path = queue_path
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def list_jobs(self, *, states: set[str] | None = None) -> list[DelegatedJob]:
        jobs = self._load_jobs()
        filtered = jobs if not states else [job for job in jobs if job.state in states]
        return sorted(filtered, key=self._claim_sort_key)

    def get(self, job_id: str) -> DelegatedJob | None:
        for job in self._load_jobs():
            if job.job_id == job_id:
                return job
        return None

    def enqueue(
        self,
        *,
        task_id: str,
        priority: int = 0,
        budget_group: str = "default",
        deadline_at: str = "",
        notes: str = "",
        runtime_overrides: dict[str, object] | None = None,
        checkpoint_path: str = "",
        report_path: str = "",
        max_queued_jobs_for_budget_group: int = 0,
        task_bank: TaskBank | None = None,
    ) -> DelegatedJob:
        with self._locked_jobs() as jobs:
            normalized_budget_group = _inferred_budget_group(
                task_id=task_id,
                budget_group=budget_group,
                runtime_overrides=runtime_overrides,
                task_bank=task_bank,
            )
            if max_queued_jobs_for_budget_group > 0:
                queued_in_group = sum(
                    1
                    for existing in jobs
                    if existing.budget_group == normalized_budget_group and existing.state not in TERMINAL_JOB_STATES
                )
                if queued_in_group >= max_queued_jobs_for_budget_group:
                    raise ValueError(
                        f"queue_budget_exceeded:{normalized_budget_group}:{max_queued_jobs_for_budget_group}"
                    )
            job = DelegatedJob(
                job_id=_job_id(task_id),
                task_id=task_id,
                priority=int(priority),
                budget_group=normalized_budget_group,
                deadline_at=deadline_at.strip(),
                notes=notes.strip(),
                runtime_overrides=dict(runtime_overrides or {}),
                checkpoint_path=checkpoint_path.strip(),
                report_path=report_path.strip(),
            )
            self._append_history(job, event="queued", detail="job added to delegated queue")
            jobs.append(job)
            return job

    def enqueue_many(
        self,
        entries: list[dict[str, object]],
        *,
        max_queued_jobs_for_budget_group: int = 0,
        task_bank: TaskBank | None = None,
    ) -> list[DelegatedJob]:
        enqueued: list[DelegatedJob] = []
        if not entries:
            return enqueued
        with self._locked_jobs() as jobs:
            queued_by_group: dict[str, int] = {}
            if max_queued_jobs_for_budget_group > 0:
                for existing in jobs:
                    if existing.state in TERMINAL_JOB_STATES:
                        continue
                    queued_by_group[existing.budget_group] = queued_by_group.get(existing.budget_group, 0) + 1
            for entry in entries:
                task_id = str(entry.get("task_id", "")).strip()
                if not task_id:
                    raise ValueError("batch enqueue entry task_id is required")
                runtime_overrides = dict(entry.get("runtime_overrides") or {})
                normalized_budget_group = _inferred_budget_group(
                    task_id=task_id,
                    budget_group=str(entry.get("budget_group", "default")),
                    runtime_overrides=runtime_overrides,
                    task_bank=task_bank,
                )
                if max_queued_jobs_for_budget_group > 0:
                    queued_in_group = queued_by_group.get(normalized_budget_group, 0)
                    if queued_in_group >= max_queued_jobs_for_budget_group:
                        raise ValueError(
                            f"queue_budget_exceeded:{normalized_budget_group}:{max_queued_jobs_for_budget_group}"
                        )
                    queued_by_group[normalized_budget_group] = queued_in_group + 1
                job = DelegatedJob(
                    job_id=_job_id(task_id),
                    task_id=task_id,
                    priority=int(entry.get("priority", 0)),
                    budget_group=normalized_budget_group,
                    deadline_at=str(entry.get("deadline_at", "")).strip(),
                    notes=str(entry.get("notes", "")).strip(),
                    runtime_overrides=runtime_overrides,
                    checkpoint_path=str(entry.get("checkpoint_path", "")).strip(),
                    report_path=str(entry.get("report_path", "")).strip(),
                )
                self._append_history(job, event="queued", detail="job added to delegated queue")
                jobs.append(job)
                enqueued.append(job)
        return enqueued

    def cancel(self, job_id: str, *, reason: str = "cancelled by operator") -> DelegatedJob:
        with self._locked_jobs() as jobs:
            for index, job in enumerate(jobs):
                if job.job_id != job_id:
                    continue
                now = _utcnow()
                if job.state in TERMINAL_JOB_STATES:
                    return job
                if job.state == "in_progress":
                    job.state = "cancel_requested"
                    job.cancel_requested_at = now
                    self._append_history(job, event="cancel_requested", detail=reason)
                else:
                    job.state = "cancelled"
                    job.cancel_requested_at = now
                    job.finished_at = now
                    job.outcome = "safe_stop"
                    job.outcome_reasons = ["job_cancelled"]
                    self._append_history(job, event="cancelled", detail=reason)
                jobs[index] = job
                return job
        raise ValueError(f"unknown job_id: {job_id}")

    @staticmethod
    def _resumable_in_progress(job: "DelegatedJob") -> bool:
        return bool(
            job.state == "in_progress"
            and (
                str(job.last_error).strip()
                or str(job.checkpoint_path).strip()
                or str(job.report_path).strip()
            )
        )

    def claim_next(
        self,
        *,
        exclude_job_ids: set[str] | None = None,
        allow_in_progress: bool = False,
    ) -> DelegatedJob | None:
        excluded = exclude_job_ids or set()
        with self._locked_jobs() as jobs:
            ordered_indices = sorted(range(len(jobs)), key=lambda index: self._claim_sort_key(jobs[index]))
            for index in ordered_indices:
                job = jobs[index]
                if job.job_id in excluded:
                    continue
                if job.state in TERMINAL_JOB_STATES:
                    continue
                if job.state == "cancel_requested":
                    job.state = "cancelled"
                    job.finished_at = _utcnow()
                    job.outcome = "safe_stop"
                    job.outcome_reasons = ["job_cancelled"]
                    self._append_history(job, event="cancelled", detail="job cancelled before resumption")
                    jobs[index] = job
                    continue
                if self._deadline_exceeded(job):
                    job.state = "expired"
                    job.finished_at = _utcnow()
                    job.outcome = "safe_stop"
                    job.outcome_reasons = ["deadline_exceeded"]
                    self._append_history(job, event="expired", detail="job deadline elapsed before execution")
                    jobs[index] = job
                    continue
                claim_time = _utcnow()
                fresh_queued_claim = job.state == "queued"
                if fresh_queued_claim or (allow_in_progress and self._resumable_in_progress(job)):
                    job.state = "in_progress"
                    job.started_at = claim_time if fresh_queued_claim else job.started_at or claim_time
                    if fresh_queued_claim:
                        job.finished_at = ""
                    job.attempt_count += 1
                    self._append_history(
                        job,
                        event="claimed",
                        detail=f"runner claimed job attempt={job.attempt_count}",
                    )
                    jobs[index] = job
                    return job
            return None

    def claim(self, job_id: str, *, allow_in_progress: bool = False) -> DelegatedJob | None:
        with self._locked_jobs() as jobs:
            for index, job in enumerate(jobs):
                if job.job_id != job_id:
                    continue
                if job.state in TERMINAL_JOB_STATES:
                    return None
                if job.state == "cancel_requested":
                    job.state = "cancelled"
                    job.finished_at = _utcnow()
                    job.outcome = "safe_stop"
                    job.outcome_reasons = ["job_cancelled"]
                    self._append_history(job, event="cancelled", detail="job cancelled before resumption")
                    jobs[index] = job
                    return None
                if self._deadline_exceeded(job):
                    job.state = "expired"
                    job.finished_at = _utcnow()
                    job.outcome = "safe_stop"
                    job.outcome_reasons = ["deadline_exceeded"]
                    self._append_history(job, event="expired", detail="job deadline elapsed before execution")
                    jobs[index] = job
                    return None
                claim_time = _utcnow()
                fresh_queued_claim = job.state == "queued"
                if fresh_queued_claim or (allow_in_progress and self._resumable_in_progress(job)):
                    job.state = "in_progress"
                    job.started_at = claim_time if fresh_queued_claim else job.started_at or claim_time
                    if fresh_queued_claim:
                        job.finished_at = ""
                    job.attempt_count += 1
                    self._append_history(job, event="claimed", detail=f"runner claimed job attempt={job.attempt_count}")
                    jobs[index] = job
                    return job
                return None
            return None

    def set_paths(self, job_id: str, *, checkpoint_path: Path, report_path: Path) -> DelegatedJob:
        return self._update_job(
            job_id,
            checkpoint_path=str(checkpoint_path),
            report_path=str(report_path),
        )

    def defer(self, job_id: str, *, reason: str) -> DelegatedJob:
        job = self._update_job(
            job_id,
            state="queued",
            last_error=reason.strip(),
        )
        self._append_history(job, event="deferred", detail=reason.strip() or "job returned to queue")
        self._persist_job(job)
        return job

    def record_scheduler_decision(self, job_id: str, *, decision: str, detail: str = "") -> DelegatedJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"unknown job_id: {job_id}")
        recorded_at = _utcnow()
        normalized = decision.strip()
        if normalized.startswith("deferred:"):
            if not job.scheduler_first_blocked_at:
                job.scheduler_first_blocked_at = recorded_at
            job.scheduler_last_blocked_at = recorded_at
            job.scheduler_blocked_count += 1
            job.scheduler_blocked_open = True
        elif normalized.startswith("ready:"):
            if job.scheduler_blocked_open:
                job.scheduler_unblock_count += 1
                job.scheduler_last_unblocked_at = recorded_at
            job.scheduler_blocked_open = False
        elif normalized.startswith("selected:"):
            job.scheduler_selected_count += 1
            job.scheduler_blocked_open = False
        self._append_history(
            job,
            event="scheduler_decision",
            detail=(f"{normalized}|{detail.strip()}").strip("|"),
        )
        self._persist_job(job)
        return job

    def mark_interrupted(
        self,
        job_id: str,
        *,
        checkpoint_path: Path,
        report_path: Path,
        error: str,
    ) -> DelegatedJob:
        job = self._update_job(
            job_id,
            state="in_progress",
            checkpoint_path=str(checkpoint_path),
            report_path=str(report_path),
            last_error=error.strip(),
        )
        self._append_history(job, event="interrupted", detail=error.strip() or "runner interrupted")
        self._persist_job(job)
        return job

    def finalize(
        self,
        job_id: str,
        *,
        state: str,
        checkpoint_path: Path,
        report_path: Path,
        outcome: str,
        outcome_reasons: list[str],
        last_error: str = "",
    ) -> DelegatedJob:
        job = self._update_job(
            job_id,
            state=state,
            finished_at=_utcnow(),
            checkpoint_path=str(checkpoint_path),
            report_path=str(report_path),
            outcome=outcome,
            outcome_reasons=list(outcome_reasons),
            last_error=last_error.strip(),
        )
        self._append_history(job, event=state, detail=f"job finished with outcome={outcome}")
        self._persist_job(job)
        return job

    def requeue_artifact_guard_backoff(
        self,
        job_id: str,
        *,
        checkpoint_path: Path,
        report_path: Path,
        reason: str,
        priority_delta: int = _ARTIFACT_GUARD_BACKOFF_PRIORITY_DELTA,
    ) -> DelegatedJob:
        with self._locked_jobs() as jobs:
            for index, job in enumerate(jobs):
                if job.job_id != job_id:
                    continue
                updated = replace(
                    job,
                    state="queued",
                    priority=int(job.priority) - max(1, int(priority_delta)),
                    finished_at="",
                    cancel_requested_at="",
                    checkpoint_path=str(checkpoint_path),
                    report_path=str(report_path),
                    outcome="",
                    outcome_reasons=[],
                    last_error="",
                    scheduler_blocked_open=False,
                    history=[dict(entry) for entry in job.history],
                )
                self._append_history(
                    updated,
                    event=_ARTIFACT_GUARD_BACKOFF_EVENT,
                    detail=reason.strip() or "artifact guard terminal backoff requeue",
                )
                jobs[index] = updated
                return updated
        raise ValueError(f"unknown job_id: {job_id}")

    def fail(
        self,
        job_id: str,
        *,
        checkpoint_path: Path,
        report_path: Path,
        error: str,
    ) -> DelegatedJob:
        return self.finalize(
            job_id,
            state="failed",
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            outcome="unsafe_ambiguous",
            outcome_reasons=["runner_exception"],
            last_error=error,
        )

    def promote(self, job_id: str, *, detail: str = "accepted for delivery") -> DelegatedJob:
        job = self._update_job(
            job_id,
            promoted_at=_utcnow(),
            promotion_detail=detail.strip() or "accepted for delivery",
        )
        self._append_history(job, event="promoted", detail=job.promotion_detail)
        self._persist_job(job)
        return job

    def retry_terminal(
        self,
        *,
        states: set[str] | None = None,
        job_ids: set[str] | None = None,
        reason: str = "retry terminal job",
        priority: int | None = None,
        limit: int = 0,
    ) -> list[DelegatedJob]:
        target_states = set(states or {"safe_stop", "failed"})
        if "completed" in target_states:
            raise ValueError("retry_terminal_refuses_completed_jobs")
        invalid_states = target_states - TERMINAL_JOB_STATES
        if invalid_states:
            raise ValueError(f"retry_terminal_invalid_states:{','.join(sorted(invalid_states))}")
        target_job_ids = {job_id.strip() for job_id in (job_ids or set()) if job_id.strip()}
        retry_reason = reason.strip() or "retry terminal job"
        retried: list[DelegatedJob] = []
        with self._locked_jobs() as jobs:
            ordered_indices = sorted(range(len(jobs)), key=lambda index: self._claim_sort_key(jobs[index]))
            for index in ordered_indices:
                if limit > 0 and len(retried) >= limit:
                    break
                job = jobs[index]
                if target_job_ids and job.job_id not in target_job_ids:
                    continue
                if job.state not in target_states:
                    continue
                updated = replace(
                    job,
                    state="queued",
                    priority=job.priority if priority is None else int(priority),
                    finished_at="",
                    cancel_requested_at="",
                    outcome="",
                    outcome_reasons=[],
                    last_error="",
                    scheduler_blocked_open=False,
                    history=[dict(entry) for entry in job.history],
                )
                self._append_history(updated, event="retry_queued", detail=retry_reason)
                jobs[index] = updated
                retried.append(updated)
        return retried

    def requeue_stale_in_progress(
        self,
        *,
        active_job_ids: set[str] | None = None,
        reason: str = "requeue stale in-progress job without active runner",
        priority: int | None = None,
        limit: int = 0,
    ) -> list[DelegatedJob]:
        active = {job_id.strip() for job_id in (active_job_ids or set()) if job_id.strip()}
        detail = reason.strip() or "requeue stale in-progress job without active runner"
        requeued: list[DelegatedJob] = []
        with self._locked_jobs() as jobs:
            ordered_indices = sorted(range(len(jobs)), key=lambda index: self._claim_sort_key(jobs[index]))
            for index in ordered_indices:
                if limit > 0 and len(requeued) >= limit:
                    break
                job = jobs[index]
                if job.state != "in_progress" or job.job_id in active:
                    continue
                if self._resumable_in_progress(job):
                    continue
                updated = replace(
                    job,
                    state="queued",
                    priority=job.priority if priority is None else int(priority),
                    started_at="",
                    finished_at="",
                    cancel_requested_at="",
                    outcome="",
                    outcome_reasons=[],
                    last_error="",
                    scheduler_blocked_open=False,
                    history=[dict(entry) for entry in job.history],
                )
                self._append_history(updated, event="stale_in_progress_requeued", detail=detail)
                jobs[index] = updated
                requeued.append(updated)
        return requeued

    def _claim_sort_key(self, job: DelegatedJob) -> tuple[int, float, int, float, str]:
        state_rank = 0 if job.state == "in_progress" else 1
        deadline = _parse_timestamp(job.deadline_at)
        queued = _parse_timestamp(job.queued_at)
        deadline_rank = deadline.timestamp() if deadline is not None else float("inf")
        queued_rank = queued.timestamp() if queued is not None else float("inf")
        return (state_rank, deadline_rank, -job.priority, queued_rank, job.job_id)

    @staticmethod
    def _append_history(job: DelegatedJob, *, event: str, detail: str) -> None:
        job.history.append(
            {
                "recorded_at": _utcnow(),
                "event": event,
                "state": job.state,
                "detail": detail,
            }
        )

    @staticmethod
    def _deadline_exceeded(job: DelegatedJob) -> bool:
        deadline = _parse_timestamp(job.deadline_at)
        if deadline is None:
            return False
        return datetime.now(UTC) > deadline

    def _persist_job(self, updated: DelegatedJob) -> None:
        with self._locked_jobs() as jobs:
            for index, job in enumerate(jobs):
                if job.job_id == updated.job_id:
                    jobs[index] = updated
                    return
            raise ValueError(f"unknown job_id: {updated.job_id}")

    def _update_job(self, job_id: str, **fields: object) -> DelegatedJob:
        with self._locked_jobs() as jobs:
            for index, job in enumerate(jobs):
                if job.job_id != job_id:
                    continue
                updated = replace(job, **fields)
                jobs[index] = updated
                return updated
            raise ValueError(f"unknown job_id: {job_id}")

    def _load_jobs(self) -> list[DelegatedJob]:
        storage_config = _storage_config_for_path(self.queue_path)
        if storage_config is not None and storage_config.uses_sqlite_storage():
            raw_jobs = storage_config.sqlite_store().load_job_queue(queue_path=self.queue_path)
            if raw_jobs:
                return [DelegatedJob.from_dict(job) for job in raw_jobs if isinstance(job, dict)]
        if not self.queue_path.exists():
            return []
        with self.queue_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                handle.seek(0)
                raw = handle.read().strip()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        raw_jobs = payload.get("jobs", [])
        if not isinstance(raw_jobs, list):
            return []
        return [DelegatedJob.from_dict(job) for job in raw_jobs if isinstance(job, dict)]

    def _write_jobs(self, jobs: list[DelegatedJob]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        storage_config = _storage_config_for_path(self.queue_path)
        _prune_terminal_jobs(jobs, storage_config)
        payload = {
            "spec_version": "asi_v1",
            "queue_kind": "delegated_job_queue",
            "jobs": [job.to_dict() for job in jobs],
        }
        if storage_config is not None and storage_config.uses_sqlite_storage():
            storage_config.sqlite_store().replace_job_queue(
                queue_path=self.queue_path,
                jobs=list(payload["jobs"]),
            )
            if not storage_config.storage_write_job_state_exports:
                return
        atomic_write_json(self.queue_path, payload, config=storage_config, govern_storage=False)

    @contextmanager
    def _locked_jobs(self) -> Any:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        storage_config = _storage_config_for_path(self.queue_path)
        with self.queue_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                if storage_config is not None and storage_config.uses_sqlite_storage():
                    raw_jobs = storage_config.sqlite_store().load_job_queue(queue_path=self.queue_path)
                    if not raw_jobs:
                        handle.seek(0)
                        raw = handle.read().strip()
                        payload = json.loads(raw) if raw else {}
                        if not isinstance(payload, dict):
                            payload = {}
                        raw_jobs = payload.get("jobs", [])
                else:
                    handle.seek(0)
                    raw = handle.read().strip()
                    payload = json.loads(raw) if raw else {}
                    if not isinstance(payload, dict):
                        payload = {}
                    raw_jobs = payload.get("jobs", [])
                jobs = (
                    [DelegatedJob.from_dict(job) for job in raw_jobs if isinstance(job, dict)]
                    if isinstance(raw_jobs, list)
                    else []
                )
                yield jobs
            except BaseException:
                raise
            else:
                _prune_terminal_jobs(jobs, storage_config)
                persisted = {
                    "spec_version": "asi_v1",
                    "queue_kind": "delegated_job_queue",
                    "jobs": [job.to_dict() for job in jobs],
                }
                if storage_config is not None and storage_config.uses_sqlite_storage():
                    storage_config.sqlite_store().replace_job_queue(
                        queue_path=self.queue_path,
                        jobs=list(persisted["jobs"]),
                    )
                    if storage_config.storage_write_job_state_exports:
                        atomic_write_json(self.queue_path, persisted, config=storage_config, govern_storage=False)
                else:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps(persisted, indent=2))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def delegated_job_paths(config: KernelConfig, job: DelegatedJob) -> tuple[Path, Path]:
    safe_job_id = _safe_name(job.job_id)
    checkpoint_path = (
        Path(job.checkpoint_path)
        if job.checkpoint_path
        else config.run_checkpoints_dir / f"{safe_job_id}.json"
    )
    report_path = (
        Path(job.report_path)
        if job.report_path
        else config.run_reports_dir / f"job_report_{safe_job_id}.json"
    )
    return checkpoint_path, report_path


def delegated_job_progress_path(config: KernelConfig, job: DelegatedJob) -> Path:
    checkpoint_path, _ = delegated_job_paths(config, job)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.progress.json")


def _stale_retry_archive_path(path: Path, *, attempt_count: int) -> Path:
    base = path.with_name(f"{path.name}.stale_before_attempt{max(1, attempt_count)}")
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.stale_before_attempt{max(1, attempt_count)}.{index}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.name}.stale_before_attempt{max(1, attempt_count)}.{uuid4().hex}")


def _archive_nonresumable_retry_artifacts(
    *,
    checkpoint_path: Path,
    report_path: Path,
    progress_path: Path,
    attempt_count: int,
) -> list[Path]:
    archived: list[Path] = []
    for path in (checkpoint_path, report_path, progress_path):
        if not path.exists() or not path.is_file():
            continue
        archive_path = _stale_retry_archive_path(path, attempt_count=attempt_count)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        path.replace(archive_path)
        archived.append(archive_path)
    return archived


def _write_delegated_job_progress(
    progress_path: Path,
    *,
    config: KernelConfig,
    job_id: str,
    task_id: str,
    payload: dict[str, object],
) -> None:
    record = {
        "recorded_at": _utcnow(),
        "job_id": job_id,
        "task_id": task_id,
    }
    record.update(dict(payload))
    try:
        atomic_write_json(progress_path, record, config=config, govern_storage=False)
    except OSError:
        return


def enqueue_with_parallel_worker_decomposition(
    queue: DelegatedJobQueue,
    *,
    bank: TaskBank,
    task_id: str,
    priority: int = 0,
    budget_group: str = "default",
    deadline_at: str = "",
    notes: str = "",
    runtime_overrides: dict[str, object] | None = None,
    max_queued_jobs_for_budget_group: int = 0,
) -> list[DelegatedJob]:
    parent_overrides = dict(runtime_overrides or {})
    try:
        target_worker_count = int(parent_overrides.get("parallel_worker_count", 0) or 0)
    except (TypeError, ValueError):
        target_worker_count = 0
    worker_tasks = bank.parallel_worker_tasks(task_id, target_worker_count=target_worker_count or None)
    if not worker_tasks:
        return [
            queue.enqueue(
                task_id=task_id,
                priority=priority,
                budget_group=budget_group,
                deadline_at=deadline_at,
                notes=notes,
                runtime_overrides=parent_overrides,
                max_queued_jobs_for_budget_group=max_queued_jobs_for_budget_group,
                task_bank=bank,
            )
        ]
    enqueued: list[DelegatedJob] = []
    dependency_job_ids: list[str] = []
    required_worker_branches: list[str] = []
    worker_overrides = {
        key: value
        for key, value in parent_overrides.items()
        if key not in {"worker_branch", "claimed_paths"}
    }
    for worker_task in worker_tasks:
        worker_guard = worker_task.metadata.get("workflow_guard", {})
        guard = dict(worker_guard) if isinstance(worker_guard, dict) else {}
        worker_branch = str(guard.get("worker_branch", "")).strip()
        worker_notes = f"{notes.strip()} [auto_worker_for:{task_id}]".strip()
        worker_runtime_overrides = dict(worker_overrides)
        if _task_is_synthetic(worker_task, bank=bank):
            worker_runtime_overrides["task_payload"] = worker_task.to_dict()
        worker_job = queue.enqueue(
            task_id=worker_task.task_id,
            priority=max(priority + 1, priority),
            budget_group=budget_group,
            deadline_at=deadline_at,
            notes=worker_notes,
            runtime_overrides=worker_runtime_overrides,
            max_queued_jobs_for_budget_group=max_queued_jobs_for_budget_group,
            task_bank=bank,
        )
        enqueued.append(worker_job)
        dependency_job_ids.append(worker_job.job_id)
        if worker_branch:
            required_worker_branches.append(worker_branch)
    integrator_overrides = dict(parent_overrides)
    if dependency_job_ids:
        integrator_overrides["dependency_job_ids"] = dependency_job_ids
    if required_worker_branches:
        integrator_overrides["required_worker_branches"] = required_worker_branches
    integrator_notes = f"{notes.strip()} [auto_integrator]".strip()
    enqueued.append(
        queue.enqueue(
            task_id=task_id,
            priority=priority,
            budget_group=budget_group,
            deadline_at=deadline_at,
            notes=integrator_notes,
            runtime_overrides=integrator_overrides,
            max_queued_jobs_for_budget_group=max_queued_jobs_for_budget_group,
            task_bank=bank,
        )
    )
    return enqueued


def resolve_job_task(bank: TaskBank, job: DelegatedJob) -> Any:
    payload = job.runtime_overrides.get("task_payload") if isinstance(job.runtime_overrides, dict) else None
    if isinstance(payload, dict):
        from ..schemas import TaskSpec

        return TaskSpec.from_dict(payload)
    return bank.get(job.task_id)


def delegated_job_dependency_status(
    queue: DelegatedJobQueue,
    job: DelegatedJob,
) -> tuple[bool, str]:
    dependency_job_ids = [
        str(value).strip()
        for value in job.runtime_overrides.get("dependency_job_ids", [])
        if str(value).strip()
    ] if isinstance(job.runtime_overrides, dict) else []
    if not dependency_job_ids:
        return True, ""
    waiting: list[str] = []
    failed: list[str] = []
    for dependency_job_id in dependency_job_ids:
        dependency = queue.get(dependency_job_id)
        if dependency is None:
            waiting.append(f"{dependency_job_id}:missing")
            continue
        if dependency.state == "completed" and dependency.outcome == "success":
            continue
        if dependency.state in TERMINAL_JOB_STATES:
            failed.append(
                f"{dependency.job_id}:{dependency.state}:{dependency.outcome or 'unknown'}"
            )
            continue
        waiting.append(f"{dependency.job_id}:{dependency.state}")
    if failed:
        return False, "dependency_failed:" + ",".join(failed)
    if waiting:
        return False, "dependency_waiting:" + ",".join(waiting)
    return True, ""


def _config_for_job(base_config: KernelConfig, job: DelegatedJob) -> KernelConfig:
    config = replace(base_config)
    retained_policy = delegation_policy_snapshot(config)
    for field, value in retained_policy.items():
        if hasattr(config, field):
            setattr(config, field, value)
    for field, value in job.runtime_overrides.items():
        if field == "task_step_floor":
            config.frontier_task_step_floor = max(1, int(value))
            continue
        if not hasattr(config, field):
            continue
        current = getattr(config, field)
        if isinstance(current, Path):
            setattr(config, field, Path(str(value)))
            continue
        if isinstance(current, tuple):
            if isinstance(value, list):
                setattr(config, field, tuple(str(item) for item in value))
            elif isinstance(value, tuple):
                setattr(config, field, tuple(str(item) for item in value))
            else:
                setattr(config, field, tuple(part for part in str(value).split(":") if part))
            continue
        setattr(config, field, value)
    config.ensure_directories()
    return config


def _worker_command_spec(job: DelegatedJob) -> dict[str, object] | None:
    overrides = dict(job.runtime_overrides) if isinstance(job.runtime_overrides, dict) else {}
    command = overrides.get("worker_command")
    if isinstance(command, str) and command.strip():
        normalized_command: str | list[str] = command.strip()
    elif isinstance(command, list) and command:
        normalized_command = [str(item) for item in command if str(item).strip()]
        if not normalized_command:
            return None
    else:
        return None
    worker_env = overrides.get("worker_env", {})
    if not isinstance(worker_env, dict):
        worker_env = {}
    return {
        "command": normalized_command,
        "cwd": str(overrides.get("worker_cwd", "")).strip(),
        "env": {str(key): str(value) for key, value in worker_env.items() if str(key).strip()},
        "timeout_seconds": int(overrides.get("worker_timeout_seconds", 0) or 0),
    }


def _checkpoint_output_tail(text: object) -> dict[str, object]:
    raw = str(text or "")
    lines = raw.splitlines()
    tail = "\n".join(lines[-_CHECKPOINT_OUTPUT_TAIL_MAX_LINES :])
    if len(tail) > _CHECKPOINT_OUTPUT_TAIL_MAX_CHARS:
        tail = tail[-_CHECKPOINT_OUTPUT_TAIL_MAX_CHARS :]
    return {
        "tail": tail,
        "line_count": len(lines),
        "char_count": len(raw),
        "truncated": len(lines) > _CHECKPOINT_OUTPUT_TAIL_MAX_LINES or len(raw) > len(tail),
    }


def _task_contract_summary(task: TaskSpec) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "workspace_subdir": task.workspace_subdir,
        "benchmark_family": str(task.metadata.get("benchmark_family", "")).strip(),
        "capability": str(task.metadata.get("capability", "")).strip(),
        "expected_files": list(task.expected_files),
        "forbidden_files": list(task.forbidden_files),
        "expected_output_substrings": list(task.expected_output_substrings),
        "forbidden_output_substrings": list(task.forbidden_output_substrings),
        "expected_file_contents": sorted(str(path).strip() for path in task.expected_file_contents),
        "success_command": task.success_command,
        "max_steps": int(task.max_steps),
    }


def _worker_command_result_payload(*, command: str, exit_code: int, stdout: str, stderr: str, timed_out: bool) -> dict[str, object]:
    return {
        "command": str(command).strip(),
        "exit_code": int(exit_code),
        "timed_out": bool(timed_out),
        "stdout_summary": _checkpoint_output_tail(stdout),
        "stderr_summary": _checkpoint_output_tail(stderr),
    }


def _worker_command_verification_result(
    *,
    task: TaskSpec,
    workspace_path: Path,
    command_result: CommandResult,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[VerificationResult, dict[str, object]]:
    verification = Verifier().verify(task, workspace_path, command_result)
    success_command_payload: dict[str, object] = {}
    if not verification.passed or not task.success_command:
        return verification, success_command_payload
    if structured_artifact_verifier_covers_success_command(task):
        return (
            verification,
            {
                "kind": "success_command_result",
                "command": task.success_command,
                "passed": True,
                "skipped": True,
                "skip_reason": "structured_artifact_verifier_covers_success_command",
            },
        )
    success_timed_out = False
    try:
        success_check = subprocess.run(
            ["bash", "-lc", task.success_command],
            cwd=workspace_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
        success_exit_code = int(success_check.returncode)
        success_stdout = success_check.stdout
        success_stderr = success_check.stderr
    except subprocess.TimeoutExpired as exc:
        success_timed_out = True
        success_exit_code = 124
        success_stdout = exc.stdout or ""
        success_stderr = exc.stderr or ""
    success_command_payload = _worker_command_result_payload(
        command=task.success_command,
        exit_code=success_exit_code,
        stdout=success_stdout,
        stderr=success_stderr,
        timed_out=success_timed_out,
    )
    if not success_timed_out and success_exit_code == 0:
        return verification, success_command_payload

    reasons = [
        reason
        for reason in verification.reasons
        if str(reason).strip() and str(reason).strip().lower() != "verification passed"
    ]
    if success_timed_out:
        success_reason = "success command timed out"
    else:
        success_reason = f"success command exited with code {success_exit_code}"
    reasons.append(success_reason)
    failure_codes = list(verification.failure_codes)
    success_code = classify_verification_reason(success_reason)
    if success_code and success_code not in failure_codes:
        failure_codes.append(success_code)
    outcome_label = success_code or (failure_codes[0] if failure_codes else "verification_failure")
    controllability = verification.controllability
    if success_timed_out:
        controllability = "runtime"
    evidence = [dict(item) for item in verification.evidence]
    evidence.append(
        {
            "kind": "success_command_result",
            "command": task.success_command,
            "exit_code": success_exit_code,
            "timed_out": success_timed_out,
        }
    )
    return (
        VerificationResult(
            passed=False,
            reasons=reasons,
            command_result=verification.command_result,
            process_score=min(float(verification.process_score), 0.95),
            outcome_label=outcome_label,
            outcome_confidence=float(verification.outcome_confidence),
            controllability=controllability,
            failure_codes=failure_codes,
            side_effects=list(verification.side_effects),
            criteria=[dict(item) for item in verification.criteria],
            evidence=evidence,
        ),
        success_command_payload,
    )


def _run_worker_command_task(
    *,
    job: DelegatedJob,
    config: KernelConfig,
    task: TaskSpec,
    runtime_task: TaskSpec,
    repo_root: Path,
    checkpoint_path: Path,
) -> EpisodeRecord:
    del task
    worker_spec = _worker_command_spec(job)
    if worker_spec is None:
        raise ValueError("worker command spec is missing")
    workspace_path = config.workspace_root / runtime_task.workspace_subdir
    workspace_path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(config.to_env())
    env.update({str(key): str(value) for key, value in worker_spec["env"].items()})
    cwd_value = str(worker_spec.get("cwd", "")).strip()
    cwd = workspace_path if not cwd_value else Path(cwd_value)
    if not cwd.is_absolute():
        cwd = repo_root / cwd
    timeout_seconds = int(worker_spec.get("timeout_seconds", 0) or 0) or config.command_timeout_seconds
    command = worker_spec["command"]
    if isinstance(command, list):
        invocation = [str(item) for item in command]
        command_text = shlex.join(invocation)
    else:
        invocation = ["bash", "-lc", str(command)]
        command_text = str(command)
    timed_out = False
    try:
        completed = subprocess.run(
            invocation,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    verification, success_command_result = _worker_command_verification_result(
        task=runtime_task,
        workspace_path=workspace_path,
        command_result=CommandResult(
            command=command_text,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        ),
        env=env,
        timeout_seconds=timeout_seconds,
    )
    verification_passed = bool(verification.passed)
    verification_reasons = list(verification.reasons)
    termination_reason = "verification_passed" if verification_passed else (
        "command_timeout" if timed_out else "worker_command_failed"
    )
    command_result_payload = _worker_command_result_payload(
        command=command_text,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "runner_kind": "delegated_worker_command",
                "job_id": job.job_id,
                "task_id": runtime_task.task_id,
                "command": command_text,
                "cwd": str(cwd),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "verification_passed": verification_passed,
                "verification_reasons": list(verification_reasons),
                "verification": verification.to_payload(),
                "termination_reason": termination_reason,
                "command_result": command_result_payload,
                "stdout_summary": dict(command_result_payload["stdout_summary"]),
                "stderr_summary": dict(command_result_payload["stderr_summary"]),
                "success_command_result": success_command_result,
                "task_contract_summary": _task_contract_summary(runtime_task),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    episode = EpisodeRecord(
        task_id=runtime_task.task_id,
        prompt=runtime_task.prompt,
        workspace=str(workspace_path),
        success=verification_passed,
        steps=[
            StepRecord(
                index=1,
                thought="execute delegated worker command",
                action="code_execute",
                content=command_text,
                selected_skill_id=None,
                command_result={
                    "command": command_text,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": timed_out,
                    "capabilities_used": [],
                },
                verification=verification.to_payload(),
            )
        ],
        task_metadata=dict(runtime_task.metadata),
        task_contract=runtime_task.to_dict(),
        termination_reason=termination_reason,
    )
    persist_episode_learning_candidates(
        episode,
        config=config,
        episode_storage={
            "phase": "delegated_worker_command",
            "relative_path": checkpoint_path.name,
            "source_group": "delegated_worker_command",
        },
    )
    return episode


def _task_is_synthetic(task: Any, *, bank: TaskBank) -> bool:
    try:
        bank.get(str(getattr(task, "task_id", "")).strip())
    except KeyError:
        return True
    return False


def _artifact_guard_backoff_count(job: DelegatedJob) -> int:
    return sum(
        1
        for entry in job.history
        if isinstance(entry, dict) and str(entry.get("event", "")).strip() == _ARTIFACT_GUARD_BACKOFF_EVENT
    )


def _artifact_guard_terminal_backoff_reason(report_path: Path, job: DelegatedJob) -> str:
    if _artifact_guard_backoff_count(job) >= _ARTIFACT_GUARD_BACKOFF_LIMIT:
        return ""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    failure = payload.get("artifact_contract_failure", {})
    if not isinstance(failure, dict):
        return ""
    mode = str(failure.get("mode", "")).strip()
    last_source = str(failure.get("last_decision_source", "")).strip()
    repairable = bool(failure.get("repairable"))
    materialization_guard = (
        mode == "artifact_materialization_guard_terminal"
        or last_source in {"artifact_materialization_guard", "swe_patch_materialization_guard"}
    )
    repairable_artifact_contract = (
        repairable
        and mode.startswith("artifact_")
        and mode not in {"artifact_contract_success", "artifact_contract_unknown"}
    )
    if not materialization_guard and not repairable_artifact_contract:
        return ""
    evidence = [
        str(value).strip()
        for value in failure.get("evidence", [])
        if str(value).strip()
    ]
    reason = (
        "artifact guard terminal backoff requeue"
        if materialization_guard
        else "repairable artifact contract backoff requeue"
    )
    if mode:
        reason += f": mode={mode}"
    if last_source:
        reason += f" source={last_source}"
    if evidence:
        reason += f" evidence={'; '.join(evidence[-3:])[:500]}"
    return reason


def _job_was_artifact_guard_backoff_requeued(job: DelegatedJob) -> bool:
    if not job.history:
        return False
    last = job.history[-1]
    return isinstance(last, dict) and str(last.get("event", "")).strip() == _ARTIFACT_GUARD_BACKOFF_EVENT


def run_next_delegated_job(
    queue: DelegatedJobQueue,
    *,
    base_config: KernelConfig | None = None,
    repo_root: Path | None = None,
    enforce_preflight: bool = True,
    runtime_controller: DelegatedRuntimeController | None = None,
) -> DelegatedJob | None:
    attempted_budget_blocked: set[str] = set()
    attempted_dependency_blocked: set[str] = set()
    attempted_preflight_blocked: set[str] = set()
    deferred_budget_blocked: DelegatedJob | None = None
    deferred_dependency_blocked: DelegatedJob | None = None
    deferred_preflight_blocked: DelegatedJob | None = None
    resolved_config = base_config or KernelConfig()
    try:
        bank = TaskBank(config=resolved_config)
    except TypeError:
        bank = TaskBank()
    controller = runtime_controller or DelegatedRuntimeController(
        resolved_config.delegated_job_runtime_state_path
    )
    while True:
        runtime_snapshot = controller.snapshot(config=resolved_config)
        active_lease_job_ids = {
            str(lease.get("job_id", "")).strip()
            for lease in runtime_snapshot.get("active_leases", [])
            if isinstance(lease, dict) and str(lease.get("job_id", "")).strip()
        }
        raw_candidates = queue.list_jobs(states={"queued", "in_progress", "cancel_requested"})
        prepared_candidates: list[tuple[DelegatedJob, KernelConfig, Any, Any, bool, bool]] = []
        repo = repo_root or Path(__file__).resolve().parents[1]
        stale_in_progress_requeued = False
        for candidate in raw_candidates:
            if candidate.job_id in attempted_budget_blocked | attempted_dependency_blocked | attempted_preflight_blocked:
                continue
            resumable = bool(
                candidate.state == "in_progress"
                and candidate.job_id not in active_lease_job_ids
                and DelegatedJobQueue._resumable_in_progress(candidate)
            )
            if (
                candidate.state == "in_progress"
                and candidate.job_id not in active_lease_job_ids
                and not resumable
            ):
                requeued = queue.requeue_stale_in_progress(
                    active_job_ids=active_lease_job_ids,
                    reason="stale in-progress job had no active lease and no resume artifact",
                    limit=1,
                )
                if requeued:
                    attempted_budget_blocked.discard(candidate.job_id)
                    attempted_dependency_blocked.discard(candidate.job_id)
                    attempted_preflight_blocked.discard(candidate.job_id)
                    stale_in_progress_requeued = True
                    break
                continue
            if candidate.state == "in_progress" and not resumable:
                continue
            candidate_config = _config_for_job(resolved_config, candidate)
            candidate_task = resolve_job_task(bank, candidate)
            candidate_runtime_task = prepare_runtime_task(
                candidate_task,
                runtime_overrides=candidate.runtime_overrides,
                job_id=candidate.job_id,
            )
            workflow_guard = (
                dict(candidate_runtime_task.metadata.get("workflow_guard", {}))
                if isinstance(candidate_runtime_task.metadata.get("workflow_guard", {}), dict)
                else {}
            )
            prepared_candidates.append(
                (
                    candidate,
                    candidate_config,
                    candidate_task,
                    candidate_runtime_task,
                    bool(str(workflow_guard.get("worker_branch", "")).strip()),
                    resumable,
                )
            )
        if stale_in_progress_requeued:
            continue
        prepared_candidates.sort(
            key=lambda item: _scheduler_candidate_sort_key(item[0], worker_job=item[4])
        )
        claimed_job: DelegatedJob | None = None
        config: KernelConfig | None = None
        task = None
        runtime_task = None
        preflight = None
        starvation_candidates: list[tuple[DelegatedJob, KernelConfig, Any, Any, bool]] = []
        scheduler_state = (
            dict(runtime_snapshot.get("scheduler", {}))
            if isinstance(runtime_snapshot.get("scheduler", {}), dict)
            else {}
        )
        for candidate, candidate_config, candidate_task, candidate_runtime_task, worker_job, resumable in prepared_candidates:
            config = candidate_config
            task = candidate_task
            dependencies_ready, dependency_reason = delegated_job_dependency_status(queue, candidate)
            if not dependencies_ready:
                queue.record_scheduler_decision(
                    candidate.job_id,
                    decision="deferred:dependency_blocked",
                    detail=dependency_reason or "delegated job waiting on incomplete dependencies",
                )
                deferred = queue.defer(
                    candidate.job_id,
                    reason=dependency_reason or "delegated job waiting on incomplete dependencies",
                )
                attempted_dependency_blocked.add(candidate.job_id)
                deferred_dependency_blocked = deferred
                continue
            runtime_task = candidate_runtime_task
            if enforce_preflight:
                preflight = run_unattended_preflight(config, runtime_task, repo_root=repo)
                if not preflight.passed:
                    detail = ",".join(check.name for check in preflight.checks if not check.passed) or "preflight_blocked"
                    queue.record_scheduler_decision(
                        candidate.job_id,
                        decision="deferred:preflight_blocked",
                        detail=detail,
                    )
                    deferred = queue.defer(candidate.job_id, reason=f"preflight_blocked:{detail}")
                    attempted_preflight_blocked.add(candidate.job_id)
                    deferred_preflight_blocked = deferred
                    continue
            max_consecutive = max(0, int(DelegatedResourcePolicy.from_config(config).max_consecutive_selections_per_budget_group))
            if (
                max_consecutive > 0
                and scheduler_state.get("last_selected_budget_group") == candidate.budget_group
                and int(scheduler_state.get("consecutive_budget_group_selections", 0)) >= max_consecutive
            ):
                queue.record_scheduler_decision(
                    candidate.job_id,
                    decision="deferred:anti_starvation_budget_group",
                    detail=(
                        f"budget_group={candidate.budget_group} "
                        f"consecutive={int(scheduler_state.get('consecutive_budget_group_selections', 0))} "
                        f"limit={max_consecutive}"
                    ),
                )
                starvation_candidates.append(
                    (candidate, candidate_config, candidate_task, candidate_runtime_task, worker_job, resumable)
                )
                preflight = None
                continue
            if bool(candidate.scheduler_blocked_open):
                queue.record_scheduler_decision(
                    candidate.job_id,
                    decision="ready:runnable",
                    detail=f"priority={candidate.priority}",
                )
            claimed_job = queue.claim(candidate.job_id, allow_in_progress=resumable)
            if claimed_job is None:
                preflight = None
                task = None
                runtime_task = None
                continue
            workflow_guard = (
                dict(runtime_task.metadata.get("workflow_guard", {}))
                if isinstance(runtime_task.metadata.get("workflow_guard", {}), dict)
                else {}
            )
            semantic_verifier = (
                dict(runtime_task.metadata.get("semantic_verifier", {}))
                if isinstance(runtime_task.metadata.get("semantic_verifier", {}), dict)
                else {}
            )
            if str(workflow_guard.get("worker_branch", "")).strip():
                decision = "selected:runnable_worker"
            elif [
                str(branch).strip()
                for branch in semantic_verifier.get("required_merged_branches", [])
                if str(branch).strip()
            ]:
                decision = "selected:runnable_integrator"
            else:
                decision = "selected:runnable_job"
            queue.record_scheduler_decision(candidate.job_id, decision=decision, detail=f"priority={candidate.priority}")
            break
        if claimed_job is None and starvation_candidates:
            for candidate, candidate_config, candidate_task, candidate_runtime_task, worker_job, resumable in starvation_candidates:
                config = candidate_config
                task = candidate_task
                runtime_task = candidate_runtime_task
                if bool(candidate.scheduler_blocked_open):
                    queue.record_scheduler_decision(
                        candidate.job_id,
                        decision="ready:runnable",
                        detail=f"priority={candidate.priority}",
                    )
                claimed_job = queue.claim(candidate.job_id, allow_in_progress=resumable)
                if claimed_job is None:
                    continue
                workflow_guard = (
                    dict(runtime_task.metadata.get("workflow_guard", {}))
                    if isinstance(runtime_task.metadata.get("workflow_guard", {}), dict)
                    else {}
                )
                semantic_verifier = (
                    dict(runtime_task.metadata.get("semantic_verifier", {}))
                    if isinstance(runtime_task.metadata.get("semantic_verifier", {}), dict)
                    else {}
                )
                if str(workflow_guard.get("worker_branch", "")).strip():
                    decision = "selected:runnable_worker"
                elif [
                    str(branch).strip()
                    for branch in semantic_verifier.get("required_merged_branches", [])
                    if str(branch).strip()
                ]:
                    decision = "selected:runnable_integrator"
                else:
                    decision = "selected:runnable_job"
                queue.record_scheduler_decision(candidate.job_id, decision=decision, detail=f"priority={candidate.priority}")
                break
        if claimed_job is None or config is None or task is None or runtime_task is None:
            return deferred_budget_blocked or deferred_dependency_blocked or deferred_preflight_blocked

        job = claimed_job
        checkpoint_path, report_path = delegated_job_paths(config, job)
        progress_path = delegated_job_progress_path(config, job)
        if not resumable and job.attempt_count > 1:
            archived_retry_artifacts = _archive_nonresumable_retry_artifacts(
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                progress_path=progress_path,
                attempt_count=job.attempt_count,
            )
            if archived_retry_artifacts:
                queue.record_scheduler_decision(
                    job.job_id,
                    decision="retry:fresh_start",
                    detail=(
                        f"attempt={job.attempt_count} "
                        f"archived_artifacts={len(archived_retry_artifacts)}"
                    ),
                )
        queue.set_paths(job.job_id, checkpoint_path=checkpoint_path, report_path=report_path)
        controller = runtime_controller or DelegatedRuntimeController(config.delegated_job_runtime_state_path)
        lease, denied_reason = controller.acquire(
            job=job,
            task=runtime_task,
            config=config,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
        )
        if lease is None:
            queue.record_scheduler_decision(
                job.job_id,
                decision="deferred:resource_blocked",
                detail=denied_reason or "runtime governance denied delegated job start",
            )
            deferred = queue.defer(job.job_id, reason=denied_reason or "runtime governance denied delegated job start")
            if denied_reason and denied_reason.startswith("resource_limit:budget_group:"):
                attempted_budget_blocked.add(job.job_id)
                deferred_budget_blocked = deferred
                continue
            return deferred

        workspace_path = config.workspace_root / runtime_task.workspace_subdir
        controller.record_budget_group_selection(job.budget_group)
        before_workspace_snapshot = capture_workspace_snapshot(workspace_path)
        workspace_snapshot_path = (
            snapshot_workspace_tree(
                workspace_path,
                config.unattended_workspace_snapshot_root,
                task_id=runtime_task.task_id,
                run_label=job.job_id,
            )
            if should_snapshot_workspace(config)
            else None
        )
        preflight = None
        artifact_bytes = 0
        release_state = "in_progress"
        release_detail = "delegated job did not reach a terminal state"
        final_job: DelegatedJob | None = None
        try:
            _write_delegated_job_progress(
                progress_path,
                config=config,
                job_id=job.job_id,
                task_id=runtime_task.task_id,
                payload={
                    "event": "delegated_job_started",
                    "step_stage": "setup_pending",
                },
            )
            if enforce_preflight and preflight is None:
                preflight = run_unattended_preflight(config, runtime_task, repo_root=repo)
                if not preflight.passed:
                    report = write_unattended_task_report(
                        task=runtime_task,
                        config=config,
                        episode=None,
                        preflight=preflight,
                        report_path=report_path,
                        before_workspace_snapshot=before_workspace_snapshot,
                    )
                    outcome, outcome_reasons = classify_run_outcome(episode=None, preflight=preflight)
                    artifact_bytes = _artifact_bytes(
                        workspace_path=workspace_path,
                        checkpoint_path=checkpoint_path,
                        report_path=report,
                    )
                    release_state = "safe_stop"
                    release_detail = "preflight failed"
                    final_job = queue.finalize(
                        job.job_id,
                        state="safe_stop",
                        checkpoint_path=checkpoint_path,
                        report_path=report,
                        outcome=outcome,
                        outcome_reasons=outcome_reasons,
                    )
                    annotate_task_report_recovery(
                        report,
                        recovery_annotation(
                            config=config,
                            workspace_snapshot_path=workspace_snapshot_path,
                            rollback_performed=False,
                            rollback_reason="preflight_failed_before_execution",
                            workspace_path=workspace_path,
                        ),
                    )
                    if report.exists():
                        write_unattended_trust_ledger(config)
                    return final_job

            worker_command = _worker_command_spec(job)
            kernel = None
            try:
                if worker_command is not None:
                    episode = _run_worker_command_task(
                        job=job,
                        config=config,
                        task=task,
                        runtime_task=runtime_task,
                        repo_root=repo,
                        checkpoint_path=checkpoint_path,
                    )
                else:
                    kernel = AgentKernel(config=config)
                    episode = kernel.run_task(
                        task,
                        checkpoint_path=checkpoint_path,
                        resume=checkpoint_path.exists(),
                        runtime_overrides=job.runtime_overrides,
                        job_id=job.job_id,
                        progress_callback=lambda payload: _write_delegated_job_progress(
                            progress_path,
                            config=config,
                            job_id=job.job_id,
                            task_id=runtime_task.task_id,
                            payload=payload,
                        ),
                    )
            except KeyboardInterrupt as exc:
                release_state = "in_progress"
                release_detail = str(exc) or "runner interrupted"
                final_job = queue.mark_interrupted(
                    job.job_id,
                    checkpoint_path=checkpoint_path,
                    report_path=report_path,
                    error=release_detail,
                )
                return final_job
            except Exception as exc:
                release_state = "failed"
                release_detail = f"{exc.__class__.__name__}: {exc}"
                report = write_unattended_task_report(
                    task=runtime_task,
                    config=config,
                    episode=None,
                    preflight=preflight,
                    report_path=report_path,
                    before_workspace_snapshot=before_workspace_snapshot,
                    outcome_override="unsafe_ambiguous",
                    outcome_reasons_override=["runner_exception"],
                    termination_reason_override=release_detail,
                    extra_uncertainties=["runner raised an exception before producing a complete episode record"],
                )
                rollback_performed = False
                if workspace_snapshot_path is not None and should_restore_on_runner_exception(config):
                    restore_workspace_tree(workspace_snapshot_path, workspace_path)
                    rollback_performed = True
                annotate_task_report_recovery(
                    report,
                    recovery_annotation(
                        config=config,
                        workspace_snapshot_path=workspace_snapshot_path,
                        rollback_performed=rollback_performed,
                        rollback_reason="runner_exception",
                        workspace_path=workspace_path,
                    ),
                )
                if report.exists():
                    write_unattended_trust_ledger(config)
                final_job = queue.fail(
                    job.job_id,
                    checkpoint_path=checkpoint_path,
                    report_path=report,
                    error=release_detail,
                )
                return final_job
            finally:
                if kernel is not None:
                    kernel.close()

            report = write_unattended_task_report(
                task=runtime_task,
                config=config,
                episode=episode,
                preflight=preflight,
                report_path=report_path,
                before_workspace_snapshot=before_workspace_snapshot,
                    )
            outcome, outcome_reasons = classify_run_outcome(episode=episode, preflight=preflight)
            artifact_bytes = _artifact_bytes(
                workspace_path=workspace_path,
                checkpoint_path=checkpoint_path,
                report_path=report,
                expected_files=list(getattr(runtime_task, "expected_files", [])),
            )
            terminal_state = "completed" if outcome == "success" else "safe_stop" if outcome == "safe_stop" else "failed"
            last_error = ""
            acceptance_gate_failed, acceptance_gate_reason = _acceptance_promotion_gate(report)
            if acceptance_gate_failed:
                terminal_state = "safe_stop"
                outcome = "safe_stop"
                outcome_reasons = list(outcome_reasons) + ["acceptance_verifier_failed"]
                last_error = acceptance_gate_reason
            if artifact_bytes > config.delegated_job_max_artifact_bytes:
                terminal_state = "safe_stop"
                outcome = "safe_stop"
                outcome_reasons = list(outcome_reasons) + ["artifact_budget_exceeded"]
                last_error = (
                    f"artifact bytes {artifact_bytes} exceeded limit {config.delegated_job_max_artifact_bytes}"
                )
            rollback_performed = False
            if workspace_snapshot_path is not None and should_restore_on_outcome(config, outcome):
                restore_workspace_tree(workspace_snapshot_path, workspace_path)
                rollback_performed = True
            annotate_task_report_recovery(
                report,
                recovery_annotation(
                    config=config,
                    workspace_snapshot_path=workspace_snapshot_path,
                    rollback_performed=rollback_performed,
                    rollback_reason="" if not rollback_performed else f"non_success_outcome:{outcome}",
                    workspace_path=workspace_path,
                ),
            )
            if report.exists():
                write_unattended_trust_ledger(config)
            artifact_guard_backoff_reason = (
                _artifact_guard_terminal_backoff_reason(report, job)
                if terminal_state == "safe_stop"
                else ""
            )
            if artifact_guard_backoff_reason:
                release_state = "queued"
                release_detail = artifact_guard_backoff_reason
                final_job = queue.requeue_artifact_guard_backoff(
                    job.job_id,
                    checkpoint_path=checkpoint_path,
                    report_path=report,
                    reason=artifact_guard_backoff_reason,
                )
                _write_delegated_job_progress(
                    progress_path,
                    config=config,
                    job_id=job.job_id,
                    task_id=runtime_task.task_id,
                    payload={
                        "event": "delegated_job_deferred",
                        "terminal_state": "queued",
                        "outcome": "artifact_guard_backoff_requeued",
                        "outcome_reasons": list(outcome_reasons),
                    },
                )
                return final_job
            release_state = terminal_state
            release_detail = last_error or f"outcome={outcome}"
            final_job = queue.finalize(
                job.job_id,
                state=terminal_state,
                checkpoint_path=checkpoint_path,
                report_path=report,
                outcome=outcome,
                outcome_reasons=outcome_reasons,
                last_error=last_error,
            )
            _write_delegated_job_progress(
                progress_path,
                config=config,
                job_id=job.job_id,
                task_id=runtime_task.task_id,
                payload={
                    "event": "delegated_job_finished",
                    "terminal_state": terminal_state,
                    "outcome": outcome,
                    "outcome_reasons": list(outcome_reasons),
                },
            )
            return final_job
        finally:
            controller.release(
                job.job_id,
                final_state=release_state,
                detail=release_detail,
                artifact_bytes=artifact_bytes,
            )


def drain_delegated_jobs(
    queue: DelegatedJobQueue,
    *,
    limit: int = 0,
    base_config: KernelConfig | None = None,
    repo_root: Path | None = None,
    enforce_preflight: bool = True,
    runtime_controller: DelegatedRuntimeController | None = None,
) -> list[DelegatedJob]:
    completed: list[DelegatedJob] = []
    while limit <= 0 or len(completed) < limit:
        job = run_next_delegated_job(
            queue,
            base_config=base_config,
            repo_root=repo_root,
            enforce_preflight=enforce_preflight,
            runtime_controller=runtime_controller,
        )
        if job is None:
            break
        completed.append(job)
        if job.state == "queued" and _job_was_artifact_guard_backoff_requeued(job):
            continue
        if job.state in {"in_progress", "queued"}:
            break
    return completed


def _artifact_bytes(
    *,
    workspace_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    expected_files: list[str] | tuple[str, ...] | None = None,
) -> int:
    expected_artifacts = [str(path).strip().strip("/") for path in expected_files or [] if str(path).strip()]
    if expected_artifacts:
        return sum(_path_bytes(workspace_path / relative_path) for relative_path in expected_artifacts)
    return _path_bytes(workspace_path)


def _path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _acceptance_promotion_gate(report_path: Path) -> tuple[bool, str]:
    if not report_path.exists():
        return False, ""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ""
    if not isinstance(payload, dict):
        return False, ""
    packet = payload.get("acceptance_packet", {})
    if not isinstance(packet, dict) or not packet:
        return False, ""
    if not _acceptance_packet_requires_promotion(packet):
        return False, ""
    verifier = packet.get("verifier_result", {})
    if not isinstance(verifier, dict):
        verifier = {}
    if bool(verifier.get("passed", False)):
        return False, ""
    target_branch = str(packet.get("target_branch", "")).strip()
    expected_branch = str(packet.get("expected_branch", "")).strip()
    return (
        True,
        "acceptance packet verifier did not pass"
        + ("" if not target_branch and not expected_branch else f" target_branch={target_branch or '-'} expected_branch={expected_branch or '-'}"),
    )


def _acceptance_packet_requires_promotion(packet: dict[str, object]) -> bool:
    for key in ("target_branch", "expected_branch", "diff_base_ref"):
        if str(packet.get(key, "")).strip():
            return True
    for key in (
        "required_merged_branches",
        "selected_edits",
        "candidate_edit_sets",
        "tests",
        "report_rules",
    ):
        values = packet.get(key, [])
        if isinstance(values, list) and values:
            return True
    return False


def _active_budget_group_counts(leases: list[DelegatedJobLease]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lease in leases:
        counts[lease.budget_group] = counts.get(lease.budget_group, 0) + 1
    return counts
