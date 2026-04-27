from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.capabilities import capability_registry_snapshot
from agent_kernel.extensions.delegation_policy import delegation_policy_snapshot
from agent_kernel.ops.job_queue import (
    TERMINAL_JOB_STATES,
    DelegatedJobQueue,
    DelegatedRuntimeController,
    delegated_job_dependency_status,
    drain_delegated_jobs,
    enqueue_with_parallel_worker_decomposition,
    resolve_job_task,
    run_next_delegated_job,
)
from agent_kernel.extensions.operator_policy import operator_policy_snapshot
from agent_kernel.ops.preflight import run_unattended_preflight
from agent_kernel.ops.shared_repo import prepare_runtime_task
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.extensions.trust import build_unattended_trust_ledger
from agent_kernel.ops.unattended_controller import discover_structural_classes, structural_class_family_aliases


def _load_report_payload(report_path: str) -> dict[str, object]:
    path = Path(str(report_path).strip())
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _acceptance_summary(payload: dict[str, object]) -> dict[str, object]:
    packet = payload.get("acceptance_packet", {})
    if not isinstance(packet, dict):
        packet = {}
    verifier = packet.get("verifier_result", {})
    if not isinstance(verifier, dict):
        verifier = {}
    return {
        "synthetic_worker": int(bool(packet.get("synthetic_worker", False))),
        "target_branch": str(packet.get("target_branch", "")).strip(),
        "expected_branch": str(packet.get("expected_branch", "")).strip(),
        "merged_branches": [
            str(branch).strip()
            for branch in packet.get("required_merged_branches", [])
            if str(branch).strip()
        ]
        if isinstance(packet.get("required_merged_branches", []), list)
        else [],
        "tests": len(packet.get("tests", [])) if isinstance(packet.get("tests", []), list) else 0,
        "selected_edits": len(packet.get("selected_edits", []))
        if isinstance(packet.get("selected_edits", []), list)
        else 0,
        "candidate_sets": len(packet.get("candidate_edit_sets", []))
        if isinstance(packet.get("candidate_edit_sets", []), list)
        else 0,
        "verifier_passed": int(bool(verifier.get("passed", False))),
    }


def _family_trust_summary(config: KernelConfig, *, family: str) -> dict[str, object]:
    ledger = build_unattended_trust_ledger(config)
    summary = (
        dict(ledger.get("family_summaries", {}).get(family, {}))
        if isinstance(ledger.get("family_summaries", {}), dict)
        else {}
    )
    assessment = (
        dict(ledger.get("family_assessments", {}).get(family, {}))
        if isinstance(ledger.get("family_assessments", {}), dict)
        else {}
    )
    return {
        "family": family,
        "status": str(assessment.get("status", "absent")).strip() or "absent",
        "detail": str(assessment.get("detail", "")).strip(),
        "reports": int(summary.get("total", 0)),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "hidden_side_effect_risk_rate": float(summary.get("hidden_side_effect_risk_rate", 0.0)),
    }


def _job_structural_summary(metadata: dict[str, object]) -> dict[str, object]:
    classes = discover_structural_classes(metadata)
    aliases = structural_class_family_aliases(classes)
    return {
        "discovered_structural_classes": classes,
        "discovered_structural_class_ids": [
            str(entry.get("class_id", "")).strip()
            for entry in classes
            if str(entry.get("class_id", "")).strip()
        ],
        "discovered_structural_class_kinds": [
            str(entry.get("class_kind", "")).strip()
            for entry in classes
            if str(entry.get("class_kind", "")).strip()
        ],
        "discovered_structural_family_aliases": aliases,
    }


def _job_readiness(
    queue: DelegatedJobQueue,
    job: object,
    *,
    config: KernelConfig,
    repo_root: Path,
    bank: TaskBank,
) -> dict[str, object]:
    acceptance = _acceptance_summary(_load_report_payload(getattr(job, "report_path", "")))
    task = resolve_job_task(bank, job)
    runtime_task = prepare_runtime_task(
        task,
        runtime_overrides=getattr(job, "runtime_overrides", {}),
        job_id=getattr(job, "job_id", ""),
    )
    metadata = dict(getattr(runtime_task, "metadata", {}))
    workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
    semantic_verifier = (
        dict(metadata.get("semantic_verifier", {}))
        if isinstance(metadata.get("semantic_verifier", {}), dict)
        else {}
    )
    structural_summary = _job_structural_summary(metadata)
    worker_job = bool(str(workflow_guard.get("worker_branch", "")).strip())
    integrator_job = bool(
        [
            str(branch).strip()
            for branch in semantic_verifier.get("required_merged_branches", [])
            if str(branch).strip()
        ]
    )
    acceptance_ready = bool(
        getattr(job, "state", "") == "completed"
        and getattr(job, "outcome", "") == "success"
        and acceptance
        and acceptance.get("verifier_passed", 0) == 1
    )
    promoted = bool(str(getattr(job, "promoted_at", "")).strip())
    promotable = bool(
        acceptance_ready
        and not promoted
        and not worker_job
        and acceptance.get("synthetic_worker", 0) != 1
        and (
            acceptance.get("merged_branches")
            or acceptance.get("target_branch")
            or acceptance.get("expected_branch")
        )
    )
    blockers: list[str] = []
    blocker_details: list[str] = []
    if getattr(job, "state", "") not in TERMINAL_JOB_STATES:
        dependencies_ready, dependency_reason = delegated_job_dependency_status(queue, job)
        if not dependencies_ready:
            blockers.append("dependency")
            blocker_details.append(dependency_reason)
        preflight = run_unattended_preflight(config, runtime_task, repo_root=repo_root)
        for check in [item for item in preflight.checks if not item.passed]:
            if check.name == "trust_posture":
                blockers.append("trust")
            elif check.name == "operator_policy":
                blockers.append("scope")
            else:
                blockers.append("preflight")
            blocker_details.append(f"{check.name}:{check.detail}")
    elif acceptance and acceptance.get("verifier_passed", 0) != 1:
        blockers.append("acceptance")
        blocker_details.append("acceptance_verifier_failed")
    elif getattr(job, "state", "") in {"failed", "safe_stop"}:
        blockers.append("outcome")
        if getattr(job, "last_error", ""):
            blocker_details.append(str(getattr(job, "last_error", "")).strip())
    return {
        "benchmark_family": str(metadata.get("benchmark_family", "bounded")).strip() or "bounded",
        **structural_summary,
        "acceptance": acceptance,
        "acceptance_ready": acceptance_ready,
        "promoted": promoted,
        "promotable": promotable,
        "worker_job": worker_job,
        "integrator_job": integrator_job,
        "runnable": getattr(job, "state", "") not in TERMINAL_JOB_STATES and not blockers,
        "blocked": bool(blockers),
        "blockers": sorted(dict.fromkeys(blockers)),
        "blocker_details": blocker_details,
    }


def _job_display_sort_key(job: object, readiness: dict[str, object]) -> tuple[int, int, int, int, str]:
    if bool(readiness.get("runnable", False)) and bool(readiness.get("worker_job", False)):
        bucket = 0
    elif bool(readiness.get("runnable", False)):
        bucket = 1
    elif bool(readiness.get("promotable", False)):
        bucket = 2
    elif bool(readiness.get("blocked", False)):
        bucket = 3
    else:
        bucket = 4
    return (
        bucket,
        0 if getattr(job, "state", "") == "in_progress" else 1,
        -int(getattr(job, "priority", 0)),
        int(getattr(job, "attempt_count", 0)),
        str(getattr(job, "job_id", "")),
    )


def _latest_scheduler_decision(job: object) -> str:
    history = getattr(job, "history", [])
    if not isinstance(history, list):
        return ""
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("event", "")).strip() != "scheduler_decision":
            continue
        detail = str(entry.get("detail", "")).strip()
        return detail.split("|", 1)[0].strip()
    return ""


def _readiness_payload(readiness: dict[str, object]) -> dict[str, object]:
    return {
        "acceptance_ready": bool(readiness.get("acceptance_ready", False)),
        "promoted": bool(readiness.get("promoted", False)),
        "promotable": bool(readiness.get("promotable", False)),
        "worker_job": bool(readiness.get("worker_job", False)),
        "integrator_job": bool(readiness.get("integrator_job", False)),
        "runnable": bool(readiness.get("runnable", False)),
        "blocked": bool(readiness.get("blocked", False)),
        "blockers": [
            str(value).strip()
            for value in readiness.get("blockers", [])
            if str(value).strip()
        ]
        if isinstance(readiness.get("blockers", []), list)
        else [],
        "blocker_details": [
            str(value).strip()
            for value in readiness.get("blocker_details", [])
            if str(value).strip()
        ]
        if isinstance(readiness.get("blocker_details", []), list)
        else [],
        "discovered_structural_class_ids": [
            str(value).strip()
            for value in readiness.get("discovered_structural_class_ids", [])
            if str(value).strip()
        ]
        if isinstance(readiness.get("discovered_structural_class_ids", []), list)
        else [],
        "discovered_structural_class_kinds": [
            str(value).strip()
            for value in readiness.get("discovered_structural_class_kinds", [])
            if str(value).strip()
        ]
        if isinstance(readiness.get("discovered_structural_class_kinds", []), list)
        else [],
        "discovered_structural_family_aliases": [
            str(value).strip()
            for value in readiness.get("discovered_structural_family_aliases", [])
            if str(value).strip()
        ]
        if isinstance(readiness.get("discovered_structural_family_aliases", []), list)
        else [],
    }


def _job_payload(job: object, readiness: dict[str, object]) -> dict[str, object]:
    acceptance = readiness.get("acceptance", {})
    return {
        "job": job.to_dict() if hasattr(job, "to_dict") else {},
        "benchmark_family": str(readiness.get("benchmark_family", "")).strip(),
        "discovered_structural_classes": [
            dict(entry)
            for entry in readiness.get("discovered_structural_classes", [])
            if isinstance(entry, dict)
        ]
        if isinstance(readiness.get("discovered_structural_classes", []), list)
        else [],
        "acceptance": dict(acceptance) if isinstance(acceptance, dict) else {},
        "readiness": _readiness_payload(readiness),
        "latest_scheduler_decision": _latest_scheduler_decision(job),
    }


def _queue_state_counts(jobs_with_readiness: list[tuple[object, dict[str, object]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job, _ in jobs_with_readiness:
        state = str(getattr(job, "state", "")).strip() or "unknown"
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _queue_blocker_counts(jobs_with_readiness: list[tuple[object, dict[str, object]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, readiness in jobs_with_readiness:
        blockers = readiness.get("blockers", [])
        if not isinstance(blockers, list):
            continue
        for blocker in blockers:
            token = str(blocker).strip()
            if not token:
                continue
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))


def _queue_role_closeout(
    jobs_with_readiness: list[tuple[object, dict[str, object]]],
    leases: list[dict[str, object]],
    trust: dict[str, object],
) -> dict[str, object]:
    jobs = [job for job, _ in jobs_with_readiness]
    active_leases = [lease for lease in leases if isinstance(lease, dict)]
    assessment = trust.get("overall_assessment", {}) if isinstance(trust, dict) else {}
    if not isinstance(assessment, dict):
        assessment = {}
    trust_status = str(assessment.get("status", "")).strip() or "unknown"
    trust_passed = bool(assessment.get("passed", False)) and trust_status == "trusted"
    completed_success_jobs = [
        job
        for job in jobs
        if str(getattr(job, "state", "")).strip() == "completed"
        and str(getattr(job, "outcome", "")).strip() == "success"
    ]
    unfinished_jobs = [
        job
        for job in jobs
        if str(getattr(job, "state", "")).strip() not in TERMINAL_JOB_STATES
    ]
    terminal_non_success_jobs = [
        job
        for job in jobs
        if str(getattr(job, "state", "")).strip() in TERMINAL_JOB_STATES
        and not (
            str(getattr(job, "state", "")).strip() == "completed"
            and str(getattr(job, "outcome", "")).strip() == "success"
        )
    ]
    blocked_jobs = [
        job
        for job, readiness in jobs_with_readiness
        if bool(readiness.get("blocked", False))
        and str(getattr(job, "state", "")).strip() not in TERMINAL_JOB_STATES
    ]
    total_jobs = len(jobs)
    queue_empty_success = (
        total_jobs > 0
        and len(completed_success_jobs) == total_jobs
        and not unfinished_jobs
        and not terminal_non_success_jobs
    )
    closeout_ready = queue_empty_success and not active_leases and trust_passed
    if closeout_ready:
        mode = "queue_empty_trusted"
    elif total_jobs == 0:
        mode = "no_jobs"
    elif active_leases:
        mode = "active_leases"
    elif terminal_non_success_jobs:
        mode = "terminal_non_success"
    elif blocked_jobs:
        mode = "blocked_open_work"
    elif unfinished_jobs:
        mode = "open_work"
    elif queue_empty_success:
        mode = "queue_empty_untrusted"
    else:
        mode = "unknown"
    return {
        "closeout_ready": closeout_ready,
        "closeout_mode": mode,
        "operator_steering_required": not closeout_ready,
        "total_jobs": total_jobs,
        "completed_success_jobs": len(completed_success_jobs),
        "unfinished_jobs": len(unfinished_jobs),
        "terminal_non_success_jobs": len(terminal_non_success_jobs),
        "blocked_open_jobs": len(blocked_jobs),
        "active_leases": len(active_leases),
        "trust_status": trust_status,
        "trust_passed": trust_passed,
        "unfinished_job_ids": [str(getattr(job, "job_id", "")).strip() for job in unfinished_jobs],
        "terminal_non_success_job_ids": [
            str(getattr(job, "job_id", "")).strip() for job in terminal_non_success_jobs
        ],
        "blocked_open_job_ids": [str(getattr(job, "job_id", "")).strip() for job in blocked_jobs],
    }


def _queue_family_rollups(jobs_with_readiness: list[tuple[object, dict[str, object]]]) -> dict[str, dict[str, object]]:
    families: dict[str, dict[str, object]] = {}
    for job, readiness in jobs_with_readiness:
        family = str(readiness.get("benchmark_family", "")).strip() or "bounded"
        rollup = families.setdefault(
            family,
            {
                "total_jobs": 0,
                "runnable_jobs": 0,
                "blocked_jobs": 0,
                "promotable_jobs": 0,
                "worker_jobs": 0,
                "integrator_jobs": 0,
                "budget_groups": set(),
            },
        )
        rollup["total_jobs"] = int(rollup["total_jobs"]) + 1
        if bool(readiness.get("runnable", False)):
            rollup["runnable_jobs"] = int(rollup["runnable_jobs"]) + 1
        if bool(readiness.get("blocked", False)):
            rollup["blocked_jobs"] = int(rollup["blocked_jobs"]) + 1
        if bool(readiness.get("promotable", False)):
            rollup["promotable_jobs"] = int(rollup["promotable_jobs"]) + 1
        if bool(readiness.get("worker_job", False)):
            rollup["worker_jobs"] = int(rollup["worker_jobs"]) + 1
        if bool(readiness.get("integrator_job", False)):
            rollup["integrator_jobs"] = int(rollup["integrator_jobs"]) + 1
        budget_group = str(getattr(job, "budget_group", "")).strip()
        if budget_group:
            budget_groups = rollup.get("budget_groups", set())
            if isinstance(budget_groups, set):
                budget_groups.add(budget_group)
    return {
        family: {
            "total_jobs": int(rollup["total_jobs"]),
            "runnable_jobs": int(rollup["runnable_jobs"]),
            "blocked_jobs": int(rollup["blocked_jobs"]),
            "promotable_jobs": int(rollup["promotable_jobs"]),
            "worker_jobs": int(rollup["worker_jobs"]),
            "integrator_jobs": int(rollup["integrator_jobs"]),
            "budget_groups": sorted(
                str(group).strip()
                for group in rollup.get("budget_groups", set())
                if str(group).strip()
            ),
        }
        for family, rollup in sorted(families.items())
    }


def _queue_structural_class_rollups(
    jobs_with_readiness: list[tuple[object, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    rollups: dict[str, dict[str, object]] = {}
    for job, readiness in jobs_with_readiness:
        classes = readiness.get("discovered_structural_classes", [])
        if not isinstance(classes, list):
            continue
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            class_id = str(entry.get("class_id", "")).strip()
            if not class_id:
                continue
            rollup = rollups.setdefault(
                class_id,
                {
                    "class_kind": str(entry.get("class_kind", "")).strip(),
                    "total_jobs": 0,
                    "runnable_jobs": 0,
                    "blocked_jobs": 0,
                    "promotable_jobs": 0,
                    "benchmark_families": set(),
                    "family_aliases": set(),
                },
            )
            rollup["total_jobs"] = int(rollup["total_jobs"]) + 1
            if bool(readiness.get("runnable", False)):
                rollup["runnable_jobs"] = int(rollup["runnable_jobs"]) + 1
            if bool(readiness.get("blocked", False)):
                rollup["blocked_jobs"] = int(rollup["blocked_jobs"]) + 1
            if bool(readiness.get("promotable", False)):
                rollup["promotable_jobs"] = int(rollup["promotable_jobs"]) + 1
            benchmark_families = rollup.get("benchmark_families", set())
            family_aliases = rollup.get("family_aliases", set())
            if isinstance(benchmark_families, set):
                benchmark_families.add(str(readiness.get("benchmark_family", "")).strip() or "bounded")
            if isinstance(family_aliases, set):
                for alias in readiness.get("discovered_structural_family_aliases", []):
                    token = str(alias).strip()
                    if token:
                        family_aliases.add(token)
    return {
        class_id: {
            "class_kind": str(rollup["class_kind"]).strip(),
            "total_jobs": int(rollup["total_jobs"]),
            "runnable_jobs": int(rollup["runnable_jobs"]),
            "blocked_jobs": int(rollup["blocked_jobs"]),
            "promotable_jobs": int(rollup["promotable_jobs"]),
            "benchmark_families": sorted(
                str(value).strip()
                for value in rollup.get("benchmark_families", set())
                if str(value).strip()
            ),
            "family_aliases": sorted(
                str(value).strip()
                for value in rollup.get("family_aliases", set())
                if str(value).strip()
            ),
        }
        for class_id, rollup in sorted(rollups.items())
    }


def _active_lease_role_rollup(
    jobs_with_readiness: list[tuple[object, dict[str, object]]],
    leases: list[dict[str, object]],
) -> dict[str, object]:
    job_readiness = {
        str(getattr(job, "job_id", "")).strip(): readiness
        for job, readiness in jobs_with_readiness
        if str(getattr(job, "job_id", "")).strip()
    }
    benchmark_families: dict[str, int] = {}
    budget_groups: dict[str, int] = {}
    shared_repo_ids: dict[str, int] = {}
    worker_jobs = 0
    integrator_jobs = 0
    other_jobs = 0
    for lease in leases:
        if not isinstance(lease, dict):
            continue
        job_id = str(lease.get("job_id", "")).strip()
        readiness = job_readiness.get(job_id, {})
        if bool(readiness.get("worker_job", False)):
            worker_jobs += 1
        elif bool(readiness.get("integrator_job", False)):
            integrator_jobs += 1
        else:
            other_jobs += 1
        family = str(readiness.get("benchmark_family", "")).strip() or "bounded"
        benchmark_families[family] = benchmark_families.get(family, 0) + 1
        budget_group = str(lease.get("budget_group", "")).strip()
        if budget_group:
            budget_groups[budget_group] = budget_groups.get(budget_group, 0) + 1
        shared_repo_id = str(lease.get("shared_repo_id", "")).strip()
        if shared_repo_id:
            shared_repo_ids[shared_repo_id] = shared_repo_ids.get(shared_repo_id, 0) + 1
    return {
        "total": worker_jobs + integrator_jobs + other_jobs,
        "worker_jobs": worker_jobs,
        "integrator_jobs": integrator_jobs,
        "other_jobs": other_jobs,
        "benchmark_families": dict(sorted(benchmark_families.items())),
        "budget_groups": dict(sorted(budget_groups.items())),
        "shared_repo_ids": dict(sorted(shared_repo_ids.items())),
    }


def _acceptance_review_entry(
    job: object,
    readiness: dict[str, object],
    *,
    config: KernelConfig,
    bank: TaskBank,
) -> dict[str, object]:
    task = resolve_job_task(bank, job)
    runtime_task = prepare_runtime_task(
        task,
        runtime_overrides=getattr(job, "runtime_overrides", {}),
        job_id=getattr(job, "job_id", ""),
    )
    metadata = dict(getattr(runtime_task, "metadata", {}))
    workflow_guard = (
        dict(metadata.get("workflow_guard", {}))
        if isinstance(metadata.get("workflow_guard", {}), dict)
        else {}
    )
    semantic_verifier = (
        dict(metadata.get("semantic_verifier", {}))
        if isinstance(metadata.get("semantic_verifier", {}), dict)
        else {}
    )
    acceptance = dict(readiness.get("acceptance", {})) if isinstance(readiness.get("acceptance", {}), dict) else {}
    merged_branches = [
        str(branch).strip()
        for branch in acceptance.get("merged_branches", [])
        if str(branch).strip()
    ] if isinstance(acceptance.get("merged_branches", []), list) else []
    if not merged_branches:
        merged_branches = [
            str(branch).strip()
            for branch in semantic_verifier.get("required_merged_branches", [])
            if str(branch).strip()
        ] if isinstance(semantic_verifier.get("required_merged_branches", []), list) else []
    benchmark_family = str(readiness.get("benchmark_family", "bounded")).strip() or "bounded"
    return {
        "job": _job_payload(job, readiness),
        "benchmark_family": benchmark_family,
        "shared_repo_id": str(workflow_guard.get("shared_repo_id", "")).strip(),
        "target_branch": (
            str(acceptance.get("target_branch", "")).strip()
            or str(workflow_guard.get("target_branch", "")).strip()
        ),
        "expected_branch": (
            str(acceptance.get("expected_branch", "")).strip()
            or str(workflow_guard.get("worker_branch", "")).strip()
        ),
        "merged_branches": merged_branches,
        "report_path": str(getattr(job, "report_path", "")).strip(),
        "checkpoint_path": str(getattr(job, "checkpoint_path", "")).strip(),
        "trust_family": _family_trust_summary(config, family=benchmark_family),
    }


def _acceptance_review_rollup(entries: list[dict[str, object]]) -> dict[str, object]:
    benchmark_families: dict[str, int] = {}
    shared_repo_ids: dict[str, int] = {}
    target_branches: dict[str, int] = {}
    promotable_jobs = 0
    promoted_jobs = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_payload = entry.get("job", {})
        readiness = job_payload.get("readiness", {}) if isinstance(job_payload, dict) else {}
        if bool(readiness.get("promotable", False)):
            promotable_jobs += 1
        if bool(readiness.get("promoted", False)):
            promoted_jobs += 1
        family = str(entry.get("benchmark_family", "")).strip() or "bounded"
        benchmark_families[family] = benchmark_families.get(family, 0) + 1
        shared_repo_id = str(entry.get("shared_repo_id", "")).strip()
        if shared_repo_id:
            shared_repo_ids[shared_repo_id] = shared_repo_ids.get(shared_repo_id, 0) + 1
        target_branch = str(entry.get("target_branch", "")).strip()
        if target_branch:
            target_branches[target_branch] = target_branches.get(target_branch, 0) + 1
    return {
        "total_jobs": len(entries),
        "promotable_jobs": promotable_jobs,
        "promoted_jobs": promoted_jobs,
        "benchmark_families": dict(sorted(benchmark_families.items())),
        "shared_repo_ids": dict(sorted(shared_repo_ids.items())),
        "target_branches": dict(sorted(target_branches.items())),
    }


def _capability_scope_entries(capability_policy: dict[str, object]) -> list[dict[str, object]]:
    entries_payload: list[dict[str, object]] = []
    capability_policies = capability_policy.get("capability_policies", {})
    if not isinstance(capability_policies, dict):
        return entries_payload
    for capability, entries in sorted(capability_policies.items()):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("source") != "module" or not bool(entry.get("enabled", False)):
                continue
            entries_payload.append(
                {
                    "capability": str(capability),
                    "module_id": str(entry.get("module_id", "")).strip(),
                    "access_tier": str(entry.get("access_tier", "")).strip(),
                    "write_tier": str(entry.get("write_tier", "")).strip(),
                    "read_only": bool(entry.get("read_only", False)),
                    "http_allowed_hosts": [
                        str(value).strip()
                        for value in entry.get("http_allowed_hosts", [])
                        if str(value).strip()
                    ]
                    if isinstance(entry.get("http_allowed_hosts", []), list)
                    else [],
                    "repo_scopes": [
                        str(value).strip()
                        for value in entry.get("repo_scopes", [])
                        if str(value).strip()
                    ]
                    if isinstance(entry.get("repo_scopes", []), list)
                    else [],
                    "account_scopes": [
                        str(value).strip()
                        for value in entry.get("account_scopes", [])
                        if str(value).strip()
                    ]
                    if isinstance(entry.get("account_scopes", []), list)
                    else [],
                }
            )
    return entries_payload


def _add_runtime_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--parallel-worker-count", type=int, default=None)
    parser.add_argument("--use-tolbert-context", choices=("0", "1"), default=None)
    parser.add_argument("--use-skills", choices=("0", "1"), default=None)
    parser.add_argument("--use-graph-memory", choices=("0", "1"), default=None)
    parser.add_argument("--use-world-model", choices=("0", "1"), default=None)
    parser.add_argument("--use-planner", choices=("0", "1"), default=None)
    parser.add_argument("--use-role-specialization", choices=("0", "1"), default=None)
    parser.add_argument("--use-prompt-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-curriculum-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-retrieval-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-state-estimation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-trust-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-recovery-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-delegation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-operator-policy-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-transition-model-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--asi-coding-require-live-llm", choices=("0", "1"), default=None)
    parser.add_argument("--allow-git-commands", choices=("0", "1"), default=None)
    parser.add_argument("--allow-http-requests", choices=("0", "1"), default=None)
    parser.add_argument("--allow-generated-path-mutations", choices=("0", "1"), default=None)
    parser.add_argument("--shared-repo-id", default=None)
    parser.add_argument("--worker-branch", default=None)
    parser.add_argument("--target-branch", default=None)
    parser.add_argument("--claim-path", action="append", default=None)


def _runtime_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model_name"] = args.model
    if getattr(args, "parallel_worker_count", None) is not None:
        overrides["parallel_worker_count"] = max(0, int(args.parallel_worker_count))
    if getattr(args, "shared_repo_id", None):
        overrides["shared_repo_id"] = str(args.shared_repo_id).strip()
    if getattr(args, "worker_branch", None):
        overrides["worker_branch"] = str(args.worker_branch).strip()
    if getattr(args, "target_branch", None):
        overrides["target_branch"] = str(args.target_branch).strip()
    claim_paths = getattr(args, "claim_path", None)
    if claim_paths:
        overrides["claimed_paths"] = [str(path).strip() for path in claim_paths if str(path).strip()]
    for arg_name, field_name in (
        ("use_tolbert_context", "use_tolbert_context"),
        ("use_skills", "use_skills"),
        ("use_graph_memory", "use_graph_memory"),
        ("use_world_model", "use_world_model"),
        ("use_planner", "use_planner"),
        ("use_role_specialization", "use_role_specialization"),
        ("use_prompt_proposals", "use_prompt_proposals"),
        ("use_curriculum_proposals", "use_curriculum_proposals"),
        ("use_retrieval_proposals", "use_retrieval_proposals"),
        ("use_state_estimation_proposals", "use_state_estimation_proposals"),
        ("use_trust_proposals", "use_trust_proposals"),
        ("use_recovery_proposals", "use_recovery_proposals"),
        ("use_delegation_proposals", "use_delegation_proposals"),
        ("use_operator_policy_proposals", "use_operator_policy_proposals"),
        ("use_transition_model_proposals", "use_transition_model_proposals"),
        ("asi_coding_require_live_llm", "asi_coding_require_live_llm"),
        ("allow_git_commands", "unattended_allow_git_commands"),
        ("allow_http_requests", "unattended_allow_http_requests"),
        ("allow_generated_path_mutations", "unattended_allow_generated_path_mutations"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value == "1"
    return overrides


def _apply_base_config_overrides(config: KernelConfig, args: argparse.Namespace) -> None:
    overrides = _runtime_overrides_from_args(args)
    for field, value in overrides.items():
        setattr(config, field, value)
    if getattr(args, "max_concurrency", None) is not None:
        config.delegated_job_max_concurrency = max(1, int(args.max_concurrency))
    if getattr(args, "max_active_per_budget_group", None) is not None:
        config.delegated_job_max_active_per_budget_group = max(0, int(args.max_active_per_budget_group))
    if getattr(args, "max_queued_per_budget_group", None) is not None:
        config.delegated_job_max_queued_per_budget_group = max(0, int(args.max_queued_per_budget_group))
    if getattr(args, "max_artifact_bytes", None) is not None:
        config.delegated_job_max_artifact_bytes = max(0, int(args.max_artifact_bytes))
    if getattr(args, "max_subprocesses_per_job", None) is not None:
        config.delegated_job_max_subprocesses_per_job = max(1, int(args.max_subprocesses_per_job))
    if getattr(args, "max_consecutive_selections_per_budget_group", None) is not None:
        config.delegated_job_max_consecutive_selections_per_budget_group = max(
            0, int(args.max_consecutive_selections_per_budget_group)
        )


def _apply_retained_delegation_policy(config: KernelConfig) -> None:
    for field, value in delegation_policy_snapshot(config).items():
        if hasattr(config, field):
            setattr(config, field, value)


def _add_governance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--max-active-per-budget-group", type=int, default=None)
    parser.add_argument("--max-queued-per-budget-group", type=int, default=None)
    parser.add_argument("--max-artifact-bytes", type=int, default=None)
    parser.add_argument("--max-subprocesses-per-job", type=int, default=None)
    parser.add_argument("--max-consecutive-selections-per-budget-group", type=int, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue")
    enqueue.add_argument("--task-id", required=True)
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--budget-group", default="default")
    enqueue.add_argument("--deadline", default="")
    enqueue.add_argument("--notes", default="")
    enqueue.add_argument("--decompose-workers", choices=("0", "1"), default="0")
    _add_runtime_override_args(enqueue)
    _add_governance_args(enqueue)

    enqueue_manifest = subparsers.add_parser("enqueue-manifest")
    enqueue_manifest.add_argument("--manifest-path", action="append", required=True)
    enqueue_manifest.add_argument("--task-id", action="append", default=None)
    enqueue_manifest.add_argument("--family", action="append", default=None)
    enqueue_manifest.add_argument("--limit", type=int, default=0)
    enqueue_manifest.add_argument("--priority-start", type=int, default=100)
    enqueue_manifest.add_argument("--priority-step", type=int, default=-1)
    enqueue_manifest.add_argument("--budget-group", default="default")
    enqueue_manifest.add_argument("--deadline", default="")
    enqueue_manifest.add_argument("--notes", default="")
    enqueue_manifest.add_argument("--decompose-workers", choices=("0", "1"), default="0")
    enqueue_manifest.add_argument("--json", action="store_true")
    _add_runtime_override_args(enqueue_manifest)
    _add_governance_args(enqueue_manifest)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--state", action="append", default=None)
    list_parser.add_argument("--show-blockers", choices=("0", "1"), default="0")
    list_parser.add_argument("--blocked-only", choices=("0", "1"), default="0")
    list_parser.add_argument("--ready-to-accept-only", choices=("0", "1"), default="0")
    list_parser.add_argument("--json", action="store_true")

    promotable = subparsers.add_parser("promotable")
    promotable.add_argument("--state", action="append", default=None)
    promotable.add_argument("--show-blockers", choices=("0", "1"), default="0")
    promotable.add_argument("--json", action="store_true")

    acceptance_review = subparsers.add_parser("acceptance-review")
    acceptance_review.add_argument("--state", action="append", default=None)
    acceptance_review.add_argument("--show-blockers", choices=("0", "1"), default="0")
    acceptance_review.add_argument("--include-promoted", choices=("0", "1"), default="0")
    acceptance_review.add_argument("--json", action="store_true")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--job-id", required=True)
    inspect.add_argument("--json", action="store_true")

    promote = subparsers.add_parser("promote")
    promote.add_argument("--job-id", required=True)
    promote.add_argument("--detail", default="accepted for delivery")

    next_runnable = subparsers.add_parser("next-runnable")
    next_runnable.add_argument("--show-blockers", choices=("0", "1"), default="0")
    next_runnable.add_argument("--json", action="store_true")

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--job-id", required=True)

    status = subparsers.add_parser("status")
    _add_governance_args(status)
    status.add_argument("--json", action="store_true")

    run_next = subparsers.add_parser("run-next")
    run_next.add_argument("--enforce-preflight", choices=("0", "1"), default="1")
    _add_runtime_override_args(run_next)
    _add_governance_args(run_next)

    drain = subparsers.add_parser("drain")
    drain.add_argument("--limit", type=int, default=0)
    drain.add_argument("--enforce-preflight", choices=("0", "1"), default="1")
    _add_runtime_override_args(drain)
    _add_governance_args(drain)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    _apply_retained_delegation_policy(config)
    _apply_base_config_overrides(config, args)
    queue = DelegatedJobQueue(config.delegated_job_queue_path)
    runtime = DelegatedRuntimeController(config.delegated_job_runtime_state_path)
    try:
        runtime_bank = TaskBank(config=config)
    except TypeError:
        runtime_bank = TaskBank()

    if args.command == "enqueue":
        runtime_bank.get(args.task_id)
        jobs = enqueue_with_parallel_worker_decomposition(
            queue,
            bank=runtime_bank,
            task_id=args.task_id,
            priority=args.priority,
            budget_group=args.budget_group,
            deadline_at=str(args.deadline).strip(),
            notes=str(args.notes).strip(),
            runtime_overrides=_runtime_overrides_from_args(args),
            max_queued_jobs_for_budget_group=max(0, int(config.delegated_job_max_queued_per_budget_group)),
        ) if args.decompose_workers == "1" else [
            queue.enqueue(
                task_id=args.task_id,
                priority=args.priority,
                budget_group=args.budget_group,
                deadline_at=str(args.deadline).strip(),
                notes=str(args.notes).strip(),
                runtime_overrides=_runtime_overrides_from_args(args),
                max_queued_jobs_for_budget_group=max(0, int(config.delegated_job_max_queued_per_budget_group)),
                task_bank=runtime_bank,
            )
        ]
        for job in jobs:
            print(
                f"job_id={job.job_id} state={job.state} task_id={job.task_id} "
                f"priority={job.priority} budget_group={job.budget_group}"
            )
        return

    if args.command == "enqueue-manifest":
        manifest_paths = tuple(str(path).strip() for path in args.manifest_path if str(path).strip())
        manifest_bank = TaskBank(config=config, external_task_manifests=manifest_paths)
        requested_task_ids = [str(task_id).strip() for task_id in (args.task_id or []) if str(task_id).strip()]
        requested_task_id_set = set(requested_task_ids)
        requested_families = {str(family).strip() for family in (args.family or []) if str(family).strip()}
        manifest_tasks = [
            task
            for task in manifest_bank.list()
            if str(task.metadata.get("task_origin", "")).strip() == "external_manifest"
            and (not requested_task_id_set or task.task_id in requested_task_id_set)
            and (not requested_families or str(task.metadata.get("benchmark_family", "")).strip() in requested_families)
        ]
        if requested_task_ids:
            task_by_id = {task.task_id: task for task in manifest_tasks}
            manifest_tasks = [task_by_id[task_id] for task_id in requested_task_ids if task_id in task_by_id]
        limit = max(0, int(args.limit))
        if limit:
            manifest_tasks = manifest_tasks[:limit]
        if not manifest_tasks:
            print("enqueue_manifest no_matching_tasks=1", file=sys.stderr)
            raise SystemExit(1)
        records: list[dict[str, object]] = []
        priority = int(args.priority_start)
        for task in manifest_tasks:
            runtime_overrides = _runtime_overrides_from_args(args)
            runtime_overrides["task_payload"] = task.to_dict()
            jobs = enqueue_with_parallel_worker_decomposition(
                queue,
                bank=manifest_bank,
                task_id=task.task_id,
                priority=priority,
                budget_group=args.budget_group,
                deadline_at=str(args.deadline).strip(),
                notes=str(args.notes).strip(),
                runtime_overrides=runtime_overrides,
                max_queued_jobs_for_budget_group=max(0, int(config.delegated_job_max_queued_per_budget_group)),
            ) if args.decompose_workers == "1" else [
                queue.enqueue(
                    task_id=task.task_id,
                    priority=priority,
                    budget_group=args.budget_group,
                    deadline_at=str(args.deadline).strip(),
                    notes=str(args.notes).strip(),
                    runtime_overrides=runtime_overrides,
                    max_queued_jobs_for_budget_group=max(0, int(config.delegated_job_max_queued_per_budget_group)),
                    task_bank=manifest_bank,
                )
            ]
            for job in jobs:
                record = {
                    "job_id": job.job_id,
                    "state": job.state,
                    "task_id": job.task_id,
                    "priority": job.priority,
                    "budget_group": job.budget_group,
                    "benchmark_family": str(task.metadata.get("benchmark_family", "")).strip(),
                    "external_manifest_path": str(task.metadata.get("external_manifest_path", "")).strip(),
                }
                records.append(record)
                if not getattr(args, "json", False):
                    print(
                        f"job_id={job.job_id} state={job.state} task_id={job.task_id} "
                        f"priority={job.priority} budget_group={job.budget_group} "
                        f"benchmark_family={record['benchmark_family']}"
                    )
            priority += int(args.priority_step)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "manifest_paths": list(manifest_paths),
                        "selected_task_count": len(manifest_tasks),
                        "enqueued_job_count": len(records),
                        "enqueued_jobs": records,
                    },
                    indent=2,
                )
            )
        return

    if args.command in {"list", "promotable", "acceptance-review", "next-runnable", "inspect", "promote"}:
        states = set(getattr(args, "state", None) or [])
        repo_root = Path(__file__).resolve().parents[1]
        show_blockers = getattr(args, "show_blockers", "0") == "1" or (
            getattr(args, "blocked_only", "0") == "1" if args.command == "list" else False
        )
        blocked_only = getattr(args, "blocked_only", "0") == "1" if args.command == "list" else False
        ready_only = (
            getattr(args, "ready_to_accept_only", "0") == "1" if args.command == "list" else False
        )
        promotable_only = args.command == "promotable"
        acceptance_review_only = args.command == "acceptance-review"
        include_promoted = (
            getattr(args, "include_promoted", "0") == "1"
            if acceptance_review_only
            else False
        )
        jobs_with_readiness = [
            (job, _job_readiness(queue, job, config=config, repo_root=repo_root, bank=runtime_bank))
            for job in queue.list_jobs(states=states or None)
        ]
        jobs_with_readiness.sort(key=lambda item: _job_display_sort_key(item[0], item[1]))
        if args.command == "inspect":
            selected = next((item for item in jobs_with_readiness if item[0].job_id == args.job_id), None)
            if selected is None:
                raise SystemExit(f"unknown job_id: {args.job_id}")
            job, readiness = selected
            report_payload = _load_report_payload(getattr(job, "report_path", ""))
            task = resolve_job_task(runtime_bank, job)
            runtime_task = prepare_runtime_task(
                task,
                runtime_overrides=getattr(job, "runtime_overrides", {}),
                job_id=getattr(job, "job_id", ""),
            )
            acceptance = dict(readiness["acceptance"])
            packet = (
                dict(report_payload.get("acceptance_packet", {}))
                if isinstance(report_payload.get("acceptance_packet", {}), dict)
                else {}
            )
            capability_usage = (
                dict(report_payload.get("capability_usage", {}))
                if isinstance(report_payload.get("capability_usage", {}), dict)
                else {}
            )
            benchmark_family = str(runtime_task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
            trust = _family_trust_summary(config, family=benchmark_family)
            if getattr(args, "json", False):
                payload = {
                    "job": _job_payload(job, readiness),
                    "task": {
                        "task_id": str(getattr(runtime_task, "task_id", "")).strip(),
                        "prompt": str(getattr(runtime_task, "prompt", "")).strip(),
                        "workspace_subdir": str(getattr(runtime_task, "workspace_subdir", "")).strip(),
                        "metadata": dict(getattr(runtime_task, "metadata", {}))
                        if isinstance(getattr(runtime_task, "metadata", {}), dict)
                        else {},
                    },
                    "trust_family": trust,
                    "capability_usage": capability_usage,
                    "acceptance_packet": packet,
                    "tests": [
                        dict(test)
                        for test in packet.get("tests", [])
                        if isinstance(test, dict)
                    ]
                    if isinstance(packet.get("tests", []), list)
                    else [],
                    "selected_edits": [
                        dict(edit)
                        for edit in packet.get("selected_edits", [])
                        if isinstance(edit, dict)
                    ]
                    if isinstance(packet.get("selected_edits", []), list)
                    else [],
                    "candidate_edit_sets": [
                        dict(edit)
                        for edit in packet.get("candidate_edit_sets", [])
                        if isinstance(edit, dict)
                    ]
                    if isinstance(packet.get("candidate_edit_sets", []), list)
                    else [],
                    "report_path": str(getattr(job, "report_path", "")).strip(),
                    "checkpoint_path": str(getattr(job, "checkpoint_path", "")).strip(),
                }
                print(json.dumps(payload, indent=2))
                return
            print(
                "job "
                f"job_id={job.job_id} state={job.state} task_id={job.task_id} outcome={job.outcome} "
                f"promotable={int(bool(readiness.get('promotable', False)))} "
                f"promoted={int(bool(readiness.get('promoted', False)))} "
                f"promoted_at={str(getattr(job, 'promoted_at', '')).strip() or '-'}"
            )
            print(
                "acceptance "
                f"verifier_passed={acceptance.get('verifier_passed', 0)} "
                f"synthetic_worker={acceptance.get('synthetic_worker', 0)} "
                f"target_branch={acceptance.get('target_branch', '') or '-'} "
                f"expected_branch={acceptance.get('expected_branch', '') or '-'} "
                f"merged_branches={','.join(acceptance.get('merged_branches', [])) or '-'} "
                f"tests={acceptance.get('tests', 0)} "
                f"selected_edits={acceptance.get('selected_edits', 0)} "
                f"candidate_sets={acceptance.get('candidate_sets', 0)}"
            )
            print(
                "readiness "
                f"runnable={int(bool(readiness.get('runnable', False)))} "
                f"worker_job={int(bool(readiness.get('worker_job', False)))} "
                f"integrator_job={int(bool(readiness.get('integrator_job', False)))} "
                f"acceptance_ready={int(bool(readiness.get('acceptance_ready', False)))} "
                f"blocked={int(bool(readiness.get('blocked', False)))} "
                f"blocked_by={','.join(readiness.get('blockers', [])) or '-'} "
                f"blocker_details={';'.join(str(value) for value in readiness.get('blocker_details', [])) or '-'}"
            )
            print(
                "structural_classes "
                f"class_ids={','.join(readiness.get('discovered_structural_class_ids', [])) or '-'} "
                f"class_kinds={','.join(readiness.get('discovered_structural_class_kinds', [])) or '-'} "
                f"family_aliases={','.join(readiness.get('discovered_structural_family_aliases', [])) or '-'}"
            )
            print(
                "trust_family "
                f"family={trust['family']} status={trust['status']} reports={trust['reports']} "
                f"success_rate={trust['success_rate']:.3f} "
                f"hidden_side_effect_risk_rate={trust['hidden_side_effect_risk_rate']:.3f} "
                f"detail={trust['detail'] or '-'}"
            )
            print(
                "capability_usage "
                f"required={','.join(str(value) for value in capability_usage.get('required_capabilities', [])) or '-'} "
                f"used={','.join(str(value) for value in capability_usage.get('used_capabilities', [])) or '-'} "
                f"external={','.join(str(value) for value in capability_usage.get('external_capabilities_used', [])) or '-'}"
            )
            for test in packet.get("tests", []) if isinstance(packet.get("tests", []), list) else []:
                if not isinstance(test, dict):
                    continue
                print(
                    "test "
                    f"label={str(test.get('label', '')).strip() or '-'} "
                    f"command={str(test.get('command', '')).strip() or '-'}"
                )
            for edit in packet.get("selected_edits", []) if isinstance(packet.get("selected_edits", []), list) else []:
                if not isinstance(edit, dict):
                    continue
                print(
                    "selected_edit "
                    f"path={str(edit.get('path', '')).strip() or '-'} "
                    f"kind={str(edit.get('kind', '')).strip() or '-'} "
                    f"score={str(edit.get('edit_score', '')).strip() or '-'}"
                )
            return
        if args.command == "promote":
            selected = next((item for item in jobs_with_readiness if item[0].job_id == args.job_id), None)
            if selected is None:
                raise SystemExit(f"unknown job_id: {args.job_id}")
            job, readiness = selected
            if not bool(readiness.get("promotable", False)):
                raise SystemExit(
                    f"job not promotable: blocked_by={','.join(readiness.get('blockers', [])) or '-'} "
                    f"state={job.state} outcome={job.outcome}"
                )
            promoted = queue.promote(job.job_id, detail=str(args.detail).strip() or "accepted for delivery")
            print(
                f"job_id={promoted.job_id} promoted=1 promoted_at={promoted.promoted_at} "
                f"task_id={promoted.task_id} detail={promoted.promotion_detail}"
            )
            return
        if args.command == "next-runnable":
            for job, readiness in jobs_with_readiness:
                if not bool(readiness.get("runnable", False)):
                    continue
                if getattr(args, "json", False):
                    print(json.dumps({"job": _job_payload(job, readiness)}, indent=2))
                    return
                print(
                    f"job_id={job.job_id} state={job.state} task_id={job.task_id} "
                    f"priority={job.priority} runnable=1 worker_job={int(bool(readiness.get('worker_job', False)))} "
                    f"integrator_job={int(bool(readiness.get('integrator_job', False)))} "
                    f"blocked_by={','.join(readiness.get('blockers', [])) or '-'}"
                    + (
                        ""
                        if not show_blockers
                        else " "
                        + f"blocker_details={';'.join(str(value) for value in readiness.get('blocker_details', [])) or '-'}"
                    )
                )
                return
            if getattr(args, "json", False):
                print(json.dumps({"job": None}, indent=2))
                return
            print("no_runnable_job=1")
            return
        if acceptance_review_only:
            review_entries: list[dict[str, object]] = []
            for job, readiness in jobs_with_readiness:
                if not bool(readiness.get("integrator_job", False)):
                    continue
                if not bool(readiness.get("acceptance_ready", False)):
                    continue
                if not include_promoted and bool(readiness.get("promoted", False)):
                    continue
                review_entries.append(
                    _acceptance_review_entry(
                        job,
                        readiness,
                        config=config,
                        bank=runtime_bank,
                    )
                )
            review_summary = _acceptance_review_rollup(review_entries)
            if getattr(args, "json", False):
                print(json.dumps({"review": review_summary, "jobs": review_entries}, indent=2))
                return
            print(
                "acceptance_review "
                f"total_jobs={int(review_summary['total_jobs'])} "
                f"promotable_jobs={int(review_summary['promotable_jobs'])} "
                f"promoted_jobs={int(review_summary['promoted_jobs'])} "
                f"families={','.join(f'{name}:{count}' for name, count in review_summary['benchmark_families'].items()) or '-'} "
                f"shared_repos={','.join(f'{name}:{count}' for name, count in review_summary['shared_repo_ids'].items()) or '-'} "
                f"target_branches={','.join(f'{name}:{count}' for name, count in review_summary['target_branches'].items()) or '-'}"
            )
            for entry in review_entries:
                if not isinstance(entry, dict):
                    continue
                job_payload = entry.get("job", {})
                if not isinstance(job_payload, dict):
                    continue
                job_dict = job_payload.get("job", {})
                readiness = job_payload.get("readiness", {})
                trust_family = entry.get("trust_family", {})
                if not isinstance(job_dict, dict) or not isinstance(readiness, dict):
                    continue
                print(
                    f"job_id={str(job_dict.get('job_id', '')).strip()} "
                    f"state={str(job_dict.get('state', '')).strip()} "
                    f"task_id={str(job_dict.get('task_id', '')).strip()} "
                    f"priority={int(job_dict.get('priority', 0))} "
                    f"outcome={str(job_dict.get('outcome', '')).strip()} "
                    f"shared_repo_id={str(entry.get('shared_repo_id', '')).strip() or '-'} "
                    f"target_branch={str(entry.get('target_branch', '')).strip() or '-'} "
                    f"merged_branches={','.join(str(value).strip() for value in entry.get('merged_branches', []) if str(value).strip()) or '-'} "
                    f"promotable={int(bool(readiness.get('promotable', False)))} "
                    f"promoted={int(bool(readiness.get('promoted', False)))} "
                    f"trust_status={str(trust_family.get('status', '')).strip() or '-'} "
                    f"report_path={str(entry.get('report_path', '')).strip() or '-'}"
                    + (
                        ""
                        if not show_blockers
                        else " "
                        + (
                            f"blocked={int(bool(readiness.get('blocked', False)))} "
                            f"blocked_by={','.join(str(value).strip() for value in readiness.get('blockers', []) if str(value).strip()) or '-'} "
                            f"blocker_details={';'.join(str(value).strip() for value in readiness.get('blocker_details', []) if str(value).strip()) or '-'}"
                        )
                    )
                )
            return
        filtered_jobs: list[dict[str, object]] = []
        for job, readiness in jobs_with_readiness:
            acceptance = dict(readiness["acceptance"])
            if promotable_only and not bool(readiness["promotable"]):
                continue
            if ready_only and not bool(readiness["promotable"]):
                continue
            if blocked_only and not bool(readiness["blocked"]):
                continue
            filtered_jobs.append(_job_payload(job, readiness))
            if getattr(args, "json", False):
                continue
            print(
                f"job_id={job.job_id} state={job.state} task_id={job.task_id} "
                f"priority={job.priority} attempts={job.attempt_count} outcome={job.outcome}"
                + (
                    ""
                    if not acceptance
                    else " "
                    + (
                        f"acceptance_verifier_passed={acceptance.get('verifier_passed', 0)} "
                        f"synthetic_worker={acceptance.get('synthetic_worker', 0)} "
                        f"target_branch={acceptance.get('target_branch', '') or '-'} "
                        f"expected_branch={acceptance.get('expected_branch', '') or '-'} "
                        f"merged_branches={','.join(acceptance.get('merged_branches', [])) or '-'} "
                        f"tests={acceptance.get('tests', 0)} "
                        f"selected_edits={acceptance.get('selected_edits', 0)} "
                        f"candidate_sets={acceptance.get('candidate_sets', 0)}"
                    )
                )
                + (
                    ""
                    if not show_blockers
                    else " "
                    + (
                        f"runnable={int(bool(readiness.get('runnable', False)))} "
                        f"worker_job={int(bool(readiness.get('worker_job', False)))} "
                        f"integrator_job={int(bool(readiness.get('integrator_job', False)))} "
                        f"acceptance_ready={int(bool(readiness.get('acceptance_ready', False)))} "
                        f"promoted={int(bool(readiness.get('promoted', False)))} "
                        f"promotable={int(bool(readiness.get('promotable', False)))} "
                        f"blocked={int(bool(readiness.get('blocked', False)))} "
                        f"blocked_by={','.join(readiness.get('blockers', [])) or '-'} "
                        f"blocker_details={';'.join(str(value) for value in readiness.get('blocker_details', [])) or '-'}"
                    )
                )
            )
        if getattr(args, "json", False):
            print(json.dumps({"jobs": filtered_jobs}, indent=2))
        return

    if args.command == "cancel":
        job = queue.cancel(args.job_id)
        print(f"job_id={job.job_id} state={job.state} task_id={job.task_id}")
        return

    runtime = DelegatedRuntimeController(config.delegated_job_runtime_state_path)

    if args.command == "status":
        json_mode = getattr(args, "json", False)
        snapshot = runtime.snapshot(config=config)
        policy = snapshot["policy"]
        leases = snapshot["active_leases"]
        operator_policy = operator_policy_snapshot(config)
        capability_policy = capability_registry_snapshot(config)
        repo_root = Path(__file__).resolve().parents[1]
        jobs_with_readiness = [
            (job, _job_readiness(queue, job, config=config, repo_root=repo_root, bank=runtime_bank))
            for job in queue.list_jobs()
        ]
        jobs_with_readiness.sort(key=lambda item: _job_display_sort_key(item[0], item[1]))
        if not json_mode:
            print(
                "policy "
                f"max_concurrency={policy.get('max_concurrent_jobs', 0)} "
                f"max_active_per_budget_group={policy.get('max_active_jobs_per_budget_group', 0)} "
                f"max_queued_per_budget_group={policy.get('max_queued_jobs_per_budget_group', 0)} "
                f"max_artifact_bytes={policy.get('max_artifact_bytes_per_job', 0)} "
                f"max_subprocesses_per_job={policy.get('max_subprocesses_per_job', 0)} "
                f"max_consecutive_selections_per_budget_group={policy.get('max_consecutive_selections_per_budget_group', 0)} "
                f"provider={policy.get('provider', '')} "
                f"model={policy.get('model_name', '')}"
            )
            print(
                "operator_policy "
                f"allow_git_commands={int(bool(operator_policy.get('unattended_allow_git_commands', False)))} "
                f"allow_http_requests={int(bool(operator_policy.get('unattended_allow_http_requests', False)))} "
                f"http_allowed_hosts={','.join(capability_policy.get('builtin', {}).get('http_request', {}).get('scope', {}).get('allowed_hosts', []))} "
                f"allow_generated_path_mutations={int(bool(operator_policy.get('unattended_allow_generated_path_mutations', False)))} "
                f"rollback_on_failure={int(config.unattended_rollback_on_failure)} "
                f"allowed_benchmark_families={','.join(operator_policy.get('unattended_allowed_benchmark_families', []))}"
            )
            print(
                "capability_policy "
                f"enabled_builtin={','.join(capability_policy.get('enabled_builtin_capabilities', []))} "
                f"enabled_external={','.join(capability_policy.get('enabled_external_capabilities', []))} "
                f"modules_path={capability_policy.get('modules_path', '')}"
            )
        modules = capability_policy.get("modules", [])
        if isinstance(modules, list) and not json_mode:
            for module in modules:
                if not isinstance(module, dict):
                    continue
                print(
                    "capability_module "
                    f"module_id={module.get('module_id', '')} "
                    f"adapter_kind={module.get('adapter_kind', '')} "
                    f"enabled={int(bool(module.get('enabled', False)))} "
                    f"valid={int(bool(module.get('valid', False)))} "
                    f"capabilities={','.join(str(value) for value in module.get('capabilities', []))} "
                    f"issues={';'.join(str(value) for value in module.get('issues', [])) or '-'}"
                )
        trust = build_unattended_trust_ledger(config)
        gated = trust["gated_summary"]
        assessment = trust["overall_assessment"]
        if not json_mode:
            print(
                "trust_policy "
                f"enforce={int(config.unattended_trust_enforce)} "
                f"required_benchmark_families={','.join(config.unattended_trust_required_benchmark_families)} "
                f"recent_report_limit={config.unattended_trust_recent_report_limit} "
                f"bootstrap_min_reports={config.unattended_trust_bootstrap_min_reports}"
            )
            print(
                "trust_status "
                f"status={assessment.get('status', '')} "
                f"reports={gated.get('total', 0)} "
                f"success_rate={float(gated.get('success_rate', 0.0)):.3f} "
                f"unsafe_ambiguous_rate={float(gated.get('unsafe_ambiguous_rate', 0.0)):.3f} "
                f"hidden_side_effect_risk_rate={float(gated.get('hidden_side_effect_risk_rate', 0.0)):.3f}"
            )
        required_families = {
            str(family).strip()
            for family in trust.get("policy", {}).get("required_benchmark_families", [])
            if str(family).strip()
        }
        family_summaries = trust.get("family_summaries", {})
        family_assessments = trust.get("family_assessments", {})
        for family in sorted(set(required_families) | set(family_summaries) | set(family_assessments)):
            summary = family_summaries.get(family, {})
            assessment_entry = family_assessments.get(family, {})
            if not json_mode:
                print(
                    "trust_family "
                    f"family={family} "
                    f"required={int(family in required_families)} "
                    f"status={assessment_entry.get('status', 'absent')} "
                    f"reports={int(summary.get('total', 0))} "
                    f"success_rate={float(summary.get('success_rate', 0.0)):.3f} "
                    f"hidden_side_effect_risk_rate={float(summary.get('hidden_side_effect_risk_rate', 0.0)):.3f}"
                )
        capability_policies = capability_policy.get("capability_policies", {})
        for capability, entries in sorted(capability_policies.items()):
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("source") != "module" or not bool(entry.get("enabled", False)):
                    continue
                if not json_mode:
                    print(
                        "capability_scope "
                        f"capability={capability} "
                        f"module_id={entry.get('module_id', '')} "
                        f"access_tier={entry.get('access_tier', '')} "
                        f"write_tier={entry.get('write_tier', '')} "
                        f"read_only={int(bool(entry.get('read_only', False)))} "
                        f"http_allowed_hosts={','.join(str(value) for value in entry.get('http_allowed_hosts', []))} "
                        f"repo_scopes={','.join(str(value) for value in entry.get('repo_scopes', []))} "
                        f"account_scopes={','.join(str(value) for value in entry.get('account_scopes', []))}"
                    )
        runnable_jobs = sum(1 for _, readiness in jobs_with_readiness if bool(readiness.get("runnable", False)))
        runnable_workers = sum(
            1
            for _, readiness in jobs_with_readiness
            if bool(readiness.get("runnable", False)) and bool(readiness.get("worker_job", False))
        )
        blocked_jobs = sum(1 for _, readiness in jobs_with_readiness if bool(readiness.get("blocked", False)))
        promotable_jobs = sum(1 for _, readiness in jobs_with_readiness if bool(readiness.get("promotable", False)))
        decision_counts: dict[str, int] = {}
        for job, _ in jobs_with_readiness:
            decision = _latest_scheduler_decision(job)
            if not decision:
                continue
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        blocked_open_jobs = sum(1 for job, _ in jobs_with_readiness if bool(getattr(job, "scheduler_blocked_open", False)))
        scheduler_selected_total = sum(int(getattr(job, "scheduler_selected_count", 0)) for job, _ in jobs_with_readiness)
        scheduler_blocked_total = sum(int(getattr(job, "scheduler_blocked_count", 0)) for job, _ in jobs_with_readiness)
        scheduler_unblock_total = sum(int(getattr(job, "scheduler_unblock_count", 0)) for job, _ in jobs_with_readiness)
        oldest_blocked_at = min(
            (
                str(getattr(job, "scheduler_first_blocked_at", "")).strip()
                for job, _ in jobs_with_readiness
                if str(getattr(job, "scheduler_first_blocked_at", "")).strip()
            ),
            default="",
        )
        scheduler = snapshot.get("scheduler", {})
        if not isinstance(scheduler, dict):
            scheduler = {}
        next_runnable = next(
            ((job, readiness) for job, readiness in jobs_with_readiness if bool(readiness.get("runnable", False))),
            None,
        )
        family_rollups = _queue_family_rollups(jobs_with_readiness)
        structural_class_rollups = _queue_structural_class_rollups(jobs_with_readiness)
        active_lease_roles = _active_lease_role_rollup(jobs_with_readiness, leases if isinstance(leases, list) else [])
        role_closeout = _queue_role_closeout(
            jobs_with_readiness,
            leases if isinstance(leases, list) else [],
            trust if isinstance(trust, dict) else {},
        )
        if json_mode:
            payload = {
                "policy": dict(policy) if isinstance(policy, dict) else {},
                "operator_policy": dict(operator_policy) if isinstance(operator_policy, dict) else {},
                "capability_policy": {
                    "enabled_builtin_capabilities": list(capability_policy.get("enabled_builtin_capabilities", [])),
                    "enabled_external_capabilities": list(capability_policy.get("enabled_external_capabilities", [])),
                    "modules_path": str(capability_policy.get("modules_path", "")).strip(),
                    "invalid_enabled_modules": [
                        dict(entry)
                        for entry in capability_policy.get("invalid_enabled_modules", [])
                        if isinstance(entry, dict)
                    ]
                    if isinstance(capability_policy.get("invalid_enabled_modules", []), list)
                    else [],
                    "modules": [
                        dict(module)
                        for module in capability_policy.get("modules", [])
                        if isinstance(module, dict)
                    ]
                    if isinstance(capability_policy.get("modules", []), list)
                    else [],
                    "capability_scopes": _capability_scope_entries(capability_policy),
                },
                "trust": trust,
                "queue": {
                    "totals": {
                        "total_jobs": len(jobs_with_readiness),
                        "runnable_jobs": runnable_jobs,
                        "runnable_workers": runnable_workers,
                        "blocked_jobs": blocked_jobs,
                        "promotable_jobs": promotable_jobs,
                    },
                    "state_counts": _queue_state_counts(jobs_with_readiness),
                    "blocker_counts": _queue_blocker_counts(jobs_with_readiness),
                    "benchmark_families": family_rollups,
                    "discovered_structural_classes": structural_class_rollups,
                    "decision_counts": dict(sorted(decision_counts.items())),
                    "fairness": {
                        "blocked_open_jobs": blocked_open_jobs,
                        "scheduler_selected_total": scheduler_selected_total,
                        "scheduler_blocked_total": scheduler_blocked_total,
                        "scheduler_unblock_total": scheduler_unblock_total,
                        "oldest_blocked_at": oldest_blocked_at,
                    },
                    "scheduler_streak": {
                        "budget_group": str(scheduler.get("last_selected_budget_group", "")).strip(),
                        "consecutive": int(scheduler.get("consecutive_budget_group_selections", 0)),
                    },
                    "budget_groups": dict(snapshot.get("budget_groups", {}))
                    if isinstance(snapshot.get("budget_groups", {}), dict)
                    else {},
                    "active_lease_roles": active_lease_roles,
                    "role_closeout": role_closeout,
                    "active_leases": [
                        dict(lease)
                        for lease in leases
                        if isinstance(lease, dict)
                    ]
                    if isinstance(leases, list)
                    else [],
                    "next_runnable": (
                        _job_payload(next_runnable[0], next_runnable[1]) if next_runnable is not None else None
                    ),
                    "jobs": [_job_payload(job, readiness) for job, readiness in jobs_with_readiness],
                },
            }
            print(json.dumps(payload, indent=2))
            return
        print(
            "queue_status "
            f"total_jobs={len(jobs_with_readiness)} "
            f"runnable_jobs={runnable_jobs} "
            f"runnable_workers={runnable_workers} "
            f"blocked_jobs={blocked_jobs} "
            f"promotable_jobs={promotable_jobs}"
        )
        if decision_counts:
            print(
                "queue_decisions "
                + " ".join(
                    f"{name}={count}"
                    for name, count in sorted(decision_counts.items(), key=lambda item: item[0])
                )
            )
        state_counts = _queue_state_counts(jobs_with_readiness)
        if state_counts:
            print("queue_states " + " ".join(f"{name}={count}" for name, count in state_counts.items()))
        blocker_counts = _queue_blocker_counts(jobs_with_readiness)
        if blocker_counts:
            print("queue_blockers " + " ".join(f"{name}={count}" for name, count in blocker_counts.items()))
        for family, rollup in family_rollups.items():
            print(
                "queue_family "
                f"family={family} "
                f"total_jobs={rollup['total_jobs']} "
                f"runnable_jobs={rollup['runnable_jobs']} "
                f"blocked_jobs={rollup['blocked_jobs']} "
                f"promotable_jobs={rollup['promotable_jobs']} "
                f"worker_jobs={rollup['worker_jobs']} "
                f"integrator_jobs={rollup['integrator_jobs']} "
                f"budget_groups={','.join(rollup['budget_groups']) or '-'}"
            )
        for class_id, rollup in structural_class_rollups.items():
            print(
                "queue_structural_class "
                f"class_id={class_id} "
                f"class_kind={rollup['class_kind'] or '-'} "
                f"total_jobs={rollup['total_jobs']} "
                f"runnable_jobs={rollup['runnable_jobs']} "
                f"blocked_jobs={rollup['blocked_jobs']} "
                f"promotable_jobs={rollup['promotable_jobs']} "
                f"benchmark_families={','.join(rollup['benchmark_families']) or '-'} "
                f"family_aliases={','.join(rollup['family_aliases']) or '-'}"
            )
        print(
            "queue_fairness "
            f"blocked_open_jobs={blocked_open_jobs} "
            f"scheduler_selected_total={scheduler_selected_total} "
            f"scheduler_blocked_total={scheduler_blocked_total} "
            f"scheduler_unblock_total={scheduler_unblock_total} "
            f"oldest_blocked_at={oldest_blocked_at or '-'}"
        )
        print(
            "scheduler_streak "
            f"budget_group={str(scheduler.get('last_selected_budget_group', '')).strip() or '-'} "
            f"consecutive={int(scheduler.get('consecutive_budget_group_selections', 0))}"
        )
        print(
            "role_closeout "
            f"ready={int(bool(role_closeout['closeout_ready']))} "
            f"mode={role_closeout['closeout_mode']} "
            f"total_jobs={int(role_closeout['total_jobs'])} "
            f"completed_success_jobs={int(role_closeout['completed_success_jobs'])} "
            f"unfinished_jobs={int(role_closeout['unfinished_jobs'])} "
            f"terminal_non_success_jobs={int(role_closeout['terminal_non_success_jobs'])} "
            f"blocked_open_jobs={int(role_closeout['blocked_open_jobs'])} "
            f"active_leases={int(role_closeout['active_leases'])} "
            f"trust_status={role_closeout['trust_status']} "
            f"operator_steering_required={int(bool(role_closeout['operator_steering_required']))}"
        )
        print(
            "active_roles "
            f"total={int(active_lease_roles['total'])} "
            f"worker_leases={int(active_lease_roles['worker_jobs'])} "
            f"integrator_leases={int(active_lease_roles['integrator_jobs'])} "
            f"other_leases={int(active_lease_roles['other_jobs'])} "
            f"families={','.join(f'{name}:{count}' for name, count in active_lease_roles['benchmark_families'].items()) or '-'} "
            f"budget_groups={','.join(f'{name}:{count}' for name, count in active_lease_roles['budget_groups'].items()) or '-'} "
            f"shared_repos={','.join(f'{name}:{count}' for name, count in active_lease_roles['shared_repo_ids'].items()) or '-'}"
        )
        if next_runnable is None:
            print("next_runnable no_runnable_job=1")
        else:
            job, readiness = next_runnable
            print(
                "next_runnable "
                f"job_id={job.job_id} task_id={job.task_id} priority={job.priority} "
                f"worker_job={int(bool(readiness.get('worker_job', False)))} "
                f"integrator_job={int(bool(readiness.get('integrator_job', False)))}"
            )
        print(f"active_leases={len(leases)}")
        budget_groups = snapshot.get("budget_groups", {})
        if isinstance(budget_groups, dict) and budget_groups:
            print(
                "budget_groups "
                + " ".join(
                    f"{name}={count}"
                    for name, count in sorted(
                        ((str(name), int(count)) for name, count in budget_groups.items()),
                        key=lambda item: item[0],
                    )
                )
            )
        for lease in leases:
            print(
                f"lease job_id={lease.get('job_id', '')} task_id={lease.get('task_id', '')} "
                f"budget_group={lease.get('budget_group', '')} "
                f"runner={lease.get('runner_id', '')} workspace={lease.get('workspace_path', '')} "
                f"shared_repo_id={lease.get('shared_repo_id', '')} "
                f"worker_branch={lease.get('worker_branch', '')} "
                f"target_branch={lease.get('target_branch', '')} "
                f"claimed_paths={','.join(str(path) for path in lease.get('claimed_paths', []))}"
            )
        return

    repo_root = Path(__file__).resolve().parents[1]
    enforce_preflight = args.enforce_preflight == "1"

    if args.command == "run-next":
        job = run_next_delegated_job(
            queue,
            base_config=config,
            repo_root=repo_root,
            enforce_preflight=enforce_preflight,
            runtime_controller=runtime,
        )
        if job is None:
            print("queue=empty")
            return
        print(f"job_id={job.job_id} state={job.state} task_id={job.task_id} outcome={job.outcome}")
        return

    drained = drain_delegated_jobs(
        queue,
        limit=max(0, args.limit),
        base_config=config,
        repo_root=repo_root,
        enforce_preflight=enforce_preflight,
        runtime_controller=runtime,
    )
    print(f"drained={len(drained)}")
    for job in drained:
        print(f"job_id={job.job_id} state={job.state} task_id={job.task_id} outcome={job.outcome}")


if __name__ == "__main__":
    main()
