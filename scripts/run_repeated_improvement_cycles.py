from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import os
import selectors
import signal
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import subprocess

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner
from agent_kernel.runtime_supervision import (
    atomic_write_json,
    install_termination_handlers,
    spawn_process_group,
    terminate_process_tree,
)
from agent_kernel.trust import build_unattended_trust_ledger

REPEATED_UNSAMPLED_PRIORITY_WEIGHT_STEP = 1.0
REPEATED_UNSAMPLED_PRIORITY_TASK_LIMIT_STEP = 1
_REPEATED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS = 120.0
_REPEATED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS = 240.0
_REPEATED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS = 180.0
_REPEATED_CHILD_GENERATED_FAILURE_ACTIVE_GRACE_SECONDS = 180.0
_REPEATED_CHILD_GENERATED_FAILURE_COMPLETION_GRACE_SECONDS = 180.0
_REPEATED_CHILD_METRICS_FINALIZE_GRACE_SECONDS = 120.0
_REPEATED_CHILD_FINALIZE_GRACE_SECONDS = 120.0


def _status_path(config: KernelConfig) -> Path:
    return config.improvement_reports_dir / "repeated_improvement_status.json"


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


def _runtime_env(config: KernelConfig) -> dict[str, str]:
    return config.to_env()


def _parse_progress_fields(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if not normalized:
        return {}
    parsed: dict[str, object] = {}
    phase_match = re.search(r"phase=(?P<phase>[A-Za-z0-9_:-]+)", normalized)
    if phase_match:
        parsed["last_progress_phase"] = str(phase_match.group("phase")).strip()
    observe_match = re.search(
        r"observe complete passed=(?P<passed>\d+)/(?P<total>\d+) pass_rate=(?P<pass_rate>[0-9.]+) "
        r"generated_pass_rate=(?P<generated_pass_rate>[0-9.]+)",
        normalized,
    )
    if observe_match:
        parsed["observe_summary"] = {
            "passed": int(observe_match.group("passed")),
            "total": int(observe_match.group("total")),
            "pass_rate": float(observe_match.group("pass_rate")),
            "generated_pass_rate": float(observe_match.group("generated_pass_rate")),
        }
        parsed["observe_completed"] = True
    selected_match = re.search(r"campaign \d+/\d+ select subsystem=(?P<subsystem>[a-z_]+)", normalized)
    if selected_match:
        parsed["selected_subsystem"] = str(selected_match.group("subsystem")).strip()
    phase_total_match = re.search(r"phase=(?P<phase>[A-Za-z0-9_:-]+) total=(?P<total>\d+)", normalized)
    if phase_total_match:
        phase_name = str(phase_total_match.group("phase")).strip()
        parsed[f"{phase_name}_total"] = int(phase_total_match.group("total"))
    variant_search_match = re.search(
        r"variant_search start subsystem=(?P<subsystem>[a-z_]+) selected_variants=(?P<count>\d+) "
        r"variant_ids=(?P<variant_ids>[A-Za-z0-9_,.-]+)",
        normalized,
    )
    if variant_search_match:
        parsed["selected_subsystem"] = str(variant_search_match.group("subsystem")).strip()
        parsed["selected_variants"] = int(variant_search_match.group("count"))
        parsed["selected_variant_ids"] = [
            token
            for token in str(variant_search_match.group("variant_ids")).split(",")
            if str(token).strip()
        ]
    generate_match = re.search(
        r"variant generate complete subsystem=(?P<subsystem>[a-z_]+) "
        r"variant=(?P<variant>[A-Za-z0-9_:-]+) artifact=(?P<artifact>\S+)",
        normalized,
    )
    if generate_match:
        parsed["selected_subsystem"] = str(generate_match.group("subsystem")).strip()
        parsed["last_candidate_variant"] = str(generate_match.group("variant")).strip()
        parsed["last_candidate_artifact_path"] = str(generate_match.group("artifact")).strip()
    finalize_match = re.search(
        r"finalize phase=(?P<phase>[A-Za-z0-9_:-]+) subsystem=(?P<subsystem>[a-z_]+)",
        normalized,
    )
    if finalize_match:
        parsed["selected_subsystem"] = str(finalize_match.group("subsystem")).strip()
        parsed["finalize_phase"] = str(finalize_match.group("phase")).strip()
    phase_task_match = re.search(
        r"phase=(?P<phase>[A-Za-z0-9_:-]+) task (?P<index>\d+)/(?P<total>\d+) "
        r"(?P<task_id>\S+) family=(?P<family>[a-z_]+)",
        normalized,
    )
    if phase_task_match:
        phase_name = str(phase_task_match.group("phase")).strip()
        parsed["current_task"] = {
            "index": int(phase_task_match.group("index")),
            "total": int(phase_task_match.group("total")),
            "task_id": str(phase_task_match.group("task_id")).strip(),
            "family": str(phase_task_match.group("family")).strip(),
            "phase": phase_name,
        }
        if phase_name:
            parsed["last_progress_phase"] = phase_name
        if int(phase_task_match.group("index")) >= int(phase_task_match.group("total")):
            parsed[f"{phase_name}_completed"] = True
    else:
        task_match = re.search(
            r"task (?P<index>\d+)/(?P<total>\d+) (?P<task_id>\S+) family=(?P<family>[a-z_]+)",
            normalized,
        )
        if task_match:
            parsed["current_task"] = {
                "index": int(task_match.group("index")),
                "total": int(task_match.group("total")),
                "task_id": str(task_match.group("task_id")).strip(),
                "family": str(task_match.group("family")).strip(),
            }
    return parsed


def _child_runtime_extension_plan(
    *,
    last_progress_phase: str,
    current_task: dict[str, object] | None,
) -> tuple[str, float]:
    normalized_phase = str(last_progress_phase).strip()
    task_payload = current_task if isinstance(current_task, dict) else {}
    task_phase = str(task_payload.get("phase", "")).strip() or normalized_phase
    task_index = int(task_payload.get("index", 0) or 0)
    task_total = int(task_payload.get("total", 0) or 0)
    task_completed = task_total > 0 and task_index >= task_total
    if task_phase == "generated_success" and task_completed:
        return (
            "generated_success_completion",
            _REPEATED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS,
        )
    if task_phase == "generated_failure_seed":
        if task_completed:
            return (
                "generated_failure_seed_completion",
                _REPEATED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS,
            )
        return (
            "generated_failure_seed_active",
            _REPEATED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS,
        )
    if task_phase == "generated_failure":
        if task_completed:
            return (
                "generated_failure_completion",
                _REPEATED_CHILD_GENERATED_FAILURE_COMPLETION_GRACE_SECONDS,
            )
        return (
            "generated_failure_active",
            _REPEATED_CHILD_GENERATED_FAILURE_ACTIVE_GRACE_SECONDS,
        )
    if normalized_phase == "metrics_finalize":
        return ("metrics_finalize", _REPEATED_CHILD_METRICS_FINALIZE_GRACE_SECONDS)
    if normalized_phase == "finalize":
        return ("finalize", _REPEATED_CHILD_FINALIZE_GRACE_SECONDS)
    return ("", 0.0)


def _progress_sampled_families(
    active_cycle_run: dict[str, object] | None,
    priority_benchmark_families: list[str],
) -> list[str]:
    if not isinstance(active_cycle_run, dict):
        return []
    current_task = active_cycle_run.get("current_task", {})
    if not isinstance(current_task, dict):
        return []
    family = str(current_task.get("family", "")).strip()
    if not family:
        return []
    if priority_benchmark_families:
        return [family] if family in set(priority_benchmark_families) else []
    return [family]


def _partial_progress_summary(
    active_cycle_run: dict[str, object] | None,
    *,
    priority_benchmark_families: list[str],
) -> dict[str, object]:
    if not isinstance(active_cycle_run, dict):
        return {}
    observe_summary = active_cycle_run.get("observe_summary", {})
    if not isinstance(observe_summary, dict):
        observe_summary = {}
    sampled_families = _ordered_unique_strings(
        active_cycle_run.get("sampled_families_from_progress", []),
        _progress_sampled_families(active_cycle_run, priority_benchmark_families),
    )
    last_progress_phase = str(active_cycle_run.get("last_progress_phase", "")).strip()
    generated_success_total = int(active_cycle_run.get("generated_success_total", 0) or 0)
    generated_success_completed = bool(active_cycle_run.get("generated_success_completed", False))
    current_task = active_cycle_run.get("current_task", {})
    if not isinstance(current_task, dict):
        current_task = {}
    if (
        not generated_success_completed
        and str(current_task.get("phase", "")).strip() == "generated_success"
        and int(current_task.get("index", 0) or 0) > 0
        and int(current_task.get("index", 0) or 0) >= int(current_task.get("total", 0) or 0)
    ):
        generated_success_completed = True
    generated_success_started = (
        generated_success_total > 0
        or str(last_progress_phase).startswith("generated_success")
        or str(current_task.get("phase", "")).strip() == "generated_success"
    )
    observe_completed = (
        bool(active_cycle_run.get("observe_completed", False))
        or int(observe_summary.get("total", 0) or 0) > 0
        or generated_success_started
    )
    partial_productive = observe_completed or generated_success_started or bool(sampled_families)
    priority_unsampled = [
        family for family in priority_benchmark_families if family not in set(sampled_families)
    ]
    return {
        "observe_completed": observe_completed,
        "observe_summary": observe_summary,
        "generated_success_started": generated_success_started,
        "generated_success_total": generated_success_total,
        "generated_success_completed": generated_success_completed,
        "last_progress_phase": last_progress_phase,
        "sampled_families_from_progress": sampled_families,
        "priority_families_without_progress_sampling": priority_unsampled,
        "candidate_generated": bool(str(active_cycle_run.get("last_candidate_artifact_path", "")).strip()),
        "productive_partial": partial_productive,
    }


def _append_partial_run_from_active_progress(
    *,
    runs: list[dict[str, object]],
    active_cycle_run: dict[str, object] | None,
    priority_benchmark_families: list[str],
    returncode: int,
    stderr: str,
    timed_out: bool = False,
    timeout_reason: str = "",
    interrupted: bool = False,
) -> dict[str, object] | None:
    progress = _partial_progress_summary(
        active_cycle_run,
        priority_benchmark_families=priority_benchmark_families,
    )
    if not bool(progress.get("productive_partial", False)):
        return None
    current_run = active_cycle_run if isinstance(active_cycle_run, dict) else {}
    synthetic_run = {
        "index": int(current_run.get("index", len(runs) + 1) or (len(runs) + 1)),
        "returncode": int(returncode),
        "stdout": "",
        "stderr": str(stderr).strip(),
        "timed_out": bool(timed_out),
        "timeout_reason": str(timeout_reason).strip(),
        "priority_benchmark_families": list(priority_benchmark_families),
        "priority_benchmark_family_weights": dict(current_run.get("priority_benchmark_family_weights", {}))
        if isinstance(current_run.get("priority_benchmark_family_weights", {}), dict)
        else {},
        "partial_progress": progress,
        "partial_productive": bool(progress.get("productive_partial", False)),
        "partial_candidate_generated": bool(progress.get("candidate_generated", False)),
        "productive": bool(progress.get("productive_partial", False)),
        "retained_gain": False,
        "campaign_cycle_ids": [],
        "runtime_managed_decisions": 0,
        "runtime_managed_retained_cycles": 0,
        "runtime_managed_rejected_cycles": 0,
        "decision_conversion_state": "partial_productive_without_decision",
    }
    if interrupted:
        synthetic_run["interrupted"] = True
    runs.append(synthetic_run)
    return synthetic_run


def _priority_benchmark_family_weights(values: list[str] | None) -> dict[str, float]:
    if not values:
        return {}
    weights: dict[str, float] = {}
    for value in values:
        token = str(value).strip()
        if "=" not in token:
            continue
        family, raw_weight = token.split("=", 1)
        family = family.strip()
        try:
            weight = float(raw_weight)
        except ValueError:
            continue
        if family and weight > 0.0:
            weights[family] = weight
    return weights


def _reroute_priority_families_for_unsampled_pressure(
    priority_families: list[str],
    priority_family_weights: dict[str, float],
    unsampled_priority_families: list[str],
) -> dict[str, object]:
    normalized_families = [str(value).strip() for value in priority_families if str(value).strip()]
    normalized_unsampled = [
        family for family in [str(value).strip() for value in unsampled_priority_families if str(value).strip()]
        if family in set(normalized_families)
    ]
    if not normalized_families or not normalized_unsampled:
        return {
            "applied": False,
            "reason": "no_unsampled_priority_families",
            "priority_benchmark_families": normalized_families,
            "priority_benchmark_family_weights": dict(priority_family_weights),
            "weight_bonus_by_family": {},
            "unsampled_priority_families": [],
        }
    reordered_families = _ordered_unique_strings(normalized_unsampled, normalized_families)
    effective_weights = {
        family: float(priority_family_weights.get(family, 0.0) or 0.0)
        for family in reordered_families
    }
    max_existing_weight = max([weight for weight in effective_weights.values() if weight > 0.0], default=1.0)
    rerouted_weights = dict(effective_weights)
    weight_bonus_by_family: dict[str, float] = {}
    for index, family in enumerate(normalized_unsampled):
        target_weight = max_existing_weight + (
            REPEATED_UNSAMPLED_PRIORITY_WEIGHT_STEP * float(len(normalized_unsampled) - index)
        )
        original_weight = rerouted_weights.get(family, 0.0)
        rerouted_weights[family] = max(original_weight, round(target_weight, 6))
        weight_bonus = round(rerouted_weights[family] - original_weight, 6)
        if weight_bonus > 0.0:
            weight_bonus_by_family[family] = weight_bonus
    return {
        "applied": True,
        "reason": "unsampled_priority_families",
        "priority_benchmark_families": reordered_families,
        "priority_benchmark_family_weights": rerouted_weights,
        "weight_bonus_by_family": weight_bonus_by_family,
        "unsampled_priority_families": normalized_unsampled,
    }


def _reroute_task_budget_for_unsampled_pressure(
    *,
    current_task_limit: int,
    max_task_limit: int,
    priority_families: list[str],
    unsampled_priority_families: list[str],
) -> dict[str, object]:
    normalized_families = [str(value).strip() for value in priority_families if str(value).strip()]
    normalized_unsampled = [
        family for family in [str(value).strip() for value in unsampled_priority_families if str(value).strip()]
        if family in set(normalized_families)
    ]
    bounded_current_task_limit = max(0, int(current_task_limit))
    bounded_max_task_limit = max(0, int(max_task_limit))
    minimum_family_task_limit = min(bounded_max_task_limit, len(normalized_families)) if bounded_max_task_limit > 0 else 0
    if (
        bounded_current_task_limit <= 0
        or bounded_max_task_limit <= bounded_current_task_limit
        or not normalized_families
        or not normalized_unsampled
    ):
        return {
            "applied": False,
            "reason": "no_unsampled_priority_budget_reroute",
            "current_task_limit": bounded_current_task_limit,
            "next_task_limit": bounded_current_task_limit,
            "task_limit_delta": 0,
            "minimum_family_task_limit": minimum_family_task_limit,
            "unsampled_priority_families": normalized_unsampled,
        }
    target_task_limit = min(
        bounded_max_task_limit,
        max(
            bounded_current_task_limit + (REPEATED_UNSAMPLED_PRIORITY_TASK_LIMIT_STEP * len(normalized_unsampled)),
            minimum_family_task_limit,
        ),
    )
    return {
        "applied": target_task_limit > bounded_current_task_limit,
        "reason": "unsampled_priority_task_budget" if target_task_limit > bounded_current_task_limit else "task_limit_already_sufficient",
        "current_task_limit": bounded_current_task_limit,
        "next_task_limit": target_task_limit,
        "task_limit_delta": max(0, target_task_limit - bounded_current_task_limit),
        "minimum_family_task_limit": minimum_family_task_limit,
        "unsampled_priority_families": normalized_unsampled,
    }


def _run_and_stream(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    progress_label: str | None = None,
    heartbeat_interval_seconds: float = 60.0,
    max_silence_seconds: float = 0.0,
    max_runtime_seconds: float = 0.0,
    max_progress_stall_seconds: float = 0.0,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    def _timeout_result(*, reason: str, details: dict[str, object]) -> dict[str, object]:
        timeout_line = f"[repeated] child={progress_label or 'run_improvement_cycle'} timeout reason={reason}"
        print(timeout_line, file=sys.stderr, flush=True)
        _emit_event(
            {
                "event": "output",
                "line": timeout_line,
                "pid": process_pid,
                "progress_label": progress_label or "run_improvement_cycle",
                "timestamp": time.time(),
            }
        )
        _emit_event(
            {
                "event": "timeout",
                "pid": process_pid,
                "progress_label": progress_label or "run_improvement_cycle",
                "timestamp": time.time(),
                "timeout_reason": reason,
                **details,
            }
        )
        return {
            "returncode": -9,
            "stdout": "".join(completed_output).strip(),
            "stderr": timeout_line,
            "timed_out": True,
            "timeout_reason": reason,
        }

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
            "progress_label": progress_label or "run_improvement_cycle",
            "started_at": time.time(),
        }
    )
    assert process.stdout is not None
    if not hasattr(process.stdout, "fileno"):
        for line in process.stdout:
            completed_output.append(line)
            print(line, end="", file=sys.stderr, flush=True)
            _emit_event(
                {
                    "event": "output",
                    "line": line.rstrip("\n"),
                    "pid": process_pid,
                    "progress_label": progress_label or "run_improvement_cycle",
                    "timestamp": time.time(),
                }
            )
        returncode = process.wait()
        _emit_event(
            {
                "event": "exit",
                "pid": process_pid,
                "progress_label": progress_label or "run_improvement_cycle",
                "returncode": returncode,
                "timestamp": time.time(),
            }
        )
        return {
            "returncode": returncode,
            "stdout": "".join(completed_output).strip(),
            "stderr": "",
            "timed_out": False,
        }
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
    current_task: dict[str, object] = {}
    runtime_grace_keys: set[str] = set()
    runtime_deadline = started_at + max_runtime if max_runtime > 0.0 else 0.0
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
                    if "[cycle:" in line or "[eval:" in line or "[repeated]" in line or "finalize phase=" in line:
                        last_progress_at = now
                        parsed_progress = _parse_progress_fields(line)
                        phase_name = str(parsed_progress.get("last_progress_phase", "")).strip()
                        if phase_name:
                            last_progress_phase = phase_name
                        parsed_task = parsed_progress.get("current_task", {})
                        if isinstance(parsed_task, dict) and parsed_task:
                            current_task = dict(parsed_task)
                    print(line, end="", file=sys.stderr, flush=True)
                    _emit_event(
                        {
                            "event": "output",
                            "line": line.rstrip("\n"),
                            "pid": process_pid,
                            "progress_label": progress_label or "run_improvement_cycle",
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
                print(
                    f"[repeated] child={progress_label or 'run_improvement_cycle'} still_running silence={int(silence)}s",
                    file=sys.stderr,
                    flush=True,
                )
                _emit_event(
                    {
                        "event": "heartbeat",
                        "pid": process_pid,
                        "progress_label": progress_label or "run_improvement_cycle",
                        "silence_seconds": int(silence),
                        "timestamp": time.time(),
                    }
                )
                last_heartbeat_at = now
            if runtime_deadline > 0.0 and now >= runtime_deadline:
                grace_key, extension_seconds = _child_runtime_extension_plan(
                    last_progress_phase=last_progress_phase,
                    current_task=current_task,
                )
                if grace_key and extension_seconds > 0.0 and grace_key not in runtime_grace_keys:
                    runtime_grace_keys.add(grace_key)
                    runtime_deadline = now + extension_seconds
                    _emit_event(
                        {
                            "event": "runtime_grace",
                            "pid": process_pid,
                            "progress_label": progress_label or "run_improvement_cycle",
                            "timestamp": time.time(),
                            "runtime_seconds": int(runtime_elapsed),
                            "grace_seconds": int(extension_seconds),
                            "grace_key": grace_key,
                            "last_progress_phase": last_progress_phase,
                            "current_task": dict(current_task),
                        }
                    )
                    continue
                terminate_process_tree(process)
                return _timeout_result(
                    reason=f"child exceeded max runtime of {int(max_runtime)} seconds",
                    details={
                        "runtime_seconds": int(runtime_elapsed),
                        "last_progress_phase": last_progress_phase,
                    },
                )
            if max_silence > 0.0 and silence >= max_silence:
                terminate_process_tree(process)
                return _timeout_result(
                    reason=f"child exceeded max silence of {int(max_silence)} seconds",
                    details={"silence_seconds": int(silence)},
                )
            if max_progress_stall > 0.0 and progress_stall >= max_progress_stall:
                terminate_process_tree(process)
                return _timeout_result(
                    reason=f"child exceeded max progress stall of {int(max_progress_stall)} seconds",
                    details={"progress_stall_seconds": int(progress_stall)},
                )
        returncode = process.wait()
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        selector.close()
    _emit_event(
        {
            "event": "exit",
            "pid": process_pid,
            "progress_label": progress_label or "run_improvement_cycle",
            "returncode": returncode,
            "timestamp": time.time(),
        }
    )
    return {
        "returncode": returncode,
        "stdout": "".join(completed_output).strip(),
        "stderr": "",
        "timed_out": False,
    }


def _is_runtime_managed_artifact_path(path: str) -> bool:
    normalized = str(path).strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    return not (lowered.startswith("/tmp/") or "pytest-" in lowered or "/tests/" in lowered)


def _reconcile_incomplete_cycles(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    progress_label: str | None,
) -> list[dict[str, object]]:
    reconciled: list[dict[str, object]] = []
    for summary in planner.incomplete_cycle_summaries(config.improvement_cycles_path, protocol="autonomous"):
        cycle_id = str(summary.get("cycle_id", "")).strip()
        subsystem = str(summary.get("subsystem", "")).strip()
        if not cycle_id or not subsystem:
            continue
        active_artifact_path = (
            str(summary.get("active_artifact_path", "")).strip()
            or str(summary.get("artifact_path", "")).strip()
        )
        candidate_artifact_path = str(summary.get("candidate_artifact_path", "")).strip()
        artifact_kind = str(summary.get("artifact_kind", "")).strip() or "retention_decision"
        reason = (
            "cycle ended before retention finalization; recorded fail-closed rejection "
            "for stale incomplete autonomous cycle"
        )
        metrics_summary = {
            "protocol": "autonomous",
            "protocol_match_id": str(summary.get("protocol_match_id", "")).strip(),
            "incomplete_cycle": True,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
            "last_state": str(summary.get("last_state", "")).strip(),
            "last_action": str(summary.get("last_action", "")).strip(),
            "selected_variant_id": str(summary.get("selected_variant_id", "")).strip(),
            "record_count": int(summary.get("record_count", 0) or 0),
            "selected_cycles": int(summary.get("selected_cycles", 0) or 0),
        }
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem=subsystem,
                action="finalize_cycle",
                artifact_path=active_artifact_path,
                artifact_kind=artifact_kind,
                reason=reason,
                metrics_summary=metrics_summary,
                candidate_artifact_path=candidate_artifact_path,
                active_artifact_path=active_artifact_path,
            ),
            govern_exports=False,
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="record",
                subsystem=subsystem,
                action="persist_retention_outcome",
                artifact_path=active_artifact_path,
                artifact_kind=artifact_kind,
                reason="persisted fail-closed outcome for incomplete autonomous cycle",
                metrics_summary=metrics_summary,
                candidate_artifact_path=candidate_artifact_path,
                active_artifact_path=active_artifact_path,
            ),
            govern_exports=False,
        )
        reconciled.append(summary)
        print(
            f"[repeated] reconciled incomplete cycle cycle_id={cycle_id} subsystem={subsystem} state=reject",
            file=sys.stderr,
            flush=True,
        )
    return reconciled


def _record_metrics_summary(record: dict[str, object]) -> dict[str, object]:
    metrics_summary = record.get("metrics_summary", {})
    return metrics_summary if isinstance(metrics_summary, dict) else {}


def _record_protocol(record: dict[str, object]) -> str:
    return str(_record_metrics_summary(record).get("protocol", "")).strip()


def _record_protocol_match_id(record: dict[str, object]) -> str:
    return str(_record_metrics_summary(record).get("protocol_match_id", "")).strip()


def _campaign_records(
    records: list[dict[str, object]],
    *,
    campaign_match_id: str,
    start_index: int = 0,
) -> list[dict[str, object]]:
    scoped = records[max(0, start_index) :]
    if campaign_match_id:
        matching_cycle_ids = {
            str(record.get("cycle_id", "")).strip()
            for record in scoped
            if _record_protocol(record) == "autonomous"
            and _record_protocol_match_id(record) == campaign_match_id
            and str(record.get("cycle_id", "")).strip()
        }
        scoped = [
            record
            for record in scoped
            if (
                _record_protocol(record) == "autonomous"
                and _record_protocol_match_id(record) == campaign_match_id
            )
            or (
                str(record.get("cycle_id", "")).strip() in matching_cycle_ids
                and str(record.get("state", "")).strip()
                in {"evaluate", "select", "retain", "reject", "record"}
            )
        ]
    return scoped


def _production_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if str(record.get("state", "")) in {"retain", "reject"}
        and _is_runtime_managed_artifact_path(str(record.get("artifact_path", "")))
    ]


def _count_decisions(records: list[dict[str, object]]) -> dict[str, int]:
    retained = 0
    rejected = 0
    for record in records:
        state = str(record.get("state", "")).strip()
        if state == "retain":
            retained += 1
        elif state == "reject":
            rejected += 1
    return {
        "retained_cycles": retained,
        "rejected_cycles": rejected,
        "total_decisions": retained + rejected,
    }


def _non_runtime_managed_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if str(record.get("state", "")) in {"retain", "reject"}
        and not _is_runtime_managed_artifact_path(str(record.get("artifact_path", "")))
    ]


def _cycle_report_safe_id(cycle_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(cycle_id).strip()).strip("._") or "cycle"


def _load_cycle_report(config: KernelConfig, cycle_id: str) -> dict[str, object] | None:
    report_path = config.improvement_reports_dir / f"cycle_report_{_cycle_report_safe_id(cycle_id)}.json"
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _report_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _decision_record_from_cycle_report(report: dict[str, object]) -> dict[str, object] | None:
    current_cycle_records = report.get("current_cycle_records", [])
    if isinstance(current_cycle_records, list):
        for record in reversed(current_cycle_records):
            if not isinstance(record, dict):
                continue
            if str(record.get("state", "")).strip() not in {"retain", "reject"}:
                continue
            if not _is_runtime_managed_artifact_path(str(record.get("artifact_path", ""))):
                continue
            return dict(record)
    final_state = str(report.get("final_state", "")).strip()
    if final_state not in {"retain", "reject"}:
        return None
    artifact_path = str(report.get("active_artifact_path", report.get("artifact_path", ""))).strip()
    if not _is_runtime_managed_artifact_path(artifact_path):
        return None
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "cycle_id": str(report.get("cycle_id", "")).strip(),
        "state": final_state,
        "subsystem": str(report.get("subsystem", "")).strip(),
        "artifact_path": artifact_path,
        "artifact_kind": str(report.get("artifact_kind", "")).strip() or "retention_decision",
        "reason": str(report.get("final_reason", "")).strip(),
        "candidate_artifact_path": str(report.get("candidate_artifact_path", "")).strip(),
        "active_artifact_path": artifact_path,
        "metrics_summary": {
            "baseline_pass_rate": _report_float(report.get("baseline_metrics", {}).get("pass_rate", 0.0))
            if isinstance(report.get("baseline_metrics", {}), dict)
            else 0.0,
            "candidate_pass_rate": _report_float(report.get("candidate_metrics", {}).get("pass_rate", 0.0))
            if isinstance(report.get("candidate_metrics", {}), dict)
            else 0.0,
            "baseline_average_steps": _report_float(report.get("baseline_metrics", {}).get("average_steps", 0.0))
            if isinstance(report.get("baseline_metrics", {}), dict)
            else 0.0,
            "candidate_average_steps": _report_float(report.get("candidate_metrics", {}).get("average_steps", 0.0))
            if isinstance(report.get("candidate_metrics", {}), dict)
            else 0.0,
            "preview_reason_code": str(report.get("preview_reason_code", "")).strip(),
            "decision_reason_code": str(report.get("decision_reason_code", "")).strip(),
            **evidence,
        },
    }


def _cycle_audit_summary_from_report(report: dict[str, object]) -> dict[str, object] | None:
    decision_record = _decision_record_from_cycle_report(report)
    if not isinstance(decision_record, dict):
        return None
    metrics_summary = decision_record.get("metrics_summary", {})
    if not isinstance(metrics_summary, dict):
        metrics_summary = {}
    return {
        "cycle_id": str(report.get("cycle_id", "")).strip(),
        "subsystem": str(report.get("subsystem", "")).strip(),
        "selected_variant_id": "",
        "prior_retained_cycle_id": "",
        "candidate_artifact_path": str(report.get("candidate_artifact_path", "")).strip(),
        "active_artifact_path": str(report.get("active_artifact_path", report.get("artifact_path", ""))).strip(),
        "final_state": str(decision_record.get("state", "")).strip(),
        "final_reason": str(decision_record.get("reason", "")).strip(),
        "preview_reason_code": str(metrics_summary.get("preview_reason_code", "")).strip(),
        "decision_reason_code": str(metrics_summary.get("decision_reason_code", "")).strip(),
        "baseline_pass_rate": _report_float(metrics_summary.get("baseline_pass_rate", 0.0)),
        "candidate_pass_rate": _report_float(metrics_summary.get("candidate_pass_rate", 0.0)),
        "baseline_average_steps": _report_float(metrics_summary.get("baseline_average_steps", 0.0)),
        "candidate_average_steps": _report_float(metrics_summary.get("candidate_average_steps", 0.0)),
        "phase_gate_passed": metrics_summary.get("phase_gate_passed"),
        "artifact_kind": str(decision_record.get("artifact_kind", "")).strip(),
        "artifact_path": str(decision_record.get("artifact_path", "")).strip(),
        "artifact_lifecycle_state": str(report.get("artifact_lifecycle_state", "")).strip(),
        "artifact_sha256": str(report.get("artifact_sha256", "")).strip(),
        "previous_artifact_sha256": str(report.get("previous_artifact_sha256", "")).strip(),
        "rollback_artifact_path": str(report.get("rollback_artifact_path", "")).strip(),
        "artifact_snapshot_path": str(report.get("artifact_snapshot_path", "")).strip(),
    }


def _recovered_runtime_managed_decisions_from_reports(
    config: KernelConfig,
    cycle_ids: list[str],
) -> list[dict[str, object]]:
    recovered: list[dict[str, object]] = []
    for cycle_id in cycle_ids:
        report = _load_cycle_report(config, cycle_id)
        if not isinstance(report, dict):
            continue
        decision_record = _decision_record_from_cycle_report(report)
        if isinstance(decision_record, dict):
            recovered.append(decision_record)
    return recovered


def _merge_decision_records(
    primary: list[dict[str, object]],
    fallback: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}

    def _decision_record_richness(record: dict[str, object]) -> tuple[int, int]:
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            metrics_summary = {}
        family_pass_rate_delta = metrics_summary.get("family_pass_rate_delta", {})
        generated_family_pass_rate_delta = metrics_summary.get("generated_family_pass_rate_delta", {})
        family_signal_count = len(family_pass_rate_delta) if isinstance(family_pass_rate_delta, dict) else 0
        generated_signal_count = len(generated_family_pass_rate_delta) if isinstance(generated_family_pass_rate_delta, dict) else 0
        return (
            family_signal_count + generated_signal_count,
            len(metrics_summary),
        )

    for record in [*primary, *fallback]:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("cycle_id", "")).strip(),
            str(record.get("state", "")).strip(),
            str(record.get("artifact_path", "")).strip(),
        )
        current = merged.get(key)
        if current is None or _decision_record_richness(record) > _decision_record_richness(current):
            merged[key] = record
    return list(merged.values())


def _candidate_isolation_summary(
    decision_records: list[dict[str, object]],
    generate_index: dict[str, dict[str, object]],
) -> dict[str, object]:
    decisions_with_candidate_path = 0
    decisions_with_active_path = 0
    distinct_paths = 0
    runtime_managed_distinct_paths = 0
    runtime_managed_same_paths = 0
    missing_path_audit_cycle_ids: list[str] = []
    for record in decision_records:
        cycle_id = str(record.get("cycle_id", "")).strip()
        generate_record = generate_index.get(cycle_id, {})
        candidate_path = str(generate_record.get("candidate_artifact_path", "")).strip()
        active_path = str(generate_record.get("active_artifact_path", "")).strip()
        runtime_managed = _is_runtime_managed_artifact_path(str(record.get("artifact_path", "")))
        if candidate_path:
            decisions_with_candidate_path += 1
        if active_path:
            decisions_with_active_path += 1
        if not candidate_path or not active_path:
            if cycle_id:
                missing_path_audit_cycle_ids.append(cycle_id)
            continue
        if candidate_path != active_path:
            distinct_paths += 1
            if runtime_managed:
                runtime_managed_distinct_paths += 1
        elif runtime_managed:
            runtime_managed_same_paths += 1
    return {
        "decision_count": len(decision_records),
        "decisions_with_candidate_path": decisions_with_candidate_path,
        "decisions_with_active_path": decisions_with_active_path,
        "decisions_with_distinct_candidate_and_active_paths": distinct_paths,
        "runtime_managed_distinct_candidate_and_active_paths": runtime_managed_distinct_paths,
        "runtime_managed_same_candidate_and_active_paths": runtime_managed_same_paths,
        "missing_path_audit_decisions": len(missing_path_audit_cycle_ids),
        "missing_path_audit_cycle_ids": missing_path_audit_cycle_ids[:10],
    }


def _yield_summary_for(
    records: list[dict[str, object]],
    generate_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    retained = [record for record in records if str(record.get("state", "")) == "retain"]
    rejected = [record for record in records if str(record.get("state", "")) == "reject"]
    retained_by_subsystem: dict[str, int] = {}
    rejected_by_subsystem: dict[str, int] = {}
    resolved_generate_index = generate_index or {}
    for record in retained:
        key = str(record.get("subsystem", "unknown"))
        retained_by_subsystem[key] = retained_by_subsystem.get(key, 0) + 1
    for record in rejected:
        key = str(record.get("subsystem", "unknown"))
        rejected_by_subsystem[key] = rejected_by_subsystem.get(key, 0) + 1

    def _average_delta(rows: list[dict[str, object]], *, baseline_key: str, candidate_key: str) -> float:
        deltas: list[float] = []
        for row in rows:
            metrics = row.get("metrics_summary", {})
            if not isinstance(metrics, dict):
                continue
            try:
                baseline = float(metrics.get(baseline_key, 0.0))
                candidate = float(metrics.get(candidate_key, 0.0))
            except (TypeError, ValueError):
                continue
            deltas.append(candidate - baseline)
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def _metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            metrics = row.get("metrics_summary", {})
            if not isinstance(metrics, dict) or key not in metrics:
                continue
            try:
                values.append(float(metrics.get(key, 0.0)))
            except (TypeError, ValueError):
                continue
        return values

    def _worst_metric(rows: list[dict[str, object]], key: str) -> float:
        values = _metric_values(rows, key)
        if not values:
            return 0.0
        return min(values)

    def _average_estimated_cost(rows: list[dict[str, object]]) -> float:
        values: list[float] = []
        for row in rows:
            cycle_id = str(row.get("cycle_id", "")).strip()
            generate_record = resolved_generate_index.get(cycle_id, {})
            metrics = generate_record.get("metrics_summary", {})
            if not isinstance(metrics, dict):
                continue
            selected_variant = metrics.get("selected_variant", {})
            estimated_cost = None
            if isinstance(selected_variant, dict):
                estimated_cost = selected_variant.get("estimated_cost")
            if estimated_cost is None:
                estimated_cost = metrics.get("selected_experiment_estimated_cost")
            try:
                values.append(float(estimated_cost))
            except (TypeError, ValueError):
                continue
        if not values:
            return 0.0
        return sum(values) / len(values)

    return {
        "retained_cycles": len(retained),
        "rejected_cycles": len(rejected),
        "total_decisions": len(records),
        "retained_by_subsystem": retained_by_subsystem,
        "rejected_by_subsystem": rejected_by_subsystem,
        "average_retained_pass_rate_delta": _average_delta(retained, baseline_key="baseline_pass_rate", candidate_key="candidate_pass_rate"),
        "average_retained_step_delta": _average_delta(retained, baseline_key="baseline_average_steps", candidate_key="candidate_average_steps"),
        "average_rejected_pass_rate_delta": _average_delta(rejected, baseline_key="baseline_pass_rate", candidate_key="candidate_pass_rate"),
        "average_rejected_step_delta": _average_delta(rejected, baseline_key="baseline_average_steps", candidate_key="candidate_average_steps"),
        "average_retained_estimated_cost": _average_estimated_cost(retained),
        "average_rejected_estimated_cost": _average_estimated_cost(rejected),
        "worst_family_delta": _worst_metric(records, "worst_family_delta"),
        "worst_generated_family_delta": _worst_metric(records, "generated_worst_family_delta"),
        "worst_failure_recovery_delta": _worst_metric(records, "failure_recovery_pass_rate_delta"),
    }


def _partial_progress_rollup(runs: list[dict[str, object]]) -> dict[str, object]:
    observed_primary_runs = 0
    generated_success_started_runs = 0
    generated_success_completed_runs = 0
    candidate_generated_runs = 0
    sampled_families: list[str] = []
    for run in runs:
        progress = run.get("partial_progress", {})
        if not isinstance(progress, dict):
            continue
        if bool(progress.get("observe_completed", False)):
            observed_primary_runs += 1
        if bool(progress.get("generated_success_started", False)):
            generated_success_started_runs += 1
        if bool(progress.get("generated_success_completed", False)):
            generated_success_completed_runs += 1
        if bool(progress.get("candidate_generated", False)):
            candidate_generated_runs += 1
        sampled_families = _ordered_unique_strings(sampled_families, progress.get("sampled_families_from_progress", []))
    return {
        "partial_productive_runs": sum(1 for run in runs if bool(run.get("partial_productive", False))),
        "partial_candidate_runs": sum(1 for run in runs if bool(run.get("partial_candidate_generated", False))),
        "observed_primary_runs": observed_primary_runs,
        "generated_success_started_runs": generated_success_started_runs,
        "generated_success_completed_runs": generated_success_completed_runs,
        "sampled_families_from_progress": sampled_families,
        "recent_partial_progress": [
            run.get("partial_progress", {})
            for run in runs[-3:]
            if isinstance(run.get("partial_progress", {}), dict) and run.get("partial_progress", {})
        ],
    }


def _estimated_cost_for_record(
    record: dict[str, object],
    generate_index: dict[str, dict[str, object]] | None = None,
) -> float:
    resolved_generate_index = generate_index or {}
    cycle_id = str(record.get("cycle_id", "")).strip()
    generate_record = resolved_generate_index.get(cycle_id, {})
    metrics = generate_record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return 0.0
    selected_variant = metrics.get("selected_variant", {})
    estimated_cost = None
    if isinstance(selected_variant, dict):
        estimated_cost = selected_variant.get("estimated_cost")
    if estimated_cost is None:
        estimated_cost = metrics.get("selected_experiment_estimated_cost")
    try:
        return float(estimated_cost)
    except (TypeError, ValueError):
        return 0.0


def _planner_pressure_summary(records: list[dict[str, object]]) -> dict[str, object]:
    campaign_pressures: list[dict[str, object]] = []
    variant_pressures: list[dict[str, object]] = []
    for record in records:
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            continue
        cycle_id = str(record.get("cycle_id", "")).strip()
        subsystem = str(record.get("subsystem", "")).strip()
        campaign_pressure = float(metrics_summary.get("campaign_breadth_pressure", 0.0) or 0.0)
        variant_pressure = float(metrics_summary.get("selected_variant_breadth_pressure", 0.0) or 0.0)
        selected_variant_id = str(metrics_summary.get("selected_variant_id", "")).strip()
        if campaign_pressure > 0.0:
            campaign_pressures.append(
                {
                    "cycle_id": cycle_id,
                    "subsystem": subsystem,
                    "campaign_breadth_pressure": campaign_pressure,
                }
            )
        if variant_pressure > 0.0:
            variant_pressures.append(
                {
                    "cycle_id": cycle_id,
                    "subsystem": subsystem,
                    "variant_id": selected_variant_id,
                    "selected_variant_breadth_pressure": variant_pressure,
                }
            )
    return {
        "campaign_breadth_pressure_cycles": len(campaign_pressures),
        "variant_breadth_pressure_cycles": len(variant_pressures),
        "recent_campaign_pressures": campaign_pressures[-10:],
        "recent_variant_pressures": variant_pressures[-10:],
    }


def _phase_gate_summary_for(records: list[dict[str, object]]) -> dict[str, object]:
    decisions = [record for record in records if str(record.get("state", "")) in {"retain", "reject"}]
    checked: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    retained_failed: list[dict[str, object]] = []
    for record in decisions:
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict) or "phase_gate_passed" not in metrics:
            continue
        checked.append(record)
        if not bool(metrics.get("phase_gate_passed", False)):
            failed.append(record)
            if str(record.get("state", "")) == "retain":
                retained_failed.append(record)
    return {
        "decision_count": len(decisions),
        "checked_decisions": len(checked),
        "failed_decisions": len(failed),
        "retained_failed_decisions": len(retained_failed),
        "all_checked_phase_gates_passed": bool(checked) and not failed,
        "all_retained_phase_gates_passed": not retained_failed,
    }


def _priority_family_yield_summary(
    records: list[dict[str, object]],
    priority_families: list[str],
    generate_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    normalized_families = [str(value).strip() for value in priority_families if str(value).strip()]
    family_summaries: dict[str, dict[str, object]] = {
        family: {
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
        for family in normalized_families
    }
    retained_delta_totals: dict[str, float] = {family: 0.0 for family in normalized_families}
    retained_delta_counts: dict[str, int] = {family: 0 for family in normalized_families}
    retained_delta_seen: set[str] = set()
    worst_delta_seen: set[str] = set()
    for record in records:
        state = str(record.get("state", "")).strip()
        if state not in {"retain", "reject"}:
            continue
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            continue
        family_deltas = metrics.get("family_pass_rate_delta", {})
        if not isinstance(family_deltas, dict):
            continue
        estimated_cost = _estimated_cost_for_record(record, generate_index)
        for family in normalized_families:
            if family not in family_deltas:
                continue
            try:
                delta = float(family_deltas.get(family, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            summary = family_summaries[family]
            summary["observed_decisions"] = int(summary["observed_decisions"]) + 1
            summary["observed_estimated_cost"] = float(summary["observed_estimated_cost"]) + estimated_cost
            if delta > 0.0:
                summary["positive_delta_decisions"] = int(summary["positive_delta_decisions"]) + 1
            elif delta < 0.0:
                summary["negative_delta_decisions"] = int(summary["negative_delta_decisions"]) + 1
            else:
                summary["neutral_delta_decisions"] = int(summary["neutral_delta_decisions"]) + 1
            if family not in worst_delta_seen:
                summary["worst_pass_rate_delta"] = delta
                worst_delta_seen.add(family)
            else:
                summary["worst_pass_rate_delta"] = min(float(summary["worst_pass_rate_delta"]), delta)
            if state == "retain":
                summary["retained_decisions"] = int(summary["retained_decisions"]) + 1
                summary["retained_estimated_cost"] = float(summary["retained_estimated_cost"]) + estimated_cost
                summary["retained_pass_rate_delta_sum"] = float(summary["retained_pass_rate_delta_sum"]) + delta
                retained_delta_totals[family] += delta
                retained_delta_counts[family] += 1
                if delta > 0.0:
                    summary["retained_positive_delta_decisions"] = int(summary["retained_positive_delta_decisions"]) + 1
                    summary["retained_positive_pass_rate_delta_sum"] = (
                        float(summary["retained_positive_pass_rate_delta_sum"]) + delta
                    )
                elif delta < 0.0:
                    summary["retained_negative_delta_decisions"] = int(summary["retained_negative_delta_decisions"]) + 1
                else:
                    summary["retained_neutral_delta_decisions"] = int(summary["retained_neutral_delta_decisions"]) + 1
                if family not in retained_delta_seen:
                    summary["best_retained_pass_rate_delta"] = delta
                    retained_delta_seen.add(family)
                else:
                    summary["best_retained_pass_rate_delta"] = max(float(summary["best_retained_pass_rate_delta"]), delta)
            else:
                summary["rejected_decisions"] = int(summary["rejected_decisions"]) + 1
                summary["rejected_estimated_cost"] = float(summary["rejected_estimated_cost"]) + estimated_cost
    for family in normalized_families:
        retained_count = retained_delta_counts[family]
        if retained_count > 0:
            family_summaries[family]["average_retained_pass_rate_delta"] = retained_delta_totals[family] / float(retained_count)
    return {
        "priority_families": normalized_families,
        "family_summaries": family_summaries,
        "priority_families_with_signal": [
            family for family in normalized_families if int(family_summaries[family]["observed_decisions"]) > 0
        ],
        "priority_families_with_retained_gain": [
            family
            for family in normalized_families
            if int(family_summaries[family]["retained_positive_delta_decisions"]) > 0
        ],
        "priority_families_without_signal": [
            family for family in normalized_families if int(family_summaries[family]["observed_decisions"]) <= 0
        ],
        "priority_families_with_signal_but_no_retained_gain": [
            family
            for family in normalized_families
            if int(family_summaries[family]["observed_decisions"]) > 0
            and int(family_summaries[family]["retained_positive_delta_decisions"]) <= 0
        ],
    }


def _priority_family_allocation_summary(
    records: list[dict[str, object]],
    priority_families: list[str],
    priority_family_weights: dict[str, float],
) -> dict[str, object]:
    normalized_families = [str(value).strip() for value in priority_families if str(value).strip()]
    weights = {
        family: float(priority_family_weights.get(family, 0.0) or 0.0)
        for family in normalized_families
        if float(priority_family_weights.get(family, 0.0) or 0.0) > 0.0
    }
    fallback_summary_weights: dict[str, float] = {}
    aggregated_task_counts = {family: 0 for family in normalized_families}
    aggregated_pass_rate_totals = {family: 0.0 for family in normalized_families}
    aggregated_pass_rate_counts = {family: 0 for family in normalized_families}
    summaries_checked = 0
    cycles_with_top_planned_family_as_top_sampled = 0
    for record in records:
        metrics = record.get("metrics_summary", {})
        if not isinstance(metrics, dict):
            continue
        summary = metrics.get("priority_family_allocation_summary", {})
        if not isinstance(summary, dict):
            continue
        summaries_checked += 1
        if not weights and not fallback_summary_weights:
            raw_summary_weights = summary.get("priority_benchmark_family_weights", {})
            if isinstance(raw_summary_weights, dict):
                for family in normalized_families:
                    try:
                        weight = float(raw_summary_weights.get(family, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        weight = 0.0
                    if weight > 0.0:
                        fallback_summary_weights[family] = weight
        actual_counts = summary.get("actual_task_counts", {})
        actual_pass_rates = summary.get("actual_pass_rates", {})
        if isinstance(actual_counts, dict):
            for family in normalized_families:
                aggregated_task_counts[family] += int(actual_counts.get(family, 0) or 0)
        if isinstance(actual_pass_rates, dict):
            for family in normalized_families:
                if family in actual_pass_rates:
                    aggregated_pass_rate_totals[family] += float(actual_pass_rates.get(family, 0.0) or 0.0)
                    aggregated_pass_rate_counts[family] += 1
        if (
            str(summary.get("top_planned_family", "")).strip()
            and str(summary.get("top_planned_family", "")).strip() == str(summary.get("top_sampled_family", "")).strip()
        ):
            cycles_with_top_planned_family_as_top_sampled += 1
    if not weights and fallback_summary_weights:
        weights = fallback_summary_weights
    if not weights and normalized_families:
        weights = {family: 1.0 for family in normalized_families}
    total_weight = sum(weights.values())
    planned_weight_shares = {
        family: 0.0 if total_weight <= 0.0 else round(weights.get(family, 0.0) / total_weight, 6)
        for family in normalized_families
    }
    total_priority_tasks = sum(aggregated_task_counts.values())
    aggregated_task_shares = {
        family: 0.0 if total_priority_tasks <= 0 else round(aggregated_task_counts[family] / total_priority_tasks, 6)
        for family in normalized_families
    }
    aggregated_average_pass_rates = {
        family: 0.0
        if aggregated_pass_rate_counts[family] <= 0
        else round(aggregated_pass_rate_totals[family] / aggregated_pass_rate_counts[family], 6)
        for family in normalized_families
    }
    top_planned_family = (
        max(
            normalized_families,
            key=lambda family: (planned_weight_shares.get(family, 0.0), weights.get(family, 0.0), family),
        )
        if normalized_families
        else ""
    )
    top_sampled_family = (
        max(
            normalized_families,
            key=lambda family: (aggregated_task_counts[family], aggregated_task_shares[family], family),
        )
        if total_priority_tasks > 0 and normalized_families
        else ""
    )
    return {
        "priority_families": normalized_families,
        "priority_benchmark_family_weights": weights,
        "planned_weight_shares": planned_weight_shares,
        "summaries_checked": summaries_checked,
        "cycles_with_top_planned_family_as_top_sampled": cycles_with_top_planned_family_as_top_sampled,
        "aggregated_task_counts": aggregated_task_counts,
        "aggregated_task_shares": aggregated_task_shares,
        "aggregated_average_pass_rates": aggregated_average_pass_rates,
        "total_priority_tasks": total_priority_tasks,
        "top_planned_family": top_planned_family,
        "top_sampled_family": top_sampled_family,
        "unsampled_priority_families": [
            family for family in normalized_families if aggregated_task_counts[family] <= 0
        ],
    }


def _trust_breadth_summary(config: KernelConfig) -> dict[str, object]:
    ledger = build_unattended_trust_ledger(config)
    overall = ledger.get("overall_summary", {})
    external = ledger.get("external_summary", {})
    coverage = ledger.get("coverage_summary", {})
    policy = ledger.get("policy", {})
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(external, dict):
        external = {}
    if not isinstance(coverage, dict):
        coverage = {}
    if not isinstance(policy, dict):
        policy = {}
    return {
        "reports_considered": int(ledger.get("reports_considered", 0) or 0),
        "distinct_benchmark_families": int(overall.get("distinct_benchmark_families", 0) or 0),
        "benchmark_families": list(overall.get("benchmark_families", []))
        if isinstance(overall.get("benchmark_families", []), list)
        else [],
        "external_report_count": int(external.get("total", external.get("external_report_count", 0)) or 0),
        "distinct_external_benchmark_families": int(
            external.get("distinct_benchmark_families", external.get("distinct_external_benchmark_families", 0)) or 0
        ),
        "external_benchmark_families": list(
            external.get("benchmark_families", external.get("external_benchmark_families", []))
        )
        if isinstance(external.get("benchmark_families", external.get("external_benchmark_families", [])), list)
        else [],
        "external_success_rate": float(external.get("success_rate", 0.0) or 0.0),
        "external_unsafe_ambiguous_rate": float(external.get("unsafe_ambiguous_rate", 0.0) or 0.0),
        "required_families": list(coverage.get("required_families", []))
        if isinstance(coverage.get("required_families", []), list)
        else [],
        "required_families_with_reports": list(coverage.get("required_families_with_reports", []))
        if isinstance(coverage.get("required_families_with_reports", []), list)
        else [],
        "missing_required_families": list(coverage.get("missing_required_families", []))
        if isinstance(coverage.get("missing_required_families", []), list)
        else [],
        "family_breadth_min_distinct_task_roots": int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0),
        "required_family_clean_task_root_counts": dict(coverage.get("required_family_clean_task_root_counts", {}))
        if isinstance(coverage.get("required_family_clean_task_root_counts", {}), dict)
        else {},
        "missing_required_family_clean_task_root_breadth": [
            family
            for family in (
                list(coverage.get("required_families", []))
                if isinstance(coverage.get("required_families", []), list)
                else []
            )
            if int(
                (
                    dict(coverage.get("required_family_clean_task_root_counts", {}))
                    if isinstance(coverage.get("required_family_clean_task_root_counts", {}), dict)
                    else {}
                ).get(str(family).strip(), 0)
                or 0
            )
            < int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0)
        ],
        "distinct_family_gap": int(coverage.get("distinct_family_gap", 0) or 0),
    }


def _load_json_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _campaign_status_snapshot(
    *,
    planner: ImprovementPlanner,
    cycles_path: Path,
    campaign_match_id: str,
    cycle_log_start_index: int,
    priority_benchmark_families: list[str],
    priority_family_weights: dict[str, float],
) -> dict[str, object]:
    records = planner.load_cycle_records(cycles_path)
    campaign_records = _campaign_records(
        records,
        campaign_match_id=campaign_match_id,
        start_index=cycle_log_start_index,
    )
    decision_records = [
        record for record in campaign_records if str(record.get("state", "")) in {"retain", "reject"}
    ]
    priority_family_allocation_summary = _priority_family_allocation_summary(
        campaign_records,
        priority_benchmark_families,
        priority_family_weights,
    )
    aggregated_task_counts = priority_family_allocation_summary.get("aggregated_task_counts", {})
    if not isinstance(aggregated_task_counts, dict):
        aggregated_task_counts = {}
    families_sampled = [
        family
        for family in priority_benchmark_families
        if int(aggregated_task_counts.get(family, 0) or 0) > 0
    ]
    priority_families_without_sampling = [
        family for family in priority_benchmark_families if family not in set(families_sampled)
    ]
    current_cycle_ids = [
        str(record.get("cycle_id", "")).strip()
        for record in campaign_records
        if str(record.get("cycle_id", "")).strip()
    ]
    return {
        "campaign_records_considered": len(campaign_records),
        "decision_records_considered": len(decision_records),
        "campaign_cycle_ids": current_cycle_ids,
        "priority_family_allocation_summary": priority_family_allocation_summary,
        "families_sampled": families_sampled,
        "priority_families_without_sampling": priority_families_without_sampling,
    }


def _ordered_new_cycle_ids(
    current_cycle_ids: list[str],
    *,
    seen_cycle_ids: set[str],
) -> list[str]:
    ordered: list[str] = []
    seen_local: set[str] = set()
    for cycle_id in current_cycle_ids:
        token = str(cycle_id).strip()
        if not token or token in seen_cycle_ids or token in seen_local:
            continue
        ordered.append(token)
        seen_local.add(token)
    return ordered


def _run_decision_conversion_state(
    *,
    final_state: str,
    runtime_managed_decisions: int,
    partial_productive: bool,
) -> str:
    normalized_state = str(final_state).strip()
    if runtime_managed_decisions > 0:
        return "runtime_managed"
    if normalized_state in {"retain", "reject"}:
        return "non_runtime_managed"
    if partial_productive:
        return "partial_productive_without_decision"
    return "no_decision"


def _summarize_run_decision_conversion(runs: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "runtime_managed_runs": 0,
        "non_runtime_managed_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "no_decision_runs": 0,
    }
    for run in runs:
        state = str(run.get("decision_conversion_state", "")).strip()
        if state == "runtime_managed":
            summary["runtime_managed_runs"] += 1
        elif state == "non_runtime_managed":
            summary["non_runtime_managed_runs"] += 1
        elif state == "partial_productive_without_decision":
            summary["partial_productive_without_decision_runs"] += 1
        else:
            summary["no_decision_runs"] += 1
    summary["decision_runs"] = summary["runtime_managed_runs"] + summary["non_runtime_managed_runs"]
    return summary


def _mirror_autonomous_parent_status(
    *,
    config: KernelConfig,
    child_status_path: Path,
    child_status_payload: dict[str, object],
) -> None:
    raw_parent_status_path = str(os.environ.get("AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH", "")).strip()
    if not raw_parent_status_path:
        return
    parent_status_path = Path(raw_parent_status_path)
    parent_payload = _load_json_payload(parent_status_path)
    if not parent_payload:
        return
    active_run = parent_payload.get("active_run", {})
    if not isinstance(active_run, dict):
        active_run = {}
    expected_run_match_id = str(os.environ.get("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID", "")).strip()
    if expected_run_match_id:
        active_run_match_id = str(active_run.get("run_match_id", "")).strip()
        if active_run_match_id and active_run_match_id != expected_run_match_id:
            return
    expected_run_index = str(os.environ.get("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX", "")).strip()
    if expected_run_index:
        try:
            expected_index_value = int(expected_run_index)
        except ValueError:
            expected_index_value = 0
        active_run_index = int(active_run.get("run_index", 0) or 0)
        if active_run_index and expected_index_value and active_run_index != expected_index_value:
            return
    requested_priority_families = [
        str(value).strip()
        for value in parent_payload.get("requested_priority_benchmark_families", [])
        if str(value).strip()
    ] if isinstance(parent_payload.get("requested_priority_benchmark_families", []), list) else []
    completed_frontier = parent_payload.get("partial_frontier_expansion_summary", {})
    if not isinstance(completed_frontier, dict):
        completed_frontier = {}
    completed_families_sampled = [
        str(value).strip()
        for value in completed_frontier.get("families_sampled", [])
        if str(value).strip()
    ] if isinstance(completed_frontier.get("families_sampled", []), list) else []
    child_families_sampled = [
        str(value).strip()
        for value in child_status_payload.get("families_sampled", [])
        if str(value).strip()
    ] if isinstance(child_status_payload.get("families_sampled", []), list) else []
    merged_families_sampled = _ordered_unique_strings(completed_families_sampled, child_families_sampled)
    if requested_priority_families:
        sampled_set = set(merged_families_sampled)
        merged_unsampled = [
            family for family in requested_priority_families if family not in sampled_set
        ]
    else:
        merged_unsampled = [
            str(value).strip()
            for value in child_status_payload.get("priority_families_without_sampling", [])
            if str(value).strip()
        ] if isinstance(child_status_payload.get("priority_families_without_sampling", []), list) else []
    active_run["child_status_path"] = str(child_status_path)
    active_run["child_status"] = child_status_payload
    parent_payload["active_run"] = active_run
    parent_payload["families_sampled"] = merged_families_sampled
    parent_payload["families_never_sampled"] = merged_unsampled
    parent_payload["pressure_families_without_sampling"] = merged_unsampled
    parent_payload["created_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(parent_status_path, parent_payload, config=config, govern_storage=False)


def _write_campaign_status(
    *,
    config: KernelConfig,
    report_path: Path,
    planner: ImprovementPlanner,
    cycles_path: Path,
    campaign_match_id: str,
    campaign_label: str,
    cycle_log_start_index: int,
    priority_benchmark_families: list[str],
    priority_family_weights: dict[str, float],
    cycles_requested: int,
    runs: list[dict[str, object]],
    state: str,
    active_cycle_run: dict[str, object] | None = None,
    snapshot: dict[str, object] | None = None,
) -> Path:
    if snapshot is None:
        snapshot = _campaign_status_snapshot(
            planner=planner,
            cycles_path=cycles_path,
            campaign_match_id=campaign_match_id,
            cycle_log_start_index=cycle_log_start_index,
            priority_benchmark_families=priority_benchmark_families,
            priority_family_weights=priority_family_weights,
        )
    progress_summary = _partial_progress_summary(
        active_cycle_run,
        priority_benchmark_families=priority_benchmark_families,
    )
    families_sampled = _ordered_unique_strings(
        snapshot.get("families_sampled", []),
        progress_summary.get("sampled_families_from_progress", []),
    )
    priority_families_without_sampling = [
        family for family in priority_benchmark_families if family not in set(families_sampled)
    ]
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "repeated_improvement_status",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "state": state,
        "campaign_label": campaign_label,
        "campaign_match_id": campaign_match_id,
        "cycles_requested": cycles_requested,
        "completed_runs": len(runs),
        "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
        "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
        "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
        "partial_productive_runs": sum(1 for run in runs if bool(run.get("partial_productive", False))),
        "partial_candidate_runs": sum(1 for run in runs if bool(run.get("partial_candidate_generated", False))),
        "priority_benchmark_families": priority_benchmark_families,
        "priority_benchmark_family_weights": dict(priority_family_weights),
        "active_cycle_run": active_cycle_run or {},
        **snapshot,
        "families_sampled": families_sampled,
        "priority_families_without_sampling": priority_families_without_sampling,
        "active_cycle_progress": progress_summary,
    }
    status_path = _status_path(config)
    atomic_write_json(status_path, payload, config=config, govern_storage=False)
    _mirror_autonomous_parent_status(
        config=config,
        child_status_path=status_path,
        child_status_payload=payload,
    )
    return status_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--campaign-width", type=int, default=2)
    parser.add_argument("--variant-width", type=int, default=1)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--priority-benchmark-family-weight", action="append", default=[])
    parser.add_argument("--child-heartbeat-seconds", type=float, default=120.0)
    parser.add_argument("--max-child-silence-seconds", type=float, default=1800.0)
    parser.add_argument("--max-child-runtime-seconds", type=float, default=14400.0)
    parser.add_argument("--max-child-progress-stall-seconds", type=float, default=1800.0)
    parser.add_argument("--campaign-label", default="")
    parser.add_argument("--campaign-match-id", default="")
    parser.add_argument("--exclude-subsystem", action="append", default=[])
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    args = parser.parse_args()

    received_signal = {"value": 0}

    def _handle_termination(signum: int) -> None:
        received_signal["value"] = int(signum)
        raise KeyboardInterrupt(f"received signal {signal.Signals(signum).name}")

    restore_signal_handlers = install_termination_handlers(_handle_termination)

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device
    config.ensure_directories()
    campaign_label = str(args.campaign_label).strip()
    campaign_match_id = str(args.campaign_match_id).strip() or f"campaign-{uuid4().hex}"
    requested_priority_benchmark_families = [
        str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
    ]
    requested_priority_family_weights = _priority_benchmark_family_weights(args.priority_benchmark_family_weight)
    current_priority_benchmark_families = list(requested_priority_benchmark_families)
    current_priority_family_weights = dict(requested_priority_family_weights)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    runs: list[dict[str, object]] = []
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    _reconcile_incomplete_cycles(
        config=config,
        planner=planner,
        progress_label=campaign_label or campaign_match_id,
    )
    cycle_log_start_index = len(planner.load_cycle_records(config.improvement_cycles_path))
    current_campaign_width = max(1, int(args.campaign_width))
    current_variant_width = max(1, int(args.variant_width))
    current_task_limit = max(0, int(args.task_limit))
    current_adaptive_search = bool(args.adaptive_search)
    max_campaign_width = max(current_campaign_width, 4)
    max_variant_width = max(current_variant_width, 4)
    max_task_limit = max(current_task_limit, int(getattr(config, "compare_feature_max_tasks", 0) or 0))
    seen_campaign_cycle_ids: set[str] = set()
    active_cycle_run: dict[str, object] = {}
    last_status_refresh_at = 0.0
    last_snapshot_refresh_at = 0.0
    status_snapshot_cache: dict[str, object] = {}

    report_path = config.improvement_reports_dir / (
        f"campaign_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    try:
        _write_campaign_status(
            config=config,
            report_path=report_path,
            planner=planner,
            cycles_path=config.improvement_cycles_path,
            campaign_match_id=campaign_match_id,
            campaign_label=campaign_label,
            cycle_log_start_index=cycle_log_start_index,
            priority_benchmark_families=[str(value).strip() for value in args.priority_benchmark_family if str(value).strip()],
            priority_family_weights=current_priority_family_weights,
            cycles_requested=max(1, args.cycles),
            runs=runs,
            state="starting",
            snapshot=status_snapshot_cache,
        )
        for index in range(1, max(1, args.cycles) + 1):
            cmd = [
                sys.executable,
                "-u",
                str(script_path),
                "--campaign-width",
                str(current_campaign_width),
                "--variant-width",
                str(current_variant_width),
                "--progress-label",
                campaign_label or f"cycle-{index}",
                "--protocol-match-id",
                campaign_match_id,
            ]
            if current_adaptive_search:
                cmd.append("--adaptive-search")
            if current_task_limit > 0:
                cmd.extend(["--task-limit", str(current_task_limit)])
            for family in current_priority_benchmark_families:
                token = str(family).strip()
                if token:
                    cmd.extend(["--priority-benchmark-family", token])
            for family in current_priority_benchmark_families:
                weight = float(current_priority_family_weights.get(family, 0.0) or 0.0)
                if weight > 0.0:
                    cmd.extend(["--priority-benchmark-family-weight", f"{family}={weight}"])
            if args.provider:
                cmd.extend(["--provider", args.provider])
            if args.model:
                cmd.extend(["--model", args.model])
            if args.tolbert_device:
                cmd.extend(["--tolbert-device", args.tolbert_device])
            for excluded_subsystem in args.exclude_subsystem:
                token = str(excluded_subsystem).strip()
                if token:
                    cmd.extend(["--exclude-subsystem", token])
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
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env.update(_runtime_env(config))
            child_reports_dir = Path(
                str(env.get("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", config.improvement_reports_dir)).strip()
                or str(config.improvement_reports_dir)
            )
            env["AGENT_KERNEL_RUNTIME_DATABASE_PATH"] = str(child_reports_dir / "agentkernel.sqlite3")
            active_cycle_run = {
                "index": index,
                "progress_label": campaign_label or f"cycle-{index}",
                "campaign_width": current_campaign_width,
                "variant_width": current_variant_width,
                "task_limit": current_task_limit,
                "adaptive_search": bool(current_adaptive_search),
                "priority_benchmark_families": list(current_priority_benchmark_families),
                "priority_benchmark_family_weights": dict(current_priority_family_weights),
            }
            _write_campaign_status(
                config=config,
                report_path=report_path,
                planner=planner,
                cycles_path=config.improvement_cycles_path,
                campaign_match_id=campaign_match_id,
                campaign_label=campaign_label,
                cycle_log_start_index=cycle_log_start_index,
                priority_benchmark_families=current_priority_benchmark_families,
                priority_family_weights=current_priority_family_weights,
                cycles_requested=max(1, args.cycles),
                runs=runs,
                state="running",
                active_cycle_run=active_cycle_run,
                snapshot=status_snapshot_cache,
            )

            def _refresh_child_status(event: dict[str, object]) -> None:
                nonlocal last_status_refresh_at, last_snapshot_refresh_at, active_cycle_run, status_snapshot_cache
                event_kind = str(event.get("event", "")).strip()
                line = str(event.get("line", "")).strip()
                active_cycle_run = {
                    **active_cycle_run,
                    "pid": int(event.get("pid", 0) or active_cycle_run.get("pid", 0) or 0),
                    "last_event": event_kind,
                    "last_event_at": float(event.get("timestamp", 0.0) or time.time()),
                }
                if event_kind == "start":
                    active_cycle_run["started_at"] = float(event.get("started_at", 0.0) or time.time())
                if event_kind == "heartbeat":
                    active_cycle_run["silence_seconds"] = int(event.get("silence_seconds", 0) or 0)
                if line:
                    active_cycle_run["last_output_line"] = line
                    if "[cycle:" in line or "[eval:" in line or "finalize phase=" in line or "[repeated]" in line:
                        active_cycle_run["last_progress_line"] = line
                    parsed_progress = _parse_progress_fields(line)
                    active_cycle_run.update(parsed_progress)
                    current_task = active_cycle_run.get("current_task", {})
                    if isinstance(current_task, dict):
                        family = str(current_task.get("family", "")).strip()
                        if family:
                            active_cycle_run["sampled_families_from_progress"] = _ordered_unique_strings(
                                active_cycle_run.get("sampled_families_from_progress", []),
                                [family],
                            )
                refresh_now = event_kind in {"start", "heartbeat", "timeout", "exit"}
                if not refresh_now and line and (
                    "[cycle:" in line or "[eval:" in line or "finalize phase=" in line or "[repeated]" in line
                ):
                    refresh_now = True
                now = time.monotonic()
                if not refresh_now and (now - last_status_refresh_at) < 2.0:
                    return
                last_status_refresh_at = now
                force_snapshot_refresh = event_kind in {"start", "timeout", "exit"}
                if not force_snapshot_refresh and (now - last_snapshot_refresh_at) >= 30.0:
                    force_snapshot_refresh = True
                if force_snapshot_refresh:
                    status_snapshot_cache = _campaign_status_snapshot(
                        planner=planner,
                        cycles_path=config.improvement_cycles_path,
                        campaign_match_id=campaign_match_id,
                        cycle_log_start_index=cycle_log_start_index,
                        priority_benchmark_families=current_priority_benchmark_families,
                        priority_family_weights=current_priority_family_weights,
                    )
                    last_snapshot_refresh_at = now
                _write_campaign_status(
                    config=config,
                    report_path=report_path,
                    planner=planner,
                    cycles_path=config.improvement_cycles_path,
                    campaign_match_id=campaign_match_id,
                    campaign_label=campaign_label,
                    cycle_log_start_index=cycle_log_start_index,
                    priority_benchmark_families=current_priority_benchmark_families,
                    priority_family_weights=current_priority_family_weights,
                    cycles_requested=max(1, args.cycles),
                    runs=runs,
                    state="running",
                    active_cycle_run=active_cycle_run,
                    snapshot=status_snapshot_cache,
                )

            completed = _run_and_stream(
                cmd,
                cwd=repo_root,
                env=env,
                progress_label=campaign_label or f"cycle-{index}",
                heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                max_silence_seconds=float(args.max_child_silence_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                on_event=_refresh_child_status,
            )
            runs.append(
                {
                    "index": index,
                    "returncode": int(completed["returncode"]),
                    "stdout": str(completed["stdout"]).strip(),
                    "stderr": str(completed["stderr"]).strip(),
                    "timed_out": bool(completed.get("timed_out", False)),
                    "timeout_reason": str(completed.get("timeout_reason", "")).strip(),
                    "priority_benchmark_families": list(current_priority_benchmark_families),
                    "priority_benchmark_family_weights": dict(current_priority_family_weights),
                }
            )
            partial_progress = _partial_progress_summary(
                active_cycle_run,
                priority_benchmark_families=current_priority_benchmark_families,
            )
            runs[-1]["partial_progress"] = partial_progress
            runs[-1]["partial_productive"] = bool(partial_progress.get("productive_partial", False))
            runs[-1]["partial_candidate_generated"] = bool(partial_progress.get("candidate_generated", False))
            active_cycle_run = {}
            _reconcile_incomplete_cycles(
                config=config,
                planner=planner,
                progress_label=campaign_label or campaign_match_id,
            )
            records = planner.load_cycle_records(config.improvement_cycles_path)
            campaign_records = _campaign_records(
                records,
                campaign_match_id=campaign_match_id,
                start_index=cycle_log_start_index,
            )
            current_cycle_ids = {
                str(record.get("cycle_id", "")).strip()
                for record in campaign_records
                if str(record.get("cycle_id", "")).strip()
            }
            ordered_current_cycle_ids = [
                str(record.get("cycle_id", "")).strip()
                for record in campaign_records
                if str(record.get("cycle_id", "")).strip()
            ]
            new_cycle_ids = _ordered_new_cycle_ids(
                ordered_current_cycle_ids,
                seen_cycle_ids=seen_campaign_cycle_ids,
            )
            new_records = [
                record
                for record in campaign_records
                if str(record.get("cycle_id", "")).strip() in set(new_cycle_ids)
            ]
            cycle_ids_for_run = list(new_cycle_ids)
            if not cycle_ids_for_run:
                fallback_cycle_id = str(runs[-1].get("cycle_id", "")).strip()
                if fallback_cycle_id:
                    cycle_ids_for_run = [fallback_cycle_id]
            new_runtime_managed_decisions = _production_decisions(new_records)
            recovered_runtime_managed_decisions = (
                _recovered_runtime_managed_decisions_from_reports(config, cycle_ids_for_run)
                if not new_runtime_managed_decisions and cycle_ids_for_run
                else []
            )
            if recovered_runtime_managed_decisions:
                new_runtime_managed_decisions = _merge_decision_records(
                    new_runtime_managed_decisions,
                    recovered_runtime_managed_decisions,
                )
            new_decision_summary = _count_decisions(new_runtime_managed_decisions)
            runs[-1]["runtime_managed_decisions"] = new_decision_summary["total_decisions"]
            runs[-1]["runtime_managed_retained_cycles"] = new_decision_summary["retained_cycles"]
            runs[-1]["runtime_managed_rejected_cycles"] = new_decision_summary["rejected_cycles"]
            runs[-1]["campaign_cycle_ids"] = list(new_cycle_ids)
            runs[-1]["productive"] = (
                new_decision_summary["total_decisions"] > 0
                or bool(runs[-1].get("partial_productive", False))
            )
            runs[-1]["retained_gain"] = new_decision_summary["retained_cycles"] > 0
            best_cycle_audit: dict[str, object] | None = None
            for cycle_id in cycle_ids_for_run:
                cycle_audit = planner.cycle_audit_summary(config.improvement_cycles_path, cycle_id=cycle_id)
                report_payload = _load_cycle_report(config, cycle_id)
                report_audit = _cycle_audit_summary_from_report(report_payload) if isinstance(report_payload, dict) else None
                if isinstance(report_audit, dict) and str(report_audit.get("final_state", "")).strip() in {"retain", "reject"}:
                    cycle_audit = report_audit
                elif not isinstance(cycle_audit, dict):
                    cycle_audit = report_audit
                if not isinstance(cycle_audit, dict):
                    continue
                best_cycle_audit = cycle_audit
                if str(cycle_audit.get("final_state", "")).strip() in {"retain", "reject"}:
                    break
            if isinstance(best_cycle_audit, dict):
                for field in (
                    "cycle_id",
                    "subsystem",
                    "selected_variant_id",
                    "prior_retained_cycle_id",
                    "candidate_artifact_path",
                    "active_artifact_path",
                    "final_state",
                    "final_reason",
                    "preview_reason_code",
                    "decision_reason_code",
                    "phase_gate_passed",
                    "artifact_kind",
                    "artifact_path",
                    "artifact_lifecycle_state",
                    "artifact_sha256",
                    "previous_artifact_sha256",
                    "rollback_artifact_path",
                    "artifact_snapshot_path",
                    "baseline_pass_rate",
                    "candidate_pass_rate",
                    "baseline_average_steps",
                    "candidate_average_steps",
                ):
                    if field in best_cycle_audit:
                        runs[-1][field] = best_cycle_audit[field]
            runs[-1]["decision_conversion_state"] = _run_decision_conversion_state(
                final_state=str(runs[-1].get("final_state", "")).strip(),
                runtime_managed_decisions=int(runs[-1].get("runtime_managed_decisions", 0) or 0),
                partial_productive=bool(runs[-1].get("partial_productive", False)),
            )
            seen_campaign_cycle_ids.update(new_cycle_ids)
            status_snapshot_cache = _campaign_status_snapshot(
                planner=planner,
                cycles_path=config.improvement_cycles_path,
                campaign_match_id=campaign_match_id,
                cycle_log_start_index=cycle_log_start_index,
                priority_benchmark_families=current_priority_benchmark_families,
                priority_family_weights=current_priority_family_weights,
            )
            last_snapshot_refresh_at = time.monotonic()
            _write_campaign_status(
                config=config,
                report_path=report_path,
                planner=planner,
                cycles_path=config.improvement_cycles_path,
                campaign_match_id=campaign_match_id,
                campaign_label=campaign_label,
                cycle_log_start_index=cycle_log_start_index,
                priority_benchmark_families=current_priority_benchmark_families,
                priority_family_weights=current_priority_family_weights,
                cycles_requested=max(1, args.cycles),
                runs=runs,
                state="running" if index < max(1, args.cycles) else "finalizing",
                snapshot=status_snapshot_cache,
            )
            rerouting_summary = _reroute_priority_families_for_unsampled_pressure(
                current_priority_benchmark_families,
                current_priority_family_weights,
                list(status_snapshot_cache.get("priority_families_without_sampling", []))
                if isinstance(status_snapshot_cache.get("priority_families_without_sampling", []), list)
                else [],
            )
            runs[-1]["priority_family_rerouting"] = rerouting_summary
            budget_rerouting_summary = _reroute_task_budget_for_unsampled_pressure(
                current_task_limit=current_task_limit,
                max_task_limit=max_task_limit,
                priority_families=current_priority_benchmark_families,
                unsampled_priority_families=list(rerouting_summary.get("unsampled_priority_families", [])),
            )
            runs[-1]["priority_family_budget_rerouting"] = budget_rerouting_summary
            if int(completed["returncode"]) != 0:
                break
            if bool(rerouting_summary.get("applied", False)) and index < max(1, args.cycles):
                current_priority_benchmark_families = list(rerouting_summary["priority_benchmark_families"])
                current_priority_family_weights = dict(rerouting_summary["priority_benchmark_family_weights"])
            if bool(budget_rerouting_summary.get("applied", False)) and index < max(1, args.cycles):
                current_task_limit = int(budget_rerouting_summary["next_task_limit"])
                print(
                    "[repeated] budget_reroute reason=unsampled_priority_task_budget "
                    f"next_task_limit={current_task_limit} "
                    f"unsampled_priority_families={','.join(budget_rerouting_summary['unsampled_priority_families'])}",
                    file=sys.stderr,
                    flush=True,
                )
            if new_decision_summary["retained_cycles"] <= 0 and index < max(1, args.cycles):
                current_adaptive_search = True
                if current_task_limit > 0 and max_task_limit > 0:
                    current_task_limit = min(max_task_limit, max(current_task_limit + 1, current_task_limit * 2))
                current_campaign_width = min(max_campaign_width, current_campaign_width + 1)
                current_variant_width = min(max_variant_width, current_variant_width + 1)
                print(
                    "[repeated] search_adapt reason=no_retained_gain "
                    f"next_campaign_width={current_campaign_width} "
                    f"next_variant_width={current_variant_width} "
                    f"next_task_limit={current_task_limit} "
                    f"adaptive_search={str(current_adaptive_search).lower()}",
                    file=sys.stderr,
                    flush=True,
                )

        records = planner.load_cycle_records(config.improvement_cycles_path)
        campaign_records = _campaign_records(
            records,
            campaign_match_id=campaign_match_id,
            start_index=cycle_log_start_index,
        )
        generate_index = {
            str(record.get("cycle_id", "")): record
            for record in campaign_records
            if str(record.get("state", "")) == "generate"
        }
        campaign_cycle_ids = {
            str(record.get("cycle_id", "")).strip()
            for record in campaign_records
            if str(record.get("cycle_id", "")).strip()
        }
        report_cycle_ids = set(campaign_cycle_ids)
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_cycle_id = str(run.get("cycle_id", "")).strip()
            if run_cycle_id:
                report_cycle_ids.add(run_cycle_id)
            for cycle_id in run.get("campaign_cycle_ids", []) if isinstance(run.get("campaign_cycle_ids", []), list) else []:
                token = str(cycle_id).strip()
                if token:
                    report_cycle_ids.add(token)
        decision_records = [
            record for record in campaign_records if str(record.get("state", "")) in {"retain", "reject"}
        ]
        recovered_runtime_managed_decisions = _recovered_runtime_managed_decisions_from_reports(
            config,
            sorted(report_cycle_ids),
        )
        decision_records = _merge_decision_records(decision_records, recovered_runtime_managed_decisions)
        recent_decisions = decision_records[-10:]
        summary = _yield_summary_for(decision_records, generate_index)
        incomplete_cycles = (
            [
                item
                for item in planner.incomplete_cycle_summaries(config.improvement_cycles_path, protocol="autonomous")
                if str(item.get("cycle_id", "")).strip() in campaign_cycle_ids
            ]
            if hasattr(planner, "incomplete_cycle_summaries")
            else []
        )
        planner_pressure_summary = _planner_pressure_summary(campaign_records)
        production_decisions = _merge_decision_records(
            _production_decisions(campaign_records),
            recovered_runtime_managed_decisions,
        )
        non_runtime_managed_decisions = _non_runtime_managed_decisions(campaign_records)
        phase_gate_summary = _phase_gate_summary_for(production_decisions)
        trust_breadth_summary = _trust_breadth_summary(config)
        priority_family_yield_summary = _priority_family_yield_summary(
            production_decisions,
            requested_priority_benchmark_families,
            generate_index,
        )
        priority_family_allocation_summary = _priority_family_allocation_summary(
            campaign_records,
            requested_priority_benchmark_families,
            requested_priority_family_weights,
        )
        inherited_decisions = 0
        runtime_managed_decisions = 0
        for record in decision_records:
            cycle_id = str(record.get("cycle_id", ""))
            generate_record = generate_index.get(cycle_id, {})
            metrics_summary = generate_record.get("metrics_summary", {})
            if isinstance(metrics_summary, dict) and str(metrics_summary.get("prior_retained_cycle_id", "")).strip():
                inherited_decisions += 1
            if _is_runtime_managed_artifact_path(str(record.get("artifact_path", ""))):
                runtime_managed_decisions += 1
        candidate_isolation_summary = _candidate_isolation_summary(decision_records, generate_index)
        partial_progress_summary = _partial_progress_rollup(runs)
        decision_conversion_summary = _summarize_run_decision_conversion(runs)

        report = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_campaign_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "campaign_label": campaign_label,
            "campaign_match_id": campaign_match_id,
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": max(0, args.task_limit),
            "effective_task_limit": max(0, current_task_limit),
            "priority_benchmark_families": requested_priority_benchmark_families,
            "priority_benchmark_family_weights": dict(requested_priority_family_weights),
            "effective_priority_benchmark_families": list(current_priority_benchmark_families),
            "effective_priority_benchmark_family_weights": dict(current_priority_family_weights),
            "excluded_subsystems": [str(value).strip() for value in args.exclude_subsystem if str(value).strip()],
            "record_scope": {
                "protocol": "autonomous",
                "campaign_match_id": campaign_match_id,
                "cycle_log_start_index": cycle_log_start_index,
                "records_considered": len(campaign_records),
                "decision_records_considered": len(decision_records),
                "cycle_ids": sorted(report_cycle_ids),
            },
            "cycles_requested": max(1, args.cycles),
            "completed_runs": len(runs),
            "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
            "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
            "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
            "runtime_managed_decisions": runtime_managed_decisions,
            "partial_productive_runs": partial_progress_summary["partial_productive_runs"],
            "partial_candidate_runs": partial_progress_summary["partial_candidate_runs"],
            "yield_summary": {
                "retained_cycles": summary["retained_cycles"],
                "rejected_cycles": summary["rejected_cycles"],
                "total_decisions": summary["total_decisions"],
                "retained_by_subsystem": summary["retained_by_subsystem"],
                "rejected_by_subsystem": summary["rejected_by_subsystem"],
                "average_retained_pass_rate_delta": summary["average_retained_pass_rate_delta"],
                "average_retained_step_delta": summary["average_retained_step_delta"],
                "average_rejected_pass_rate_delta": summary["average_rejected_pass_rate_delta"],
                "average_rejected_step_delta": summary["average_rejected_step_delta"],
            },
            "production_yield_summary": _yield_summary_for(production_decisions, generate_index),
            "decision_stream_summary": {
                "runtime_managed": _yield_summary_for(production_decisions, generate_index),
                "non_runtime_managed": _yield_summary_for(non_runtime_managed_decisions, generate_index),
            },
            "phase_gate_summary": phase_gate_summary,
            "trust_breadth_summary": trust_breadth_summary,
            "priority_family_yield_summary": priority_family_yield_summary,
            "priority_family_allocation_summary": priority_family_allocation_summary,
            "incomplete_cycle_summary": {
                "count": len(incomplete_cycles),
                "cycle_ids": [str(item.get("cycle_id", "")) for item in incomplete_cycles],
                "subsystems": [str(item.get("subsystem", "")) for item in incomplete_cycles],
            },
            "planner_pressure_summary": planner_pressure_summary,
            "inheritance_summary": {
                "decision_count": len(decision_records),
                "inherited_decisions": inherited_decisions,
                "runtime_managed_decisions": runtime_managed_decisions,
                "non_runtime_managed_decisions": max(0, len(decision_records) - runtime_managed_decisions),
            },
            "candidate_isolation_summary": candidate_isolation_summary,
            "decision_conversion_summary": decision_conversion_summary,
            "partial_progress_summary": partial_progress_summary,
            "recent_decisions": recent_decisions,
            "recent_runtime_managed_decisions": production_decisions[-10:],
            "recent_production_decisions": production_decisions[-10:],
            "recent_non_runtime_decisions": non_runtime_managed_decisions[-10:],
            "runs": runs,
        }
        atomic_write_json(report_path, report, config=config)
        status_snapshot_cache = _campaign_status_snapshot(
            planner=planner,
            cycles_path=config.improvement_cycles_path,
            campaign_match_id=campaign_match_id,
            cycle_log_start_index=cycle_log_start_index,
            priority_benchmark_families=current_priority_benchmark_families,
            priority_family_weights=current_priority_family_weights,
        )
        _write_campaign_status(
                config=config,
                report_path=report_path,
                planner=planner,
                cycles_path=config.improvement_cycles_path,
                campaign_match_id=campaign_match_id,
                campaign_label=campaign_label,
                cycle_log_start_index=cycle_log_start_index,
                priority_benchmark_families=current_priority_benchmark_families,
                priority_family_weights=current_priority_family_weights,
                cycles_requested=max(1, args.cycles),
                runs=runs,
                state="finished",
                snapshot=status_snapshot_cache,
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=f"campaign:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
                state="record",
                subsystem="campaign",
                action="summarize_repeated_cycles",
                artifact_path=str(report_path),
                artifact_kind="improvement_campaign_report",
                reason="record repeated-cycle yield summary for runtime-managed artifacts",
                metrics_summary={
                    "campaign_label": campaign_label,
                    "campaign_match_id": campaign_match_id,
                    "adaptive_search": bool(args.adaptive_search),
                    "task_limit": max(0, args.task_limit),
                    "effective_task_limit": max(0, current_task_limit),
                    "priority_benchmark_families": requested_priority_benchmark_families,
                    "priority_benchmark_family_weights": dict(requested_priority_family_weights),
                    "effective_priority_benchmark_families": list(current_priority_benchmark_families),
                    "effective_priority_benchmark_family_weights": dict(current_priority_family_weights),
                    "cycles_requested": max(1, args.cycles),
                    "completed_runs": len(runs),
                    "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
                    "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
                    "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
                    "partial_productive_runs": partial_progress_summary["partial_productive_runs"],
                    "partial_candidate_runs": partial_progress_summary["partial_candidate_runs"],
                    "observed_primary_runs": partial_progress_summary["observed_primary_runs"],
                    "generated_success_started_runs": partial_progress_summary["generated_success_started_runs"],
                    "generated_success_completed_runs": partial_progress_summary["generated_success_completed_runs"],
                    "production_total_decisions": report["production_yield_summary"]["total_decisions"],
                    "production_retained_cycles": report["production_yield_summary"]["retained_cycles"],
                    "production_rejected_cycles": report["production_yield_summary"]["rejected_cycles"],
                    "all_retained_phase_gates_passed": phase_gate_summary["all_retained_phase_gates_passed"],
                    "incomplete_cycle_count": len(incomplete_cycles),
                    "campaign_breadth_pressure_cycles": planner_pressure_summary["campaign_breadth_pressure_cycles"],
                    "variant_breadth_pressure_cycles": planner_pressure_summary["variant_breadth_pressure_cycles"],
                    "external_report_count": trust_breadth_summary["external_report_count"],
                    "distinct_external_benchmark_families": trust_breadth_summary["distinct_external_benchmark_families"],
                    "priority_families_with_retained_gain": priority_family_yield_summary["priority_families_with_retained_gain"],
                    "priority_families_without_signal": priority_family_yield_summary["priority_families_without_signal"],
                    "priority_families_with_signal_but_no_retained_gain": priority_family_yield_summary[
                        "priority_families_with_signal_but_no_retained_gain"
                    ],
                    "priority_family_allocation_top_planned_family": priority_family_allocation_summary["top_planned_family"],
                    "priority_family_allocation_top_sampled_family": priority_family_allocation_summary["top_sampled_family"],
                    "priority_family_allocation_total_priority_tasks": priority_family_allocation_summary["total_priority_tasks"],
                },
            ),
            govern_exports=False,
        )
        print(report_path)
    except KeyboardInterrupt:
        synthetic_run = _append_partial_run_from_active_progress(
            runs=runs,
            active_cycle_run=active_cycle_run,
            priority_benchmark_families=current_priority_benchmark_families,
            returncode=-9,
            stderr=(
                f"received signal {signal.Signals(received_signal['value']).name}"
                if int(received_signal["value"]) > 0
                else "operator interrupted repeated improvement campaign"
            ),
            interrupted=True,
        )
        partial_progress_summary = _partial_progress_rollup(runs)
        interrupted_report = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_campaign_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "campaign_label": campaign_label,
            "campaign_match_id": campaign_match_id,
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": max(0, args.task_limit),
            "effective_task_limit": max(0, current_task_limit),
            "priority_benchmark_families": requested_priority_benchmark_families,
            "priority_benchmark_family_weights": dict(requested_priority_family_weights),
            "effective_priority_benchmark_families": list(current_priority_benchmark_families),
            "effective_priority_benchmark_family_weights": dict(current_priority_family_weights),
            "excluded_subsystems": [str(value).strip() for value in args.exclude_subsystem if str(value).strip()],
            "cycles_requested": max(1, args.cycles),
            "completed_runs": len(runs),
            "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
            "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
            "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
            "partial_productive_runs": partial_progress_summary["partial_productive_runs"],
            "partial_candidate_runs": partial_progress_summary["partial_candidate_runs"],
            "status": "interrupted",
            "reason": (
                f"received signal {signal.Signals(received_signal['value']).name}"
                if int(received_signal["value"]) > 0
                else "operator interrupted repeated improvement campaign"
            ),
            "partial_progress_summary": partial_progress_summary,
            "interrupted_active_cycle_run": active_cycle_run,
            "synthetic_interrupted_run_index": synthetic_run.get("index") if isinstance(synthetic_run, dict) else 0,
            "runs": runs,
        }
        atomic_write_json(report_path, interrupted_report, config=config)
        _write_campaign_status(
            config=config,
            report_path=report_path,
            planner=planner,
            cycles_path=config.improvement_cycles_path,
            campaign_match_id=campaign_match_id,
            campaign_label=campaign_label,
            cycle_log_start_index=cycle_log_start_index,
            priority_benchmark_families=current_priority_benchmark_families,
            priority_family_weights=current_priority_family_weights,
            cycles_requested=max(1, args.cycles),
            runs=runs,
            state="interrupted",
            active_cycle_run=active_cycle_run,
            snapshot=status_snapshot_cache,
        )
        print(report_path)
        raise SystemExit(130)
    finally:
        restore_signal_handlers()


if __name__ == "__main__":
    main()
