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


def _runtime_env(config: KernelConfig) -> dict[str, str]:
    return config.to_env()


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
            if max_runtime > 0.0 and runtime_elapsed >= max_runtime:
                terminate_process_tree(process)
                return _timeout_result(
                    reason=f"child exceeded max runtime of {int(max_runtime)} seconds",
                    details={"runtime_seconds": int(runtime_elapsed)},
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
        scoped = [
            record
            for record in scoped
            if _record_protocol(record) == "autonomous"
            and _record_protocol_match_id(record) == campaign_match_id
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
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(external, dict):
        external = {}
    if not isinstance(coverage, dict):
        coverage = {}
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
        "distinct_family_gap": int(coverage.get("distinct_family_gap", 0) or 0),
    }


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
    priority_family_weights = _priority_benchmark_family_weights(args.priority_benchmark_family_weight)

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
    cycle_log_start_index = len(planner.load_cycle_records(config.improvement_cycles_path))
    current_campaign_width = max(1, int(args.campaign_width))
    current_variant_width = max(1, int(args.variant_width))
    current_task_limit = max(0, int(args.task_limit))
    current_adaptive_search = bool(args.adaptive_search)
    max_campaign_width = max(current_campaign_width, 4)
    max_variant_width = max(current_variant_width, 4)
    max_task_limit = max(current_task_limit, int(getattr(config, "compare_feature_max_tasks", 0) or 0))
    seen_campaign_cycle_ids: set[str] = set()

    report_path = config.improvement_reports_dir / (
        f"campaign_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    try:
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
            for family in args.priority_benchmark_family:
                token = str(family).strip()
                if token:
                    cmd.extend(["--priority-benchmark-family", token])
            for weighted_family in args.priority_benchmark_family_weight:
                token = str(weighted_family).strip()
                if token:
                    cmd.extend(["--priority-benchmark-family-weight", token])
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
            completed = _run_and_stream(
                cmd,
                cwd=repo_root,
                env=env,
                progress_label=campaign_label or f"cycle-{index}",
                heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                max_silence_seconds=float(args.max_child_silence_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
            )
            runs.append(
                {
                    "index": index,
                    "returncode": int(completed["returncode"]),
                    "stdout": str(completed["stdout"]).strip(),
                    "stderr": str(completed["stderr"]).strip(),
                    "timed_out": bool(completed.get("timed_out", False)),
                    "timeout_reason": str(completed.get("timeout_reason", "")).strip(),
                }
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
            new_cycle_ids = {
                cycle_id for cycle_id in current_cycle_ids
                if cycle_id and cycle_id not in seen_campaign_cycle_ids
            }
            new_records = [
                record
                for record in campaign_records
                if str(record.get("cycle_id", "")).strip() in new_cycle_ids
            ]
            new_runtime_managed_decisions = _production_decisions(new_records)
            new_decision_summary = _count_decisions(new_runtime_managed_decisions)
            runs[-1]["runtime_managed_decisions"] = new_decision_summary["total_decisions"]
            runs[-1]["runtime_managed_retained_cycles"] = new_decision_summary["retained_cycles"]
            runs[-1]["runtime_managed_rejected_cycles"] = new_decision_summary["rejected_cycles"]
            runs[-1]["campaign_cycle_ids"] = sorted(new_cycle_ids)
            runs[-1]["productive"] = new_decision_summary["total_decisions"] > 0
            runs[-1]["retained_gain"] = new_decision_summary["retained_cycles"] > 0
            seen_campaign_cycle_ids.update(new_cycle_ids)
            if int(completed["returncode"]) != 0:
                break
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
        recent_decisions = [
            record
            for record in campaign_records
            if str(record.get("state", "")) in {"retain", "reject"}
        ][-10:]
        generate_index = {
            str(record.get("cycle_id", "")): record
            for record in campaign_records
            if str(record.get("state", "")) == "generate"
        }
        decision_records = [
            record for record in campaign_records if str(record.get("state", "")) in {"retain", "reject"}
        ]
        summary = _yield_summary_for(decision_records, generate_index)
        campaign_cycle_ids = {
            str(record.get("cycle_id", "")).strip()
            for record in campaign_records
            if str(record.get("cycle_id", "")).strip()
        }
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
        production_decisions = _production_decisions(campaign_records)
        non_runtime_managed_decisions = _non_runtime_managed_decisions(campaign_records)
        phase_gate_summary = _phase_gate_summary_for(production_decisions)
        trust_breadth_summary = _trust_breadth_summary(config)
        priority_family_yield_summary = _priority_family_yield_summary(
            production_decisions,
            [str(value).strip() for value in args.priority_benchmark_family if str(value).strip()],
            generate_index,
        )
        priority_family_allocation_summary = _priority_family_allocation_summary(
            campaign_records,
            [str(value).strip() for value in args.priority_benchmark_family if str(value).strip()],
            priority_family_weights,
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

        report = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_campaign_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "campaign_label": campaign_label,
            "campaign_match_id": campaign_match_id,
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": max(0, args.task_limit),
            "priority_benchmark_families": [
                str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
            ],
            "priority_benchmark_family_weights": dict(priority_family_weights),
            "excluded_subsystems": [str(value).strip() for value in args.exclude_subsystem if str(value).strip()],
            "record_scope": {
                "protocol": "autonomous",
                "campaign_match_id": campaign_match_id,
                "cycle_log_start_index": cycle_log_start_index,
                "records_considered": len(campaign_records),
                "decision_records_considered": len(decision_records),
                "cycle_ids": sorted(campaign_cycle_ids),
            },
            "cycles_requested": max(1, args.cycles),
            "completed_runs": len(runs),
            "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
            "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
            "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
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
            "recent_decisions": recent_decisions,
            "recent_runtime_managed_decisions": production_decisions[-10:],
            "recent_production_decisions": production_decisions[-10:],
            "recent_non_runtime_decisions": non_runtime_managed_decisions[-10:],
            "runs": runs,
        }
        atomic_write_json(report_path, report)
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
                    "priority_benchmark_families": [
                        str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
                    ],
                    "priority_benchmark_family_weights": dict(priority_family_weights),
                    "cycles_requested": max(1, args.cycles),
                    "completed_runs": len(runs),
                    "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
                    "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
                    "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
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
        )
        print(report_path)
    except KeyboardInterrupt:
        interrupted_report = {
            "spec_version": "asi_v1",
            "report_kind": "improvement_campaign_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "campaign_label": campaign_label,
            "campaign_match_id": campaign_match_id,
            "adaptive_search": bool(args.adaptive_search),
            "task_limit": max(0, args.task_limit),
            "priority_benchmark_families": [
                str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
            ],
            "priority_benchmark_family_weights": dict(priority_family_weights),
            "excluded_subsystems": [str(value).strip() for value in args.exclude_subsystem if str(value).strip()],
            "cycles_requested": max(1, args.cycles),
            "completed_runs": len(runs),
            "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
            "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
            "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
            "status": "interrupted",
            "reason": (
                f"received signal {signal.Signals(received_signal['value']).name}"
                if int(received_signal["value"]) > 0
                else "operator interrupted repeated improvement campaign"
            ),
            "runs": runs,
        }
        atomic_write_json(report_path, interrupted_report)
        print(report_path)
        raise SystemExit(130)
    finally:
        restore_signal_handlers()


if __name__ == "__main__":
    main()
