from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
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
from agent_kernel.ops.improvement_reporting import semantic_progress_state
from agent_kernel.ops.runtime_supervision import (
    atomic_write_json,
    install_termination_handlers,
    spawn_process_group,
    terminate_process_tree,
)
from agent_kernel.extensions.trust import build_unattended_trust_ledger

REPEATED_UNSAMPLED_PRIORITY_WEIGHT_STEP = 1.0
REPEATED_UNSAMPLED_PRIORITY_TASK_LIMIT_STEP = 1
_REPEATED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS = 120.0
_REPEATED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS = 240.0
_REPEATED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS = 180.0
_REPEATED_CHILD_GENERATED_FAILURE_ACTIVE_GRACE_SECONDS = 180.0
_REPEATED_CHILD_GENERATED_FAILURE_COMPLETION_GRACE_SECONDS = 180.0
_REPEATED_CHILD_METRICS_FINALIZE_GRACE_SECONDS = 120.0
_REPEATED_CHILD_FINALIZE_GRACE_SECONDS = 120.0
_REPEATED_CHILD_APPLY_DECISION_GRACE_SECONDS = 120.0
_REPEATED_CHILD_PREVIEW_COMPLETION_GRACE_SECONDS = 120.0
_REPEATED_CHILD_OBSERVE_HANDOFF_GRACE_SECONDS = 120.0
_RETRIEVAL_RETAINED_GAIN_REASON_SNIPPETS = (
    "improved broad coding-family support without regressing the base lane",
    "broadened coding-family support without regressing the base lane",
    "strengthened complementary retrieval support without regressing the base lane",
    "increased verified long-horizon trusted-retrieval carryover without regressing the base lane",
    "preserved verified long-horizon trusted-retrieval carryover without regressing the base lane",
    "increased trusted retrieval usage without regressing the base lane",
)
_SUPPORT_RETAINED_GAIN_REASON_SNIPPETS = _RETRIEVAL_RETAINED_GAIN_REASON_SNIPPETS + (
    "improved broad coding-family support without regressing verified task performance",
    "broadened coding-family support without regressing verified task performance",
    "strengthened complementary support without regressing the base lane",
    "preserved complementary support without regressing the base lane",
)
_REPEATED_PHASE_A_LANE_ORDER = (
    "lane_runtime_authority",
    "lane_decision_closure",
    "lane_trust_and_carryover",
    "lane_repo_generalization",
    "lane_strategy_memory",
)
_DEFAULT_EXTERNAL_TASK_MANIFEST_BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "task_manifests"
    / "default_external_breadth_tasks.json"
)
_REPEATED_LANE_PACKET_SPECS: dict[str, dict[str, object]] = {
    "lane_runtime_authority": {
        "owned_paths": [
            "scripts/run_repeated_improvement_cycles.py",
            "tests/test_finalize_cycle.py",
        ],
        "relevant_tests": ["tests/test_finalize_cycle.py"],
        "done_when": "repeated improvement runs preserve authoritative status and report surfaces across interruptions",
    },
    "lane_decision_closure": {
        "owned_paths": [
            "agent_kernel/cycle_runner.py",
            "scripts/run_repeated_improvement_cycles.py",
            "tests/test_finalize_cycle.py",
        ],
        "relevant_tests": ["tests/test_finalize_cycle.py"],
        "done_when": "runtime-managed child decisions convert into retained gains more often",
    },
    "lane_trust_and_carryover": {
        "owned_paths": [
            "agent_kernel/extensions/improvement/retrieval_improvement.py",
            "scripts/run_repeated_improvement_cycles.py",
            "tests/test_improvement.py",
        ],
        "relevant_tests": [
            "tests/test_improvement.py",
            "tests/test_finalize_cycle.py",
        ],
        "done_when": "required-family trust breadth and clean task-root breadth are both closed",
    },
    "lane_repo_generalization": {
        "owned_paths": [
            "scripts/run_repeated_improvement_cycles.py",
            "scripts/run_autonomous_compounding_check.py",
            "tests/test_autonomous_compounding.py",
        ],
        "relevant_tests": ["tests/test_autonomous_compounding.py"],
        "done_when": "held-out or external open-world slices exist and remain non-regressing against the retained baseline",
    },
    "lane_strategy_memory": {
        "owned_paths": [
            "agent_kernel/extensions/improvement/planner_runtime_state.py",
            "scripts/run_repeated_improvement_cycles.py",
            "tests/test_improvement.py",
        ],
        "relevant_tests": ["tests/test_improvement.py"],
        "done_when": "strategy memory and resource lineage expose retained mutation pressure instead of ad hoc local wins",
    },
}


def _default_external_task_manifests_paths(*, repo_root: Path | None = None) -> tuple[str, ...]:
    bundle_path = (
        Path(repo_root) / "datasets" / "task_manifests" / "default_external_breadth_tasks.json"
        if repo_root is not None
        else _DEFAULT_EXTERNAL_TASK_MANIFEST_BUNDLE
    )
    if bundle_path.is_file():
        return (str(bundle_path),)
    return ()


def _effective_external_task_manifests_paths(
    config: KernelConfig,
    *,
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    explicit = tuple(
        str(path).strip()
        for path in getattr(config, "external_task_manifests_paths", ())
        if str(path).strip()
    )
    if explicit:
        return explicit
    return _default_external_task_manifests_paths(repo_root=repo_root)


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


def _parse_iso_datetime(value: object) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_tolbert_runtime_summary(
    summary: Mapping[str, object] | None,
    *,
    configured_to_use_tolbert: bool | None = None,
) -> dict[str, object]:
    payload = dict(summary) if isinstance(summary, Mapping) else {}
    configured = (
        bool(payload.get("configured_to_use_tolbert", False))
        if configured_to_use_tolbert is None
        else bool(configured_to_use_tolbert)
    )
    normalized = {
        "configured_to_use_tolbert": configured,
        "stages_attempted": _ordered_unique_strings(payload.get("stages_attempted", [])),
        "successful_tolbert_stages": _ordered_unique_strings(payload.get("successful_tolbert_stages", [])),
        "startup_failure_stages": _ordered_unique_strings(payload.get("startup_failure_stages", [])),
        "recovered_without_tolbert_stages": _ordered_unique_strings(payload.get("recovered_without_tolbert_stages", [])),
        "bypassed_stages": _ordered_unique_strings(payload.get("bypassed_stages", [])),
    }
    normalized["startup_failure_count"] = len(normalized["startup_failure_stages"])
    if normalized["startup_failure_stages"]:
        outcome = "failed_recovered"
    elif normalized["successful_tolbert_stages"]:
        outcome = "succeeded"
    elif normalized["bypassed_stages"] or not configured:
        outcome = "bypassed"
    elif normalized["stages_attempted"]:
        outcome = "pending"
    else:
        outcome = "not_exercised"
    normalized["outcome"] = outcome
    normalized["used_tolbert_successfully"] = bool(normalized["successful_tolbert_stages"])
    normalized["recovered_without_tolbert"] = bool(normalized["recovered_without_tolbert_stages"])
    normalized["bypassed"] = outcome == "bypassed"
    return normalized


def _merge_tolbert_runtime_summaries(*summaries: object) -> dict[str, object]:
    configured = False
    stages_attempted: list[str] = []
    successful: list[str] = []
    startup_failures: list[str] = []
    recovered: list[str] = []
    bypassed: list[str] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        normalized = _normalize_tolbert_runtime_summary(summary)
        configured = configured or bool(normalized.get("configured_to_use_tolbert", False))
        stages_attempted = _ordered_unique_strings(stages_attempted, normalized.get("stages_attempted", []))
        successful = _ordered_unique_strings(successful, normalized.get("successful_tolbert_stages", []))
        startup_failures = _ordered_unique_strings(startup_failures, normalized.get("startup_failure_stages", []))
        recovered = _ordered_unique_strings(recovered, normalized.get("recovered_without_tolbert_stages", []))
        bypassed = _ordered_unique_strings(bypassed, normalized.get("bypassed_stages", []))
    return _normalize_tolbert_runtime_summary(
        {
            "configured_to_use_tolbert": configured,
            "stages_attempted": stages_attempted,
            "successful_tolbert_stages": successful,
            "startup_failure_stages": startup_failures,
            "recovered_without_tolbert_stages": recovered,
            "bypassed_stages": bypassed,
        }
    )


def _tolbert_runtime_status_summary(runs: list[dict[str, object]], active_cycle_run: Mapping[str, object] | None = None) -> dict[str, object]:
    outcome_counts = {
        "succeeded": 0,
        "failed_recovered": 0,
        "bypassed": 0,
        "not_exercised": 0,
        "pending": 0,
    }
    startup_failures = 0
    configured_runs = 0
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        summary = _normalize_tolbert_runtime_summary(run.get("tolbert_runtime_summary"))
        if bool(summary.get("configured_to_use_tolbert", False)):
            configured_runs += 1
        outcome = str(summary.get("outcome", "")).strip() or "not_exercised"
        if outcome not in outcome_counts:
            outcome = "not_exercised"
        outcome_counts[outcome] += 1
        startup_failures += int(summary.get("startup_failure_count", 0) or 0)
    active_summary = _normalize_tolbert_runtime_summary(active_cycle_run.get("tolbert_runtime_summary")) if isinstance(active_cycle_run, Mapping) else {}
    return {
        "configured_runs": configured_runs,
        "startup_failure_count": startup_failures,
        "outcome_counts": outcome_counts,
        "active": active_summary,
    }


def _record_state(record: Mapping[str, object] | None) -> str:
    if not isinstance(record, Mapping):
        return ""
    for key in ("state", "final_state"):
        token = str(record.get(key, "")).strip()
        if token:
            return token
    return ""


def _record_metrics_summary(record: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(record, Mapping):
        return {}
    metrics = record.get("metrics_summary", {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _record_float(record: Mapping[str, object] | None, key: str) -> float | None:
    if not isinstance(record, Mapping):
        return None
    value = record.get(key)
    if value is None:
        value = _record_metrics_summary(record).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_phase_gate_passed(record: Mapping[str, object] | None) -> bool | None:
    if not isinstance(record, Mapping):
        return None
    if "phase_gate_passed" in record:
        return bool(record.get("phase_gate_passed"))
    metrics = _record_metrics_summary(record)
    if "phase_gate_passed" in metrics:
        return bool(metrics.get("phase_gate_passed"))
    return None


def _record_reason_text(record: Mapping[str, object] | None) -> str:
    if not isinstance(record, Mapping):
        return ""
    for key in ("final_reason", "reason"):
        token = str(record.get(key, "")).strip()
        if token:
            return token
    return str(_record_metrics_summary(record).get("final_reason", "")).strip()


def _record_is_non_regressive_retain(record: Mapping[str, object] | None) -> bool:
    if _record_state(record) != "retain":
        return False
    phase_gate_passed = _record_phase_gate_passed(record)
    if phase_gate_passed is False:
        return False
    baseline_pass_rate = _record_float(record, "baseline_pass_rate")
    candidate_pass_rate = _record_float(record, "candidate_pass_rate")
    if (
        baseline_pass_rate is not None
        and candidate_pass_rate is not None
        and candidate_pass_rate < baseline_pass_rate
    ):
        return False
    baseline_average_steps = _record_float(record, "baseline_average_steps")
    candidate_average_steps = _record_float(record, "candidate_average_steps")
    if (
        baseline_average_steps is not None
        and candidate_average_steps is not None
        and candidate_average_steps > baseline_average_steps
    ):
        return False
    return True


def _record_support_signal_summary(record: Mapping[str, object] | None) -> dict[str, object]:
    metrics = _record_metrics_summary(record)
    family_signal_present = False
    positive_signal = False
    negative_signal = False
    transition_model_scoring_control_delta_count = 0
    try:
        transition_model_scoring_control_delta_count = max(
            0,
            int(metrics.get("transition_model_scoring_control_delta_count", 0) or 0),
        )
    except (TypeError, ValueError):
        transition_model_scoring_control_delta_count = 0
    for key in ("family_pass_rate_delta", "generated_family_pass_rate_delta"):
        payload = metrics.get(key, {})
        if not isinstance(payload, Mapping):
            continue
        for value in payload.values():
            try:
                delta = float(value)
            except (TypeError, ValueError):
                continue
            family_signal_present = True
            if delta > 0.0:
                positive_signal = True
            elif delta < 0.0:
                negative_signal = True
    failure_recovery_delta = _record_float(record, "failure_recovery_pass_rate_delta")
    if failure_recovery_delta is not None:
        family_signal_present = True
        if failure_recovery_delta > 0.0:
            positive_signal = True
        elif failure_recovery_delta < 0.0:
            negative_signal = True
    tolbert_summary = _normalize_tolbert_runtime_summary(
        metrics.get("tolbert_runtime_summary", (record or {}).get("tolbert_runtime_summary", {}))
    )
    return {
        "family_signal_present": family_signal_present,
        "positive_signal": positive_signal,
        "negative_signal": negative_signal,
        "used_tolbert_successfully": bool(tolbert_summary.get("used_tolbert_successfully", False)),
        "transition_model_scoring_control_delta_count": transition_model_scoring_control_delta_count,
    }


def _record_counts_as_reject_learning_opportunity(record: Mapping[str, object] | None) -> bool:
    if _record_state(record) != "reject":
        return False
    phase_gate_passed = _record_phase_gate_passed(record)
    if phase_gate_passed is False:
        return False
    baseline_pass_rate = _record_float(record, "baseline_pass_rate")
    candidate_pass_rate = _record_float(record, "candidate_pass_rate")
    if (
        baseline_pass_rate is not None
        and candidate_pass_rate is not None
        and candidate_pass_rate < baseline_pass_rate
    ):
        return False
    baseline_average_steps = _record_float(record, "baseline_average_steps")
    candidate_average_steps = _record_float(record, "candidate_average_steps")
    if (
        baseline_average_steps is not None
        and candidate_average_steps is not None
        and candidate_average_steps > baseline_average_steps
    ):
        return False
    support_signal = _record_support_signal_summary(record)
    return bool(support_signal.get("positive_signal", False) or support_signal.get("family_signal_present", False))


def _record_counts_as_retained_gain(record: Mapping[str, object] | None) -> bool:
    if not _record_is_non_regressive_retain(record):
        return False
    baseline_pass_rate = _record_float(record, "baseline_pass_rate")
    candidate_pass_rate = _record_float(record, "candidate_pass_rate")
    pass_delta: float | None = None
    if baseline_pass_rate is not None and candidate_pass_rate is not None:
        pass_delta = candidate_pass_rate - baseline_pass_rate
    if pass_delta is not None and pass_delta > 0.0:
        return True
    support_signal = _record_support_signal_summary(record)
    if bool(support_signal.get("positive_signal", False)):
        return True
    return _record_counts_as_support_retained_gain(record)


def _record_counts_as_support_retained_gain(record: Mapping[str, object] | None) -> bool:
    if not _record_is_non_regressive_retain(record):
        return False
    reason = _record_reason_text(record).lower()
    if any(snippet in reason for snippet in _SUPPORT_RETAINED_GAIN_REASON_SNIPPETS):
        return True
    subsystem = str((record or {}).get("subsystem", "")).strip()
    support_signal = _record_support_signal_summary(record)
    if subsystem == "tolbert_model" and bool(support_signal.get("used_tolbert_successfully", False)):
        return True
    if (
        subsystem == "transition_model"
        and int(support_signal.get("transition_model_scoring_control_delta_count", 0) or 0) > 0
    ):
        return True
    return bool(
        support_signal.get("family_signal_present", False)
        and not support_signal.get("negative_signal", False)
        and subsystem in {"retrieval", "tolbert_model", "transition_model"}
    )


def _runtime_env(config: KernelConfig) -> dict[str, str]:
    return config.to_env()


def _initialize_child_runtime_database(env: Mapping[str, str]) -> None:
    storage_backend = str(env.get("AGENT_KERNEL_STORAGE_BACKEND", "sqlite")).strip().lower()
    if storage_backend != "sqlite":
        return
    db_path = str(env.get("AGENT_KERNEL_RUNTIME_DATABASE_PATH", "")).strip()
    if not db_path:
        return
    from agent_kernel.storage import SQLiteKernelStore

    SQLiteKernelStore(Path(db_path))


def _parse_progress_fields(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if not normalized:
        return {}
    parsed: dict[str, object] = {}
    observe_start_match = re.search(r"phase=observe\s+start(?:\s|$)", normalized)
    if observe_start_match:
        parsed["last_progress_phase"] = "observe"
        parsed["current_task"] = {}
        parsed["selected_subsystem"] = ""
        parsed["pending_decision_state"] = ""
        parsed["preview_state"] = ""
        parsed["finalize_phase"] = ""
        parsed["last_candidate_variant"] = ""
        parsed["last_candidate_artifact_path"] = ""
    phase_match = re.search(r"phase=(?P<phase>[A-Za-z0-9_:-]+)", normalized)
    if phase_match:
        parsed["last_progress_phase"] = str(phase_match.group("phase")).strip()
    phase_complete_match = re.search(r"phase=(?P<phase>[A-Za-z0-9_:-]+)\s+complete(?:\s|$)", normalized)
    if phase_complete_match:
        phase_name = str(phase_complete_match.group("phase")).strip()
        if phase_name:
            parsed[f"{phase_name}_completed"] = True
    observe_match = re.search(
        r"observe complete passed=(?P<passed>\d+)/(?P<total>\d+) pass_rate=(?P<pass_rate>[0-9.]+) "
        r"generated_pass_rate=(?P<generated_pass_rate>[0-9.]+)",
        normalized,
    )
    if observe_match:
        parsed["last_progress_phase"] = "observe"
        parsed["observe_summary"] = {
            "passed": int(observe_match.group("passed")),
            "total": int(observe_match.group("total")),
            "pass_rate": float(observe_match.group("pass_rate")),
            "generated_pass_rate": float(observe_match.group("generated_pass_rate")),
        }
        parsed["observe_completed"] = True
        parsed["current_task"] = {}
    selected_match = re.search(r"campaign \d+/\d+ select subsystem=(?P<subsystem>[a-z_]+)", normalized)
    if selected_match:
        parsed["last_progress_phase"] = "campaign_select"
        parsed["selected_subsystem"] = str(selected_match.group("subsystem")).strip()
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
    phase_total_match = re.search(r"phase=(?P<phase>[A-Za-z0-9_:-]+) total=(?P<total>\d+)", normalized)
    if phase_total_match:
        phase_name = str(phase_total_match.group("phase")).strip()
        parsed[f"{phase_name}_total"] = int(phase_total_match.group("total"))
        parsed[f"{phase_name}_completed"] = False
    variant_search_match = re.search(
        r"variant_search start subsystem=(?P<subsystem>[a-z_]+) selected_variants=(?P<count>\d+) "
        r"variant_ids=(?P<variant_ids>[A-Za-z0-9_,.-]+)",
        normalized,
    )
    if variant_search_match:
        parsed["last_progress_phase"] = "variant_search"
        parsed["selected_subsystem"] = str(variant_search_match.group("subsystem")).strip()
        parsed["selected_variants"] = int(variant_search_match.group("count"))
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
        parsed["selected_variant_ids"] = [
            token
            for token in str(variant_search_match.group("variant_ids")).split(",")
            if str(token).strip()
        ]
    generate_start_match = re.search(
        r"variant generate start subsystem=(?P<subsystem>[a-z_]+) "
        r"variant=(?P<variant>[A-Za-z0-9_:-]+) rank=(?P<rank>\d+)/(?P<total>\d+)",
        normalized,
    )
    if generate_start_match:
        parsed["last_progress_phase"] = "variant_generate"
        parsed["selected_subsystem"] = str(generate_start_match.group("subsystem")).strip()
        parsed["last_candidate_variant"] = str(generate_start_match.group("variant")).strip()
        parsed["variant_rank"] = int(generate_start_match.group("rank"))
        parsed["variant_total"] = int(generate_start_match.group("total"))
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
    generate_heartbeat_match = re.search(
        r"variant generate heartbeat subsystem=(?P<subsystem>[a-z_]+) "
        r"variant=(?P<variant>[A-Za-z0-9_:-]+)\s+stage=(?P<stage>[A-Za-z0-9_:-]+)",
        normalized,
    )
    if generate_heartbeat_match:
        parsed["last_progress_phase"] = "variant_generate"
        parsed["selected_subsystem"] = str(generate_heartbeat_match.group("subsystem")).strip()
        parsed["last_candidate_variant"] = str(generate_heartbeat_match.group("variant")).strip()
        parsed["variant_generate_stage"] = str(generate_heartbeat_match.group("stage")).strip()
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
    generate_match = re.search(
        r"variant generate complete subsystem=(?P<subsystem>[a-z_]+) "
        r"variant=(?P<variant>[A-Za-z0-9_:-]+) artifact=(?P<artifact>\S+)",
        normalized,
    )
    if generate_match:
        parsed["last_progress_phase"] = "variant_generate"
        parsed["selected_subsystem"] = str(generate_match.group("subsystem")).strip()
        parsed["last_candidate_variant"] = str(generate_match.group("variant")).strip()
        parsed["last_candidate_artifact_path"] = str(generate_match.group("artifact")).strip()
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
    generate_failed_match = re.search(
        r"variant generate failed subsystem=(?P<subsystem>[a-z_]+) "
        r"variant=(?P<variant>[A-Za-z0-9_:-]+) reason=(?P<reason>\S+)",
        normalized,
    )
    if generate_failed_match:
        parsed["last_progress_phase"] = "variant_generate"
        parsed["selected_subsystem"] = str(generate_failed_match.group("subsystem")).strip()
        parsed["last_candidate_variant"] = str(generate_failed_match.group("variant")).strip()
        parsed["variant_generate_failure_reason"] = str(generate_failed_match.group("reason")).strip()
        parsed["preview_state"] = ""
        parsed["current_task"] = {}
    finalize_match = re.search(
        r"finalize phase=(?P<phase>[A-Za-z0-9_:-]+) subsystem=(?P<subsystem>[a-z_]+)",
        normalized,
    )
    if finalize_match:
        parsed["selected_subsystem"] = str(finalize_match.group("subsystem")).strip()
        parsed["finalize_phase"] = str(finalize_match.group("phase")).strip()
        parsed["last_progress_phase"] = str(finalize_match.group("phase")).strip()
        parsed["current_task"] = {}
        finalize_phase = str(finalize_match.group("phase")).strip()
        if finalize_phase.startswith("preview") or finalize_phase.endswith("_eval"):
            parsed["preview_state"] = ""
        if finalize_phase.endswith("_eval"):
            tolbert_stage = finalize_phase[: -len("_eval")]
            if tolbert_stage:
                parsed["tolbert_runtime_summary"] = {
                    "stages_attempted": [tolbert_stage],
                    "outcome": "pending",
                }
    tolbert_retry_match = re.search(
        r"finalize phase=(?P<phase>[A-Za-z0-9_:-]+)_retry subsystem=(?P<subsystem>[a-z_]+) "
        r"reason=tolbert_startup_failure use_tolbert_context=0",
        normalized,
    )
    if tolbert_retry_match:
        stage_name = str(tolbert_retry_match.group("phase")).strip()
        parsed["selected_subsystem"] = str(tolbert_retry_match.group("subsystem")).strip()
        parsed["tolbert_runtime_summary"] = {
            "configured_to_use_tolbert": True,
            "stages_attempted": [stage_name],
            "startup_failure_stages": [stage_name],
            "recovered_without_tolbert_stages": [stage_name],
            "bypassed_stages": [stage_name],
            "startup_failure_count": 1,
            "outcome": "failed_recovered",
        }
    preview_state_match = re.search(
        r"preview_complete subsystem=(?P<subsystem>[a-z_]+) preview_state=(?P<state>[a-z_]+)",
        normalized,
    )
    if preview_state_match:
        parsed["selected_subsystem"] = str(preview_state_match.group("subsystem")).strip()
        parsed["preview_state"] = str(preview_state_match.group("state")).strip()
    preview_complete_decision_match = re.search(
        r"preview complete subsystem=(?P<subsystem>[a-z_]+)(?:\s+variant=(?P<variant>[A-Za-z0-9_:-]+))?"
        r"\s+state=(?P<state>[a-z_]+)",
        normalized,
    )
    if preview_complete_decision_match:
        parsed["selected_subsystem"] = str(preview_complete_decision_match.group("subsystem")).strip()
        parsed["preview_state"] = str(preview_complete_decision_match.group("state")).strip()
        variant_name = str(preview_complete_decision_match.group("variant") or "").strip()
        if variant_name:
            parsed["last_candidate_variant"] = variant_name
    apply_decision_match = re.search(
        r"apply_decision subsystem=(?P<subsystem>[a-z_]+) state=(?P<state>[a-z_]+)",
        normalized,
    )
    if apply_decision_match:
        parsed["selected_subsystem"] = str(apply_decision_match.group("subsystem")).strip()
        parsed["pending_decision_state"] = str(apply_decision_match.group("state")).strip()
    done_match = re.search(
        r"phase=done subsystem=(?P<subsystem>[a-z_]+) state=(?P<state>[a-z_]+)",
        normalized,
    )
    if done_match:
        parsed["selected_subsystem"] = str(done_match.group("subsystem")).strip()
        parsed["pending_decision_state"] = str(done_match.group("state")).strip()
    phase_task_match = re.search(
        r"phase=(?P<phase>[A-Za-z0-9_:-]+) task (?P<index>\d+)/(?P<total>\d+) "
        r"(?P<task_id>\S+) family=(?P<family>[a-z_]+)",
        normalized,
    )
    if phase_task_match:
        phase_name = str(phase_task_match.group("phase")).strip()
        current_task = {
            "index": int(phase_task_match.group("index")),
            "total": int(phase_task_match.group("total")),
            "task_id": str(phase_task_match.group("task_id")).strip(),
            "family": str(phase_task_match.group("family")).strip(),
            "phase": phase_name,
        }
        task_origin_match = re.search(r"\btask_origin=(?P<task_origin>[A-Za-z0-9_:-]+)", normalized)
        if task_origin_match:
            current_task["task_origin"] = str(task_origin_match.group("task_origin")).strip()
        source_task_match = re.search(r"\bsource_task=(?P<source_task>\S+)", normalized)
        if source_task_match:
            current_task["source_task"] = str(source_task_match.group("source_task")).strip()
        parsed["current_task"] = current_task
        if phase_name:
            parsed["last_progress_phase"] = phase_name
    else:
        task_match = re.search(
            r"task (?P<index>\d+)/(?P<total>\d+) (?P<task_id>\S+) family=(?P<family>[a-z_]+)",
            normalized,
        )
        if task_match:
            current_task = {
                "index": int(task_match.group("index")),
                "total": int(task_match.group("total")),
                "task_id": str(task_match.group("task_id")).strip(),
                "family": str(task_match.group("family")).strip(),
            }
            task_origin_match = re.search(r"\btask_origin=(?P<task_origin>[A-Za-z0-9_:-]+)", normalized)
            if task_origin_match:
                current_task["task_origin"] = str(task_origin_match.group("task_origin")).strip()
            source_task_match = re.search(r"\bsource_task=(?P<source_task>\S+)", normalized)
            if source_task_match:
                current_task["source_task"] = str(source_task_match.group("source_task")).strip()
            parsed["current_task"] = current_task
    return parsed


def _parse_cognitive_progress_fields(line: str) -> dict[str, object]:
    normalized = str(line).strip()
    if not normalized or "cognitive_stage=" not in normalized:
        return {}
    stage_match = re.search(r"cognitive_stage=(?P<stage>[A-Za-z0-9_:-]+)", normalized)
    if not stage_match:
        return {}
    payload: dict[str, object] = {
        "stage": str(stage_match.group("stage")).strip(),
        "recorded_at": round(time.time(), 4),
    }
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


def _cognitive_stage_step_index(cognitive_stage: Mapping[str, object] | None) -> int:
    payload = cognitive_stage if isinstance(cognitive_stage, Mapping) else {}
    try:
        return int(payload.get("step_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _updated_task_verification_state(
    *,
    current_task_verification_passed: bool | None,
    previous_cognitive_stage: Mapping[str, object] | None,
    next_cognitive_stage: Mapping[str, object] | None,
) -> bool | None:
    next_stage = next_cognitive_stage if isinstance(next_cognitive_stage, Mapping) else {}
    if "verification_passed" in next_stage:
        return bool(next_stage.get("verification_passed"))
    if current_task_verification_passed is not False:
        return current_task_verification_passed
    previous_step_index = _cognitive_stage_step_index(previous_cognitive_stage)
    next_step_index = _cognitive_stage_step_index(next_stage)
    if previous_step_index > 0 and next_step_index > previous_step_index:
        return None
    return current_task_verification_passed


def _tolbert_runtime_summary_from_cognitive_progress(
    cognitive_progress: Mapping[str, object] | None,
    *,
    active_cycle_run: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(cognitive_progress, Mapping):
        return {}
    subphase = str(cognitive_progress.get("step_subphase", "")).strip()
    decision_source = str(cognitive_progress.get("decision_source", "")).strip()
    if subphase != "tolbert_query" and "tolbert" not in decision_source:
        return {}
    run_payload = dict(active_cycle_run or {})
    current_task = run_payload.get("current_task", {})
    phase_name = ""
    if isinstance(current_task, Mapping):
        phase_name = str(current_task.get("phase", "")).strip()
    if not phase_name:
        phase_name = str(run_payload.get("last_progress_phase", "")).strip() or "observe"
    return {
        "configured_to_use_tolbert": True,
        "stages_attempted": [phase_name],
        "successful_tolbert_stages": [phase_name],
        "outcome": "succeeded",
    }


def _child_runtime_extension_plan(
    *,
    last_progress_phase: str,
    current_task: dict[str, object] | None,
    active_cycle_run: dict[str, object] | None = None,
) -> tuple[str, float]:
    normalized_phase = str(last_progress_phase).strip()
    task_payload = current_task if isinstance(current_task, dict) else {}
    task_index = int(task_payload.get("index", 0) or 0)
    task_total = int(task_payload.get("total", 0) or 0)
    task_completed = task_total > 0 and task_index >= task_total
    task_phase = str(task_payload.get("phase", "")).strip() or normalized_phase
    run_payload = active_cycle_run if isinstance(active_cycle_run, dict) else {}
    raw_task_phase = str(task_payload.get("raw_phase", "")).strip()
    raw_last_progress_phase = str(run_payload.get("raw_last_progress_phase", "")).strip()
    effective_task_phase = raw_task_phase or raw_last_progress_phase or task_phase
    if normalized_phase in {"variant_search", "variant_generate"}:
        return ("", 0.0)
    if normalized_phase == "observe" and (
        bool(run_payload.get("observe_completed", False)) or not task_payload or task_completed
    ):
        return ("observe_handoff", _REPEATED_CHILD_OBSERVE_HANDOFF_GRACE_SECONDS)
    if normalized_phase in {"preview_baseline_complete", "preview_candidate_complete", "preview_complete"}:
        return ("preview_completion", _REPEATED_CHILD_PREVIEW_COMPLETION_GRACE_SECONDS)
    if normalized_phase in {"apply_decision", "done"}:
        return ("apply_decision", _REPEATED_CHILD_APPLY_DECISION_GRACE_SECONDS)
    if effective_task_phase.startswith("generated_success") and (
        task_completed or bool(run_payload.get("generated_success_completed", False))
    ):
        return (
            "generated_success_completion",
            _REPEATED_CHILD_GENERATED_SUCCESS_COMPLETION_GRACE_SECONDS,
        )
    if effective_task_phase == "generated_failure_seed":
        if task_completed or bool(run_payload.get("generated_failure_seed_completed", False)):
            return (
                "generated_failure_seed_completion",
                _REPEATED_CHILD_GENERATED_FAILURE_SEED_COMPLETION_GRACE_SECONDS,
            )
        return (
            "generated_failure_seed_active",
            _REPEATED_CHILD_GENERATED_FAILURE_SEED_ACTIVE_GRACE_SECONDS,
        )
    if effective_task_phase == "generated_failure":
        if task_completed or bool(run_payload.get("generated_failure_completed", False)):
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
    generated_failure_seed_total = int(active_cycle_run.get("generated_failure_seed_total", 0) or 0)
    generated_failure_seed_completed = bool(active_cycle_run.get("generated_failure_seed_completed", False))
    generated_failure_total = int(active_cycle_run.get("generated_failure_total", 0) or 0)
    generated_failure_completed = bool(active_cycle_run.get("generated_failure_completed", False))
    current_task = active_cycle_run.get("current_task", {})
    if not isinstance(current_task, dict):
        current_task = {}
    current_phase = str(current_task.get("phase", "")).strip()
    generated_success_started = (
        generated_success_total > 0
        or str(last_progress_phase).startswith("generated_success")
        or current_phase == "generated_success"
    )
    generated_failure_seed_started = (
        generated_failure_seed_total > 0
        or str(last_progress_phase).startswith("generated_failure_seed")
        or current_phase == "generated_failure_seed"
    )
    generated_failure_started = (
        generated_failure_total > 0
        or str(last_progress_phase).startswith("generated_failure")
        or current_phase == "generated_failure"
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
        "generated_failure_seed_started": generated_failure_seed_started,
        "generated_failure_seed_total": generated_failure_seed_total,
        "generated_failure_seed_completed": generated_failure_seed_completed,
        "generated_failure_started": generated_failure_started,
        "generated_failure_total": generated_failure_total,
        "generated_failure_completed": generated_failure_completed,
        "last_progress_phase": last_progress_phase,
        "sampled_families_from_progress": sampled_families,
        "priority_families_without_progress_sampling": priority_unsampled,
        "candidate_generated": bool(str(active_cycle_run.get("last_candidate_artifact_path", "")).strip()),
        "productive_partial": partial_productive,
    }


def _verification_outcome_summary(active_cycle_run: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(active_cycle_run, dict):
        return {
            "verified_task_count": 0,
            "failed_task_count": 0,
            "successful_task_count": 0,
            "verified_families": [],
            "failed_families": [],
            "successful_families": [],
            "all_verified_tasks_failed": False,
        }
    successful_task_ids = _ordered_unique_strings(active_cycle_run.get("successful_verification_task_ids", []))
    failed_task_ids = _ordered_unique_strings(active_cycle_run.get("failed_verification_task_ids", []))
    task_outcomes = active_cycle_run.get("verification_outcomes_by_task", {})
    task_families = active_cycle_run.get("verification_task_families", {})
    if isinstance(task_outcomes, Mapping) and task_outcomes:
        normalized_task_outcomes: list[tuple[str, bool]] = []
        for raw_task_id, raw_outcome in task_outcomes.items():
            task_id = str(raw_task_id).strip()
            if not task_id:
                continue
            normalized_task_outcomes.append((task_id, bool(raw_outcome)))
        verified_task_ids = [task_id for task_id, _ in normalized_task_outcomes]
        successful_task_ids = [task_id for task_id, outcome in normalized_task_outcomes if outcome]
        failed_task_ids = [task_id for task_id, outcome in normalized_task_outcomes if not outcome]
    else:
        successful_task_set = set(successful_task_ids)
        failed_task_ids = [task_id for task_id in failed_task_ids if task_id not in successful_task_set]
        verified_task_ids = _ordered_unique_strings(
            active_cycle_run.get("verified_task_ids", []),
            failed_task_ids,
            successful_task_ids,
        )
    verified_families = _ordered_unique_strings(active_cycle_run.get("verified_families", []))
    failed_families = _ordered_unique_strings(active_cycle_run.get("failed_verification_families", []))
    successful_families = _ordered_unique_strings(active_cycle_run.get("successful_verification_families", []))
    if isinstance(task_families, Mapping) and task_families:
        derived_verified_families = _ordered_unique_strings(
            [
                str(task_families.get(task_id, "")).strip()
                for task_id in verified_task_ids
                if str(task_families.get(task_id, "")).strip()
            ]
        )
        derived_failed_families = _ordered_unique_strings(
            [
                str(task_families.get(task_id, "")).strip()
                for task_id in failed_task_ids
                if str(task_families.get(task_id, "")).strip()
            ]
        )
        derived_successful_families = _ordered_unique_strings(
            [
                str(task_families.get(task_id, "")).strip()
                for task_id in successful_task_ids
                if str(task_families.get(task_id, "")).strip()
            ]
        )
        has_task_level_family_evidence = bool(verified_task_ids or successful_task_ids or failed_task_ids)
        verified_families = derived_verified_families or verified_families
        failed_families = (
            derived_failed_families if has_task_level_family_evidence else failed_families
        )
        successful_families = (
            derived_successful_families if has_task_level_family_evidence else successful_families
        )
    verified_task_count = len(verified_task_ids)
    failed_task_count = len(failed_task_ids)
    successful_task_count = len(successful_task_ids)
    return {
        "verified_task_count": verified_task_count,
        "failed_task_count": failed_task_count,
        "successful_task_count": successful_task_count,
        "verified_task_ids": verified_task_ids,
        "failed_task_ids": failed_task_ids,
        "successful_task_ids": successful_task_ids,
        "verified_families": verified_families,
        "failed_families": failed_families,
        "successful_families": successful_families,
        "all_verified_tasks_failed": (
            verified_task_count > 0
            and failed_task_count >= verified_task_count
            and successful_task_count <= 0
        ),
    }


def _reset_active_cycle_run_verification_state(active_cycle_run: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(active_cycle_run, dict):
        active_cycle_run = {}
    cleaned = dict(active_cycle_run)
    cleaned["verified_task_ids"] = []
    cleaned["failed_verification_task_ids"] = []
    cleaned["successful_verification_task_ids"] = []
    cleaned["verified_families"] = []
    cleaned["failed_verification_families"] = []
    cleaned["successful_verification_families"] = []
    cleaned["verification_outcomes_by_task"] = {}
    cleaned["verification_task_families"] = {}
    cleaned["verification_outcome_summary"] = _verification_outcome_summary({})
    cleaned.pop("current_task_verification_passed", None)
    return cleaned


def _structured_child_decision_from_run(
    *,
    run: Mapping[str, object] | None,
    priority_benchmark_families: list[str],
    reason: str = "",
) -> dict[str, object]:
    payload = run if isinstance(run, Mapping) else {}
    if int(payload.get("runtime_managed_decisions", 0) or 0) > 0:
        return {}
    verification_summary = payload.get("verification_outcome_summary", {})
    if not isinstance(verification_summary, Mapping):
        verification_summary = {}
    if not bool(verification_summary.get("all_verified_tasks_failed", False)):
        return {}
    partial_progress = payload.get("partial_progress", {})
    if not isinstance(partial_progress, Mapping):
        partial_progress = {}
    sampled_progress_families = _ordered_unique_strings(partial_progress.get("sampled_families_from_progress", []))
    sampled_priority_families = [
        family for family in sampled_progress_families if family in set(priority_benchmark_families)
    ]
    if not sampled_priority_families:
        return {}
    if priority_benchmark_families and len(sampled_priority_families) < len(priority_benchmark_families):
        generated_success_started = bool(partial_progress.get("generated_success_started", False))
        if not generated_success_started:
            return {}
    current_task = payload.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    reason_text = str(reason).strip() or str(payload.get("stderr", "")).strip() or (
        "all verified primary tasks failed before a runtime-managed retention decision was produced"
    )
    failed_task_ids = _ordered_unique_strings(verification_summary.get("failed_task_ids", []))
    failed_families = _ordered_unique_strings(verification_summary.get("failed_families", []))
    subsystem = str(payload.get("selected_subsystem", "")).strip() or str(payload.get("subsystem", "")).strip()
    if not subsystem:
        subsystem = str(current_task.get("family", "")).strip()
    return {
        "cycle_id": str(payload.get("cycle_id", "")).strip(),
        "subsystem": subsystem,
        "state": "reject",
        "decision_kind": "structured_child_outcome",
        "decision_source": "structured_child_verification_envelope",
        "reason_code": "all_primary_verifications_failed",
        "reason": reason_text,
        "artifact_kind": "structured_child_outcome",
        "artifact_path": "",
        "metrics_summary": {
            "phase_gate_passed": False,
            "all_verified_tasks_failed": True,
            "verified_task_count": int(verification_summary.get("verified_task_count", 0) or 0),
            "failed_task_count": int(verification_summary.get("failed_task_count", 0) or 0),
            "successful_task_count": int(verification_summary.get("successful_task_count", 0) or 0),
            "sampled_priority_families": sampled_priority_families,
            "failed_verification_families": failed_families,
            "failed_verification_task_ids": failed_task_ids,
            "generated_success_started": bool(partial_progress.get("generated_success_started", False)),
            "generated_success_completed": bool(partial_progress.get("generated_success_completed", False)),
            "partial_productive": bool(payload.get("partial_productive", False)),
        },
    }


def _apply_structured_child_decision(
    *,
    run: dict[str, object],
    priority_benchmark_families: list[str],
    reason: str = "",
) -> dict[str, object]:
    decision = _structured_child_decision_from_run(
        run=run,
        priority_benchmark_families=priority_benchmark_families,
        reason=reason,
    )
    if not decision:
        return run
    run["structured_child_decision"] = dict(decision)
    run["non_runtime_managed_decisions"] = max(1, int(run.get("non_runtime_managed_decisions", 0) or 0))
    run["decision_records_considered"] = max(1, int(run.get("decision_records_considered", 0) or 0))
    run["final_state"] = "reject"
    if not str(run.get("final_reason", "")).strip():
        run["final_reason"] = str(decision.get("reason", "")).strip()
    run["decision_conversion_state"] = "non_runtime_managed"
    run["decision_state"] = _decision_state_from_run(run)
    decision["decision_state"] = dict(run["decision_state"])
    run["structured_child_decision"] = dict(decision)
    return run


def _resolved_active_cycle_phase(active_cycle_run: dict[str, object] | None) -> str:
    if not isinstance(active_cycle_run, dict):
        return ""
    finalize_phase = str(active_cycle_run.get("finalize_phase", "")).strip()
    current_task = active_cycle_run.get("current_task", {})
    current_phase = ""
    if isinstance(current_task, Mapping):
        current_phase = str(current_task.get("phase", "")).strip()
    if finalize_phase == "holdout_eval":
        return finalize_phase
    if current_phase == "primary" or current_phase == "observe" or current_phase.startswith("generated_"):
        return current_phase
    if finalize_phase:
        return finalize_phase
    if current_phase:
        return current_phase
    return finalize_phase or str(active_cycle_run.get("last_progress_phase", "")).strip()


def _active_cycle_semantic_progress_state(
    active_cycle_run: dict[str, object] | None,
    *,
    now: float | None = None,
    max_progress_stall_seconds: float = 0.0,
    max_runtime_seconds: float = 0.0,
) -> dict[str, object]:
    if not isinstance(active_cycle_run, dict):
        return {}
    started_at = float(active_cycle_run.get("started_at", 0.0) or 0.0)
    last_event_at = float(active_cycle_run.get("last_event_at", started_at) or started_at or 0.0)
    current_task = active_cycle_run.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    observe_summary = active_cycle_run.get("observe_summary", {})
    if not isinstance(observe_summary, Mapping):
        observe_summary = {}
    current_time = float(now if now is not None else last_event_at or time.time())
    last_progress_at = float(
        active_cycle_run.get("last_progress_at", 0.0) or last_event_at or current_time
    )
    if bool(active_cycle_run.get("child_exit_detected", False)):
        runtime_elapsed_seconds = max(0.0, current_time - float(started_at or current_time))
        progress_silence_seconds = max(0.0, current_time - last_progress_at)
        prior_phase = _resolved_active_cycle_phase(active_cycle_run)
        detail = "child exited; draining buffered output"
        if prior_phase:
            detail = f"child exited; draining buffered output from {prior_phase}"
        return {
            "phase": "buffer_drain_after_exit",
            "phase_family": "finalize",
            "status": "exited",
            "progress_class": "buffer_drain",
            "decision_distance": "closed",
            "progress_silence_seconds": progress_silence_seconds,
            "runtime_elapsed_seconds": runtime_elapsed_seconds,
            "detail": detail,
        }
    verification_value = (
        active_cycle_run.get("current_task_verification_passed")
        if "current_task_verification_passed" in active_cycle_run
        else None
    )
    state = semantic_progress_state(
        phase=_resolved_active_cycle_phase(active_cycle_run),
        now=current_time,
        started_at=float(started_at or current_time),
        last_progress_at=last_progress_at,
        max_progress_stall_seconds=max_progress_stall_seconds,
        max_runtime_seconds=max_runtime_seconds,
        current_task=current_task,
        observe_summary=observe_summary,
        pending_decision_state=str(active_cycle_run.get("pending_decision_state", "")).strip(),
        preview_state="",
        current_task_verification_passed=(
            bool(verification_value) if verification_value is not None else None
        ),
    )
    return state if isinstance(state, dict) else {}


def _mid_cycle_intervention_signal(active_cycle_run: Mapping[str, object] | None) -> dict[str, object]:
    payload = active_cycle_run if isinstance(active_cycle_run, Mapping) else {}
    pending_decision_state = str(payload.get("pending_decision_state", "")).strip()
    decision_state = pending_decision_state
    if decision_state not in {"retain", "reject"}:
        return {"triggered": False, "reason": "no_decision_emitted"}
    current_task = payload.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    current_phase = str(current_task.get("phase", "")).strip()
    last_progress_phase = str(payload.get("last_progress_phase", "")).strip()
    active_phase = current_phase or last_progress_phase
    if not active_phase.startswith("generated_failure"):
        return {"triggered": False, "reason": "decision_not_in_failure_recovery"}
    selected_subsystem = str(payload.get("selected_subsystem", "")).strip()
    return {
        "triggered": True,
        "reason": (
            "mid-cycle controller intervention: durable "
            f"{decision_state} decision already emitted under {selected_subsystem or 'the active subsystem'}, "
            f"but the cycle continued into {active_phase} recovery without ending"
        ),
        "reason_code": "post_decision_failure_recovery",
        "decision_state": decision_state,
        "phase": active_phase,
        "subsystem": selected_subsystem,
    }


def _normalize_active_cycle_run_for_status(
    active_cycle_run: dict[str, object] | None,
    *,
    semantic_state: Mapping[str, object] | None,
) -> dict[str, object]:
    normalized = dict(active_cycle_run or {})
    if not normalized:
        return normalized
    resolved_phase = str((semantic_state or {}).get("phase", "")).strip()
    phase_family = str((semantic_state or {}).get("phase_family", "")).strip()
    if not resolved_phase:
        return normalized
    current_task = normalized.get("current_task", {})
    if isinstance(current_task, Mapping):
        task_payload = dict(current_task)
        raw_phase = str(task_payload.get("phase", "")).strip()
        if (
            phase_family in {"preview", "finalize", "holdout"}
            and raw_phase.startswith("generated_")
        ):
            task_payload["raw_phase"] = raw_phase
            task_payload["phase"] = resolved_phase
            normalized["current_task"] = task_payload
    raw_last_progress_phase = str(normalized.get("last_progress_phase", "")).strip()
    if phase_family in {"preview", "finalize", "holdout"} and raw_last_progress_phase.startswith("generated_"):
        normalized["raw_last_progress_phase"] = raw_last_progress_phase
        normalized["last_progress_phase"] = resolved_phase
    return normalized


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
    for key in (
        "cycle_id",
        "subsystem",
        "selected_subsystem",
        "selected_variant_id",
        "strategy_candidate_id",
        "strategy_candidate_kind",
        "strategy_origin",
        "last_progress_phase",
        "current_task",
        "current_cognitive_stage",
        "verification_outcome_summary",
    ):
        value = current_run.get(key)
        if value:
            synthetic_run[key] = value
    if isinstance(current_run.get("tolbert_runtime_summary"), Mapping):
        synthetic_run["tolbert_runtime_summary"] = _normalize_tolbert_runtime_summary(
            current_run.get("tolbert_runtime_summary")
        )
    if interrupted:
        synthetic_run["interrupted"] = True
    synthetic_run = _apply_structured_child_decision(
        run=synthetic_run,
        priority_benchmark_families=priority_benchmark_families,
        reason=str(stderr).strip(),
    )
    synthetic_run["decision_state"] = _decision_state_from_run(synthetic_run)
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
    on_event: Callable[[dict[str, object]], dict[str, object] | None] | None = None,
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
            "progress_label": progress_label or "run_improvement_cycle",
            "started_at": time.time(),
        }
    )
    assert process.stdout is not None
    if not hasattr(process.stdout, "fileno"):
        for line in process.stdout:
            completed_output.append(line)
            print(line, end="", file=sys.stderr, flush=True)
            action = _emit_event(
                {
                    "event": "output",
                    "line": line.rstrip("\n"),
                    "pid": process_pid,
                    "progress_label": progress_label or "run_improvement_cycle",
                    "timestamp": time.time(),
                }
            )
            if bool(action.get("terminate", False)):
                terminate_process_tree(process)
                return _timeout_result(
                    reason=str(action.get("reason", "")).strip() or "child terminated by controller intervention",
                    details={"intervention_reason": str(action.get("reason", "")).strip()},
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
    active_phase_progress: dict[str, object] = {}
    runtime_grace_keys: set[str] = set()
    runtime_deadline = started_at + max_runtime if max_runtime > 0.0 else 0.0
    exit_detected = False

    def _emit_exit_detected_if_needed() -> None:
        nonlocal exit_detected
        if exit_detected:
            return
        polled = process.poll()
        if polled is None:
            return
        exit_detected = True
        _emit_event(
            {
                "event": "exit_detected",
                "pid": process_pid,
                "progress_label": progress_label or "run_improvement_cycle",
                "returncode": int(polled),
                "timestamp": time.time(),
            }
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
                    _emit_exit_detected_if_needed()
                    completed_output.append(line)
                    now = time.monotonic()
                    last_output_at = now
                    if "[cycle:" in line or "[eval:" in line or "[repeated]" in line or "finalize phase=" in line:
                        last_progress_at = now
                        parsed_progress = _parse_progress_fields(line)
                        previous_task_identity = _task_progress_identity(current_task)
                        active_phase_progress.update(parsed_progress)
                        phase_name = str(parsed_progress.get("last_progress_phase", "")).strip()
                        if phase_name:
                            last_progress_phase = phase_name
                        parsed_task = parsed_progress.get("current_task", {})
                        if isinstance(parsed_task, dict) and parsed_task:
                            current_task = dict(parsed_task)
                            if _task_progress_identity(current_task) != previous_task_identity:
                                active_phase_progress.pop("current_task_verification_passed", None)
                    print(line, end="", file=sys.stderr, flush=True)
                    action = _emit_event(
                        {
                            "event": "output",
                            "line": line.rstrip("\n"),
                            "pid": process_pid,
                            "progress_label": progress_label or "run_improvement_cycle",
                            "timestamp": time.time(),
                        }
                    )
                    if bool(action.get("terminate", False)):
                        terminate_process_tree(process)
                        return _timeout_result(
                            reason=str(action.get("reason", "")).strip() or "child terminated by controller intervention",
                            details={"intervention_reason": str(action.get("reason", "")).strip()},
                        )
            elif process.poll() is not None:
                _emit_exit_detected_if_needed()
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
                action = _emit_event(
                    {
                        "event": "heartbeat",
                        "pid": process_pid,
                        "progress_label": progress_label or "run_improvement_cycle",
                        "silence_seconds": int(silence),
                        "timestamp": time.time(),
                    }
                )
                if bool(action.get("terminate", False)):
                    terminate_process_tree(process)
                    return _timeout_result(
                        reason=str(action.get("reason", "")).strip() or "child terminated by controller intervention",
                        details={"intervention_reason": str(action.get("reason", "")).strip()},
                    )
                last_heartbeat_at = now
            _emit_exit_detected_if_needed()
            if runtime_deadline > 0.0 and now >= runtime_deadline:
                grace_key, extension_seconds = _child_runtime_extension_plan(
                    last_progress_phase=last_progress_phase,
                    current_task=current_task,
                    active_cycle_run=active_phase_progress,
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


def _record_decision_conversion_state(record: Mapping[str, object] | None) -> str:
    payload = record if isinstance(record, Mapping) else {}
    decision_state = payload.get("decision_state", {})
    if isinstance(decision_state, Mapping):
        state = str(decision_state.get("decision_conversion_state", "")).strip()
        if state:
            return state
    artifact_path = (
        str(payload.get("active_artifact_path", "")).strip()
        or str(payload.get("artifact_path", "")).strip()
    )
    return "runtime_managed" if _is_runtime_managed_artifact_path(artifact_path) else "non_runtime_managed"


def _record_is_runtime_managed_decision(record: Mapping[str, object] | None) -> bool:
    return _record_decision_conversion_state(record) == "runtime_managed"


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
        existing_cycle_report = _load_cycle_report(config, cycle_id)
        if isinstance(existing_cycle_report, dict):
            existing_decision_record = _decision_record_from_cycle_report(existing_cycle_report)
            if isinstance(existing_decision_record, dict) and str(existing_decision_record.get("state", "")).strip() in {
                "retain",
                "reject",
            }:
                continue
        active_artifact_path = (
            str(summary.get("active_artifact_path", "")).strip()
            or str(summary.get("artifact_path", "")).strip()
        )
        candidate_artifact_path = str(summary.get("candidate_artifact_path", "")).strip()
        artifact_kind = str(summary.get("artifact_kind", "")).strip() or "retention_decision"
        strategy_candidate_id = str(summary.get("strategy_candidate_id", "")).strip()
        strategy_candidate_kind = str(summary.get("strategy_candidate_kind", "")).strip()
        strategy_origin = str(summary.get("strategy_origin", "")).strip()
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
            "strategy_candidate_id": strategy_candidate_id,
            "strategy_candidate_kind": strategy_candidate_kind,
            "strategy_origin": strategy_origin,
            "record_count": int(summary.get("record_count", 0) or 0),
            "selected_cycles": int(summary.get("selected_cycles", 0) or 0),
        }
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="incomplete",
                subsystem=subsystem,
                action="finalize_cycle",
                artifact_path=active_artifact_path,
                artifact_kind=artifact_kind,
                reason="incomplete autonomous cycle was reconciled without a retention decision",
                metrics_summary=metrics_summary,
                strategy_candidate_id=strategy_candidate_id,
                strategy_candidate_kind=strategy_candidate_kind,
                strategy_origin=strategy_origin,
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
                strategy_candidate_id=strategy_candidate_id,
                strategy_candidate_kind=strategy_candidate_kind,
                strategy_origin=strategy_origin,
                candidate_artifact_path=candidate_artifact_path,
                active_artifact_path=active_artifact_path,
            ),
            govern_exports=False,
        )
        reconciled.append(summary)
        print(
            f"[repeated] reconciled incomplete cycle cycle_id={cycle_id} subsystem={subsystem} state=incomplete",
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
                in {"evaluate", "select", "retain", "reject", "incomplete", "record"}
            )
        ]
    return scoped


def _campaign_records_with_report_fallback(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    cycles_path: Path,
    campaign_match_id: str,
    start_index: int = 0,
) -> list[dict[str, object]]:
    records = planner.load_cycle_records(cycles_path)
    campaign_records = _campaign_records(
        records,
        campaign_match_id=campaign_match_id,
        start_index=start_index,
    )
    recovered_report_records = _report_records_for_campaign(
        config,
        campaign_match_id=campaign_match_id,
    )
    if recovered_report_records:
        campaign_records = _merge_campaign_records(campaign_records, recovered_report_records)
    return campaign_records


def _production_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if str(record.get("state", "")) in {"retain", "reject"}
        and _record_is_runtime_managed_decision(record)
    ]


def _decision_record_richness(record: Mapping[str, object] | None) -> tuple[int, int, int]:
    payload = record if isinstance(record, Mapping) else {}
    metrics_summary = payload.get("metrics_summary", {})
    if not isinstance(metrics_summary, Mapping):
        metrics_summary = {}
    decision_state = payload.get("decision_state")
    family_pass_rate_delta = metrics_summary.get("family_pass_rate_delta", {})
    generated_family_pass_rate_delta = metrics_summary.get("generated_family_pass_rate_delta", {})
    family_signal_count = len(family_pass_rate_delta) if isinstance(family_pass_rate_delta, Mapping) else 0
    generated_signal_count = len(generated_family_pass_rate_delta) if isinstance(generated_family_pass_rate_delta, Mapping) else 0
    return (
        1 if isinstance(decision_state, Mapping) and bool(decision_state) else 0,
        family_signal_count + generated_signal_count,
        len(metrics_summary),
    )


def _runtime_managed_cycle_decisions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[object, dict[str, object]] = {}
    fallback_index = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        cycle_id = str(record.get("cycle_id", "")).strip()
        key: object
        if cycle_id:
            key = cycle_id
        else:
            key = (
                "__uncycled__",
                str(record.get("state", "")).strip(),
                str(record.get("artifact_path", "")).strip(),
                fallback_index,
            )
            fallback_index += 1
        current = merged.get(key)
        if current is None or _decision_record_richness(record) > _decision_record_richness(current):
            merged[key] = record
    return list(merged.values())


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
        and not _record_is_runtime_managed_decision(record)
    ]


def _reconciled_runtime_managed_decision_from_run(run: Mapping[str, object] | None) -> dict[str, object]:
    payload = run if isinstance(run, Mapping) else {}
    if int(payload.get("incomplete_cycle_count", 0) or 0) <= 0:
        return {}
    if str(payload.get("decision_conversion_state", "")).strip() != "runtime_managed":
        return {}
    state = str(payload.get("final_state", "")).strip()
    if state not in {"retain", "reject"}:
        return {}
    artifact_path = (
        str(payload.get("active_artifact_path", "")).strip()
        or str(payload.get("artifact_path", "")).strip()
    )
    if not _is_runtime_managed_artifact_path(artifact_path):
        return {}
    reason = (
        str(payload.get("decision_reason_code", "")).strip()
        or str(payload.get("preview_reason_code", "")).strip()
        or str(payload.get("final_reason", "")).strip()
        or "incomplete autonomous cycle was reconciled into a fail-closed runtime-managed decision"
    )
    decision_state = _decision_state_from_run(payload)
    return {
        "cycle_id": str(payload.get("cycle_id", "")).strip(),
        "state": state,
        "subsystem": str(payload.get("subsystem", "")).strip(),
        "action": "finalize_cycle",
        "artifact_path": artifact_path,
        "artifact_kind": str(payload.get("artifact_kind", "")).strip() or "retention_decision",
        "reason": reason,
        "candidate_artifact_path": str(payload.get("candidate_artifact_path", "")).strip(),
        "active_artifact_path": artifact_path,
        "metrics_summary": {
            "protocol": "autonomous",
            "protocol_match_id": str(payload.get("campaign_match_id", "")).strip(),
            "incomplete_cycle": True,
            "reconciled_runtime_managed_decision": True,
        },
        "decision_state": decision_state,
    }


def _reconciled_runtime_managed_decisions_from_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for run in runs:
        decision = _reconciled_runtime_managed_decision_from_run(run)
        if decision:
            decisions.append(decision)
    return decisions


def _structured_non_runtime_decisions(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        decision = run.get("structured_child_decision", {})
        if isinstance(decision, Mapping) and str(decision.get("state", "")).strip() in {"retain", "reject"}:
            decisions.append(dict(decision))
    return decisions


def _run_execution_source_task_ids(run: Mapping[str, object] | None) -> list[str]:
    payload = run if isinstance(run, Mapping) else {}
    task_ids: list[str] = []
    verification_summary = payload.get("verification_outcome_summary", {})
    if isinstance(verification_summary, Mapping):
        for key in ("verified_task_ids", "failed_task_ids", "successful_task_ids"):
            task_ids = _ordered_unique_strings(task_ids, verification_summary.get(key, []))
    current_task = payload.get("current_task", {})
    if isinstance(current_task, Mapping):
        current_task_id = str(current_task.get("task_id", "")).strip()
        if current_task_id:
            task_ids = _ordered_unique_strings(task_ids, [current_task_id])
    partial_progress = payload.get("partial_progress", {})
    if isinstance(partial_progress, Mapping):
        current_task_id = str(partial_progress.get("current_task_id", "")).strip()
        if current_task_id:
            task_ids = _ordered_unique_strings(task_ids, [current_task_id])
    return task_ids


def _execution_source_summary_for_task_ids(
    config: KernelConfig,
    *,
    task_ids: list[str],
    since_iso: str = "",
) -> dict[str, int]:
    summary = {
        "decoder_generated": 0,
        "llm_generated": 0,
        "bounded_decoder_generated": 0,
        "synthetic_plan": 0,
        "deterministic_or_other": 0,
        "total_executed_commands": 0,
    }
    seen_attempt_ids: set[str] = set()
    normalized_since = str(since_iso).strip()
    normalized_since_dt = _parse_iso_datetime(normalized_since)
    if config.uses_sqlite_storage():
        for task_id in _ordered_unique_strings(task_ids):
            for payload in config.sqlite_store().load_episode_attempt_documents(task_id):
                storage = dict(payload.get("episode_storage", {})) if isinstance(payload.get("episode_storage", {}), Mapping) else {}
                updated_at = str(storage.get("updated_at", "")).strip()
                if normalized_since and updated_at and updated_at < normalized_since:
                    continue
                attempt_id = str(storage.get("episode_id", "")).strip()
                if attempt_id and attempt_id in seen_attempt_ids:
                    continue
                if attempt_id:
                    seen_attempt_ids.add(attempt_id)
                execution_summary = dict(payload.get("summary", {})).get("execution_source_summary", {})
                execution_summary = dict(execution_summary) if isinstance(execution_summary, Mapping) else {}
                decoder_generated = int(
                    execution_summary.get("decoder_generated", execution_summary.get("llm_generated", 0)) or 0
                )
                llm_generated = int(execution_summary.get("llm_generated", 0) or 0)
                bounded_decoder_generated = int(
                    execution_summary.get(
                        "bounded_decoder_generated",
                        max(0, decoder_generated - llm_generated),
                    )
                    or 0
                )
                summary["decoder_generated"] += decoder_generated
                summary["llm_generated"] += llm_generated
                summary["bounded_decoder_generated"] += bounded_decoder_generated
                summary["synthetic_plan"] += int(execution_summary.get("synthetic_plan", 0) or 0)
                summary["deterministic_or_other"] += int(execution_summary.get("deterministic_or_other", 0) or 0)
                summary["total_executed_commands"] += int(execution_summary.get("total_executed_commands", 0) or 0)
        if summary["total_executed_commands"] > 0:
            return summary
    task_id_set = set(_ordered_unique_strings(task_ids))
    if not task_id_set or not config.run_reports_dir.exists():
        return summary
    latest_reports: dict[str, tuple[datetime | None, dict[str, object]]] = {}
    for report_path in sorted(config.run_reports_dir.glob("task_report_*.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("report_kind", "")).strip() != "unattended_task_report":
            continue
        task_id = str(payload.get("task_id", "")).strip()
        if task_id not in task_id_set:
            continue
        generated_at_dt = _parse_iso_datetime(payload.get("generated_at", ""))
        if normalized_since_dt is not None and generated_at_dt is not None and generated_at_dt < normalized_since_dt:
            continue
        previous = latest_reports.get(task_id)
        if previous is not None and previous[0] is not None and generated_at_dt is not None and generated_at_dt <= previous[0]:
            continue
        latest_reports[task_id] = (generated_at_dt, dict(payload))
    for _, payload in latest_reports.values():
        summary_payload = dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), Mapping) else {}
        execution_summary = dict(summary_payload.get("execution_source_summary", {})) if isinstance(
            summary_payload.get("execution_source_summary", {}), Mapping
        ) else {}
        if not execution_summary:
            commands = payload.get("commands", [])
            execution_summary = {
                "decoder_generated": 0,
                "llm_generated": 0,
                "bounded_decoder_generated": 0,
                "synthetic_plan": 0,
                "deterministic_or_other": 0,
                "total_executed_commands": 0,
            }
            if isinstance(commands, list):
                for command in commands:
                    if not isinstance(command, Mapping):
                        continue
                    decision_source = str(command.get("decision_source", "")).strip()
                    execution_summary["total_executed_commands"] += 1
                    if decision_source == "llm":
                        execution_summary["decoder_generated"] += 1
                        execution_summary["llm_generated"] += 1
                    elif decision_source.endswith("_decoder"):
                        execution_summary["decoder_generated"] += 1
                        execution_summary["bounded_decoder_generated"] += 1
                    elif decision_source == "synthetic_edit_plan_direct":
                        execution_summary["synthetic_plan"] += 1
                    else:
                        execution_summary["deterministic_or_other"] += 1
        summary["decoder_generated"] += int(
            execution_summary.get("decoder_generated", execution_summary.get("llm_generated", 0)) or 0
        )
        summary["llm_generated"] += int(execution_summary.get("llm_generated", 0) or 0)
        summary["bounded_decoder_generated"] += int(
            execution_summary.get(
                "bounded_decoder_generated",
                max(
                    0,
                    int(execution_summary.get("decoder_generated", execution_summary.get("llm_generated", 0)) or 0)
                    - int(execution_summary.get("llm_generated", 0) or 0),
                ),
            )
            or 0
        )
        summary["synthetic_plan"] += int(execution_summary.get("synthetic_plan", 0) or 0)
        summary["deterministic_or_other"] += int(execution_summary.get("deterministic_or_other", 0) or 0)
        summary["total_executed_commands"] += int(execution_summary.get("total_executed_commands", 0) or 0)
    return summary


def _merge_execution_source_summaries(*summaries: Mapping[str, object] | None) -> dict[str, int]:
    merged = {
        "decoder_generated": 0,
        "llm_generated": 0,
        "bounded_decoder_generated": 0,
        "synthetic_plan": 0,
        "deterministic_or_other": 0,
        "total_executed_commands": 0,
    }
    for payload in summaries:
        if not isinstance(payload, Mapping):
            continue
        merged["decoder_generated"] += max(0, int(payload.get("decoder_generated", 0) or 0))
        merged["llm_generated"] += max(0, int(payload.get("llm_generated", 0) or 0))
        merged["bounded_decoder_generated"] += max(
            0,
            int(payload.get("bounded_decoder_generated", 0) or 0),
        )
        merged["synthetic_plan"] += max(0, int(payload.get("synthetic_plan", 0) or 0))
        merged["deterministic_or_other"] += max(0, int(payload.get("deterministic_or_other", 0) or 0))
        merged["total_executed_commands"] += max(0, int(payload.get("total_executed_commands", 0) or 0))
    return merged


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


def _iter_cycle_reports(config: KernelConfig) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for path in sorted(config.improvement_reports_dir.glob("cycle_report_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            reports.append(payload)
    return reports


def _report_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _controller_intervention_reason_code_for_text(value: object) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return ""
    if "test-only rejection" in normalized or "all verified primary tasks failed" in normalized:
        return "all_primary_verifications_failed"
    if "controller intervention" in normalized:
        return "post_decision_failure_recovery"
    if "max silence" in normalized:
        return "child_max_silence"
    if "max runtime" in normalized:
        return "child_max_runtime"
    if "stalled" in normalized or "progress stall" in normalized:
        return "child_progress_stall"
    if "received signal" in normalized:
        return "received_signal"
    return ""


def _decision_state_payload(
    *,
    decision_owner: str,
    decision_credit: str,
    decision_conversion_state: str,
    retention_state: str,
    retention_basis: str,
    closeout_mode: str,
    controller_intervention_reason_code: str = "",
    recorded_at: str = "",
) -> dict[str, object]:
    return {
        "decision_owner": str(decision_owner).strip(),
        "decision_credit": str(decision_credit).strip(),
        "decision_conversion_state": str(decision_conversion_state).strip(),
        "retention_state": str(retention_state).strip(),
        "retention_basis": str(retention_basis).strip(),
        "closeout_mode": str(closeout_mode).strip(),
        "controller_intervention_reason_code": str(controller_intervention_reason_code).strip(),
        "recorded_at": str(recorded_at).strip(),
    }


def _decision_state_from_record(record: Mapping[str, object] | None) -> dict[str, object]:
    payload = record if isinstance(record, Mapping) else {}
    existing = payload.get("decision_state", {})
    if isinstance(existing, Mapping) and existing:
        decision_state = _decision_state_payload(
            decision_owner=str(existing.get("decision_owner", "")).strip() or "child_native",
            decision_credit=str(existing.get("decision_credit", "")).strip() or "child_native",
            decision_conversion_state=(
                str(existing.get("decision_conversion_state", "")).strip() or "runtime_managed"
            ),
            retention_state=(
                str(existing.get("retention_state", "")).strip()
                or str(payload.get("state", "")).strip()
                or str(payload.get("final_state", "")).strip()
            ),
            retention_basis=(
                str(existing.get("retention_basis", "")).strip()
                or str(payload.get("reason", "")).strip()
                or str(payload.get("final_reason", "")).strip()
            ),
            closeout_mode=str(existing.get("closeout_mode", "")).strip() or "natural",
            controller_intervention_reason_code=str(
                existing.get("controller_intervention_reason_code", "")
            ).strip(),
            recorded_at=str(existing.get("recorded_at", "")).strip(),
        )
        return decision_state
    metrics_summary = payload.get("metrics_summary", {})
    metrics_summary = dict(metrics_summary) if isinstance(metrics_summary, Mapping) else {}
    retention_state = str(payload.get("state", "")).strip() or str(payload.get("final_state", "")).strip()
    retention_basis = (
        str(metrics_summary.get("decision_reason_code", "")).strip()
        or str(metrics_summary.get("preview_reason_code", "")).strip()
        or str(payload.get("reason_code", "")).strip()
        or str(payload.get("reason", "")).strip()
        or str(payload.get("final_reason", "")).strip()
    )
    active_artifact_path = (
        str(payload.get("active_artifact_path", "")).strip()
        or str(payload.get("artifact_path", "")).strip()
    )
    runtime_managed = _is_runtime_managed_artifact_path(active_artifact_path)
    reconciled_incomplete_runtime_decision = runtime_managed and bool(metrics_summary.get("incomplete_cycle", False))
    controller_reason_code = ""
    if reconciled_incomplete_runtime_decision:
        controller_reason_code = _controller_intervention_reason_code_for_text(retention_basis)
        if not controller_reason_code:
            controller_reason_code = "incomplete_cycle_without_durable_decision"
    elif not runtime_managed:
        controller_reason_code = _controller_intervention_reason_code_for_text(retention_basis)
    return _decision_state_payload(
        decision_owner=(
            "controller_runtime_manager"
            if reconciled_incomplete_runtime_decision or not runtime_managed
            else "child_native"
        ),
        decision_credit=(
            "controller_fail_closed"
            if reconciled_incomplete_runtime_decision
            else ("child_native" if runtime_managed else "controller_runtime_manager")
        ),
        decision_conversion_state="runtime_managed" if runtime_managed else "non_runtime_managed",
        retention_state=retention_state,
        retention_basis=retention_basis,
        closeout_mode="forced_reject" if reconciled_incomplete_runtime_decision or not runtime_managed else "natural",
        controller_intervention_reason_code=controller_reason_code,
        recorded_at=str(payload.get("recorded_at", "")).strip(),
    )


def _decision_state_from_run(run: Mapping[str, object] | None) -> dict[str, object]:
    payload = run if isinstance(run, Mapping) else {}
    existing = payload.get("decision_state", {})
    if isinstance(existing, Mapping) and existing:
        return _decision_state_payload(
            decision_owner=str(existing.get("decision_owner", "")).strip(),
            decision_credit=str(existing.get("decision_credit", "")).strip(),
            decision_conversion_state=str(existing.get("decision_conversion_state", "")).strip(),
            retention_state=str(existing.get("retention_state", "")).strip(),
            retention_basis=str(existing.get("retention_basis", "")).strip(),
            closeout_mode=str(existing.get("closeout_mode", "")).strip(),
            controller_intervention_reason_code=str(
                existing.get("controller_intervention_reason_code", "")
            ).strip(),
            recorded_at=str(existing.get("recorded_at", "")).strip(),
        )
    decision_conversion_state = str(payload.get("decision_conversion_state", "")).strip()
    final_state = str(payload.get("final_state", "")).strip()
    retention_state = final_state if final_state else "undecided"
    if retention_state not in {"retain", "reject", "incomplete"}:
        if decision_conversion_state == "partial_productive_without_decision":
            retention_state = "incomplete"
        else:
            retention_state = "undecided"
    timeout_reason = str(payload.get("timeout_reason", "")).strip()
    controller_reason_code = ""
    structured_child_decision = payload.get("structured_child_decision", {})
    if isinstance(structured_child_decision, Mapping):
        controller_reason_code = str(structured_child_decision.get("reason_code", "")).strip()
    if not controller_reason_code:
        controller_reason_code = _controller_intervention_reason_code_for_text(timeout_reason)
    if not controller_reason_code:
        controller_reason_code = _controller_intervention_reason_code_for_text(
            str(payload.get("final_reason", "")).strip()
            or str(payload.get("decision_reason_code", "")).strip()
            or str(payload.get("preview_reason_code", "")).strip()
        )
    if not controller_reason_code and retention_state == "incomplete":
        controller_reason_code = "incomplete_cycle_without_durable_decision"
    incomplete_cycle_count = int(payload.get("incomplete_cycle_count", 0) or 0)
    reconciled_incomplete_runtime_decision = (
        incomplete_cycle_count > 0 and int(payload.get("runtime_managed_decisions", 0) or 0) > 0
    )
    decision_owner = "controller_runtime_manager"
    decision_credit = "controller_runtime_manager"
    closeout_mode = "forced_reject"
    if reconciled_incomplete_runtime_decision:
        decision_credit = "controller_fail_closed"
        closeout_mode = "forced_reject"
        if not controller_reason_code:
            controller_reason_code = "incomplete_cycle_without_durable_decision"
    elif int(payload.get("runtime_managed_decisions", 0) or 0) > 0:
        decision_owner = "child_native"
        if timeout_reason:
            decision_credit = "child_emitted_decision"
            closeout_mode = "child_native_before_partial_timeout"
        else:
            decision_credit = "child_native"
            closeout_mode = "natural"
    elif decision_conversion_state == "partial_productive_without_decision":
        decision_owner = "none"
        decision_credit = "partial_productive_evidence_only"
        closeout_mode = "partial_timeout_evidence_only"
    elif isinstance(structured_child_decision, Mapping):
        decision_credit = (
            str(structured_child_decision.get("decision_source", "")).strip()
            or str(structured_child_decision.get("decision_kind", "")).strip()
            or "controller_runtime_manager"
        )
    elif retention_state == "incomplete":
        if timeout_reason:
            decision_owner = "none"
            decision_credit = "partial_productive_evidence_only"
            closeout_mode = "partial_timeout_evidence_only"
        else:
            decision_credit = "controller_fail_closed"
            closeout_mode = "forced_reject"
    retention_basis = (
        str(payload.get("decision_reason_code", "")).strip()
        or str(payload.get("preview_reason_code", "")).strip()
        or (
            str(structured_child_decision.get("reason_code", "")).strip()
            if isinstance(structured_child_decision, Mapping)
            else ""
        )
        or str(payload.get("final_reason", "")).strip()
        or timeout_reason
        or decision_conversion_state
    )
    return _decision_state_payload(
        decision_owner=decision_owner,
        decision_credit=decision_credit,
        decision_conversion_state=decision_conversion_state or "no_decision",
        retention_state=retention_state,
        retention_basis=retention_basis,
        closeout_mode=closeout_mode,
        controller_intervention_reason_code=(
            "" if decision_owner == "child_native" else controller_reason_code
        ),
    )


def _terminal_campaign_run(runs: list[dict[str, object]]) -> dict[str, object]:
    for run in reversed(runs):
        if not isinstance(run, Mapping):
            continue
        final_state = str(run.get("final_state", "")).strip()
        decision_conversion_state = str(run.get("decision_conversion_state", "")).strip()
        if final_state in {"retain", "reject", "incomplete"}:
            return dict(run)
        if decision_conversion_state in {
            "runtime_managed",
            "non_runtime_managed",
            "partial_productive_without_decision",
            "incomplete",
        }:
            return dict(run)
        if _run_has_campaign_evidence(run):
            return dict(run)
    return {}


def _terminal_campaign_projection(
    runs: list[dict[str, object]],
    *,
    include_semantic_progress_state: bool = False,
) -> dict[str, object]:
    terminal_run = _terminal_campaign_run(runs)
    if not terminal_run:
        return {}
    final_state = str(terminal_run.get("final_state", "")).strip()
    decision_conversion_state = str(terminal_run.get("decision_conversion_state", "")).strip() or _run_decision_conversion_state(
        final_state=final_state,
        runtime_managed_decisions=_run_runtime_managed_decision_count(terminal_run),
        non_runtime_managed_decisions=max(0, int(terminal_run.get("non_runtime_managed_decisions", 0) or 0)),
        partial_productive=bool(terminal_run.get("partial_productive", False)),
        incomplete_cycle_count=max(0, int(terminal_run.get("incomplete_cycle_count", 0) or 0)),
    )
    subsystem = str(terminal_run.get("selected_subsystem", "")).strip() or str(
        terminal_run.get("subsystem", "")
    ).strip()
    final_reason = str(terminal_run.get("final_reason", "")).strip() or str(
        terminal_run.get("timeout_reason", "")
    ).strip()
    reason_code = str(terminal_run.get("decision_reason_code", "")).strip() or str(
        terminal_run.get("preview_reason_code", "")
    ).strip()
    verification_outcome_summary = terminal_run.get("verification_outcome_summary", {})
    projection: dict[str, object] = {
        "decision_conversion_state": decision_conversion_state,
        "productive": bool(terminal_run.get("productive", False)),
        "partial_productive": bool(terminal_run.get("partial_productive", False)),
        "retained_gain": bool(terminal_run.get("retained_gain", False)),
        "decision_state": _decision_state_from_run(terminal_run),
    }
    if final_state:
        projection["final_state"] = final_state
    if final_reason:
        projection["final_reason"] = final_reason
        projection["reason"] = final_reason
    if reason_code:
        projection["decision_reason_code"] = reason_code
        projection["preview_reason_code"] = reason_code
        projection["reason_code"] = reason_code
    if subsystem:
        projection["selected_subsystem"] = subsystem
        projection["subsystem"] = subsystem
    for key in (
        "artifact_path",
        "active_artifact_path",
        "candidate_artifact_path",
        "artifact_kind",
    ):
        value = str(terminal_run.get(key, "")).strip()
        if value:
            projection[key] = value
    if isinstance(verification_outcome_summary, Mapping) and verification_outcome_summary:
        projection["verification_outcome_summary"] = dict(verification_outcome_summary)
    if include_semantic_progress_state:
        now = time.time()
        semantic_state = semantic_progress_state(
            phase="done",
            now=now,
            started_at=now,
            last_progress_at=now,
            max_progress_stall_seconds=0.0,
        )
        if final_state in {"retain", "reject"}:
            semantic_state["detail"] = f"campaign finished with {final_state} decision"
        elif decision_conversion_state == "partial_productive_without_decision":
            semantic_state["detail"] = "campaign finished without a durable decision"
        else:
            semantic_state["detail"] = "campaign finished"
        projection["semantic_progress_state"] = semantic_state
    return projection


def _decision_state_summary(
    runs: list[dict[str, object]],
    decision_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    owner_counts = {
        "child_native": 0,
        "controller_runtime_manager": 0,
        "none": 0,
    }
    closeout_mode_counts = {
        "natural": 0,
        "accepted_partial_timeout": 0,
        "child_native_before_partial_timeout": 0,
        "partial_timeout_evidence_only": 0,
        "forced_reject": 0,
    }
    retention_state_counts = {
        "retain": 0,
        "reject": 0,
        "incomplete": 0,
        "undecided": 0,
    }
    for run in runs:
        decision_state = _decision_state_from_run(run)
        owner = str(decision_state.get("decision_owner", "")).strip()
        closeout_mode = str(decision_state.get("closeout_mode", "")).strip()
        retention_state = str(decision_state.get("retention_state", "")).strip()
        if owner in owner_counts:
            owner_counts[owner] += 1
        if closeout_mode in closeout_mode_counts:
            closeout_mode_counts[closeout_mode] += 1
        if retention_state in retention_state_counts:
            retention_state_counts[retention_state] += 1
    summary = {
        "run_decisions": {
            "child_native": owner_counts["child_native"],
            "controller_runtime_manager": owner_counts["controller_runtime_manager"],
            "none": owner_counts["none"],
        },
        "run_closeout_modes": closeout_mode_counts,
        "run_retention_states": retention_state_counts,
    }
    if decision_records is not None:
        summary["record_decisions"] = {
            "child_native": 0,
            "controller_runtime_manager": 0,
        }
        for record in decision_records:
            decision_state = _decision_state_from_record(record)
            owner = str(decision_state.get("decision_owner", "")).strip()
            if owner in summary["record_decisions"]:
                summary["record_decisions"][owner] += 1
    return summary


def _decision_record_from_cycle_report(report: dict[str, object]) -> dict[str, object] | None:
    current_cycle_records = report.get("current_cycle_records", [])
    if isinstance(current_cycle_records, list):
        for record in reversed(current_cycle_records):
            if not isinstance(record, dict):
                continue
            if str(record.get("state", "")).strip() not in {"retain", "reject"}:
                continue
            payload = dict(record)
            if not isinstance(payload.get("decision_state", {}), Mapping) or not payload.get("decision_state"):
                payload["decision_state"] = _decision_state_from_record(
                    {
                        **payload,
                        "decision_state": report.get("decision_state", {}),
                        "final_state": report.get("final_state", ""),
                        "final_reason": report.get("final_reason", ""),
                    }
                )
            if not _record_is_runtime_managed_decision(payload):
                continue
            return payload
    final_state = str(report.get("final_state", "")).strip()
    if final_state not in {"retain", "reject"}:
        return None
    artifact_path = str(report.get("active_artifact_path", report.get("artifact_path", ""))).strip()
    decision_state = _decision_state_from_record(report)
    if not _record_is_runtime_managed_decision(
        {
            "active_artifact_path": artifact_path,
            "artifact_path": artifact_path,
            "decision_state": decision_state,
        }
    ):
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
        "decision_state": decision_state,
    }


def _cycle_audit_summary_from_report(report: dict[str, object]) -> dict[str, object] | None:
    decision_record = _decision_record_from_cycle_report(report)
    if not isinstance(decision_record, dict):
        return None
    metrics_summary = decision_record.get("metrics_summary", {})
    if not isinstance(metrics_summary, dict):
        metrics_summary = {}
    audit = {
        "cycle_id": str(report.get("cycle_id", "")).strip(),
        "subsystem": str(report.get("subsystem", "")).strip(),
        "strategy_candidate_id": str(report.get("strategy_candidate_id", "")).strip(),
        "strategy_candidate_kind": str(report.get("strategy_candidate_kind", "")).strip(),
        "strategy_origin": str(report.get("strategy_origin", "")).strip(),
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
        "tolbert_runtime_summary": _normalize_tolbert_runtime_summary(report.get("tolbert_runtime_summary")),
        "decision_state": _decision_state_from_record(report),
    }
    for field in _CONFIRMATION_STRONG_BASELINE_INT_FIELDS:
        if field in metrics_summary:
            audit[field] = int(metrics_summary.get(field, 0) or 0)
    for field in _CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS:
        if field in metrics_summary:
            audit[field] = _report_float(metrics_summary.get(field, 0.0))
    return audit


def _report_records_for_campaign(
    config: KernelConfig,
    *,
    campaign_match_id: str,
) -> list[dict[str, object]]:
    if not campaign_match_id:
        return []
    recovered: list[dict[str, object]] = []
    for report in _iter_cycle_reports(config):
        current_cycle_records = report.get("current_cycle_records", [])
        if not isinstance(current_cycle_records, list):
            continue
        for record in current_cycle_records:
            if not isinstance(record, dict):
                continue
            if _record_protocol(record) != "autonomous":
                continue
            if _record_protocol_match_id(record) != campaign_match_id:
                continue
            recovered.append(dict(record))
    return recovered


def _merge_campaign_records(
    primary: list[dict[str, object]],
    fallback: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str, str], dict[str, object]] = {}

    def _key(record: dict[str, object]) -> tuple[str, str, str, str]:
        return (
            str(record.get("cycle_id", "")).strip(),
            str(record.get("state", "")).strip(),
            str(record.get("action", "")).strip(),
            str(record.get("artifact_path", "")).strip(),
        )

    for record in primary:
        if isinstance(record, dict):
            merged[_key(record)] = record
    for record in fallback:
        if not isinstance(record, dict):
            continue
        merged.setdefault(_key(record), record)
    return list(merged.values())


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
        candidate_path = (
            str(record.get("candidate_artifact_path", "")).strip()
            or str(generate_record.get("candidate_artifact_path", "")).strip()
        )
        active_path = (
            str(record.get("active_artifact_path", "")).strip()
            or str(generate_record.get("active_artifact_path", "")).strip()
        )
        runtime_managed = _record_is_runtime_managed_decision(record)
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


def _merge_incomplete_cycle_summaries(
    planner_summaries: list[dict[str, object]] | None,
    runs: list[dict[str, object]],
    decision_records: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    durable_decision_cycle_ids = {
        str(record.get("cycle_id", "")).strip()
        for record in (decision_records or [])
        if isinstance(record, Mapping)
        and str(record.get("state", "")).strip() in {"retain", "reject"}
        and not bool(
            record.get("metrics_summary", {}).get("incomplete_cycle", False)
            if isinstance(record.get("metrics_summary", {}), Mapping)
            else False
        )
    }

    def _summary_richness(summary: Mapping[str, object]) -> tuple[int, int]:
        signal_fields = (
            "subsystem",
            "selected_variant_id",
            "strategy_candidate_id",
            "strategy_candidate_kind",
            "strategy_origin",
            "artifact_kind",
            "artifact_path",
            "active_artifact_path",
            "candidate_artifact_path",
        )
        populated = sum(1 for field in signal_fields if str(summary.get(field, "")).strip())
        return (populated, len(summary))

    def _merge_summary(summary: Mapping[str, object]) -> None:
        cycle_id = str(summary.get("cycle_id", "")).strip()
        if not cycle_id:
            return
        candidate = {key: value for key, value in dict(summary).items() if value not in (None, "", [], {})}
        current = merged.get(cycle_id)
        if current is None or _summary_richness(candidate) >= _summary_richness(current):
            merged[cycle_id] = candidate

    for summary in planner_summaries or []:
        if isinstance(summary, Mapping):
            cycle_id = str(summary.get("cycle_id", "")).strip()
            if cycle_id and cycle_id in durable_decision_cycle_ids:
                continue
            _merge_summary(summary)
    for run in runs:
        if not isinstance(run, Mapping) or int(run.get("incomplete_cycle_count", 0) or 0) <= 0:
            continue
        cycle_id = str(run.get("cycle_id", "")).strip()
        if not cycle_id:
            campaign_cycle_ids = run.get("campaign_cycle_ids", [])
            if isinstance(campaign_cycle_ids, list):
                for raw_cycle_id in campaign_cycle_ids:
                    token = str(raw_cycle_id).strip()
                    if token:
                        cycle_id = token
                        break
        if not cycle_id:
            continue
        _merge_summary(
            {
                "cycle_id": cycle_id,
                "subsystem": str(run.get("subsystem", "")).strip(),
                "selected_variant_id": str(run.get("selected_variant_id", "")).strip(),
                "strategy_candidate_id": str(run.get("strategy_candidate_id", "")).strip(),
                "strategy_candidate_kind": str(run.get("strategy_candidate_kind", "")).strip(),
                "strategy_origin": str(run.get("strategy_origin", "")).strip(),
                "artifact_kind": str(run.get("artifact_kind", "")).strip(),
                "artifact_path": str(run.get("artifact_path", "")).strip(),
                "active_artifact_path": str(run.get("active_artifact_path", "")).strip(),
                "candidate_artifact_path": str(run.get("candidate_artifact_path", "")).strip(),
            }
        )
    for record in decision_records or []:
        if not isinstance(record, Mapping):
            continue
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, Mapping) or not bool(metrics_summary.get("incomplete_cycle", False)):
            continue
        _merge_summary(
            {
                "cycle_id": str(record.get("cycle_id", "")).strip(),
                "subsystem": str(record.get("subsystem", "")).strip(),
                "selected_variant_id": str(metrics_summary.get("selected_variant_id", "")).strip(),
                "strategy_candidate_id": str(metrics_summary.get("strategy_candidate_id", "")).strip(),
                "strategy_candidate_kind": str(metrics_summary.get("strategy_candidate_kind", "")).strip(),
                "strategy_origin": str(metrics_summary.get("strategy_origin", "")).strip(),
                "artifact_kind": str(record.get("artifact_kind", "")).strip(),
                "artifact_path": str(record.get("artifact_path", "")).strip(),
                "active_artifact_path": str(record.get("active_artifact_path", "")).strip(),
                "candidate_artifact_path": str(record.get("candidate_artifact_path", "")).strip(),
            }
        )
    return list(merged.values())


def _yield_summary_for(
    records: list[dict[str, object]],
    generate_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    retained = [record for record in records if str(record.get("state", "")) == "retain"]
    rejected = [record for record in records if str(record.get("state", "")) == "reject"]
    rejected_learning_opportunities = [
        record for record in rejected if _record_counts_as_reject_learning_opportunity(record)
    ]
    retained_by_subsystem: dict[str, int] = {}
    rejected_by_subsystem: dict[str, int] = {}
    rejected_learning_opportunity_by_subsystem: dict[str, int] = {}
    resolved_generate_index = generate_index or {}
    for record in retained:
        key = str(record.get("subsystem", "unknown"))
        retained_by_subsystem[key] = retained_by_subsystem.get(key, 0) + 1
    for record in rejected:
        key = str(record.get("subsystem", "unknown"))
        rejected_by_subsystem[key] = rejected_by_subsystem.get(key, 0) + 1
    for record in rejected_learning_opportunities:
        key = str(record.get("subsystem", "unknown"))
        rejected_learning_opportunity_by_subsystem[key] = (
            rejected_learning_opportunity_by_subsystem.get(key, 0) + 1
        )

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
        "rejected_learning_opportunity_cycles": len(rejected_learning_opportunities),
        "total_decisions": len(records),
        "retained_by_subsystem": retained_by_subsystem,
        "rejected_by_subsystem": rejected_by_subsystem,
        "rejected_learning_opportunity_by_subsystem": rejected_learning_opportunity_by_subsystem,
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
            "rejected_learning_opportunity_decisions": 0,
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
            "retained_support_gain_decisions": 0,
            "retained_support_gain_score": 0.0,
            "average_retained_pass_rate_delta": 0.0,
            "best_retained_pass_rate_delta": 0.0,
            "worst_pass_rate_delta": 0.0,
        }
        for family in normalized_families
    }
    retained_delta_totals: dict[str, float] = {family: 0.0 for family in normalized_families}
    retained_delta_counts: dict[str, int] = {family: 0 for family in normalized_families}
    retained_delta_seen: set[str] = set()
    retained_gain_seen: set[str] = set()
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
        record_counts_as_gain = _record_counts_as_support_retained_gain(record)
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
                if delta > 0.0 or (record_counts_as_gain and delta >= 0.0):
                    summary["retained_positive_delta_decisions"] = int(summary["retained_positive_delta_decisions"]) + 1
                    if delta > 0.0:
                        summary["retained_positive_pass_rate_delta_sum"] = (
                            float(summary["retained_positive_pass_rate_delta_sum"]) + delta
                        )
                    elif record_counts_as_gain:
                        support_signal = _record_support_signal_summary(record)
                        control_delta_count = int(
                            support_signal.get("transition_model_scoring_control_delta_count", 0) or 0
                        )
                        support_gain_score = min(0.05, 0.01 * max(1, control_delta_count))
                        summary["retained_support_gain_decisions"] = (
                            int(summary["retained_support_gain_decisions"]) + 1
                        )
                        summary["retained_support_gain_score"] = (
                            float(summary["retained_support_gain_score"]) + support_gain_score
                        )
                    retained_gain_seen.add(family)
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
                if _record_counts_as_reject_learning_opportunity(record):
                    summary["rejected_learning_opportunity_decisions"] = (
                        int(summary["rejected_learning_opportunity_decisions"]) + 1
                    )
    for family in normalized_families:
        retained_count = retained_delta_counts[family]
        if retained_count > 0:
            family_summaries[family]["average_retained_pass_rate_delta"] = retained_delta_totals[family] / float(retained_count)
    retained_gain_conversion_families = [
        family
        for family in sorted(
            normalized_families,
            key=lambda item: (
                -int(family_summaries[item]["observed_decisions"]),
                -int(family_summaries[item]["rejected_decisions"]),
                -float(family_summaries[item]["observed_estimated_cost"]),
                item,
            ),
        )
        if int(family_summaries[family]["observed_decisions"]) > 0
        and family not in retained_gain_seen
    ]
    return {
        "priority_families": normalized_families,
        "family_summaries": family_summaries,
        "priority_families_with_signal": [
            family for family in normalized_families if int(family_summaries[family]["observed_decisions"]) > 0
        ],
        "priority_families_with_retained_gain": [family for family in normalized_families if family in retained_gain_seen],
        "priority_families_without_signal": [
            family for family in normalized_families if int(family_summaries[family]["observed_decisions"]) <= 0
        ],
        "priority_families_with_signal_but_no_retained_gain": [
            family
            for family in normalized_families
            if int(family_summaries[family]["observed_decisions"]) > 0
            and family not in retained_gain_seen
        ],
        "priority_families_needing_retained_gain_conversion": retained_gain_conversion_families,
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


def _trust_breadth_summary_from_ledger(ledger: Mapping[str, object] | None) -> dict[str, object]:
    payload = ledger if isinstance(ledger, Mapping) else {}
    overall = payload.get("overall_summary", {})
    external = payload.get("external_summary", {})
    coverage = payload.get("coverage_summary", {})
    policy = payload.get("policy", {})
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(external, dict):
        external = {}
    if not isinstance(coverage, dict):
        coverage = {}
    if not isinstance(policy, dict):
        policy = {}
    return {
        "reports_considered": int(payload.get("reports_considered", 0) or 0),
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


def _open_world_breadth_summary_from_ledger(ledger: Mapping[str, object] | None) -> dict[str, object]:
    payload = ledger if isinstance(ledger, Mapping) else {}
    coverage = payload.get("coverage_summary", {})
    external = payload.get("external_summary", {})
    if not isinstance(coverage, dict):
        coverage = {}
    if not isinstance(external, dict):
        external = {}
    task_yield_summary = coverage.get("task_yield_bucket_summary", {})
    if not isinstance(task_yield_summary, dict):
        task_yield_summary = {}
    semantic_hub_summary = task_yield_summary.get("semantic_hub", {})
    replay_derived_summary = coverage.get("replay_derived_task_yield_summary", {})
    if not isinstance(semantic_hub_summary, dict):
        semantic_hub_summary = {}
    if not isinstance(replay_derived_summary, dict):
        replay_derived_summary = {}
    semantic_hub_families = _ordered_unique_strings(semantic_hub_summary.get("benchmark_families", []))
    external_families = _ordered_unique_strings(external.get("benchmark_families", []))
    replay_derived_families = _ordered_unique_strings(replay_derived_summary.get("benchmark_families", []))
    open_world_families = _ordered_unique_strings(semantic_hub_families, external_families)
    return {
        "semantic_hub_report_count": int(semantic_hub_summary.get("reports", 0) or 0),
        "distinct_semantic_hub_benchmark_families": int(
            semantic_hub_summary.get("distinct_benchmark_families", 0) or 0
        ),
        "semantic_hub_benchmark_families": semantic_hub_families,
        "external_report_count": int(external.get("total", external.get("external_report_count", 0)) or 0),
        "distinct_external_benchmark_families": int(
            external.get("distinct_benchmark_families", external.get("distinct_external_benchmark_families", 0)) or 0
        ),
        "external_benchmark_families": external_families,
        "replay_derived_report_count": int(replay_derived_summary.get("reports", 0) or 0),
        "distinct_replay_derived_benchmark_families": int(
            replay_derived_summary.get("distinct_benchmark_families", 0) or 0
        ),
        "replay_derived_benchmark_families": replay_derived_families,
        "open_world_report_count": int(semantic_hub_summary.get("reports", 0) or 0)
        + int(external.get("total", external.get("external_report_count", 0)) or 0),
        "distinct_open_world_benchmark_families": len(open_world_families),
        "benchmark_families": open_world_families,
        "open_world_benchmark_families": open_world_families,
        "retained_open_world_gain": False,
        "open_world_task_yield_present": bool(
            int(semantic_hub_summary.get("reports", 0) or 0) > 0
            or int(external.get("total", external.get("external_report_count", 0)) or 0) > 0
        ),
        "semantic_hub_summary": dict(semantic_hub_summary),
        "external_summary": dict(external),
        "replay_derived_summary": dict(replay_derived_summary),
    }


def _trust_breadth_summary(config: KernelConfig) -> dict[str, object]:
    ledger = build_unattended_trust_ledger(config)
    return _trust_breadth_summary_from_ledger(ledger)


def _open_world_breadth_summary(config: KernelConfig) -> dict[str, object]:
    ledger = build_unattended_trust_ledger(config)
    return _open_world_breadth_summary_from_ledger(ledger)


def _trust_and_open_world_breadth_summaries(config: KernelConfig) -> tuple[dict[str, object], dict[str, object]]:
    ledger = build_unattended_trust_ledger(config)
    return _trust_breadth_summary_from_ledger(ledger), _open_world_breadth_summary_from_ledger(ledger)


def _normalize_benchmark_families(values: object) -> list[str]:
    return _ordered_unique_strings(values)


def _repeated_report_status(payload: Mapping[str, object] | None) -> str:
    report = payload if isinstance(payload, Mapping) else {}
    status = str(report.get("status", "")).strip()
    return status or "finished"


def _repeated_closure_gap_summary(report_payload: Mapping[str, object] | None) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    trust = report.get("trust_breadth_summary", {})
    if not isinstance(trust, Mapping):
        trust = {}
    open_world = report.get("open_world_breadth_summary", {})
    if not isinstance(open_world, Mapping):
        open_world = {}
    partial_progress = report.get("partial_progress_summary", {})
    if not isinstance(partial_progress, Mapping):
        partial_progress = {}
    required_families = _normalize_benchmark_families(
        trust.get("required_families", report.get("priority_benchmark_families", []))
    )
    missing_required_families = _normalize_benchmark_families(trust.get("missing_required_families", []))
    missing_clean_task_root_breadth = _normalize_benchmark_families(
        trust.get("missing_required_family_clean_task_root_breadth", [])
    )
    retained_conversion_closed = (
        int(report.get("retained_gain_runs", 0) or 0) > 0
        and int(report.get("runtime_managed_decisions", 0) or 0) > 0
    )
    trust_breadth_closed = (
        not missing_required_families
        and not missing_clean_task_root_breadth
        and int(trust.get("distinct_family_gap", 0) or 0) <= 0
    )
    open_world_transfer_closed = (
        int(open_world.get("open_world_report_count", 0) or 0) > 0
        and int(open_world.get("distinct_open_world_benchmark_families", 0) or 0) > 0
    )
    long_horizon_started = int(partial_progress.get("generated_success_started_runs", 0) or 0) > 0
    long_horizon_completed = int(partial_progress.get("generated_success_completed_runs", 0) or 0) > 0
    long_horizon_state = "closed" if long_horizon_completed else "partial" if long_horizon_started else "open"
    retrieval_carryover_state = (
        "partial"
        if int(report.get("runtime_managed_decisions", 0) or 0) > 0
        or int(report.get("retained_gain_runs", 0) or 0) > 0
        else "open"
    )
    return {
        "retained_conversion": "closed" if retained_conversion_closed else "open",
        "trust_breadth": "closed" if trust_breadth_closed else "open",
        "retrieval_carryover": retrieval_carryover_state,
        "long_horizon_finalize_execution": long_horizon_state,
        "open_world_transfer": "closed" if open_world_transfer_closed else "open",
        "required_families": required_families,
        "missing_required_families": missing_required_families,
        "missing_required_family_clean_task_root_breadth": missing_clean_task_root_breadth,
    }


def _repeated_benchmark_dominance_summary(
    report_payload: Mapping[str, object] | None,
    *,
    closure_gap_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    closure = closure_gap_summary if isinstance(closure_gap_summary, Mapping) else {}
    trust = report.get("trust_breadth_summary", {})
    if not isinstance(trust, Mapping):
        trust = {}
    required_families = _normalize_benchmark_families(
        trust.get("required_families", report.get("priority_benchmark_families", []))
    )
    required_families_with_reports = _normalize_benchmark_families(trust.get("required_families_with_reports", []))
    missing_required_families = _normalize_benchmark_families(trust.get("missing_required_families", []))
    missing_clean_task_root_breadth = _normalize_benchmark_families(
        trust.get("missing_required_family_clean_task_root_breadth", [])
    )
    contract_state = (
        "closed"
        if (
            str(closure.get("retained_conversion", "")).strip() == "closed"
            and str(closure.get("trust_breadth", "")).strip() == "closed"
        )
        else "open"
    )
    return {
        "required_families": required_families,
        "required_families_with_reports": required_families_with_reports,
        "missing_required_families": missing_required_families,
        "missing_required_family_clean_task_root_breadth": missing_clean_task_root_breadth,
        "contract_state": contract_state,
    }


def _repeated_asi_campaign_lane_recommendation(
    report_payload: Mapping[str, object] | None,
    *,
    closure_gap_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    closure = closure_gap_summary if isinstance(closure_gap_summary, Mapping) else {}
    status = _repeated_report_status(report)
    trust = report.get("trust_breadth_summary", {})
    if not isinstance(trust, Mapping):
        trust = {}
    blockers: list[str] = []
    if int(report.get("runtime_managed_decisions", 0) or 0) <= 0:
        blockers.append("no_runtime_managed_decisions")
    if int(report.get("retained_gain_runs", 0) or 0) <= 0:
        blockers.append("no_retained_gain_runs")
    missing_required_families = _normalize_benchmark_families(trust.get("missing_required_families", []))
    if missing_required_families:
        blockers.extend(
            f"missing_required_families:{family}" for family in missing_required_families
        )
    missing_clean_task_root_breadth = _normalize_benchmark_families(
        trust.get("missing_required_family_clean_task_root_breadth", [])
    )
    if missing_clean_task_root_breadth:
        blockers.extend(
            f"missing_clean_task_root_breadth:{family}" for family in missing_clean_task_root_breadth
        )
    if str(closure.get("open_world_transfer", "")).strip() != "closed":
        blockers.append("unfamiliar_environment_transfer_not_closed")
    if status == "interrupted":
        blockers.append("campaign_interrupted")

    decision_gap = str(closure.get("retained_conversion", "")).strip() != "closed"
    trust_gap = str(closure.get("trust_breadth", "")).strip() != "closed"
    open_world_gap = str(closure.get("open_world_transfer", "")).strip() != "closed"
    runtime_gap = status == "interrupted" or int(report.get("completed_runs", 0) or 0) <= 0

    supporting_lanes: list[str] = []

    def _add_support(lane_id: str) -> None:
        if lane_id and lane_id not in supporting_lanes:
            supporting_lanes.append(lane_id)

    primary_lane = "lane_runtime_authority"
    primary_batch_goal = "authoritative_runtime_state"
    rationale = "repeated-cycle status still needs to stay authoritative before stronger claims can be trusted"
    if decision_gap:
        primary_lane = "lane_decision_closure"
        primary_batch_goal = "retained_conversion"
        rationale = "repeated cycles still lack retained runtime-managed conversion, so decision closure remains primary"
        if trust_gap:
            _add_support("lane_trust_and_carryover")
        if open_world_gap:
            _add_support("lane_repo_generalization")
    elif trust_gap:
        primary_lane = "lane_trust_and_carryover"
        primary_batch_goal = "counted_trust_breadth"
        rationale = "required-family breadth is still open, so trust and carryover becomes the primary lane"
        if open_world_gap:
            _add_support("lane_repo_generalization")
    elif open_world_gap:
        primary_lane = "lane_repo_generalization"
        primary_batch_goal = "open_world_transfer"
        rationale = "A8-oriented pressure still lacks open-world breadth, so repo generalization becomes primary"
    elif runtime_gap:
        primary_lane = "lane_runtime_authority"
        primary_batch_goal = "resilient_campaign_completion"
        rationale = "campaign completion is still fragile, so runtime authority remains the active lane"
    else:
        primary_lane = "lane_strategy_memory"
        primary_batch_goal = "strategic_superiority_proof"
        rationale = "closure is locally stable, so the next gains should target broader retained strategic pressure"

    if runtime_gap and primary_lane != "lane_runtime_authority":
        _add_support("lane_runtime_authority")
    if trust_gap and primary_lane not in {"lane_trust_and_carryover", "lane_decision_closure"}:
        _add_support("lane_trust_and_carryover")
    if open_world_gap and primary_lane != "lane_repo_generalization":
        _add_support("lane_repo_generalization")

    supporting_lanes = [
        lane_id
        for lane_id in _REPEATED_PHASE_A_LANE_ORDER
        if lane_id in supporting_lanes
    ]
    return {
        "primary_lane": primary_lane,
        "primary_batch_goal": primary_batch_goal,
        "supporting_lanes": supporting_lanes,
        "target_level": "A8",
        "blockers": blockers,
        "rationale": rationale,
    }


def _repeated_substantive_gaps(
    report_payload: Mapping[str, object] | None,
    *,
    closure_gap_summary: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    closure = closure_gap_summary if isinstance(closure_gap_summary, Mapping) else {}
    gaps: list[dict[str, object]] = []
    if str(closure.get("retained_conversion", "")).strip() != "closed":
        gaps.append(
            {
                "gap_id": "retained_conversion",
                "rationale": "repeated cycles are not yet converting child-native runtime decisions into retained gains",
                "recommended_lane": "lane_decision_closure",
            }
        )
    if str(closure.get("trust_breadth", "")).strip() != "closed":
        gaps.append(
            {
                "gap_id": "counted_trust_breadth",
                "rationale": "required-family trust breadth and clean task-root breadth remain open",
                "recommended_lane": "lane_trust_and_carryover",
            }
        )
    if str(closure.get("open_world_transfer", "")).strip() != "closed":
        gaps.append(
            {
                "gap_id": "open_world_transfer",
                "rationale": "held-out or external open-world evidence is still too thin for A8-oriented pressure",
                "recommended_lane": "lane_repo_generalization",
            }
        )
    if _repeated_report_status(report) == "interrupted":
        gaps.append(
            {
                "gap_id": "campaign_resilience",
                "rationale": "the latest repeated-cycle campaign ended in an interrupted state before a clean closeout",
                "recommended_lane": "lane_runtime_authority",
            }
        )
    return gaps


def _repeated_codex_input_packet_paths(
    *,
    report_path: Path,
    status_path: Path,
    report_payload: Mapping[str, object] | None,
    lane_recommendation: Mapping[str, object] | None,
) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    lane = lane_recommendation if isinstance(lane_recommendation, Mapping) else {}
    lane_ids = _ordered_unique_strings(
        lane.get("primary_lane", ""),
        lane.get("supporting_lanes", []),
    )
    lane_packet_paths = {
        lane_id: str(report_path.parent / f"{report_path.stem}.{lane_id}.codex_lane_packet.json")
        for lane_id in lane_ids
    }
    return {
        "official_anchor": {
            "report_path": str(report_path),
            "status_path": str(status_path),
            "repeated_status_path": str(status_path),
            "child_report_path": str(report_path),
            "anchor_date": str(report.get("created_at", "")).strip(),
        },
        "batch_packet_path": str(report_path.parent / f"{report_path.stem}.codex_batch_packet.json"),
        "frontier_packet_path": str(report_path.parent / f"{report_path.stem}.codex_frontier_packet.json"),
        "baseline_packet_path": str(report_path.parent / f"{report_path.stem}.codex_baseline_packet.json"),
        "lane_packet_paths": lane_packet_paths,
    }


_CONFIRMATION_STRONG_BASELINE_INT_FIELDS = (
    "confirmation_run_count",
    "confirmation_regressed_task_count",
    "confirmation_regressed_trace_task_count",
    "confirmation_regressed_trajectory_task_count",
    "confirmation_regressed_family_conservative_count",
)
_CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS = (
    "confirmation_pass_rate_delta_lower_bound",
    "confirmation_pass_rate_delta_conservative_lower_bound",
    "confirmation_paired_task_non_regression_rate_lower_bound",
    "confirmation_paired_trace_non_regression_rate_lower_bound",
    "confirmation_paired_trajectory_non_regression_rate_lower_bound",
    "confirmation_paired_trajectory_exact_match_rate_lower_bound",
    "confirmation_worst_family_conservative_lower_bound",
)


def _confirmation_strong_baseline_evidence(payload: Mapping[str, object] | None) -> dict[str, object]:
    evidence = payload if isinstance(payload, Mapping) else {}
    confirmation_run_count = int(evidence.get("confirmation_run_count", 0) or 0)
    comparison_fields_present = [
        field
        for field in (
            *_CONFIRMATION_STRONG_BASELINE_INT_FIELDS[1:],
            *_CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS,
        )
        if field in evidence
    ]
    summary: dict[str, object] = {
        "confirmation_run_count": confirmation_run_count,
        "comparison_fields_present": comparison_fields_present,
        "ready": confirmation_run_count > 0 and bool(comparison_fields_present),
    }
    for field in _CONFIRMATION_STRONG_BASELINE_INT_FIELDS[1:]:
        if field in evidence:
            summary[field] = int(evidence.get(field, 0) or 0)
    for field in _CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS:
        if field in evidence:
            summary[field] = _report_float(evidence.get(field, 0.0))
    return summary


def _repeated_strong_baseline_comparison_summary(report_payload: Mapping[str, object] | None) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    evidence_candidates: list[tuple[int, dict[str, object]]] = []
    root_evidence = _confirmation_strong_baseline_evidence(report)
    if bool(root_evidence.get("ready", False)):
        evidence_candidates.append((0, root_evidence))
    for run in report.get("runs", []) if isinstance(report.get("runs", []), list) else []:
        if not isinstance(run, Mapping):
            continue
        run_evidence = _confirmation_strong_baseline_evidence(run)
        if not bool(run_evidence.get("ready", False)):
            continue
        evidence_candidates.append((int(run.get("index", 0) or 0), run_evidence))
    if not evidence_candidates:
        return {
            "ready": False,
            "supporting_run_indices": [],
            "comparison_fields_present": [],
            "confirmation_run_count": 0,
        }
    strongest = max(
        evidence_candidates,
        key=lambda item: (
            int(item[1].get("confirmation_run_count", 0) or 0),
            len(item[1].get("comparison_fields_present", [])),
            item[0],
        ),
    )[1]
    return {
        "ready": True,
        "supporting_run_indices": [run_index for run_index, _ in evidence_candidates if run_index > 0],
        "comparison_fields_present": sorted(
            {
                str(field).strip()
                for _, evidence in evidence_candidates
                for field in evidence.get("comparison_fields_present", [])
                if str(field).strip()
            }
        ),
        "confirmation_run_count": int(strongest.get("confirmation_run_count", 0) or 0),
        **{
            field: strongest[field]
            for field in (
                *_CONFIRMATION_STRONG_BASELINE_INT_FIELDS[1:],
                *_CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS,
            )
            if field in strongest
        },
    }


def _repeated_codex_input_packet_summary(report_payload: Mapping[str, object] | None) -> dict[str, object]:
    report = report_payload if isinstance(report_payload, Mapping) else {}
    codex_input_packets = report.get("codex_input_packets", {})
    if not isinstance(codex_input_packets, Mapping):
        codex_input_packets = {}
    closure = report.get("closure_gap_summary", {})
    if not isinstance(closure, Mapping):
        closure = {}
    open_world = report.get("open_world_breadth_summary", {})
    if not isinstance(open_world, Mapping):
        open_world = {}
    lane = report.get("asi_campaign_lane_recommendation", {})
    if not isinstance(lane, Mapping):
        lane = {}
    official_anchor = codex_input_packets.get("official_anchor", {})
    if not isinstance(official_anchor, Mapping):
        official_anchor = {}
    batch_packet_path = str(codex_input_packets.get("batch_packet_path", "")).strip()
    frontier_packet_path = str(codex_input_packets.get("frontier_packet_path", "")).strip()
    baseline_packet_path = str(codex_input_packets.get("baseline_packet_path", "")).strip()
    raw_lane_packet_paths = codex_input_packets.get("lane_packet_paths", {})
    lane_packet_paths = (
        {
            str(key).strip(): str(value).strip()
            for key, value in raw_lane_packet_paths.items()
            if str(key).strip() and str(value).strip()
        }
        if isinstance(raw_lane_packet_paths, Mapping)
        else {}
    )

    batch_blockers: list[str] = []
    if not str(official_anchor.get("report_path", "")).strip():
        batch_blockers.append("missing_official_anchor_report_path")
    if not str(official_anchor.get("status_path", "")).strip():
        batch_blockers.append("missing_official_anchor_status_path")
    if not str(official_anchor.get("child_report_path", "")).strip():
        batch_blockers.append("missing_official_anchor_child_report_path")
    if not batch_packet_path:
        batch_blockers.append("missing_frozen_batch_packet_path")
    if not lane:
        batch_blockers.append("missing_lane_recommendation")
    batch_packet = {
        "ready": not batch_blockers,
        "packet_path": batch_packet_path,
        "blockers": batch_blockers,
    }

    external_summary = open_world.get("external_summary", {})
    if not isinstance(external_summary, Mapping):
        external_summary = {}
    strong_baseline_summary = _repeated_strong_baseline_comparison_summary(report)
    unfamiliar_domain_slice_ready = int(open_world.get("open_world_report_count", 0) or 0) > 0
    strong_baseline_slice_ready = bool(strong_baseline_summary.get("ready", False))
    long_horizon_transfer_slice_ready = str(closure.get("long_horizon_finalize_execution", "")).strip() in {
        "partial",
        "closed",
    }
    conservative_comparison_ready = bool(external_summary.get("success_rate_confidence_interval"))
    frontier_blockers: list[str] = []
    if not unfamiliar_domain_slice_ready:
        frontier_blockers.append("missing_unfamiliar_domain_slice")
    if not strong_baseline_slice_ready:
        frontier_blockers.append("missing_strong_baseline_comparison_slice")
    if not long_horizon_transfer_slice_ready:
        frontier_blockers.append("missing_long_horizon_transfer_slice")
    if not conservative_comparison_ready:
        frontier_blockers.append("missing_conservative_comparison_report")
    if not frontier_packet_path:
        frontier_blockers.append("missing_frozen_frontier_packet_path")
    frontier_packet = {
        "ready": not frontier_blockers,
        "packet_path": frontier_packet_path,
        "unfamiliar_domain_slice_ready": unfamiliar_domain_slice_ready,
        "strong_baseline_slice_ready": strong_baseline_slice_ready,
        "strong_baseline_summary": strong_baseline_summary,
        "long_horizon_transfer_slice_ready": long_horizon_transfer_slice_ready,
        "conservative_comparison_ready": conservative_comparison_ready,
        "blockers": frontier_blockers,
    }

    trust = report.get("trust_breadth_summary", {})
    if not isinstance(trust, Mapping):
        trust = {}
    baseline_blockers: list[str] = []
    if not str(official_anchor.get("report_path", "")).strip():
        baseline_blockers.append("missing_baseline_artifact_anchor")
    if not baseline_packet_path:
        baseline_blockers.append("missing_frozen_baseline_packet_path")
    if not trust:
        baseline_blockers.append("missing_baseline_trust_summary")
    baseline_packet = {
        "ready": not baseline_blockers,
        "packet_path": baseline_packet_path,
        "blockers": baseline_blockers,
    }

    primary_lane = str(lane.get("primary_lane", "")).strip()
    supporting_lanes = _normalize_benchmark_families(lane.get("supporting_lanes", []))
    lane_ids = _ordered_unique_strings(primary_lane, supporting_lanes)
    lane_blockers: list[str] = []
    if not primary_lane:
        lane_blockers.append("missing_primary_lane")
    for lane_id in lane_ids:
        if lane_id not in lane_packet_paths:
            lane_blockers.append(f"missing_lane_packet_path:{lane_id}")
    lane_packets = {
        "ready": not lane_blockers,
        "primary_lane": primary_lane,
        "supporting_lanes": supporting_lanes,
        "lane_packet_paths": lane_packet_paths,
        "blockers": lane_blockers,
    }

    packet_set_blockers: list[str] = []
    if not batch_packet["ready"]:
        packet_set_blockers.append("CodexBatchPacket")
    if not frontier_packet["ready"]:
        packet_set_blockers.append("CodexFrontierPacket")
    if not baseline_packet["ready"]:
        packet_set_blockers.append("CodexBaselinePacket")
    if not lane_packets["ready"]:
        packet_set_blockers.append("CodexLanePacket")

    suggested_actions: list[str] = []
    if not batch_packet["ready"]:
        suggested_actions.append("freeze_codex_batch_packet")
    if not frontier_packet["ready"]:
        if not unfamiliar_domain_slice_ready:
            suggested_actions.append("add_held_out_unfamiliar_domain_slice")
        if not strong_baseline_slice_ready:
            suggested_actions.append("add_strong_baseline_comparison_slice")
        if not long_horizon_transfer_slice_ready:
            suggested_actions.append("add_long_horizon_transfer_slice")
        if not conservative_comparison_ready:
            suggested_actions.append("add_conservative_comparison_report")
        suggested_actions.append("freeze_codex_frontier_packet")
    if not baseline_packet["ready"]:
        suggested_actions.append("freeze_codex_baseline_packet")
    if not lane_packets["ready"]:
        suggested_actions.append("freeze_codex_lane_packets")

    return {
        "packet_set_ready": not packet_set_blockers,
        "packet_set_blockers": list(dict.fromkeys(packet_set_blockers)),
        "batch_packet": batch_packet,
        "frontier_packet": frontier_packet,
        "baseline_packet": baseline_packet,
        "lane_packets": lane_packets,
        "suggested_actions": list(dict.fromkeys(suggested_actions)),
        "a8_minimum_frontier_inputs_ready": frontier_packet["ready"],
    }


def _build_repeated_codex_batch_packet(
    *,
    report_path: Path,
    status_path: Path,
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    report = payload if isinstance(payload, Mapping) else {}
    lane = report.get("asi_campaign_lane_recommendation", {})
    if not isinstance(lane, Mapping):
        lane = {}
    closure = report.get("closure_gap_summary", {})
    if not isinstance(closure, Mapping):
        closure = {}
    benchmark = report.get("benchmark_dominance_summary", {})
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    trust = report.get("trust_breadth_summary", {})
    if not isinstance(trust, Mapping):
        trust = {}
    gaps = _repeated_substantive_gaps(report, closure_gap_summary=closure)
    allowed_lanes = _ordered_unique_strings(
        lane.get("primary_lane", ""),
        lane.get("supporting_lanes", []),
    )
    non_upgrading_runs: list[dict[str, object]] = []
    for run in report.get("runs", []) if isinstance(report.get("runs", []), list) else []:
        if not isinstance(run, Mapping):
            continue
        if bool(run.get("retained_gain", False)):
            continue
        reason = "run produced no retained improvement"
        if int(run.get("returncode", 0) or 0) != 0:
            reason = "child run did not complete cleanly"
        elif str(run.get("decision_conversion_state", "")).strip() == "partial_productive_without_decision":
            reason = "run produced partial progress without a durable decision"
        elif str(run.get("final_state", "")).strip() == "reject":
            reason = "candidate was rejected and did not upgrade the retained baseline"
        non_upgrading_runs.append(
            {
                "run_index": int(run.get("index", 0) or 0),
                "decision_conversion_state": str(run.get("decision_conversion_state", "")).strip(),
                "final_state": str(run.get("final_state", "")).strip(),
                "why_not_anchor": reason,
            }
        )
    return {
        "packet_kind": "CodexBatchPacket",
        "packet_id": f"CodexBatchPacket:{report_path.stem}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_goal": str(lane.get("primary_batch_goal", "")).strip(),
        "batch_scope": "repeated_improvement_campaign",
        "official_anchor": {
            "status_path": str(status_path),
            "report_path": str(report_path),
            "repeated_status_path": str(status_path),
            "child_report_path": str(report_path),
            "anchor_date": str(report.get("created_at", "")).strip(),
        },
        "non_upgrading_runs": non_upgrading_runs[:3],
        "top_live_bottlenecks": [str(item.get("gap_id", "")).strip() for item in gaps[:3]],
        "top_substantive_gaps": gaps[:4],
        "closure_scoreboard": {
            "retained_conversion": str(closure.get("retained_conversion", "")).strip(),
            "trust_breadth": str(closure.get("trust_breadth", "")).strip(),
            "retrieval_carryover": str(closure.get("retrieval_carryover", "")).strip(),
            "long_horizon_finalize_execution": str(closure.get("long_horizon_finalize_execution", "")).strip(),
            "required_families": _normalize_benchmark_families(benchmark.get("required_families", [])),
            "contract_state": str(benchmark.get("contract_state", "")).strip(),
        },
        "frontier_scoreboard": {
            "a8_coding_frontier_ready": bool(
                isinstance(report.get("codex_input_packet_summary", {}), Mapping)
                and report.get("codex_input_packet_summary", {}).get("a8_minimum_frontier_inputs_ready", False)
            ),
            "packet_set_ready": bool(
                isinstance(report.get("codex_input_packet_summary", {}), Mapping)
                and report.get("codex_input_packet_summary", {}).get("packet_set_ready", False)
            ),
            "frontier_packet_path": str(report.get("codex_input_packets", {}).get("frontier_packet_path", "")).strip()
            if isinstance(report.get("codex_input_packets", {}), Mapping)
            else "",
            "baseline_packet_path": str(report.get("codex_input_packets", {}).get("baseline_packet_path", "")).strip()
            if isinstance(report.get("codex_input_packets", {}), Mapping)
            else "",
        },
        "reflection_records": [],
        "selection_records": [],
        "top_retained_reports": [],
        "top_rejected_reports": [],
        "trust_summary": dict(trust),
        "strategy_memory_summary": {
            "versioned_resource_lineage": False,
        },
        "frontier_packet_path": str(report.get("codex_input_packets", {}).get("frontier_packet_path", "")).strip()
        if isinstance(report.get("codex_input_packets", {}), Mapping)
        else "",
        "baseline_packet_path": str(report.get("codex_input_packets", {}).get("baseline_packet_path", "")).strip()
        if isinstance(report.get("codex_input_packets", {}), Mapping)
        else "",
        "allowed_lanes": allowed_lanes,
        "merge_order": [lane_id for lane_id in _REPEATED_PHASE_A_LANE_ORDER if lane_id in allowed_lanes],
        "compute_budget": {
            "cycles": int(report.get("cycles_requested", 0) or 0),
            "campaign_width": 0,
            "variant_width": 0,
            "task_limit": int(report.get("effective_task_limit", report.get("task_limit", 0)) or 0),
            "task_step_floor": 0,
        },
        "rerun_budget": {
            "rounds_requested": int(report.get("cycles_requested", 0) or 0),
            "child_failure_recovery_budget": 0,
        },
    }


def _build_repeated_codex_frontier_packet(
    *,
    report_path: Path,
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    report = payload if isinstance(payload, Mapping) else {}
    open_world = report.get("open_world_breadth_summary", {})
    if not isinstance(open_world, Mapping):
        open_world = {}
    partial_progress = report.get("partial_progress_summary", {})
    if not isinstance(partial_progress, Mapping):
        partial_progress = {}
    packet_summary = report.get("codex_input_packet_summary", {})
    if not isinstance(packet_summary, Mapping):
        packet_summary = {}
    frontier_summary = packet_summary.get("frontier_packet", {})
    if not isinstance(frontier_summary, Mapping):
        frontier_summary = {}
    strong_baseline_summary = _repeated_strong_baseline_comparison_summary(report)
    external_summary = open_world.get("external_summary", {})
    if not isinstance(external_summary, Mapping):
        external_summary = {}
    codex_input_packets = report.get("codex_input_packets", {})
    if not isinstance(codex_input_packets, Mapping):
        codex_input_packets = {}
    return {
        "packet_kind": "CodexFrontierPacket",
        "packet_id": f"CodexFrontierPacket:{report_path.stem}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frontier_version": report_path.stem,
        "held_out_task_manifest": {
            "benchmark_families": _normalize_benchmark_families(open_world.get("benchmark_families", [])),
            "external_benchmark_families": _normalize_benchmark_families(open_world.get("external_benchmark_families", [])),
            "semantic_hub_report_count": int(open_world.get("semantic_hub_report_count", 0) or 0),
            "external_report_count": int(open_world.get("external_report_count", 0) or 0),
        },
        "max_packet_age_seconds": 86400,
        "slice_freeze_window": "freeze until next repeated-cycle campaign report is emitted",
        "unfamiliar_domain_slices": {
            "open_world_report_count": int(open_world.get("open_world_report_count", 0) or 0),
            "benchmark_families": _normalize_benchmark_families(open_world.get("benchmark_families", [])),
        },
        "withheld_rotation_policy": "refresh only when the next repeated campaign report is written",
        "train_exclusion_manifest": [],
        "novelty_audit": {
            "distinct_external_benchmark_families": int(open_world.get("distinct_external_benchmark_families", 0) or 0),
            "distinct_open_world_benchmark_families": int(open_world.get("distinct_open_world_benchmark_families", 0) or 0),
        },
        "long_horizon_transfer_slices": {
            "generated_success_started": bool(int(partial_progress.get("generated_success_started_runs", 0) or 0)),
            "generated_success_completed": bool(int(partial_progress.get("generated_success_completed_runs", 0) or 0)),
            "partial_productive_runs": int(report.get("partial_productive_runs", 0) or 0),
        },
        "baseline_bundle": {
            "retained_kernel_baseline": str(codex_input_packets.get("baseline_packet_path", "")).strip(),
            "current_codex_guided_baseline": str(report_path),
            "artifact_anchor": str(report_path),
        },
        "comparison_commands": {
            "retained_baseline_compare": ["python", "scripts/compare_retained_baseline.py"],
            "frontier_eval": ["python", "scripts/run_autonomous_compounding_check.py"],
        },
        "family_conservative_bounds": {
            "external_summary": dict(external_summary),
            "strong_baseline_summary": dict(strong_baseline_summary),
        },
        "paired_task_trace_report": "",
        "paired_trajectory_report": "",
        "strong_baseline_comparison_summary": dict(strong_baseline_summary),
        "transfer_alignment_evidence": {},
        "long_horizon_evidence": dict(partial_progress),
        "frontier_acceptance_policy": {
            "minimum_non_regression_rate": 1.0,
            "minimum_baseline_win_rate": 0.0,
            "maximum_allowed_regressions": 0,
        },
        "blockers": [
            str(item).strip()
            for item in frontier_summary.get("blockers", [])
            if str(item).strip()
        ],
    }


def _build_repeated_codex_baseline_packet(
    *,
    report_path: Path,
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    report = payload if isinstance(payload, Mapping) else {}
    benchmark = report.get("benchmark_dominance_summary", {})
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    packet_summary = report.get("codex_input_packet_summary", {})
    if not isinstance(packet_summary, Mapping):
        packet_summary = {}
    baseline_summary = packet_summary.get("baseline_packet", {})
    if not isinstance(baseline_summary, Mapping):
        baseline_summary = {}
    codex_input_packets = report.get("codex_input_packets", {})
    if not isinstance(codex_input_packets, Mapping):
        codex_input_packets = {}
    return {
        "packet_kind": "CodexBaselinePacket",
        "packet_id": f"CodexBaselinePacket:{report_path.stem}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_type": "codex_guided_kernel",
        "artifact_paths": {
            "report_path": str(report_path),
            "child_report_path": str(report_path),
            "active_resource_path": "",
            "candidate_artifact_path": "",
        },
        "compare_commands": {
            "retained_baseline_compare": ["python", "scripts/compare_retained_baseline.py"],
            "frontier_eval": ["python", "scripts/run_autonomous_compounding_check.py"],
        },
        "baseline_metrics": {
            "runtime_managed_decisions": int(report.get("runtime_managed_decisions", 0) or 0),
            "retained_gain_runs": int(report.get("retained_gain_runs", 0) or 0),
        },
        "baseline_task_outcomes": dict(report.get("decision_conversion_summary", {}))
        if isinstance(report.get("decision_conversion_summary", {}), Mapping)
        else {},
        "baseline_trajectory_signature_summary": {
            "execution_source_summary": dict(report.get("execution_source_summary", {}))
            if isinstance(report.get("execution_source_summary", {}), Mapping)
            else {},
            "partial_progress_summary": dict(report.get("partial_progress_summary", {}))
            if isinstance(report.get("partial_progress_summary", {}), Mapping)
            else {},
        },
        "baseline_trust_summary": dict(report.get("trust_breadth_summary", {}))
        if isinstance(report.get("trust_breadth_summary", {}), Mapping)
        else {},
        "baseline_frontier_summary": dict(packet_summary.get("frontier_packet", {}))
        if isinstance(packet_summary.get("frontier_packet", {}), Mapping)
        else {},
        "matched_budget_policy": {
            "priority_benchmark_families": _normalize_benchmark_families(
                report.get("effective_priority_benchmark_families", report.get("priority_benchmark_families", []))
            ),
        },
        "baseline_freeze_window": "freeze until next repeated-cycle campaign report is emitted",
        "tool_permission_profile": {
            "claimed_runtime_shape": "repeated_improvement_campaign",
            "decoder_runtime_posture": {},
            "required_families": _normalize_benchmark_families(benchmark.get("required_families", [])),
        },
        "blockers": [
            str(item).strip()
            for item in baseline_summary.get("blockers", [])
            if str(item).strip()
        ],
    }


def _build_repeated_codex_lane_packets(
    *,
    report_path: Path,
    payload: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    report = payload if isinstance(payload, Mapping) else {}
    lane = report.get("asi_campaign_lane_recommendation", {})
    if not isinstance(lane, Mapping):
        lane = {}
    lanes = _ordered_unique_strings(lane.get("primary_lane", ""), lane.get("supporting_lanes", []))
    packets: dict[str, dict[str, object]] = {}
    for lane_id in lanes:
        spec = _REPEATED_LANE_PACKET_SPECS.get(lane_id, {})
        packets[lane_id] = {
            "packet_kind": "CodexLanePacket",
            "packet_id": f"CodexLanePacket:{lane_id}:{report_path.stem}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lane_id": lane_id,
            "question": str(lane.get("rationale", "")).strip(),
            "owned_paths": list(spec.get("owned_paths", [])) if isinstance(spec.get("owned_paths", []), list) else [],
            "explicitly_not_owned": [
                other_lane
                for other_lane in _REPEATED_PHASE_A_LANE_ORDER
                if other_lane != lane_id and other_lane in lanes
            ],
            "relevant_tests": list(spec.get("relevant_tests", [])) if isinstance(spec.get("relevant_tests", []), list) else [],
            "relevant_reports": [str(report_path)],
            "relevant_reflection_records": [],
            "relevant_selection_records": [],
            "relevant_strategy_memory_nodes": [],
            "failure_motifs": [
                str(item).strip()
                for item in lane.get("blockers", [])
                if str(item).strip()
            ],
            "expected_signals": [
                str(lane.get("primary_batch_goal", "")).strip(),
                str(report.get("codex_input_packet_summary", {}).get("a8_minimum_frontier_inputs_ready", ""))
                if isinstance(report.get("codex_input_packet_summary", {}), Mapping)
                else "",
            ],
            "done_when": str(spec.get("done_when", "")).strip(),
            "verification_plan": {
                "tests": list(spec.get("relevant_tests", [])) if isinstance(spec.get("relevant_tests", []), list) else [],
                "compare_commands": [
                    ["python", "scripts/compare_retained_baseline.py"],
                    ["python", "scripts/run_autonomous_compounding_check.py"],
                ],
            },
            "restart_requirement": (
                "resume from the latest repeated report and status packet because the last campaign ended interrupted"
                if _repeated_report_status(report) == "interrupted"
                else "continue from the latest repeated report and rerun the owned verification plan after edits"
            ),
        }
    return packets


def _attach_repeated_codex_packets(
    *,
    report_path: Path,
    status_path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    report = dict(payload)
    report["report_path"] = str(report_path)
    report["status_path"] = str(status_path)
    closure_gap_summary = _repeated_closure_gap_summary(report)
    benchmark_dominance_summary = _repeated_benchmark_dominance_summary(
        report,
        closure_gap_summary=closure_gap_summary,
    )
    lane_recommendation = _repeated_asi_campaign_lane_recommendation(
        report,
        closure_gap_summary=closure_gap_summary,
    )
    codex_input_packets = _repeated_codex_input_packet_paths(
        report_path=report_path,
        status_path=status_path,
        report_payload=report,
        lane_recommendation=lane_recommendation,
    )
    report["closure_gap_summary"] = closure_gap_summary
    report["benchmark_dominance_summary"] = benchmark_dominance_summary
    report["asi_campaign_lane_recommendation"] = lane_recommendation
    report["codex_input_packets"] = codex_input_packets
    report["codex_input_packet_summary"] = _repeated_codex_input_packet_summary(report)
    return report


def _write_repeated_codex_input_packets(
    *,
    config: KernelConfig,
    report_path: Path,
    status_path: Path,
    payload: Mapping[str, object] | None,
) -> None:
    report = payload if isinstance(payload, Mapping) else {}
    codex_input_packets = report.get("codex_input_packets", {})
    if not isinstance(codex_input_packets, Mapping) or not codex_input_packets:
        return
    batch_packet_path = str(codex_input_packets.get("batch_packet_path", "")).strip()
    frontier_packet_path = str(codex_input_packets.get("frontier_packet_path", "")).strip()
    baseline_packet_path = str(codex_input_packets.get("baseline_packet_path", "")).strip()
    lane_packet_paths = codex_input_packets.get("lane_packet_paths", {})
    if not isinstance(lane_packet_paths, Mapping):
        lane_packet_paths = {}
    if batch_packet_path:
        atomic_write_json(
            Path(batch_packet_path),
            _build_repeated_codex_batch_packet(report_path=report_path, status_path=status_path, payload=report),
            config=config,
            govern_storage=False,
        )
    if frontier_packet_path:
        atomic_write_json(
            Path(frontier_packet_path),
            _build_repeated_codex_frontier_packet(report_path=report_path, payload=report),
            config=config,
            govern_storage=False,
        )
    if baseline_packet_path:
        atomic_write_json(
            Path(baseline_packet_path),
            _build_repeated_codex_baseline_packet(report_path=report_path, payload=report),
            config=config,
            govern_storage=False,
        )
    lane_packets = _build_repeated_codex_lane_packets(report_path=report_path, payload=report)
    for lane_id, packet in lane_packets.items():
        lane_path = str(lane_packet_paths.get(lane_id, "")).strip()
        if lane_path:
            atomic_write_json(
                Path(lane_path),
                packet,
                config=config,
                govern_storage=False,
            )


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
    config: KernelConfig,
    planner: ImprovementPlanner,
    cycles_path: Path,
    campaign_match_id: str,
    cycle_log_start_index: int,
    priority_benchmark_families: list[str],
    priority_family_weights: dict[str, float],
) -> dict[str, object]:
    campaign_records = _campaign_records_with_report_fallback(
        config=config,
        planner=planner,
        cycles_path=cycles_path,
        campaign_match_id=campaign_match_id,
        start_index=cycle_log_start_index,
    )
    raw_decision_records = [
        record for record in campaign_records if str(record.get("state", "")) in {"retain", "reject"}
    ]
    runtime_managed_decision_records = _runtime_managed_cycle_decisions(
        _production_decisions(raw_decision_records)
    )
    non_runtime_managed_decision_records = _merge_decision_records(
        _non_runtime_managed_decisions(raw_decision_records),
        [],
    )
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
    current_cycle_ids = _ordered_unique_strings(current_cycle_ids)
    return {
        "campaign_records_considered": len(campaign_records),
        "decision_records_considered": (
            len(runtime_managed_decision_records) + len(non_runtime_managed_decision_records)
        ),
        "campaign_cycle_ids": current_cycle_ids,
        "priority_family_allocation_summary": priority_family_allocation_summary,
        "families_sampled": families_sampled,
        "priority_families_without_sampling": priority_families_without_sampling,
    }


def _preserved_campaign_status_snapshot_fields(snapshot: Mapping[str, object] | None) -> dict[str, object]:
    payload = snapshot if isinstance(snapshot, Mapping) else {}
    preserved: dict[str, object] = {}
    if "campaign_records_considered" in payload:
        preserved["campaign_records_considered"] = max(
            0,
            int(payload.get("campaign_records_considered", 0) or 0),
        )
    if "decision_records_considered" in payload:
        preserved["decision_records_considered"] = max(
            0,
            int(payload.get("decision_records_considered", 0) or 0),
        )
    campaign_cycle_ids = _ordered_unique_strings(payload.get("campaign_cycle_ids", []))
    if campaign_cycle_ids:
        preserved["campaign_cycle_ids"] = campaign_cycle_ids
    priority_family_allocation_summary = payload.get("priority_family_allocation_summary", {})
    if isinstance(priority_family_allocation_summary, Mapping) and priority_family_allocation_summary:
        preserved["priority_family_allocation_summary"] = dict(priority_family_allocation_summary)
    return preserved


def _run_has_campaign_evidence(run: Mapping[str, object]) -> bool:
    if not isinstance(run, Mapping):
        return False
    if bool(run.get("productive", False)) or bool(run.get("partial_productive", False)):
        return True
    if bool(run.get("retained_gain", False)):
        return True
    if int(run.get("runtime_managed_decisions", 0) or 0) > 0:
        return True
    if int(run.get("decision_records_considered", 0) or 0) > 0:
        return True
    if str(run.get("final_state", "")).strip() in {"retain", "reject"}:
        return True
    if str(run.get("final_state", "")).strip() == "incomplete":
        return True
    if str(run.get("decision_conversion_state", "")).strip() in {"runtime_managed", "non_runtime_managed"}:
        return True
    cycle_ids = run.get("campaign_cycle_ids", [])
    if isinstance(cycle_ids, list) and any(str(value).strip() for value in cycle_ids):
        return True
    return bool(str(run.get("cycle_id", "")).strip())


def _run_campaign_cycle_ids(run: Mapping[str, object] | None) -> list[str]:
    payload = run if isinstance(run, Mapping) else {}
    cycle_ids = payload.get("campaign_cycle_ids", [])
    if isinstance(cycle_ids, list):
        normalized = _ordered_unique_strings(
            [str(value).strip() for value in cycle_ids if str(value).strip()]
        )
        if normalized:
            return normalized
    cycle_id = str(payload.get("cycle_id", "")).strip()
    return [cycle_id] if cycle_id else []


def _run_runtime_managed_decision_count(run: Mapping[str, object] | None) -> int:
    payload = run if isinstance(run, Mapping) else {}
    explicit_count = max(0, int(payload.get("runtime_managed_decisions", 0) or 0))
    if explicit_count <= 0:
        return 0
    cycle_ids = _run_campaign_cycle_ids(payload)
    if cycle_ids:
        return min(explicit_count, len(cycle_ids))
    if (
        str(payload.get("final_state", "")).strip() in {"retain", "reject"}
        or str(payload.get("decision_conversion_state", "")).strip() == "runtime_managed"
    ):
        return 1
    return explicit_count


def _run_decision_evidence_count(run: Mapping[str, object]) -> int:
    if not isinstance(run, Mapping):
        return 0
    explicit_count = max(0, int(run.get("decision_records_considered", 0) or 0))
    runtime_managed_decisions = _run_runtime_managed_decision_count(run)
    if explicit_count > 0:
        if runtime_managed_decisions > 0 and max(0, int(run.get("non_runtime_managed_decisions", 0) or 0)) <= 0:
            return min(explicit_count, runtime_managed_decisions)
        return explicit_count
    if runtime_managed_decisions > 0:
        return runtime_managed_decisions
    if bool(run.get("retained_gain", False)):
        return 1
    if str(run.get("final_state", "")).strip() in {"retain", "reject"}:
        return 1
    if str(run.get("decision_conversion_state", "")).strip() in {"runtime_managed", "non_runtime_managed"}:
        return 1
    return 0


def _run_status_evidence_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    campaign_records_considered = 0
    decision_records_considered = 0
    runtime_managed_decisions = 0
    non_runtime_managed_decisions = 0
    sampled_families_from_progress: list[str] = []
    execution_source_summary = {
        "decoder_generated": 0,
        "llm_generated": 0,
        "bounded_decoder_generated": 0,
        "synthetic_plan": 0,
        "deterministic_or_other": 0,
        "total_executed_commands": 0,
    }
    for run in runs:
        if not isinstance(run, dict):
            continue
        if _run_has_campaign_evidence(run):
            campaign_records_considered += 1
        decision_records_considered += _run_decision_evidence_count(run)
        runtime_managed_decisions += _run_runtime_managed_decision_count(run)
        non_runtime_managed_decisions += max(0, int(run.get("non_runtime_managed_decisions", 0) or 0))
        run_execution_summary = run.get("execution_source_summary", {})
        if isinstance(run_execution_summary, Mapping):
            decoder_generated = max(
                0,
                int(
                    run_execution_summary.get(
                        "decoder_generated",
                        run_execution_summary.get("llm_generated", 0),
                    )
                    or 0
                ),
            )
            llm_generated = max(0, int(run_execution_summary.get("llm_generated", 0) or 0))
            bounded_decoder_generated = max(
                0,
                int(
                    run_execution_summary.get(
                        "bounded_decoder_generated",
                        max(0, decoder_generated - llm_generated),
                    )
                    or 0
                ),
            )
            execution_source_summary["decoder_generated"] += max(
                0,
                decoder_generated,
            )
            execution_source_summary["llm_generated"] += llm_generated
            execution_source_summary["bounded_decoder_generated"] += bounded_decoder_generated
            execution_source_summary["synthetic_plan"] += max(
                0,
                int(run_execution_summary.get("synthetic_plan", 0) or 0),
            )
            execution_source_summary["deterministic_or_other"] += max(
                0,
                int(run_execution_summary.get("deterministic_or_other", 0) or 0),
            )
            execution_source_summary["total_executed_commands"] += max(
                0,
                int(run_execution_summary.get("total_executed_commands", 0) or 0),
            )
        partial_progress = run.get("partial_progress", {})
        if isinstance(partial_progress, dict):
            sampled_families_from_progress = _ordered_unique_strings(
                sampled_families_from_progress,
                partial_progress.get("sampled_families_from_progress", []),
            )
    return {
        "campaign_records_considered": campaign_records_considered,
        "decision_records_considered": decision_records_considered,
        "runtime_managed_decisions": runtime_managed_decisions,
        "non_runtime_managed_decisions": non_runtime_managed_decisions,
        "execution_source_summary": execution_source_summary,
        "sampled_families_from_progress": sampled_families_from_progress,
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
    non_runtime_managed_decisions: int = 0,
    partial_productive: bool,
    incomplete_cycle_count: int = 0,
) -> str:
    normalized_state = str(final_state).strip()
    if runtime_managed_decisions > 0:
        return "runtime_managed"
    if non_runtime_managed_decisions > 0:
        return "non_runtime_managed"
    if normalized_state == "incomplete" or incomplete_cycle_count > 0:
        return "incomplete"
    if normalized_state in {"retain", "reject"}:
        return "non_runtime_managed"
    if partial_productive:
        return "partial_productive_without_decision"
    return "no_decision"


def _summarize_run_decision_conversion(runs: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "runtime_managed_runs": 0,
        "non_runtime_managed_runs": 0,
        "incomplete_runs": 0,
        "partial_productive_without_decision_runs": 0,
        "no_decision_runs": 0,
    }
    for run in runs:
        state = str(run.get("decision_conversion_state", "")).strip()
        if state == "runtime_managed":
            summary["runtime_managed_runs"] += 1
        elif state == "non_runtime_managed":
            summary["non_runtime_managed_runs"] += 1
        elif state == "incomplete":
            summary["incomplete_runs"] += 1
        elif state == "partial_productive_without_decision":
            summary["partial_productive_without_decision_runs"] += 1
        else:
            summary["no_decision_runs"] += 1
    summary["decision_runs"] = summary["runtime_managed_runs"] + summary["non_runtime_managed_runs"]
    return summary


def _preferred_parent_live_progress_state(child_status_payload: Mapping[str, object] | None) -> dict[str, object]:
    payload = child_status_payload if isinstance(child_status_payload, Mapping) else {}
    for key in ("canonical_progress_state", "semantic_progress_state", "adaptive_progress_state"):
        state = payload.get(key, {})
        if isinstance(state, Mapping) and state:
            return dict(state)
    active_cycle_run = payload.get("active_cycle_run", {})
    if isinstance(active_cycle_run, Mapping):
        for key in ("semantic_progress_state", "adaptive_progress_state"):
            state = active_cycle_run.get(key, {})
            if isinstance(state, Mapping) and state:
                return dict(state)
    return {}


def _parent_live_task_projection(child_status_payload: Mapping[str, object] | None) -> dict[str, object]:
    payload = child_status_payload if isinstance(child_status_payload, Mapping) else {}
    active_cycle_run = payload.get("active_cycle_run", {})
    if not isinstance(active_cycle_run, Mapping):
        active_cycle_run = {}
    current_task = payload.get("current_task", {})
    if not isinstance(current_task, Mapping) or not current_task:
        current_task = active_cycle_run.get("current_task", {})
    if not isinstance(current_task, Mapping):
        current_task = {}
    current_cognitive_stage = payload.get("current_cognitive_stage", {})
    if not isinstance(current_cognitive_stage, Mapping) or not current_cognitive_stage:
        current_cognitive_stage = active_cycle_run.get("current_cognitive_stage", {})
    if not isinstance(current_cognitive_stage, Mapping):
        current_cognitive_stage = {}
    progress_state = _preferred_parent_live_progress_state(payload)
    last_progress_phase = str(payload.get("last_progress_phase", "")).strip()
    if not last_progress_phase:
        last_progress_phase = str(active_cycle_run.get("last_progress_phase", "")).strip()
    if not last_progress_phase and current_task:
        last_progress_phase = str(current_task.get("phase", "")).strip()
    projection = {
        "last_progress_phase": last_progress_phase,
        "current_task": dict(current_task),
        "current_cognitive_stage": dict(current_cognitive_stage),
        "canonical_progress_state": dict(progress_state),
        "semantic_progress_state": dict(progress_state),
    }
    for key in (
        "current_task_verification_passed",
        "selected_subsystem",
        "verification_outcome_summary",
        "runtime_managed_decisions",
        "non_runtime_managed_decisions",
        "retained_gain_runs",
    ):
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, Mapping):
            projection[key] = dict(value)
        else:
            projection[key] = value
    return projection


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
    parent_status = str(parent_payload.get("status", "")).strip()
    parent_phase = str(parent_payload.get("phase", "")).strip()
    if parent_status in {"completed", "safe_stop"} or parent_phase == "completed":
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
    parent_payload["active_child"] = dict(child_status_payload)
    parent_payload.update(_parent_live_task_projection(child_status_payload))
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
    reason: str = "",
    active_cycle_run: dict[str, object] | None = None,
    snapshot: dict[str, object] | None = None,
    max_progress_stall_seconds: float = 0.0,
    max_runtime_seconds: float = 0.0,
) -> Path:
    if snapshot is None:
        snapshot = _campaign_status_snapshot(
            config=config,
            planner=planner,
            cycles_path=cycles_path,
            campaign_match_id=campaign_match_id,
            cycle_log_start_index=cycle_log_start_index,
            priority_benchmark_families=priority_benchmark_families,
            priority_family_weights=priority_family_weights,
        )
    snapshot_payload = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    progress_summary = _partial_progress_summary(
        active_cycle_run,
        priority_benchmark_families=priority_benchmark_families,
    )
    semantic_state = _active_cycle_semantic_progress_state(
        active_cycle_run,
        max_progress_stall_seconds=max_progress_stall_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )
    normalized_active_cycle_run = _normalize_active_cycle_run_for_status(
        active_cycle_run,
        semantic_state=semantic_state,
    )
    active_verification_summary = _verification_outcome_summary(active_cycle_run)
    active_started_at = float((active_cycle_run or {}).get("started_at", 0.0) or 0.0)
    active_since_iso = (
        datetime.fromtimestamp(active_started_at, timezone.utc).isoformat()
        if active_started_at > 0.0
        else ""
    )
    active_execution_source_summary = _execution_source_summary_for_task_ids(
        config,
        task_ids=_run_execution_source_task_ids(
            {
                **(normalized_active_cycle_run if isinstance(normalized_active_cycle_run, dict) else {}),
                "verification_outcome_summary": active_verification_summary,
                "partial_progress": progress_summary,
            }
        ),
        since_iso=active_since_iso,
    )
    active_trust_breadth_summary, active_open_world_breadth_summary = _trust_and_open_world_breadth_summaries(config)
    resolved_snapshot = _preserved_campaign_status_snapshot_fields(snapshot_payload)
    run_evidence_summary = _run_status_evidence_summary(runs)
    merged_execution_source_summary = _merge_execution_source_summaries(
        run_evidence_summary.get("execution_source_summary", {}),
        active_execution_source_summary,
    )
    resolved_snapshot["campaign_records_considered"] = max(
        int(resolved_snapshot.get("campaign_records_considered", 0) or 0),
        int(run_evidence_summary.get("campaign_records_considered", 0) or 0),
    )
    resolved_snapshot["decision_records_considered"] = max(
        int(resolved_snapshot.get("decision_records_considered", 0) or 0),
        int(run_evidence_summary.get("decision_records_considered", 0) or 0),
    )
    pending_decision_state = ""
    if isinstance(active_cycle_run, dict):
        pending_decision_state = str(active_cycle_run.get("pending_decision_state", "")).strip()
    if pending_decision_state in {"retain", "reject"}:
        resolved_snapshot["pending_decision_state"] = pending_decision_state
        if int(resolved_snapshot.get("decision_records_considered", 0) or 0) <= 0:
            resolved_snapshot["decision_records_considered"] = 1
    preview_state = ""
    if isinstance(active_cycle_run, dict):
        preview_state = str(active_cycle_run.get("preview_state", "")).strip()
    if preview_state in {"retain", "reject"}:
        resolved_snapshot["preview_state"] = preview_state
    families_sampled = _ordered_unique_strings(
        snapshot_payload.get("families_sampled", []),
        run_evidence_summary.get("sampled_families_from_progress", []),
        progress_summary.get("sampled_families_from_progress", []),
    )
    priority_families_without_sampling = [
        family for family in priority_benchmark_families if family not in set(families_sampled)
    ]
    active_current_task = normalized_active_cycle_run.get("current_task", {})
    if not isinstance(active_current_task, Mapping):
        active_current_task = {}
    active_current_cognitive_stage = normalized_active_cycle_run.get("current_cognitive_stage", {})
    if not isinstance(active_current_cognitive_stage, Mapping):
        active_current_cognitive_stage = {}
    active_current_task_progress_timeline = normalized_active_cycle_run.get("current_task_progress_timeline", [])
    if not isinstance(active_current_task_progress_timeline, list):
        active_current_task_progress_timeline = []
    selected_subsystem = str(normalized_active_cycle_run.get("selected_subsystem", "")).strip()
    finalize_phase = str(normalized_active_cycle_run.get("finalize_phase", "")).strip()
    last_progress_phase = str(normalized_active_cycle_run.get("last_progress_phase", "")).strip()
    last_progress_line = str(normalized_active_cycle_run.get("last_progress_line", "")).strip()
    last_output_line = str(
        normalized_active_cycle_run.get("last_output_line", "") or last_progress_line
    ).strip()
    incomplete_cycles = _merge_incomplete_cycle_summaries([], runs, [])
    terminal_campaign_projection = (
        _terminal_campaign_projection(
            runs,
            include_semantic_progress_state=True,
        )
        if state == "finished" and not normalized_active_cycle_run
        else {}
    )
    resolved_report_path = str(report_path) if report_path.exists() else ""
    pending_report_path = "" if resolved_report_path else str(report_path)
    report_payload = _load_json_payload(report_path) if resolved_report_path else {}
    report_codex_input_packets = report_payload.get("codex_input_packets", {})
    if not isinstance(report_codex_input_packets, Mapping):
        report_codex_input_packets = {}
    report_codex_input_packet_summary = report_payload.get("codex_input_packet_summary", {})
    if not isinstance(report_codex_input_packet_summary, Mapping):
        report_codex_input_packet_summary = {}
    report_lane_recommendation = report_payload.get("asi_campaign_lane_recommendation", {})
    if not isinstance(report_lane_recommendation, Mapping):
        report_lane_recommendation = {}
    report_closure_gap_summary = report_payload.get("closure_gap_summary", {})
    if not isinstance(report_closure_gap_summary, Mapping):
        report_closure_gap_summary = {}
    report_benchmark_dominance_summary = report_payload.get("benchmark_dominance_summary", {})
    if not isinstance(report_benchmark_dominance_summary, Mapping):
        report_benchmark_dominance_summary = {}
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "repeated_improvement_status",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_path": resolved_report_path,
        **({"pending_report_path": pending_report_path} if pending_report_path else {}),
        "state": state,
        "status": state,
        **({"reason": str(reason).strip()} if str(reason).strip() else {}),
        "campaign_label": campaign_label,
        "campaign_match_id": campaign_match_id,
        "cycles_requested": cycles_requested,
        "completed_runs": len(runs),
        "successful_runs": sum(1 for run in runs if int(run.get("returncode", 1)) == 0),
        "productive_runs": sum(1 for run in runs if bool(run.get("productive", False))),
        "retained_gain_runs": sum(1 for run in runs if bool(run.get("retained_gain", False))),
        "partial_productive_runs": sum(1 for run in runs if bool(run.get("partial_productive", False))),
        "partial_candidate_runs": sum(1 for run in runs if bool(run.get("partial_candidate_generated", False))),
        "runtime_managed_decisions": int(run_evidence_summary.get("runtime_managed_decisions", 0) or 0),
        "non_runtime_managed_decisions": int(run_evidence_summary.get("non_runtime_managed_decisions", 0) or 0),
        "execution_source_summary": merged_execution_source_summary,
        "current_task": (
            dict(active_current_task)
            if active_current_task
            else {}
        ),
        "verification_outcome_summary": active_verification_summary,
        "trust_breadth_summary": active_trust_breadth_summary,
        "open_world_breadth_summary": active_open_world_breadth_summary,
        "decision_conversion_summary": _summarize_run_decision_conversion(runs),
        "decision_state_summary": _decision_state_summary(runs),
        "incomplete_cycle_summary": {
            "count": len(incomplete_cycles),
            "cycle_ids": [str(item.get("cycle_id", "")).strip() for item in incomplete_cycles],
            "subsystems": [str(item.get("subsystem", "")).strip() for item in incomplete_cycles],
        },
        "priority_benchmark_families": priority_benchmark_families,
        "priority_benchmark_family_weights": dict(priority_family_weights),
        "external_task_manifests_paths": [
            str(path).strip()
            for path in getattr(config, "external_task_manifests_paths", ())
            if str(path).strip()
        ],
        "active_cycle_run": {
            **normalized_active_cycle_run,
            "verification_outcome_summary": active_verification_summary,
            "execution_source_summary": active_execution_source_summary,
            "trust_breadth_summary": active_trust_breadth_summary,
            "open_world_breadth_summary": active_open_world_breadth_summary,
            **(
                {
                    "tolbert_runtime_summary": _normalize_tolbert_runtime_summary(
                        normalized_active_cycle_run.get("tolbert_runtime_summary")
                    )
                }
                if isinstance(normalized_active_cycle_run.get("tolbert_runtime_summary"), Mapping)
                else {}
            ),
            **({"semantic_progress_state": semantic_state} if semantic_state else {}),
        },
        "recent_structured_child_decisions": [
            {
                **dict(record),
                "decision_state": _decision_state_from_record(record),
            }
            for record in _structured_non_runtime_decisions(runs)[-10:]
        ],
        "progress_events_observed": int((active_cycle_run or {}).get("progress_event_count", 0) or 0),
        "progress_output_lines_observed": int((active_cycle_run or {}).get("progress_output_count", 0) or 0),
        "tolbert_runtime_summary": _tolbert_runtime_status_summary(runs, active_cycle_run),
        **({"selected_subsystem": selected_subsystem} if selected_subsystem else {}),
        **({"finalize_phase": finalize_phase} if finalize_phase else {}),
        **({"last_progress_phase": last_progress_phase} if last_progress_phase else {}),
        **({"last_progress_line": last_progress_line} if last_progress_line else {}),
        **({"last_output_line": last_output_line} if last_output_line else {}),
        **(
            {"current_cognitive_stage": dict(active_current_cognitive_stage)}
            if active_current_cognitive_stage
            else {}
        ),
        **(
            {"current_task_progress_timeline": list(active_current_task_progress_timeline)}
            if "current_task_progress_timeline" in normalized_active_cycle_run
            else {}
        ),
        **(
            {
                "current_task_verification_passed": normalized_active_cycle_run.get(
                    "current_task_verification_passed"
                )
            }
            if "current_task_verification_passed" in normalized_active_cycle_run
            else {}
        ),
        **({"pending_decision_state": pending_decision_state} if pending_decision_state else {}),
        **({"preview_state": preview_state} if preview_state else {}),
        **resolved_snapshot,
        **terminal_campaign_projection,
        "families_sampled": families_sampled,
        "priority_families_without_sampling": priority_families_without_sampling,
        "active_cycle_progress": progress_summary,
        "semantic_progress_state": terminal_campaign_projection.get("semantic_progress_state", semantic_state),
        **({"codex_input_packets": dict(report_codex_input_packets)} if report_codex_input_packets else {}),
        **(
            {"codex_input_packet_summary": dict(report_codex_input_packet_summary)}
            if report_codex_input_packet_summary
            else {}
        ),
        **(
            {"asi_campaign_lane_recommendation": dict(report_lane_recommendation)}
            if report_lane_recommendation
            else {}
        ),
        **({"closure_gap_summary": dict(report_closure_gap_summary)} if report_closure_gap_summary else {}),
        **(
            {"benchmark_dominance_summary": dict(report_benchmark_dominance_summary)}
            if report_benchmark_dominance_summary
            else {}
        ),
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
    parser.add_argument("--semantic-only-runtime", action="store_true")
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
    parser.add_argument("--allow-transition-model-fallback", action="store_true")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--prefer-retrieval-observe", action="store_true")
    args = parser.parse_args()
    args.exclude_subsystem = _ordered_unique_strings(args.exclude_subsystem)
    if args.semantic_only_runtime:
        args.exclude_subsystem = _ordered_unique_strings(args.exclude_subsystem, ["tolbert_model"])

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
    if args.semantic_only_runtime:
        # Semantic-only runtime still needs Tolbert retrieval/context at execution time;
        # it only disables model-artifact mutation lanes such as Tolbert fine-tuning.
        config.use_tolbert_model_artifacts = False
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
    effective_external_task_manifests = _effective_external_task_manifests_paths(
        config,
        repo_root=repo_root,
    )
    if effective_external_task_manifests:
        config.external_task_manifests_paths = effective_external_task_manifests
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
    campaign_started_at_iso = datetime.now(timezone.utc).isoformat()

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
            max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
            max_runtime_seconds=float(args.max_child_runtime_seconds),
        )
        generate_index: dict[str, dict[str, object]] = {}
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
            if args.allow_transition_model_fallback:
                cmd.append("--allow-transition-model-fallback")
            for flag, enabled in (
                ("--include-episode-memory", args.include_episode_memory),
                ("--include-skill-memory", args.include_skill_memory),
                ("--include-skill-transfer", args.include_skill_transfer),
                ("--include-operator-memory", args.include_operator_memory),
                ("--include-tool-memory", args.include_tool_memory),
                ("--include-verifier-memory", args.include_verifier_memory),
                ("--include-curriculum", args.include_curriculum),
                ("--include-failure-curriculum", args.include_failure_curriculum),
                ("--prefer-retrieval-observe", args.prefer_retrieval_observe),
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
            _initialize_child_runtime_database(env)
            active_cycle_run = {
                "index": index,
                "progress_label": campaign_label or f"cycle-{index}",
                "campaign_width": current_campaign_width,
                "variant_width": current_variant_width,
                "task_limit": current_task_limit,
                "adaptive_search": bool(current_adaptive_search),
                "priority_benchmark_families": list(current_priority_benchmark_families),
                "priority_benchmark_family_weights": dict(current_priority_family_weights),
                "tolbert_runtime_summary": {
                    "configured_to_use_tolbert": bool(config.use_tolbert_context),
                },
            }
            status_path = _write_campaign_status(
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
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
            )
            status_snapshot_cache = _load_json_payload(status_path)

            def _refresh_child_status(event: dict[str, object]) -> dict[str, object] | None:
                nonlocal last_status_refresh_at, last_snapshot_refresh_at, active_cycle_run, status_snapshot_cache
                event_kind = str(event.get("event", "")).strip()
                line = str(event.get("line", "")).strip()
                event_timestamp = float(event.get("timestamp", 0.0) or time.time())
                active_cycle_run = {
                    **active_cycle_run,
                    "pid": int(event.get("pid", 0) or active_cycle_run.get("pid", 0) or 0),
                    "last_event": event_kind,
                    "last_event_at": event_timestamp,
                    "progress_event_count": int(active_cycle_run.get("progress_event_count", 0) or 0) + 1,
                }
                if event_kind == "start":
                    active_cycle_run = _reset_active_cycle_run_verification_state(active_cycle_run)
                    started_at = float(event.get("started_at", 0.0) or time.time())
                    active_cycle_run["started_at"] = started_at
                    active_cycle_run["last_progress_at"] = started_at
                    active_cycle_run.pop("child_exit_detected", None)
                    active_cycle_run.pop("child_exit_detected_at", None)
                    active_cycle_run.pop("child_exit_confirmed_at", None)
                    active_cycle_run.pop("output_buffer_draining_after_exit", None)
                    active_cycle_run.pop("child_returncode", None)
                if event_kind == "exit_detected":
                    active_cycle_run["child_exit_detected"] = True
                    active_cycle_run["child_exit_detected_at"] = event_timestamp
                    active_cycle_run["output_buffer_draining_after_exit"] = True
                    active_cycle_run["child_returncode"] = int(
                        event.get("returncode", 0) or active_cycle_run.get("child_returncode", 0) or 0
                    )
                if event_kind == "exit":
                    active_cycle_run["child_exit_detected"] = True
                    active_cycle_run["child_exit_confirmed_at"] = event_timestamp
                    active_cycle_run["output_buffer_draining_after_exit"] = False
                    active_cycle_run["child_returncode"] = int(
                        event.get("returncode", 0) or active_cycle_run.get("child_returncode", 0) or 0
                    )
                if event_kind == "heartbeat":
                    active_cycle_run["silence_seconds"] = int(event.get("silence_seconds", 0) or 0)
                if line:
                    active_cycle_run["progress_output_count"] = int(active_cycle_run.get("progress_output_count", 0) or 0) + 1
                    active_cycle_run["last_output_line"] = line
                    if (
                        ("[cycle:" in line or "[eval:" in line or "finalize phase=" in line)
                        and "still_running silence=" not in line
                    ):
                        active_cycle_run["last_progress_line"] = line
                        active_cycle_run["last_progress_at"] = event_timestamp
                    previous_task = active_cycle_run.get("current_task", {})
                    parsed_progress = _parse_progress_fields(line)
                    parsed_tolbert_summary = parsed_progress.pop("tolbert_runtime_summary", None)
                    if isinstance(parsed_tolbert_summary, Mapping):
                        active_cycle_run["tolbert_runtime_summary"] = _merge_tolbert_runtime_summaries(
                            active_cycle_run.get("tolbert_runtime_summary"),
                            parsed_tolbert_summary,
                        )
                    active_cycle_run.update(parsed_progress)
                    current_task = active_cycle_run.get("current_task", {})
                    if isinstance(current_task, dict):
                        task_transition = _task_progress_identity(current_task) != _task_progress_identity(previous_task)
                        if task_transition:
                            active_cycle_run.pop("current_task_verification_passed", None)
                            active_cycle_run["current_task_progress_timeline"] = []
                            active_cycle_run["current_cognitive_stage"] = {}
                        family = str(current_task.get("family", "")).strip()
                        if family:
                            active_cycle_run["sampled_families_from_progress"] = _ordered_unique_strings(
                                active_cycle_run.get("sampled_families_from_progress", []),
                                [family],
                            )
                    cognitive_progress = _parse_cognitive_progress_fields(line)
                    if cognitive_progress:
                        previous_cognitive_stage = active_cycle_run.get("current_cognitive_stage", {})
                        if not isinstance(previous_cognitive_stage, Mapping):
                            previous_cognitive_stage = {}
                        timeline = list(active_cycle_run.get("current_task_progress_timeline", []) or [])
                        timeline.append(cognitive_progress)
                        if len(timeline) > 64:
                            timeline = timeline[-64:]
                        active_cycle_run["current_task_progress_timeline"] = timeline
                        verification_state = _updated_task_verification_state(
                            current_task_verification_passed=active_cycle_run.get(
                                "current_task_verification_passed"
                            ),
                            previous_cognitive_stage=previous_cognitive_stage,
                            next_cognitive_stage=cognitive_progress,
                        )
                        if verification_state is None:
                            active_cycle_run.pop("current_task_verification_passed", None)
                        else:
                            active_cycle_run["current_task_verification_passed"] = verification_state
                        active_cycle_run["current_cognitive_stage"] = dict(cognitive_progress)
                        if "verification_passed" in cognitive_progress:
                            current_task_payload = active_cycle_run.get("current_task", {})
                            if not isinstance(current_task_payload, Mapping):
                                current_task_payload = {}
                            task_id = str(current_task_payload.get("task_id", "")).strip()
                            family = str(current_task_payload.get("family", "")).strip()
                            if task_id:
                                active_cycle_run["verified_task_ids"] = _ordered_unique_strings(
                                    active_cycle_run.get("verified_task_ids", []),
                                    [task_id],
                                )
                                verification_outcomes_by_task = active_cycle_run.get(
                                    "verification_outcomes_by_task", {}
                                )
                                if not isinstance(verification_outcomes_by_task, Mapping):
                                    verification_outcomes_by_task = {}
                                verification_outcomes_by_task = dict(verification_outcomes_by_task)
                                verification_outcomes_by_task[task_id] = bool(
                                    cognitive_progress.get("verification_passed")
                                )
                                active_cycle_run["verification_outcomes_by_task"] = verification_outcomes_by_task
                            if family:
                                active_cycle_run["verified_families"] = _ordered_unique_strings(
                                    active_cycle_run.get("verified_families", []),
                                    [family],
                                )
                            if task_id and family:
                                verification_task_families = active_cycle_run.get("verification_task_families", {})
                                if not isinstance(verification_task_families, Mapping):
                                    verification_task_families = {}
                                verification_task_families = dict(verification_task_families)
                                verification_task_families[task_id] = family
                                active_cycle_run["verification_task_families"] = verification_task_families
                            if bool(cognitive_progress.get("verification_passed")):
                                if task_id:
                                    active_cycle_run["successful_verification_task_ids"] = _ordered_unique_strings(
                                        active_cycle_run.get("successful_verification_task_ids", []),
                                        [task_id],
                                    )
                                    active_cycle_run["failed_verification_task_ids"] = [
                                        existing_task_id
                                        for existing_task_id in active_cycle_run.get(
                                            "failed_verification_task_ids", []
                                        )
                                        if str(existing_task_id).strip() != task_id
                                    ]
                                if family:
                                    active_cycle_run["successful_verification_families"] = _ordered_unique_strings(
                                        active_cycle_run.get("successful_verification_families", []),
                                        [family],
                                    )
                            else:
                                if task_id:
                                    active_cycle_run["failed_verification_task_ids"] = _ordered_unique_strings(
                                        active_cycle_run.get("failed_verification_task_ids", []),
                                        [task_id],
                                    )
                                    active_cycle_run["successful_verification_task_ids"] = [
                                        existing_task_id
                                        for existing_task_id in active_cycle_run.get(
                                            "successful_verification_task_ids", []
                                        )
                                        if str(existing_task_id).strip() != task_id
                                    ]
                                if family:
                                    active_cycle_run["failed_verification_families"] = _ordered_unique_strings(
                                        active_cycle_run.get("failed_verification_families", []),
                                        [family],
                                    )
                            active_cycle_run["verification_outcome_summary"] = _verification_outcome_summary(
                                active_cycle_run
                            )
                        inferred_tolbert_summary = _tolbert_runtime_summary_from_cognitive_progress(
                            cognitive_progress,
                            active_cycle_run=active_cycle_run,
                        )
                        if inferred_tolbert_summary:
                            active_cycle_run["tolbert_runtime_summary"] = _merge_tolbert_runtime_summaries(
                                active_cycle_run.get("tolbert_runtime_summary"),
                                inferred_tolbert_summary,
                            )
                refresh_now = event_kind in {"start", "heartbeat", "timeout", "exit"}
                if not refresh_now and line and (
                    "[cycle:" in line or "[eval:" in line or "finalize phase=" in line or "[repeated]" in line
                ):
                    refresh_now = True
                now = time.monotonic()
                if not refresh_now and (now - last_status_refresh_at) < 2.0:
                    intervention = _mid_cycle_intervention_signal(active_cycle_run)
                    if bool(intervention.get("triggered", False)):
                        return {
                            "terminate": True,
                            "reason": str(intervention.get("reason", "")).strip(),
                        }
                    return None
                last_status_refresh_at = now
                force_snapshot_refresh = event_kind in {"start", "timeout", "exit"}
                if not force_snapshot_refresh and (now - last_snapshot_refresh_at) >= 30.0:
                    force_snapshot_refresh = True
                if force_snapshot_refresh:
                    status_snapshot_cache = _campaign_status_snapshot(
                        config=config,
                        planner=planner,
                        cycles_path=config.improvement_cycles_path,
                        campaign_match_id=campaign_match_id,
                        cycle_log_start_index=cycle_log_start_index,
                        priority_benchmark_families=current_priority_benchmark_families,
                        priority_family_weights=current_priority_family_weights,
                    )
                    last_snapshot_refresh_at = now
                status_path = _write_campaign_status(
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
                    max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                    max_runtime_seconds=float(args.max_child_runtime_seconds),
                )
                status_snapshot_cache = _load_json_payload(status_path)
                intervention = _mid_cycle_intervention_signal(active_cycle_run)
                if bool(intervention.get("triggered", False)):
                    return {
                        "terminate": True,
                        "reason": str(intervention.get("reason", "")).strip(),
                    }
                return None

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
            if isinstance(active_cycle_run.get("tolbert_runtime_summary"), Mapping):
                runs[-1]["tolbert_runtime_summary"] = _normalize_tolbert_runtime_summary(
                    active_cycle_run.get("tolbert_runtime_summary")
                )
            if isinstance(active_cycle_run.get("verification_outcome_summary"), Mapping):
                runs[-1]["verification_outcome_summary"] = dict(active_cycle_run.get("verification_outcome_summary"))
            runs[-1]["partial_productive"] = bool(partial_progress.get("productive_partial", False))
            runs[-1]["partial_candidate_generated"] = bool(partial_progress.get("candidate_generated", False))
            active_cycle_run = {}
            reconciled_incomplete_cycles = _reconcile_incomplete_cycles(
                config=config,
                planner=planner,
                progress_label=campaign_label or campaign_match_id,
            )
            campaign_records = _campaign_records_with_report_fallback(
                config=config,
                planner=planner,
                cycles_path=config.improvement_cycles_path,
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
            new_runtime_managed_decisions = _runtime_managed_cycle_decisions(
                _production_decisions(new_records)
            )
            recovered_runtime_managed_decisions = (
                _recovered_runtime_managed_decisions_from_reports(config, cycle_ids_for_run)
                if cycle_ids_for_run
                else []
            )
            if recovered_runtime_managed_decisions:
                new_runtime_managed_decisions = _runtime_managed_cycle_decisions(
                    _merge_decision_records(
                        new_runtime_managed_decisions,
                        recovered_runtime_managed_decisions,
                    )
                )
            new_decision_summary = _count_decisions(new_runtime_managed_decisions)
            runs[-1]["runtime_managed_decisions"] = new_decision_summary["total_decisions"]
            runs[-1]["runtime_managed_retained_cycles"] = new_decision_summary["retained_cycles"]
            runs[-1]["runtime_managed_rejected_cycles"] = new_decision_summary["rejected_cycles"]
            runs[-1]["decision_records_considered"] = new_decision_summary["total_decisions"]
            runs[-1]["non_runtime_managed_decisions"] = 0
            runs[-1]["recovered_runtime_managed_decisions"] = len(
                _runtime_managed_cycle_decisions(recovered_runtime_managed_decisions)
            )
            runs[-1]["campaign_cycle_ids"] = list(cycle_ids_for_run)
            runs[-1]["productive"] = (
                new_decision_summary["total_decisions"] > 0
                or bool(runs[-1].get("partial_productive", False))
            )
            runs[-1]["retained_gain"] = new_decision_summary["retained_cycles"] > 0
            runs[-1]["execution_source_summary"] = _execution_source_summary_for_task_ids(
                config,
                task_ids=_run_execution_source_task_ids(runs[-1]),
                since_iso=campaign_started_at_iso,
            )
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
                    "strategy_candidate_id",
                    "strategy_candidate_kind",
                    "strategy_origin",
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
                    "tolbert_runtime_summary",
                    *_CONFIRMATION_STRONG_BASELINE_INT_FIELDS,
                    *_CONFIRMATION_STRONG_BASELINE_FLOAT_FIELDS,
                ):
                    if field in best_cycle_audit:
                        runs[-1][field] = best_cycle_audit[field]
                definitive_state = _record_state(best_cycle_audit)
                if definitive_state in {"retain", "reject"} and int(runs[-1].get("runtime_managed_decisions", 0) or 0) > 0:
                    runs[-1]["runtime_managed_retained_cycles"] = 1 if definitive_state == "retain" else 0
                    runs[-1]["runtime_managed_rejected_cycles"] = 1 if definitive_state == "reject" else 0
                if _record_counts_as_retained_gain(best_cycle_audit):
                    runs[-1]["retained_gain"] = True
            runs[-1] = _apply_structured_child_decision(
                run=runs[-1],
                priority_benchmark_families=current_priority_benchmark_families,
                reason=str(runs[-1].get("stderr", "")).strip(),
            )
            reconciled_cycle_summaries_for_run = [
                item
                for item in reconciled_incomplete_cycles
                if str(item.get("cycle_id", "")).strip() in set(cycle_ids_for_run)
            ]
            incomplete_cycle_summaries: list[dict[str, object]] = []
            if hasattr(planner, "incomplete_cycle_summaries"):
                incomplete_cycle_summaries = [
                    item
                    for item in planner.incomplete_cycle_summaries(
                        config.improvement_cycles_path,
                        protocol="autonomous",
                    )
                    if str(item.get("cycle_id", "")).strip() in set(cycle_ids_for_run)
                ]
            durable_cycle_ids_for_run = {
                str(record.get("cycle_id", "")).strip()
                for record in [*new_runtime_managed_decisions, *recovered_runtime_managed_decisions]
                if str(record.get("state", "")).strip() in {"retain", "reject"}
                and not bool(
                    record.get("metrics_summary", {}).get("incomplete_cycle", False)
                    if isinstance(record.get("metrics_summary", {}), Mapping)
                    else False
                )
            }
            if durable_cycle_ids_for_run:
                incomplete_cycle_summaries = [
                    item
                    for item in incomplete_cycle_summaries
                    if str(item.get("cycle_id", "")).strip() not in durable_cycle_ids_for_run
                ]
            if reconciled_cycle_summaries_for_run:
                seen_reconciled_cycle_ids = {
                    str(item.get("cycle_id", "")).strip() for item in reconciled_cycle_summaries_for_run
                }
                incomplete_cycle_summaries = [
                    *reconciled_cycle_summaries_for_run,
                    *[
                        item
                        for item in incomplete_cycle_summaries
                        if str(item.get("cycle_id", "")).strip() not in seen_reconciled_cycle_ids
                    ],
                ]
            incomplete_cycle_count = len(incomplete_cycle_summaries)
            runs[-1]["incomplete_cycle_count"] = incomplete_cycle_count
            if incomplete_cycle_summaries:
                incomplete_summary = incomplete_cycle_summaries[0]
                for field in (
                    "cycle_id",
                    "subsystem",
                    "selected_variant_id",
                    "strategy_candidate_id",
                    "strategy_candidate_kind",
                    "strategy_origin",
                    "artifact_kind",
                    "artifact_path",
                    "active_artifact_path",
                    "candidate_artifact_path",
                ):
                    if not str(runs[-1].get(field, "")).strip() and str(incomplete_summary.get(field, "")).strip():
                        runs[-1][field] = str(incomplete_summary.get(field, "")).strip()
            if (
                str(runs[-1].get("final_state", "")).strip() not in {"retain", "reject"}
                and incomplete_cycle_count > 0
            ):
                active_artifact_path = (
                    str(runs[-1].get("active_artifact_path", "")).strip()
                    or str(runs[-1].get("artifact_path", "")).strip()
                )
                if not active_artifact_path:
                    for cycle_id in cycle_ids_for_run:
                        generate_record = generate_index.get(cycle_id, {})
                        active_artifact_path = (
                            str(generate_record.get("active_artifact_path", "")).strip()
                            or str(generate_record.get("artifact_path", "")).strip()
                        )
                        if active_artifact_path:
                            break
                if _is_runtime_managed_artifact_path(active_artifact_path):
                    runs[-1]["active_artifact_path"] = active_artifact_path
                    if not str(runs[-1].get("artifact_path", "")).strip():
                        runs[-1]["artifact_path"] = active_artifact_path
                    runs[-1]["final_state"] = "reject"
                    runs[-1]["runtime_managed_decisions"] = max(
                        1,
                        int(runs[-1].get("runtime_managed_decisions", 0) or 0),
                    )
                    runs[-1]["runtime_managed_retained_cycles"] = 0
                    runs[-1]["runtime_managed_rejected_cycles"] = max(
                        1,
                        int(runs[-1].get("runtime_managed_rejected_cycles", 0) or 0),
                    )
                    runs[-1]["decision_records_considered"] = max(
                        1,
                        int(runs[-1].get("decision_records_considered", 0) or 0),
                    )
                    if not str(runs[-1].get("final_reason", "")).strip():
                        runs[-1]["final_reason"] = (
                            "incomplete autonomous cycle was reconciled into a fail-closed runtime-managed rejection"
                        )
                else:
                    runs[-1]["final_state"] = "incomplete"
                    if not str(runs[-1].get("final_reason", "")).strip():
                        runs[-1]["final_reason"] = (
                            "incomplete autonomous cycle did not reach a durable retention decision"
                        )
            runs[-1]["decision_conversion_state"] = _run_decision_conversion_state(
                final_state=str(runs[-1].get("final_state", "")).strip(),
                runtime_managed_decisions=int(runs[-1].get("runtime_managed_decisions", 0) or 0),
                non_runtime_managed_decisions=int(runs[-1].get("non_runtime_managed_decisions", 0) or 0),
                partial_productive=bool(runs[-1].get("partial_productive", False)),
                incomplete_cycle_count=incomplete_cycle_count,
            )
            runs[-1]["decision_state"] = _decision_state_from_run(runs[-1])
            seen_campaign_cycle_ids.update(new_cycle_ids)
            status_snapshot_cache = _campaign_status_snapshot(
                config=config,
                planner=planner,
                cycles_path=config.improvement_cycles_path,
                campaign_match_id=campaign_match_id,
                cycle_log_start_index=cycle_log_start_index,
                priority_benchmark_families=current_priority_benchmark_families,
                priority_family_weights=current_priority_family_weights,
            )
            last_snapshot_refresh_at = time.monotonic()
            status_path = _write_campaign_status(
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
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
            )
            status_snapshot_cache = _load_json_payload(status_path)
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
                if str(runs[-1].get("decision_conversion_state", "")).strip() == "partial_productive_without_decision":
                    print(
                        "[repeated] search_adapt reason=partial_productive_without_decision "
                        f"next_campaign_width={current_campaign_width} "
                        f"next_variant_width={current_variant_width} "
                        f"next_task_limit={current_task_limit} "
                        f"adaptive_search={str(current_adaptive_search).lower()}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
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

        campaign_records = _campaign_records_with_report_fallback(
            config=config,
            planner=planner,
            cycles_path=config.improvement_cycles_path,
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
        structured_non_runtime_decisions = _structured_non_runtime_decisions(runs)
        decision_records = _merge_decision_records(decision_records, structured_non_runtime_decisions)
        reconciled_runtime_managed_decisions = _reconciled_runtime_managed_decisions_from_runs(runs)
        decision_records = _merge_decision_records(decision_records, reconciled_runtime_managed_decisions)
        recovered_runtime_managed_decisions = _recovered_runtime_managed_decisions_from_reports(
            config,
            sorted(report_cycle_ids),
        )
        decision_records = _merge_decision_records(decision_records, recovered_runtime_managed_decisions)
        recent_decisions = decision_records[-10:]
        recent_decisions = [
            {
                **dict(record),
                "decision_state": _decision_state_from_record(record),
            }
            for record in recent_decisions
        ]
        summary = _yield_summary_for(decision_records, generate_index)
        planner_incomplete_cycles = (
            [
                item
                for item in planner.incomplete_cycle_summaries(config.improvement_cycles_path, protocol="autonomous")
                if str(item.get("cycle_id", "")).strip() in campaign_cycle_ids
            ]
            if hasattr(planner, "incomplete_cycle_summaries")
            else []
        )
        incomplete_cycles = _merge_incomplete_cycle_summaries(
            planner_incomplete_cycles,
            runs,
            decision_records,
        )
        planner_pressure_summary = _planner_pressure_summary(campaign_records)
        production_decisions = _merge_decision_records(
            _merge_decision_records(
                _production_decisions(campaign_records),
                reconciled_runtime_managed_decisions,
            ),
            recovered_runtime_managed_decisions,
        )
        recent_runtime_managed_decisions = [
            {
                **dict(record),
                "decision_state": _decision_state_from_record(record),
            }
            for record in production_decisions[-10:]
        ]
        non_runtime_managed_decisions = _merge_decision_records(
            _non_runtime_managed_decisions(campaign_records),
            structured_non_runtime_decisions,
        )
        phase_gate_summary = _phase_gate_summary_for(production_decisions)
        trust_breadth_summary = _trust_breadth_summary(config)
        open_world_breadth_summary = _open_world_breadth_summary(config)
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
            if _record_is_runtime_managed_decision(record):
                runtime_managed_decisions += 1
        non_runtime_managed_decision_count = max(0, len(decision_records) - runtime_managed_decisions)
        candidate_isolation_summary = _candidate_isolation_summary(decision_records, generate_index)
        partial_progress_summary = _partial_progress_rollup(runs)
        decision_conversion_summary = _summarize_run_decision_conversion(runs)
        execution_source_summary = _execution_source_summary_for_task_ids(
            config,
            task_ids=[
                task_id
                for run in runs
                for task_id in _run_execution_source_task_ids(run)
            ],
            since_iso=campaign_started_at_iso,
        )
        terminal_campaign_projection = _terminal_campaign_projection(runs)

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
            "non_runtime_managed_decisions": non_runtime_managed_decision_count,
            "partial_productive_runs": partial_progress_summary["partial_productive_runs"],
            "partial_candidate_runs": partial_progress_summary["partial_candidate_runs"],
            "execution_source_summary": execution_source_summary,
            "decision_state_summary": _decision_state_summary(runs, decision_records),
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
            "open_world_breadth_summary": open_world_breadth_summary,
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
                "non_runtime_managed_decisions": non_runtime_managed_decision_count,
            },
            "candidate_isolation_summary": candidate_isolation_summary,
            "decision_conversion_summary": decision_conversion_summary,
            "partial_progress_summary": partial_progress_summary,
            "tolbert_runtime_summary": _tolbert_runtime_status_summary(runs),
            "recent_decisions": recent_decisions,
            "recent_runtime_managed_decisions": recent_runtime_managed_decisions,
            "recent_production_decisions": recent_runtime_managed_decisions,
            "recent_non_runtime_decisions": [
                {
                    **dict(record),
                    "decision_state": _decision_state_from_record(record),
                }
                for record in non_runtime_managed_decisions[-10:]
            ],
            **terminal_campaign_projection,
            "runs": runs,
        }
        report = _attach_repeated_codex_packets(
            report_path=report_path,
            status_path=_status_path(config),
            payload=report,
        )
        atomic_write_json(report_path, report, config=config)
        _write_repeated_codex_input_packets(
            config=config,
            report_path=report_path,
            status_path=_status_path(config),
            payload=report,
        )
        status_snapshot_cache = _campaign_status_snapshot(
            config=config,
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
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
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
        run_evidence_summary = _run_status_evidence_summary(runs)
        decision_conversion_summary = _summarize_run_decision_conversion(runs)
        trust_breadth_summary = _trust_breadth_summary(config)
        open_world_breadth_summary = _open_world_breadth_summary(config)
        structured_non_runtime_decisions = _structured_non_runtime_decisions(runs)
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
            "runtime_managed_decisions": int(run_evidence_summary.get("runtime_managed_decisions", 0) or 0),
            "non_runtime_managed_decisions": int(run_evidence_summary.get("non_runtime_managed_decisions", 0) or 0),
            "execution_source_summary": dict(run_evidence_summary.get("execution_source_summary", {})),
            "partial_productive_runs": partial_progress_summary["partial_productive_runs"],
            "partial_candidate_runs": partial_progress_summary["partial_candidate_runs"],
            "decision_state_summary": _decision_state_summary(runs, structured_non_runtime_decisions),
            "inheritance_summary": {
                "decision_count": len(structured_non_runtime_decisions),
                "inherited_decisions": 0,
                "runtime_managed_decisions": int(run_evidence_summary.get("runtime_managed_decisions", 0) or 0),
                "non_runtime_managed_decisions": int(run_evidence_summary.get("non_runtime_managed_decisions", 0) or 0),
            },
            "decision_stream_summary": {
                "runtime_managed": _yield_summary_for([], {}),
                "non_runtime_managed": _yield_summary_for(structured_non_runtime_decisions, {}),
            },
            "decision_conversion_summary": decision_conversion_summary,
            "trust_breadth_summary": trust_breadth_summary,
            "open_world_breadth_summary": open_world_breadth_summary,
            "recent_non_runtime_decisions": [
                {
                    **dict(record),
                    "decision_state": _decision_state_from_record(record),
                }
                for record in structured_non_runtime_decisions[-10:]
            ],
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
        interrupted_report = _attach_repeated_codex_packets(
            report_path=report_path,
            status_path=_status_path(config),
            payload=interrupted_report,
        )
        atomic_write_json(report_path, interrupted_report, config=config)
        _write_repeated_codex_input_packets(
            config=config,
            report_path=report_path,
            status_path=_status_path(config),
            payload=interrupted_report,
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
            state="interrupted",
            reason=(
                f"received signal {signal.Signals(received_signal['value']).name}"
                if int(received_signal["value"]) > 0
                else "operator interrupted repeated improvement campaign"
            ),
            active_cycle_run=active_cycle_run,
            snapshot=status_snapshot_cache,
            max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
            max_runtime_seconds=float(args.max_child_runtime_seconds),
        )
        print(report_path)
        raise SystemExit(130)
    finally:
        restore_signal_handlers()


if __name__ == "__main__":
    main()
