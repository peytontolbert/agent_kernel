from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import KernelConfig
from ...extensions.capabilities import load_capability_modules
from ...extensions.improvement.capability_improvement import capability_surface_summary as capability_surface_summary_impl
from ...extensions.delegation_policy import delegation_policy_snapshot
from ...extensions.operator_policy import operator_policy_snapshot


def failure_counts(planner: Any) -> dict[str, int]:
    if planner.memory is None:
        return {}
    counts: dict[str, int] = {}
    for document in planner.memory.list_documents():
        summary = document.get("summary", {})
        for failure_type in summary.get("failure_types", []):
            label = str(failure_type)
            counts[label] = counts.get(label, 0) + 1
        for failure_type in summary.get("transition_failures", []):
            label = str(failure_type)
            counts[label] = counts.get(label, 0) + 1
    return counts


def transition_failure_counts(planner: Any) -> dict[str, int]:
    if planner.memory is None:
        return {}
    counts: dict[str, int] = {}
    for document in planner.memory.list_documents():
        summary = document.get("summary", {})
        for failure_type in summary.get("transition_failures", []):
            label = str(failure_type).strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def transition_summary(planner: Any) -> dict[str, object]:
    if planner.memory is None:
        return {}
    final_completion_ratios: list[float] = []
    net_progress_deltas: list[float] = []
    state_regression_steps = 0
    state_progress_gain_steps = 0
    for document in planner.memory.list_documents():
        summary = document.get("summary", {})
        try:
            final_completion_ratios.append(float(summary.get("final_completion_ratio", 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            net_progress_deltas.append(float(summary.get("net_state_progress_delta", 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            state_regression_steps += int(summary.get("state_regression_steps", 0))
        except (TypeError, ValueError):
            pass
        try:
            state_progress_gain_steps += int(summary.get("state_progress_gain_steps", 0))
        except (TypeError, ValueError):
            pass
    average_completion = (
        round(sum(final_completion_ratios) / len(final_completion_ratios), 4)
        if final_completion_ratios
        else 0.0
    )
    average_progress_delta = (
        round(sum(net_progress_deltas) / len(net_progress_deltas), 4)
        if net_progress_deltas
        else 0.0
    )
    return {
        "average_final_completion_ratio": average_completion,
        "average_net_state_progress_delta": average_progress_delta,
        "state_regression_steps": state_regression_steps,
        "state_progress_gain_steps": state_progress_gain_steps,
    }


def environment_violation_summary(planner: Any) -> dict[str, object]:
    if planner.memory is None:
        return {
            "violation_counts": {},
            "alignment_failure_counts": {},
            "observed_environment_modes": {},
            "violation_total": 0,
            "alignment_failure_total": 0,
        }
    violation_counts: dict[str, int] = {}
    alignment_failure_counts: dict[str, int] = {}
    observed_environment_modes: dict[str, dict[str, int]] = {
        "network_access_mode": {},
        "git_write_mode": {},
        "workspace_write_scope": {},
    }
    for document in planner.memory.list_documents():
        summary = document.get("summary", {})
        for label, value in dict(summary.get("environment_violation_counts", {})).items():
            key = str(label).strip()
            if not key:
                continue
            try:
                violation_counts[key] = violation_counts.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue
        for label in summary.get("environment_alignment_failures", []):
            key = str(label).strip()
            if key:
                alignment_failure_counts[key] = alignment_failure_counts.get(key, 0) + 1
        snapshot = summary.get("environment_snapshot", {})
        if isinstance(snapshot, dict):
            for field in observed_environment_modes:
                value = str(snapshot.get(field, "")).strip().lower()
                if value:
                    observed_environment_modes[field][value] = observed_environment_modes[field].get(value, 0) + 1
    return {
        "violation_counts": violation_counts,
        "alignment_failure_counts": alignment_failure_counts,
        "observed_environment_modes": observed_environment_modes,
        "violation_total": sum(violation_counts.values()),
        "alignment_failure_total": sum(alignment_failure_counts.values()),
    }


def capability_surface_summary(planner: Any) -> dict[str, int]:
    if planner.capability_modules_path is None:
        return {"module_count": 0, "enabled_module_count": 0, "external_capability_count": 0, "improvement_surface_count": 0}
    payload = {"modules": load_capability_modules(planner.capability_modules_path)}
    return capability_surface_summary_impl(payload)


def trust_ledger_payload(planner: Any) -> dict[str, object]:
    path = planner.trust_ledger_path
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def delegation_policy_summary(planner: Any) -> dict[str, object]:
    snapshot = delegation_policy_snapshot(planner.runtime_config or KernelConfig())
    return {key: int(value) for key, value in snapshot.items()}


def operator_policy_summary(planner: Any) -> dict[str, object]:
    snapshot = operator_policy_snapshot(planner.runtime_config or KernelConfig())
    return {
        "unattended_allowed_benchmark_families": list(snapshot.get("unattended_allowed_benchmark_families", [])),
        "unattended_allow_git_commands": bool(snapshot.get("unattended_allow_git_commands", False)),
        "unattended_allow_http_requests": bool(snapshot.get("unattended_allow_http_requests", False)),
        "unattended_http_allowed_hosts": list(snapshot.get("unattended_http_allowed_hosts", [])),
        "unattended_http_timeout_seconds": int(snapshot.get("unattended_http_timeout_seconds", 1)),
        "unattended_http_max_body_bytes": int(snapshot.get("unattended_http_max_body_bytes", 1)),
        "unattended_allow_generated_path_mutations": bool(
            snapshot.get("unattended_allow_generated_path_mutations", False)
        ),
        "unattended_generated_path_prefixes": list(snapshot.get("unattended_generated_path_prefixes", [])),
    }


def decision_records(planner: Any, output_path: Path | None = None) -> list[dict[str, object]]:
    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return []
    return [
        record
        for record in planner.load_cycle_records(resolved)
        if str(record.get("state", "")) in {"retain", "reject"}
    ]


def load_retained_universe_payload_from_record(planner: Any, record: dict[str, object]) -> dict[str, object]:
    from ... import improvement as core

    for raw_path in (
        str(record.get("active_artifact_path", "")).strip(),
        str(record.get("artifact_path", "")).strip(),
        str(record.get("candidate_artifact_path", "")).strip(),
    ):
        if not raw_path:
            continue
        payload = core._load_json_payload(Path(raw_path))
        artifact_kind = str(payload.get("artifact_kind", "")).strip()
        if artifact_kind not in {"universe_contract", "universe_constitution", "operating_envelope"}:
            continue
        retained = core.retained_artifact_payload(payload, artifact_kind=artifact_kind)
        if isinstance(retained, dict):
            return retained
    return {}


def resolve_cycles_path(planner: Any, output_path: Path | None = None) -> Path | None:
    if output_path is not None:
        return output_path
    return planner.cycles_path


def improvement_planner_controls(planner: Any) -> dict[str, object]:
    from ... import improvement as core

    if not planner.use_prompt_proposals:
        return {}
    path = planner.prompt_proposals_path
    if path is None or not path.exists():
        return {}
    payload = core._load_json_payload(path)
    return core.retained_improvement_planner_controls(payload)


def record_has_regression_signal(planner_cls: Any, record: dict[str, object]) -> bool:
    if planner_cls._record_has_phase_gate_failure(record):
        return True
    metrics = record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return False
    regression_count_fields = (
        "regressed_task_count",
        "confirmation_regressed_task_count",
        "regressed_trace_task_count",
        "confirmation_regressed_trace_task_count",
        "regressed_trajectory_task_count",
        "confirmation_regressed_trajectory_task_count",
        "regressed_family_count",
        "generated_regressed_family_count",
        "confirmation_regressed_family_conservative_count",
        "prior_retained_regressed_family_count",
        "prior_retained_generated_regressed_family_count",
    )
    for field in regression_count_fields:
        if int(metrics.get(field, 0) or 0) > 0:
            return True
    baseline_pass_rate = metrics.get("baseline_pass_rate")
    candidate_pass_rate = metrics.get("candidate_pass_rate")
    try:
        if baseline_pass_rate is not None and candidate_pass_rate is not None:
            if float(candidate_pass_rate) < float(baseline_pass_rate):
                return True
    except (TypeError, ValueError):
        return False
    return False


def active_artifact_payload_from_generation_context(payload: dict[str, object] | None) -> dict[str, object] | None:
    from ... import improvement as core

    def _retained_or_legacy_runtime_payload(candidate: object) -> dict[str, object] | None:
        if not isinstance(candidate, dict):
            return None
        artifact_kind = str(candidate.get("artifact_kind", "")).strip()
        has_contract_metadata = (
            "spec_version" in candidate
            or "lifecycle_state" in candidate
            or "retention_gate" in candidate
            or "retention_decision" in candidate
        )
        if artifact_kind and has_contract_metadata:
            retained = core.retained_artifact_payload(candidate, artifact_kind=artifact_kind)
            if isinstance(retained, dict):
                return retained
            retention_decision = candidate.get("retention_decision", {})
            if isinstance(retention_decision, dict) and str(retention_decision.get("state", "")).strip() == "reject":
                return None
            if str(candidate.get("lifecycle_state", "")).strip() == "rejected":
                return None
        return candidate

    if not isinstance(payload, dict):
        return None
    context = payload.get("generation_context", {})
    if not isinstance(context, dict):
        return None
    inline_payload = context.get("active_artifact_payload")
    if isinstance(inline_payload, dict):
        return _retained_or_legacy_runtime_payload(inline_payload)
    active_artifact_value = str(context.get("active_artifact_path", "")).strip()
    if not active_artifact_value:
        return None
    active_artifact_path = Path(active_artifact_value)
    if not active_artifact_path.exists():
        return None
    try:
        loaded = core._load_json_payload(active_artifact_path)
    except (OSError, json.JSONDecodeError):
        return None
    return _retained_or_legacy_runtime_payload(loaded)
