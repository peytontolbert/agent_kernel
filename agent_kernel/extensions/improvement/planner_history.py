from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...improvement_engine import ImprovementYieldSummary
from .artifact_protocol_support import average_metric_delta


def incomplete_cycle_summaries(
    planner: Any,
    output_path: Path | None = None,
    *,
    protocol: str | None = None,
    before_cycle_id: str | None = None,
) -> list[dict[str, object]]:
    from ... import improvement as core

    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return []
    records = planner.load_cycle_records(resolved)
    if not records:
        return []
    if before_cycle_id is not None:
        cutoff_index = next(
            (
                index
                for index, record in enumerate(records)
                if str(record.get("cycle_id", "")) == before_cycle_id
            ),
            None,
        )
        if cutoff_index is not None:
            records = records[:cutoff_index]
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        cycle_id = str(record.get("cycle_id", "")).strip()
        if not cycle_id:
            continue
        grouped.setdefault(cycle_id, []).append(record)
    summaries: list[dict[str, object]] = []
    for cycle_id, cycle_records in grouped.items():
        states = {str(record.get("state", "")).strip() for record in cycle_records}
        if states & {"retain", "reject"}:
            continue
        if not states & {"select", "generate", "evaluate"}:
            continue
        observe_record = next(
            (record for record in cycle_records if str(record.get("state", "")) == "observe"),
            None,
        )
        select_record = next(
            (record for record in cycle_records if str(record.get("state", "")) == "select"),
            None,
        )
        generate_record = next(
            (record for record in reversed(cycle_records) if str(record.get("state", "")) == "generate"),
            None,
        )
        latest_record = cycle_records[-1]
        metrics_sources = [
            record.get("metrics_summary", {})
            for record in (observe_record, select_record, generate_record, latest_record)
            if isinstance(record, dict)
        ]
        protocol_value = ""
        protocol_match_id = ""
        for metrics_summary in metrics_sources:
            if not isinstance(metrics_summary, dict):
                continue
            token = str(metrics_summary.get("protocol", "")).strip()
            if token:
                protocol_value = token
            match_token = str(metrics_summary.get("protocol_match_id", "")).strip()
            if match_token and not protocol_match_id:
                protocol_match_id = match_token
            if protocol_value and protocol_match_id:
                break
        if protocol is not None and protocol_value != protocol:
            continue
        candidate_artifact_path = ""
        active_artifact_path = ""
        artifact_kind = str(latest_record.get("artifact_kind", "")).strip()
        artifact_path = str(latest_record.get("artifact_path", "")).strip()
        for record in reversed(cycle_records):
            candidate_value = str(record.get("candidate_artifact_path", "")).strip()
            if candidate_value and not candidate_artifact_path:
                candidate_artifact_path = candidate_value
            active_value = str(record.get("active_artifact_path", "")).strip()
            if active_value and not active_artifact_path:
                active_artifact_path = active_value
            artifact_value = str(record.get("artifact_path", "")).strip()
            if artifact_value and not artifact_path:
                artifact_path = artifact_value
            kind_value = str(record.get("artifact_kind", "")).strip()
            if kind_value and not artifact_kind:
                artifact_kind = kind_value
        summaries.append(
            {
                "cycle_id": cycle_id,
                "subsystem": str(latest_record.get("subsystem", "")).strip(),
                "protocol": protocol_value,
                "protocol_match_id": protocol_match_id,
                "last_state": str(latest_record.get("state", "")).strip(),
                "last_action": str(latest_record.get("action", "")).strip(),
                "reason": str(latest_record.get("reason", "")).strip(),
                "artifact_kind": artifact_kind,
                "artifact_path": artifact_path,
                "active_artifact_path": active_artifact_path,
                "candidate_artifact_path": candidate_artifact_path,
                "selected_variant_id": core._record_selected_variant_id(latest_record)
                or core._record_selected_variant_id(select_record or {})
                or core._record_selected_variant_id(generate_record or {}),
                "strategy_candidate_id": core._record_strategy_candidate_id(latest_record)
                or core._record_strategy_candidate_id(select_record or {})
                or core._record_strategy_candidate_id(generate_record or {}),
                "strategy_candidate_kind": core._record_strategy_candidate_kind(latest_record)
                or core._record_strategy_candidate_kind(select_record or {})
                or core._record_strategy_candidate_kind(generate_record or {}),
                "strategy_origin": core._record_strategy_origin(latest_record)
                or core._record_strategy_origin(select_record or {})
                or core._record_strategy_origin(generate_record or {}),
                "prior_retained_cycle_id": core._record_prior_retained_cycle_id(latest_record)
                or core._record_prior_retained_cycle_id(generate_record or {}),
                "selected_cycles": len(
                    {
                        str(record.get("cycle_id", ""))
                        for record in cycle_records
                        if str(record.get("state", "")) == "select"
                    }
                ),
                "record_count": len(cycle_records),
            }
        )
    return summaries


def load_cycle_records(planner: Any, output_path: Path) -> list[dict[str, object]]:
    if planner.runtime_config is not None and planner.runtime_config.uses_sqlite_storage():
        records = planner.runtime_config.sqlite_store().load_cycle_records(output_path=output_path)
        if records:
            return records
    if not output_path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def cycle_history(
    planner: Any,
    output_path: Path,
    *,
    cycle_id: str | None = None,
    subsystem: str | None = None,
    state: str | None = None,
) -> list[dict[str, object]]:
    records = planner.load_cycle_records(output_path)
    if cycle_id is not None:
        records = [record for record in records if str(record.get("cycle_id", "")) == cycle_id]
    if subsystem is not None:
        records = [record for record in records if str(record.get("subsystem", "")) == subsystem]
    if state is not None:
        records = [record for record in records if str(record.get("state", "")) == state]
    return records


def cycle_audit_summary(planner: Any, output_path: Path, *, cycle_id: str) -> dict[str, object] | None:
    from ... import improvement as core

    records = cycle_history(planner, output_path, cycle_id=cycle_id)
    if not records:
        return None
    latest = records[-1]
    selected_variant_id = ""
    prior_retained_cycle_id = ""
    candidate_artifact_path = ""
    active_artifact_path = ""
    decision_record: dict[str, object] | None = None
    for record in records:
        if not selected_variant_id:
            selected_variant_id = core._record_selected_variant_id(record)
        if not prior_retained_cycle_id:
            prior_retained_cycle_id = core._record_prior_retained_cycle_id(record)
        if not candidate_artifact_path:
            candidate_artifact_path = str(record.get("candidate_artifact_path", "")).strip()
        if not active_artifact_path:
            active_artifact_path = str(record.get("active_artifact_path", "")).strip()
        if str(record.get("state", "")).strip() in {"retain", "reject"}:
            decision_record = record
    decision_record = decision_record or latest
    return {
        "cycle_id": cycle_id,
        "subsystem": str(latest.get("subsystem", "")).strip(),
        "record_count": len(records),
        "states": [str(record.get("state", "")).strip() for record in records],
        "selected_variant_id": selected_variant_id,
        "strategy_candidate_id": core._record_strategy_candidate_id(decision_record),
        "strategy_candidate_kind": core._record_strategy_candidate_kind(decision_record),
        "strategy_origin": core._record_strategy_origin(decision_record),
        "prior_retained_cycle_id": prior_retained_cycle_id,
        "candidate_artifact_path": candidate_artifact_path,
        "active_artifact_path": active_artifact_path,
        "final_state": str(decision_record.get("state", "")).strip(),
        "final_reason": str(decision_record.get("reason", "")).strip(),
        "preview_reason_code": str(core._record_metrics_summary(decision_record).get("preview_reason_code", "")).strip(),
        "decision_reason_code": str(core._record_metrics_summary(decision_record).get("decision_reason_code", "")).strip(),
        "baseline_pass_rate": core._record_float_value(decision_record, "baseline_pass_rate"),
        "candidate_pass_rate": core._record_float_value(decision_record, "candidate_pass_rate"),
        "baseline_average_steps": core._record_float_value(decision_record, "baseline_average_steps"),
        "candidate_average_steps": core._record_float_value(decision_record, "candidate_average_steps"),
        "phase_gate_passed": core._record_phase_gate_passed(decision_record),
        "artifact_kind": str(decision_record.get("artifact_kind", "")).strip(),
        "artifact_path": str(decision_record.get("artifact_path", "")).strip(),
        "artifact_lifecycle_state": str(decision_record.get("artifact_lifecycle_state", "")).strip(),
        "artifact_sha256": str(decision_record.get("artifact_sha256", "")).strip(),
        "previous_artifact_sha256": str(decision_record.get("previous_artifact_sha256", "")).strip(),
        "rollback_artifact_path": str(decision_record.get("rollback_artifact_path", "")).strip(),
        "artifact_snapshot_path": str(decision_record.get("artifact_snapshot_path", "")).strip(),
    }


def retained_gain_summary(
    planner: Any,
    output_path: Path,
    *,
    subsystem: str | None = None,
) -> ImprovementYieldSummary:
    from ... import improvement as core

    decisions = [
        record
        for record in planner.load_cycle_records(output_path)
        if str(record.get("state", "")) in {"retain", "reject"}
    ]
    if subsystem is not None:
        decisions = [
            record
            for record in decisions
            if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
        ]

    retained = [record for record in decisions if str(record.get("state", "")) == "retain"]
    rejected = [record for record in decisions if str(record.get("state", "")) == "reject"]

    retained_by_subsystem: dict[str, int] = {}
    rejected_by_subsystem: dict[str, int] = {}
    for record in retained:
        key = str(record.get("subsystem", "unknown"))
        retained_by_subsystem[key] = retained_by_subsystem.get(key, 0) + 1
    for record in rejected:
        key = str(record.get("subsystem", "unknown"))
        rejected_by_subsystem[key] = rejected_by_subsystem.get(key, 0) + 1

    return ImprovementYieldSummary(
        retained_cycles=len(retained),
        rejected_cycles=len(rejected),
        total_decisions=len(decisions),
        retained_by_subsystem=retained_by_subsystem,
        rejected_by_subsystem=rejected_by_subsystem,
        average_retained_pass_rate_delta=average_metric_delta(
            retained,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        average_retained_step_delta=average_metric_delta(
            retained,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
        average_rejected_pass_rate_delta=average_metric_delta(
            rejected,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        average_rejected_step_delta=average_metric_delta(
            rejected,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
    )


def _empty_universe_cycle_feedback_summary() -> dict[str, object]:
    return {
        "retained_cycle_count": 0,
        "selected_variant_counts": {},
        "selected_variant_weights": {},
        "successful_environment_assumptions": {},
        "successful_environment_assumption_weights": {},
        "successful_action_risk_control_floor": {},
        "successful_action_risk_control_weighted_mean": {},
        "average_retained_pass_rate_delta": 0.0,
        "average_retained_step_delta": 0.0,
        "broad_support_cycle_count": 0,
        "narrow_support_cycle_count": 0,
        "phase_gate_warning_count": 0,
        "constitution_retained_cycle_count": 0,
        "operating_envelope_retained_cycle_count": 0,
        "dominant_variant_share": 0.0,
    }


def universe_cycle_feedback_summary(
    planner: Any,
    *,
    recent_cycle_window: int = 8,
    output_path: Path | None = None,
) -> dict[str, object]:
    from ... import improvement as core

    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return _empty_universe_cycle_feedback_summary()
    decision_records = [
        record
        for record in planner._decision_records(resolved)
        if planner._subsystems_match(str(record.get("subsystem", "")), "universe")
        and str(record.get("state", "")).strip() == "retain"
    ]
    if not decision_records:
        return _empty_universe_cycle_feedback_summary()
    recent_records = decision_records[-max(1, recent_cycle_window) :]
    selected_variant_counts: dict[str, int] = {}
    selected_variant_weights: dict[str, float] = {}
    successful_environment_assumptions: dict[str, dict[str, int]] = {
        "network_access_mode": {},
        "git_write_mode": {},
        "workspace_write_scope": {},
    }
    successful_environment_assumption_weights: dict[str, dict[str, float]] = {
        "network_access_mode": {},
        "git_write_mode": {},
        "workspace_write_scope": {},
    }
    successful_action_risk_control_floor: dict[str, int] = {}
    successful_action_risk_control_weighted_sum: dict[str, float] = {}
    successful_action_risk_control_weighted_weight: dict[str, float] = {}
    pass_rate_deltas: list[float] = []
    step_deltas: list[float] = []
    broad_support_cycle_count = 0
    narrow_support_cycle_count = 0
    phase_gate_warning_count = 0
    constitution_retained_cycle_count = 0
    operating_envelope_retained_cycle_count = 0
    for record in recent_records:
        variant_id = core._record_selected_variant_id(record)
        baseline_pass_rate = core._record_float_value(record, "baseline_pass_rate")
        candidate_pass_rate = core._record_float_value(record, "candidate_pass_rate")
        if baseline_pass_rate is not None and candidate_pass_rate is not None:
            pass_rate_deltas.append(candidate_pass_rate - baseline_pass_rate)
        baseline_average_steps = core._record_float_value(record, "baseline_average_steps")
        candidate_average_steps = core._record_float_value(record, "candidate_average_steps")
        if baseline_average_steps is not None and candidate_average_steps is not None:
            step_deltas.append(candidate_average_steps - baseline_average_steps)
        non_regressed_family_support = core._record_non_regressed_family_support(record)
        if non_regressed_family_support <= 1:
            narrow_support_cycle_count += 1
        support_confidence = 0.35 if non_regressed_family_support <= 1 else (0.75 if non_regressed_family_support == 2 else 1.0)
        pass_rate_delta = 0.0
        if baseline_pass_rate is not None and candidate_pass_rate is not None:
            pass_rate_delta = candidate_pass_rate - baseline_pass_rate
        step_gain = 0.0
        if baseline_average_steps is not None and candidate_average_steps is not None:
            step_gain = baseline_average_steps - candidate_average_steps
        phase_gate_passed = core._record_phase_gate_passed(record)
        phase_gate_failures = core._record_phase_gate_failures(record)
        phase_gate_confidence = 1.0
        if phase_gate_passed is False or phase_gate_failures:
            phase_gate_warning_count += 1
            phase_gate_confidence = 0.5
        regressed_family_count = 0
        metrics_summary = core._record_metrics_summary(record)
        if "regressed_family_count" in metrics_summary:
            try:
                regressed_family_count = max(0, int(metrics_summary.get("regressed_family_count", 0) or 0))
            except (TypeError, ValueError):
                regressed_family_count = 0
        regression_confidence = max(0.55, 1.0 - min(0.45, regressed_family_count * 0.15))
        outcome_weight = max(
            0.4,
            1.0 + max(0.0, min(0.05, pass_rate_delta)) * 8.0 + max(0.0, min(1.0, step_gain)) * 2.0,
        )
        weighted_support = round(outcome_weight * support_confidence * phase_gate_confidence * regression_confidence, 4)
        if non_regressed_family_support >= 2:
            broad_support_cycle_count += 1
        if variant_id:
            selected_variant_counts[variant_id] = selected_variant_counts.get(variant_id, 0) + 1
            selected_variant_weights[variant_id] = round(
                selected_variant_weights.get(variant_id, 0.0) + weighted_support,
                4,
            )
        payload = planner._load_retained_universe_payload_from_record(record)
        artifact_kind = str(payload.get("artifact_kind", "")).strip()
        if artifact_kind == "universe_constitution":
            constitution_retained_cycle_count += 1
        elif artifact_kind == "operating_envelope":
            operating_envelope_retained_cycle_count += 1
        environment_assumptions = planner._plugin_layer.retained_universe_environment_assumptions(payload)
        for field, counts in successful_environment_assumptions.items():
            value = str(environment_assumptions.get(field, "")).strip().lower()
            if value:
                counts[value] = counts.get(value, 0) + 1
                successful_environment_assumption_weights[field][value] = round(
                    successful_environment_assumption_weights[field].get(value, 0.0) + weighted_support,
                    4,
                )
        action_risk_controls = planner._plugin_layer.retained_universe_action_risk_controls(payload)
        for key, value in action_risk_controls.items():
            successful_action_risk_control_floor[key] = max(
                int(value),
                int(successful_action_risk_control_floor.get(key, 0)),
            )
            successful_action_risk_control_weighted_sum[key] = (
                successful_action_risk_control_weighted_sum.get(key, 0.0) + (float(value) * weighted_support)
            )
            successful_action_risk_control_weighted_weight[key] = (
                successful_action_risk_control_weighted_weight.get(key, 0.0) + weighted_support
            )
    dominant_variant_share = 0.0
    if recent_records and selected_variant_counts:
        dominant_variant_share = round(max(selected_variant_counts.values()) / len(recent_records), 4)
    return {
        "retained_cycle_count": len(recent_records),
        "selected_variant_counts": selected_variant_counts,
        "selected_variant_weights": selected_variant_weights,
        "successful_environment_assumptions": {
            field: core._dominant_weight_label(successful_environment_assumption_weights.get(field, counts))
            for field, counts in successful_environment_assumptions.items()
            if core._dominant_weight_label(successful_environment_assumption_weights.get(field, counts))
        },
        "successful_environment_assumption_weights": successful_environment_assumption_weights,
        "successful_action_risk_control_floor": successful_action_risk_control_floor,
        "successful_action_risk_control_weighted_mean": {
            key: round(
                successful_action_risk_control_weighted_sum[key] / successful_action_risk_control_weighted_weight[key],
                3,
            )
            for key in successful_action_risk_control_weighted_sum
            if successful_action_risk_control_weighted_weight.get(key, 0.0) > 0.0
        },
        "average_retained_pass_rate_delta": round(sum(pass_rate_deltas) / len(pass_rate_deltas), 4)
        if pass_rate_deltas
        else 0.0,
        "average_retained_step_delta": round(sum(step_deltas) / len(step_deltas), 4) if step_deltas else 0.0,
        "broad_support_cycle_count": broad_support_cycle_count,
        "narrow_support_cycle_count": narrow_support_cycle_count,
        "phase_gate_warning_count": phase_gate_warning_count,
        "constitution_retained_cycle_count": constitution_retained_cycle_count,
        "operating_envelope_retained_cycle_count": operating_envelope_retained_cycle_count,
        "dominant_variant_share": dominant_variant_share,
    }


def trust_ledger_summary(planner: Any) -> dict[str, object]:
    payload = planner.trust_ledger_payload()
    if not payload:
        return {}
    overall = payload.get("overall_summary", {}) if isinstance(payload.get("overall_summary", {}), dict) else {}
    gated = payload.get("gated_summary", {}) if isinstance(payload.get("gated_summary", {}), dict) else {}
    assessment = payload.get("overall_assessment", {}) if isinstance(payload.get("overall_assessment", {}), dict) else {}
    coverage = payload.get("coverage_summary", {}) if isinstance(payload.get("coverage_summary", {}), dict) else {}
    required_family_clean_task_root_counts = (
        coverage.get("required_family_clean_task_root_counts", {})
        if isinstance(coverage.get("required_family_clean_task_root_counts", {}), dict)
        else {}
    )
    clean_success_task_roots = gated.get("clean_success_task_roots", overall.get("clean_success_task_roots", []))
    return {
        "reports_considered": int(payload.get("reports_considered", 0) or 0),
        "overall_status": str(assessment.get("status", "")).strip(),
        "overall_passed": bool(assessment.get("passed", False)),
        "failing_thresholds": [
            str(value).strip()
            for value in assessment.get("failing_thresholds", [])
            if str(value).strip()
        ]
        if isinstance(assessment.get("failing_thresholds", []), list)
        else [],
        "success_rate": float(gated.get("success_rate", overall.get("success_rate", 0.0)) or 0.0),
        "unsafe_ambiguous_rate": float(
            gated.get("unsafe_ambiguous_rate", overall.get("unsafe_ambiguous_rate", 0.0)) or 0.0
        ),
        "hidden_side_effect_risk_rate": float(
            gated.get("hidden_side_effect_risk_rate", overall.get("hidden_side_effect_risk_rate", 0.0)) or 0.0
        ),
        "rollback_performed_rate": float(
            gated.get("rollback_performed_rate", overall.get("rollback_performed_rate", 0.0)) or 0.0
        ),
        "success_hidden_side_effect_risk_rate": float(
            gated.get(
                "success_hidden_side_effect_risk_rate",
                overall.get("success_hidden_side_effect_risk_rate", 0.0),
            )
            or 0.0
        ),
        "false_pass_risk_rate": float(gated.get("false_pass_risk_rate", overall.get("false_pass_risk_rate", 0.0)) or 0.0),
        "unexpected_change_report_rate": float(
            gated.get("unexpected_change_report_rate", overall.get("unexpected_change_report_rate", 0.0)) or 0.0
        ),
        "clean_success_rate": float(gated.get("clean_success_rate", overall.get("clean_success_rate", 0.0)) or 0.0),
        "distinct_benchmark_families": int(overall.get("distinct_benchmark_families", 0) or 0),
        "distinct_clean_success_task_roots": int(
            gated.get("distinct_clean_success_task_roots", overall.get("distinct_clean_success_task_roots", 0)) or 0
        ),
        "clean_success_task_roots": [
            str(value).strip()
            for value in clean_success_task_roots
            if str(value).strip()
        ]
        if isinstance(clean_success_task_roots, list)
        else [],
        "distinct_family_gap": int(coverage.get("distinct_family_gap", 0) or 0),
        "missing_required_families": [
            str(value).strip()
            for value in coverage.get("missing_required_families", [])
            if str(value).strip()
        ]
        if isinstance(coverage.get("missing_required_families", []), list)
        else [],
        "restricted_required_families": [
            str(value).strip()
            for value in coverage.get("restricted_required_families", [])
            if str(value).strip()
        ]
        if isinstance(coverage.get("restricted_required_families", []), list)
        else [],
        "family_breadth_min_distinct_task_roots": int(coverage.get("family_breadth_min_distinct_task_roots", 0) or 0),
        "required_family_clean_task_root_counts": {
            str(key).strip(): int(value or 0)
            for key, value in required_family_clean_task_root_counts.items()
            if str(key).strip()
        },
        "missing_required_family_clean_task_root_breadth": [
            str(value).strip()
            for value in coverage.get("required_families_missing_clean_task_root_breadth", [])
            if str(value).strip()
        ]
        if isinstance(coverage.get("required_families_missing_clean_task_root_breadth", []), list)
        else [],
        "external_report_count": int(coverage.get("external_report_count", 0) or 0),
        "distinct_external_benchmark_families": int(coverage.get("distinct_external_benchmark_families", 0) or 0),
    }


def decision_summary(records: list[dict[str, object]], *, recent_decision_window: int = 3) -> dict[str, object]:
    from ... import improvement as core

    retained = [record for record in records if str(record.get("state", "")) == "retain"]
    rejected = [record for record in records if str(record.get("state", "")) == "reject"]
    total = len(records)
    retention_rate = 0.0 if total == 0 else len(retained) / total
    rejection_rate = 0.0 if total == 0 else len(rejected) / total
    recent_records = records[-max(1, recent_decision_window) :] if records else []
    recent_retained = [record for record in recent_records if str(record.get("state", "")) == "retain"]
    recent_rejected = [record for record in recent_records if str(record.get("state", "")) == "reject"]
    last_decision = records[-1] if records else None
    summary = {
        "total_decisions": total,
        "retained_cycles": len(retained),
        "rejected_cycles": len(rejected),
        "retention_rate": round(retention_rate, 4),
        "rejection_rate": round(rejection_rate, 4),
        "average_retained_pass_rate_delta": average_metric_delta(
            retained,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        "average_retained_step_delta": average_metric_delta(
            retained,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
        "average_rejected_pass_rate_delta": average_metric_delta(
            rejected,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        "average_rejected_step_delta": average_metric_delta(
            rejected,
            baseline_key="baseline_average_steps",
            candidate_key="candidate_average_steps",
        ),
    }
    recent_total = len(recent_records)
    summary["recent_decision_window"] = max(1, recent_decision_window)
    summary["recent_retained_cycles"] = len(recent_retained)
    summary["recent_rejected_cycles"] = len(recent_rejected)
    summary["recent_retention_rate"] = round(0.0 if recent_total == 0 else len(recent_retained) / recent_total, 4)
    summary["recent_rejection_rate"] = round(0.0 if recent_total == 0 else len(recent_rejected) / recent_total, 4)
    summary["last_decision_state"] = "" if last_decision is None else str(last_decision.get("state", ""))
    summary["last_cycle_id"] = "" if last_decision is None else str(last_decision.get("cycle_id", ""))
    summary["net_retained_cycle_advantage"] = len(retained) - len(rejected)
    summary["decision_quality_score"] = decision_quality_score(summary)
    return summary


def recent_subsystem_activity_summary(
    planner: Any,
    *,
    subsystem: str,
    recent_cycle_window: int = 6,
    output_path: Path | None = None,
) -> dict[str, object]:
    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return empty_recent_activity_summary(recent_cycle_window)
    records = planner.load_cycle_records(resolved)
    if not records:
        return empty_recent_activity_summary(recent_cycle_window)
    recent_cycle_ids: list[str] = []
    seen_cycle_ids: set[str] = set()
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if not cycle_id or cycle_id in seen_cycle_ids:
            continue
        seen_cycle_ids.add(cycle_id)
        recent_cycle_ids.append(cycle_id)
        if len(recent_cycle_ids) >= max(1, recent_cycle_window):
            break
    recent_cycle_id_set = set(recent_cycle_ids)
    relevant = [
        record
        for record in records
        if str(record.get("cycle_id", "")) in recent_cycle_id_set
        and planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
    ]
    return _activity_summary_from_records(
        planner,
        relevant=relevant,
        recent_cycle_window=recent_cycle_window,
    )


def recent_campaign_surface_activity_summary(
    planner: Any,
    *,
    subsystem: str,
    recent_cycle_window: int = 6,
    output_path: Path | None = None,
) -> dict[str, object]:
    surface_key = planner._campaign_surface_key(subsystem)
    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return empty_recent_activity_summary(recent_cycle_window)
    records = planner.load_cycle_records(resolved)
    if not records:
        return empty_recent_activity_summary(recent_cycle_window)
    recent_cycle_ids: list[str] = []
    seen_cycle_ids: set[str] = set()
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if not cycle_id or cycle_id in seen_cycle_ids:
            continue
        seen_cycle_ids.add(cycle_id)
        recent_cycle_ids.append(cycle_id)
        if len(recent_cycle_ids) >= max(1, recent_cycle_window):
            break
    recent_cycle_id_set = set(recent_cycle_ids)
    relevant = [
        record
        for record in records
        if str(record.get("cycle_id", "")) in recent_cycle_id_set
        and planner._campaign_surface_key(str(record.get("subsystem", ""))) == surface_key
    ]
    if not relevant:
        return empty_recent_activity_summary(recent_cycle_window)
    return _activity_summary_from_records(
        planner,
        relevant=relevant,
        recent_cycle_window=recent_cycle_window,
    )


def recent_variant_activity_summary(
    planner: Any,
    *,
    subsystem: str,
    variant_id: str,
    recent_cycle_window: int = 6,
    output_path: Path | None = None,
) -> dict[str, object]:
    resolved = planner._resolve_cycles_path(output_path)
    if resolved is None:
        return empty_recent_activity_summary(recent_cycle_window)
    cycle_variants = planner._cycle_variant_index(output_path)
    activity = planner.recent_subsystem_activity_summary(
        subsystem=subsystem,
        recent_cycle_window=recent_cycle_window,
        output_path=output_path,
    )
    if int(activity.get("selected_cycles", 0)) == 0 and int(activity.get("total_decisions", 0)) == 0:
        return activity
    records = planner.load_cycle_records(resolved)
    relevant_cycle_ids = {
        str(record.get("cycle_id", ""))
        for record in records
        if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
        and cycle_variants.get(str(record.get("cycle_id", ""))) == variant_id
    }
    if not relevant_cycle_ids:
        return empty_recent_activity_summary(recent_cycle_window)
    relevant = [
        record
        for record in records
        if str(record.get("cycle_id", "")) in relevant_cycle_ids
        and planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
    ]
    recent_cycle_ids = list(
        dict.fromkeys(
            reversed(
                [str(record.get("cycle_id", "")) for record in relevant if str(record.get("cycle_id", "")).strip()]
            )
        )
    )
    recent_cycle_ids = list(reversed(recent_cycle_ids[: max(1, recent_cycle_window)]))
    recent_set = set(recent_cycle_ids)
    recent_relevant = [record for record in relevant if str(record.get("cycle_id", "")) in recent_set]
    return _activity_summary_from_records(
        planner,
        relevant=recent_relevant,
        recent_cycle_window=recent_cycle_window,
    )


def history_bonus(summary: dict[str, object], *, variant_specific: bool = False) -> float:
    total_decisions = int(summary.get("total_decisions", 0))
    if total_decisions == 0:
        return 0.0
    retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
    retained_step_gain = max(0.0, -float(summary.get("average_retained_step_delta", 0.0))) * 0.25
    rejected_pass_penalty = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
    rejected_step_penalty = max(0.0, float(summary.get("average_rejected_step_delta", 0.0))) * 0.25
    retention_signal = float(summary.get("retention_rate", 0.0)) * (0.03 if variant_specific else 0.02)
    rejection_signal = float(summary.get("rejection_rate", 0.0)) * (0.02 if variant_specific else 0.015)
    recent_retention_signal = float(summary.get("recent_retention_rate", 0.0)) * (0.02 if variant_specific else 0.015)
    recent_rejection_signal = float(summary.get("recent_rejection_rate", 0.0)) * (0.018 if variant_specific else 0.012)
    decision_quality_signal = float(summary.get("decision_quality_score", 0.0)) * (0.4 if variant_specific else 0.3)
    incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0.0)) * (
        0.016 if variant_specific else 0.012
    )
    reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0.0)) * (
        0.018 if variant_specific else 0.014
    )
    last_decision_state = str(summary.get("last_decision_state", "")).strip()
    recency_bias = 0.0
    if last_decision_state == "retain":
        recency_bias = 0.01 if variant_specific else 0.0075
    elif last_decision_state == "reject":
        recency_bias = -0.012 if variant_specific else -0.008
    bonus = (
        retained_gain
        + retained_step_gain
        + retention_signal
        + recent_retention_signal
        + decision_quality_signal
        + recency_bias
        - rejected_pass_penalty
        - rejected_step_penalty
        - rejection_signal
        - recent_rejection_signal
        - incomplete_cycle_penalty
        - reconciled_failure_penalty
    )
    return round(max(-0.1, min(0.1, bonus)), 4)


def recent_history_bonus(summary: dict[str, object], *, variant_specific: bool = False) -> float:
    total_decisions = int(summary.get("total_decisions", 0))
    selected_cycles = int(summary.get("selected_cycles", 0))
    if total_decisions <= 0 and selected_cycles <= 0:
        return 0.0
    retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
    rejection_penalty = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
    quality = float(summary.get("decision_quality_score", 0.0))
    no_yield_penalty = float(summary.get("no_yield_cycles", 0)) * (0.012 if variant_specific else 0.009)
    incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0)) * (
        0.014 if variant_specific else 0.011
    )
    regression_penalty = float(summary.get("recent_regression_cycles", 0)) * (
        0.01 if variant_specific else 0.008
    )
    phase_gate_penalty = float(summary.get("recent_phase_gate_failure_cycles", 0)) * (
        0.008 if variant_specific else 0.006
    )
    reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0)) * (
        0.016 if variant_specific else 0.012
    )
    scale = 0.25 if variant_specific else 0.2
    bonus = (
        ((retained_gain - rejection_penalty + quality) * scale)
        - no_yield_penalty
        - incomplete_cycle_penalty
        - regression_penalty
        - phase_gate_penalty
        - reconciled_failure_penalty
    )
    return round(max(-0.04, min(0.04, bonus)), 4)


def decision_quality_score(summary: dict[str, object]) -> float:
    retained_gain = max(0.0, float(summary.get("average_retained_pass_rate_delta", 0.0)))
    retained_efficiency = max(0.0, -float(summary.get("average_retained_step_delta", 0.0))) * 0.2
    rejected_regression = max(0.0, -float(summary.get("average_rejected_pass_rate_delta", 0.0)))
    rejected_efficiency_regression = max(0.0, float(summary.get("average_rejected_step_delta", 0.0))) * 0.2
    retention_bias = float(summary.get("retention_rate", 0.0)) * 0.02
    rejection_bias = float(summary.get("rejection_rate", 0.0)) * 0.02
    recent_retention_bias = float(summary.get("recent_retention_rate", 0.0)) * 0.015
    recent_rejection_bias = float(summary.get("recent_rejection_rate", 0.0)) * 0.015
    no_yield_penalty = float(summary.get("no_yield_cycles", 0)) * 0.01
    incomplete_cycle_penalty = float(summary.get("recent_incomplete_cycles", 0)) * 0.012
    regression_penalty = float(summary.get("recent_regression_cycles", 0)) * 0.008
    phase_gate_penalty = float(summary.get("recent_phase_gate_failure_cycles", 0)) * 0.006
    reconciled_failure_penalty = float(summary.get("recent_reconciled_failure_cycles", 0)) * 0.014
    quality = (
        retained_gain
        + retained_efficiency
        + retention_bias
        + recent_retention_bias
        - rejected_regression
        - rejected_efficiency_regression
        - rejection_bias
        - recent_rejection_bias
        - no_yield_penalty
        - incomplete_cycle_penalty
        - regression_penalty
        - phase_gate_penalty
        - reconciled_failure_penalty
    )
    return round(max(-0.1, min(0.1, quality)), 4)


def empty_recent_activity_summary(recent_cycle_window: int) -> dict[str, object]:
    return {
        "recent_cycle_window": max(1, recent_cycle_window),
        "selected_cycles": 0,
        "retained_cycles": 0,
        "rejected_cycles": 0,
        "no_yield_cycles": 0,
        "recent_incomplete_cycles": 0,
        "recent_observation_timeout_cycles": 0,
        "recent_budgeted_observation_timeout_cycles": 0,
        "last_observation_timeout_budget_source": "",
        "repeated_observation_timeout_budget_source_count": 0,
        "last_decision_state": "",
        "last_cycle_id": "",
        "average_retained_pass_rate_delta": 0.0,
        "average_rejected_pass_rate_delta": 0.0,
        "total_decisions": 0,
        "retention_rate": 0.0,
        "rejection_rate": 0.0,
        "recent_retention_rate": 0.0,
        "recent_rejection_rate": 0.0,
        "recent_regression_cycles": 0,
        "recent_phase_gate_failure_cycles": 0,
        "repeated_phase_gate_reason_count": 0,
        "last_phase_gate_failure_reason": "",
        "recent_reconciled_failure_cycles": 0,
        "net_retained_cycle_advantage": 0,
        "decision_quality_score": 0.0,
    }


def campaign_breadth_pressure(summary: dict[str, object]) -> float:
    no_yield_cycles = int(summary.get("no_yield_cycles", 0) or 0)
    incomplete_cycles = int(summary.get("recent_incomplete_cycles", 0) or 0)
    reconciled_failures = int(summary.get("recent_reconciled_failure_cycles", 0) or 0)
    pressure = no_yield_cycles * 0.2 + incomplete_cycles * 0.35 + reconciled_failures * 0.45
    return round(max(0.0, min(1.0, pressure)), 4)


def variant_breadth_pressure(summary: dict[str, object]) -> float:
    return campaign_breadth_pressure(summary)


def record_has_phase_gate_failure(record: dict[str, object]) -> bool:
    from ... import improvement as core

    phase_gate_passed = core._record_phase_gate_passed(record)
    if phase_gate_passed is False:
        return True
    metrics = record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return False
    return int(metrics.get("phase_gate_failure_count", 0) or 0) > 0


def record_is_reconciled_failure(record: dict[str, object]) -> bool:
    if str(record.get("state", "")).strip() != "reject":
        return False
    metrics = record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return False
    return bool(metrics.get("incomplete_cycle", False) or metrics.get("finalize_exception", False))


def record_has_observation_timeout(record: dict[str, object]) -> bool:
    if str(record.get("state", "")).strip() != "observe":
        return False
    metrics = record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return False
    return bool(metrics.get("observation_timed_out", False) or metrics.get("observation_budget_exceeded", False))


def record_observation_timeout_budget_source(record: dict[str, object]) -> str:
    if not record_has_observation_timeout(record):
        return ""
    metrics = record.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        return ""
    for field in (
        "observation_current_task_timeout_budget_source",
        "current_task_timeout_budget_source",
    ):
        source = str(metrics.get(field, "")).strip()
        if source and source != "none":
            return source
    return ""


def _activity_summary_from_records(
    planner: Any,
    *,
    relevant: list[dict[str, object]],
    recent_cycle_window: int,
) -> dict[str, object]:
    from ... import improvement as core

    if not relevant:
        return empty_recent_activity_summary(recent_cycle_window)
    selected_cycle_ids = {
        str(record.get("cycle_id", ""))
        for record in relevant
        if str(record.get("state", "")) == "select"
    }
    latest_by_cycle: dict[str, dict[str, object]] = {}
    for record in relevant:
        cycle_id = str(record.get("cycle_id", "")).strip()
        if cycle_id:
            latest_by_cycle[cycle_id] = record
    decision_records = [
        record for record in relevant if str(record.get("state", "")) in {"retain", "reject"}
    ]
    retained = [record for record in decision_records if str(record.get("state", "")) == "retain"]
    rejected = [record for record in decision_records if str(record.get("state", "")) == "reject"]
    last_decision = decision_records[-1] if decision_records else None
    summary = {
        "recent_cycle_window": max(1, recent_cycle_window),
        "selected_cycles": len(selected_cycle_ids),
        "retained_cycles": len(retained),
        "rejected_cycles": len(rejected),
        "last_decision_state": "" if last_decision is None else str(last_decision.get("state", "")),
        "last_cycle_id": "" if last_decision is None else str(last_decision.get("cycle_id", "")),
        "average_retained_pass_rate_delta": average_metric_delta(
            retained,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
        "average_rejected_pass_rate_delta": average_metric_delta(
            rejected,
            baseline_key="baseline_pass_rate",
            candidate_key="candidate_pass_rate",
        ),
    }
    total_decisions = len(decision_records)
    selected_cycles = len(selected_cycle_ids)
    summary["no_yield_cycles"] = max(0, selected_cycles - total_decisions)
    decision_cycle_ids = {str(record.get("cycle_id", "")) for record in decision_records}
    summary["recent_incomplete_cycles"] = len(
        {
            cycle_id
            for cycle_id in selected_cycle_ids
            if cycle_id not in decision_cycle_ids
            and str(latest_by_cycle.get(cycle_id, {}).get("state", "")).strip() in {"generate", "evaluate"}
        }
    )
    summary["total_decisions"] = total_decisions
    summary["retention_rate"] = round(0.0 if total_decisions == 0 else len(retained) / total_decisions, 4)
    summary["rejection_rate"] = round(0.0 if total_decisions == 0 else len(rejected) / total_decisions, 4)
    summary["recent_retention_rate"] = round(
        0.0 if selected_cycles == 0 else len(retained) / selected_cycles,
        4,
    )
    summary["recent_rejection_rate"] = round(
        0.0 if selected_cycles == 0 else len(rejected) / selected_cycles,
        4,
    )
    summary["recent_regression_cycles"] = len(
        {
            str(record.get("cycle_id", ""))
            for record in relevant
            if planner._record_has_regression_signal(record)
        }
    )
    summary["recent_phase_gate_failure_cycles"] = len(
        {
            str(record.get("cycle_id", ""))
            for record in relevant
            if planner._record_has_phase_gate_failure(record)
        }
    )
    phase_gate_failure_reasons = [
        reason
        for record in relevant
        for reason in core._record_phase_gate_failures(record)
    ]
    last_phase_gate_failure_reason = phase_gate_failure_reasons[-1] if phase_gate_failure_reasons else ""
    repeated_phase_gate_reason_count = 0
    if last_phase_gate_failure_reason:
        repeated_phase_gate_reason_count = sum(
            1 for reason in phase_gate_failure_reasons if reason == last_phase_gate_failure_reason
        )
    summary["last_phase_gate_failure_reason"] = last_phase_gate_failure_reason
    summary["repeated_phase_gate_reason_count"] = repeated_phase_gate_reason_count
    summary["recent_reconciled_failure_cycles"] = len(
        {
            str(record.get("cycle_id", ""))
            for record in decision_records
            if planner._record_is_reconciled_failure(record)
        }
    )
    observation_timeout_records = [
        record for record in relevant if planner._record_has_observation_timeout(record)
    ]
    summary["recent_observation_timeout_cycles"] = len(
        {str(record.get("cycle_id", "")) for record in observation_timeout_records}
    )
    timeout_budget_sources = [
        source
        for record in observation_timeout_records
        if (source := planner._record_observation_timeout_budget_source(record))
    ]
    summary["recent_budgeted_observation_timeout_cycles"] = len(
        {
            str(record.get("cycle_id", ""))
            for record in observation_timeout_records
            if planner._record_observation_timeout_budget_source(record)
        }
    )
    last_observation_timeout_budget_source = timeout_budget_sources[-1] if timeout_budget_sources else ""
    repeated_observation_timeout_budget_source_count = 0
    if last_observation_timeout_budget_source:
        repeated_observation_timeout_budget_source_count = sum(
            1 for source in timeout_budget_sources if source == last_observation_timeout_budget_source
        )
    summary["last_observation_timeout_budget_source"] = last_observation_timeout_budget_source
    summary["repeated_observation_timeout_budget_source_count"] = repeated_observation_timeout_budget_source_count
    summary["net_retained_cycle_advantage"] = len(retained) - len(rejected)
    summary["decision_quality_score"] = decision_quality_score(summary)
    return summary
