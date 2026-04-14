from __future__ import annotations

from .improvement_engine import ImprovementCycleRecord


def validate_cycle_record_consistency(record: ImprovementCycleRecord) -> None:
    expected_kinds = {
        "benchmark": "benchmark_candidate_set",
        "retrieval": "retrieval_policy_set",
        "tolbert_model": "tolbert_model_bundle",
        "qwen_adapter": "qwen_adapter_bundle",
        "verifier": "verifier_candidate_set",
        "policy": "prompt_proposal_set",
        "universe": "universe_contract",
        "universe_constitution": "universe_constitution",
        "operating_envelope": "operating_envelope",
        "world_model": "world_model_policy_set",
        "state_estimation": "state_estimation_policy_set",
        "trust": "trust_policy_set",
        "recovery": "recovery_policy_set",
        "delegation": "delegated_runtime_policy_set",
        "transition_model": "transition_model_policy_set",
        "curriculum": "curriculum_proposal_set",
        "capabilities": "capability_module_set",
        "tooling": "tool_candidate_set",
        "skills": "skill_set",
        "operators": "operator_class_set",
    }
    if record.state != "generate":
        return
    expected_kind = expected_kinds.get(record.subsystem)
    if expected_kind and record.artifact_kind and record.artifact_kind != expected_kind:
        raise ValueError(
            f"generate record artifact kind {record.artifact_kind!r} does not match subsystem "
            f"{record.subsystem!r} expected kind {expected_kind!r}"
        )


def record_metrics_summary(record: dict[str, object] | ImprovementCycleRecord) -> dict[str, object]:
    metrics_summary = record.metrics_summary if isinstance(record, ImprovementCycleRecord) else record.get("metrics_summary", {})
    return metrics_summary if isinstance(metrics_summary, dict) else {}


def record_selected_variant_id(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        if str(record.selected_variant_id).strip():
            return str(record.selected_variant_id).strip()
    else:
        direct = str(record.get("selected_variant_id", "")).strip()
        if direct:
            return direct
    metrics_summary = record_metrics_summary(record)
    direct = str(metrics_summary.get("selected_variant_id", "")).strip()
    if direct:
        return direct
    selected_variant = metrics_summary.get("selected_variant", {})
    if isinstance(selected_variant, dict):
        return str(selected_variant.get("variant_id", "")).strip()
    return ""


def record_strategy_candidate_id(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        direct = str(record.strategy_candidate_id).strip()
        if direct:
            return direct
    else:
        direct = str(record.get("strategy_candidate_id", "")).strip()
        if direct:
            return direct
    metrics_summary = record_metrics_summary(record)
    direct = str(metrics_summary.get("strategy_candidate_id", "")).strip()
    if direct:
        return direct
    direct = str(metrics_summary.get("strategy_id", "")).strip()
    if direct:
        return direct
    strategy_candidate = metrics_summary.get("strategy_candidate", {})
    if isinstance(strategy_candidate, dict):
        return str(
            strategy_candidate.get("strategy_candidate_id", "")
            or strategy_candidate.get("strategy_id", "")
        ).strip()
    return ""


def record_strategy_candidate_kind(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        direct = str(record.strategy_candidate_kind).strip()
        if direct:
            return direct
    else:
        direct = str(record.get("strategy_candidate_kind", "")).strip()
        if direct:
            return direct
    metrics_summary = record_metrics_summary(record)
    direct = str(metrics_summary.get("strategy_candidate_kind", "")).strip()
    if direct:
        return direct
    direct = str(metrics_summary.get("strategy_kind", "")).strip()
    if direct:
        return direct
    strategy_candidate = metrics_summary.get("strategy_candidate", {})
    if isinstance(strategy_candidate, dict):
        return str(
            strategy_candidate.get("strategy_candidate_kind", "")
            or strategy_candidate.get("strategy_kind", "")
        ).strip()
    return ""


def record_strategy_origin(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        direct = str(record.strategy_origin).strip()
        if direct:
            return direct
    else:
        direct = str(record.get("strategy_origin", "")).strip()
        if direct:
            return direct
    metrics_summary = record_metrics_summary(record)
    direct = str(metrics_summary.get("strategy_origin", "")).strip()
    if direct:
        return direct
    strategy_candidate = metrics_summary.get("strategy_candidate", {})
    if isinstance(strategy_candidate, dict):
        return str(
            strategy_candidate.get("origin", "")
            or strategy_candidate.get("strategy_origin", "")
        ).strip()
    return ""


def record_prior_retained_cycle_id(record: dict[str, object] | ImprovementCycleRecord) -> str:
    if isinstance(record, ImprovementCycleRecord):
        if str(record.prior_retained_cycle_id).strip():
            return str(record.prior_retained_cycle_id).strip()
    else:
        direct = str(record.get("prior_retained_cycle_id", "")).strip()
        if direct:
            return direct
    metrics_summary = record_metrics_summary(record)
    return str(metrics_summary.get("prior_retained_cycle_id", "")).strip()


def record_float_value(record: dict[str, object] | ImprovementCycleRecord, key: str) -> float | None:
    if isinstance(record, ImprovementCycleRecord):
        direct = getattr(record, key)
        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError):
                return None
    else:
        if key in record and record.get(key) is not None:
            try:
                return float(record.get(key))
            except (TypeError, ValueError):
                return None
    metrics_summary = record_metrics_summary(record)
    if key in metrics_summary and metrics_summary.get(key) is not None:
        try:
            return float(metrics_summary.get(key))
        except (TypeError, ValueError):
            return None
    return None


def record_non_regressed_family_support(record: dict[str, object] | ImprovementCycleRecord) -> int:
    metrics_summary = record_metrics_summary(record)
    support = 0
    for key in ("family_pass_rate_delta", "generated_family_pass_rate_delta"):
        delta_map = metrics_summary.get(key, {})
        if not isinstance(delta_map, dict):
            continue
        for value in delta_map.values():
            try:
                if float(value) >= 0.0:
                    support += 1
            except (TypeError, ValueError):
                continue
    if support > 0:
        return support
    regressed = 0
    if "regressed_family_count" in metrics_summary:
        try:
            regressed = int(metrics_summary.get("regressed_family_count", 0))
        except (TypeError, ValueError):
            regressed = 0
    return max(1, 1 - regressed)


def record_phase_gate_passed(record: dict[str, object] | ImprovementCycleRecord) -> bool | None:
    if isinstance(record, ImprovementCycleRecord):
        if record.phase_gate_passed is not None:
            return bool(record.phase_gate_passed)
    else:
        if "phase_gate_passed" in record:
            return bool(record.get("phase_gate_passed"))
    metrics_summary = record_metrics_summary(record)
    if "phase_gate_passed" in metrics_summary:
        return bool(metrics_summary.get("phase_gate_passed"))
    return None


def record_phase_gate_failures(record: dict[str, object] | ImprovementCycleRecord) -> list[str]:
    metrics_summary = record_metrics_summary(record)
    failures = metrics_summary.get("phase_gate_failures", [])
    if not isinstance(failures, list):
        return []
    return [str(failure).strip() for failure in failures if str(failure).strip()]


def dominant_count_label(counts: dict[str, int]) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    ranked = sorted(
        (
            (int(value), str(label).strip())
            for label, value in counts.items()
            if str(label).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0:
        return ""
    return ranked[0][1]


def dominant_weight_label(counts: dict[str, object]) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    ranked = sorted(
        (
            (float(value), str(label).strip())
            for label, value in counts.items()
            if str(label).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0.0:
        return ""
    return ranked[0][1]


def normalized_cycle_record(record: ImprovementCycleRecord) -> ImprovementCycleRecord:
    return ImprovementCycleRecord(
        cycle_id=record.cycle_id,
        state=record.state,
        subsystem=record.subsystem,
        action=record.action,
        artifact_path=record.artifact_path,
        artifact_kind=record.artifact_kind,
        reason=record.reason,
        metrics_summary=record.metrics_summary,
        candidate_artifact_path=record.candidate_artifact_path,
        active_artifact_path=record.active_artifact_path,
        artifact_lifecycle_state=record.artifact_lifecycle_state,
        artifact_sha256=record.artifact_sha256,
        previous_artifact_sha256=record.previous_artifact_sha256,
        rollback_artifact_path=record.rollback_artifact_path,
        artifact_snapshot_path=record.artifact_snapshot_path,
        selected_variant_id=record.selected_variant_id or record_selected_variant_id(record),
        strategy_candidate_id=record.strategy_candidate_id or record_strategy_candidate_id(record),
        strategy_candidate_kind=record.strategy_candidate_kind or record_strategy_candidate_kind(record),
        strategy_origin=record.strategy_origin or record_strategy_origin(record),
        prior_retained_cycle_id=record.prior_retained_cycle_id or record_prior_retained_cycle_id(record),
        baseline_pass_rate=record.baseline_pass_rate
        if record.baseline_pass_rate is not None
        else record_float_value(record, "baseline_pass_rate"),
        candidate_pass_rate=record.candidate_pass_rate
        if record.candidate_pass_rate is not None
        else record_float_value(record, "candidate_pass_rate"),
        baseline_average_steps=record.baseline_average_steps
        if record.baseline_average_steps is not None
        else record_float_value(record, "baseline_average_steps"),
        candidate_average_steps=record.candidate_average_steps
        if record.candidate_average_steps is not None
        else record_float_value(record, "candidate_average_steps"),
        phase_gate_passed=record.phase_gate_passed
        if record.phase_gate_passed is not None
        else record_phase_gate_passed(record),
        compatibility=record.compatibility,
    )


__all__ = [
    "dominant_count_label",
    "dominant_weight_label",
    "normalized_cycle_record",
    "record_float_value",
    "record_metrics_summary",
    "record_non_regressed_family_support",
    "record_phase_gate_failures",
    "record_phase_gate_passed",
    "record_prior_retained_cycle_id",
    "record_selected_variant_id",
    "record_strategy_candidate_id",
    "record_strategy_candidate_kind",
    "record_strategy_origin",
    "validate_cycle_record_consistency",
]
