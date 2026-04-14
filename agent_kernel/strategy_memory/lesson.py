from __future__ import annotations

from typing import Any


def _family_gains(report: dict[str, object]) -> tuple[list[str], list[str]]:
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        return [], []
    family_deltas = evidence.get("family_pass_rate_delta", {})
    if not isinstance(family_deltas, dict):
        family_deltas = {}
    improved = sorted(str(family) for family, value in family_deltas.items() if float(value or 0.0) > 0.0)
    regressed = sorted(str(family) for family, value in family_deltas.items() if float(value or 0.0) < 0.0)
    return improved, regressed


def _phase_gate_summary(report: dict[str, object]) -> tuple[bool, list[str]]:
    phase_gate = report.get("phase_gate_report", {})
    if not isinstance(phase_gate, dict):
        return False, []
    failures = [str(value).strip() for value in list(phase_gate.get("failures", []) or []) if str(value).strip()]
    return bool(phase_gate.get("passed", False)), failures


def estimate_retained_gain(report: dict[str, object]) -> float:
    final_state = str(report.get("final_state", "")).strip()
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    pass_rate_delta = float(evidence.get("pass_rate_delta", 0.0) or 0.0)
    step_delta = float(evidence.get("average_step_delta", 0.0) or 0.0)
    trusted_delta = float(evidence.get("trusted_carryover_repair_rate_delta", 0.0) or 0.0)
    family_gain = float(evidence.get("family_discrimination_gain", 0.0) or 0.0)
    raw = pass_rate_delta + max(0.0, -step_delta) * 0.05 + trusted_delta + family_gain * 0.25
    if final_state == "retain":
        return round(max(0.15 if raw <= 0.0 else raw, raw), 4)
    if final_state == "reject":
        return round(min(-0.2, raw - 0.2), 4)
    return round(raw, 4)


def synthesize_strategy_lesson(report: dict[str, object]) -> dict[str, object]:
    final_state = str(report.get("final_state", "")).strip() or "pending"
    final_reason = str(report.get("final_reason", "")).strip()
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    improved_families, regressed_families = _family_gains(report)
    family_breadth_gain = len(improved_families)
    pass_rate_delta = float(evidence.get("pass_rate_delta", 0.0) or 0.0)
    step_delta = float(evidence.get("average_step_delta", 0.0) or 0.0)
    trusted_delta = float(evidence.get("trusted_carryover_repair_rate_delta", 0.0) or 0.0)
    false_failure_rate = float(evidence.get("false_failure_rate", 0.0) or 0.0)
    closeout_mode = str(report.get("closeout_mode", "")).strip()
    phase_gate_passed, phase_gate_failures = _phase_gate_summary(report)
    unattended_closeout_ready = (
        final_state == "retain"
        and phase_gate_passed
        and closeout_mode in {"natural", "child_native_before_partial_timeout"}
    )
    lesson_parts: list[str] = []
    if final_state == "retain":
        lesson_parts.append("retained strategy")
    elif final_state == "reject":
        lesson_parts.append("rejected strategy")
    else:
        lesson_parts.append("pending strategy")
    if final_reason:
        lesson_parts.append(final_reason)
    if pass_rate_delta > 0.0:
        lesson_parts.append(f"pass_rate_delta={pass_rate_delta:.3f}")
    if step_delta < 0.0:
        lesson_parts.append(f"step_delta={step_delta:.3f}")
    if trusted_delta > 0.0:
        lesson_parts.append(f"trusted_carryover_delta={trusted_delta:.3f}")
    if improved_families:
        lesson_parts.append(f"improved_families={','.join(improved_families)}")
    if family_breadth_gain >= 2:
        lesson_parts.append(f"family_breadth_gain={family_breadth_gain}")
    if regressed_families:
        lesson_parts.append(f"regressed_families={','.join(regressed_families)}")
    if closeout_mode:
        lesson_parts.append(f"closeout_mode={closeout_mode}")
    if phase_gate_passed:
        lesson_parts.append("phase_gate=pass")
    elif phase_gate_failures:
        lesson_parts.append(f"phase_gate_failures={','.join(phase_gate_failures[:3])}")
    if false_failure_rate > 0.0:
        lesson_parts.append(f"false_failure_rate={false_failure_rate:.3f}")
    reuse_conditions: list[str] = []
    avoid_conditions: list[str] = []
    if final_state == "retain":
        reuse_conditions.append("reuse when the same subsystem needs a retained baseline-preserving intervention")
        if improved_families:
            reuse_conditions.append(f"prefer for family coverage near {','.join(improved_families)}")
        if family_breadth_gain >= 2:
            reuse_conditions.append("prefer when unattended runs need broader family coverage from one retained change")
        if trusted_delta > 0.0:
            reuse_conditions.append("prefer when trusted retrieval carryover must influence the next repair step")
        if unattended_closeout_ready:
            reuse_conditions.append("prefer when unattended closeout should finish cleanly without extra supervisor recovery")
    if final_state == "reject":
        avoid_conditions.append(f"avoid when the strategy still terminates as {final_reason or 'reject'}")
        if false_failure_rate >= 0.5:
            avoid_conditions.append("avoid when verifier discrimination is weak or false-failure pressure is high")
        if pass_rate_delta == 0.0 and step_delta == 0.0 and trusted_delta == 0.0 and not improved_families:
            avoid_conditions.append("avoid replaying unchanged zero-yield variants without new evidence")
        if closeout_mode and closeout_mode not in {"natural", "child_native_before_partial_timeout"}:
            avoid_conditions.append("avoid when unattended closeout already degrades into partial or recovery-heavy completion")
        if phase_gate_failures:
            avoid_conditions.append("avoid when the same phase-gate failure pattern is still unresolved")
    return {
        "analysis_lesson": "; ".join(lesson_parts),
        "reuse_conditions": reuse_conditions,
        "avoid_conditions": avoid_conditions,
        "retained_gain": estimate_retained_gain(report),
    }
