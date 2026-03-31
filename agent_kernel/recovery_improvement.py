from __future__ import annotations

from copy import deepcopy

from evals.metrics import EvalMetrics
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)


def recovery_behavior_controls(
    metrics: EvalMetrics,
    trust_ledger: dict[str, object] | None,
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    del metrics
    ledger = trust_ledger if isinstance(trust_ledger, dict) else {}
    overall = ledger.get("overall_summary", {}) if isinstance(ledger.get("overall_summary", {}), dict) else {}
    gated = ledger.get("gated_summary", {}) if isinstance(ledger.get("gated_summary", {}), dict) else {}
    rollback_rate = float(gated.get("rollback_performed_rate", overall.get("rollback_performed_rate", 0.0)) or 0.0)
    hidden_rate = float(gated.get("hidden_side_effect_risk_rate", overall.get("hidden_side_effect_risk_rate", 0.0)) or 0.0)
    controls: dict[str, object] = {
        "snapshot_before_execution": True,
        "rollback_on_runner_exception": True,
        "rollback_on_failed_outcome": True,
        "rollback_on_safe_stop": False,
        "verify_post_rollback_file_count": False,
        "max_post_rollback_file_count": 0,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if rollback_rate > 0.0 or hidden_rate > 0.0 or focus == "rollback_safety":
        controls["rollback_on_safe_stop"] = True
        controls["verify_post_rollback_file_count"] = True
    if focus == "snapshot_coverage":
        controls["snapshot_before_execution"] = True
        controls["verify_post_rollback_file_count"] = True
    return controls


def build_recovery_proposal_artifact(
    metrics: EvalMetrics,
    trust_ledger: dict[str, object] | None,
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    controls = recovery_behavior_controls(
        metrics,
        trust_ledger,
        focus=None if generation_focus == "balanced" else generation_focus,
        baseline=retained_recovery_controls(current_payload),
    )
    return build_standard_proposal_artifact(
        artifact_kind="recovery_policy_set",
        generation_focus=generation_focus,
        control_schema="workspace_recovery_controls_v1",
        retention_gate=retention_gate_preset("recovery"),
        controls=controls,
        proposals=_proposals(trust_ledger, generation_focus),
        extra_sections={"ledger_summary": _ledger_summary(trust_ledger)},
    )


def retained_recovery_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="recovery_policy_set", section="controls")


def _ledger_summary(trust_ledger: dict[str, object] | None) -> dict[str, object]:
    ledger = trust_ledger if isinstance(trust_ledger, dict) else {}
    overall = ledger.get("overall_summary", {}) if isinstance(ledger.get("overall_summary", {}), dict) else {}
    gated = ledger.get("gated_summary", {}) if isinstance(ledger.get("gated_summary", {}), dict) else {}
    return {
        "rollback_performed_rate": float(
            gated.get("rollback_performed_rate", overall.get("rollback_performed_rate", 0.0)) or 0.0
        ),
        "hidden_side_effect_risk_rate": float(
            gated.get("hidden_side_effect_risk_rate", overall.get("hidden_side_effect_risk_rate", 0.0)) or 0.0
        ),
        "reports_considered": int(ledger.get("reports_considered", 0) or 0),
    }


def _proposals(trust_ledger: dict[str, object] | None, focus: str) -> list[dict[str, object]]:
    summary = _ledger_summary(trust_ledger)
    proposals: list[dict[str, object]] = []
    if float(summary.get("rollback_performed_rate", 0.0)) > 0.0 or focus == "rollback_safety":
        proposals.append(
            {
                "area": "rollback_safety",
                "priority": 5,
                "reason": "unattended runs still require rollback or leave hidden-side-effect risk after non-success outcomes",
                "suggestion": "Enable rollback on safe-stop outcomes and verify the workspace is clean after restoration.",
            }
        )
    if int(summary.get("reports_considered", 0)) > 0 or focus == "snapshot_coverage":
        proposals.append(
            {
                "area": "snapshot_coverage",
                "priority": 4,
                "reason": "workspace recovery should remain explicit even when rollback pressure is currently low",
                "suggestion": "Keep execution snapshots and post-rollback verification as retained unattended recovery controls.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "snapshot_coverage",
            "priority": 3,
            "reason": "recovery policy should remain explicit as a retained runtime surface",
            "suggestion": "Preserve bounded snapshot and rollback policy as a retained autonomous subsystem.",
        },
    )
