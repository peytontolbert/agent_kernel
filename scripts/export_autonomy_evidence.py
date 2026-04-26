from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import hashlib
import json
from typing import Any


REQUIRED_A4_FAMILIES = ("integration", "project", "repository", "repo_chore", "repo_sandbox")
SUPPORTED_STATIC_LEVELS = ("A4", "A5_substrate", "A5", "A6")
MIN_A5_PRODUCTION_ROLE_SECONDS = 7200
MIN_A6_RETAINED_GAIN_RUNS = 2
MIN_A6_CANDIDATE_BASELINE_REPORTS = 2


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def build_a4_evidence_packet(
    report: dict[str, Any],
    *,
    source_report_path: str,
    source_report_sha256: str = "",
) -> dict[str, Any]:
    evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    decision_state = report.get("decision_state") if isinstance(report.get("decision_state"), dict) else {}
    phase_gate_report = report.get("phase_gate_report") if isinstance(report.get("phase_gate_report"), dict) else {}
    compatibility = report.get("compatibility") if isinstance(report.get("compatibility"), dict) else {}
    candidate_isolation = (
        report.get("candidate_isolation_summary")
        if isinstance(report.get("candidate_isolation_summary"), dict)
        else {}
    )
    family_delta = evidence.get("family_pass_rate_delta") if isinstance(evidence.get("family_pass_rate_delta"), dict) else {}
    generated_family_delta = (
        evidence.get("generated_family_pass_rate_delta")
        if isinstance(evidence.get("generated_family_pass_rate_delta"), dict)
        else {}
    )
    required_families_present = sorted(
        family for family in REQUIRED_A4_FAMILIES if family in family_delta or family in generated_family_delta
    )
    missing_required_families = [
        family for family in REQUIRED_A4_FAMILIES if family not in required_families_present
    ]

    gates = {
        "retained": report.get("final_state") == "retain"
        and decision_state.get("retention_state") == "retain",
        "child_native_decision": decision_state.get("decision_owner") == "child_native",
        "runtime_managed_conversion": decision_state.get("decision_conversion_state") == "runtime_managed",
        "natural_closeout": decision_state.get("closeout_mode") == "natural",
        "phase_gates_passed": phase_gate_report.get("passed") is True,
        "artifact_compatible": compatibility.get("compatible") is True,
        "runtime_managed_artifact_path": candidate_isolation.get("runtime_managed_artifact_path") is True,
        "required_family_surface_present": not missing_required_families,
        "non_regressive_trace": int(evidence.get("confirmation_regressed_trace_task_count") or 0) == 0,
        "non_regressive_trajectory": int(evidence.get("confirmation_regressed_trajectory_task_count") or 0) == 0,
        "transition_model_improved": int(evidence.get("transition_model_improvement_count") or 0) > 0,
    }
    supported = all(gates.values())

    return {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "created_at": datetime.now(UTC).isoformat(),
        "claim": {
            "level": "A4",
            "scope": "narrow_direct_transition_model_route",
            "status": "supported" if supported else "not_supported",
            "summary": (
                "Retained child-native runtime-managed natural closeout on the direct five-family "
                "transition_model route."
                if supported
                else "Source report does not satisfy every narrow A4 evidence gate."
            ),
        },
        "source": {
            "cycle_id": report.get("cycle_id", ""),
            "subsystem": report.get("subsystem", ""),
            "source_report_path": source_report_path,
            "source_report_sha256": source_report_sha256,
            "created_at": report.get("created_at", ""),
        },
        "decision_state": {
            "final_state": report.get("final_state", ""),
            "final_reason": report.get("final_reason", ""),
            "decision_owner": decision_state.get("decision_owner", ""),
            "decision_conversion_state": decision_state.get("decision_conversion_state", ""),
            "closeout_mode": decision_state.get("closeout_mode", ""),
            "retention_basis": decision_state.get("retention_basis", ""),
        },
        "gates": gates,
        "required_families": {
            "required": list(REQUIRED_A4_FAMILIES),
            "present": required_families_present,
            "missing": missing_required_families,
        },
        "metrics": {
            "baseline_pass_rate": _number(report.get("baseline_metrics", {}) or {}, "pass_rate"),
            "candidate_pass_rate": _number(report.get("candidate_metrics", {}) or {}, "pass_rate"),
            "pass_rate_delta": _number(evidence, "pass_rate_delta"),
            "average_step_delta": _number(evidence, "average_step_delta"),
            "transition_model_improvement_count": int(evidence.get("transition_model_improvement_count") or 0),
            "transition_signature_count": int(evidence.get("transition_signature_count") or 0),
            "confirmation_run_count": int(evidence.get("confirmation_run_count") or 0),
            "paired_task_non_regression_lower_bound": _number(
                evidence, "confirmation_paired_task_non_regression_rate_lower_bound"
            ),
            "paired_trace_non_regression_lower_bound": _number(
                evidence, "confirmation_paired_trace_non_regression_rate_lower_bound"
            ),
            "paired_trajectory_non_regression_lower_bound": _number(
                evidence, "confirmation_paired_trajectory_non_regression_rate_lower_bound"
            ),
            "regressed_trace_task_count": int(evidence.get("confirmation_regressed_trace_task_count") or 0),
            "regressed_trajectory_task_count": int(
                evidence.get("confirmation_regressed_trajectory_task_count") or 0
            ),
        },
        "artifacts": {
            "active_artifact_path": report.get("active_artifact_path", ""),
            "candidate_artifact_path": report.get("candidate_artifact_path", ""),
            "artifact_sha256": report.get("artifact_sha256", ""),
            "artifact_lifecycle_state": report.get("artifact_lifecycle_state", ""),
        },
        "open_limits": [
            "This packet supports a narrow direct A4 crossing anchor, not a broad A5 claim.",
            "Fresh official campaigns still need to repeat or package this route reliably.",
            "Patch55 is negative hardening evidence for a later candidate, not a disproof of this source cycle.",
        ],
    }


def verify_a4_evidence_packet(packet: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for field_path in (
        ("claim", "level"),
        ("claim", "scope"),
        ("claim", "status"),
        ("source", "cycle_id"),
        ("source", "source_report_sha256"),
        ("decision_state", "final_state"),
        ("decision_state", "decision_owner"),
        ("decision_state", "decision_conversion_state"),
        ("decision_state", "closeout_mode"),
    ):
        observed: Any = packet
        wanted: Any = expected
        for key in field_path:
            observed = observed.get(key, {}) if isinstance(observed, dict) else {}
            wanted = wanted.get(key, {}) if isinstance(wanted, dict) else {}
        if observed != wanted:
            failures.append(
                f"{'.'.join(field_path)} mismatch: observed={observed!r} expected={wanted!r}"
            )
    if packet.get("gates") != expected.get("gates"):
        failures.append("gates mismatch")
    if packet.get("required_families") != expected.get("required_families"):
        failures.append("required_families mismatch")
    return failures


def verify_static_autonomy_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if packet.get("report_kind") != "autonomy_evidence_packet":
        failures.append("report_kind must be autonomy_evidence_packet")
    claim = packet.get("claim") if isinstance(packet.get("claim"), dict) else {}
    level = str(claim.get("level", "")).strip()
    if level not in SUPPORTED_STATIC_LEVELS:
        failures.append(f"claim.level must be one of {SUPPORTED_STATIC_LEVELS!r}")
    if claim.get("status") != "supported":
        failures.append("claim.status must be supported")
    if not str(claim.get("scope", "")).strip():
        failures.append("claim.scope is required")
    if not str(claim.get("summary", "")).strip():
        failures.append("claim.summary is required")
    open_limits = packet.get("open_limits")
    if not isinstance(open_limits, list) or not open_limits:
        failures.append("open_limits must be a non-empty list")

    if level == "A4":
        gates = packet.get("gates") if isinstance(packet.get("gates"), dict) else {}
        if not gates:
            failures.append("A4 gates object is required")
        elif not all(value is True for value in gates.values()):
            failures.append("A4 gates must all be true for supported packet")
        metrics = packet.get("metrics") if isinstance(packet.get("metrics"), dict) else {}
        if int(metrics.get("regressed_trace_task_count") or 0) != 0:
            failures.append("A4 regressed_trace_task_count must be zero")
        if int(metrics.get("regressed_trajectory_task_count") or 0) != 0:
            failures.append("A4 regressed_trajectory_task_count must be zero")
        required_families = (
            packet.get("required_families") if isinstance(packet.get("required_families"), dict) else {}
        )
        if required_families.get("missing") not in ([], None):
            failures.append("A4 required_families.missing must be empty")

    if level == "A5_substrate":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        if not evidence:
            failures.append("A5_substrate evidence object is required")
        focused_tests = evidence.get("focused_tests") if isinstance(evidence.get("focused_tests"), dict) else {}
        if not str(focused_tests.get("command", "")).strip():
            failures.append("A5_substrate focused_tests.command is required")
        if "passed" not in str(focused_tests.get("result", "")):
            failures.append("A5_substrate focused_tests.result must record passing tests")
        has_cli_evidence = any(str(key).startswith("isolated_cli_") for key in evidence)
        if not has_cli_evidence:
            failures.append("A5_substrate requires at least one isolated_cli_* evidence block")

    if level == "A5":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        confirmation = (
            evidence.get("a5_confirmation")
            if isinstance(evidence.get("a5_confirmation"), dict)
            else {}
        )
        if not confirmation:
            failures.append("A5 evidence.a5_confirmation object is required")
        duration_seconds = _number(confirmation, "role_duration_seconds")
        if duration_seconds < MIN_A5_PRODUCTION_ROLE_SECONDS:
            failures.append(
                f"A5 role_duration_seconds must be at least {MIN_A5_PRODUCTION_ROLE_SECONDS}"
            )
        if confirmation.get("product_native_intake") is not True:
            failures.append("A5 product_native_intake must be true")
        if confirmation.get("product_user_workstream_intake") is not True:
            failures.append("A5 product_user_workstream_intake must be true")
        closeout = (
            confirmation.get("native_role_closeout")
            if isinstance(confirmation.get("native_role_closeout"), dict)
            else {}
        )
        if closeout.get("closeout_ready") is not True:
            failures.append("A5 native_role_closeout.closeout_ready must be true")
        if closeout.get("closeout_mode") != "queue_empty_trusted":
            failures.append("A5 native_role_closeout.closeout_mode must be queue_empty_trusted")
        if closeout.get("operator_steering_required") is not False:
            failures.append("A5 native_role_closeout.operator_steering_required must be false")
        if int(closeout.get("active_leases") or 0) != 0:
            failures.append("A5 native_role_closeout.active_leases must be zero")
        counts = (
            confirmation.get("required_family_counted_gated_report_counts")
            if isinstance(confirmation.get("required_family_counted_gated_report_counts"), dict)
            else {}
        )
        missing_or_empty = [
            family for family in REQUIRED_A4_FAMILIES if int(counts.get(family, 0) or 0) <= 0
        ]
        if missing_or_empty:
            failures.append(
                "A5 required_family_counted_gated_report_counts missing or zero for: "
                + ",".join(missing_or_empty)
            )
        if confirmation.get("trust_status") != "trusted":
            failures.append("A5 trust_status must be trusted")
        if int(confirmation.get("active_lease_count") or 0) != 0:
            failures.append("A5 active_lease_count must be zero")
        if confirmation.get("interruption_resume_proven") is not True:
            failures.append("A5 interruption_resume_proven must be true")

    if level == "A6":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        confirmation = (
            evidence.get("a6_confirmation")
            if isinstance(evidence.get("a6_confirmation"), dict)
            else {}
        )
        if not confirmation:
            failures.append("A6 evidence.a6_confirmation object is required")
        retained_gain_runs = int(confirmation.get("retained_gain_runs") or 0)
        if retained_gain_runs < MIN_A6_RETAINED_GAIN_RUNS:
            failures.append(f"A6 retained_gain_runs must be at least {MIN_A6_RETAINED_GAIN_RUNS}")
        candidate_baseline_reports = int(confirmation.get("candidate_baseline_report_count") or 0)
        if candidate_baseline_reports < MIN_A6_CANDIDATE_BASELINE_REPORTS:
            failures.append(
                "A6 candidate_baseline_report_count must be at least "
                f"{MIN_A6_CANDIDATE_BASELINE_REPORTS}"
            )
        if confirmation.get("regression_gates_survived") is not True:
            failures.append("A6 regression_gates_survived must be true")
        if confirmation.get("retained_changes_affect_runtime") is not True:
            failures.append("A6 retained_changes_affect_runtime must be true")
        if confirmation.get("autonomous_compounding_claim_ready") is not True:
            failures.append("A6 autonomous_compounding_claim_ready must be true")
        if confirmation.get("non_collapsing_repeated_runs") is not True:
            failures.append("A6 non_collapsing_repeated_runs must be true")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-report", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--verify-packet", default=None)
    parser.add_argument("--verify-static-packet", default=None)
    args = parser.parse_args()
    if args.verify_static_packet:
        verify_path = Path(args.verify_static_packet)
        failures = verify_static_autonomy_packet(_read_json(verify_path))
        if failures:
            raise SystemExit("static autonomy evidence packet verification failed: " + "; ".join(failures))
        print(f"verified_static_packet={verify_path}")
        return
    if not args.cycle_report:
        raise SystemExit("--cycle-report is required unless --verify-static-packet is used")
    if not args.output_json and not args.verify_packet:
        raise SystemExit("one of --output-json or --verify-packet is required")

    report_path = Path(args.cycle_report)
    packet = build_a4_evidence_packet(
        _read_json(report_path),
        source_report_path=str(report_path),
        source_report_sha256=_sha256(report_path),
    )
    if args.verify_packet:
        verify_path = Path(args.verify_packet)
        failures = verify_a4_evidence_packet(_read_json(verify_path), packet)
        if failures:
            raise SystemExit("autonomy evidence packet verification failed: " + "; ".join(failures))
        print(
            f"verified_packet={verify_path} "
            f"status={packet['claim']['status']} "
            f"cycle_id={packet['source']['cycle_id']}"
        )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"claim_level={packet['claim']['level']} "
            f"claim_scope={packet['claim']['scope']} "
            f"status={packet['claim']['status']} "
            f"cycle_id={packet['source']['cycle_id']} "
            f"output_json={output_path}"
        )


if __name__ == "__main__":
    main()
