from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import hashlib
import json
from typing import Any


REQUIRED_A4_FAMILIES = ("integration", "project", "repository", "repo_chore", "repo_sandbox")
SUPPORTED_STATIC_LEVELS = ("A4", "A5_substrate", "A5", "A6", "A7_readiness", "A7", "A8")
MIN_A5_PRODUCTION_ROLE_SECONDS = 7200
MIN_A6_RETAINED_GAIN_RUNS = 2
MIN_A6_CANDIDATE_BASELINE_REPORTS = 2
MIN_A7_READINESS_ROTATIONS = 2
MIN_A7_READINESS_HELD_OUT_REPORTS = 10
MIN_A7_READINESS_DISTINCT_FRONTIER_SLICES = 10
MIN_A7_READINESS_CLEAN_SUCCESS_RATE = 0.8
MIN_A7_READINESS_HARD_HINTLESS_ROTATIONS = 2
MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESSES = 15
MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESS_RATE = 0.9
MIN_A7_READINESS_STATEFUL_REPAIR_ROTATIONS = 1
MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESSES = 5
MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESS_RATE = 0.9
MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_ROTATIONS = 1
MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESSES = 5
MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESS_RATE = 0.9
MIN_A7_FULL_UNFAMILIAR_DOMAIN_SLICES = 5
MIN_A7_FULL_BROAD_TRANSFER_CLEAN_SUCCESS_RATE = 0.8
MIN_A7_FULL_STRONG_BASELINE_WIN_RATE = 0.5
MIN_A7_FULL_LONG_HORIZON_TRANSFER_SLICES = 1
MIN_A8_STRONG_BASELINE_COMPARISON_SLICES = 5
MIN_A8_SUPERHUMAN_CODING_WINDOW_COUNT = 3
MIN_A8_SUPERHUMAN_CODING_TASK_COUNT = 100
MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE = 0.8
MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE_LOWER_BOUND = 0.6
MIN_A8_UNFAMILIAR_DOMAIN_SLICES = 5
MIN_A8_LONG_HORIZON_TRANSFER_SLICES = 3
MIN_A8_RECURSIVE_COMPOUNDING_RETAINED_GAIN_RUNS = 5
MIN_A8_RECURSIVE_COMPOUNDING_WINDOW_COUNT = 3
MAX_A8_REGRESSION_RATE = 0.02
MIN_A8_CODEFORCES_RATING_EQUIVALENT = 3000
MIN_A8_MLE_BENCH_GOLD_MEDAL_RATE = 0.2
MIN_A8_SWE_BENCH_VERIFIED_RESOLVE_RATE = 0.8
MIN_A8_SWE_REBENCH_RESOLVE_RATE = 0.6
MIN_A8_RE_BENCH_HUMAN_EXPERT_WIN_RATE = 0.5


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


def _a8_target_thresholds() -> dict[str, object]:
    return {
        "codeforces_rating_equivalent": MIN_A8_CODEFORCES_RATING_EQUIVALENT,
        "mle_bench_gold_medal_rate": MIN_A8_MLE_BENCH_GOLD_MEDAL_RATE,
        "swe_bench_verified_resolve_rate": MIN_A8_SWE_BENCH_VERIFIED_RESOLVE_RATE,
        "swe_rebench_resolve_rate": MIN_A8_SWE_REBENCH_RESOLVE_RATE,
        "re_bench_human_expert_win_rate": MIN_A8_RE_BENCH_HUMAN_EXPERT_WIN_RATE,
        "strong_baseline_comparison_slice_count": MIN_A8_STRONG_BASELINE_COMPARISON_SLICES,
        "superhuman_coding_window_count": MIN_A8_SUPERHUMAN_CODING_WINDOW_COUNT,
        "superhuman_coding_task_count": MIN_A8_SUPERHUMAN_CODING_TASK_COUNT,
        "strong_human_baseline_win_rate": MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE,
        "strong_human_baseline_win_rate_lower_bound": MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE_LOWER_BOUND,
        "unfamiliar_domain_slice_count": MIN_A8_UNFAMILIAR_DOMAIN_SLICES,
        "long_horizon_transfer_slice_count": MIN_A8_LONG_HORIZON_TRANSFER_SLICES,
        "recursive_compounding_retained_gain_runs": MIN_A8_RECURSIVE_COMPOUNDING_RETAINED_GAIN_RUNS,
        "recursive_compounding_window_count": MIN_A8_RECURSIVE_COMPOUNDING_WINDOW_COUNT,
        "max_regression_rate": MAX_A8_REGRESSION_RATE,
    }


A8_BENCHMARK_REQUIRED_METRICS = {
    "codeforces": ("rating_equivalent",),
    "mle_bench": ("gold_medal_rate",),
    "swe_bench_verified": ("resolve_rate",),
    "swe_rebench": ("resolve_rate",),
    "re_bench": ("human_expert_win_rate",),
    "sustained_coding_window": (
        "window_count",
        "task_count",
        "strong_human_baseline_win_rate",
        "strong_human_baseline_win_rate_lower_bound",
        "unfamiliar_domain_slice_count",
        "long_horizon_transfer_slice_count",
        "strong_baseline_comparison_slice_count",
        "regression_rate",
    ),
    "recursive_compounding": ("retained_gain_runs", "window_count"),
}


def _max_metric(results: list[dict[str, Any]], benchmark: str, metric: str, default: float = 0.0) -> float:
    values = [
        _number(result.get("metrics", {}) or {}, metric, default)
        for result in results
        if result.get("benchmark") == benchmark
    ]
    return max(values) if values else default


def _min_metric(results: list[dict[str, Any]], benchmark: str, metric: str, default: float = 0.0) -> float:
    values = [
        _number(result.get("metrics", {}) or {}, metric, default)
        for result in results
        if result.get("benchmark") == benchmark
    ]
    return min(values) if values else default


def _any_metric_true(results: list[dict[str, Any]], benchmark: str, metric: str) -> bool:
    return any(
        (result.get("metrics") if isinstance(result.get("metrics"), dict) else {}).get(metric) is True
        for result in results
        if result.get("benchmark") == benchmark
    )


def _all_benchmark_results_mark_conservative(results: list[dict[str, Any]]) -> bool:
    required = set(A8_BENCHMARK_REQUIRED_METRICS)
    present = {str(result.get("benchmark", "")).strip() for result in results}
    if not required.issubset(present):
        return False
    return all(
        (result.get("metrics") if isinstance(result.get("metrics"), dict) else {}).get(
            "conservative_comparison_report"
        )
        is True
        for result in results
    )


def verify_a8_benchmark_result_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if packet.get("report_kind") != "a8_benchmark_result":
        failures.append("report_kind must be a8_benchmark_result")
    benchmark = str(packet.get("benchmark", "")).strip()
    if benchmark not in A8_BENCHMARK_REQUIRED_METRICS:
        failures.append(
            "benchmark must be one of "
            + ",".join(sorted(A8_BENCHMARK_REQUIRED_METRICS))
        )
    metrics = packet.get("metrics") if isinstance(packet.get("metrics"), dict) else {}
    if not metrics:
        failures.append("metrics object is required")
    for metric in A8_BENCHMARK_REQUIRED_METRICS.get(benchmark, ()):
        if metric not in metrics:
            failures.append(f"metrics.{metric} is required for benchmark {benchmark}")
    return failures


def build_a8_autonomy_packet_from_benchmark_results(
    results: list[dict[str, Any]],
    *,
    source_paths: list[str] | None = None,
) -> dict[str, Any]:
    validation_failures = [
        failure
        for result in results
        for failure in verify_a8_benchmark_result_packet(result)
    ]
    a8 = {
        "strong_baseline_comparison_slice_count": int(
            _max_metric(results, "sustained_coding_window", "strong_baseline_comparison_slice_count")
        ),
        "superhuman_coding_window_count": int(_max_metric(results, "sustained_coding_window", "window_count")),
        "superhuman_coding_task_count": int(_max_metric(results, "sustained_coding_window", "task_count")),
        "strong_human_baseline_win_rate": _max_metric(
            results, "sustained_coding_window", "strong_human_baseline_win_rate"
        ),
        "strong_human_baseline_win_rate_lower_bound": _max_metric(
            results, "sustained_coding_window", "strong_human_baseline_win_rate_lower_bound"
        ),
        "unfamiliar_domain_slice_count": int(
            _max_metric(results, "sustained_coding_window", "unfamiliar_domain_slice_count")
        ),
        "long_horizon_transfer_slice_count": int(
            _max_metric(results, "sustained_coding_window", "long_horizon_transfer_slice_count")
        ),
        "recursive_compounding_retained_gain_runs": int(
            _max_metric(results, "recursive_compounding", "retained_gain_runs")
        ),
        "recursive_compounding_window_count": int(_max_metric(results, "recursive_compounding", "window_count")),
        "regression_rate": _min_metric(results, "sustained_coding_window", "regression_rate", 1.0),
        "codeforces_rating_equivalent": int(_max_metric(results, "codeforces", "rating_equivalent")),
        "mle_bench_gold_medal_rate": _max_metric(results, "mle_bench", "gold_medal_rate"),
        "swe_bench_verified_resolve_rate": _max_metric(results, "swe_bench_verified", "resolve_rate"),
        "swe_rebench_resolve_rate": _max_metric(results, "swe_rebench", "resolve_rate"),
        "re_bench_human_expert_win_rate": _max_metric(results, "re_bench", "human_expert_win_rate"),
        "decisive_outperformance": False,
        "conservative_comparison_report": _all_benchmark_results_mark_conservative(results),
        "verified_recursive_compounding": _any_metric_true(
            results, "recursive_compounding", "verified_recursive_compounding"
        ),
        "sustained_superhuman_coding": False,
        "a8_claim_ready": False,
    }
    a8["decisive_outperformance"] = (
        a8["codeforces_rating_equivalent"] >= MIN_A8_CODEFORCES_RATING_EQUIVALENT
        and a8["mle_bench_gold_medal_rate"] >= MIN_A8_MLE_BENCH_GOLD_MEDAL_RATE
        and a8["swe_bench_verified_resolve_rate"] >= MIN_A8_SWE_BENCH_VERIFIED_RESOLVE_RATE
        and a8["swe_rebench_resolve_rate"] >= MIN_A8_SWE_REBENCH_RESOLVE_RATE
        and a8["re_bench_human_expert_win_rate"] >= MIN_A8_RE_BENCH_HUMAN_EXPERT_WIN_RATE
        and a8["strong_human_baseline_win_rate"] >= MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE
        and a8["strong_human_baseline_win_rate_lower_bound"]
        >= MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE_LOWER_BOUND
    )
    a8["sustained_superhuman_coding"] = (
        a8["strong_baseline_comparison_slice_count"] >= MIN_A8_STRONG_BASELINE_COMPARISON_SLICES
        and a8["superhuman_coding_window_count"] >= MIN_A8_SUPERHUMAN_CODING_WINDOW_COUNT
        and a8["superhuman_coding_task_count"] >= MIN_A8_SUPERHUMAN_CODING_TASK_COUNT
        and a8["unfamiliar_domain_slice_count"] >= MIN_A8_UNFAMILIAR_DOMAIN_SLICES
        and a8["long_horizon_transfer_slice_count"] >= MIN_A8_LONG_HORIZON_TRANSFER_SLICES
        and a8["regression_rate"] <= MAX_A8_REGRESSION_RATE
    )
    a8["verified_recursive_compounding"] = (
        a8["verified_recursive_compounding"]
        and a8["recursive_compounding_retained_gain_runs"]
        >= MIN_A8_RECURSIVE_COMPOUNDING_RETAINED_GAIN_RUNS
        and a8["recursive_compounding_window_count"] >= MIN_A8_RECURSIVE_COMPOUNDING_WINDOW_COUNT
    )
    a8["a8_claim_ready"] = (
        not validation_failures
        and a8["decisive_outperformance"]
        and a8["conservative_comparison_report"]
        and a8["verified_recursive_compounding"]
        and a8["sustained_superhuman_coding"]
    )
    packet = {
        "spec_version": "asi_v1",
        "report_kind": "autonomy_evidence_packet",
        "created_at": datetime.now(UTC).isoformat(),
        "claim": {
            "level": "A8",
            "scope": "coding_superhuman_composite_benchmark",
            "status": "supported" if a8["a8_claim_ready"] else "not_supported",
            "summary": (
                "Composite coding benchmark evidence meets the A8 superhuman threshold."
                if a8["a8_claim_ready"]
                else "Composite coding benchmark evidence is incomplete or below the A8 threshold."
            ),
        },
        "source": {
            "benchmark_result_paths": source_paths or [],
            "benchmark_result_count": len(results),
        },
        "evidence": {"a8": a8},
        "open_limits": [
            "A8 requires real external benchmark adapters; synthetic or readiness-only packets are insufficient.",
            "The aggregate remains not_supported until every required benchmark, sustained window, and compounding gate passes.",
        ],
    }
    if validation_failures:
        packet["validation_failures"] = validation_failures
    return packet


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


def verify_a8_benchmark_target_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if packet.get("report_kind") != "a8_coding_superhuman_target_packet":
        failures.append("report_kind must be a8_coding_superhuman_target_packet")
    target = packet.get("target") if isinstance(packet.get("target"), dict) else {}
    if target.get("level") != "A8":
        failures.append("target.level must be A8")
    if target.get("domain") != "coding":
        failures.append("target.domain must be coding")
    thresholds = target.get("thresholds") if isinstance(target.get("thresholds"), dict) else {}
    expected_thresholds = _a8_target_thresholds()
    for key, expected in expected_thresholds.items():
        observed = thresholds.get(key)
        if observed != expected:
            failures.append(f"target.thresholds.{key} mismatch: observed={observed!r} expected={expected!r}")
    benchmark_sources = target.get("benchmark_sources") if isinstance(target.get("benchmark_sources"), dict) else {}
    required_sources = (
        "codeforces",
        "mle_bench",
        "swe_bench_verified",
        "swe_rebench",
        "re_bench",
    )
    for source in required_sources:
        source_payload = benchmark_sources.get(source)
        if not isinstance(source_payload, dict):
            failures.append(f"target.benchmark_sources.{source} object is required")
        elif not str(source_payload.get("evidence_metric", "")).strip():
            failures.append(f"target.benchmark_sources.{source}.evidence_metric is required")
    acceptance_policy = target.get("acceptance_policy") if isinstance(target.get("acceptance_policy"), dict) else {}
    if acceptance_policy.get("requires_sustained_windows") is not True:
        failures.append("target.acceptance_policy.requires_sustained_windows must be true")
    if acceptance_policy.get("requires_conservative_comparison") is not True:
        failures.append("target.acceptance_policy.requires_conservative_comparison must be true")
    if acceptance_policy.get("requires_recursive_compounding") is not True:
        failures.append("target.acceptance_policy.requires_recursive_compounding must be true")
    if acceptance_policy.get("forbids_readiness_only_promotion") is not True:
        failures.append("target.acceptance_policy.forbids_readiness_only_promotion must be true")
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

    if level == "A7_readiness":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        readiness = (
            evidence.get("a7_readiness")
            if isinstance(evidence.get("a7_readiness"), dict)
            else {}
        )
        if not readiness:
            failures.append("A7_readiness evidence.a7_readiness object is required")
        rotation_count = int(readiness.get("rotation_count") or 0)
        if rotation_count < MIN_A7_READINESS_ROTATIONS:
            failures.append(
                f"A7_readiness rotation_count must be at least {MIN_A7_READINESS_ROTATIONS}"
            )
        total_reports = int(readiness.get("total_held_out_frontier_reports") or 0)
        if total_reports < MIN_A7_READINESS_HELD_OUT_REPORTS:
            failures.append(
                "A7_readiness total_held_out_frontier_reports must be at least "
                f"{MIN_A7_READINESS_HELD_OUT_REPORTS}"
            )
        clean_success_rate = float(readiness.get("aggregate_clean_success_rate") or 0.0)
        if clean_success_rate < MIN_A7_READINESS_CLEAN_SUCCESS_RATE:
            failures.append(
                "A7_readiness aggregate_clean_success_rate must be at least "
                f"{MIN_A7_READINESS_CLEAN_SUCCESS_RATE:g}"
            )
        repeated_families = {
            str(family).strip()
            for family in readiness.get("families_repeated_in_both_rotations", [])
            if str(family).strip()
        }
        missing_families = [family for family in REQUIRED_A4_FAMILIES if family not in repeated_families]
        if missing_families:
            failures.append(
                "A7_readiness families_repeated_in_both_rotations missing: "
                + ",".join(missing_families)
            )
        distinct_frontier_slices = int(readiness.get("distinct_frontier_slice_count") or 0)
        if distinct_frontier_slices < MIN_A7_READINESS_DISTINCT_FRONTIER_SLICES:
            failures.append(
                "A7_readiness distinct_frontier_slice_count must be at least "
                f"{MIN_A7_READINESS_DISTINCT_FRONTIER_SLICES}"
            )
        hard_hintless_fields_present = any(
            key in readiness
            for key in (
                "hard_hintless_rotation_count",
                "hard_hintless_clean_success_count",
                "hard_hintless_clean_success_rate",
            )
        )
        if hard_hintless_fields_present:
            hard_rotations = int(readiness.get("hard_hintless_rotation_count") or 0)
            if hard_rotations < MIN_A7_READINESS_HARD_HINTLESS_ROTATIONS:
                failures.append(
                    "A7_readiness hard_hintless_rotation_count must be at least "
                    f"{MIN_A7_READINESS_HARD_HINTLESS_ROTATIONS}"
                )
            hard_clean_successes = int(readiness.get("hard_hintless_clean_success_count") or 0)
            if hard_clean_successes < MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESSES:
                failures.append(
                    "A7_readiness hard_hintless_clean_success_count must be at least "
                    f"{MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESSES}"
                )
            hard_clean_rate = float(readiness.get("hard_hintless_clean_success_rate") or 0.0)
            if hard_clean_rate < MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESS_RATE:
                failures.append(
                    "A7_readiness hard_hintless_clean_success_rate must be at least "
                    f"{MIN_A7_READINESS_HARD_HINTLESS_CLEAN_SUCCESS_RATE:g}"
                )
        stateful_repair_fields_present = any(
            key in readiness
            for key in (
                "stateful_repair_rotation_count",
                "stateful_repair_clean_success_count",
                "stateful_repair_clean_success_rate",
            )
        )
        if stateful_repair_fields_present:
            stateful_rotations = int(readiness.get("stateful_repair_rotation_count") or 0)
            if stateful_rotations < MIN_A7_READINESS_STATEFUL_REPAIR_ROTATIONS:
                failures.append(
                    "A7_readiness stateful_repair_rotation_count must be at least "
                    f"{MIN_A7_READINESS_STATEFUL_REPAIR_ROTATIONS}"
                )
            stateful_clean_successes = int(readiness.get("stateful_repair_clean_success_count") or 0)
            if stateful_clean_successes < MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESSES:
                failures.append(
                    "A7_readiness stateful_repair_clean_success_count must be at least "
                    f"{MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESSES}"
                )
            stateful_clean_rate = float(readiness.get("stateful_repair_clean_success_rate") or 0.0)
            if stateful_clean_rate < MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESS_RATE:
                failures.append(
                    "A7_readiness stateful_repair_clean_success_rate must be at least "
                    f"{MIN_A7_READINESS_STATEFUL_REPAIR_CLEAN_SUCCESS_RATE:g}"
                )
        diagnostic_synthesis_fields_present = any(
            key in readiness
            for key in (
                "diagnostic_synthesis_rotation_count",
                "diagnostic_synthesis_clean_success_count",
                "diagnostic_synthesis_clean_success_rate",
            )
        )
        if diagnostic_synthesis_fields_present:
            diagnostic_rotations = int(readiness.get("diagnostic_synthesis_rotation_count") or 0)
            if diagnostic_rotations < MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_ROTATIONS:
                failures.append(
                    "A7_readiness diagnostic_synthesis_rotation_count must be at least "
                    f"{MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_ROTATIONS}"
                )
            diagnostic_clean_successes = int(readiness.get("diagnostic_synthesis_clean_success_count") or 0)
            if diagnostic_clean_successes < MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESSES:
                failures.append(
                    "A7_readiness diagnostic_synthesis_clean_success_count must be at least "
                    f"{MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESSES}"
                )
            diagnostic_clean_rate = float(readiness.get("diagnostic_synthesis_clean_success_rate") or 0.0)
            if diagnostic_clean_rate < MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESS_RATE:
                failures.append(
                    "A7_readiness diagnostic_synthesis_clean_success_rate must be at least "
                    f"{MIN_A7_READINESS_DIAGNOSTIC_SYNTHESIS_CLEAN_SUCCESS_RATE:g}"
                )
        if readiness.get("non_identical_manifests") is not True:
            failures.append("A7_readiness non_identical_manifests must be true")
        if readiness.get("existing_path_only") is not True:
            failures.append("A7_readiness existing_path_only must be true")
        if readiness.get("per_task_architecture_changes") is not False:
            failures.append("A7_readiness per_task_architecture_changes must be false")
        provider = readiness.get("provider") if isinstance(readiness.get("provider"), dict) else {}
        provider_name = str(provider.get("provider", "")).strip().lower()
        if provider_name in {"", "mock"}:
            failures.append("A7_readiness provider.provider must be a live non-mock provider")

    if level == "A7":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        a7_full = evidence.get("a7") if isinstance(evidence.get("a7"), dict) else {}
        if not a7_full:
            failures.append("A7 evidence.a7 object is required")
        unfamiliar_domain_slices = int(a7_full.get("unfamiliar_domain_slice_count") or 0)
        if unfamiliar_domain_slices < MIN_A7_FULL_UNFAMILIAR_DOMAIN_SLICES:
            failures.append(
                "A7 unfamiliar_domain_slice_count must be at least "
                f"{MIN_A7_FULL_UNFAMILIAR_DOMAIN_SLICES}"
            )
        broad_transfer_rate = float(a7_full.get("broad_transfer_clean_success_rate") or 0.0)
        if broad_transfer_rate < MIN_A7_FULL_BROAD_TRANSFER_CLEAN_SUCCESS_RATE:
            failures.append(
                "A7 broad_transfer_clean_success_rate must be at least "
                f"{MIN_A7_FULL_BROAD_TRANSFER_CLEAN_SUCCESS_RATE:g}"
            )
        strong_baseline_win_rate = float(a7_full.get("strong_baseline_win_rate") or 0.0)
        if strong_baseline_win_rate < MIN_A7_FULL_STRONG_BASELINE_WIN_RATE:
            failures.append(
                "A7 strong_baseline_win_rate must be at least "
                f"{MIN_A7_FULL_STRONG_BASELINE_WIN_RATE:g}"
            )
        long_horizon_slices = int(a7_full.get("long_horizon_transfer_slice_count") or 0)
        if long_horizon_slices < MIN_A7_FULL_LONG_HORIZON_TRANSFER_SLICES:
            failures.append(
                "A7 long_horizon_transfer_slice_count must be at least "
                f"{MIN_A7_FULL_LONG_HORIZON_TRANSFER_SLICES}"
            )
        if a7_full.get("limited_redesign") is not True:
            failures.append("A7 limited_redesign must be true")
        if a7_full.get("conservative_comparison_report") is not True:
            failures.append("A7 conservative_comparison_report must be true")
        if a7_full.get("full_a7_claim_ready") is not True:
            failures.append("A7 full_a7_claim_ready must be true")

    if level == "A8":
        evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
        a8 = evidence.get("a8") if isinstance(evidence.get("a8"), dict) else {}
        if not a8:
            failures.append("A8 evidence.a8 object is required")
        baseline_slices = int(a8.get("strong_baseline_comparison_slice_count") or 0)
        if baseline_slices < MIN_A8_STRONG_BASELINE_COMPARISON_SLICES:
            failures.append(
                "A8 strong_baseline_comparison_slice_count must be at least "
                f"{MIN_A8_STRONG_BASELINE_COMPARISON_SLICES}"
            )
        coding_windows = int(a8.get("superhuman_coding_window_count") or 0)
        if coding_windows < MIN_A8_SUPERHUMAN_CODING_WINDOW_COUNT:
            failures.append(
                "A8 superhuman_coding_window_count must be at least "
                f"{MIN_A8_SUPERHUMAN_CODING_WINDOW_COUNT}"
            )
        coding_tasks = int(a8.get("superhuman_coding_task_count") or 0)
        if coding_tasks < MIN_A8_SUPERHUMAN_CODING_TASK_COUNT:
            failures.append(
                "A8 superhuman_coding_task_count must be at least "
                f"{MIN_A8_SUPERHUMAN_CODING_TASK_COUNT}"
            )
        baseline_win_rate = float(a8.get("strong_human_baseline_win_rate") or 0.0)
        if baseline_win_rate < MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE:
            failures.append(
                "A8 strong_human_baseline_win_rate must be at least "
                f"{MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE:g}"
            )
        baseline_win_rate_lower_bound = float(a8.get("strong_human_baseline_win_rate_lower_bound") or 0.0)
        if baseline_win_rate_lower_bound < MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE_LOWER_BOUND:
            failures.append(
                "A8 strong_human_baseline_win_rate_lower_bound must be at least "
                f"{MIN_A8_STRONG_HUMAN_BASELINE_WIN_RATE_LOWER_BOUND:g}"
            )
        unfamiliar_domain_slices = int(a8.get("unfamiliar_domain_slice_count") or 0)
        if unfamiliar_domain_slices < MIN_A8_UNFAMILIAR_DOMAIN_SLICES:
            failures.append(
                "A8 unfamiliar_domain_slice_count must be at least "
                f"{MIN_A8_UNFAMILIAR_DOMAIN_SLICES}"
            )
        long_horizon_slices = int(a8.get("long_horizon_transfer_slice_count") or 0)
        if long_horizon_slices < MIN_A8_LONG_HORIZON_TRANSFER_SLICES:
            failures.append(
                "A8 long_horizon_transfer_slice_count must be at least "
                f"{MIN_A8_LONG_HORIZON_TRANSFER_SLICES}"
            )
        retained_gain_runs = int(a8.get("recursive_compounding_retained_gain_runs") or 0)
        if retained_gain_runs < MIN_A8_RECURSIVE_COMPOUNDING_RETAINED_GAIN_RUNS:
            failures.append(
                "A8 recursive_compounding_retained_gain_runs must be at least "
                f"{MIN_A8_RECURSIVE_COMPOUNDING_RETAINED_GAIN_RUNS}"
            )
        compounding_windows = int(a8.get("recursive_compounding_window_count") or 0)
        if compounding_windows < MIN_A8_RECURSIVE_COMPOUNDING_WINDOW_COUNT:
            failures.append(
                "A8 recursive_compounding_window_count must be at least "
                f"{MIN_A8_RECURSIVE_COMPOUNDING_WINDOW_COUNT}"
            )
        regression_rate = float(a8.get("regression_rate") or 0.0)
        if regression_rate > MAX_A8_REGRESSION_RATE:
            failures.append(f"A8 regression_rate must be at most {MAX_A8_REGRESSION_RATE:g}")
        codeforces_rating = int(a8.get("codeforces_rating_equivalent") or 0)
        if codeforces_rating < MIN_A8_CODEFORCES_RATING_EQUIVALENT:
            failures.append(
                "A8 codeforces_rating_equivalent must be at least "
                f"{MIN_A8_CODEFORCES_RATING_EQUIVALENT}"
            )
        mle_gold_rate = float(a8.get("mle_bench_gold_medal_rate") or 0.0)
        if mle_gold_rate < MIN_A8_MLE_BENCH_GOLD_MEDAL_RATE:
            failures.append(
                "A8 mle_bench_gold_medal_rate must be at least "
                f"{MIN_A8_MLE_BENCH_GOLD_MEDAL_RATE:g}"
            )
        swe_verified_rate = float(a8.get("swe_bench_verified_resolve_rate") or 0.0)
        if swe_verified_rate < MIN_A8_SWE_BENCH_VERIFIED_RESOLVE_RATE:
            failures.append(
                "A8 swe_bench_verified_resolve_rate must be at least "
                f"{MIN_A8_SWE_BENCH_VERIFIED_RESOLVE_RATE:g}"
            )
        swe_rebench_rate = float(a8.get("swe_rebench_resolve_rate") or 0.0)
        if swe_rebench_rate < MIN_A8_SWE_REBENCH_RESOLVE_RATE:
            failures.append(
                "A8 swe_rebench_resolve_rate must be at least "
                f"{MIN_A8_SWE_REBENCH_RESOLVE_RATE:g}"
            )
        re_bench_win_rate = float(a8.get("re_bench_human_expert_win_rate") or 0.0)
        if re_bench_win_rate < MIN_A8_RE_BENCH_HUMAN_EXPERT_WIN_RATE:
            failures.append(
                "A8 re_bench_human_expert_win_rate must be at least "
                f"{MIN_A8_RE_BENCH_HUMAN_EXPERT_WIN_RATE:g}"
            )
        if a8.get("decisive_outperformance") is not True:
            failures.append("A8 decisive_outperformance must be true")
        if a8.get("conservative_comparison_report") is not True:
            failures.append("A8 conservative_comparison_report must be true")
        if a8.get("verified_recursive_compounding") is not True:
            failures.append("A8 verified_recursive_compounding must be true")
        if a8.get("sustained_superhuman_coding") is not True:
            failures.append("A8 sustained_superhuman_coding must be true")
        if a8.get("a8_claim_ready") is not True:
            failures.append("A8 a8_claim_ready must be true")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-report", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--verify-packet", default=None)
    parser.add_argument("--verify-static-packet", default=None)
    parser.add_argument("--verify-a8-target-packet", default=None)
    parser.add_argument("--verify-a8-benchmark-result", default=None)
    parser.add_argument("--a8-benchmark-result", action="append", default=[])
    args = parser.parse_args()
    if args.a8_benchmark_result:
        result_paths = [Path(path) for path in args.a8_benchmark_result]
        packet = build_a8_autonomy_packet_from_benchmark_results(
            [_read_json(path) for path in result_paths],
            source_paths=[str(path) for path in result_paths],
        )
        if not args.output_json:
            raise SystemExit("--output-json is required with --a8-benchmark-result")
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"claim_level={packet['claim']['level']} "
            f"claim_scope={packet['claim']['scope']} "
            f"status={packet['claim']['status']} "
            f"benchmark_result_count={packet['source']['benchmark_result_count']} "
            f"output_json={output_path}"
        )
        return
    if args.verify_a8_benchmark_result:
        verify_path = Path(args.verify_a8_benchmark_result)
        failures = verify_a8_benchmark_result_packet(_read_json(verify_path))
        if failures:
            raise SystemExit("A8 benchmark result packet verification failed: " + "; ".join(failures))
        print(f"verified_a8_benchmark_result={verify_path}")
        return
    if args.verify_a8_target_packet:
        verify_path = Path(args.verify_a8_target_packet)
        failures = verify_a8_benchmark_target_packet(_read_json(verify_path))
        if failures:
            raise SystemExit("A8 benchmark target packet verification failed: " + "; ".join(failures))
        print(f"verified_a8_target_packet={verify_path}")
        return
    if args.verify_static_packet:
        verify_path = Path(args.verify_static_packet)
        failures = verify_static_autonomy_packet(_read_json(verify_path))
        if failures:
            raise SystemExit("static autonomy evidence packet verification failed: " + "; ".join(failures))
        print(f"verified_static_packet={verify_path}")
        return
    if not args.cycle_report:
        raise SystemExit(
            "--cycle-report is required unless verification or --a8-benchmark-result mode is used"
        )
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
