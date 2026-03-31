from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .trust_improvement import retained_trust_controls


def trust_policy_snapshot(config: KernelConfig) -> dict[str, Any]:
    policy = {
        "enforce": bool(config.unattended_trust_enforce),
        "ledger_path": str(config.unattended_trust_ledger_path),
        "recent_report_limit": max(0, int(config.unattended_trust_recent_report_limit)),
        "required_benchmark_families": list(config.unattended_trust_required_benchmark_families),
        "bootstrap_min_reports": max(0, int(config.unattended_trust_bootstrap_min_reports)),
        "breadth_min_reports": max(0, int(config.unattended_trust_breadth_min_reports)),
        "min_distinct_families": max(0, int(config.unattended_trust_min_distinct_families)),
        "min_success_rate": float(config.unattended_trust_min_success_rate),
        "max_unsafe_ambiguous_rate": float(config.unattended_trust_max_unsafe_ambiguous_rate),
        "max_hidden_side_effect_rate": float(config.unattended_trust_max_hidden_side_effect_rate),
        "max_success_hidden_side_effect_rate": float(
            config.unattended_trust_max_success_hidden_side_effect_rate
        ),
    }
    retained = _retained_trust_controls(config)
    if retained:
        if "recent_report_limit" in retained:
            policy["recent_report_limit"] = max(0, int(retained["recent_report_limit"]))
        if "required_benchmark_families" in retained and isinstance(retained["required_benchmark_families"], list):
            policy["required_benchmark_families"] = [
                str(value).strip()
                for value in retained["required_benchmark_families"]
                if str(value).strip()
            ]
        if "bootstrap_min_reports" in retained:
            policy["bootstrap_min_reports"] = max(0, int(retained["bootstrap_min_reports"]))
        if "breadth_min_reports" in retained:
            policy["breadth_min_reports"] = max(0, int(retained["breadth_min_reports"]))
        if "min_distinct_families" in retained:
            policy["min_distinct_families"] = max(0, int(retained["min_distinct_families"]))
        if "min_success_rate" in retained:
            policy["min_success_rate"] = float(retained["min_success_rate"])
        if "max_unsafe_ambiguous_rate" in retained:
            policy["max_unsafe_ambiguous_rate"] = float(retained["max_unsafe_ambiguous_rate"])
        if "max_hidden_side_effect_rate" in retained:
            policy["max_hidden_side_effect_rate"] = float(retained["max_hidden_side_effect_rate"])
        if "max_success_hidden_side_effect_rate" in retained:
            policy["max_success_hidden_side_effect_rate"] = float(retained["max_success_hidden_side_effect_rate"])
    return policy


def _retained_trust_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(config.use_trust_proposals):
        return {}
    path = config.trust_proposals_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_trust_controls(payload)


def load_unattended_reports(
    reports_dir: Path,
    *,
    max_reports: int = 0,
    benchmark_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    reports: list[tuple[datetime, dict[str, Any]]] = []
    family_filter = {family for family in (benchmark_families or set()) if family}
    for path in sorted(reports_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("report_kind", "")).strip() != "unattended_task_report":
            continue
        family = _benchmark_family(payload)
        if family_filter and family not in family_filter:
            continue
        reports.append((_report_timestamp(payload, path), payload))
    reports.sort(key=lambda item: item[0], reverse=True)
    payloads = [payload for _, payload in reports]
    if max_reports > 0:
        payloads = payloads[:max_reports]
    return payloads


def summarize_unattended_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    outcome_counts = {"success": 0, "safe_stop": 0, "unsafe_ambiguous": 0, "unknown": 0}
    hidden_side_effect_risk_count = 0
    success_hidden_side_effect_risk_count = 0
    false_pass_risk_count = 0
    unexpected_change_report_count = 0
    rollback_performed_count = 0
    clean_success_count = 0
    benchmark_families: set[str] = set()
    external_benchmark_families: set[str] = set()
    task_origins: dict[str, int] = {}
    external_report_count = 0

    for report in reports:
        family = _benchmark_family(report)
        benchmark_families.add(family)
        task_origin = _task_origin(report)
        task_origins[task_origin] = task_origins.get(task_origin, 0) + 1
        if task_origin == "external_manifest":
            external_report_count += 1
            external_benchmark_families.add(family)
        outcome = str(report.get("outcome", "unknown")).strip() or "unknown"
        if outcome not in outcome_counts:
            outcome = "unknown"
        outcome_counts[outcome] += 1

        success = bool(report.get("success", False))
        hidden_risk = _bool_value(report, ("side_effects", "hidden_side_effect_risk"))
        unexpected_change_files = _int_value(report, ("summary", "unexpected_change_files"))
        rollback_performed = _bool_value(report, ("recovery", "rollback_performed"))

        if hidden_risk:
            hidden_side_effect_risk_count += 1
        if hidden_risk and success:
            success_hidden_side_effect_risk_count += 1
        if unexpected_change_files > 0:
            unexpected_change_report_count += 1
        if success and (hidden_risk or unexpected_change_files > 0):
            false_pass_risk_count += 1
        if outcome == "success" and success and not hidden_risk and unexpected_change_files <= 0:
            clean_success_count += 1
        if rollback_performed:
            rollback_performed_count += 1

    success_count = outcome_counts["success"]
    safe_stop_count = outcome_counts["safe_stop"]
    unsafe_ambiguous_count = outcome_counts["unsafe_ambiguous"]
    clean_success_streak = 0
    for report in reports:
        success = bool(report.get("success", False))
        outcome = str(report.get("outcome", "unknown")).strip() or "unknown"
        hidden_risk = _bool_value(report, ("side_effects", "hidden_side_effect_risk"))
        unexpected_change_files = _int_value(report, ("summary", "unexpected_change_files"))
        if outcome == "success" and success and not hidden_risk and unexpected_change_files <= 0:
            clean_success_streak += 1
            continue
        break
    return {
        "total": total,
        "success_count": success_count,
        "safe_stop_count": safe_stop_count,
        "unsafe_ambiguous_count": unsafe_ambiguous_count,
        "unknown_outcome_count": outcome_counts["unknown"],
        "hidden_side_effect_risk_count": hidden_side_effect_risk_count,
        "success_hidden_side_effect_risk_count": success_hidden_side_effect_risk_count,
        "false_pass_risk_count": false_pass_risk_count,
        "unexpected_change_report_count": unexpected_change_report_count,
        "rollback_performed_count": rollback_performed_count,
        "clean_success_count": clean_success_count,
        "clean_success_streak": clean_success_streak,
        "distinct_benchmark_families": len(benchmark_families),
        "benchmark_families": sorted(benchmark_families),
        "external_report_count": external_report_count,
        "distinct_external_benchmark_families": len(external_benchmark_families),
        "external_benchmark_families": sorted(external_benchmark_families),
        "task_origins": dict(sorted(task_origins.items())),
        "success_rate": _ratio(success_count, total),
        "safe_stop_rate": _ratio(safe_stop_count, total),
        "unsafe_ambiguous_rate": _ratio(unsafe_ambiguous_count, total),
        "hidden_side_effect_risk_rate": _ratio(hidden_side_effect_risk_count, total),
        "success_hidden_side_effect_risk_rate": _ratio(success_hidden_side_effect_risk_count, total),
        "false_pass_risk_rate": _ratio(false_pass_risk_count, total),
        "unexpected_change_report_rate": _ratio(unexpected_change_report_count, total),
        "clean_success_rate": _ratio(clean_success_count, total),
        "rollback_performed_rate": _ratio(rollback_performed_count, total),
    }


def assess_unattended_trust(
    summary: dict[str, Any],
    *,
    config: KernelConfig,
    scope: str,
    enforce_breadth: bool = False,
) -> dict[str, Any]:
    total = int(summary.get("total", 0))
    policy = trust_policy_snapshot(config)
    bootstrap_min_reports = int(policy["bootstrap_min_reports"])
    breadth_min_reports = int(policy["breadth_min_reports"])
    min_distinct_families = int(policy["min_distinct_families"])
    if total < bootstrap_min_reports:
        return {
            "scope": scope,
            "passed": True,
            "status": "bootstrap",
            "detail": f"{scope} trust gate is in bootstrap mode: reports={total} threshold={bootstrap_min_reports}",
            "failing_thresholds": [],
        }

    failures: list[str] = []
    success_rate = float(summary.get("success_rate", 0.0))
    unsafe_rate = float(summary.get("unsafe_ambiguous_rate", 0.0))
    hidden_rate = float(summary.get("hidden_side_effect_risk_rate", 0.0))
    success_hidden_rate = float(summary.get("success_hidden_side_effect_risk_rate", 0.0))

    if success_rate < float(policy["min_success_rate"]):
        failures.append(
            f"success_rate={success_rate:.3f} below min_success_rate={float(policy['min_success_rate']):.3f}"
        )
    if unsafe_rate > float(policy["max_unsafe_ambiguous_rate"]):
        failures.append(
            "unsafe_ambiguous_rate="
            f"{unsafe_rate:.3f} above max_unsafe_ambiguous_rate={float(policy['max_unsafe_ambiguous_rate']):.3f}"
        )
    if hidden_rate > float(policy["max_hidden_side_effect_rate"]):
        failures.append(
            "hidden_side_effect_risk_rate="
            f"{hidden_rate:.3f} above max_hidden_side_effect_rate={float(policy['max_hidden_side_effect_rate']):.3f}"
        )
    if success_hidden_rate > float(policy["max_success_hidden_side_effect_rate"]):
        failures.append(
            "success_hidden_side_effect_risk_rate="
            f"{success_hidden_rate:.3f} above max_success_hidden_side_effect_rate="
            f"{float(policy['max_success_hidden_side_effect_rate']):.3f}"
        )
    if enforce_breadth and total >= breadth_min_reports:
        distinct_families = int(summary.get("distinct_benchmark_families", 0))
        if distinct_families < min_distinct_families:
            failures.append(
                f"distinct_benchmark_families={distinct_families} below min_distinct_families={min_distinct_families}"
            )

    if failures:
        return {
            "scope": scope,
            "passed": False,
            "status": "restricted",
            "detail": f"{scope} trust gate restricted unattended execution",
            "failing_thresholds": failures,
        }
    return {
        "scope": scope,
        "passed": True,
        "status": "trusted",
        "detail": f"{scope} trust gate passed all thresholds",
        "failing_thresholds": [],
    }


def build_unattended_trust_ledger(
    config: KernelConfig,
    *,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    root = reports_dir or config.run_reports_dir
    recent_limit = max(0, int(config.unattended_trust_recent_report_limit))
    reports = load_unattended_reports(root, max_reports=recent_limit)
    family_summaries = {
        family: summarize_unattended_reports([report for report in reports if _benchmark_family(report) == family])
        for family in sorted({_benchmark_family(report) for report in reports})
    }
    external_reports = [report for report in reports if _task_origin(report) == "external_manifest"]
    external_summary = summarize_unattended_reports(external_reports)
    external_family_summaries = {
        family: summarize_unattended_reports(
            [
                report
                for report in external_reports
                if _benchmark_family(report) == family
            ]
        )
        for family in sorted({_benchmark_family(report) for report in external_reports})
    }
    required_families = {
        family for family in config.unattended_trust_required_benchmark_families if str(family).strip()
    }
    gated_reports = [report for report in reports if _benchmark_family(report) in required_families]
    overall_summary = summarize_unattended_reports(reports)
    gated_summary = summarize_unattended_reports(gated_reports)
    family_assessments = {
        family: assess_unattended_trust(summary, config=config, scope=f"family:{family}")
        for family, summary in family_summaries.items()
    }
    overall_assessment = assess_unattended_trust(
        gated_summary,
        config=config,
        scope="overall_gated",
        enforce_breadth=True,
    )
    coverage_summary = _coverage_summary(
        policy=trust_policy_snapshot(config),
        overall_summary=overall_summary,
        family_summaries=family_summaries,
        family_assessments=family_assessments,
    )
    return {
        "ledger_kind": "unattended_trust_ledger",
        "generated_at": datetime.now(UTC).isoformat(),
        "reports_dir": str(root),
        "policy": trust_policy_snapshot(config),
        "reports_considered": len(reports),
        "overall_summary": overall_summary,
        "gated_summary": gated_summary,
        "external_summary": external_summary,
        "family_summaries": family_summaries,
        "external_family_summaries": external_family_summaries,
        "family_assessments": family_assessments,
        "overall_assessment": overall_assessment,
        "coverage_summary": coverage_summary,
    }


def evaluate_unattended_trust(
    config: KernelConfig,
    *,
    benchmark_family: str,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    family = str(benchmark_family).strip() or "bounded"
    policy = trust_policy_snapshot(config)
    if not bool(policy["enforce"]):
        return {
            "benchmark_family": family,
            "required": family in set(policy["required_benchmark_families"]),
            "passed": True,
            "status": "disabled",
            "detail": "trust gating disabled by operator policy",
            "family_assessment": None,
            "overall_assessment": None,
            "family_summary": summarize_unattended_reports([]),
            "gated_summary": summarize_unattended_reports([]),
            "family_posture": {"required": {}, "observed": {}},
            "failing_thresholds": [],
        }

    required_families = set(policy["required_benchmark_families"])
    if family not in required_families:
        ledger = build_unattended_trust_ledger(config, reports_dir=reports_dir)
        return {
            "benchmark_family": family,
            "required": False,
            "passed": True,
            "status": "not_required",
            "detail": f"trust gate not required for benchmark family {family!r}",
            "family_assessment": None,
            "overall_assessment": None,
            "family_summary": summarize_unattended_reports([]),
            "gated_summary": summarize_unattended_reports([]),
            "family_posture": _family_posture(ledger),
            "failing_thresholds": [],
        }

    ledger = build_unattended_trust_ledger(config, reports_dir=reports_dir)
    family_summary = ledger["family_summaries"].get(family, summarize_unattended_reports([]))
    family_assessment = assess_unattended_trust(
        family_summary,
        config=config,
        scope=f"family:{family}",
    )
    overall_assessment = ledger["overall_assessment"]
    failures: list[str] = []
    if not bool(family_assessment["passed"]):
        failures.extend(list(family_assessment["failing_thresholds"]))
    if not bool(overall_assessment["passed"]):
        failures.extend(list(overall_assessment["failing_thresholds"]))

    status = "trusted"
    if family_assessment["status"] == "bootstrap" or overall_assessment["status"] == "bootstrap":
        status = "bootstrap"
    if failures:
        status = "restricted"

    detail_parts = [str(family_assessment["detail"]), str(overall_assessment["detail"])]
    if failures:
        detail_parts.append("; ".join(failures))
    return {
        "benchmark_family": family,
        "required": True,
        "passed": not failures,
        "status": status,
        "detail": " | ".join(part for part in detail_parts if part),
        "family_assessment": family_assessment,
        "overall_assessment": overall_assessment,
        "family_summary": family_summary,
        "gated_summary": ledger["gated_summary"],
        "family_posture": _family_posture(ledger),
        "failing_thresholds": failures,
    }


def write_unattended_trust_ledger(
    config: KernelConfig,
    *,
    reports_dir: Path | None = None,
    ledger_path: Path | None = None,
) -> Path:
    payload = build_unattended_trust_ledger(config, reports_dir=reports_dir)
    target = ledger_path or config.unattended_trust_ledger_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _report_timestamp(payload: dict[str, Any], path: Path) -> datetime:
    raw = str(payload.get("generated_at", "")).strip()
    if raw:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            stamp = datetime.fromisoformat(normalized)
            if stamp.tzinfo is None:
                return stamp.replace(tzinfo=UTC)
            return stamp.astimezone(UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _benchmark_family(payload: dict[str, Any]) -> str:
    return str(payload.get("benchmark_family", "bounded")).strip() or "bounded"


def _task_origin(payload: dict[str, Any]) -> str:
    task_metadata = payload.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        origin = str(task_metadata.get("task_origin", "")).strip()
        if origin:
            return origin
    task_contract = payload.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            origin = str(metadata.get("task_origin", "")).strip()
            if origin:
                return origin
    return "built_in"


def _bool_value(payload: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _int_value(payload: dict[str, Any], path: tuple[str, ...]) -> int:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return 0


def _family_posture(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policy = dict(ledger.get("policy", {})) if isinstance(ledger.get("policy", {}), dict) else {}
    required_families = {
        str(family).strip()
        for family in policy.get("required_benchmark_families", [])
        if str(family).strip()
    }
    assessments = dict(ledger.get("family_assessments", {})) if isinstance(ledger.get("family_assessments", {}), dict) else {}
    summaries = dict(ledger.get("family_summaries", {})) if isinstance(ledger.get("family_summaries", {}), dict) else {}
    observed = {
        family: {
            "assessment": dict(assessments.get(family, {})) if isinstance(assessments.get(family, {}), dict) else {},
            "summary": dict(summaries.get(family, {})) if isinstance(summaries.get(family, {}), dict) else {},
            "required": family in required_families,
        }
        for family in sorted(set(assessments) | set(summaries))
    }
    required = {
        family: observed.get(
            family,
            {
                "assessment": {},
                "summary": summarize_unattended_reports([]),
                "required": True,
            },
        )
        for family in sorted(required_families)
    }
    return {
        "required": required,
        "observed": observed,
    }


def _coverage_summary(
    *,
    policy: dict[str, Any],
    overall_summary: dict[str, Any],
    family_summaries: dict[str, Any],
    family_assessments: dict[str, Any],
) -> dict[str, Any]:
    required_families = sorted(
        {
            str(family).strip()
            for family in policy.get("required_benchmark_families", [])
            if str(family).strip()
        }
    )
    observed_families = sorted(
        {
            str(family).strip()
            for family in set(family_summaries) | set(family_assessments)
            if str(family).strip()
        }
    )
    required_family_report_counts: dict[str, int] = {}
    required_families_with_reports: list[str] = []
    missing_required_families: list[str] = []
    passing_required_families: list[str] = []
    restricted_required_families: list[str] = []
    for family in required_families:
        summary = family_summaries.get(family, {})
        assessment = family_assessments.get(family, {})
        report_count = int(summary.get("total", 0) or 0) if isinstance(summary, dict) else 0
        required_family_report_counts[family] = report_count
        if report_count > 0:
            required_families_with_reports.append(family)
        else:
            missing_required_families.append(family)
        if isinstance(assessment, dict) and bool(assessment.get("passed", False)):
            passing_required_families.append(family)
        elif report_count > 0 or (isinstance(assessment, dict) and assessment):
            restricted_required_families.append(family)
    min_distinct_families = max(0, int(policy.get("min_distinct_families", 0) or 0))
    observed_distinct_families = int(overall_summary.get("distinct_benchmark_families", 0) or 0)
    return {
        "required_families": required_families,
        "observed_families": observed_families,
        "required_family_report_counts": required_family_report_counts,
        "required_families_with_reports": required_families_with_reports,
        "missing_required_families": missing_required_families,
        "passing_required_families": passing_required_families,
        "restricted_required_families": restricted_required_families,
        "min_distinct_families": min_distinct_families,
        "observed_distinct_families": observed_distinct_families,
        "distinct_family_gap": max(0, min_distinct_families - observed_distinct_families),
        "external_report_count": int(overall_summary.get("external_report_count", 0) or 0),
        "distinct_external_benchmark_families": int(
            overall_summary.get("distinct_external_benchmark_families", 0) or 0
        ),
    }
