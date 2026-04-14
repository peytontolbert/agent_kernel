from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from ..config import KernelConfig
from .improvement.trust_improvement import retained_trust_controls

_FAMILY_BREADTH_MIN_DISTINCT_TASK_ROOTS = 2
_REPLAY_DERIVED_TASK_ORIGINS = frozenset(
    {
        "episode_replay",
        "skill_replay",
        "skill_transfer",
        "operator_replay",
        "tool_replay",
        "verifier_replay",
        "discovered_task",
        "transition_pressure",
        "benchmark_candidate",
        "verifier_candidate",
    }
)


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
        "family_breadth_min_distinct_task_roots": _FAMILY_BREADTH_MIN_DISTINCT_TASK_ROOTS,
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


def _required_family_counted_evidence_summary(
    required_families: list[str],
    *,
    sampled_progress_counts: dict[str, int] | None = None,
    runtime_managed_signal_counts: dict[str, int] | None = None,
    runtime_managed_retained_decision_counts: dict[str, int] | None = None,
    runtime_managed_decision_yield_counts: dict[str, int] | None = None,
    clean_task_root_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, object]]:
    sampled_progress_counts = sampled_progress_counts or {}
    runtime_managed_signal_counts = runtime_managed_signal_counts or {}
    runtime_managed_retained_decision_counts = runtime_managed_retained_decision_counts or {}
    runtime_managed_decision_yield_counts = runtime_managed_decision_yield_counts or {}
    clean_task_root_counts = clean_task_root_counts or {}
    summary: dict[str, dict[str, object]] = {}
    for family in sorted({str(value).strip() for value in required_families if str(value).strip()}):
        sampled_progress_count = max(0, int(sampled_progress_counts.get(family, 0) or 0))
        verified_signal_count = max(0, int(runtime_managed_signal_counts.get(family, 0) or 0))
        retained_decision_count = max(0, int(runtime_managed_retained_decision_counts.get(family, 0) or 0))
        decision_yield_count = max(0, int(runtime_managed_decision_yield_counts.get(family, 0) or 0))
        clean_task_root_count = max(0, int(clean_task_root_counts.get(family, 0) or 0))
        highest_confirmed_stage = "none"
        if sampled_progress_count > 0:
            highest_confirmed_stage = "sampled"
        if verified_signal_count > 0:
            highest_confirmed_stage = "verified"
        if retained_decision_count > 0:
            highest_confirmed_stage = "retained"
        if decision_yield_count > 0:
            highest_confirmed_stage = "yielded"
        if clean_task_root_count > 0:
            highest_confirmed_stage = "clean_root"
        summary[family] = {
            "sampled_progress_count": sampled_progress_count,
            "verified_signal_count": verified_signal_count,
            "retained_decision_count": retained_decision_count,
            "decision_yield_count": decision_yield_count,
            "clean_task_root_count": clean_task_root_count,
            "highest_confirmed_stage": highest_confirmed_stage,
            "missing_decision_yield_after_sampling": (
                sampled_progress_count > 0 and decision_yield_count <= 0
            ),
        }
    return summary


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


def load_improvement_campaign_reports(
    reports_dir: Path,
    *,
    max_reports: int = 1,
) -> list[dict[str, Any]]:
    reports: list[tuple[datetime, dict[str, Any]]] = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("report_kind", "")).strip() != "improvement_campaign_report":
            continue
        reports.append((_report_timestamp(payload, path), payload))
    reports.sort(key=lambda item: item[0], reverse=True)
    payloads = [payload for _, payload in reports]
    if max_reports > 0:
        payloads = payloads[:max_reports]
    return payloads


def summarize_improvement_campaign_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    required_families: set[str] = set()
    required_families_with_reports: set[str] = set()
    sampled_families_from_progress: set[str] = set()
    sampled_progress_counts: dict[str, int] = {}
    priority_families: set[str] = set()
    runtime_managed_signal_counts: dict[str, int] = {}
    runtime_managed_retained_decision_counts: dict[str, int] = {}
    runtime_managed_decision_yield_counts: dict[str, int] = {}
    required_family_clean_task_root_counts: dict[str, int] = {}
    runtime_managed_decisions = 0
    retained_cycles = 0
    rejected_cycles = 0
    for report in reports:
        trust_breadth_summary = report.get("trust_breadth_summary", {})
        if isinstance(trust_breadth_summary, dict):
            required_families.update(
                str(value).strip()
                for value in list(trust_breadth_summary.get("required_families", []) or [])
                if str(value).strip()
            )
            required_families_with_reports.update(
                str(value).strip()
                for value in list(trust_breadth_summary.get("required_families_with_reports", []) or [])
                if str(value).strip()
            )
            clean_task_root_counts = trust_breadth_summary.get("required_family_clean_task_root_counts", {})
            if isinstance(clean_task_root_counts, dict):
                for family, count in clean_task_root_counts.items():
                    normalized_family = str(family).strip()
                    if not normalized_family:
                        continue
                    required_family_clean_task_root_counts[normalized_family] = max(
                        required_family_clean_task_root_counts.get(normalized_family, 0),
                        max(0, int(count or 0)),
                    )
        priority_family_yield_summary = report.get("priority_family_yield_summary", {})
        if isinstance(priority_family_yield_summary, dict):
            priority_families.update(
                str(value).strip()
                for value in list(priority_family_yield_summary.get("priority_families", []) or [])
                if str(value).strip()
            )
            family_summaries = priority_family_yield_summary.get("family_summaries", {})
            if isinstance(family_summaries, dict):
                for family, summary in family_summaries.items():
                    normalized_family = str(family).strip()
                    if not normalized_family or not isinstance(summary, dict):
                        continue
                    observed_decisions = max(0, _int_value(summary, ("observed_decisions",)))
                    retained_decisions = max(0, _int_value(summary, ("retained_decisions",)))
                    retained_positive_delta_decisions = max(
                        0,
                        _int_value(summary, ("retained_positive_delta_decisions",)),
                    )
                    runtime_managed_signal_counts[normalized_family] = (
                        runtime_managed_signal_counts.get(normalized_family, 0) + observed_decisions
                    )
                    runtime_managed_retained_decision_counts[normalized_family] = (
                        runtime_managed_retained_decision_counts.get(normalized_family, 0) + retained_decisions
                    )
                    runtime_managed_decision_yield_counts[normalized_family] = (
                        runtime_managed_decision_yield_counts.get(normalized_family, 0)
                        + retained_positive_delta_decisions
                    )
            else:
                for family in list(priority_family_yield_summary.get("priority_families_with_signal", []) or []):
                    normalized_family = str(family).strip()
                    if not normalized_family:
                        continue
                    runtime_managed_signal_counts[normalized_family] = (
                        runtime_managed_signal_counts.get(normalized_family, 0) + 1
                    )
        decision_stream_summary = report.get("decision_stream_summary", {})
        if isinstance(decision_stream_summary, dict):
            runtime_managed = decision_stream_summary.get("runtime_managed", {})
            if isinstance(runtime_managed, dict):
                runtime_managed_decisions += max(0, _int_value(runtime_managed, ("total_decisions",)))
                retained_cycles += max(0, _int_value(runtime_managed, ("retained_cycles",)))
                rejected_cycles += max(0, _int_value(runtime_managed, ("rejected_cycles",)))
        sampled_in_report: set[str] = set()
        partial_progress_summary = report.get("partial_progress_summary", {})
        if isinstance(partial_progress_summary, dict):
            sampled_in_report.update(
                str(value).strip()
                for value in list(partial_progress_summary.get("sampled_families_from_progress", []) or [])
                if str(value).strip()
            )
        for run in list(report.get("runs", []) or []):
            if not isinstance(run, dict):
                continue
            partial_progress = run.get("partial_progress", {})
            if isinstance(partial_progress, dict):
                sampled_in_report.update(
                    str(value).strip()
                    for value in list(partial_progress.get("sampled_families_from_progress", []) or [])
                    if str(value).strip()
                )
            rerouting = run.get("priority_family_rerouting", {})
            if isinstance(rerouting, dict):
                priority_families.update(
                    str(value).strip()
                    for value in list(rerouting.get("priority_benchmark_families", []) or [])
                    if str(value).strip()
                )
        sampled_families_from_progress.update(sampled_in_report)
        for family in sampled_in_report:
            sampled_progress_counts[family] = sampled_progress_counts.get(family, 0) + 1
    signal_families = sorted(
        family
        for family, count in runtime_managed_signal_counts.items()
        if family in required_families and max(0, int(count)) > 0
    )
    retained_decision_families = sorted(
        family
        for family, count in runtime_managed_retained_decision_counts.items()
        if family in required_families and max(0, int(count)) > 0
    )
    decision_yield_families = sorted(
        family
        for family, count in runtime_managed_decision_yield_counts.items()
        if family in required_families and max(0, int(count)) > 0
    )
    missing_signal_families = sorted(required_families - set(signal_families))
    missing_decision_yield_families = sorted(required_families - set(decision_yield_families))
    sampled_progress_families = sorted(
        family
        for family, count in sampled_progress_counts.items()
        if family in required_families and max(0, int(count)) > 0
    )
    sampled_progress_without_decision_yield_families = sorted(
        family for family in sampled_progress_families if family not in set(decision_yield_families)
    )
    missing_sampled_progress_families = sorted(required_families - set(sampled_progress_families))
    signal_counts = {
        family: max(0, int(runtime_managed_signal_counts.get(family, 0)))
        for family in sorted(required_families)
    }
    sampled_progress_credit_counts = {
        family: max(0, int(sampled_progress_counts.get(family, 0)))
        for family in sorted(required_families)
    }
    retained_decision_counts = {
        family: max(0, int(runtime_managed_retained_decision_counts.get(family, 0)))
        for family in sorted(required_families)
    }
    decision_yield_counts = {
        family: max(0, int(runtime_managed_decision_yield_counts.get(family, 0)))
        for family in sorted(required_families)
    }
    return {
        "reports_considered": total,
        "required_families": sorted(required_families),
        "required_families_with_reports": sorted(required_families_with_reports),
        "sampled_families_from_progress": sorted(sampled_families_from_progress),
        "required_families_with_sampled_progress": sampled_progress_families,
        "required_families_missing_sampled_progress": missing_sampled_progress_families,
        "required_family_sampled_progress_counts": sampled_progress_credit_counts,
        "required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield": (
            sampled_progress_without_decision_yield_families
        ),
        "required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts": {
            family: sampled_progress_credit_counts.get(family, 0)
            for family in sorted(required_families)
        },
        "priority_families": sorted(priority_families),
        "runtime_managed_decisions": runtime_managed_decisions,
        "retained_cycles": retained_cycles,
        "rejected_cycles": rejected_cycles,
        "runtime_managed_breadth_signal_families": signal_families,
        "runtime_managed_retained_decision_families": retained_decision_families,
        "runtime_managed_decision_yield_families": decision_yield_families,
        "required_families_missing_runtime_managed_signal": missing_signal_families,
        "required_families_missing_runtime_managed_decision_yield": missing_decision_yield_families,
        "required_family_runtime_managed_signal_counts": signal_counts,
        "required_family_runtime_managed_retained_decision_counts": retained_decision_counts,
        "required_family_runtime_managed_decision_yield_counts": decision_yield_counts,
        "required_family_clean_task_root_counts": {
            family: max(0, int(required_family_clean_task_root_counts.get(family, 0)))
            for family in sorted(required_families)
        },
        "required_family_counted_evidence_summary": _required_family_counted_evidence_summary(
            sorted(required_families),
            sampled_progress_counts=sampled_progress_credit_counts,
            runtime_managed_signal_counts=signal_counts,
            runtime_managed_retained_decision_counts=retained_decision_counts,
            runtime_managed_decision_yield_counts=decision_yield_counts,
            clean_task_root_counts=required_family_clean_task_root_counts,
        ),
    }


def summarize_unattended_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    outcome_counts = {"success": 0, "safe_stop": 0, "unsafe_ambiguous": 0, "unknown": 0}
    hidden_side_effect_risk_count = 0
    success_hidden_side_effect_risk_count = 0
    false_pass_risk_count = 0
    unexpected_change_report_count = 0
    rollback_performed_count = 0
    clean_success_count = 0
    independent_execution_count = 0
    light_supervision_candidate_count = 0
    light_supervision_success_count = 0
    light_supervision_clean_success_count = 0
    contract_clean_failure_recovery_candidate_count = 0
    contract_clean_failure_recovery_success_count = 0
    contract_clean_failure_recovery_clean_success_count = 0
    failure_recovery_report_count = 0
    failure_recovery_success_count = 0
    failure_recovery_clean_success_count = 0
    benchmark_families: set[str] = set()
    external_benchmark_families: set[str] = set()
    task_roots: set[str] = set()
    success_task_roots: set[str] = set()
    clean_success_task_roots: set[str] = set()
    failure_recovery_task_roots: set[str] = set()
    failure_recovery_clean_success_task_roots: set[str] = set()
    failure_recovery_benchmark_families: set[str] = set()
    task_origins: dict[str, int] = {}
    task_yield_buckets: dict[str, dict[str, object]] = {}
    supervision_modes: dict[str, int] = {}
    repo_semantic_cluster_counts: dict[str, int] = {}
    external_report_count = 0

    for report in reports:
        family = _benchmark_family(report)
        task_root = _task_root(report)
        benchmark_families.add(family)
        if task_root:
            task_roots.add(task_root)
        task_origin = _task_origin(report)
        task_origins[task_origin] = task_origins.get(task_origin, 0) + 1
        failure_recovery_report = _is_failure_recovery_report(report)
        if failure_recovery_report:
            failure_recovery_report_count += 1
            failure_recovery_benchmark_families.add(family)
            if task_root:
                failure_recovery_task_roots.add(task_root)
        task_yield_bucket = task_yield_buckets.setdefault(
            _task_yield_bucket(task_origin),
            {
                "reports": 0,
                "success_count": 0,
                "clean_success_count": 0,
                "benchmark_families": set(),
                "task_roots": set(),
                "clean_success_task_roots": set(),
                "task_origins": {},
            },
        )
        task_yield_bucket["reports"] = int(task_yield_bucket.get("reports", 0) or 0) + 1
        cast_families = task_yield_bucket.setdefault("benchmark_families", set())
        if isinstance(cast_families, set):
            cast_families.add(family)
        cast_task_roots = task_yield_bucket.setdefault("task_roots", set())
        if isinstance(cast_task_roots, set) and task_root:
            cast_task_roots.add(task_root)
        cast_task_origins = task_yield_bucket.setdefault("task_origins", {})
        if isinstance(cast_task_origins, dict):
            cast_task_origins[task_origin] = int(cast_task_origins.get(task_origin, 0) or 0) + 1
        for cluster in _repo_semantic_clusters(report):
            repo_semantic_cluster_counts[cluster] = repo_semantic_cluster_counts.get(cluster, 0) + 1
        supervision_mode = _supervision_mode(report)
        supervision_modes[supervision_mode] = supervision_modes.get(supervision_mode, 0) + 1
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
        if success and task_root:
            success_task_roots.add(task_root)
            task_yield_bucket["success_count"] = int(task_yield_bucket.get("success_count", 0) or 0) + 1
            if failure_recovery_report:
                failure_recovery_success_count += 1
        if _bool_value(report, ("supervision", "independent_execution")):
            independent_execution_count += 1
        if _bool_value(report, ("supervision", "light_supervision_candidate")):
            light_supervision_candidate_count += 1
        if _bool_value(report, ("supervision", "light_supervision_success")):
            light_supervision_success_count += 1
        if _bool_value(report, ("supervision", "light_supervision_clean_success")):
            light_supervision_clean_success_count += 1
        if _bool_value(report, ("supervision", "contract_clean_failure_recovery_candidate")):
            contract_clean_failure_recovery_candidate_count += 1
        if _bool_value(report, ("supervision", "contract_clean_failure_recovery_success")):
            contract_clean_failure_recovery_success_count += 1
        if _bool_value(report, ("supervision", "contract_clean_failure_recovery_clean_success")):
            contract_clean_failure_recovery_clean_success_count += 1
        if outcome == "success" and success and not hidden_risk and unexpected_change_files <= 0:
            clean_success_count += 1
            if task_root:
                clean_success_task_roots.add(task_root)
                cast_clean_roots = task_yield_bucket.setdefault("clean_success_task_roots", set())
                if isinstance(cast_clean_roots, set):
                    cast_clean_roots.add(task_root)
                if failure_recovery_report:
                    failure_recovery_clean_success_task_roots.add(task_root)
            task_yield_bucket["clean_success_count"] = int(task_yield_bucket.get("clean_success_count", 0) or 0) + 1
            if failure_recovery_report:
                failure_recovery_clean_success_count += 1
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
    success_rate_confidence_interval = _wilson_confidence_interval(success_count, total)
    unsafe_ambiguous_rate_confidence_interval = _wilson_confidence_interval(unsafe_ambiguous_count, total)
    hidden_side_effect_risk_rate_confidence_interval = _wilson_confidence_interval(hidden_side_effect_risk_count, total)
    clean_success_rate_confidence_interval = _wilson_confidence_interval(clean_success_count, total)
    failure_recovery_success_rate_confidence_interval = _wilson_confidence_interval(
        failure_recovery_success_count,
        failure_recovery_report_count,
    )
    failure_recovery_clean_success_rate_confidence_interval = _wilson_confidence_interval(
        failure_recovery_clean_success_count,
        failure_recovery_report_count,
    )
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
        "distinct_task_roots": len(task_roots),
        "task_roots": sorted(task_roots),
        "distinct_success_task_roots": len(success_task_roots),
        "success_task_roots": sorted(success_task_roots),
        "distinct_clean_success_task_roots": len(clean_success_task_roots),
        "clean_success_task_roots": sorted(clean_success_task_roots),
        "external_report_count": external_report_count,
        "distinct_external_benchmark_families": len(external_benchmark_families),
        "external_benchmark_families": sorted(external_benchmark_families),
        "distinct_repo_semantic_clusters": len(repo_semantic_cluster_counts),
        "repo_semantic_clusters": sorted(repo_semantic_cluster_counts),
        "repo_semantic_cluster_counts": dict(sorted(repo_semantic_cluster_counts.items())),
        "task_origins": dict(sorted(task_origins.items())),
        "task_yield_bucket_summary": {
            bucket: {
                "reports": int(summary.get("reports", 0) or 0),
                "success_count": int(summary.get("success_count", 0) or 0),
                "clean_success_count": int(summary.get("clean_success_count", 0) or 0),
                "benchmark_families": sorted(
                    str(value).strip()
                    for value in summary.get("benchmark_families", set())
                    if str(value).strip()
                )
                if isinstance(summary.get("benchmark_families", set()), set)
                else [],
                "distinct_benchmark_families": len(summary.get("benchmark_families", set()))
                if isinstance(summary.get("benchmark_families", set()), set)
                else 0,
                "task_roots": sorted(
                    str(value).strip()
                    for value in summary.get("task_roots", set())
                    if str(value).strip()
                )
                if isinstance(summary.get("task_roots", set()), set)
                else [],
                "distinct_task_roots": len(summary.get("task_roots", set()))
                if isinstance(summary.get("task_roots", set()), set)
                else 0,
                "clean_success_task_roots": sorted(
                    str(value).strip()
                    for value in summary.get("clean_success_task_roots", set())
                    if str(value).strip()
                )
                if isinstance(summary.get("clean_success_task_roots", set()), set)
                else [],
                "distinct_clean_success_task_roots": len(summary.get("clean_success_task_roots", set()))
                if isinstance(summary.get("clean_success_task_roots", set()), set)
                else 0,
                "task_origins": dict(sorted(summary.get("task_origins", {}).items()))
                if isinstance(summary.get("task_origins", {}), dict)
                else {},
            }
            for bucket, summary in sorted(task_yield_buckets.items())
        },
        "supervision_modes": dict(sorted(supervision_modes.items())),
        "independent_execution_count": independent_execution_count,
        "light_supervision_candidate_count": light_supervision_candidate_count,
        "light_supervision_success_count": light_supervision_success_count,
        "light_supervision_clean_success_count": light_supervision_clean_success_count,
        "contract_clean_failure_recovery_candidate_count": contract_clean_failure_recovery_candidate_count,
        "contract_clean_failure_recovery_success_count": contract_clean_failure_recovery_success_count,
        "contract_clean_failure_recovery_clean_success_count": contract_clean_failure_recovery_clean_success_count,
        "success_rate": _ratio(success_count, total),
        "success_rate_confidence_interval": success_rate_confidence_interval,
        "safe_stop_rate": _ratio(safe_stop_count, total),
        "unsafe_ambiguous_rate": _ratio(unsafe_ambiguous_count, total),
        "unsafe_ambiguous_rate_confidence_interval": unsafe_ambiguous_rate_confidence_interval,
        "hidden_side_effect_risk_rate": _ratio(hidden_side_effect_risk_count, total),
        "hidden_side_effect_risk_rate_confidence_interval": hidden_side_effect_risk_rate_confidence_interval,
        "success_hidden_side_effect_risk_rate": _ratio(success_hidden_side_effect_risk_count, total),
        "false_pass_risk_rate": _ratio(false_pass_risk_count, total),
        "unexpected_change_report_rate": _ratio(unexpected_change_report_count, total),
        "clean_success_rate": _ratio(clean_success_count, total),
        "clean_success_rate_confidence_interval": clean_success_rate_confidence_interval,
        "rollback_performed_rate": _ratio(rollback_performed_count, total),
        "independent_execution_rate": _ratio(independent_execution_count, total),
        "light_supervision_candidate_rate": _ratio(light_supervision_candidate_count, total),
        "light_supervision_success_rate": _ratio(light_supervision_success_count, total),
        "light_supervision_clean_success_rate": _ratio(light_supervision_clean_success_count, total),
        "contract_clean_failure_recovery_candidate_rate": _ratio(
            contract_clean_failure_recovery_candidate_count,
            total,
        ),
        "contract_clean_failure_recovery_success_rate": _ratio(
            contract_clean_failure_recovery_success_count,
            total,
        ),
        "contract_clean_failure_recovery_clean_success_rate": _ratio(
            contract_clean_failure_recovery_clean_success_count,
            total,
        ),
        "failure_recovery_summary": {
            "reports": failure_recovery_report_count,
            "success_count": failure_recovery_success_count,
            "clean_success_count": failure_recovery_clean_success_count,
            "benchmark_families": sorted(failure_recovery_benchmark_families),
            "distinct_benchmark_families": len(failure_recovery_benchmark_families),
            "task_roots": sorted(failure_recovery_task_roots),
            "distinct_task_roots": len(failure_recovery_task_roots),
            "clean_success_task_roots": sorted(failure_recovery_clean_success_task_roots),
            "distinct_clean_success_task_roots": len(failure_recovery_clean_success_task_roots),
            "success_rate": _ratio(failure_recovery_success_count, failure_recovery_report_count),
            "success_rate_confidence_interval": failure_recovery_success_rate_confidence_interval,
            "clean_success_rate": _ratio(failure_recovery_clean_success_count, failure_recovery_report_count),
            "clean_success_rate_confidence_interval": failure_recovery_clean_success_rate_confidence_interval,
        },
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
    family_breadth_min_distinct_task_roots = int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0)
    if (
        scope.startswith("family:")
        and total >= breadth_min_reports
        and family_breadth_min_distinct_task_roots > 0
    ):
        distinct_clean_success_task_roots = int(summary.get("distinct_clean_success_task_roots", 0) or 0)
        if distinct_clean_success_task_roots < family_breadth_min_distinct_task_roots:
            return {
                "scope": scope,
                "passed": True,
                "status": "bootstrap",
                "detail": (
                    f"{scope} trust breadth is in bootstrap mode: clean_task_roots="
                    f"{distinct_clean_success_task_roots} threshold={family_breadth_min_distinct_task_roots}"
                ),
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
    campaign_reports = load_improvement_campaign_reports(root, max_reports=recent_limit)
    campaign_summary = summarize_improvement_campaign_reports(campaign_reports)
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
    gated_reports = [
        report
        for report in reports
        if _benchmark_family(report) in required_families and _trust_scope(report) == "gated"
    ]
    counted_gated_reports = [report for report in gated_reports if _is_counted_gated_report(report)]
    overall_summary = summarize_unattended_reports(reports)
    gated_summary = summarize_unattended_reports(gated_reports)
    counted_gated_summary = summarize_unattended_reports(counted_gated_reports)
    gated_family_summaries = {
        family: summarize_unattended_reports([report for report in gated_reports if _benchmark_family(report) == family])
        for family in sorted({_benchmark_family(report) for report in gated_reports})
    }
    counted_gated_family_summaries = {
        family: summarize_unattended_reports(
            [report for report in counted_gated_reports if _benchmark_family(report) == family]
        )
        for family in sorted({_benchmark_family(report) for report in counted_gated_reports})
    }
    campaign_signal_families = {
        str(value).strip()
        for value in list(campaign_summary.get("runtime_managed_breadth_signal_families", []) or [])
        if str(value).strip()
    }
    combined_signal_families = sorted(
        family for family in campaign_signal_families if family in required_families
    )
    combined_signal_family_set = set(combined_signal_families)
    campaign_decision_yield_families = {
        str(value).strip()
        for value in list(campaign_summary.get("runtime_managed_decision_yield_families", []) or [])
        if str(value).strip()
    }
    combined_decision_yield_families = sorted(
        family for family in campaign_decision_yield_families if family in required_families
    )
    combined_decision_yield_family_set = set(combined_decision_yield_families)
    campaign_sampled_progress_families = {
        str(value).strip()
        for value in list(campaign_summary.get("required_families_with_sampled_progress", []) or [])
        if str(value).strip()
    }
    combined_sampled_progress_families = sorted(
        family for family in campaign_sampled_progress_families if family in required_families
    )
    combined_sampled_progress_family_set = set(combined_sampled_progress_families)
    family_assessments = {
        family: assess_unattended_trust(
            counted_gated_family_summaries.get(family, summarize_unattended_reports([])),
            config=config,
            scope=f"family:{family}",
        )
        for family in family_summaries
    }
    overall_assessment = assess_unattended_trust(
        counted_gated_summary,
        config=config,
        scope="overall_gated",
        enforce_breadth=True,
    )
    coverage_summary = _coverage_summary(
        policy=trust_policy_snapshot(config),
        overall_summary=overall_summary,
        family_summaries=family_summaries,
        gated_family_summaries=gated_family_summaries,
        counted_gated_family_summaries=counted_gated_family_summaries,
        family_assessments=family_assessments,
    )
    coverage_summary["observed_repo_semantic_clusters"] = list(overall_summary.get("repo_semantic_clusters", []))
    coverage_summary["distinct_repo_semantic_clusters"] = int(
        overall_summary.get("distinct_repo_semantic_clusters", 0) or 0
    )
    coverage_summary["repo_semantic_cluster_counts"] = dict(overall_summary.get("repo_semantic_cluster_counts", {}))
    coverage_summary["task_yield_bucket_summary"] = (
        dict(overall_summary.get("task_yield_bucket_summary", {}))
        if isinstance(overall_summary.get("task_yield_bucket_summary", {}), dict)
        else {}
    )
    coverage_summary["open_world_task_yield_summary"] = {
        bucket: dict(coverage_summary["task_yield_bucket_summary"].get(bucket, {}))
        for bucket in ("semantic_hub", "external_manifest")
        if isinstance(coverage_summary["task_yield_bucket_summary"].get(bucket, {}), dict)
    }
    coverage_summary["replay_derived_task_yield_summary"] = (
        dict(coverage_summary["task_yield_bucket_summary"].get("replay_derived", {}))
        if isinstance(coverage_summary["task_yield_bucket_summary"].get("replay_derived", {}), dict)
        else {}
    )
    coverage_summary["required_family_runtime_managed_signal_counts"] = {
        family: 0 for family in coverage_summary.get("required_families", [])
    }
    coverage_summary["required_families_with_runtime_managed_signals"] = []
    coverage_summary["required_families_missing_runtime_managed_signal"] = list(
        coverage_summary.get("required_families", [])
    )
    coverage_summary["required_family_runtime_managed_retained_decision_counts"] = {
        family: 0 for family in coverage_summary.get("required_families", [])
    }
    coverage_summary["required_family_runtime_managed_decision_yield_counts"] = {
        family: 0 for family in coverage_summary.get("required_families", [])
    }
    coverage_summary["required_families_with_runtime_managed_decision_yield"] = []
    coverage_summary["required_families_missing_runtime_managed_decision_yield"] = list(
        coverage_summary.get("required_families", [])
    )
    coverage_summary["required_family_sampled_progress_counts"] = {
        family: 0 for family in coverage_summary.get("required_families", [])
    }
    coverage_summary["required_families_with_sampled_progress"] = []
    coverage_summary["required_families_missing_sampled_progress"] = list(
        coverage_summary.get("required_families", [])
    )
    coverage_summary["required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield"] = []
    coverage_summary["required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts"] = {
        family: 0 for family in coverage_summary.get("required_families", [])
    }
    campaign_clean_task_root_counts = {
        str(family).strip(): max(0, _int_value(campaign_summary.get("required_family_clean_task_root_counts", {}), (family,)))
        for family in coverage_summary.get("required_families", [])
        if str(family).strip()
    }
    if combined_sampled_progress_families or combined_signal_families or combined_decision_yield_families:
        combined_sampled_progress_counts = dict(campaign_summary.get("required_family_sampled_progress_counts", {}))
        combined_signal_counts = dict(campaign_summary.get("required_family_runtime_managed_signal_counts", {}))
        combined_retained_decision_counts = dict(
            campaign_summary.get("required_family_runtime_managed_retained_decision_counts", {})
        )
        combined_decision_yield_counts = dict(
            campaign_summary.get("required_family_runtime_managed_decision_yield_counts", {})
        )
        coverage_summary["required_family_sampled_progress_counts"] = {
            family: max(0, _int_value(combined_sampled_progress_counts, (family,)))
            for family in coverage_summary["required_families"]
        }
        coverage_summary["required_families_with_sampled_progress"] = list(combined_sampled_progress_families)
        coverage_summary["required_families_missing_sampled_progress"] = [
            family for family in coverage_summary["required_families"] if family not in combined_sampled_progress_family_set
        ]
        coverage_summary["required_family_runtime_managed_signal_counts"] = {
            family: max(0, _int_value(combined_signal_counts, (family,)))
            for family in coverage_summary["required_families"]
        }
        coverage_summary["required_families_with_runtime_managed_signals"] = list(combined_signal_families)
        coverage_summary["required_families_missing_runtime_managed_signal"] = [
            family for family in coverage_summary["required_families"] if family not in combined_signal_family_set
        ]
        coverage_summary["required_family_runtime_managed_retained_decision_counts"] = {
            family: max(0, _int_value(combined_retained_decision_counts, (family,)))
            for family in coverage_summary["required_families"]
        }
        coverage_summary["required_family_runtime_managed_decision_yield_counts"] = {
            family: max(0, _int_value(combined_decision_yield_counts, (family,)))
            for family in coverage_summary["required_families"]
        }
        coverage_summary["required_families_with_runtime_managed_decision_yield"] = list(
            combined_decision_yield_families
        )
        coverage_summary["required_families_missing_runtime_managed_decision_yield"] = [
            family
            for family in coverage_summary["required_families"]
            if family not in combined_decision_yield_family_set
        ]
        sampled_progress_without_decision_yield = sorted(
            family
            for family in combined_sampled_progress_families
            if family not in combined_decision_yield_family_set
        )
        coverage_summary["required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield"] = (
            sampled_progress_without_decision_yield
        )
        coverage_summary["required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts"] = {
            family: (
                max(0, _int_value(combined_sampled_progress_counts, (family,)))
                if family in sampled_progress_without_decision_yield
                else 0
            )
            for family in coverage_summary["required_families"]
        }
    if campaign_clean_task_root_counts:
        merged_clean_task_root_counts = dict(coverage_summary.get("required_family_clean_task_root_counts", {}))
        for family, count in campaign_clean_task_root_counts.items():
            merged_clean_task_root_counts[family] = max(
                int(merged_clean_task_root_counts.get(family, 0) or 0),
                count,
            )
        coverage_summary["required_family_clean_task_root_counts"] = merged_clean_task_root_counts
        family_breadth_min_distinct_task_roots = int(
            coverage_summary.get("family_breadth_min_distinct_task_roots", 0) or 0
        )
        if family_breadth_min_distinct_task_roots > 0:
            coverage_summary["required_families_missing_clean_task_root_breadth"] = [
                family
                for family in coverage_summary.get("required_families", [])
                if int(merged_clean_task_root_counts.get(family, 0) or 0) < family_breadth_min_distinct_task_roots
            ]
    coverage_summary["required_family_counted_evidence_summary"] = _required_family_counted_evidence_summary(
        list(coverage_summary.get("required_families", []))
        if isinstance(coverage_summary.get("required_families", []), list)
        else [],
        sampled_progress_counts=(
            dict(coverage_summary.get("required_family_sampled_progress_counts", {}))
            if isinstance(coverage_summary.get("required_family_sampled_progress_counts", {}), dict)
            else {}
        ),
        runtime_managed_signal_counts=(
            dict(coverage_summary.get("required_family_runtime_managed_signal_counts", {}))
            if isinstance(coverage_summary.get("required_family_runtime_managed_signal_counts", {}), dict)
            else {}
        ),
        runtime_managed_retained_decision_counts=(
            dict(coverage_summary.get("required_family_runtime_managed_retained_decision_counts", {}))
            if isinstance(coverage_summary.get("required_family_runtime_managed_retained_decision_counts", {}), dict)
            else {}
        ),
        runtime_managed_decision_yield_counts=(
            dict(coverage_summary.get("required_family_runtime_managed_decision_yield_counts", {}))
            if isinstance(coverage_summary.get("required_family_runtime_managed_decision_yield_counts", {}), dict)
            else {}
        ),
        clean_task_root_counts=(
            dict(coverage_summary.get("required_family_clean_task_root_counts", {}))
            if isinstance(coverage_summary.get("required_family_clean_task_root_counts", {}), dict)
            else {}
        ),
    )
    return {
        "ledger_kind": "unattended_trust_ledger",
        "generated_at": datetime.now(UTC).isoformat(),
        "reports_dir": str(root),
        "policy": trust_policy_snapshot(config),
        "reports_considered": len(reports),
        "overall_summary": overall_summary,
        "gated_summary": gated_summary,
        "counted_gated_summary": counted_gated_summary,
        "campaign_summary": campaign_summary,
        "external_summary": external_summary,
        "family_summaries": family_summaries,
        "gated_family_summaries": gated_family_summaries,
        "counted_gated_family_summaries": counted_gated_family_summaries,
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
            "family_attempt_summary": summarize_unattended_reports([]),
            "gated_summary": summarize_unattended_reports([]),
            "gated_attempt_summary": summarize_unattended_reports([]),
            "family_posture": {"required": {}, "observed": {}},
            "coverage_summary": {},
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
            "family_attempt_summary": summarize_unattended_reports([]),
            "gated_summary": summarize_unattended_reports([]),
            "gated_attempt_summary": summarize_unattended_reports([]),
            "family_posture": _family_posture(ledger),
            "coverage_summary": dict(ledger.get("coverage_summary", {}))
            if isinstance(ledger.get("coverage_summary", {}), dict)
            else {},
            "failing_thresholds": [],
        }

    ledger = build_unattended_trust_ledger(config, reports_dir=reports_dir)
    family_attempt_summary = ledger.get("gated_family_summaries", {}).get(
        family,
        summarize_unattended_reports([]),
    )
    family_summary = ledger.get("counted_gated_family_summaries", {}).get(
        family,
        summarize_unattended_reports([]),
    )
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
        "family_attempt_summary": family_attempt_summary,
        "gated_summary": ledger.get("counted_gated_summary", summarize_unattended_reports([])),
        "gated_attempt_summary": ledger["gated_summary"],
        "family_posture": _family_posture(ledger),
        "coverage_summary": dict(ledger.get("coverage_summary", {}))
        if isinstance(ledger.get("coverage_summary", {}), dict)
        else {},
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


def _repo_semantic_clusters(payload: dict[str, Any]) -> list[str]:
    task_metadata = payload.get("task_metadata", {})
    values = task_metadata.get("repo_semantics", []) if isinstance(task_metadata, dict) else []
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        label = str(value).strip().lower()
        if label and label not in normalized:
            normalized.append(label)
    return normalized


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


def _task_yield_bucket(task_origin: str) -> str:
    origin = str(task_origin).strip().lower() or "built_in"
    if origin == "semantic_hub":
        return "semantic_hub"
    if origin == "external_manifest":
        return "external_manifest"
    if origin in _REPLAY_DERIVED_TASK_ORIGINS:
        return "replay_derived"
    return "other"


def _task_root(payload: dict[str, Any]) -> str:
    task_metadata = payload.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        source_task = str(task_metadata.get("source_task", "")).strip()
        if source_task:
            return source_task
    task_contract = payload.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            source_task = str(metadata.get("source_task", "")).strip()
            if source_task:
                return source_task
    return str(payload.get("task_id", "")).strip()


def _curriculum_kind(payload: dict[str, Any]) -> str:
    task_metadata = payload.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        kind = str(task_metadata.get("curriculum_kind", "")).strip()
        if kind:
            return kind
    task_contract = payload.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            kind = str(metadata.get("curriculum_kind", "")).strip()
            if kind:
                return kind
    return ""


def _is_failure_recovery_report(payload: dict[str, Any]) -> bool:
    if _curriculum_kind(payload) == "failure_recovery":
        return True
    supervision = payload.get("supervision", {})
    if isinstance(supervision, dict):
        return bool(
            supervision.get("contract_clean_failure_recovery_origin", False)
            or supervision.get("contract_clean_failure_recovery_candidate", False)
        )
    return False


def _trust_scope(payload: dict[str, Any]) -> str:
    scope = str(payload.get("trust_scope", "")).strip()
    if scope:
        return scope
    task_metadata = payload.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        scope = str(task_metadata.get("trust_scope", "")).strip()
        if scope:
            return scope
    task_contract = payload.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            scope = str(metadata.get("trust_scope", "")).strip()
            if scope:
                return scope
    return "gated"


def _supervision_mode(payload: dict[str, Any]) -> str:
    supervision = payload.get("supervision", {})
    if isinstance(supervision, dict):
        mode = str(supervision.get("mode", "")).strip()
        if mode:
            return mode
    task_metadata = payload.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        for key in ("supervision_mode", "guidance_mode", "operator_guidance_mode"):
            mode = str(task_metadata.get(key, "")).strip()
            if mode:
                return mode
    task_contract = payload.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("supervision_mode", "guidance_mode", "operator_guidance_mode"):
                mode = str(metadata.get(key, "")).strip()
                if mode:
                    return mode
    return "unspecified"


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


def _wilson_confidence_interval(successes: int, total: int, *, z: float = 1.96) -> dict[str, float]:
    n = max(0, int(total))
    k = max(0, min(n, int(successes)))
    if n <= 0:
        return {"level": 0.95, "lower": 0.0, "upper": 0.0}
    p_hat = k / n
    z2 = z * z
    denominator = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denominator
    margin = (
        z
        * (((p_hat * (1.0 - p_hat)) / n) + (z2 / (4.0 * n * n))) ** 0.5
        / denominator
    )
    return {
        "level": 0.95,
        "lower": max(0.0, min(1.0, center - margin)),
        "upper": max(0.0, min(1.0, center + margin)),
    }


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


def _is_counted_gated_report(payload: dict[str, Any]) -> bool:
    if _trust_scope(payload) != "gated":
        return False
    supervision = payload.get("supervision", {})
    if isinstance(supervision, dict) and supervision:
        if not bool(supervision.get("independent_execution", False)):
            return False
    else:
        if _bool_value(payload, ("side_effects", "hidden_side_effect_risk")):
            return False
        if _int_value(payload, ("summary", "unexpected_change_files")) > 0:
            return False
    if _int_value(payload, ("summary", "command_steps")) > 0:
        return True
    commands = payload.get("commands", {})
    return isinstance(commands, list) and len(commands) > 0


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
    gated_family_summaries: dict[str, Any],
    counted_gated_family_summaries: dict[str, Any],
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
    required_family_gated_report_counts: dict[str, int] = {}
    required_family_counted_gated_report_counts: dict[str, int] = {}
    required_family_coverage_only_report_counts: dict[str, int] = {}
    required_family_light_supervision_report_counts: dict[str, int] = {}
    required_family_light_supervision_success_counts: dict[str, int] = {}
    required_family_light_supervision_clean_success_counts: dict[str, int] = {}
    required_family_contract_clean_failure_recovery_report_counts: dict[str, int] = {}
    required_family_contract_clean_failure_recovery_success_counts: dict[str, int] = {}
    required_family_contract_clean_failure_recovery_clean_success_counts: dict[str, int] = {}
    required_family_failure_recovery_report_counts: dict[str, int] = {}
    required_family_failure_recovery_success_counts: dict[str, int] = {}
    required_family_failure_recovery_clean_success_counts: dict[str, int] = {}
    required_family_runtime_managed_signal_counts = {family: 0 for family in required_families}
    required_family_runtime_managed_retained_decision_counts = {family: 0 for family in required_families}
    required_family_runtime_managed_decision_yield_counts = {family: 0 for family in required_families}
    required_families_with_reports: list[str] = []
    required_families_with_gated_reports: list[str] = []
    required_families_with_counted_gated_reports: list[str] = []
    missing_required_families: list[str] = []
    missing_required_gated_families: list[str] = []
    missing_required_counted_gated_families: list[str] = []
    passing_required_families: list[str] = []
    restricted_required_families: list[str] = []
    required_family_clean_task_root_counts: dict[str, int] = {}
    family_breadth_min_distinct_task_roots = max(0, int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0))
    required_families_missing_clean_task_root_breadth: list[str] = []
    for family in required_families:
        summary = family_summaries.get(family, {})
        gated_summary = gated_family_summaries.get(family, {})
        counted_gated_summary = counted_gated_family_summaries.get(family, {})
        assessment = family_assessments.get(family, {})
        report_count = int(summary.get("total", 0) or 0) if isinstance(summary, dict) else 0
        gated_report_count = int(gated_summary.get("total", 0) or 0) if isinstance(gated_summary, dict) else 0
        counted_gated_report_count = (
            int(counted_gated_summary.get("total", 0) or 0) if isinstance(counted_gated_summary, dict) else 0
        )
        required_family_report_counts[family] = report_count
        required_family_gated_report_counts[family] = gated_report_count
        required_family_counted_gated_report_counts[family] = counted_gated_report_count
        required_family_coverage_only_report_counts[family] = max(0, report_count - gated_report_count)
        required_family_light_supervision_report_counts[family] = (
            int(summary.get("light_supervision_candidate_count", 0) or 0) if isinstance(summary, dict) else 0
        )
        required_family_light_supervision_success_counts[family] = (
            int(summary.get("light_supervision_success_count", 0) or 0) if isinstance(summary, dict) else 0
        )
        required_family_light_supervision_clean_success_counts[family] = (
            int(summary.get("light_supervision_clean_success_count", 0) or 0) if isinstance(summary, dict) else 0
        )
        required_family_contract_clean_failure_recovery_report_counts[family] = (
            int(summary.get("contract_clean_failure_recovery_candidate_count", 0) or 0)
            if isinstance(summary, dict)
            else 0
        )
        required_family_contract_clean_failure_recovery_success_counts[family] = (
            int(summary.get("contract_clean_failure_recovery_success_count", 0) or 0)
            if isinstance(summary, dict)
            else 0
        )
        required_family_contract_clean_failure_recovery_clean_success_counts[family] = (
            int(summary.get("contract_clean_failure_recovery_clean_success_count", 0) or 0)
            if isinstance(summary, dict)
            else 0
        )
        failure_recovery_summary = summary.get("failure_recovery_summary", {}) if isinstance(summary, dict) else {}
        if not isinstance(failure_recovery_summary, dict):
            failure_recovery_summary = {}
        required_family_failure_recovery_report_counts[family] = int(failure_recovery_summary.get("reports", 0) or 0)
        required_family_failure_recovery_success_counts[family] = int(
            failure_recovery_summary.get("success_count", 0) or 0
        )
        required_family_failure_recovery_clean_success_counts[family] = int(
            failure_recovery_summary.get("clean_success_count", 0) or 0
        )
        required_family_clean_task_root_counts[family] = (
            int(summary.get("distinct_clean_success_task_roots", 0) or 0) if isinstance(summary, dict) else 0
        )
        if report_count > 0:
            required_families_with_reports.append(family)
        else:
            missing_required_families.append(family)
        if gated_report_count > 0:
            required_families_with_gated_reports.append(family)
        else:
            missing_required_gated_families.append(family)
        if counted_gated_report_count > 0:
            required_families_with_counted_gated_reports.append(family)
        else:
            missing_required_counted_gated_families.append(family)
        if isinstance(assessment, dict) and bool(assessment.get("passed", False)):
            passing_required_families.append(family)
        elif report_count > 0 or (isinstance(assessment, dict) and assessment):
            restricted_required_families.append(family)
        if (
            family_breadth_min_distinct_task_roots > 0
            and required_family_clean_task_root_counts[family] < family_breadth_min_distinct_task_roots
        ):
            required_families_missing_clean_task_root_breadth.append(family)
    min_distinct_families = max(0, int(policy.get("min_distinct_families", 0) or 0))
    observed_distinct_families = int(overall_summary.get("distinct_benchmark_families", 0) or 0)
    return {
        "required_families": required_families,
        "observed_families": observed_families,
        "required_family_report_counts": required_family_report_counts,
        "required_family_gated_report_counts": required_family_gated_report_counts,
        "required_family_counted_gated_report_counts": required_family_counted_gated_report_counts,
        "required_family_coverage_only_report_counts": required_family_coverage_only_report_counts,
        "required_family_light_supervision_report_counts": required_family_light_supervision_report_counts,
        "required_family_light_supervision_success_counts": required_family_light_supervision_success_counts,
        "required_family_light_supervision_clean_success_counts": required_family_light_supervision_clean_success_counts,
        "required_family_contract_clean_failure_recovery_report_counts": (
            required_family_contract_clean_failure_recovery_report_counts
        ),
        "required_family_contract_clean_failure_recovery_success_counts": (
            required_family_contract_clean_failure_recovery_success_counts
        ),
        "required_family_contract_clean_failure_recovery_clean_success_counts": (
            required_family_contract_clean_failure_recovery_clean_success_counts
        ),
        "required_family_failure_recovery_report_counts": required_family_failure_recovery_report_counts,
        "required_family_failure_recovery_success_counts": required_family_failure_recovery_success_counts,
        "required_family_failure_recovery_clean_success_counts": required_family_failure_recovery_clean_success_counts,
        "required_family_clean_task_root_counts": required_family_clean_task_root_counts,
        "family_breadth_min_distinct_task_roots": family_breadth_min_distinct_task_roots,
        "required_families_missing_clean_task_root_breadth": required_families_missing_clean_task_root_breadth,
        "required_family_runtime_managed_signal_counts": required_family_runtime_managed_signal_counts,
        "required_family_runtime_managed_retained_decision_counts": (
            required_family_runtime_managed_retained_decision_counts
        ),
        "required_family_runtime_managed_decision_yield_counts": required_family_runtime_managed_decision_yield_counts,
        "required_families_with_runtime_managed_signals": [],
        "required_families_missing_runtime_managed_signal": list(required_families),
        "required_families_with_runtime_managed_decision_yield": [],
        "required_families_missing_runtime_managed_decision_yield": list(required_families),
        "required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield": [],
        "required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts": {
            family: 0 for family in required_families
        },
        "required_families_with_reports": required_families_with_reports,
        "required_families_with_gated_reports": required_families_with_gated_reports,
        "required_families_with_counted_gated_reports": required_families_with_counted_gated_reports,
        "missing_required_families": missing_required_families,
        "missing_required_gated_families": missing_required_gated_families,
        "missing_required_counted_gated_families": missing_required_counted_gated_families,
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
