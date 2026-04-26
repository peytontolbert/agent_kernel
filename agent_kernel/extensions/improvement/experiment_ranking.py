from __future__ import annotations

from pathlib import Path

from evals.metrics import EvalMetrics

from ...config import KernelConfig
from ...extensions.trust import load_improvement_campaign_reports, trust_policy_snapshot
from ...improvement_engine import ImprovementExperiment, sort_experiments as engine_sort_experiments


def _improvement_reports_dir(planner) -> Path | None:
    runtime_config = getattr(planner, "runtime_config", None)
    if runtime_config is not None:
        candidate = getattr(runtime_config, "improvement_reports_dir", None)
        if candidate is not None:
            resolved = Path(candidate)
            if resolved.exists():
                return resolved
    cycles_path = getattr(planner, "cycles_path", None)
    if cycles_path is not None:
        resolved_cycles_path = Path(cycles_path)
        candidate = resolved_cycles_path.parent / "reports"
        if candidate.exists():
            return candidate
    return None


def _closure_progression_summary(planner) -> dict[str, object]:
    reports_dir = _improvement_reports_dir(planner)
    if reports_dir is None:
        return {}
    reports = load_improvement_campaign_reports(reports_dir, max_reports=12)
    if not reports:
        return {}
    latest = reports[0]
    for report in reports:
        status = str(report.get("status", report.get("state", ""))).strip().lower()
        retained_gain_runs = int(report.get("retained_gain_runs", 0) or 0)
        runtime_managed_decisions = int(report.get("runtime_managed_decisions", 0) or 0)
        if retained_gain_runs > 0:
            latest = report
            break
        if runtime_managed_decisions > 0 and status not in {"interrupted", "running", "starting"}:
            latest = report
            break
    closure = latest.get("closure_gap_summary", {})
    if not isinstance(closure, dict):
        return {}
    retained_conversion = str(closure.get("retained_conversion", "")).strip().lower()
    trust_breadth = str(closure.get("trust_breadth", "")).strip().lower()
    retrieval_carryover = str(closure.get("retrieval_carryover", "")).strip().lower()
    if not retained_conversion and not trust_breadth and not retrieval_carryover:
        return {}
    return {
        "retained_conversion": retained_conversion,
        "trust_breadth": trust_breadth,
        "retrieval_carryover": retrieval_carryover,
        "retrieval_carryover_priority": (
            retained_conversion == "closed"
            and trust_breadth == "closed"
            and retrieval_carryover not in {"", "closed"}
        ),
        "runtime_managed_decisions": int(latest.get("runtime_managed_decisions", 0) or 0),
        "retained_gain_runs": int(latest.get("retained_gain_runs", 0) or 0),
        "retained_by_subsystem": dict(latest.get("yield_summary", {}).get("retained_by_subsystem", {}) or {})
        if isinstance(latest.get("yield_summary", {}), dict)
        else {},
    }


def _current_trust_coverage_summary(planner, metrics: EvalMetrics) -> dict[str, object]:
    config = planner.runtime_config or KernelConfig()
    policy = trust_policy_snapshot(config)
    required_families = [
        str(value).strip()
        for value in policy.get("required_benchmark_families", [])
        if str(value).strip()
    ]
    observed_family_totals = {
        str(family).strip(): int(total or 0)
        for family, total in metrics.total_by_benchmark_family.items()
        if str(family).strip()
        and str(family).strip() != "benchmark_candidate"
        and int(total or 0) > 0
    }
    observed_families = sorted(observed_family_totals)
    min_distinct_families = max(0, int(policy.get("min_distinct_families", 0) or 0))
    family_breadth_min_distinct_task_roots = max(
        0,
        int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0),
    )
    current_clean_task_root_counts: dict[str, int] = {family: 0 for family in required_families}
    clean_success_task_roots_by_family: dict[str, set[str]] = {family: set() for family in required_families}
    current_required_families_with_reports: set[str] = set()
    current_required_families_with_external_reports: set[str] = set()
    current_external_benchmark_families: set[str] = set()
    current_external_report_count = 0
    current_external_unsafe_ambiguous_count = 0
    for task_id, summary in metrics.task_outcomes.items():
        if not isinstance(summary, dict):
            continue
        family = str(summary.get("benchmark_family", "")).strip()
        if family and family in required_families:
            current_required_families_with_reports.add(family)
        task_origin = str(summary.get("task_origin", "")).strip().lower()
        if task_origin == "external_manifest":
            current_external_report_count += 1
            if family:
                current_external_benchmark_families.add(family)
                if family in required_families:
                    current_required_families_with_external_reports.add(family)
            if bool(summary.get("unsafe_ambiguous", False)):
                current_external_unsafe_ambiguous_count += 1
        if not family or family not in clean_success_task_roots_by_family:
            continue
        if not bool(summary.get("clean_success", False)):
            continue
        task_root = str(summary.get("task_root", "")).strip() or str(task_id).strip()
        if not task_root:
            continue
        clean_success_task_roots_by_family[family].add(task_root)
    current_clean_task_root_counts.update(
        {
            family: len(task_roots)
            for family, task_roots in clean_success_task_roots_by_family.items()
        }
    )
    return {
        "required_families": required_families,
        "observed_families": observed_families,
        "observed_distinct_families": len(observed_families),
        "missing_required_families": [
            family for family in required_families if family not in observed_family_totals
        ],
        "required_families_with_reports": [
            family for family in required_families if family in current_required_families_with_reports
        ],
        "required_families_with_external_reports": [
            family for family in required_families if family in current_required_families_with_external_reports
        ],
        "distinct_family_gap": max(0, min_distinct_families - len(observed_families)),
        "family_breadth_min_distinct_task_roots": family_breadth_min_distinct_task_roots,
        "required_family_clean_task_root_counts": current_clean_task_root_counts,
        "missing_required_family_clean_task_root_breadth": [
            family
            for family, total in current_clean_task_root_counts.items()
            if total < family_breadth_min_distinct_task_roots
        ],
        "external_report_count": current_external_report_count,
        "distinct_external_benchmark_families": len(current_external_benchmark_families),
        "external_benchmark_families": sorted(current_external_benchmark_families),
        "external_unsafe_ambiguous_rate": (
            round(current_external_unsafe_ambiguous_count / current_external_report_count, 4)
            if current_external_report_count > 0
            else 0.0
        ),
    }


def _coverage_only_trust_failure(threshold: object) -> bool:
    normalized = str(threshold).strip()
    return (
        "distinct_benchmark_families=" in normalized
        or "clean_task_roots=" in normalized
        or "missing_required_families" in normalized
    )


def _external_only_bootstrap_trust_gap(trust_summary: dict[str, object]) -> bool:
    if not trust_summary:
        return False
    if str(trust_summary.get("overall_status", "")).strip() not in {"bootstrap", "restricted"}:
        return False
    if int(trust_summary.get("external_report_count", 0) or 0) > 0:
        return False
    if int(trust_summary.get("distinct_external_benchmark_families", 0) or 0) > 0:
        return False
    if int(trust_summary.get("distinct_family_gap", 0) or 0) > 0:
        return False
    if list(trust_summary.get("missing_required_families", [])):
        return False
    if list(trust_summary.get("missing_required_family_clean_task_root_breadth", [])):
        return False
    return not any(
        float(trust_summary.get(field, 0.0) or 0.0) > 0.0
        for field in (
            "unsafe_ambiguous_rate",
            "hidden_side_effect_risk_rate",
            "success_hidden_side_effect_risk_rate",
            "false_pass_risk_rate",
            "unexpected_change_report_rate",
            "rollback_performed_rate",
        )
    )


def _counted_external_breadth_aligned(trust_summary: dict[str, object]) -> bool:
    if not trust_summary:
        return False
    required_families = [
        str(value).strip()
        for value in trust_summary.get("required_families", [])
        if str(value).strip()
    ]
    required_families_with_reports = {
        str(value).strip()
        for value in trust_summary.get("required_families_with_external_reports", [])
        if str(value).strip()
    }
    if not required_families_with_reports:
        required_families_with_reports = {
        str(value).strip()
        for value in trust_summary.get("required_families_with_reports", [])
        if str(value).strip()
        }
    if int(trust_summary.get("external_report_count", 0) or 0) <= 0:
        return False
    if int(trust_summary.get("distinct_external_benchmark_families", 0) or 0) <= 0:
        return False
    if int(trust_summary.get("distinct_family_gap", 0) or 0) > 0:
        return False
    if list(trust_summary.get("missing_required_families", [])):
        return False
    if list(trust_summary.get("missing_required_family_clean_task_root_breadth", [])):
        return False
    if required_families and not all(
        family in required_families_with_reports for family in required_families
    ):
        return False
    return not any(
        float(trust_summary.get(field, 0.0) or 0.0) > 0.0
        for field in (
            "unsafe_ambiguous_rate",
            "hidden_side_effect_risk_rate",
            "success_hidden_side_effect_risk_rate",
            "false_pass_risk_rate",
            "unexpected_change_report_rate",
            "rollback_performed_rate",
            "external_unsafe_ambiguous_rate",
        )
    )


def _retained_conversion_gap_summary(planner, *, recent_cycle_window: int = 8) -> dict[str, object]:
    summaries: dict[str, dict[str, object]] = {}
    for subsystem in ("curriculum", "transition_model"):
        summaries[subsystem] = planner.recent_subsystem_activity_summary(
            subsystem=subsystem,
            recent_cycle_window=recent_cycle_window,
        )
    transition_model_summary = dict(summaries.get("transition_model", {}) or {})
    if (
        int(transition_model_summary.get("total_decisions", 0) or 0) <= 0
        and (
            int(transition_model_summary.get("selected_cycles", 0) or 0) > 0
            or int(transition_model_summary.get("no_yield_cycles", 0) or 0) > 0
        )
    ):
        wider_window = max(12, recent_cycle_window * 2)
        extended_transition_model_summary = planner.recent_subsystem_activity_summary(
            subsystem="transition_model",
            recent_cycle_window=wider_window,
        )
        if isinstance(extended_transition_model_summary, dict):
            summaries["transition_model"] = dict(extended_transition_model_summary)
    selected_cycles = sum(int(summary.get("selected_cycles", 0) or 0) for summary in summaries.values())
    retained_cycles = sum(int(summary.get("retained_cycles", 0) or 0) for summary in summaries.values())
    total_decisions = sum(int(summary.get("total_decisions", 0) or 0) for summary in summaries.values())
    rejected_cycles = sum(int(summary.get("rejected_cycles", 0) or 0) for summary in summaries.values())
    no_yield_cycles = sum(int(summary.get("no_yield_cycles", 0) or 0) for summary in summaries.values())
    incomplete_cycles = sum(int(summary.get("recent_incomplete_cycles", 0) or 0) for summary in summaries.values())
    return {
        "recent_cycle_window": max(1, recent_cycle_window),
        "selected_cycles": selected_cycles,
        "retained_cycles": retained_cycles,
        "total_decisions": total_decisions,
        "rejected_cycles": rejected_cycles,
        "no_yield_cycles": no_yield_cycles,
        "recent_incomplete_cycles": incomplete_cycles,
        "active": (
            selected_cycles > 0
            and retained_cycles == 0
            and (
                rejected_cycles > 0
                or (
                    total_decisions == 0
                    and (no_yield_cycles > 0 or incomplete_cycles > 0)
                )
            )
        ),
        "subsystem_activity": summaries,
    }


def _a4_runtime_conversion_transition_model_priority(
    retained_conversion_gap: dict[str, object],
) -> bool:
    subsystem_activity = retained_conversion_gap.get("subsystem_activity", {})
    subsystem_activity = dict(subsystem_activity) if isinstance(subsystem_activity, dict) else {}
    transition_model_activity = subsystem_activity.get("transition_model", {})
    transition_model_activity = (
        dict(transition_model_activity) if isinstance(transition_model_activity, dict) else {}
    )
    curriculum_activity = subsystem_activity.get("curriculum", {})
    curriculum_activity = dict(curriculum_activity) if isinstance(curriculum_activity, dict) else {}
    transition_model_has_decision_reject = (
        int(transition_model_activity.get("total_decisions", 0) or 0) > 0
        and int(transition_model_activity.get("rejected_cycles", 0) or 0) > 0
        and str(transition_model_activity.get("last_decision_state", "")).strip() == "reject"
    )
    curriculum_is_still_unresolved = (
        (
            int(curriculum_activity.get("no_yield_cycles", 0) or 0) > 0
            or int(curriculum_activity.get("recent_incomplete_cycles", 0) or 0) > 0
        )
        and str(curriculum_activity.get("last_decision_state", "")).strip() == ""
    )
    curriculum_already_retained = (
        int(curriculum_activity.get("retained_cycles", 0) or 0) > 0
        and int(curriculum_activity.get("total_decisions", 0) or 0) > 0
        and str(curriculum_activity.get("last_decision_state", "")).strip() == "retain"
    )
    return transition_model_has_decision_reject and (
        curriculum_is_still_unresolved or curriculum_already_retained
    )


def _effective_trust_summary(planner, metrics: EvalMetrics) -> dict[str, object]:
    trust_summary = planner.trust_ledger_summary()
    if not trust_summary:
        return {}
    effective = dict(trust_summary)
    live_coverage = _current_trust_coverage_summary(planner, metrics)
    original_missing_required_families = [
        str(value).strip()
        for value in effective.get("missing_required_families", [])
        if str(value).strip()
    ]
    original_required_families_with_reports = [
        str(value).strip()
        for value in effective.get("required_families_with_reports", [])
        if str(value).strip()
    ]
    original_required_families_with_external_reports = [
        str(value).strip()
        for value in effective.get("required_families_with_external_reports", [])
        if str(value).strip()
    ]
    live_missing_required_families = set(live_coverage["missing_required_families"])
    effective_missing_required_families = [
        family for family in original_missing_required_families if family in live_missing_required_families
    ]
    original_distinct_family_gap = int(effective.get("distinct_family_gap", 0) or 0)
    effective_distinct_family_gap = min(
        original_distinct_family_gap,
        int(live_coverage.get("distinct_family_gap", 0) or 0),
    )
    original_distinct_benchmark_families = int(effective.get("distinct_benchmark_families", 0) or 0)
    effective_distinct_benchmark_families = max(
        original_distinct_benchmark_families,
        int(live_coverage.get("observed_distinct_families", 0) or 0),
    )
    original_required_family_clean_task_root_counts = {
        str(family).strip(): int(total or 0)
        for family, total in dict(effective.get("required_family_clean_task_root_counts", {})).items()
        if str(family).strip()
    }
    live_required_family_clean_task_root_counts = {
        str(family).strip(): int(total or 0)
        for family, total in dict(live_coverage.get("required_family_clean_task_root_counts", {})).items()
        if str(family).strip()
    }
    effective_required_family_clean_task_root_counts = dict(original_required_family_clean_task_root_counts)
    for family, live_total in live_required_family_clean_task_root_counts.items():
        effective_required_family_clean_task_root_counts[family] = max(
            int(effective_required_family_clean_task_root_counts.get(family, 0) or 0),
            live_total,
        )
    breadth_threshold = int(
        effective.get(
            "family_breadth_min_distinct_task_roots",
            live_coverage.get("family_breadth_min_distinct_task_roots", 0),
        )
        or 0
    )
    original_missing_required_task_root_breadth = [
        str(value).strip()
        for value in effective.get("missing_required_family_clean_task_root_breadth", [])
        if str(value).strip()
    ]
    effective_missing_required_task_root_breadth = [
        family
        for family in original_missing_required_task_root_breadth
        if int(effective_required_family_clean_task_root_counts.get(family, 0) or 0) < breadth_threshold
    ]
    effective_required_families_with_reports = list(
        dict.fromkeys(
            original_required_families_with_reports
            + [
                family
                for family in list(live_coverage.get("required_families_with_reports", []))
                if str(family).strip()
            ]
        )
    )
    effective_required_families_with_external_reports = list(
        dict.fromkeys(
            original_required_families_with_external_reports
            + [
                family
                for family in list(live_coverage.get("required_families_with_external_reports", []))
                if str(family).strip()
            ]
        )
    )
    original_external_report_count = int(effective.get("external_report_count", 0) or 0)
    effective_external_report_count = max(
        original_external_report_count,
        int(live_coverage.get("external_report_count", 0) or 0),
    )
    original_distinct_external_benchmark_families = int(
        effective.get("distinct_external_benchmark_families", 0) or 0
    )
    effective_distinct_external_benchmark_families = max(
        original_distinct_external_benchmark_families,
        int(live_coverage.get("distinct_external_benchmark_families", 0) or 0),
    )
    original_external_benchmark_families = [
        str(value).strip()
        for value in effective.get("external_benchmark_families", [])
        if str(value).strip()
    ]
    effective_external_benchmark_families = list(
        dict.fromkeys(
            original_external_benchmark_families
            + [
                family
                for family in list(live_coverage.get("external_benchmark_families", []))
                if str(family).strip()
            ]
        )
    )
    original_external_unsafe_ambiguous_rate = float(effective.get("external_unsafe_ambiguous_rate", 0.0) or 0.0)
    effective_external_unsafe_ambiguous_rate = max(
        original_external_unsafe_ambiguous_rate,
        float(live_coverage.get("external_unsafe_ambiguous_rate", 0.0) or 0.0),
    )
    effective["missing_required_families"] = effective_missing_required_families
    effective["required_families_with_reports"] = effective_required_families_with_reports
    effective["required_families_with_external_reports"] = effective_required_families_with_external_reports
    effective["distinct_family_gap"] = effective_distinct_family_gap
    effective["distinct_benchmark_families"] = effective_distinct_benchmark_families
    effective["required_family_clean_task_root_counts"] = effective_required_family_clean_task_root_counts
    effective["missing_required_family_clean_task_root_breadth"] = effective_missing_required_task_root_breadth
    effective["external_report_count"] = effective_external_report_count
    effective["distinct_external_benchmark_families"] = effective_distinct_external_benchmark_families
    effective["external_benchmark_families"] = effective_external_benchmark_families
    effective["external_unsafe_ambiguous_rate"] = effective_external_unsafe_ambiguous_rate
    effective["current_observed_benchmark_families"] = list(live_coverage["observed_families"])
    effective["current_missing_required_families"] = list(live_coverage["missing_required_families"])
    effective["current_required_families_with_reports"] = list(live_coverage.get("required_families_with_reports", []))
    effective["current_required_families_with_external_reports"] = list(
        live_coverage.get("required_families_with_external_reports", [])
    )
    effective["current_distinct_family_gap"] = int(live_coverage.get("distinct_family_gap", 0) or 0)
    effective["current_required_family_clean_task_root_counts"] = live_required_family_clean_task_root_counts
    effective["current_missing_required_family_clean_task_root_breadth"] = list(
        live_coverage.get("missing_required_family_clean_task_root_breadth", [])
    )
    effective["current_external_report_count"] = int(live_coverage.get("external_report_count", 0) or 0)
    effective["current_distinct_external_benchmark_families"] = int(
        live_coverage.get("distinct_external_benchmark_families", 0) or 0
    )
    effective["current_external_benchmark_families"] = list(live_coverage.get("external_benchmark_families", []))
    effective["current_external_unsafe_ambiguous_rate"] = float(
        live_coverage.get("external_unsafe_ambiguous_rate", 0.0) or 0.0
    )
    effective["coverage_override_applied"] = (
        effective_missing_required_families != original_missing_required_families
        or effective_required_families_with_reports != original_required_families_with_reports
        or effective_required_families_with_external_reports != original_required_families_with_external_reports
        or effective_distinct_family_gap != original_distinct_family_gap
        or effective_distinct_benchmark_families != original_distinct_benchmark_families
        or effective_required_family_clean_task_root_counts != original_required_family_clean_task_root_counts
        or effective_missing_required_task_root_breadth != original_missing_required_task_root_breadth
        or effective_external_report_count != original_external_report_count
        or effective_distinct_external_benchmark_families != original_distinct_external_benchmark_families
        or effective_external_benchmark_families != original_external_benchmark_families
        or effective_external_unsafe_ambiguous_rate != original_external_unsafe_ambiguous_rate
    )
    residual_coverage_signal = (
        effective_distinct_benchmark_families < 2
        or effective_distinct_family_gap > 0
        or bool(effective_missing_required_families)
        or bool(list(effective.get("missing_required_family_clean_task_root_breadth", [])))
    )
    residual_risk_signal = any(
        float(effective.get(field, 0.0) or 0.0) > 0.0
        for field in (
            "unsafe_ambiguous_rate",
            "hidden_side_effect_risk_rate",
            "success_hidden_side_effect_risk_rate",
            "false_pass_risk_rate",
            "unexpected_change_report_rate",
            "rollback_performed_rate",
        )
    )
    non_coverage_failures = [
        str(value).strip()
        for value in effective.get("failing_thresholds", [])
        if str(value).strip() and not _coverage_only_trust_failure(value)
    ]
    if (
        str(effective.get("overall_status", "")).strip() in {"bootstrap", "restricted"}
        and not residual_coverage_signal
        and not residual_risk_signal
        and not non_coverage_failures
    ):
        effective["overall_status"] = "coverage_aligned"
        effective["overall_passed"] = True
        effective["failing_thresholds"] = []
        effective["status_override_reason"] = "current_metrics_closed_coverage_only_trust_gap"
    return effective


def rank_experiments(planner, metrics: EvalMetrics) -> list[ImprovementExperiment]:
    failure_counts = planner.failure_counts()
    transition_failure_counts = planner.transition_failure_counts()
    transition_summary = planner.transition_summary()
    trust_summary = _effective_trust_summary(planner, metrics)
    closure_progression = _closure_progression_summary(planner)
    broad_observe_signal = planner._broad_coding_observe_diversification_signal(metrics)
    primary_only_broad_observe = bool(broad_observe_signal.get("primary_only_broad_observe", False))
    retained_conversion_gap = _retained_conversion_gap_summary(planner)
    direct_transition_model_priority = _a4_runtime_conversion_transition_model_priority(
        retained_conversion_gap,
    )
    broad_observe_curriculum_priority = (
        bool(broad_observe_signal.get("active", False))
        and primary_only_broad_observe
        and not bool(broad_observe_signal.get("retrieval_emergency", False))
        and (
            direct_transition_model_priority
            or
            bool(retained_conversion_gap.get("active", False))
            or (
            _external_only_bootstrap_trust_gap(trust_summary)
            or (
                _counted_external_breadth_aligned(trust_summary)
                and bool(retained_conversion_gap.get("active", False))
            )
            )
        )
    )
    candidates: list[ImprovementExperiment] = []
    if (failure_counts or transition_failure_counts) and (
        metrics.total_by_benchmark_family.get("benchmark_candidate", 0) == 0
        or metrics.generated_pass_rate < metrics.pass_rate
    ):
        benchmark_gain_raw = max(
            0.02,
            (
                failure_counts.get("missing_expected_file", 0)
                + transition_failure_counts.get("no_state_progress", 0)
                + transition_failure_counts.get("state_regression", 0)
            )
            / max(1, sum(failure_counts.values())),
        )
        benchmark_gain = round(max(0.02, min(0.05, benchmark_gain_raw)), 4)
        candidates.append(
            ImprovementExperiment(
                subsystem="benchmark",
                reason="failure clusters, stalled transitions, and environment patterns can be turned into benchmark proposals, but those proposals must expand operator-relevant coverage instead of only matching the current validator shape",
                priority=5,
                expected_gain=benchmark_gain,
                estimated_cost=3,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "benchmark_candidate_total": metrics.total_by_benchmark_family.get("benchmark_candidate", 0),
                    "generated_pass_rate": metrics.generated_pass_rate,
                    "pass_rate": metrics.pass_rate,
                    "benchmark_expected_gain_raw": round(float(benchmark_gain_raw), 4),
                    "benchmark_expected_gain_capped": benchmark_gain,
                    "benchmark_candidate_share": round(
                        metrics.total_by_benchmark_family.get("benchmark_candidate", 0) / max(1, metrics.total),
                        4,
                    ),
                },
            )
        )
    low_confidence_state_signal = 0
    if metrics.low_confidence_episodes > 0 or metrics.trusted_retrieval_steps < metrics.total // 2:
        confidence_gap = max(
            metrics.low_confidence_episodes / max(1, metrics.total),
            0.0 if metrics.total == 0 else 0.5 - (metrics.trusted_retrieval_steps / max(1, metrics.total)),
        )
        retrieval_deficit = max(0, (metrics.total // 2) - int(metrics.trusted_retrieval_steps or 0))
        if confidence_gap >= 0.1 or retrieval_deficit > 0:
            low_confidence_state_signal = max(
                int(metrics.low_confidence_episodes or 0),
                retrieval_deficit,
            )
        allow_qwen_support_runtime, coding_strength = planner._allow_qwen_adapter_support_runtime(metrics)
        candidates.append(
            ImprovementExperiment(
                subsystem="retrieval",
                reason="low-confidence retrieval remains common relative to trusted retrieval usage",
                priority=5,
                expected_gain=round(max(0.02, confidence_gap), 4),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "retrieval_selected_steps": metrics.retrieval_selected_steps,
                    "retrieval_influenced_steps": metrics.retrieval_influenced_steps,
                    "proposal_selected_steps": metrics.proposal_selected_steps,
                    "total": metrics.total,
                },
            )
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="tolbert_model",
                reason="persistent retrieval weakness may still need a learned Tolbert checkpoint, but runtime retrieval and state controls should be repaired first",
                priority=4,
                expected_gain=round(max(0.02, confidence_gap), 4),
                estimated_cost=4,
                score=0.0,
                evidence={
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                    "total": metrics.total,
                    "coding_strength": coding_strength,
                    "cross_cycle_weight_update": True,
                },
            )
        )
        if allow_qwen_support_runtime:
            candidates.append(
                ImprovementExperiment(
                    subsystem="qwen_adapter",
                    reason="current coding weakness may eventually need a stronger adapted Qwen support runtime, but prompt and state controls should lead the low-confidence repair loop",
                    priority=4,
                    expected_gain=round(max(0.015, confidence_gap * 0.6), 4),
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                        "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                        "total": metrics.total,
                        "support_runtime_only": True,
                        "coding_strength": coding_strength,
                        "cross_cycle_weight_update": True,
                    },
                )
            )
    if metrics.total_by_memory_source.get("verifier", 0) == 0 or transition_failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="verifier",
                reason="verifier-memory lane is not yet populated enough to discriminate stalled or regressive intermediate states",
                priority=5,
                expected_gain=round(max(0.03, min(0.05, 0.01 * sum(transition_failure_counts.values()))), 4),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "verifier_memory_total": metrics.total_by_memory_source.get("verifier", 0),
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if bool(closure_progression.get("retrieval_carryover_priority", False)):
        retrieval_reason = (
            "retained conversion and counted trust breadth are already closed in the latest official report, "
            "so the next official batch should prove retrieval carryover instead of reopening closure-first lanes"
        )
        retrieval_evidence = {
            "closure_progression": dict(closure_progression),
            "retrieval_carryover_priority": True,
        }
        existing_retrieval_index = next(
            (index for index, candidate in enumerate(candidates) if candidate.subsystem == "retrieval"),
            None,
        )
        if existing_retrieval_index is None:
            candidates.append(
                ImprovementExperiment(
                    subsystem="retrieval",
                    reason=retrieval_reason,
                    priority=7,
                    expected_gain=0.05,
                    estimated_cost=2,
                    score=0.0,
                    evidence=retrieval_evidence,
                )
            )
        else:
            existing_retrieval = candidates[existing_retrieval_index]
            merged_evidence = dict(existing_retrieval.evidence)
            merged_evidence.update(retrieval_evidence)
            candidates[existing_retrieval_index] = ImprovementExperiment(
                subsystem="retrieval",
                reason=retrieval_reason,
                priority=max(7, int(existing_retrieval.priority)),
                expected_gain=max(0.05, float(existing_retrieval.expected_gain)),
                estimated_cost=min(2, int(existing_retrieval.estimated_cost)),
                score=0.0,
                evidence=merged_evidence,
            )
    if failure_counts.get("command_failure", 0) >= failure_counts.get("missing_expected_file", 0) and failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="tooling",
                reason="command failures dominate stored failures, suggesting missing reusable procedures or tools",
                priority=4,
                expected_gain=round(
                    max(0.01, failure_counts.get("command_failure", 0) / max(1, sum(failure_counts.values()))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "command_failure_count": failure_counts.get("command_failure", 0),
                    "failure_counts": failure_counts,
                },
            )
        )
    if metrics.retrieval_ranked_skill_steps < max(1, metrics.skill_selected_steps // 2):
        candidates.append(
            ImprovementExperiment(
                subsystem="skills",
                reason="skill usage is high but retrieval-ranked skill selection is comparatively weak",
                priority=4,
                expected_gain=round(
                    max(
                        0.01,
                        (metrics.skill_selected_steps - metrics.retrieval_ranked_skill_steps) / max(1, metrics.skill_selected_steps),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "skill_selected_steps": metrics.skill_selected_steps,
                    "retrieval_ranked_skill_steps": metrics.retrieval_ranked_skill_steps,
                },
            )
        )
    if metrics.total_by_memory_source.get("skill_transfer", 0) == 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="operators",
                reason="cross-task operator transfer is not yet populated or measured against raw skill transfer",
                priority=4,
                expected_gain=0.03,
                estimated_cost=4,
                score=0.0,
                evidence={
                    "skill_transfer_total": metrics.total_by_memory_source.get("skill_transfer", 0),
                    "operator_total": metrics.total_by_memory_source.get("operator", 0),
                },
            )
        )
    if (
        (metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate)
        or transition_failure_counts.get("no_state_progress", 0) > 0
        or transition_failure_counts.get("state_regression", 0) > 0
    ):
        generated_gap = 0.0
        if metrics.generated_total:
            generated_gap = max(0.0, metrics.pass_rate - metrics.generated_pass_rate)
        candidates.append(
            ImprovementExperiment(
                subsystem="curriculum",
                reason="generated-task pressure should target repeated no-progress and regression transitions, not only coarse task failures",
                priority=4,
                expected_gain=round(
                    max(
                        0.02,
                        generated_gap,
                        min(
                            0.05,
                            (
                                transition_failure_counts.get("no_state_progress", 0)
                                + transition_failure_counts.get("state_regression", 0)
                            )
                            / max(1, sum(transition_failure_counts.values())),
                        ),
                    ),
                    4,
                ),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "pass_rate": metrics.pass_rate,
                    "generated_pass_rate": metrics.generated_pass_rate,
                    "generated_total": metrics.generated_total,
                    "generated_pass_rate_gap": round(generated_gap, 4),
                    "transition_failure_counts": transition_failure_counts,
                },
            )
        )
    if broad_observe_curriculum_priority:
        observed_family_count = int(broad_observe_signal.get("observed_family_count", 0) or 0)
        external_only_trust_bootstrap = _external_only_bootstrap_trust_gap(trust_summary)
        counted_external_breadth_aligned = _counted_external_breadth_aligned(trust_summary)
        retained_conversion_gap_active = bool(retained_conversion_gap.get("active", False))
        curriculum_reason = (
            "primary-only broad observe should be converted into a retained integrated rerun and "
            "generated repair pressure before external-only trust bootstrap closure"
        )
        if counted_external_breadth_aligned:
            curriculum_reason = (
                "counted external breadth is now live, so broad observe should convert into a "
                "retained integrated rerun instead of another non-decision diversification lane"
            )
        elif retained_conversion_gap_active:
            curriculum_reason = (
                "clean broad observe should be converted into a retained integrated rerun because "
                "retained conversion is still unresolved"
            )
        curriculum_evidence = {
            "broad_observe_diversification": dict(broad_observe_signal),
            "retained_conversion_priority": True,
            "external_only_trust_bootstrap": external_only_trust_bootstrap,
            "counted_external_breadth_aligned": counted_external_breadth_aligned,
            "retained_conversion_gap_active": retained_conversion_gap_active,
            "retained_conversion_gap": retained_conversion_gap,
            "external_report_count": int(trust_summary.get("external_report_count", 0) or 0),
            "distinct_external_benchmark_families": int(
                trust_summary.get("distinct_external_benchmark_families", 0) or 0
            ),
        }
        curriculum_expected_gain = round(max(0.03, min(0.05, observed_family_count * 0.01)), 4)
        if direct_transition_model_priority:
            transition_model_reason = (
                "required-family breadth is already counted and transition-model already reached a "
                "decision-bearing reject, so the next retained-conversion rerun should ratchet "
                "transition-model before replaying curriculum"
            )
            transition_model_evidence = {
                **curriculum_evidence,
                "a4_runtime_conversion_priority": True,
                "decision_bearing_transition_model_retry": True,
            }
            existing_transition_model_index = next(
                (index for index, candidate in enumerate(candidates) if candidate.subsystem == "transition_model"),
                None,
            )
            if existing_transition_model_index is None:
                candidates.append(
                    ImprovementExperiment(
                        subsystem="transition_model",
                        reason=transition_model_reason,
                        priority=6,
                        expected_gain=max(0.05, curriculum_expected_gain),
                        estimated_cost=2,
                        score=0.0,
                        evidence=transition_model_evidence,
                    )
                )
            else:
                existing_transition_model = candidates[existing_transition_model_index]
                merged_evidence = dict(existing_transition_model.evidence)
                merged_evidence.update(transition_model_evidence)
                candidates[existing_transition_model_index] = ImprovementExperiment(
                    subsystem="transition_model",
                    reason=transition_model_reason,
                    priority=max(6, int(existing_transition_model.priority)),
                    expected_gain=max(0.05, curriculum_expected_gain, float(existing_transition_model.expected_gain)),
                    estimated_cost=min(2, int(existing_transition_model.estimated_cost)),
                    score=0.0,
                    evidence=merged_evidence,
                )
        else:
            existing_curriculum_index = next(
                (index for index, candidate in enumerate(candidates) if candidate.subsystem == "curriculum"),
            None,
                )
            if existing_curriculum_index is None:
                candidates.append(
                    ImprovementExperiment(
                        subsystem="curriculum",
                        reason=curriculum_reason,
                        priority=5,
                        expected_gain=curriculum_expected_gain,
                        estimated_cost=3,
                        score=0.0,
                        evidence=curriculum_evidence,
                    )
                )
            else:
                existing_curriculum = candidates[existing_curriculum_index]
                merged_evidence = dict(existing_curriculum.evidence)
                merged_evidence.update(curriculum_evidence)
                candidates[existing_curriculum_index] = ImprovementExperiment(
                    subsystem="curriculum",
                    reason=curriculum_reason,
                    priority=max(5, int(existing_curriculum.priority)),
                    expected_gain=max(curriculum_expected_gain, float(existing_curriculum.expected_gain)),
                    estimated_cost=min(3, int(existing_curriculum.estimated_cost)),
                    score=0.0,
                    evidence=merged_evidence,
                )
    repo_world_model_total = sum(
        int(metrics.total_by_benchmark_family.get(family, 0))
        for family in ("repo_sandbox", "repo_chore", "repository", "project", "integration")
    )
    world_model_failure_signal = (
        failure_counts.get("missing_expected_file", 0)
        + failure_counts.get("command_failure", 0)
        + transition_failure_counts.get("no_state_progress", 0)
        + transition_failure_counts.get("state_regression", 0)
    )
    if repo_world_model_total > 0 and world_model_failure_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="world_model",
                reason="repo-workflow failures and bad transitions suggest command scoring, progress estimation, and preserved-path modeling are still weak",
                priority=3,
                expected_gain=round(
                    max(0.01, min(0.03, world_model_failure_signal / max(1, sum(failure_counts.values())))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "repo_world_model_total": repo_world_model_total,
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if (
        transition_failure_counts
        or int(transition_summary.get("state_regression_steps", 0) or 0) > 0
        or low_confidence_state_signal > 0
    ):
        state_estimation_signal = (
            transition_failure_counts.get("no_state_progress", 0)
            + transition_failure_counts.get("state_regression", 0)
            + int(transition_summary.get("state_regression_steps", 0) or 0)
            + low_confidence_state_signal
        )
        state_estimation_reason = (
            "low-confidence routing should first be repaired through runtime state summarization and recovery cues before cross-cycle weight updates"
            if low_confidence_state_signal > 0
            else "state summarization should separate stalls, regressions, and recovery opportunities more explicitly before policy scoring"
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="state_estimation",
                reason=state_estimation_reason,
                priority=4,
                expected_gain=round(
                    max(
                        0.015,
                        min(
                            0.05,
                            state_estimation_signal
                            / max(1, (sum(transition_failure_counts.values()) or 0) + low_confidence_state_signal),
                        ),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                    "total": metrics.total,
                    "runtime_learning_priority": low_confidence_state_signal > 0,
                },
            )
        )
    universe_constitution_signal = (
        failure_counts.get("command_failure", 0)
        + transition_failure_counts.get("no_state_progress", 0)
        + transition_failure_counts.get("state_regression", 0)
        + metrics.low_confidence_episodes
    )
    environment_violation_summary = planner.environment_violation_summary()
    universe_cycle_feedback = planner.universe_cycle_feedback_summary()
    operating_envelope_signal = (
        int(environment_violation_summary.get("violation_total", 0))
        + int(environment_violation_summary.get("alignment_failure_total", 0))
    )
    broad_support_bonus = min(
        2,
        int(
            universe_cycle_feedback.get(
                "broad_support_cycle_count",
                universe_cycle_feedback.get("retained_cycle_count", 0),
            )
        ),
    )
    universe_constitution_signal += min(
        1,
        int(universe_cycle_feedback.get("constitution_retained_cycle_count", 0) or 0),
    )
    operating_envelope_signal += broad_support_bonus
    operating_envelope_signal += min(
        1,
        int(universe_cycle_feedback.get("operating_envelope_retained_cycle_count", 0) or 0),
    )
    if universe_constitution_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="universe_constitution",
                reason="constitutional machine-law should tighten verifier, bounded-action, and destructive-reset rules above task-local world state",
                priority=4,
                expected_gain=round(
                    max(0.012, min(0.025, universe_constitution_signal / max(1, metrics.total or 1))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "universe_cycle_feedback": universe_cycle_feedback,
                    "total": metrics.total,
                },
            )
        )
    if operating_envelope_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="operating_envelope",
                reason="the retained operating envelope should calibrate to repeated environment conflicts and attested runtime mismatch",
                priority=4,
                expected_gain=round(
                    max(0.015, min(0.03, operating_envelope_signal / max(1, metrics.total or 1))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "environment_violation_summary": environment_violation_summary,
                    "universe_cycle_feedback": universe_cycle_feedback,
                    "total": metrics.total,
                },
            )
        )
    if transition_failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="transition_model",
                reason="retained bad-transition signatures should directly penalize repeated stalled or regressive commands",
                priority=5,
                expected_gain=round(
                    max(
                        0.02,
                        min(
                            0.05,
                            (
                                transition_failure_counts.get("no_state_progress", 0)
                                + transition_failure_counts.get("state_regression", 0)
                            )
                            / max(1, sum(transition_failure_counts.values())),
                        ),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if trust_summary and not broad_observe_curriculum_priority and (
        str(trust_summary.get("overall_status", "")).strip() in {"bootstrap", "restricted"}
        or float(trust_summary.get("unsafe_ambiguous_rate", 0.0)) > 0.0
        or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("false_pass_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("unexpected_change_report_rate", 0.0)) > 0.0
        or int(trust_summary.get("distinct_benchmark_families", 0)) < 2
        or int(trust_summary.get("distinct_family_gap", 0)) > 0
        or list(trust_summary.get("missing_required_families", []))
        or list(trust_summary.get("missing_required_family_clean_task_root_breadth", []))
    ):
        clean_success_rate = float(trust_summary.get("clean_success_rate", 0.0))
        false_pass_risk_rate = float(trust_summary.get("false_pass_risk_rate", 0.0))
        hidden_side_effect_risk_rate = float(trust_summary.get("hidden_side_effect_risk_rate", 0.0))
        unexpected_change_report_rate = float(trust_summary.get("unexpected_change_report_rate", 0.0))
        missing_required_task_root_breadth = list(
            trust_summary.get("missing_required_family_clean_task_root_breadth", [])
        )
        breadth_threshold = int(trust_summary.get("family_breadth_min_distinct_task_roots", 0) or 0)
        breadth_gap = 0.0
        if breadth_threshold > 0 and missing_required_task_root_breadth:
            counts = trust_summary.get("required_family_clean_task_root_counts", {})
            if not isinstance(counts, dict):
                counts = {}
            deficits = [
                max(0, breadth_threshold - int(counts.get(family, 0) or 0))
                for family in missing_required_task_root_breadth
            ]
            breadth_gap = min(0.08, sum(deficits) * 0.02)
        false_pass_contamination = max(0.0, false_pass_risk_rate - max(0.0, clean_success_rate))
        trust_risk_signal = max(
            float(trust_summary.get("unsafe_ambiguous_rate", 0.0)),
            hidden_side_effect_risk_rate,
            float(trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)),
            false_pass_risk_rate,
            unexpected_change_report_rate,
            false_pass_contamination,
            breadth_gap,
            min(
                0.08,
                (int(trust_summary.get("distinct_family_gap", 0) or 0) * 0.01)
                + (len(list(trust_summary.get("missing_required_families", []))) * 0.005),
            ),
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="trust",
                reason="unattended trust gating remains restricted, coverage-misaligned, or exposed to hidden-risk and false-pass outcomes",
                priority=4,
                expected_gain=round(max(0.01, trust_risk_signal or 0.02), 4),
                estimated_cost=2,
                score=0.0,
                evidence=trust_summary,
            )
        )
    if trust_summary and (
        float(trust_summary.get("rollback_performed_rate", 0.0)) > 0.0
        or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("false_pass_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("unexpected_change_report_rate", 0.0)) > 0.0
    ):
        rollback_performed_rate = float(trust_summary.get("rollback_performed_rate", 0.0))
        hidden_side_effect_risk_rate = float(trust_summary.get("hidden_side_effect_risk_rate", 0.0))
        success_hidden_side_effect_risk_rate = float(
            trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)
        )
        false_pass_risk_rate = float(trust_summary.get("false_pass_risk_rate", 0.0))
        unexpected_change_report_rate = float(trust_summary.get("unexpected_change_report_rate", 0.0))
        recovery_signal = max(
            rollback_performed_rate,
            hidden_side_effect_risk_rate,
            success_hidden_side_effect_risk_rate,
            false_pass_risk_rate,
            unexpected_change_report_rate,
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="recovery",
                reason="unattended runs still depend on rollback or leave residual side-effect uncertainty after restore paths",
                priority=4,
                expected_gain=round(max(0.01, recovery_signal), 4),
                estimated_cost=2,
                score=0.0,
                evidence=trust_summary,
            )
        )
    delegation_summary = planner.delegation_policy_summary()
    if delegation_summary and (
        int(delegation_summary.get("delegated_job_max_concurrency", 1)) < 3
        or int(delegation_summary.get("delegated_job_max_active_per_budget_group", 0)) < 2
        or int(delegation_summary.get("delegated_job_max_queued_per_budget_group", 0)) < 8
        or int(delegation_summary.get("delegated_job_max_subprocesses_per_job", 1)) < 2
        or int(delegation_summary.get("max_steps", 5)) < 12
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="delegation",
                reason="delegated execution remains throttled by shallow worker budgets or narrow queue policy",
                priority=4,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=delegation_summary,
            )
        )
    operator_policy_summary = planner.operator_policy_summary()
    if operator_policy_summary and (
        len(list(operator_policy_summary.get("unattended_allowed_benchmark_families", []))) < 5
        or not bool(operator_policy_summary.get("unattended_allow_git_commands", False))
        or not bool(operator_policy_summary.get("unattended_allow_http_requests", False))
        or not bool(operator_policy_summary.get("unattended_allow_generated_path_mutations", False))
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="operator_policy",
                reason="unattended operator-boundary policy still limits family breadth or critical execution scopes",
                priority=4,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=operator_policy_summary,
            )
        )
    capability_summary = planner.capability_surface_summary()
    if capability_summary and (
        int(capability_summary.get("enabled_module_count", 0)) == 0
        or int(capability_summary.get("improvement_surface_count", 0)) == 0
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="capabilities",
                reason="capability registry does not yet expose retained autonomous improvement surfaces",
                priority=3,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=capability_summary,
            )
        )
    external_subsystems = {candidate.subsystem for candidate in candidates}
    for external in planner._plugin_layer.external_experiments(planner.capability_modules_path):
        subsystem = str(external.get("subsystem", "")).strip()
        if not subsystem or subsystem in external_subsystems:
            continue
        candidates.append(
            ImprovementExperiment(
                subsystem=subsystem,
                reason=str(external.get("reason", "")).strip() or "module-defined improvement surface",
                priority=int(external.get("priority", 3)),
                expected_gain=float(external.get("expected_gain", 0.01)),
                estimated_cost=int(external.get("estimated_cost", 2)),
                score=0.0,
                evidence=dict(external.get("evidence", {}))
                if isinstance(external.get("evidence", {}), dict)
                else {},
            )
        )
        external_subsystems.add(subsystem)
    candidates.append(
        ImprovementExperiment(
            subsystem="policy",
            reason="no sharper subsystem deficit dominates current metrics",
            priority=3,
            expected_gain=0.01,
            estimated_cost=2,
            score=0.0,
            evidence={"default_fallback": True},
        )
    )
    planner_controls = planner._improvement_planner_controls()
    learning_candidate_summary = planner._learning_candidate_summary()
    scored_candidates = [
        planner._score_experiment(
            candidate,
            metrics=metrics,
            planner_controls=planner_controls,
            learning_candidate_summary=learning_candidate_summary,
            trust_summary=trust_summary,
        )
        for candidate in candidates
    ]
    return engine_sort_experiments(scored_candidates)


__all__ = ["rank_experiments"]
