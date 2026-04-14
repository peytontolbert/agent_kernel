from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.trust import build_unattended_trust_ledger, write_unattended_trust_ledger

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


def _load_reports(root: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("report_kind", "")).strip() != "unattended_task_report":
            continue
        reports.append(payload)
    return reports


def _bool_value(payload: dict[str, object], path: tuple[str, ...]) -> bool:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def _int_value(payload: dict[str, object], path: tuple[str, ...]) -> int:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    return int(current) if isinstance(current, int) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--benchmark-family", action="append", default=None)
    parser.add_argument("--write-ledger", action="store_true")
    parser.add_argument("--ledger-path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    reports_dir = Path(args.reports_dir) if args.reports_dir else config.run_reports_dir
    families = set(args.benchmark_family or [])
    reports = []
    for report in _load_reports(reports_dir):
        family = str(report.get("benchmark_family", "bounded")).strip() or "bounded"
        if families and family not in families:
            continue
        reports.append(report)

    outcomes = Counter()
    family_totals: Counter[str] = Counter()
    family_hidden_risk: Counter[str] = Counter()
    family_success_hidden_risk: Counter[str] = Counter()
    family_external_totals: Counter[str] = Counter()
    family_semantic_hub_totals: Counter[str] = Counter()
    family_replay_derived_totals: Counter[str] = Counter()
    hidden_side_effect_risk_count = 0
    success_hidden_side_effect_risk_count = 0
    false_pass_risk_count = 0
    unexpected_change_files_total = 0
    unexpected_change_report_count = 0
    acceptance_packet_count = 0
    synthetic_worker_count = 0
    selected_edit_total = 0
    candidate_edit_set_total = 0
    required_merged_branch_total = 0
    test_command_total = 0

    for report in reports:
        family = str(report.get("benchmark_family", "bounded")).strip() or "bounded"
        task_metadata = report.get("task_metadata", {})
        task_origin = (
            str(task_metadata.get("task_origin", "")).strip()
            if isinstance(task_metadata, dict)
            else ""
        ) or "built_in"
        outcome = str(report.get("outcome", "unknown")).strip() or "unknown"
        hidden_risk = _bool_value(report, ("side_effects", "hidden_side_effect_risk"))
        success = bool(report.get("success", False))
        unexpected_change_files = _int_value(report, ("summary", "unexpected_change_files"))
        acceptance_packet = report.get("acceptance_packet", {})
        if not isinstance(acceptance_packet, dict):
            acceptance_packet = {}

        outcomes[outcome] += 1
        family_totals[family] += 1
        if task_origin == "external_manifest":
            family_external_totals[family] += 1
        if task_origin == "semantic_hub":
            family_semantic_hub_totals[family] += 1
        if task_origin in _REPLAY_DERIVED_TASK_ORIGINS:
            family_replay_derived_totals[family] += 1
        unexpected_change_files_total += unexpected_change_files
        if unexpected_change_files > 0:
            unexpected_change_report_count += 1
        if acceptance_packet:
            acceptance_packet_count += 1
            synthetic_worker_count += int(bool(acceptance_packet.get("synthetic_worker", False)))
            selected_edit_total += len(acceptance_packet.get("selected_edits", [])) if isinstance(
                acceptance_packet.get("selected_edits", []), list
            ) else 0
            candidate_edit_set_total += len(acceptance_packet.get("candidate_edit_sets", [])) if isinstance(
                acceptance_packet.get("candidate_edit_sets", []), list
            ) else 0
            required_merged_branch_total += len(acceptance_packet.get("required_merged_branches", [])) if isinstance(
                acceptance_packet.get("required_merged_branches", []), list
            ) else 0
            test_command_total += len(acceptance_packet.get("tests", [])) if isinstance(
                acceptance_packet.get("tests", []), list
            ) else 0
        if hidden_risk:
            hidden_side_effect_risk_count += 1
            family_hidden_risk[family] += 1
        if hidden_risk and success:
            success_hidden_side_effect_risk_count += 1
            family_success_hidden_risk[family] += 1
        if success and (hidden_risk or unexpected_change_files > 0):
            false_pass_risk_count += 1

    print(f"reports_total={len(reports)}")
    print(f"hidden_side_effect_risk_count={hidden_side_effect_risk_count}")
    print(f"success_hidden_side_effect_risk_count={success_hidden_side_effect_risk_count}")
    print(f"false_pass_risk_count={false_pass_risk_count}")
    print(f"unexpected_change_files_total={unexpected_change_files_total}")
    print(f"unexpected_change_report_count={unexpected_change_report_count}")
    print(f"acceptance_packet_count={acceptance_packet_count}")
    print(f"synthetic_worker_count={synthetic_worker_count}")
    print(f"selected_edit_total={selected_edit_total}")
    print(f"candidate_edit_set_total={candidate_edit_set_total}")
    print(f"required_merged_branch_total={required_merged_branch_total}")
    print(f"test_command_total={test_command_total}")
    print(f"external_report_count={sum(family_external_totals.values())}")
    print(f"external_distinct_families={len(family_external_totals)}")
    print(f"semantic_hub_report_count={sum(family_semantic_hub_totals.values())}")
    print(f"semantic_hub_distinct_families={len(family_semantic_hub_totals)}")
    print(f"replay_derived_report_count={sum(family_replay_derived_totals.values())}")
    print(f"replay_derived_distinct_families={len(family_replay_derived_totals)}")
    for outcome in sorted(outcomes):
        print(f"outcome_count outcome={outcome} count={outcomes[outcome]}")
    for family in sorted(family_totals):
        print(
            f"benchmark_family={family} total={family_totals[family]} "
            f"external_total={family_external_totals.get(family, 0)} "
            f"semantic_hub_total={family_semantic_hub_totals.get(family, 0)} "
            f"replay_derived_total={family_replay_derived_totals.get(family, 0)} "
            f"hidden_side_effect_risk={family_hidden_risk.get(family, 0)} "
            f"success_hidden_side_effect_risk={family_success_hidden_risk.get(family, 0)}"
        )

    ledger = build_unattended_trust_ledger(config, reports_dir=reports_dir)
    overall = ledger["overall_summary"]
    gated = ledger["gated_summary"]
    overall_assessment = ledger["overall_assessment"]
    print(
        "trust_overall "
        f"status={overall_assessment.get('status', '')} "
        f"reports={gated.get('total', 0)} "
        f"success_rate={float(gated.get('success_rate', 0.0)):.3f} "
        f"unsafe_ambiguous_rate={float(gated.get('unsafe_ambiguous_rate', 0.0)):.3f} "
        f"hidden_side_effect_risk_rate={float(gated.get('hidden_side_effect_risk_rate', 0.0)):.3f} "
        f"false_pass_risk_rate={float(gated.get('false_pass_risk_rate', 0.0)):.3f} "
        f"clean_success_streak={int(gated.get('clean_success_streak', 0))} "
        f"distinct_families={int(gated.get('distinct_benchmark_families', 0))}"
    )
    success_ci = overall.get("success_rate_confidence_interval", {})
    if not isinstance(success_ci, dict):
        success_ci = {}
    unsafe_ci = overall.get("unsafe_ambiguous_rate_confidence_interval", {})
    if not isinstance(unsafe_ci, dict):
        unsafe_ci = {}
    hidden_ci = overall.get("hidden_side_effect_risk_rate_confidence_interval", {})
    if not isinstance(hidden_ci, dict):
        hidden_ci = {}
    print(
        "trust_confidence "
        f"success_rate_low={float(success_ci.get('lower', 0.0)):.3f} "
        f"success_rate_high={float(success_ci.get('upper', 0.0)):.3f} "
        f"unsafe_ambiguous_rate_low={float(unsafe_ci.get('lower', 0.0)):.3f} "
        f"unsafe_ambiguous_rate_high={float(unsafe_ci.get('upper', 0.0)):.3f} "
        f"hidden_side_effect_risk_rate_low={float(hidden_ci.get('lower', 0.0)):.3f} "
        f"hidden_side_effect_risk_rate_high={float(hidden_ci.get('upper', 0.0)):.3f}"
    )
    print(
        "trust_coverage "
        f"reports={overall.get('total', 0)} "
        f"families={','.join(str(name) for name in overall.get('benchmark_families', []))} "
        f"external_reports={int(overall.get('external_report_count', 0))} "
        f"external_families={','.join(str(name) for name in overall.get('external_benchmark_families', []))}"
    )
    task_yield_bucket_summary = overall.get("task_yield_bucket_summary", {})
    if not isinstance(task_yield_bucket_summary, dict):
        task_yield_bucket_summary = {}
    print(
        "trust_task_yield "
        f"semantic_hub_reports={int(dict(task_yield_bucket_summary.get('semantic_hub', {})).get('reports', 0) or 0)} "
        f"external_manifest_reports={int(dict(task_yield_bucket_summary.get('external_manifest', {})).get('reports', 0) or 0)} "
        f"replay_derived_reports={int(dict(task_yield_bucket_summary.get('replay_derived', {})).get('reports', 0) or 0)}"
    )
    failure_recovery_summary = overall.get("failure_recovery_summary", {})
    if not isinstance(failure_recovery_summary, dict):
        failure_recovery_summary = {}
    failure_recovery_success_ci = failure_recovery_summary.get("success_rate_confidence_interval", {})
    if not isinstance(failure_recovery_success_ci, dict):
        failure_recovery_success_ci = {}
    failure_recovery_clean_success_ci = failure_recovery_summary.get("clean_success_rate_confidence_interval", {})
    if not isinstance(failure_recovery_clean_success_ci, dict):
        failure_recovery_clean_success_ci = {}
    print(
        "trust_failure_recovery "
        f"reports={int(failure_recovery_summary.get('reports', 0) or 0)} "
        f"successes={int(failure_recovery_summary.get('success_count', 0) or 0)} "
        f"clean_successes={int(failure_recovery_summary.get('clean_success_count', 0) or 0)} "
        f"success_rate={float(failure_recovery_summary.get('success_rate', 0.0)):.3f} "
        f"success_rate_low={float(failure_recovery_success_ci.get('lower', 0.0)):.3f} "
        f"success_rate_high={float(failure_recovery_success_ci.get('upper', 0.0)):.3f} "
        f"clean_success_rate={float(failure_recovery_summary.get('clean_success_rate', 0.0)):.3f} "
        f"clean_success_rate_low={float(failure_recovery_clean_success_ci.get('lower', 0.0)):.3f} "
        f"clean_success_rate_high={float(failure_recovery_clean_success_ci.get('upper', 0.0)):.3f}"
    )
    coverage = ledger.get("coverage_summary", {})
    if not isinstance(coverage, dict):
        coverage = {}
    print(
        "trust_required_coverage "
        f"required={','.join(str(name) for name in coverage.get('required_families', []))} "
        f"observed={','.join(str(name) for name in coverage.get('observed_families', []))} "
        f"with_reports={','.join(str(name) for name in coverage.get('required_families_with_reports', []))} "
        f"missing={','.join(str(name) for name in coverage.get('missing_required_families', []))} "
        f"passing={','.join(str(name) for name in coverage.get('passing_required_families', []))} "
        f"restricted={','.join(str(name) for name in coverage.get('restricted_required_families', []))} "
        f"distinct_family_gap={int(coverage.get('distinct_family_gap', 0) or 0)}"
    )
    required_families = {
        str(family).strip()
        for family in ledger.get("policy", {}).get("required_benchmark_families", [])
        if str(family).strip()
    }
    for family in sorted(set(required_families) | set(ledger.get("family_summaries", {}))):
        summary = ledger.get("family_summaries", {}).get(family, {})
        assessment = ledger.get("family_assessments", {}).get(family, {})
        print(
            "trust_family "
            f"family={family} "
            f"required={int(family in required_families)} "
            f"status={assessment.get('status', 'absent')} "
            f"reports={int(summary.get('total', 0))}"
        )
    for family in sorted(required_families):
        summary = ledger.get("family_summaries", {}).get(family, {})
        assessment = ledger.get("family_assessments", {}).get(family, {})
        print(
            "trust_required_family "
            f"family={family} "
            f"reports={int(summary.get('total', 0) if isinstance(summary, dict) else 0)} "
            f"status={assessment.get('status', 'absent') if isinstance(assessment, dict) else 'absent'} "
            f"passed={int(bool(assessment.get('passed', False)) if isinstance(assessment, dict) else False)} "
            f"success_rate={float(summary.get('success_rate', 0.0) if isinstance(summary, dict) else 0.0):.3f} "
            f"unsafe_ambiguous_rate={float(summary.get('unsafe_ambiguous_rate', 0.0) if isinstance(summary, dict) else 0.0):.3f} "
            f"hidden_side_effect_risk_rate={float(summary.get('hidden_side_effect_risk_rate', 0.0) if isinstance(summary, dict) else 0.0):.3f} "
            f"false_pass_risk_rate={float(summary.get('false_pass_risk_rate', 0.0) if isinstance(summary, dict) else 0.0):.3f}"
        )
    if args.write_ledger:
        ledger_target = write_unattended_trust_ledger(
            config,
            reports_dir=reports_dir,
            ledger_path=Path(args.ledger_path) if args.ledger_path else None,
        )
        print(f"trust_ledger={ledger_target}")
    if args.json:
        print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
