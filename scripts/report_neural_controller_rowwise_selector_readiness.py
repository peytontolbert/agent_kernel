from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_neural_controller_rowwise_frontier import _read_json_object


def _summary_counts(report: dict[str, Any]) -> dict[str, int]:
    summary = report.get("summary", report.get("frontier_summary", {}))
    if not isinstance(summary, dict):
        summary = {}
    total = int(summary.get("content_comparison_steps", 0) or 0)
    return {
        "total": total,
        "strict": int(summary.get("content_exact_agreement_steps", 0) or 0),
        "contract": int(summary.get("contract_content_agreement_steps", 0) or 0),
    }


def _source_switch_count(report: dict[str, Any], baseline_label: str) -> int:
    counts = report.get("source_counts", {})
    if not isinstance(counts, dict):
        return 0
    total = sum(int(value or 0) for value in counts.values())
    baseline = int(counts.get(baseline_label, 0) or 0)
    return max(0, total - baseline)


def report_selector_readiness(
    *,
    baseline_report_path: Path,
    selector_report_path: Path,
    retained_report_path: Path,
    output_path: Path,
    baseline_label: str = "baseline",
) -> dict[str, Any]:
    baseline = _read_json_object(baseline_report_path)
    selector = _read_json_object(selector_report_path)
    retained = _read_json_object(retained_report_path)
    baseline_counts = _summary_counts(baseline)
    selector_counts = _summary_counts(selector)
    retained_counts = _summary_counts(retained)
    strict_gap_before = retained_counts["strict"] - baseline_counts["strict"]
    strict_gap_after = retained_counts["strict"] - selector_counts["strict"]
    contract_gap_before = retained_counts["contract"] - baseline_counts["contract"]
    contract_gap_after = retained_counts["contract"] - selector_counts["contract"]
    report = {
        "report_kind": "neural_controller_rowwise_selector_readiness",
        "baseline_report_path": str(baseline_report_path),
        "selector_report_path": str(selector_report_path),
        "retained_report_path": str(retained_report_path),
        "baseline_label": baseline_label,
        "baseline": baseline_counts,
        "selector": selector_counts,
        "retained": retained_counts,
        "strict_gain": selector_counts["strict"] - baseline_counts["strict"],
        "contract_gain": selector_counts["contract"] - baseline_counts["contract"],
        "strict_gap_before": strict_gap_before,
        "strict_gap_after": strict_gap_after,
        "contract_gap_before": contract_gap_before,
        "contract_gap_after": contract_gap_after,
        "reaches_retained_strict": strict_gap_after <= 0,
        "reaches_retained_contract": contract_gap_after <= 0,
        "source_switch_count": _source_switch_count(selector, baseline_label),
        "primary_authority_ready": False,
        "recommendation": (
            "promote_selector_to_retained_candidate_packet"
            if strict_gap_after <= 0 and contract_gap_after <= 0
            else "continue_selector_hardening"
        ),
        "authority_note": (
            "Selector reaches the current retained comparison surface, but primary authority still requires "
            "a retained promotion gate and verifier-backed runtime integration."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--retained-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    args = parser.parse_args()
    report = report_selector_readiness(
        baseline_report_path=Path(args.baseline_report),
        selector_report_path=Path(args.selector_report),
        retained_report_path=Path(args.retained_report),
        output_path=Path(args.output),
        baseline_label=str(args.baseline_label),
    )
    print(
        "neural_controller_rowwise_selector_readiness "
        f"strict={report['selector']['strict']}/{report['selector']['total']} "
        f"contract={report['selector']['contract']}/{report['selector']['total']} "
        f"strict_gap_after={report['strict_gap_after']} "
        f"contract_gap_after={report['contract_gap_after']} "
        f"recommendation={report['recommendation']}"
    )


if __name__ == "__main__":
    main()
