from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("content_exact_rate", "exec_kind_agreement_rate")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _family_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("family_metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return metrics


def _summary_content_rate(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return 0.0
    return float(summary.get("content_exact_agreement_rate", 0.0) or 0.0)


def _rate(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def compare_family_metrics(
    *,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    min_family_total: int = 3,
    tolerance: float = 0.0,
    require_overall_non_regression: bool = True,
    require_macro_non_regression: bool = True,
) -> dict[str, Any]:
    baseline_families = _family_metrics(baseline_report)
    candidate_families = _family_metrics(candidate_report)
    family_names = sorted(
        family
        for family, row in baseline_families.items()
        if not family.startswith("_") and int(row.get("total", 0) or 0) >= min_family_total
    )
    family_deltas: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for family in family_names:
        baseline_row = baseline_families.get(family, {})
        candidate_row = candidate_families.get(family, {})
        if not isinstance(baseline_row, dict):
            baseline_row = {}
        if not isinstance(candidate_row, dict):
            candidate_row = {}
        row_delta = {
            "baseline_total": int(baseline_row.get("total", 0) or 0),
            "candidate_total": int(candidate_row.get("total", 0) or 0),
        }
        for key in METRIC_KEYS:
            baseline_rate = _rate(baseline_row, key)
            candidate_rate = _rate(candidate_row, key)
            delta = round(candidate_rate - baseline_rate, 6)
            row_delta[key] = {
                "baseline": baseline_rate,
                "candidate": candidate_rate,
                "delta": delta,
            }
            if delta < -abs(tolerance):
                regressions.append(
                    {
                        "scope": "family",
                        "family": family,
                        "metric": key,
                        "baseline": baseline_rate,
                        "candidate": candidate_rate,
                        "delta": delta,
                    }
                )
            elif delta > abs(tolerance):
                improvements.append(
                    {
                        "scope": "family",
                        "family": family,
                        "metric": key,
                        "baseline": baseline_rate,
                        "candidate": candidate_rate,
                        "delta": delta,
                    }
                )
        family_deltas[family] = row_delta

    baseline_summary_rate = _summary_content_rate(baseline_report)
    candidate_summary_rate = _summary_content_rate(candidate_report)
    summary_delta = round(candidate_summary_rate - baseline_summary_rate, 6)
    if require_overall_non_regression and summary_delta < -abs(tolerance):
        regressions.append(
            {
                "scope": "summary",
                "metric": "content_exact_agreement_rate",
                "baseline": baseline_summary_rate,
                "candidate": candidate_summary_rate,
                "delta": summary_delta,
            }
        )

    macro_deltas: dict[str, Any] = {}
    baseline_macro = baseline_families.get("_macro", {})
    candidate_macro = candidate_families.get("_macro", {})
    if not isinstance(baseline_macro, dict):
        baseline_macro = {}
    if not isinstance(candidate_macro, dict):
        candidate_macro = {}
    for key in ("macro_content_exact_rate", "macro_exec_kind_agreement_rate"):
        baseline_rate = _rate(baseline_macro, key)
        candidate_rate = _rate(candidate_macro, key)
        delta = round(candidate_rate - baseline_rate, 6)
        macro_deltas[key] = {
            "baseline": baseline_rate,
            "candidate": candidate_rate,
            "delta": delta,
        }
        if require_macro_non_regression and delta < -abs(tolerance):
            regressions.append(
                {
                    "scope": "macro",
                    "metric": key,
                    "baseline": baseline_rate,
                    "candidate": candidate_rate,
                    "delta": delta,
                }
            )

    accepted = not regressions
    if not family_names:
        accepted = False
        regressions.append(
            {
                "scope": "gate",
                "metric": "family_coverage",
                "reason": "no baseline families met min_family_total",
            }
        )

    return {
        "report_kind": "neural_controller_family_metric_compare",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "min_family_total": int(min_family_total),
        "tolerance": float(tolerance),
        "accepted": accepted,
        "recommendation": "accept_candidate" if accepted else "keep_baseline",
        "summary_delta": {
            "content_exact_agreement_rate": {
                "baseline": baseline_summary_rate,
                "candidate": candidate_summary_rate,
                "delta": summary_delta,
            }
        },
        "macro_deltas": macro_deltas,
        "family_deltas": family_deltas,
        "regressions": regressions,
        "improvements": improvements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--min-family-total", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--allow-summary-regression", action="store_true")
    parser.add_argument("--allow-macro-regression", action="store_true")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_report)
    candidate_path = Path(args.candidate_report)
    comparison = compare_family_metrics(
        baseline_report=_read_json_object(baseline_path),
        candidate_report=_read_json_object(candidate_path),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        min_family_total=args.min_family_total,
        tolerance=args.tolerance,
        require_overall_non_regression=not args.allow_summary_regression,
        require_macro_non_regression=not args.allow_macro_regression,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "neural_controller_family_metric_compare "
        f"baseline={args.baseline_label} "
        f"candidate={args.candidate_label} "
        f"accepted={str(comparison['accepted']).lower()} "
        f"regressions={len(comparison['regressions'])} "
        f"improvements={len(comparison['improvements'])}"
    )


if __name__ == "__main__":
    main()
