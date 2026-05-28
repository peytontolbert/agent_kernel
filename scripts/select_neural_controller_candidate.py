from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_neural_controller_family_metrics import compare_family_metrics


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _summary_rate(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return 0.0
    return float(summary.get("content_exact_agreement_rate", 0.0) or 0.0)


def _summary_exact(report: dict[str, Any]) -> int:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return 0
    return int(summary.get("content_exact_agreement_steps", 0) or 0)


def _macro(report: dict[str, Any], key: str) -> float:
    families = report.get("family_metrics", {})
    if not isinstance(families, dict):
        return 0.0
    macro = families.get("_macro", {})
    if not isinstance(macro, dict):
        return 0.0
    return float(macro.get(key, 0.0) or 0.0)


def _regression_penalty(comparison: dict[str, Any]) -> float:
    penalty = 0.0
    for regression in comparison.get("regressions", []):
        if not isinstance(regression, dict):
            continue
        delta = abs(float(regression.get("delta", 0.0) or 0.0))
        scope = str(regression.get("scope", ""))
        if scope == "family":
            penalty += 10.0 * delta
        elif scope == "macro":
            penalty += 5.0 * delta
        else:
            penalty += delta
    return round(penalty, 6)


def _diagnostic_score(
    *,
    report: dict[str, Any],
    comparison: dict[str, Any],
    baseline_report: dict[str, Any],
) -> float:
    baseline_rate = _summary_rate(baseline_report)
    summary_gain = _summary_rate(report) - baseline_rate
    macro_content_gain = _macro(report, "macro_content_exact_rate") - _macro(
        baseline_report, "macro_content_exact_rate"
    )
    macro_exec_gain = _macro(report, "macro_exec_kind_agreement_rate") - _macro(
        baseline_report, "macro_exec_kind_agreement_rate"
    )
    score = summary_gain + 0.5 * macro_content_gain + 0.25 * macro_exec_gain
    score -= _regression_penalty(comparison)
    return round(score, 6)


def select_candidate(
    *,
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    baseline_label: str = "baseline",
    min_family_total: int = 3,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    baseline_report = _read_json_object(baseline_report_path)
    rows: list[dict[str, Any]] = []
    for candidate_path in candidate_report_paths:
        candidate_report = _read_json_object(candidate_path)
        candidate_label = candidate_path.stem.replace("_slot_eval132_shadow_report", "")
        comparison = compare_family_metrics(
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            min_family_total=min_family_total,
            tolerance=tolerance,
        )
        rows.append(
            {
                "candidate_label": candidate_label,
                "candidate_report_path": str(candidate_path),
                "accepted": bool(comparison.get("accepted", False)),
                "summary_exact": _summary_exact(candidate_report),
                "summary_rate": _summary_rate(candidate_report),
                "macro_content_exact_rate": _macro(candidate_report, "macro_content_exact_rate"),
                "macro_exec_kind_agreement_rate": _macro(candidate_report, "macro_exec_kind_agreement_rate"),
                "regression_count": len(comparison.get("regressions", [])),
                "improvement_count": len(comparison.get("improvements", [])),
                "regression_penalty": _regression_penalty(comparison),
                "diagnostic_score": _diagnostic_score(
                    report=candidate_report,
                    comparison=comparison,
                    baseline_report=baseline_report,
                ),
                "comparison": comparison,
            }
        )

    accepted_rows = sorted(
        [row for row in rows if row["accepted"]],
        key=lambda row: (
            float(row["summary_rate"]),
            float(row["macro_content_exact_rate"]),
            float(row["macro_exec_kind_agreement_rate"]),
        ),
        reverse=True,
    )
    diagnostic_rows = sorted(
        rows,
        key=lambda row: (
            float(row["diagnostic_score"]),
            float(row["summary_rate"]),
            float(row["macro_exec_kind_agreement_rate"]),
        ),
        reverse=True,
    )
    selected = accepted_rows[0] if accepted_rows else None
    return {
        "report_kind": "neural_controller_candidate_selection",
        "baseline_label": baseline_label,
        "baseline_report_path": str(baseline_report_path),
        "min_family_total": int(min_family_total),
        "tolerance": float(tolerance),
        "accepted_candidate_label": selected["candidate_label"] if selected else "",
        "accepted_candidate_report_path": selected["candidate_report_path"] if selected else "",
        "strict_recommendation": "accept_candidate" if selected else "keep_baseline",
        "best_diagnostic_candidate_label": diagnostic_rows[0]["candidate_label"] if diagnostic_rows else "",
        "best_diagnostic_candidate_report_path": diagnostic_rows[0]["candidate_report_path"] if diagnostic_rows else "",
        "candidates": rows,
        "accepted_rank": accepted_rows,
        "diagnostic_rank": diagnostic_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--min-family-total", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    if not args.candidate_report:
        raise ValueError("at least one --candidate-report is required")
    selection = select_candidate(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_paths=[Path(path) for path in args.candidate_report],
        baseline_label=args.baseline_label,
        min_family_total=args.min_family_total,
        tolerance=args.tolerance,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "neural_controller_candidate_selection "
        f"strict_recommendation={selection['strict_recommendation']} "
        f"accepted={selection['accepted_candidate_label'] or 'none'} "
        f"best_diagnostic={selection['best_diagnostic_candidate_label'] or 'none'} "
        f"candidates={len(selection['candidates'])}"
    )


if __name__ == "__main__":
    main()
