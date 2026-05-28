from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import (
    EXEC_KIND_FAMILY,
    neural_controller_shadow_promotion_readiness,
    summarize_neural_controller_shadow_documents,
)
from scripts.compare_neural_controller_family_metrics import compare_family_metrics
from scripts.evaluate_neural_controller_shadow_dataset import summarize_family_metrics


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _shadow_by_example(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    documents = report.get("documents", [])
    if not isinstance(documents, list):
        return out
    for document in documents:
        if not isinstance(document, dict):
            continue
        steps = document.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        metadata = steps[0].get("proposal_metadata", {}) if isinstance(steps[0], dict) else {}
        if not isinstance(metadata, dict):
            continue
        shadow = metadata.get("neural_controller_shadow", {})
        if not isinstance(shadow, dict):
            continue
        example_id = str(shadow.get("example_id") or document.get("task_id") or "").strip()
        if example_id:
            out[example_id] = dict(shadow)
    return out


def _fallback_families_from_comparison(comparison: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for regression in comparison.get("regressions", []):
        if not isinstance(regression, dict):
            continue
        if str(regression.get("scope", "")) != "family":
            continue
        family = str(regression.get("family", "")).strip()
        if family:
            families.add(family)
    return families


def _family(shadow: dict[str, Any]) -> str:
    return EXEC_KIND_FAMILY.get(str(shadow.get("target_exec_kind", "")).strip(), "unknown")


def compose_guarded_report(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    output_path: Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    min_family_total: int = 3,
    tolerance: float = 0.0,
    fallback_family: tuple[str, ...] = (),
) -> dict[str, Any]:
    baseline_report = _read_json_object(baseline_report_path)
    candidate_report = _read_json_object(candidate_report_path)
    comparison = compare_family_metrics(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        min_family_total=min_family_total,
        tolerance=tolerance,
    )
    fallback_families = _fallback_families_from_comparison(comparison)
    fallback_families.update(str(item).strip() for item in fallback_family if str(item).strip())

    baseline_by_id = _shadow_by_example(baseline_report)
    candidate_by_id = _shadow_by_example(candidate_report)
    documents: list[dict[str, Any]] = []
    source_counts = {"baseline": 0, "candidate": 0}
    missing_candidate = 0
    for example_id, baseline_shadow in sorted(baseline_by_id.items()):
        candidate_shadow = candidate_by_id.get(example_id)
        family = _family(baseline_shadow)
        use_baseline = family in fallback_families or candidate_shadow is None
        if candidate_shadow is None:
            missing_candidate += 1
        selected_shadow = dict(baseline_shadow if use_baseline else candidate_shadow)
        selected_shadow["guarded_source"] = "baseline" if use_baseline else "candidate"
        selected_shadow["guarded_family"] = family
        selected_shadow["guarded_fallback_families"] = sorted(fallback_families)
        source_counts["baseline" if use_baseline else "candidate"] += 1
        documents.append(
            {
                "task_id": example_id,
                "steps": [
                    {
                        "proposal_metadata": {"neural_controller_shadow": selected_shadow},
                        "verification": {"passed": True},
                    }
                ],
            }
        )

    summary = summarize_neural_controller_shadow_documents(documents)
    family_metrics = summarize_family_metrics(documents)
    report = {
        "report_kind": "neural_controller_guarded_composition",
        "baseline_report_path": str(baseline_report_path),
        "candidate_report_path": str(candidate_report_path),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "fallback_families": sorted(fallback_families),
        "source_counts": source_counts,
        "missing_candidate_examples": missing_candidate,
        "candidate_comparison": comparison,
        "documents": documents,
        "summary": summary,
        "family_metrics": family_metrics,
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--min-family-total", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--fallback-family", action="append", default=[])
    args = parser.parse_args()
    report = compose_guarded_report(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_path=Path(args.candidate_report),
        output_path=Path(args.output),
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
        min_family_total=int(args.min_family_total),
        tolerance=float(args.tolerance),
        fallback_family=tuple(str(item) for item in args.fallback_family),
    )
    print(
        "neural_controller_guarded_composition "
        f"fallback_families={','.join(report['fallback_families']) or 'none'} "
        f"baseline_steps={report['source_counts']['baseline']} "
        f"candidate_steps={report['source_counts']['candidate']} "
        f"content_exact={report['summary'].get('content_exact_agreement_steps', 0)}/"
        f"{report['summary'].get('content_comparison_steps', 0)}"
    )


if __name__ == "__main__":
    main()
