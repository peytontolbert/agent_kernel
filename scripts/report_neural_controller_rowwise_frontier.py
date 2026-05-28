from __future__ import annotations

import argparse
import json
from collections import Counter
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
        step = steps[0] if isinstance(steps[0], dict) else {}
        metadata = step.get("proposal_metadata", {}) if isinstance(step, dict) else {}
        if not isinstance(metadata, dict):
            continue
        shadow = metadata.get("neural_controller_shadow", {})
        if not isinstance(shadow, dict):
            continue
        example_id = str(shadow.get("example_id") or document.get("task_id") or "").strip()
        if example_id:
            out[example_id] = dict(shadow)
    return out


def _strict_exact(shadow: dict[str, Any] | None) -> bool:
    return bool(shadow and shadow.get("content_exact_agreement", False))


def _contract_success(shadow: dict[str, Any] | None) -> bool:
    return _strict_exact(shadow) or (
        bool(shadow) and str(shadow.get("artifact_failure_mode", "")).strip() == "artifact_contract_success"
    )


def _family(shadow: dict[str, Any] | None) -> str:
    if not shadow:
        return "unknown"
    return EXEC_KIND_FAMILY.get(str(shadow.get("target_exec_kind", "")).strip(), "unknown")


def _shadow_score(shadow: dict[str, Any] | None) -> tuple[int, int, int, float, int]:
    if not shadow:
        return (0, 0, 0, 0.0, 0)
    return (
        1 if _strict_exact(shadow) else 0,
        1 if _contract_success(shadow) else 0,
        1 if bool(shadow.get("exec_kind_agreement", False)) else 0,
        float(shadow.get("slot_agreement_rate", 0.0) or 0.0),
        1 if not shadow.get("warnings") else 0,
    )


def _make_document(example_id: str, shadow: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": example_id,
        "steps": [
            {
                "proposal_metadata": {"neural_controller_shadow": shadow},
                "verification": {"passed": True},
            }
        ],
    }


def _feature_payload(shadow: dict[str, Any] | None) -> dict[str, Any]:
    if not shadow:
        return {}
    policy_heads = shadow.get("policy_heads", {})
    if not isinstance(policy_heads, dict):
        policy_heads = {}
    control_tokens = shadow.get("control_tokens", [])
    if not isinstance(control_tokens, list):
        control_tokens = []
    return {
        "strict_exact": _strict_exact(shadow),
        "contract_success": _contract_success(shadow),
        "family": _family(shadow),
        "artifact_failure_mode": str(shadow.get("artifact_failure_mode", "")).strip(),
        "predicted_exec_kind": str(shadow.get("predicted_exec_kind", "")).strip(),
        "target_exec_kind": str(shadow.get("target_exec_kind", "")).strip(),
        "generated_token_count": int(shadow.get("generated_token_count", 0) or 0),
        "slot_agreement_rate": float(shadow.get("slot_agreement_rate", 0.0) or 0.0),
        "control_tokens": [str(token) for token in control_tokens if str(token).startswith("<AK_")],
        "policy_heads": {
            str(key): float(value)
            for key, value in policy_heads.items()
            if isinstance(value, int | float)
        },
        "warning_count": len(list(shadow.get("warnings", []) or []))
        if isinstance(shadow.get("warnings", []), list)
        else 0,
    }


def report_rowwise_frontier(
    *,
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    output_path: Path,
    retained_report_path: Path | None = None,
    baseline_label: str = "baseline",
    candidate_labels: list[str] | None = None,
    preserve_baseline_exact: bool = True,
    selector_dataset_output_path: Path | None = None,
) -> dict[str, Any]:
    baseline_report = _read_json_object(baseline_report_path)
    retained_report = _read_json_object(retained_report_path) if retained_report_path else None
    baseline_by_id = _shadow_by_example(baseline_report)
    retained_by_id = _shadow_by_example(retained_report) if retained_report else {}
    candidate_reports = [_read_json_object(path) for path in candidate_report_paths]
    candidate_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    labels = candidate_labels or []
    for index, (path, report) in enumerate(zip(candidate_report_paths, candidate_reports, strict=True)):
        label = labels[index] if index < len(labels) and labels[index] else path.stem
        candidate_by_label[label] = _shadow_by_example(report)

    example_ids = sorted(baseline_by_id)
    oracle_documents: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}

    for example_id in example_ids:
        baseline_shadow = baseline_by_id.get(example_id)
        retained_shadow = retained_by_id.get(example_id)
        candidates = {
            label: shadows.get(example_id)
            for label, shadows in candidate_by_label.items()
            if shadows.get(example_id) is not None
        }
        best_label = baseline_label
        best_shadow = baseline_shadow
        if not (preserve_baseline_exact and _strict_exact(baseline_shadow)):
            for label, shadow in candidates.items():
                improves_strict = (not _strict_exact(best_shadow)) and _strict_exact(shadow)
                improves_contract = (not _contract_success(best_shadow)) and _contract_success(shadow)
                if (improves_strict or improves_contract) and _shadow_score(shadow) > _shadow_score(best_shadow):
                    best_label = label
                    best_shadow = shadow
        selected_shadow = dict(best_shadow or baseline_shadow or {})
        selected_shadow["rowwise_frontier_source"] = best_label
        selected_shadow["rowwise_frontier_baseline_label"] = baseline_label
        source_counts[best_label] += 1
        oracle_documents.append(_make_document(example_id, selected_shadow))

        family = _family(baseline_shadow or retained_shadow or best_shadow)
        family_row = family_counts.setdefault(family, Counter())
        baseline_exact = _strict_exact(baseline_shadow)
        retained_exact = _strict_exact(retained_shadow)
        selected_exact = _strict_exact(selected_shadow)
        if baseline_exact:
            family_row["baseline_exact"] += 1
        if retained_exact:
            family_row["retained_exact"] += 1
        if selected_exact:
            family_row["frontier_exact"] += 1
        if not baseline_exact and selected_exact:
            family_row["baseline_miss_recovered"] += 1
        if retained_exact and not selected_exact:
            family_row["retained_only_unrecovered"] += 1
        family_row["total"] += 1

        rows.append(
            {
                "example_id": example_id,
                "family": family,
                "baseline_exact": baseline_exact,
                "retained_exact": retained_exact if retained_shadow else None,
                "frontier_exact": selected_exact,
                "selected_source": best_label,
                "candidate_exact_sources": sorted(
                    label for label, shadow in candidates.items() if _strict_exact(shadow)
                ),
                "candidate_contract_sources": sorted(
                    label for label, shadow in candidates.items() if _contract_success(shadow)
                ),
                "baseline_miss_recovered": (not baseline_exact and selected_exact),
                "retained_only_unrecovered": bool(retained_shadow and retained_exact and not selected_exact),
            }
        )
        baseline_features = _feature_payload(baseline_shadow)
        for label, candidate_shadow in sorted(candidates.items()):
            candidate_features = _feature_payload(candidate_shadow)
            candidate_improves_strict = (not baseline_exact) and _strict_exact(candidate_shadow)
            candidate_improves_contract = (not _contract_success(baseline_shadow)) and _contract_success(
                candidate_shadow
            )
            selector_rows.append(
                {
                    "example_id": example_id,
                    "candidate_label": label,
                    "baseline_label": baseline_label,
                    "family": family,
                    "accept_candidate": candidate_improves_strict or candidate_improves_contract,
                    "frontier_selected_source": best_label,
                    "baseline_miss": not baseline_exact,
                    "candidate_improves_strict": candidate_improves_strict,
                    "candidate_improves_contract": candidate_improves_contract,
                    "baseline": baseline_features,
                    "candidate": candidate_features,
                }
            )

    summary = summarize_neural_controller_shadow_documents(oracle_documents)
    family_metrics = summarize_family_metrics(oracle_documents)
    selector_summary = {
        "rows": len(selector_rows),
        "accepted": sum(1 for row in selector_rows if row["accept_candidate"]),
        "rejected": sum(1 for row in selector_rows if not row["accept_candidate"]),
        "accepted_by_candidate": dict(
            sorted(Counter(row["candidate_label"] for row in selector_rows if row["accept_candidate"]).items())
        ),
    }
    if selector_dataset_output_path:
        selector_dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        selector_dataset_output_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selector_rows),
            encoding="utf-8",
        )
    report = {
        "report_kind": "neural_controller_rowwise_frontier",
        "baseline_report_path": str(baseline_report_path),
        "retained_report_path": str(retained_report_path) if retained_report_path else "",
        "candidate_report_paths": [str(path) for path in candidate_report_paths],
        "baseline_label": baseline_label,
        "preserve_baseline_exact": bool(preserve_baseline_exact),
        "source_counts": dict(sorted(source_counts.items())),
        "frontier_summary": summary,
        "frontier_family_metrics": family_metrics,
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
        "family_recovery_counts": {
            family: dict(sorted(counter.items())) for family, counter in sorted(family_counts.items())
        },
        "selector_dataset_output_path": str(selector_dataset_output_path) if selector_dataset_output_path else "",
        "selector_dataset_summary": selector_summary,
        "rows": rows,
        "documents": oracle_documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", action="append", default=[])
    parser.add_argument("--candidate-label", action="append", default=[])
    parser.add_argument("--retained-report", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector-dataset-output", default="")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--allow-baseline-exact-switches",
        action="store_true",
        help="Allow candidates to replace rows where the baseline is already strict-exact.",
    )
    args = parser.parse_args()
    if not args.candidate_report:
        raise ValueError("at least one --candidate-report is required")
    report = report_rowwise_frontier(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_paths=[Path(path) for path in args.candidate_report],
        retained_report_path=Path(args.retained_report) if args.retained_report else None,
        output_path=Path(args.output),
        baseline_label=str(args.baseline_label),
        candidate_labels=[str(label) for label in args.candidate_label],
        preserve_baseline_exact=not bool(args.allow_baseline_exact_switches),
        selector_dataset_output_path=Path(args.selector_dataset_output)
        if args.selector_dataset_output
        else None,
    )
    summary = report["frontier_summary"]
    print(
        "neural_controller_rowwise_frontier "
        f"content_exact={summary.get('content_exact_agreement_steps', 0)}/"
        f"{summary.get('content_comparison_steps', 0)} "
        f"contract={summary.get('contract_content_agreement_steps', 0)}/"
        f"{summary.get('content_comparison_steps', 0)} "
        f"sources={json.dumps(report['source_counts'], sort_keys=True)} "
        f"selector_rows={report['selector_dataset_summary']['rows']} "
        f"selector_accepts={report['selector_dataset_summary']['accepted']}"
    )


if __name__ == "__main__":
    main()
