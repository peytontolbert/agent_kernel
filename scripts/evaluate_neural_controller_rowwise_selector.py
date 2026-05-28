from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import (
    neural_controller_shadow_promotion_readiness,
    select_verified_neural_controller_shadow,
    summarize_neural_controller_shadow_documents,
)
from scripts.evaluate_neural_controller_shadow_dataset import summarize_family_metrics
from scripts.report_neural_controller_rowwise_frontier import _contract_success, _make_document, _read_json_object
from scripts.report_neural_controller_rowwise_frontier import _shadow_by_example, _strict_exact


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _candidate_label_map(paths: list[Path], labels: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for index, path in enumerate(paths):
        label = labels[index] if index < len(labels) and labels[index] else path.stem
        out[label] = path
    return out


def _policy_accepts(row: dict[str, Any], policy: str) -> bool:
    baseline = row.get("baseline", {})
    candidate = row.get("candidate", {})
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return False
    if policy == "positive_label":
        return bool(row.get("accept_candidate", False))
    if policy == "candidate_strict_improves":
        return (not bool(baseline.get("strict_exact", False))) and bool(candidate.get("strict_exact", False))
    if policy == "candidate_contract_improves":
        return (not bool(baseline.get("contract_success", False))) and bool(
            candidate.get("contract_success", False)
        )
    if policy == "candidate_strict_or_contract_improves":
        strict = (not bool(baseline.get("strict_exact", False))) and bool(candidate.get("strict_exact", False))
        contract = (not bool(baseline.get("contract_success", False))) and bool(
            candidate.get("contract_success", False)
        )
        return strict or contract
    raise ValueError(f"unknown selector policy: {policy}")


def _select_with_runtime_contract_policy(
    *,
    baseline_label: str,
    baseline_shadow: dict[str, Any],
    candidate_paths_by_label: dict[str, Path],
    candidate_shadows_by_label: dict[str, dict[str, dict[str, Any]]],
    example_id: str,
) -> tuple[str, dict[str, Any]]:
    selection = select_verified_neural_controller_shadow(
        baseline_label=baseline_label,
        baseline_shadow=baseline_shadow,
        candidate_shadows=[
            (label, candidate_shadows_by_label.get(label, {}).get(example_id, {}))
            for label in candidate_paths_by_label
        ],
        policy="candidate_contract_improves",
    )
    shadow = selection.get("shadow", {})
    return str(selection.get("source", baseline_label)).strip() or baseline_label, dict(shadow if isinstance(shadow, dict) else baseline_shadow)


def evaluate_rowwise_selector(
    *,
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    selector_dataset_path: Path,
    output_path: Path,
    candidate_labels: list[str] | None = None,
    baseline_label: str = "baseline",
    policy: str = "candidate_strict_or_contract_improves",
) -> dict[str, Any]:
    baseline_report = _read_json_object(baseline_report_path)
    baseline_by_id = _shadow_by_example(baseline_report)
    labels = candidate_labels or []
    candidate_paths_by_label = _candidate_label_map(candidate_report_paths, labels)
    candidate_shadows_by_label = {
        label: _shadow_by_example(_read_json_object(path)) for label, path in candidate_paths_by_label.items()
    }
    selector_rows = _iter_jsonl(selector_dataset_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        example_id = str(row.get("example_id", "")).strip()
        if example_id:
            grouped[example_id].append(row)

    documents: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    selected_rows: list[dict[str, Any]] = []
    for example_id in sorted(baseline_by_id):
        baseline_shadow = baseline_by_id[example_id]
        if policy == "candidate_contract_improves":
            selected_label, selected_shadow = _select_with_runtime_contract_policy(
                baseline_label=baseline_label,
                baseline_shadow=baseline_shadow,
                candidate_paths_by_label=candidate_paths_by_label,
                candidate_shadows_by_label=candidate_shadows_by_label,
                example_id=example_id,
            )
        else:
            selected_label = baseline_label
            selected_shadow = baseline_shadow
            accepted_candidates = [row for row in grouped.get(example_id, []) if _policy_accepts(row, policy)]
            accepted_candidates.sort(
                key=lambda row: (
                    1 if bool(row.get("candidate_improves_strict", False)) else 0,
                    1 if bool(row.get("candidate_improves_contract", False)) else 0,
                    -list(candidate_paths_by_label).index(str(row.get("candidate_label", "")))
                    if str(row.get("candidate_label", "")) in candidate_paths_by_label
                    else -9999,
                ),
                reverse=True,
            )
            if accepted_candidates:
                selected_label = str(accepted_candidates[0].get("candidate_label", "")).strip()
                selected_shadow = candidate_shadows_by_label.get(selected_label, {}).get(example_id, baseline_shadow)
        shadow = dict(selected_shadow)
        shadow["rowwise_selector_source"] = selected_label
        shadow["rowwise_selector_policy"] = policy
        shadow["rowwise_selector_baseline_label"] = baseline_label
        source_counts[selected_label] += 1
        documents.append(_make_document(example_id, shadow))
        selected_rows.append(
            {
                "example_id": example_id,
                "selected_source": selected_label,
                "baseline_exact": _strict_exact(baseline_shadow),
                "selected_exact": _strict_exact(shadow),
                "baseline_contract": _contract_success(baseline_shadow),
                "selected_contract": _contract_success(shadow),
            }
        )

    summary = summarize_neural_controller_shadow_documents(documents)
    report = {
        "report_kind": "neural_controller_rowwise_selector_eval",
        "baseline_report_path": str(baseline_report_path),
        "candidate_report_paths": [str(path) for path in candidate_report_paths],
        "selector_dataset_path": str(selector_dataset_path),
        "baseline_label": baseline_label,
        "policy": policy,
        "source_counts": dict(sorted(source_counts.items())),
        "summary": summary,
        "family_metrics": summarize_family_metrics(documents),
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
        "rows": selected_rows,
        "documents": documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", action="append", default=[])
    parser.add_argument("--candidate-label", action="append", default=[])
    parser.add_argument("--selector-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--policy",
        choices=[
            "positive_label",
            "candidate_strict_improves",
            "candidate_contract_improves",
            "candidate_strict_or_contract_improves",
        ],
        default="candidate_strict_or_contract_improves",
    )
    args = parser.parse_args()
    if not args.candidate_report:
        raise ValueError("at least one --candidate-report is required")
    report = evaluate_rowwise_selector(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_paths=[Path(path) for path in args.candidate_report],
        selector_dataset_path=Path(args.selector_dataset),
        output_path=Path(args.output),
        candidate_labels=[str(label) for label in args.candidate_label],
        baseline_label=str(args.baseline_label),
        policy=str(args.policy),
    )
    summary = report["summary"]
    print(
        "neural_controller_rowwise_selector_eval "
        f"policy={report['policy']} "
        f"content_exact={summary.get('content_exact_agreement_steps', 0)}/"
        f"{summary.get('content_comparison_steps', 0)} "
        f"contract={summary.get('contract_content_agreement_steps', 0)}/"
        f"{summary.get('content_comparison_steps', 0)} "
        f"sources={json.dumps(report['source_counts'], sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
