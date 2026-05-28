from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import neural_controller_shadow_promotion_readiness


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _family_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("family_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _rate(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _readiness(summary: dict[str, Any], *, min_content_exact_agreement_rate: float) -> dict[str, Any]:
    return neural_controller_shadow_promotion_readiness(
        summary,
        min_content_exact_agreement_rate=float(min_content_exact_agreement_rate),
    )


def _report_row(label: str, path: Path, *, min_content_exact_agreement_rate: float) -> dict[str, Any]:
    report = _read_json_object(path)
    summary = _summary(report)
    families = _family_metrics(report)
    readiness = _readiness(summary, min_content_exact_agreement_rate=min_content_exact_agreement_rate)
    return {
        "label": label,
        "path": str(path),
        "manifest_path": str(report.get("manifest_path", "")),
        "content_exact": _int(summary, "content_exact_agreement_steps"),
        "content_steps": _int(summary, "content_comparison_steps"),
        "content_exact_rate": _rate(summary, "content_exact_agreement_rate"),
        "contract_content": _int(summary, "contract_content_agreement_steps"),
        "contract_content_rate": _rate(summary, "contract_content_agreement_rate"),
        "action_agreement_rate": _rate(summary, "action_agreement_rate"),
        "verified_action_agreement_rate": _rate(summary, "verified_action_agreement_rate"),
        "error_rate": _rate(summary, "error_rate"),
        "warning_rate": _rate(summary, "warning_rate"),
        "macro_content_exact_rate": _rate(families.get("_macro", {}), "macro_content_exact_rate")
        if isinstance(families.get("_macro", {}), dict)
        else 0.0,
        "macro_exec_kind_agreement_rate": _rate(families.get("_macro", {}), "macro_exec_kind_agreement_rate")
        if isinstance(families.get("_macro", {}), dict)
        else 0.0,
        "shadow_compare_ready": bool(readiness.get("shadow_compare_ready", False)),
        "kernel_guarded_content_ready": bool(readiness.get("kernel_guarded_content_ready", False)),
        "content_authority_ready": bool(readiness.get("content_authority_ready", False)),
        "primary_authority_ready": bool(readiness.get("primary_authority_ready", False)),
        "blockers": list(readiness.get("blockers", []) or []),
        "content_authority_blockers": list(readiness.get("content_authority_blockers", []) or []),
        "primary_authority_blocker": str(readiness.get("primary_authority_blocker", "")),
    }


def _family_gap_rows(report: dict[str, Any], *, min_content_exact_agreement_rate: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, raw in _family_metrics(report).items():
        if str(family).startswith("_") or not isinstance(raw, dict):
            continue
        total = _int(raw, "total")
        content_exact = _int(raw, "content_exact")
        content_rate = _rate(raw, "content_exact_rate")
        exec_rate = _rate(raw, "exec_kind_agreement_rate")
        rows.append(
            {
                "family": str(family),
                "total": total,
                "content_exact": content_exact,
                "content_miss": max(0, total - content_exact),
                "content_exact_rate": content_rate,
                "exec_kind_agreement_rate": exec_rate,
                "below_content_gate": content_rate < float(min_content_exact_agreement_rate),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            bool(row["below_content_gate"]),
            int(row["content_miss"]),
            int(row["total"]),
        ),
        reverse=True,
    )


def build_flip_readiness_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = Path(args.baseline_report)
    guarded_path = Path(args.guarded_report)
    candidates = [Path(path) for path in args.candidate_report]
    min_content_rate = float(args.min_content_exact_agreement_rate)
    baseline = _report_row("baseline", baseline_path, min_content_exact_agreement_rate=min_content_rate)
    guarded = _report_row("guarded", guarded_path, min_content_exact_agreement_rate=min_content_rate)
    candidate_rows = [
        _report_row(f"candidate:{path.stem}", path, min_content_exact_agreement_rate=min_content_rate)
        for path in candidates
    ]
    guarded_report = _read_json_object(guarded_path)
    flip_ready = bool(guarded.get("content_authority_ready", False) and guarded.get("primary_authority_ready", False))
    blockers = []
    if not guarded.get("content_authority_ready", False):
        blockers.extend(guarded.get("content_authority_blockers", []))
    if not guarded.get("primary_authority_ready", False):
        blockers.append(guarded.get("primary_authority_blocker", "primary_authority_not_ready"))
    report = {
        "report_kind": "neural_controller_flip_readiness",
        "target": "replace_tolbert_qwen_combo",
        "flip_ready": flip_ready,
        "recommended_runtime_mode": "primary" if flip_ready else "guarded_shadow_advisory",
        "baseline": baseline,
        "guarded": guarded,
        "candidates": candidate_rows,
        "blockers": list(dict.fromkeys(str(item) for item in blockers if str(item))),
        "family_gaps": _family_gap_rows(
            guarded_report,
            min_content_exact_agreement_rate=min_content_rate,
        ),
        "next_targets": [
            row["family"]
            for row in _family_gap_rows(
                guarded_report,
                min_content_exact_agreement_rate=min_content_rate,
            )[:3]
        ],
        "thresholds": {
            "min_content_exact_agreement_rate": min_content_rate,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--guarded-report", required=True)
    parser.add_argument("--candidate-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-content-exact-agreement-rate", type=float, default=0.80)
    args = parser.parse_args()
    report = build_flip_readiness_report(args)
    print(
        "neural_controller_flip_readiness "
        f"flip_ready={str(report['flip_ready']).lower()} "
        f"mode={report['recommended_runtime_mode']} "
        f"guarded_exact={report['guarded']['content_exact']}/{report['guarded']['content_steps']} "
        f"blockers={','.join(report['blockers']) or 'none'} "
        f"next_targets={','.join(report['next_targets']) or 'none'}"
    )


if __name__ == "__main__":
    main()
