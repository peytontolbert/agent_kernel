from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rate(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _guarded_family_metrics(flip_report: dict[str, Any]) -> dict[str, Any]:
    guarded = flip_report.get("guarded", {})
    if not isinstance(guarded, dict):
        return {}
    path_text = str(guarded.get("path", "")).strip()
    if not path_text:
        return {}
    try:
        guarded_report = _read_json_object(Path(path_text))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    metrics = guarded_report.get("family_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _family_authority_profile(
    *,
    family_metrics: dict[str, Any],
    min_content_rate: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    blockers: list[str] = []
    for family, raw in sorted(family_metrics.items()):
        if str(family).startswith("_") or not isinstance(raw, dict):
            continue
        exact_rate = _rate(raw, "content_exact_rate")
        contract_rate = _rate(raw, "contract_content_rate")
        if family == "materialize_artifact":
            ready = contract_rate >= min_content_rate
            metric = "contract_content_rate"
        else:
            ready = exact_rate >= min_content_rate
            metric = "content_exact_rate"
        if not ready:
            blockers.append(f"{family}_{metric}_below_gate")
        rows[str(family)] = {
            "authority_metric": metric,
            "content_exact_rate": exact_rate,
            "contract_content_rate": contract_rate,
            "ready": ready,
        }
    return {
        "profile": "contract_materialize_strict_other_families",
        "ready": bool(rows) and not blockers,
        "blockers": blockers,
        "families": rows,
    }


def build_retained_promotion_gate(args: argparse.Namespace) -> dict[str, Any]:
    flip_report_path = Path(args.flip_report)
    flip_report = _read_json_object(flip_report_path)
    guarded = flip_report.get("guarded", {})
    if not isinstance(guarded, dict):
        guarded = {}
    strict_content_rate = _rate(guarded, "content_exact_rate")
    contract_content_rate = _rate(guarded, "contract_content_rate")
    min_content_rate = float(args.min_content_rate)
    strict_content_ready = strict_content_rate >= min_content_rate
    contract_content_ready = contract_content_rate >= min_content_rate
    family_profile = _family_authority_profile(
        family_metrics=_guarded_family_metrics(flip_report),
        min_content_rate=min_content_rate,
    )
    retained_evidence_ready = bool(args.retained_evidence_ready)
    allow_contract_content_primary = bool(args.allow_contract_content_primary)
    primary_content_ready = strict_content_ready or (
        allow_contract_content_primary and contract_content_ready
    )
    primary_authority_ready = bool(retained_evidence_ready and primary_content_ready)
    blockers: list[str] = []
    if not retained_evidence_ready:
        blockers.append("retained_evidence_not_confirmed")
    if not primary_content_ready:
        blockers.append("primary_content_gate_not_met")
    if contract_content_ready and not strict_content_ready and not allow_contract_content_primary:
        blockers.append("contract_content_ready_but_not_authorized_for_primary")
    report = {
        "report_kind": "neural_controller_retained_promotion_gate",
        "flip_report_path": str(flip_report_path),
        "strict_content_ready": strict_content_ready,
        "contract_content_ready": contract_content_ready,
        "retained_evidence_ready": retained_evidence_ready,
        "allow_contract_content_primary": allow_contract_content_primary,
        "primary_authority_ready": primary_authority_ready,
        "recommended_runtime_mode": "primary" if primary_authority_ready else "guarded",
        "blockers": blockers,
        "family_authority_profile": family_profile,
        "metrics": {
            "strict_content_rate": strict_content_rate,
            "contract_content_rate": contract_content_rate,
            "min_content_rate": min_content_rate,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-content-rate", type=float, default=0.80)
    parser.add_argument("--retained-evidence-ready", action="store_true")
    parser.add_argument("--allow-contract-content-primary", action="store_true")
    args = parser.parse_args()
    report = build_retained_promotion_gate(args)
    print(
        "neural_controller_retained_promotion_gate "
        f"primary_authority_ready={str(report['primary_authority_ready']).lower()} "
        f"recommended_mode={report['recommended_runtime_mode']} "
        f"strict_content_ready={str(report['strict_content_ready']).lower()} "
        f"contract_content_ready={str(report['contract_content_ready']).lower()} "
        f"blockers={','.join(report['blockers']) or 'none'}"
    )


if __name__ == "__main__":
    main()
