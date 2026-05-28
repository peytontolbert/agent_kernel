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


def build_selector_retained_candidate_packet(
    *,
    readiness_path: Path,
    selector_dataset_path: Path,
    selector_report_path: Path,
    output_path: Path,
    selector_policy: str = "candidate_contract_improves",
) -> dict[str, Any]:
    readiness = _read_json_object(readiness_path)
    reaches_retained = bool(
        readiness.get("reaches_retained_strict", False)
        and readiness.get("reaches_retained_contract", False)
    )
    blockers: list[str] = []
    if not reaches_retained:
        blockers.append("selector_does_not_reach_retained_surface")
    if bool(readiness.get("primary_authority_ready", False)):
        blockers.append("unexpected_primary_authority_claim")
    packet = {
        "report_kind": "neural_controller_selector_retained_candidate_packet",
        "readiness_path": str(readiness_path),
        "selector_dataset_path": str(selector_dataset_path),
        "selector_report_path": str(selector_report_path),
        "selector_policy": str(selector_policy).strip() or "candidate_contract_improves",
        "retained_candidate_ready": reaches_retained and not blockers,
        "primary_authority_ready": False,
        "recommended_runtime_mode": "guarded",
        "blockers": blockers,
        "metrics": {
            "strict": readiness.get("selector", {}).get("strict", 0)
            if isinstance(readiness.get("selector", {}), dict)
            else 0,
            "contract": readiness.get("selector", {}).get("contract", 0)
            if isinstance(readiness.get("selector", {}), dict)
            else 0,
            "total": readiness.get("selector", {}).get("total", 0)
            if isinstance(readiness.get("selector", {}), dict)
            else 0,
            "strict_gain": int(readiness.get("strict_gain", 0) or 0),
            "contract_gain": int(readiness.get("contract_gain", 0) or 0),
            "source_switch_count": int(readiness.get("source_switch_count", 0) or 0),
        },
        "authority_note": (
            "Retain as guarded selector candidate only. Primary authority remains blocked until "
            "verifier-backed runtime integration and a dedicated retained promotion gate are complete."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--selector-dataset", required=True)
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector-policy", default="candidate_contract_improves")
    args = parser.parse_args()
    packet = build_selector_retained_candidate_packet(
        readiness_path=Path(args.readiness),
        selector_dataset_path=Path(args.selector_dataset),
        selector_report_path=Path(args.selector_report),
        output_path=Path(args.output),
        selector_policy=str(args.selector_policy),
    )
    print(
        "neural_controller_selector_retained_candidate_packet "
        f"ready={str(packet['retained_candidate_ready']).lower()} "
        f"mode={packet['recommended_runtime_mode']} "
        f"primary_authority_ready={str(packet['primary_authority_ready']).lower()} "
        f"blockers={','.join(packet['blockers']) or 'none'}"
    )


if __name__ == "__main__":
    main()
