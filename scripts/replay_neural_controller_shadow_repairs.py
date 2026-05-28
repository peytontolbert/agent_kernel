#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_kernel.neural_controller import (  # noqa: E402
    neural_controller_shadow_promotion_readiness,
    repair_line_protocol_with_command_copy_target,
    summarize_neural_controller_shadow_documents,
)
from scripts.evaluate_neural_controller_shadow_dataset import (  # noqa: E402
    summarize_family_metrics,
)


def _load_eval_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows[str(row.get("example_id", ""))] = row
    return rows


def _shadow_to_line_protocol(shadow: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokens": shadow.get("control_tokens", []),
        "action": str(shadow.get("predicted_action", "")),
        "content": str(shadow.get("predicted_content") or shadow.get("predicted_content_preview", "")),
        "target_path": str(shadow.get("predicted_target_path", "")),
        "target_content": str(shadow.get("predicted_target_content", "")),
        "verify_polarity": str(shadow.get("predicted_verify_polarity", "")),
    }


def replay_repairs(*, report_path: Path, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_eval_rows(dataset_path)
    changed = 0
    improved = 0
    regressed = 0
    by_family: dict[str, list[int]] = {}

    for document in report.get("documents", []):
        if not isinstance(document, dict):
            continue
        for step in document.get("steps", []):
            if not isinstance(step, dict):
                continue
            shadow = step.get("proposal_metadata", {}).get("neural_controller_shadow", {})
            if not isinstance(shadow, dict) or not shadow.get("ready"):
                continue
            row = rows.get(str(shadow.get("example_id", "")))
            if not row:
                continue
            line_protocol = _shadow_to_line_protocol(shadow)
            repaired, _warnings = repair_line_protocol_with_command_copy_target(
                line_protocol,
                encoder_text=str(row.get("encoder_text", "")),
            )
            if str(repaired.get("content", "")) == str(line_protocol.get("content", "")):
                continue
            before = bool(shadow.get("content_exact_agreement", False))
            after = str(repaired.get("content", "")) == str(shadow.get("target_content_preview", ""))
            family = str(shadow.get("target_exec_kind", ""))
            by_family.setdefault(family, [0, 0, 0])
            by_family[family][0] += 1
            changed += 1
            if after and not before:
                improved += 1
                by_family[family][1] += 1
            if before and not after:
                regressed += 1
                by_family[family][2] += 1
            repaired_content = str(repaired.get("content", ""))
            shadow["predicted_content"] = repaired_content
            shadow["predicted_content_preview"] = repaired_content[:240]
            shadow["selected_content"] = repaired_content
            shadow["selected_content_preview"] = repaired_content[:240]
            shadow["content_exact_agreement"] = after
            for flag in (
                "source_inspection_candidate_repaired",
                "validation_command_repaired",
                "artifact_command_target_repaired",
            ):
                if repaired.get(flag):
                    shadow[flag] = True
            shadow["postprocess_replay_source"] = str(report_path)

    report["summary"] = summarize_neural_controller_shadow_documents(report.get("documents", []))
    report["family_metrics"] = summarize_family_metrics(report.get("documents", []))
    report["promotion_readiness"] = neural_controller_shadow_promotion_readiness(report["summary"])
    report["postprocess_replay"] = {
        "source_report": str(report_path),
        "changed_steps": changed,
        "improved_steps": improved,
        "regressed_steps": regressed,
        "by_family": by_family,
        "repair": "neural_controller_shadow_repairs",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = replay_repairs(
        report_path=Path(args.report),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
    )
    summary = report.get("summary", {})
    replay = report.get("postprocess_replay", {})
    print(
        "neural_controller_shadow_repair_replay "
        f"exact={summary.get('content_exact_agreement_steps', 0)}/"
        f"{summary.get('content_comparison_steps', 0)} "
        f"changed={replay.get('changed_steps', 0)} "
        f"improved={replay.get('improved_steps', 0)} "
        f"regressed={replay.get('regressed_steps', 0)}"
    )


if __name__ == "__main__":
    main()
