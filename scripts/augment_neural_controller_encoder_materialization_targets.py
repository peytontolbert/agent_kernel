from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import (  # noqa: E402
    augment_encoder_with_active_materialization_target,
    augment_encoder_with_plan_source_inspection_candidates,
)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def augment_dataset(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _iter_jsonl(input_path)
    changed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            copied = dict(row)
            before = str(copied.get("encoder_text", ""))
            after = augment_encoder_with_active_materialization_target(before)
            after = augment_encoder_with_plan_source_inspection_candidates(after)
            if after != before:
                changed += 1
                copied["encoder_text"] = after
                copied["active_materialization_target_augmented"] = True
                copied["source_inspection_candidates_augmented"] = True
            handle.write(json.dumps(copied, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "artifact_kind": "neural_controller_active_materialization_encoder_augmentation",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "changed_rows": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = augment_dataset(Path(args.input), Path(args.output))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
