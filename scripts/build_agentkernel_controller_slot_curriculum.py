#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_agentkernel_lite_datasets import DatasetWriter
from scripts.merge_agentkernel_lite_datasets import _iter_dataset_rows


SLOT_FIELDS = (
    "Target-Path:",
    "Target-Content:",
    "Verify-Polarity:",
    "Edit-Old:",
    "Edit-New:",
)

SLOT_NAMES = {
    "Target-Path:": "target_path",
    "Target-Content:": "target_content",
    "Verify-Polarity:": "verify_polarity",
    "Edit-Old:": "edit_old",
    "Edit-New:": "edit_new",
}

EXEC_KIND_FAMILIES = {
    "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>": "materialize_artifact",
    "<AK_EXEC_KIND_VERIFY_PRESENT>": "verify_present",
    "<AK_EXEC_KIND_VERIFY_ABSENT>": "verify_absent",
    "<AK_EXEC_KIND_INSPECT_SOURCE>": "inspect_source",
    "<AK_EXEC_KIND_LOCALIZED_EDIT>": "localized_edit",
    "<AK_EXEC_KIND_RUN_CHECK>": "run_check",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_slot_names(row: dict[str, Any]) -> list[str]:
    decoder_text = str(row.get("decoder_text") or "")
    names: list[str] = []
    for prefix in SLOT_FIELDS:
        if prefix in decoder_text:
            names.append(SLOT_NAMES[prefix])
    if "Verify-Polarity: present" in decoder_text:
        names.append("verify_present")
    if "Verify-Polarity: absent" in decoder_text:
        names.append("verify_absent")
    return names


def _row_operation_family(row: dict[str, Any]) -> str:
    decoder_text = str(row.get("decoder_text") or "")
    for token, family in EXEC_KIND_FAMILIES.items():
        if token in decoder_text:
            return family
    return "unknown"


def _parse_family_bonus(value: str) -> dict[str, int]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--family-bonus-json must be a JSON object")
    bonuses: dict[str, int] = {}
    for family, bonus in payload.items():
        bonuses[str(family)] = max(0, int(bonus))
    return bonuses


def _repeat_count_for_slots(
    slots: list[str],
    *,
    operation_family: str,
    family_bonus: dict[str, int],
    base_repeat: int,
    target_content_bonus: int,
    verify_bonus: int,
    verify_present_bonus: int,
    edit_bonus: int,
    max_repeat: int,
) -> int:
    if not slots:
        return 0
    repeat = max(1, int(base_repeat))
    if "target_content" in slots:
        repeat += max(0, int(target_content_bonus))
    if "verify_polarity" in slots:
        repeat += max(0, int(verify_bonus))
    if "verify_present" in slots:
        repeat += max(0, int(verify_present_bonus))
    if "edit_old" in slots or "edit_new" in slots:
        repeat += max(0, int(edit_bonus))
    repeat += max(0, int(family_bonus.get(operation_family, 0)))
    return max(1, min(int(max_repeat), repeat))


def _with_repeat_metadata(
    row: dict[str, Any],
    *,
    slots: list[str],
    operation_family: str,
    repeat_index: int,
    repeat_count: int,
) -> dict[str, Any]:
    copied = dict(row)
    base_id = str(copied.get("example_id") or copied.get("source_id") or "slot_row")
    copied["example_id"] = f"{base_id}:slot_curriculum:{repeat_index:02d}"
    copied["source_type"] = str(copied.get("source_type") or "agentkernel_slot_curriculum")
    copied["task_type"] = "controller_long_horizon_argument_slots"
    copied["slot_curriculum_repeat_index"] = repeat_index
    copied["slot_curriculum_repeat_count"] = repeat_count
    copied["slot_curriculum_slots"] = slots
    copied["slot_curriculum_operation_family"] = operation_family
    try:
        weight = float(copied.get("weight") or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    copied["weight"] = max(weight, min(float(repeat_count), 8.0))
    return copied


def build_slot_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    source_manifest = _load_json(source_manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    family_bonus = _parse_family_bonus(str(getattr(args, "family_bonus_json", "") or ""))
    writer = DatasetWriter(
        output_dir,
        output_format=str(args.output_format),
        parquet_shard_size=int(args.parquet_shard_size),
    )

    slot_counts = {"train": {}, "eval": {}}
    family_counts = {"train": {}, "eval": {}}
    repeated_family_counts = {"train": {}, "eval": {}}
    source_counts = {
        "train_rows_seen": 0,
        "eval_rows_seen": 0,
        "train_slot_rows": 0,
        "eval_slot_rows": 0,
        "train_repeated_rows": 0,
    }

    def record_slots(split: str, slots: list[str]) -> None:
        for slot in slots:
            slot_counts[split][slot] = int(slot_counts[split].get(slot, 0)) + 1

    def record_family(split: str, family: str) -> None:
        family_counts[split][family] = int(family_counts[split].get(family, 0)) + 1

    def record_repeated_family(split: str, family: str, count: int) -> None:
        repeated_family_counts[split][family] = int(repeated_family_counts[split].get(family, 0)) + int(count)

    try:
        for row in _iter_dataset_rows(Path(str(source_manifest["train_dataset_path"]))):
            source_counts["train_rows_seen"] += 1
            slots = _row_slot_names(row)
            if not slots:
                continue
            operation_family = _row_operation_family(row)
            source_counts["train_slot_rows"] += 1
            record_slots("train", slots)
            record_family("train", operation_family)
            repeat_count = _repeat_count_for_slots(
                slots,
                operation_family=operation_family,
                family_bonus=family_bonus,
                base_repeat=args.base_repeat,
                target_content_bonus=args.target_content_bonus,
                verify_bonus=args.verify_bonus,
                verify_present_bonus=args.verify_present_bonus,
                edit_bonus=args.edit_bonus,
                max_repeat=args.max_repeat,
            )
            record_repeated_family("train", operation_family, repeat_count)
            for repeat_index in range(repeat_count):
                writer.write(
                    "train",
                    _with_repeat_metadata(
                        row,
                        slots=slots,
                        operation_family=operation_family,
                        repeat_index=repeat_index,
                        repeat_count=repeat_count,
                    ),
                )
                source_counts["train_repeated_rows"] += 1

        for row in _iter_dataset_rows(Path(str(source_manifest["eval_dataset_path"]))):
            source_counts["eval_rows_seen"] += 1
            slots = _row_slot_names(row)
            if not slots:
                continue
            operation_family = _row_operation_family(row)
            source_counts["eval_slot_rows"] += 1
            record_slots("eval", slots)
            record_family("eval", operation_family)
            record_repeated_family("eval", operation_family, 1)
            writer.write(
                "eval",
                _with_repeat_metadata(
                    row,
                    slots=slots,
                    operation_family=operation_family,
                    repeat_index=0,
                    repeat_count=1,
                ),
            )
    finally:
        writer.close()

    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        **source_manifest,
        "artifact_kind": "agentkernel_controller_slot_curriculum_dataset",
        "objective": str(args.objective),
        "dataset_format": writer.output_format,
        "decoder_format": source_manifest.get("decoder_format", "line"),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(writer.train_path),
        "eval_dataset_path": str(writer.eval_path),
        "source_manifest_path": str(source_manifest_path),
        "total_examples": writer.counts["train"] + writer.counts["eval"],
        "train_examples": writer.counts["train"],
        "eval_examples": writer.counts["eval"],
        "slot_curriculum": {
            "slot_fields": list(SLOT_NAMES.values()),
            "operation_families": sorted(set(EXEC_KIND_FAMILIES.values())),
            "source_counts": source_counts,
            "slot_counts": slot_counts,
            "family_counts": family_counts,
            "repeated_family_counts": repeated_family_counts,
            "repeat_policy": {
                "base_repeat": int(args.base_repeat),
                "target_content_bonus": int(args.target_content_bonus),
                "verify_bonus": int(args.verify_bonus),
                "verify_present_bonus": int(args.verify_present_bonus),
                "edit_bonus": int(args.edit_bonus),
                "family_bonus": family_bonus,
                "max_repeat": int(args.max_repeat),
            },
            "eval_policy": "slot_bearing_rows_once",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="agentkernel_controller_long_horizon_argument_slot_curriculum")
    parser.add_argument("--base-repeat", type=int, default=2)
    parser.add_argument("--target-content-bonus", type=int, default=4)
    parser.add_argument("--verify-bonus", type=int, default=4)
    parser.add_argument("--verify-present-bonus", type=int, default=4)
    parser.add_argument("--edit-bonus", type=int, default=8)
    parser.add_argument(
        "--family-bonus-json",
        default="",
        help="JSON object mapping operation family names to additional repeat counts.",
    )
    parser.add_argument("--max-repeat", type=int, default=12)
    parser.add_argument("--output-format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--parquet-shard-size", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(build_slot_curriculum(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
