from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def _load_override_targets(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    for row in _iter_jsonl(path):
        source_id = str(row.get("source_id") or "").strip()
        decoder_text = str(row.get("decoder_text") or "").strip()
        if source_id and decoder_text:
            targets[source_id] = decoder_text
    return targets


def _base_example_id(row: dict[str, Any]) -> str:
    example_id = str(row.get("example_id") or "").strip()
    if ":retained_output:" in example_id:
        example_id = example_id.split(":retained_output:", 1)[0]
    return example_id


def _write_split(*, source_path: Path, output_path: Path, overrides: dict[str, str], split: str) -> tuple[int, int]:
    total = 0
    rewritten = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl(source_path):
            total += 1
            base_id = _base_example_id(row)
            if base_id in overrides:
                row = dict(row)
                row["decoder_text"] = overrides[base_id]
                row["source_type"] = "agentkernel_retained_output_override"
                row["retained_output_override_source_id"] = base_id
                row["retained_output_override_split"] = split
                rewritten += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return total, rewritten


def build_override_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    override_manifest_path = Path(args.override_manifest).expanduser().resolve()
    source_manifest = _read_json(source_manifest_path)
    override_manifest = _read_json(override_manifest_path)
    overrides = _load_override_targets(Path(str(override_manifest["train_dataset_path"])))
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    train_total, train_rewritten = _write_split(
        source_path=Path(str(source_manifest["train_dataset_path"])),
        output_path=train_path,
        overrides=overrides,
        split="train",
    )
    eval_total, eval_rewritten = _write_split(
        source_path=Path(str(source_manifest["eval_dataset_path"])),
        output_path=eval_path,
        overrides=overrides,
        split="eval",
    )
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = dict(source_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_controller_override_dataset",
            "objective": str(args.objective),
            "manifest_path": str(manifest_path),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "source_manifest_path": str(source_manifest_path),
            "override_manifest_path": str(override_manifest_path),
            "train_examples": train_total,
            "eval_examples": eval_total,
            "total_examples": train_total + eval_total,
            "retained_output_override": {
                "override_targets": len(overrides),
                "train_rewritten": train_rewritten,
                "eval_rewritten": eval_rewritten,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--override-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="agentkernel_controller_retained_output_override")
    args = parser.parse_args()
    print(json.dumps(build_override_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
