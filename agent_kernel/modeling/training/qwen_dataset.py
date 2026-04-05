from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
from typing import Any

from agent_kernel.config import KernelConfig

from .universal_dataset import collect_universal_decoder_examples


def materialize_qwen_sft_dataset(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    eval_fraction: float = 0.1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = collect_universal_decoder_examples(config=config, repo_root=repo_root)
    train_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []
    for example in examples:
        record = _qwen_sft_record(example)
        if not record:
            continue
        if _is_eval_example(record, eval_fraction=eval_fraction):
            eval_examples.append(record)
        else:
            train_examples.append(record)
    if not train_examples and eval_examples:
        train_examples.append(eval_examples.pop(0))
    if not eval_examples and len(train_examples) > 1:
        eval_examples.append(train_examples.pop())

    train_path = output_dir / "qwen_sft_train.jsonl"
    eval_path = output_dir / "qwen_sft_eval.jsonl"
    manifest_path = output_dir / "qwen_sft_dataset_manifest.json"
    _write_jsonl(train_path, train_examples)
    _write_jsonl(eval_path, eval_examples)

    difficulty_counts = Counter(
        str(example.get("difficulty", "seed")).strip() or "seed"
        for example in [*train_examples, *eval_examples]
    )
    source_counts = Counter(
        str(example.get("source_type", "unknown")).strip() or "unknown"
        for example in [*train_examples, *eval_examples]
    )
    manifest = {
        "artifact_kind": "qwen_sft_dataset",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_examples) + len(eval_examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "long_horizon_example_count": int(difficulty_counts.get("long_horizon", 0) or 0),
        "source_counts": dict(sorted(source_counts.items())),
        "format": {
            "type": "chat_jsonl",
            "messages_key": "messages",
            "weight_key": "weight",
            "metadata_key": "metadata",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _qwen_sft_record(example: dict[str, Any]) -> dict[str, Any]:
    prompt = str(example.get("prompt", "")).strip()
    target = str(example.get("target", "")).strip()
    if not prompt or not target:
        return {}
    difficulty = str(example.get("difficulty", "seed")).strip() or "seed"
    return {
        "messages": [
            {"role": "system", "content": "You are a coding agent operating inside the kernel runtime."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "weight": float(example.get("example_weight", 1.0) or 1.0),
        "metadata": {
            "source_type": str(example.get("source_type", "")).strip(),
            "source_id": str(example.get("source_id", "")).strip(),
            "difficulty": difficulty,
        },
        "source_type": str(example.get("source_type", "")).strip(),
        "difficulty": difficulty,
    }


def _is_eval_example(example: dict[str, Any], *, eval_fraction: float) -> bool:
    stable_key = json.dumps(example.get("metadata", {}), sort_keys=True)
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < max(0.0, min(0.9, float(eval_fraction)))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
