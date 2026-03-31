from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
from typing import Any

from agent_kernel.config import KernelConfig

from ..tolbert.config import HybridTolbertSSMConfig
from ..tolbert.tokenization import build_decoder_vocabulary, decoder_tokenizer_stats
from .corpus_acquisition import external_corpus_examples, materialize_external_corpus


def materialize_universal_decoder_dataset(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    hybrid_config: HybridTolbertSSMConfig | None = None,
    eval_fraction: float = 0.1,
) -> dict[str, Any]:
    hybrid_config = hybrid_config or HybridTolbertSSMConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = collect_universal_decoder_examples(config=config, repo_root=repo_root)
    external_corpus_manifest = materialize_external_corpus(
        config=config,
        output_dir=output_dir / "external_corpus",
    )
    external_examples = external_corpus_examples(external_corpus_manifest)
    examples.extend(external_examples)
    train_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []
    for example in examples:
        if _is_eval_example(example, eval_fraction=eval_fraction):
            eval_examples.append(example)
        else:
            train_examples.append(example)
    if not train_examples and eval_examples:
        train_examples.append(eval_examples.pop(0))
    if not eval_examples and len(train_examples) > 1:
        eval_examples.append(train_examples.pop())
    decoder_vocab = build_decoder_vocabulary(
        [str(example.get("target", "")) for example in train_examples],
        hybrid_config,
    )
    tokenizer_stats = decoder_tokenizer_stats(
        [str(example.get("prompt", "")) for example in examples] + [str(example.get("target", "")) for example in examples]
    )
    train_path = output_dir / "universal_decoder_train.jsonl"
    eval_path = output_dir / "universal_decoder_eval.jsonl"
    decoder_vocab_path = output_dir / "universal_decoder_vocab.json"
    decoder_tokenizer_manifest_path = output_dir / "universal_decoder_tokenizer_manifest.json"
    manifest_path = output_dir / "universal_decoder_dataset_manifest.json"
    _write_jsonl(train_path, train_examples)
    _write_jsonl(eval_path, eval_examples)
    train_shard_paths = _write_jsonl_shards(output_dir / "shards", "universal_decoder_train", train_examples)
    eval_shard_paths = _write_jsonl_shards(output_dir / "shards", "universal_decoder_eval", eval_examples)
    decoder_vocab_path.write_text(json.dumps(decoder_vocab, indent=2, sort_keys=True), encoding="utf-8")
    decoder_tokenizer_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_decoder_tokenizer_manifest",
                "decoder_vocab_path": str(decoder_vocab_path),
                "decoder_vocab_size": len(decoder_vocab),
                **tokenizer_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_counts = Counter(str(example.get("source_type", "unknown")) for example in examples)
    manifest = {
        "artifact_kind": "tolbert_universal_decoder_dataset",
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "decoder_vocab_path": str(decoder_vocab_path),
        "decoder_tokenizer_manifest_path": str(decoder_tokenizer_manifest_path),
        "decoder_vocab_size": len(decoder_vocab),
        "decoder_tokenizer_stats": tokenizer_stats,
        "external_corpus_manifest_path": str(external_corpus_manifest.get("manifest_path", "")),
        "external_corpus_fetched_count": int(external_corpus_manifest.get("fetched_count", 0) or 0),
        "external_corpus_example_count": len(external_examples),
        "external_corpus_status_counts": dict(external_corpus_manifest.get("status_counts", {}))
        if isinstance(external_corpus_manifest.get("status_counts", {}), dict)
        else {},
        "external_corpus_quality": dict(external_corpus_manifest.get("quality", {}))
        if isinstance(external_corpus_manifest.get("quality", {}), dict)
        else {},
        "manifest_path": str(manifest_path),
        "config": hybrid_config.to_dict(),
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_shard_paths": [str(path) for path in train_shard_paths],
        "eval_shard_paths": [str(path) for path in eval_shard_paths],
        "source_counts": dict(sorted(source_counts.items())),
        "sources": sorted(source_counts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if config.uses_sqlite_storage():
        config.sqlite_store().record_export_manifest(
            export_key=f"dataset:universal_decoder:{output_dir.resolve()}",
            export_kind="universal_decoder_dataset",
            payload=manifest,
        )
    return manifest


def collect_universal_decoder_examples(
    *,
    config: KernelConfig,
    repo_root: Path,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    examples.extend(_trajectory_examples(config))
    examples.extend(_markdown_examples(repo_root / "docs"))
    examples.extend(_code_examples(repo_root / "agent_kernel"))
    examples.extend(_code_examples(repo_root / "scripts"))
    examples.extend(_report_examples(config.run_reports_dir))
    deduped: dict[str, dict[str, Any]] = {}
    for example in examples:
        prompt = str(example.get("prompt", "")).strip()
        target = str(example.get("target", "")).strip()
        if not prompt or not target:
            continue
        key = hashlib.sha256(f"{prompt}\n-->\n{target}".encode("utf-8")).hexdigest()
        deduped[key] = example
    return sorted(deduped.values(), key=lambda item: (str(item.get("source_type", "")), str(item.get("source_id", ""))))


def _trajectory_examples(config: KernelConfig) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    if config.uses_sqlite_storage():
        payloads = config.sqlite_store().iter_trajectory_payloads()
    else:
        for path in sorted(config.trajectories_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    for payload in payloads:
        task_id = str(payload.get("task_id", ""))
        prompt = str(payload.get("prompt", "")).strip()
        task_metadata = payload.get("task_metadata", {})
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        benchmark_family = str(task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        episode_success = bool(payload.get("success", False))
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            continue
        history: list[str] = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "")).strip() or "unknown"
            content = str(step.get("content", "")).strip()
            if not content:
                continue
            history_slice = "\n".join(history[-2:])
            examples.append(
                {
                    "source_type": "trajectory_step",
                    "source_id": f"{task_id}:{index}",
                    "prompt": "\n".join(
                        value
                        for value in [
                            f"task: {prompt}" if prompt else "",
                            f"action: {action}",
                            f"context:\n{history_slice}" if history_slice else "",
                        ]
                        if value
                    ),
                    "target": content,
                }
            )
            verification = step.get("verification", {})
            if (
                action == "code_execute"
                and content
                and (episode_success or (isinstance(verification, dict) and bool(verification.get("passed", False))))
            ):
                examples.append(
                    {
                        "source_type": "trajectory_success_command",
                        "source_id": f"{task_id}:{index}:success_command",
                        "prompt": "\n".join(
                            value
                            for value in [
                                f"task: {prompt}" if prompt else "",
                                f"benchmark_family: {benchmark_family}",
                                "goal: emit a verifier-aligned bounded shell command",
                                f"context:\n{history_slice}" if history_slice else "",
                            ]
                            if value
                        ),
                        "target": content,
                    }
                )
            history.append(content)
    return examples


def _markdown_examples(root: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not root.exists():
        return examples
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for index, block in enumerate(_text_blocks(text)):
            pair = _split_prompt_target(block)
            if pair is None:
                continue
            prompt, target = pair
            examples.append(
                {
                    "source_type": "docs_markdown",
                    "source_id": f"{path}:{index}",
                    "prompt": prompt,
                    "target": target,
                }
            )
    return examples


def _code_examples(root: Path, *, max_file_chars: int = 16000, max_blocks_per_file: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not root.exists():
        return examples
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".sh", ".toml", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        blocks = _code_blocks(text[:max_file_chars])[:max_blocks_per_file]
        for index, block in enumerate(blocks):
            pair = _split_prompt_target(block)
            if pair is None:
                continue
            prompt, target = pair
            examples.append(
                {
                    "source_type": "repo_source",
                    "source_id": f"{path}:{index}",
                    "prompt": prompt,
                    "target": target,
                }
            )
    return examples


def _report_examples(root: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not root.exists():
        return examples
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        acceptance = payload.get("acceptance_packet", {})
        if not isinstance(acceptance, dict):
            acceptance = {}
        selected_edits = acceptance.get("selected_edits", [])
        if not isinstance(selected_edits, list):
            selected_edits = []
        if not selected_edits:
            continue
        prompt = json.dumps(
            {
                "task_id": payload.get("task_id", path.stem),
                "benchmark_family": payload.get("task_metadata", {}).get("benchmark_family", ""),
                "selected_edits": selected_edits,
            },
            sort_keys=True,
        )
        target = json.dumps(
            {
                "verifier_result": acceptance.get("verifier_result", {}),
                "capability_usage": acceptance.get("capability_usage", {}),
                "tests": acceptance.get("tests", []),
            },
            sort_keys=True,
        )
        examples.append(
            {
                "source_type": "acceptance_report",
                "source_id": str(path),
                "prompt": prompt,
                "target": target,
            }
        )
    return examples


def _text_blocks(text: str, *, max_blocks: int = 12) -> list[str]:
    blocks = [block.strip() for block in str(text).split("\n\n")]
    return [block for block in blocks if len(block.split()) >= 8][:max_blocks]


def _code_blocks(text: str, *, max_lines: int = 24) -> list[str]:
    lines = [line.rstrip() for line in str(text).splitlines()]
    blocks: list[str] = []
    for start in range(0, len(lines), max_lines):
        block = "\n".join(lines[start : start + max_lines]).strip()
        if len(block.split()) >= 8:
            blocks.append(block)
    return blocks


def _split_prompt_target(text: str) -> tuple[str, str] | None:
    tokens = str(text).split()
    if len(tokens) < 8:
        return None
    split_index = max(4, int(len(tokens) * 0.6))
    split_index = min(split_index, len(tokens) - 2)
    prompt = " ".join(tokens[:split_index]).strip()
    target = " ".join(tokens[split_index:]).strip()
    if not prompt or not target:
        return None
    return prompt, target


def _is_eval_example(example: dict[str, Any], *, eval_fraction: float) -> bool:
    source_id = str(example.get("source_id", ""))
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:2], "big") / 65535.0
    return bucket < max(0.0, min(0.5, float(eval_fraction)))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_jsonl_shards(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    *,
    shard_size: int = 5000,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        shard_path = output_dir / f"{stem}_0000.jsonl"
        shard_path.write_text("", encoding="utf-8")
        return [shard_path]
    shard_paths: list[Path] = []
    normalized_size = max(1, int(shard_size))
    for shard_index, start in enumerate(range(0, len(rows), normalized_size)):
        shard_rows = rows[start : start + normalized_size]
        shard_path = output_dir / f"{stem}_{shard_index:04d}.jsonl"
        shard_path.write_text("".join(json.dumps(row) + "\n" for row in shard_rows), encoding="utf-8")
        shard_paths.append(shard_path)
    return shard_paths
