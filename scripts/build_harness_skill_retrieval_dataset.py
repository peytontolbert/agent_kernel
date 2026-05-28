#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


SCHEMA_VERSION = "agentkernel_harness_skill_retrieval_dataset_v1"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [part.strip() for part in raw.splitlines() if part.strip()]
        return _list_text(parsed)
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            if isinstance(item, (list, tuple, set)):
                items.extend(_list_text(item))
            elif str(item).strip():
                items.append(f"{key}: {item}")
        return list(dict.fromkeys(items))
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    raw = str(value).strip()
    return [raw] if raw else []


def _compact(text: Any, *, limit: int) -> str:
    value = " ".join(_text(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _stable_float(key: str) -> float:
    return int(_sha256_text(key)[:12], 16) / float(16**12)


def _load_rows(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("building the harness skill dataset requires pyarrow") from exc
    rows: list[dict[str, Any]] = []
    sources = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    for source in sources:
        parquet_file = pq.ParquetFile(source)
        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    rows.append(row)
                    if max_rows > 0 and len(rows) >= max_rows:
                        return rows
    return rows


def _skill_id(row: dict[str, Any]) -> str:
    for key in ("id", "skill_id", "source_id"):
        value = _text(row.get(key))
        if value:
            return value
    basis = "|".join(
        _text(row.get(key))
        for key in ("source_repo", "source_path", "skill_kind", "qualname", "line_start", "line_end")
    )
    return _sha256_text(basis)[:24]


def _label_values(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    labels.extend(_list_text(row.get("annotation_primitive_labels")))
    for key in ("primitive_type", "primitive_subtype", "skill_kind", "language"):
        value = _text(row.get(key))
        if value:
            labels.append(value)
    return list(dict.fromkeys(label.lower() for label in labels if label))


def _query_text(row: dict[str, Any]) -> str:
    path = _text(row.get("source_path"))
    repo = _text(row.get("source_repo") or row.get("dataset_repo_label"))
    summary = _compact(row.get("annotation_summary"), limit=280)
    use_when = "; ".join(_compact(item, limit=180) for item in _list_text(row.get("annotation_use_when"))[:4])
    patch_relevance = "; ".join(
        _compact(item, limit=180) for item in _list_text(row.get("annotation_patch_relevance"))[:4]
    )
    verification = "; ".join(
        _compact(item, limit=140) for item in _list_text(row.get("annotation_verification_hints"))[:3]
    )
    labels = ", ".join(_label_values(row)[:8])
    parts = [
        "<AK_USER> Retrieve harness skills that can improve an AgentKernel patch attempt.",
        f"repo: {repo}" if repo else "",
        f"path: {path}" if path else "",
        f"skill labels: {labels}" if labels else "",
        f"summary: {summary}" if summary else "",
        f"use when: {use_when}" if use_when else "",
        f"patch relevance: {patch_relevance}" if patch_relevance else "",
        f"verification: {verification}" if verification else "",
        "<AK_RETRIEVE> <AK_RET_SKILLS> <AK_RET_SEMANTIC>",
    ]
    return "\n".join(part for part in parts if part)


def _doc_text(row: dict[str, Any], *, max_excerpt_chars: int) -> str:
    skill_id = _skill_id(row)
    repo = _text(row.get("source_repo") or row.get("dataset_repo_label"))
    path = _text(row.get("source_path"))
    qualname = _text(row.get("qualname") or row.get("module"))
    language = _text(row.get("language"))
    primitive = _text(row.get("primitive_type"))
    subtype = _text(row.get("primitive_subtype"))
    kind = _text(row.get("skill_kind"))
    summary = _compact(row.get("annotation_summary"), limit=420)
    use_when = "; ".join(_compact(item, limit=180) for item in _list_text(row.get("annotation_use_when"))[:5])
    patch_relevance = "; ".join(
        _compact(item, limit=180) for item in _list_text(row.get("annotation_patch_relevance"))[:5]
    )
    risks = "; ".join(_compact(item, limit=140) for item in _list_text(row.get("annotation_risks"))[:4])
    verification = "; ".join(
        _compact(item, limit=140) for item in _list_text(row.get("annotation_verification_hints"))[:4]
    )
    side_effects = ", ".join(_list_text(row.get("side_effects"))[:8])
    permissions = ", ".join(_list_text(row.get("required_permissions"))[:8])
    excerpt = _text(row.get("source_excerpt"))
    if max_excerpt_chars > 0 and len(excerpt) > max_excerpt_chars:
        excerpt = excerpt[:max_excerpt_chars].rstrip() + "\n..."
    parts = [
        f"skill_id: {skill_id}",
        f"repo: {repo}" if repo else "",
        f"path: {path}" if path else "",
        f"symbol: {qualname}" if qualname else "",
        f"language: {language}" if language else "",
        f"kind: {kind}" if kind else "",
        f"primitive: {primitive}/{subtype}" if primitive or subtype else "",
        f"summary: {summary}" if summary else "",
        f"use_when: {use_when}" if use_when else "",
        f"patch_relevance: {patch_relevance}" if patch_relevance else "",
        f"verification: {verification}" if verification else "",
        f"side_effects: {side_effects}" if side_effects else "",
        f"permissions: {permissions}" if permissions else "",
        f"risks: {risks}" if risks else "",
        "source_excerpt:",
        excerpt,
    ]
    return "\n".join(part for part in parts if part)


def _negative_groups(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        labels = _label_values(row)
        for label in labels[:8]:
            groups[f"label:{label}"].append(index)
        for key in ("language", "skill_kind", "primitive_type", "dataset_repo_label"):
            value = _text(row.get(key)).lower()
            if value:
                groups[f"{key}:{value}"].append(index)
    return groups


def _choose_negatives(
    index: int,
    row: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    docs: list[str],
    groups: dict[str, list[int]],
    negative_count: int,
    rng: random.Random,
) -> list[str]:
    if negative_count <= 0 or len(rows) < 2:
        return []
    candidates: list[int] = []

    def add_from_group(group_key: str, *, limit: int) -> None:
        group = groups.get(group_key, [])
        if not group:
            return
        # Common labels such as documentation can contain tens of thousands of
        # rows. Sample a bounded window so full dataset builds remain linear.
        draw_count = min(len(group), max(limit, negative_count * 3))
        if draw_count >= len(group):
            sample = list(group)
        else:
            sample = rng.sample(group, draw_count)
        candidates.extend(candidate for candidate in sample if candidate != index)

    for label in _label_values(row)[:5]:
        add_from_group(f"label:{label}", limit=64)
    for key in ("language", "skill_kind", "primitive_type"):
        value = _text(row.get(key)).lower()
        if value:
            add_from_group(f"{key}:{value}", limit=48)
    candidates = [candidate for candidate in dict.fromkeys(candidates) if candidate != index]
    rng.shuffle(candidates)
    negatives: list[int] = []
    for candidate in candidates:
        if candidate not in negatives:
            negatives.append(candidate)
        if len(negatives) >= negative_count:
            break
    while len(negatives) < negative_count:
        candidate = rng.randrange(len(rows))
        if candidate != index and candidate not in negatives:
            negatives.append(candidate)
    return [docs[candidate] for candidate in negatives]


def _training_row(
    row: dict[str, Any],
    *,
    doc_text: str,
    negatives: list[str],
) -> dict[str, Any]:
    skill_id = _skill_id(row)
    labels = _label_values(row)
    confidence_raw = row.get("annotation_confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.65
    confidence = max(0.05, min(1.0, confidence))
    query = _query_text(row)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "skill_id": skill_id,
        "source_repo": _text(row.get("source_repo") or row.get("dataset_repo_label")),
        "source_path": _text(row.get("source_path")),
        "language": _text(row.get("language")),
        "skill_kind": _text(row.get("skill_kind")),
        "primitive_type": _text(row.get("primitive_type")),
        "primitive_subtype": _text(row.get("primitive_subtype")),
        "annotation_primitive_labels": labels,
    }
    return {
        "source_id": skill_id,
        "encoder_text": query,
        "decoder_text": "<AK_GATHER_CONTEXT> <AK_RETRIEVE> <AK_RET_SKILLS> <AK_CONF_MEDIUM>",
        "action": "gather_context",
        "task_type": "harness_skill_retrieval",
        "weight": 0.0,
        "distill_loss_weight": 0.0,
        "retrieval_query_text": query,
        "retrieval_doc_text": doc_text,
        "retrieval_negative_doc_texts": json.dumps(negatives, ensure_ascii=False),
        "retrieval_loss_weight": confidence,
        "query_confidence_target": confidence,
        "retrieval_coverage_target": 0.85 if labels else 0.65,
        "ood_query_target": 0.05,
        "ood_evidence_target": 0.05,
        "answer_confidence_target": confidence,
        "needs_verification_target": 0.35,
        "paper_action_validity_target": 1.0,
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("writing harness skill Parquet requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("source_id", pa.string()),
            pa.field("encoder_text", pa.string()),
            pa.field("decoder_text", pa.string()),
            pa.field("action", pa.string()),
            pa.field("task_type", pa.string()),
            pa.field("weight", pa.float32()),
            pa.field("distill_loss_weight", pa.float32()),
            pa.field("retrieval_query_text", pa.string()),
            pa.field("retrieval_doc_text", pa.string()),
            pa.field("retrieval_negative_doc_texts", pa.string()),
            pa.field("retrieval_loss_weight", pa.float32()),
            pa.field("query_confidence_target", pa.float32()),
            pa.field("retrieval_coverage_target", pa.float32()),
            pa.field("ood_query_target", pa.float32()),
            pa.field("ood_evidence_target", pa.float32()),
            pa.field("answer_confidence_target", pa.float32()),
            pa.field("needs_verification_target", pa.float32()),
            pa.field("paper_action_validity_target", pa.float32()),
            pa.field("metadata", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _write_split_shards(
    output_dir: Path,
    *,
    split: str,
    rows: list[dict[str, Any]],
    shard_size: int,
) -> list[str]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for offset in range(0, len(rows), shard_size):
        shard_index = offset // shard_size
        path = split_dir / f"part-{shard_index:05d}.parquet"
        _write_parquet(path, rows[offset : offset + shard_size])
        paths.append(str(path))
    if not paths:
        path = split_dir / "part-00000.parquet"
        _write_parquet(path, [])
        paths.append(str(path))
    return paths


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.skills_parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = _load_rows(source, max_rows=max(0, int(args.max_rows)))
    rows = [row for row in rows if _text(row.get("source_excerpt")) or _text(row.get("annotation_summary"))]
    if len(rows) < 2:
        raise SystemExit("need at least two skill rows with source or annotation text")
    rng = random.Random(int(args.seed))
    docs = [_doc_text(row, max_excerpt_chars=int(args.max_excerpt_chars)) for row in rows]
    groups = _negative_groups(rows)
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        negatives = _choose_negatives(
            index,
            row,
            rows=rows,
            docs=docs,
            groups=groups,
            negative_count=int(args.negative_count),
            rng=rng,
        )
        examples.append(_training_row(row, doc_text=docs[index], negatives=negatives))
    rng.shuffle(examples)
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    eval_fraction = max(0.0, min(0.5, float(args.eval_fraction)))
    for row in examples:
        key = str(row.get("source_id", "")) or str(len(train_rows) + len(eval_rows))
        if _stable_float(f"{int(args.seed)}:{key}") < eval_fraction:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    if not eval_rows:
        eval_rows.append(train_rows.pop())
    if not train_rows:
        train_rows.append(eval_rows.pop())
    train_paths = _write_split_shards(output_dir, split="train", rows=train_rows, shard_size=int(args.shard_size))
    eval_paths = _write_split_shards(output_dir, split="eval", rows=eval_rows, shard_size=int(args.shard_size))
    source_counts = Counter(_text(row.get("source_repo") or row.get("dataset_repo_label")) or "unknown" for row in rows)
    skill_kind_counts = Counter(_text(row.get("skill_kind")) or "unknown" for row in rows)
    primitive_counts = Counter(_text(row.get("primitive_type")) or "unknown" for row in rows)
    manifest_path = output_dir / "agentkernel_harness_skill_retrieval_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "schema_version": SCHEMA_VERSION,
        "objective": "harness_skill_retrieval",
        "dataset_format": "parquet",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(output_dir / "train"),
        "eval_dataset_path": str(output_dir / "eval"),
        "train_shards": len(train_paths),
        "eval_shards": len(eval_paths),
        "source_dataset_path": str(source),
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "retrieval_pair_count": len(examples),
        "negative_count": int(args.negative_count),
        "max_excerpt_chars": int(args.max_excerpt_chars),
        "eval_fraction": float(args.eval_fraction),
        "seed": int(args.seed),
        "source_counts": dict(sorted(source_counts.items())),
        "skill_kind_counts": dict(sorted(skill_kind_counts.items())),
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "task_type_counts": {"harness_skill_retrieval": len(examples)},
        "action_counts": {"gather_context": len(examples)},
        "agentkernel_special_tokens": [
            "<AK_USER>",
            "<AK_RETRIEVE>",
            "<AK_RET_SKILLS>",
            "<AK_RET_SEMANTIC>",
            "<AK_GATHER_CONTEXT>",
            "<AK_CONF_MEDIUM>",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills-parquet",
        default="/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet",
        help="Repo-skills-miner Hugging Face style Parquet file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--max-excerpt-chars", type=int, default=3200)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
