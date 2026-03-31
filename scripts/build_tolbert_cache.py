from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "other_repos" / "TOLBERT"))
sys.path.insert(0, str(_repo_root()))

from agent_kernel.modeling.tolbert.delta import materialize_tolbert_checkpoint_from_delta


def _build_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    from tolbert.modeling import TOLBERT, TOLBERTConfig

    model_cfg = TOLBERTConfig(
        base_model_name=cfg["base_model_name"],
        level_sizes=cfg["level_sizes"],
        proj_dim=cfg.get("proj_dim", 256),
        lambda_hier=cfg.get("lambda_hier", 1.0),
        lambda_path=cfg.get("lambda_path", 0.0),
        lambda_contrast=cfg.get("lambda_contrast", 0.0),
    )
    model = TOLBERT(model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _resolved_checkpoint_path(
    *,
    checkpoint: Path,
    parent_checkpoint: Path | None,
    checkpoint_delta: Path | None,
) -> Path:
    if checkpoint.exists():
        return checkpoint
    if parent_checkpoint is None or checkpoint_delta is None:
        return checkpoint
    materialized = checkpoint_delta.parent / ".materialized_checkpoints" / f"{checkpoint_delta.stem}__cache_materialized.pt"
    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_checkpoint_path=checkpoint_delta,
        output_checkpoint_path=materialized,
    )
    return materialized


def _checkpoint_mtime(*, checkpoint: Path, parent_checkpoint: Path | None, checkpoint_delta: Path | None) -> float:
    candidates = [checkpoint]
    if parent_checkpoint is not None:
        candidates.append(parent_checkpoint)
    if checkpoint_delta is not None:
        candidates.append(checkpoint_delta)
    mtimes: list[float] = []
    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _encode_batch(
    *,
    model: TOLBERT,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not records:
        return torch.empty(0), []

    tokens = tokenizer(
        [str(record.get("text", "")) for record in records],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    with torch.no_grad():
        out = model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )
        batch_embs = out["proj"].detach().cpu()
    return batch_embs, list(records)


def _encode_spans(
    *,
    model: TOLBERT,
    tokenizer: Any,
    spans_path: Path,
    max_length: int,
    device: torch.device,
    batch_size: int,
    progress_every: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    embs: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    encoded = 0

    with spans_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch_records.append(json.loads(line))
            if len(batch_records) < batch_size:
                continue
            batch_embs, batch_metas = _encode_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch_records,
                max_length=max_length,
                device=device,
            )
            embs.append(batch_embs)
            metas.extend(batch_metas)
            encoded += len(batch_metas)
            if progress_every > 0 and encoded % progress_every == 0:
                print(f"encoded={encoded}")
            batch_records = []

    if batch_records:
        batch_embs, batch_metas = _encode_batch(
            model=model,
            tokenizer=tokenizer,
            records=batch_records,
            max_length=max_length,
            device=device,
        )
        embs.append(batch_embs)
        metas.extend(batch_metas)
        encoded += len(batch_metas)
        if progress_every > 0:
            print(f"encoded={encoded}")

    if not embs:
        return torch.empty(0), []
    return torch.cat(embs, dim=0), metas


def _save_cache_payload(
    *,
    out_path: Path,
    embs: torch.Tensor,
    metas: list[dict[str, Any]],
    spans_mtime: float,
    ckpt_mtime: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embs": embs,
            "metas": metas,
            "spans_mtime": spans_mtime,
            "ckpt_mtime": ckpt_mtime,
        },
        out_path,
    )


def _shard_layout(out_path: Path) -> tuple[Path, Path, str]:
    if out_path.suffix:
        shard_dir = out_path.parent
        manifest_path = out_path.with_suffix(".json")
        shard_prefix = out_path.stem
    else:
        shard_dir = out_path
        manifest_path = shard_dir / "cache_manifest.json"
        shard_prefix = "cache"
    return shard_dir, manifest_path, shard_prefix


def _branch_presence(metas: list[dict[str, Any]]) -> dict[str, list[int]]:
    presence: dict[int, set[int]] = {}
    for meta in metas:
        node_path = meta.get("node_path")
        if not isinstance(node_path, list):
            continue
        for position, node_id in enumerate(node_path):
            if position == 0:
                continue
            presence.setdefault(position, set()).add(int(node_id))
    return {
        str(position): sorted(node_ids)
        for position, node_ids in sorted(presence.items())
    }


def _write_sharded_cache(
    *,
    model: TOLBERT,
    tokenizer: Any,
    spans_path: Path,
    out_path: Path,
    max_length: int,
    device: torch.device,
    batch_size: int,
    shard_size: int,
    progress_every: int,
    checkpoint: Path,
) -> tuple[list[Path], Path, int]:
    shard_dir, manifest_path, shard_prefix = _shard_layout(out_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    spans_mtime = spans_path.stat().st_mtime
    ckpt_mtime = checkpoint.stat().st_mtime
    batch_records: list[dict[str, Any]] = []
    shard_embs: list[torch.Tensor] = []
    shard_metas: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    total_encoded = 0
    shard_index = 0

    def _flush_shard() -> None:
        nonlocal shard_embs, shard_metas, shard_index
        if not shard_metas:
            return
        shard_index += 1
        shard_path = shard_dir / f"{shard_prefix}.shard{shard_index:05d}.pt"
        _save_cache_payload(
            out_path=shard_path,
            embs=torch.cat(shard_embs, dim=0),
            metas=list(shard_metas),
            spans_mtime=spans_mtime,
            ckpt_mtime=ckpt_mtime,
        )
        shard_paths.append(shard_path)
        shard_records.append(
            {
                "path": str(shard_path),
                "name": shard_path.name,
                "branch_presence": _branch_presence(shard_metas),
                "span_count": len(shard_metas),
            }
        )
        print(f"cache_shard={shard_path} spans={len(shard_metas)}")
        shard_embs = []
        shard_metas = []

    with spans_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch_records.append(json.loads(line))
            if len(batch_records) < batch_size:
                continue
            batch_embs, batch_metas = _encode_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch_records,
                max_length=max_length,
                device=device,
            )
            shard_embs.append(batch_embs)
            shard_metas.extend(batch_metas)
            total_encoded += len(batch_metas)
            if progress_every > 0 and total_encoded % progress_every == 0:
                print(f"encoded={total_encoded}")
            if len(shard_metas) >= shard_size:
                _flush_shard()
            batch_records = []

    if batch_records:
        batch_embs, batch_metas = _encode_batch(
            model=model,
            tokenizer=tokenizer,
            records=batch_records,
            max_length=max_length,
            device=device,
        )
        shard_embs.append(batch_embs)
        shard_metas.extend(batch_metas)
        total_encoded += len(batch_metas)
        if progress_every > 0:
            print(f"encoded={total_encoded}")

    _flush_shard()

    manifest = {
        "artifact_kind": "tolbert_cache_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spans_path": str(spans_path),
        "checkpoint_path": str(checkpoint),
        "cache_paths": [str(path) for path in shard_paths],
        "shards": shard_records,
        "num_shards": len(shard_paths),
        "total_spans": total_encoded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"cache_manifest={manifest_path}")
    return shard_paths, manifest_path, total_encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--checkpoint-delta", default=None)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    parent_checkpoint = Path(args.parent_checkpoint) if args.parent_checkpoint else None
    checkpoint_delta = Path(args.checkpoint_delta) if args.checkpoint_delta else None
    spans_path = Path(args.spans)
    out_path = Path(args.out)
    device = torch.device(args.device)

    from tolbert.config import load_tolbert_config
    from transformers import AutoTokenizer

    config = load_tolbert_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], cache_dir="/data/checkpoints/")
    resolved_checkpoint = _resolved_checkpoint_path(
        checkpoint=checkpoint,
        parent_checkpoint=parent_checkpoint,
        checkpoint_delta=checkpoint_delta,
    )
    model = _build_model(config, resolved_checkpoint, device)
    shard_size = max(0, int(args.shard_size))
    if shard_size > 0:
        shard_paths, manifest_path, total_spans = _write_sharded_cache(
            model=model,
            tokenizer=tokenizer,
            spans_path=spans_path,
            out_path=out_path,
            max_length=int(config.get("max_length", 256)),
            device=device,
            batch_size=max(1, args.batch_size),
            shard_size=shard_size,
            progress_every=max(0, args.progress_every),
            checkpoint=resolved_checkpoint,
        )
        print(f"cache_shards={len(shard_paths)}")
        print(f"cache_manifest={manifest_path}")
        print(f"spans={total_spans}")
        return
    embs, metas = _encode_spans(
        model=model,
        tokenizer=tokenizer,
        spans_path=spans_path,
        max_length=int(config.get("max_length", 256)),
        device=device,
        batch_size=max(1, args.batch_size),
        progress_every=max(0, args.progress_every),
    )

    _save_cache_payload(
        out_path=out_path,
        embs=embs,
        metas=metas,
        spans_mtime=spans_path.stat().st_mtime,
        ckpt_mtime=_checkpoint_mtime(
            checkpoint=resolved_checkpoint,
            parent_checkpoint=parent_checkpoint,
            checkpoint_delta=checkpoint_delta,
        ),
    )
    print(f"cache={out_path}")
    print(f"spans={embs.size(0)}")


if __name__ == "__main__":
    main()
