#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_stack_root(repo_root: Path) -> Path:
    return repo_root / "other_repos" / "model-stack"


def _install_model_stack_path(repo_root: Path) -> None:
    root = str(_model_stack_root(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        del padding, truncation
        ids = [byte + 4 for byte in str(text).encode("utf-8", errors="replace")]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        ids = ids[:max_length]
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention.append(0)
        return {"input_ids": ids, "attention_mask": attention}

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_kind": "byte_fallback_v1",
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "unk_token_id": self.unk_token_id,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class LocalBpeTokenizer:
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    default_agentkernel_special_tokens = [
        "<AK_USER>",
        "<AK_CHAT>",
        "<AK_THINK>",
        "<AK_DEEP_RESEARCH>",
        "<AK_CONTEXT>",
        "<AK_EVIDENCE>",
        "<AK_CANDIDATE>",
        "<AK_QUERY_REWRITE>",
        "<AK_RERANK>",
        "<AK_GATHER_CONTEXT>",
        "<AK_RESPOND>",
        "<AK_SUFFICIENT>",
        "<AK_INSUFFICIENT>",
        "<AK_ANSWER>",
        "<AK_JSON>",
    ]

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.token_to_id(self.pad_token))
        self.bos_token_id = int(tokenizer.token_to_id(self.bos_token))
        self.eos_token_id = int(tokenizer.token_to_id(self.eos_token))
        self.unk_token_id = int(tokenizer.token_to_id(self.unk_token))
        self.vocab_size = int(tokenizer.get_vocab_size())
        self.agentkernel_special_tokens = [
            token
            for token in self.default_agentkernel_special_tokens
            if tokenizer.token_to_id(token) is not None
        ]

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        encoding = self.tokenizer.encode(str(text), add_special_tokens=add_special_tokens)
        ids = list(encoding.ids)
        if truncation:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length":
            while len(ids) < max_length:
                ids.append(self.pad_token_id)
                attention.append(0)
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=True)

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_kind": "agentkernel_bytelevel_bpe_v1",
                    "pad_token": self.pad_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "unk_token": self.unk_token,
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "unk_token_id": self.unk_token_id,
                    "agentkernel_special_tokens": self.agentkernel_special_tokens,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _iter_jsonl_texts(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("encoder_text", "decoder_text"):
                text = str(row.get(key, "") or "").strip()
                if text:
                    yield text


def _tokenizer_training_texts(
    dataset_manifest: dict[str, Any],
    *,
    max_texts: int,
    special_tokens: list[str] | tuple[str, ...] = (),
) -> Iterator[str]:
    emitted = 0
    if special_tokens:
        yield " ".join(str(token) for token in special_tokens)
        emitted += 1
        if max_texts > 0 and emitted >= max_texts:
            return
    for key in ("train_dataset_path", "eval_dataset_path"):
        path_value = str(dataset_manifest.get(key, "") or "")
        if not path_value:
            continue
        for text in _iter_jsonl_texts(Path(path_value)):
            yield text
            emitted += 1
            if max_texts > 0 and emitted >= max_texts:
                return


def _train_agentkernel_bpe(
    dataset_manifest: dict[str, Any],
    *,
    vocab_size: int,
    max_texts: int,
    use_agentkernel_special_tokens: bool,
) -> LocalBpeTokenizer:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "AgentKernel BPE training requires the 'tokenizers' package. "
            "Install tokenizers or run with --tokenizer-kind byte."
        ) from exc

    special_tokens = [
        LocalBpeTokenizer.pad_token,
        LocalBpeTokenizer.bos_token,
        LocalBpeTokenizer.eos_token,
        LocalBpeTokenizer.unk_token,
    ]
    if use_agentkernel_special_tokens:
        special_tokens.extend(LocalBpeTokenizer.default_agentkernel_special_tokens)
    tokenizer = Tokenizer(BPE(unk_token=LocalBpeTokenizer.unk_token))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=max(int(vocab_size), len(special_tokens)),
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _tokenizer_training_texts(
            dataset_manifest,
            max_texts=int(max_texts),
            special_tokens=LocalBpeTokenizer.default_agentkernel_special_tokens if use_agentkernel_special_tokens else (),
        ),
        trainer=trainer,
    )
    bos_id = tokenizer.token_to_id(LocalBpeTokenizer.bos_token)
    eos_id = tokenizer.token_to_id(LocalBpeTokenizer.eos_token)
    if bos_id is None or eos_id is None:
        raise RuntimeError("trained AgentKernel BPE tokenizer is missing BOS/EOS special tokens")
    tokenizer.post_processor = TemplateProcessing(
        single=f"{LocalBpeTokenizer.bos_token} $A {LocalBpeTokenizer.eos_token}",
        pair=(
            f"{LocalBpeTokenizer.bos_token} $A {LocalBpeTokenizer.eos_token} "
            f"$B:1 {LocalBpeTokenizer.eos_token}:1"
        ),
        special_tokens=[
            (LocalBpeTokenizer.bos_token, int(bos_id)),
            (LocalBpeTokenizer.eos_token, int(eos_id)),
        ],
    )
    return LocalBpeTokenizer(tokenizer)


def _ensure_agentkernel_special_tokens(tokenizer) -> None:
    missing = [
        token
        for token in LocalBpeTokenizer.default_agentkernel_special_tokens
        if tokenizer.token_to_id(token) is None
    ]
    if missing:
        tokenizer.add_special_tokens(missing)


def _load_tokenizer(args: argparse.Namespace, *, dataset_manifest: dict[str, Any]):
    if bool(getattr(args, "byte_tokenizer", 0)):
        args.tokenizer_kind = "byte"
    tokenizer_kind = str(args.tokenizer_kind).strip().lower()
    if tokenizer_kind == "byte":
        return ByteTokenizer()
    if tokenizer_kind == "agentkernel-bpe":
        source_dir = str(getattr(args, "tokenizer_source_dir", "") or "").strip()
        source_tokenizer = Path(source_dir).expanduser().resolve() / "tokenizer.json" if source_dir else None
        existing_tokenizer = Path(str(args.output_dir)).expanduser().resolve() / "tokenizer" / "tokenizer.json"
        tokenizer_path = existing_tokenizer if existing_tokenizer.exists() else source_tokenizer
        if tokenizer_path is not None and tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Loading an existing AgentKernel BPE tokenizer requires the 'tokenizers' package."
                ) from exc
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            if bool(args.agentkernel_special_tokens):
                _ensure_agentkernel_special_tokens(tokenizer)
            return LocalBpeTokenizer(tokenizer)
        return _train_agentkernel_bpe(
            dataset_manifest,
            vocab_size=int(args.tokenizer_vocab_size),
            max_texts=int(args.tokenizer_max_texts),
            use_agentkernel_special_tokens=bool(args.agentkernel_special_tokens),
        )
    if tokenizer_kind != "hf":
        raise ValueError(f"unknown tokenizer kind: {args.tokenizer_kind}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_name), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return tokenizer


class EncDecJsonlDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_encoder_tokens: int,
        max_decoder_tokens: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("encoder_text", "")).strip() and str(row.get("decoder_text", "")).strip():
                    self.rows.append(row)
        self.tokenizer = tokenizer
        self.max_encoder_tokens = int(max_encoder_tokens)
        self.max_decoder_tokens = int(max_decoder_tokens)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.decoder_start_token_id = int(
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or self.pad_token_id
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        enc = self.tokenizer(
            str(row["encoder_text"]),
            max_length=self.max_encoder_tokens,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        target = self.tokenizer(
            str(row["decoder_text"]),
            max_length=self.max_decoder_tokens,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        labels = list(target["input_ids"])
        if labels and int(labels[0]) == self.decoder_start_token_id:
            labels = labels[1:]
        labels = labels[: self.max_decoder_tokens]
        while len(labels) < self.max_decoder_tokens:
            labels.append(self.pad_token_id)
        decoder_input_ids = [self.decoder_start_token_id, *labels[:-1]]
        label_ids = [token if token != self.pad_token_id else -100 for token in labels]
        return {
            "enc_input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "enc_attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "dec_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "loss_weight": torch.tensor(float(row.get("weight", 1.0) or 1.0), dtype=torch.float32),
        }


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _weighted_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        batch["enc_input_ids"],
        batch["dec_input_ids"],
        batch["enc_attention_mask"],
        None,
    )
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_losses = token_losses.reshape(batch["labels"].shape)
    valid_tokens = (batch["labels"] != -100).to(dtype=token_losses.dtype)
    per_example = (token_losses * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1.0)
    weights = batch["loss_weight"].to(dtype=per_example.dtype).clamp_min(0.0)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _evaluate(
    model: torch.nn.Module,
    dataset: EncDecJsonlDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    if len(dataset) <= 0:
        return {"eval_examples": 0, "eval_batches": 0, "eval_loss": None}
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    was_training = model.training
    model.eval()
    losses: list[float] = []
    seen_examples = 0
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            if max_batches > 0 and index > max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = _weighted_loss(model, batch)
            losses.append(float(loss.detach().cpu().item()))
            seen_examples += int(batch["labels"].shape[0])
    if was_training:
        model.train()
    return {
        "eval_examples": seen_examples,
        "eval_batches": len(losses),
        "eval_loss": sum(losses) / len(losses) if losses else None,
    }


def _preset_config(name: str, vocab_size: int):
    from specs.config import ModelConfig

    normalized = str(name).strip().lower()
    if normalized == "tiny":
        return ModelConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=int(vocab_size),
            attn_impl="eager",
            activation="silu",
            norm="layer",
            max_position_embeddings=512,
        )
    if normalized in {"100m", "agentkernel-lite-100m", "lite-100m"}:
        return ModelConfig(
            d_model=640,
            n_heads=10,
            n_layers=6,
            d_ff=2048,
            vocab_size=int(vocab_size),
            attn_impl="eager",
            activation="silu",
            norm="layer",
            max_position_embeddings=4096,
        )
    raise ValueError(f"unknown preset: {name}")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{int(step):08d}.pt"


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def _save_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[float],
    eval_history: list[dict[str, Any]],
    include_optimizer: bool,
) -> Path:
    path = _checkpoint_path(output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "losses": [float(loss) for loss in losses],
        "eval_history": eval_history,
        "include_optimizer": bool(include_optimizer),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    latest = path.parent / "latest.json"
    latest.write_text(json.dumps({"step": int(step), "path": str(path)}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    vocab_mismatch: str,
) -> tuple[int, list[float], list[dict[str, Any]]]:
    payload = torch.load(path, map_location=device)
    state = payload["model_state_dict"]
    if str(vocab_mismatch).strip().lower() == "expand":
        current = model.state_dict()
        patched: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key not in current:
                continue
            target = current[key]
            if tuple(tensor.shape) == tuple(target.shape):
                patched[key] = tensor
                continue
            if key in {"enc_embed.weight", "dec_embed.weight", "lm_head.weight"} and len(tensor.shape) == 2:
                if tensor.shape[1] != target.shape[1] or tensor.shape[0] > target.shape[0]:
                    raise RuntimeError(
                        f"cannot expand checkpoint tensor {key}: checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
                    )
                expanded = target.detach().clone()
                expanded[: tensor.shape[0], :].copy_(tensor.to(dtype=expanded.dtype, device=expanded.device))
                patched[key] = expanded
                continue
            raise RuntimeError(
                f"checkpoint tensor shape mismatch for {key}: checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
            )
        missing, unexpected = model.load_state_dict(patched, strict=False)
        unexpected = list(unexpected)
        missing = [key for key in missing if key not in {"enc_embed.weight", "dec_embed.weight", "lm_head.weight"}]
        if unexpected or missing:
            raise RuntimeError(f"partial checkpoint load mismatch: missing={missing} unexpected={unexpected}")
        optimizer = None
    else:
        model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    losses = [float(loss) for loss in payload.get("losses", [])]
    eval_history = payload.get("eval_history", [])
    if not isinstance(eval_history, list):
        eval_history = []
    return int(payload.get("step", 0) or 0), losses, eval_history


def _save_manifest(
    *,
    output_dir: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    dataset_manifest: dict[str, Any],
    config,
    parameter_count: int,
    tokenizer_kind: str,
    tokenizer_name: str,
    training_summary: dict[str, Any],
    browser_bitnet_manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "agentkernel_lite_encdec_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_bundle",
        "model_family": "agentkernel_lite_encdec_v1",
        "chat_contract": {
            "primary_action": "respond",
            "code_execution": False,
            "structured_decision": True,
            "extensions_may_be_suggested": True,
        },
        "manifest_path": str(manifest_path),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_name": tokenizer_name,
        "dataset_manifest_path": str(dataset_manifest.get("manifest_path", "")),
        "model_config": asdict(config),
        "parameter_count": int(parameter_count),
        "training_summary": training_summary,
        "runtime_targets": {
            "browser": "model_stack_browser_bitnet_encoder_decoder",
            "kernel": "agentkernel_lite_rust_wasm_loop",
        },
        "replaces_surfaces": ["chat_decision_generation", "context_grounded_reply_generation"],
    }
    if browser_bitnet_manifest_path is not None:
        manifest["browser_bitnet_manifest_path"] = str(browser_bitnet_manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _attach_tokenizer_to_browser_bitnet(
    *,
    browser_bitnet_manifest_path: Path,
    tokenizer_dir: Path,
    tokenizer_kind: str,
    tokenizer,
) -> None:
    if not browser_bitnet_manifest_path.exists():
        return
    browser_dir = browser_bitnet_manifest_path.parent
    target_dir = browser_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in tokenizer_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)
    payload = json.loads(browser_bitnet_manifest_path.read_text(encoding="utf-8"))
    payload["tokenizer"] = {
        "kind": str(tokenizer_kind),
        "path": "tokenizer/tokenizer.json" if (target_dir / "tokenizer.json").exists() else "tokenizer/tokenizer_config.json",
        "config_path": "tokenizer/tokenizer_config.json",
        "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        "bos_token_id": int(getattr(tokenizer, "bos_token_id", 1) or 1),
        "eos_token_id": int(getattr(tokenizer, "eos_token_id", 2) or 2),
        "unk_token_id": int(getattr(tokenizer, "unk_token_id", 3) or 3),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer)),
    }
    browser_bitnet_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    _install_model_stack_path(repo_root)
    _seed_everything(int(args.seed))

    from runtime.checkpoint import save_pretrained
    from runtime.seq2seq import EncoderDecoderLM
    from export.exporter import export_model
    from specs.export import ExportConfig

    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / "model"
    tokenizer_dir = output_dir / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args, dataset_manifest=dataset_manifest)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    config = _preset_config(str(args.preset), vocab_size=vocab_size)
    config.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=vocab_size)
    _materialize_lazy_modules(model)
    parameter_count = _parameter_count(model)
    tokenizer_kind = "byte" if bool(args.byte_tokenizer) else str(args.tokenizer_kind)
    tokenizer_name = str(args.tokenizer_name) if tokenizer_kind == "hf" else tokenizer_kind

    dry_summary = {
        "dry_run": bool(args.dry_run),
        "preset": str(args.preset),
        "device": str(args.device),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "parameter_count": int(parameter_count),
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_vocab_size": int(vocab_size),
        "dataset_objective": str(dataset_manifest.get("objective", "")),
        "primary_action": "respond" if str(dataset_manifest.get("objective", "")).lower() in {"chat", "text"} else "mixed",
        "code_execution": False,
        "extension_counts": dict(dataset_manifest.get("extension_counts", {}) or {}),
        "checkpoint_every": int(args.checkpoint_every),
        "eval_every": int(args.eval_every),
        "train_dataset_path": str(dataset_manifest.get("train_dataset_path", "")),
        "eval_dataset_path": str(dataset_manifest.get("eval_dataset_path", "")),
    }
    if bool(args.dry_run):
        tokenizer.save_pretrained(str(tokenizer_dir))
        return _save_manifest(
            output_dir=output_dir,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            dataset_manifest=dataset_manifest,
            config=config,
            parameter_count=parameter_count,
            tokenizer_kind=tokenizer_kind,
            tokenizer_name=tokenizer_name,
            training_summary=dry_summary,
        )

    train_dataset = EncDecJsonlDataset(
        Path(str(dataset_manifest["train_dataset_path"])),
        tokenizer,
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_decoder_tokens=int(args.max_decoder_tokens),
    )
    if len(train_dataset) <= 0:
        raise SystemExit("training dataset is empty")
    eval_dataset = None
    eval_path_value = str(dataset_manifest.get("eval_dataset_path", "") or "")
    eval_path = Path(eval_path_value) if eval_path_value else None
    if eval_path is not None and eval_path.exists():
        eval_dataset = EncDecJsonlDataset(
            eval_path,
            tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
        )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
    )
    device = torch.device(str(args.device))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    resume_path: Path | None = None
    if str(args.resume_from).strip():
        resume_path = Path(str(args.resume_from)).expanduser().resolve()
    elif bool(args.resume_latest):
        resume_path = _latest_checkpoint(output_dir)
    losses: list[float] = []
    eval_history: list[dict[str, Any]] = []
    start_step = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_step, losses, eval_history = _load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            vocab_mismatch=str(args.checkpoint_vocab_mismatch),
        )
        print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "step": start_step}, sort_keys=True))
    iterator = iter(loader)
    last_step = start_step
    for step in range(start_step + 1, int(args.max_steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = _weighted_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        last_step = step
        if step % max(1, int(args.log_every)) == 0:
            print(json.dumps({"step": step, "loss": loss_value}, sort_keys=True))
        if eval_dataset is not None and int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
            eval_result = _evaluate(
                model,
                eval_dataset,
                batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size),
                device=device,
                max_batches=int(args.max_eval_batches),
            )
            eval_result = {"step": step, **eval_result}
            eval_history.append(eval_result)
            print(json.dumps({"event": "eval", **eval_result}, sort_keys=True))
        if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
            checkpoint_path = _save_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                losses=losses,
                eval_history=eval_history,
                include_optimizer=bool(args.checkpoint_include_optimizer),
            )
            print(json.dumps({"event": "checkpoint", "path": str(checkpoint_path), "step": step}, sort_keys=True))

    if bool(args.save_final_checkpoint) and last_step > start_step:
        checkpoint_path = _save_training_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            step=last_step,
            losses=losses,
            eval_history=eval_history,
            include_optimizer=bool(args.checkpoint_include_optimizer),
        )
        print(json.dumps({"event": "checkpoint", "final": True, "path": str(checkpoint_path), "step": last_step}, sort_keys=True))

    tokenizer.save_pretrained(str(tokenizer_dir))
    save_pretrained(model.eval().cpu(), config, str(model_dir))
    browser_bitnet_manifest_path = None
    if bool(args.export_browser_bitnet):
        browser_bitnet_manifest_path = export_model(
            model,
            ExportConfig(
                target="browser-bitnet",
                outdir=str(output_dir / "browser_bitnet"),
                quantize="bitnet",
                quant_spin=False,
                quant_weight_opt="none",
                quant_activation_quant=None,
                max_seq_len=max(int(args.max_encoder_tokens), int(args.max_decoder_tokens)),
            ),
            model_cfg=config,
        )
        _attach_tokenizer_to_browser_bitnet(
            browser_bitnet_manifest_path=Path(browser_bitnet_manifest_path),
            tokenizer_dir=tokenizer_dir,
            tokenizer_kind=tokenizer_kind,
            tokenizer=tokenizer,
        )
    training_summary = {
        **dry_summary,
        "dry_run": False,
        "completed_steps": int(last_step),
        "start_step": int(start_step),
        "resumed_from": "" if resume_path is None else str(resume_path),
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "eval_history": eval_history,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "browser_bitnet_exported": browser_bitnet_manifest_path is not None,
    }
    return _save_manifest(
        output_dir=output_dir,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        dataset_manifest=dataset_manifest,
        config=config,
        parameter_count=parameter_count,
        tokenizer_kind=tokenizer_kind,
        tokenizer_name=tokenizer_name,
        training_summary=training_summary,
        browser_bitnet_manifest_path=browser_bitnet_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/run")
    parser.add_argument("--preset", default="agentkernel-lite-100m", choices=("agentkernel-lite-100m", "100m", "tiny"))
    parser.add_argument("--tokenizer-kind", default="byte", choices=("agentkernel-bpe", "hf", "byte"))
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer-source-dir", default="")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-max-texts", type=int, default=200000)
    parser.add_argument("--agentkernel-special-tokens", type=int, choices=(0, 1), default=1)
    parser.add_argument("--byte-tokenizer", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-decoder-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-include-optimizer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--checkpoint-vocab-mismatch", choices=("strict", "expand"), default="strict")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-latest", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-final-checkpoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    parser.add_argument("--export-browser-bitnet", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    manifest = train(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
