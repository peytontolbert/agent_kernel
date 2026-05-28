from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any


class TokenizersBpe:
    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = int(self.tokenizer.token_to_id("<pad>") or 0)
        self.bos_token_id = int(self.tokenizer.token_to_id("<s>") or 1)
        self.eos_token_id = int(self.tokenizer.token_to_id("</s>") or 2)
        self.unk_token_id = int(self.tokenizer.token_to_id("<unk>") or 3)

    def encode(self, text: str, *, max_length: int) -> list[int]:
        return list(self.tokenizer.encode(str(text), add_special_tokens=True).ids)[:max_length]

    def decode(self, ids: list[int]) -> str:
        clean: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            clean.append(token_id)
        return self.tokenizer.decode(clean, skip_special_tokens=False)


def generate_neural_controller_text(
    *,
    manifest_path: Path,
    encoder_text: str,
    repo_root: Path,
    device: str = "cpu",
    max_new_tokens: int = 512,
    max_encoder_tokens: int = 1024,
) -> dict[str, Any]:
    model, tokenizer, manifest = _load_bundle_cached(
        str(manifest_path.resolve()),
        str(repo_root.resolve()),
        str(device),
    )
    import torch

    enc_ids = tokenizer.encode(encoder_text, max_length=int(max_encoder_tokens))
    dec_ids = [int(tokenizer.bos_token_id)]
    forbidden = {int(tokenizer.pad_token_id), int(tokenizer.unk_token_id)}
    policy_heads: dict[str, float] = {}
    with torch.no_grad():
        enc = torch.tensor([enc_ids], dtype=torch.long, device=torch.device(device))
        enc_mask = torch.ones_like(enc, dtype=torch.long, device=torch.device(device))
        policy_logits = getattr(model, "agent_policy_logits", None)
        if callable(policy_logits):
            for name, value in policy_logits(enc, enc_mask).items():
                policy_name = "action_validity" if str(name) == "paper_action_validity" else str(name)
                policy_heads[policy_name] = round(float(torch.sigmoid(value).detach().cpu().item()), 4)
        for _ in range(int(max_new_tokens)):
            dec = torch.tensor([dec_ids], dtype=torch.long, device=torch.device(device))
            logits = model(enc, dec, enc_mask, None)[0, -1].float()
            for token_id in forbidden:
                if 0 <= token_id < logits.numel():
                    logits[token_id] = -float("inf")
            next_id = int(torch.argmax(logits).item())
            if next_id == int(tokenizer.eos_token_id):
                break
            dec_ids.append(next_id)
            if len(dec_ids) > 16 and len(dec_ids) % 8 == 0:
                partial_text = tokenizer.decode(dec_ids[1:])
                if _line_protocol_prediction_complete(partial_text):
                    break
    generated_ids = dec_ids[1:]
    scalar_diagnostics = {}
    diagnostics_fn = getattr(model, "scalar_control_diagnostics", None)
    if callable(diagnostics_fn):
        scalar_diagnostics = diagnostics_fn()
    return {
        "generated_text": tokenizer.decode(generated_ids),
        "generated_token_count": len(generated_ids),
        "policy_heads": policy_heads,
        "scalar_control": scalar_diagnostics,
        "manifest_path": str(manifest_path),
        "model_family": str(manifest.get("model_family", "")),
    }


@lru_cache(maxsize=2)
def _load_bundle_cached(manifest_path_raw: str, repo_root_raw: str, device_raw: str):
    import torch

    repo_root = Path(repo_root_raw)
    model_stack = repo_root / "other_repos" / "model-stack"
    for path in (repo_root, model_stack):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    manifest_path = Path(manifest_path_raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"neural controller manifest is not an object: {manifest_path}")
    model_dir = Path(str(manifest.get("model_dir", "")))
    tokenizer_dir = Path(str(manifest.get("tokenizer_dir", "")))
    config = load_config(str(model_dir))
    tokenizer_kind = str(manifest.get("tokenizer_kind", "agentkernel-bpe")).lower()
    if tokenizer_kind != "agentkernel-bpe":
        raise ValueError(f"unsupported neural controller tokenizer_kind: {tokenizer_kind}")
    tokenizer = TokenizersBpe(tokenizer_dir / "tokenizer.json")
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.to(torch.device(device_raw)).eval()
    return model, tokenizer, manifest


def _materialize_lazy_modules(model) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _line_protocol_prediction_complete(text: str) -> bool:
    seen: set[str] = set()
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip().lower()
        if key in {"thought", "action", "content", "done"}:
            seen.add(key)
    return {"action", "content", "done"}.issubset(seen)
