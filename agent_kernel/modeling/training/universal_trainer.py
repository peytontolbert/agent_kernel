from __future__ import annotations

from pathlib import Path
import json
from typing import Any

try:
    import torch
    nn = torch.nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..adapter_training import InjectedLoRAState
from ..tolbert import build_prompt_condition_batch
from ..tolbert.checkpoint import save_hybrid_runtime_bundle
from ..tolbert.config import HybridTolbertSSMConfig
from ..tolbert.delta import load_tolbert_checkpoint_state, write_tolbert_checkpoint_delta
from ..tolbert.hybrid_model import HybridTolbertSSMModel
from ..tolbert.tokenization import encode_decoder_sequence


def train_hybrid_decoder_from_universal_dataset(
    *,
    dataset_manifest_path: Path,
    output_dir: Path,
    config: HybridTolbertSSMConfig,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    device: str = "cpu",
    parent_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    _require_torch()
    manifest = _load_json(dataset_manifest_path)
    train_path = Path(str(manifest.get("train_dataset_path", "")).strip())
    decoder_vocab_path = Path(str(manifest.get("decoder_vocab_path", "")).strip())
    decoder_vocab = _load_json(decoder_vocab_path)
    examples = _load_jsonl(train_path)
    model = HybridTolbertSSMModel(config).to(device)
    if parent_checkpoint_path is not None and not parent_checkpoint_path.exists():
        parent_checkpoint_path = None
    adapter_state: InjectedLoRAState | None = None
    training_mode = "full_checkpoint"
    if parent_checkpoint_path is not None:
        parent_state_dict, _, _ = load_tolbert_checkpoint_state(parent_checkpoint_path)
        model.load_state_dict(parent_state_dict, strict=False)
        adapter_state = InjectedLoRAState(
            model,
            rank=8,
            alpha=8.0,
            module_filter=_should_wrap_lora_module,
            direct_parameter_filter=_should_train_direct_delta_parameter,
        )
        optimizer = torch.optim.Adam(list(adapter_state.parameters()), lr=lr)
        training_mode = "injected_lora_plus_structured_adapters"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(max(1, epochs)):
        for offset in range(0, len(examples), max(1, batch_size)):
            batch_examples = examples[offset : offset + max(1, batch_size)]
            if not batch_examples:
                continue
            prompts = [str(example.get("prompt", "")) for example in batch_examples]
            condition = build_prompt_condition_batch(prompts=prompts, config=config, device=device)
            decoder_inputs = []
            decoder_targets = []
            for example in batch_examples:
                input_ids, target_ids = encode_decoder_sequence(
                    str(example.get("target", "")),
                    config,
                    decoder_vocab=decoder_vocab,
                )
                decoder_inputs.append(input_ids)
                decoder_targets.append(target_ids)
            optimizer.zero_grad()
            output = model(
                command_token_ids=condition["command_token_ids"],
                scalar_features=condition["scalar_features"],
                family_ids=condition["family_ids"],
                path_level_ids=condition["path_level_ids"],
                decoder_input_ids=torch.tensor(decoder_inputs, dtype=torch.long, device=device),
                prefer_python_ref=not str(device).startswith("cuda"),
                prefer_python_world_ref=not str(device).startswith("cuda"),
            )
            target_tensor = torch.tensor(decoder_targets, dtype=torch.long, device=device)
            example_weight = torch.tensor(
                [float(example.get("example_weight", 1.0) or 1.0) for example in batch_examples],
                dtype=torch.float32,
                device=device,
            )
            loss = _weighted_decoder_cross_entropy(
                output.decoder_logits,
                target_tensor,
                example_weight,
                pad_token_id=config.decoder_pad_token_id,
            )
            loss.backward()
            if adapter_state is not None:
                torch.nn.utils.clip_grad_norm_(list(adapter_state.parameters()), max_norm=1.0)
            optimizer.step()
    delta_checkpoint_payload = None
    delta_checkpoint_path = None
    if adapter_state is not None and parent_checkpoint_path is not None:
        state_dict_adapters, state_dict_delta = adapter_state.mutation_components()
        delta_checkpoint_path = output_dir / "hybrid_checkpoint__delta.pt"
        delta_checkpoint_payload = {
            "artifact_kind": "tolbert_checkpoint_delta",
            "format_version": "tolbert_delta_v1",
            "checkpoint_format": "wrapped_state_dict",
            "parent_checkpoint_path": str(parent_checkpoint_path),
            "parent_checkpoint_sha256": "",
            "child_checkpoint_sha256": "",
            "config": config.to_dict(),
            "state_dict_adapters": state_dict_adapters,
            "state_dict_delta": state_dict_delta,
            "override_state_dict": {},
            "removed_state_keys": [],
            "stats": {
                **adapter_state.stats(),
                "changed_key_count": len(state_dict_adapters) + len(state_dict_delta),
                "unchanged_key_count": 0,
                "override_key_count": 0,
                "removed_key_count": 0,
                "total_key_count": len(state_dict_adapters) + len(state_dict_delta),
            },
        }
        delta_metadata = write_tolbert_checkpoint_delta(
            parent_checkpoint_path=parent_checkpoint_path,
            delta_output_path=delta_checkpoint_path,
            state_dict_adapters=state_dict_adapters,
            state_dict_delta=state_dict_delta,
            config=config.to_dict(),
            checkpoint_format="wrapped_state_dict",
        )
        delta_checkpoint_payload["parent_checkpoint_sha256"] = delta_metadata["parent_checkpoint_sha256"]
    manifest_path = save_hybrid_runtime_bundle(
        output_dir=output_dir,
        model=model,
        config=config,
        metadata={
            "training_objective": "universal_decoder_only",
            "dataset_manifest_path": str(dataset_manifest_path),
            "train_dataset_path": str(train_path),
            "decoder_vocab_path": str(decoder_vocab_path),
            "decoder_tokenizer_manifest_path": str(manifest.get("decoder_tokenizer_manifest_path", "")).strip(),
            "decoder_vocab_size": int(manifest.get("decoder_vocab_size", 0) or 0),
            "difficulty_counts": dict(manifest.get("difficulty_counts", {}))
            if isinstance(manifest.get("difficulty_counts", {}), dict)
            else {},
            "long_horizon_example_count": int(manifest.get("long_horizon_example_count", 0) or 0),
            "example_count": len(examples),
            "epochs": max(1, epochs),
            "batch_size": max(1, batch_size),
            "lr": float(lr),
            "training_mode": training_mode,
            "adapter_training_stats": dict(adapter_state.stats()) if adapter_state is not None else {},
        },
        decoder_vocab_path=decoder_vocab_path,
        parent_checkpoint_path=parent_checkpoint_path,
        delta_checkpoint_path=delta_checkpoint_path,
        delta_checkpoint_payload=delta_checkpoint_payload,
    )
    return _load_json(manifest_path)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("Full PyTorch with torch.nn is required for universal decoder training")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _should_wrap_lora_module(module_path: str, module: nn.Module) -> bool:
    del module
    return bool(module_path)


def _should_train_direct_delta_parameter(parameter_path: str, parameter: nn.Parameter) -> bool:
    del parameter
    return parameter_path in {
        "log_a",
        "d_skip",
        "delta_bias",
        "world_transition_logits",
        "world_transition_gate",
        "decoder_log_a",
        "decoder_d_skip",
        "decoder_delta_bias",
    }


def _weighted_decoder_cross_entropy(
    decoder_logits: torch.Tensor,
    decoder_target_ids: torch.Tensor,
    example_weight: torch.Tensor,
    *,
    pad_token_id: int,
) -> torch.Tensor:
    per_token = nn.functional.cross_entropy(
        decoder_logits.reshape(-1, int(decoder_logits.shape[-1])),
        decoder_target_ids.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).reshape(decoder_target_ids.shape[0], -1)
    mask = (decoder_target_ids != pad_token_id).float()
    per_example = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return (per_example * example_weight).sum() / example_weight.sum().clamp_min(1.0)
