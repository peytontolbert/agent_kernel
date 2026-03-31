from __future__ import annotations

from pathlib import Path
import json
from typing import Any

try:
    import torch
    import torch.nn.functional as F  # noqa: F401
    nn = torch.nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..adapter_training import InjectedLoRAState
from ..tolbert.checkpoint import save_hybrid_runtime_bundle
from ..tolbert.config import HybridTolbertSSMConfig
from ..tolbert.delta import load_tolbert_checkpoint_state, write_tolbert_checkpoint_delta
from ..tolbert.hybrid_model import HybridTolbertSSMModel
from .hybrid_dataset import materialize_hybrid_training_dataset


def train_hybrid_runtime_from_dataset(
    *,
    dataset_path: Path,
    output_dir: Path,
    config: HybridTolbertSSMConfig,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    device: str = "cpu",
    runtime_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    _require_torch()
    dataset_manifest = _load_dataset_manifest(dataset_path)
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = HybridTolbertSSMModel(config).to(device)
    parent_checkpoint_path = _optional_runtime_path(runtime_paths, "parent_checkpoint_path")
    adapter_state: InjectedLoRAState | None = None
    training_mode = "full_checkpoint"
    if parent_checkpoint_path is not None and parent_checkpoint_path.exists():
        parent_state_dict, _, _ = load_tolbert_checkpoint_state(parent_checkpoint_path)
        _load_compatible_state_dict(model, parent_state_dict)
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
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction="batchmean")
    for _ in range(max(1, epochs)):
        for offset in range(0, len(examples), max(1, batch_size)):
            batch_examples = examples[offset : offset + max(1, batch_size)]
            if not batch_examples:
                continue
            batch = _tensorize_batch(batch_examples, device=device)
            optimizer.zero_grad()
            output = model(
                command_token_ids=batch["command_token_ids"],
                scalar_features=batch["scalar_features"],
                family_ids=batch["family_ids"],
                path_level_ids=batch["path_level_ids"],
                decoder_input_ids=batch["decoder_input_ids"],
                prefer_python_ref=_prefer_python_ref_for_device(device),
                prefer_python_world_ref=_prefer_python_ref_for_device(device),
            )
            loss = (
                mse(output.score, batch["score_target"])
                + bce(output.policy_logits, batch["policy_target"])
                + mse(output.value, batch["value_target"])
                + bce(output.stop_logits, batch["stop_target"])
                + bce(output.risk_logits, batch["risk_target"])
                + mse(output.transition, batch["transition_target"])
                + _decoder_cross_entropy(output.decoder_logits, batch["decoder_target_ids"])
            )
            if output.world_final_belief is not None:
                loss = loss + kl_div(output.world_final_belief, batch["world_target"])
            loss.backward()
            if adapter_state is not None:
                torch.nn.utils.clip_grad_norm_(list(adapter_state.parameters()), max_norm=1.0)
            optimizer.step()
    delta_checkpoint_payload = None
    delta_checkpoint_path = None
    if adapter_state is not None and parent_checkpoint_path is not None:
        state_dict_adapters, state_dict_delta = adapter_state.mutation_components()
        delta_checkpoint_path = (_optional_runtime_path(runtime_paths, "checkpoint_delta_path") or output_dir / "hybrid_checkpoint__delta.pt")
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
    else:
        delta_metadata = {}
    manifest_path = save_hybrid_runtime_bundle(
        output_dir=output_dir,
        model=model,
        config=config,
        metadata={
            "dataset_path": str(dataset_path),
            "example_count": len(examples),
            "epochs": max(1, epochs),
            "batch_size": max(1, batch_size),
            "lr": float(lr),
            "decoder_vocab_path": str(dataset_manifest.get("decoder_vocab_path", "")).strip(),
            "decoder_vocab_entry_count": int(dataset_manifest.get("decoder_vocab_entry_count", 0) or 0),
            "training_mode": training_mode,
            "adapter_training_stats": dict(adapter_state.stats()) if adapter_state is not None else {},
        },
        config_path=_optional_runtime_path(runtime_paths, "config_path"),
        checkpoint_path=_optional_runtime_path(runtime_paths, "checkpoint_path"),
        manifest_path=_optional_runtime_path(runtime_paths, "bundle_manifest_path"),
        decoder_vocab_path=_optional_dataset_path(dataset_manifest, "decoder_vocab_path"),
        parent_checkpoint_path=parent_checkpoint_path,
        delta_checkpoint_path=delta_checkpoint_path,
        delta_checkpoint_payload=delta_checkpoint_payload,
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def train_hybrid_runtime_from_trajectories(
    *,
    trajectories_root: Path,
    output_dir: Path,
    config: HybridTolbertSSMConfig,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    device: str = "cpu",
    runtime_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    _require_torch()
    dataset_path = output_dir / "hybrid_training_dataset.jsonl"
    materialize_hybrid_training_dataset(
        trajectories_root=trajectories_root,
        output_path=dataset_path,
        config=config,
    )
    return train_hybrid_runtime_from_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        runtime_paths=runtime_paths,
    )


def _tensorize_batch(examples: list[dict[str, Any]], *, device: str) -> dict[str, torch.Tensor]:
    return {
        "family_ids": torch.tensor([int(example["family_id"]) for example in examples], dtype=torch.long, device=device),
        "path_level_ids": torch.tensor([example["path_level_ids"] for example in examples], dtype=torch.long, device=device),
        "command_token_ids": torch.tensor([example["command_token_ids"] for example in examples], dtype=torch.long, device=device),
        "decoder_input_ids": torch.tensor([example["decoder_input_ids"] for example in examples], dtype=torch.long, device=device),
        "decoder_target_ids": torch.tensor([example["decoder_target_ids"] for example in examples], dtype=torch.long, device=device),
        "scalar_features": torch.tensor([example["scalar_features"] for example in examples], dtype=torch.float32, device=device),
        "score_target": torch.tensor([float(example["score_target"]) for example in examples], dtype=torch.float32, device=device),
        "policy_target": torch.tensor([float(example["policy_target"]) for example in examples], dtype=torch.float32, device=device),
        "value_target": torch.tensor([float(example["value_target"]) for example in examples], dtype=torch.float32, device=device),
        "stop_target": torch.tensor([float(example["stop_target"]) for example in examples], dtype=torch.float32, device=device),
        "risk_target": torch.tensor([float(example["risk_target"]) for example in examples], dtype=torch.float32, device=device),
        "transition_target": torch.tensor([example["transition_target"] for example in examples], dtype=torch.float32, device=device),
        "world_target": torch.tensor([example["world_target"] for example in examples], dtype=torch.float32, device=device),
    }


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("Full PyTorch with torch.nn is required for hybrid runtime training")


def _prefer_python_ref_for_device(device: str) -> bool:
    return not str(device).strip().lower().startswith("cuda")


def _optional_runtime_path(runtime_paths: dict[str, str] | None, key: str) -> Path | None:
    if not isinstance(runtime_paths, dict):
        return None
    raw = str(runtime_paths.get(key, "")).strip()
    if not raw:
        return None
    return Path(raw)


def _optional_dataset_path(dataset_manifest: dict[str, Any], key: str) -> Path | None:
    raw = str(dataset_manifest.get(key, "")).strip()
    if not raw:
        return None
    return Path(raw)


def _load_dataset_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_compatible_state_dict(model: HybridTolbertSSMModel, state_dict: dict[str, object]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"score_head.weight", "score_head.bias"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected parameters in hybrid runtime parent checkpoint: {sorted(unexpected)}")
    disallowed_missing = missing - allowed_missing
    if disallowed_missing:
        raise RuntimeError(f"missing parameters in hybrid runtime parent checkpoint: {sorted(disallowed_missing)}")


def _decoder_cross_entropy(decoder_logits: torch.Tensor, decoder_target_ids: torch.Tensor) -> torch.Tensor:
    vocab_size = int(decoder_logits.shape[-1])
    return nn.functional.cross_entropy(
        decoder_logits.reshape(-1, vocab_size),
        decoder_target_ids.reshape(-1),
        ignore_index=0,
    )


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
