from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import shutil
from typing import Any

import torch

from .config import HybridTolbertSSMConfig
from .delta import resolve_tolbert_runtime_checkpoint_path
from .hybrid_model import HybridTolbertSSMModel


def save_hybrid_runtime_bundle(
    *,
    output_dir: Path,
    model: HybridTolbertSSMModel,
    config: HybridTolbertSSMConfig,
    metadata: dict[str, Any] | None = None,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    manifest_path: Path | None = None,
    decoder_vocab_path: Path | None = None,
    parent_checkpoint_path: Path | None = None,
    delta_checkpoint_path: Path | None = None,
    delta_checkpoint_payload: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path or (output_dir / "hybrid_config.json")
    checkpoint_path = checkpoint_path or (output_dir / "hybrid_checkpoint.pt")
    manifest_path = manifest_path or (output_dir / "hybrid_bundle_manifest.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    checkpoint_value = str(checkpoint_path)
    if delta_checkpoint_payload is not None and parent_checkpoint_path is not None:
        resolved_delta_path = delta_checkpoint_path or checkpoint_path.with_name(f"{checkpoint_path.stem}__delta.pt")
        resolved_delta_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta_checkpoint_payload, resolved_delta_path)
        checkpoint_value = ""
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config.to_dict()}, checkpoint_path)
    metadata_payload = dict(metadata or {})
    if decoder_vocab_path is not None and decoder_vocab_path.exists():
        bundled_decoder_vocab_path = output_dir / "hybrid_decoder_vocab.json"
        if decoder_vocab_path.resolve() != bundled_decoder_vocab_path.resolve():
            bundled_decoder_vocab_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(decoder_vocab_path, bundled_decoder_vocab_path)
        else:
            bundled_decoder_vocab_path = decoder_vocab_path
        metadata_payload["decoder_vocab_path"] = str(bundled_decoder_vocab_path)
        metadata_payload["relative_decoder_vocab_path"] = bundled_decoder_vocab_path.name
    manifest = {
        "artifact_kind": "tolbert_hybrid_runtime_bundle",
        "model_family": config.model_family,
        "config_path": str(config_path),
        "checkpoint_path": checkpoint_value,
        "relative_config_path": config_path.name,
        "relative_checkpoint_path": "" if not checkpoint_value else checkpoint_path.name,
        "metadata": metadata_payload,
    }
    if delta_checkpoint_payload is not None and parent_checkpoint_path is not None:
        resolved_delta_path = delta_checkpoint_path or checkpoint_path.with_name(f"{checkpoint_path.stem}__delta.pt")
        manifest["parent_checkpoint_path"] = str(parent_checkpoint_path)
        manifest["checkpoint_delta_path"] = str(resolved_delta_path)
        manifest["checkpoint_mutation"] = {
            "mode": "parent_plus_structured_adapter_training",
            "parent_checkpoint_path": str(parent_checkpoint_path),
            "checkpoint_delta_path": str(resolved_delta_path),
            "stats": dict(delta_checkpoint_payload.get("stats", {}))
            if isinstance(delta_checkpoint_payload.get("stats", {}), dict)
            else {},
        }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_hybrid_runtime_bundle(
    manifest_path: Path,
    *,
    device: str = "cpu",
) -> tuple[HybridTolbertSSMModel, HybridTolbertSSMConfig, dict[str, Any]]:
    return _load_hybrid_runtime_bundle_cached(str(manifest_path.resolve()), str(device))


@lru_cache(maxsize=8)
def _load_hybrid_runtime_bundle_cached(
    manifest_path_str: str,
    device: str,
) -> tuple[HybridTolbertSSMModel, HybridTolbertSSMConfig, dict[str, Any]]:
    manifest_path = Path(manifest_path_str)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = _resolve_manifest_path(
        manifest_path=manifest_path,
        relative_key="relative_config_path",
        absolute_key="config_path",
    )
    checkpoint_path = Path(
        resolve_tolbert_runtime_checkpoint_path(manifest, artifact_path=manifest_path)
        or _resolve_manifest_path(
            manifest_path=manifest_path,
            relative_key="relative_checkpoint_path",
            absolute_key="checkpoint_path",
        )
    )
    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict):
        profile_key = str(metadata.get("causal_world_profile_path", "")).strip()
        if profile_key:
            metadata["causal_world_profile_path"] = str(_resolve_optional_relative_path(manifest_path, profile_key))
        decoder_vocab_key = str(metadata.get("decoder_vocab_path", "")).strip()
        relative_decoder_vocab_key = str(metadata.get("relative_decoder_vocab_path", "")).strip()
        if decoder_vocab_key or relative_decoder_vocab_key:
            metadata["decoder_vocab_path"] = str(
                _resolve_optional_relative_path(
                    manifest_path,
                    relative_decoder_vocab_key or decoder_vocab_key,
                )
            )
    config = HybridTolbertSSMConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    payload = torch.load(checkpoint_path, map_location=device)
    model = HybridTolbertSSMModel(config)
    _load_compatible_state_dict(model, payload["state_dict"])
    model.to(device)
    model.eval()
    manifest["config_path"] = str(config_path)
    manifest["checkpoint_path"] = str(checkpoint_path)
    return model, config, manifest


def _resolve_manifest_path(*, manifest_path: Path, relative_key: str, absolute_key: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_value = str(manifest.get(relative_key, "")).strip()
    if relative_value:
        candidate = (manifest_path.parent / relative_value).resolve()
        if candidate.exists():
            return candidate
    absolute_value = str(manifest.get(absolute_key, "")).strip()
    return _resolve_optional_relative_path(manifest_path, absolute_value)


def _resolve_optional_relative_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _load_compatible_state_dict(model: HybridTolbertSSMModel, state_dict: dict[str, object]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"score_head.weight", "score_head.bias"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected parameters in hybrid runtime bundle: {sorted(unexpected)}")
    disallowed_missing = missing - allowed_missing
    if disallowed_missing:
        raise RuntimeError(f"missing parameters in hybrid runtime bundle: {sorted(disallowed_missing)}")
