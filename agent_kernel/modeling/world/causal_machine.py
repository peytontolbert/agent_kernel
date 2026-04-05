from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CausalWorldProfile:
    profile_path: Path
    sidecar_path: Path
    num_states: int
    horizons: tuple[int, ...]
    sketch_dim: int
    log_probs: np.ndarray
    state_masses: np.ndarray
    centroids_sketch: np.ndarray


@dataclass(frozen=True, slots=True)
class CausalWorldPrior:
    log_prior: np.ndarray
    backend: str
    matched_state_index: int
    matched_state_probability: float
    signature_norm: float
    token_count: int
    horizon_hint: str = ""
    applied_bias_strength: float = 1.0


def load_causal_world_profile(profile_json_path: str | Path) -> CausalWorldProfile:
    profile_path = Path(profile_json_path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"causal world profile not found: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    future_profile = payload.get("future_signature_profile")
    if not isinstance(future_profile, dict) or not bool(future_profile.get("available")):
        raise ValueError(f"{profile_path} does not contain an available future_signature_profile")
    horizons = tuple(int(value) for value in future_profile.get("horizons", []) if int(value) > 0)
    if not horizons:
        raise ValueError(f"{profile_path} does not contain valid future horizons")
    signature_dim = int(future_profile.get("signature_dim", 0))
    if signature_dim <= 0 or signature_dim % len(horizons) != 0:
        raise ValueError(f"{profile_path} has invalid signature_dim={signature_dim} for horizons={horizons}")
    per_horizon_dim = signature_dim // len(horizons)
    sketch_dim = per_horizon_dim - 1
    if sketch_dim <= 0:
        raise ValueError(f"{profile_path} implies non-positive sketch_dim={sketch_dim}")
    spectral_block = payload.get("spectral_eigenbases")
    if not isinstance(spectral_block, dict):
        raise ValueError(f"{profile_path} is missing spectral_eigenbases metadata")
    sidecar_path = _resolve_sidecar_path(profile_path, spectral_block)
    with np.load(sidecar_path) as arrays:
        if "causal_machine_signature_centroids" not in arrays or "causal_machine_log_probs" not in arrays:
            raise ValueError(f"{sidecar_path} is missing causal-machine arrays")
        centroids = arrays["causal_machine_signature_centroids"].astype(np.float32, copy=False)
        log_probs = arrays["causal_machine_log_probs"].astype(np.float32, copy=False)
        state_masses = (
            arrays["causal_machine_state_masses"].astype(np.float32, copy=False)
            if "causal_machine_state_masses" in arrays
            else np.zeros((log_probs.shape[0],), dtype=np.float32)
        )
    keep_cols: list[int] = []
    for index in range(len(horizons)):
        base = index * per_horizon_dim
        keep_cols.extend(range(base, base + sketch_dim))
    centroids_sketch = centroids[:, keep_cols].astype(np.float32, copy=False)
    return CausalWorldProfile(
        profile_path=profile_path,
        sidecar_path=sidecar_path,
        num_states=int(log_probs.shape[0]),
        horizons=horizons,
        sketch_dim=int(sketch_dim),
        log_probs=log_probs,
        state_masses=state_masses,
        centroids_sketch=centroids_sketch,
    )


def build_causal_state_signature(
    text_fragments: list[str] | tuple[str, ...],
    *,
    sketch_dim: int,
) -> tuple[np.ndarray, int]:
    width = max(int(sketch_dim), 1)
    signature = np.zeros((width,), dtype=np.float32)
    token_count = 0
    for fragment in text_fragments:
        for token in _tokenize_fragment(fragment):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "little", signed=False) % width
            sign = 1.0 if (digest[8] & 1) == 0 else -1.0
            signature[bucket] += sign
            token_count += 1
    norm = float(np.linalg.norm(signature))
    if norm > 0.0:
        signature /= norm
    return signature, token_count


def condition_causal_world_prior(
    profile: CausalWorldProfile,
    *,
    text_fragments: list[str] | tuple[str, ...],
    bias_strength: float = 1.0,
    horizon_hint: str | None = None,
) -> CausalWorldPrior:
    base_logits = _base_profile_logits(profile)
    resolved_horizon_hint = str(horizon_hint or "").strip()
    applied_bias_strength = float(bias_strength)
    if resolved_horizon_hint == "long_horizon":
        max_horizon = max(profile.horizons, default=1)
        if max_horizon > 1:
            applied_bias_strength *= 1.0 + min(0.75, 0.2 * math.log1p(float(max_horizon)))
    signature_per_horizon, token_count = build_causal_state_signature(
        text_fragments,
        sketch_dim=profile.sketch_dim,
    )
    if token_count <= 0:
        return _fallback_world_prior(
            base_logits,
            backend="mass_only",
            horizon_hint=resolved_horizon_hint,
            applied_bias_strength=applied_bias_strength,
        )
    signature = np.tile(signature_per_horizon, len(profile.horizons)).astype(np.float32, copy=False)
    if signature.shape != (profile.centroids_sketch.shape[1],):
        return _fallback_world_prior(
            base_logits,
            backend="mass_only",
            horizon_hint=resolved_horizon_hint,
            applied_bias_strength=applied_bias_strength,
        )
    centroid_rows = profile.centroids_sketch
    if resolved_horizon_hint == "long_horizon" and len(profile.horizons) > 1:
        max_horizon = max(profile.horizons, default=1)
        weighted_columns: list[np.ndarray] = []
        for weight_index, horizon in enumerate(profile.horizons):
            weight = 1.0 + (0.5 * float(horizon) / float(max_horizon))
            start = weight_index * profile.sketch_dim
            stop = start + profile.sketch_dim
            weighted_columns.append(profile.centroids_sketch[:, start:stop] * weight)
        centroid_rows = np.concatenate(weighted_columns, axis=1).astype(np.float32, copy=False)
    centroid_norms = np.linalg.norm(centroid_rows, axis=1)
    safe_norms = np.where(centroid_norms > 0.0, centroid_norms, 1.0).astype(np.float32, copy=False)
    similarities = (centroid_rows @ signature) / safe_norms
    logits = base_logits + (applied_bias_strength * similarities.astype(np.float32, copy=False))
    return _finalize_world_prior(
        logits,
        backend="profile_conditioned",
        signature_norm=float(np.linalg.norm(signature)),
        token_count=token_count,
        horizon_hint=resolved_horizon_hint,
        applied_bias_strength=applied_bias_strength,
    )


def summarize_causal_machine_adoption() -> dict[str, object]:
    return {
        "recommended_modules": [
            "agent_kernel/modeling/world/causal_machine.py",
            "agent_kernel/modeling/world/counterfactual.py",
            "agent_kernel/modeling/world/kernels/",
        ],
        "recommended_capabilities": [
            "profile_loaded_state_priors",
            "profile_conditioned_state_priors",
            "chunked_belief_scan_kernel",
            "counterfactual_group_eval",
            "incremental_cache_equivalence_checks",
        ],
        "non_goals": [
            "lexical_shortcut_trainer_port",
            "competition_recipe_defaults",
            "full_gpt_trainer_transplant",
        ],
    }


def _tokenize_fragment(fragment: object) -> list[str]:
    text = str(fragment or "").strip().lower()
    if not text:
        return []
    return [token for token in re.findall(r"[a-z0-9_./:-]+", text) if len(token) >= 2]


def _base_profile_logits(profile: CausalWorldProfile) -> np.ndarray:
    masses = np.asarray(profile.state_masses, dtype=np.float32)
    if masses.shape == (profile.num_states,) and float(masses.sum()) > 0.0:
        prior = masses / masses.sum()
        return np.log(prior.clip(min=1.0e-8)).astype(np.float32, copy=False)
    log_probs = np.asarray(profile.log_probs, dtype=np.float32)
    if log_probs.shape == (profile.num_states,):
        return log_probs.astype(np.float32, copy=False)
    return np.zeros((profile.num_states,), dtype=np.float32)


def _fallback_world_prior(
    logits: np.ndarray,
    *,
    backend: str,
    horizon_hint: str,
    applied_bias_strength: float,
) -> CausalWorldPrior:
    return _finalize_world_prior(
        logits,
        backend=backend,
        signature_norm=0.0,
        token_count=0,
        horizon_hint=horizon_hint,
        applied_bias_strength=applied_bias_strength,
    )


def _finalize_world_prior(
    logits: np.ndarray,
    *,
    backend: str,
    signature_norm: float,
    token_count: int,
    horizon_hint: str,
    applied_bias_strength: float,
) -> CausalWorldPrior:
    clipped = np.asarray(logits, dtype=np.float32)
    clipped = clipped - np.max(clipped)
    probs = np.exp(clipped)
    total = float(probs.sum())
    if total <= 0.0:
        probs = np.full_like(clipped, fill_value=1.0 / max(int(clipped.shape[0]), 1))
    else:
        probs = probs / total
    matched_state_index = int(np.argmax(probs)) if probs.size > 0 else 0
    matched_state_probability = float(probs[matched_state_index]) if probs.size > 0 else 0.0
    return CausalWorldPrior(
        log_prior=np.log(probs.clip(min=1.0e-8)).astype(np.float32, copy=False),
        backend=str(backend),
        matched_state_index=matched_state_index,
        matched_state_probability=matched_state_probability,
        signature_norm=float(signature_norm),
        token_count=max(int(token_count), 0),
        horizon_hint=str(horizon_hint).strip(),
        applied_bias_strength=float(applied_bias_strength),
    )


def _resolve_sidecar_path(profile_path: Path, spectral_block: dict[str, Any]) -> Path:
    sidecar_path = spectral_block.get("sidecar_npz")
    if not isinstance(sidecar_path, str) or not sidecar_path.strip():
        candidate = profile_path.with_name(f"{profile_path.stem}_spectral_eigenbases.npz")
        sidecar = candidate
    else:
        sidecar = Path(sidecar_path)
    if not sidecar.is_absolute():
        sidecar = (profile_path.parent / sidecar).resolve()
    if not sidecar.is_file():
        raise FileNotFoundError(f"causal world sidecar not found: {sidecar}")
    return sidecar
