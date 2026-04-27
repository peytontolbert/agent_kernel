from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def build_context_provider(*, config, repo_root: Path):
    base_provider = None
    if bool(config.use_tolbert_context):
        from .tolbert import MockTolbertContextCompiler, TolbertContextCompiler

        provider = str(config.normalized_provider()).strip()
        if provider == "mock":
            base_provider = MockTolbertContextCompiler(config=config, repo_root=repo_root)
        else:
            base_provider = TolbertContextCompiler(config=config, repo_root=repo_root)
    if not bool(getattr(config, "use_research_library_context", False)):
        return base_provider
    if base_provider is None and not bool(getattr(config, "research_library_standalone_context", False)):
        return None
    from ..research_library.context import ResearchLibraryContextProvider

    return ResearchLibraryContextProvider(
        config=config,
        repo_root=repo_root,
        base_provider=base_provider,
    )


def load_model_artifact(path: Path) -> dict[str, object]:
    from ..modeling.artifacts import load_model_artifact as _load_model_artifact

    return _load_model_artifact(path)


def retained_tolbert_hybrid_runtime(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_hybrid_runtime as _retained_tolbert_hybrid_runtime

    return _retained_tolbert_hybrid_runtime(payload)


def retained_tolbert_universal_decoder_runtime(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import (
        retained_tolbert_universal_decoder_runtime as _retained_tolbert_universal_decoder_runtime,
    )

    return _retained_tolbert_universal_decoder_runtime(payload)


def retained_tolbert_active_decoder_runtime(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_active_decoder_runtime as _retained_tolbert_active_decoder_runtime

    return _retained_tolbert_active_decoder_runtime(payload)


def retained_tolbert_action_generation_policy(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import (
        retained_tolbert_action_generation_policy as _retained_tolbert_action_generation_policy,
    )

    return _retained_tolbert_action_generation_policy(payload)


def retained_tolbert_decoder_policy(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_decoder_policy as _retained_tolbert_decoder_policy

    return _retained_tolbert_decoder_policy(payload)


def retained_tolbert_model_surfaces(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_model_surfaces as _retained_tolbert_model_surfaces

    return _retained_tolbert_model_surfaces(payload)


def retained_tolbert_rollout_policy(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_rollout_policy as _retained_tolbert_rollout_policy

    return _retained_tolbert_rollout_policy(payload)


def retained_tolbert_runtime_policy(payload: dict[str, object]) -> dict[str, object]:
    from ..modeling.artifacts import retained_tolbert_runtime_policy as _retained_tolbert_runtime_policy

    return _retained_tolbert_runtime_policy(payload)


def infer_hybrid_world_signal(*, state, bundle_manifest_path: Path, device: str = "cpu", scoring_policy: Any = None):
    from ..modeling.tolbert.runtime import infer_hybrid_world_signal as _infer_hybrid_world_signal

    return _infer_hybrid_world_signal(
        state=state,
        bundle_manifest_path=bundle_manifest_path,
        device=device,
        scoring_policy=scoring_policy,
    )


def score_hybrid_candidates(*, state, candidates: list[dict[str, object]], bundle_manifest_path: Path, device: str = "cpu", scoring_policy: Any = None) -> list[dict[str, object]]:
    from ..modeling.tolbert.runtime import score_hybrid_candidates as _score_hybrid_candidates

    return _score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=bundle_manifest_path,
        device=device,
        scoring_policy=scoring_policy,
    )


def generate_hybrid_decoder_text(*, state, bundle_manifest_path: Path, device: str = "cpu", max_new_tokens: int | None = None) -> dict[str, object]:
    from ..modeling.tolbert.runtime import generate_hybrid_decoder_text as _generate_hybrid_decoder_text

    return _generate_hybrid_decoder_text(
        state=state,
        bundle_manifest_path=bundle_manifest_path,
        device=device,
        max_new_tokens=max_new_tokens,
    )


def build_latent_state_summary(**kwargs) -> dict[str, object]:
    from ..modeling.world.latent_state import build_latent_state_summary as _build_latent_state_summary

    if "state" in kwargs:
        state = kwargs.pop("state")
        kwargs.setdefault("world_model_summary", getattr(state, "world_model_summary", {}))
        kwargs.setdefault("latest_transition", getattr(state, "latest_state_transition", {}))
        task = getattr(state, "task", None)
        kwargs.setdefault("task_metadata", getattr(task, "metadata", {}) if task is not None else {})
        kwargs.setdefault(
            "recent_history",
            [dict(item) for item in kwargs.pop("recent_history", [])]
            if "recent_history" in kwargs
            else [
                asdict(step) if is_dataclass(step) else dict(step)
                for step in getattr(state, "history", [])[-3:]
            ],
        )
        context_packet = getattr(state, "context_packet", None)
        kwargs.setdefault("context_control", getattr(context_packet, "control", {}) if context_packet is not None else {})
    return _build_latent_state_summary(**kwargs)


def latent_command_bias(latent_state_summary: dict[str, object], command: str) -> int:
    from ..modeling.world.latent_state import latent_command_bias as _latent_command_bias

    return _latent_command_bias(latent_state_summary, command)


def choose_tolbert_route(*, runtime_policy: dict[str, object], benchmark_family: str, path_confidence: float, trust_retrieval: bool):
    from ..modeling.policy.runtime import choose_tolbert_route as _choose_tolbert_route

    return _choose_tolbert_route(
        runtime_policy=runtime_policy,
        benchmark_family=benchmark_family,
        path_confidence=path_confidence,
        trust_retrieval=trust_retrieval,
    )


def decode_action_generation_candidates(**kwargs) -> list[dict[str, object]]:
    from ..modeling.policy.decoder import decode_action_generation_candidates as _decode_action_generation_candidates

    return _decode_action_generation_candidates(**kwargs)


def decode_bounded_action_candidates(**kwargs) -> list[dict[str, object]]:
    from ..modeling.policy.decoder import decode_bounded_action_candidates as _decode_bounded_action_candidates

    return _decode_bounded_action_candidates(**kwargs)


def render_structured_edit_command(step: dict[str, object]) -> str:
    from ..modeling.policy.decoder import _render_structured_edit_command

    return _render_structured_edit_command(step)


__all__ = [
    "build_context_provider",
    "build_latent_state_summary",
    "choose_tolbert_route",
    "decode_action_generation_candidates",
    "decode_bounded_action_candidates",
    "generate_hybrid_decoder_text",
    "infer_hybrid_world_signal",
    "latent_command_bias",
    "load_model_artifact",
    "render_structured_edit_command",
    "retained_tolbert_action_generation_policy",
    "retained_tolbert_active_decoder_runtime",
    "retained_tolbert_decoder_policy",
    "retained_tolbert_hybrid_runtime",
    "retained_tolbert_model_surfaces",
    "retained_tolbert_rollout_policy",
    "retained_tolbert_runtime_policy",
    "retained_tolbert_universal_decoder_runtime",
    "score_hybrid_candidates",
]
