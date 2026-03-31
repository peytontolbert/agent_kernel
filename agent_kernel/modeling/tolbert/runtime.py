from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from ..world.causal_machine import condition_causal_world_prior, load_causal_world_profile
from ..world.latent_state import latent_command_bias
from agent_kernel.state import AgentState

from .checkpoint import load_hybrid_runtime_bundle
from .config import HybridTolbertSSMConfig
from .tokenization import (
    decode_decoder_ids,
    encode_command_tokens,
    encode_decoder_sequence,
    hashed_id,
)


@dataclass(slots=True)
class HybridDecoderCache:
    pooled_state: torch.Tensor
    decoder_last_state: torch.Tensor | None = None
    cache_mode: str = "full_recompute"
    steps_reused: int = 0
    cache_equivalent: bool | None = None
    max_abs_diff: float = 0.0
    fallback_used: bool = False
    step_diagnostics: list[dict[str, object]] = field(default_factory=list)
    world_initial_belief: torch.Tensor | None = None
    world_last_belief: torch.Tensor | None = None
    world_cache_mode: str = "disabled"
    world_steps_reused: int = 0
    world_cache_equivalent: bool | None = None
    world_max_abs_diff: float = 0.0
    world_fallback_used: bool = False
    world_step_diagnostics: list[dict[str, object]] = field(default_factory=list)


def score_hybrid_candidates(
    *,
    state: AgentState,
    candidates: list[dict[str, object]],
    bundle_manifest_path: Path,
    device: str = "cpu",
    scoring_policy: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    model, config, manifest = load_hybrid_runtime_bundle(bundle_manifest_path, device=device)
    decoder_vocab = _load_decoder_vocabulary(manifest)
    batch = build_candidate_batch(
        state=state,
        candidates=candidates,
        config=config,
        device=device,
        decoder_vocab=decoder_vocab,
    )
    world_prior = _world_initial_belief_payload(
        batch_size=len(candidates),
        state=state,
        config=config,
        manifest=manifest,
        device=device,
    )
    with torch.no_grad():
        output = model(
            command_token_ids=batch["command_token_ids"],
            scalar_features=batch["scalar_features"],
            family_ids=batch["family_ids"],
            path_level_ids=batch["path_level_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            prefer_python_ref=_prefer_python_ref_for_device(device),
            prefer_python_world_ref=_prefer_python_ref_for_device(device),
            world_initial_log_belief=world_prior["log_belief"],
        )
    decoder_diagnostics = getattr(output, "decoder_diagnostics", {})
    if not isinstance(decoder_diagnostics, dict):
        decoder_diagnostics = {}
    ssm_diagnostics = getattr(output, "ssm_diagnostics", {})
    if not isinstance(ssm_diagnostics, dict):
        ssm_diagnostics = {}
    world_diagnostics = getattr(output, "world_diagnostics", {})
    if not isinstance(world_diagnostics, dict):
        world_diagnostics = {}
    weights = _normalized_scoring_policy(scoring_policy)
    scored: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        learned_score = float(torch.sigmoid(output.score[index]).item())
        policy_score = float(torch.sigmoid(output.policy_logits[index]).item())
        value_score = float(torch.sigmoid(output.value[index]).item())
        risk_score = float(torch.sigmoid(output.risk_logits[index]).item())
        stop_score = float(torch.sigmoid(output.stop_logits[index]).item())
        transition_progress = float(output.transition[index, 0].item())
        transition_regression = float(output.transition[index, 1].item())
        transition_score = (
            (weights["transition_progress_weight"] * transition_progress)
            - (weights["transition_regression_penalty_weight"] * max(0.0, transition_regression))
        )
        decoder_logprob = _decoder_sequence_logprob(
            output.decoder_logits[index],
            batch["decoder_target_ids"][index],
        )
        decoder_world_progress_score = 0.0
        decoder_world_risk_score = 0.0
        decoder_world_entropy_mean = 0.0
        decoder_world_final = decoder_diagnostics.get("world_final_belief")
        if isinstance(decoder_world_final, torch.Tensor) and decoder_world_final.dim() == 2:
            decoder_world_probs = decoder_world_final[index].exp()
            decoder_world_progress_score = (
                float(decoder_world_probs[0].item()) if decoder_world_probs.numel() > 0 else 0.0
            )
            decoder_world_risk_score = (
                float(decoder_world_probs[1:].sum().item()) if decoder_world_probs.numel() > 1 else 0.0
            )
            safe_probs = decoder_world_probs.clamp_min(1.0e-8)
            decoder_world_entropy_mean = float((-(safe_probs * safe_probs.log()).sum()).item())
        world_progress_score = 0.0
        world_risk_score = 0.0
        if output.world_final_belief is not None:
            world_probs = output.world_final_belief[index].exp()
            world_progress_score = float(world_probs[0].item()) if world_probs.numel() > 0 else 0.0
            world_risk_score = float(world_probs[1].item()) if world_probs.numel() > 1 else 0.0
            if world_probs.numel() > 2:
                world_risk_score += float(world_probs[2].item())
            if world_probs.numel() > 3:
                world_risk_score += float(world_probs[3].item())
        latent_bias = float(latent_command_bias(state.latent_state_summary, str(candidate.get("command", candidate.get("content", "")))))
        total_score = (
            (weights["learned_score_weight"] * learned_score)
            + (weights["policy_weight"] * policy_score)
            + (weights["value_weight"] * value_score)
            - (weights["risk_penalty_weight"] * risk_score)
            + transition_score
            + (weights["world_progress_weight"] * world_progress_score)
            - (weights["world_risk_penalty_weight"] * world_risk_score)
            + (weights["decoder_world_progress_weight"] * decoder_world_progress_score)
            - (weights["decoder_world_risk_penalty_weight"] * decoder_world_risk_score)
            + (weights["latent_bias_weight"] * latent_bias)
            + (weights["decoder_logprob_weight"] * decoder_logprob)
        )
        if str(candidate.get("action", "code_execute")) == "respond":
            total_score = (
                (weights["respond_learned_score_weight"] * learned_score)
                + (weights["respond_policy_weight"] * policy_score)
                + (weights["respond_value_weight"] * value_score)
                - (weights["respond_risk_penalty_weight"] * risk_score)
                + (weights["respond_stop_weight"] * stop_score)
                + (weights["respond_transition_progress_weight"] * transition_progress)
                - (weights["respond_transition_regression_penalty_weight"] * max(0.0, transition_regression))
                + (weights["respond_world_progress_weight"] * world_progress_score)
                - (weights["respond_world_risk_penalty_weight"] * world_risk_score)
                + (weights["respond_decoder_world_progress_weight"] * decoder_world_progress_score)
                - (weights["respond_decoder_world_risk_penalty_weight"] * decoder_world_risk_score)
                + (weights["respond_latent_bias_weight"] * latent_bias)
                + (weights["respond_decoder_logprob_weight"] * decoder_logprob)
            )
        enriched = dict(candidate)
        enriched.update(
            {
                "hybrid_policy_score": policy_score,
                "hybrid_learned_score": learned_score,
                "hybrid_value_score": value_score,
                "hybrid_risk_score": risk_score,
                "hybrid_stop_score": stop_score,
                "hybrid_transition_progress": transition_progress,
                "hybrid_transition_regression": transition_regression,
                "hybrid_transition_score": transition_score,
                "hybrid_decoder_logprob": decoder_logprob,
                "hybrid_world_progress_score": world_progress_score,
                "hybrid_world_risk_score": world_risk_score,
                "hybrid_decoder_world_progress_score": decoder_world_progress_score,
                "hybrid_decoder_world_risk_score": decoder_world_risk_score,
                "hybrid_decoder_world_entropy_mean": decoder_world_entropy_mean,
                "hybrid_total_score": total_score,
                "hybrid_model_family": str(manifest.get("model_family", "")),
                "hybrid_ssm_backend": str(output.ssm_backend),
                "hybrid_ssm_last_state_norm_mean": float(
                    ssm_diagnostics.get("last_state_norm_mean", 0.0) or 0.0
                ),
                "hybrid_ssm_pooled_state_norm_mean": float(
                    ssm_diagnostics.get("pooled_norm_mean", 0.0) or 0.0
                ),
                "hybrid_world_backend": str(output.world_backend),
                "hybrid_world_transition_family": str(world_diagnostics.get("transition_family", "")),
                "hybrid_world_transition_bandwidth": int(
                    world_diagnostics.get("transition_bandwidth", 0) or 0
                ),
                "hybrid_world_transition_gate": float(world_diagnostics.get("transition_gate", 0.0) or 0.0),
                "hybrid_world_final_entropy_mean": float(
                    world_diagnostics.get("final_entropy_mean", 0.0) or 0.0
                ),
                "hybrid_world_final_top_states": list(world_diagnostics.get("final_top_states", [])),
                "hybrid_world_prior_backend": str(world_prior["diagnostics"].get("backend", "")),
                "hybrid_world_prior_top_state": int(world_prior["diagnostics"].get("matched_state_index", -1)),
                "hybrid_world_prior_top_probability": float(
                    world_prior["diagnostics"].get("matched_state_probability", 0.0) or 0.0
                ),
                "hybrid_world_profile_horizons": list(world_prior["diagnostics"].get("horizons", [])),
                "hybrid_world_signature_token_count": int(world_prior["diagnostics"].get("token_count", 0) or 0),
                "hybrid_latent_bias": latent_bias,
                "hybrid_scoring_policy": dict(weights),
            }
        )
        scored.append(enriched)
    return sorted(scored, key=lambda item: (-float(item["hybrid_total_score"]), str(item.get("command", ""))))


def infer_hybrid_world_signal(
    *,
    state: AgentState,
    bundle_manifest_path: Path,
    device: str = "cpu",
    scoring_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    probes = [
        {"command": "__probe_recover_state__", "action": "code_execute"},
        {"command": "__probe_continue_progress__", "action": "code_execute"},
        {"content": "__probe_stop__", "action": "respond"},
    ]
    scored = score_hybrid_candidates(
        state=state,
        candidates=probes,
        bundle_manifest_path=bundle_manifest_path,
        device=device,
        scoring_policy=scoring_policy,
    )
    if not scored:
        return {}
    world_progress = max(float(item.get("hybrid_world_progress_score", 0.0) or 0.0) for item in scored)
    world_risk = max(float(item.get("hybrid_world_risk_score", 0.0) or 0.0) for item in scored)
    decoder_world_progress = max(
        float(item.get("hybrid_decoder_world_progress_score", 0.0) or 0.0)
        for item in scored
    )
    decoder_world_risk = max(
        float(item.get("hybrid_decoder_world_risk_score", 0.0) or 0.0)
        for item in scored
    )
    transition_progress = max(float(item.get("hybrid_transition_progress", 0.0) or 0.0) for item in scored)
    transition_regression = max(
        max(0.0, float(item.get("hybrid_transition_regression", 0.0) or 0.0))
        for item in scored
    )
    progress_signal = max(world_progress, decoder_world_progress, transition_progress)
    risk_signal = max(world_risk, decoder_world_risk, transition_regression)
    top = scored[0]
    return {
        "source": "tolbert_hybrid_runtime",
        "model_family": str(top.get("hybrid_model_family", "")).strip(),
        "world_progress_score": round(world_progress, 4),
        "world_risk_score": round(world_risk, 4),
        "decoder_world_progress_score": round(decoder_world_progress, 4),
        "decoder_world_risk_score": round(decoder_world_risk, 4),
        "decoder_world_entropy_mean": round(float(top.get("hybrid_decoder_world_entropy_mean", 0.0) or 0.0), 4),
        "transition_progress_score": round(transition_progress, 4),
        "transition_regression_score": round(transition_regression, 4),
        "progress_signal": round(progress_signal, 4),
        "risk_signal": round(risk_signal, 4),
        "top_probe_reason": str(top.get("reason", "")).strip(),
        "probe_count": len(scored),
        "world_prior_backend": str(top.get("hybrid_world_prior_backend", "")).strip(),
        "world_prior_top_state": int(top.get("hybrid_world_prior_top_state", -1) or -1),
        "world_prior_top_probability": round(
            float(top.get("hybrid_world_prior_top_probability", 0.0) or 0.0),
            4,
        ),
        "world_profile_horizons": list(top.get("hybrid_world_profile_horizons", [])),
        "world_signature_token_count": int(top.get("hybrid_world_signature_token_count", 0) or 0),
        "world_transition_family": str(top.get("hybrid_world_transition_family", "")).strip(),
        "world_transition_bandwidth": int(top.get("hybrid_world_transition_bandwidth", 0) or 0),
        "world_transition_gate": round(float(top.get("hybrid_world_transition_gate", 0.0) or 0.0), 4),
        "world_final_entropy_mean": round(float(top.get("hybrid_world_final_entropy_mean", 0.0) or 0.0), 4),
        "ssm_last_state_norm_mean": round(float(top.get("hybrid_ssm_last_state_norm_mean", 0.0) or 0.0), 4),
        "ssm_pooled_state_norm_mean": round(
            float(top.get("hybrid_ssm_pooled_state_norm_mean", 0.0) or 0.0),
            4,
        ),
    }


def build_candidate_batch(
    *,
    state: AgentState,
    candidates: list[dict[str, object]],
    config: HybridTolbertSSMConfig,
    device: str = "cpu",
    decoder_vocab: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    family_ids = []
    path_ids = []
    token_batches = []
    scalar_batches = []
    recent_steps = [
        _history_step_frame(step)
        for step in state.history[-max(0, config.sequence_length - 1) :]
    ]
    for candidate in candidates:
        step_frames = recent_steps + [_candidate_step_frame(state, candidate)]
        frames = step_frames[-config.sequence_length :]
        while len(frames) < config.sequence_length:
            frames.insert(0, {"command": "", "features": [0.0] * config.scalar_feature_dim})
        token_rows = []
        scalar_rows = []
        for frame in frames:
            token_rows.append(encode_command_tokens(str(frame.get("command", "")), config))
            features = list(frame.get("features", []))
            scalar_rows.append((features + [0.0] * config.scalar_feature_dim)[: config.scalar_feature_dim])
        token_batches.append(token_rows)
        scalar_batches.append(scalar_rows)
        family_ids.append(hashed_id(str(state.task.metadata.get("benchmark_family", "bounded")), config.family_vocab_size))
        path_ids.append(
            [
                hashed_id(value, config.path_vocab_size)
                for value in _path_signature_values(state, config)
            ]
        )
    decoder_inputs = []
    decoder_targets = []
    for candidate in candidates:
        input_ids, target_ids = encode_decoder_sequence(
            str(candidate.get("command", candidate.get("content", ""))),
            config,
            decoder_vocab=decoder_vocab,
        )
        decoder_inputs.append(input_ids)
        decoder_targets.append(target_ids)
    return {
        "command_token_ids": torch.tensor(token_batches, dtype=torch.long, device=device),
        "decoder_input_ids": torch.tensor(decoder_inputs, dtype=torch.long, device=device),
        "decoder_target_ids": torch.tensor(decoder_targets, dtype=torch.long, device=device),
        "scalar_features": torch.tensor(scalar_batches, dtype=torch.float32, device=device),
        "family_ids": torch.tensor(family_ids, dtype=torch.long, device=device),
        "path_level_ids": torch.tensor(path_ids, dtype=torch.long, device=device),
    }


def build_prompt_condition_batch(
    *,
    prompts: list[str],
    config: HybridTolbertSSMConfig,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    token_batches = []
    scalar_batches = []
    family_ids = []
    path_ids = []
    zero_features = [0.0] * config.scalar_feature_dim
    for prompt in prompts:
        frames = [
            {"command": "", "features": zero_features}
            for _ in range(max(0, config.sequence_length - 1))
        ]
        frames.append({"command": str(prompt).strip(), "features": zero_features})
        token_batches.append([encode_command_tokens(str(frame["command"]), config) for frame in frames])
        scalar_batches.append([list(frame["features"]) for frame in frames])
        family_ids.append(hashed_id("universal", config.family_vocab_size))
        path_ids.append(
            [
                hashed_id(value, config.path_vocab_size)
                for value in ("universal", "language_model", "decoder", "prompt")
            ][: config.max_path_levels]
        )
    while path_ids and len(path_ids[0]) < config.max_path_levels:
        for row in path_ids:
            row.append(0)
    return {
        "command_token_ids": torch.tensor(token_batches, dtype=torch.long, device=device),
        "scalar_features": torch.tensor(scalar_batches, dtype=torch.float32, device=device),
        "family_ids": torch.tensor(family_ids, dtype=torch.long, device=device),
        "path_level_ids": torch.tensor(path_ids, dtype=torch.long, device=device),
    }


def _candidate_step_frame(state: AgentState, candidate: dict[str, object]) -> dict[str, object]:
    heuristic_score = float(candidate.get("score", candidate.get("hybrid_total_score", 0.0)) or 0.0)
    trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False)) if state.context_packet else False
    command = str(candidate.get("command", candidate.get("content", ""))).strip()
    learned = _runtime_world_feedback(state.latent_state_summary)
    return {
        "command": command,
        "features": [
            float(state.context_packet.control.get("path_confidence", 0.0)) if state.context_packet else 0.0,
            1.0 if trust_retrieval else 0.0,
            float(state.latest_state_transition.get("progress_delta", 0.0) or 0.0),
            float(len(state.latest_state_transition.get("regressions", []))),
            1.0 if bool(candidate.get("retrieval_influenced", False)) else 0.0,
            1.0 if bool(candidate.get("retrieval_ranked_skill", False)) else 0.0,
            min(1.0, len(state.available_skills) / 5.0),
            min(1.0, len(state.retrieval_direct_candidates) / 5.0),
            1.0 if str(candidate.get("reason", "")).strip().startswith("retrieval") else 0.0,
            1.0 if str(candidate.get("action", "code_execute")) == "code_execute" else 0.0,
            1.0 if str(candidate.get("action", "")) == "respond" else 0.0,
            float(latent_command_bias(state.latent_state_summary, command)),
            heuristic_score / 10.0 if heuristic_score > 0.0 else 0.0,
            learned["progress_signal"],
            learned["risk_signal"],
        ],
    }


def _history_step_frame(step: Any) -> dict[str, object]:
    latent_state_summary = getattr(step, "latent_state_summary", {})
    proposal_metadata = getattr(step, "proposal_metadata", {})
    learned = _runtime_world_feedback(
        latent_state_summary if isinstance(latent_state_summary, dict) else {},
        proposal_metadata if isinstance(proposal_metadata, dict) else {},
    )
    verification = getattr(step, "verification", {})
    if not isinstance(verification, dict):
        verification = {}
    state_transition = getattr(step, "state_transition", {})
    if not isinstance(state_transition, dict):
        state_transition = {}
    return {
        "command": str(getattr(step, "content", "")).strip(),
        "features": [
            float(getattr(step, "path_confidence", 0.0) or 0.0),
            1.0 if bool(getattr(step, "trust_retrieval", False)) else 0.0,
            float(getattr(step, "state_progress_delta", 0.0) or 0.0),
            float(getattr(step, "state_regression_count", 0.0) or 0.0),
            1.0 if bool(getattr(step, "retrieval_influenced", False)) else 0.0,
            1.0 if bool(getattr(step, "retrieval_ranked_skill", False)) else 0.0,
            min(1.0, float(getattr(step, "available_skill_count", 0) or 0) / 5.0),
            min(1.0, float(getattr(step, "retrieval_candidate_count", 0) or 0) / 5.0),
            min(1.0, float(getattr(step, "retrieval_evidence_count", 0) or 0) / 5.0),
            1.0 if str(getattr(step, "action", "")).strip() == "code_execute" else 0.0,
            1.0 if str(getattr(step, "action", "")).strip() == "respond" else 0.0,
            1.0 if bool(verification.get("passed", False)) else 0.0,
            float(getattr(step, "latent_command_bias", 0.0) or 0.0),
            1.0 if bool(state_transition.get("no_progress", False)) else 0.0,
            learned["progress_signal"],
            learned["risk_signal"],
        ],
    }


def _runtime_world_feedback(
    latent_state_summary: dict[str, object],
    proposal_metadata: dict[str, object] | None = None,
) -> dict[str, float]:
    learned = latent_state_summary.get("learned_world_state", {})
    if not isinstance(learned, dict):
        learned = {}
    proposal = proposal_metadata if isinstance(proposal_metadata, dict) else {}
    progress_signal = max(
        float(learned.get("progress_signal", 0.0) or 0.0),
        float(learned.get("world_progress_score", 0.0) or 0.0),
        float(learned.get("decoder_world_progress_score", 0.0) or 0.0),
        float(proposal.get("hybrid_world_progress_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_progress_score", 0.0) or 0.0),
    )
    risk_signal = max(
        float(learned.get("risk_signal", 0.0) or 0.0),
        float(learned.get("world_risk_score", 0.0) or 0.0),
        float(learned.get("decoder_world_risk_score", 0.0) or 0.0),
        float(proposal.get("hybrid_world_risk_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_risk_score", 0.0) or 0.0),
    )
    return {
        "progress_signal": max(0.0, min(1.0, progress_signal)),
        "risk_signal": max(0.0, min(1.0, risk_signal)),
    }


def _path_signature_values(state: AgentState, config: HybridTolbertSSMConfig) -> list[str]:
    values = [
        str(state.task.metadata.get("benchmark_family", "bounded")).strip() or "bounded",
        str(state.task.metadata.get("capability", "generic")).strip() or "generic",
        str(state.task.metadata.get("difficulty", "seed")).strip() or "seed",
        state.task.task_id,
    ]
    return values[: config.max_path_levels]


def _prefer_python_ref_for_device(device: str) -> bool:
    return not str(device).strip().lower().startswith("cuda")


def _decoder_sequence_logprob(
    decoder_logits: torch.Tensor,
    decoder_target_ids: torch.Tensor,
) -> float:
    log_probs = torch.log_softmax(decoder_logits, dim=-1)
    total = 0.0
    count = 0
    for index, token_id in enumerate(decoder_target_ids.tolist()):
        token = int(token_id)
        if token <= 0:
            continue
        total += float(log_probs[index, token].item())
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _normalized_scoring_policy(policy: dict[str, object] | None) -> dict[str, float]:
    defaults = {
        "learned_score_weight": 1.5,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "risk_penalty_weight": 1.0,
        "stop_weight": 1.0,
        "transition_progress_weight": 0.15,
        "transition_regression_penalty_weight": 0.20,
        "world_progress_weight": 0.20,
        "world_risk_penalty_weight": 0.20,
        "decoder_world_progress_weight": 0.20,
        "decoder_world_risk_penalty_weight": 0.20,
        "latent_bias_weight": 0.10,
        "decoder_logprob_weight": 0.10,
        "respond_learned_score_weight": 1.25,
        "respond_policy_weight": 0.0,
        "respond_value_weight": 1.0,
        "respond_risk_penalty_weight": 1.0,
        "respond_stop_weight": 1.0,
        "respond_transition_progress_weight": 0.0,
        "respond_transition_regression_penalty_weight": 0.0,
        "respond_world_progress_weight": 0.10,
        "respond_world_risk_penalty_weight": 0.20,
        "respond_decoder_world_progress_weight": 0.20,
        "respond_decoder_world_risk_penalty_weight": 0.30,
        "respond_latent_bias_weight": 0.0,
        "respond_decoder_logprob_weight": 0.05,
    }
    if not isinstance(policy, dict):
        return defaults
    normalized = dict(defaults)
    for key, default in defaults.items():
        try:
            normalized[key] = float(policy.get(key, default))
        except (TypeError, ValueError):
            normalized[key] = default
    return normalized


def _world_initial_belief_payload(
    *,
    batch_size: int,
    state: AgentState | None,
    config: HybridTolbertSSMConfig,
    manifest: dict[str, object],
    device: str,
) -> dict[str, object]:
    payload = {
        "log_belief": None,
        "diagnostics": {
            "backend": "",
            "matched_state_index": -1,
            "matched_state_probability": 0.0,
            "signature_norm": 0.0,
            "token_count": 0,
            "horizons": [],
        },
    }
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        return payload
    profile_path = str(metadata.get("causal_world_profile_path", "")).strip()
    if not profile_path:
        return payload
    profile = load_causal_world_profile(profile_path)
    if profile.num_states != config.world_state_dim:
        return payload
    prior = condition_causal_world_prior(
        profile,
        text_fragments=_state_profile_fragments(state),
    )
    log_prior = torch.tensor(prior.log_prior, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
    payload["log_belief"] = log_prior
    payload["diagnostics"] = {
        "backend": prior.backend,
        "matched_state_index": prior.matched_state_index,
        "matched_state_probability": prior.matched_state_probability,
        "signature_norm": prior.signature_norm,
        "token_count": prior.token_count,
        "horizons": list(profile.horizons),
        "profile_path": str(profile.profile_path),
    }
    return payload


def _world_initial_log_belief(
    *,
    batch_size: int,
    state: AgentState | None,
    config: HybridTolbertSSMConfig,
    manifest: dict[str, object],
    device: str,
) -> torch.Tensor | None:
    payload = _world_initial_belief_payload(
        batch_size=batch_size,
        state=state,
        config=config,
        manifest=manifest,
        device=device,
    )
    log_belief = payload.get("log_belief")
    return log_belief if isinstance(log_belief, torch.Tensor) else None


def generate_hybrid_decoder_text(
    *,
    state: AgentState,
    bundle_manifest_path: Path,
    device: str = "cpu",
    max_new_tokens: int | None = None,
    use_incremental_cache: bool = True,
    verify_cache_equivalence: bool = True,
    cache_equivalence_steps: int = 2,
    cache_equivalence_atol: float = 1.0e-4,
) -> dict[str, object]:
    model, config, manifest = load_hybrid_runtime_bundle(bundle_manifest_path, device=device)
    decoder_vocab = _load_decoder_vocabulary(manifest)
    batch = build_candidate_batch(
        state=state,
        candidates=[{"command": ""}],
        config=config,
        device=device,
        decoder_vocab=decoder_vocab,
    )
    world_prior = _world_initial_belief_payload(
        batch_size=1,
        state=state,
        config=config,
        manifest=manifest,
        device=device,
    )
    encoded = model.encode_condition(
        command_token_ids=batch["command_token_ids"],
        scalar_features=batch["scalar_features"],
        family_ids=batch["family_ids"],
        path_level_ids=batch["path_level_ids"],
        prefer_python_ref=_prefer_python_ref_for_device(device),
        prefer_python_world_ref=_prefer_python_ref_for_device(device),
        world_initial_log_belief=world_prior["log_belief"],
    )
    generated_ids, token_log_probs, cache = _generate_decoder_ids(
        model=model,
        config=config,
        pooled_state=encoded["pooled_state"],
        initial_world_belief=_resolved_initial_world_belief(
            encoded_world_belief=encoded.get("world_final_belief"),
            prior_world_belief=world_prior.get("log_belief"),
        ),
        device=device,
        max_new_tokens=max_new_tokens,
        use_incremental_cache=use_incremental_cache,
        verify_cache_equivalence=verify_cache_equivalence,
        cache_equivalence_steps=cache_equivalence_steps,
        cache_equivalence_atol=cache_equivalence_atol,
    )
    return {
        "model_family": str(manifest.get("model_family", "")),
        "generated_ids": generated_ids,
        "generated_text": decode_decoder_ids(generated_ids, config, decoder_vocab=decoder_vocab),
        "avg_logprob": (sum(token_log_probs) / len(token_log_probs)) if token_log_probs else 0.0,
        "cache_mode": cache.cache_mode,
        "cache_equivalent": cache.cache_equivalent,
        "cache_max_abs_diff": cache.max_abs_diff,
        "cache_steps_reused": cache.steps_reused,
        "cache_fallback_used": cache.fallback_used,
        "world_cache_mode": cache.world_cache_mode,
        "world_cache_equivalent": cache.world_cache_equivalent,
        "world_cache_max_abs_diff": cache.world_max_abs_diff,
        "world_cache_steps_reused": cache.world_steps_reused,
        "world_cache_fallback_used": cache.world_fallback_used,
        **_world_belief_summary(cache.world_last_belief),
    }


def generate_hybrid_decoder_completion(
    *,
    prompt: str,
    bundle_manifest_path: Path,
    device: str = "cpu",
    max_new_tokens: int | None = None,
    use_incremental_cache: bool = True,
    verify_cache_equivalence: bool = True,
    cache_equivalence_steps: int = 2,
    cache_equivalence_atol: float = 1.0e-4,
) -> dict[str, object]:
    model, config, manifest = load_hybrid_runtime_bundle(bundle_manifest_path, device=device)
    decoder_vocab = _load_decoder_vocabulary(manifest)
    batch = build_prompt_condition_batch(
        prompts=[prompt],
        config=config,
        device=device,
    )
    world_initial_log_belief = _world_initial_log_belief(
        batch_size=1,
        state=None,
        config=config,
        manifest=manifest,
        device=device,
    )
    encoded = model.encode_condition(
        command_token_ids=batch["command_token_ids"],
        scalar_features=batch["scalar_features"],
        family_ids=batch["family_ids"],
        path_level_ids=batch["path_level_ids"],
        prefer_python_ref=_prefer_python_ref_for_device(device),
        prefer_python_world_ref=_prefer_python_ref_for_device(device),
        world_initial_log_belief=world_initial_log_belief,
    )
    generated_ids, token_log_probs, cache = _generate_decoder_ids(
        model=model,
        config=config,
        pooled_state=encoded["pooled_state"],
        initial_world_belief=_resolved_initial_world_belief(
            encoded_world_belief=encoded.get("world_final_belief"),
            prior_world_belief=world_initial_log_belief,
        ),
        device=device,
        max_new_tokens=max_new_tokens,
        use_incremental_cache=use_incremental_cache,
        verify_cache_equivalence=verify_cache_equivalence,
        cache_equivalence_steps=cache_equivalence_steps,
        cache_equivalence_atol=cache_equivalence_atol,
    )
    return {
        "model_family": str(manifest.get("model_family", "")),
        "prompt": str(prompt),
        "generated_ids": generated_ids,
        "generated_text": decode_decoder_ids(generated_ids, config, decoder_vocab=decoder_vocab),
        "avg_logprob": (sum(token_log_probs) / len(token_log_probs)) if token_log_probs else 0.0,
        "cache_mode": cache.cache_mode,
        "cache_equivalent": cache.cache_equivalent,
        "cache_max_abs_diff": cache.max_abs_diff,
        "cache_steps_reused": cache.steps_reused,
        "cache_fallback_used": cache.fallback_used,
        "world_cache_mode": cache.world_cache_mode,
        "world_cache_equivalent": cache.world_cache_equivalent,
        "world_cache_max_abs_diff": cache.world_max_abs_diff,
        "world_cache_steps_reused": cache.world_steps_reused,
        "world_cache_fallback_used": cache.world_fallback_used,
        **_world_belief_summary(cache.world_last_belief),
    }


def _generate_decoder_ids(
    *,
    model: Any,
    config: HybridTolbertSSMConfig,
    pooled_state: torch.Tensor,
    initial_world_belief: torch.Tensor | None,
    device: str,
    max_new_tokens: int | None,
    use_incremental_cache: bool,
    verify_cache_equivalence: bool,
    cache_equivalence_steps: int,
    cache_equivalence_atol: float,
) -> tuple[list[int], list[float], HybridDecoderCache]:
    input_ids = torch.full(
        (pooled_state.shape[0], config.max_command_tokens),
        config.decoder_pad_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = config.decoder_bos_token_id
    token_limit = min(config.max_command_tokens, max(1, int(max_new_tokens or config.max_command_tokens)))
    world_enabled = bool(config.use_world_model and isinstance(initial_world_belief, torch.Tensor))
    cache = HybridDecoderCache(
        pooled_state=pooled_state,
        cache_mode="incremental_recurrent" if use_incremental_cache else "full_recompute",
        cache_equivalent=True if use_incremental_cache and verify_cache_equivalence else None,
        world_initial_belief=initial_world_belief.clone() if world_enabled else None,
        world_last_belief=initial_world_belief.clone() if world_enabled else None,
        world_cache_mode=(
            "incremental_belief"
            if world_enabled and use_incremental_cache
            else ("full_recompute" if world_enabled else "disabled")
        ),
        world_cache_equivalent=True if world_enabled and use_incremental_cache and verify_cache_equivalence else None,
    )
    generated_ids: list[int] = []
    token_log_probs: list[float] = []
    incremental_enabled = bool(use_incremental_cache)
    incremental_world_enabled = bool(use_incremental_cache and world_enabled)
    prefer_python_ref = _prefer_python_ref_for_device(device)
    with torch.no_grad():
        for position in range(token_limit):
            if incremental_enabled:
                logits, next_state, next_world_belief, step_diagnostics = model.decode_decoder_step(
                    pooled_state=cache.pooled_state,
                    token_ids=input_ids[:, position],
                    position=position,
                    decoder_state=cache.decoder_last_state,
                    prefer_python_ref=prefer_python_ref,
                    world_log_belief=cache.world_last_belief if world_enabled else None,
                    prefer_python_world_ref=prefer_python_ref,
                )
                cache.decoder_last_state = next_state
                cache.steps_reused += 1
                cache.step_diagnostics.append(step_diagnostics)
                if world_enabled and incremental_world_enabled:
                    cache.world_last_belief = next_world_belief
                    cache.world_steps_reused += 1
                    cache.world_step_diagnostics.append(step_diagnostics)
                if verify_cache_equivalence and position < max(0, int(cache_equivalence_steps)):
                    full_logits, _full_diag = model.decode_decoder_tokens(
                        pooled_state=cache.pooled_state,
                        decoder_input_ids=input_ids,
                        prefer_python_ref=prefer_python_ref,
                        initial_world_log_belief=cache.world_initial_belief if world_enabled else None,
                        prefer_python_world_ref=prefer_python_ref,
                    )
                    diff = float(torch.max(torch.abs(full_logits[:, position, :] - logits)).item())
                    cache.max_abs_diff = max(cache.max_abs_diff, diff)
                    if diff > float(cache_equivalence_atol):
                        cache.cache_equivalent = False
                        cache.fallback_used = True
                        incremental_enabled = False
                        logits = full_logits[:, position, :]
                    elif cache.cache_equivalent is not False:
                        cache.cache_equivalent = True
            else:
                full_logits, full_diag = model.decode_decoder_tokens(
                    pooled_state=cache.pooled_state,
                    decoder_input_ids=input_ids,
                    prefer_python_ref=prefer_python_ref,
                    initial_world_log_belief=cache.world_initial_belief if world_enabled else None,
                    prefer_python_world_ref=prefer_python_ref,
                )
                logits = full_logits[:, position, :]
                if world_enabled:
                    full_world_belief = full_diag.get("world_final_belief") if isinstance(full_diag, dict) else None
                    cache.world_last_belief = full_world_belief
                    cache.world_step_diagnostics.append(dict(full_diag, position=int(position)))
            if world_enabled:
                prefix_ids = input_ids[:, : position + 1]
                if incremental_world_enabled:
                    if verify_cache_equivalence and position < max(0, int(cache_equivalence_steps)):
                        _full_logits, full_diag = model.decode_decoder_tokens(
                            pooled_state=cache.pooled_state,
                            decoder_input_ids=prefix_ids,
                            initial_world_log_belief=cache.world_initial_belief,
                            prefer_python_ref=prefer_python_ref,
                            prefer_python_world_ref=prefer_python_ref,
                        )
                        full_world_belief = full_diag.get("world_final_belief") if isinstance(full_diag, dict) else None
                        if isinstance(full_world_belief, torch.Tensor) and isinstance(next_world_belief, torch.Tensor):
                            diff = float(torch.max(torch.abs(full_world_belief - next_world_belief)).item())
                            cache.world_max_abs_diff = max(cache.world_max_abs_diff, diff)
                            if diff > float(cache_equivalence_atol):
                                cache.world_cache_equivalent = False
                                cache.world_fallback_used = True
                                incremental_world_enabled = False
                                incremental_enabled = False
                                cache.fallback_used = True
                                cache.world_last_belief = full_world_belief
                            elif cache.world_cache_equivalent is not False:
                                cache.world_cache_equivalent = True
                else:
                    _full_logits, full_world_diag = model.decode_decoder_tokens(
                        pooled_state=cache.pooled_state,
                        decoder_input_ids=prefix_ids,
                        initial_world_log_belief=cache.world_initial_belief,
                        prefer_python_ref=prefer_python_ref,
                        prefer_python_world_ref=prefer_python_ref,
                    )
                    full_world_belief = full_world_diag.get("world_final_belief") if isinstance(full_world_diag, dict) else None
                    cache.world_last_belief = full_world_belief
                    cache.world_steps_reused += 1
                    cache.world_step_diagnostics.append(dict(full_world_diag, position=int(position)))
            token_id = int(torch.argmax(logits[0]).item())
            generated_ids.append(token_id)
            token_log_probs.append(float(torch.log_softmax(logits[0], dim=-1)[token_id].item()))
            if token_id in {config.decoder_pad_token_id, config.decoder_eos_token_id}:
                break
            if position + 1 < config.max_command_tokens:
                input_ids[0, position + 1] = token_id
    return generated_ids, token_log_probs, cache


def _resolved_initial_world_belief(
    *,
    encoded_world_belief: object,
    prior_world_belief: object,
) -> torch.Tensor | None:
    if isinstance(encoded_world_belief, torch.Tensor):
        return encoded_world_belief
    if isinstance(prior_world_belief, torch.Tensor):
        return prior_world_belief
    return None


def _world_belief_summary(world_log_belief: torch.Tensor | None) -> dict[str, object]:
    if not isinstance(world_log_belief, torch.Tensor):
        return {
            "world_final_top_states": [],
            "world_final_top_state_probs": [],
            "world_final_entropy_mean": 0.0,
        }
    probs = world_log_belief.exp()
    top = torch.topk(probs[0], k=min(3, probs.shape[-1]))
    safe = probs.clamp_min(1.0e-8)
    entropy = -(safe * safe.log()).sum(dim=-1).mean()
    return {
        "world_final_top_states": [int(value) for value in top.indices.tolist()],
        "world_final_top_state_probs": [round(float(value), 4) for value in top.values.tolist()],
        "world_final_entropy_mean": round(float(entropy.item()), 4),
    }


def _state_profile_fragments(state: AgentState | None) -> list[str]:
    if state is None:
        return []
    fragments: list[str] = [str(state.task.prompt)]
    fragments.extend(str(command) for command in state.task.suggested_commands[:4])
    fragments.extend(str(goal) for goal in state.plan[:4])
    fragments.extend(str(goal) for goal in state.initial_plan[:4])
    fragments.extend(str(item) for item in state.world_model_summary.get("missing_expected_artifacts", [])[:6])
    fragments.extend(str(item) for item in state.world_model_summary.get("present_forbidden_artifacts", [])[:6])
    fragments.extend(str(item) for item in state.latest_state_transition.get("regressions", [])[:4])
    fragments.extend(str(step.content) for step in state.history[-4:] if str(step.content).strip())
    return [fragment.strip() for fragment in fragments if str(fragment).strip()]


def _load_decoder_vocabulary(manifest: dict[str, object]) -> dict[str, int]:
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    raw_path = str(metadata.get("decoder_vocab_path", "")).strip()
    if not raw_path:
        return {}
    vocab_path = Path(raw_path)
    if not vocab_path.exists():
        return {}
    try:
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for token, value in payload.items():
        try:
            normalized[str(token)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized
