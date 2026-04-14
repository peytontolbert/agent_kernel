from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from ..world.causal_machine import condition_causal_world_prior, load_causal_world_profile
from ..world.latent_state import latent_command_bias
from agent_kernel.state import AgentState, _software_work_objective_phase, _software_work_phase_rank

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
    task_horizon = str(state.world_model_summary.get("horizon", "")).strip() or str(
        state.task.metadata.get("difficulty", "")
    ).strip()
    is_long_horizon = task_horizon == "long_horizon"
    horizon_values: list[float] = []
    for value in world_prior["diagnostics"].get("horizons", []):
        try:
            horizon_values.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            continue
    horizon_extent = max(horizon_values, default=0.0)
    horizon_scale = 1.0 + (
        weights["long_horizon_horizon_scale_weight"] * min(3.0, math.log1p(horizon_extent))
    )
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
        decoder_world_belief_vector: list[float] = []
        decoder_world_final = decoder_diagnostics.get("world_final_belief")
        if isinstance(decoder_world_final, torch.Tensor) and decoder_world_final.dim() == 2:
            decoder_world_probs = decoder_world_final[index].exp()
            decoder_world_belief_vector = _tensor_row_to_float_list(decoder_world_probs)
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
        world_belief_vector: list[float] = []
        if output.world_final_belief is not None:
            world_probs = output.world_final_belief[index].exp()
            world_belief_vector = _tensor_row_to_float_list(world_probs)
            world_progress_score = float(world_probs[0].item()) if world_probs.numel() > 0 else 0.0
            world_risk_score = float(world_probs[1].item()) if world_probs.numel() > 1 else 0.0
            if world_probs.numel() > 2:
                world_risk_score += float(world_probs[2].item())
            if world_probs.numel() > 3:
                world_risk_score += float(world_probs[3].item())
        latent_bias = float(latent_command_bias(state.latent_state_summary, str(candidate.get("command", candidate.get("content", "")))))
        recovery_stage_alignment, recovery_stage_rank, recovery_stage_objective = _planner_recovery_stage_alignment(
            state,
            candidate,
        )
        software_work_alignment, software_work_rank, software_work_objective = _software_work_stage_alignment(
            state,
            candidate,
        )
        software_work_transition_alignment, software_work_stage_status = _software_work_transition_alignment(
            state,
            candidate,
            objective=software_work_objective,
        )
        software_work_phase_alignment, software_work_candidate_phase, software_work_suggested_phase = (
            _software_work_phase_boundary_alignment(
                state,
                candidate,
                objective=software_work_objective,
            )
        )
        software_work_phase_gate_alignment, software_work_gate_objective = _software_work_phase_gate_alignment(
            state,
            candidate,
            objective=software_work_objective,
        )
        campaign_contract_alignment, campaign_contract_rank, campaign_contract_objective = _campaign_contract_alignment(
            state,
            candidate,
        )
        campaign_drift_penalty, campaign_drift_reason = _campaign_contract_drift_penalty(
            state,
            candidate,
        )
        trusted_retrieval_alignment = _trusted_retrieval_alignment(state, candidate)
        trusted_retrieval_procedure_alignment = _trusted_retrieval_procedure_alignment(state, candidate)
        graph_environment_alignment = _graph_environment_alignment(state, candidate)
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
            + (weights["recovery_stage_alignment_weight"] * recovery_stage_alignment)
            + (weights["software_work_alignment_weight"] * software_work_alignment)
            + (weights["software_work_transition_weight"] * software_work_transition_alignment)
            + (weights["software_work_phase_handoff_weight"] * software_work_phase_alignment)
            + (weights["software_work_phase_gate_weight"] * software_work_phase_gate_alignment)
            + (weights["campaign_contract_alignment_weight"] * campaign_contract_alignment)
            - (weights["campaign_drift_penalty_weight"] * max(0.0, campaign_drift_penalty))
            + (weights["trusted_retrieval_alignment_weight"] * trusted_retrieval_alignment)
            + (weights["trusted_retrieval_procedure_alignment_weight"] * trusted_retrieval_procedure_alignment)
            + (weights["graph_environment_alignment_weight"] * graph_environment_alignment)
        )
        long_horizon_progress_signal = max(
            transition_progress,
            world_progress_score,
            decoder_world_progress_score,
        )
        long_horizon_risk_signal = max(
            max(0.0, transition_regression),
            world_risk_score,
            decoder_world_risk_score,
            risk_score,
        )
        if is_long_horizon and str(candidate.get("action", "code_execute")) != "respond":
            total_score += horizon_scale * (
                (weights["long_horizon_progress_bonus_weight"] * long_horizon_progress_signal)
                - (weights["long_horizon_risk_penalty_weight"] * long_horizon_risk_signal)
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
                "hybrid_world_belief_vector": world_belief_vector,
                "hybrid_world_belief_top_states": _belief_vector_top_states(world_belief_vector),
                "hybrid_world_belief_top_state_probs": _belief_vector_top_state_probs(world_belief_vector),
                "hybrid_decoder_world_progress_score": decoder_world_progress_score,
                "hybrid_decoder_world_risk_score": decoder_world_risk_score,
                "hybrid_decoder_world_entropy_mean": decoder_world_entropy_mean,
                "hybrid_decoder_world_belief_vector": decoder_world_belief_vector,
                "hybrid_decoder_world_belief_top_states": _belief_vector_top_states(decoder_world_belief_vector),
                "hybrid_decoder_world_belief_top_state_probs": _belief_vector_top_state_probs(
                    decoder_world_belief_vector
                ),
                "hybrid_task_horizon": task_horizon,
                "hybrid_long_horizon_scale": horizon_scale if is_long_horizon else 1.0,
                "hybrid_long_horizon_progress_signal": long_horizon_progress_signal,
                "hybrid_long_horizon_risk_signal": long_horizon_risk_signal,
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
                "hybrid_world_prior_horizon_hint": str(world_prior["diagnostics"].get("horizon_hint", "")),
                "hybrid_world_prior_bias_strength": float(
                    world_prior["diagnostics"].get("applied_bias_strength", 1.0) or 1.0
                ),
                "hybrid_world_profile_horizons": list(world_prior["diagnostics"].get("horizons", [])),
                "hybrid_world_signature_token_count": int(world_prior["diagnostics"].get("token_count", 0) or 0),
                "hybrid_latent_bias": latent_bias,
                "hybrid_recovery_stage_alignment": recovery_stage_alignment,
                "hybrid_recovery_stage_rank": recovery_stage_rank,
                "hybrid_recovery_stage_objective": recovery_stage_objective,
                "hybrid_software_work_alignment": software_work_alignment,
                "hybrid_software_work_rank": software_work_rank,
                "hybrid_software_work_objective": software_work_objective,
                "hybrid_software_work_transition_alignment": software_work_transition_alignment,
                "hybrid_software_work_stage_status": software_work_stage_status,
                "hybrid_software_work_phase_alignment": software_work_phase_alignment,
                "hybrid_software_work_candidate_phase": software_work_candidate_phase,
                "hybrid_software_work_suggested_phase": software_work_suggested_phase,
                "hybrid_software_work_phase_gate_alignment": software_work_phase_gate_alignment,
                "hybrid_software_work_gate_objective": software_work_gate_objective,
                "hybrid_campaign_contract_alignment": campaign_contract_alignment,
                "hybrid_campaign_contract_rank": campaign_contract_rank,
                "hybrid_campaign_contract_objective": campaign_contract_objective,
                "hybrid_campaign_drift_penalty": campaign_drift_penalty,
                "hybrid_campaign_drift_reason": campaign_drift_reason,
                "hybrid_trusted_retrieval_alignment": trusted_retrieval_alignment,
                "hybrid_trusted_retrieval_procedure_alignment": trusted_retrieval_procedure_alignment,
                "hybrid_graph_environment_alignment": graph_environment_alignment,
                "hybrid_transfer_novelty": _historical_environment_novelty(state),
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
    controller_scores: dict[str, float] = {}
    controller_world_vectors: dict[str, list[float]] = {}
    controller_decoder_vectors: dict[str, list[float]] = {}
    for item in scored:
        controller_mode_key = _controller_mode_for_probe(item)
        if not controller_mode_key:
            continue
        score = float(item.get("hybrid_total_score", 0.0) or 0.0)
        if controller_mode_key not in controller_scores or score > controller_scores[controller_mode_key]:
            controller_scores[controller_mode_key] = score
            controller_world_vectors[controller_mode_key] = _float_list(item.get("hybrid_world_belief_vector", []))
            controller_decoder_vectors[controller_mode_key] = _float_list(
                item.get("hybrid_decoder_world_belief_vector", [])
            )
    controller_belief = _controller_belief_distribution(controller_scores)
    controller_mode = max(controller_belief, key=controller_belief.get) if controller_belief else ""
    controller_expected_world_belief = _weighted_belief_vector(controller_world_vectors, controller_belief)
    controller_expected_decoder_world_belief = _weighted_belief_vector(
        controller_decoder_vectors,
        controller_belief,
    )
    controller_world_summary = _belief_vector_summary(controller_expected_world_belief)
    controller_decoder_summary = _belief_vector_summary(controller_expected_decoder_world_belief)
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
        "controller_belief": controller_belief,
        "controller_mode": controller_mode,
        "controller_mode_probability": round(float(controller_belief.get(controller_mode, 0.0) or 0.0), 4)
        if controller_mode
        else 0.0,
        "controller_expected_world_belief": controller_expected_world_belief,
        "controller_expected_world_top_states": controller_world_summary["top_states"],
        "controller_expected_world_top_state_probs": controller_world_summary["top_state_probs"],
        "controller_expected_world_entropy_mean": controller_world_summary["entropy_mean"],
        "controller_expected_decoder_world_belief": controller_expected_decoder_world_belief,
        "controller_expected_decoder_world_top_states": controller_decoder_summary["top_states"],
        "controller_expected_decoder_world_top_state_probs": controller_decoder_summary["top_state_probs"],
        "controller_expected_decoder_world_entropy_mean": controller_decoder_summary["entropy_mean"],
        "world_prior_backend": str(top.get("hybrid_world_prior_backend", "")).strip(),
        "world_prior_top_state": int(top.get("hybrid_world_prior_top_state", -1) or -1),
        "world_prior_top_probability": round(
            float(top.get("hybrid_world_prior_top_probability", 0.0) or 0.0),
            4,
        ),
        "world_prior_horizon_hint": str(top.get("hybrid_world_prior_horizon_hint", "")).strip(),
        "world_prior_bias_strength": round(float(top.get("hybrid_world_prior_bias_strength", 1.0) or 1.0), 4),
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
    trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False)) if state.context_packet else False
    command = str(candidate.get("command", candidate.get("content", ""))).strip()
    learned = _runtime_world_feedback(state.latent_state_summary)
    no_progress = bool(state.latest_state_transition.get("no_progress", False))
    return {
        "command": command,
        "features": [
            float(state.context_packet.control.get("path_confidence", 0.0)) if state.context_packet else 0.0,
            1.0 if trust_retrieval else 0.0,
            float(state.latest_state_transition.get("progress_delta", 0.0) or 0.0),
            float(len(state.latest_state_transition.get("regressions", []))),
            1.0 if bool(candidate.get("retrieval_influenced", False)) else 0.0,
            1.0 if bool(candidate.get("retrieval_ranked_skill", False)) else 0.0,
            learned["trusted_retrieval_alignment"],
            learned["graph_environment_alignment"],
            learned["transfer_novelty"],
            1.0 if str(candidate.get("action", "code_execute")) == "code_execute" else 0.0,
            1.0 if str(candidate.get("action", "")) == "respond" else 0.0,
            0.0,
            float(latent_command_bias(state.latent_state_summary, command)),
            1.0 if no_progress else 0.0,
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
            learned["trusted_retrieval_alignment"],
            learned["graph_environment_alignment"],
            learned["transfer_novelty"],
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
    trusted_retrieval_alignment = max(
        float(learned.get("trusted_retrieval_alignment", 0.0) or 0.0),
        float(proposal.get("hybrid_trusted_retrieval_alignment", 0.0) or 0.0),
    )
    graph_environment_alignment = float(
        proposal.get(
            "hybrid_graph_environment_alignment",
            learned.get("graph_environment_alignment", 0.0),
        )
        or 0.0
    )
    transfer_novelty = max(
        float(learned.get("transfer_novelty", 0.0) or 0.0),
        float(proposal.get("hybrid_transfer_novelty", 0.0) or 0.0),
    )
    return {
        "progress_signal": max(0.0, min(1.0, progress_signal)),
        "risk_signal": max(0.0, min(1.0, risk_signal)),
        "trusted_retrieval_alignment": max(0.0, min(1.0, trusted_retrieval_alignment)),
        "graph_environment_alignment": max(-1.0, min(1.0, graph_environment_alignment)),
        "transfer_novelty": max(0.0, min(1.0, transfer_novelty)),
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
        "recovery_stage_alignment_weight": 0.35,
        "software_work_alignment_weight": 0.30,
        "software_work_transition_weight": 0.35,
        "software_work_phase_handoff_weight": 0.32,
        "software_work_phase_gate_weight": 0.42,
        "campaign_contract_alignment_weight": 0.38,
        "campaign_drift_penalty_weight": 0.30,
        "trusted_retrieval_alignment_weight": 0.45,
        "trusted_retrieval_procedure_alignment_weight": 0.40,
        "graph_environment_alignment_weight": 0.35,
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
        "long_horizon_progress_bonus_weight": 0.0,
        "long_horizon_risk_penalty_weight": 0.0,
        "long_horizon_horizon_scale_weight": 0.0,
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


def _controller_mode_for_probe(candidate: dict[str, object]) -> str:
    if str(candidate.get("action", "")).strip() == "respond":
        return "stop"
    command = str(candidate.get("command", candidate.get("content", ""))).strip()
    if command == "__probe_recover_state__":
        return "recover"
    if command == "__probe_continue_progress__":
        return "continue"
    return ""


def _controller_belief_distribution(scores: dict[str, float]) -> dict[str, float]:
    belief = {"recover": 0.0, "continue": 0.0, "stop": 0.0}
    if not scores:
        return belief
    max_score = max(scores.values())
    scaled = {key: math.exp(float(value) - max_score) for key, value in scores.items()}
    total = sum(scaled.values())
    if total <= 0.0:
        return belief
    for key, value in scaled.items():
        belief[key] = round(float(value / total), 4)
    return belief


def _weighted_belief_vector(
    vectors: dict[str, list[float]],
    weights: dict[str, float],
) -> list[float]:
    if not vectors:
        return []
    width = max((len(vector) for vector in vectors.values()), default=0)
    if width <= 0:
        return []
    aggregated = [0.0] * width
    total = 0.0
    for key, vector in vectors.items():
        weight = float(weights.get(key, 0.0) or 0.0)
        if weight <= 0.0 or not vector:
            continue
        total += weight
        for index, value in enumerate(vector[:width]):
            aggregated[index] += weight * float(value)
    if total <= 0.0:
        return []
    normalized = [max(0.0, value / total) for value in aggregated]
    norm_total = sum(normalized)
    if norm_total <= 0.0:
        return []
    return [round(value / norm_total, 4) for value in normalized]


def _belief_vector_summary(vector: list[float]) -> dict[str, object]:
    if not vector:
        return {
            "top_states": [],
            "top_state_probs": [],
            "entropy_mean": 0.0,
        }
    safe = [max(float(value), 1.0e-8) for value in vector]
    total = sum(safe)
    if total <= 0.0:
        return {
            "top_states": [],
            "top_state_probs": [],
            "entropy_mean": 0.0,
        }
    normalized = [value / total for value in safe]
    ranked = sorted(enumerate(normalized), key=lambda item: item[1], reverse=True)[:3]
    entropy = -sum(value * math.log(value) for value in normalized)
    return {
        "top_states": [int(index) for index, _ in ranked],
        "top_state_probs": [round(float(value), 4) for _, value in ranked],
        "entropy_mean": round(float(entropy), 4),
    }


def _belief_vector_top_states(vector: list[float]) -> list[int]:
    return list(_belief_vector_summary(vector)["top_states"])


def _belief_vector_top_state_probs(vector: list[float]) -> list[float]:
    return list(_belief_vector_summary(vector)["top_state_probs"])


def _tensor_row_to_float_list(row: torch.Tensor) -> list[float]:
    return [round(float(value), 4) for value in row.detach().cpu().tolist()]


def _float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    normalized: list[float] = []
    for value in values:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _planner_recovery_stage_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> tuple[float, int, str]:
    artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
    if str(artifact.get("kind", "")).strip() != "planner_recovery_rewrite":
        return 0.0, -1, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, -1, ""
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return 0.0, -1, ""
    ranked = artifact.get("ranked_objectives", [])
    if not isinstance(ranked, list):
        ranked = []
    for index, item in enumerate(ranked[:4], start=1):
        if not isinstance(item, dict):
            continue
        objective = str(item.get("objective", "")).strip()
        if not objective or not _candidate_matches_planner_recovery_objective(command, objective):
            continue
        raw_score = float(item.get("score", 0.0) or 0.0)
        scaled = max(0.0, min(1.5, raw_score / 100.0))
        scaled *= max(0.4, 1.0 - ((index - 1) * 0.15))
        return scaled, index, objective
    next_stage = str(artifact.get("next_stage_objective", "")).strip()
    if next_stage and _candidate_matches_planner_recovery_objective(command, next_stage):
        return 0.5, 1, next_stage
    return 0.0, -1, ""


def _software_work_stage_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> tuple[float, int, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, -1, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, -1, ""
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return 0.0, -1, ""
    objectives = state.software_work_plan_update()
    for index, objective in enumerate(objectives[:5], start=1):
        if not _candidate_matches_software_work_objective(command, objective):
            continue
        scale = max(0.35, 1.0 - ((index - 1) * 0.12))
        return scale, index, objective
    return 0.0, -1, ""


def _software_work_transition_alignment(
    state: AgentState,
    candidate: dict[str, object],
    *,
    objective: str,
) -> tuple[float, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, ""
    normalized_objective = str(objective).strip()
    if not normalized_objective:
        return 0.0, ""
    overview = state.software_work_stage_overview()
    objective_states = overview.get("objective_states", {})
    objective_states = objective_states if isinstance(objective_states, dict) else {}
    attempt_counts = overview.get("attempt_counts", {})
    attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}
    status = str(objective_states.get(normalized_objective, "pending")).strip() or "pending"
    attempts = max(0, int(attempt_counts.get(normalized_objective, 0) or 0))
    command = _canonicalize_command(str(candidate.get("command", candidate.get("content", ""))))
    failed_commands = state.all_failed_command_signatures()
    if status == "completed":
        return 0.0, status
    if status == "advanced":
        return 0.35, status
    if status == "pending":
        return 0.18, status
    if status == "stalled":
        if command in failed_commands:
            return -0.22 - (min(3, attempts) * 0.03), status
        return 0.05, status
    if status == "regressed":
        if command in failed_commands:
            return -0.30 - (min(3, attempts) * 0.04), status
        return 0.08, status
    return 0.0, status


def _software_work_phase_boundary_alignment(
    state: AgentState,
    candidate: dict[str, object],
    *,
    objective: str,
) -> tuple[float, str, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, "", ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, "", ""
    phase_state = state.software_work_phase_state()
    if not isinstance(phase_state, dict) or not phase_state:
        return 0.0, "", ""
    current_phase = str(phase_state.get("current_phase", "")).strip()
    suggested_phase = str(phase_state.get("suggested_phase", "")).strip()
    if not suggested_phase:
        return 0.0, "", ""
    candidate_phase = _software_work_candidate_phase(state, candidate, objective=objective)
    if not candidate_phase:
        return 0.0, "", suggested_phase
    if candidate_phase == suggested_phase:
        if bool(phase_state.get("handoff_ready", False)) and current_phase and suggested_phase != current_phase:
            return 0.34, candidate_phase, suggested_phase
        return 0.18, candidate_phase, suggested_phase
    if bool(phase_state.get("handoff_ready", False)) and current_phase and candidate_phase == current_phase:
        return -0.20, candidate_phase, suggested_phase
    if current_phase and _software_work_phase_rank(candidate_phase) > _software_work_phase_rank(suggested_phase):
        return -0.10, candidate_phase, suggested_phase
    if current_phase and not bool(phase_state.get("handoff_ready", False)):
        if _software_work_phase_rank(candidate_phase) > _software_work_phase_rank(current_phase):
            return -0.24, candidate_phase, suggested_phase
    return 0.0, candidate_phase, suggested_phase


def _software_work_phase_gate_alignment(
    state: AgentState,
    candidate: dict[str, object],
    *,
    objective: str,
) -> tuple[float, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, ""
    gate_state = state.software_work_phase_gate_state()
    if not isinstance(gate_state, dict) or not gate_state:
        return 0.0, ""
    gate_phase = str(gate_state.get("gate_phase", "")).strip()
    gate_objectives = [
        str(item).strip()
        for item in gate_state.get("gate_objectives", [])
        if str(item).strip()
    ]
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    for gate_objective in gate_objectives:
        if _candidate_matches_software_work_objective(command, gate_objective):
            return 0.42, gate_objective
    candidate_phase = _software_work_candidate_phase(state, candidate, objective=objective)
    if gate_phase and candidate_phase and _software_work_phase_rank(candidate_phase) > _software_work_phase_rank(gate_phase):
        return -0.28, candidate_phase
    return (-0.08, "") if gate_objectives else (0.0, "")


def _campaign_contract_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> tuple[float, int, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, -1, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, -1, ""
    contract = state.campaign_contract_state()
    if not isinstance(contract, dict) or not contract:
        return 0.0, -1, ""
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return 0.0, -1, ""
    regressed = {
        str(item).strip()
        for item in contract.get("regressed_objectives", [])
        if str(item).strip()
    }
    stalled = {
        str(item).strip()
        for item in contract.get("stalled_objectives", [])
        if str(item).strip()
    }
    for index, objective in enumerate(contract.get("anchor_objectives", [])[:5], start=1):
        normalized = str(objective).strip()
        if not normalized or not _candidate_matches_software_work_objective(command, normalized):
            continue
        scale = max(0.3, 1.0 - ((index - 1) * 0.14))
        if normalized in regressed:
            scale += 0.18
        elif normalized in stalled:
            scale += 0.08
        return min(1.4, scale), index, normalized
    required_paths = {
        str(item).strip().lower()
        for item in contract.get("required_paths", [])
        if str(item).strip()
    }
    if required_paths and any(path in command for path in required_paths):
        return 0.18, 6, next(iter(sorted(required_paths)))
    return 0.0, -1, ""


def _campaign_contract_drift_penalty(
    state: AgentState,
    candidate: dict[str, object],
) -> tuple[float, str]:
    if state.world_horizon() != "long_horizon":
        return 0.0, ""
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0, ""
    contract = state.campaign_contract_state()
    if not isinstance(contract, dict) or not contract:
        return 0.0, ""
    drift_pressure = max(0, int(contract.get("drift_pressure", 0) or 0))
    if drift_pressure <= 0:
        return 0.0, ""
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return 0.0, ""
    for objective in contract.get("anchor_objectives", [])[:5]:
        normalized = str(objective).strip()
        if normalized and _candidate_matches_software_work_objective(command, normalized):
            return 0.0, ""
    required_paths = {
        str(item).strip().lower()
        for item in contract.get("required_paths", [])
        if str(item).strip()
    }
    if required_paths and any(path in command for path in required_paths):
        return 0.0, ""
    if _is_read_only_discovery(command) or _is_verification_command(command):
        return 0.0, ""
    return min(1.0, 0.18 + (0.08 * drift_pressure)), "off_contract_mutation"


def _software_work_candidate_phase(
    state: AgentState,
    candidate: dict[str, object],
    *,
    objective: str,
) -> str:
    normalized_objective = str(objective).strip()
    if normalized_objective:
        return _software_work_objective_phase(
            normalized_objective,
            test_attempted=_software_work_test_attempted(state),
            regression_signaled=bool(state.latest_state_transition.get("regressed", False)),
        )
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return ""
    objectives = state.software_work_plan_update()
    for objective in objectives[:6]:
        if _candidate_matches_software_work_objective(command, objective):
            return _software_work_objective_phase(
                objective,
                test_attempted=_software_work_test_attempted(state),
                regression_signaled=bool(state.latest_state_transition.get("regressed", False)),
            )
    if any(token in command for token in ("pytest", "unittest", "nose", "tox", "smoke", "test")):
        return "test"
    if any(token in command for token in ("git merge", "git cherry-pick", "git rebase", "codegen", "generate")):
        return "migration"
    if any(token in command for token in ("report", "summary", "postmortem", "fix", "repair")):
        return "follow_up_fix"
    return "implementation"


def _software_work_test_attempted(state: AgentState) -> bool:
    overview = state.software_work_stage_overview()
    recent_outcomes = overview.get("recent_outcomes", [])
    if not isinstance(recent_outcomes, list):
        return False
    return any(
        _software_work_objective_phase(str(item.get("objective", "")).strip()) == "test"
        for item in recent_outcomes
        if isinstance(item, dict)
    )


def _trusted_retrieval_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> float:
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    trusted_commands = graph_summary.get("trusted_retrieval_command_counts", {})
    if not isinstance(trusted_commands, dict) or not trusted_commands:
        return 0.0
    command = _canonicalize_command(str(candidate.get("command", candidate.get("content", ""))))
    if not command:
        return 0.0
    command_tokens = _command_target_tokens(command)
    repair_tokens = _repair_surface_tokens(state)
    best = 0.0
    for prior_command, raw_count in trusted_commands.items():
        prior = _canonicalize_command(str(prior_command))
        count = _safe_int(raw_count)
        if not prior or count <= 0:
            continue
        if prior == command:
            best = max(best, min(1.6, 0.45 + (0.35 * count)))
            continue
        overlap = repair_tokens.intersection(command_tokens).intersection(_command_target_tokens(prior))
        if overlap:
            best = max(best, min(1.0, 0.2 + (0.15 * count) + (0.1 * len(overlap))))
    return best


def _trusted_retrieval_procedure_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> float:
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    procedures = graph_summary.get("trusted_retrieval_procedures", {})
    if not isinstance(procedures, list) or not procedures:
        return 0.0
    command = _canonicalize_command(str(candidate.get("command", candidate.get("content", ""))))
    if not command:
        return 0.0
    recent_commands = _recent_successful_command_suffix(state)
    if not recent_commands:
        return 0.0
    best = 0.0
    for item in procedures[:4]:
        if not isinstance(item, dict):
            continue
        commands = [_canonicalize_command(str(value)) for value in item.get("commands", []) if str(value).strip()]
        commands = [value for value in commands if value]
        if len(commands) < 2:
            continue
        prefix_len = _trusted_retrieval_procedure_prefix_match(recent_commands, commands)
        if prefix_len <= 0 or prefix_len >= len(commands):
            continue
        if commands[prefix_len] != command:
            continue
        count = _safe_int(item.get("count", 0))
        bonus = 0.24 + (0.14 * prefix_len) + (0.08 * min(3, count))
        if _is_verification_command(command):
            bonus += 0.08
        best = max(best, min(1.2, bonus))
    return best


def _graph_environment_alignment(
    state: AgentState,
    candidate: dict[str, object],
) -> float:
    if str(candidate.get("action", "code_execute")).strip() == "respond":
        return 0.0
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    universe_summary = state.universe_summary if isinstance(state.universe_summary, dict) else {}
    if not graph_summary or not universe_summary:
        return 0.0
    snapshot = universe_summary.get("environment_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    command = str(candidate.get("command", candidate.get("content", ""))).strip().lower()
    if not command:
        return 0.0
    novelty = _historical_environment_novelty(state)
    failures = _graph_environment_alignment_failures(graph_summary)
    read_only_discovery = _is_read_only_discovery(command)
    verification_aligned = _is_verification_command(command)
    git_mutation = _is_git_mutation_command(command)
    network_fetch = _is_network_fetch_command(command)
    workspace_scope_escape = _is_workspace_scope_escape_command(command)
    destructive_mutation = _is_destructive_command(command)

    score = 0.0
    if novelty > 0:
        if read_only_discovery:
            score += 0.55 * novelty
        if verification_aligned:
            score += 0.35 * novelty
        if git_mutation and _environment_conflicts(snapshot, graph_summary, "git_write_mode"):
            score -= 0.75 * novelty
        if network_fetch and _environment_conflicts(snapshot, graph_summary, "network_access_mode"):
            score -= 0.75 * novelty
        if (workspace_scope_escape or destructive_mutation) and _environment_conflicts(
            snapshot,
            graph_summary,
            "workspace_write_scope",
        ):
            score -= 0.85 * novelty

    if git_mutation:
        score -= min(0.9, 0.2 * failures.get("git_write_aligned", 0))
    if network_fetch:
        score -= min(0.9, 0.2 * failures.get("network_access_aligned", 0))
    if workspace_scope_escape or destructive_mutation:
        score -= min(1.0, 0.25 * failures.get("workspace_scope_aligned", 0))
    if failures and verification_aligned:
        score += 0.2
    return score


def _candidate_matches_software_work_objective(command: str, objective: str) -> bool:
    normalized = str(objective).strip().lower()
    if not normalized:
        return False
    if _candidate_matches_planner_recovery_objective(command, normalized):
        return True
    for prefix in (
        "apply planned edit ",
        "complete implementation for ",
        "revise implementation for ",
        "materialize expected artifact ",
        "remove forbidden artifact ",
        "preserve required artifact ",
    ):
        if normalized.startswith(prefix):
            target = normalized.removeprefix(prefix).strip()
            return bool(target) and target in command
    return False


def _canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())


def _repair_surface_tokens(state: AgentState) -> set[str]:
    tokens: set[str] = set()
    world = state.world_model_summary if isinstance(state.world_model_summary, dict) else {}
    diagnosis = state.active_subgoal_diagnosis()
    for value in (
        diagnosis.get("path", ""),
        *world.get("missing_expected_artifacts", []),
        *world.get("present_forbidden_artifacts", []),
        *world.get("unsatisfied_expected_contents", []),
        *state.task.expected_files,
        *state.task.forbidden_files,
        *state.task.expected_file_contents.keys(),
    ):
        tokens.update(_path_tokens(str(value)))
    return tokens


def _command_target_tokens(command: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_.\\-/]+", str(command).lower())
        if "/" in token or "." in token
    }


def _path_tokens(path: str) -> set[str]:
    normalized = str(path).strip().lower()
    if not normalized:
        return set()
    parts = [part for part in re.split(r"[\\/]", normalized) if part and part not in {".", ".."}]
    tokens = set(parts)
    tokens.add(normalized)
    if parts:
        tokens.add(parts[-1])
    return tokens


def _historical_environment_novelty(state: AgentState) -> int:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    universe_summary = state.universe_summary if isinstance(state.universe_summary, dict) else {}
    snapshot = universe_summary.get("environment_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    novelty = 0
    for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
        current = str(snapshot.get(field, "")).strip().lower()
        dominant = _dominant_graph_environment_mode(graph_summary, field)
        if current and dominant and current != dominant:
            novelty += 1
    return novelty


def _recent_successful_command_suffix(state: AgentState) -> list[str]:
    commands: list[str] = []
    for step in state.history[-6:]:
        if str(getattr(step, "action", "")).strip() != "code_execute":
            continue
        command = _canonicalize_command(str(getattr(step, "content", "")).strip())
        if not command:
            continue
        command_result = getattr(step, "command_result", {})
        if isinstance(command_result, dict) and not bool(command_result.get("timed_out", False)):
            try:
                if int(command_result.get("exit_code", 1)) == 0:
                    commands.append(command)
                    continue
            except (TypeError, ValueError):
                pass
        verification = getattr(step, "verification", {})
        if isinstance(verification, dict) and bool(verification.get("passed", False)):
            commands.append(command)
    return commands


def _trusted_retrieval_procedure_prefix_match(
    recent_commands: list[str],
    procedure_commands: list[str],
) -> int:
    if len(procedure_commands) < 2:
        return 0
    max_prefix = min(len(recent_commands), len(procedure_commands) - 1)
    for prefix_len in range(max_prefix, 0, -1):
        if recent_commands[-prefix_len:] == procedure_commands[:prefix_len]:
            return prefix_len
    return 0


def _graph_environment_alignment_failures(graph_summary: dict[str, object]) -> dict[str, int]:
    failures = graph_summary.get("environment_alignment_failures", {})
    if not isinstance(failures, dict):
        return {}
    return {
        str(key).strip(): value
        for key, value in (
            (str(key).strip(), _safe_int(raw_value))
            for key, raw_value in failures.items()
        )
        if key and value > 0
    }


def _dominant_graph_environment_mode(graph_summary: dict[str, object], field: str) -> str:
    observed_modes = graph_summary.get("observed_environment_modes", {})
    if not isinstance(observed_modes, dict):
        return ""
    values = observed_modes.get(field, {})
    if not isinstance(values, dict):
        return ""
    ranked = sorted(
        (
            (str(mode).strip().lower(), _safe_int(count))
            for mode, count in values.items()
            if str(mode).strip()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0] if ranked and ranked[0][1] > 0 else ""


def _environment_conflicts(
    snapshot: dict[str, object],
    graph_summary: dict[str, object],
    field: str,
) -> bool:
    current = str(snapshot.get(field, "")).strip().lower()
    dominant = _dominant_graph_environment_mode(graph_summary, field)
    return bool(current and dominant and current != dominant)


def _is_read_only_discovery(command: str) -> bool:
    return any(
        command.startswith(prefix)
        for prefix in (
            "ls",
            "find ",
            "rg ",
            "grep ",
            "cat ",
            "sed -n",
            "git status",
            "git diff",
            "git log",
            "git rev-parse",
            "pwd",
            "test ",
        )
    )


def _is_verification_command(command: str) -> bool:
    return any(token in command for token in ("pytest", " test ", "grep -q", "diff ", "git diff", "python -m pytest"))


def _is_git_mutation_command(command: str) -> bool:
    return any(
        command.startswith(prefix)
        for prefix in (
            "git commit",
            "git merge",
            "git rebase",
            "git cherry-pick",
            "git push",
            "git pull",
            "git reset",
            "git checkout ",
            "git switch ",
            "git branch ",
        )
    )


def _is_network_fetch_command(command: str) -> bool:
    return any(token in command for token in ("curl ", "wget ", "pip install", "npm install", "uv pip install"))


def _is_workspace_scope_escape_command(command: str) -> bool:
    return "../" in command or " /tmp/" in command or " cd .." in command


def _is_destructive_command(command: str) -> bool:
    return "rm -rf" in command or "git reset --hard" in command


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_matches_planner_recovery_objective(command: str, objective: str) -> bool:
    normalized_command = str(command).strip().lower()
    normalized_objective = str(objective).strip().lower()
    if not normalized_command or not normalized_objective:
        return False
    for prefix in (
        "update workflow path ",
        "regenerate generated artifact ",
        "write workflow report ",
        "accept required branch ",
        "prepare workflow branch ",
        "run workflow test ",
    ):
        if normalized_objective.startswith(prefix):
            target = normalized_objective.removeprefix(prefix).strip()
            if not target:
                return False
            if prefix == "accept required branch ":
                return "git merge" in normalized_command and target in normalized_command
            if prefix == "prepare workflow branch ":
                return target in normalized_command and any(
                    token in normalized_command for token in ("git checkout", "git switch", "git branch")
                )
            if prefix == "run workflow test ":
                objective_tokens = {token for token in target.split() if len(token) > 2}
                return (
                    ("test" in normalized_command or "pytest" in normalized_command)
                    and bool(objective_tokens.intersection(set(normalized_command.split())))
                )
            return target in normalized_command
    return any(token in normalized_command for token in normalized_objective.split() if len(token) > 3)


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
        horizon_hint=_state_horizon_hint(state),
    )
    log_prior = torch.tensor(prior.log_prior, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
    payload["log_belief"] = log_prior
    payload["diagnostics"] = {
        "backend": prior.backend,
        "matched_state_index": prior.matched_state_index,
        "matched_state_probability": prior.matched_state_probability,
        "signature_norm": prior.signature_norm,
        "token_count": prior.token_count,
        "horizon_hint": prior.horizon_hint,
        "applied_bias_strength": prior.applied_bias_strength,
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
    benchmark_family = str(state.task.metadata.get("benchmark_family", "")).strip()
    if benchmark_family:
        fragments.append(f"benchmark_family:{benchmark_family}")
    horizon_hint = _state_horizon_hint(state)
    if horizon_hint:
        fragments.append(f"horizon:{horizon_hint}")
    fragments.extend(str(command) for command in state.task.suggested_commands[:4])
    fragments.extend(str(goal) for goal in state.plan[:4])
    fragments.extend(str(goal) for goal in state.initial_plan[:4])
    fragments.extend(str(item) for item in state.world_model_summary.get("missing_expected_artifacts", [])[:6])
    fragments.extend(str(item) for item in state.world_model_summary.get("present_forbidden_artifacts", [])[:6])
    fragments.extend(str(item) for item in state.world_model_summary.get("preserved_artifacts", [])[:6])
    fragments.extend(str(item) for item in state.latest_state_transition.get("regressions", [])[:4])
    if str(state.active_subgoal).strip():
        fragments.append(str(state.active_subgoal))
    fragments.extend(str(step.content) for step in state.history[-4:] if str(step.content).strip())
    return [fragment.strip() for fragment in fragments if str(fragment).strip()]


def _state_horizon_hint(state: AgentState | None) -> str:
    if state is None:
        return ""
    horizon = str(state.world_model_summary.get("horizon", "")).strip()
    if horizon:
        return horizon
    return str(state.task.metadata.get("difficulty", state.task.metadata.get("task_difficulty", ""))).strip()


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
