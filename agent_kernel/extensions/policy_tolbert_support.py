from __future__ import annotations

from pathlib import Path
from typing import Any

from ..actions import CODE_EXECUTE
from ..extensions.policy_command_utils import (
    canonicalize_command as _canonicalize_command,
    normalize_command_for_workspace as _normalize_command_for_workspace,
)
from ..extensions.policy_runtime_support import SkillLibrary
from ..extensions.runtime_modeling_adapter import (
    decode_action_generation_candidates,
    decode_bounded_action_candidates,
    generate_hybrid_decoder_text,
)
from ..llm import coerce_decoder_text_decision
from ..modeling.policy.decoder import DecodedActionCandidate
from ..modeling.world.rollout import rollout_action_value
from ..schemas import ActionDecision
from ..state import AgentState


def tolbert_direct_decision(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    recommended_commands: list[dict[str, str]],
    blocked_commands: list[str],
    route_mode: str,
) -> ActionDecision | None:
    if route_mode == "shadow" or state.context_packet is None or not policy._tolbert_direct_command_enabled():
        return None
    trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False))
    confidence = float(state.context_packet.control.get("path_confidence", 0.0))
    normalized_blocked = {SkillLibrary._normalize_command(command) for command in blocked_commands}
    ranked_commands = policy._rank_direct_retrieval_commands(
        state,
        recommended_commands,
        blocked_commands=normalized_blocked,
    )
    state.retrieval_direct_candidates = ranked_commands
    for entry in ranked_commands:
        command = str(entry.get("command", ""))
        if policy._can_execute_direct_retrieval_command(
            state,
            top_skill=top_skill,
            command=command,
            span_id=str(entry.get("span_id", "")).strip(),
            control_score=int(entry.get("control_score", 0)),
            confidence=confidence,
            trust_retrieval=trust_retrieval,
        ):
            return ActionDecision(
                thought="Use Tolbert retrieval-guided command.",
                action=CODE_EXECUTE,
                content=_normalize_command_for_workspace(command, state.task.workspace_subdir),
                done=False,
                selected_retrieval_span_id=str(entry.get("span_id", "")).strip() or None,
                retrieval_influenced=True,
                decision_source="tolbert_direct",
            )
    if (
        top_skill is not None
        and policy._tolbert_skill_ranking_enabled()
        and confidence >= policy._retrieval_float("tolbert_deterministic_command_confidence")
    ):
        matched_span_id = policy._matching_retrieval_span_id(
            policy.skill_library._commands_for_skill(top_skill)[0],
            recommended_commands,
        )
        return policy._skill_action_decision(
            state,
            top_skill,
            thought_prefix="Use Tolbert-validated skill",
            retrieval_influenced=True,
            retrieval_ranked_skill=True,
            selected_retrieval_span_id=matched_span_id,
        )
    return None


def screened_tolbert_retrieval_guidance(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
) -> dict[str, list[str]]:
    spans = retrieval_guidance.get("recommended_command_spans", [])
    if state.context_packet is None or not isinstance(spans, list) or not spans:
        return retrieval_guidance
    trust_retrieval = bool(state.context_packet.control.get("trust_retrieval", False))
    confidence = float(state.context_packet.control.get("path_confidence", 0.0))
    normalized_blocked = {SkillLibrary._normalize_command(command) for command in blocked_commands}
    filtered_spans: list[dict[str, str]] = []
    filtered_commands: list[str] = []
    seen: set[str] = set()
    for entry in spans:
        command = str(entry.get("command", "")).strip()
        normalized = SkillLibrary._normalize_command(command)
        if not normalized or normalized in normalized_blocked or normalized in seen:
            continue
        control_score = policy._command_control_score(state, command)
        if not policy._can_execute_direct_retrieval_command(
            state,
            top_skill=top_skill,
            command=command,
            span_id=str(entry.get("span_id", "")).strip(),
            control_score=control_score,
            confidence=confidence,
            trust_retrieval=trust_retrieval,
        ):
            continue
        seen.add(normalized)
        filtered_spans.append(
            {
                "span_id": str(entry.get("span_id", "")).strip(),
                "command": command,
            }
        )
        filtered_commands.append(command)
    screened = dict(retrieval_guidance)
    screened["recommended_commands"] = filtered_commands
    screened["recommended_command_spans"] = filtered_spans
    return screened


def rank_direct_retrieval_commands(
    policy: Any,
    state: AgentState,
    recommended_commands: list[dict[str, str]],
    *,
    blocked_commands: set[str],
) -> list[dict[str, str]]:
    verifier = state.task
    ranked: list[tuple[int, dict[str, str]]] = []
    for entry in recommended_commands:
        command = str(entry.get("command", "")).strip()
        normalized = SkillLibrary._normalize_command(command)
        if not normalized or normalized in blocked_commands:
            continue
        score = 0
        if any(path and path in command for path in verifier.expected_files):
            score += 4
        if any(path and path in command for path in verifier.expected_file_contents):
            score += 4
        if any(path and path in command for path in verifier.forbidden_files):
            score -= 6
        if verifier.workspace_subdir and f"{verifier.workspace_subdir}/" in command:
            score -= 3
        if "printf " in command:
            score += 1
        control_score = policy._command_control_score(state, command)
        score += control_score
        score += policy._trusted_retrieval_carryover_match_bonus(state, command)
        ranked.append(
            (
                score,
                {
                    "span_id": str(entry.get("span_id", "")),
                    "command": command,
                    "control_score": control_score,
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]["span_id"], item[1]["command"]))
    return [entry for _, entry in ranked]


def can_execute_direct_retrieval_command(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    command: str,
    span_id: str,
    control_score: int,
    confidence: float,
    trust_retrieval: bool,
) -> bool:
    if control_score < policy._retrieval_float("tolbert_direct_command_min_score"):
        return False
    if state.history:
        return trust_retrieval or confidence >= policy._retrieval_float("tolbert_deterministic_command_confidence")
    if not policy._first_step_command_covers_required_artifacts(state, command):
        return False
    if not trust_retrieval:
        return False
    if (
        confidence < max(
            policy._retrieval_float("tolbert_deterministic_command_confidence"),
            policy._retrieval_float("tolbert_first_step_direct_command_confidence"),
            policy._retrieval_float("tolbert_first_step_direct_command_confidence")
            + policy._policy_control_float("direct_command_confidence_boost"),
        )
        and not policy._trusted_workflow_guidance_bypasses_first_step_confidence(
            state,
            command=command,
            span_id=span_id,
        )
    ):
        return False
    if top_skill is not None:
        skill_commands = policy.skill_library._commands_for_skill(top_skill)
        if skill_commands:
            skill_score = policy._command_control_score(state, skill_commands[0])
            required_artifact_bias = policy._policy_control_int("required_artifact_first_step_bias")
            if skill_score > control_score + max(0, required_artifact_bias):
                return False
    return True


def _resolved_decoder_bundle_manifest_path(
    policy: Any,
    runtime: dict[str, object],
) -> Path:
    manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
    if not manifest_raw:
        policy.runtime_support.last_hybrid_runtime_error = "missing bundle_manifest_path"
        raise RuntimeError("retained decoder runtime is enabled but bundle_manifest_path is missing")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_absolute():
        manifest_path = policy.runtime_support.repo_root / manifest_path
    if not manifest_path.exists():
        policy.runtime_support.last_hybrid_runtime_error = f"bundle manifest does not exist: {manifest_path}"
        raise RuntimeError(f"retained decoder bundle manifest does not exist: {manifest_path}")
    return manifest_path


def _decoder_runtime_ready_for_primary(runtime: dict[str, object]) -> bool:
    runtime_key = str(runtime.get("runtime_key", "hybrid_runtime")).strip() or "hybrid_runtime"
    if runtime_key == "universal_decoder_runtime":
        return bool(runtime.get("materialized", False))
    return bool(runtime.get("primary_enabled", False))


def _decoder_generation_source(runtime: dict[str, object]) -> str:
    runtime_key = str(runtime.get("runtime_key", "hybrid_runtime")).strip() or "hybrid_runtime"
    if runtime_key == "universal_decoder_runtime":
        return "universal_decoder_generation"
    return "hybrid_decoder_generation"


def _hybrid_generated_candidate(
    policy: Any,
    state: AgentState,
    *,
    runtime_policy: dict[str, object],
    decoder_runtime: dict[str, object],
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    existing_candidates: list[DecodedActionCandidate],
) -> DecodedActionCandidate | None:
    if not _decoder_runtime_ready_for_primary(decoder_runtime):
        return None
    if not bool(decoder_runtime.get("supports_decoder_surface", False)):
        return None
    try:
        manifest_path = _resolved_decoder_bundle_manifest_path(policy, decoder_runtime)
        policy.runtime_support.last_hybrid_runtime_error = ""
        generated = generate_hybrid_decoder_text(
            state=state,
            bundle_manifest_path=manifest_path,
            device=str(decoder_runtime.get("preferred_device", "cpu")).strip() or "cpu",
            max_new_tokens=64,
        )
    except Exception as exc:
        policy.runtime_support.last_hybrid_runtime_error = str(exc).strip() or exc.__class__.__name__
        return None
    generation_source = _decoder_generation_source(decoder_runtime)
    runtime_key = str(decoder_runtime.get("runtime_key", "hybrid_runtime")).strip() or "hybrid_runtime"
    decision = coerce_decoder_text_decision(
        str(generated.get("generated_text", "")),
        default_command_thought="Execute the retained Tolbert decoder generation.",
        default_response_thought="Stop because the retained Tolbert decoder emitted a terminal response.",
    )
    if decision is None:
        return None
    action = str(decision.get("action", CODE_EXECUTE)).strip() or CODE_EXECUTE
    content = str(decision.get("content", "")).strip()
    if action == CODE_EXECUTE:
        content = _normalize_command_for_workspace(content, state.task.workspace_subdir)
        normalized = _canonicalize_command(content)
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        if not normalized or normalized in blocked:
            return None
        existing_commands = {
            _canonicalize_command(candidate.content)
            for candidate in existing_candidates
            if candidate.action == CODE_EXECUTE
        }
        proposal_policy = policy._tolbert_action_generation_policy()
        score = float(policy._command_control_score(state, content))
        score += rollout_action_value(
            world_model_summary=state.world_model_summary,
            latent_state_summary=state.latent_state_summary,
            latest_transition=state.latest_state_transition,
            action=CODE_EXECUTE,
            content=content,
            rollout_policy=policy._tolbert_rollout_policy(),
            world_model=policy.world_model,
        )
        score += float(proposal_policy.get("proposal_score_bias", 0.0) or 0.0)
        proposal_novel = normalized not in existing_commands
        if proposal_novel:
            score += float(proposal_policy.get("novel_command_bonus", 0.0) or 0.0)
        matched_span_id = policy._matching_retrieval_span_id(
            content,
            retrieval_guidance.get("recommended_command_spans", []),
        )
        return DecodedActionCandidate(
            action=CODE_EXECUTE,
            content=content,
            thought=str(decision.get("thought", "")).strip() or "Execute the retained Tolbert decoder generation.",
            score=score,
            reason=generation_source,
            selected_skill_id=None,
            selected_retrieval_span_id=matched_span_id,
            retrieval_influenced=matched_span_id is not None,
            retrieval_ranked_skill=False,
            proposal_source=generation_source,
            proposal_novel=proposal_novel,
            proposal_metadata={
                "decoder_model_family": str(generated.get("model_family", "")).strip(),
                "decoder_generated_text": str(generated.get("generated_text", "")).strip(),
                "decoder_avg_logprob": float(generated.get("avg_logprob", 0.0) or 0.0),
                "decoder_bundle_manifest_path": str(manifest_path),
                "decoder_runtime_key": runtime_key,
                "decoder_training_objective": str(decoder_runtime.get("training_objective", "")).strip(),
            },
        )
    score = rollout_action_value(
        world_model_summary=state.world_model_summary,
        latent_state_summary=state.latent_state_summary,
        latest_transition=state.latest_state_transition,
        action="respond",
        content=content,
        rollout_policy=policy._tolbert_rollout_policy(),
        world_model=policy.world_model,
    )
    score += max(0.0, float(runtime_policy.get("primary_min_command_score", 2) or 2))
    return DecodedActionCandidate(
        action="respond",
        content=content,
        thought=str(decision.get("thought", "")).strip() or "Stop because the retained Tolbert decoder emitted a terminal response.",
        score=score,
        reason=generation_source,
        proposal_source=generation_source,
        proposal_novel=False,
        proposal_metadata={
            "decoder_model_family": str(generated.get("model_family", "")).strip(),
            "decoder_generated_text": str(generated.get("generated_text", "")).strip(),
            "decoder_avg_logprob": float(generated.get("avg_logprob", 0.0) or 0.0),
            "decoder_bundle_manifest_path": str(manifest_path),
            "decoder_runtime_key": runtime_key,
            "decoder_training_objective": str(decoder_runtime.get("training_objective", "")).strip(),
        },
    )


def tolbert_primary_decision(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
) -> ActionDecision | None:
    runtime_policy = policy._tolbert_runtime_policy()
    screened_retrieval_guidance = policy._screened_tolbert_retrieval_guidance(
        state,
        top_skill=top_skill,
        retrieval_guidance=retrieval_guidance,
        blocked_commands=blocked_commands,
    )
    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=top_skill,
        retrieval_guidance=screened_retrieval_guidance,
        blocked_commands=blocked_commands,
        proposal_policy=policy._tolbert_action_generation_policy(),
        rollout_policy=policy._tolbert_rollout_policy(),
        command_score_fn=policy._command_control_score,
        normalize_command_fn=_normalize_command_for_workspace,
        canonicalize_command_fn=_canonicalize_command,
    )
    candidates.extend(
        decode_bounded_action_candidates(
            state=state,
            world_model=policy.world_model,
            skill_library=policy.skill_library,
            top_skill=top_skill,
            retrieval_guidance=screened_retrieval_guidance,
            blocked_commands=blocked_commands,
            decoder_policy=policy._tolbert_decoder_policy(),
            rollout_policy=policy._tolbert_rollout_policy(),
            command_score_fn=policy._command_control_score,
            match_span_fn=policy._matching_retrieval_span_id,
            normalize_command_fn=_normalize_command_for_workspace,
            canonicalize_command_fn=_canonicalize_command,
    )
    )
    hybrid_runtime = policy._tolbert_hybrid_runtime()
    decoder_runtime = policy._tolbert_active_decoder_runtime()
    generated_candidate = _hybrid_generated_candidate(
        policy,
        state,
        runtime_policy=runtime_policy,
        decoder_runtime=decoder_runtime,
        top_skill=top_skill,
        retrieval_guidance=screened_retrieval_guidance,
        blocked_commands=blocked_commands,
        existing_candidates=candidates,
    )
    if generated_candidate is not None:
        candidates.append(generated_candidate)
    candidates = sorted(candidates, key=lambda item: (-item.score, item.action, item.content))
    if not candidates:
        return None
    require_trusted_retrieval = bool(runtime_policy.get("require_trusted_retrieval", True))
    allow_direct_primary = bool(runtime_policy.get("allow_direct_command_primary", True))
    allow_skill_primary = bool(runtime_policy.get("allow_skill_primary", True))
    best = candidates[0]
    raw_best = best
    if bool(hybrid_runtime.get("primary_enabled", False)):
        candidate_lookup = {
            f"{candidate.action}:{candidate.content}": candidate
            for candidate in candidates
        }
        scored = policy._hybrid_scored_candidates(
            state,
            [
                {
                    "action": candidate.action,
                    "command": candidate.content,
                    "content": candidate.content,
                    "score": float(candidate.score),
                    "reason": candidate.reason,
                    "selected_skill_id": candidate.selected_skill_id,
                    "span_id": candidate.selected_retrieval_span_id,
                    "retrieval_influenced": candidate.retrieval_influenced,
                    "retrieval_ranked_skill": candidate.retrieval_ranked_skill,
                    "proposal_source": candidate.proposal_source,
                    "proposal_novel": candidate.proposal_novel,
                }
                for candidate in candidates
            ],
        )
        if scored:
            selected_key = (
                f"{str(scored[0].get('action', 'code_execute')).strip()}:"
                f"{str(scored[0].get('content', scored[0].get('command', ''))).strip()}"
            )
            best = candidate_lookup.get(selected_key, best)
            best.proposal_metadata = {
                **dict(best.proposal_metadata or {}),
                **hybrid_candidate_metadata(scored[0]),
            }
    if require_trusted_retrieval and not policy._trust_retrieval(state) and (allow_direct_primary or allow_skill_primary):
        retrieval_guided_best = next(
            (
                candidate
                for candidate in candidates
                if bool(candidate.retrieval_influenced) or bool(candidate.retrieval_ranked_skill)
            ),
            None,
        )
        if retrieval_guided_best is not None and not (
            bool(best.retrieval_influenced) or bool(best.retrieval_ranked_skill)
        ):
            best = retrieval_guided_best
    if (
        raw_best.proposal_source == "expected_file_content"
        and bool(dict(raw_best.proposal_metadata or {}).get("fallback_from_workspace_preview", False))
        and not bool(dict(raw_best.proposal_metadata or {}).get("workspace_preview_has_complete_current_proof", False))
        and not bool(dict(raw_best.proposal_metadata or {}).get("workspace_preview_has_explicit_hidden_gap_current_proof", False))
        and not bool(dict(raw_best.proposal_metadata or {}).get("workspace_preview_has_bridged_hidden_gap_region_block", False))
        and bool(best.proposal_source)
        and best.proposal_source.startswith("structured_edit:")
    ):
        best = raw_best
    if best.action == CODE_EXECUTE and float(best.score) < float(runtime_policy.get("primary_min_command_score", 2)):
        return None
    return ActionDecision(
        thought=best.thought,
        action=best.action,
        content=best.content,
        done=best.action != CODE_EXECUTE,
        selected_skill_id=best.selected_skill_id,
        selected_retrieval_span_id=best.selected_retrieval_span_id,
        retrieval_influenced=best.retrieval_influenced,
        retrieval_ranked_skill=best.retrieval_ranked_skill,
        proposal_source=best.proposal_source,
        proposal_novel=best.proposal_novel,
        proposal_metadata=dict(best.proposal_metadata or {}),
        decision_source=(
            "tolbert_retained_proposal_decoder"
            if best.proposal_source
            and str(decoder_runtime.get("runtime_key", "hybrid_runtime")).strip() == "universal_decoder_runtime"
            else
            "tolbert_hybrid_proposal_decoder"
            if best.proposal_source and bool(hybrid_runtime.get("primary_enabled", False))
            else "tolbert_proposal_decoder"
            if best.proposal_source
            else "tolbert_retained_decoder"
            if str(decoder_runtime.get("runtime_key", "hybrid_runtime")).strip() == "universal_decoder_runtime"
            else "tolbert_hybrid_decoder"
            if bool(hybrid_runtime.get("primary_enabled", False))
            else "tolbert_decoder"
        ),
        tolbert_route_mode="primary",
    )


def tolbert_shadow_decision(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    route_mode: str,
) -> dict[str, object]:
    if route_mode not in {"shadow", "primary"}:
        return {}
    screened_retrieval_guidance = policy._screened_tolbert_retrieval_guidance(
        state,
        top_skill=top_skill,
        retrieval_guidance=retrieval_guidance,
        blocked_commands=blocked_commands,
    )
    candidates = policy._tolbert_ranked_candidates(
        state,
        top_skill=top_skill,
        retrieval_guidance=screened_retrieval_guidance,
        blocked_commands=blocked_commands,
    )
    if not candidates:
        return {}
    hybrid_runtime = policy._tolbert_hybrid_runtime()
    if bool(hybrid_runtime.get("shadow_enabled", False)):
        try:
            scored = policy._hybrid_scored_candidates(state, candidates)
        except RuntimeError as exc:
            return {
                "mode": route_mode,
                "command": "",
                "score": 0.0,
                "reason": f"hybrid_runtime_error:{str(exc)}",
                "span_id": "",
                "model_family": str(hybrid_runtime.get("model_family", "")).strip(),
            }
        if scored:
            best = scored[0]
            return {
                "mode": route_mode,
                "command": str(best.get("command", "")),
                "score": float(best.get("hybrid_total_score", best.get("score", 0.0))),
                "reason": f"hybrid_runtime:{str(best.get('reason', '')).strip() or 'candidate'}",
                "span_id": str(best.get("span_id", "")).strip(),
                "model_family": str(best.get("hybrid_model_family", "")).strip(),
                "hybrid_metadata": hybrid_candidate_metadata(best),
            }
    best = candidates[0]
    return {
        "mode": route_mode,
        "command": str(best["command"]),
        "score": int(best["score"]),
        "reason": str(best["reason"]),
        "span_id": str(best.get("span_id", "")).strip(),
    }


def hybrid_candidate_metadata(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "hybrid_total_score": float(candidate.get("hybrid_total_score", candidate.get("score", 0.0)) or 0.0),
        "hybrid_learned_score": float(candidate.get("hybrid_learned_score", 0.0) or 0.0),
        "hybrid_policy_score": float(candidate.get("hybrid_policy_score", 0.0) or 0.0),
        "hybrid_value_score": float(candidate.get("hybrid_value_score", 0.0) or 0.0),
        "hybrid_risk_score": float(candidate.get("hybrid_risk_score", 0.0) or 0.0),
        "hybrid_stop_score": float(candidate.get("hybrid_stop_score", 0.0) or 0.0),
        "hybrid_transition_progress": float(candidate.get("hybrid_transition_progress", 0.0) or 0.0),
        "hybrid_transition_regression": float(candidate.get("hybrid_transition_regression", 0.0) or 0.0),
        "hybrid_world_progress_score": float(candidate.get("hybrid_world_progress_score", 0.0) or 0.0),
        "hybrid_world_risk_score": float(candidate.get("hybrid_world_risk_score", 0.0) or 0.0),
        "hybrid_world_belief_vector": [
            float(value)
            for value in candidate.get("hybrid_world_belief_vector", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_world_belief_top_states": [
            int(value)
            for value in candidate.get("hybrid_world_belief_top_states", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_world_belief_top_state_probs": [
            float(value)
            for value in candidate.get("hybrid_world_belief_top_state_probs", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_decoder_world_progress_score": float(
            candidate.get("hybrid_decoder_world_progress_score", 0.0) or 0.0
        ),
        "hybrid_decoder_world_risk_score": float(
            candidate.get("hybrid_decoder_world_risk_score", 0.0) or 0.0
        ),
        "hybrid_decoder_world_entropy_mean": float(
            candidate.get("hybrid_decoder_world_entropy_mean", 0.0) or 0.0
        ),
        "hybrid_decoder_world_belief_vector": [
            float(value)
            for value in candidate.get("hybrid_decoder_world_belief_vector", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_decoder_world_belief_top_states": [
            int(value)
            for value in candidate.get("hybrid_decoder_world_belief_top_states", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_decoder_world_belief_top_state_probs": [
            float(value)
            for value in candidate.get("hybrid_decoder_world_belief_top_state_probs", [])
            if isinstance(value, (int, float))
        ],
        "hybrid_world_prior_backend": str(candidate.get("hybrid_world_prior_backend", "")).strip(),
        "hybrid_world_prior_top_state": int(candidate.get("hybrid_world_prior_top_state", -1) or -1),
        "hybrid_world_prior_top_probability": float(
            candidate.get("hybrid_world_prior_top_probability", 0.0) or 0.0
        ),
        "hybrid_world_transition_family": str(candidate.get("hybrid_world_transition_family", "")).strip(),
        "hybrid_world_transition_bandwidth": int(candidate.get("hybrid_world_transition_bandwidth", 0) or 0),
        "hybrid_world_transition_gate": float(candidate.get("hybrid_world_transition_gate", 0.0) or 0.0),
        "hybrid_world_final_entropy_mean": float(candidate.get("hybrid_world_final_entropy_mean", 0.0) or 0.0),
        "hybrid_recovery_stage_alignment": float(candidate.get("hybrid_recovery_stage_alignment", 0.0) or 0.0),
        "hybrid_recovery_stage_rank": int(candidate.get("hybrid_recovery_stage_rank", -1) or -1),
        "hybrid_recovery_stage_objective": str(candidate.get("hybrid_recovery_stage_objective", "")).strip(),
        "hybrid_ssm_last_state_norm_mean": float(candidate.get("hybrid_ssm_last_state_norm_mean", 0.0) or 0.0),
        "hybrid_ssm_pooled_state_norm_mean": float(candidate.get("hybrid_ssm_pooled_state_norm_mean", 0.0) or 0.0),
        "hybrid_model_family": str(candidate.get("hybrid_model_family", "")).strip(),
        "hybrid_reason": str(candidate.get("reason", "")).strip(),
    }


def tolbert_ranked_candidates(
    policy: Any,
    state: AgentState,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
) -> list[dict[str, object]]:
    runtime_policy = policy._tolbert_runtime_policy()
    allow_direct = bool(runtime_policy.get("allow_direct_command_primary", True))
    allow_skill = bool(runtime_policy.get("allow_skill_primary", True))
    blocked = {_canonicalize_command(command) for command in blocked_commands}
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    if allow_direct:
        for item in retrieval_guidance.get("recommended_command_spans", []):
            command = _normalize_command_for_workspace(str(item.get("command", "")), state.task.workspace_subdir)
            normalized = _canonicalize_command(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "command": command,
                    "reason": "retrieval guidance",
                    "span_id": str(item.get("span_id", "")).strip(),
                    "retrieval_influenced": True,
                    "retrieval_ranked_skill": False,
                }
            )
    if allow_skill and top_skill is not None:
        for command in policy.skill_library._commands_for_skill(top_skill)[:1]:
            command = _normalize_command_for_workspace(command, state.task.workspace_subdir)
            normalized = _canonicalize_command(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "command": command,
                    "reason": "retained skill",
                    "span_id": policy._matching_retrieval_span_id(
                        command,
                        retrieval_guidance.get("recommended_command_spans", []),
                    ),
                    "retrieval_influenced": False,
                    "retrieval_ranked_skill": True,
                }
            )
    for item in policy._trusted_retrieval_carryover_candidates(state, blocked_commands=blocked):
        command = str(item.get("command", "")).strip()
        normalized = _canonicalize_command(command)
        if not normalized or normalized in seen or normalized in blocked:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "command": command,
                "reason": "trusted retrieval carryover",
                "span_id": str(item.get("span_id", "")).strip(),
                "retrieval_influenced": True,
                "retrieval_ranked_skill": False,
            }
        )
    for command in state.task.suggested_commands[:3]:
        command = _normalize_command_for_workspace(command, state.task.workspace_subdir)
        normalized = _canonicalize_command(command)
        if not normalized or normalized in seen or normalized in blocked:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "command": command,
                "reason": "task suggestion",
                "span_id": policy._matching_retrieval_span_id(
                    command,
                    retrieval_guidance.get("recommended_command_spans", []),
                ),
                "retrieval_influenced": False,
                "retrieval_ranked_skill": False,
            }
        )
    for candidate in candidates:
        command = str(candidate["command"])
        candidate["score"] = (
            policy._command_control_score(state, command)
            + policy._trusted_retrieval_carryover_match_bonus(state, command)
        )
        reason = str(candidate.get("reason", "")).strip()
        if reason == "trusted retrieval carryover":
            candidate["priority_rank"] = 0
        elif bool(candidate.get("retrieval_influenced", False)):
            candidate["priority_rank"] = 1
        elif bool(candidate.get("retrieval_ranked_skill", False)) or str(candidate.get("span_id", "")).strip():
            candidate["priority_rank"] = 2
        else:
            candidate["priority_rank"] = 3
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("priority_rank", 99)),
            -int(item["score"]),
            str(item["command"]),
        ),
    )
