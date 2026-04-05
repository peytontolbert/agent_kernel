from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_kernel.world_model import WorldModel


def rollout_action_value(
    *,
    world_model_summary: dict[str, object],
    latent_state_summary: dict[str, object],
    latest_transition: dict[str, object],
    action: str,
    content: str,
    rollout_policy: dict[str, object],
    world_model: "WorldModel",
) -> float:
    if action == "respond":
        return _stop_value(
            world_model_summary=world_model_summary,
            latent_state_summary=latent_state_summary,
            rollout_policy=rollout_policy,
        )
    effect = world_model.simulate_command_effect(world_model_summary, content)
    progress_weight = _float_value(rollout_policy.get("predicted_progress_gain_weight"), 3.0)
    conflict_penalty = _float_value(rollout_policy.get("predicted_conflict_penalty_weight"), 4.0)
    preserved_bonus = _float_value(rollout_policy.get("predicted_preserved_bonus_weight"), 1.0)
    workflow_bonus = _float_value(rollout_policy.get("predicted_workflow_bonus_weight"), 1.5)
    latent_bonus = _float_value(rollout_policy.get("latent_progress_bonus_weight"), 1.0)
    latent_risk_penalty = _float_value(rollout_policy.get("latent_risk_penalty_weight"), 2.0)
    learned_progress_bonus = _float_value(rollout_policy.get("learned_world_progress_bonus_weight"), 1.25)
    learned_recovery_bonus = _float_value(rollout_policy.get("learned_world_recovery_bonus_weight"), 1.5)
    learned_continue_penalty = _float_value(rollout_policy.get("learned_world_continue_penalty_weight"), 1.25)
    score = 0.0
    predicted_progress_gain = float(effect.get("predicted_progress_gain", 0) or 0.0)
    predicted_conflicts = float(len(effect.get("predicted_conflicts", [])))
    score += predicted_progress_gain * progress_weight
    score -= predicted_conflicts * conflict_penalty
    score += float(len(effect.get("predicted_preserved", []))) * preserved_bonus
    score += float(len(effect.get("predicted_workflow_paths", []))) * workflow_bonus
    progress_band = str(latent_state_summary.get("progress_band", "flat"))
    risk_band = str(latent_state_summary.get("risk_band", "stable"))
    learned_progress_signal = _learned_progress_signal(latent_state_summary)
    learned_risk_signal = _learned_risk_signal(latent_state_summary)
    is_long_horizon = str(world_model_summary.get("horizon", "")).strip() == "long_horizon"
    if progress_band in {"advancing", "improving"}:
        score += latent_bonus
    if risk_band in {"blocked", "stalled"} and predicted_progress_gain > 0:
        score += latent_bonus
    if risk_band == "regressive" and not predicted_progress_gain:
        score -= latent_risk_penalty
    if predicted_progress_gain > 0:
        score += learned_progress_signal * learned_progress_bonus
    if bool(latest_transition.get("no_progress", False)) and predicted_progress_gain > 0:
        score += _float_value(rollout_policy.get("recover_from_stall_bonus_weight"), 1.5)
        score += learned_risk_signal * learned_recovery_bonus
    elif not predicted_progress_gain and learned_risk_signal > 0.0:
        score -= learned_risk_signal * learned_continue_penalty
    if is_long_horizon:
        score += predicted_progress_gain * _float_value(
            rollout_policy.get("long_horizon_progress_bonus_weight"),
            0.0,
        )
        score += float(len(effect.get("predicted_preserved", []))) * _float_value(
            rollout_policy.get("long_horizon_preserved_bonus_weight"),
            0.0,
        )
        if risk_band in {"blocked", "stalled", "regressive"} and not predicted_progress_gain:
            score -= _float_value(rollout_policy.get("long_horizon_risk_penalty_weight"), 0.0)
        if learned_risk_signal >= 0.55 and predicted_progress_gain > 0 and predicted_conflicts <= 0:
            score += learned_risk_signal * learned_recovery_bonus
        elif learned_risk_signal >= 0.55 and not predicted_progress_gain:
            score -= learned_risk_signal * learned_continue_penalty
    return score


def _stop_value(
    *,
    world_model_summary: dict[str, object],
    latent_state_summary: dict[str, object],
    rollout_policy: dict[str, object],
) -> float:
    completion_ratio = _float_value(world_model_summary.get("completion_ratio"), 0.0)
    missing_expected = len(list(world_model_summary.get("missing_expected_artifacts", [])))
    present_forbidden = len(list(world_model_summary.get("present_forbidden_artifacts", [])))
    changed_preserved = len(list(world_model_summary.get("changed_preserved_artifacts", [])))
    learned_progress_signal = _learned_progress_signal(latent_state_summary)
    learned_risk_signal = _learned_risk_signal(latent_state_summary)
    score = completion_ratio * _float_value(rollout_policy.get("stop_completion_weight"), 8.0)
    score -= missing_expected * _float_value(rollout_policy.get("stop_missing_expected_penalty_weight"), 6.0)
    score -= present_forbidden * _float_value(rollout_policy.get("stop_forbidden_penalty_weight"), 6.0)
    score -= changed_preserved * _float_value(rollout_policy.get("stop_preserved_penalty_weight"), 4.0)
    score += learned_progress_signal * _float_value(rollout_policy.get("stop_learned_progress_weight"), 1.5)
    score -= learned_risk_signal * _float_value(rollout_policy.get("stop_learned_risk_penalty_weight"), 4.0)
    if str(world_model_summary.get("horizon", "")).strip() == "long_horizon":
        score -= max(0.0, 1.0 - completion_ratio) * _float_value(
            rollout_policy.get("long_horizon_stop_penalty_weight"),
            0.0,
        )
        score -= max(0.0, learned_risk_signal - learned_progress_signal) * _float_value(
            rollout_policy.get("long_horizon_stop_risk_gap_penalty_weight"),
            3.0,
        )
    if str(latent_state_summary.get("risk_band", "stable")) == "stable":
        score += _float_value(rollout_policy.get("stable_stop_bonus_weight"), 1.5)
    return score


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _learned_progress_signal(latent_state_summary: dict[str, object]) -> float:
    learned = latent_state_summary.get("learned_world_state", {})
    learned = learned if isinstance(learned, dict) else {}
    return max(
        0.0,
        min(
            1.0,
            max(
                _float_value(learned.get("progress_signal"), 0.0),
                _float_value(learned.get("world_progress_score"), 0.0),
                _float_value(learned.get("decoder_world_progress_score"), 0.0),
                _float_value(learned.get("transition_progress_score"), 0.0),
            ),
        ),
    )


def _learned_risk_signal(latent_state_summary: dict[str, object]) -> float:
    learned = latent_state_summary.get("learned_world_state", {})
    learned = learned if isinstance(learned, dict) else {}
    return max(
        0.0,
        min(
            1.0,
            max(
                _float_value(learned.get("risk_signal"), 0.0),
                _float_value(learned.get("world_risk_score"), 0.0),
                _float_value(learned.get("decoder_world_risk_score"), 0.0),
                _float_value(learned.get("transition_regression_score"), 0.0),
            ),
        ),
    )
