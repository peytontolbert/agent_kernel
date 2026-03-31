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
    score = 0.0
    score += float(effect.get("predicted_progress_gain", 0)) * progress_weight
    score -= float(len(effect.get("predicted_conflicts", []))) * conflict_penalty
    score += float(len(effect.get("predicted_preserved", []))) * preserved_bonus
    score += float(len(effect.get("predicted_workflow_paths", []))) * workflow_bonus
    progress_band = str(latent_state_summary.get("progress_band", "flat"))
    risk_band = str(latent_state_summary.get("risk_band", "stable"))
    if progress_band in {"advancing", "improving"}:
        score += latent_bonus
    if risk_band in {"blocked", "stalled"} and float(effect.get("predicted_progress_gain", 0)) > 0:
        score += latent_bonus
    if risk_band == "regressive" and not effect.get("predicted_progress_gain", 0):
        score -= latent_risk_penalty
    if bool(latest_transition.get("no_progress", False)) and float(effect.get("predicted_progress_gain", 0)) > 0:
        score += _float_value(rollout_policy.get("recover_from_stall_bonus_weight"), 1.5)
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
    score = completion_ratio * _float_value(rollout_policy.get("stop_completion_weight"), 8.0)
    score -= missing_expected * _float_value(rollout_policy.get("stop_missing_expected_penalty_weight"), 6.0)
    score -= present_forbidden * _float_value(rollout_policy.get("stop_forbidden_penalty_weight"), 6.0)
    score -= changed_preserved * _float_value(rollout_policy.get("stop_preserved_penalty_weight"), 4.0)
    if str(latent_state_summary.get("risk_band", "stable")) == "stable":
        score += _float_value(rollout_policy.get("stable_stop_bonus_weight"), 1.5)
    return score


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
