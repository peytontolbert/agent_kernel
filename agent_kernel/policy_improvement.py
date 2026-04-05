from __future__ import annotations

from copy import deepcopy

from evals.metrics import EvalMetrics

from .improvement_common import retained_mapping_section
from .prompt_policy_shared import (
    build_prompt_proposal_artifact,
    dedupe_prompt_adjustments,
    improvement_planner_controls,
    planner_mutation_controls,
    policy_behavior_controls,
    propose_prompt_adjustments,
    retained_improvement_planner_controls,
    retained_planner_controls,
    retained_policy_controls,
    retained_role_directives,
    role_directive_overrides,
)


def tolbert_runtime_policy_overrides(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {
        "min_path_confidence": 0.75,
        "require_trusted_retrieval": True,
        "fallback_to_vllm_on_low_confidence": True,
        "primary_min_command_score": 2,
        "allow_direct_command_primary": True,
        "allow_skill_primary": True,
        "use_encoder_context": True,
        "use_decoder_head": True,
        "use_value_head": True,
        "use_transition_head": True,
        "use_world_model_head": True,
        "use_risk_head": True,
        "use_stop_head": True,
        "use_policy_head": True,
        "use_latent_state": True,
    }
    if isinstance(baseline, dict):
        overrides.update(deepcopy(baseline))
    if metrics.low_confidence_episodes > 0:
        overrides["min_path_confidence"] = max(_float_value(overrides.get("min_path_confidence"), 0.75), 0.8)
    if failure_counts.get("missing_expected_file", 0) > 0:
        overrides["primary_min_command_score"] = max(_int_value(overrides.get("primary_min_command_score"), 2), 3)
    if focus == "retrieval_caution":
        overrides["min_path_confidence"] = max(_float_value(overrides.get("min_path_confidence"), 0.75), 0.85)
        overrides["primary_min_command_score"] = max(_int_value(overrides.get("primary_min_command_score"), 2), 3)
        overrides["fallback_to_vllm_on_low_confidence"] = True
        overrides["require_trusted_retrieval"] = True
    if focus == "verifier_alignment":
        overrides["primary_min_command_score"] = max(_int_value(overrides.get("primary_min_command_score"), 2), 3)
        overrides["use_world_model_head"] = True
        overrides["use_stop_head"] = True
        overrides["use_transition_head"] = True
    if focus == "long_horizon_success":
        overrides["min_path_confidence"] = max(_float_value(overrides.get("min_path_confidence"), 0.75), 0.8)
        overrides["primary_min_command_score"] = max(_int_value(overrides.get("primary_min_command_score"), 2), 3)
        overrides["use_world_model_head"] = True
        overrides["use_stop_head"] = True
        overrides["use_transition_head"] = True
        overrides["use_risk_head"] = True
    return overrides


def tolbert_decoder_policy_overrides(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {
        "allow_retrieval_guidance": True,
        "allow_skill_commands": True,
        "allow_task_suggestions": True,
        "allow_stop_decision": True,
        "min_stop_completion_ratio": 0.95,
        "max_task_suggestions": 3,
    }
    if isinstance(baseline, dict):
        overrides.update(deepcopy(baseline))
    if metrics.low_confidence_episodes > 0:
        overrides["max_task_suggestions"] = max(_int_value(overrides.get("max_task_suggestions"), 3), 4)
    if focus == "retrieval_caution":
        overrides["max_task_suggestions"] = max(_int_value(overrides.get("max_task_suggestions"), 3), 4)
        overrides["min_stop_completion_ratio"] = max(
            _float_value(overrides.get("min_stop_completion_ratio"), 0.95),
            0.98,
        )
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "verifier_alignment":
        overrides["min_stop_completion_ratio"] = max(
            _float_value(overrides.get("min_stop_completion_ratio"), 0.95),
            0.98,
        )
    if focus == "long_horizon_success":
        overrides["min_stop_completion_ratio"] = max(
            _float_value(overrides.get("min_stop_completion_ratio"), 0.95),
            0.99,
        )
        overrides["max_task_suggestions"] = max(_int_value(overrides.get("max_task_suggestions"), 3), 4)
    return overrides


def tolbert_rollout_policy_overrides(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {
        "predicted_progress_gain_weight": 3.0,
        "predicted_conflict_penalty_weight": 4.0,
        "predicted_preserved_bonus_weight": 1.0,
        "predicted_workflow_bonus_weight": 1.5,
        "latent_progress_bonus_weight": 1.0,
        "latent_risk_penalty_weight": 2.0,
        "recover_from_stall_bonus_weight": 1.5,
        "stop_completion_weight": 8.0,
        "stop_missing_expected_penalty_weight": 6.0,
        "stop_forbidden_penalty_weight": 6.0,
        "stop_preserved_penalty_weight": 4.0,
        "stable_stop_bonus_weight": 1.5,
    }
    if isinstance(baseline, dict):
        overrides.update(deepcopy(baseline))
    if metrics.low_confidence_episodes > 0:
        overrides["latent_risk_penalty_weight"] = max(
            _float_value(overrides.get("latent_risk_penalty_weight"), 2.0),
            2.5,
        )
    if failure_counts.get("command_failure", 0) > 0:
        overrides["predicted_conflict_penalty_weight"] = max(
            _float_value(overrides.get("predicted_conflict_penalty_weight"), 4.0),
            4.5,
        )
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "verifier_alignment":
        overrides["stop_missing_expected_penalty_weight"] = max(
            _float_value(overrides.get("stop_missing_expected_penalty_weight"), 6.0),
            7.5,
        )
        overrides["predicted_workflow_bonus_weight"] = max(
            _float_value(overrides.get("predicted_workflow_bonus_weight"), 1.5),
            2.0,
        )
    if focus == "retrieval_caution":
        overrides["predicted_workflow_bonus_weight"] = max(
            _float_value(overrides.get("predicted_workflow_bonus_weight"), 1.5),
            2.0,
        )
        overrides["latent_risk_penalty_weight"] = max(
            _float_value(overrides.get("latent_risk_penalty_weight"), 2.0),
            2.75,
        )
        overrides["stop_missing_expected_penalty_weight"] = max(
            _float_value(overrides.get("stop_missing_expected_penalty_weight"), 6.0),
            7.0,
        )
    if focus == "long_horizon_success":
        overrides["predicted_preserved_bonus_weight"] = max(
            _float_value(overrides.get("predicted_preserved_bonus_weight"), 1.0),
            1.5,
        )
        overrides["predicted_workflow_bonus_weight"] = max(
            _float_value(overrides.get("predicted_workflow_bonus_weight"), 1.5),
            2.0,
        )
        overrides["latent_risk_penalty_weight"] = max(
            _float_value(overrides.get("latent_risk_penalty_weight"), 2.0),
            2.5,
        )
        overrides["stop_missing_expected_penalty_weight"] = max(
            _float_value(overrides.get("stop_missing_expected_penalty_weight"), 6.0),
            7.0,
        )
        overrides["stop_preserved_penalty_weight"] = max(
            _float_value(overrides.get("stop_preserved_penalty_weight"), 4.0),
            5.5,
        )
        overrides["stable_stop_bonus_weight"] = max(
            _float_value(overrides.get("stable_stop_bonus_weight"), 1.5),
            2.0,
        )
        overrides["long_horizon_progress_bonus_weight"] = max(
            _float_value(overrides.get("long_horizon_progress_bonus_weight"), 0.0),
            1.0,
        )
        overrides["long_horizon_preserved_bonus_weight"] = max(
            _float_value(overrides.get("long_horizon_preserved_bonus_weight"), 0.0),
            1.0,
        )
        overrides["long_horizon_risk_penalty_weight"] = max(
            _float_value(overrides.get("long_horizon_risk_penalty_weight"), 0.0),
            2.0,
        )
        overrides["long_horizon_stop_penalty_weight"] = max(
            _float_value(overrides.get("long_horizon_stop_penalty_weight"), 0.0),
            2.5,
        )
    return overrides


def tolbert_hybrid_scoring_policy_overrides(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, float]:
    overrides: dict[str, float] = {
        "learned_score_weight": 1.5,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "risk_penalty_weight": 1.0,
        "stop_weight": 1.0,
        "transition_progress_weight": 0.15,
        "transition_regression_penalty_weight": 0.20,
        "world_progress_weight": 0.20,
        "world_risk_penalty_weight": 0.20,
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
        "respond_latent_bias_weight": 0.0,
        "respond_decoder_logprob_weight": 0.05,
    }
    if isinstance(baseline, dict):
        for key, value in baseline.items():
            overrides[str(key)] = _float_value(value, overrides.get(str(key), 0.0))
    if metrics.low_confidence_episodes > 0:
        overrides["risk_penalty_weight"] = max(float(overrides["risk_penalty_weight"]), 1.1)
        overrides["world_risk_penalty_weight"] = max(float(overrides["world_risk_penalty_weight"]), 0.25)
    if failure_counts.get("missing_expected_file", 0) > 0:
        overrides["world_progress_weight"] = max(float(overrides["world_progress_weight"]), 0.25)
        overrides["stop_weight"] = max(float(overrides["stop_weight"]), 1.1)
    if focus == "retrieval_caution":
        overrides["decoder_logprob_weight"] = min(float(overrides["decoder_logprob_weight"]), 0.08)
        overrides["risk_penalty_weight"] = max(float(overrides["risk_penalty_weight"]), 1.2)
        overrides["stop_weight"] = max(float(overrides["stop_weight"]), 1.1)
        overrides["world_progress_weight"] = max(float(overrides["world_progress_weight"]), 0.25)
        overrides["world_risk_penalty_weight"] = max(float(overrides["world_risk_penalty_weight"]), 0.25)
        overrides["transition_regression_penalty_weight"] = max(
            float(overrides["transition_regression_penalty_weight"]),
            0.28,
        )
    if focus == "verifier_alignment":
        overrides["world_progress_weight"] = max(float(overrides["world_progress_weight"]), 0.28)
        overrides["transition_progress_weight"] = max(float(overrides["transition_progress_weight"]), 0.2)
        overrides["stop_weight"] = max(float(overrides["stop_weight"]), 1.15)
    if focus == "long_horizon_success":
        overrides["risk_penalty_weight"] = max(float(overrides["risk_penalty_weight"]), 1.15)
        overrides["stop_weight"] = max(float(overrides["stop_weight"]), 1.15)
        overrides["transition_progress_weight"] = max(float(overrides["transition_progress_weight"]), 0.22)
        overrides["world_progress_weight"] = max(float(overrides["world_progress_weight"]), 0.3)
        overrides["world_risk_penalty_weight"] = max(float(overrides["world_risk_penalty_weight"]), 0.25)
        overrides["long_horizon_progress_bonus_weight"] = max(
            float(overrides.get("long_horizon_progress_bonus_weight", 0.0) or 0.0),
            0.35,
        )
        overrides["long_horizon_risk_penalty_weight"] = max(
            float(overrides.get("long_horizon_risk_penalty_weight", 0.0) or 0.0),
            0.3,
        )
        overrides["long_horizon_horizon_scale_weight"] = max(
            float(overrides.get("long_horizon_horizon_scale_weight", 0.0) or 0.0),
            0.15,
        )
    return overrides


def build_policy_proposal_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    artifact = build_prompt_proposal_artifact(
        metrics,
        failure_counts,
        focus=focus,
        current_payload=current_payload,
    )
    artifact["policy_schema"] = "agentic_policy_controls_v1"
    artifact["tolbert_runtime_policy_overrides"] = tolbert_runtime_policy_overrides(
        metrics,
        failure_counts,
        focus=focus,
        baseline=retained_tolbert_runtime_policy_overrides(current_payload),
    )
    artifact["tolbert_decoder_policy_overrides"] = tolbert_decoder_policy_overrides(
        metrics,
        failure_counts,
        focus=focus,
        baseline=retained_tolbert_decoder_policy_overrides(current_payload),
    )
    artifact["tolbert_rollout_policy_overrides"] = tolbert_rollout_policy_overrides(
        metrics,
        failure_counts,
        focus=focus,
        baseline=retained_tolbert_rollout_policy_overrides(current_payload),
    )
    artifact["tolbert_hybrid_scoring_policy_overrides"] = tolbert_hybrid_scoring_policy_overrides(
        metrics,
        failure_counts,
        focus=focus,
        baseline=retained_tolbert_hybrid_scoring_policy_overrides(current_payload),
    )
    return artifact


def retained_tolbert_runtime_policy_overrides(payload: object) -> dict[str, object]:
    return retained_mapping_section(
        payload,
        artifact_kind="prompt_proposal_set",
        section="tolbert_runtime_policy_overrides",
    )


def retained_tolbert_decoder_policy_overrides(payload: object) -> dict[str, object]:
    return retained_mapping_section(
        payload,
        artifact_kind="prompt_proposal_set",
        section="tolbert_decoder_policy_overrides",
    )


def retained_tolbert_rollout_policy_overrides(payload: object) -> dict[str, object]:
    return retained_mapping_section(
        payload,
        artifact_kind="prompt_proposal_set",
        section="tolbert_rollout_policy_overrides",
    )


def retained_tolbert_hybrid_scoring_policy_overrides(payload: object) -> dict[str, float]:
    overrides = retained_mapping_section(
        payload,
        artifact_kind="prompt_proposal_set",
        section="tolbert_hybrid_scoring_policy_overrides",
    )
    if not isinstance(overrides, dict):
        return {}
    return {str(key): _float_value(value, 0.0) for key, value in overrides.items() if str(key).strip()}


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "build_policy_proposal_artifact",
    "improvement_planner_controls",
    "planner_mutation_controls",
    "policy_behavior_controls",
    "propose_prompt_adjustments",
    "retained_improvement_planner_controls",
    "retained_planner_controls",
    "retained_policy_controls",
    "retained_role_directives",
    "retained_tolbert_decoder_policy_overrides",
    "retained_tolbert_hybrid_scoring_policy_overrides",
    "retained_tolbert_rollout_policy_overrides",
    "retained_tolbert_runtime_policy_overrides",
    "role_directive_overrides",
    "tolbert_decoder_policy_overrides",
    "tolbert_hybrid_scoring_policy_overrides",
    "tolbert_rollout_policy_overrides",
    "tolbert_runtime_policy_overrides",
]
