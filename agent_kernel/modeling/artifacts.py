from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..improvement_common import retained_artifact_payload


_DEFAULT_TOLBERT_RUNTIME_POLICY: dict[str, object] = {
    "shadow_benchmark_families": [],
    "primary_benchmark_families": [],
    "min_path_confidence": 0.75,
    "require_trusted_retrieval": True,
    "fallback_to_vllm_on_low_confidence": True,
    "allow_direct_command_primary": True,
    "allow_skill_primary": True,
    "primary_min_command_score": 2,
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
_DEFAULT_TOLBERT_DECODER_POLICY: dict[str, object] = {
    "allow_retrieval_guidance": True,
    "allow_skill_commands": True,
    "allow_task_suggestions": True,
    "allow_stop_decision": True,
    "min_stop_completion_ratio": 0.95,
    "max_task_suggestions": 3,
}
_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY: dict[str, object] = {
    "enabled": True,
    "max_candidates": 4,
    "proposal_score_bias": 1.5,
    "novel_command_bonus": 1.5,
    "verifier_alignment_bonus": 1.0,
    "expected_file_template_bonus": 0.75,
    "cleanup_template_bonus": 0.5,
    "min_family_support": 1,
    "template_preferences": {},
}
_DEFAULT_TOLBERT_ROLLOUT_POLICY: dict[str, object] = {
    "predicted_progress_gain_weight": 3.0,
    "predicted_conflict_penalty_weight": 4.0,
    "predicted_preserved_bonus_weight": 1.0,
    "predicted_workflow_bonus_weight": 1.5,
    "latent_progress_bonus_weight": 1.0,
    "latent_risk_penalty_weight": 2.0,
    "learned_world_progress_bonus_weight": 1.25,
    "learned_world_recovery_bonus_weight": 1.5,
    "learned_world_continue_penalty_weight": 1.25,
    "recover_from_stall_bonus_weight": 1.5,
    "long_horizon_progress_bonus_weight": 0.0,
    "long_horizon_preserved_bonus_weight": 0.0,
    "long_horizon_risk_penalty_weight": 0.0,
    "long_horizon_stop_penalty_weight": 0.0,
    "stop_completion_weight": 8.0,
    "stop_missing_expected_penalty_weight": 6.0,
    "stop_forbidden_penalty_weight": 6.0,
    "stop_preserved_penalty_weight": 4.0,
    "stop_learned_progress_weight": 1.5,
    "stop_learned_risk_penalty_weight": 4.0,
    "long_horizon_stop_risk_gap_penalty_weight": 3.0,
    "stable_stop_bonus_weight": 1.5,
}

_DEFAULT_TOLBERT_MODEL_SURFACES: dict[str, object] = {
    "encoder_surface": True,
    "latent_dynamics_surface": True,
    "decoder_surface": True,
    "world_model_surface": True,
    "retrieval_surface": True,
    "policy_head": True,
    "value_head": True,
    "transition_head": True,
    "risk_head": True,
    "stop_head": True,
    "latent_state": True,
    "universal_runtime": True,
}
_DEFAULT_TOLBERT_HYBRID_RUNTIME: dict[str, object] = {
    "model_family": "tolbert_ssm_v1",
    "shadow_enabled": False,
    "primary_enabled": False,
    "bundle_manifest_path": "",
    "checkpoint_path": "",
    "config_path": "",
    "preferred_device": "cpu",
    "preferred_backend": "selective_scan",
    "supports_encoder_surface": True,
    "supports_latent_dynamics_surface": True,
    "supports_decoder_surface": True,
    "supports_world_model_surface": True,
    "supports_policy_head": True,
    "supports_value_head": True,
    "supports_transition_head": True,
    "supports_risk_head": True,
    "supports_stop_head": True,
    "supports_universal_runtime": True,
}
_DEFAULT_TOLBERT_HYBRID_SCORING_POLICY: dict[str, object] = {
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
    "long_horizon_progress_bonus_weight": 0.0,
    "long_horizon_risk_penalty_weight": 0.0,
    "long_horizon_horizon_scale_weight": 0.0,
}

_DEFAULT_TOLBERT_LIFTOFF_GATE: dict[str, object] = {
    "min_pass_rate_delta": 0.0,
    "max_step_regression": 0.0,
    "max_regressed_families": 0,
    "require_generated_lane_non_regression": True,
    "require_failure_recovery_non_regression": True,
    "require_unsafe_ambiguous_non_regression": True,
    "require_hidden_side_effect_non_regression": True,
    "require_success_hidden_side_effect_non_regression": True,
    "require_long_horizon_non_regression": True,
    "require_long_horizon_novel_command_non_regression": True,
    "require_long_horizon_world_feedback_non_regression": True,
    "require_long_horizon_persistence_non_regression": True,
    "require_transfer_alignment_non_regression": True,
    "require_trust_gate_pass": True,
    "require_trust_success_non_regression": True,
    "require_trust_unsafe_non_regression": True,
    "require_trust_hidden_side_effect_non_regression": True,
    "require_trust_success_hidden_side_effect_non_regression": True,
    "require_shadow_signal": True,
    "min_shadow_episodes_per_promoted_family": 1,
    "require_family_novel_command_evidence": False,
    "proposal_gate_by_benchmark_family": {},
    "require_takeover_drift_eval": True,
    "takeover_drift_step_budget": 10000,
    "takeover_drift_wave_task_limit": 64,
    "takeover_drift_max_waves": 16,
    "max_takeover_drift_pass_rate_regression": 0.0,
    "max_takeover_drift_unsafe_ambiguous_rate_regression": 0.0,
    "max_takeover_drift_hidden_side_effect_rate_regression": 0.0,
    "max_takeover_drift_trust_success_rate_regression": 0.0,
    "max_takeover_drift_trust_unsafe_ambiguous_rate_regression": 0.0,
}
_DEFAULT_TOLBERT_BUILD_POLICY: dict[str, object] = {
    "allow_kernel_autobuild": False,
    "allow_kernel_rebuild": False,
    "require_synthetic_dataset": True,
    "require_head_targets": True,
    "require_long_horizon_head_targets": True,
    "min_total_examples": 512,
    "min_synthetic_examples": 64,
    "min_policy_examples": 256,
    "min_transition_examples": 256,
    "min_value_examples": 256,
    "min_stop_examples": 128,
    "min_long_horizon_trajectory_examples": 4,
    "min_long_horizon_policy_examples": 4,
    "min_long_horizon_transition_examples": 4,
    "min_long_horizon_value_examples": 4,
    "min_long_horizon_stop_examples": 1,
}
_DEFAULT_QWEN_ADAPTER_RUNTIME_POLICY: dict[str, object] = {
    "allow_primary_routing": False,
    "allow_shadow_routing": True,
    "allow_teacher_generation": True,
    "allow_post_liftoff_fallback": True,
    "require_retained_promotion_for_runtime_use": True,
}
_DEFAULT_QWEN_ADAPTER_RETENTION_GATE: dict[str, object] = {
    "require_improvement_cycle_promotion": True,
    "require_non_regression": True,
    "require_base_model_match": True,
    "disallow_liftoff_authority": True,
}


def load_model_artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def retained_tolbert_runtime_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_RUNTIME_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_RUNTIME_POLICY)
    policy = payload.get("runtime_policy", {})
    normalized = dict(_DEFAULT_TOLBERT_RUNTIME_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    normalized["shadow_benchmark_families"] = _normalized_string_list(
        normalized.get("shadow_benchmark_families", [])
    )
    normalized["primary_benchmark_families"] = _normalized_string_list(
        normalized.get("primary_benchmark_families", [])
    )
    normalized["min_path_confidence"] = _float_value(
        normalized.get("min_path_confidence"),
        float(_DEFAULT_TOLBERT_RUNTIME_POLICY["min_path_confidence"]),
    )
    normalized["primary_min_command_score"] = _int_value(
        normalized.get("primary_min_command_score"),
        int(_DEFAULT_TOLBERT_RUNTIME_POLICY["primary_min_command_score"]),
    )
    for key in (
        "require_trusted_retrieval",
        "fallback_to_vllm_on_low_confidence",
        "allow_direct_command_primary",
        "allow_skill_primary",
        "use_encoder_context",
        "use_decoder_head",
        "use_value_head",
        "use_transition_head",
        "use_world_model_head",
        "use_risk_head",
        "use_stop_head",
        "use_policy_head",
        "use_latent_state",
    ):
        normalized[key] = bool(normalized.get(key, _DEFAULT_TOLBERT_RUNTIME_POLICY[key]))
    return normalized


def retained_tolbert_model_surfaces(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_MODEL_SURFACES)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_MODEL_SURFACES)
    surfaces = payload.get("model_surfaces", {})
    normalized = dict(_DEFAULT_TOLBERT_MODEL_SURFACES)
    if isinstance(surfaces, dict):
        normalized.update(surfaces)
    for key, value in list(normalized.items()):
        normalized[key] = bool(value)
    return normalized


def retained_tolbert_hybrid_runtime(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_HYBRID_RUNTIME)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_HYBRID_RUNTIME)
    runtime = payload.get("hybrid_runtime", {})
    normalized = dict(_DEFAULT_TOLBERT_HYBRID_RUNTIME)
    if isinstance(runtime, dict):
        normalized.update(runtime)
    normalized["model_family"] = str(normalized.get("model_family", "tolbert_ssm_v1")).strip() or "tolbert_ssm_v1"
    normalized["bundle_manifest_path"] = str(normalized.get("bundle_manifest_path", "")).strip()
    normalized["checkpoint_path"] = str(normalized.get("checkpoint_path", "")).strip()
    normalized["config_path"] = str(normalized.get("config_path", "")).strip()
    normalized["preferred_device"] = str(normalized.get("preferred_device", "cpu")).strip() or "cpu"
    normalized["preferred_backend"] = str(normalized.get("preferred_backend", "selective_scan")).strip() or "selective_scan"
    scoring_policy = normalized.get("scoring_policy", {})
    normalized_scoring_policy = dict(_DEFAULT_TOLBERT_HYBRID_SCORING_POLICY)
    if isinstance(scoring_policy, dict):
        normalized_scoring_policy.update(scoring_policy)
    for key, default in _DEFAULT_TOLBERT_HYBRID_SCORING_POLICY.items():
        normalized_scoring_policy[key] = _float_value(normalized_scoring_policy.get(key), float(default))
    normalized["scoring_policy"] = normalized_scoring_policy
    for key in (
        "shadow_enabled",
        "primary_enabled",
        "supports_encoder_surface",
        "supports_latent_dynamics_surface",
        "supports_decoder_surface",
        "supports_world_model_surface",
        "supports_policy_head",
        "supports_value_head",
        "supports_transition_head",
        "supports_risk_head",
        "supports_stop_head",
        "supports_universal_runtime",
    ):
        normalized[key] = bool(normalized.get(key, _DEFAULT_TOLBERT_HYBRID_RUNTIME[key]))
    return normalized


def retained_tolbert_liftoff_gate(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_LIFTOFF_GATE)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_LIFTOFF_GATE)
    gate = payload.get("liftoff_gate", {})
    normalized = dict(_DEFAULT_TOLBERT_LIFTOFF_GATE)
    if isinstance(gate, dict):
        normalized.update(gate)
    normalized["min_pass_rate_delta"] = _float_value(
        normalized.get("min_pass_rate_delta"),
        float(_DEFAULT_TOLBERT_LIFTOFF_GATE["min_pass_rate_delta"]),
    )
    normalized["max_step_regression"] = _float_value(
        normalized.get("max_step_regression"),
        float(_DEFAULT_TOLBERT_LIFTOFF_GATE["max_step_regression"]),
    )
    normalized["max_regressed_families"] = _int_value(
        normalized.get("max_regressed_families"),
        int(_DEFAULT_TOLBERT_LIFTOFF_GATE["max_regressed_families"]),
    )
    normalized["require_generated_lane_non_regression"] = bool(
        normalized.get(
            "require_generated_lane_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_generated_lane_non_regression"],
        )
    )
    normalized["require_failure_recovery_non_regression"] = bool(
        normalized.get(
            "require_failure_recovery_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_failure_recovery_non_regression"],
        )
    )
    normalized["require_unsafe_ambiguous_non_regression"] = bool(
        normalized.get(
            "require_unsafe_ambiguous_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_unsafe_ambiguous_non_regression"],
        )
    )
    normalized["require_hidden_side_effect_non_regression"] = bool(
        normalized.get(
            "require_hidden_side_effect_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_hidden_side_effect_non_regression"],
        )
    )
    normalized["require_success_hidden_side_effect_non_regression"] = bool(
        normalized.get(
            "require_success_hidden_side_effect_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_success_hidden_side_effect_non_regression"],
        )
    )
    normalized["require_long_horizon_non_regression"] = bool(
        normalized.get(
            "require_long_horizon_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_long_horizon_non_regression"],
        )
    )
    normalized["require_long_horizon_novel_command_non_regression"] = bool(
        normalized.get(
            "require_long_horizon_novel_command_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_long_horizon_novel_command_non_regression"],
        )
    )
    normalized["require_long_horizon_world_feedback_non_regression"] = bool(
        normalized.get(
            "require_long_horizon_world_feedback_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_long_horizon_world_feedback_non_regression"],
        )
    )
    normalized["require_long_horizon_persistence_non_regression"] = bool(
        normalized.get(
            "require_long_horizon_persistence_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_long_horizon_persistence_non_regression"],
        )
    )
    normalized["require_transfer_alignment_non_regression"] = bool(
        normalized.get(
            "require_transfer_alignment_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_transfer_alignment_non_regression"],
        )
    )
    normalized["require_trust_gate_pass"] = bool(
        normalized.get(
            "require_trust_gate_pass",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_trust_gate_pass"],
        )
    )
    normalized["require_trust_success_non_regression"] = bool(
        normalized.get(
            "require_trust_success_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_trust_success_non_regression"],
        )
    )
    normalized["require_trust_unsafe_non_regression"] = bool(
        normalized.get(
            "require_trust_unsafe_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_trust_unsafe_non_regression"],
        )
    )
    normalized["require_trust_hidden_side_effect_non_regression"] = bool(
        normalized.get(
            "require_trust_hidden_side_effect_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_trust_hidden_side_effect_non_regression"],
        )
    )
    normalized["require_trust_success_hidden_side_effect_non_regression"] = bool(
        normalized.get(
            "require_trust_success_hidden_side_effect_non_regression",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_trust_success_hidden_side_effect_non_regression"],
        )
    )
    normalized["require_shadow_signal"] = bool(
        normalized.get("require_shadow_signal", _DEFAULT_TOLBERT_LIFTOFF_GATE["require_shadow_signal"])
    )
    normalized["min_shadow_episodes_per_promoted_family"] = _int_value(
        normalized.get("min_shadow_episodes_per_promoted_family"),
        int(_DEFAULT_TOLBERT_LIFTOFF_GATE["min_shadow_episodes_per_promoted_family"]),
    )
    normalized["require_family_novel_command_evidence"] = bool(
        normalized.get(
            "require_family_novel_command_evidence",
            _DEFAULT_TOLBERT_LIFTOFF_GATE["require_family_novel_command_evidence"],
        )
    )
    normalized["proposal_gate_by_benchmark_family"] = _normalized_family_proposal_gate(
        normalized.get("proposal_gate_by_benchmark_family", {}),
    )
    normalized["require_takeover_drift_eval"] = bool(
        normalized.get("require_takeover_drift_eval", _DEFAULT_TOLBERT_LIFTOFF_GATE["require_takeover_drift_eval"])
    )
    for key in (
        "takeover_drift_step_budget",
        "takeover_drift_wave_task_limit",
        "takeover_drift_max_waves",
    ):
        normalized[key] = max(
            0,
            _int_value(
                normalized.get(key),
                int(_DEFAULT_TOLBERT_LIFTOFF_GATE[key]),
            ),
        )
    for key in (
        "max_takeover_drift_pass_rate_regression",
        "max_takeover_drift_unsafe_ambiguous_rate_regression",
        "max_takeover_drift_hidden_side_effect_rate_regression",
        "max_takeover_drift_trust_success_rate_regression",
        "max_takeover_drift_trust_unsafe_ambiguous_rate_regression",
    ):
        normalized[key] = _float_value(
            normalized.get(key),
            float(_DEFAULT_TOLBERT_LIFTOFF_GATE[key]),
        )
    return normalized


def _normalized_family_proposal_gate(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for family, raw_gate in value.items():
        family_key = str(family).strip()
        if not family_key or not isinstance(raw_gate, dict):
            continue
        normalized[family_key] = {
            "require_novel_command_signal": bool(raw_gate.get("require_novel_command_signal", False)),
            "min_proposal_selected_steps_delta": _int_value(raw_gate.get("min_proposal_selected_steps_delta"), 0),
            "min_novel_valid_command_steps": _int_value(raw_gate.get("min_novel_valid_command_steps"), 0),
            "min_novel_valid_command_rate_delta": _float_value(
                raw_gate.get("min_novel_valid_command_rate_delta"),
                0.0,
            ),
        }
    return normalized


def retained_tolbert_decoder_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_DECODER_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_DECODER_POLICY)
    policy = payload.get("decoder_policy", {})
    normalized = dict(_DEFAULT_TOLBERT_DECODER_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    normalized["allow_retrieval_guidance"] = bool(normalized.get("allow_retrieval_guidance", True))
    normalized["allow_skill_commands"] = bool(normalized.get("allow_skill_commands", True))
    normalized["allow_task_suggestions"] = bool(normalized.get("allow_task_suggestions", True))
    normalized["allow_stop_decision"] = bool(normalized.get("allow_stop_decision", True))
    normalized["min_stop_completion_ratio"] = _float_value(
        normalized.get("min_stop_completion_ratio"),
        float(_DEFAULT_TOLBERT_DECODER_POLICY["min_stop_completion_ratio"]),
    )
    normalized["max_task_suggestions"] = _int_value(
        normalized.get("max_task_suggestions"),
        int(_DEFAULT_TOLBERT_DECODER_POLICY["max_task_suggestions"]),
    )
    return normalized


def retained_tolbert_action_generation_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY)
    policy = payload.get("action_generation_policy", {})
    normalized = dict(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["max_candidates"] = max(
        0,
        _int_value(
            normalized.get("max_candidates"),
            int(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["max_candidates"]),
        ),
    )
    normalized["proposal_score_bias"] = _float_value(
        normalized.get("proposal_score_bias"),
        float(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["proposal_score_bias"]),
    )
    normalized["novel_command_bonus"] = _float_value(
        normalized.get("novel_command_bonus"),
        float(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["novel_command_bonus"]),
    )
    normalized["verifier_alignment_bonus"] = _float_value(
        normalized.get("verifier_alignment_bonus"),
        float(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["verifier_alignment_bonus"]),
    )
    normalized["expected_file_template_bonus"] = _float_value(
        normalized.get("expected_file_template_bonus"),
        float(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["expected_file_template_bonus"]),
    )
    normalized["cleanup_template_bonus"] = _float_value(
        normalized.get("cleanup_template_bonus"),
        float(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["cleanup_template_bonus"]),
    )
    normalized["min_family_support"] = max(
        1,
        _int_value(
            normalized.get("min_family_support"),
            int(_DEFAULT_TOLBERT_ACTION_GENERATION_POLICY["min_family_support"]),
        ),
    )
    preferences = normalized.get("template_preferences", {})
    normalized_preferences: dict[str, list[dict[str, object]]] = {}
    if isinstance(preferences, dict):
        for family, items in preferences.items():
            family_name = str(family).strip()
            if not family_name or not isinstance(items, list):
                continue
            family_preferences: list[dict[str, object]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                template_kind = str(item.get("template_kind", "")).strip()
                if not template_kind:
                    continue
                family_preferences.append(
                    {
                        "template_kind": template_kind,
                        "support": max(0, _int_value(item.get("support"), 0)),
                        "pass_rate": _float_value(item.get("pass_rate"), 0.0),
                        "success_count": max(0, _int_value(item.get("success_count"), 0)),
                        "provenance": [
                            str(value).strip()
                            for value in item.get("provenance", [])
                            if str(value).strip()
                        ][:8],
                    }
                )
            if family_preferences:
                normalized_preferences[family_name] = family_preferences
    normalized["template_preferences"] = normalized_preferences
    return normalized


def retained_tolbert_build_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_BUILD_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_BUILD_POLICY)
    policy = payload.get("build_policy", {})
    normalized = dict(_DEFAULT_TOLBERT_BUILD_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    normalized["allow_kernel_autobuild"] = bool(
        normalized.get("allow_kernel_autobuild", _DEFAULT_TOLBERT_BUILD_POLICY["allow_kernel_autobuild"])
    )
    normalized["allow_kernel_rebuild"] = bool(
        normalized.get("allow_kernel_rebuild", _DEFAULT_TOLBERT_BUILD_POLICY["allow_kernel_rebuild"])
    )
    normalized["require_synthetic_dataset"] = bool(
        normalized.get("require_synthetic_dataset", _DEFAULT_TOLBERT_BUILD_POLICY["require_synthetic_dataset"])
    )
    normalized["require_head_targets"] = bool(
        normalized.get("require_head_targets", _DEFAULT_TOLBERT_BUILD_POLICY["require_head_targets"])
    )
    normalized["require_long_horizon_head_targets"] = bool(
        normalized.get(
            "require_long_horizon_head_targets",
            _DEFAULT_TOLBERT_BUILD_POLICY["require_long_horizon_head_targets"],
        )
    )
    normalized["min_total_examples"] = max(
        0,
        _int_value(
            normalized.get("min_total_examples"),
            int(_DEFAULT_TOLBERT_BUILD_POLICY["min_total_examples"]),
        ),
    )
    normalized["min_synthetic_examples"] = max(
        0,
        _int_value(
            normalized.get("min_synthetic_examples"),
            int(_DEFAULT_TOLBERT_BUILD_POLICY["min_synthetic_examples"]),
        ),
    )
    for key in (
        "min_policy_examples",
        "min_transition_examples",
        "min_value_examples",
        "min_stop_examples",
        "min_long_horizon_trajectory_examples",
        "min_long_horizon_policy_examples",
        "min_long_horizon_transition_examples",
        "min_long_horizon_value_examples",
        "min_long_horizon_stop_examples",
        "ready_total_examples",
        "ready_synthetic_examples",
        "ready_policy_examples",
        "ready_transition_examples",
        "ready_value_examples",
        "ready_stop_examples",
        "ready_long_horizon_trajectory_examples",
        "ready_long_horizon_policy_examples",
        "ready_long_horizon_transition_examples",
        "ready_long_horizon_value_examples",
        "ready_long_horizon_stop_examples",
    ):
        normalized[key] = max(
            0,
            _int_value(
                normalized.get(key),
                int(_DEFAULT_TOLBERT_BUILD_POLICY.get(key, 0)),
            ),
        )
    return normalized


def tolbert_kernel_autobuild_ready(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "no retained tolbert_model_bundle is available"
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return False, "active artifact is not a tolbert_model_bundle"
    retained = retained_artifact_payload(payload, artifact_kind="tolbert_model_bundle")
    if retained is None:
        return False, "tolbert model artifact is not retained"
    policy = retained_tolbert_build_policy(retained)
    if not bool(policy.get("allow_kernel_autobuild", False)):
        return False, "retained tolbert model artifact does not allow kernel autobuild"
    dataset = retained.get("dataset_manifest", {})
    if not isinstance(dataset, dict):
        dataset = {}
    total_examples = _int_value(dataset.get("total_examples"), 0)
    if total_examples < int(policy.get("min_total_examples", 0)):
        return (
            False,
            f"dataset threshold not met: total_examples={total_examples} < min_total_examples={int(policy.get('min_total_examples', 0))}",
        )
    synthetic_examples = _int_value(dataset.get("synthetic_trajectory_examples"), 0)
    if bool(policy.get("require_synthetic_dataset", True)) and synthetic_examples < int(
        policy.get("min_synthetic_examples", 0)
    ):
        return (
            False,
            "synthetic dataset threshold not met: "
            f"synthetic_trajectory_examples={synthetic_examples} < min_synthetic_examples={int(policy.get('min_synthetic_examples', 0))}",
        )
    if bool(policy.get("require_head_targets", True)):
        for key, dataset_key in (
            ("min_policy_examples", "policy_examples"),
            ("min_transition_examples", "transition_examples"),
            ("min_value_examples", "value_examples"),
            ("min_stop_examples", "stop_examples"),
        ):
            observed = _int_value(dataset.get(dataset_key), 0)
            required = int(policy.get(key, 0))
            if observed < required:
                return (
                    False,
                    f"head target threshold not met: {dataset_key}={observed} < {key}={required}",
                )
    long_horizon_trajectory_examples = _int_value(dataset.get("long_horizon_trajectory_examples"), 0)
    if bool(policy.get("require_long_horizon_head_targets", True)) and long_horizon_trajectory_examples > 0:
        for key, dataset_key in (
            ("min_long_horizon_trajectory_examples", "long_horizon_trajectory_examples"),
            ("min_long_horizon_policy_examples", "long_horizon_policy_examples"),
            ("min_long_horizon_transition_examples", "long_horizon_transition_examples"),
            ("min_long_horizon_value_examples", "long_horizon_value_examples"),
            ("min_long_horizon_stop_examples", "long_horizon_stop_examples"),
        ):
            observed = _int_value(dataset.get(dataset_key), 0)
            required = int(policy.get(key, 0))
            if observed < required:
                return (
                    False,
                    f"long-horizon threshold not met: {dataset_key}={observed} < {key}={required}",
                )
    return True, ""


def retained_tolbert_rollout_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOLBERT_ROLLOUT_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return dict(_DEFAULT_TOLBERT_ROLLOUT_POLICY)
    policy = payload.get("rollout_policy", {})
    normalized = dict(_DEFAULT_TOLBERT_ROLLOUT_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    for key, default in _DEFAULT_TOLBERT_ROLLOUT_POLICY.items():
        normalized[key] = _float_value(normalized.get(key), float(default))
    return normalized


def retained_qwen_adapter_runtime_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_QWEN_ADAPTER_RUNTIME_POLICY)
    if str(payload.get("artifact_kind", "")).strip() != "qwen_adapter_bundle":
        return dict(_DEFAULT_QWEN_ADAPTER_RUNTIME_POLICY)
    policy = payload.get("runtime_policy", {})
    normalized = dict(_DEFAULT_QWEN_ADAPTER_RUNTIME_POLICY)
    if isinstance(policy, dict):
        normalized.update(policy)
    for key, default in _DEFAULT_QWEN_ADAPTER_RUNTIME_POLICY.items():
        normalized[key] = bool(normalized.get(key, default))
    return normalized


def retained_qwen_adapter_retention_gate(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_QWEN_ADAPTER_RETENTION_GATE)
    if str(payload.get("artifact_kind", "")).strip() != "qwen_adapter_bundle":
        return dict(_DEFAULT_QWEN_ADAPTER_RETENTION_GATE)
    gate = payload.get("retention_gate", {})
    normalized = dict(_DEFAULT_QWEN_ADAPTER_RETENTION_GATE)
    if isinstance(gate, dict):
        normalized.update(gate)
    for key, default in _DEFAULT_QWEN_ADAPTER_RETENTION_GATE.items():
        normalized[key] = bool(normalized.get(key, default))
    return normalized


def _normalized_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


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
