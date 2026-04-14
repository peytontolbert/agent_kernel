from __future__ import annotations

from typing import Any

from evals.metrics import EvalMetrics

from .improvement_common import normalized_control_mapping
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER
from .state_estimation_improvement import (
    STATE_ESTIMATION_LATENT_CONTROL_KEYS,
    STATE_ESTIMATION_POLICY_CONTROL_KEYS,
    STATE_ESTIMATION_TRANSITION_CONTROL_KEYS,
    retained_state_estimation_latent_controls,
    retained_state_estimation_policy_controls,
    retained_state_estimation_transition_controls,
)
from .transition_model_improvement import (
    retained_transition_model_controls,
    retained_transition_model_signatures,
)
from .universe_improvement import (
    UNIVERSE_ACTION_RISK_CONTROL_KEYS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS,
    UNIVERSE_GOVERNANCE_KEYS,
)

_STATE_ESTIMATION_CONTROL_KEYS = set(STATE_ESTIMATION_TRANSITION_CONTROL_KEYS)
_STATE_ESTIMATION_LATENT_KEYS = set(STATE_ESTIMATION_LATENT_CONTROL_KEYS)
_STATE_ESTIMATION_POLICY_KEYS = set(STATE_ESTIMATION_POLICY_CONTROL_KEYS)
_UNIVERSE_ACTION_RISK_KEYS = set(UNIVERSE_ACTION_RISK_CONTROL_KEYS)
_UNIVERSE_ENVIRONMENT_ENUM_FIELDS = {
    str(key): set(values) for key, values in UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS.items()
}
_UNIVERSE_ENVIRONMENT_BOOL_FIELDS = set(UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS)
_UNIVERSE_GOVERNANCE_KEYS_SET = set(UNIVERSE_GOVERNANCE_KEYS)


def state_estimation_transition_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_transition_controls(None)
    return retained_state_estimation_transition_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "controls": payload.get("controls", {}),
        }
    )


def state_estimation_latent_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_latent_controls(None)
    return retained_state_estimation_latent_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "latent_controls": payload.get("latent_controls", {}),
        }
    )


def state_estimation_policy_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return retained_state_estimation_policy_controls(None)
    return retained_state_estimation_policy_controls(
        {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "policy_controls": payload.get("policy_controls", {}),
        }
    )


def state_estimation_improvement_count(
    *,
    baseline_transition_controls: dict[str, object],
    candidate_transition_controls: dict[str, object],
    baseline_latent_controls: dict[str, object],
    candidate_latent_controls: dict[str, object],
    baseline_policy_controls: dict[str, object],
    candidate_policy_controls: dict[str, object],
) -> int:
    improvements = 0
    for key in sorted(_STATE_ESTIMATION_CONTROL_KEYS):
        if float(candidate_transition_controls.get(key, 0.0)) != float(baseline_transition_controls.get(key, 0.0)):
            improvements += 1
    for key in sorted(_STATE_ESTIMATION_LATENT_KEYS):
        if float(candidate_latent_controls.get(key, 0.0)) != float(baseline_latent_controls.get(key, 0.0)):
            improvements += 1
    for key in sorted(_STATE_ESTIMATION_POLICY_KEYS):
        if int(candidate_policy_controls.get(key, 0)) != int(baseline_policy_controls.get(key, 0)):
            improvements += 1
    return improvements


def trajectory_has_regression(payload: dict[str, object]) -> bool:
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if int(step.get("state_regression_count", 0) or 0) > 0:
            return True
        signals = {
            str(value).strip()
            for value in step.get("failure_signals", [])
            if str(value).strip()
        }
        if "state_regression" in signals:
            return True
    return False


def state_regression_trace_count(metrics: EvalMetrics) -> int:
    return sum(
        1
        for payload in (metrics.task_trajectories or {}).values()
        if isinstance(payload, dict) and trajectory_has_regression(payload)
    )


def regressive_recovery_rate(metrics: EvalMetrics) -> float:
    trajectories = [
        payload
        for payload in (metrics.task_trajectories or {}).values()
        if isinstance(payload, dict) and trajectory_has_regression(payload)
    ]
    if not trajectories:
        return 0.0
    recovered = sum(1 for payload in trajectories if bool(payload.get("success", False)))
    return recovered / len(trajectories)


def paired_trajectory_non_regression_rate(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> tuple[float, int]:
    baseline_trajectories = baseline_metrics.task_trajectories or {}
    candidate_trajectories = candidate_metrics.task_trajectories or {}
    if not isinstance(baseline_trajectories, dict) or not isinstance(candidate_trajectories, dict):
        return (0.0, 0)
    shared_task_ids = sorted(set(baseline_trajectories) & set(candidate_trajectories))
    if not shared_task_ids:
        return (0.0, 0)
    non_regressions = 0
    pair_count = 0
    for task_id in shared_task_ids:
        baseline_payload = baseline_trajectories.get(task_id, {})
        candidate_payload = candidate_trajectories.get(task_id, {})
        if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
            continue
        pair_count += 1
        if state_estimation_trajectory_score(candidate_payload) <= state_estimation_trajectory_score(baseline_payload):
            non_regressions += 1
    if pair_count <= 0:
        return (0.0, 0)
    return (non_regressions / pair_count, pair_count)


def state_estimation_trajectory_score(payload: dict[str, object]) -> float:
    score = 0.0
    if not bool(payload.get("success", False)):
        score += 5.0
    termination_reason = str(payload.get("termination_reason", "")).strip()
    if termination_reason == "no_state_progress":
        score += 3.0
    elif termination_reason == "repeated_failed_action":
        score += 2.0
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        score += 0.5 * int(step.get("state_regression_count", 0) or 0)
        if bool(step.get("timed_out", False)):
            score += 1.0
        if int(step.get("exit_code", 0) or 0) != 0:
            score += 0.25
        signals = {
            str(value).strip()
            for value in step.get("failure_signals", [])
            if str(value).strip()
        }
        if "no_state_progress" in signals:
            score += 1.0
        if "state_regression" in signals:
            score += 1.0
    return score


def state_estimation_evidence(
    *,
    baseline_transition_controls: dict[str, object],
    candidate_transition_controls: dict[str, object],
    baseline_latent_controls: dict[str, object],
    candidate_latent_controls: dict[str, object],
    baseline_policy_controls: dict[str, object],
    candidate_policy_controls: dict[str, object],
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> dict[str, object]:
    paired_non_regression_rate, paired_trajectory_pair_count = paired_trajectory_non_regression_rate(
        baseline_metrics,
        candidate_metrics,
    )
    return {
        "state_estimation_improvement_count": state_estimation_improvement_count(
            baseline_transition_controls=baseline_transition_controls,
            candidate_transition_controls=candidate_transition_controls,
            baseline_latent_controls=baseline_latent_controls,
            candidate_latent_controls=candidate_latent_controls,
            baseline_policy_controls=baseline_policy_controls,
            candidate_policy_controls=candidate_policy_controls,
        ),
        "no_state_progress_termination_delta": int(
            candidate_metrics.termination_reasons.get("no_state_progress", 0)
        )
        - int(baseline_metrics.termination_reasons.get("no_state_progress", 0)),
        "state_regression_trace_delta": state_regression_trace_count(candidate_metrics)
        - state_regression_trace_count(baseline_metrics),
        "paired_trajectory_non_regression_rate": paired_non_regression_rate,
        "paired_trajectory_pair_count": paired_trajectory_pair_count,
        "regressive_recovery_rate_delta": regressive_recovery_rate(candidate_metrics)
        - regressive_recovery_rate(baseline_metrics),
        "transition_control_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_CONTROL_KEYS)
            if float(candidate_transition_controls.get(key, 0.0))
            != float(baseline_transition_controls.get(key, 0.0))
        ),
        "latent_control_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_LATENT_KEYS)
            if float(candidate_latent_controls.get(key, 0.0))
            != float(baseline_latent_controls.get(key, 0.0))
        ),
        "state_estimation_policy_delta_count": sum(
            1
            for key in sorted(_STATE_ESTIMATION_POLICY_KEYS)
            if int(candidate_policy_controls.get(key, 0))
            != int(baseline_policy_controls.get(key, 0))
        ),
    }


def trust_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(controls, list_fields=("required_benchmark_families",))


def universe_governance_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_governance(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_governance(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "governance": payload.get("governance", {}),
        }
    )


def universe_invariants_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_invariants(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_invariants(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "invariants": payload.get("invariants", []),
        }
    )


def universe_forbidden_patterns_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_forbidden_command_patterns(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_forbidden_command_patterns(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "forbidden_command_patterns": payload.get("forbidden_command_patterns", []),
        }
    )


def universe_preferred_prefixes_from_payload(payload: dict[str, object] | None) -> list[str]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_preferred_command_prefixes(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_preferred_command_prefixes(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "preferred_command_prefixes": payload.get("preferred_command_prefixes", []),
        }
    )


def universe_action_risk_controls_from_payload(payload: dict[str, object] | None) -> dict[str, int]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_action_risk_controls(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_action_risk_controls(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "action_risk_controls": payload.get("action_risk_controls", {}),
        }
    )


def universe_environment_assumptions_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_environment_assumptions(None)
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.retained_universe_environment_assumptions(
        {
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "environment_assumptions": payload.get("environment_assumptions", {}),
        }
    )


def universe_change_scope(payload: dict[str, object] | None) -> str:
    if not isinstance(payload, dict):
        return "combined"
    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    if artifact_kind == "universe_constitution":
        return "constitution"
    if artifact_kind == "operating_envelope":
        return "operating_envelope"
    return "combined"


def universe_cross_family_support(evidence: dict[str, object]) -> int:
    support = 0
    for key in ("family_pass_rate_delta", "generated_family_pass_rate_delta"):
        delta_map = evidence.get(key, {})
        if not isinstance(delta_map, dict):
            continue
        for value in delta_map.values():
            try:
                if float(value) >= 0.0:
                    support += 1
            except (TypeError, ValueError):
                continue
    if support > 0:
        return support
    return max(1, 1 - int(evidence.get("regressed_family_count", 0)))


def universe_outcome_weighted_support(evidence: dict[str, object]) -> float:
    pass_gain = max(0.0, float(evidence.get("pass_rate_delta", 0.0) or 0.0))
    step_gain = max(0.0, -float(evidence.get("average_step_delta", 0.0) or 0.0))
    cross_family_support = max(1, universe_cross_family_support(evidence))
    support_discount = 0.5 if cross_family_support <= 1 else 1.0
    return round((1.0 + pass_gain * 20.0 + step_gain * 5.0) * cross_family_support * support_discount, 4)


def enabled_flag_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if bool(candidate_controls.get(key, False)) and not bool(baseline_controls.get(key, False))
    )


def increased_int_control_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if int(candidate_controls.get(key, 0)) > int(baseline_controls.get(key, 0))
    )


def expanded_sequence_control_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> int:
    return sum(
        1
        for key in keys
        if set(candidate_controls.get(key, [])) > set(baseline_controls.get(key, []))
    )


def boolean_control_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_delta": int(bool(candidate_controls.get(key, False))) - int(bool(baseline_controls.get(key, False)))
        for key in keys
    }


def integer_control_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_delta": int(candidate_controls.get(key, 0)) - int(baseline_controls.get(key, 0))
        for key in keys
    }


def sequence_length_deltas(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    return {
        f"{key}_count_delta": len(candidate_controls.get(key, [])) - len(baseline_controls.get(key, []))
        for key in keys
    }


def universe_improvement_count(
    *,
    baseline_governance: dict[str, object],
    candidate_governance: dict[str, object],
    baseline_action_risk_controls: dict[str, int],
    candidate_action_risk_controls: dict[str, int],
    baseline_environment_assumptions: dict[str, object],
    candidate_environment_assumptions: dict[str, object],
    baseline_invariants: list[str],
    candidate_invariants: list[str],
    baseline_forbidden_patterns: list[str],
    candidate_forbidden_patterns: list[str],
    baseline_preferred_prefixes: list[str],
    candidate_preferred_prefixes: list[str],
) -> int:
    improvements = enabled_flag_improvement_count(
        baseline_governance,
        candidate_governance,
        keys=tuple(sorted(_UNIVERSE_GOVERNANCE_KEYS_SET)),
    )
    improvements += increased_int_control_count(
        baseline_action_risk_controls,
        candidate_action_risk_controls,
        keys=tuple(sorted(_UNIVERSE_ACTION_RISK_KEYS)),
    )
    improvements += more_restrictive_environment_assumption_count(
        baseline_environment_assumptions,
        candidate_environment_assumptions,
    )
    if set(candidate_invariants) > set(baseline_invariants):
        improvements += 1
    if set(candidate_forbidden_patterns) > set(baseline_forbidden_patterns):
        improvements += 1
    if set(candidate_preferred_prefixes) > set(baseline_preferred_prefixes):
        improvements += 1
    return improvements


def environment_assumption_delta_count(
    baseline_assumptions: dict[str, object],
    candidate_assumptions: dict[str, object],
) -> int:
    deltas = 0
    for key in sorted(_UNIVERSE_ENVIRONMENT_ENUM_FIELDS):
        if str(candidate_assumptions.get(key, "")).strip().lower() != str(baseline_assumptions.get(key, "")).strip().lower():
            deltas += 1
    for key in sorted(_UNIVERSE_ENVIRONMENT_BOOL_FIELDS):
        if bool(candidate_assumptions.get(key, False)) != bool(baseline_assumptions.get(key, False)):
            deltas += 1
    return deltas


def more_restrictive_environment_assumption_count(
    baseline_assumptions: dict[str, object],
    candidate_assumptions: dict[str, object],
) -> int:
    improvements = 0
    for key, allowed_values in _UNIVERSE_ENVIRONMENT_ENUM_FIELDS.items():
        if key == "network_access_mode":
            order = ("blocked", "allowlist_only", "open")
        elif key == "git_write_mode":
            order = ("blocked", "operator_gated", "task_scoped")
        else:
            order = ("task_only", "generated_only", "shared_repo_gated")
        rank = {value: index for index, value in enumerate(order) if value in allowed_values}
        baseline_value = str(baseline_assumptions.get(key, order[0])).strip().lower()
        candidate_value = str(candidate_assumptions.get(key, order[0])).strip().lower()
        if baseline_value in rank and candidate_value in rank and rank[candidate_value] < rank[baseline_value]:
            improvements += 1
    for key in sorted(_UNIVERSE_ENVIRONMENT_BOOL_FIELDS):
        if not bool(baseline_assumptions.get(key, False)) and bool(candidate_assumptions.get(key, False)):
            improvements += 1
    return improvements


def universe_control_evidence(
    *,
    baseline_governance: dict[str, object],
    candidate_governance: dict[str, object],
    baseline_action_risk_controls: dict[str, int],
    candidate_action_risk_controls: dict[str, int],
    baseline_environment_assumptions: dict[str, object],
    candidate_environment_assumptions: dict[str, object],
    baseline_invariants: list[str],
    candidate_invariants: list[str],
    baseline_forbidden_patterns: list[str],
    candidate_forbidden_patterns: list[str],
    baseline_preferred_prefixes: list[str],
    candidate_preferred_prefixes: list[str],
) -> dict[str, object]:
    return {
        "universe_improvement_count": universe_improvement_count(
            baseline_governance=baseline_governance,
            candidate_governance=candidate_governance,
            baseline_action_risk_controls=baseline_action_risk_controls,
            candidate_action_risk_controls=candidate_action_risk_controls,
            baseline_environment_assumptions=baseline_environment_assumptions,
            candidate_environment_assumptions=candidate_environment_assumptions,
            baseline_invariants=baseline_invariants,
            candidate_invariants=candidate_invariants,
            baseline_forbidden_patterns=baseline_forbidden_patterns,
            candidate_forbidden_patterns=candidate_forbidden_patterns,
            baseline_preferred_prefixes=baseline_preferred_prefixes,
            candidate_preferred_prefixes=candidate_preferred_prefixes,
        ),
        "universe_governance_delta_count": sum(
            1
            for key in sorted(_UNIVERSE_GOVERNANCE_KEYS_SET)
            if bool(candidate_governance.get(key, False)) != bool(baseline_governance.get(key, False))
        ),
        "universe_action_risk_delta_count": sum(
            1
            for key in sorted(_UNIVERSE_ACTION_RISK_KEYS)
            if int(candidate_action_risk_controls.get(key, 0)) != int(baseline_action_risk_controls.get(key, 0))
        ),
        "universe_environment_delta_count": environment_assumption_delta_count(
            baseline_environment_assumptions,
            candidate_environment_assumptions,
        ),
        "universe_invariant_delta_count": len(set(candidate_invariants) - set(baseline_invariants)),
        "universe_forbidden_pattern_delta_count": len(
            set(candidate_forbidden_patterns) - set(baseline_forbidden_patterns)
        ),
        "universe_preferred_prefix_delta_count": len(
            set(candidate_preferred_prefixes) - set(baseline_preferred_prefixes)
        ),
    }


def trust_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    improvements = 0
    if float(candidate_controls.get("min_success_rate", 0.0)) > float(baseline_controls.get("min_success_rate", 0.0)):
        improvements += 1
    if float(candidate_controls.get("max_unsafe_ambiguous_rate", 1.0)) < float(
        baseline_controls.get("max_unsafe_ambiguous_rate", 1.0)
    ):
        improvements += 1
    if float(candidate_controls.get("max_hidden_side_effect_rate", 1.0)) < float(
        baseline_controls.get("max_hidden_side_effect_rate", 1.0)
    ):
        improvements += 1
    if float(candidate_controls.get("max_success_hidden_side_effect_rate", 1.0)) < float(
        baseline_controls.get("max_success_hidden_side_effect_rate", 1.0)
    ):
        improvements += 1
    if int(candidate_controls.get("min_distinct_families", 0)) > int(baseline_controls.get("min_distinct_families", 0)):
        improvements += 1
    if int(candidate_controls.get("breadth_min_reports", 0)) > int(baseline_controls.get("breadth_min_reports", 0)):
        improvements += 1
    candidate_families = set(candidate_controls.get("required_benchmark_families", []))
    baseline_families = set(baseline_controls.get("required_benchmark_families", []))
    if candidate_families > baseline_families:
        improvements += 1
    return improvements


def trust_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    return {
        "trust_control_improvement_count": trust_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
        "required_family_count_delta": len(candidate_controls.get("required_benchmark_families", []))
        - len(baseline_controls.get("required_benchmark_families", [])),
        "min_success_rate_delta": float(candidate_controls.get("min_success_rate", 0.0))
        - float(baseline_controls.get("min_success_rate", 0.0)),
        "max_unsafe_ambiguous_rate_delta": float(candidate_controls.get("max_unsafe_ambiguous_rate", 0.0))
        - float(baseline_controls.get("max_unsafe_ambiguous_rate", 0.0)),
        "max_hidden_side_effect_rate_delta": float(candidate_controls.get("max_hidden_side_effect_rate", 0.0))
        - float(baseline_controls.get("max_hidden_side_effect_rate", 0.0)),
        "max_success_hidden_side_effect_rate_delta": float(
            candidate_controls.get("max_success_hidden_side_effect_rate", 0.0)
        )
        - float(baseline_controls.get("max_success_hidden_side_effect_rate", 0.0)),
    }


def recovery_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(
        controls,
        bool_fields=(
            "snapshot_before_execution",
            "rollback_on_runner_exception",
            "rollback_on_failed_outcome",
            "rollback_on_safe_stop",
            "verify_post_rollback_file_count",
        ),
        nonnegative_int_fields=("max_post_rollback_file_count",),
    )


def recovery_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    improvements = enabled_flag_improvement_count(
        baseline_controls,
        candidate_controls,
        keys=(
            "snapshot_before_execution",
            "rollback_on_runner_exception",
            "rollback_on_failed_outcome",
            "rollback_on_safe_stop",
            "verify_post_rollback_file_count",
        ),
    )
    if int(candidate_controls.get("max_post_rollback_file_count", 0)) < int(
        baseline_controls.get("max_post_rollback_file_count", 0)
    ):
        improvements += 1
    return improvements


def recovery_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    evidence = {
        "recovery_control_improvement_count": recovery_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(
        boolean_control_deltas(
            baseline_controls,
            candidate_controls,
            keys=(
                "snapshot_before_execution",
                "rollback_on_runner_exception",
                "rollback_on_failed_outcome",
                "rollback_on_safe_stop",
                "verify_post_rollback_file_count",
            ),
        )
    )
    evidence.update(
        integer_control_deltas(
            baseline_controls,
            candidate_controls,
            keys=("max_post_rollback_file_count",),
        )
    )
    return evidence


def delegation_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    delegation_keys = tuple(sorted(str(key) for key in payload.get("controls", {}).keys())) if isinstance(controls, dict) else ()
    return normalized_control_mapping(controls, int_fields=delegation_keys)


def delegation_control_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    return increased_int_control_count(
        baseline_controls,
        candidate_controls,
        keys=tuple(sorted(set(baseline_controls) | set(candidate_controls))),
    )


def delegation_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    keys = tuple(sorted(set(baseline_controls) | set(candidate_controls)))
    evidence = {
        "delegation_control_improvement_count": delegation_control_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(integer_control_deltas(baseline_controls, candidate_controls, keys=keys))
    return evidence


def operator_policy_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    controls = payload.get("controls", {})
    return normalized_control_mapping(
        controls,
        bool_fields=(
            "unattended_allow_git_commands",
            "unattended_allow_http_requests",
            "unattended_allow_generated_path_mutations",
        ),
        positive_int_fields=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        list_fields=("unattended_allowed_benchmark_families", "unattended_generated_path_prefixes"),
        lowercase_list_fields=("unattended_http_allowed_hosts",),
    )


def operator_policy_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> int:
    return (
        expanded_sequence_control_count(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allowed_benchmark_families",
                "unattended_http_allowed_hosts",
                "unattended_generated_path_prefixes",
            ),
        )
        + enabled_flag_improvement_count(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allow_git_commands",
                "unattended_allow_http_requests",
                "unattended_allow_generated_path_mutations",
            ),
        )
        + increased_int_control_count(
            baseline_controls,
            candidate_controls,
            keys=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        )
    )


def operator_policy_control_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
) -> dict[str, object]:
    evidence = {
        "operator_policy_improvement_count": operator_policy_improvement_count(
            baseline_controls,
            candidate_controls,
        ),
    }
    evidence.update(
        sequence_length_deltas(
            baseline_controls,
            candidate_controls,
            keys=(
                "unattended_allowed_benchmark_families",
                "unattended_http_allowed_hosts",
                "unattended_generated_path_prefixes",
            ),
        )
    )
    rename_map = {
        "unattended_allowed_benchmark_families_count_delta": "allowed_benchmark_family_count_delta",
        "unattended_http_allowed_hosts_count_delta": "http_allowed_host_count_delta",
        "unattended_generated_path_prefixes_count_delta": "generated_path_prefix_count_delta",
    }
    for source_key, target_key in rename_map.items():
        if source_key in evidence:
            evidence[target_key] = evidence.pop(source_key)
    return evidence


def transition_model_controls_from_payload(payload: dict[str, object] | None) -> dict[str, object]:
    return retained_transition_model_controls(payload)


def transition_model_signatures_from_payload(payload: dict[str, object] | None) -> list[dict[str, object]]:
    return retained_transition_model_signatures(payload)


def transition_model_improvement_count(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    baseline_signatures: list[dict[str, object]],
    candidate_signatures: list[dict[str, object]],
) -> int:
    improvements = 0
    for key in (
        "repeat_command_penalty",
        "regressed_path_command_penalty",
        "recovery_command_bonus",
        "progress_command_bonus",
    ):
        if int(candidate_controls.get(key, 0)) > int(baseline_controls.get(key, 0)):
            improvements += 1
    if int(candidate_controls.get("max_signatures", 0)) > int(baseline_controls.get("max_signatures", 0)):
        improvements += 1
    baseline_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in baseline_signatures
    }
    candidate_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in candidate_signatures
    }
    if candidate_signature_keys > baseline_signature_keys:
        improvements += 1
    if len(candidate_signatures) > len(baseline_signatures):
        improvements += 1
    return improvements


def transition_model_evidence(
    baseline_controls: dict[str, object],
    candidate_controls: dict[str, object],
    *,
    baseline_signatures: list[dict[str, object]],
    candidate_signatures: list[dict[str, object]],
) -> dict[str, object]:
    baseline_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in baseline_signatures
    }
    candidate_signature_keys = {
        (str(signature.get("signal", "")), str(signature.get("command", "")))
        for signature in candidate_signatures
    }
    return {
        "transition_model_improvement_count": transition_model_improvement_count(
            baseline_controls,
            candidate_controls,
            baseline_signatures=baseline_signatures,
            candidate_signatures=candidate_signatures,
        ),
        "transition_signature_count": len(candidate_signatures),
        "transition_signature_count_delta": len(candidate_signatures) - len(baseline_signatures),
        "transition_signature_growth": len(candidate_signature_keys - baseline_signature_keys),
    }


def capability_surface_evidence(
    baseline_summary: dict[str, int],
    candidate_summary: dict[str, int],
) -> dict[str, object]:
    return {
        "module_count": int(candidate_summary.get("module_count", 0)),
        "enabled_module_count": int(candidate_summary.get("enabled_module_count", 0)),
        "external_capability_count": int(candidate_summary.get("external_capability_count", 0)),
        "improvement_surface_count": int(candidate_summary.get("improvement_surface_count", 0)),
        "module_count_delta": int(candidate_summary.get("module_count", 0))
        - int(baseline_summary.get("module_count", 0)),
        "enabled_module_count_delta": int(candidate_summary.get("enabled_module_count", 0))
        - int(baseline_summary.get("enabled_module_count", 0)),
        "external_capability_count_delta": int(candidate_summary.get("external_capability_count", 0))
        - int(baseline_summary.get("external_capability_count", 0)),
        "improvement_surface_count_delta": int(candidate_summary.get("improvement_surface_count", 0))
        - int(baseline_summary.get("improvement_surface_count", 0)),
    }


__all__ = [
    "boolean_control_deltas",
    "capability_surface_evidence",
    "delegation_control_evidence",
    "delegation_control_improvement_count",
    "delegation_controls_from_payload",
    "enabled_flag_improvement_count",
    "environment_assumption_delta_count",
    "expanded_sequence_control_count",
    "increased_int_control_count",
    "integer_control_deltas",
    "more_restrictive_environment_assumption_count",
    "operator_policy_control_evidence",
    "operator_policy_controls_from_payload",
    "operator_policy_improvement_count",
    "paired_trajectory_non_regression_rate",
    "recovery_control_evidence",
    "recovery_control_improvement_count",
    "recovery_controls_from_payload",
    "regressive_recovery_rate",
    "sequence_length_deltas",
    "state_estimation_evidence",
    "state_estimation_improvement_count",
    "state_estimation_latent_controls_from_payload",
    "state_estimation_policy_controls_from_payload",
    "state_estimation_trajectory_score",
    "state_estimation_transition_controls_from_payload",
    "state_regression_trace_count",
    "trajectory_has_regression",
    "transition_model_controls_from_payload",
    "transition_model_evidence",
    "transition_model_improvement_count",
    "transition_model_signatures_from_payload",
    "trust_control_evidence",
    "trust_control_improvement_count",
    "trust_controls_from_payload",
    "universe_action_risk_controls_from_payload",
    "universe_change_scope",
    "universe_control_evidence",
    "universe_cross_family_support",
    "universe_environment_assumptions_from_payload",
    "universe_forbidden_patterns_from_payload",
    "universe_governance_from_payload",
    "universe_improvement_count",
    "universe_invariants_from_payload",
    "universe_outcome_weighted_support",
    "universe_preferred_prefixes_from_payload",
]
