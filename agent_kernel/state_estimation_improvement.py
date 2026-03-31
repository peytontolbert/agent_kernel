from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from evals.metrics import EvalMetrics

from .config import KernelConfig
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)

STATE_ESTIMATION_PROPOSAL_AREAS = {
    "transition_normalization",
    "risk_sensitivity",
    "recovery_bias",
}
STATE_ESTIMATION_GENERATION_FOCI = {
    "balanced",
    "transition_normalization",
    "risk_sensitivity",
    "recovery_bias",
}
STATE_ESTIMATION_TRANSITION_CONTROL_KEYS = {
    "no_progress_progress_epsilon",
    "min_state_change_score_for_progress",
    "regression_path_budget",
    "regression_severity_weight",
    "progress_recovery_credit",
}
STATE_ESTIMATION_LATENT_CONTROL_KEYS = {
    "advancing_completion_ratio",
    "advancing_progress_delta",
    "improving_progress_delta",
    "regressing_progress_delta",
    "regressive_regression_count",
    "blocked_forbidden_count",
    "active_path_budget",
    "learned_world_progress_threshold",
    "learned_world_risk_threshold",
    "learned_world_blend_weight",
}
STATE_ESTIMATION_POLICY_CONTROL_KEYS = {
    "regressive_path_match_bonus",
    "regressive_cleanup_bonus",
    "blocked_command_bonus",
    "advancing_path_match_bonus",
    "trusted_retrieval_path_bonus",
    "learned_world_progress_bonus",
    "learned_world_risk_penalty",
}


def build_state_estimation_proposal_artifact(
    metrics: EvalMetrics,
    transition_summary: dict[str, object],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    normalized_focus = None if generation_focus == "balanced" else generation_focus
    return build_standard_proposal_artifact(
        artifact_kind="state_estimation_policy_set",
        generation_focus=generation_focus,
        control_schema="state_estimation_controls_v1",
        retention_gate=retention_gate_preset("state_estimation"),
        controls=state_estimation_transition_controls(
            metrics,
            transition_summary,
            focus=normalized_focus,
            baseline=retained_state_estimation_transition_controls(current_payload),
        ),
        proposals=_proposals(metrics, transition_summary, focus=normalized_focus),
        extra_sections={
            "latent_controls": state_estimation_latent_controls(
                metrics,
                transition_summary,
                focus=normalized_focus,
                baseline=retained_state_estimation_latent_controls(current_payload),
            ),
            "policy_controls": state_estimation_policy_controls(
                metrics,
                transition_summary,
                focus=normalized_focus,
                baseline=retained_state_estimation_policy_controls(current_payload),
            ),
            "transition_summary": dict(transition_summary),
        },
    )


def state_estimation_transition_controls(
    metrics: EvalMetrics,
    transition_summary: dict[str, object],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = dict(transition_summary or {})
    controls: dict[str, object] = {
        "no_progress_progress_epsilon": 0.0,
        "min_state_change_score_for_progress": 1,
        "regression_path_budget": 6,
        "regression_severity_weight": 1.0,
        "progress_recovery_credit": 1.0,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if int(summary.get("state_regression_steps", 0) or 0) > 0 or focus == "transition_normalization":
        controls["regression_path_budget"] = max(int(controls["regression_path_budget"]), 8)
        controls["regression_severity_weight"] = max(float(controls["regression_severity_weight"]), 1.25)
    if int(summary.get("state_progress_gain_steps", 0) or 0) <= 0:
        controls["no_progress_progress_epsilon"] = max(float(controls["no_progress_progress_epsilon"]), 0.02)
        controls["min_state_change_score_for_progress"] = max(
            int(controls["min_state_change_score_for_progress"]),
            2,
        )
    if focus == "recovery_bias" or metrics.low_confidence_episodes > 0:
        controls["progress_recovery_credit"] = max(float(controls["progress_recovery_credit"]), 1.5)
        controls["no_progress_progress_epsilon"] = max(float(controls["no_progress_progress_epsilon"]), 0.03)
    return _normalize_transition_controls(controls)


def state_estimation_latent_controls(
    metrics: EvalMetrics,
    transition_summary: dict[str, object],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = dict(transition_summary or {})
    controls: dict[str, object] = {
        "advancing_completion_ratio": 0.8,
        "advancing_progress_delta": 0.2,
        "improving_progress_delta": 0.0,
        "regressing_progress_delta": -0.05,
        "regressive_regression_count": 1,
        "blocked_forbidden_count": 1,
        "active_path_budget": 6,
        "learned_world_progress_threshold": 0.55,
        "learned_world_risk_threshold": 0.55,
        "learned_world_blend_weight": 0.6,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if int(summary.get("state_regression_steps", 0) or 0) > 0 or focus == "risk_sensitivity":
        controls["regressive_regression_count"] = 1
        controls["active_path_budget"] = max(int(controls["active_path_budget"]), 8)
    if float(summary.get("average_net_state_progress_delta", 0.0) or 0.0) <= 0.0:
        controls["advancing_progress_delta"] = min(float(controls["advancing_progress_delta"]), 0.15)
        controls["improving_progress_delta"] = min(float(controls["improving_progress_delta"]), 0.0)
    if focus == "recovery_bias" or metrics.low_confidence_episodes > 0:
        controls["blocked_forbidden_count"] = 1
        controls["active_path_budget"] = max(int(controls["active_path_budget"]), 8)
        controls["learned_world_blend_weight"] = max(float(controls["learned_world_blend_weight"]), 0.7)
    return _normalize_latent_controls(controls)


def state_estimation_policy_controls(
    metrics: EvalMetrics,
    transition_summary: dict[str, object],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = dict(transition_summary or {})
    controls: dict[str, object] = {
        "regressive_path_match_bonus": 2,
        "regressive_cleanup_bonus": 1,
        "blocked_command_bonus": 1,
        "advancing_path_match_bonus": 1,
        "trusted_retrieval_path_bonus": 1,
        "learned_world_progress_bonus": 2,
        "learned_world_risk_penalty": 2,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if int(summary.get("state_regression_steps", 0) or 0) > 0 or focus == "risk_sensitivity":
        controls["regressive_path_match_bonus"] = max(int(controls["regressive_path_match_bonus"]), 3)
    if int(summary.get("state_progress_gain_steps", 0) or 0) <= 0 or focus == "recovery_bias":
        controls["blocked_command_bonus"] = max(int(controls["blocked_command_bonus"]), 2)
        controls["regressive_cleanup_bonus"] = max(int(controls["regressive_cleanup_bonus"]), 2)
    if metrics.trusted_retrieval_steps > 0:
        controls["trusted_retrieval_path_bonus"] = max(int(controls["trusted_retrieval_path_bonus"]), 2)
    if int(summary.get("state_regression_steps", 0) or 0) > 0:
        controls["learned_world_risk_penalty"] = max(int(controls["learned_world_risk_penalty"]), 3)
    if int(summary.get("state_progress_gain_steps", 0) or 0) > 0:
        controls["learned_world_progress_bonus"] = max(int(controls["learned_world_progress_bonus"]), 3)
    return _normalize_policy_controls(controls)


def retained_state_estimation_transition_controls(payload: object) -> dict[str, object]:
    return _normalize_transition_controls(
        retained_mapping_section(
            payload,
            artifact_kind="state_estimation_policy_set",
            section="controls",
        )
    )


def retained_state_estimation_latent_controls(payload: object) -> dict[str, object]:
    return _normalize_latent_controls(
        retained_mapping_section(
            payload,
            artifact_kind="state_estimation_policy_set",
            section="latent_controls",
        )
    )


def retained_state_estimation_policy_controls(payload: object) -> dict[str, object]:
    return _normalize_policy_controls(
        retained_mapping_section(
            payload,
            artifact_kind="state_estimation_policy_set",
            section="policy_controls",
        )
    )


def retained_state_estimation_payload(config: KernelConfig | None) -> dict[str, object]:
    resolved = config or KernelConfig()
    if not bool(resolved.use_state_estimation_proposals):
        return {}
    path = resolved.state_estimation_proposals_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_state_transition(
    raw_transition: dict[str, object] | None,
    *,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    transition = dict(raw_transition or {})
    controls = retained_state_estimation_transition_controls(payload)
    progress_delta = _float_value(transition.get("progress_delta"), 0.0)
    regressions = [
        str(value).strip()
        for value in transition.get("regressions", [])
        if str(value).strip()
    ]
    regression_path_budget = max(1, int(controls.get("regression_path_budget", 6)))
    transition["regressions"] = regressions[:regression_path_budget]
    transition["regression_count"] = len(regressions)
    state_change_score = _float_value(transition.get("state_change_score"), 0.0)
    progress_epsilon = _float_value(controls.get("no_progress_progress_epsilon"), 0.0)
    min_state_change_score = int(controls.get("min_state_change_score_for_progress", 1) or 1)
    no_progress = (
        progress_delta <= progress_epsilon
        and state_change_score < float(min_state_change_score)
        and not regressions
    )
    transition["no_progress"] = bool(no_progress)
    severity_weight = _float_value(controls.get("regression_severity_weight"), 1.0)
    recovery_credit = _float_value(controls.get("progress_recovery_credit"), 1.0)
    severity_score = max(0.0, len(regressions) * severity_weight) - max(0.0, state_change_score * recovery_credit)
    transition["severity_score"] = round(severity_score, 3)
    if regressions:
        transition["severity_band"] = "regressive"
    elif no_progress:
        transition["severity_band"] = "stalled"
    elif progress_delta > progress_epsilon:
        transition["severity_band"] = "improving"
    else:
        transition["severity_band"] = "stable"
    transition["recovery_opportunity"] = bool(regressions or no_progress)
    return transition


def state_estimation_policy_bias(
    latent_state_summary: dict[str, object],
    command: str,
    policy_controls: dict[str, object] | None = None,
) -> int:
    controls = _normalize_policy_controls(policy_controls or {})
    normalized = str(command).strip()
    if not normalized:
        return 0
    active_paths = {
        str(path).strip()
        for path in latent_state_summary.get("active_paths", [])
        if str(path).strip()
    }
    risk_band = str(latent_state_summary.get("risk_band", "stable"))
    progress_band = str(latent_state_summary.get("progress_band", "flat"))
    retrieval_mode = str(latent_state_summary.get("retrieval_mode", "blind"))
    learned_world_state = (
        dict(latent_state_summary.get("learned_world_state", {}))
        if isinstance(latent_state_summary.get("learned_world_state", {}), dict)
        else {}
    )
    learned_progress_signal = _float_value(learned_world_state.get("progress_signal"), 0.0)
    learned_risk_signal = _float_value(learned_world_state.get("risk_signal"), 0.0)
    score = 0
    if risk_band == "regressive" and any(path in normalized for path in active_paths):
        score += int(controls.get("regressive_path_match_bonus", 2))
        if "rm " in normalized or "unlink " in normalized:
            score += int(controls.get("regressive_cleanup_bonus", 1))
    if risk_band == "blocked" and normalized:
        if "mkdir -p " in normalized or "printf " in normalized or "> " in normalized:
            score += int(controls.get("blocked_command_bonus", 1))
    if progress_band == "advancing" and any(path in normalized for path in active_paths):
        score += int(controls.get("advancing_path_match_bonus", 1))
    if retrieval_mode == "trusted" and any(path in normalized for path in active_paths):
        score += int(controls.get("trusted_retrieval_path_bonus", 1))
    if learned_progress_signal >= 0.55:
        if any(path in normalized for path in active_paths) or "printf " in normalized or "> " in normalized:
            score += int(controls.get("learned_world_progress_bonus", 2))
    if learned_risk_signal >= 0.55:
        if "rm -rf" in normalized or "git clean -fd" in normalized:
            score -= int(controls.get("learned_world_risk_penalty", 2))
    return score


def _proposals(
    metrics: EvalMetrics,
    transition_summary: dict[str, object],
    *,
    focus: str | None = None,
) -> list[dict[str, object]]:
    summary = dict(transition_summary or {})
    proposals: list[dict[str, object]] = []
    if int(summary.get("state_regression_steps", 0) or 0) > 0 or focus == "transition_normalization":
        proposals.append(
            {
                "area": "transition_normalization",
                "priority": 5,
                "reason": "raw transitions are under-describing regressive path changes and recovery opportunities",
                "suggestion": "Normalize transition summaries so regressions, state-change score, and no-progress states are retained explicitly for downstream policy use.",
            }
        )
    if int(summary.get("state_regression_steps", 0) or 0) > 0 or metrics.low_confidence_episodes > 0 or focus == "risk_sensitivity":
        proposals.append(
            {
                "area": "risk_sensitivity",
                "priority": 5,
                "reason": "latent state should separate regressive, blocked, and merely flat episodes more sharply",
                "suggestion": "Tighten risk-band thresholds and carry more active regression paths into latent state so policy can recognize risky commands earlier.",
            }
        )
    if int(summary.get("state_progress_gain_steps", 0) or 0) <= 0 or focus == "recovery_bias":
        proposals.append(
            {
                "area": "recovery_bias",
                "priority": 4,
                "reason": "policy scoring should bias more strongly toward recovery commands after stalls or regressive transitions",
                "suggestion": "Increase command bias for cleanup and remediation actions when latent state reports blocked or regressive active paths.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "transition_normalization",
            "priority": 3,
            "reason": "state estimation should stay explicit even when recent transition failures are sparse",
            "suggestion": "Keep transition summarization, latent-state thresholds, and policy-bias controls explicit so retained state-estimation updates have a stable runtime surface.",
        },
    )


def _normalize_transition_controls(controls: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(controls, dict):
        return {}
    normalized: dict[str, object] = {}
    normalized["no_progress_progress_epsilon"] = max(
        -1.0,
        min(1.0, _float_value(controls.get("no_progress_progress_epsilon"), 0.0)),
    )
    normalized["min_state_change_score_for_progress"] = max(
        0,
        int(_float_value(controls.get("min_state_change_score_for_progress"), 1)),
    )
    normalized["regression_path_budget"] = max(1, int(_float_value(controls.get("regression_path_budget"), 6)))
    normalized["regression_severity_weight"] = max(
        0.0,
        _float_value(controls.get("regression_severity_weight"), 1.0),
    )
    normalized["progress_recovery_credit"] = max(
        0.0,
        _float_value(controls.get("progress_recovery_credit"), 1.0),
    )
    return normalized


def _normalize_latent_controls(controls: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(controls, dict):
        return {}
    normalized: dict[str, object] = {}
    normalized["advancing_completion_ratio"] = max(
        0.0,
        min(1.0, _float_value(controls.get("advancing_completion_ratio"), 0.8)),
    )
    normalized["advancing_progress_delta"] = _float_value(controls.get("advancing_progress_delta"), 0.2)
    normalized["improving_progress_delta"] = _float_value(controls.get("improving_progress_delta"), 0.0)
    normalized["regressing_progress_delta"] = _float_value(controls.get("regressing_progress_delta"), -0.05)
    normalized["regressive_regression_count"] = max(
        1,
        int(_float_value(controls.get("regressive_regression_count"), 1)),
    )
    normalized["blocked_forbidden_count"] = max(
        1,
        int(_float_value(controls.get("blocked_forbidden_count"), 1)),
    )
    normalized["active_path_budget"] = max(1, int(_float_value(controls.get("active_path_budget"), 6)))
    normalized["learned_world_progress_threshold"] = max(
        0.0,
        min(1.0, _float_value(controls.get("learned_world_progress_threshold"), 0.55)),
    )
    normalized["learned_world_risk_threshold"] = max(
        0.0,
        min(1.0, _float_value(controls.get("learned_world_risk_threshold"), 0.55)),
    )
    normalized["learned_world_blend_weight"] = max(
        0.0,
        min(1.0, _float_value(controls.get("learned_world_blend_weight"), 0.6)),
    )
    return normalized


def _normalize_policy_controls(controls: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(controls, dict):
        return {}
    return {
        "regressive_path_match_bonus": max(
            0,
            int(_float_value(controls.get("regressive_path_match_bonus"), 2)),
        ),
        "regressive_cleanup_bonus": max(
            0,
            int(_float_value(controls.get("regressive_cleanup_bonus"), 1)),
        ),
        "blocked_command_bonus": max(0, int(_float_value(controls.get("blocked_command_bonus"), 1))),
        "advancing_path_match_bonus": max(
            0,
            int(_float_value(controls.get("advancing_path_match_bonus"), 1)),
        ),
        "trusted_retrieval_path_bonus": max(
            0,
            int(_float_value(controls.get("trusted_retrieval_path_bonus"), 1)),
        ),
        "learned_world_progress_bonus": max(
            0,
            int(_float_value(controls.get("learned_world_progress_bonus"), 2)),
        ),
        "learned_world_risk_penalty": max(
            0,
            int(_float_value(controls.get("learned_world_risk_penalty"), 2)),
        ),
    }


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
