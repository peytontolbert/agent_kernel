from __future__ import annotations

import hashlib
import json
from typing import Any


def build_latent_state_summary(
    *,
    world_model_summary: dict[str, object],
    latest_transition: dict[str, object] | None,
    task_metadata: dict[str, object] | None,
    recent_history: list[dict[str, object]] | None = None,
    context_control: dict[str, object] | None = None,
    latent_controls: dict[str, object] | None = None,
    learned_world_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = dict(world_model_summary or {})
    transition = dict(latest_transition or {})
    metadata = dict(task_metadata or {})
    history = list(recent_history or [])
    control = dict(context_control or {})
    controls = dict(latent_controls or {})
    learned = dict(learned_world_signal or {})
    completion_ratio = _float_value(summary.get("completion_ratio"), 0.0)
    progress_delta = _float_value(transition.get("progress_delta"), 0.0)
    regressions = [str(value).strip() for value in transition.get("regressions", []) if str(value).strip()]
    missing_expected = [
        str(value).strip() for value in summary.get("missing_expected_artifacts", []) if str(value).strip()
    ]
    present_forbidden = [
        str(value).strip() for value in summary.get("present_forbidden_artifacts", []) if str(value).strip()
    ]
    active_paths = missing_expected[:4] + present_forbidden[:4]
    if regressions:
        active_paths.extend(path for path in regressions[:4] if path not in active_paths)
    path_confidence = _float_value(control.get("path_confidence"), 0.0)
    trust_retrieval = bool(control.get("trust_retrieval", False))
    active_path_budget = max(1, int(controls.get("active_path_budget", 6) or 6))
    learned_progress_signal = _float_value(learned.get("progress_signal"), 0.0)
    learned_risk_signal = _float_value(learned.get("risk_signal"), 0.0)
    heuristic_progress_band = _progress_band(completion_ratio, progress_delta, controls=controls)
    heuristic_risk_band = _risk_band(
        regressions=regressions,
        present_forbidden=present_forbidden,
        no_progress=bool(transition.get("no_progress", False)),
        controls=controls,
    )
    progress_band = _blended_progress_band(
        heuristic_progress_band,
        learned_progress_signal=learned_progress_signal,
        controls=controls,
    )
    risk_band = _blended_risk_band(
        heuristic_risk_band,
        learned_risk_signal=learned_risk_signal,
        controls=controls,
    )
    state_payload = {
        "benchmark_family": str(metadata.get("benchmark_family", "bounded")).strip() or "bounded",
        "completion_ratio": round(completion_ratio, 3),
        "progress_delta": round(progress_delta, 3),
        "no_progress": bool(transition.get("no_progress", False)),
        "regressions": regressions[:6],
        "missing_expected": missing_expected[:6],
        "present_forbidden": present_forbidden[:6],
        "path_confidence": round(path_confidence, 3),
        "trust_retrieval": trust_retrieval,
        "learned_progress_signal": round(learned_progress_signal, 3),
        "learned_risk_signal": round(learned_risk_signal, 3),
        "recent_commands": [
            str(item.get("content", "")).strip()
            for item in history[-3:]
            if isinstance(item, dict) and str(item.get("content", "")).strip()
        ],
    }
    state_id = hashlib.sha256(json.dumps(state_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "state_id": state_id,
        "benchmark_family": state_payload["benchmark_family"],
        "progress_band": progress_band,
        "risk_band": risk_band,
        "retrieval_mode": "trusted" if trust_retrieval else "explore" if path_confidence > 0.0 else "blind",
        "path_confidence": path_confidence,
        "completion_ratio": completion_ratio,
        "transition_signature": {
            "progress_delta": progress_delta,
            "regression_count": len(regressions),
            "no_progress": bool(transition.get("no_progress", False)),
            "severity_band": str(transition.get("severity_band", "")).strip() or "stable",
            "severity_score": _float_value(transition.get("severity_score"), 0.0),
        },
        "learned_world_state": {
            "source": str(learned.get("source", "")).strip(),
            "model_family": str(learned.get("model_family", "")).strip(),
            "progress_signal": learned_progress_signal,
            "risk_signal": learned_risk_signal,
            "world_progress_score": _float_value(learned.get("world_progress_score"), 0.0),
            "world_risk_score": _float_value(learned.get("world_risk_score"), 0.0),
            "decoder_world_progress_score": _float_value(learned.get("decoder_world_progress_score"), 0.0),
            "decoder_world_risk_score": _float_value(learned.get("decoder_world_risk_score"), 0.0),
            "decoder_world_entropy_mean": _float_value(learned.get("decoder_world_entropy_mean"), 0.0),
            "transition_progress_score": _float_value(learned.get("transition_progress_score"), 0.0),
            "transition_regression_score": _float_value(learned.get("transition_regression_score"), 0.0),
            "top_probe_reason": str(learned.get("top_probe_reason", "")).strip(),
            "probe_count": int(learned.get("probe_count", 0) or 0),
            "controller_belief": {
                "recover": _float_value(
                    _mapping_value(learned.get("controller_belief"), "recover"),
                    0.0,
                ),
                "continue": _float_value(
                    _mapping_value(learned.get("controller_belief"), "continue"),
                    0.0,
                ),
                "stop": _float_value(
                    _mapping_value(learned.get("controller_belief"), "stop"),
                    0.0,
                ),
            },
            "controller_mode": str(learned.get("controller_mode", "")).strip(),
            "controller_mode_probability": _float_value(learned.get("controller_mode_probability"), 0.0),
            "controller_expected_world_belief": [
                round(_float_value(value, 0.0), 4)
                for value in learned.get("controller_expected_world_belief", [])
                if _is_float_like(value)
            ],
            "controller_expected_world_top_states": [
                int(value)
                for value in learned.get("controller_expected_world_top_states", [])
                if _is_int_like(value)
            ],
            "controller_expected_world_top_state_probs": [
                round(_float_value(value, 0.0), 4)
                for value in learned.get("controller_expected_world_top_state_probs", [])
                if _is_float_like(value)
            ],
            "controller_expected_world_entropy_mean": _float_value(
                learned.get("controller_expected_world_entropy_mean"),
                0.0,
            ),
            "controller_expected_decoder_world_belief": [
                round(_float_value(value, 0.0), 4)
                for value in learned.get("controller_expected_decoder_world_belief", [])
                if _is_float_like(value)
            ],
            "controller_expected_decoder_world_top_states": [
                int(value)
                for value in learned.get("controller_expected_decoder_world_top_states", [])
                if _is_int_like(value)
            ],
            "controller_expected_decoder_world_top_state_probs": [
                round(_float_value(value, 0.0), 4)
                for value in learned.get("controller_expected_decoder_world_top_state_probs", [])
                if _is_float_like(value)
            ],
            "controller_expected_decoder_world_entropy_mean": _float_value(
                learned.get("controller_expected_decoder_world_entropy_mean"),
                0.0,
            ),
            "world_prior_backend": str(learned.get("world_prior_backend", "")).strip(),
            "world_prior_top_state": int(learned.get("world_prior_top_state", -1) or -1),
            "world_prior_top_probability": _float_value(learned.get("world_prior_top_probability"), 0.0),
            "world_prior_horizon_hint": str(learned.get("world_prior_horizon_hint", "")).strip(),
            "world_prior_bias_strength": _float_value(learned.get("world_prior_bias_strength"), 1.0),
            "world_profile_horizons": [
                int(value)
                for value in learned.get("world_profile_horizons", [])
                if _is_int_like(value)
            ],
            "world_signature_token_count": int(learned.get("world_signature_token_count", 0) or 0),
            "world_transition_family": str(learned.get("world_transition_family", "")).strip(),
            "world_transition_bandwidth": int(learned.get("world_transition_bandwidth", 0) or 0),
            "world_transition_gate": _float_value(learned.get("world_transition_gate"), 0.0),
            "world_final_entropy_mean": _float_value(learned.get("world_final_entropy_mean"), 0.0),
            "ssm_last_state_norm_mean": _float_value(learned.get("ssm_last_state_norm_mean"), 0.0),
            "ssm_pooled_state_norm_mean": _float_value(learned.get("ssm_pooled_state_norm_mean"), 0.0),
        },
        "active_paths": active_paths[:active_path_budget],
        "recent_command_count": len(state_payload["recent_commands"]),
    }


def latent_command_bias(latent_state_summary: dict[str, object], command: str) -> int:
    normalized = str(command).strip()
    if not normalized:
        return 0
    risk_band = str(latent_state_summary.get("risk_band", "stable"))
    active_paths = {
        str(path).strip()
        for path in latent_state_summary.get("active_paths", [])
        if str(path).strip()
    }
    progress_band = str(latent_state_summary.get("progress_band", "flat"))
    learned = latent_state_summary.get("learned_world_state", {})
    learned = learned if isinstance(learned, dict) else {}
    learned_progress_signal = _learned_progress_signal(learned)
    learned_risk_signal = _learned_risk_signal(learned)
    targets_active_path = any(path in normalized for path in active_paths)
    write_like_command = any(token in normalized for token in ("mkdir -p ", "printf ", "> ", "touch "))
    cleanup_like_command = "rm " in normalized or "unlink " in normalized
    risky_broad_cleanup = (
        "rm -rf" in normalized
        or "git reset --hard" in normalized
        or "git checkout --" in normalized
    )
    score = 0
    if risk_band == "regressive" and any(path in normalized for path in active_paths):
        score += 2
        if "rm " in normalized or "unlink " in normalized:
            score += 1
    if risk_band == "blocked" and normalized:
        if "mkdir -p " in normalized or "printf " in normalized or "> " in normalized:
            score += 1
    if progress_band == "advancing" and any(path in normalized for path in active_paths):
        score += 1
    if learned_progress_signal >= 0.55 and targets_active_path:
        score += 1
        if write_like_command:
            score += 1
    if learned_risk_signal >= 0.55:
        if targets_active_path:
            score += 1
            if cleanup_like_command:
                score += 2
        if risky_broad_cleanup:
            score -= 4
    return score


def _progress_band(
    completion_ratio: float,
    progress_delta: float,
    *,
    controls: dict[str, object] | None = None,
) -> str:
    config = dict(controls or {})
    advancing_completion_ratio = _float_value(config.get("advancing_completion_ratio"), 0.8)
    advancing_progress_delta = _float_value(config.get("advancing_progress_delta"), 0.2)
    improving_progress_delta = _float_value(config.get("improving_progress_delta"), 0.0)
    regressing_progress_delta = _float_value(config.get("regressing_progress_delta"), -0.05)
    if progress_delta > advancing_progress_delta or completion_ratio >= advancing_completion_ratio:
        return "advancing"
    if progress_delta > improving_progress_delta:
        return "improving"
    if progress_delta < regressing_progress_delta:
        return "regressing"
    return "flat"


def _risk_band(
    *,
    regressions: list[str],
    present_forbidden: list[str],
    no_progress: bool,
    controls: dict[str, object] | None = None,
) -> str:
    config = dict(controls or {})
    regressive_regression_count = max(1, int(config.get("regressive_regression_count", 1) or 1))
    blocked_forbidden_count = max(1, int(config.get("blocked_forbidden_count", 1) or 1))
    if len(regressions) >= regressive_regression_count:
        return "regressive"
    if len(present_forbidden) >= blocked_forbidden_count:
        return "blocked"
    if no_progress:
        return "stalled"
    return "stable"


def _blended_progress_band(
    heuristic_progress_band: str,
    *,
    learned_progress_signal: float,
    controls: dict[str, object] | None = None,
) -> str:
    config = dict(controls or {})
    threshold = _float_value(config.get("learned_world_progress_threshold"), 0.55)
    blend = _float_value(config.get("learned_world_blend_weight"), 0.6)
    if blend <= 0.0 or learned_progress_signal < threshold:
        return heuristic_progress_band
    if heuristic_progress_band in {"flat", "regressing"}:
        return "improving"
    return "advancing"


def _blended_risk_band(
    heuristic_risk_band: str,
    *,
    learned_risk_signal: float,
    controls: dict[str, object] | None = None,
) -> str:
    config = dict(controls or {})
    threshold = _float_value(config.get("learned_world_risk_threshold"), 0.55)
    blend = _float_value(config.get("learned_world_blend_weight"), 0.6)
    if blend <= 0.0 or learned_risk_signal < threshold:
        return heuristic_risk_band
    if heuristic_risk_band == "stable":
        return "regressive"
    if heuristic_risk_band == "stalled":
        return "blocked"
    return heuristic_risk_band


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _learned_progress_signal(learned_world_state: dict[str, object]) -> float:
    return max(
        0.0,
        min(
            1.0,
            max(
                _float_value(learned_world_state.get("progress_signal"), 0.0),
                _float_value(learned_world_state.get("world_progress_score"), 0.0),
                _float_value(learned_world_state.get("decoder_world_progress_score"), 0.0),
                _float_value(learned_world_state.get("transition_progress_score"), 0.0),
            ),
        ),
    )


def _learned_risk_signal(learned_world_state: dict[str, object]) -> float:
    return max(
        0.0,
        min(
            1.0,
            max(
                _float_value(learned_world_state.get("risk_signal"), 0.0),
                _float_value(learned_world_state.get("world_risk_score"), 0.0),
                _float_value(learned_world_state.get("decoder_world_risk_score"), 0.0),
                _float_value(learned_world_state.get("transition_regression_score"), 0.0),
            ),
        ),
    )


def _is_int_like(value: object) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_float_like(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _mapping_value(value: object, key: str) -> object:
    if not isinstance(value, dict):
        return 0.0
    return value.get(key, 0.0)
