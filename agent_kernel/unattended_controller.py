from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .kernel_catalog import kernel_catalog_record_list, kernel_catalog_string_list, kernel_catalog_string_set


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_FOCUSES = tuple(kernel_catalog_string_list("unattended_controller", "focuses"))
_STATE_FEATURE_ORDER = tuple(kernel_catalog_string_list("unattended_controller", "state_feature_order"))
_ACTION_FEATURE_ORDER = tuple(kernel_catalog_string_list("unattended_controller", "action_feature_order"))
_TRANSITION_CONTEXT_STATE_FEATURES = tuple(
    kernel_catalog_string_list("unattended_controller", "transition_context_state_features")
)
_PRIORITY_BROAD_REQUIRED_FAMILIES = kernel_catalog_string_set(
    "unattended_controller",
    "priority_broad_required_families",
)
_RECENT_STATE_FEATURE_MEMORY_LIMIT = 12
_DISCOVERED_STATE_FEATURE_SPECS = tuple(kernel_catalog_record_list("unattended_controller", "discovered_state_features"))
_DISCOVERED_STATE_FEATURES = tuple(
    token
    for token in (str(item.get("name", "")).strip() for item in _DISCOVERED_STATE_FEATURE_SPECS)
    if token and token not in set(_STATE_FEATURE_ORDER)
)
_ALL_STATE_FEATURE_ORDER = (*_STATE_FEATURE_ORDER, *_DISCOVERED_STATE_FEATURES)
_FEATURE_REWARD_WEIGHTS = {
    str(item.get("name", "")).strip(): _safe_float(item.get("reward_weight"))
    for item in _DISCOVERED_STATE_FEATURE_SPECS
    if str(item.get("name", "")).strip()
}
_BINARY_FEATURES = kernel_catalog_string_set("unattended_controller", "binary_features") | {
    str(item.get("name", "")).strip()
    for item in _DISCOVERED_STATE_FEATURE_SPECS
    if bool(item.get("binary", False)) and str(item.get("name", "")).strip()
}
_NONNEGATIVE_FEATURES = kernel_catalog_string_set("unattended_controller", "nonnegative_features") | {
    str(item.get("name", "")).strip()
    for item in _DISCOVERED_STATE_FEATURE_SPECS
    if bool(item.get("nonnegative", False)) and str(item.get("name", "")).strip()
}
_STRUCTURAL_CLASS_SIGNAL_KEYS = (
    "repo_semantics",
    "artifact_kinds",
    "workflow_shape",
    "contract_shape",
    "workflow_guard",
    "semantic_verifier",
    "discovered_structural_classes",
)
_STRUCTURAL_SIGNAL_FAMILY_ALIASES = {
    "project": "project",
    "repository": "repository",
    "integration": "integration",
    "tooling": "tooling",
    "workflow": "workflow",
    "cleanup": "repo_chore",
    "shared_repo": "integration",
    "validation": "integration",
    "semantic_verifier": "integration",
    "workspace_acceptance": "repository",
    "command_acceptance": "workflow",
    "shared_repo_parallel": "integration",
    "multi_branch_validation": "integration",
    "multi_artifact_workspace": "repository",
    "command_driven": "workflow",
}


def _normalize_benchmark_families(values: object) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return normalized


def _normalize_signal_tokens(values: object) -> list[str]:
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return []
    return _normalize_benchmark_families(values)


def _unique_signal_tokens(*groups: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for value in group:
            token = str(value).strip()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
    return ordered


def _canonical_structural_class_id(prefix: str, signals: Sequence[str]) -> str:
    normalized_prefix = str(prefix).strip().lower().replace(" ", "_") or "class"
    normalized_signals = [str(signal).strip().lower().replace(" ", "_") for signal in signals if str(signal).strip()]
    if not normalized_signals:
        return normalized_prefix
    return f"{normalized_prefix}:{'+'.join(normalized_signals)}"


def _structural_class_record(
    *,
    class_kind: str,
    derivation_basis: Sequence[str],
    repo_topology_signals: Sequence[str],
    verifier_shape_signals: Sequence[str],
    edit_pattern_signals: Sequence[str],
    workflow_shape_signals: Sequence[str],
) -> dict[str, object]:
    return {
        "class_id": _canonical_structural_class_id(
            class_kind,
            _unique_signal_tokens(
                repo_topology_signals,
                verifier_shape_signals,
                edit_pattern_signals,
                workflow_shape_signals,
            ),
        ),
        "class_kind": str(class_kind).strip(),
        "derivation_basis": [str(value).strip() for value in derivation_basis if str(value).strip()],
        "repo_topology_signals": [str(value).strip() for value in repo_topology_signals if str(value).strip()],
        "verifier_shape_signals": [str(value).strip() for value in verifier_shape_signals if str(value).strip()],
        "edit_pattern_signals": [str(value).strip() for value in edit_pattern_signals if str(value).strip()],
        "workflow_shape_signals": [str(value).strip() for value in workflow_shape_signals if str(value).strip()],
    }


def _merge_structural_class_records(*groups: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for group in groups:
        for raw_record in group:
            if not isinstance(raw_record, Mapping):
                continue
            class_kind = str(raw_record.get("class_kind", "")).strip()
            class_id = str(raw_record.get("class_id", "")).strip()
            if not class_kind and not class_id:
                continue
            repo_topology_signals = _normalize_signal_tokens(raw_record.get("repo_topology_signals", []))
            verifier_shape_signals = _normalize_signal_tokens(raw_record.get("verifier_shape_signals", []))
            edit_pattern_signals = _normalize_signal_tokens(raw_record.get("edit_pattern_signals", []))
            workflow_shape_signals = _normalize_signal_tokens(raw_record.get("workflow_shape_signals", []))
            derivation_basis = _normalize_signal_tokens(raw_record.get("derivation_basis", []))
            record = _structural_class_record(
                class_kind=class_kind or class_id.split(":", 1)[0],
                derivation_basis=derivation_basis,
                repo_topology_signals=repo_topology_signals,
                verifier_shape_signals=verifier_shape_signals,
                edit_pattern_signals=edit_pattern_signals,
                workflow_shape_signals=workflow_shape_signals,
            )
            merged[record["class_id"]] = record
    return [merged[key] for key in sorted(merged)]


def discover_structural_classes(payload: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    raw_classes = payload.get("discovered_structural_classes", [])
    existing_records = raw_classes if isinstance(raw_classes, Sequence) and not isinstance(raw_classes, (str, bytes)) else []
    repo_semantics = _normalize_signal_tokens(payload.get("repo_semantics", []))
    artifact_kinds = _normalize_signal_tokens(payload.get("artifact_kinds", []))
    workflow_guard = payload.get("workflow_guard", {})
    workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, Mapping) else {}
    semantic_verifier = payload.get("semantic_verifier", {})
    semantic_verifier = dict(semantic_verifier) if isinstance(semantic_verifier, Mapping) else {}
    workflow_shape = str(payload.get("workflow_shape", "")).strip().lower()
    contract_shape = str(payload.get("contract_shape", "")).strip().lower()

    repo_topology_signals = list(repo_semantics)
    verifier_shape_signals = [contract_shape] if contract_shape else []
    edit_pattern_signals = list(artifact_kinds)
    workflow_shape_signals = [workflow_shape] if workflow_shape else []

    if bool(workflow_guard.get("shared_repo_id")) and "shared_repo" not in repo_topology_signals:
        repo_topology_signals.append("shared_repo")
    if bool(workflow_guard.get("shared_repo_id")) and "shared_repo_parallel" not in workflow_shape_signals:
        workflow_shape_signals.append("shared_repo_parallel")
    required_merged_branches = _normalize_signal_tokens(semantic_verifier.get("required_merged_branches", []))
    if required_merged_branches and "semantic_verifier" not in verifier_shape_signals:
        verifier_shape_signals.append("semantic_verifier")
    if required_merged_branches and "multi_branch_validation" not in workflow_shape_signals:
        workflow_shape_signals.append("multi_branch_validation")

    derived_records: list[dict[str, object]] = []
    if repo_topology_signals:
        derived_records.append(
            _structural_class_record(
                class_kind="repo_topology",
                derivation_basis=["task_metadata:repo_semantics", "task_metadata:workflow_guard"],
                repo_topology_signals=repo_topology_signals,
                verifier_shape_signals=[],
                edit_pattern_signals=[],
                workflow_shape_signals=[],
            )
        )
    if verifier_shape_signals:
        derived_records.append(
            _structural_class_record(
                class_kind="verifier_shape",
                derivation_basis=["task_metadata:contract_shape", "task_metadata:semantic_verifier"],
                repo_topology_signals=[],
                verifier_shape_signals=verifier_shape_signals,
                edit_pattern_signals=[],
                workflow_shape_signals=[],
            )
        )
    if edit_pattern_signals:
        derived_records.append(
            _structural_class_record(
                class_kind="edit_pattern",
                derivation_basis=["task_metadata:artifact_kinds"],
                repo_topology_signals=[],
                verifier_shape_signals=[],
                edit_pattern_signals=edit_pattern_signals,
                workflow_shape_signals=[],
            )
        )
    if workflow_shape_signals:
        derived_records.append(
            _structural_class_record(
                class_kind="workflow_shape",
                derivation_basis=["task_metadata:workflow_shape", "task_metadata:semantic_verifier", "task_metadata:workflow_guard"],
                repo_topology_signals=[],
                verifier_shape_signals=[],
                edit_pattern_signals=[],
                workflow_shape_signals=workflow_shape_signals,
            )
        )
    return _merge_structural_class_records(existing_records, derived_records)


def structural_class_family_aliases(classes: Sequence[Mapping[str, object]]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for record in classes:
        if not isinstance(record, Mapping):
            continue
        for key in (
            "repo_topology_signals",
            "verifier_shape_signals",
            "edit_pattern_signals",
            "workflow_shape_signals",
        ):
            for raw_value in _normalize_signal_tokens(record.get(key, [])):
                alias = _STRUCTURAL_SIGNAL_FAMILY_ALIASES.get(raw_value)
                if alias and alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
    return aliases


def structural_class_summary(*payloads: Mapping[str, object] | None) -> dict[str, object]:
    records = _merge_structural_class_records(
        *(
            discover_structural_classes(payload)
            for payload in payloads
            if isinstance(payload, Mapping)
        )
    )
    family_aliases = structural_class_family_aliases(records)
    return {
        "classes": records,
        "class_ids": [str(record.get("class_id", "")).strip() for record in records if str(record.get("class_id", "")).strip()],
        "class_kinds": [
            str(record.get("class_kind", "")).strip()
            for record in records
            if str(record.get("class_kind", "")).strip()
        ],
        "family_aliases": family_aliases,
    }


def _ordered_feature_names(*feature_maps: Mapping[str, object] | None) -> tuple[str, ...]:
    ordered = list(_ALL_STATE_FEATURE_ORDER)
    seen = set(ordered)
    for feature_map in feature_maps:
        if not isinstance(feature_map, Mapping):
            continue
        for raw_name in feature_map:
            name = str(raw_name).strip()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def _declared_controller_features(*signals: Mapping[str, object] | None) -> dict[str, float]:
    features: dict[str, float] = {}
    for signal in signals:
        if not isinstance(signal, Mapping):
            continue
        for key in ("controller_features", "discovered_features"):
            raw_features = signal.get(key, {})
            if not isinstance(raw_features, Mapping):
                continue
            for raw_name, raw_value in raw_features.items():
                name = str(raw_name).strip()
                if not name:
                    continue
                features[name] = _safe_float(raw_value)
    return features


def _apply_feature_constraints(features: Mapping[str, object] | None) -> dict[str, float]:
    normalized = {
        str(name).strip(): _safe_float(value)
        for name, value in dict(features or {}).items()
        if str(name).strip()
    }
    normalized["bias"] = 1.0
    for feature in _BINARY_FEATURES:
        if feature in normalized:
            normalized[feature] = max(0.0, min(1.0, normalized.get(feature, 0.0)))
    for feature in _NONNEGATIVE_FEATURES:
        if feature in normalized:
            normalized[feature] = max(0.0, normalized.get(feature, 0.0))
    return normalized


def default_controller_state(
    *,
    gamma: float = 0.85,
    value_learning_rate: float = 0.08,
    transition_learning_rate: float = 0.15,
    exploration_bonus: float = 2.5,
    uncertainty_penalty: float = 1.5,
    min_action_support: int = 3,
    thin_evidence_penalty: float = 2.0,
    support_confidence_power: float = 0.5,
    rollout_depth: int = 2,
    rollout_beam_width: int = 6,
    repeat_action_penalty: float = 5.0,
    state_repeat_penalty: float = 0.25,
    state_novelty_bonus: float = 0.1,
) -> dict[str, object]:
    return {
        "spec_version": "asi_v1",
        "controller_kind": "unattended_model_based_controller",
        "gamma": float(gamma),
        "value_learning_rate": float(value_learning_rate),
        "transition_learning_rate": float(transition_learning_rate),
        "exploration_bonus": float(exploration_bonus),
        "uncertainty_penalty": float(uncertainty_penalty),
        "min_action_support": max(1, int(min_action_support)),
        "thin_evidence_penalty": max(0.0, float(thin_evidence_penalty)),
        "support_confidence_power": max(0.1, float(support_confidence_power)),
        "rollout_depth": max(1, int(rollout_depth)),
        "rollout_beam_width": max(1, int(rollout_beam_width)),
        "repeat_action_penalty": max(0.0, float(repeat_action_penalty)),
        "state_repeat_penalty": max(0.0, float(state_repeat_penalty)),
        "state_novelty_bonus": max(0.0, float(state_novelty_bonus)),
        "updates": 0,
        "value_weights": {},
        "policy_value_weights": {},
        "transition_context_weights": {},
        "transition_context_error_ema": {},
        "action_models": {},
        "recent_rewards": [],
        "recent_state_features": [],
        "last_action_key": "",
        "repo_setting_policy_priors": {},
        "strategy_memory_priors": {},
    }


def normalize_controller_state(payload: Mapping[str, object] | None) -> dict[str, object]:
    state = default_controller_state()
    if not isinstance(payload, Mapping):
        return state
    state["gamma"] = max(0.0, min(0.99, _safe_float(payload.get("gamma"), float(state["gamma"]))))
    state["value_learning_rate"] = max(
        0.001,
        min(1.0, _safe_float(payload.get("value_learning_rate"), float(state["value_learning_rate"]))),
    )
    state["transition_learning_rate"] = max(
        0.001,
        min(1.0, _safe_float(payload.get("transition_learning_rate"), float(state["transition_learning_rate"]))),
    )
    state["exploration_bonus"] = max(
        0.0,
        _safe_float(payload.get("exploration_bonus"), float(state["exploration_bonus"])),
    )
    state["uncertainty_penalty"] = max(
        0.0,
        _safe_float(payload.get("uncertainty_penalty"), float(state["uncertainty_penalty"])),
    )
    state["min_action_support"] = max(
        1,
        _safe_int(payload.get("min_action_support"), _safe_int(state["min_action_support"], 3)),
    )
    state["thin_evidence_penalty"] = max(
        0.0,
        _safe_float(payload.get("thin_evidence_penalty"), _safe_float(state["thin_evidence_penalty"], 2.0)),
    )
    state["support_confidence_power"] = max(
        0.1,
        _safe_float(payload.get("support_confidence_power"), _safe_float(state.get("support_confidence_power"), 0.5)),
    )
    state["rollout_depth"] = max(1, _safe_int(payload.get("rollout_depth"), _safe_int(state["rollout_depth"], 2)))
    state["rollout_beam_width"] = max(
        1,
        _safe_int(payload.get("rollout_beam_width"), _safe_int(state["rollout_beam_width"], 6)),
    )
    state["repeat_action_penalty"] = max(
        0.0,
        _safe_float(payload.get("repeat_action_penalty"), _safe_float(state["repeat_action_penalty"], 5.0)),
    )
    state["state_repeat_penalty"] = max(
        0.0,
        _safe_float(payload.get("state_repeat_penalty"), _safe_float(state["state_repeat_penalty"], 0.25)),
    )
    state["state_novelty_bonus"] = max(
        0.0,
        _safe_float(payload.get("state_novelty_bonus"), _safe_float(state["state_novelty_bonus"], 0.1)),
    )
    state["updates"] = max(0, _safe_int(payload.get("updates"), 0))
    value_weights = payload.get("value_weights", {})
    if isinstance(value_weights, Mapping):
        state["value_weights"] = {
            str(key): _safe_float(value)
            for key, value in value_weights.items()
            if str(key).strip()
        }
    policy_value_weights = payload.get("policy_value_weights", {})
    if isinstance(policy_value_weights, Mapping):
        state["policy_value_weights"] = {
            str(key): _safe_float(value)
            for key, value in policy_value_weights.items()
            if str(key).strip()
        }
    transition_context_weights = payload.get("transition_context_weights", {})
    if isinstance(transition_context_weights, Mapping):
        normalized_context_weights: dict[str, dict[str, float]] = {}
        for target_feature, raw_weights in transition_context_weights.items():
            target = str(target_feature).strip()
            if not target or not isinstance(raw_weights, Mapping):
                continue
            normalized_context_weights[target] = {
                str(source_feature): _safe_float(weight)
                for source_feature, weight in raw_weights.items()
                if str(source_feature).strip()
            }
        state["transition_context_weights"] = normalized_context_weights
    transition_context_error_ema = payload.get("transition_context_error_ema", {})
    if isinstance(transition_context_error_ema, Mapping):
        state["transition_context_error_ema"] = {
            str(key): max(0.0, _safe_float(value))
            for key, value in transition_context_error_ema.items()
            if str(key).strip()
        }
    action_models = payload.get("action_models", {})
    if isinstance(action_models, Mapping):
        normalized_models: dict[str, object] = {}
        for action_key, raw_stats in action_models.items():
            token = str(action_key).strip()
            if not token or not isinstance(raw_stats, Mapping):
                continue
            transition_mean = raw_stats.get("transition_mean", {})
            transition_mean_dict = (
                {
                    str(feature): _safe_float(value)
                    for feature, value in transition_mean.items()
                    if str(feature).strip()
                }
                if isinstance(transition_mean, Mapping)
                else {}
            )
            normalized_models[token] = {
                "count": max(0, _safe_int(raw_stats.get("count"), 0)),
                "reward_mean": _safe_float(raw_stats.get("reward_mean")),
                "reward_m2": max(0.0, _safe_float(raw_stats.get("reward_m2"))),
                "transition_mean": transition_mean_dict,
                "transition_error_ema": {
                    str(feature): max(0.0, _safe_float(value))
                    for feature, value in dict(raw_stats.get("transition_error_ema", {})).items()
                    if str(feature).strip()
                }
                if isinstance(raw_stats.get("transition_error_ema", {}), Mapping)
                else {},
                "policy_template": dict(raw_stats.get("policy_template", {}))
                if isinstance(raw_stats.get("policy_template", {}), Mapping)
                else {},
                "last_reward": _safe_float(raw_stats.get("last_reward")),
            }
        state["action_models"] = normalized_models
    recent_rewards = payload.get("recent_rewards", [])
    if isinstance(recent_rewards, Sequence) and not isinstance(recent_rewards, (str, bytes)):
        state["recent_rewards"] = [_safe_float(value) for value in list(recent_rewards)[-20:]]
    recent_state_features = payload.get("recent_state_features", [])
    if isinstance(recent_state_features, Sequence) and not isinstance(recent_state_features, (str, bytes)):
        normalized_recent_states: list[dict[str, float]] = []
        for raw_features in recent_state_features:
            if not isinstance(raw_features, Mapping):
                continue
            normalized_recent_states.append(_latent_world_feature_snapshot(raw_features))
        state["recent_state_features"] = normalized_recent_states[-_RECENT_STATE_FEATURE_MEMORY_LIMIT:]
    state["last_action_key"] = str(payload.get("last_action_key", "")).strip()
    repo_setting_policy_priors = payload.get("repo_setting_policy_priors", {})
    if isinstance(repo_setting_policy_priors, Mapping):
        normalized_priors: dict[str, dict[str, object]] = {}
        for signal, raw_entry in repo_setting_policy_priors.items():
            token = str(signal).strip().lower()
            if not token or not isinstance(raw_entry, Mapping):
                continue
            campaign_width_stats = raw_entry.get("campaign_width_stats", {})
            task_step_floor_stats = raw_entry.get("task_step_floor_stats", {})
            adaptive_search_stats = raw_entry.get("adaptive_search_stats", {})
            normalized_priors[token] = {
                "observations": max(0, _safe_int(raw_entry.get("observations"), 0)),
                "last_outcome_score": _safe_float(raw_entry.get("last_outcome_score")),
                "campaign_width_stats": {
                    "observations": max(0, _safe_int(campaign_width_stats.get("observations"), 0)),
                    "sum_w": max(0.0, _safe_float(campaign_width_stats.get("sum_w"))),
                    "sum_x": _safe_float(campaign_width_stats.get("sum_x")),
                    "sum_y": _safe_float(campaign_width_stats.get("sum_y")),
                    "sum_x2": max(0.0, _safe_float(campaign_width_stats.get("sum_x2"))),
                    "sum_xy": _safe_float(campaign_width_stats.get("sum_xy")),
                }
                if isinstance(campaign_width_stats, Mapping)
                else {
                    "observations": 0,
                    "sum_w": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_xy": 0.0,
                },
                "task_step_floor_stats": {
                    "observations": max(0, _safe_int(task_step_floor_stats.get("observations"), 0)),
                    "sum_w": max(0.0, _safe_float(task_step_floor_stats.get("sum_w"))),
                    "sum_x": _safe_float(task_step_floor_stats.get("sum_x")),
                    "sum_y": _safe_float(task_step_floor_stats.get("sum_y")),
                    "sum_x2": max(0.0, _safe_float(task_step_floor_stats.get("sum_x2"))),
                    "sum_xy": _safe_float(task_step_floor_stats.get("sum_xy")),
                }
                if isinstance(task_step_floor_stats, Mapping)
                else {
                    "observations": 0,
                    "sum_w": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_xy": 0.0,
                },
                "adaptive_search_stats": {
                    "observations": max(0, _safe_int(adaptive_search_stats.get("observations"), 0)),
                    "true_count": max(0, _safe_int(adaptive_search_stats.get("true_count"), 0)),
                    "false_count": max(0, _safe_int(adaptive_search_stats.get("false_count"), 0)),
                    "true_score_sum": _safe_float(adaptive_search_stats.get("true_score_sum")),
                    "false_score_sum": _safe_float(adaptive_search_stats.get("false_score_sum")),
                }
                if isinstance(adaptive_search_stats, Mapping)
                else {
                    "observations": 0,
                    "true_count": 0,
                    "false_count": 0,
                    "true_score_sum": 0.0,
                    "false_score_sum": 0.0,
                },
                "family_priors": {
                    str(family).strip().lower(): {
                        "observations": max(0, _safe_int(family_entry.get("observations"), 0)),
                        "last_outcome_score": _safe_float(family_entry.get("last_outcome_score")),
                        "campaign_width_stats": {
                            "observations": max(
                                0,
                                _safe_int(
                                    dict(family_entry.get("campaign_width_stats", {})).get("observations", 0)
                                ),
                            ),
                            "sum_w": max(
                                0.0,
                                _safe_float(dict(family_entry.get("campaign_width_stats", {})).get("sum_w")),
                            ),
                            "sum_x": _safe_float(dict(family_entry.get("campaign_width_stats", {})).get("sum_x")),
                            "sum_y": _safe_float(dict(family_entry.get("campaign_width_stats", {})).get("sum_y")),
                            "sum_x2": max(
                                0.0,
                                _safe_float(dict(family_entry.get("campaign_width_stats", {})).get("sum_x2")),
                            ),
                            "sum_xy": _safe_float(dict(family_entry.get("campaign_width_stats", {})).get("sum_xy")),
                        }
                        if isinstance(family_entry, Mapping)
                        and isinstance(family_entry.get("campaign_width_stats", {}), Mapping)
                        else {
                            "observations": 0,
                            "sum_w": 0.0,
                            "sum_x": 0.0,
                            "sum_y": 0.0,
                            "sum_x2": 0.0,
                            "sum_xy": 0.0,
                        },
                        "task_step_floor_stats": {
                            "observations": max(
                                0,
                                _safe_int(
                                    dict(family_entry.get("task_step_floor_stats", {})).get("observations", 0)
                                ),
                            ),
                            "sum_w": max(
                                0.0,
                                _safe_float(dict(family_entry.get("task_step_floor_stats", {})).get("sum_w")),
                            ),
                            "sum_x": _safe_float(dict(family_entry.get("task_step_floor_stats", {})).get("sum_x")),
                            "sum_y": _safe_float(dict(family_entry.get("task_step_floor_stats", {})).get("sum_y")),
                            "sum_x2": max(
                                0.0,
                                _safe_float(dict(family_entry.get("task_step_floor_stats", {})).get("sum_x2")),
                            ),
                            "sum_xy": _safe_float(dict(family_entry.get("task_step_floor_stats", {})).get("sum_xy")),
                        }
                        if isinstance(family_entry, Mapping)
                        and isinstance(family_entry.get("task_step_floor_stats", {}), Mapping)
                        else {
                            "observations": 0,
                            "sum_w": 0.0,
                            "sum_x": 0.0,
                            "sum_y": 0.0,
                            "sum_x2": 0.0,
                            "sum_xy": 0.0,
                        },
                        "adaptive_search_stats": {
                            "observations": max(
                                0,
                                _safe_int(
                                    dict(family_entry.get("adaptive_search_stats", {})).get("observations", 0)
                                ),
                            ),
                            "true_count": max(
                                0,
                                _safe_int(dict(family_entry.get("adaptive_search_stats", {})).get("true_count", 0)),
                            ),
                            "false_count": max(
                                0,
                                _safe_int(dict(family_entry.get("adaptive_search_stats", {})).get("false_count", 0)),
                            ),
                            "true_score_sum": _safe_float(
                                dict(family_entry.get("adaptive_search_stats", {})).get("true_score_sum")
                            ),
                            "false_score_sum": _safe_float(
                                dict(family_entry.get("adaptive_search_stats", {})).get("false_score_sum")
                            ),
                        }
                        if isinstance(family_entry, Mapping)
                        and isinstance(family_entry.get("adaptive_search_stats", {}), Mapping)
                        else {
                            "observations": 0,
                            "true_count": 0,
                            "false_count": 0,
                            "true_score_sum": 0.0,
                            "false_score_sum": 0.0,
                        },
                    }
                    for family, family_entry in dict(raw_entry.get("family_priors", {})).items()
                    if str(family).strip() and isinstance(family_entry, Mapping)
                }
                if isinstance(raw_entry.get("family_priors", {}), Mapping)
                else {},
            }
        state["repo_setting_policy_priors"] = normalized_priors
    strategy_memory_priors = payload.get("strategy_memory_priors", {})
    if isinstance(strategy_memory_priors, Mapping):
        normalized_strategy_priors: dict[str, object] = {}
        for key, value in strategy_memory_priors.items():
            token = str(key).strip()
            if not token:
                continue
            if token in {
                "retained_family_gains",
                "recent_rejects_by_family",
                "retained_subsystems",
                "recent_rejects_by_subsystem",
            } and isinstance(value, Mapping):
                normalized_strategy_priors[token] = {
                    str(name).strip(): (
                        max(0, _safe_int(item))
                        if token.startswith("recent_rejects")
                        else max(0.0, _safe_float(item))
                    )
                    for name, item in value.items()
                    if str(name).strip()
                }
                continue
            if token in {"node_count", "retained_count", "rejected_count", "recent_rejects", "recent_retains"}:
                normalized_strategy_priors[token] = max(0, _safe_int(value))
                continue
            if token == "best_retained_gain":
                normalized_strategy_priors[token] = max(0.0, _safe_float(value))
                continue
        state["strategy_memory_priors"] = normalized_strategy_priors
    return state


def action_key_for_policy(policy: Mapping[str, object]) -> str:
    focus = str(policy.get("focus", "balanced")).strip() or "balanced"
    adaptive = 1 if bool(policy.get("adaptive_search", False)) else 0
    cycles = max(1, _safe_int(policy.get("cycles"), 1))
    campaign_width = max(1, _safe_int(policy.get("campaign_width"), 1))
    variant_width = max(1, _safe_int(policy.get("variant_width"), 1))
    task_limit = max(1, _safe_int(policy.get("task_limit"), 64))
    task_step_floor = max(1, _safe_int(policy.get("task_step_floor"), 1))
    priority_families = ",".join(_normalize_benchmark_families(policy.get("priority_benchmark_families", [])))
    return (
        f"focus={focus}|adaptive={adaptive}|cycles={cycles}|campaign={campaign_width}|"
        f"variant={variant_width}|task_limit={task_limit}|step_floor={task_step_floor}|priority={priority_families}"
    )


def _policy_features(policy: Mapping[str, object]) -> dict[str, float]:
    focus = str(policy.get("focus", "balanced")).strip() or "balanced"
    cycles = max(1, _safe_int(policy.get("cycles"), 1))
    campaign_width = max(1, _safe_int(policy.get("campaign_width"), 1))
    variant_width = max(1, _safe_int(policy.get("variant_width"), 1))
    task_limit = max(1, _safe_int(policy.get("task_limit"), 64))
    task_step_floor = max(1, _safe_int(policy.get("task_step_floor"), 1))
    priority_families = _normalize_benchmark_families(policy.get("priority_benchmark_families", []))
    return {
        "bias": 1.0,
        "focus_balanced": 1.0 if focus == "balanced" else 0.0,
        "focus_recovery_alignment": 1.0 if focus == "recovery_alignment" else 0.0,
        "focus_discovered_task_adaptation": 1.0 if focus == "discovered_task_adaptation" else 0.0,
        "adaptive_search": 1.0 if bool(policy.get("adaptive_search", False)) else 0.0,
        "cycles_norm": min(1.0, float(cycles) / 4.0),
        "campaign_width_norm": min(1.0, float(campaign_width) / 4.0),
        "variant_width_norm": min(1.0, float(variant_width) / 4.0),
        "task_limit_log2_norm": min(1.0, math.log2(float(task_limit)) / 12.0),
        "task_step_floor_log2_norm": min(1.0, math.log2(float(task_step_floor + 1)) / 12.0),
        "priority_family_count_norm": min(1.0, float(len(priority_families)) / 3.0),
        "priority_broad_required": 1.0 if set(priority_families) & _PRIORITY_BROAD_REQUIRED_FAMILIES else 0.0,
    }


def _policy_pressure_alignment_bonus(
    observation_features: Mapping[str, object] | None,
    policy: Mapping[str, object],
) -> float:
    features = observation_features if isinstance(observation_features, Mapping) else {}
    policy_features = _policy_features(policy)
    subsystem_reject_pressure = _safe_float(features.get("subsystem_reject_pressure"))
    subsystem_retain_pressure = _safe_float(features.get("subsystem_retain_pressure"))
    reject_dominance = max(0.0, subsystem_reject_pressure - subsystem_retain_pressure)
    breadth_pressure = max(
        0.0,
        _safe_float(features.get("breadth_pressure")) + _safe_float(features.get("priority_family_yield_gap")),
    )
    no_retained_gain_pressure = max(0.0, _safe_float(features.get("no_retained_gain_pressure")))
    retained_gain_conversion_pressure = max(0.0, _safe_float(features.get("retained_gain_conversion_pressure")))
    broad_observe_then_retrieval_first = max(
        0.0,
        _safe_float(features.get("broad_observe_then_retrieval_first")),
    )
    live_decision_credit_gap = max(0.0, _safe_float(features.get("live_decision_credit_gap")))
    observability_pressure = max(0.0, _safe_float(features.get("observability_pressure")))
    subsystem_robustness_pressure = max(0.0, _safe_float(features.get("subsystem_robustness_pressure")))
    structural_class_gap = max(0.0, _safe_float(features.get("structural_class_gap")))
    structural_class_alignment = max(0.0, _safe_float(features.get("structural_class_alignment")))
    retrieval_pressure = max(
        0.0,
        _safe_float(features.get("low_confidence_pressure")) + _safe_float(features.get("retrieval_regression")),
    )
    focus_discovered = _safe_float(policy_features.get("focus_discovered_task_adaptation"))
    focus_recovery = _safe_float(policy_features.get("focus_recovery_alignment"))
    adaptive_search = _safe_float(policy_features.get("adaptive_search"))
    campaign_width = _safe_float(policy_features.get("campaign_width_norm"))
    variant_width = _safe_float(policy_features.get("variant_width_norm"))
    priority_breadth = _safe_float(policy_features.get("priority_broad_required"))

    bonus = 0.0
    if reject_dominance > 0.0:
        bonus += reject_dominance * (
            1.25 * adaptive_search
            + 1.0 * focus_discovered
            + 0.85 * focus_recovery
            + 0.75 * priority_breadth
            + 0.5 * campaign_width
            + 0.5 * variant_width
        )
    if breadth_pressure > 0.0:
        bonus += breadth_pressure * (
            0.7 * focus_discovered
            + 0.6 * adaptive_search
            + 0.5 * priority_breadth
            + 0.45 * campaign_width
            + 0.35 * variant_width
        )
    if no_retained_gain_pressure > 0.0:
        bonus += no_retained_gain_pressure * (
            1.8 * adaptive_search
            + 1.6 * focus_discovered
            + 1.25 * focus_recovery
            + 1.15 * priority_breadth
            + 0.85 * campaign_width
            + 0.85 * variant_width
            + 0.5 * breadth_pressure
            + 0.4 * retrieval_pressure
        )
    if retained_gain_conversion_pressure > 0.0:
        bonus += retained_gain_conversion_pressure * (
            1.35 * adaptive_search
            + 1.25 * focus_discovered
            + 0.95 * focus_recovery
            + 0.9 * priority_breadth
            + 0.65 * campaign_width
            + 0.65 * variant_width
        )
    if broad_observe_then_retrieval_first > 0.0:
        bonus += broad_observe_then_retrieval_first * (
            1.6 * focus_discovered
            + 1.1 * adaptive_search
            + 0.95 * priority_breadth
            + 0.8 * campaign_width
            + 0.65 * variant_width
            + 0.35 * no_retained_gain_pressure
        )
    if live_decision_credit_gap > 0.0:
        bonus += live_decision_credit_gap * (
            1.2 * adaptive_search
            + 1.05 * focus_discovered
            + 0.75 * campaign_width
            + 0.6 * variant_width
        )
    if observability_pressure > 0.0:
        bonus += observability_pressure * (
            1.25 * adaptive_search
            + 1.1 * focus_discovered
            + 0.85 * focus_recovery
            + 0.8 * campaign_width
            + 0.7 * variant_width
            + 0.35 * priority_breadth
        )
    if subsystem_robustness_pressure > 0.0:
        bonus += subsystem_robustness_pressure * (
            1.2 * adaptive_search
            + 1.05 * focus_recovery
            + 0.9 * focus_discovered
            + 0.65 * campaign_width
            + 0.5 * variant_width
        )
    if structural_class_gap > 0.0:
        bonus += structural_class_gap * (
            1.25 * adaptive_search
            + 1.15 * focus_discovered
            + 0.8 * priority_breadth
            + 0.6 * campaign_width
            + 0.45 * variant_width
        )
    if structural_class_alignment > 0.0:
        bonus += structural_class_alignment * (
            0.45 * focus_discovered
            + 0.35 * adaptive_search
            + 0.25 * priority_breadth
        )
    if retrieval_pressure > 0.0:
        bonus += retrieval_pressure * (
            0.9 * focus_discovered
            + 0.75 * adaptive_search
            + 0.55 * variant_width
        )
    return bonus


def _transition_context_features(
    state_features: Mapping[str, float],
    policy: Mapping[str, object],
) -> dict[str, float]:
    features: dict[str, float] = {"context_bias": 1.0}
    for feature in _TRANSITION_CONTEXT_STATE_FEATURES:
        features[f"state::{feature}"] = _safe_float(state_features.get(feature))
    for feature, value in _policy_features(policy).items():
        features[f"policy::{feature}"] = _safe_float(value)
    return features


def build_round_observation(
    *,
    campaign_signal: Mapping[str, object] | None,
    subsystem_signal: Mapping[str, object] | None = None,
    planner_pressure_signal: Mapping[str, object] | None = None,
    liftoff_signal: Mapping[str, object] | None = None,
    round_signal: Mapping[str, object] | None = None,
) -> dict[str, object]:
    campaign = campaign_signal if isinstance(campaign_signal, Mapping) else {}
    subsystem = subsystem_signal if isinstance(subsystem_signal, Mapping) else {}
    pressure = planner_pressure_signal if isinstance(planner_pressure_signal, Mapping) else {}
    liftoff = liftoff_signal if isinstance(liftoff_signal, Mapping) else {}
    round_state = round_signal if isinstance(round_signal, Mapping) else {}
    worst_family_delta = _safe_float(campaign.get("worst_family_delta"))
    worst_generated_family_delta = _safe_float(campaign.get("worst_generated_family_delta"))
    retained_cycles = max(0, _safe_int(campaign.get("retained_cycles"), 0))
    rejected_cycles = max(0, _safe_int(campaign.get("rejected_cycles"), 0))
    priority_families_with_retained_gain = _normalize_benchmark_families(
        campaign.get("priority_families_with_retained_gain", [])
    )
    priority_families_without_signal = _normalize_benchmark_families(
        campaign.get("priority_families_without_signal", [])
    )
    priority_families_with_signal_but_no_retained_gain = _normalize_benchmark_families(
        campaign.get("priority_families_with_signal_but_no_retained_gain", [])
    )
    priority_families_needing_retained_gain_conversion = _normalize_benchmark_families(
        campaign.get(
            "priority_families_needing_retained_gain_conversion",
            priority_families_with_signal_but_no_retained_gain,
        )
    )
    priority_families = _normalize_benchmark_families(campaign.get("priority_families", []))
    frontier_failure_motif_families = _normalize_benchmark_families(
        pressure.get("frontier_failure_motif_families", [])
    )
    frontier_repo_setting_families = _normalize_benchmark_families(
        pressure.get("frontier_repo_setting_families", [])
    )
    structural_summary = structural_class_summary(campaign, subsystem, pressure, liftoff, round_state)
    structural_family_aliases = _normalize_benchmark_families(structural_summary.get("family_aliases", []))
    structural_family_alias_set = set(structural_family_aliases)
    priority_family_yield_gap = {
        *priority_families_without_signal,
        *priority_families_with_signal_but_no_retained_gain,
    }
    productive_priority_family_count = len(priority_families_with_retained_gain)
    unresolved_priority_family_count = max(
        len(priority_family_yield_gap),
        max(0, len(priority_families) - productive_priority_family_count),
    )
    no_retained_gain_pressure = (
        min(
            1.0,
            float(max(unresolved_priority_family_count, rejected_cycles))
            / float(max(1, min(4, len(priority_families) or 3))),
        )
        if unresolved_priority_family_count > 0 or rejected_cycles > 0
        else 0.0
    )
    target_priority_family_count = max(
        len(priority_families),
        productive_priority_family_count + len(priority_family_yield_gap),
    )
    generalization_gain = min(1.0, float(max(0, productive_priority_family_count - 1)) / 2.0)
    if target_priority_family_count <= 0:
        generalization_gap = 0.0
    else:
        unresolved_generalization = max(0, len(priority_family_yield_gap) - max(0, productive_priority_family_count - 1))
        generalization_gap = min(1.0, float(unresolved_generalization) / float(target_priority_family_count))
    frontier_failure_motif_gain = min(
        1.0,
        float(len(set(priority_families_with_retained_gain) & set(frontier_failure_motif_families))) / 3.0,
    )
    frontier_failure_motif_pressure = min(
        1.0,
        float(len(set(frontier_failure_motif_families) - set(priority_families_with_retained_gain))) / 3.0,
    )
    frontier_repo_setting_gain = min(
        1.0,
        float(len(set(priority_families_with_retained_gain) & set(frontier_repo_setting_families))) / 3.0,
    )
    frontier_repo_setting_pressure = min(
        1.0,
        float(len(set(frontier_repo_setting_families) - set(priority_families_with_retained_gain))) / 3.0,
    )
    productive_depth_retained_cycles = max(0, _safe_int(campaign.get("productive_depth_retained_cycles"), 0))
    depth_drift_cycles = max(0, _safe_int(campaign.get("depth_drift_cycles"), 0))
    long_horizon_retained_cycles = max(0, _safe_int(campaign.get("long_horizon_retained_cycles"), 0))
    average_productive_depth_step_delta = max(
        0.0,
        _safe_float(campaign.get("average_productive_depth_step_delta")),
    )
    average_depth_drift_step_delta = max(
        0.0,
        _safe_float(campaign.get("average_depth_drift_step_delta")),
    )
    productive_depth_gain = min(
        1.0,
        min(1.0, float(productive_depth_retained_cycles) / 3.0)
        + min(1.0, average_productive_depth_step_delta / 12.0),
    )
    depth_drift_pressure = min(
        1.0,
        min(1.0, float(depth_drift_cycles) / 3.0)
        + min(1.0, average_depth_drift_step_delta / 12.0),
    )
    partial_productive_without_decision_runs = max(
        0,
        _safe_int(campaign.get("partial_productive_without_decision_runs"), 0),
    )
    incomplete_cycle_count = max(0, _safe_int(campaign.get("incomplete_cycle_count"), 0))
    failed_run_present = 1.0 if isinstance(campaign.get("recent_failed_run", {}), Mapping) and campaign.get("recent_failed_run", {}) else 0.0
    failed_decision_present = (
        1.0
        if isinstance(campaign.get("recent_failed_decision", {}), Mapping) and campaign.get("recent_failed_decision", {})
        else 0.0
    )
    broad_observe_then_retrieval_first = 1.0 if bool(round_state.get("broad_observe_then_retrieval_first", False)) else 0.0
    live_decision_credit_gap = 1.0 if bool(round_state.get("live_decision_credit_gap", False)) else 0.0
    semantic_progress_drift = 1.0 if bool(round_state.get("semantic_progress_drift", False)) else 0.0
    intermediate_decision_evidence = bool(campaign.get("intermediate_decision_evidence", False))
    intermediate_decision_count = max(
        0,
        _safe_int(campaign.get("non_runtime_managed_decisions"), 0),
        _safe_int(campaign.get("non_runtime_managed_runs"), 0),
    )
    if intermediate_decision_count <= 0 and intermediate_decision_evidence:
        intermediate_decision_count = max(
            0,
            sum(max(0, _safe_int(value, 0)) for value in dict(subsystem.get("retained_by_subsystem", {})).values())
            + sum(max(0, _safe_int(value, 0)) for value in dict(subsystem.get("rejected_by_subsystem", {})).values()),
        )
    intermediate_decision_gain = min(1.0, float(intermediate_decision_count) / 3.0)
    observability_pressure = min(
        1.0,
        (
            live_decision_credit_gap
            + semantic_progress_drift
            + min(1.0, float(partial_productive_without_decision_runs) / 2.0)
            + (1.0 if incomplete_cycle_count > 0 else 0.0)
        )
        / 4.0,
    )
    subsystem_robustness_pressure = min(
        1.0,
        (
            failed_run_present
            + failed_decision_present
            + min(1.0, float(partial_productive_without_decision_runs) / 2.0)
            + (1.0 if incomplete_cycle_count > 0 else 0.0)
            + min(1.0, float(max(0, _safe_int(campaign.get("failed_decisions"), 0))) / 2.0)
        )
        / 5.0,
    )
    structural_class_coverage = min(1.0, float(len(structural_summary.get("class_ids", []))) / 4.0)
    structural_class_alignment = min(
        1.0,
        float(len(set(priority_families_with_retained_gain) & structural_family_alias_set)) / 3.0,
    )
    structural_class_gap = min(
        1.0,
        float(len(priority_family_yield_gap & structural_family_alias_set)) / 3.0,
    )
    features = {
        "bias": 1.0,
        "retained_cycles": float(retained_cycles),
        "rejected_cycles": float(rejected_cycles),
        "pass_rate_delta": _safe_float(campaign.get("average_retained_pass_rate_delta")),
        "step_delta": _safe_float(campaign.get("average_retained_step_delta")),
        "productive_depth_gain": productive_depth_gain,
        "depth_drift_pressure": depth_drift_pressure,
        "long_horizon_retained_gain": min(1.0, float(long_horizon_retained_cycles) / 3.0),
        "failed_decisions": float(max(0, _safe_int(campaign.get("failed_decisions"), 0))),
        "family_regression": max(0.0, -worst_family_delta),
        "generated_family_regression": max(0.0, -worst_generated_family_delta),
        "family_regression_count": float(max(0, _safe_int(campaign.get("max_regressed_families"), 0))),
        "generated_family_regression_count": float(
            max(0, _safe_int(campaign.get("max_generated_regressed_families"), 0))
        ),
        "low_confidence_pressure": float(max(0, _safe_int(campaign.get("max_low_confidence_episode_delta"), 0))),
        "retrieval_regression": float(max(0, -_safe_int(campaign.get("min_trusted_retrieval_step_delta"), 0))),
        "breadth_pressure": float(
            max(0, _safe_int(pressure.get("campaign_breadth_pressure_cycles"), 0))
            + max(0, _safe_int(pressure.get("variant_breadth_pressure_cycles"), 0))
        ),
        "priority_family_retained_gain": min(1.0, float(len(priority_families_with_retained_gain)) / 3.0),
        "priority_family_yield_gap": min(1.0, float(len(priority_family_yield_gap)) / 3.0),
        "no_retained_gain_pressure": no_retained_gain_pressure,
        "retained_gain_conversion_pressure": min(
            1.0,
            float(len(priority_families_needing_retained_gain_conversion)) / 3.0,
        ),
        "generalization_gain": generalization_gain,
        "generalization_gap": generalization_gap,
        "frontier_failure_motif_gain": frontier_failure_motif_gain,
        "frontier_failure_motif_pressure": frontier_failure_motif_pressure,
        "frontier_repo_setting_gain": frontier_repo_setting_gain,
        "frontier_repo_setting_pressure": frontier_repo_setting_pressure,
        "intermediate_decision_gain": intermediate_decision_gain,
        "broad_observe_then_retrieval_first": broad_observe_then_retrieval_first,
        "live_decision_credit_gap": live_decision_credit_gap,
        "observability_pressure": observability_pressure,
        "subsystem_robustness_pressure": subsystem_robustness_pressure,
        "structural_class_coverage": structural_class_coverage,
        "structural_class_alignment": structural_class_alignment,
        "structural_class_gap": structural_class_gap,
        "subsystem_reject_pressure": float(
            sum(max(0, _safe_int(value, 0)) for value in dict(subsystem.get("rejected_by_subsystem", {})).values())
        ),
        "subsystem_retain_pressure": float(
            sum(max(0, _safe_int(value, 0)) for value in dict(subsystem.get("retained_by_subsystem", {})).values())
        ),
        "allow_kernel_autobuild": 1.0 if bool(liftoff.get("allow_kernel_autobuild", False)) else 0.0,
        "liftoff_shadow": 1.0 if str(liftoff.get("state", "")).strip() == "shadow_only" else 0.0,
        "liftoff_reject": 1.0 if str(liftoff.get("state", "")).strip() == "reject" else 0.0,
        "liftoff_retain": 1.0 if str(liftoff.get("state", "")).strip() == "retain" else 0.0,
    }
    features.update(_declared_controller_features(campaign, subsystem, pressure, liftoff, round_state))
    features = _apply_feature_constraints(features)
    return {
        "features": features,
        "campaign_signal": dict(campaign),
        "subsystem_signal": dict(subsystem),
        "planner_pressure_signal": dict(pressure),
        "liftoff_signal": dict(liftoff),
        "round_signal": dict(round_state),
        "discovered_structural_classes": list(structural_summary.get("classes", [])),
        "discovered_structural_family_aliases": structural_family_aliases,
    }


def build_failure_observation(
    *,
    phase: str,
    reason: str,
    subsystem: str = "",
    supplemental_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_phase = str(phase).strip()
    normalized_reason = str(reason).lower()
    normalized_subsystem = str(subsystem).strip()
    failure_features = {
        "bias": 1.0,
        "retained_cycles": 0.0,
        "rejected_cycles": 1.0,
        "pass_rate_delta": -0.08,
        "step_delta": 1.5,
        "productive_depth_gain": 0.0,
        "depth_drift_pressure": 0.0,
        "long_horizon_retained_gain": 0.0,
        "failed_decisions": 1.0,
        "family_regression": 0.2,
        "generated_family_regression": 0.2 if "generated" in normalized_reason else 0.0,
        "family_regression_count": 1.0,
        "generated_family_regression_count": 1.0 if "generated" in normalized_reason else 0.0,
        "low_confidence_pressure": 1.0 if "confidence" in normalized_reason else 0.0,
        "retrieval_regression": 1.0 if normalized_subsystem == "retrieval" else 0.0,
        "breadth_pressure": 1.0 if "stalled" in normalized_reason or "timeout" in normalized_reason else 0.0,
        "priority_family_retained_gain": 0.0,
        "priority_family_yield_gap": 0.0,
        "retained_gain_conversion_pressure": 0.0,
        "generalization_gain": 0.0,
        "generalization_gap": 1.0 if "generalization" in normalized_reason else 0.0,
        "frontier_failure_motif_gain": 0.0,
        "frontier_failure_motif_pressure": 0.0,
        "frontier_repo_setting_gain": 0.0,
        "frontier_repo_setting_pressure": 0.0,
        "structural_class_coverage": 0.0,
        "structural_class_alignment": 0.0,
        "structural_class_gap": 0.0,
        "subsystem_reject_pressure": 1.0,
        "subsystem_retain_pressure": 0.0,
        "allow_kernel_autobuild": 0.0,
        "liftoff_shadow": 0.0,
        "liftoff_reject": 1.0 if normalized_phase == "liftoff" else 0.0,
        "liftoff_retain": 0.0,
    }
    supplemental_features = supplemental_observation.get("features", {}) if isinstance(supplemental_observation, Mapping) else {}
    if not isinstance(supplemental_features, Mapping):
        supplemental_features = {}
    features = {
        str(name).strip(): _safe_float(value)
        for name, value in supplemental_features.items()
        if str(name).strip()
    }
    max_merge_features = {
        "retained_cycles",
        "rejected_cycles",
        "productive_depth_gain",
        "depth_drift_pressure",
        "long_horizon_retained_gain",
        "failed_decisions",
        "family_regression",
        "generated_family_regression",
        "family_regression_count",
        "generated_family_regression_count",
        "low_confidence_pressure",
        "retrieval_regression",
        "breadth_pressure",
        "priority_family_retained_gain",
        "priority_family_yield_gap",
        "no_retained_gain_pressure",
        "retained_gain_conversion_pressure",
        "generalization_gain",
        "generalization_gap",
        "frontier_failure_motif_gain",
        "frontier_failure_motif_pressure",
        "frontier_repo_setting_gain",
        "frontier_repo_setting_pressure",
        "subsystem_reject_pressure",
        "subsystem_retain_pressure",
        "allow_kernel_autobuild",
        "liftoff_shadow",
        "liftoff_reject",
        "liftoff_retain",
        "intermediate_decision_gain",
        "broad_observe_then_retrieval_first",
        "live_decision_credit_gap",
        "observability_pressure",
        "subsystem_robustness_pressure",
        "structural_class_coverage",
        "structural_class_alignment",
        "structural_class_gap",
    }
    for feature, value in failure_features.items():
        if feature == "bias":
            continue
        if feature == "pass_rate_delta":
            features[feature] = min(_safe_float(features.get(feature)), _safe_float(value))
            continue
        if feature == "step_delta":
            features[feature] = max(_safe_float(features.get(feature)), _safe_float(value))
            continue
        if feature in max_merge_features:
            features[feature] = max(_safe_float(features.get(feature)), _safe_float(value))
            continue
        if feature not in features:
            features[feature] = _safe_float(value)
    features["bias"] = 1.0
    return {
        "features": _apply_feature_constraints(features),
        "failure": {
            "phase": normalized_phase,
            "reason": str(reason).strip(),
            "subsystem": normalized_subsystem,
        },
    }


def observation_reward(observation: Mapping[str, object] | None) -> float:
    if not isinstance(observation, Mapping):
        return 0.0
    features = observation.get("features", {})
    if not isinstance(features, Mapping):
        return 0.0
    reward = 0.0
    reward += _safe_float(features.get("pass_rate_delta")) * 100.0
    reward -= max(0.0, _safe_float(features.get("step_delta"))) * 0.25
    reward += _safe_float(features.get("retained_cycles")) * 4.0
    reward -= _safe_float(features.get("rejected_cycles")) * 2.0
    reward += _safe_float(features.get("productive_depth_gain")) * 4.0
    reward -= _safe_float(features.get("depth_drift_pressure")) * 4.0
    reward += _safe_float(features.get("long_horizon_retained_gain")) * 2.5
    reward -= _safe_float(features.get("failed_decisions")) * 6.0
    reward -= _safe_float(features.get("family_regression")) * 120.0
    reward -= _safe_float(features.get("generated_family_regression")) * 120.0
    reward -= _safe_float(features.get("family_regression_count")) * 5.0
    reward -= _safe_float(features.get("generated_family_regression_count")) * 5.0
    reward -= _safe_float(features.get("low_confidence_pressure")) * 0.5
    reward -= _safe_float(features.get("retrieval_regression")) * 0.5
    reward -= _safe_float(features.get("subsystem_reject_pressure")) * 1.5
    reward += _safe_float(features.get("subsystem_retain_pressure")) * 1.0
    reward += _safe_float(features.get("priority_family_retained_gain")) * 3.0
    reward -= _safe_float(features.get("priority_family_yield_gap")) * 3.0
    reward -= _safe_float(features.get("no_retained_gain_pressure")) * 5.0
    reward -= _safe_float(features.get("retained_gain_conversion_pressure")) * 4.0
    reward += _safe_float(features.get("generalization_gain")) * 4.0
    reward -= _safe_float(features.get("generalization_gap")) * 4.0
    reward += _safe_float(features.get("frontier_failure_motif_gain")) * 3.0
    reward -= _safe_float(features.get("frontier_failure_motif_pressure")) * 3.0
    reward += _safe_float(features.get("frontier_repo_setting_gain")) * 4.0
    reward -= _safe_float(features.get("frontier_repo_setting_pressure")) * 4.0
    reward += _safe_float(features.get("intermediate_decision_gain")) * 2.5
    reward -= _safe_float(features.get("broad_observe_then_retrieval_first")) * 4.5
    reward -= _safe_float(features.get("live_decision_credit_gap")) * 3.5
    reward -= _safe_float(features.get("observability_pressure")) * 4.0
    reward -= _safe_float(features.get("subsystem_robustness_pressure")) * 4.5
    reward += _safe_float(features.get("structural_class_coverage")) * 2.5
    reward += _safe_float(features.get("structural_class_alignment")) * 3.5
    reward -= _safe_float(features.get("structural_class_gap")) * 4.0
    reward += _safe_float(features.get("allow_kernel_autobuild")) * 2.0
    reward += _safe_float(features.get("liftoff_shadow")) * 3.0
    reward += _safe_float(features.get("liftoff_retain")) * 12.0
    reward -= _safe_float(features.get("liftoff_reject")) * 8.0
    for feature, weight in _FEATURE_REWARD_WEIGHTS.items():
        reward += _safe_float(features.get(feature)) * _safe_float(weight)
    return reward


def estimate_state_value(controller_state: Mapping[str, object], features: Mapping[str, float]) -> float:
    weights = controller_state.get("value_weights", {})
    if not isinstance(weights, Mapping):
        return 0.0
    return sum(_safe_float(weights.get(name)) * _safe_float(value) for name, value in features.items())


def _estimate_policy_value(controller_state: Mapping[str, object], features: Mapping[str, float]) -> float:
    weights = controller_state.get("policy_value_weights", {})
    if not isinstance(weights, Mapping):
        return 0.0
    return sum(_safe_float(weights.get(name)) * _safe_float(value) for name, value in features.items())


def _reward_stddev(action_stats: Mapping[str, object]) -> float:
    count = max(0, _safe_int(action_stats.get("count"), 0))
    if count <= 1:
        return abs(_safe_float(action_stats.get("last_reward")) - _safe_float(action_stats.get("reward_mean")))
    variance = max(0.0, _safe_float(action_stats.get("reward_m2")) / float(max(1, count - 1)))
    return math.sqrt(variance)


def _latent_world_feature_snapshot(features: Mapping[str, object] | None) -> dict[str, float]:
    payload = features if isinstance(features, Mapping) else {}
    return _apply_feature_constraints(
        {
            feature: _safe_float(value)
            for feature, value in payload.items()
            if str(feature).strip() and str(feature).strip() != "bias"
        }
    )


def _latent_world_similarity(
    left_features: Mapping[str, object] | None,
    right_features: Mapping[str, object] | None,
) -> float:
    left = _latent_world_feature_snapshot(left_features)
    right = _latent_world_feature_snapshot(right_features)
    active_features = sorted(
        feature
        for feature in set(left) | set(right)
        if feature != "bias"
        and (abs(_safe_float(left.get(feature))) > 1e-6 or abs(_safe_float(right.get(feature))) > 1e-6)
    )
    if not active_features:
        return 1.0
    distance = 0.0
    for feature in active_features:
        left_value = math.tanh(_safe_float(left.get(feature)))
        right_value = math.tanh(_safe_float(right.get(feature)))
        distance += min(1.0, abs(left_value - right_value))
    return max(0.0, 1.0 - (distance / float(len(active_features))))


def _latent_world_repeat_novelty(
    predicted_features: Mapping[str, object] | None,
    reference_states: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    if not reference_states:
        return 0.0, 0.0
    max_similarity = max(
        _latent_world_similarity(predicted_features, reference_features)
        for reference_features in reference_states
    )
    return max_similarity, max(0.0, 1.0 - max_similarity)


def _transition_uncertainty(action_stats: Mapping[str, object]) -> float:
    transition_error = action_stats.get("transition_error_ema", {})
    if not isinstance(transition_error, Mapping):
        return 0.0
    if not transition_error:
        return 0.0
    return sum(max(0.0, _safe_float(value)) for value in transition_error.values()) / float(len(transition_error))


def _context_transition_uncertainty(controller_state: Mapping[str, object]) -> float:
    error_map = controller_state.get("transition_context_error_ema", {})
    if not isinstance(error_map, Mapping) or not error_map:
        return 0.0
    return sum(max(0.0, _safe_float(value)) for value in error_map.values()) / float(len(error_map))


def _predict_context_transition_residual(
    controller_state: Mapping[str, object],
    *,
    target_feature: str,
    context_features: Mapping[str, float],
) -> float:
    all_weights = controller_state.get("transition_context_weights", {})
    if not isinstance(all_weights, Mapping):
        return 0.0
    weights = all_weights.get(str(target_feature).strip(), {})
    if not isinstance(weights, Mapping):
        return 0.0
    return sum(_safe_float(weights.get(name)) * _safe_float(value) for name, value in context_features.items())


def predict_next_observation(
    controller_state: Mapping[str, object],
    current_observation: Mapping[str, object] | None,
    policy: Mapping[str, object],
) -> dict[str, object]:
    observation = current_observation if isinstance(current_observation, Mapping) else {}
    current_features = observation.get("features", {})
    if not isinstance(current_features, Mapping):
        current_features = {}
    predicted_features = _apply_feature_constraints(
        {
            feature: _safe_float(current_features.get(feature))
            for feature in _ordered_feature_names(current_features)
        }
    )
    context_features = _transition_context_features(predicted_features, policy)
    action_models = controller_state.get("action_models", {})
    if isinstance(action_models, Mapping):
        action_stats = action_models.get(action_key_for_policy(policy), {})
    else:
        action_stats = {}
    transition_mean = action_stats.get("transition_mean", {}) if isinstance(action_stats, Mapping) else {}
    if isinstance(transition_mean, Mapping):
        for feature, delta in transition_mean.items():
            token = str(feature).strip()
            if not token or token == "bias":
                continue
            residual = _predict_context_transition_residual(
                controller_state,
                target_feature=token,
                context_features=context_features,
            )
            predicted_features[token] = predicted_features.get(token, 0.0) + _safe_float(delta) + residual
    return {"features": _apply_feature_constraints(predicted_features)}


def update_controller_state(
    controller_state: Mapping[str, object] | None,
    *,
    start_observation: Mapping[str, object] | None,
    action_policy: Mapping[str, object],
    end_observation: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    state = normalize_controller_state(controller_state)
    start_features = (
        dict(start_observation.get("features", {}))
        if isinstance(start_observation, Mapping) and isinstance(start_observation.get("features", {}), Mapping)
        else {}
    )
    end_features = (
        dict(end_observation.get("features", {}))
        if isinstance(end_observation, Mapping) and isinstance(end_observation.get("features", {}), Mapping)
        else {}
    )
    feature_order = _ordered_feature_names(start_features, end_features)
    for feature in feature_order:
        start_features.setdefault(feature, 0.0)
        end_features.setdefault(feature, 0.0)
    start_features = _apply_feature_constraints(start_features)
    end_features = _apply_feature_constraints(end_features)
    reward = observation_reward(end_observation)
    action_key = action_key_for_policy(action_policy)
    action_models = state.setdefault("action_models", {})
    action_stats = action_models.setdefault(
        action_key,
        {
            "count": 0,
            "reward_mean": 0.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": {
                "focus": str(action_policy.get("focus", "balanced")).strip() or "balanced",
                "adaptive_search": bool(action_policy.get("adaptive_search", False)),
                "cycles": max(1, _safe_int(action_policy.get("cycles"), 1)),
                "campaign_width": max(1, _safe_int(action_policy.get("campaign_width"), 1)),
                "variant_width": max(1, _safe_int(action_policy.get("variant_width"), 1)),
                "task_limit": max(1, _safe_int(action_policy.get("task_limit"), 64)),
                "task_step_floor": max(1, _safe_int(action_policy.get("task_step_floor"), 1)),
                "priority_benchmark_families": _normalize_benchmark_families(
                    action_policy.get("priority_benchmark_families", [])
                ),
            },
            "last_reward": 0.0,
        },
    )
    count = max(0, _safe_int(action_stats.get("count"), 0)) + 1
    previous_mean = _safe_float(action_stats.get("reward_mean"))
    reward_mean = previous_mean + ((reward - previous_mean) / float(count))
    reward_m2 = max(0.0, _safe_float(action_stats.get("reward_m2"))) + (reward - previous_mean) * (reward - reward_mean)
    action_stats["count"] = count
    action_stats["reward_mean"] = reward_mean
    action_stats["reward_m2"] = reward_m2
    action_stats["last_reward"] = reward
    transition_mean = action_stats.setdefault("transition_mean", {})
    transition_error_ema = action_stats.setdefault("transition_error_ema", {})
    transition_context_weights = state.setdefault("transition_context_weights", {})
    transition_context_error_ema = state.setdefault("transition_context_error_ema", {})
    transition_learning_rate = _safe_float(state.get("transition_learning_rate"), 0.15)
    context_features = _transition_context_features(start_features, action_policy)
    for feature in feature_order:
        if feature == "bias":
            continue
        delta = _safe_float(end_features.get(feature)) - _safe_float(start_features.get(feature))
        old_mean = _safe_float(transition_mean.get(feature))
        predicted_residual = _predict_context_transition_residual(
            state,
            target_feature=feature,
            context_features=context_features,
        )
        prediction_error = delta - (old_mean + predicted_residual)
        new_mean = old_mean + (transition_learning_rate * (delta - old_mean))
        transition_mean[feature] = new_mean
        weights = transition_context_weights.setdefault(feature, {})
        for source_feature, value in context_features.items():
            old_weight = _safe_float(weights.get(source_feature))
            weights[source_feature] = old_weight + (transition_learning_rate * prediction_error * _safe_float(value))
        error = abs(prediction_error)
        old_error = max(0.0, _safe_float(transition_error_ema.get(feature)))
        transition_error_ema[feature] = old_error + (transition_learning_rate * (error - old_error))
        old_context_error = max(0.0, _safe_float(transition_context_error_ema.get(feature)))
        transition_context_error_ema[feature] = old_context_error + (
            transition_learning_rate * (error - old_context_error)
        )
    gamma = _safe_float(state.get("gamma"), 0.85)
    learning_rate = _safe_float(state.get("value_learning_rate"), 0.08)
    value_weights = state.setdefault("value_weights", {})
    policy_value_weights = state.setdefault("policy_value_weights", {})
    policy_features = _policy_features(action_policy)
    current_value = estimate_state_value(state, start_features)
    next_value = estimate_state_value(state, end_features)
    td_target = reward + (gamma * next_value)
    td_error = td_target - current_value
    for feature, value in start_features.items():
        weight = _safe_float(value_weights.get(feature))
        value_weights[feature] = weight + (learning_rate * td_error * _safe_float(value))
    for feature, value in policy_features.items():
        weight = _safe_float(policy_value_weights.get(feature))
        policy_value_weights[feature] = weight + (learning_rate * td_error * _safe_float(value))
    state["updates"] = max(0, _safe_int(state.get("updates"), 0)) + 1
    recent_rewards = list(state.get("recent_rewards", []))
    recent_rewards.append(reward)
    state["recent_rewards"] = recent_rewards[-20:]
    recent_state_features = list(state.get("recent_state_features", []))
    recent_state_features.append(_latent_world_feature_snapshot(end_features))
    state["recent_state_features"] = recent_state_features[-_RECENT_STATE_FEATURE_MEMORY_LIMIT:]
    state["last_action_key"] = action_key
    diagnostics = {
        "action_key": action_key,
        "reward": reward,
        "td_error": td_error,
        "count": count,
        "reward_mean": reward_mean,
        "reward_stddev": _reward_stddev(action_stats),
        "transition_uncertainty": _transition_uncertainty(action_stats),
    }
    return state, diagnostics


def plan_next_policy(
    controller_state: Mapping[str, object] | None,
    *,
    current_observation: Mapping[str, object] | None,
    candidate_policies: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    state = normalize_controller_state(controller_state)
    observation = current_observation if isinstance(current_observation, Mapping) else {}
    current_features = observation.get("features", {})
    if not isinstance(current_features, Mapping):
        current_features = {}
    best_policy: dict[str, object] | None = None
    best_score = float("-inf")
    diagnostics: list[dict[str, object]] = []
    gamma = _safe_float(state.get("gamma"), 0.85)
    exploration_bonus = _safe_float(state.get("exploration_bonus"), 2.5)
    uncertainty_penalty = _safe_float(state.get("uncertainty_penalty"), 1.5)
    min_action_support = max(1, _safe_int(state.get("min_action_support"), 3))
    thin_evidence_penalty = max(0.0, _safe_float(state.get("thin_evidence_penalty"), 2.0))
    support_confidence_power = max(0.1, _safe_float(state.get("support_confidence_power"), 0.5))
    rollout_depth = max(1, _safe_int(state.get("rollout_depth"), 2))
    rollout_beam_width = max(1, _safe_int(state.get("rollout_beam_width"), 6))
    repeat_action_penalty = max(0.0, _safe_float(state.get("repeat_action_penalty"), 5.0))
    state_repeat_penalty = max(0.0, _safe_float(state.get("state_repeat_penalty"), 0.25))
    state_novelty_bonus = max(0.0, _safe_float(state.get("state_novelty_bonus"), 0.1))
    last_action_key = str(state.get("last_action_key", "")).strip()
    recent_state_features = state.get("recent_state_features", [])
    if not isinstance(recent_state_features, Sequence) or isinstance(recent_state_features, (str, bytes)):
        recent_state_features = []
    recent_state_feature_memory = [
        _latent_world_feature_snapshot(features)
        for features in recent_state_features
        if isinstance(features, Mapping)
    ][-_RECENT_STATE_FEATURE_MEMORY_LIMIT:]
    state_feature_search_enabled = max(0, _safe_int(state.get("updates"), 0)) >= 2
    historical_state_feature_memory = (
        recent_state_feature_memory
        if state_feature_search_enabled and len(recent_state_feature_memory) >= 2
        else []
    )
    action_models = state.get("action_models", {})
    if not isinstance(action_models, Mapping):
        action_models = {}

    def _candidate_exploitation(
        observation_payload: Mapping[str, object] | None,
        policy: Mapping[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        key = action_key_for_policy(policy)
        predicted = predict_next_observation(state, observation_payload, policy)
        observation_features = (
            dict(observation_payload.get("features", {}))
            if isinstance(observation_payload, Mapping) and isinstance(observation_payload.get("features", {}), Mapping)
            else {}
        )
        action_stats = action_models.get(key, {})
        count = max(0, _safe_int(action_stats.get("count"), 0)) if isinstance(action_stats, Mapping) else 0
        policy_prior = _estimate_policy_value(state, _policy_features(policy))
        pressure_alignment_bonus = _policy_pressure_alignment_bonus(observation_features, policy)
        learned_reward = _safe_float(action_stats.get("reward_mean")) if isinstance(action_stats, Mapping) else 0.0
        reward_stddev = _reward_stddev(action_stats) if isinstance(action_stats, Mapping) else 0.0
        transition_uncertainty = _transition_uncertainty(action_stats) if isinstance(action_stats, Mapping) else 0.0
        context_transition_uncertainty = _context_transition_uncertainty(state)
        rollout_value = estimate_state_value(state, dict(predicted.get("features", {})))
        effective_count = math.sqrt(float(count + 1))
        reward_lcb = learned_reward - ((uncertainty_penalty * reward_stddev) / effective_count)
        rollout_lcb = rollout_value - (uncertainty_penalty * (transition_uncertainty + context_transition_uncertainty))
        if count <= 0:
            support_confidence = 0.0
        elif count >= min_action_support:
            support_confidence = 1.0
        else:
            support_confidence = math.pow(float(count) / float(min_action_support), support_confidence_power)
        support_shortfall = max(0, min_action_support - count)
        support_penalty = (
            thin_evidence_penalty * (float(support_shortfall) / float(min_action_support))
            if 0 < count < min_action_support
            else 0.0
        )
        supported_reward_lcb = reward_lcb * support_confidence
        supported_rollout_lcb = rollout_lcb * support_confidence
        exploitation = policy_prior + pressure_alignment_bonus + supported_reward_lcb - support_penalty
        details = {
            "action_key": key,
            "policy": dict(policy),
            "policy_prior": policy_prior,
            "pressure_alignment_bonus": pressure_alignment_bonus,
            "learned_reward": learned_reward,
            "reward_lcb": reward_lcb,
            "supported_reward_lcb": supported_reward_lcb,
            "reward_stddev": reward_stddev,
            "rollout_value": rollout_value,
            "rollout_lcb": rollout_lcb,
            "supported_rollout_lcb": supported_rollout_lcb,
            "transition_uncertainty": transition_uncertainty,
            "context_transition_uncertainty": context_transition_uncertainty,
            "count": count,
            "min_action_support": min_action_support,
            "support_confidence_power": support_confidence_power,
            "support_confidence": support_confidence,
            "support_shortfall": support_shortfall,
            "support_penalty": support_penalty,
        }
        return exploitation, predicted, details

    def _rollout_score(
        observation_payload: Mapping[str, object] | None,
        policy: Mapping[str, object],
        *,
        depth: int,
        include_exploration: bool,
        previous_action_key: str,
        latent_history: Sequence[Mapping[str, object]],
    ) -> tuple[float, dict[str, object]]:
        exploitation, predicted, details = _candidate_exploitation(observation_payload, policy)
        action_key = str(details.get("action_key", "")).strip()
        count = int(details.get("count", 0) or 0)
        exploration = (exploration_bonus / math.sqrt(float(count + 1))) if include_exploration else 0.0
        repeat_penalty = repeat_action_penalty if previous_action_key and previous_action_key == action_key else 0.0
        predicted_features = predicted.get("features", {})
        if not isinstance(predicted_features, Mapping):
            predicted_features = {}
        reference_states = (
            [*historical_state_feature_memory, *latent_history]
            if state_feature_search_enabled and (latent_history or historical_state_feature_memory)
            else []
        )
        latent_repeat_similarity, latent_novelty = _latent_world_repeat_novelty(
            predicted_features,
            reference_states,
        )
        latent_repeat_penalty = state_repeat_penalty * latent_repeat_similarity
        latent_novelty_bonus = state_novelty_bonus * latent_novelty
        leaf_value = float(details.get("supported_rollout_lcb", 0.0) or 0.0)
        future_score = leaf_value
        future_action_key = ""
        next_latent_history = [
            *list(latent_history)[-_RECENT_STATE_FEATURE_MEMORY_LIMIT + 1 :],
            _latent_world_feature_snapshot(predicted_features),
        ]
        if depth > 1 and candidate_policies:
            ranked_children: list[tuple[float, dict[str, object]]] = []
            for child_policy in candidate_policies:
                child_exploitation, _child_predicted, child_details = _candidate_exploitation(predicted, child_policy)
                ranked_children.append((child_exploitation, dict(child_policy)))
            ranked_children.sort(key=lambda item: float(item[0]), reverse=True)
            child_scores: list[tuple[float, dict[str, object]]] = []
            for _, child_policy in ranked_children[:rollout_beam_width]:
                child_score, child_details = _rollout_score(
                    predicted,
                    child_policy,
                    depth=depth - 1,
                    include_exploration=False,
                    previous_action_key=action_key,
                    latent_history=next_latent_history,
                )
                child_scores.append((child_score, child_details))
            if child_scores:
                best_child_score, best_child_details = max(child_scores, key=lambda item: float(item[0]))
                future_score = best_child_score
                future_action_key = str(best_child_details.get("action_key", "")).strip()
        total_score = exploitation + (gamma * future_score) + exploration + latent_novelty_bonus
        total_score -= repeat_penalty + latent_repeat_penalty
        details["exploration"] = exploration
        details["repeat_penalty"] = repeat_penalty
        details["state_repeat_similarity"] = latent_repeat_similarity
        details["state_repeat_penalty"] = latent_repeat_penalty
        details["state_novelty_score"] = latent_novelty
        details["state_novelty_bonus"] = latent_novelty_bonus
        details["future_score"] = future_score
        details["future_action_key"] = future_action_key
        details["score"] = total_score
        return total_score, details

    for raw_policy in candidate_policies:
        policy = dict(raw_policy)
        score, details = _rollout_score(
            observation,
            policy,
            depth=rollout_depth,
            include_exploration=True,
            previous_action_key=last_action_key,
            latent_history=[],
        )
        diagnostics.append(details)
        if score > best_score:
            best_score = score
            best_policy = policy
    if best_policy is None:
        best_policy = dict(candidate_policies[0]) if candidate_policies else {
            "focus": "balanced",
            "adaptive_search": False,
            "cycles": 1,
            "campaign_width": 1,
            "variant_width": 1,
            "task_limit": 64,
            "task_step_floor": 1,
        }
    diagnostics.sort(key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
    return best_policy, {
        "controller_kind": str(state.get("controller_kind", "")).strip(),
        "updates": max(0, _safe_int(state.get("updates"), 0)),
        "rollout_depth": rollout_depth,
        "rollout_beam_width": rollout_beam_width,
        "min_action_support": min_action_support,
        "thin_evidence_penalty": thin_evidence_penalty,
        "support_confidence_power": support_confidence_power,
        "state_repeat_penalty": state_repeat_penalty,
        "state_novelty_bonus": state_novelty_bonus,
        "last_action_key": last_action_key,
        "selected_action_key": action_key_for_policy(best_policy),
        "selected_score": best_score,
        "candidates": diagnostics[:12],
    }


def controller_state_summary(controller_state: Mapping[str, object] | None) -> dict[str, object]:
    state = normalize_controller_state(controller_state)
    action_models = state.get("action_models", {})
    if not isinstance(action_models, Mapping):
        action_models = {}
    ranked = sorted(
        (
            {
                "action_key": str(action_key),
                "count": max(0, _safe_int(stats.get("count"), 0)),
                "reward_mean": _safe_float(stats.get("reward_mean")),
                "reward_stddev": _reward_stddev(stats),
                "transition_uncertainty": _transition_uncertainty(stats),
            }
            for action_key, stats in action_models.items()
            if isinstance(stats, Mapping)
        ),
        key=lambda item: (-int(item["count"]), str(item["action_key"])),
    )
    return {
        "controller_kind": str(state.get("controller_kind", "")).strip(),
        "updates": max(0, _safe_int(state.get("updates"), 0)),
        "known_actions": len(ranked),
        "core_state_features": list(_STATE_FEATURE_ORDER),
        "discovered_state_features": list(_DISCOVERED_STATE_FEATURES),
        "last_action_key": str(state.get("last_action_key", "")).strip(),
        "recent_rewards": list(state.get("recent_rewards", [])),
        "rollout_depth": max(1, _safe_int(state.get("rollout_depth"), 2)),
        "rollout_beam_width": max(1, _safe_int(state.get("rollout_beam_width"), 6)),
        "min_action_support": max(1, _safe_int(state.get("min_action_support"), 3)),
        "thin_evidence_penalty": max(0.0, _safe_float(state.get("thin_evidence_penalty"), 2.0)),
        "support_confidence_power": max(0.1, _safe_float(state.get("support_confidence_power"), 0.5)),
        "repeat_action_penalty": max(0.0, _safe_float(state.get("repeat_action_penalty"), 5.0)),
        "state_repeat_penalty": max(0.0, _safe_float(state.get("state_repeat_penalty"), 0.25)),
        "state_novelty_bonus": max(0.0, _safe_float(state.get("state_novelty_bonus"), 0.1)),
        "recent_state_feature_count": len(list(state.get("recent_state_features", []))),
        "repo_setting_policy_priors": {
            str(signal): dict(entry)
            for signal, entry in dict(state.get("repo_setting_policy_priors", {})).items()
            if str(signal).strip() and isinstance(entry, Mapping)
        },
        "strategy_memory_priors": {
            str(key): (
                {
                    str(name): (
                        max(0, _safe_int(item))
                        if str(key).startswith("recent_rejects")
                        else max(0.0, _safe_float(item))
                    )
                    for name, item in dict(value).items()
                    if str(name).strip()
                }
                if isinstance(value, Mapping)
                else (max(0.0, _safe_float(value)) if "gain" in str(key) else max(0, _safe_int(value)))
            )
            for key, value in dict(state.get("strategy_memory_priors", {})).items()
            if str(key).strip()
        },
        "policy_value_weights": {
            feature: _safe_float(weight)
            for feature, weight in dict(state.get("policy_value_weights", {})).items()
            if str(feature).strip()
        },
        "transition_context_error_ema": {
            feature: max(0.0, _safe_float(value))
            for feature, value in dict(state.get("transition_context_error_ema", {})).items()
            if str(feature).strip()
        },
        "top_actions": ranked[:5],
    }
