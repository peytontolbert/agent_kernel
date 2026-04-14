from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

from ...config import KernelConfig
from .improvement_catalog import catalog_mapping, catalog_object, catalog_string_set
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER
from .semantic_compatibility import semantic_compatibility_violations
from .improvement_support_validation import build_default_task_contract_catalog
from .state_estimation_improvement import STATE_ESTIMATION_POLICY_CONTROL_KEYS

_TOLBERT_MODEL_GENERATION_FOCI = catalog_string_set("improvement", "tolbert_model_generation_foci")
_TOLBERT_MODEL_SURFACE_KEYS = catalog_string_set("improvement", "tolbert_model_surface_keys")
_TOLBERT_RUNTIME_POLICY_KEYS = catalog_string_set("improvement", "tolbert_runtime_policy_keys")
_TOLBERT_DECODER_POLICY_KEYS = catalog_string_set("improvement", "tolbert_decoder_policy_keys")
_TOLBERT_ACTION_GENERATION_POLICY_KEYS = catalog_string_set("improvement", "tolbert_action_generation_policy_keys")
_TOLBERT_ROLLOUT_POLICY_KEYS = catalog_string_set("improvement", "tolbert_rollout_policy_keys")
_TOLBERT_LIFTOFF_GATE_KEYS = catalog_string_set("improvement", "tolbert_liftoff_gate_keys")
_TOLBERT_BUILD_POLICY_KEYS = catalog_string_set("improvement", "tolbert_build_policy_keys")
_ARTIFACT_VALIDATION_PROFILES = catalog_object("improvement", "artifact_validation_profiles")
_ARTIFACT_CONTRACTS = catalog_mapping("improvement", "artifact_contracts")
_CAPABILITY_SURFACE_ARTIFACT_KINDS = catalog_mapping("improvement", "capability_surface_artifact_kinds")
_STATE_ESTIMATION_POLICY_KEYS = set(STATE_ESTIMATION_POLICY_CONTROL_KEYS)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _operator_task_contract(record: dict[str, object]) -> dict[str, object]:
    task_contract = record.get("task_contract")
    if isinstance(task_contract, dict) and task_contract:
        return task_contract
    template_contract = record.get("template_contract")
    if isinstance(template_contract, dict) and template_contract:
        return template_contract
    return {}


def _operator_steps(record: dict[str, object]) -> list[str]:
    steps = record.get("steps")
    if isinstance(steps, list) and any(str(step).strip() for step in steps):
        return [str(step) for step in steps if str(step).strip()]
    template_procedure = record.get("template_procedure")
    if isinstance(template_procedure, dict):
        commands = template_procedure.get("commands", [])
        if isinstance(commands, list):
            return [str(command) for command in commands if str(command).strip()]
    return []


def _operator_benchmark_families(record: dict[str, object]) -> list[str]:
    benchmark_families = record.get("benchmark_families")
    if isinstance(benchmark_families, list) and any(str(value).strip() for value in benchmark_families):
        return [str(value) for value in benchmark_families if str(value).strip()]
    applicable_families = record.get("applicable_benchmark_families")
    if isinstance(applicable_families, list):
        return [str(value) for value in applicable_families if str(value).strip()]
    return []


def _operator_support(record: dict[str, object]) -> int:
    support = record.get("support")
    if _is_positive_int(support):
        return int(support)
    support_count = record.get("support_count")
    if _is_positive_int(support_count):
        return int(support_count)
    return 0


def _tool_candidate_stage(stage: object) -> str:
    return str(stage).strip()


def _tool_candidate_lifecycle_state(candidate: dict[str, object]) -> str:
    return str(candidate.get("lifecycle_state", "")).strip()


def _tool_candidates_have_stage(payload: dict[str, object] | None, stage: str) -> bool:
    if not isinstance(payload, dict):
        return False
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return False
    return all(
        isinstance(candidate, dict) and str(candidate.get("promotion_stage", "")).strip() == stage
        for candidate in candidates
    )


def _artifact_contract(subsystem: str) -> dict[str, object]:
    value = _ARTIFACT_CONTRACTS.get(subsystem, {})
    return dict(value) if isinstance(value, dict) else {}


def _allowed_artifact_lifecycle_states(subsystem: str) -> set[str]:
    contract = _artifact_contract(subsystem)
    raw_states = contract.get("lifecycle_states", [])
    if not isinstance(raw_states, list):
        return set()
    return {state for state in (str(value).strip() for value in raw_states) if state}


def _artifact_validation_profile(subsystem: str) -> dict[str, object]:
    if not isinstance(_ARTIFACT_VALIDATION_PROFILES, dict):
        return {}
    value = _ARTIFACT_VALIDATION_PROFILES.get(subsystem, {})
    if not isinstance(value, dict):
        return {}
    profile = deepcopy(value)
    if subsystem == "state_estimation":
        sections = profile.get("sections", [])
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                if str(section.get("field", "")).strip() != "policy_controls":
                    continue
                allowed_keys = {
                    str(item).strip()
                    for item in section.get("allowed_keys", [])
                    if str(item).strip()
                }
                allowed_keys.update(_STATE_ESTIMATION_POLICY_KEYS)
                section["allowed_keys"] = sorted(allowed_keys)
    return profile


def _rule_error(rule: dict[str, object], key: str, *, range_error: bool = False) -> str:
    template_key = "range_error" if range_error and isinstance(rule.get("range_error"), str) else "type_error"
    if not isinstance(rule.get(template_key), str):
        template_key = "error"
    template = str(rule.get(template_key, "")).strip()
    return template.replace("{key}", key) if template else ""


def _string_list_has_content(value: object) -> bool:
    return isinstance(value, list) and bool([str(item).strip() for item in value if str(item).strip()])


def _validate_profile_rule(value: object, rule: dict[str, object], *, key: str) -> str:
    rule_kind = str(rule.get("kind", "")).strip()
    skip_values = rule.get("skip_values", [])
    if isinstance(skip_values, list) and any(value == item for item in skip_values):
        return ""
    if rule_kind == "boolean":
        return "" if isinstance(value, bool) else _rule_error(rule, key)
    if rule_kind == "int":
        minimum = rule.get("min")
        maximum = rule.get("max")
        if isinstance(value, bool) or not isinstance(value, int):
            return _rule_error(rule, key)
        if minimum is not None and int(value) < int(minimum):
            return _rule_error(rule, key, range_error=True)
        if maximum is not None and int(value) > int(maximum):
            return _rule_error(rule, key, range_error=True)
        return ""
    if rule_kind == "number":
        minimum = rule.get("min")
        maximum = rule.get("max")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _rule_error(rule, key)
        numeric_value = float(value)
        if minimum is not None and numeric_value < float(minimum):
            return _rule_error(rule, key, range_error=True)
        if maximum is not None and numeric_value > float(maximum):
            return _rule_error(rule, key, range_error=True)
        return ""
    if rule_kind == "enum":
        raw_value = str(value).strip()
        if str(rule.get("normalize", "")).strip() == "lower":
            raw_value = raw_value.lower()
        allowed_values = {str(item).strip() for item in rule.get("values", []) if str(item).strip()}
        return "" if raw_value in allowed_values else _rule_error(rule, key)
    if rule_kind == "list":
        return "" if isinstance(value, list) else _rule_error(rule, key)
    if rule_kind == "list_nonempty_strings":
        return "" if _string_list_has_content(value) else _rule_error(rule, key)
    if rule_kind == "object":
        return "" if isinstance(value, dict) else _rule_error(rule, key)
    if rule_kind == "object_or_null":
        return "" if value is None or isinstance(value, dict) else _rule_error(rule, key)
    if rule_kind == "string_nonempty":
        return "" if isinstance(value, str) and value.strip() else _rule_error(rule, key)
    if rule_kind == "string_nonempty_if_present":
        return "" if isinstance(value, str) and value.strip() else _rule_error(rule, key)
    return ""


def _validate_profile_object_section(
    *,
    section: dict[str, object],
    value: dict[str, object],
    violations: list[str],
) -> None:
    field_rules = section.get("field_rules", {})
    field_rules = field_rules if isinstance(field_rules, dict) else {}
    allowed_keys = section.get("allowed_keys")
    allowed = {str(item).strip() for item in allowed_keys if str(item).strip()} if isinstance(allowed_keys, list) else None
    unknown_key_error = str(section.get("unknown_key_error", "")).strip()
    default_rule = section.get("default_rule", {})
    default_rule = default_rule if isinstance(default_rule, dict) else {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        rule = field_rules.get(key)
        if rule is None and allowed is not None and key not in allowed:
            if unknown_key_error:
                violations.append(unknown_key_error.replace("{key}", key))
            continue
        normalized_rule = dict(rule) if isinstance(rule, dict) else dict(default_rule)
        if not normalized_rule:
            continue
        error = _validate_profile_rule(raw_value, normalized_rule, key=key)
        if error:
            violations.append(error)
    for raw_key, raw_rule in field_rules.items():
        key = str(raw_key).strip()
        if not key or key in value or not isinstance(raw_rule, dict) or not bool(raw_rule.get("required", False)):
            continue
        error = _validate_profile_rule(None, raw_rule, key=key)
        if error:
            violations.append(error)


def _validate_profile_list_section(
    *,
    section: dict[str, object],
    value: list[object],
    violations: list[str],
) -> None:
    item_fields = section.get("item_fields", {})
    item_fields = item_fields if isinstance(item_fields, dict) else {}
    item_error = str(section.get("item_error", "")).strip()
    for item in value:
        if not item_fields:
            continue
        if not isinstance(item, dict):
            if item_error:
                violations.append(item_error)
            continue
        for raw_key, raw_rule in item_fields.items():
            key = str(raw_key).strip()
            if not key or not isinstance(raw_rule, dict):
                continue
            if key not in item and not bool(raw_rule.get("required", False)):
                continue
            error = _validate_profile_rule(item.get(key), raw_rule, key=key)
            if error:
                violations.append(error)


def _validate_artifact_profile(
    *,
    subsystem: str,
    payload: dict[str, object],
    checks: list[str],
    violations: list[str],
) -> None:
    profile = _artifact_validation_profile(subsystem)
    if not profile:
        return
    generation_focuses = {
        str(item).strip() for item in profile.get("generation_focuses", []) if str(item).strip()
    }
    generation_focus = str(payload.get("generation_focus", "")).strip()
    if generation_focus and generation_focuses and generation_focus not in generation_focuses:
        violations.append(f"generation_focus must be a supported {subsystem} focus")
    control_schema = str(profile.get("control_schema", "")).strip()
    if control_schema and str(payload.get("control_schema", "")).strip() != control_schema:
        violations.append(f"{subsystem} artifacts must declare control_schema {control_schema}")
    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    sections = profile.get("sections", [])
    if isinstance(sections, list):
        for raw_section in sections:
            if not isinstance(raw_section, dict):
                continue
            skip_kinds = {
                str(item).strip() for item in raw_section.get("skip_for_artifact_kinds", []) if str(item).strip()
            }
            only_kinds = {
                str(item).strip() for item in raw_section.get("only_for_artifact_kinds", []) if str(item).strip()
            }
            if artifact_kind in skip_kinds or (only_kinds and artifact_kind not in only_kinds):
                continue
            field = str(raw_section.get("field", "")).strip()
            if not field:
                continue
            value = payload.get(field)
            section_kind = str(raw_section.get("kind", "")).strip()
            missing_error = str(raw_section.get("missing_error", "")).strip()
            required = bool(raw_section.get("required", False))
            non_empty = bool(raw_section.get("non_empty", False))
            if section_kind == "object":
                if not isinstance(value, dict) or (non_empty and not value):
                    if required and missing_error:
                        violations.append(missing_error)
                    continue
                _validate_profile_object_section(section=raw_section, value=value, violations=violations)
                continue
            if section_kind == "list":
                if not isinstance(value, list) or (non_empty and not value):
                    if required and missing_error:
                        violations.append(missing_error)
                    continue
                _validate_profile_list_section(section=raw_section, value=value, violations=violations)
                continue
            if section_kind == "list_nonempty_strings":
                if not _string_list_has_content(value):
                    if required and missing_error:
                        violations.append(missing_error)
                continue
    for raw_check in profile.get("checks", []):
        check = str(raw_check).strip()
        if check:
            checks.append(check)


def assess_artifact_compatibility(
    *,
    subsystem: str,
    payload: Any,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    from ... import improvement as core

    subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem, capability_modules_path)
    checks: list[str] = []
    violations: list[str] = []
    task_catalog = build_default_task_contract_catalog()

    if not isinstance(payload, dict):
        return {
            "compatible": False,
            "checked_rules": ["artifact payload must be a JSON object"],
            "violations": ["artifact payload is not a JSON object"],
        }

    if str(payload.get("spec_version", "")).strip() != "asi_v1":
        violations.append("spec_version must be asi_v1")
    checks.append("spec_version")

    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    contract = _artifact_contract(subsystem)
    expected_kind = contract.get("artifact_kind")
    if isinstance(expected_kind, list):
        expected_kinds = [kind for kind in (str(value).strip() for value in expected_kind) if kind]
        if expected_kinds and artifact_kind not in expected_kinds:
            violations.append(f"artifact_kind must be one of: {', '.join(expected_kinds)}")
    elif str(expected_kind).strip() and artifact_kind != str(expected_kind).strip():
        violations.append(f"artifact_kind must be {str(expected_kind).strip()}")
    checks.append("artifact_kind")

    lifecycle_state = str(payload.get("lifecycle_state", "")).strip()
    if not lifecycle_state:
        violations.append("artifact must contain a lifecycle_state")
    else:
        allowed_lifecycle_states = _allowed_artifact_lifecycle_states(subsystem)
        normalized_lifecycle_state = lifecycle_state
        if lifecycle_state == "proposed" and "candidate" in allowed_lifecycle_states:
            normalized_lifecycle_state = "candidate"
        if allowed_lifecycle_states and normalized_lifecycle_state not in allowed_lifecycle_states:
            allowed_text = ", ".join(sorted(allowed_lifecycle_states))
            violations.append(f"artifact lifecycle_state must be one of: {allowed_text}")
    checks.append("lifecycle_state")

    if bool(contract.get("requires_retention_gate", False)):
        retention_gate = payload.get("retention_gate", {})
        if not isinstance(retention_gate, dict) or not retention_gate:
            violations.append("artifact must contain a retention_gate")
        checks.append("retention_gate")

    if subsystem in {
        "benchmark",
        "retrieval",
        "verifier",
        "policy",
        "universe",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "curriculum",
    }:
        proposals = payload.get("proposals", [])
        if not isinstance(proposals, list) or not proposals:
            violations.append("artifact must contain a non-empty proposals list")
        checks.append("proposals")
    if subsystem == "tolbert_model":
        proposals = payload.get("proposals", [])
        runtime_paths = payload.get("runtime_paths", {})
        dataset_manifest = payload.get("dataset_manifest", {})
        if not isinstance(proposals, list) or not proposals:
            violations.append("Tolbert model artifact must contain a non-empty proposals list")
        if not isinstance(runtime_paths, dict):
            violations.append("Tolbert model artifact must contain runtime_paths")
        if not isinstance(dataset_manifest, dict) or int(dataset_manifest.get("total_examples", 0)) <= 0:
            violations.append("Tolbert model artifact must contain a non-empty dataset_manifest")
        checks.append("tolbert_model_surface")
    if subsystem == "qwen_adapter":
        runtime_paths = payload.get("runtime_paths", {})
        training_dataset_manifest = payload.get("training_dataset_manifest", {})
        runtime_policy = payload.get("runtime_policy", {})
        supported_benchmark_families = payload.get("supported_benchmark_families", [])
        if not str(payload.get("base_model_name", "")).strip():
            violations.append("Qwen adapter artifact must contain a base_model_name")
        if not isinstance(runtime_paths, dict):
            violations.append("Qwen adapter artifact must contain runtime_paths")
        if not isinstance(training_dataset_manifest, dict) or int(training_dataset_manifest.get("total_examples", 0) or 0) <= 0:
            violations.append("Qwen adapter artifact must contain a non-empty training_dataset_manifest")
        if not isinstance(runtime_policy, dict) or not runtime_policy:
            violations.append("Qwen adapter artifact must contain runtime_policy")
        if not isinstance(supported_benchmark_families, list) or not supported_benchmark_families:
            violations.append("Qwen adapter artifact must contain supported_benchmark_families")
        checks.append("qwen_adapter_surface")
    if subsystem == "tooling":
        candidates = payload.get("candidates", [])
    if subsystem == "operators":
        operators = payload.get("operators", [])
        if not isinstance(operators, list) or not operators:
            violations.append("artifact must contain a non-empty operators list")
        checks.append("operators")

    _validate_artifact_profile(
        subsystem=subsystem,
        payload=payload,
        checks=checks,
        violations=violations,
    )

    if subsystem == "benchmark":
        proposals = payload.get("proposals", [])
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            if not (
                isinstance(proposal.get("failure_types"), list)
                or isinstance(proposal.get("transition_failures"), list)
                or proposal.get("command_count") is not None
            ):
                violations.append(
                    "every benchmark proposal must contain discriminative source details such as failure_types, transition_failures, or command_count"
                )

    if subsystem == "tolbert_model":
        generation_focus = str(payload.get("generation_focus", "")).strip()
        if generation_focus and generation_focus not in _TOLBERT_MODEL_GENERATION_FOCI:
            violations.append("generation_focus must be a supported tolbert_model focus")
        training_controls = payload.get("training_controls", {})
        if not isinstance(training_controls, dict) or not training_controls:
            violations.append("Tolbert model artifact must contain training_controls")
        model_surfaces = payload.get("model_surfaces", {})
        if not isinstance(model_surfaces, dict) or not model_surfaces:
            violations.append("Tolbert model artifact must contain model_surfaces")
        else:
            for key, value in model_surfaces.items():
                if key not in _TOLBERT_MODEL_SURFACE_KEYS:
                    violations.append(f"Tolbert model surface is unsupported: {key}")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert model surface {key} must be boolean")
        runtime_policy = payload.get("runtime_policy", {})
        if not isinstance(runtime_policy, dict) or not runtime_policy:
            violations.append("Tolbert model artifact must contain runtime_policy")
        else:
            extra_runtime_policy_keys = {
                "allow_trusted_primary_without_min_confidence",
                "trusted_primary_min_confidence",
            }
            for key, value in runtime_policy.items():
                if key not in _TOLBERT_RUNTIME_POLICY_KEYS and key not in extra_runtime_policy_keys:
                    violations.append(f"Tolbert runtime policy is unsupported: {key}")
                    continue
                if key in {"shadow_benchmark_families", "primary_benchmark_families"}:
                    if not isinstance(value, list):
                        violations.append(f"Tolbert runtime policy {key} must be a list")
                elif key in {"min_path_confidence", "trusted_primary_min_confidence"}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert runtime policy {key} must be numeric")
                elif key in {"primary_min_command_score"}:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert runtime policy {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert runtime policy {key} must be boolean")
        decoder_policy = payload.get("decoder_policy", {})
        if not isinstance(decoder_policy, dict) or not decoder_policy:
            violations.append("Tolbert model artifact must contain decoder_policy")
        else:
            for key, value in decoder_policy.items():
                if key not in _TOLBERT_DECODER_POLICY_KEYS:
                    violations.append(f"Tolbert decoder policy is unsupported: {key}")
                    continue
                if key == "min_stop_completion_ratio":
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert decoder policy {key} must be numeric")
                elif key == "max_task_suggestions":
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert decoder policy {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert decoder policy {key} must be boolean")
        action_generation_policy = payload.get("action_generation_policy", {})
        if action_generation_policy:
            if not isinstance(action_generation_policy, dict):
                violations.append("Tolbert action generation policy must be an object")
            else:
                for key, value in action_generation_policy.items():
                    if key not in _TOLBERT_ACTION_GENERATION_POLICY_KEYS:
                        violations.append(f"Tolbert action generation policy is unsupported: {key}")
                        continue
                    if key == "template_preferences":
                        if not isinstance(value, dict):
                            violations.append("Tolbert action generation policy template_preferences must be an object")
                            continue
                        for family, items in value.items():
                            if not str(family).strip():
                                violations.append("Tolbert action generation policy family keys must be non-empty")
                            if not isinstance(items, list):
                                violations.append(
                                    "Tolbert action generation policy template_preferences values must be lists"
                                )
                                continue
                            for item in items:
                                if not isinstance(item, dict):
                                    violations.append(
                                        "Tolbert action generation policy template preference entries must be objects"
                                    )
                                    continue
                                if not str(item.get("template_kind", "")).strip():
                                    violations.append(
                                        "Tolbert action generation policy template preference must include template_kind"
                                    )
                                support = item.get("support", 0)
                                if isinstance(support, bool) or not isinstance(support, int):
                                    violations.append(
                                        "Tolbert action generation policy template preference support must be an integer"
                                    )
                                pass_rate = item.get("pass_rate", 0.0)
                                if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
                                    violations.append(
                                        "Tolbert action generation policy template preference pass_rate must be numeric"
                                    )
                    elif key in {"max_candidates", "min_family_support"}:
                        if isinstance(value, bool) or not isinstance(value, int):
                            violations.append(f"Tolbert action generation policy {key} must be an integer")
                    elif key == "enabled":
                        if not isinstance(value, bool):
                            violations.append(f"Tolbert action generation policy {key} must be boolean")
                    elif isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert action generation policy {key} must be numeric")
        rollout_policy = payload.get("rollout_policy", {})
        if not isinstance(rollout_policy, dict) or not rollout_policy:
            violations.append("Tolbert model artifact must contain rollout_policy")
        else:
            for key, value in rollout_policy.items():
                if key not in _TOLBERT_ROLLOUT_POLICY_KEYS:
                    violations.append(f"Tolbert rollout policy is unsupported: {key}")
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(f"Tolbert rollout policy {key} must be numeric")
        liftoff_gate = payload.get("liftoff_gate", {})
        if not isinstance(liftoff_gate, dict) or not liftoff_gate:
            violations.append("Tolbert model artifact must contain liftoff_gate")
        else:
            extra_liftoff_gate_keys = {
                "require_primary_routing_signal",
                "min_primary_episodes",
                "allow_selection_signal_fallback",
            }
            for key, value in liftoff_gate.items():
                if key not in _TOLBERT_LIFTOFF_GATE_KEYS and key not in extra_liftoff_gate_keys:
                    violations.append(f"Tolbert liftoff gate is unsupported: {key}")
                    continue
                if key == "proposal_gate_by_benchmark_family":
                    if not isinstance(value, dict):
                        violations.append("Tolbert liftoff gate proposal_gate_by_benchmark_family must be an object")
                        continue
                    for family, family_gate in value.items():
                        if not str(family).strip():
                            violations.append(
                                "Tolbert liftoff gate proposal_gate_by_benchmark_family keys must be non-empty"
                            )
                        if not isinstance(family_gate, dict):
                            violations.append(
                                "Tolbert liftoff gate proposal_gate_by_benchmark_family values must be objects"
                            )
                            continue
                        for family_key, family_value in family_gate.items():
                            if family_key not in {
                                "require_novel_command_signal",
                                "min_proposal_selected_steps_delta",
                                "min_novel_valid_command_steps",
                                "min_novel_valid_command_rate_delta",
                                "allow_primary_routing_signal",
                                "min_primary_episodes",
                            }:
                                violations.append(
                                    f"Tolbert liftoff family proposal gate is unsupported: {family_key}"
                                )
                                continue
                            if family_key in {"require_novel_command_signal", "allow_primary_routing_signal"}:
                                if not isinstance(family_value, bool):
                                    violations.append(
                                        f"Tolbert liftoff family proposal gate {family_key} must be boolean"
                                    )
                            elif family_key == "min_novel_valid_command_rate_delta":
                                if isinstance(family_value, bool) or not isinstance(family_value, (int, float)):
                                    violations.append(
                                        "Tolbert liftoff family proposal gate min_novel_valid_command_rate_delta must be numeric"
                                    )
                            elif isinstance(family_value, bool) or not isinstance(family_value, int):
                                violations.append(
                                    f"Tolbert liftoff family proposal gate {family_key} must be an integer"
                                )
                elif key in {
                    "min_pass_rate_delta",
                    "max_step_regression",
                    "max_takeover_drift_pass_rate_regression",
                    "max_takeover_drift_unsafe_ambiguous_rate_regression",
                    "max_takeover_drift_hidden_side_effect_rate_regression",
                    "max_takeover_drift_trust_success_rate_regression",
                    "max_takeover_drift_trust_unsafe_ambiguous_rate_regression",
                }:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        violations.append(f"Tolbert liftoff gate {key} must be numeric")
                elif key in {
                    "max_regressed_families",
                    "min_shadow_episodes_per_promoted_family",
                    "min_primary_episodes",
                    "takeover_drift_step_budget",
                    "takeover_drift_wave_task_limit",
                    "takeover_drift_max_waves",
                }:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert liftoff gate {key} must be an integer")
                elif not isinstance(value, bool):
                    violations.append(f"Tolbert liftoff gate {key} must be boolean")
        build_policy = payload.get("build_policy", {})
        if not isinstance(build_policy, dict) or not build_policy:
            violations.append("Tolbert model artifact must contain build_policy")
        else:
            for key, value in build_policy.items():
                if key not in _TOLBERT_BUILD_POLICY_KEYS:
                    violations.append(f"Tolbert build policy is unsupported: {key}")
                    continue
                if key in {
                    "allow_kernel_autobuild",
                    "allow_kernel_rebuild",
                    "require_synthetic_dataset",
                    "require_head_targets",
                    "require_long_horizon_head_targets",
                }:
                    if not isinstance(value, bool):
                        violations.append(f"Tolbert build policy {key} must be boolean")
                else:
                    if isinstance(value, bool) or not isinstance(value, int):
                        violations.append(f"Tolbert build policy {key} must be an integer")
        runtime_paths = payload.get("runtime_paths", {})
        if isinstance(runtime_paths, dict):
            for key in (
                "config_path",
                "checkpoint_path",
                "nodes_path",
                "label_map_path",
                "source_spans_paths",
                "cache_paths",
            ):
                if not runtime_paths.get(key):
                    violations.append(f"Tolbert model runtime_paths must include {key}")
        checks.append("tolbert_model_surface")
        checks.append("tolbert_model_runtime_paths")
    if subsystem == "qwen_adapter":
        runtime_policy = payload.get("runtime_policy", {})
        retention_gate = payload.get("retention_gate", {})
        runtime_paths = payload.get("runtime_paths", {})
        if isinstance(runtime_policy, dict):
            if bool(runtime_policy.get("allow_primary_routing", False)):
                violations.append("Qwen adapter runtime_policy.allow_primary_routing must remain false before liftoff")
            if not bool(runtime_policy.get("allow_teacher_generation", False)):
                violations.append("Qwen adapter runtime_policy.allow_teacher_generation must be true")
        if isinstance(retention_gate, dict) and not bool(retention_gate.get("disallow_liftoff_authority", False)):
            violations.append("Qwen adapter retention_gate.disallow_liftoff_authority must be true")
        if isinstance(runtime_paths, dict):
            runtime_target = str(
                runtime_paths.get("served_model_name")
                or runtime_paths.get("merged_output_dir")
                or runtime_paths.get("adapter_output_dir")
            ).strip()
            if not runtime_target:
                violations.append("Qwen adapter runtime_paths must declare served_model_name, merged_output_dir, or adapter_output_dir")
        checks.append("qwen_adapter_runtime_paths")
    if subsystem == "universe":
        artifact_kind = str(payload.get("artifact_kind", "")).strip()
        control_schema = str(payload.get("control_schema", "")).strip()
        if artifact_kind == "universe_constitution":
            if control_schema != "universe_constitution_v1":
                violations.append("universe constitution artifacts must declare control_schema universe_constitution_v1")
        elif artifact_kind == "operating_envelope":
            if control_schema != "operating_envelope_v1":
                violations.append("operating envelope artifacts must declare control_schema operating_envelope_v1")
        elif control_schema != "universe_contract_v1":
            violations.append("universe artifacts must declare control_schema universe_contract_v1")
    if subsystem == "tooling":
        candidates = payload.get("candidates", [])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_lifecycle_state = _tool_candidate_lifecycle_state(candidate)
            promotion_stage = _tool_candidate_stage(candidate.get("promotion_stage", ""))
            if promotion_stage:
                allowed_stage_states = {
                    "candidate_procedure": "candidate",
                    "replay_verified": "replay_verified",
                    "promoted_tool": "retained",
                    "rejected": "rejected",
                }
                expected_lifecycle_state = allowed_stage_states.get(promotion_stage, "")
                if not expected_lifecycle_state:
                    violations.append("every tool candidate promotion_stage must be candidate_procedure, replay_verified, promoted_tool, or rejected")
                elif candidate_lifecycle_state and candidate_lifecycle_state != expected_lifecycle_state:
                    violations.append(
                        f"tool candidate lifecycle_state must be {expected_lifecycle_state} when promotion_stage is {promotion_stage}"
                    )
        expected_top_level_stage = {
            "candidate": "candidate_procedure",
            "replay_verified": "replay_verified",
            "retained": "promoted_tool",
            "rejected": "rejected",
        }.get(lifecycle_state, "")
        if expected_top_level_stage and isinstance(candidates, list) and candidates and not _tool_candidates_have_stage(
            payload,
            expected_top_level_stage,
        ):
            violations.append(
                f"tool artifact lifecycle_state {lifecycle_state} requires all candidates to be in promotion_stage {expected_top_level_stage}"
            )
    if subsystem == "operators":
        operators = payload.get("operators", [])
        for operator in operators:
            if not isinstance(operator, dict):
                violations.append("every operator must be an object")
                continue
            if not str(operator.get("operator_id", "")).strip():
                violations.append("every operator must contain an operator_id")
            if _operator_support(operator) <= 0:
                violations.append("every operator must contain a positive support value")
            if not _operator_benchmark_families(operator):
                violations.append("every operator must contain benchmark_families")
            if not _operator_steps(operator):
                violations.append("every operator must contain non-empty steps")
            if not _operator_task_contract(operator):
                violations.append("every operator must contain a task_contract")
        checks.append("operator_ids")
        checks.append("operator_contracts")
    if subsystem == "capabilities":
        config_field_names = {entry.name for entry in fields(KernelConfig)}
        modules = payload.get("modules", [])
        for module in modules:
            if not isinstance(module, dict):
                continue
            settings = module.get("settings", {})
            if not isinstance(settings, dict):
                continue
            improvement_subsystems = settings.get("improvement_subsystems", [])
            if improvement_subsystems is None:
                continue
            if not isinstance(improvement_subsystems, list):
                violations.append("capability module improvement_subsystems must be a list")
                continue
            for surface in improvement_subsystems:
                if not isinstance(surface, dict):
                    violations.append("every capability improvement surface must be an object")
                    continue
                if not str(surface.get("subsystem_id", "")).strip():
                    violations.append("every capability improvement surface must contain a subsystem_id")
                base_subsystem = str(surface.get("base_subsystem", "")).strip()
                if base_subsystem not in {
                    "benchmark",
                    "retrieval",
                    "tolbert_model",
                    "verifier",
                    "policy",
                    "universe",
                    "world_model",
                    "state_estimation",
                    "trust",
                    "recovery",
                    "delegation",
                    "operator_policy",
                    "transition_model",
                    "curriculum",
                    "tooling",
                    "skills",
                    "operators",
                    "capabilities",
                }:
                    violations.append("capability improvement surfaces must declare a supported base_subsystem")
                artifact_path_attr = str(surface.get("artifact_path_attr", "")).strip()
                if artifact_path_attr and artifact_path_attr not in config_field_names:
                    violations.append("capability improvement surfaces must declare a valid config artifact_path_attr")
                proposal_toggle_attr = str(surface.get("proposal_toggle_attr", "")).strip()
                if proposal_toggle_attr and proposal_toggle_attr not in config_field_names:
                    violations.append("capability improvement surfaces must declare a valid config proposal_toggle_attr")
                generator_kind = str(surface.get("generator_kind", "")).strip()
                if generator_kind and generator_kind != base_subsystem:
                    violations.append("capability improvement surfaces must use the base_subsystem generator_kind")
                artifact_kind_override = str(surface.get("artifact_kind", "")).strip()
                expected_artifact_kind = str(_CAPABILITY_SURFACE_ARTIFACT_KINDS.get(base_subsystem, "")).strip()
                if artifact_kind_override and artifact_kind_override != expected_artifact_kind:
                    violations.append("capability improvement surfaces must use the base_subsystem artifact_kind")
        checks.append("capability_modules")

    semantic_violations = semantic_compatibility_violations(
        subsystem=subsystem,
        payload=payload,
        task_catalog=task_catalog,
    )
    if semantic_violations:
        violations.extend(semantic_violations)
    checks.append("semantic_source_contract")

    return {
        "compatible": not violations,
        "checked_rules": checks,
        "violations": violations,
    }


__all__ = ["assess_artifact_compatibility"]
