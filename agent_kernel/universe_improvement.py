from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from evals.metrics import EvalMetrics

from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    filter_proposals_by_area,
    merged_string_lists,
    normalized_generation_focus,
    normalized_control_mapping,
    normalized_string_list,
    retained_mapping_section,
    retained_sequence_section,
    retention_gate_preset,
)

UNIVERSE_PROPOSAL_AREAS = {
    "governance",
    "verification",
    "operator_scope",
    "environment_envelope",
}
UNIVERSE_GENERATION_FOCI = {
    "balanced",
    "governance",
    "verification",
    "operator_scope",
    "environment_envelope",
}
UNIVERSE_GOVERNANCE_KEYS = {
    "require_verification",
    "require_bounded_steps",
    "prefer_reversible_actions",
    "respect_task_forbidden_artifacts",
    "respect_preserved_artifacts",
}
UNIVERSE_ACTION_RISK_CONTROL_KEYS = {
    "destructive_mutation_penalty",
    "git_mutation_penalty",
    "inline_destructive_interpreter_penalty",
    "network_fetch_penalty",
    "privileged_command_penalty",
    "read_only_discovery_bonus",
    "remote_execution_penalty",
    "reversible_file_operation_bonus",
    "scope_escape_penalty",
    "unbounded_execution_penalty",
    "verification_bonus",
}
UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS = {
    "git_write_mode": {"blocked", "operator_gated", "task_scoped"},
    "network_access_mode": {"blocked", "allowlist_only", "open"},
    "workspace_write_scope": {"task_only", "generated_only", "shared_repo_gated"},
}
UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS = {
    "require_path_scoped_mutations",
    "require_rollback_on_mutation",
}

_DEFAULT_INVARIANTS = [
    "preserve verifier contract alignment",
    "prefer measurable reversible progress",
    "avoid destructive workspace resets",
    "keep autonomous actions inside bounded runtime policy",
]
_DEFAULT_FORBIDDEN_PATTERNS = [
    "rm -rf /",
    "git reset --hard",
    "git checkout --",
    "curl -s",
    "curl -fsSL",
    "wget -qO-",
    "sudo ",
]
_DEFAULT_PREFERRED_PREFIXES = [
    "rg ",
    "sed -n ",
    "cat ",
    "ls ",
    "test ",
    "pytest",
    "python -m pytest",
    "git diff",
    "git status",
]
DEFAULT_UNIVERSE_ACTION_RISK_CONTROLS = {
    "destructive_mutation_penalty": 12,
    "git_mutation_penalty": 6,
    "inline_destructive_interpreter_penalty": 8,
    "network_fetch_penalty": 4,
    "privileged_command_penalty": 10,
    "read_only_discovery_bonus": 3,
    "remote_execution_penalty": 8,
    "reversible_file_operation_bonus": 2,
    "scope_escape_penalty": 9,
    "unbounded_execution_penalty": 7,
    "verification_bonus": 4,
}
DEFAULT_UNIVERSE_ENVIRONMENT_ASSUMPTIONS = {
    "git_write_mode": "operator_gated",
    "network_access_mode": "blocked",
    "workspace_write_scope": "task_only",
    "require_path_scoped_mutations": True,
    "require_rollback_on_mutation": True,
}
_UNIVERSE_CONSTITUTION_PROPOSAL_AREAS = {"governance", "verification", "operator_scope"}
_OPERATING_ENVELOPE_PROPOSAL_AREAS = {"environment_envelope", "operator_scope"}
_UNIVERSE_BUNDLE_FILENAMES = {
    "universe": "universe_contract.json",
    "universe_constitution": "universe_constitution.json",
    "operating_envelope": "operating_envelope.json",
}


def build_universe_contract_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    environment_violation_summary: dict[str, object] | None = None,
    cycle_feedback_summary: dict[str, object] | None = None,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    normalized_focus = None if generation_focus == "balanced" else generation_focus
    environment_evidence = environment_violation_summary if isinstance(environment_violation_summary, dict) else {}
    cycle_feedback = cycle_feedback_summary if isinstance(cycle_feedback_summary, dict) else {}
    return build_standard_proposal_artifact(
        artifact_kind="universe_contract",
        generation_focus=generation_focus,
        control_schema="universe_contract_v1",
        retention_gate=retention_gate_preset("universe"),
        proposals=_proposals(
            metrics,
            failure_counts,
            environment_evidence=environment_evidence,
            cycle_feedback=cycle_feedback,
            focus=normalized_focus,
        ),
        extra_sections={
            "invariants": universe_invariants(metrics, failure_counts, focus=normalized_focus, baseline=retained_universe_invariants(current_payload)),
            "forbidden_command_patterns": universe_forbidden_command_patterns(
                metrics,
                failure_counts,
                focus=normalized_focus,
                baseline=retained_universe_forbidden_command_patterns(current_payload),
            ),
            "preferred_command_prefixes": universe_preferred_command_prefixes(
                metrics,
                failure_counts,
                focus=normalized_focus,
                baseline=retained_universe_preferred_command_prefixes(current_payload),
            ),
            "governance": universe_governance_controls(
                metrics,
                failure_counts,
                focus=normalized_focus,
                baseline=retained_universe_governance(current_payload),
            ),
            "action_risk_controls": universe_action_risk_controls(
                metrics,
                failure_counts,
                environment_evidence=environment_evidence,
                cycle_feedback=cycle_feedback,
                focus=normalized_focus,
                baseline=retained_universe_action_risk_controls(current_payload),
            ),
            "environment_assumptions": universe_environment_assumptions(
                metrics,
                failure_counts,
                environment_evidence=environment_evidence,
                cycle_feedback=cycle_feedback,
                focus=normalized_focus,
                baseline=retained_universe_environment_assumptions(current_payload),
            ),
        },
    )


def build_universe_constitution_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    baseline_payload = _universe_payload_from_any(current_payload)
    combined = build_universe_contract_artifact(
        metrics,
        failure_counts,
        focus=focus,
        current_payload=baseline_payload,
    )
    return {
        "spec_version": "asi_v1",
        "artifact_kind": "universe_constitution",
        "lifecycle_state": str(combined.get("lifecycle_state", "proposed")).strip() or "proposed",
        "generation_focus": str(combined.get("generation_focus", "balanced")).strip() or "balanced",
        "control_schema": "universe_constitution_v1",
        "retention_gate": dict(combined.get("retention_gate", retention_gate_preset("universe"))),
        "governance": dict(combined.get("governance", {})),
        "invariants": list(combined.get("invariants", [])),
        "forbidden_command_patterns": list(combined.get("forbidden_command_patterns", [])),
        "preferred_command_prefixes": list(combined.get("preferred_command_prefixes", [])),
        "proposals": filter_proposals_by_area(
            list(combined.get("proposals", [])),
            allowed_areas=_UNIVERSE_CONSTITUTION_PROPOSAL_AREAS,
        ),
    }


def build_operating_envelope_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    environment_violation_summary: dict[str, object] | None = None,
    cycle_feedback_summary: dict[str, object] | None = None,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    baseline_payload = _universe_payload_from_any(current_payload)
    combined = build_universe_contract_artifact(
        metrics,
        failure_counts,
        environment_violation_summary=environment_violation_summary,
        cycle_feedback_summary=cycle_feedback_summary,
        focus=focus,
        current_payload=baseline_payload,
    )
    return {
        "spec_version": "asi_v1",
        "artifact_kind": "operating_envelope",
        "lifecycle_state": str(combined.get("lifecycle_state", "proposed")).strip() or "proposed",
        "generation_focus": str(combined.get("generation_focus", "balanced")).strip() or "balanced",
        "control_schema": "operating_envelope_v1",
        "retention_gate": dict(combined.get("retention_gate", retention_gate_preset("universe"))),
        "action_risk_controls": dict(combined.get("action_risk_controls", {})),
        "environment_assumptions": dict(combined.get("environment_assumptions", {})),
        "allowed_http_hosts": [],
        "writable_path_prefixes": [],
        "toolchain_requirements": ["git", "python", "pytest", "rg"],
        "learned_calibration_priors": {
            "selected_variant_weights": dict(
                cycle_feedback_summary.get("selected_variant_weights", {})
                if isinstance(cycle_feedback_summary, dict)
                else {}
            ),
            "successful_environment_assumption_weights": dict(
                cycle_feedback_summary.get("successful_environment_assumption_weights", {})
                if isinstance(cycle_feedback_summary, dict)
                else {}
            ),
        },
        "proposals": filter_proposals_by_area(
            list(combined.get("proposals", [])),
            allowed_areas=_OPERATING_ENVELOPE_PROPOSAL_AREAS,
        ),
    }


def materialize_universe_constitution_payload(
    payload: object | None,
    *,
    lifecycle_state: str = "retained",
) -> dict[str, object]:
    combined = _universe_payload_from_any(payload) or {}
    source = payload if isinstance(payload, dict) else combined
    materialized = {
        "spec_version": str(combined.get("spec_version", "asi_v1")).strip() or "asi_v1",
        "artifact_kind": "universe_constitution",
        "lifecycle_state": str(lifecycle_state).strip() or "retained",
        "generation_focus": str(combined.get("generation_focus", "balanced")).strip() or "balanced",
        "control_schema": "universe_constitution_v1",
        "retention_gate": dict(source.get("retention_gate", retention_gate_preset("universe"))),
        "governance": dict(combined.get("governance", {})),
        "invariants": list(combined.get("invariants", [])),
        "forbidden_command_patterns": list(combined.get("forbidden_command_patterns", [])),
        "preferred_command_prefixes": list(combined.get("preferred_command_prefixes", [])),
        "proposals": filter_proposals_by_area(
            list(combined.get("proposals", [])),
            allowed_areas=_UNIVERSE_CONSTITUTION_PROPOSAL_AREAS,
        ),
    }
    if isinstance(source, dict) and isinstance(source.get("retention_decision"), dict):
        materialized["retention_decision"] = deepcopy(source.get("retention_decision", {}))
    return materialized


def materialize_operating_envelope_payload(
    payload: object | None,
    *,
    lifecycle_state: str = "retained",
) -> dict[str, object]:
    combined = _universe_payload_from_any(payload) or {}
    source = payload if isinstance(payload, dict) else combined
    materialized = {
        "spec_version": str(combined.get("spec_version", "asi_v1")).strip() or "asi_v1",
        "artifact_kind": "operating_envelope",
        "lifecycle_state": str(lifecycle_state).strip() or "retained",
        "generation_focus": str(combined.get("generation_focus", "balanced")).strip() or "balanced",
        "control_schema": "operating_envelope_v1",
        "retention_gate": dict(source.get("retention_gate", retention_gate_preset("universe"))),
        "action_risk_controls": dict(combined.get("action_risk_controls", {})),
        "environment_assumptions": dict(combined.get("environment_assumptions", {})),
        "allowed_http_hosts": list(source.get("allowed_http_hosts", combined.get("allowed_http_hosts", []))),
        "writable_path_prefixes": list(source.get("writable_path_prefixes", combined.get("writable_path_prefixes", []))),
        "toolchain_requirements": list(source.get("toolchain_requirements", combined.get("toolchain_requirements", []))),
        "learned_calibration_priors": dict(
            source.get("learned_calibration_priors", combined.get("learned_calibration_priors", {}))
        ),
        "proposals": filter_proposals_by_area(
            list(combined.get("proposals", [])),
            allowed_areas=_OPERATING_ENVELOPE_PROPOSAL_AREAS,
        ),
    }
    if isinstance(source, dict) and isinstance(source.get("retention_decision"), dict):
        materialized["retention_decision"] = deepcopy(source.get("retention_decision", {}))
    return materialized


def compose_universe_contract_payload(
    *,
    constitution_payload: object | None = None,
    operating_envelope_payload: object | None = None,
    baseline_payload: object | None = None,
    lifecycle_state: str = "retained",
) -> dict[str, object]:
    constitution = materialize_universe_constitution_payload(
        constitution_payload if constitution_payload is not None else baseline_payload,
        lifecycle_state=lifecycle_state,
    )
    envelope = materialize_operating_envelope_payload(
        operating_envelope_payload if operating_envelope_payload is not None else baseline_payload,
        lifecycle_state=lifecycle_state,
    )
    baseline = _universe_payload_from_any(baseline_payload) or {}
    retention_gate = retention_gate_preset("universe")
    for candidate in (baseline_payload, constitution_payload, operating_envelope_payload):
        if isinstance(candidate, dict) and isinstance(candidate.get("retention_gate"), dict):
            retention_gate = dict(candidate.get("retention_gate", {}))
            break
    proposals = _merged_universe_proposals(
        baseline.get("proposals", []),
        constitution.get("proposals", []),
        envelope.get("proposals", []),
    )
    combined = {
        "spec_version": str(baseline.get("spec_version", "asi_v1")).strip() or "asi_v1",
        "artifact_kind": "universe_contract",
        "lifecycle_state": str(lifecycle_state).strip() or "retained",
        "generation_focus": _merged_universe_generation_focus(
            baseline,
            constitution_payload,
            operating_envelope_payload,
        ),
        "control_schema": "universe_contract_v1",
        "retention_gate": retention_gate,
        "governance": dict(constitution.get("governance", {})),
        "invariants": list(constitution.get("invariants", [])),
        "forbidden_command_patterns": list(constitution.get("forbidden_command_patterns", [])),
        "preferred_command_prefixes": list(constitution.get("preferred_command_prefixes", [])),
        "action_risk_controls": dict(envelope.get("action_risk_controls", {})),
        "environment_assumptions": dict(envelope.get("environment_assumptions", {})),
        "allowed_http_hosts": list(envelope.get("allowed_http_hosts", [])),
        "writable_path_prefixes": list(envelope.get("writable_path_prefixes", [])),
        "toolchain_requirements": list(envelope.get("toolchain_requirements", [])),
        "learned_calibration_priors": dict(envelope.get("learned_calibration_priors", {})),
        "proposals": proposals,
    }
    for candidate in (constitution_payload, operating_envelope_payload, baseline_payload):
        if isinstance(candidate, dict) and isinstance(candidate.get("retention_decision"), dict):
            combined["retention_decision"] = deepcopy(candidate.get("retention_decision", {}))
            break
    return combined


def universe_bundle_lifecycle_state(*payloads: object | None, default: str = "retained") -> str:
    for payload in payloads:
        if isinstance(payload, dict):
            value = str(payload.get("lifecycle_state", "")).strip()
            if value:
                return value
    return default


def compose_universe_bundle_payloads(
    *,
    constitution_payload: object | None = None,
    operating_envelope_payload: object | None = None,
    baseline_payload: object | None = None,
    lifecycle_state: str | None = None,
) -> dict[str, dict[str, object]]:
    resolved_lifecycle_state = str(
        lifecycle_state
        or universe_bundle_lifecycle_state(
            constitution_payload,
            operating_envelope_payload,
            baseline_payload,
        )
    ).strip() or "retained"
    constitution = materialize_universe_constitution_payload(
        constitution_payload if constitution_payload is not None else baseline_payload,
        lifecycle_state=resolved_lifecycle_state,
    )
    operating_envelope = materialize_operating_envelope_payload(
        operating_envelope_payload if operating_envelope_payload is not None else baseline_payload,
        lifecycle_state=resolved_lifecycle_state,
    )
    combined = compose_universe_contract_payload(
        constitution_payload=constitution,
        operating_envelope_payload=operating_envelope,
        baseline_payload=baseline_payload,
        lifecycle_state=resolved_lifecycle_state,
    )
    return {
        "universe": combined,
        "universe_constitution": constitution,
        "operating_envelope": operating_envelope,
    }


def write_universe_bundle_files(
    *,
    contract_path: Path,
    constitution_path: Path,
    operating_envelope_path: Path,
    bundle: dict[str, dict[str, object]],
) -> dict[str, str]:
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    constitution_path.parent.mkdir(parents=True, exist_ok=True)
    operating_envelope_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(bundle.get("universe", {}), indent=2), encoding="utf-8")
    constitution_path.write_text(json.dumps(bundle.get("universe_constitution", {}), indent=2), encoding="utf-8")
    operating_envelope_path.write_text(
        json.dumps(bundle.get("operating_envelope", {}), indent=2),
        encoding="utf-8",
    )
    return {
        "universe": str(contract_path),
        "universe_constitution": str(constitution_path),
        "operating_envelope": str(operating_envelope_path),
    }


def universe_bundle_paths(
    *,
    universe_contract_path: Path,
    universe_constitution_path: Path,
    operating_envelope_path: Path,
) -> dict[str, Path]:
    return {
        "universe": universe_contract_path,
        "universe_constitution": universe_constitution_path,
        "operating_envelope": operating_envelope_path,
    }


def sibling_universe_bundle_paths(anchor_path: Path) -> dict[str, Path]:
    parent = anchor_path.parent
    return {
        key: parent / filename
        for key, filename in _UNIVERSE_BUNDLE_FILENAMES.items()
    }


def comparison_shadow_universe_bundle_paths(artifact_path: Path) -> dict[str, Path]:
    shadow_root = artifact_path.parent / ".comparison_artifacts"
    return {
        "universe": shadow_root / f"{artifact_path.stem}.universe_contract.json",
        "universe_constitution": shadow_root / f"{artifact_path.stem}.universe_constitution.json",
        "operating_envelope": shadow_root / f"{artifact_path.stem}.operating_envelope.json",
    }


def universe_bundle_contains_path(bundle_paths: dict[str, Path], candidate_path: Path) -> bool:
    resolved_candidate = candidate_path.resolve(strict=False)
    return any(resolved_candidate == path.resolve(strict=False) for path in bundle_paths.values())


def universe_governance_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "require_verification": True,
        "require_bounded_steps": True,
        "prefer_reversible_actions": True,
        "respect_task_forbidden_artifacts": True,
        "respect_preserved_artifacts": True,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if focus == "verification" or metrics.low_confidence_episodes > 0:
        controls["require_verification"] = True
    if focus == "governance" or failure_counts.get("command_failure", 0) > 0:
        controls["prefer_reversible_actions"] = True
        controls["require_bounded_steps"] = True
    if failure_counts.get("state_regression", 0) > 0:
        controls["respect_preserved_artifacts"] = True
    return {key: bool(controls.get(key, False)) for key in sorted(UNIVERSE_GOVERNANCE_KEYS)}


def universe_invariants(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: object | None = None,
) -> list[str]:
    invariants = merged_string_lists(_DEFAULT_INVARIANTS, baseline)
    if metrics.low_confidence_episodes > 0 or focus == "verification":
        invariants = merged_string_lists(invariants, ["verify before accepting terminal success"])
    if failure_counts.get("state_regression", 0) > 0 or focus == "governance":
        invariants = merged_string_lists(invariants, ["protect preserved artifacts before widening action scope"])
    return invariants


def universe_forbidden_command_patterns(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: object | None = None,
) -> list[str]:
    patterns = merged_string_lists(_DEFAULT_FORBIDDEN_PATTERNS, baseline)
    if failure_counts.get("command_failure", 0) > 0 or focus == "governance":
        patterns = merged_string_lists(patterns, ["curl | sh", "git clean -fd"])
    if metrics.low_confidence_episodes > 0 or focus == "operator_scope":
        patterns = merged_string_lists(patterns, ["python -c \"import shutil; shutil.rmtree"])
    return patterns


def universe_preferred_command_prefixes(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: object | None = None,
) -> list[str]:
    prefixes = merged_string_lists(_DEFAULT_PREFERRED_PREFIXES, baseline)
    if metrics.low_confidence_episodes > 0 or focus == "verification":
        prefixes = merged_string_lists(prefixes, ["python -m pytest -q", "test -f "])
    if failure_counts.get("missing_expected_file", 0) > 0:
        prefixes = merged_string_lists(prefixes, ["mkdir -p ", "cp "])
    return prefixes


def universe_action_risk_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    environment_evidence: dict[str, object] | None = None,
    cycle_feedback: dict[str, object] | None = None,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, int]:
    controls = dict(DEFAULT_UNIVERSE_ACTION_RISK_CONTROLS)
    evidence = environment_evidence if isinstance(environment_evidence, dict) else {}
    feedback = cycle_feedback if isinstance(cycle_feedback, dict) else {}
    violation_counts = evidence.get("violation_counts", {})
    if not isinstance(violation_counts, dict):
        violation_counts = {}
    successful_action_risk_control_floor = feedback.get("successful_action_risk_control_floor", {})
    if not isinstance(successful_action_risk_control_floor, dict):
        successful_action_risk_control_floor = {}
    successful_action_risk_control_weighted_mean = feedback.get("successful_action_risk_control_weighted_mean", {})
    if not isinstance(successful_action_risk_control_weighted_mean, dict):
        successful_action_risk_control_weighted_mean = {}
    controls.update(
        normalized_control_mapping(
            baseline,
            positive_int_fields=tuple(sorted(UNIVERSE_ACTION_RISK_CONTROL_KEYS)),
        )
    )
    if focus == "verification" or metrics.low_confidence_episodes > 0:
        controls["verification_bonus"] = max(int(controls["verification_bonus"]), 6)
        controls["read_only_discovery_bonus"] = max(int(controls["read_only_discovery_bonus"]), 4)
    if focus == "governance" or failure_counts.get("command_failure", 0) > 0:
        controls["destructive_mutation_penalty"] = max(int(controls["destructive_mutation_penalty"]), 14)
        controls["git_mutation_penalty"] = max(int(controls["git_mutation_penalty"]), 8)
        controls["remote_execution_penalty"] = max(int(controls["remote_execution_penalty"]), 10)
    if focus == "operator_scope" or failure_counts.get("state_regression", 0) > 0:
        controls["inline_destructive_interpreter_penalty"] = max(
            int(controls["inline_destructive_interpreter_penalty"]),
            10,
        )
        controls["unbounded_execution_penalty"] = max(int(controls["unbounded_execution_penalty"]), 8)
    if failure_counts.get("missing_expected_file", 0) > 0:
        controls["reversible_file_operation_bonus"] = max(int(controls["reversible_file_operation_bonus"]), 3)
    if int(violation_counts.get("network_access_conflict", 0)) > 0:
        controls["network_fetch_penalty"] = max(int(controls["network_fetch_penalty"]), 6)
        controls["remote_execution_penalty"] = max(int(controls["remote_execution_penalty"]), 10)
    if int(violation_counts.get("git_write_conflict", 0)) > 0:
        controls["git_mutation_penalty"] = max(int(controls["git_mutation_penalty"]), 8)
    if int(violation_counts.get("path_scope_conflict", 0)) > 0:
        controls["scope_escape_penalty"] = max(int(controls["scope_escape_penalty"]), 11)
    for key, value in successful_action_risk_control_floor.items():
        if key not in UNIVERSE_ACTION_RISK_CONTROL_KEYS:
            continue
        try:
            controls[key] = max(int(controls.get(key, 0)), int(value))
        except (TypeError, ValueError):
            continue
    for key, value in successful_action_risk_control_weighted_mean.items():
        if key not in UNIVERSE_ACTION_RISK_CONTROL_KEYS:
            continue
        try:
            controls[key] = max(int(controls.get(key, 0)), int(round(float(value))))
        except (TypeError, ValueError):
            continue
    return {
        key: int(controls.get(key, DEFAULT_UNIVERSE_ACTION_RISK_CONTROLS[key]))
        for key in sorted(UNIVERSE_ACTION_RISK_CONTROL_KEYS)
    }


def universe_environment_assumptions(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    environment_evidence: dict[str, object] | None = None,
    cycle_feedback: dict[str, object] | None = None,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    assumptions = dict(DEFAULT_UNIVERSE_ENVIRONMENT_ASSUMPTIONS)
    evidence = environment_evidence if isinstance(environment_evidence, dict) else {}
    feedback = cycle_feedback if isinstance(cycle_feedback, dict) else {}
    observed_modes = evidence.get("observed_environment_modes", {})
    if not isinstance(observed_modes, dict):
        observed_modes = {}
    alignment_failure_counts = evidence.get("alignment_failure_counts", {})
    if not isinstance(alignment_failure_counts, dict):
        alignment_failure_counts = {}
    successful_environment_assumptions = feedback.get("successful_environment_assumptions", {})
    if not isinstance(successful_environment_assumptions, dict):
        successful_environment_assumptions = {}
    assumptions.update(_normalized_environment_assumptions(baseline))
    for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
        if int(alignment_failure_counts.get(_alignment_key_for_field(field), 0)) > 0:
            observed = _dominant_observed_environment_mode(observed_modes, field)
            if observed:
                assumptions[field] = observed
        elif str(successful_environment_assumptions.get(field, "")).strip():
            assumptions[field] = str(successful_environment_assumptions.get(field, "")).strip().lower()
    if focus == "governance" or failure_counts.get("command_failure", 0) > 0:
        assumptions["network_access_mode"] = _more_restrictive_mode(
            "network_access_mode",
            str(assumptions.get("network_access_mode", "blocked")),
            "allowlist_only",
        )
        assumptions["git_write_mode"] = _more_restrictive_mode(
            "git_write_mode",
            str(assumptions.get("git_write_mode", "operator_gated")),
            "operator_gated",
        )
    if focus == "operator_scope" or failure_counts.get("state_regression", 0) > 0:
        assumptions["workspace_write_scope"] = _more_restrictive_mode(
            "workspace_write_scope",
            str(assumptions.get("workspace_write_scope", "task_only")),
            "task_only",
        )
        assumptions["require_path_scoped_mutations"] = True
    if focus == "verification" or metrics.low_confidence_episodes > 0:
        assumptions["require_rollback_on_mutation"] = True
    return _normalized_environment_assumptions(assumptions)


def retained_universe_governance(payload: object) -> dict[str, object]:
    controls = retained_mapping_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="governance",
    )
    return {key: bool(controls.get(key, False)) for key in sorted(UNIVERSE_GOVERNANCE_KEYS) if key in controls}


def retained_universe_invariants(payload: object) -> list[str]:
    values = retained_sequence_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="invariants",
    )
    return normalized_string_list(values)


def retained_universe_forbidden_command_patterns(payload: object) -> list[str]:
    values = retained_sequence_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="forbidden_command_patterns",
    )
    return normalized_string_list(values)


def retained_universe_preferred_command_prefixes(payload: object) -> list[str]:
    values = retained_sequence_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="preferred_command_prefixes",
    )
    return normalized_string_list(values)


def retained_universe_action_risk_controls(payload: object) -> dict[str, int]:
    controls = retained_mapping_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="action_risk_controls",
    )
    return {
        key: int(value)
        for key, value in normalized_control_mapping(
            controls,
            positive_int_fields=tuple(sorted(UNIVERSE_ACTION_RISK_CONTROL_KEYS)),
        ).items()
        if key in UNIVERSE_ACTION_RISK_CONTROL_KEYS
    }


def retained_universe_environment_assumptions(payload: object) -> dict[str, object]:
    assumptions = retained_mapping_section(
        _universe_payload_from_any(payload),
        artifact_kind="universe_contract",
        section="environment_assumptions",
    )
    return _normalized_environment_assumptions(assumptions)


def _universe_payload_from_any(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    artifact_kind = str(payload.get("artifact_kind", "")).strip()
    if artifact_kind == "universe_contract":
        return payload
    if artifact_kind == "universe_constitution":
        rebound = {
            "spec_version": str(payload.get("spec_version", "asi_v1")).strip() or "asi_v1",
            "artifact_kind": "universe_contract",
            "lifecycle_state": str(payload.get("lifecycle_state", "retained")).strip() or "retained",
            "governance": dict(payload.get("governance", {})),
            "invariants": list(payload.get("invariants", [])),
            "forbidden_command_patterns": list(payload.get("forbidden_command_patterns", [])),
            "preferred_command_prefixes": list(payload.get("preferred_command_prefixes", [])),
        }
        if "retention_decision" in payload:
            rebound["retention_decision"] = payload.get("retention_decision", {})
        return rebound
    if artifact_kind == "operating_envelope":
        rebound = {
            "spec_version": str(payload.get("spec_version", "asi_v1")).strip() or "asi_v1",
            "artifact_kind": "universe_contract",
            "lifecycle_state": str(payload.get("lifecycle_state", "retained")).strip() or "retained",
            "action_risk_controls": dict(payload.get("action_risk_controls", {})),
            "environment_assumptions": dict(payload.get("environment_assumptions", {})),
        }
        if "retention_decision" in payload:
            rebound["retention_decision"] = payload.get("retention_decision", {})
        return rebound
    return payload


def _normalized_environment_assumptions(assumptions: object) -> dict[str, object]:
    if not isinstance(assumptions, dict):
        return {}
    normalized: dict[str, object] = {}
    for key, allowed_values in UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS.items():
        value = str(assumptions.get(key, "")).strip().lower()
        if value in allowed_values:
            normalized[key] = value
    for key in UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS:
        if key in assumptions:
            normalized[key] = bool(assumptions.get(key, False))
    return normalized


def _merged_universe_generation_focus(*payloads: object) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        focus = str(payload.get("generation_focus", "")).strip()
        if focus:
            return focus
    return "balanced"


def _merged_universe_proposals(*proposal_groups: object) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for group in proposal_groups:
        if not isinstance(group, list):
            continue
        for proposal in group:
            if not isinstance(proposal, dict):
                continue
            key = repr(
                (
                    str(proposal.get("area", "")).strip(),
                    int(proposal.get("priority", 0) or 0),
                    str(proposal.get("reason", "")).strip(),
                    str(proposal.get("suggestion", "")).strip(),
                )
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(deepcopy(proposal))
    return merged


def _more_restrictive_mode(field: str, left: str, right: str) -> str:
    allowed_values = UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS[field]
    if field == "network_access_mode":
        order = ("blocked", "allowlist_only", "open")
    elif field == "git_write_mode":
        order = ("blocked", "operator_gated", "task_scoped")
    else:
        order = ("task_only", "generated_only", "shared_repo_gated")
    rank = {value: index for index, value in enumerate(order) if value in allowed_values}
    normalized_left = left if left in rank else order[0]
    normalized_right = right if right in rank else order[0]
    return normalized_left if rank[normalized_left] <= rank[normalized_right] else normalized_right


def _dominant_observed_environment_mode(observed_modes: dict[str, object], field: str) -> str:
    field_counts = observed_modes.get(field, {})
    if not isinstance(field_counts, dict) or not field_counts:
        return ""
    ranked = sorted(
        (
            (int(value), str(mode).strip().lower())
            for mode, value in field_counts.items()
            if str(mode).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0:
        return ""
    return ranked[0][1]


def _alignment_key_for_field(field: str) -> str:
    if field == "network_access_mode":
        return "network_access_aligned"
    if field == "git_write_mode":
        return "git_write_aligned"
    return "workspace_scope_aligned"


def _dominant_count_label(counts: dict[str, object]) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    ranked = sorted(
        (
            (float(value), str(label).strip())
            for label, value in counts.items()
            if str(label).strip()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0.0:
        return ""
    return ranked[0][1]


def _proposals(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    environment_evidence: dict[str, object] | None = None,
    cycle_feedback: dict[str, object] | None = None,
    focus: str | None = None,
) -> list[dict[str, object]]:
    evidence = environment_evidence if isinstance(environment_evidence, dict) else {}
    feedback = cycle_feedback if isinstance(cycle_feedback, dict) else {}
    violation_counts = evidence.get("violation_counts", {})
    if not isinstance(violation_counts, dict):
        violation_counts = {}
    alignment_failure_counts = evidence.get("alignment_failure_counts", {})
    if not isinstance(alignment_failure_counts, dict):
        alignment_failure_counts = {}
    observed_modes = evidence.get("observed_environment_modes", {})
    if not isinstance(observed_modes, dict):
        observed_modes = {}
    selected_variant_counts = feedback.get("selected_variant_counts", {})
    if not isinstance(selected_variant_counts, dict):
        selected_variant_counts = {}
    selected_variant_weights = feedback.get("selected_variant_weights", {})
    if not isinstance(selected_variant_weights, dict):
        selected_variant_weights = {}
    proposals: list[dict[str, object]] = []
    if failure_counts.get("command_failure", 0) > 0 or failure_counts.get("state_regression", 0) > 0 or focus == "governance":
        proposals.append(
            {
                "area": "governance",
                "priority": 5,
                "reason": "runtime failures suggest the stable command-governance contract is too weak",
                "suggestion": "Tighten destructive-command exclusions and strengthen preserved-artifact governance above task-local world state.",
            }
        )
    if metrics.low_confidence_episodes > 0 or focus == "verification":
        proposals.append(
            {
                "area": "verification",
                "priority": 4,
                "reason": "low-confidence episodes indicate the stable universe contract should favor explicit verification more strongly",
                "suggestion": "Increase preference for verification-aligned commands and keep bounded success criteria explicit in the stable contract.",
            }
        )
    if sum(int(value) for value in violation_counts.values()) + sum(int(value) for value in alignment_failure_counts.values()) > 0:
        proposals.append(
            {
                "area": "environment_envelope",
                "priority": 5,
                "reason": "repeated runtime envelope conflicts indicate the retained universe contract is misaligned with the observed operating environment",
                "suggestion": (
                    "Calibrate environment assumptions toward the observed runtime envelope"
                    f" (network={_dominant_observed_environment_mode(observed_modes, 'network_access_mode') or 'unknown'},"
                    f" git={_dominant_observed_environment_mode(observed_modes, 'git_write_mode') or 'unknown'},"
                    f" workspace={_dominant_observed_environment_mode(observed_modes, 'workspace_write_scope') or 'unknown'})"
                    " while preserving path-scoped, rollback-ready mutation requirements."
                ),
            }
        )
    elif int(feedback.get("retained_cycle_count", 0)) > 0:
        proposals.append(
            {
                "area": "environment_envelope",
                "priority": 4,
                "reason": "retained universe cycles already indicate successful environment-envelope calibrations that should remain part of the prior",
                "suggestion": (
                    "Preserve successful retained universe calibration priors"
                    f" with dominant variant={_dominant_count_label(selected_variant_weights or selected_variant_counts) or 'unknown'}"
                    " unless new runtime evidence contradicts them."
                ),
            }
        )
    if focus == "operator_scope":
        proposals.append(
            {
                "area": "operator_scope",
                "priority": 4,
                "reason": "selected variant targets stable runtime boundaries rather than task-local heuristics",
                "suggestion": "Refine stable command prefixes and forbidden patterns so autonomous execution remains bounded before policy-level mutation.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "governance",
            "priority": 3,
            "reason": "the universe layer should retain a stable machine-governance contract even when recent failures are sparse",
            "suggestion": "Keep verification, reversibility, and bounded-action invariants explicit above the world model.",
        },
    )
