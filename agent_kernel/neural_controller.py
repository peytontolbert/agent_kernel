from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable


FULL_KERNEL_CONTROL_TOKENS = (
    "<AK_BOOTSTRAP>",
    "<AK_MEMORY_READ>",
    "<AK_WORLD_STATE>",
    "<AK_GOVERNANCE>",
    "<AK_CONTEXT_COMPILE>",
    "<AK_PLAN>",
    "<AK_DECIDE>",
    "<AK_ACTION_SPACE_CODE>",
    "<AK_ACTION_SPACE_ARTIFACT>",
    "<AK_ACTION_SPACE_RETRIEVAL>",
    "<AK_ACTION_SPACE_RESPOND>",
    "<AK_ACTION_SPACE_DELEGATE>",
    "<AK_ACTION_SPACE_IMPROVE>",
    "<AK_RETRIEVE>",
    "<AK_NO_RETRIEVAL>",
    "<AK_RET_CODE>",
    "<AK_RET_MEMORY>",
    "<AK_RET_EPISODE>",
    "<AK_RET_GRAPH>",
    "<AK_RET_WORLD>",
    "<AK_RET_ARTIFACT>",
    "<AK_RET_EXACT>",
    "<AK_RET_SEMANTIC>",
    "<AK_EXECUTE>",
    "<AK_VERIFY>",
    "<AK_WORLD_UPDATE>",
    "<AK_MEMORY_WRITE>",
    "<AK_LEARN_COMPILE>",
    "<AK_IMPROVE_SELECT>",
    "<AK_IMPROVE_GENERATE>",
    "<AK_IMPROVE_EVALUATE>",
    "<AK_RETAIN>",
    "<AK_REJECT>",
    "<AK_CONF_HIGH>",
    "<AK_CONF_MEDIUM>",
    "<AK_CONF_LOW>",
    "<AK_OOD>",
    "<AK_ARTIFACT_REPAIR>",
    "<AK_SOURCE_INSPECT>",
    "<AK_VALIDATE_PRESENT>",
    "<AK_VALIDATE_ABSENT>",
    "<AK_READ_SOURCE>",
    "<AK_PATCH_BUILD>",
    "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
    "<AK_EXEC_KIND_VERIFY_PRESENT>",
    "<AK_EXEC_KIND_VERIFY_ABSENT>",
    "<AK_EXEC_KIND_INSPECT_SOURCE>",
    "<AK_EXEC_KIND_LOCALIZED_EDIT>",
    "<AK_EXEC_KIND_RUN_CHECK>",
    "<AK_COPY_COMMAND_TARGET>",
    "<AK_COPY_ARTIFACT_TARGET>",
    "<AK_COPY_ARTIFACT_PATH>",
    "<AK_COPY_ARTIFACT_CONTENT>",
    *tuple(f"<AK_COPY_MATERIALIZE_CANDIDATE_{index}>" for index in range(1, 25)),
    *tuple(f"<AK_COPY_SOURCE_INSPECT_CANDIDATE_{index}>" for index in range(1, 25)),
    *tuple(f"<AK_COPY_VALIDATE_PRESENT_CANDIDATE_{index}>" for index in range(1, 25)),
    *tuple(f"<AK_COPY_VALIDATE_ABSENT_CANDIDATE_{index}>" for index in range(1, 25)),
    *tuple(f"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_{index}>" for index in range(1, 25)),
    "<AK_RESPOND>",
    "<AK_SAFE_STOP>",
    "<AK_CLOSEOUT>",
)


@dataclass(slots=True)
class NeuralControllerManifest:
    manifest_path: Path
    artifact_kind: str
    model_family: str
    model_dir: str
    tokenizer_dir: str
    dataset_manifest_path: str
    parameter_count: int = 0
    retrieval_head_dim: int = 0
    agent_policy_heads: bool = False
    training_objective: str = ""
    completed_steps: int = 0
    replaces_surfaces: tuple[str, ...] = field(default_factory=tuple)
    special_tokens: tuple[str, ...] = field(default_factory=tuple)
    runtime_targets: dict[str, Any] = field(default_factory=dict)

    @property
    def has_neural_retrieval(self) -> bool:
        return self.retrieval_head_dim > 0

    @property
    def has_policy_heads(self) -> bool:
        return bool(self.agent_policy_heads)

    @property
    def is_full_kernel_controller_trace(self) -> bool:
        objective = self.training_objective.lower()
        dataset = self.dataset_manifest_path.lower()
        artifact_kind = self.artifact_kind.lower()
        runtime_targets = json.dumps(self.runtime_targets, sort_keys=True).lower()
        return (
            "agentkernel_controller" in objective
            or "controller_trace" in dataset
            or artifact_kind == "agentkernel_controller_seq2seq_bundle"
            or "full_agent_kernel" in runtime_targets
        )


@dataclass(slots=True)
class NeuralControllerAdvisory:
    enabled: bool
    mode: str
    source: str
    manifest_path: str = ""
    ready: bool = False
    surfaces: tuple[str, ...] = field(default_factory=tuple)
    action_space_tokens: tuple[str, ...] = field(default_factory=tuple)
    retrieval_tokens: tuple[str, ...] = field(default_factory=tuple)
    policy_heads: tuple[str, ...] = field(default_factory=tuple)
    guarded_fallback_families: tuple[str, ...] = field(default_factory=tuple)
    guarded_candidate_manifest_path: str = ""
    guarded_selector_policy: str = "family_fallback"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "source": self.source,
            "manifest_path": self.manifest_path,
            "ready": self.ready,
            "surfaces": list(self.surfaces),
            "action_space_tokens": list(self.action_space_tokens),
            "retrieval_tokens": list(self.retrieval_tokens),
            "policy_heads": list(self.policy_heads),
            "guarded_fallback_families": list(self.guarded_fallback_families),
            "guarded_candidate_manifest_path": self.guarded_candidate_manifest_path,
            "guarded_selector_policy": self.guarded_selector_policy,
            "warnings": list(self.warnings),
        }


def load_neural_controller_manifest(path: Path) -> NeuralControllerManifest | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return manifest_from_payload(payload, manifest_path=path)


def manifest_from_payload(payload: dict[str, Any], *, manifest_path: Path) -> NeuralControllerManifest:
    model_config = payload.get("model_config") if isinstance(payload.get("model_config"), dict) else {}
    training_summary = payload.get("training_summary") if isinstance(payload.get("training_summary"), dict) else {}
    replaces_surfaces = tuple(
        str(item).strip()
        for item in payload.get("replaces_surfaces", [])
        if str(item).strip()
    )
    special_tokens = tuple(
        token
        for token in FULL_KERNEL_CONTROL_TOKENS
        if token in json.dumps(payload.get("training_summary", {})) or token in json.dumps(payload.get("model_config", {}))
    )
    dataset_manifest_path = str(payload.get("dataset_manifest_path", "")).strip()
    dataset_tokens = _dataset_special_tokens(dataset_manifest_path)
    if dataset_tokens:
        special_tokens = tuple(dict.fromkeys([*special_tokens, *dataset_tokens]))
    return NeuralControllerManifest(
        manifest_path=manifest_path,
        artifact_kind=str(payload.get("artifact_kind", "")).strip(),
        model_family=str(payload.get("model_family", "")).strip(),
        model_dir=str(payload.get("model_dir", "")).strip(),
        tokenizer_dir=str(payload.get("tokenizer_dir", "")).strip(),
        dataset_manifest_path=dataset_manifest_path,
        parameter_count=_int_value(payload.get("parameter_count"), _int_value(training_summary.get("parameter_count"), 0)),
        retrieval_head_dim=_int_value(model_config.get("retrieval_head_dim"), _int_value(training_summary.get("retrieval_head_dim"), 0)),
        agent_policy_heads=bool(model_config.get("agent_policy_heads", training_summary.get("agent_policy_heads", False))),
        training_objective=str(training_summary.get("dataset_objective", "")).strip(),
        completed_steps=_int_value(training_summary.get("completed_steps"), 0),
        replaces_surfaces=replaces_surfaces,
        special_tokens=special_tokens,
        runtime_targets=dict(payload.get("runtime_targets", {})) if isinstance(payload.get("runtime_targets", {}), dict) else {},
    )


def neural_controller_surfaces(manifest: NeuralControllerManifest) -> tuple[str, ...]:
    surfaces = list(manifest.replaces_surfaces)
    if manifest.has_neural_retrieval:
        surfaces.extend(
            [
                "encoder_neural_retrieval_query_embedding",
                "encoder_neural_retrieval_doc_embedding",
                "retrieval_namespace_policy",
                "retrieval_coverage_estimation",
            ]
        )
    if manifest.has_policy_heads:
        surfaces.extend(
            [
                "controller_confidence_head",
                "controller_ood_head",
                "controller_verification_need_head",
                "controller_action_validity_head",
            ]
        )
    if manifest.is_full_kernel_controller_trace:
        surfaces.extend(
            [
                "full_kernel_action_space_policy",
                "full_kernel_artifact_repair_policy",
                "full_kernel_source_inspection_policy",
                "full_kernel_safe_stop_policy",
            ]
        )
    return tuple(dict.fromkeys(item for item in surfaces if item))


def build_neural_controller_advisory(
    *,
    manifest: NeuralControllerManifest | None,
    mode: str,
    source: str = "manifest",
    guarded_fallback_families: tuple[str, ...] = (),
    guarded_candidate_manifest_path: str = "",
    guarded_selector_policy: str = "family_fallback",
) -> NeuralControllerAdvisory:
    normalized_mode = _normalize_mode(mode)
    if manifest is None:
        return NeuralControllerAdvisory(
            enabled=False,
            mode=normalized_mode,
            source=source,
            ready=False,
            warnings=("manifest_missing_or_unreadable",),
        )
    warnings: list[str] = []
    if not manifest.has_neural_retrieval:
        warnings.append("retrieval_heads_missing")
    if not manifest.has_policy_heads:
        warnings.append("policy_heads_missing")
    if not manifest.is_full_kernel_controller_trace:
        warnings.append("full_kernel_controller_trace_not_detected")
    tokens = manifest.special_tokens or FULL_KERNEL_CONTROL_TOKENS
    return NeuralControllerAdvisory(
        enabled=normalized_mode != "disabled",
        mode=normalized_mode,
        source=source,
        manifest_path=str(manifest.manifest_path),
        ready=not warnings,
        surfaces=neural_controller_surfaces(manifest),
        action_space_tokens=tuple(token for token in tokens if token.startswith("<AK_ACTION_SPACE_")),
        retrieval_tokens=tuple(token for token in tokens if token.startswith("<AK_RET_") or token in {"<AK_RETRIEVE>", "<AK_NO_RETRIEVAL>"}),
        policy_heads=(
            "confidence",
            "ood_query",
            "ood_response",
            "retrieval_coverage",
            "verification_need",
            "action_validity",
        )
        if manifest.has_policy_heads
        else tuple(),
        guarded_fallback_families=tuple(
            dict.fromkeys(str(item).strip() for item in guarded_fallback_families if str(item).strip())
        ),
        guarded_candidate_manifest_path=str(guarded_candidate_manifest_path).strip(),
        guarded_selector_policy=str(guarded_selector_policy or "family_fallback").strip(),
        warnings=tuple(warnings),
    )


EXEC_KIND_FAMILY = {
    "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>": "materialize_artifact",
    "<AK_EXEC_KIND_VERIFY_PRESENT>": "verify_present",
    "<AK_EXEC_KIND_VERIFY_ABSENT>": "verify_absent",
    "<AK_EXEC_KIND_INSPECT_SOURCE>": "inspect_source",
    "<AK_EXEC_KIND_LOCALIZED_EDIT>": "localized_edit",
    "<AK_EXEC_KIND_RUN_CHECK>": "run_check",
}


def neural_controller_exec_kind_family(tokens: object) -> str:
    if isinstance(tokens, str):
        token_values = tokens.split()
    elif isinstance(tokens, (list, tuple, set)):
        token_values = [str(token) for token in tokens]
    else:
        token_values = []
    for token in token_values:
        family = EXEC_KIND_FAMILY.get(str(token).strip())
        if family:
            return family
    return ""


def guarded_neural_controller_source(
    *,
    candidate_line_protocol: dict[str, Any],
    baseline_line_protocol: dict[str, Any],
    fallback_families: tuple[str, ...],
) -> dict[str, Any]:
    candidate_family = neural_controller_exec_kind_family(candidate_line_protocol.get("tokens", []))
    fallback_set = {str(item).strip() for item in fallback_families if str(item).strip()}
    use_baseline = bool(candidate_family and candidate_family in fallback_set)
    selected = baseline_line_protocol if use_baseline else candidate_line_protocol
    return {
        "source": "baseline" if use_baseline else "candidate",
        "candidate_family": candidate_family,
        "fallback_families": sorted(fallback_set),
        "line_protocol": dict(selected),
    }


def neural_controller_shadow_contract_success(shadow: dict[str, Any] | None) -> bool:
    if not isinstance(shadow, dict):
        return False
    return bool(shadow.get("content_exact_agreement", False)) or (
        str(shadow.get("artifact_failure_mode", "")).strip() == "artifact_contract_success"
    )


def neural_controller_verified_contract_mode(verification: dict[str, Any]) -> str:
    if not isinstance(verification, dict):
        return ""
    if bool(verification.get("passed", False)):
        return "artifact_contract_success"
    failure_codes = verification.get("failure_codes", [])
    if isinstance(failure_codes, list):
        for code in failure_codes:
            normalized = str(code).strip()
            if normalized:
                return normalized
    reasons = verification.get("reasons", [])
    if isinstance(reasons, list):
        for reason in reasons:
            text = str(reason).strip()
            if text and text.lower() != "verification passed":
                return text
    return "verification_failure"


def attach_neural_controller_verified_contract_metadata(
    proposal_metadata: dict[str, Any],
    *,
    verification: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(proposal_metadata or {})
    shadow = metadata.get("neural_controller_shadow", {})
    if not isinstance(shadow, dict) or not shadow:
        return metadata
    mode = neural_controller_verified_contract_mode(verification)
    updated_shadow = dict(shadow)
    updated_shadow["runtime_artifact_failure_mode"] = mode
    updated_shadow["runtime_contract_success"] = mode == "artifact_contract_success"
    guarded_selected_source = str(updated_shadow.get("guarded_selected_source", "")).strip()
    guarded_selector_policy = str(updated_shadow.get("guarded_selector_policy", "")).strip()
    if guarded_selected_source and not str(updated_shadow.get("rowwise_selector_source", "")).strip():
        updated_shadow["rowwise_selector_source"] = guarded_selected_source
    if guarded_selector_policy and not str(updated_shadow.get("rowwise_selector_policy", "")).strip():
        updated_shadow["rowwise_selector_policy"] = guarded_selector_policy
    if guarded_selected_source:
        updated_shadow["runtime_selector_selected_source"] = guarded_selected_source
        updated_shadow["runtime_selector_selected_contract_success"] = mode == "artifact_contract_success"
    metadata["neural_controller_shadow"] = updated_shadow
    return metadata


def select_verified_neural_controller_shadow(
    *,
    baseline_label: str,
    baseline_shadow: dict[str, Any],
    candidate_shadows: list[tuple[str, dict[str, Any]]],
    policy: str = "candidate_contract_improves",
) -> dict[str, Any]:
    """Select a candidate shadow only when verified evidence improves the baseline.

    This is intentionally verifier/contract based. It does not require target
    exact-match labels, so it can be used as the runtime shape for rowwise
    neural-controller gating after candidate dry-runs have produced contract
    metadata.
    """
    baseline_source = str(baseline_label).strip() or "baseline"
    baseline_contract = neural_controller_shadow_contract_success(baseline_shadow)
    selected_label = baseline_source
    selected_shadow = dict(baseline_shadow)
    accepted_sources: list[str] = []
    if policy != "candidate_contract_improves":
        raise ValueError(f"unknown neural-controller rowwise selector policy: {policy}")
    if not baseline_contract:
        for label, shadow in candidate_shadows:
            candidate_label = str(label).strip()
            if not candidate_label or not isinstance(shadow, dict):
                continue
            if neural_controller_shadow_contract_success(shadow):
                accepted_sources.append(candidate_label)
                if selected_label == baseline_source:
                    selected_label = candidate_label
                    selected_shadow = dict(shadow)
    selected_shadow["rowwise_selector_source"] = selected_label
    selected_shadow["rowwise_selector_policy"] = policy
    selected_shadow["rowwise_selector_baseline_label"] = baseline_source
    return {
        "source": selected_label,
        "policy": policy,
        "baseline_contract_success": baseline_contract,
        "accepted_candidate_sources": accepted_sources,
        "shadow": selected_shadow,
    }


def build_neural_controller_encoder_text(*, state_payload: dict[str, Any]) -> str:
    task = state_payload.get("task") if isinstance(state_payload.get("task"), dict) else {}
    metadata = task.get("metadata", {}) if isinstance(task.get("metadata", {}), dict) else {}
    history = state_payload.get("history") if isinstance(state_payload.get("history"), list) else []
    retrieval_plan = state_payload.get("retrieval_plan") if isinstance(state_payload.get("retrieval_plan"), dict) else {}
    context_packet = state_payload.get("context_packet")
    world = state_payload.get("world_model_summary") if isinstance(state_payload.get("world_model_summary"), dict) else {}
    plan = state_payload.get("plan") if isinstance(state_payload.get("plan"), list) else []
    chunks = state_payload.get("state_context_chunks") if isinstance(state_payload.get("state_context_chunks"), list) else []
    task_id = _compact(task.get("task_id", ""))
    prompt = _compact(task.get("prompt", ""), limit=1200)
    workspace = _compact(task.get("workspace_subdir", ""))
    active_subgoal = str(state_payload.get("active_subgoal", "direct bounded workspace action"))
    trajectory_step_index = state_payload.get("trajectory_step_index")
    trajectory_step_count = state_payload.get("trajectory_step_count")
    expected_contents = task.get("expected_file_contents", {})
    expected_contents = expected_contents if isinstance(expected_contents, dict) else {}
    expected_files = task.get("expected_files", [])
    expected_files = expected_files if isinstance(expected_files, list) else []
    artifact_path = ""
    artifact_content = ""
    if expected_contents:
        artifact_path, artifact_content = artifact_slot_target_from_task(
            active_subgoal=active_subgoal,
            expected_file_contents=expected_contents,
        )
    validation_present_targets = _unique_strings(
        world.get("missing_expected_artifacts", []),
        world.get("expected_artifacts", []),
        expected_files,
    )
    validation_absent_targets = _unique_strings(
        world.get("present_forbidden_artifacts", []),
        world.get("forbidden_artifacts", []),
    )
    next_step_candidates = _next_step_target_candidates(
        world=world,
        expected_files=expected_files,
        expected_contents=expected_contents,
        active_artifact_path=artifact_path,
    )
    suggested_commands = task.get("suggested_commands", [])
    source_inspection_candidates = _source_inspection_candidate_commands(
        suggested_commands=suggested_commands,
        history=history,
        plan=plan,
    )
    localized_edit_candidates = _localized_edit_candidate_commands(
        suggested_commands=suggested_commands,
        history=history,
        expected_contents=expected_contents,
        expected_files=expected_files,
        success_command=task.get("success_command", ""),
    )
    lines = [
        f"<AK_USER> task_id={task_id}",
        "<AK_LOOP> <AK_BOOTSTRAP> <AK_MEMORY_READ> <AK_WORLD_STATE> <AK_GOVERNANCE> <AK_CONTEXT_COMPILE> <AK_PLAN> <AK_DECIDE>",
        "<AK_STATE> outcome=unknown failure_mode=unknown",
        "<AK_NO_RETRIEVAL>" if not retrieval_plan else "<AK_RETRIEVE>",
        "subgoal=" + _compact(active_subgoal, limit=300),
        "<AK_CONTEXT>",
        f"Task prompt: {prompt}",
        f"Workspace: {workspace}",
        "Benchmark family: " + _compact(metadata.get("benchmark_family", "bounded")),
        "<AK_STATE>",
        f"Task: {task_id}",
        f"Prompt: {prompt}",
        f"Workspace: {workspace}",
        f"Active subgoal: {_compact(active_subgoal, limit=300)}",
    ]
    try:
        step_index_int = int(trajectory_step_index)
        step_count_int = int(trajectory_step_count)
    except (TypeError, ValueError):
        step_index_int = -1
        step_count_int = -1
    if step_index_int >= 0 and step_count_int > 0:
        lines.append(f"Trajectory position: step {step_index_int + 1} of {step_count_int}")
    if artifact_path:
        lines.append("Current artifact target path: " + _compact_command(artifact_path, limit=500))
        lines.append("Current artifact target content: " + _compact_command(artifact_content, limit=1200))
    if validation_present_targets:
        lines.append("Validation target present paths: " + _compact(", ".join(validation_present_targets[:8]), limit=700))
        lines.append(
            "Validation present commands: "
            + _compact(" | ".join(f"test -f {path}" for path in validation_present_targets[:8]), limit=900)
        )
    if validation_absent_targets:
        lines.append("Validation target absent paths: " + _compact(", ".join(validation_absent_targets[:8]), limit=700))
        lines.append(
            "Validation absent commands: "
            + _compact(" | ".join(f"test ! -f {path}" for path in validation_absent_targets[:8]), limit=900)
        )
    if next_step_candidates:
        lines.append("Next-step target candidates: " + _compact(" | ".join(next_step_candidates[:8]), limit=900))
    if source_inspection_candidates:
        lines.append(
            "Source inspection candidate commands: "
            + _compact(" | ".join(source_inspection_candidates[:8]), limit=900)
        )
        for index, command in enumerate(source_inspection_candidates[:24], start=1):
            lines.append(f"Source inspection candidate {index}: " + _compact_command(command, limit=900))
    validation_present_commands = validation_command_candidates_from_encoder("\n".join(lines), polarity="present")
    validation_absent_commands = validation_command_candidates_from_encoder("\n".join(lines), polarity="absent")
    for index, command in enumerate(validation_present_commands[:24], start=1):
        lines.append(f"Validation present candidate {index}: " + _compact_command(command, limit=360))
    for index, command in enumerate(validation_absent_commands[:24], start=1):
        lines.append(f"Validation absent candidate {index}: " + _compact_command(command, limit=360))
    if localized_edit_candidates:
        lines.append(
            "Localized edit candidate commands: "
            + _compact(" | ".join(localized_edit_candidates[:24]), limit=2200)
        )
        for index, command in enumerate(localized_edit_candidates[:24], start=1):
            edit_old, edit_new, edit_path = _infer_sed_edit_slots(command)
            lines.append(f"Localized edit candidate {index}: " + _compact_command(command, limit=900))
            fields = []
            if edit_path:
                fields.append(f"path={_compact_command(edit_path, limit=220)}")
            if edit_old:
                fields.append(f"old={_compact_command(edit_old, limit=260)}")
            if edit_new:
                fields.append(f"new={_compact_command(edit_new, limit=360)}")
            if fields:
                lines.append(f"Localized edit candidate {index} fields: " + " ; ".join(fields))
    first_command = ""
    if isinstance(suggested_commands, list):
        command_text = " | ".join(_compact_command(command, limit=260) for command in suggested_commands[:3])
        if command_text:
            first_command = _compact_command(suggested_commands[0], limit=1800)
            if first_command:
                lines.append("Command copy target: " + first_command)
            lines.append("Suggested commands: " + _compact(command_text, limit=900))
    success_command = _compact_command(task.get("success_command", ""), limit=360)
    if success_command:
        lines.append("Success command: " + success_command)
    if expected_files:
        lines.append("Expected files: " + _compact(", ".join(str(path) for path in expected_files[:8]), limit=500))
    artifact_target = ""
    if expected_contents:
        previews = [
            f"{path}={_compact(content, limit=100)}"
            for path, content in list(expected_contents.items())[:5]
        ]
        lines.append("Expected file contents: " + _compact(" | ".join(previews), limit=700))
        artifact_target = artifact_command_target_from_task(
            active_subgoal=active_subgoal,
            expected_file_contents=expected_contents,
        )
        if artifact_target:
            lines.append("Artifact command target: " + artifact_target)
            if artifact_path:
                lines.append("Artifact target path: " + _compact_command(artifact_path, limit=500))
                lines.append("Artifact target content: " + _compact_command(artifact_content, limit=1200))
    materialization_candidates = materialization_candidates_from_encoder("\n".join(lines))
    active_materialization_target = _active_materialization_target_command(
        command_copy_target=first_command if isinstance(suggested_commands, list) and suggested_commands else "",
        artifact_target=artifact_target if expected_contents else "",
        active_subgoal=active_subgoal,
    )
    if active_materialization_target:
        lines.append("Active materialization target: " + _compact_command(active_materialization_target, limit=1800))
        materialization_candidates = list(dict.fromkeys([active_materialization_target, *materialization_candidates]))
    if materialization_candidates:
        lines.append(
            "Materialization candidate commands: "
            + _compact(" | ".join(materialization_candidates[:8]), limit=1200)
        )
        for index, command in enumerate(materialization_candidates[:24], start=1):
            lines.append(f"Materialization candidate {index}: " + _compact_command(command, limit=1800))
    if not retrieval_plan and suggested_commands:
        lines.append("Contract: use the direct task command; do not inspect source_lines or retrieve code context.")
    if plan:
        lines.append("Plan: " + _compact(" | ".join(str(item) for item in plan[:6]), limit=700))
    if retrieval_plan:
        lines.append("<AK_CONTEXT>")
        lines.append("Retrieval plan: " + _compact(json.dumps(retrieval_plan, sort_keys=True), limit=900))
    if context_packet is not None:
        lines.append("Context packet: available")
    if world:
        lines.append("World: " + _compact(json.dumps(world, sort_keys=True), limit=900))
    if chunks:
        chunk_text = " | ".join(_compact(chunk, limit=260) for chunk in chunks[:4])
        lines.append("State context chunks: " + _compact(chunk_text, limit=1100))
    if history:
        lines.append("<AK_HISTORY>")
        for step in history[-4:]:
            if not isinstance(step, dict):
                continue
            lines.append(
                "Step: "
                f"action={_compact(step.get('action', ''), limit=80)} "
                f"source={_compact(step.get('decision_source', ''), limit=100)} "
                f"content={_compact(step.get('content', ''), limit=220)}"
            )
    return "\n".join(line for line in lines if line.strip())


def _unique_strings(*groups: object) -> list[str]:
    values: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            text = str(item or "").strip()
            if text and text not in values:
                values.append(text)
    return values


def _next_step_target_candidates(
    *,
    world: dict[str, Any],
    expected_files: list[Any],
    expected_contents: dict[Any, Any],
    active_artifact_path: str = "",
) -> list[str]:
    candidates: list[str] = []

    def add(kind: str, path: object) -> None:
        text = str(path or "").strip()
        if not text:
            return
        item = f"{kind}:{text}"
        if item not in candidates:
            candidates.append(item)

    if active_artifact_path:
        add("materialize", active_artifact_path)
    for path in _unique_strings(world.get("present_forbidden_artifacts", [])):
        add("verify_absent", path)
    for path in _unique_strings(world.get("missing_expected_artifacts", [])):
        add("materialize", path)
    for path in _unique_strings(world.get("unsatisfied_expected_contents", [])):
        add("materialize", path)
    for path in _unique_strings(world.get("expected_artifacts", []), expected_files):
        add("verify_present", path)
    for path in expected_contents:
        add("materialize", path)
    return candidates


def _source_inspection_candidate_commands(*, suggested_commands: object, history: list[Any], plan: list[Any] | None = None) -> list[str]:
    candidates: list[str] = []

    def add(command: object) -> None:
        text = _compact_command(command, limit=360)
        if not text:
            return
        if not _is_source_inspection_command(text):
            return
        if text not in candidates:
            candidates.append(text)

    if isinstance(suggested_commands, list):
        for command in suggested_commands:
            add(command)
    plan_source_paths = _source_inspection_paths_from_plan(plan or [])
    if len(plan_source_paths) > 1:
        add("cat " + " ".join(plan_source_paths[:6]))
    for path in plan_source_paths:
        add(f"cat {path}")
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        add(item.get("content", ""))
    return candidates


def _source_inspection_paths_from_plan(plan: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in plan:
        command = _source_inspection_command_from_active_subgoal(str(item or ""))
        if not command:
            continue
        path = _inspect_path_from_command(command)
        if path and path not in paths:
            paths.append(path)
    return paths


def _localized_edit_candidate_commands(
    *,
    suggested_commands: object,
    history: list[Any],
    expected_contents: dict[str, Any] | None = None,
    expected_files: list[Any] | None = None,
    success_command: object = "",
) -> list[str]:
    candidates: list[str] = []
    completed: set[str] = set()
    completed_paths: list[str] = []
    completed_edits: list[tuple[str, str]] = []

    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        command = _normalize_localized_edit_candidate_text(_compact_command(item.get("content", ""), limit=900))
        if _is_complete_localized_edit_candidate(command):
            completed.add(command)
            _old, edit_new, path = _infer_sed_edit_slots(command)
            if path:
                completed_paths.append(path)
            if path and edit_new:
                completed_edits.append((path, edit_new))

    def add(command: object) -> None:
        text = _normalize_localized_edit_candidate_text(_compact_command(command, limit=900))
        if not text or not _is_localized_edit_command(text):
            return
        if text in completed:
            return
        if text not in candidates:
            candidates.append(text)

    if isinstance(suggested_commands, list):
        for command in suggested_commands:
            add(command)
    for command in _localized_edit_candidates_from_expected_contents(expected_contents or {}):
        add(command)
    return _sort_localized_edit_candidates(
        candidates,
        completed_paths=completed_paths,
        completed_edits=completed_edits,
        expected_files=expected_files or [],
        success_command=success_command,
    )


def _sort_localized_edit_candidates(
    candidates: list[str],
    *,
    completed_paths: list[str],
    completed_edits: list[tuple[str, str]],
    expected_files: list[Any],
    success_command: object = "",
) -> list[str]:
    if not candidates:
        return candidates
    expected_path_ranked = _sort_localized_edit_candidates_by_expected_path_frontier(
        candidates,
        completed_paths=completed_paths,
        expected_files=expected_files,
    )
    if expected_path_ranked:
        return expected_path_ranked
    residual_append_ranked = _sort_localized_edit_candidates_by_residual_append_frontier(
        candidates,
        completed_paths=completed_paths,
        completed_edits=completed_edits,
        expected_files=expected_files,
    )
    if residual_append_ranked:
        return residual_append_ranked
    success_ranked = _sort_localized_edit_candidates_by_success_frontier(
        candidates,
        completed_edits=completed_edits,
        success_command=success_command,
    )
    if success_ranked:
        return success_ranked
    return candidates


def _sort_localized_edit_candidates_by_expected_path_frontier(
    candidates: list[str],
    *,
    completed_paths: list[str],
    expected_files: list[Any],
) -> list[str]:
    if not completed_paths or not expected_files:
        return []
    candidate_paths = {
        path
        for command in candidates
        for _old, _new, path in [_infer_sed_edit_slots(command)]
        if path
    }
    if not candidate_paths:
        return []
    path_order = {
        _clean_shell_path(path): index
        for index, path in enumerate(expected_files)
        if _clean_shell_path(path)
    }
    if not path_order:
        return []
    last_completed_path = ""
    for path in reversed(completed_paths):
        if path in path_order:
            last_completed_path = path
            break
    if not last_completed_path:
        return []
    last_index = path_order[last_completed_path]
    next_indices = [
        index
        for path, index in path_order.items()
        if index > last_index and path in candidate_paths
    ]
    if not next_indices:
        return []
    next_index = min(next_indices)

    def rank(command: str) -> tuple[int, int, int, int]:
        _old, edit_new, path = _infer_sed_edit_slots(command)
        index = path_order.get(path, 10_000)
        is_full_replacement = _localized_edit_replaces_multiple_lines(command, edit_new)
        original_index = candidates.index(command)
        if index == next_index:
            return (0, 0 if is_full_replacement else 1, index, original_index)
        if index > last_index:
            return (1, 0 if is_full_replacement else 1, index, original_index)
        return (2, 0 if is_full_replacement else 1, index, original_index)

    return sorted(candidates, key=rank)


def _localized_edit_replaces_multiple_lines(command: str, edit_new: str) -> bool:
    normalized = str(command or "")
    if re.search(r"sed\s+-i\s+(['\"])\d+,\d+c", normalized):
        return True
    return "\\n" in str(edit_new or "")


def _sort_localized_edit_candidates_by_residual_append_frontier(
    candidates: list[str],
    *,
    completed_paths: list[str],
    completed_edits: list[tuple[str, str]],
    expected_files: list[Any],
) -> list[str]:
    if not completed_paths or not expected_files:
        return []
    expected_path_order = [
        _clean_shell_path(path)
        for path in expected_files
        if _clean_shell_path(path)
    ]
    expected_path_order = list(dict.fromkeys(expected_path_order))
    if not expected_path_order:
        return []
    completed_path_set = set(completed_paths)
    path_order = {path: index for index, path in enumerate(expected_path_order)}
    completed_indices = [
        path_order[path]
        for path in completed_path_set
        if path in path_order
    ]
    if not completed_indices or max(completed_indices) < len(expected_path_order) - 1:
        return []
    append_candidates = {
        command
        for command in candidates
        if re.search(r"sed\s+-i\s+(['\"])\$a", command)
    }
    if not append_candidates:
        return []
    completed_keys = {
        (path, _canonical_localized_success_text(value))
        for path, value in completed_edits
        if path and _canonical_localized_success_text(value)
    }
    unseen_append_candidates = {
        command
        for command in append_candidates
        for _old, edit_new, path in [_infer_sed_edit_slots(command)]
        if (path, _canonical_localized_success_text(edit_new)) not in completed_keys
    }
    if unseen_append_candidates:
        append_candidates = unseen_append_candidates

    def rank(command: str) -> tuple[int, int, int]:
        _old, _new, path = _infer_sed_edit_slots(command)
        return (
            0 if command in append_candidates else 1,
            path_order.get(path, 10_000),
            candidates.index(command),
        )

    return sorted(candidates, key=rank)


def _sort_localized_edit_candidates_by_legacy_path_order(
    candidates: list[str],
    *,
    completed_paths: list[str],
    expected_files: list[Any],
) -> list[str]:
    if not completed_paths or not expected_files:
        return candidates
    path_order = {
        _clean_shell_path(path): index
        for index, path in enumerate(expected_files)
        if _clean_shell_path(path)
    }
    if not path_order:
        return candidates
    last_completed_path = ""
    for path in reversed(completed_paths):
        if path in path_order:
            last_completed_path = path
            break
    if not last_completed_path:
        return candidates
    last_index = path_order[last_completed_path]

    def rank(command: str) -> tuple[int, int, int]:
        _old, _new, path = _infer_sed_edit_slots(command)
        index = path_order.get(path, 10_000)
        if path == last_completed_path:
            return (0, index, candidates.index(command))
        if index > last_index:
            return (1, index, candidates.index(command))
        return (2, index, candidates.index(command))

    return sorted(candidates, key=rank)


def _sort_localized_edit_candidates_by_success_frontier(
    candidates: list[str],
    *,
    completed_edits: list[tuple[str, str]],
    success_command: object,
) -> list[str]:
    checks = _localized_success_checks(success_command)
    if not checks:
        return []
    completed_keys = {
        (path, _canonical_localized_success_text(value))
        for path, value in completed_edits
        if path and _canonical_localized_success_text(value)
    }
    cursor = -1
    for index, (path, expected_text) in enumerate(checks):
        if expected_text and (path, expected_text) in completed_keys:
            cursor = index
    frontier_index = -1
    for index, (path, expected_text) in enumerate(checks):
        if index <= cursor:
            continue
        if expected_text and (path, expected_text) not in completed_keys:
            frontier_index = index
            break
    if frontier_index < 0:
        return []
    path_order: dict[str, int] = {}
    text_order: dict[tuple[str, str], int] = {}
    for index, (path, expected_text) in enumerate(checks):
        path_order.setdefault(path, index)
        if expected_text:
            text_order.setdefault((path, expected_text), index)
    frontier_path, frontier_text = checks[frontier_index]

    def rank(command: str) -> tuple[int, int, int, int]:
        _old, edit_new, path = _infer_sed_edit_slots(command)
        normalized_new = _canonical_localized_success_text(edit_new)
        exact_index = text_order.get((path, normalized_new), 10_000)
        original_index = candidates.index(command)
        if path == frontier_path and normalized_new == frontier_text:
            return (0, frontier_index, exact_index, original_index)
        if path == frontier_path:
            return (1, frontier_index, exact_index, original_index)
        return (2, path_order.get(path, 10_000), exact_index, original_index)

    return sorted(candidates, key=rank)


def _localized_success_checks(success_command: object) -> list[tuple[str, str]]:
    command = str(success_command or "")
    checks: list[tuple[str, str]] = []
    grep_pattern = re.compile(
        r"grep\s+(?:-[A-Za-z]+\s+)*(['\"])(.*?)\1\s+([^;&|]+)"
    )
    for match in grep_pattern.finditer(command):
        path = _clean_shell_path(match.group(3))
        expected_text = _canonical_localized_success_text(match.group(2))
        if path and expected_text:
            checks.append((path, expected_text))
    return checks


def _canonical_localized_success_text(value: object) -> str:
    text = str(value or "")
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = text.replace(r"\ ", " ")
    text = text.replace(r"\-", "-").replace(r"\:", ":").replace(r"\.", ".")
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = text.strip()
    if text.startswith("^"):
        text = text[1:]
    if text.endswith("$"):
        text = text[:-1]
    return " ".join(text.strip().split())


def _localized_edit_candidates_from_expected_contents(expected_contents: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for raw_path, raw_content in expected_contents.items():
        path = _clean_shell_path(raw_path)
        content = str(raw_content or "").replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
        if not path or not content:
            continue
        lines = content.split("\n")
        if len(lines) > 1:
            escaped_content = content.replace("\n", r"\\n")
            newline_escape = r"\\n"
            candidates.append(f"sed -i '1,2c{newline_escape}{escaped_content}' {path}")
        first_line = lines[0].strip()
        if first_line:
            old_pattern = _expected_content_old_pattern(first_line)
            if old_pattern:
                candidates.append(f"sed -i '1s#{old_pattern}#{first_line}#' {path}")
        for line in lines[1:]:
            normalized = line.strip()
            if normalized.startswith("- "):
                candidates.append(f"sed -i '$a\\\\n{normalized}' {path}")
    return candidates


def _expected_content_old_pattern(new_line: str) -> str:
    text = str(new_line or "").strip()
    if not text:
        return ""
    if text.startswith("title: ") and "handoff" in text:
        return r"^title:\ draft\ handoff$"
    if text.startswith("title: ") and "validation" in text:
        return r"^title:\ draft\ validation$"
    first_word = text.split()[0].rstrip(":")
    if first_word:
        return rf"^{_sed_search_escape(first_word)}\ pending$"
    return ""


def _sed_search_escape(value: str) -> str:
    return re.sub(r"([\\.^$*+?{}\[\]|()#\-\s])", lambda match: "\\" + match.group(1), str(value or ""))


def _is_source_inspection_command(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().split())
    lowered = normalized.lower()
    return lowered.startswith(("cat ", "head ", "tail ", "grep ")) or lowered.startswith("sed -n ")


def _is_localized_edit_command(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().split())
    return normalized.startswith("sed -i ")


def _is_complete_localized_edit_candidate(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().split())
    if not _is_localized_edit_command(normalized):
        return False
    if normalized.count("'") % 2 or normalized.count('"') % 2:
        return False
    _old, _new, path = _infer_sed_edit_slots(normalized)
    return bool(path and not path.startswith("sed") and not path.startswith("-"))


def _normalize_localized_edit_candidate_text(command: str) -> str:
    text = str(command or "").strip()
    # Encoder candidate lines may already contain escaped shell commands from
    # dataset rows. Collapse one duplicated escape layer so candidates compare
    # against decoder line-protocol content.
    return text.replace("\\\\\\\\n", "\\\\n").replace("\\\\\\\\t", "\\\\t")


def parse_neural_controller_line_protocol(text: str) -> dict[str, Any]:
    action = ""
    content = ""
    failure_mode = ""
    target_path = ""
    target_content = ""
    edit_old = ""
    edit_new = ""
    verify_polarity = ""
    tokens: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<AK_"):
            tokens.extend(part for part in line.split() if part.startswith("<AK_"))
            continue
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("content:"):
            content = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("artifact-failure-mode:"):
            failure_mode = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("target-path:"):
            target_path = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("target-content:"):
            target_content = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("edit-old:"):
            edit_old = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("edit-new:"):
            edit_new = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("verify-polarity:"):
            verify_polarity = line.split(":", 1)[1].strip()
    inferred_slots = _infer_line_protocol_slots(action=action, content=content)
    if not target_path:
        target_path = inferred_slots.get("target_path", "")
    if not target_content:
        target_content = inferred_slots.get("target_content", "")
    if not edit_old:
        edit_old = inferred_slots.get("edit_old", "")
    if not edit_new:
        edit_new = inferred_slots.get("edit_new", "")
    if not verify_polarity:
        verify_polarity = inferred_slots.get("verify_polarity", "")
    return {
        "tokens": tokens,
        "action": action,
        "content": content,
        "artifact_failure_mode": failure_mode,
        "target_path": target_path,
        "target_content": target_content,
        "edit_old": edit_old,
        "edit_new": edit_new,
        "verify_polarity": verify_polarity,
    }


def _infer_line_protocol_slots(*, action: str, content: str) -> dict[str, str]:
    if str(action or "").strip() != "code_execute":
        return {}
    text = str(content or "").strip()
    if not text or text.startswith("<AK_"):
        return {}
    slots: dict[str, str] = {}
    absent_path = _first_shell_path_after_patterns(
        text,
        (
            r"\btest\s+!\s+-[fe]\s+([^;&|]+)",
            r"\[\s+!\s+-[fe]\s+([^\]]+)\]",
        ),
    )
    if absent_path:
        return {"target_path": absent_path, "verify_polarity": "absent"}
    present_path = _first_shell_path_after_patterns(
        text,
        (
            r"\btest\s+-[fe]\s+([^;&|]+)",
            r"\[\s+-[fe]\s+([^\]]+)\]",
        ),
    )
    if present_path:
        return {"target_path": present_path, "verify_polarity": "present"}
    if text.startswith("sed -i "):
        edit_old, edit_new, path = _infer_sed_edit_slots(text)
        if path:
            slots["target_path"] = path
        if edit_old:
            slots["edit_old"] = edit_old
        if edit_new:
            slots["edit_new"] = edit_new
        return slots
    redirect_path = _redirect_path_from_command(text)
    if redirect_path:
        slots["target_path"] = redirect_path
        materialized = _materialized_content_from_command(text)
        if materialized:
            slots["target_content"] = materialized
        return slots
    inspect_path = _inspect_path_from_command(text)
    if inspect_path:
        return {"target_path": inspect_path}
    return {}


def _first_shell_path_after_patterns(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_shell_path(match.group(1))
    return ""


def _redirect_path_from_command(text: str) -> str:
    match = re.search(r">\s*([^;&|]+)", text)
    return _clean_shell_path(match.group(1)) if match else ""


def _redirect_paths_from_command(text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r">\s*([^;&|]+)", str(text or "")):
        path = _clean_shell_path(match.group(1))
        if path and path not in paths:
            paths.append(path)
    return paths


def _active_artifact_path_from_text(text: str) -> str:
    value = str(text or "")
    match = re.search(r"materialize expected artifact\s+([^\s|]+)", value, flags=re.IGNORECASE)
    if match:
        return _clean_shell_path(match.group(1))
    match = re.search(r"write\s+`([^`]+)`", value, flags=re.IGNORECASE)
    if match:
        return _clean_shell_path(match.group(1))
    return ""


def _active_materialization_target_command(
    *,
    command_copy_target: str,
    artifact_target: str,
    active_subgoal: str,
) -> str:
    active_path = _active_artifact_path_from_text(active_subgoal)
    if not active_path:
        return ""
    for command in (artifact_target, command_copy_target):
        if active_path in _redirect_paths_from_command(command):
            return _compact_command(command, limit=1800)
    return ""


def _materialized_content_from_command(text: str) -> str:
    printf_match = re.search(r"printf\s+(?:%s\s+)?(['\"])(.*?)\1\s*>", text)
    if printf_match:
        return printf_match.group(2)
    echo_match = re.search(r"echo\s+(['\"])(.*?)\1\s*>", text)
    if echo_match:
        return echo_match.group(2)
    return ""


def _inspect_path_from_command(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("sed -n "):
        return _clean_shell_path(stripped.split()[-1])
    parts = stripped.split()
    if not parts or parts[0] not in {"cat", "head", "tail", "grep"}:
        return ""
    return _clean_shell_path(parts[-1]) if len(parts) >= 2 else ""


def _infer_sed_edit_slots(text: str) -> tuple[str, str, str]:
    path = _clean_shell_path(text.split()[-1]) if text.split() else ""
    match = re.search(r"sed\s+-i\s+(['\"])(?:\d+(?:,\d+)?)?s(.)(.*?)\2(.*?)\2", text)
    if match:
        return match.group(3), match.group(4), path
    change_match = re.search(r"sed\s+-i\s+(['\"])(?:\d+(?:,\d+)?)c\\\\n(.*?)\1", text)
    if change_match:
        return "", change_match.group(2), path
    append_match = re.search(r"sed\s+-i\s+(['\"])\$a\\\\n(.*?)\1", text)
    if append_match:
        return "", append_match.group(2), path
    return "", "", path


def _clean_shell_path(value: object) -> str:
    text = str(value or "").strip().strip("'\"")
    return text.rstrip(";|&").strip()


def command_copy_target_from_encoder(encoder_text: str) -> str:
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("command copy target:"):
            return line.split(":", 1)[1].strip()
    return ""


def artifact_command_target_from_encoder(encoder_text: str) -> str:
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("artifact command target:"):
            return line.split(":", 1)[1].strip()
    return ""


def source_inspection_candidates_from_encoder(encoder_text: str) -> list[str]:
    candidates: list[str] = []
    active_subgoal_candidates: list[str] = []
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("active subgoal:"):
            command = _source_inspection_command_from_active_subgoal(line.split(":", 1)[1].strip())
            if command:
                active_subgoal_candidates.append(command)
            continue
        numbered_match = re.match(r"source inspection candidate\s+\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if numbered_match:
            command = _compact_command(numbered_match.group(1), limit=900)
            if command and _is_source_inspection_command(command):
                candidates.append(command)
                mirrored = _source_lines_candidate_for_command(command)
                if mirrored:
                    candidates.append(mirrored)
            continue
        if not lowered.startswith("source inspection candidate commands:"):
            continue
        value = line.split(":", 1)[1].strip()
        for item in value.split("|"):
            command = _compact_command(item, limit=360)
            if command and _is_source_inspection_command(command):
                candidates.append(command)
                mirrored = _source_lines_candidate_for_command(command)
                if mirrored:
                    candidates.append(mirrored)
    candidates.extend(active_subgoal_candidates)
    return list(dict.fromkeys(candidates))


def plan_source_inspection_candidates_from_encoder(encoder_text: str) -> list[str]:
    plan_paths: list[str] = []
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("plan:"):
            plan_paths.extend(_source_inspection_paths_from_plan_text(line.split(":", 1)[1].strip()))
    candidates: list[str] = []
    if len(plan_paths) > 1:
        candidates.append(_compact_command("cat " + " ".join(plan_paths[:6]), limit=900))
        candidates.append(
            _compact_command(
                "cat " + " ".join(plan_paths[:6]) + " 2>/dev/null || echo 'Files not found or empty'",
                limit=900,
            )
        )
    for path in plan_paths:
        command = _compact_command(f"cat {path}", limit=360)
        candidates.append(command)
        mirrored = _source_lines_candidate_for_command(command)
        if mirrored:
            candidates.append(mirrored)
        candidates.append(_compact_command(f"cat {path} | head -50", limit=360))
    return list(dict.fromkeys(candidates))


def _source_inspection_paths_from_plan_text(plan_text: str) -> list[str]:
    paths: list[str] = []
    for item in str(plan_text or "").split("|"):
        command = _source_inspection_command_from_active_subgoal(item.strip())
        path = _inspect_path_from_command(command) if command else ""
        if path and path not in paths:
            paths.append(path)
    return paths


def _source_inspection_command_from_active_subgoal(active_subgoal: str) -> str:
    text = str(active_subgoal or "").strip()
    prefix = "update workflow path "
    if not text.lower().startswith(prefix):
        return ""
    path = _clean_shell_path(text[len(prefix):])
    if not path:
        return ""
    return _compact_command(f"cat {path}", limit=360)


def _source_lines_candidate_for_command(command: str) -> str:
    if not " ".join(str(command or "").strip().split()).startswith("cat "):
        return ""
    path = _inspect_path_from_command(command)
    if not path or path.startswith("source_lines/"):
        return ""
    return _compact_command(f"cat source_lines/{path}.lines", limit=360)


def artifact_slot_target_from_encoder(encoder_text: str) -> tuple[str, str]:
    path = ""
    content = ""
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("artifact target path:"):
            path = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("artifact target content:"):
            content = line.split(":", 1)[1].strip()
            continue
    return path, content


def artifact_slot_target_from_task(
    *,
    active_subgoal: str,
    expected_file_contents: dict[Any, Any],
) -> tuple[str, str]:
    path = _artifact_path_from_subgoal(active_subgoal, expected_file_contents)
    if not path:
        return "", ""
    return path, _compact_command(expected_file_contents.get(path, ""), limit=1200)


def artifact_command_target_from_task(
    *,
    active_subgoal: str,
    expected_file_contents: dict[Any, Any],
) -> str:
    path = _artifact_path_from_subgoal(active_subgoal, expected_file_contents)
    if not path:
        return ""
    content = str(expected_file_contents.get(path, ""))
    parent = str(Path(path).parent)
    escaped_content = content.replace("'", "'\"'\"'")
    command = f"printf %s '{escaped_content}' > {path}"
    if parent and parent != ".":
        command = f"mkdir -p {parent} && {command}"
    return _compact_command(command, limit=1800)


def _first_matching_control_token(tokens: Iterable[object], pattern: str) -> str:
    compiled = re.compile(pattern)
    for token in tokens:
        text = str(token or "").strip()
        if compiled.fullmatch(text):
            return text
    return ""


def repair_line_protocol_with_command_copy_target(
    line_protocol: dict[str, Any],
    *,
    encoder_text: str,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(line_protocol, dict) or not line_protocol:
        return {}, []
    repaired = dict(line_protocol)
    warnings: list[str] = []
    target = command_copy_target_from_encoder(encoder_text)
    artifact_target = artifact_command_target_from_encoder(encoder_text)
    source_inspection_candidates = source_inspection_candidates_from_encoder(encoder_text)
    localized_edit_candidates = localized_edit_candidates_from_encoder(encoder_text)
    validation_present_candidates = validation_command_candidates_from_encoder(encoder_text, polarity="present")
    validation_absent_candidates = validation_command_candidates_from_encoder(encoder_text, polarity="absent")
    artifact_path, artifact_content = artifact_slot_target_from_encoder(encoder_text)
    action = str(repaired.get("action", "")).strip()
    content = str(repaired.get("content", "")).strip()
    pointer_tokens = tuple(str(token).strip() for token in repaired.get("tokens", []) if str(token).strip())
    if action == "code_execute":
        present_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_VALIDATE_PRESENT_CANDIDATE_\d+>",
        )
        present_pointer_candidate = validation_candidate_from_pointer_token(
            present_pointer_token,
            encoder_text,
            polarity="present",
        )
        if present_pointer_candidate:
            repaired["content"] = present_pointer_candidate
            repaired["target_path"] = _validation_path_from_command(present_pointer_candidate)
            repaired["verify_polarity"] = "present"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_PRESENT>",
            )
            repaired["validation_command_expanded"] = True
            return repaired, warnings
        absent_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_VALIDATE_ABSENT_CANDIDATE_\d+>",
        )
        absent_pointer_candidate = validation_candidate_from_pointer_token(
            absent_pointer_token,
            encoder_text,
            polarity="absent",
        )
        if absent_pointer_candidate:
            repaired["content"] = absent_pointer_candidate
            repaired["target_path"] = _validation_path_from_command(absent_pointer_candidate)
            repaired["verify_polarity"] = "absent"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_ABSENT>",
            )
            repaired["validation_command_expanded"] = True
            return repaired, warnings
        if validation_present_candidates and "<AK_VALIDATE_PRESENT>" in pointer_tokens:
            present_candidate = _preferred_validation_candidate_from_world(
                encoder_text,
                candidates=validation_present_candidates,
                polarity="present",
            )
            if present_candidate and content != present_candidate:
                repaired["content"] = present_candidate
                repaired["target_path"] = _validation_path_from_command(present_candidate)
                repaired["verify_polarity"] = "present"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_PRESENT>",
                )
                repaired["validation_command_repaired"] = True
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
        if validation_absent_candidates and "<AK_VALIDATE_ABSENT>" in pointer_tokens:
            absent_candidate = _preferred_validation_candidate_from_world(
                encoder_text,
                candidates=validation_absent_candidates,
                polarity="absent",
            )
            if absent_candidate and content != absent_candidate:
                repaired["content"] = absent_candidate
                repaired["target_path"] = _validation_path_from_command(absent_candidate)
                repaired["verify_polarity"] = "absent"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_ABSENT>",
                )
                repaired["validation_command_repaired"] = True
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
        materialization_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_MATERIALIZE_CANDIDATE_\d+>",
        )
        materialization_pointer_candidate = materialization_candidate_from_pointer_token(
            materialization_pointer_token,
            encoder_text,
        )
        if materialization_pointer_candidate:
            repaired["content"] = materialization_pointer_candidate
            target_path = _redirect_path_from_command(materialization_pointer_candidate)
            target_content = _materialized_content_from_command(materialization_pointer_candidate)
            if target_path:
                repaired["target_path"] = target_path
            if target_content:
                repaired["target_content"] = target_content
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
            )
            repaired["materialization_candidate_expanded"] = True
            return repaired, warnings
        localized_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_\d+>",
        )
        localized_pointer_candidate = localized_edit_candidate_from_pointer_token(localized_pointer_token, encoder_text)
        if localized_pointer_candidate:
            repaired["content"] = localized_pointer_candidate
            _old, _new, path = _infer_sed_edit_slots(localized_pointer_candidate)
            if path:
                repaired["target_path"] = path
            if _old:
                repaired["edit_old"] = _old
            if _new:
                repaired["edit_new"] = _new
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_LOCALIZED_EDIT>",
            )
            repaired["localized_edit_candidate_expanded"] = True
            return repaired, warnings
        source_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_SOURCE_INSPECT_CANDIDATE_\d+>",
        )
        source_pointer_candidate = source_inspection_candidate_from_pointer_token(source_pointer_token, encoder_text)
        if source_pointer_candidate:
            repaired["content"] = source_pointer_candidate
            repaired["target_path"] = _inspect_path_from_command(source_pointer_candidate)
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_INSPECT_SOURCE>",
            )
            repaired["source_inspection_candidate_expanded"] = True
            return repaired, warnings
        present_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_VALIDATE_PRESENT_CANDIDATE_\d+>",
        )
        present_pointer_candidate = validation_candidate_from_pointer_token(
            present_pointer_token,
            encoder_text,
            polarity="present",
        )
        if present_pointer_candidate:
            repaired["content"] = present_pointer_candidate
            repaired["target_path"] = _validation_path_from_command(present_pointer_candidate)
            repaired["verify_polarity"] = "present"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_PRESENT>",
            )
            repaired["validation_command_expanded"] = True
            return repaired, warnings
        absent_pointer_token = _first_matching_control_token(
            (content, *pointer_tokens),
            r"<AK_COPY_VALIDATE_ABSENT_CANDIDATE_\d+>",
        )
        absent_pointer_candidate = validation_candidate_from_pointer_token(
            absent_pointer_token,
            encoder_text,
            polarity="absent",
        )
        if absent_pointer_candidate:
            repaired["content"] = absent_pointer_candidate
            repaired["target_path"] = _validation_path_from_command(absent_pointer_candidate)
            repaired["verify_polarity"] = "absent"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_ABSENT>",
            )
            repaired["validation_command_expanded"] = True
            return repaired, warnings
        if (
            validation_present_candidates
            and (
                "<AK_VALIDATE_PRESENT>" in pointer_tokens
                or _line_protocol_has_exec_kind(repaired, "verify_present")
            )
        ):
            present_candidate = _preferred_validation_candidate_from_world(
                encoder_text,
                candidates=validation_present_candidates,
                polarity="present",
            )
            if present_candidate and content != present_candidate:
                repaired["content"] = present_candidate
                repaired["target_path"] = _validation_path_from_command(present_candidate)
                repaired["verify_polarity"] = "present"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_PRESENT>",
                )
                repaired["validation_command_repaired"] = True
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
        if (
            validation_absent_candidates
            and (
                "<AK_VALIDATE_ABSENT>" in pointer_tokens
                or _line_protocol_has_exec_kind(repaired, "verify_absent")
            )
        ):
            absent_candidate = _preferred_validation_candidate_from_world(
                encoder_text,
                candidates=validation_absent_candidates,
                polarity="absent",
            )
            if absent_candidate and content != absent_candidate:
                repaired["content"] = absent_candidate
                repaired["target_path"] = _validation_path_from_command(absent_candidate)
                repaired["verify_polarity"] = "absent"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_ABSENT>",
                )
                repaired["validation_command_repaired"] = True
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
    if artifact_path and str(repaired.get("target_path", "")).strip() == "<AK_COPY_ARTIFACT_PATH>":
        repaired["target_path"] = artifact_path
        repaired["artifact_path_target_expanded"] = True
    if artifact_path and str(repaired.get("target_path", "")).strip() == "<AK_COPY_ARTIFACT_TARGET>":
        repaired["target_path"] = artifact_path
        repaired["artifact_path_target_expanded"] = True
        repaired["artifact_pointer_slot_normalized"] = True
    if artifact_content and str(repaired.get("target_content", "")).strip() == "<AK_COPY_ARTIFACT_CONTENT>":
        repaired["target_content"] = artifact_content
        repaired["artifact_content_target_expanded"] = True
    if artifact_content and str(repaired.get("target_content", "")).strip() == "<AK_COPY_ARTIFACT_TARGET>":
        repaired["target_content"] = artifact_content
        repaired["artifact_content_target_expanded"] = True
        repaired["artifact_pointer_slot_normalized"] = True
    if action == "code_execute" and artifact_target and content == "<AK_COPY_ARTIFACT_TARGET>":
        repaired["content"] = artifact_target
        repaired["artifact_command_target_expanded"] = True
        return repaired, warnings
    if (
        action == "code_execute"
        and artifact_target
        and _line_protocol_has_exec_kind(repaired, "materialize_artifact")
        and _line_protocol_targets_artifact_slot(
            repaired,
            artifact_path=artifact_path,
            artifact_content=artifact_content,
        )
        and not (
            artifact_path
            and _redirect_path_from_command(content) == artifact_path
            and _looks_like_materialization_command(content)
        )
        and _should_repair_to_artifact_command_target(content, artifact_content=artifact_content)
        and content != artifact_target
    ):
        repaired["content"] = artifact_target
        repaired["artifact_command_target_repaired"] = True
        return repaired, warnings
    if action == "code_execute" and target and content == "<AK_COPY_COMMAND_TARGET>":
        repaired["content"] = target
        repaired["command_copy_target_expanded"] = True
        return repaired, warnings
    if (
        action == "code_execute"
        and target
        and _low_conf_artifact_repair_tokens(repaired)
        and _should_repair_to_command_copy_target(
            content,
            command_copy_target=target,
            target_path=str(repaired.get("target_path", "")).strip(),
        )
        and _redirect_path_from_command(target)
    ):
        repaired["content"] = target
        repaired["target_path"] = _redirect_path_from_command(target)
        repaired["tokens"] = _replace_exec_kind_token(
            repaired.get("tokens", []),
            "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
        )
        repaired["command_copy_target_repaired"] = True
        repaired["materialize_exec_kind_repaired"] = True
        return repaired, warnings
    if (
        action == "code_execute"
        and target
        and _line_protocol_has_exec_kind(repaired, "materialize_artifact")
        and _line_protocol_targets_command_copy_target(
            repaired,
            command_copy_target=target,
        )
        and _should_repair_to_command_copy_target(
            content,
            command_copy_target=target,
            target_path=str(repaired.get("target_path", "")).strip(),
        )
        and content != target
    ):
        repaired["content"] = target
        repaired["command_copy_target_repaired"] = True
        return repaired, warnings
    localized_pointer_token = _first_matching_control_token(
        (content, *pointer_tokens),
        r"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_\d+>",
    )
    localized_pointer_candidate = localized_edit_candidate_from_pointer_token(localized_pointer_token, encoder_text)
    if action == "code_execute" and localized_pointer_candidate:
        repaired["content"] = localized_pointer_candidate
        _old, _new, path = _infer_sed_edit_slots(localized_pointer_candidate)
        if path:
            repaired["target_path"] = path
        if _old:
            repaired["edit_old"] = _old
        if _new:
            repaired["edit_new"] = _new
        repaired["localized_edit_candidate_expanded"] = True
        return repaired, warnings
    source_pointer_token = _first_matching_control_token(
        (content, *pointer_tokens),
        r"<AK_COPY_SOURCE_INSPECT_CANDIDATE_\d+>",
    )
    source_pointer_candidate = source_inspection_candidate_from_pointer_token(source_pointer_token, encoder_text)
    if action == "code_execute" and source_pointer_candidate:
        repaired["content"] = source_pointer_candidate
        repaired["target_path"] = _inspect_path_from_command(source_pointer_candidate)
        repaired["tokens"] = _replace_exec_kind_token(
            repaired.get("tokens", []),
            "<AK_EXEC_KIND_INSPECT_SOURCE>",
        )
        repaired["source_inspection_candidate_expanded"] = True
        return repaired, warnings
    present_pointer_token = _first_matching_control_token(
        (content, *pointer_tokens),
        r"<AK_COPY_VALIDATE_PRESENT_CANDIDATE_\d+>",
    )
    present_pointer_candidate = validation_candidate_from_pointer_token(present_pointer_token, encoder_text, polarity="present")
    if action == "code_execute" and present_pointer_candidate:
        repaired["content"] = present_pointer_candidate
        repaired["target_path"] = _validation_path_from_command(present_pointer_candidate)
        repaired["verify_polarity"] = "present"
        repaired["tokens"] = _replace_exec_kind_token(
            repaired.get("tokens", []),
            "<AK_EXEC_KIND_VERIFY_PRESENT>",
        )
        repaired["validation_command_expanded"] = True
        return repaired, warnings
    absent_pointer_token = _first_matching_control_token(
        (content, *pointer_tokens),
        r"<AK_COPY_VALIDATE_ABSENT_CANDIDATE_\d+>",
    )
    absent_pointer_candidate = validation_candidate_from_pointer_token(absent_pointer_token, encoder_text, polarity="absent")
    if action == "code_execute" and absent_pointer_candidate:
        repaired["content"] = absent_pointer_candidate
        repaired["target_path"] = _validation_path_from_command(absent_pointer_candidate)
        repaired["verify_polarity"] = "absent"
        repaired["tokens"] = _replace_exec_kind_token(
            repaired.get("tokens", []),
            "<AK_EXEC_KIND_VERIFY_ABSENT>",
        )
        repaired["validation_command_expanded"] = True
        return repaired, warnings
    if action == "code_execute" and _line_protocol_has_exec_kind(repaired, "verify_present"):
        candidate = _matching_validation_command(
            content=content,
            target_path=str(repaired.get("target_path", "")).strip(),
            candidates=validation_present_candidates,
        )
        if candidate and content != candidate and _can_repair_validation_probe(content):
            repaired["content"] = candidate
            repaired["target_path"] = _validation_path_from_command(candidate)
            repaired["verify_polarity"] = "present"
            repaired["validation_command_repaired"] = True
            return repaired, warnings
    if action == "code_execute" and _line_protocol_has_exec_kind(repaired, "verify_absent"):
        candidate = _matching_validation_command(
            content=content,
            target_path=str(repaired.get("target_path", "")).strip(),
            candidates=validation_absent_candidates,
        )
        if candidate and content != candidate and _can_repair_validation_probe(content):
            repaired["content"] = candidate
            repaired["target_path"] = _validation_path_from_command(candidate)
            repaired["verify_polarity"] = "absent"
            repaired["validation_command_repaired"] = True
            return repaired, warnings
        present_candidate = _opposite_present_validation_command(
            content=content,
            candidates=validation_present_candidates,
            absent_candidates=validation_absent_candidates,
        )
        if present_candidate:
            repaired["content"] = present_candidate
            repaired["target_path"] = _validation_path_from_command(present_candidate)
            repaired["verify_polarity"] = "present"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_PRESENT>",
            )
            repaired["validation_command_repaired"] = True
            repaired["validation_exec_kind_repaired"] = True
            return repaired, warnings
    if action == "code_execute" and _line_protocol_has_exec_kind(repaired, "inspect_source"):
        if _looks_like_validation_command(content):
            present_validation = _matching_validation_command(
                content=content,
                target_path=str(repaired.get("target_path", "")).strip(),
                candidates=validation_present_candidates,
            )
            absent_validation = _matching_validation_command(
                content=content,
                target_path=str(repaired.get("target_path", "")).strip(),
                candidates=validation_absent_candidates,
            )
            if present_validation == content:
                repaired["verify_polarity"] = "present"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_PRESENT>",
                )
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
            if absent_validation == content:
                repaired["verify_polarity"] = "absent"
                repaired["tokens"] = _replace_exec_kind_token(
                    repaired.get("tokens", []),
                    "<AK_EXEC_KIND_VERIFY_ABSENT>",
                )
                repaired["validation_exec_kind_repaired"] = True
                return repaired, warnings
        present_candidate = _source_probe_to_present_validation_command(
            content=content,
            candidates=validation_present_candidates,
            absent_candidates=validation_absent_candidates,
            source_inspection_candidates=source_inspection_candidates,
            direct_artifact_contract=_has_direct_artifact_validation_contract(encoder_text),
        )
        if present_candidate:
            repaired["content"] = present_candidate
            repaired["target_path"] = _validation_path_from_command(present_candidate)
            repaired["verify_polarity"] = "present"
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_VERIFY_PRESENT>",
            )
            repaired["validation_command_repaired"] = True
            repaired["validation_exec_kind_repaired"] = True
            return repaired, warnings
        candidate = _matching_source_inspection_candidate(
            content=content,
            target_path=str(repaired.get("target_path", "")).strip(),
            candidates=source_inspection_candidates,
        )
        if candidate and content != candidate:
            repaired["content"] = candidate
            repaired["target_path"] = _inspect_path_from_command(candidate)
            repaired["source_inspection_candidate_repaired"] = True
            return repaired, warnings
    if action == "code_execute" and _low_conf_artifact_repair_tokens(repaired):
        candidate = _matching_source_inspection_candidate(
            content=content,
            target_path=str(repaired.get("target_path", "")).strip(),
            candidates=source_inspection_candidates,
        )
        if (
            candidate
            and _looks_like_validation_command(content)
            and not validation_present_candidates
            and not validation_absent_candidates
        ):
            repaired["content"] = candidate
            repaired["target_path"] = _inspect_path_from_command(candidate)
            repaired["tokens"] = _replace_exec_kind_token(
                repaired.get("tokens", []),
                "<AK_EXEC_KIND_INSPECT_SOURCE>",
            )
            repaired["source_inspection_candidate_repaired"] = True
            repaired["low_conf_artifact_repair_source_inspection_repaired"] = True
            return repaired, warnings
    if action == "code_execute" and _line_protocol_has_exec_kind(repaired, "localized_edit"):
        candidate = _matching_localized_edit_candidate(
            content=content,
            target_path=str(repaired.get("target_path", "")).strip(),
            candidates=localized_edit_candidates,
        )
        if candidate and content != candidate:
            repaired["content"] = candidate
            _old, _new, path = _infer_sed_edit_slots(candidate)
            if path:
                repaired["target_path"] = path
            if _old:
                repaired["edit_old"] = _old
            if _new:
                repaired["edit_new"] = _new
            repaired["localized_edit_candidate_repaired"] = True
            return repaired, warnings
    return repaired, warnings


def _line_protocol_has_exec_kind(line_protocol: dict[str, Any], family: str) -> bool:
    return neural_controller_exec_kind_family(line_protocol.get("tokens", [])) == family


def _line_protocol_targets_artifact_slot(
    line_protocol: dict[str, Any],
    *,
    artifact_path: str,
    artifact_content: str,
) -> bool:
    target_path = str(line_protocol.get("target_path", "")).strip()
    target_content = str(line_protocol.get("target_content", "")).strip()
    if artifact_path and target_path == artifact_path:
        return True
    if artifact_content and target_content == artifact_content:
        return True
    return False


def _line_protocol_targets_command_copy_target(
    line_protocol: dict[str, Any],
    *,
    command_copy_target: str,
) -> bool:
    target_path = str(line_protocol.get("target_path", "")).strip()
    command_path = _redirect_path_from_command(command_copy_target)
    if target_path and command_path and target_path == command_path:
        return True
    return _low_conf_artifact_repair_tokens(line_protocol)


def _should_repair_to_artifact_command_target(content: str, *, artifact_content: str = "") -> bool:
    text = str(content or "").strip()
    if not text:
        return True
    if text in {"<AK_COPY_ARTIFACT_TARGET>", "<AK_COPY_COMMAND_TARGET>"}:
        return True
    if text.count("'") % 2 == 1 or text.count('"') % 2 == 1:
        return True
    if _looks_like_validation_command(text):
        return True
    if not _looks_like_materialization_command(text):
        return True
    return False


def _should_repair_to_command_copy_target(
    content: str,
    *,
    command_copy_target: str = "",
    target_path: str = "",
) -> bool:
    text = str(content or "").strip()
    if not text:
        return True
    if text in {"<AK_COPY_COMMAND_TARGET>", "<AK_COPY_ARTIFACT_TARGET>"}:
        return True
    if text.count("'") % 2 == 1 or text.count('"') % 2 == 1:
        return True
    if _looks_like_validation_command(text):
        return False
    content_path = _redirect_path_from_command(text)
    declared_path = _clean_shell_path(target_path)
    if (
        content_path
        and declared_path
        and content_path == declared_path
        and _materialization_redirect_count(command_copy_target) > 1
    ):
        return False
    if _looks_like_materialization_command(text) and _materialization_redirect_count(command_copy_target) > 1:
        return False
    target_path = _redirect_path_from_command(command_copy_target)
    if content_path and target_path and content_path != target_path:
        return True
    return not _looks_like_materialization_command(text) and not _is_source_inspection_command(text)


def _looks_like_materialization_command(content: str) -> bool:
    text = " ".join(str(content or "").strip().split())
    if not text:
        return False
    return bool(re.search(r"(^|[;&|]\s*)(mkdir|printf|echo|cat)\b.*(?:>|>>|tee\b)", text))


def _materialization_redirect_count(content: str) -> int:
    text = str(content or "")
    return len(re.findall(r"(?:^|[^>])>{1,2}(?!>)", text))


def validation_command_candidates_from_encoder(encoder_text: str, *, polarity: str) -> list[str]:
    wanted = "validation present commands:" if polarity == "present" else "validation absent commands:"
    numbered_prefix = "validation present candidate" if polarity == "present" else "validation absent candidate"
    candidates: list[str] = []
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        numbered_match = re.match(rf"{re.escape(numbered_prefix)}\s+\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if numbered_match:
            command = _compact_command(numbered_match.group(1), limit=360)
            if _validation_path_from_command(command):
                candidates.append(command)
            continue
        if not line.lower().startswith(wanted):
            continue
        value = line.split(":", 1)[1].strip()
        for item in value.split("|"):
            command = _compact_command(item, limit=360)
            if _validation_path_from_command(command):
                candidates.append(command)
    return list(dict.fromkeys(candidates))


def materialization_candidates_from_encoder(encoder_text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        numbered_match = re.match(r"materialization candidate\s+\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if numbered_match:
            command = _compact_command(numbered_match.group(1), limit=1800)
            if _looks_like_materialization_command(command):
                candidates.append(command)
            continue
        for prefix in ("Artifact command target:", "Command copy target:"):
            if line.lower().startswith(prefix.lower()):
                command = _compact_command(line.split(":", 1)[1].strip(), limit=1800)
                if _looks_like_materialization_command(command):
                    candidates.append(command)
                break
        if line.lower().startswith("active materialization target:"):
            command = _compact_command(line.split(":", 1)[1].strip(), limit=1800)
            if _looks_like_materialization_command(command):
                candidates.append(command)
    return list(dict.fromkeys(candidates))


def augment_encoder_with_active_materialization_target(encoder_text: str) -> str:
    text = str(encoder_text or "")
    if "Active materialization target:" in text:
        return text
    active_subgoal = ""
    command_copy_target = ""
    artifact_target = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("active subgoal:"):
            active_subgoal = line.split(":", 1)[1].strip()
        elif line.lower().startswith("subgoal=") and not active_subgoal:
            active_subgoal = line.split("=", 1)[1].strip()
        elif line.lower().startswith("command copy target:"):
            command_copy_target = line.split(":", 1)[1].strip()
        elif line.lower().startswith("artifact command target:"):
            artifact_target = line.split(":", 1)[1].strip()
    target = _active_materialization_target_command(
        command_copy_target=command_copy_target,
        artifact_target=artifact_target,
        active_subgoal=active_subgoal,
    )
    if not target:
        return text
    lines = text.splitlines()
    insert_at = len(lines)
    for index, raw_line in enumerate(lines):
        if raw_line.strip().lower().startswith(("materialization candidate commands:", "materialization candidate 1:")):
            insert_at = index
            break
    lines.insert(insert_at, "Active materialization target: " + _compact_command(target, limit=1800))
    return "\n".join(lines)


def augment_encoder_with_plan_source_inspection_candidates(encoder_text: str) -> str:
    text = str(encoder_text or "")
    candidates = plan_source_inspection_candidates_from_encoder(text)
    plan_derived = [
        candidate
        for candidate in candidates
        if candidate and candidate not in text
    ]
    if not plan_derived:
        return text
    lines = text.splitlines()
    insert_at = len(lines)
    for index, raw_line in enumerate(lines):
        if raw_line.strip().lower().startswith(("validation present candidate", "validation absent candidate", "localized edit candidate", "command copy target:", "success command:")):
            insert_at = index
            break
    additions = ["Source inspection candidate commands: " + _compact(" | ".join(plan_derived[:8]), limit=900)]
    for index, command in enumerate(plan_derived[:24], start=1):
        additions.append(f"Source inspection candidate {index}: " + _compact_command(command, limit=900))
    lines[insert_at:insert_at] = additions
    return "\n".join(lines)


def materialization_candidate_index_from_token(token: str) -> int:
    match = re.fullmatch(r"<AK_COPY_MATERIALIZE_CANDIDATE_(\d+)>", str(token or "").strip())
    if not match:
        return 0
    return int(match.group(1))


def materialization_candidate_from_pointer_token(token: str, encoder_text: str) -> str:
    index = materialization_candidate_index_from_token(token)
    if index <= 0:
        return ""
    candidates = materialization_candidates_from_encoder(encoder_text)
    if index > len(candidates):
        return ""
    return candidates[index - 1]


def _encoder_world_list_values(encoder_text: str, key: str) -> list[str]:
    raw_payload = ""
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("World:"):
            raw_payload = line.split(":", 1)[1].strip()
            break
    if not raw_payload:
        return []
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(\[[^\]]*\])', raw_payload)
        if not match:
            return []
        try:
            values = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        return _unique_strings(values) if isinstance(values, list) else []
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return _unique_strings(values)


def _preferred_validation_candidate_from_world(
    encoder_text: str,
    *,
    candidates: list[str],
    polarity: str,
) -> str:
    if not candidates:
        return ""
    preferred_paths = (
        _encoder_world_list_values(encoder_text, "existing_expected_artifacts")
        if polarity == "present"
        else _encoder_world_list_values(encoder_text, "present_forbidden_artifacts")
    )
    for path in preferred_paths:
        for candidate in candidates:
            if _validation_path_from_command(candidate) == path:
                return candidate
    if not preferred_paths and candidates:
        return candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    return ""


def localized_edit_candidates_from_encoder(encoder_text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(encoder_text or "").splitlines():
        line = raw_line.strip()
        numbered_match = re.match(r"localized edit candidate\s+\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if numbered_match:
            command = _normalize_localized_edit_candidate_text(_compact_command(numbered_match.group(1), limit=900))
            if _is_complete_localized_edit_candidate(command):
                candidates.append(command)
            continue
        if not line.lower().startswith("localized edit candidate commands:"):
            continue
        value = line.split(":", 1)[1].strip()
        for item in value.split("|"):
            command = _normalize_localized_edit_candidate_text(_compact_command(item, limit=900))
            if _is_complete_localized_edit_candidate(command):
                candidates.append(command)
    return list(dict.fromkeys(candidates))


def localized_edit_candidate_pointer_token(index: int) -> str:
    if 1 <= int(index) <= 24:
        return f"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_{int(index)}>"
    return ""


def localized_edit_candidate_index_from_token(token: str) -> int:
    match = re.fullmatch(r"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_(\d+)>", str(token or "").strip())
    if not match:
        return 0
    index = int(match.group(1))
    return index if 1 <= index <= 24 else 0


def localized_edit_candidate_from_pointer_token(token: str, encoder_text: str) -> str:
    index = localized_edit_candidate_index_from_token(token)
    if not index:
        return ""
    candidates = localized_edit_candidates_from_encoder(encoder_text)
    if index > len(candidates):
        return ""
    return candidates[index - 1]


def source_inspection_candidate_index_from_token(token: str) -> int:
    match = re.fullmatch(r"<AK_COPY_SOURCE_INSPECT_CANDIDATE_(\d+)>", str(token or "").strip())
    if not match:
        return 0
    index = int(match.group(1))
    return index if 1 <= index <= 24 else 0


def source_inspection_candidate_from_pointer_token(token: str, encoder_text: str) -> str:
    index = source_inspection_candidate_index_from_token(token)
    if not index:
        return ""
    candidates = source_inspection_candidates_from_encoder(encoder_text)
    if index > len(candidates):
        return ""
    return candidates[index - 1]


def validation_candidate_index_from_token(token: str, *, polarity: str) -> int:
    prefix = "PRESENT" if polarity == "present" else "ABSENT"
    match = re.fullmatch(rf"<AK_COPY_VALIDATE_{prefix}_CANDIDATE_(\d+)>", str(token or "").strip())
    if not match:
        return 0
    index = int(match.group(1))
    return index if 1 <= index <= 24 else 0


def validation_candidate_from_pointer_token(token: str, encoder_text: str, *, polarity: str) -> str:
    index = validation_candidate_index_from_token(token, polarity=polarity)
    if not index:
        return ""
    candidates = validation_command_candidates_from_encoder(encoder_text, polarity=polarity)
    if index > len(candidates):
        return ""
    return candidates[index - 1]


def _matching_validation_command(
    *,
    content: str,
    target_path: str,
    candidates: list[str],
) -> str:
    if not candidates:
        return ""
    predicted_path = target_path or _validation_path_from_command(content)
    if predicted_path:
        for candidate in candidates:
            if _validation_path_from_command(candidate) == predicted_path:
                return candidate
        if _is_absent_validation_candidate_set(candidates) and _looks_like_validation_command(content):
            return candidates[0]
    if len(candidates) == 1 and (
        _looks_like_unstable_validation_probe(content, target_path)
        or _is_single_absent_validation_candidate(candidates[0])
    ):
        return candidates[0]
    return ""


def _is_absent_validation_candidate_set(candidates: list[str]) -> bool:
    return bool(candidates) and all(_validation_command_polarity(candidate) == "absent" for candidate in candidates)


def _validation_command_polarity(command: str) -> str:
    text = " ".join(str(command or "").strip().split())
    if re.match(r"^(test|\[)\s+!\s+-[fe]\s+", text):
        return "absent"
    if re.match(r"^(test|\[)\s+-[fe]\s+", text):
        return "present"
    return ""


def _matching_localized_edit_candidate(
    *,
    content: str,
    target_path: str,
    candidates: list[str],
) -> str:
    if not candidates:
        return ""
    normalized_content = _normalize_localized_edit_candidate_text(_compact_command(content, limit=900))
    normalized_candidates = {
        _normalize_localized_edit_candidate_text(_compact_command(candidate, limit=900))
        for candidate in candidates
    }
    if normalized_content in normalized_candidates:
        return ""
    if len(candidates) == 1 and _looks_like_malformed_localized_edit(content, target_path):
        return candidates[0]
    if _looks_like_ungrounded_localized_edit(content):
        return candidates[0]
    return ""


def _looks_like_ungrounded_localized_edit(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return True
    return text.startswith("sed -i ")


def _looks_like_malformed_localized_edit(content: str, target_path: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return True
    if not text.startswith("sed -i "):
        return False
    if text.count("'") % 2 == 1 or text.count('"') % 2 == 1:
        return True
    path = _clean_shell_path(target_path or (text.split()[-1] if text.split() else ""))
    parts = [part for part in path.split("/") if part]
    return any(left == right for left, right in zip(parts, parts[1:]))


def _validation_path_from_command(command: str) -> str:
    parts = str(command or "").strip().split()
    if len(parts) < 3 or parts[0] != "test":
        return ""
    if parts[1] == "-f":
        return _clean_shell_path(" ".join(parts[2:]))
    if len(parts) >= 4 and parts[1] == "!" and parts[2] == "-f":
        return _clean_shell_path(" ".join(parts[3:]))
    return ""


def _looks_like_unstable_validation_probe(content: str, target_path: str) -> bool:
    text = f"{content} {target_path}".strip().lower()
    if not text:
        return True
    return any(marker in text for marker in ("test_info", "/null", " old_", "episode_replay"))


def _can_repair_validation_probe(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return True
    return text.startswith("test ")


def _opposite_present_validation_command(
    *,
    content: str,
    candidates: list[str],
    absent_candidates: list[str],
) -> str:
    if not candidates or absent_candidates:
        return ""
    path = _negative_validation_path_from_command(content)
    if not path:
        return ""
    for candidate in candidates:
        if _validation_path_from_command(candidate) == path:
            return candidate
    return ""


def _source_probe_to_present_validation_command(
    *,
    content: str,
    candidates: list[str],
    absent_candidates: list[str],
    source_inspection_candidates: list[str],
    direct_artifact_contract: bool,
) -> str:
    if not candidates or absent_candidates:
        return ""
    if source_inspection_candidates and not direct_artifact_contract:
        return ""
    path = _inspect_path_from_command(content)
    if not path:
        return ""
    for candidate in candidates:
        if _validation_path_from_command(candidate) == path:
            return candidate
    return ""


def _is_single_absent_validation_candidate(command: str) -> bool:
    return bool(_negative_validation_path_from_command(command))


def _has_direct_artifact_validation_contract(encoder_text: str) -> bool:
    lowered = str(encoder_text or "").lower()
    return "do not inspect source_lines" in lowered or "use the direct task command" in lowered


def _negative_validation_path_from_command(command: str) -> str:
    parts = str(command or "").strip().split()
    if len(parts) >= 4 and parts[0] == "test" and parts[1] == "!" and parts[2] == "-f":
        return _clean_shell_path(" ".join(parts[3:]))
    return ""


def _matching_source_inspection_candidate(
    *,
    content: str,
    target_path: str,
    candidates: list[str],
) -> str:
    if not candidates:
        return ""
    predicted_path = target_path or _inspect_path_from_command(content)
    if not _is_source_inspection_command(content):
        return candidates[0]
    preferred_candidates = _preferred_source_inspection_candidates(
        candidates=candidates,
        content=content,
        target_path=target_path,
    )
    prefix_candidate = _source_inspection_prefix_candidate(
        content=content,
        candidates=preferred_candidates,
    )
    if prefix_candidate:
        return prefix_candidate
    if predicted_path:
        for candidate in candidates:
            if _inspect_path_from_command(candidate) == predicted_path:
                return candidate
        if _looks_like_unstable_source_inspection(content, target_path):
            return preferred_candidates[0] if len(preferred_candidates) == 1 else ""
        return ""
    return preferred_candidates[0] if len(preferred_candidates) == 1 else ""


def _source_inspection_prefix_candidate(*, content: str, candidates: list[str]) -> str:
    normalized_content = " ".join(str(content or "").strip().split())
    if not normalized_content or not _is_source_inspection_command(normalized_content):
        return ""
    matches = []
    for candidate in candidates:
        normalized_candidate = " ".join(str(candidate or "").strip().split())
        if not normalized_candidate or normalized_candidate == normalized_content:
            continue
        if normalized_candidate.startswith(normalized_content + " "):
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else ""


def _preferred_source_inspection_candidates(
    *,
    candidates: list[str],
    content: str,
    target_path: str,
) -> list[str]:
    if _looks_like_unstable_source_inspection(content, target_path):
        source_line_candidates = [
            candidate
            for candidate in candidates
            if _inspect_path_from_command(candidate).startswith("source_lines/")
        ]
        if source_line_candidates:
            return source_line_candidates
    wants_source_lines = "source_lines/" in f"{target_path} {content}"
    filtered = [
        candidate
        for candidate in candidates
        if _inspect_path_from_command(candidate).startswith("source_lines/") == wants_source_lines
    ]
    return filtered or candidates


def _looks_like_unstable_source_inspection(content: str, target_path: str) -> bool:
    text = f"{content} {target_path}".strip()
    if not text:
        return True
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "source_lines/",
            "/null",
            " null ",
            "test_info/test_info",
            "files not found or empty",
        )
    ):
        return True
    for path in (_inspect_path_from_command(content), target_path):
        parts = [part for part in str(path or "").split("/") if part]
        for left, right in zip(parts, parts[1:]):
            if left == right:
                return True
    return False


def _low_conf_artifact_repair_tokens(line_protocol: dict[str, Any]) -> bool:
    tokens = {str(token).strip() for token in line_protocol.get("tokens", [])}
    return "<AK_ARTIFACT_REPAIR>" in tokens and "<AK_CONF_LOW>" in tokens


def _looks_like_validation_command(content: str) -> bool:
    text = " ".join(str(content or "").strip().split())
    return bool(re.match(r"^(test|\[)\s+!?\s*-[fe]\s+", text))


def _replace_exec_kind_token(tokens: object, replacement: str) -> list[str]:
    token_values = [str(token).strip() for token in tokens if str(token).strip()] if isinstance(tokens, list) else []
    replaced: list[str] = []
    inserted = False
    for token in token_values:
        if token in EXEC_KIND_FAMILY:
            if not inserted:
                replaced.append(replacement)
                inserted = True
            continue
        replaced.append(token)
    if not inserted:
        replaced.append(replacement)
    return replaced


def compact_neural_controller_shadow(
    shadow: dict[str, Any],
    *,
    selected_action: str = "",
    selected_content: str = "",
) -> dict[str, Any]:
    if not shadow:
        return {}
    line_protocol = shadow.get("line_protocol", {})
    if not isinstance(line_protocol, dict):
        line_protocol = {}
    policy_heads = shadow.get("policy_heads", {})
    if not isinstance(policy_heads, dict):
        policy_heads = {}
    scalar_control = shadow.get("scalar_control", {})
    if not isinstance(scalar_control, dict):
        scalar_control = {}
    predicted_action = str(line_protocol.get("action", "")).strip()
    predicted_content = str(line_protocol.get("content", "")).strip()
    selected_action = str(selected_action).strip()
    selected_content = str(selected_content).strip()
    payload = {
        "ready": bool(shadow.get("ready", False)),
        "manifest_path": str(shadow.get("manifest_path", "")).strip(),
        "generated_token_count": _int_value(shadow.get("generated_token_count"), 0),
        "predicted_action": predicted_action,
        "predicted_content": predicted_content,
        "predicted_content_preview": _compact(predicted_content, limit=240),
        "control_tokens": [
            str(token).strip()
            for token in line_protocol.get("tokens", [])
            if str(token).strip().startswith("<AK_")
        ][:16],
        "artifact_failure_mode": str(line_protocol.get("artifact_failure_mode", "")).strip(),
        "policy_heads": {
            str(key): round(_float_value(value, 0.0), 4)
            for key, value in sorted(policy_heads.items())
        },
    }
    if scalar_control:
        payload["scalar_control"] = {
            str(name): {
                str(key): round(_float_value(value, 0.0), 6)
                for key, value in sorted(values.items())
            }
            for name, values in sorted(scalar_control.items())
            if isinstance(values, dict)
        }
    guarded = shadow.get("guarded", {})
    if isinstance(guarded, dict) and guarded:
        payload["guarded_selected_source"] = str(guarded.get("selected_source", "")).strip()
        payload["guarded_selector_policy"] = str(guarded.get("selector_policy", "")).strip()
        payload["guarded_candidate_family"] = str(guarded.get("candidate_family", "")).strip()
        payload["guarded_fallback_families"] = [
            str(value).strip()
            for value in guarded.get("fallback_families", [])
            if str(value).strip()
        ][:16]
        payload["guarded_candidate_generated_token_count"] = _int_value(
            guarded.get("candidate_generated_token_count"),
            0,
        )
        baseline_prediction = _compact_line_protocol_prediction(guarded.get("baseline_line_protocol", {}))
        candidate_prediction = _compact_line_protocol_prediction(guarded.get("candidate_line_protocol", {}))
        if baseline_prediction:
            payload["guarded_baseline_prediction"] = baseline_prediction
        if candidate_prediction:
            payload["guarded_candidate_prediction"] = candidate_prediction
    if bool(line_protocol.get("command_copy_target_repaired", False)):
        payload["command_copy_target_repaired"] = True
    if bool(line_protocol.get("command_copy_target_expanded", False)):
        payload["command_copy_target_expanded"] = True
    if bool(line_protocol.get("artifact_command_target_expanded", False)):
        payload["artifact_command_target_expanded"] = True
    if bool(line_protocol.get("artifact_command_target_repaired", False)):
        payload["artifact_command_target_repaired"] = True
    if bool(line_protocol.get("artifact_path_target_expanded", False)):
        payload["artifact_path_target_expanded"] = True
    if bool(line_protocol.get("artifact_content_target_expanded", False)):
        payload["artifact_content_target_expanded"] = True
    if bool(line_protocol.get("artifact_pointer_slot_normalized", False)):
        payload["artifact_pointer_slot_normalized"] = True
    if bool(line_protocol.get("materialization_candidate_expanded", False)):
        payload["materialization_candidate_expanded"] = True
    if bool(line_protocol.get("source_inspection_candidate_repaired", False)):
        payload["source_inspection_candidate_repaired"] = True
    if bool(line_protocol.get("source_inspection_candidate_expanded", False)):
        payload["source_inspection_candidate_expanded"] = True
    if bool(line_protocol.get("low_conf_artifact_repair_source_inspection_repaired", False)):
        payload["low_conf_artifact_repair_source_inspection_repaired"] = True
    if bool(line_protocol.get("validation_command_repaired", False)):
        payload["validation_command_repaired"] = True
    if bool(line_protocol.get("validation_exec_kind_repaired", False)):
        payload["validation_exec_kind_repaired"] = True
    if bool(line_protocol.get("validation_command_expanded", False)):
        payload["validation_command_expanded"] = True
    if bool(line_protocol.get("localized_edit_candidate_repaired", False)):
        payload["localized_edit_candidate_repaired"] = True
    if bool(line_protocol.get("localized_edit_candidate_expanded", False)):
        payload["localized_edit_candidate_expanded"] = True
    error = str(shadow.get("error", "")).strip()
    if error:
        payload["error"] = _compact(error, limit=240)
    warnings = [
        str(value).strip()
        for value in shadow.get("warnings", [])
        if str(value).strip()
    ]
    if warnings:
        payload["warnings"] = warnings[:6]
    if selected_action:
        payload["selected_action"] = selected_action
        payload["action_agreement"] = bool(predicted_action and predicted_action == selected_action)
    if selected_content:
        payload["selected_content"] = selected_content
        payload["selected_content_preview"] = _compact(selected_content, limit=240)
        payload["content_comparison_evaluated"] = True
        payload["content_exact_agreement"] = bool(
            predicted_content and _canonical_text(predicted_content) == _canonical_text(selected_content)
        )
    return payload


def _compact_line_protocol_prediction(line_protocol: object) -> dict[str, Any]:
    if not isinstance(line_protocol, dict):
        return {}
    action = str(line_protocol.get("action", "")).strip()
    content = str(line_protocol.get("content", "")).strip()
    tokens = [
        str(token).strip()
        for token in line_protocol.get("tokens", [])
        if str(token).strip().startswith("<AK_")
    ][:16]
    payload: dict[str, Any] = {}
    if action:
        payload["action"] = action
    if content:
        payload["content_preview"] = _compact(content, limit=240)
        if len(content) <= 2000:
            payload["content"] = content
    if tokens:
        payload["control_tokens"] = tokens
        family = neural_controller_exec_kind_family(tokens)
        if family:
            payload["exec_kind_family"] = family
    failure_mode = str(line_protocol.get("artifact_failure_mode", "")).strip()
    if failure_mode:
        payload["artifact_failure_mode"] = failure_mode
    return payload


def summarize_neural_controller_shadow_steps(steps: list[Any]) -> dict[str, Any]:
    shadow_steps = 0
    ready_steps = 0
    action_agreement_steps = 0
    content_comparison_steps = 0
    content_exact_agreement_steps = 0
    contract_content_agreement_steps = 0
    command_copy_target_repaired_steps = 0
    unrepaired_content_exact_agreement_steps = 0
    verified_ready_steps = 0
    verified_action_agreement_steps = 0
    error_steps = 0
    warning_steps = 0
    token_counts: dict[str, int] = {}
    for step in steps:
        metadata = _step_field(step, "proposal_metadata", {})
        if not isinstance(metadata, dict):
            continue
        shadow = metadata.get("neural_controller_shadow", {})
        if not isinstance(shadow, dict) or not shadow:
            continue
        shadow_steps += 1
        ready = bool(shadow.get("ready", False))
        if ready:
            ready_steps += 1
        if str(shadow.get("error", "")).strip():
            error_steps += 1
        warnings = shadow.get("warnings", [])
        if isinstance(warnings, list) and any(str(value).strip() for value in warnings):
            warning_steps += 1
        command_copy_repaired = bool(shadow.get("command_copy_target_repaired", False))
        if not command_copy_repaired and isinstance(warnings, list):
            command_copy_repaired = any(
                str(value).strip() == "command_copy_target_repaired"
                for value in warnings
            )
        if bool(shadow.get("action_agreement", False)):
            action_agreement_steps += 1
        has_content_comparison = bool(shadow.get("content_comparison_evaluated", False)) or "content_exact_agreement" in shadow
        if has_content_comparison:
            content_comparison_steps += 1
        if has_content_comparison and bool(shadow.get("content_exact_agreement", False)):
            content_exact_agreement_steps += 1
            if not command_copy_repaired:
                unrepaired_content_exact_agreement_steps += 1
        if has_content_comparison and (
            bool(shadow.get("content_exact_agreement", False))
            or str(shadow.get("artifact_failure_mode", "")).strip() == "artifact_contract_success"
        ):
            contract_content_agreement_steps += 1
        if command_copy_repaired:
            command_copy_target_repaired_steps += 1
        verification = _step_field(step, "verification", {})
        verified = bool(verification.get("passed", False)) if isinstance(verification, dict) else False
        if verified and ready:
            verified_ready_steps += 1
        if verified and bool(shadow.get("action_agreement", False)):
            verified_action_agreement_steps += 1
        for token in shadow.get("control_tokens", []):
            normalized = str(token).strip()
            if normalized:
                token_counts[normalized] = token_counts.get(normalized, 0) + 1
    return {
        "shadow_steps": shadow_steps,
        "ready_steps": ready_steps,
        "action_agreement_steps": action_agreement_steps,
        "content_comparison_steps": content_comparison_steps,
        "content_exact_agreement_steps": content_exact_agreement_steps,
        "contract_content_agreement_steps": contract_content_agreement_steps,
        "unrepaired_content_exact_agreement_steps": unrepaired_content_exact_agreement_steps,
        "command_copy_target_repaired_steps": command_copy_target_repaired_steps,
        "verified_ready_steps": verified_ready_steps,
        "verified_action_agreement_steps": verified_action_agreement_steps,
        "error_steps": error_steps,
        "warning_steps": warning_steps,
        "control_token_counts": dict(sorted(token_counts.items())),
    }


def summarize_neural_controller_shadow_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = 0
    episodes_with_shadow = 0
    totals = _empty_shadow_summary()
    for document in documents:
        if not isinstance(document, dict):
            continue
        episode_count += 1
        summary = document.get("summary", {})
        shadow_summary = summary.get("neural_controller_shadow", {}) if isinstance(summary, dict) else {}
        if not _has_shadow_summary(shadow_summary):
            steps = document.get("steps", [])
            shadow_summary = summarize_neural_controller_shadow_steps(steps if isinstance(steps, list) else [])
        if not _has_shadow_summary(shadow_summary):
            trace_steps = _steps_from_policy_trace(document.get("policy_trace", []))
            shadow_summary = summarize_neural_controller_shadow_steps(trace_steps)
        if not _has_shadow_summary(shadow_summary):
            continue
        episodes_with_shadow += 1
        _merge_shadow_summary(totals, shadow_summary)
    totals["episode_count"] = episode_count
    totals["episodes_with_shadow"] = episodes_with_shadow
    totals["ready_rate"] = _rate(totals["ready_steps"], totals["shadow_steps"])
    totals["action_agreement_rate"] = _rate(totals["action_agreement_steps"], totals["ready_steps"])
    totals["content_exact_agreement_rate"] = _rate(
        totals["content_exact_agreement_steps"],
        totals["content_comparison_steps"],
    )
    totals["contract_content_agreement_rate"] = _rate(
        totals["contract_content_agreement_steps"],
        totals["content_comparison_steps"],
    )
    totals["unrepaired_content_exact_agreement_rate"] = _rate(
        totals["unrepaired_content_exact_agreement_steps"],
        totals["content_comparison_steps"],
    )
    totals["command_copy_target_repaired_rate"] = _rate(
        totals["command_copy_target_repaired_steps"],
        totals["content_comparison_steps"],
    )
    totals["error_rate"] = _rate(totals["error_steps"], totals["shadow_steps"])
    totals["warning_rate"] = _rate(totals["warning_steps"], totals["shadow_steps"])
    totals["verified_action_agreement_rate"] = _rate(
        totals["verified_action_agreement_steps"],
        totals["verified_ready_steps"],
    )
    return totals


def neural_controller_shadow_promotion_readiness(
    summary: dict[str, Any],
    *,
    min_episodes: int = 5,
    min_ready_steps: int = 25,
    min_content_comparison_steps: int = 5,
    min_action_agreement_rate: float = 0.70,
    min_verified_action_agreement_rate: float = 0.80,
    min_content_exact_agreement_rate: float = 0.80,
    max_error_rate: float = 0.0,
    max_warning_rate: float = 0.20,
) -> dict[str, Any]:
    blockers: list[str] = []
    authority_blockers: list[str] = []
    episode_count = _int_value(summary.get("episodes_with_shadow"), 0)
    ready_steps = _int_value(summary.get("ready_steps"), 0)
    content_comparison_steps = _int_value(summary.get("content_comparison_steps"), 0)
    action_rate = _float_value(summary.get("action_agreement_rate"), 0.0)
    verified_rate = _float_value(summary.get("verified_action_agreement_rate"), 0.0)
    content_rate = _float_value(summary.get("content_exact_agreement_rate"), 0.0)
    unrepaired_content_rate = _float_value(summary.get("unrepaired_content_exact_agreement_rate"), content_rate)
    repaired_rate = _float_value(summary.get("command_copy_target_repaired_rate"), 0.0)
    error_rate = _float_value(summary.get("error_rate"), 0.0)
    warning_rate = _float_value(summary.get("warning_rate"), 0.0)
    if episode_count < min_episodes:
        blockers.append("insufficient_shadow_episodes")
    if ready_steps < min_ready_steps:
        blockers.append("insufficient_ready_shadow_steps")
    if action_rate < min_action_agreement_rate:
        blockers.append("action_agreement_rate_below_gate")
    if verified_rate < min_verified_action_agreement_rate:
        blockers.append("verified_action_agreement_rate_below_gate")
    if error_rate > max_error_rate:
        blockers.append("shadow_error_rate_above_gate")
    if warning_rate > max_warning_rate:
        blockers.append("shadow_warning_rate_above_gate")
    if content_comparison_steps < min_content_comparison_steps:
        authority_blockers.append("insufficient_content_comparison_steps")
    if content_rate < min_content_exact_agreement_rate:
        authority_blockers.append("content_exact_agreement_rate_below_gate")
    pure_authority_blockers = list(authority_blockers)
    if unrepaired_content_rate < min_content_exact_agreement_rate:
        pure_authority_blockers.append("unrepaired_content_exact_agreement_rate_below_gate")
    if repaired_rate > 0.0:
        pure_authority_blockers.append("command_copy_target_repairs_present")
    return {
        "shadow_compare_ready": not blockers,
        "kernel_guarded_content_ready": not blockers and not authority_blockers,
        "content_authority_ready": not blockers and not pure_authority_blockers,
        "pure_content_authority_ready": not blockers and not pure_authority_blockers,
        "primary_authority_ready": False,
        "primary_authority_blocker": "requires_retained_promotion_gate",
        "blockers": blockers,
        "content_authority_blockers": authority_blockers,
        "pure_content_authority_blockers": pure_authority_blockers,
        "thresholds": {
            "min_episodes": min_episodes,
            "min_ready_steps": min_ready_steps,
            "min_content_comparison_steps": min_content_comparison_steps,
            "min_action_agreement_rate": min_action_agreement_rate,
            "min_verified_action_agreement_rate": min_verified_action_agreement_rate,
            "min_content_exact_agreement_rate": min_content_exact_agreement_rate,
            "max_error_rate": max_error_rate,
            "max_warning_rate": max_warning_rate,
        },
    }


def _dataset_special_tokens(dataset_manifest_path: str) -> tuple[str, ...]:
    path = Path(dataset_manifest_path)
    if not path.exists():
        return tuple()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return tuple()
    if not isinstance(payload, dict):
        return tuple()
    return tuple(
        token
        for token in payload.get("agentkernel_special_tokens", [])
        if str(token).strip() in FULL_KERNEL_CONTROL_TOKENS
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "shadow").strip().lower().replace("-", "_")
    if normalized not in {"disabled", "shadow", "advisory", "guarded", "primary"}:
        return "shadow"
    return normalized


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact(value: object, *, limit: int = 600) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()[:limit].rstrip()


def _compact_command(value: object, *, limit: int = 600) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\n")
    return text.strip()[:limit].rstrip()


def _artifact_path_from_subgoal(active_subgoal: str, expected_file_contents: dict[Any, Any]) -> str:
    subgoal = str(active_subgoal or "")
    paths = [str(path).strip() for path in expected_file_contents if str(path).strip()]
    for path in paths:
        if path in subgoal:
            return path
    if len(paths) == 1 and "artifact" in subgoal.lower():
        return paths[0]
    return ""


def _canonical_text(value: object) -> str:
    text = str(value or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return " ".join(text.split())


def _empty_shadow_summary() -> dict[str, Any]:
    return {
        "shadow_steps": 0,
        "ready_steps": 0,
        "action_agreement_steps": 0,
        "content_comparison_steps": 0,
        "content_exact_agreement_steps": 0,
        "contract_content_agreement_steps": 0,
        "unrepaired_content_exact_agreement_steps": 0,
        "command_copy_target_repaired_steps": 0,
        "verified_ready_steps": 0,
        "verified_action_agreement_steps": 0,
        "error_steps": 0,
        "warning_steps": 0,
        "control_token_counts": {},
    }


def _has_shadow_summary(summary: object) -> bool:
    return isinstance(summary, dict) and _int_value(summary.get("shadow_steps"), 0) > 0


def _merge_shadow_summary(total: dict[str, Any], summary: dict[str, Any]) -> None:
    if (
        "unrepaired_content_exact_agreement_steps" not in summary
        and _int_value(summary.get("content_exact_agreement_steps"), 0) > 0
    ):
        summary = dict(summary)
        summary["unrepaired_content_exact_agreement_steps"] = max(
            0,
            _int_value(summary.get("content_exact_agreement_steps"), 0)
            - _int_value(summary.get("command_copy_target_repaired_steps"), 0),
        )
    if (
        "content_comparison_steps" not in summary
        and _int_value(summary.get("content_exact_agreement_steps"), 0) > 0
    ):
        summary = dict(summary)
        summary["content_comparison_steps"] = _int_value(summary.get("content_exact_agreement_steps"), 0)
    if "contract_content_agreement_steps" not in summary:
        summary = dict(summary)
        summary["contract_content_agreement_steps"] = _int_value(summary.get("content_exact_agreement_steps"), 0)
    for key in (
        "shadow_steps",
        "ready_steps",
        "action_agreement_steps",
        "content_comparison_steps",
        "content_exact_agreement_steps",
        "contract_content_agreement_steps",
        "unrepaired_content_exact_agreement_steps",
        "command_copy_target_repaired_steps",
        "verified_ready_steps",
        "verified_action_agreement_steps",
        "error_steps",
        "warning_steps",
    ):
        total[key] = _int_value(total.get(key), 0) + _int_value(summary.get(key), 0)
    total_tokens = total.get("control_token_counts", {})
    if not isinstance(total_tokens, dict):
        total_tokens = {}
    tokens = summary.get("control_token_counts", {})
    if isinstance(tokens, dict):
        for token, count in tokens.items():
            normalized = str(token).strip()
            if normalized:
                total_tokens[normalized] = _int_value(total_tokens.get(normalized), 0) + _int_value(count, 0)
    total["control_token_counts"] = dict(sorted(total_tokens.items()))


def _step_field(step: Any, key: str, default: Any) -> Any:
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def _steps_from_policy_trace(policy_trace: object) -> list[dict[str, Any]]:
    if not isinstance(policy_trace, list):
        return []
    steps: list[dict[str, Any]] = []
    for item in policy_trace:
        if not isinstance(item, dict):
            continue
        neural = item.get("neural_controller", {})
        if not isinstance(neural, dict):
            continue
        shadow = neural.get("shadow", {})
        if not isinstance(shadow, dict) or not shadow:
            continue
        steps.append(
            {
                "proposal_metadata": {"neural_controller_shadow": shadow},
                "verification": {"passed": bool(item.get("verification_passed", False))},
            }
        )
    return steps


def _rate(numerator: object, denominator: object) -> float:
    denom = _int_value(denominator, 0)
    if denom <= 0:
        return 0.0
    return round(_int_value(numerator, 0) / denom, 6)
