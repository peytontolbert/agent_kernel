from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Any

from ..schemas import TaskSpec
from ..state import AgentState


PATCH_BUILDER_COMMANDS = ("patch_builder", "swe_patch_builder")
SOURCE_INSPECTION_EXECUTABLES = {"cat", "sed", "head", "tail", "ls", "find", "grep"}
REQUIRED_IDENTIFIER_PREFIXES = (
    "Artifact does not reference required identifier: ",
    "SWE patch does not reference required issue identifier: ",
)
NOOP_REPAIR_MARKERS = (
    "Artifact AST unchanged after ignoring docstrings/comments",
    "Artifact diff has no meaningful content change",
    "SWE patch python AST unchanged after ignoring docstrings/comments",
    "SWE patch diff has no meaningful content change",
    "swe_patch_builder replacement produced no meaningful change",
    "patch_builder replacement produced no meaningful change",
)
REPAIRABLE_VERIFIER_MARKERS = (
    "Artifact apply check failed",
    "Artifact syntax check failed",
    "Artifact AST unchanged after ignoring docstrings/comments",
    "Artifact diff has no meaningful content change",
    "Artifact removes existing Python definitions",
    "Artifact adds unused production function parameters",
    "Artifact does not reference required identifier",
    "SWE patch apply check failed",
    "SWE patch python syntax check failed",
    "SWE patch python AST unchanged after ignoring docstrings/comments",
    "SWE patch diff has no meaningful content change",
    "SWE patch removes existing Python definitions",
    "SWE patch adds unused production function parameters",
    "SWE patch does not reference required issue identifier",
    "SWE patch leaves invalid __init__ return values",
    "SWE patch leaves invalid __init__ generators",
    "SWE patch introduces local use before assignment",
)
ARTIFACT_DECISION_SOURCES = (
    "artifact_",
    "llm_artifact_",
    "swe_patch_",
    "llm_swe_patch_",
)


@dataclass(frozen=True, slots=True)
class ArtifactRepairContract:
    artifact_path: str
    builder_commands: tuple[str, ...] = PATCH_BUILDER_COMMANDS
    source_context_by_path: dict[str, str] | None = None
    required_identifiers: tuple[str, ...] = ()


def contract_from_state(state: AgentState) -> ArtifactRepairContract | None:
    task = state.task
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    explicit = metadata.get("artifact_repair_contract", {})
    if isinstance(explicit, dict):
        explicit_contract = _explicit_contract(task, explicit)
        if explicit_contract is not None:
            return explicit_contract

    # Compatibility normalization for existing patch-artifact tasks. Benchmark
    # adapters should eventually write artifact_repair_contract directly.
    artifact_path = _patch_artifact_path(task)
    if not artifact_path:
        return None
    return ArtifactRepairContract(
        artifact_path=artifact_path,
        builder_commands=PATCH_BUILDER_COMMANDS,
        source_context_by_path=source_context_by_path(task),
        required_identifiers=_required_identifiers(metadata),
    )


def source_context_by_path(task: TaskSpec) -> dict[str, str]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    setup_file_contents = metadata.get("setup_file_contents", {})
    if not isinstance(setup_file_contents, dict):
        return {}
    contexts: dict[str, str] = {}
    nested_context = setup_file_contents.get("source_context", {})
    if isinstance(nested_context, dict):
        contexts.update(
            {
                str(path).strip().strip("/"): str(content)
                for path, content in nested_context.items()
                if str(path).strip() and isinstance(content, str)
            }
        )
    for raw_path, content in setup_file_contents.items():
        if not isinstance(content, str):
            continue
        path = str(raw_path).strip().strip("/")
        if not path:
            continue
        if path.startswith("source_context/"):
            contexts.setdefault(path.removeprefix("source_context/").strip("/"), content)
            continue
        if path.startswith("source_lines/"):
            continue
        contexts.setdefault(path, content)
    return contexts


def parse_builder_command(
    command: str,
    *,
    artifact_path: str = "patch.diff",
    builder_commands: tuple[str, ...] = PATCH_BUILDER_COMMANDS,
    require_redirect: bool = True,
) -> dict[str, object] | None:
    normalized = str(command).strip()
    if not normalized:
        return None
    try:
        parts = shlex.split(normalized)
    except ValueError:
        return None
    if not parts or parts[0] not in set(builder_commands):
        return None
    args = parts[1:]
    if ">" in args:
        redirect_index = args.index(">")
        if redirect_index + 1 >= len(args) or args[redirect_index + 1] != artifact_path:
            return None
        if redirect_index + 2 != len(args):
            return None
        args = args[:redirect_index]
    elif require_redirect:
        return None
    path = ""
    operations: list[tuple[int, int, list[str]]] = []
    current_start_line = 0
    current_end_line = 0
    current_replacement_lines: list[str] = []

    def flush_operation() -> bool:
        nonlocal current_start_line, current_end_line, current_replacement_lines
        if not current_start_line:
            return True
        if not current_replacement_lines:
            return False
        operations.append((current_start_line, current_end_line, list(current_replacement_lines)))
        current_start_line = 0
        current_end_line = 0
        current_replacement_lines = []
        return True

    index = 0
    while index < len(args):
        token = args[index]
        if token == "--path" and index + 1 < len(args):
            path = str(args[index + 1]).strip()
            index += 2
            continue
        if token == "--replace-line" and index + 1 < len(args):
            if not flush_operation():
                return None
            try:
                current_start_line = int(args[index + 1])
            except ValueError:
                return None
            current_end_line = current_start_line
            index += 2
            continue
        if token == "--replace-lines" and index + 2 < len(args):
            if not flush_operation():
                return None
            try:
                current_start_line = int(args[index + 1])
                current_end_line = int(args[index + 2])
            except ValueError:
                return None
            index += 3
            continue
        if token == "--with" and index + 1 < len(args):
            if not current_start_line:
                return None
            current_replacement_lines.extend(str(args[index + 1]).split("\n"))
            index += 2
            continue
        return None
    if not flush_operation():
        return None
    return {"builder": parts[0], "artifact_path": artifact_path, "path": path, "operations": operations}


def command_is_source_inspection(command: str) -> bool:
    normalized = str(command).strip()
    if not normalized:
        return False
    first = normalized.split(maxsplit=1)[0]
    if first in SOURCE_INSPECTION_EXECUTABLES:
        return True
    if first in {"python", "python3"} and (
        "open(" in normalized or "readlines" in normalized or "read_text" in normalized
    ):
        return True
    return any(token in normalized for token in (" | ", " && ", " || ", " > /tmp/", " < "))


def preferred_builder_command(contract: ArtifactRepairContract | None) -> str:
    if contract is None:
        return PATCH_BUILDER_COMMANDS[0]
    if "patch_builder" in contract.builder_commands:
        return "patch_builder"
    for command in contract.builder_commands:
        if command:
            return command
    return PATCH_BUILDER_COMMANDS[0]


def required_command_shape(contract: ArtifactRepairContract | None) -> str:
    artifact_path = contract.artifact_path if contract is not None else "patch.diff"
    builder = preferred_builder_command(contract)
    return f"{builder} --path <allowed-path> --replace-line <line> --with '<replacement>' > {artifact_path}"


def reason_text(reasons: object) -> str:
    if isinstance(reasons, list):
        return "\n".join(str(reason) for reason in reasons)
    return ""


def is_noop_repair_reason_text(text: str) -> bool:
    return any(marker in str(text) for marker in NOOP_REPAIR_MARKERS)


def is_repairable_reason_text(text: str, *, artifact_path: str = "patch.diff") -> bool:
    normalized = str(text)
    missing_markers = (
        f"Artifact verifier missing artifact file: {artifact_path}",
        f"SWE patch verifier missing patch file: {artifact_path}",
        f"missing expected file: {artifact_path}",
    )
    return any(marker in normalized for marker in missing_markers) or any(
        marker in normalized for marker in REPAIRABLE_VERIFIER_MARKERS
    )


def required_identifier_from_reasons(reasons: object) -> str:
    if not isinstance(reasons, list):
        return ""
    for reason in reasons:
        text = str(reason).strip()
        for prefix in REQUIRED_IDENTIFIER_PREFIXES:
            if text.startswith(prefix):
                return text.removeprefix(prefix).strip()
    return ""


def classify_artifact_contract_failure_report(report: dict[str, Any]) -> dict[str, Any]:
    """Classify artifact-contract failures without depending on benchmark names."""

    if not isinstance(report, dict):
        return _artifact_failure_classification(
            mode="not_artifact_contract",
            repairable=False,
            evidence=[],
        )
    outcome = str(report.get("outcome", "")).strip()
    last_decision_source = str(report.get("last_decision_source", "")).strip()
    trace = [item for item in report.get("policy_trace", []) if isinstance(item, dict)]
    outcome_reasons = [str(value).strip() for value in report.get("outcome_reasons", []) if str(value).strip()]
    termination_reason = str(report.get("termination_reason", "")).strip()
    evidence = _artifact_failure_evidence(
        trace=trace,
        outcome_reasons=outcome_reasons,
        termination_reason=termination_reason,
    )
    if outcome in {"success", "completed"}:
        return _artifact_failure_classification(
            mode="artifact_contract_success" if _has_artifact_contract_evidence(report, trace, evidence) else "not_artifact_contract",
            repairable=False,
            evidence=evidence,
            last_decision_source=last_decision_source,
        )
    if not _has_artifact_contract_evidence(report, trace, evidence):
        return _artifact_failure_classification(
            mode="not_artifact_contract",
            repairable=False,
            evidence=evidence,
            last_decision_source=last_decision_source,
        )
    joined = "\n".join(evidence).lower()
    if last_decision_source in {"artifact_materialization_guard", "swe_patch_materialization_guard"}:
        mode = "artifact_materialization_guard_terminal"
    elif last_decision_source == "artifact_inference_failure_source_context_fallback":
        mode = "artifact_inference_failure_source_context_exhausted"
    elif "inference_failure" in joined:
        mode = "artifact_inference_failure"
    elif "removes existing python definitions" in joined:
        mode = "artifact_definition_header_or_destructive_replacement"
    elif "syntax" in joined or "indentationerror" in joined or "malformed" in joined:
        mode = "artifact_syntax_or_malformed_builder"
    elif "no meaningful content" in joined or "ast unchanged" in joined:
        mode = "artifact_semantic_noop"
    elif "does not reference required" in joined or "required identifier" in joined:
        mode = "artifact_missing_required_identifier"
    elif "invalid __init__ return values" in joined:
        mode = "artifact_invalid_init_return_value"
    elif "invalid __init__ generators" in joined:
        mode = "artifact_invalid_init_generator"
    elif "local use before assignment" in joined:
        mode = "artifact_local_use_before_assignment"
    elif "apply check failed" in joined:
        mode = "artifact_apply_check_failed"
    elif "missing expected file" in joined or "missing artifact file" in joined or "missing patch file" in joined:
        mode = "artifact_missing_after_response"
    elif "no_state_progress" in joined:
        mode = "artifact_repair_no_state_progress"
    elif "policy_terminated" in joined or "policy terminated" in joined:
        mode = "artifact_policy_terminated"
    else:
        mode = "artifact_contract_unknown"
    return _artifact_failure_classification(
        mode=mode,
        repairable=mode != "artifact_contract_unknown",
        evidence=evidence,
        last_decision_source=last_decision_source,
    )


def _explicit_contract(task: TaskSpec, payload: dict[str, Any]) -> ArtifactRepairContract | None:
    artifact_path = str(payload.get("artifact_path", "")).strip()
    if not artifact_path:
        return None
    raw_builders = payload.get("builder_commands", PATCH_BUILDER_COMMANDS)
    if isinstance(raw_builders, str):
        raw_builders = [raw_builders]
    builder_commands = tuple(str(value).strip() for value in raw_builders if str(value).strip())
    if not builder_commands:
        builder_commands = PATCH_BUILDER_COMMANDS
    return ArtifactRepairContract(
        artifact_path=artifact_path,
        builder_commands=builder_commands,
        source_context_by_path=source_context_by_path(task),
        required_identifiers=tuple(str(value).strip() for value in payload.get("required_identifiers", []) if str(value).strip()),
    )


def _artifact_failure_classification(
    *,
    mode: str,
    repairable: bool,
    evidence: list[str],
    last_decision_source: str = "",
) -> dict[str, Any]:
    return {
        "kind": "artifact_contract_failure",
        "mode": mode,
        "repairable": bool(repairable),
        "last_decision_source": last_decision_source,
        "evidence": evidence[:8],
    }


def _artifact_failure_evidence(
    *,
    trace: list[dict[str, Any]],
    outcome_reasons: list[str],
    termination_reason: str,
) -> list[str]:
    evidence: list[str] = []
    for item in trace:
        decision_source = str(item.get("decision_source", "")).strip()
        if decision_source:
            evidence.append(f"decision_source:{decision_source}")
        failure_origin = str(item.get("failure_origin", "")).strip()
        if failure_origin:
            evidence.append(f"failure_origin:{failure_origin}")
        reasons = item.get("verification_reasons", [])
        if isinstance(reasons, list):
            evidence.extend(str(reason).strip() for reason in reasons if str(reason).strip())
        proposal_metadata = item.get("proposal_metadata", {})
        if isinstance(proposal_metadata, dict):
            for key in (
                "retry_rejected_reason",
                "rejected_command",
                "retry_command",
                "fallback_path",
                "source_lines_path",
            ):
                value = str(proposal_metadata.get(key, "")).strip()
                if value:
                    evidence.append(f"{key}:{value}")
    evidence.extend(outcome_reasons)
    if termination_reason:
        evidence.append(termination_reason)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _has_artifact_contract_evidence(
    report: dict[str, Any],
    trace: list[dict[str, Any]],
    evidence: list[str],
) -> bool:
    metadata = report.get("task_metadata", {})
    if isinstance(metadata, dict):
        if isinstance(metadata.get("artifact_repair_contract"), dict):
            return True
        if metadata.get("patch_artifact_path") or metadata.get("swe_patch_workspace_relpath"):
            return True
        semantic_verifier = metadata.get("semantic_verifier", {})
        if isinstance(semantic_verifier, dict) and semantic_verifier.get("patch_path"):
            return True
    if any(str(item.get("decision_source", "")).startswith(ARTIFACT_DECISION_SOURCES) for item in trace):
        return True
    joined = "\n".join(evidence)
    return is_repairable_reason_text(joined) or "patch.diff" in joined


def _patch_artifact_path(task: TaskSpec) -> str:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    semantic_verifier = metadata.get("semantic_verifier", {})
    artifact_path = ""
    if isinstance(semantic_verifier, dict):
        artifact_path = str(semantic_verifier.get("patch_path", "")).strip()
    if not artifact_path:
        artifact_path = str(metadata.get("patch_artifact_path", "")).strip()
    if not artifact_path:
        artifact_path = str(metadata.get("swe_patch_workspace_relpath", "")).strip()
    if not artifact_path and "patch.diff" in task.expected_files:
        artifact_path = "patch.diff"
    if not artifact_path:
        return ""
    if artifact_path in task.expected_files or artifact_path == "patch.diff" or bool(metadata.get("swe_patch_output_path")):
        return artifact_path
    return ""


def _required_identifiers(metadata: dict[str, Any]) -> tuple[str, ...]:
    semantic_verifier = metadata.get("semantic_verifier", {})
    if not isinstance(semantic_verifier, dict):
        return ()
    return tuple(
        str(value).strip()
        for value in semantic_verifier.get("required_patch_identifiers", [])
        if str(value).strip()
    )
