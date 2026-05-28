from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping


ActionType = str


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    prompt: str
    workspace_subdir: str
    setup_commands: list[str] = field(default_factory=list)
    success_command: str = ""
    suggested_commands: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    expected_output_substrings: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    forbidden_output_substrings: list[str] = field(default_factory=list)
    expected_file_contents: dict[str, str] = field(default_factory=dict)
    max_steps: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = self.task_id.strip()
        self.prompt = self.prompt.strip()
        self.workspace_subdir = self.workspace_subdir.strip()
        self.setup_commands = [command.strip() for command in self.setup_commands if command.strip()]
        self.success_command = self.success_command.strip()
        self.suggested_commands = [command.strip() for command in self.suggested_commands if command.strip()]
        self.expected_files = [path.strip() for path in self.expected_files if path.strip()]
        self.expected_output_substrings = [
            needle.strip() for needle in self.expected_output_substrings if needle.strip()
        ]
        self.forbidden_files = [path.strip() for path in self.forbidden_files if path.strip()]
        self.forbidden_output_substrings = [
            needle.strip() for needle in self.forbidden_output_substrings if needle.strip()
        ]
        self.expected_file_contents = {
            path.strip(): content
            for path, content in self.expected_file_contents.items()
            if path.strip()
        }
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.workspace_subdir:
            raise ValueError("workspace_subdir must not be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_subdir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "workspace_subdir": self.workspace_subdir,
            "setup_commands": list(self.setup_commands),
            "success_command": self.success_command,
            "suggested_commands": list(self.suggested_commands),
            "expected_files": list(self.expected_files),
            "expected_output_substrings": list(self.expected_output_substrings),
            "forbidden_files": list(self.forbidden_files),
            "forbidden_output_substrings": list(self.forbidden_output_substrings),
            "expected_file_contents": dict(self.expected_file_contents),
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(payload.get("task_id", "")).strip(),
            prompt=str(payload.get("prompt", "")).strip(),
            workspace_subdir=str(payload.get("workspace_subdir", "")).strip(),
            setup_commands=[str(value) for value in payload.get("setup_commands", [])],
            success_command=str(payload.get("success_command", "")).strip(),
            suggested_commands=[str(value) for value in payload.get("suggested_commands", [])],
            expected_files=[str(value) for value in payload.get("expected_files", [])],
            expected_output_substrings=[str(value) for value in payload.get("expected_output_substrings", [])],
            forbidden_files=[str(value) for value in payload.get("forbidden_files", [])],
            forbidden_output_substrings=[str(value) for value in payload.get("forbidden_output_substrings", [])],
            expected_file_contents={
                str(path): str(content)
                for path, content in payload.get("expected_file_contents", {}).items()
            }
            if isinstance(payload.get("expected_file_contents", {}), dict)
            else {},
            max_steps=int(payload.get("max_steps", 5)),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), dict)
            else {},
        )


@dataclass(slots=True)
class ActionDecision:
    thought: str
    action: ActionType
    content: str
    done: bool = False
    selected_skill_id: str | None = None
    selected_retrieval_span_id: str | None = None
    retrieval_influenced: bool = False
    retrieval_ranked_skill: bool = False
    decision_source: str = "llm"
    tolbert_route_mode: str = ""
    proposal_source: str = ""
    proposal_novel: bool = False
    proposal_metadata: dict[str, Any] = field(default_factory=dict)
    shadow_decision: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thought = self.thought.strip()
        self.action = self.action.strip()
        self.content = self.content.strip()


@dataclass(slots=True)
class NodePathPrediction:
    tree_version: str
    decode_mode: str
    levels: list[int]
    predicted_level_ids: dict[str, int]
    confidence_by_level: dict[str, float]
    labels_by_level: dict[str, str]
    fallbacks: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedSpan:
    span_id: str
    text: str
    source_id: str
    span_type: str
    score: float
    node_path: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPacket:
    request_id: str
    created_at: str
    task: dict[str, str]
    control: dict[str, Any]
    tolbert: dict[str, Any]
    retrieval: dict[str, list[RetrievedSpan]]
    verifier_contract: dict[str, Any]


@dataclass(slots=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    capabilities_used: list[str] = field(default_factory=list)


def classify_command_result_failure(result: CommandResult | None) -> dict[str, Any]:
    """Classify terminal-boundary failures into generic recovery routes."""
    if result is None:
        return {}
    command = str(result.command).strip()
    combined = f"{result.stderr}\n{result.stdout}".strip()
    lowered = combined.lower()
    if result.timed_out:
        return _command_failure("timeout", command, combined, confidence=0.98)
    if result.exit_code == 0:
        return {}
    if "syntaxerror" in lowered or "parseerror" in lowered or "invalid syntax" in lowered:
        return _command_failure("syntax_error", command, combined, confidence=0.95)
    if "no meaningful change" in lowered or "patch is empty" in lowered or "empty patch" in lowered:
        return _command_failure("no_op_edit", command, combined, confidence=0.94)
    if "modulenotfounderror" in lowered or "importerror" in lowered or "cannot find module" in lowered:
        return _command_failure("import_error", command, combined, confidence=0.93)
    if "permission denied" in lowered or "eacces" in lowered:
        return _command_failure("permission_error", command, combined, confidence=0.93)
    if "no such file or directory" in lowered or "enoent" in lowered:
        return _command_failure("missing_file", command, combined, confidence=0.9)
    if "unexpected keyword argument" in lowered or "has no attribute" in lowered or "missing 1 required" in lowered:
        return _command_failure("api_mismatch", command, combined, confidence=0.82)
    test_like = any(token in lowered for token in ("failed", "assertionerror", "short test summary info"))
    command_test_like = any(token in command for token in ("pytest", "tox", "unittest", "npm test", "go test"))
    if test_like and command_test_like:
        return _command_failure("test_assertion_failure", command, combined, confidence=0.9)
    if "traceback (most recent call last)" in lowered or "runtimeerror" in lowered or "typeerror" in lowered:
        return _command_failure("runtime_exception", command, combined, confidence=0.78)
    if "version conflict" in lowered or "resolutionimpossible" in lowered or "dependency conflict" in lowered:
        return _command_failure("dependency_conflict", command, combined, confidence=0.86)
    if any(token in lowered for token in ("ruff", "flake8", "eslint", "prettier", "gofmt")):
        return _command_failure("lint_style_failure", command, combined, confidence=0.74)
    return _command_failure("runtime_or_command_failure", command, combined, confidence=0.55)


def _command_failure(mode: str, command: str, output: str, *, confidence: float) -> dict[str, Any]:
    signature_source = output or command
    signature = " ".join(signature_source.split())[:240]
    return {
        "kind": "terminal_failure_classification",
        "failure_class": mode,
        "failure_code": f"terminal_{mode}",
        "confidence": round(confidence, 3),
        "command": command[:240],
        "signature": signature,
        "evidence_tail": output[-2000:],
    }


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    reasons: list[str]
    command_result: CommandResult | None = None
    process_score: float = 0.0
    outcome_label: str = "failure"
    outcome_confidence: float = 1.0
    controllability: str = "agent"
    failure_codes: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "process_score": float(self.process_score),
            "outcome_label": str(self.outcome_label).strip() or ("success" if self.passed else "failure"),
            "outcome_confidence": float(self.outcome_confidence),
            "controllability": str(self.controllability).strip() or "agent",
            "failure_codes": list(self.failure_codes),
            "side_effects": list(self.side_effects),
            "criteria": [dict(item) for item in self.criteria],
            "evidence": [dict(item) for item in self.evidence],
        }


@dataclass(slots=True)
class StepRecord:
    index: int
    thought: str
    action: str
    content: str
    selected_skill_id: str | None
    command_result: dict[str, Any] | None
    verification: dict[str, Any]
    available_skill_count: int = 0
    retrieval_candidate_count: int = 0
    retrieval_evidence_count: int = 0
    retrieval_command_match: bool = False
    selected_retrieval_span_id: str | None = None
    retrieval_influenced: bool = False
    retrieval_ranked_skill: bool = False
    path_confidence: float = 0.0
    trust_retrieval: bool = False
    retrieval_direct_candidate_count: int = 0
    research_context_chunk_count: int = 0
    llm_visible_research_context_chunk_count: int = 0
    research_retrieval_evidence_count: int = 0
    research_model_asset_count: int = 0
    research_repository_match_count: int = 0
    research_algorithm_match_count: int = 0
    active_subgoal: str = ""
    subgoal_diagnoses: dict[str, Any] = field(default_factory=dict)
    acting_role: str = "executor"
    world_model_horizon: str = ""
    state_progress_delta: float = 0.0
    state_regression_count: int = 0
    state_transition: dict[str, Any] = field(default_factory=dict)
    failure_signals: list[str] = field(default_factory=list)
    failure_origin: str = ""
    command_governance: dict[str, Any] = field(default_factory=dict)
    runtime_attestation: dict[str, Any] = field(default_factory=dict)
    decision_source: str = "llm"
    tolbert_route_mode: str = ""
    proposal_source: str = ""
    proposal_novel: bool = False
    proposal_metadata: dict[str, Any] = field(default_factory=dict)
    shadow_decision: dict[str, Any] = field(default_factory=dict)
    latent_state_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeRecord:
    task_id: str
    prompt: str
    workspace: str
    success: bool
    steps: list[StepRecord]
    task_metadata: dict[str, Any] = field(default_factory=dict)
    task_contract: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    graph_summary: dict[str, Any] = field(default_factory=dict)
    universe_summary: dict[str, Any] = field(default_factory=dict)
    world_model_summary: dict[str, Any] = field(default_factory=dict)
    history_archive: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "workspace": self.workspace,
            "success": self.success,
            "task_metadata": self.task_metadata,
            "task_contract": self.task_contract,
            "plan": self.plan,
            "graph_summary": self.graph_summary,
            "universe_summary": self.universe_summary,
            "world_model_summary": self.world_model_summary,
            "history_archive": self.history_archive,
            "termination_reason": self.termination_reason,
            "steps": [asdict(step) for step in self.steps],
        }


def step_verification_passed(step: StepRecord | Mapping[str, Any] | None) -> bool:
    payload: Mapping[str, Any]
    if isinstance(step, StepRecord):
        payload = step.verification
    elif isinstance(step, Mapping):
        verification = step.get("verification", {})
        payload = verification if isinstance(verification, Mapping) else {}
    else:
        payload = {}
    return bool(payload.get("passed", False))


def classify_verification_reason(reason: str) -> str:
    normalized = str(reason).strip()
    if not normalized or normalized.lower() == "verification passed":
        return ""
    lowered = normalized.lower()
    if "timed out" in lowered:
        return "timeout"
    if "exit code" in lowered:
        return "command_failure"
    if "missing expected file content target" in lowered:
        return "missing_expected_file_content_target"
    if "missing expected file" in lowered:
        return "missing_expected_file"
    if "missing expected output" in lowered:
        return "missing_expected_output"
    if "forbidden file present" in lowered:
        return "forbidden_file_present"
    if "unexpected file content" in lowered:
        return "unexpected_file_content"
    if "forbidden output present" in lowered:
        return "forbidden_output_present"
    if "semantic report missing phrase" in lowered:
        return "semantic_report_missing_phrase"
    if "semantic report missing" in lowered:
        return "semantic_report_missing"
    if "semantic report does not cover" in lowered:
        return "semantic_report_missing_coverage"
    if "git repository missing" in lowered:
        return "git_repository_missing"
    if "git branch mismatch" in lowered:
        return "git_branch_mismatch"
    if "git branch inspection failed" in lowered:
        return "git_branch_inspection_failed"
    if "git diff missing expected path" in lowered:
        return "git_diff_missing_expected_path"
    if "git diff includes unexpected path" in lowered:
        return "git_diff_unexpected_path"
    if "required worker branch not accepted" in lowered:
        return "required_branch_unaccepted"
    if "git diff unexpectedly changed preserved path" in lowered:
        return "preserved_path_changed"
    if "generated artifact missing" in lowered:
        return "generated_artifact_missing"
    if "generated artifact not recorded in git diff" in lowered:
        return "generated_artifact_not_in_diff"
    if "git conflict remains unresolved" in lowered:
        return "unresolved_git_conflict"
    if "conflict markers still present" in lowered:
        return "conflict_markers_present"
    if "git worktree not clean" in lowered:
        return "git_worktree_not_clean"
    if "failed to execute" in lowered:
        return "verification_command_execution_failed"
    if "test command" in lowered and "exited with code" in lowered:
        return "verification_test_failed"
    if "success command" in lowered and "timed out" in lowered:
        return "success_command_timeout"
    if "success command" in lowered and "exited with code" in lowered:
        return "success_command_failed"
    if "semantic verifier contract malformed" in lowered:
        return "semantic_verifier_contract_malformed"
    if "swe patch replaces real behavior with placeholder output" in lowered:
        return "placeholder_output_replacement"
    if "swe patch makes suspicious config key replacements" in lowered:
        return "config_key_replacement"
    if "swe patch duplicates surrounding call wrappers" in lowered:
        return "duplicate_call_wrapper"
    if "swe patch inserts python-looking code into non-python file" in lowered:
        return "non_python_language_mismatch"
    if "swe patch makes suspicious semantic token flips" in lowered:
        return "semantic_token_flip"
    if "swe patch makes suspicious unknown attribute replacements" in lowered:
        return "unknown_attribute_replacement"
    if "swe patch introduces private attribute reads" in lowered:
        return "private_attribute_read"
    if "swe patch makes suspicious exception contract changes" in lowered:
        return "exception_contract_regression"
    if "swe patch adds redundant decorated normalizations" in lowered:
        return "redundant_decorated_normalization"
    if "swe patch changes only python string literal values" in lowered:
        return "string_literal_only_change"
    if "swe patch changes only python type annotations" in lowered:
        return "annotation_only_change"
    if "swe patch makes indentation-only statement moves" in lowered:
        return "indentation_only_statement_move"
    if "swe patch introduces none return value paths" in lowered:
        return "none_return_value_introduced"
    if "swe patch introduces none container misuse" in lowered:
        return "none_container_misuse"
    if "swe patch introduces function object arithmetic" in lowered:
        return "function_object_arithmetic"
    if "swe patch leaves invalid __init__ return values" in lowered:
        return "invalid_init_return_value"
    if "swe patch leaves invalid __init__ generators" in lowered:
        return "invalid_init_generator"
    if "swe patch introduces local use before assignment" in lowered:
        return "local_use_before_assignment"
    if "swe patch changes only tests or auxiliary update artifacts" in lowered:
        return "disallowed_swe_solution_path"
    if "swe patch removes module registration assignments" in lowered:
        return "module_registration_removed"
    if "swe patch introduces unresolved name reads" in lowered:
        return "unresolved_name_read"
    if "swe patch removes production return value paths" in lowered:
        return "return_value_path_removed"
    if "swe patch makes suspicious isolated boolean return flips" in lowered:
        return "suspicious_boolean_return_flip"
    if "swe patch makes suspicious python statement-kind replacements" in lowered:
        return "statement_kind_replacement"
    if "swe patch json syntax check failed" in lowered:
        return "json_syntax_error"
    if "policy terminated" in lowered:
        return "policy_terminated"
    if "governance rejected command" in lowered:
        return "governance_rejected"
    if "repeated failed action" in lowered:
        return "repeated_failed_action"
    if "no state progress" in lowered:
        return "no_state_progress"
    return "verification_failure"


def verification_failure_codes(verification: Mapping[str, Any] | None) -> list[str]:
    payload = verification if isinstance(verification, Mapping) else {}
    normalized: list[str] = []
    raw_codes = payload.get("failure_codes", [])
    if isinstance(raw_codes, list):
        for value in raw_codes:
            code = str(value).strip()
            if code and code not in normalized:
                normalized.append(code)
    if normalized:
        return normalized
    raw_reasons = payload.get("reasons", [])
    if isinstance(raw_reasons, list):
        for value in raw_reasons:
            code = classify_verification_reason(str(value))
            if code and code not in normalized:
                normalized.append(code)
    return normalized


def episode_success_criteria(episode: EpisodeRecord) -> dict[str, bool]:
    steps = list(episode.steps or [])
    terminal_verifier_passed = step_verification_passed(steps[-1]) if steps else bool(episode.success)
    all_steps_verified = all(step_verification_passed(step) for step in steps) if steps else terminal_verifier_passed
    task_success = bool(episode.success) or terminal_verifier_passed
    verifier_aligned_task_success = task_success and terminal_verifier_passed
    return {
        "task_success": task_success,
        "terminal_verifier_passed": terminal_verifier_passed,
        "all_steps_verified": all_steps_verified,
        "verifier_aligned_task_success": verifier_aligned_task_success,
    }
