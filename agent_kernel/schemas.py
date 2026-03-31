from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    reasons: list[str]
    command_result: CommandResult | None = None


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
    active_subgoal: str = ""
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
            "termination_reason": self.termination_reason,
            "steps": [asdict(step) for step in self.steps],
        }
