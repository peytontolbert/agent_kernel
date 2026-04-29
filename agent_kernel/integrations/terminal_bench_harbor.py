from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from agent_kernel.actions import CODE_EXECUTE
from agent_kernel.config import KernelConfig
from agent_kernel.extensions.runtime_modeling_adapter import build_context_provider
from agent_kernel.llm import HybridDecoderClient, MockLLMClient, ModelStackClient, OllamaClient, VLLMClient
from agent_kernel.ops.loop_runtime_support import build_default_policy
from agent_kernel.policy import LLMDecisionPolicy, SkillLibrary
from agent_kernel.schemas import ActionDecision, CommandResult, StepRecord, TaskSpec, VerificationResult
from agent_kernel.state import AgentState

try:  # Harbor is an optional dependency for normal AgentKernel development.
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # pragma: no cover - exercised by import tests without Harbor installed.
    class BaseAgent:  # type: ignore[no-redef]
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name

    BaseEnvironment = Any  # type: ignore[misc, assignment]
    AgentContext = Any  # type: ignore[misc, assignment]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _agentkernel_version() -> str | None:
    try:
        return importlib.metadata.version("agent-kernel")
    except importlib.metadata.PackageNotFoundError:
        return None


def build_terminal_bench_config(model_name: str | None = None) -> KernelConfig:
    config = KernelConfig()
    config.provider = os.getenv("AGENT_KERNEL_PROVIDER", "vllm")
    config.model_name = model_name or os.getenv("AGENT_KERNEL_MODEL", config.model_name)
    config.vllm_autostart = _env_bool("AGENT_KERNEL_VLLM_AUTOSTART", False)
    config.max_steps = _env_int("AGENT_KERNEL_TBENCH_MAX_STEPS", 80)
    config.max_task_steps_hard_cap = max(config.max_task_steps_hard_cap, config.max_steps)
    config.llm_timeout_seconds = _env_int("AGENT_KERNEL_TBENCH_LLM_TIMEOUT_SECONDS", config.llm_timeout_seconds)
    config.command_timeout_seconds = _env_int(
        "AGENT_KERNEL_TBENCH_COMMAND_TIMEOUT_SECONDS",
        max(config.command_timeout_seconds, 120),
    )
    config.payload_history_step_window = _env_int("AGENT_KERNEL_TBENCH_HISTORY_STEPS", 12)
    config.runtime_history_step_window = max(config.runtime_history_step_window, config.payload_history_step_window)

    config.use_tolbert_context = _env_bool("AGENT_KERNEL_TBENCH_USE_TOLBERT_CONTEXT", False)
    config.use_graph_memory = _env_bool("AGENT_KERNEL_TBENCH_USE_GRAPH_MEMORY", False)
    config.use_world_model = _env_bool("AGENT_KERNEL_TBENCH_USE_WORLD_MODEL", False)
    config.use_universe_model = _env_bool("AGENT_KERNEL_TBENCH_USE_UNIVERSE_MODEL", False)
    config.use_planner = _env_bool("AGENT_KERNEL_TBENCH_USE_PLANNER", False)
    config.use_role_specialization = _env_bool("AGENT_KERNEL_TBENCH_USE_ROLE_SPECIALIZATION", False)
    config.use_skills = _env_bool("AGENT_KERNEL_TBENCH_USE_SKILLS", True)
    config.persist_learning_candidates = False
    config.persist_episode_memory = False
    return config


def build_terminal_bench_task(instruction: str, *, max_steps: int) -> TaskSpec:
    prompt = (
        "You are running inside a Terminal-Bench task container through Harbor. "
        "Use shell commands to inspect and modify the task workspace. "
        "Do not change benchmark timeouts, resources, /tests, /solution, or verifier files. "
        "When the task is complete, respond with done=true.\n\n"
        f"{instruction}"
    )
    return TaskSpec(
        task_id="terminal_bench_task",
        prompt=prompt,
        workspace_subdir="terminal_bench",
        max_steps=max_steps,
        metadata={
            "benchmark_family": "terminal-bench",
            "horizon": "long_horizon",
            "execution_environment": "harbor",
        },
    )


def _verification_payload(command_result: CommandResult | None, *, done: bool = False) -> dict[str, object]:
    if done:
        return VerificationResult(
            passed=False,
            reasons=["agent marked task complete; Harbor verifier determines final success"],
            command_result=command_result,
            outcome_label="agent_done",
            controllability="external_verifier",
        ).to_payload()
    if command_result is None:
        return VerificationResult(
            passed=False,
            reasons=["no command executed"],
            outcome_label="no_command_executed",
        ).to_payload()
    passed = command_result.exit_code == 0 and not command_result.timed_out
    return VerificationResult(
        passed=passed,
        reasons=["command completed"] if passed else [f"command exited {command_result.exit_code}"],
        command_result=command_result,
        process_score=1.0 if passed else 0.0,
        outcome_label="command_completed" if passed else "command_failed",
    ).to_payload()


class AgentKernelTerminalBenchAgent(BaseAgent):
    """Harbor external agent that runs AgentKernel's decision policy on Terminal-Bench."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "agent_kernel"

    def version(self) -> str | None:
        return _agentkernel_version()

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        await environment.exec("mkdir -p /logs/agent", timeout_sec=10)

    def _build_policy(self, config: KernelConfig):
        repo_root = Path(__file__).resolve().parents[2]
        return build_default_policy(
            config=config,
            repo_root=repo_root,
            skill_library_cls=SkillLibrary,
            llm_decision_policy_cls=LLMDecisionPolicy,
            context_provider_factory=build_context_provider,
            ollama_client_cls=OllamaClient,
            vllm_client_cls=VLLMClient,
            model_stack_client_cls=ModelStackClient,
            mock_client_factory=MockLLMClient,
            hybrid_client_factory=HybridDecoderClient,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        config = build_terminal_bench_config(self.model_name)
        task = build_terminal_bench_task(instruction, max_steps=config.max_steps)
        state = AgentState(task=task, current_role="executor")
        policy = self._build_policy(config)
        cwd = os.getenv("AGENT_KERNEL_TBENCH_CWD", "").strip() or None
        transcript: list[dict[str, object]] = []
        termination_reason = "max_steps_reached"

        try:
            for step_index in range(config.max_steps):
                decision = policy.decide(state)
                command_result = None
                if decision.action == CODE_EXECUTE and decision.content.strip():
                    result = await environment.exec(
                        decision.content,
                        cwd=cwd,
                        timeout_sec=config.command_timeout_seconds,
                    )
                    command_result = CommandResult(
                        command=decision.content,
                        exit_code=int(result.return_code),
                        stdout=result.stdout or "",
                        stderr=result.stderr or "",
                        timed_out=False,
                    )
                elif decision.done:
                    termination_reason = "policy_terminated"
                else:
                    decision = ActionDecision(
                        thought=decision.thought,
                        action="respond",
                        content=decision.content or "No command selected.",
                        done=True,
                        decision_source=decision.decision_source,
                    )
                    termination_reason = "policy_terminated"

                step = StepRecord(
                    index=step_index,
                    thought=decision.thought,
                    action=decision.action,
                    content=decision.content,
                    selected_skill_id=decision.selected_skill_id,
                    command_result=asdict(command_result) if command_result else None,
                    verification=_verification_payload(command_result, done=decision.done),
                    active_subgoal=state.active_subgoal,
                    acting_role=state.current_role,
                    decision_source=decision.decision_source,
                    tolbert_route_mode=decision.tolbert_route_mode,
                    proposal_source=decision.proposal_source,
                    proposal_novel=decision.proposal_novel,
                    proposal_metadata=dict(decision.proposal_metadata or {}),
                    shadow_decision=dict(decision.shadow_decision or {}),
                )
                state.history.append(step)
                transcript.append(asdict(step))
                self._write_transcript(transcript, termination_reason=termination_reason)

                if decision.done:
                    break
            else:
                termination_reason = "max_steps_reached"
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()

        self._write_transcript(transcript, termination_reason=termination_reason)
        self._populate_context(context, config=config, steps=len(transcript), termination_reason=termination_reason)

    def _write_transcript(self, transcript: list[dict[str, object]], *, termination_reason: str) -> None:
        payload = {
            "agent": self.name(),
            "model": self.model_name,
            "termination_reason": termination_reason,
            "steps": transcript,
        }
        path = self.logs_dir / "agent_kernel_terminal_bench_transcript.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _populate_context(
        self,
        context: AgentContext,
        *,
        config: KernelConfig,
        steps: int,
        termination_reason: str,
    ) -> None:
        metadata = {
            "agent": self.name(),
            "agent_version": self.version(),
            "model": config.model_name,
            "provider": config.provider,
            "steps": steps,
            "termination_reason": termination_reason,
            "transcript_path": str(self.logs_dir / "agent_kernel_terminal_bench_transcript.json"),
        }
        if hasattr(context, "metadata"):
            existing = getattr(context, "metadata", None)
            if isinstance(existing, dict):
                metadata = {**existing, **metadata}
            setattr(context, "metadata", metadata)


AgentKernelHarborAgent = AgentKernelTerminalBenchAgent

