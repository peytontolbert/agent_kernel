from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import ActionDecision, CommandResult, ContextPacket, StepRecord, TaskSpec


@dataclass(slots=True)
class AgentState:
    task: TaskSpec
    history: list[StepRecord] = field(default_factory=list)
    recent_workspace_summary: str = ""
    context_packet: ContextPacket | None = None
    available_skills: list[dict[str, object]] = field(default_factory=list)
    retrieval_direct_candidates: list[dict[str, str]] = field(default_factory=list)
    graph_summary: dict[str, object] = field(default_factory=dict)
    universe_summary: dict[str, object] = field(default_factory=dict)
    world_model_summary: dict[str, object] = field(default_factory=dict)
    workspace_snapshot: dict[str, str] = field(default_factory=dict)
    latest_state_transition: dict[str, object] = field(default_factory=dict)
    latent_state_summary: dict[str, object] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    initial_plan: list[str] = field(default_factory=list)
    active_subgoal: str = ""
    current_role: str = "executor"
    consecutive_failures: int = 0
    repeated_action_count: int = 0
    consecutive_no_progress_steps: int = 0
    last_action_signature: str = ""
    termination_reason: str = ""

    def next_step_index(self) -> int:
        return len(self.history) + 1

    def update_after_step(
        self,
        *,
        decision: ActionDecision,
        command_result: CommandResult | None,
        verification_passed: bool,
        step_index: int,
        progress_delta: float = 0.0,
        state_regressed: bool = False,
        state_transition: dict[str, object] | None = None,
    ) -> None:
        signature = f"{decision.action}:{decision.content}"
        if signature == self.last_action_signature:
            self.repeated_action_count += 1
        else:
            self.last_action_signature = signature
            self.repeated_action_count = 1

        if verification_passed:
            self.consecutive_failures = 0
            self.plan = []
            self.active_subgoal = ""
        else:
            self.consecutive_failures += 1
        if command_result is not None:
            if progress_delta > 0:
                self.consecutive_no_progress_steps = 0
            else:
                self.consecutive_no_progress_steps += 1
        self.latest_state_transition = dict(state_transition or {})
        if state_regressed:
            self.latest_state_transition["regressed"] = True

        if command_result is None:
            self.recent_workspace_summary = (
                f"step {step_index}: action={decision.action} without workspace mutation"
            )
            return

        outcome = "ok" if command_result.exit_code == 0 and not command_result.timed_out else "failed"
        self.recent_workspace_summary = (
            f"step {step_index}: command={command_result.command!r} "
            f"exit_code={command_result.exit_code} timed_out={command_result.timed_out} outcome={outcome}"
        )

    def refresh_plan_progress(self, world_model_summary: dict[str, object]) -> None:
        if not self.plan:
            self.active_subgoal = ""
            return
        remaining = [goal for goal in self.plan if not self._subgoal_satisfied(goal, world_model_summary)]
        self.plan = remaining
        self.active_subgoal = remaining[0] if remaining else ""

    def _subgoal_satisfied(self, goal: str, world_model_summary: dict[str, object]) -> bool:
        normalized = str(goal).strip()
        if not normalized:
            return True
        if normalized.startswith("materialize expected artifact "):
            path = normalized.removeprefix("materialize expected artifact ").strip()
            existing = {str(item) for item in world_model_summary.get("existing_expected_artifacts", [])}
            unsatisfied_contents = {str(item) for item in world_model_summary.get("unsatisfied_expected_contents", [])}
            if path not in existing:
                return False
            return path not in unsatisfied_contents
        if normalized.startswith("remove forbidden artifact "):
            path = normalized.removeprefix("remove forbidden artifact ").strip()
            present = {str(item) for item in world_model_summary.get("present_forbidden_artifacts", [])}
            return path not in present
        if normalized.startswith("preserve required artifact "):
            path = normalized.removeprefix("preserve required artifact ").strip()
            intact = {str(item) for item in world_model_summary.get("intact_preserved_artifacts", [])}
            return path in intact
        if normalized.startswith("update workflow path "):
            path = normalized.removeprefix("update workflow path ").strip()
            updated = {str(item) for item in world_model_summary.get("updated_workflow_paths", [])}
            return path in updated
        if normalized.startswith("regenerate generated artifact "):
            path = normalized.removeprefix("regenerate generated artifact ").strip()
            updated = {str(item) for item in world_model_summary.get("updated_generated_paths", [])}
            return path in updated
        if normalized.startswith("write workflow report "):
            path = normalized.removeprefix("write workflow report ").strip()
            updated = {str(item) for item in world_model_summary.get("updated_report_paths", [])}
            return path in updated
        if normalized.startswith("run workflow test "):
            return any(step.verification.get("passed", False) for step in self.history)
        return False

    def should_stop_for_stuckness(self) -> bool:
        if self.repeated_action_count >= 2 and self.consecutive_failures >= 2:
            self.termination_reason = "repeated_failed_action"
            return True
        if self.consecutive_no_progress_steps >= 3:
            self.termination_reason = "no_state_progress"
            return True
        return False
