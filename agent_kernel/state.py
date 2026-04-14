from __future__ import annotations

from dataclasses import dataclass, field

from .ops import state_workflow_support
from .schemas import ActionDecision, CommandResult, ContextPacket, StepRecord, TaskSpec


def _canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())


def _software_work_phase_rank(phase: str) -> int:
    return state_workflow_support.software_work_phase_rank(phase)


def _software_work_objective_phase(
    objective: str,
    *,
    test_attempted: bool = False,
    regression_signaled: bool = False,
) -> str:
    return state_workflow_support.software_work_objective_phase(
        objective,
        test_attempted=test_attempted,
        regression_signaled=regression_signaled,
    )


@dataclass(slots=True)
class AgentState:
    task: TaskSpec
    history: list[StepRecord] = field(default_factory=list)
    history_archive: dict[str, object] = field(default_factory=dict)
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
    subgoal_diagnoses: dict[str, dict[str, object]] = field(default_factory=dict)
    planner_recovery_artifact: dict[str, object] = field(default_factory=dict)
    software_work_stage_state: dict[str, object] = field(default_factory=dict)
    current_role: str = "executor"
    consecutive_failures: int = 0
    repeated_action_count: int = 0
    consecutive_no_progress_steps: int = 0
    last_action_signature: str = ""
    termination_reason: str = ""

    def next_step_index(self) -> int:
        return self.completed_step_count() + 1

    def completed_step_count(self) -> int:
        return self.archived_step_count() + len(self.history)

    def archived_step_count(self) -> int:
        return int(self.history_archive.get("archived_step_count", 0) or 0)

    def all_executed_command_signatures(self) -> set[str]:
        signatures = {
            str(value)
            for value in self.history_archive.get("executed_command_signatures", [])
            if str(value).strip()
        }
        signatures.update(
            _canonicalize_command(str(step.content))
            for step in self.history
            if str(step.action).strip() and str(step.content).strip()
        )
        signatures.discard("")
        return signatures

    def all_successful_command_signatures(self) -> set[str]:
        signatures = {
            str(value)
            for value in self.history_archive.get("successful_command_signatures", [])
            if str(value).strip()
        }
        signatures.update(
            _canonicalize_command(str(step.content))
            for step in self.history
            if str(step.action).strip()
            and str(step.content).strip()
            and step.command_result
            and int(step.command_result.get("exit_code", 1)) == 0
            and not bool(step.command_result.get("timed_out", False))
        )
        signatures.discard("")
        return signatures

    def all_failed_command_signatures(self) -> set[str]:
        signatures = {
            str(value)
            for value in self.history_archive.get("failed_command_signatures", [])
            if str(value).strip()
        }
        signatures.update(
            _canonicalize_command(str(step.content))
            for step in self.history
            if str(step.content).strip() and not step.verification.get("passed", False)
        )
        signatures.discard("")
        return signatures

    def compact_history(self, *, max_recent_steps: int, summary_char_limit: int) -> None:
        if max_recent_steps < 1 or len(self.history) <= max_recent_steps:
            return
        archived_steps = self.history[:-max_recent_steps]
        self.history = self.history[-max_recent_steps:]
        self._append_history_archive(archived_steps, summary_char_limit=summary_char_limit)

    def _append_history_archive(self, steps: list[StepRecord], *, summary_char_limit: int) -> None:
        if not steps:
            return
        archive = dict(self.history_archive)
        archive["archived_step_count"] = int(archive.get("archived_step_count", 0) or 0) + len(steps)
        action_counts = dict(archive.get("action_counts", {})) if isinstance(archive.get("action_counts", {}), dict) else {}
        executed_signatures = {
            str(value)
            for value in archive.get("executed_command_signatures", [])
            if str(value).strip()
        }
        successful_signatures = {
            str(value)
            for value in archive.get("successful_command_signatures", [])
            if str(value).strip()
        }
        failed_signatures = {
            str(value)
            for value in archive.get("failed_command_signatures", [])
            if str(value).strip()
        }
        summary_lines = [
            str(line)
            for line in archive.get("recent_archived_summaries", [])
            if str(line).strip()
        ]
        for step in steps:
            action = str(step.action).strip() or "unknown"
            action_counts[action] = int(action_counts.get(action, 0) or 0) + 1
            canonical = _canonicalize_command(str(step.content))
            if canonical:
                executed_signatures.add(canonical)
                if step.command_result and int(step.command_result.get("exit_code", 1)) == 0 and not bool(
                    step.command_result.get("timed_out", False)
                ):
                    successful_signatures.add(canonical)
                if not step.verification.get("passed", False):
                    failed_signatures.add(canonical)
            summary_lines.append(self._history_summary_line(step))
        archive["action_counts"] = action_counts
        archive["executed_command_signatures"] = sorted(executed_signatures)
        archive["successful_command_signatures"] = sorted(successful_signatures)
        archive["failed_command_signatures"] = sorted(failed_signatures)
        archive["recent_archived_summaries"] = _trim_summary_lines(summary_lines, summary_char_limit=summary_char_limit)
        self.history_archive = archive

    @staticmethod
    def _history_summary_line(step: StepRecord) -> str:
        outcome = "pending"
        if step.command_result:
            exit_code = int(step.command_result.get("exit_code", 1))
            timed_out = bool(step.command_result.get("timed_out", False))
            outcome = "timeout" if timed_out else ("ok" if exit_code == 0 else f"exit={exit_code}")
        elif step.verification.get("passed", False):
            outcome = "verified"
        content = " ".join(str(step.content).split())
        if len(content) > 96:
            content = content[:93].rstrip() + "..."
        return f"{step.index}:{step.action}:{outcome}:{content}".strip(":")

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
        software_work_objective: str = "",
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
            self.subgoal_diagnoses = {}
            self.planner_recovery_artifact = {}
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
        if command_result is not None:
            self._record_software_work_outcome(
                objective=software_work_objective,
                step_index=step_index,
                command=str(decision.content),
                verification_passed=verification_passed,
                progress_delta=progress_delta,
                state_regressed=state_regressed,
                state_transition=self.latest_state_transition,
            )

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

    def refresh_plan_progress(
        self,
        world_model_summary: dict[str, object],
        *,
        expand_long_horizon: bool = True,
    ) -> None:
        state_workflow_support.refresh_plan_progress(
            self,
            world_model_summary,
            expand_long_horizon=expand_long_horizon,
        )

    def diagnosis_for_subgoal(self, goal: str) -> dict[str, object]:
        normalized = str(goal).strip()
        diagnosis = self.subgoal_diagnoses.get(normalized, {})
        return dict(diagnosis) if isinstance(diagnosis, dict) else {}

    def active_subgoal_diagnosis(self) -> dict[str, object]:
        if not self.active_subgoal:
            return {}
        return self.diagnosis_for_subgoal(self.active_subgoal)

    def world_horizon(self) -> str:
        return str(
            self.world_model_summary.get(
                "horizon",
                self.task.metadata.get("difficulty", self.task.metadata.get("horizon", "")),
            )
        ).strip()

    def learned_world_progress_and_risk(self) -> tuple[float, float]:
        learned = self.latent_state_summary.get("learned_world_state", {})
        learned = learned if isinstance(learned, dict) else {}
        progress = max(
            _float_value(learned.get("progress_signal"), 0.0),
            _float_value(learned.get("world_progress_score"), 0.0),
            _float_value(learned.get("decoder_world_progress_score"), 0.0),
            _float_value(learned.get("transition_progress_score"), 0.0),
        )
        risk = max(
            _float_value(learned.get("risk_signal"), 0.0),
            _float_value(learned.get("world_risk_score"), 0.0),
            _float_value(learned.get("decoder_world_risk_score"), 0.0),
            _float_value(learned.get("transition_regression_score"), 0.0),
        )
        return progress, risk

    def long_horizon_recovery_pressure(self) -> int:
        return max(
            int(self.consecutive_failures > 0),
            max(0, int(self.repeated_action_count) - 1),
            int(self.consecutive_no_progress_steps),
        )

    def software_work_plan_update(self) -> list[str]:
        return state_workflow_support.software_work_plan_update(self)

    def campaign_contract_state(self) -> dict[str, object]:
        return state_workflow_support.campaign_contract_state(self)

    def software_work_phase_gate_state(self) -> dict[str, object]:
        return state_workflow_support.software_work_phase_gate_state(self)

    def current_software_work_objective(self) -> str:
        return state_workflow_support.current_software_work_objective(self)

    def software_work_stage_overview(self) -> dict[str, object]:
        return state_workflow_support.software_work_stage_overview(self)

    def software_work_phase_state(self) -> dict[str, object]:
        return state_workflow_support.software_work_phase_state(self)

    def _software_work_base_objectives(self) -> list[str]:
        return state_workflow_support.software_work_base_objectives(self)

    def _reconcile_long_horizon_plan(
        self,
        remaining: list[str],
        world_model_summary: dict[str, object],
    ) -> list[str]:
        return state_workflow_support.reconcile_long_horizon_plan(self, remaining, world_model_summary)

    def _artifact_stage_objectives(self, world_model_summary: dict[str, object]) -> list[str]:
        return state_workflow_support.artifact_stage_objectives(self, world_model_summary)

    def _refresh_planner_recovery_progress(
        self,
        remaining: list[str],
        world_model_summary: dict[str, object],
    ) -> None:
        state_workflow_support.refresh_planner_recovery_progress(self, remaining, world_model_summary)

    def _record_software_work_outcome(
        self,
        *,
        objective: str,
        step_index: int,
        command: str,
        verification_passed: bool,
        progress_delta: float,
        state_regressed: bool,
        state_transition: dict[str, object],
    ) -> None:
        state_workflow_support.record_software_work_outcome(
            self,
            objective=objective,
            step_index=step_index,
            command=command,
            verification_passed=verification_passed,
            progress_delta=progress_delta,
            state_regressed=state_regressed,
            state_transition=state_transition,
        )

    def _pending_synthetic_edit_objectives(self) -> list[str]:
        return state_workflow_support.pending_synthetic_edit_objectives(self)

    def _pending_world_software_objectives(self) -> list[str]:
        return state_workflow_support.pending_world_software_objectives(self)

    def _campaign_contract_required_paths(self, objectives: list[str]) -> list[str]:
        return state_workflow_support.campaign_contract_required_paths(self, objectives)

    @staticmethod
    def _is_plan_trackable_subgoal(goal: str) -> bool:
        return state_workflow_support.is_plan_trackable_subgoal(goal)

    @staticmethod
    def _is_generic_contract_subgoal(goal: str) -> bool:
        return state_workflow_support.is_generic_contract_subgoal(goal)

    def _subgoal_satisfied(self, goal: str, world_model_summary: dict[str, object]) -> bool:
        return state_workflow_support.subgoal_satisfied(self, goal, world_model_summary)

    @staticmethod
    def _validation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
        return state_workflow_support.validation_contract_satisfied(world_model_summary)

    @staticmethod
    def _preservation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
        return state_workflow_support.preservation_contract_satisfied(world_model_summary)

    def _verifier_contract_satisfied(self, world_model_summary: dict[str, object]) -> bool:
        return state_workflow_support.verifier_contract_satisfied(self, world_model_summary)

    def _workflow_test_satisfied(self, label: str) -> bool:
        return state_workflow_support.workflow_test_satisfied(self, label)

    @staticmethod
    def _workflow_test_match_tokens(label: str, verifier: dict[str, object]) -> list[str]:
        return state_workflow_support.workflow_test_match_tokens(label, verifier)

    def _recent_command_mentions(self, token: str, *, prefixes: tuple[str, ...] = ()) -> bool:
        return state_workflow_support.recent_command_mentions(self, token, prefixes=prefixes)

    def should_stop_for_stuckness(self) -> bool:
        if self.repeated_action_count >= 2 and self.consecutive_failures >= 2:
            self.termination_reason = "repeated_failed_action"
            return True
        if self.consecutive_no_progress_steps >= 3:
            self.termination_reason = "no_state_progress"
            return True
        return False


def _trim_summary_lines(lines: list[str], *, summary_char_limit: int) -> list[str]:
    max_chars = max(128, int(summary_char_limit or 0))
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        normalized = str(line).strip()
        if not normalized:
            continue
        cost = len(normalized) + (1 if kept else 0)
        if kept and total + cost > max_chars:
            break
        kept.append(normalized)
        total += cost
    kept.reverse()
    return kept


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
